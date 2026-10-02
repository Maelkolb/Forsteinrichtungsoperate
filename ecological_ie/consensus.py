import json
import shutil
from pathlib import Path

from .checks import BAD_FLAGS, check_unit
from .gemini import Gemini
from .grid import read_grid
from .normalize import KeyState, normalise_table
from .pipeline import grid_path, run_parallel, unit_dirs
from .spec import load_spec, load_unit, page_plans


def page_score(grid: dict, form: dict) -> dict:
    state, previous, tables = KeyState(), {}, []
    score = {"ok": 0, "mismatch": 0, "unparseable": 0, "shape": 0, "out_of_range": 0, "bad_numbers": 0, "rows": 0}
    numeric = {c["id"] for c in form.get("columns", []) if c["type"] in ("number", "pair_major", "pair_minor")}
    for index, table in enumerate(grid["result"].get("tables", [])):
        context = {"unit": "", "position": 0, "page_id": "", "table": index, "form": "", "data_index": 0}
        rows, cells = normalise_table(table, form, state, previous, context)
        tables.append((form, rows))
        score["rows"] += len(rows)
        score["shape"] += sum(not row["shape_ok"] for row in rows)
        score["out_of_range"] += sum(cell["flag"] == "pair_minor_out_of_range" for cell in cells)
        score["bad_numbers"] += sum(cell["flag"] in BAD_FLAGS and cell["canonical"] in numeric for cell in cells)
    for check in check_unit(tables):
        if check["status"] in score:
            score[check["status"]] += 1
    score["value"] = (score["ok"] - 2 * score["mismatch"] - score["unparseable"] - score["shape"]
                      - score["out_of_range"] - 0.5 * score["bad_numbers"])
    return score


def needs_second_reading(score: dict) -> bool:
    problems = score["mismatch"] + score["shape"] + score["out_of_range"] + score["bad_numbers"]
    return problems >= 3 or (score["mismatch"] and score["mismatch"] >= score["ok"])


def problem_hint(score: dict) -> str:
    parts = []
    if score["shape"]:
        parts.append(f"{score['shape']} rows had a different number of cells than columns")
    if score["out_of_range"]:
        parts.append(f"{score['out_of_range']} cells had Kreuzer/Pfennig or decimal parts out of range – a sign that "
                     "the columns were shifted against each other")
    if score["bad_numbers"]:
        parts.append(f"{score['bad_numbers']} number cells could not be read as numbers")
    if score["mismatch"]:
        parts.append(f"{score['mismatch']} sums did not match the rows above them")
    return ("A first reading of this page had problems: " + "; ".join(parts) + ". Rebuild the column structure "
            "carefully from the scan (every sub-column, left to right) and keep each value in its own column.")


def second_reading(plan, spec: dict, unit_dir: Path, run_dir: Path, gemini: Gemini, model: str | None,
                   thinking: str) -> dict | None:
    unit_id = spec["unit"]
    path = grid_path(run_dir, unit_id, plan.position)
    form = spec.get("forms", {}).get(plan.form, {})
    first = json.loads(path.read_text(encoding="utf-8"))
    if first.get("second_reading_done"):
        return None
    first_score = page_score(first, form)
    if not needs_second_reading(first_score):
        return None
    alt_path = run_dir / "tables" / "grids_alt" / unit_id / f"p{plan.position:03d}.json"
    second = read_grid(plan, spec, unit_dir, gemini, alt_path, force=False, model=model, thinking=thinking,
                       hint=problem_hint(first_score))
    second_score = page_score(second, form)
    winner = "second" if second_score["value"] > first_score["value"] else "first"
    if winner == "second":
        shutil.copy(path, alt_path.with_suffix(".first.json"))
        second["second_reading_done"] = True
        second["selection"] = {"first": first_score, "second": second_score, "winner": winner}
        path.write_text(json.dumps(second, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        first["second_reading_done"] = True
        first["selection"] = {"first": first_score, "second": second_score, "winner": winner}
        path.write_text(json.dumps(first, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"unit": unit_id, "position": plan.position, "winner": winner, "first": first_score, "second": second_score}


def run_second_readings(units_dir: Path, run_dir: Path, gemini: Gemini, sections=None, model=None,
                        thinking: str = "high") -> list[dict]:
    tasks = []
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec:
            continue
        for plan in page_plans(unit, spec):
            if plan.profile == "table" and grid_path(run_dir, unit["id"], plan.position).exists():
                tasks.append((f"{unit['id']} p{plan.position}",
                              lambda plan=plan, spec=spec, unit_dir=unit_dir:
                              second_reading(plan, spec, unit_dir, run_dir, gemini, model, thinking)))
    results, _ = run_parallel(tasks, gemini.settings.workers, "second readings")
    results = [r for r in results if r]
    replaced = {(r["unit"], r["position"]) for r in results if r["winner"] == "second"}
    overrides_path = run_dir / "tables" / "overrides.json"
    if replaced and overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        overrides = [o for o in overrides if (o["unit"], o["position"]) not in replaced]
        overrides_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(run_dir / "tables" / "second_readings.jsonl", "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")
    return results
