import json
import threading
from pathlib import Path

import pandas as pd
from PIL import Image

from .gemini import Gemini
from .spec import load_spec, load_unit

RECHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "row": {"type": "integer", "description": "our row index from the task"},
            "label": {"type": "string", "description": "first cell of the row as written (year, name, 'Su' …)"},
            "values": {"type": "array", "items": {"type": "object", "properties": {
                "column": {"type": "string", "description": "column id from the task"},
                "text": {"type": "string", "description": "cell exactly as written; '' if empty"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
                "required": ["column", "text", "confidence"]}}},
            "required": ["row", "label", "values"]}},
        "clerk_error_likely": {"type": "boolean", "description": "after re-reading, the source itself does not add up"},
        "note": {"type": "string"}},
    "required": ["rows", "clerk_error_likely"],
}

RECHECK_PROMPT = '''You verify readings of a handwritten table from a 19th-century Bavarian forest management plan.

The images show part of one page: image 1 over the full page width, image 2 (if present) zoomed into the columns to
check. Find each listed row in the images by its first cell and its current readings; the row numbers are only our
indexes and are not written on the page.

COLUMNS TO RE-READ (column id: header as printed):
{columns}

ROWS (our index: first cell – current reading per column id):
{rows}

WHY: these arithmetic checks fail with the current readings:
{failures}
So some readings are probably wrong – or the clerk made an error in the original.

TASK: For every listed row and every listed column, read the cell again from the images, digit by digit, and report
it exactly as written ("" if empty). Report what is written even if the numbers then still do not add up; never adjust
a value to make a check pass. Use the column ids above in "column"; put the row's first cell into "label", not into a
value.
'''

lock = threading.Lock()


def box_to_pixels(box, size):
    width, height = size
    ymin, xmin, ymax, xmax = box
    return xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height


def band_crop(image: Image.Image, rows: dict, x_range=None, longest: int = 2000) -> Image.Image:
    boxes = [box_to_pixels(box, image.size) for box in rows.values()]
    row_height = sum(b[3] - b[1] for b in boxes) / len(boxes)
    top = max(0, min(b[1] for b in boxes) - 3 * row_height)
    bottom = min(image.height, max(b[3] for b in boxes) + 3 * row_height)
    left, right = 0, image.width
    if x_range:
        left = max(0, x_range[0] / 1000 * image.width - 0.03 * image.width)
        right = min(image.width, x_range[1] / 1000 * image.width + 0.03 * image.width)
    crop = image.crop((int(left), int(top), int(right), int(bottom))).convert("RGB")
    factor = max(1.0, min(4.0, longest / max(crop.size)))
    return crop.resize((int(crop.width * factor), int(crop.height * factor)), Image.LANCZOS)


def column_ids_for(form: dict, column_id: str) -> list[str]:
    minors = [c["id"] for c in form.get("columns", []) if c["type"] == "pair_minor" and c.get("of") == column_id]
    return [column_id, *minors]


def form_column(form: dict, column_id: str) -> dict:
    return next((c for c in form.get("columns", []) if c["id"] == column_id), {})


def recheck_page(checks: list[dict], unit_dir: Path, run_dir: Path, gemini: Gemini, model: str, thinking: str) -> dict:
    first = checks[0]
    spec, unit = load_spec(unit_dir), load_unit(unit_dir)
    grid = json.loads((run_dir / "tables" / "grids" / first["unit"] / f"p{first['position']:03d}.json")
                      .read_text(encoding="utf-8"))
    table = grid["result"]["tables"][first["table"]]
    form = spec["forms"].get(grid["form"], {})
    canonical_ids, row_indexes = [], set()
    for check in checks:
        involved = [check["column"]]
        if check["rule"].startswith("row_sum"):
            involved += check["rule"].split(" ", 1)[1].split("+")
        for column_id in involved:
            for page_id in column_ids_for(form, column_id):
                if page_id not in canonical_ids:
                    canonical_ids.append(page_id)
        row_indexes |= {r for p, t, r in check["rows_used"] if p == check["position"] and t == check["table"]}
        row_indexes.add(check["row"])
    page_columns = {cid: [i for i, c in enumerate(table["columns"]) if c.get("canonical") == cid] for cid in canonical_ids}
    page_columns = {cid: cols[0] for cid, cols in page_columns.items() if cols}
    rows = {i: table["rows"][i]["box_2d"] for i in sorted(row_indexes)
            if i < len(table["rows"]) and len(table["rows"][i].get("box_2d") or []) == 4}
    if not page_columns or not rows:
        return {"checks": checks, "overrides": [], "skipped": "no columns or row boxes"}
    page = next(p for p in unit["pages"] if p["position"] == first["position"])
    images = []
    if page["image"]:
        image = Image.open(unit_dir / page["image"])
        images.append(band_crop(image, rows))
        ranges = [table["columns"][c].get("x_range") for c in page_columns.values()]
        ranges = [r for r in ranges if r and len(r) == 2]
        if ranges:
            images.append(band_crop(image, rows, (min(r[0] for r in ranges), max(r[1] for r in ranges))))
    column_lines = "\n".join(f"- {cid}: {table['columns'][col]['header']}"
                             + (f" ({form_column(form, cid).get('label')})" if form_column(form, cid) else "")
                             for cid, col in page_columns.items())
    row_lines = "\n".join(
        f"{i}: {next((c for c in table['rows'][i]['cells'] if c.strip()), '')[:40]} – "
        + "; ".join(f"{cid} = {table['rows'][i]['cells'][col] if col < len(table['rows'][i]['cells']) else ''}"
                    for cid, col in page_columns.items())
        for i in rows)
    failures = "\n".join(f"- row {c['row']}, {c['column']}: {c['rule']} – row reads {c['found']}, current readings "
                         f"give {', '.join(f'{k} = {v}' for k, v in c['expected'].items())}" for c in checks)
    prompt = RECHECK_PROMPT.format(columns=column_lines, rows=row_lines, failures=failures)
    data, usage, _ = gemini.extract(prompt, RECHECK_SCHEMA, images=images, model=model, thinking=thinking)
    overrides = []
    for answer in data.get("rows", []):
        index = answer.get("row")
        if index not in rows:
            continue
        cells = table["rows"][index]["cells"]
        label = next((c for c in cells if c.strip()), "").strip()
        for value in answer.get("values", []):
            col = page_columns.get(value.get("column"))
            text = (value.get("text") or "").strip()
            if col is None or col >= len(cells) or text == cells[col].strip():
                continue
            if text and text == label and col != 0:
                continue
            overrides.append({"unit": first["unit"], "position": first["position"], "table": first["table"],
                              "row": index, "col": col, "column": value["column"], "raw": text,
                              "before": cells[col], "confidence": value.get("confidence"), "model": model})
    return {"checks": [{k: c[k] for k in ("unit", "position", "table", "row", "column", "rule", "found")}
                       for c in checks],
            "overrides": overrides, "clerk_error_likely": data.get("clerk_error_likely"),
            "note": data.get("note", ""), "usage": usage}


def check_key(check: dict) -> tuple:
    return tuple(check[k] for k in ("unit", "position", "table", "row", "column", "rule"))


def read_log(run_dir: Path) -> list[dict]:
    path = run_dir / "tables" / "recheck_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def run_recheck(units_dir: Path, run_dir: Path, gemini: Gemini, model: str, thinking: str = "high",
                round_number: int = 1, max_checks_per_page: int = 25) -> list[dict]:
    from .pipeline import run_parallel
    checks = pd.read_json(run_dir / "derived" / "table_checks.jsonl", lines=True)
    pending = checks[checks.status.isin(["mismatch", "mismatch_after_recheck"])
                     & ~checks.rule.eq("carry_over_from_previous_page")].to_dict("records")
    done = {check_key(c) for entry in read_log(run_dir) if entry.get("round") == round_number for c in entry["checks"]}
    groups = {}
    for check in pending:
        if check_key(check) not in done:
            groups.setdefault((check["unit"], check["position"], check["table"]), []).append(check)
    unit_dirs = {json.loads(p.read_text(encoding="utf-8"))["id"]: p.parent for p in units_dir.glob("*/unit.json")}
    tasks = [(f"{unit} p{position} t{table}",
              lambda group=group[:max_checks_per_page], unit=unit: recheck_page(group, unit_dirs[unit], run_dir, gemini,
                                                                                model, thinking))
             for (unit, position, table), group in groups.items()]
    results, _ = run_parallel(tasks, gemini.settings.workers, f"re-check round {round_number}")
    overrides_path = run_dir / "tables" / "overrides.json"
    existing = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else []
    keyed = {(o["unit"], o["position"], o["table"], o["row"], o["col"]): o for o in existing}
    for result in results:
        for override in result["overrides"]:
            keyed[(override["unit"], override["position"], override["table"], override["row"], override["col"])] = \
                {**override, "round": round_number}
    overrides_path.write_text(json.dumps(list(keyed.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    with lock, open(run_dir / "tables" / "recheck_log.jsonl", "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({**result, "round": round_number}, ensure_ascii=False) + "\n")
    return results


def mark_discrepancies(checks: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    log = read_log(run_dir)
    if checks.empty or not log:
        return checks
    changed_cells = {(o["unit"], o["position"], o["table"], o["row"]) for entry in log for o in entry["overrides"]}
    rechecked = {check_key(c) for entry in log for c in entry["checks"]}

    def status(row):
        key = check_key(row)
        if key not in rechecked:
            return row["status"]
        if row["status"] == "ok":
            return "ok_after_recheck"
        if row["status"] != "mismatch":
            return row["status"]
        touched = any((row["unit"], row["position"], row["table"], r) in changed_cells
                      for p, t, r in [*row["rows_used"], (row["position"], row["table"], row["row"])])
        return "mismatch_after_recheck" if touched else "reading_confirmed"

    checks = checks.copy()
    checks["status"] = checks.apply(status, axis=1)
    return checks
