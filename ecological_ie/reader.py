import base64
import json
from pathlib import Path

import pandas as pd

from .edition import plain_text, render_grid, render_transcript
from .layout import apply_row_layout, load_layout
from .pipeline import apply_overrides, load_overrides
from .publish import text_of
from .spec import load_spec, load_unit, page_plans

FAILING = {"mismatch", "mismatch_after_recheck"}
ROMAN = {"I": 1, "II": 2}


def changed_pairs(pairs) -> list:
    cleaned = ([plain_text(before), plain_text(after)] for before, after in pairs)
    return [pair for pair in cleaned if pair[0] != pair[1]]


def toc_order(unit_id: str) -> tuple:
    heft, _, number = unit_id.partition("-")
    return ROMAN.get(heft, 9), int(number) if number.isdigit() else 99


def parse_json(value, default):
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def table_of_contents(units_dir: Path, units: list[dict]) -> dict:
    index_path = units_dir / "units.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    linked = {u["id"]: u for u in units}
    entries = [{"id": u["id"], "heft": u["heft"], "nr": u["nr"], "title": u["title"], "linked": True} for u in units]
    for section in index.get("unlinked_sections", []):
        if section["id"] == "kopf" or section["id"] in linked:
            continue
        heft, _, nr = section["id"].partition("-")
        entries.append({"id": section["id"], "heft": f"{heft}. Heft", "nr": nr.lstrip("0") if nr.isdigit() else nr,
                        "title": section["title"], "linked": False})
    entries.sort(key=lambda e: toc_order(e["id"]))
    image = units_dir / "toc_page.jpg"
    data_uri = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode() if image.exists() else ""
    return {"title": index.get("toc_title", ""), "toc_page": index.get("toc", ""), "entries": entries, "img": data_uri}


def form_models(spec: dict) -> dict:
    return {name: {"name": form["name"],
                   "columns": [[c["id"], c["label"], c["type"], c.get("unit", ""), c.get("variable", "")]
                               for c in form["columns"]]}
            for name, form in (spec.get("forms") or {}).items()}


def page_checks(checks: pd.DataFrame, unit_id: str, position: int) -> pd.DataFrame:
    if checks.empty:
        return checks
    return checks[(checks["unit"] == unit_id) & (checks["position"] == position)]


def failing_list(checks: pd.DataFrame, grid: dict, form: dict) -> list:
    labels = {c["id"]: c["label"] for c in form.get("columns", [])}
    tables = grid["result"].get("tables", [])
    out = []
    for check in checks.to_dict("records"):
        if check["status"] not in FAILING and check["status"] != "reading_confirmed":
            continue
        row_label = ""
        if check["table"] < len(tables) and check["row"] < len(tables[check["table"]].get("rows", [])):
            cells = tables[check["table"]]["rows"][check["row"]].get("cells", [])
            row_label = next((c for c in cells if str(c).strip()), "")
        expected = parse_json(check.get("expected"), {})
        best = min(expected.values(), key=lambda v: abs(v - check["found"])) if expected else None
        out.append([int(check["table"]), int(check["row"]), str(row_label)[:30], labels.get(check["column"], check["column"]),
                    check["rule"].split(" ")[0], check["found"], best, check["status"]])
    return out


def reader_pages(units_dir: Path, run_dir: Path, checks: pd.DataFrame, statements: list, events: list) -> tuple[dict, dict]:
    overrides = load_overrides(run_dir)
    raw_overrides = json.loads((run_dir / "tables" / "overrides.json").read_text(encoding="utf-8")) \
        if (run_dir / "tables" / "overrides.json").exists() else []
    quotes = {}
    for i, s in enumerate(statements):
        quotes.setdefault((s["u"], s["p"]), []).append((str(i), s["quote"]))
    for i, e in enumerate(events):
        quotes.setdefault((e["u"], e["p"]), []).append((f"e{i}", e["quote"]))
    reader, forms = {}, {}
    for unit_file in sorted(units_dir.glob("*/unit.json"), key=lambda p: toc_order(load_unit(p.parent)["id"])):
        unit_dir = unit_file.parent
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec:
            continue
        forms[unit["id"]] = form_models(spec)
        notes = {int(k): v.get("note", "") for k, v in (spec.get("pages") or {}).items()}
        pages = []
        for plan in page_plans(unit, spec):
            record = {"p": plan.position, "profile": plan.profile, "role": plan.role, "note": notes.get(plan.position, ""),
                      "form": plan.form or "", "stmts": [i for i, s in enumerate(statements) if s["u"] == unit["id"] and s["p"] == plan.position],
                      "events": [i for i, e in enumerate(events) if e["u"] == unit["id"] and e["p"] == plan.position]}
            proof_path = run_dir / "text" / unit["id"] / "proof" / f"p{plan.position:03d}.json"
            grid_path = run_dir / "tables" / "grids" / unit["id"] / f"p{plan.position:03d}.json"
            layout = load_layout(run_dir, unit["id"], plan.position) or {}
            if plan.profile == "text" and proof_path.exists():
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                page_quotes = quotes.get((unit["id"], plan.position), [])
                record.update({
                    "kind": "text",
                    "html": render_transcript(proof["corrected"], proof["applied"], page_quotes, line_anchors=True),
                    "htr": render_transcript(proof["text"]),
                    "corr": changed_pairs([c.get("transcript_reads", ""), c.get("image_reads", "")] for c in proof["applied"]
                                          if c.get("status") == "applied"),
                    "quality": proof["result"].get("page_quality", ""),
                    "lines": layout.get("lines", {}),
                    "regions": [[r["type"].removesuffix("Region"), r["box"]] for r in layout.get("regions", [])]})
            elif plan.profile == "table" and grid_path.exists():
                grid = apply_overrides(json.loads(grid_path.read_text(encoding="utf-8")),
                                       overrides.get((unit["id"], plan.position), {}))
                grid = apply_row_layout(grid, layout)
                record["cols"] = {str(t): [c.get("x_range") or [0, 0] for c in table.get("columns", [])]
                                  for t, table in enumerate(grid["result"].get("tables", []))}
                form = (spec.get("forms") or {}).get(plan.form, {})
                these = page_checks(checks, unit["id"], plan.position)
                bad = {(int(c["table"]), int(c["row"]), c["column"]) for c in these.to_dict("records") if c["status"] in FAILING}
                reread = {(o["table"], o["row"], o["col"]): o.get("before", "") for o in raw_overrides
                          if o["unit"] == unit["id"] and o["position"] == plan.position}
                record.update({
                    "kind": "table",
                    "html": render_grid(grid, form, bad, reread),
                    "outside": [[item.get("kind", ""), item.get("text", "")] for item in grid["result"].get("text_outside_tables", [])],
                    "scan_corr": [[c.get("where", ""), *pair] for c in grid["result"].get("image_corrections", [])
                                  for pair in changed_pairs([[c.get("transcript_reads", ""), c.get("image_reads", "")]])],
                    "checks": these.groupby("status").size().to_dict() if not these.empty else {},
                    "fails": failing_list(these, grid, form),
                    "reread": len(reread)})
            elif plan.profile == "map":
                record["kind"] = "map"
            else:
                record["kind"] = "none"
            pages.append(record)
        reader[unit["id"]] = pages
    return reader, forms
