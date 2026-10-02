import html
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .publish import load
from .spec import load_spec, load_unit, page_plans

STYLE = """
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a64;--line:#dcd9cf;--bad:#c62828;--fix:#f0b400;--ok:#2e7d32;--sum:#1565c0}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--panel:#252523;--ink:#ecebe6;--muted:#a3a29b;--line:#3b3a36}}
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:18px;margin:0 0 4px} h2{font-size:15px;margin:18px 0 6px} .muted{color:var(--muted)}
main{padding:12px 16px} .page{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:12px;
border-top:1px solid var(--line);padding:12px 0} @media (max-width:900px){.page{grid-template-columns:1fr}}
.scan{position:relative} .scan img{width:100%;display:block;border:1px solid var(--line)}
.scan svg{position:absolute;inset:0;width:100%;height:100%} .data{overflow:auto;max-height:90vh}
table{border-collapse:collapse;font-size:12px} td,th{border:1px solid var(--line);padding:2px 4px;vertical-align:top}
th{background:var(--bg);position:sticky;top:0} td.bad{background:color-mix(in srgb,var(--bad) 22%,transparent)}
td.fix{background:color-mix(in srgb,var(--fix) 30%,transparent)} tr.sum td{font-weight:600}
.red{color:var(--bad)} mark{background:color-mix(in srgb,var(--fix) 40%,transparent)} pre{white-space:pre-wrap;font-size:12.5px}
.chip{display:inline-block;padding:0 6px;border-radius:8px;border:1px solid var(--line);margin-right:4px;font-size:12px}
a{color:inherit}
"""


def esc(value) -> str:
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def image_src(unit_dir: Path, page: dict, out_dir: Path, copy_images: bool = False) -> str:
    if not page["image"]:
        return ""
    if copy_images:
        target = out_dir / "images" / unit_dir.name / page["image"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(unit_dir / page["image"], target)
        return target.relative_to(out_dir).as_posix()
    return Path(os.path.relpath(unit_dir / page["image"], out_dir)).as_posix()


def overlay(boxes: list[tuple[list, str, str]]) -> str:
    shapes = []
    for box, color, title in boxes:
        if not box or len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        shapes.append(f'<rect x="{xmin / 10:.2f}%" y="{ymin / 10:.2f}%" width="{max(0, xmax - xmin) / 10:.2f}%" '
                      f'height="{max(0, ymax - ymin) / 10:.2f}%" fill="{color}" fill-opacity="0.06" stroke="{color}" '
                      f'stroke-width="1.5"><title>{esc(title)}</title></rect>')
    return f'<svg xmlns="http://www.w3.org/2000/svg">{"".join(shapes)}</svg>' if shapes else ""


def table_page(grid: dict, checks: pd.DataFrame, overrides: dict) -> tuple[str, list]:
    parts, boxes = [], []
    bad = {(c["table"], c["row"], c["column"]) for c in checks.to_dict("records")
           if c["status"] in ("mismatch", "mismatch_after_recheck", "reading_confirmed")}
    for t, table in enumerate(grid["result"].get("tables", [])):
        columns = table.get("columns", [])
        head = "".join(f"<th title='{esc(c.get('canonical'))}'>{esc(c.get('header'))}</th>" for c in columns)
        body = []
        for r, row in enumerate(table.get("rows", [])):
            red = {item["col"]: item["text"] for item in row.get("red", []) if isinstance(item, dict)}
            failed = any((t, r, c.get("canonical")) in bad for c in columns)
            color = "#c62828" if failed else ("#1565c0" if row.get("row_type") in ("sum", "carry_over") else "#8a8a80")
            boxes.append((row.get("box_2d"), color, f"t{t} r{r} {row.get('row_type')}"))
            cells = []
            for c, column in enumerate(columns):
                raw = row["cells"][c] if c < len(row["cells"]) else ""
                classes = []
                if (t, r, column.get("canonical")) in bad:
                    classes.append("bad")
                if (t, r, c) in overrides:
                    classes.append("fix")
                title = f"before re-check: {overrides[(t, r, c)]}" if (t, r, c) in overrides else ""
                extra = f"<span class='red'> {esc(red[c])}</span>" if c in red else ""
                cells.append(f"<td class='{' '.join(classes)}' title='{esc(title)}'>{esc(raw)}{extra}</td>")
            body.append(f"<tr class='{esc(row.get('row_type'))}'><td class='muted'>{r} {esc(row.get('row_type'))}</td>"
                        f"{''.join(cells)}</tr>")
        parts.append(f"<table><tr><th>row</th>{head}</tr>{''.join(body)}</table>")
    corrections = grid["result"].get("image_corrections", [])
    if corrections:
        parts.append("<h2>Scan vs. transcript</h2><table><tr><th>where</th><th>transcript</th><th>scan</th></tr>" +
                     "".join(f"<tr><td>{esc(c.get('where'))}</td><td>{esc(c.get('transcript_reads'))}</td>"
                             f"<td>{esc(c.get('image_reads'))}</td></tr>" for c in corrections) + "</table>")
    return "".join(parts), boxes


def text_page(proof: dict, statements: pd.DataFrame) -> str:
    text = esc(proof["corrected"])
    for correction in proof["applied"]:
        if correction["status"] == "applied" and correction["image_reads"]:
            target = esc(correction["image_reads"])
            text = text.replace(target, f"<mark title='transcript: {esc(correction['transcript_reads'])}'>{target}</mark>", 1)
    items = "".join(f"<tr><td>{esc(s['category'])}</td><td>{esc(s['subject'])}</td><td>{esc(s.get('value'))} "
                    f"{esc(s.get('value_unit'))}</td><td>{esc(s.get('time_text'))}</td><td>{esc(s.get('place_text'))}</td>"
                    f"<td>{esc(s['status'])}</td><td class='muted'>{esc(s['quote'])}</td></tr>"
                    for s in statements.to_dict("records"))
    table = (f"<h2>Statements ({len(statements)})</h2><table><tr><th>category</th><th>subject</th><th>value</th>"
             f"<th>time</th><th>place</th><th>status</th><th>quote</th></tr>{items}</table>") if items else ""
    return f"<pre>{text}</pre>{table}"


def map_page(record: dict) -> tuple[str, list]:
    width, height = record["image_size"]
    boxes = [([l["box_px"][1] / height * 1000, l["box_px"][0] / width * 1000, l["box_px"][3] / height * 1000,
               l["box_px"][2] / width * 1000], "#c62828", l["text"]) for l in record["labels"]]
    overview = record["overview"]
    legend = "".join(f"<li>{esc(i['symbol'])}: {esc(i['meaning'])}</li>" for i in overview.get("legend", []))
    labels = "".join(f"<tr><td>{esc(l['text'])}</td><td>{esc(l['class'])}</td><td>{esc(l.get('ink'))}</td>"
                     f"<td>{esc(l['confidence'])}</td></tr>" for l in record["labels"])
    return (f"<p><b>{esc(overview.get('title'))}</b> · {esc(overview.get('map_type'))} · {esc(overview.get('date_text'))} · "
            f"{esc(overview.get('scale_text'))}</p><ul>{legend}</ul><table><tr><th>label</th><th>class</th><th>ink</th>"
            f"<th>conf.</th></tr>{labels}</table>"), boxes


def write_unit_page(unit_dir: Path, run_dir: Path, out_dir: Path, checks: pd.DataFrame, statements: pd.DataFrame,
                    overrides: list, copy_images: bool = False) -> dict:
    unit, spec = load_unit(unit_dir), load_spec(unit_dir)
    unit_checks = checks[checks["unit"] == unit["id"]] if not checks.empty else checks
    sections = []
    for plan in page_plans(unit, spec):
        src = image_src(unit_dir, plan.page, out_dir, copy_images)
        content, boxes = "<p class='muted'>not extracted</p>", []
        grid_file = run_dir / "tables" / "grids" / unit["id"] / f"p{plan.position:03d}.json"
        proof_file = run_dir / "text" / unit["id"] / "proof" / f"p{plan.position:03d}.json"
        map_file = run_dir / "maps" / unit["id"] / f"p{plan.position:03d}" / "map.json"
        if plan.profile == "table" and grid_file.exists():
            page_checks = unit_checks[unit_checks["position"] == plan.position] if not unit_checks.empty else unit_checks
            page_overrides = {(o["table"], o["row"], o["col"]): o["before"] for o in overrides
                              if o["unit"] == unit["id"] and o["position"] == plan.position}
            content, boxes = table_page(json.loads(grid_file.read_text(encoding="utf-8")), page_checks, page_overrides)
        elif plan.profile == "text" and proof_file.exists():
            page_statements = statements[(statements["unit"] == unit["id"]) &
                                         (statements["page"] == f"p{plan.position:03d}")] if not statements.empty else statements
            content = text_page(json.loads(proof_file.read_text(encoding="utf-8")), page_statements)
        elif plan.profile == "map" and map_file.exists():
            content, boxes = map_page(json.loads(map_file.read_text(encoding="utf-8")))
        scan = f"<div class='scan'><img src='{esc(src)}' loading='lazy' alt='scan'>{overlay(boxes)}</div>" if src else \
            "<div class='muted'>no image</div>"
        sections.append(f"<section class='page' id='p{plan.position}'><div>{scan}<p class='muted'>p{plan.position} · "
                        f"{esc(plan.page['pid'])} · {plan.profile}/{plan.role}{' · form ' + esc(plan.form) if plan.form else ''}"
                        f"</p></div><div class='data'>{content}</div></section>")
    status = unit_checks.groupby("status").size().to_dict() if not unit_checks.empty else {}
    chips = "".join(f"<span class='chip'>{esc(k)} {v}</span>" for k, v in status.items())
    document = (f"<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' "
                f"content='width=device-width,initial-scale=1'><title>{esc(unit['id'])} {esc(unit['title'])}</title>"
                f"<style>{STYLE}</style></head><body><header><h1>{esc(unit['id'])} · {esc(unit['title'])}</h1>"
                f"<div class='muted'>{esc(spec.get('summary'))}</div><div>{chips}</div><a href='index.html'>← all units</a>"
                f"</header><main>{''.join(sections)}</main></body></html>")
    (out_dir / f"{unit['id']}.html").write_text(document, encoding="utf-8")
    return {"id": unit["id"], "title": unit["title"], "pages": len(unit["pages"]), "checks": status}


def write_review_site(units_dir: Path, run_dir: Path, out_dir: Path | None = None, copy_images: bool = False) -> Path:
    out_dir = out_dir or run_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = run_dir / "derived"
    checks, statements = load(derived, "table_checks"), load(derived, "text_statements")
    overrides_path = run_dir / "tables" / "overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else []
    entries = [write_unit_page(path.parent, run_dir, out_dir, checks, statements, overrides, copy_images)
               for path in sorted(units_dir.glob("*/unit.json")) if (path.parent / "spec.yaml").exists()]
    rows = "".join(f"<tr><td><a href='{esc(e['id'])}.html'>{esc(e['id'])}</a></td><td>{esc(e['title'])}</td>"
                   f"<td>{e['pages']}</td><td>{' '.join(f'{esc(k)} {v}' for k, v in e['checks'].items())}</td></tr>"
                   for e in entries)
    (out_dir / "index.html").write_text(
        f"<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>Waldstandsrevision review</title><style>{STYLE}</style></head><body><header><h1>"
        f"Waldstandsrevision Ilzertrift-Komplex 1878/90 – review</h1></header><main><table><tr><th>unit</th><th>title</th>"
        f"<th>pages</th><th>checks</th></tr>{rows}</table></main></body></html>", encoding="utf-8")
    return out_dir / "index.html"
