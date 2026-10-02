import html
import json
import os
import re
from pathlib import Path

COLORS = {"tree_species": "#c8e6c9", "stand_structure": "#dcedc8", "site_conditions": "#ffe0b2",
          "ground_vegetation": "#f0f4c3", "climate_weather": "#b3e5fc", "damage_event": "#ffcdd2",
          "regeneration": "#d1c4e9", "silvicultural_measure": "#e1bee7", "non_timber_use": "#ffecb3",
          "wildlife": "#f8bbd0", "land_use_hydrology": "#b2dfdb", "quantity": "#e0e0e0", "other_ecological": "#eeeeee"}

STYLE = """body{font-family:system-ui,sans-serif;margin:0;padding:12px} .cols{display:flex;gap:12px}
.left,.right{flex:1;min-width:0;overflow:auto;max-height:85vh;border:1px solid #ccc;padding:8px}
.src{white-space:pre-wrap;font-size:13px;line-height:1.45} table{border-collapse:collapse;font-size:12px}
td,th{border:1px solid #bbb;padding:3px 5px;vertical-align:top} mark sup{font-size:9px} .page{display:none}
.page.active{display:block} pre{font-size:11px;white-space:pre-wrap} .right table td{font-size:11px}
.src table{font-size:11px} .src td,.src th{border:1px solid #999;padding:2px 4px} .src .red{color:#c00}
img.scan{max-width:100%;border:1px solid #999}"""


def escape(value) -> str:
    return html.escape(str(value if value is not None else ""))


def highlight(text: str, items) -> str:
    spans = []
    for quote, category, idx in sorted(items, key=lambda item: -len(item[0] or "")):
        if not quote:
            continue
        start = text.find(quote)
        if start < 0:
            match = re.search(re.escape(" ".join(quote.split()[:5])).replace(r"\ ", r"\s+"), text)
            if not match:
                continue
            start, end = match.start(), min(len(text), match.start() + len(quote))
        else:
            end = start + len(quote)
        if any(not (end <= s or start >= e) for s, e, *_ in spans):
            continue
        spans.append((start, end, category, idx))
    out, pos = [], 0
    for start, end, category, idx in sorted(spans):
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark style="background:{COLORS.get(category, "#eee")}" title="{escape(category)} #{idx}">'
                   f'{html.escape(text[start:end])}<sup>{idx}</sup></mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def species_text(record) -> str:
    species = (record.get("stand") or {}).get("species", []) or []
    return ", ".join(f"{s.get('name_de')} {s.get('share_value') if s.get('share_value') is not None else '?'}"
                     for s in species)


def finding_row(i, finding) -> str:
    return (f'<tr style="background:{COLORS.get(finding.get("category"), "#eee")}"><td>{i}</td>'
            f'<td>{escape(finding.get("category"))}<br><small>{escape(finding.get("subtype"))}</small></td>'
            f'<td><b>{escape(finding.get("entity"))}</b><br><small>{escape(finding.get("entity_en"))}</small></td>'
            f'<td>{escape(finding.get("value"))} {escape(finding.get("unit"))}</td><td>{escape(finding.get("date"))}</td>'
            f'<td>{escape(finding.get("location"))}</td><td>{escape(finding.get("status"))}</td>'
            f'<td>{escape(finding.get("confidence"))}</td><td><small>{escape(finding.get("note"))}</small></td></tr>')


def corrections_html(result) -> str:
    corrections = result.get("image_corrections") or []
    if not corrections:
        return ""
    rows = "".join(f"<tr><td>{escape(c.get('where'))}</td><td>{escape(c.get('transcript_reads'))}</td>"
                   f"<td>{escape(c.get('image_reads'))}</td><td>{escape(c.get('confidence'))}</td></tr>"
                   for c in corrections)
    return (f"<p><b>Image corrections ({len(corrections)})</b></p><table><tr><th>where</th><th>transcript</th>"
            f"<th>scan</th><th>conf</th></tr>{rows}</table>")


def text_panels(page, result):
    items = [(f.get("quote"), f.get("category"), i) for i, f in enumerate(result.get("findings", []))]
    left = f'<pre class="src">{highlight(page["text"], items)}</pre>'
    species = ", ".join(f'{s.get("name_de")} ({s.get("role")}{"" if s.get("share_value") is None else ", " + str(s.get("share_value"))})'
                        for s in result.get("tree_species", []))
    rows = "".join(finding_row(i, f) for i, f in enumerate(result.get("findings", [])))
    right = (f'<p><b>{escape(result.get("document_type"))}</b> · quality {escape(result.get("transcription_quality"))}'
             f'<br>{escape(result.get("summary_en"))}</p><p><b>Species:</b> {escape(species)}</p>'
             f'<table><tr><th>#</th><th>category</th><th>entity</th><th>value</th><th>date</th><th>location</th>'
             f'<th>status</th><th>conf</th><th>note</th></tr>{rows}</table>')
    return left, right


def table_panels(page, result):
    text = page["text"]
    left = (f'<div class="src">{text.replace("```html", "").replace("```", "")}</div>' if "<table" in text
            else f'<pre class="src">{html.escape(text)}</pre>')
    records = "".join(
        f'<details {"open" if i < 3 else ""}><summary><b>#{i} {escape(rec.get("record_kind"))}</b> '
        f'{escape(rec.get("district_no"))} {escape(rec.get("district_name"))} / {escape(rec.get("compartment_no"))} '
        f'{escape(rec.get("subcompartment"))} · {escape(rec.get("area_value"))} {escape(rec.get("area_unit"))} · '
        f'species {escape(species_text(rec))}</summary>'
        f'<pre>{escape(json.dumps({k: v for k, v in rec.items() if v not in ("", None, [], {})}, ensure_ascii=False, indent=1))}</pre></details>'
        for i, rec in enumerate(result.get("records", [])))
    right = (f'<p><b>{escape(result.get("table_type"))}</b> · {escape(result.get("forest_office"))} · '
             f'{escape(result.get("operating_class"))}<br><small>headers: {escape(" | ".join(result.get("column_headers", []) or []))}'
             f'</small><br><small>notes: {escape(result.get("parsing_notes"))}</small></p>{records}')
    return left, right


def page_section(seq, page, record, out_dir: Path) -> str:
    result = record["result"]
    left, right = (text_panels if page["source_type"] == "text" else table_panels)(page, result)
    if record.get("image"):
        image_src = Path(os.path.relpath(record["image"], out_dir)).as_posix()
        left = f'<details><summary>scan</summary><img class="scan" src="{escape(image_src)}"></details>' + left
    right += corrections_html(result)
    return (f'<section id="{escape(seq)}" class="page"><h2>{escape(seq)} · {escape(page["subtype"])} · '
            f'{escape(page["page_id"])}</h2><div class="cols"><div class="left">{left}</div>'
            f'<div class="right">{right}</div></div></section>')


def write_review_viewer(results: dict, pages_by_seq: dict, out_dir: Path, title: str) -> Path:
    options = "".join(f'<option value="{escape(seq)}">{escape(seq)} · {escape(pages_by_seq[seq]["subtype"])} · '
                      f'{escape(pages_by_seq[seq]["page_id"])}</option>' for seq in results)
    legend = " ".join(f'<span style="background:{color};padding:2px 6px;border-radius:3px">{name}</span>'
                      for name, color in COLORS.items())
    sections = "".join(page_section(seq, pages_by_seq[seq], record, out_dir) for seq, record in results.items())
    document = (f'<!doctype html><html><head><meta charset="utf-8"><title>Forsteinrichtung IE review</title>'
                f'<style>{STYLE}</style></head><body><h1>{escape(title)}</h1><p>{legend}</p>'
                f'<select id="sel" onchange="show(this.value)" style="width:100%;font-size:14px">{options}</select>'
                f'{sections}<script>function show(id){{document.querySelectorAll(".page").forEach('
                f'e=>e.classList.toggle("active",e.id===id));}} show(document.getElementById("sel").value);</script>'
                f'</body></html>')
    path = out_dir / "review_viewer.html"
    path.write_text(document, encoding="utf-8")
    return path
