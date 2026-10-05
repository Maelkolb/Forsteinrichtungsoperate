import html
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .explorer import events_list, statements_list
from .publish import load
from .reader import reader_pages, table_of_contents
from .spec import load_spec, load_unit

STATUS = {
    "ok": ("sum agrees", "#0ca30c"),
    "ok_after_recheck": ("agrees after re-reading", "#0ca30c"),
    "reading_confirmed": ("reading confirmed, sum differs", "#fab219"),
    "mismatch_after_recheck": ("sum differs", "#d03b3b"),
    "unparseable": ("number unreadable", "#ec835a"),
}
CATEGORY = {"tree_species": "Species", "stand_structure": "Stand", "site_conditions": "Site", "ground_vegetation": "Ground flora",
            "climate_weather": "Climate", "damage_event": "Damage", "regeneration": "Regeneration",
            "silvicultural_measure": "Silviculture", "non_timber_use": "Use", "wildlife": "Wildlife",
            "land_use_hydrology": "Land and water", "quantity": "Quantity", "other_ecological": "Other"}
EVENT_TYPES = {"windthrow": "Windthrow", "bark_beetle": "Bark beetle", "snow_break": "Snow break", "ice_break": "Rime break",
               "frost": "Frost", "fungal": "Rot", "game_damage": "Game browsing", "grazing_damage": "Grazing damage",
               "fire": "Fire", "drought": "Drought", "flood": "Flood", "other_insects": "Insects", "other": "Other"}
PROFILE = {"text": "Text", "table": "Table", "map": "Map"}
ROLE = {"title": "title page", "copy": "copy of another page", "empty": "empty page", "other": "other"}
OUTSIDE = {"marginalia": "Margin", "signature": "Signature", "stamp": "Stamp", "note": "Note", "other": "Other"}
STATIC = Path(__file__).parent / "static"
SCRIPTS = "\n".join((STATIC / name).read_text(encoding="utf-8") for name in ("zoom.js", "review.js"))
FONTS = ("https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400"
         "&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap")

STYLE = """
:root{--bg:#fafaf8;--surface:#fff;--surface-2:#f2f2ef;--ink:#1a1c1e;--ink-2:#4d5358;--muted:#868b8f;--rule:#e4e4df;--rule-2:#cbccc6;
--accent:#1f5a4e;--accent-wash:#e5eeea;--redink:#ad2a20;--corr-wash:#f8ecd2;--quote:#e3efe9;--crit-wash:#fbe5e2;--grid:#e8e8e4;
--serif:"Source Serif 4","Iowan Old Style","Palatino Linotype",Georgia,serif;--sans:"Source Sans 3","Segoe UI",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121413;--surface:#181b1a;--surface-2:#1e2221;--ink:#e7eae8;--ink-2:#b0b7b3;--muted:#838b87;
--rule:#2a302e;--rule-2:#3b4441;--accent:#8cc4b0;--accent-wash:#1d2b27;--redink:#f07a6f;--corr-wash:#3a3120;--quote:#203a32;
--crit-wash:#3a1d1b;--grid:#252b29;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-underline-offset:3px}
h1,h2,h3{margin:0;font-weight:600}
h1{font-family:var(--serif);font-size:clamp(1.7rem,3vw,2.2rem);line-height:1.15}
h2{font-family:var(--serif);font-size:1.12rem}
h3{font-size:.86rem;color:var(--ink-2);padding-bottom:6px;border-bottom:1px solid var(--ink-2);margin:22px 0 4px}
p{margin:0}
.muted{color:var(--muted)}
.top{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--rule-2)}
.top-inner{display:flex;align-items:center;gap:24px;padding:9px 24px;min-height:54px}
.brand{display:grid;line-height:1.15;text-decoration:none;color:var(--ink)}
.brand b{font-family:var(--serif);font-size:1.05rem}
.brand span{font-size:.78rem;color:var(--muted)}
.pager{margin-left:auto;display:flex;gap:18px;font-size:.9rem;flex-wrap:wrap}
.page{max-width:1500px;margin:0 auto;padding:36px 24px 80px;display:grid;gap:30px}
.head{display:grid;gap:8px;max-width:980px}
.head .no{color:var(--muted);font-weight:400;margin-right:12px}
.head .src{color:var(--ink-2);font-size:.92rem}
.head .summary{color:var(--ink-2);font-size:.95rem;line-height:1.55}
.overview{display:grid;gap:10px;padding-top:16px;border-top:1px solid var(--ink)}
.overview .big{font-family:var(--serif);font-size:1.1rem;font-weight:600}
.statusbar{display:flex;height:4px;border-radius:2px;overflow:hidden;background:var(--grid);gap:1px;max-width:520px}
.statusbar span{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.85rem;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.jump{display:flex;flex-wrap:wrap;gap:2px 4px;font-size:.84rem}
.jump a{display:inline-block;min-width:34px;text-align:center;padding:1px 4px;border-radius:3px;text-decoration:none;color:var(--ink-2);position:relative}
.jump a:hover{background:var(--surface-2);color:var(--ink)}
.jump a.flag::after{content:"";position:absolute;top:1px;right:2px;width:5px;height:5px;border-radius:50%;background:#d03b3b}
.pg{display:grid;gap:12px;padding-top:16px;border-top:1px solid var(--rule-2);scroll-margin-top:70px}
.pg-head{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.pg-head .kind{color:var(--muted);font-size:.88rem}
.pg-body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:28px;align-items:start}
@media (max-width:980px){.pg-body{grid-template-columns:minmax(0,1fr)}}
.viewer{position:sticky;top:70px;height:calc(100vh - 90px);border:1px solid var(--rule)}
@media (max-width:980px){.viewer{position:relative;top:auto;height:70vh}}
.scan{margin:0;background:var(--surface-2);border:1px solid var(--rule)}
.scan .none{padding:40px 20px;color:var(--muted);font-style:italic;font-family:var(--serif)}
.zoom-overlay .row-outline{fill:none;stroke:#9aa09c;stroke-opacity:.7;stroke-width:1}
.zoom-overlay .row-outline.failed{stroke:#d03b3b;stroke-opacity:1;stroke-width:1.5}
.zoom-overlay .band{fill:rgba(173,42,32,.1);stroke:var(--redink);stroke-width:1.5}
.zoom-overlay .cell{fill:none;stroke:var(--accent);stroke-width:2.5}
.zoom-overlay .line-hl{fill:var(--quote);fill-opacity:.6;stroke:var(--accent);stroke-width:1.2}
.zoom-overlay .region{fill:none;stroke:var(--ink-2);stroke-width:1;stroke-dasharray:6 4;stroke-opacity:.55}
.zoom-overlay .label{fill:transparent;stroke:var(--redink);stroke-width:1.2;pointer-events:all;cursor:help}
.tx .ln{display:block;padding-left:1.5em;text-indent:-1.5em}
.tx .ln .marg{text-indent:0}
.tx .ln.hl,.tx [data-l].hl{background:var(--accent-wash)}
table.grid tbody tr{cursor:pointer}
table.grid tbody tr.hl td,table.grid tbody tr:hover td{background-color:var(--accent-wash)}
.content{min-width:0;display:grid;gap:4px;align-content:start}
.note{color:var(--muted);font-style:italic;font-family:var(--serif)}
.tx{font-family:var(--serif);font-size:1.02rem;line-height:1.7;max-width:760px}
.tx h1,.tx h2,.tx h3,.tx h4{font-family:var(--serif);font-size:1.08rem;line-height:1.45;margin:1em 0 .5em;text-align:center;color:var(--ink);border:0;padding:0}
.tx p{margin:0 0 .9em}
.tx .lb{display:block}
.tx u{text-decoration-thickness:1px;text-underline-offset:3px}
.tx .red,.grid .red{color:var(--redink)}
.tx .marg{float:right;clear:right;display:block;width:30%;margin:.3em 0 .8em 22px;padding-left:10px;border-left:1px solid var(--rule-2);font-size:.84rem;line-height:1.45;color:var(--ink-2);font-style:italic}
.tx .unc{font-size:.72em;vertical-align:super;color:var(--accent);font-family:var(--sans);font-weight:600}
.tx .unc-w{text-decoration:underline dotted var(--accent);text-underline-offset:3px}
.tx .ed{font-style:italic;color:var(--muted);font-family:var(--sans);font-size:.86em}
.tx .ed::before{content:"\\27E8"}.tx .ed::after{content:"\\27E9"}
.tx .sup-text,.tx del{color:var(--muted)}
.tx mark.corr{background:var(--corr-wash);color:inherit;border-radius:2px}
.tx .q{background:var(--quote);border-radius:2px}
.tx table{border-collapse:collapse;margin:.6em 0 1.2em;font:.88rem/1.4 var(--sans);font-variant-numeric:tabular-nums;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}
.tx table th,.tx table td{border-bottom:1px solid var(--rule);border-left:1px solid var(--rule);padding:4px 8px;vertical-align:top}
.tx table th:first-child,.tx table td:first-child{border-left:0}
.tx table th{font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--ink-2)}
.form-head{display:grid;gap:4px;margin:0 0 14px}
.form-head .form-nr{justify-self:end;font-size:.86rem;color:var(--ink-2);font-family:var(--serif)}
.form-head .ttl{font-family:var(--serif);font-size:1.04rem;line-height:1.45;white-space:pre-line;text-align:center}
.grid-wrap{overflow-x:auto;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink);margin-bottom:12px}
table.grid{border-collapse:collapse;font:12.5px/1.35 var(--sans);font-variant-numeric:tabular-nums;width:max-content;min-width:100%}
table.grid th{font-weight:600;color:var(--ink-2);font-size:11.5px;text-align:center;vertical-align:bottom;padding:5px 7px;border-bottom:1px solid var(--rule-2);border-left:1px solid var(--rule);max-width:150px}
table.grid th:first-child,table.grid td:first-child{border-left:0}
table.grid tr.canon-row{display:none}
table.grid td{padding:3px 7px;border-left:1px solid var(--rule);border-bottom:1px solid var(--rule);vertical-align:top;max-width:320px}
table.grid td.n{text-align:right;white-space:nowrap}
table.grid tr.sum td{font-weight:600;border-top:1px solid var(--ink-2)}
table.grid tr.carry_over td,table.grid tr.note td{font-style:italic;color:var(--ink-2)}
table.grid tr.group_header td,table.grid tr.heading td{font-weight:600;background:var(--surface-2)}
table.grid td.bad{background:var(--crit-wash);box-shadow:inset 0 -2px 0 #d03b3b}
table.grid td.rr{background:var(--corr-wash)}
table.grid td.unc-c{text-decoration:underline dotted var(--accent);text-underline-offset:3px}
.page-notes{display:grid;gap:8px;margin-top:6px}
.page-notes div{display:grid;grid-template-columns:84px minmax(0,1fr);gap:12px;font-size:.88rem}
.page-notes .k{color:var(--muted);font-size:.8rem;padding-top:2px}
.page-notes .t{font-family:var(--serif);font-style:italic;white-space:pre-line;color:var(--ink-2)}
table.list{border-collapse:collapse;width:100%;font-size:.86rem}
table.list th{text-align:left;font-weight:600;color:var(--muted);font-size:.78rem;padding:4px 8px 4px 0;border-bottom:1px solid var(--rule-2)}
table.list td{padding:5px 8px 5px 0;border-bottom:1px solid var(--rule);vertical-align:top}
table.list td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.list .quote{font-family:var(--serif);color:var(--ink)}
.lemma{display:flex;flex-wrap:wrap;gap:0 8px;align-items:baseline;padding:5px 0;border-bottom:1px solid var(--rule);font-size:.9rem}
.lemma .l{font-family:var(--serif)}
.lemma .br,.lemma .sig{color:var(--muted)}
.lemma .sig{font-size:.74rem}
.lemma .r{font-family:var(--serif);font-style:italic;color:var(--ink-2)}
.lemma .w{flex-basis:100%;font-size:.78rem;color:var(--muted)}
.summary-line{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:16px;font-size:.9rem}
.contents{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 240px;gap:44px;align-items:start}
@media (max-width:1100px){.contents{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.toc-scan{display:none}}
@media (max-width:760px){.contents{grid-template-columns:minmax(0,1fr)}}
.heft h2{padding-bottom:8px;border-bottom:1px solid var(--ink)}
.toc-row{display:grid;grid-template-columns:26px minmax(0,1fr) 150px;gap:12px;align-items:baseline;padding:9px 2px;border-bottom:1px solid var(--rule);text-decoration:none;color:var(--ink)}
.toc-row:hover .ti{color:var(--accent)}
.toc-row .nr{font-family:var(--serif);text-align:right;color:var(--ink-2)}
.toc-row .ti{font-family:var(--serif);line-height:1.38}
.toc-row .meta{display:grid;gap:5px;justify-items:end;font-size:.78rem;color:var(--muted);text-align:right}
.toc-row .meta .statusbar{width:70px}
.toc-row.off{color:var(--muted)}
.toc-row.off .ti,.toc-row.off .nr{color:var(--muted)}
.toc-scan img{width:100%;border:1px solid var(--rule)}
"""


def esc(value) -> str:
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def image_src(unit_dir: Path, page: dict, out_dir: Path, copy_images: bool = False, image_urls: dict | None = None) -> str:
    if not page["image"]:
        return ""
    if image_urls and f"{unit_dir.name}/{page['image']}" in image_urls:
        return image_urls[f"{unit_dir.name}/{page['image']}"]
    if copy_images:
        target = out_dir / "images" / unit_dir.name / page["image"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(unit_dir / page["image"], target)
        return target.relative_to(out_dir).as_posix()
    return Path(os.path.relpath(unit_dir / page["image"], out_dir)).as_posix()


def merged_status(counts: dict) -> dict:
    merged = {k: v for k, v in counts.items() if k != "mismatch"}
    merged["mismatch_after_recheck"] = merged.get("mismatch_after_recheck", 0) + counts.get("mismatch", 0)
    return {k: merged[k] for k in STATUS if merged.get(k)}


def status_bar(counts: dict) -> str:
    merged = merged_status(counts)
    total = sum(merged.values())
    if not total:
        return ""
    spans = "".join(f'<span style="width:{v / total * 100:.2f}%;background:{STATUS[k][1]}{";opacity:.55" if k == "ok_after_recheck" else ""}"></span>'
                    for k, v in merged.items())
    return f'<div class="statusbar">{spans}</div>'


def status_legend(counts: dict) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="dot" style="background:{STATUS[k][1]}"></span><b>{v}</b> {STATUS[k][0]}</span>'
        for k, v in merged_status(counts).items()) + "</div>"


def agree_line(counts: dict) -> str:
    total = sum(counts.values())
    good = counts.get("ok", 0) + counts.get("ok_after_recheck", 0)
    return f"{good} of {total} sums agree" if total else ""


def number(value) -> str:
    return "–" if value is None or (isinstance(value, float) and pd.isna(value)) else f"{value:,.2f}"


def archival(page: dict | None) -> str:
    return f"{page['sig']}, p. {page['num']}" if page else ""


def page_range(pages: list[dict]) -> str:
    if not pages:
        return ""
    first, last = pages[0], pages[-1]
    if first["sig"] == last["sig"] and first["num"] == last["num"]:
        return archival(first)
    if first["sig"] == last["sig"]:
        return f"{first['sig']}, pp. {first['num']}–{last['num']}"
    return f"{archival(first)} – {archival(last)}"


def section_number(unit: dict) -> str:
    heft = "II" if unit["id"].startswith("II") else "I"
    return f"{heft}. {unit['nr']}" if unit.get("nr") else heft


def kind_label(record: dict) -> str:
    profile = PROFILE.get(record["profile"], "Not extracted")
    role = ROLE.get(record["role"], "")
    return f"{profile}, {role}" if role else profile


def lemmata(pairs: list, label: str) -> str:
    return "".join(f'<div class="lemma"><span class="l">{esc(lemma) or "–"}</span><span class="br">]</span>'
                   f'<span class="r">{esc(reading) or "–"}</span><span class="sig">{label}</span>'
                   + (f'<span class="w">{esc(where)}</span>' if where else "") + "</div>"
                   for lemma, reading, where in pairs)


def table_content(record: dict) -> str:
    heads = [o for o in record["outside"] if o[0] in ("form_number", "heading")]
    notes = [o for o in record["outside"] if o[0] in OUTSIDE]
    parts = []
    if heads:
        parts.append('<div class="form-head">' + "".join(
            f'<span class="form-nr">{esc(t)}</span>' if k == "form_number" else f'<div class="ttl">{esc(t)}</div>' for k, t in heads) + "</div>")
    parts.append(record["html"])
    if notes:
        parts.append('<div class="page-notes">' + "".join(f'<div><span class="k">{OUTSIDE[k]}</span><span class="t">{esc(t)}</span></div>'
                                                         for k, t in notes) + "</div>")
    if record["checks"]:
        parts.append(f'<div class="summary-line"><b>{agree_line(record["checks"])}</b>{status_legend(record["checks"])}</div>')
    if record["reread"]:
        parts.append(f'<p class="muted" style="font-size:.86rem">{record["reread"]} cells changed on re-reading, shaded yellow; '
                     "hover a cell for the first reading.</p>")
    if record["fails"]:
        rows = "".join(f'<tr><td>{esc(str(label).rstrip(" ,.;:")) or f"Row {r + 1}"}</td><td>{esc(column)}</td>'
                       f'<td class="n">{number(found)}</td><td class="n">{number(expected)}</td>'
                       f'<td><span class="dot" style="background:{STATUS.get(status, STATUS["mismatch_after_recheck"])[1]}"></span> '
                       f'{STATUS.get(status, STATUS["mismatch_after_recheck"])[0]}</td></tr>'
                       for _, r, label, column, _, found, expected, status in record["fails"])
        parts.append('<h3>Open sums</h3><table class="list"><tr><th>Row</th><th>Column</th><th>Written</th><th>Rows give</th>'
                     f"<th>Status</th></tr>{rows}</table>")
    if record["scan_corr"]:
        parts.append("<h3>Read from the facsimile</h3>" + lemmata([(scan, htr, where) for where, htr, scan in record["scan_corr"]], "HTR"))
    return "".join(parts)


def text_content(record: dict, statements: list, events: list) -> str:
    parts = [f'<div class="tx">{record["html"]}</div>']
    if record["corr"]:
        parts.append("<h3>Corrections</h3>" + lemmata([(checked, htr, "") for htr, checked in record["corr"]], "HTR"))

    def located(key: str) -> bool:
        return f'data-q="{key}"' in record["html"]

    if record["events"]:
        rows = "".join(f'<tr><td>{EVENT_TYPES.get(events[i]["type"], esc(events[i]["type"]))}</td><td>{esc(events[i]["date"])}</td>'
                       f'<td>{esc(events[i]["place"])}</td><td>{esc(events[i]["extent"])}</td>'
                       f'<td class="quote">{esc(events[i]["quote"])}</td><td>{"" if located(f"e{i}") else "not located"}</td></tr>'
                       for i in record["events"])
        parts.append('<h3>Events</h3><table class="list"><tr><th>Event</th><th>Date</th><th>Place</th><th>Extent</th><th>Passage</th>'
                     f"<th></th></tr>{rows}</table>")
    if record["stmts"]:
        rows = []
        for i in record["stmts"]:
            s = statements[i]
            subject = f"{s['subj']}, {s['attr']}" if s["attr"] else s["subj"]
            value = f"{s['val']} {s['vu']}".strip()
            rows.append(f"<tr><td>{CATEGORY.get(s['cat'], esc(s['cat']))}</td><td><b>{esc(subject)}</b></td><td>{esc(value)}</td>"
                        f"<td>{esc(s['time'])}</td><td>{esc(s['place'])}</td><td>{esc(s['status'])}</td>"
                        f"<td class='quote'>{esc(s['quote'])}</td><td class='muted'>{'' if located(str(i)) else 'not located'}</td></tr>")
        parts.append(f'<h3>Statements</h3><table class="list"><tr><th>Category</th><th>Statement</th><th>Value</th><th>Time</th>'
                     f"<th>Place</th><th>Status</th><th>Passage, shaded in the text</th><th></th></tr>{''.join(rows)}</table>")
    return "".join(parts)


def map_content(map_record: dict) -> tuple[str, list]:
    width, height = map_record["image_size"]
    boxes = [[l["text"], [round(l["box_px"][1] / height * 1000), round(l["box_px"][0] / width * 1000),
                           round(l["box_px"][3] / height * 1000), round(l["box_px"][2] / width * 1000)]] for l in map_record["labels"]]
    overview = map_record["overview"]
    facts = ", ".join(esc(overview.get(k)) for k in ("map_type", "date_text", "scale_text") if overview.get(k))
    legend = "".join(f'<div class="lemma"><span class="l">{esc(i["meaning"])}</span><span class="r">{esc(i["symbol"])}</span></div>'
                     for i in overview.get("legend", []))
    labels = "".join(f"<tr><td>{esc(l['text'])}</td><td>{esc(l['class']).replace('_', ' ')}</td><td>{esc(l.get('ink'))}</td>"
                     f"<td>{esc(l['confidence'])}</td></tr>" for l in map_record["labels"])
    return (f'<h2 style="margin-bottom:4px">{esc(overview.get("title"))}</h2><p class="muted">{facts}</p>'
            + (f"<h3>Legend</h3>{legend}" if legend else "")
            + f'<h3>Labels</h3><table class="list"><tr><th>Label</th><th>Class</th><th>Ink</th><th>Confidence</th></tr>{labels}</table>'), boxes


def document(title: str, body: str, pager: str = "") -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link rel="stylesheet" href="{FONTS}"><style>{STYLE}</style></head><body>'
            f'<header class="top"><div class="top-inner"><a class="brand" href="index.html"><b>Waldstandsrevision Ilzertrift-Komplex</b>'
            f'<span>Review of the extraction</span></a><nav class="pager">{pager}</nav></div></header>'
            f'<main class="page">{body}</main><script>{SCRIPTS}</script></body></html>')


def write_unit_page(unit: dict, unit_dir: Path, run_dir: Path, out_dir: Path, records: list, checks: pd.DataFrame,
                    statements: list, events: list, neighbours: tuple, copy_images: bool = False,
                    image_urls: dict | None = None) -> dict:
    pages = {p["position"]: p for p in unit["pages"]}
    unit_checks = checks[checks["unit"] == unit["id"]] if not checks.empty else checks
    counts = unit_checks.groupby("status").size().to_dict() if not unit_checks.empty else {}
    sections, jump = [], []
    for record in records:
        position, page = record["p"], pages.get(record["p"])
        src = image_src(unit_dir, page, out_dir, copy_images, image_urls) if page else ""
        width, height = (page or {}).get("image_size") or [0, 0]
        view = {"src": src, "alt": f"Scan {page['pid']}" if page else "", "w": width, "h": height}
        map_file = run_dir / "maps" / unit["id"] / f"p{position:03d}" / "map.json"
        if record["kind"] == "table":
            content = table_content(record)
            view["cols"] = record.get("cols", {})
        elif record["kind"] == "text":
            content = text_content(record, statements, events)
            view.update({"lines": record.get("lines", {}), "regions": record.get("regions", [])})
        elif record["kind"] == "map" and map_file.exists():
            content, view["labels"] = map_content(json.loads(map_file.read_text(encoding="utf-8")))
        else:
            content = f'<p class="note">{esc(record["note"]) or "Not transcribed."}</p>'
        data = json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        scan = (f'<div class="viewer"></div><script type="application/json" class="page-data">{data}</script>'
                if src else '<figure class="scan"><p class="none">No image of this page in the data set.</p></figure>')
        failing = any(f[7] != "reading_confirmed" for f in record.get("fails", []))
        jump.append(f'<a href="#p{position}"{" class=flag" if failing else ""}>{esc(page["num"]) if page else position}</a>')
        sections.append(f'<section class="pg" id="p{position}"><div class="pg-head"><h2>{esc(archival(page))}</h2>'
                        f'<span class="kind">{kind_label(record)}</span></div>'
                        f'<div class="pg-body">{scan}<div class="content">{content}</div></div></section>')
    previous, following = neighbours
    pager = "".join([f'<a href="{p["id"]}.html">← {esc(section_number(p))}</a>' if p else "" for p in [previous]]
                    + ['<a href="index.html">Contents</a>']
                    + [f'<a href="{n["id"]}.html">{esc(section_number(n))} →</a>' if n else "" for n in [following]])
    overview = (f'<div class="overview">' + (f'<span class="big">{agree_line(counts)}</span>{status_bar(counts)}{status_legend(counts)}' if counts else "")
                + f'<div class="jump">{"".join(jump)}</div></div>')
    head = (f'<div class="head"><h1><span class="no">{esc(section_number(unit))}</span>{esc(unit["title"])}</h1>'
            f'<p class="src">{esc(page_range([pages[r["p"]] for r in records if r["p"] in pages]))}</p>'
            + (f'<p class="summary">{esc(unit.get("summary") or "")}</p>' if unit.get("summary") else "") + "</div>")
    (out_dir / f"{unit['id']}.html").write_text(document(f"{section_number(unit)} {unit['title']}", head + overview + "".join(sections), pager),
                                                encoding="utf-8")
    return {"id": unit["id"], "pages": len(records), "checks": counts,
            "range": page_range([pages[r["p"]] for r in records if r["p"] in pages])}


def write_review_site(units_dir: Path, run_dir: Path, out_dir: Path | None = None, copy_images: bool = False,
                      image_urls: dict | None = None) -> Path:
    out_dir = out_dir or run_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = run_dir / "derived"
    checks = load(derived, "table_checks")
    statements, events = statements_list(load(derived, "text_statements")), events_list(load(derived, "text_events"))
    reader, _ = reader_pages(units_dir, run_dir, checks, statements, events)
    unit_dirs = {load_unit(path.parent)["id"]: path.parent for path in units_dir.glob("*/unit.json")}
    units = [dict(load_unit(unit_dirs[u]), summary=(load_spec(unit_dirs[u]).get("summary") or "").strip()) for u in reader]
    entries = {}
    for i, unit in enumerate(units):
        neighbours = (units[i - 1] if i else None, units[i + 1] if i + 1 < len(units) else None)
        entries[unit["id"]] = write_unit_page(unit, unit_dirs[unit["id"]], run_dir, out_dir, reader[unit["id"]], checks,
                                              statements, events, neighbours, copy_images, image_urls)
    toc = table_of_contents(units_dir, [{"id": u["id"], "heft": u["heft"], "nr": u["nr"], "title": u["title"]} for u in units])

    def row(entry: dict) -> str:
        number = f'{esc(entry["nr"])}.' if entry["nr"] and entry["nr"] != "WHB" else ""
        if not entry["linked"]:
            return (f'<div class="toc-row off"><span class="nr">{number}</span><span class="ti">{esc(entry["title"])}</span>'
                    f'<span class="meta">not linked</span></div>')
        info = entries[entry["id"]]
        return (f'<a class="toc-row" href="{entry["id"]}.html"><span class="nr">{number}</span><span class="ti">{esc(entry["title"])}</span>'
                f'<span class="meta">{esc(info["range"])}{status_bar(info["checks"])}</span></a>')

    heft_one = "".join(row(e) for e in toc["entries"] if not e["id"].startswith("II"))
    heft_two = "".join(row(e) for e in toc["entries"] if e["id"].startswith("II"))
    all_counts = checks.groupby("status").size().to_dict() if not checks.empty else {}
    body = ('<div class="head"><h1>Waldstandsrevision für den Ilzertrift-Komplex</h1>'
            '<p class="src">Reviere Schönau, St. Oswald und Klingenbrunn, Forstamt Schönberg, 1878 mit 1890</p></div>'
            f'<div class="overview"><span class="big">{agree_line(all_counts)}</span>{status_bar(all_counts)}{status_legend(all_counts)}</div>'
            f'<div class="contents"><section class="heft"><h2>I. Heft</h2>{heft_one}</section>'
            f'<section class="heft"><h2>II. Heft</h2>{heft_two}</section>'
            + (f'<figure class="toc-scan" style="margin:0"><img src="{toc["img"]}" alt="Inhaltsverzeichnis"></figure>' if toc["img"] else "")
            + "</div>")
    (out_dir / "index.html").write_text(document("Waldstandsrevision Ilzertrift-Komplex, review", body), encoding="utf-8")
    return out_dir / "index.html"
