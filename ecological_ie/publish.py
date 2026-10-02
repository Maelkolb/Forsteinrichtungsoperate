import json
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from .spec import load_spec, load_unit

REVIER_NAMES = [("schönberg", "Schönberg"), ("schoenberg", "Schönberg"), ("oswald", "St. Oswald"),
                ("klingenbrunn", "Klingenbrunn"), ("schön", "Schönau"), ("schoen", "Schönau"), ("schonau", "Schönau")]
STANDARD_UNITS = {"tagwerk": ("ha", 0.3407, "1 bayer. Tagwerk = 0.3407 ha"),
                  "tgw": ("ha", 0.3407, "1 bayer. Tagwerk = 0.3407 ha"),
                  "klafter": ("Ster", 3.1325, "1 Klafter = 3.1325 Ster, the factor used in the source (I-16, II-05)"),
                  "fl": ("M", 12 / 7, "1 Gulden = 12/7 Mark, the factor used in the source (I-20)"),
                  "gulden": ("M", 12 / 7, "1 Gulden = 12/7 Mark, the factor used in the source (I-20)")}
ROMAN = re.compile(r"^[IVXLC]+$")

FIELD_DOCS = {
    "unit": "TOC item of the Operat (e.g. I-11 = I. Heft, Nr. 11)",
    "position": "page position inside the unit (volume order)",
    "page_id": "archival page id: [page] archive, fonds signature",
    "table": "index of the table on the page", "row": "row index in the table (top to bottom)",
    "col": "column index on the page (left to right)", "row_type": "data, group_header, sum, carry_over, heading, note",
    "raw": "cell as written (black ink)", "red": "red-ink content of the cell", "value": "normalised number",
    "unit_of_measure": "unit of the value (row heading overrides the column unit)",
    "measure_unit": "unit of the column as defined in the unit spec",
    "variable": "observed quantity (English name from the unit spec)", "period": "year or period of the value",
    "scope": "Revier, class or area the value refers to", "check_status": "result of the arithmetic checks covering this cell",
    "stand_id": "normalised stand key Revier|Distrikt|Abteilung|Unterabteilung",
    "value_std": "value converted to ha / Ster / Mark with the factor in 'conversion' (empty factor = unchanged)",
    "page_role": "role of the page in its unit; 'copy' = duplicate copy, exclude from totals",
    "box_2d": "approximate row box on the page image, [ymin, xmin, ymax, xmax] in 0-1000",
    "quote": "verbatim evidence from the (proofread) text", "quote_check": "where the quote was found",
}


def text_of(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()


def revier_name(text) -> str:
    text = text_of(text)
    plain = text.lower().replace("sct.", "").replace("st.", "").replace("revier", "").strip(" .,")
    for key, name in REVIER_NAMES:
        if key in plain:
            return name
    return text


UNIT_FAMILIES = {
    "money": {"fl", "fl.", "gulden", "m", "m.", "mark", "kr", "pf"},
    "volume": {"klafter", "klftr", "ster", "str", "kubikmeter", "raummeter", "fmtr", "fm", "rm"},
    "area": {"tagwerk", "tgw", "ha", "hektar", "dez", "ar"},
    "length": {"ruthen", "meter", "m", "km"},
}


def unit_family(unit) -> str:
    unit = text_of(unit).lower().strip(" .")
    return next((name for name, members in UNIT_FAMILIES.items() if unit in members or unit + "." in members), "")


def effective_unit(row_unit, column_unit) -> str:
    row_unit, column_unit = text_of(row_unit), text_of(column_unit)
    if row_unit and (not column_unit or unit_family(row_unit) == unit_family(column_unit) != ""):
        return row_unit
    return column_unit


def standard_value(value, unit: str):
    rule = STANDARD_UNITS.get(text_of(unit).lower().strip(" ."))
    if rule is None or value is None or pd.isna(value):
        return value, text_of(unit), ""
    return round(value * rule[1], 6), rule[0], rule[2]


def stand_id(row: dict) -> str:
    district = text_of(row.get("key_district_no")).strip(" .").upper()
    compartment = re.sub(r"\D", "", text_of(row.get("key_compartment_no")))
    sub = re.sub(r"[^a-zäöü]", "", text_of(row.get("key_subcompartment")).lower())
    if not ROMAN.match(district):
        return ""
    revier = revier_name(row.get("key_revier")) or revier_name(row.get("scope"))
    return "|".join([revier if revier in ("Schönau", "St. Oswald", "Klingenbrunn", "Schönberg") else revier,
                     district, compartment, sub])


def load(derived: Path, name: str) -> pd.DataFrame:
    path = derived / f"{name}.jsonl"
    return pd.read_json(path, lines=True) if path.exists() and path.stat().st_size else pd.DataFrame()


def cell_check_status(checks: pd.DataFrame) -> dict:
    status = {}
    rank = {"mismatch": 4, "mismatch_after_recheck": 4, "reading_confirmed": 3, "unparseable": 2,
            "ok_after_recheck": 1, "ok": 0}
    for check in checks.to_dict("records"):
        cells = [(p, t, r) for p, t, r in check["rows_used"]] + [(check["position"], check["table"], check["row"])]
        for position, table, row in cells:
            key = (check["unit"], position, table, row, check["column"])
            if rank.get(check["status"], 0) >= rank.get(status.get(key, "ok"), 0) or key not in status:
                status[key] = check["status"]
    return status


def observations(cells: pd.DataFrame, rows: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return cells
    key_columns = [c for c in rows.columns if c.startswith("key_")]
    merged = cells[(cells["variable"].fillna("") != "") & cells["value"].notna()].merge(
        rows[["unit", "position", "table", "row", "period", "box_2d", *key_columns]].rename(columns={"period": "row_period"}),
        on=["unit", "position", "table", "row"], how="left")
    merged = merged.drop_duplicates(["unit", "position", "table", "row", "canonical"])
    status = cell_check_status(checks) if not checks.empty else {}
    merged["check_status"] = [status.get((r.unit, r.position, r.table, r.row, r.canonical), "unchecked")
                              for r in merged.itertuples()]
    merged["period"] = merged["row_period"].fillna("").astype(str).where(merged["row_period"].fillna("") != "",
                                                                         merged["period"].fillna(""))
    row_units = merged["key_unit"] if "key_unit" in merged else pd.Series("", index=merged.index)
    merged["unit_of_measure"] = [effective_unit(r, c) for r, c in zip(row_units, merged["measure_unit"])]
    merged["scope"] = merged["scope"].fillna("")
    if "key_revier" in merged:
        merged["scope"] = merged["scope"].where(merged["scope"] != "", merged["key_revier"].fillna("").map(revier_name))
    merged["stand_id"] = [stand_id(r) for r in merged.to_dict("records")]
    standard = [standard_value(r["value"], r["unit_of_measure"]) for r in merged.to_dict("records")]
    merged["value_std"] = [v for v, _, _ in standard]
    merged["unit_std"] = [u for _, u, _ in standard]
    merged["conversion"] = [c for _, _, c in standard]
    columns = ["unit", "position", "page_id", "table", "row", "row_type", "variable", "value", "unit_of_measure",
               "value_std", "unit_std", "conversion",
               "period", "scope", "stand_id", *key_columns, "raw", "red", "uncertain", "flag", "check_status", "box_2d",
               "canonical", "header", "form", "page_role"]
    return merged[[c for c in columns if c in merged.columns]].rename(columns={"canonical": "column_id"})


def stands(rows: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    data = rows[(rows["row_type"] == "data") & (rows.get("page_role", pd.Series("", index=rows.index)) != "copy")].copy()
    scopes = cells[cells["scope"].fillna("") != ""].drop_duplicates(["unit", "position", "table", "row"])
    data = data.merge(scopes[["unit", "position", "table", "row", "scope"]], on=["unit", "position", "table", "row"],
                      how="left")
    data["stand_id"] = [stand_id(r) for r in data.to_dict("records")]
    data = data[data["stand_id"] != ""]
    area = cells[cells["variable"].fillna("").isin(["area", "stand_area"]) & cells["value"].notna()]
    area = area.drop_duplicates(["unit", "position", "table", "row"])[["unit", "position", "table", "row", "value",
                                                                      "measure_unit"]]
    data = data.merge(area.rename(columns={"value": "area", "measure_unit": "area_unit"}),
                      on=["unit", "position", "table", "row"], how="left")
    data["area_ha"] = [standard_value(a, u)[0] if text_of(u).lower() in ("tagwerk", "tgw", "ha", "hektar") else None
                       for a, u in zip(data["area"], data["area_unit"])]
    grouped = data.groupby("stand_id").agg(
        revier=("stand_id", lambda s: s.iloc[0].split("|")[0]),
        district_no=("stand_id", lambda s: s.iloc[0].split("|")[1]),
        compartment_no=("stand_id", lambda s: s.iloc[0].split("|")[2]),
        subcompartment=("stand_id", lambda s: s.iloc[0].split("|")[3]),
        district_names=("key_district_name", lambda s: "; ".join(sorted({str(v) for v in s.dropna() if str(v)}))
                        ) if "key_district_name" in data else ("stand_id", lambda s: ""),
        sources=("unit", lambda s: "; ".join(sorted(set(s)))),
        n_rows=("row", "count"),
        areas=("area", lambda s: "; ".join(f"{v:g}" for v in s.dropna())),
        area_units=("area_unit", lambda s: "; ".join(sorted({str(v) for v in s.dropna()}))),
        areas_ha=("area_ha", lambda s: "; ".join(f"{v:.3f}" for v in s.dropna())),
    ).reset_index()
    return grouped


def field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    return "string"


def write_resource(frame: pd.DataFrame, name: str, description: str, package: Path, resources: list):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].map(lambda v: isinstance(v, (list, dict))).any():
            frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    path = package / "data" / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    resources.append({"name": name, "path": f"data/{name}.csv", "format": "csv", "encoding": "utf-8",
                      "description": description, "count_of_rows": len(frame),
                      "schema": {"fields": [{"name": c, "type": field_type(frame[c]),
                                             **({"description": FIELD_DOCS[c]} if c in FIELD_DOCS else {})}
                                            for c in frame.columns]}})


def units_table(units_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    units, pages = [], []
    for unit_file in sorted(units_dir.glob("*/unit.json")):
        unit, spec = load_unit(unit_file.parent), load_spec(unit_file.parent)
        units.append({"unit": unit["id"], "heft": unit["heft"], "nr": unit["nr"], "title": unit["title"],
                      "summary": (spec.get("summary") or "").strip(), "default_profile": spec.get("default_profile", ""),
                      "pages": len(unit["pages"])})
        for page in unit["pages"]:
            pages.append({"unit": unit["id"], "position": page["position"], "page_id": page["pid"],
                          "volume": page["sig"], "page_number": page["num"], "kind": page["kind"],
                          "category": page["category"], "image_source": page["image_source"],
                          "image": f"{unit_file.parent.name}/{page['image']}" if page["image"] else "",
                          "image_size": page["image_size"]})
    return pd.DataFrame(units), pd.DataFrame(pages)


def spec_codebook(units_dir: Path) -> str:
    lines = []
    for unit_file in sorted(units_dir.glob("*/unit.json")):
        spec = load_spec(unit_file.parent)
        for name, form in (spec.get("forms") or {}).items():
            lines.append(f"\n### {spec['unit']} · {name}: {form['name']}\n")
            lines.append("| column | label | type | unit | variable |\n|---|---|---|---|---|")
            for column in form["columns"]:
                lines.append(f"| {column['id']} | {column['label']} | {column['type']} | {column.get('unit', '')} "
                             f"| {column.get('variable', '')} |")
    return "\n".join(lines)


def publish(units_dir: Path, run_dir: Path, package: Path, title: str, ledger: list[dict] | None = None) -> Path:
    derived = run_dir / "derived"
    (package / "data").mkdir(parents=True, exist_ok=True)
    for stale in [*package.glob("*.json"), *package.glob("*.md"), *(package / "data").glob("*")]:
        if stale.is_file():
            stale.unlink()
    rows, cells, checks = load(derived, "table_rows"), load(derived, "table_cells"), load(derived, "table_checks")
    resources = []
    units, pages = units_table(units_dir)
    write_resource(units, "units", "TOC items of the Operat with the spec summary", package, resources)
    write_resource(pages, "pages", "pages per unit with image reference", package, resources)
    if not rows.empty:
        write_resource(rows.drop(columns=["values", "decimals", "flags"], errors="ignore"), "table_rows",
                       "faithful layer: every table row with type, keys and approximate box", package, resources)
        write_resource(cells, "table_cells", "faithful layer: every table cell as written, normalised value, flags",
                       package, resources)
        write_resource(checks, "table_checks", "arithmetic checks (column totals, row sums, carry-overs) and results",
                       package, resources)
        write_resource(observations(cells, rows, checks), "observations",
                       "interpreted layer: tidy numeric observations with period, scope, stand and check status",
                       package, resources)
        write_resource(stands(rows, cells), "stands", "stand register from all table units", package, resources)
    for name, description in [("table_corrections", "table readings where the scan differs from the HTR transcript"),
                              ("table_outside_text", "titles, form numbers, marginalia, signatures around tables"),
                              ("stand_descriptions", "ecological decomposition of description cells"),
                              ("text_pages", "text pages: HTR transcript and proofread text"),
                              ("text_corrections", "text corrections from proofreading against the scan"),
                              ("text_statements", "atomic ecological statements from the text units"),
                              ("text_events", "dated disturbance events from the text units"),
                              ("maps", "map metadata and legends"), ("map_labels", "map labels with pixel boxes"),
                              ("map_symbols", "map symbols with pixel boxes")]:
        frame = load(derived, name)
        if not frame.empty:
            write_resource(frame, name, description, package, resources)
    if (derived / "map_labels.geojson").exists():
        shutil.copy(derived / "map_labels.geojson", package / "data" / "map_labels.geojson")
        resources.append({"name": "map_labels_geojson", "path": "data/map_labels.geojson", "format": "geojson",
                          "description": "map labels as polygons in pixel coordinates (y negated)"})
    spent = sum(entry.get("cost_usd", 0) for entry in ledger or [])
    descriptor = {"name": "forsteinrichtung-ilzertrift-1878-90", "title": title, "version": "0.1.0",
                  "created": date.today().isoformat(), "licenses": [{"name": "TO BE DECIDED"}],
                  "description": "Structured data extracted from the annotated parts of the Waldstandsrevision 1878/90 "
                                 "of the Ilzertrift-Komplex (Reviere Schönau, St. Oswald, Klingenbrunn; Forstamt "
                                 "Schönberg). Faithful layer (cells, proofread text, map labels) and interpreted layer "
                                 "(observations, stands, statements, events), each value with page provenance.",
                  "sources": [{"title": "Staatsarchiv Landshut / Regierung von Niederbayern, Kammer der Forsten, "
                                        "A 383 I – A 385"}],
                  "resources": resources}
    (package / "datapackage.json").write_text(json.dumps(descriptor, ensure_ascii=False, indent=1), encoding="utf-8")
    readme = [f"# {title}", "", descriptor["description"], "",
              "| resource | rows | content |", "|---|---|---|"]
    readme += [f"| {r['name']} | {r.get('count_of_rows', '')} | {r['description']} |" for r in resources]
    readme += ["", "## Validation", ""]
    if not checks.empty:
        readme += ["| status | checks |", "|---|---|"]
        readme += [f"| {status} | {count} |" for status, count in checks.groupby("status").size().items()]
    readme += ["", "## Form codebook (from the unit specs)", spec_codebook(units_dir), "",
               f"Extraction cost of this run: ${spent:.2f}."]
    (package / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return package
