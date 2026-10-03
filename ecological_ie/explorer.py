import json
import math
import re
from datetime import date
from pathlib import Path

import pandas as pd

from .publish import revier_name, text_of
from .reader import reader_pages, table_of_contents
from .spec import load_spec, load_unit

TEMPLATE = Path(__file__).parent / "explorer_template.html"
REVIERE = ["Schönau", "St. Oswald", "Klingenbrunn"]
OBS_COLUMNS = ["unit", "position", "table", "row", "variable", "value", "value_std", "unit_std", "period", "scope",
               "stand_id", "check_status", "raw", "box_2d"]


def clean(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_csv(package: Path, name: str) -> pd.DataFrame:
    path = package / "data" / f"{name}.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def period_order(periods: list[str]) -> list[str]:
    def key(period: str):
        years = [int(y) for y in re.findall(r"1[89]\d\d", period)]
        return (years[0] if years else 9999, period)
    return sorted(set(periods), key=key)


def units_and_pages(units_dir: Path, checks: pd.DataFrame) -> tuple[list, dict]:
    units, pages = [], {}
    status = checks.groupby(["unit", "status"]).size().unstack(fill_value=0) if not checks.empty else pd.DataFrame()
    for unit_file in sorted(units_dir.glob("*/unit.json")):
        unit, spec = load_unit(unit_file.parent), load_spec(unit_file.parent)
        kinds = {kind: sum(p["kind"] == kind for p in unit["pages"]) for kind in ("Te", "Ta", "Ka")}
        units.append({"id": unit["id"], "heft": unit["heft"], "nr": unit["nr"], "title": unit["title"],
                      "summary": (spec.get("summary") or "").strip(), "pages": len(unit["pages"]), "kinds": kinds,
                      "checks": {k: int(v) for k, v in status.loc[unit["id"]].items() if v} if unit["id"] in status.index else {}})
        for page in unit["pages"]:
            size = page.get("image_size") or [0, 0]
            pages[f"{unit['id']}/{page['position']}"] = [
                page["pid"], f"{unit_file.parent.name}/{page['image']}" if page["image"] else "", size[0], size[1],
                page["kind"], page["sig"], page["num"]]
    return units, pages


def observation_table(observations: pd.DataFrame) -> tuple[dict, dict]:
    frame = observations[observations["page_role"].fillna("") != "copy"] if "page_role" in observations else observations
    frame = frame[OBS_COLUMNS].reset_index(drop=True)
    rows = []
    for record in frame.itertuples(index=False):
        box = json.loads(record.box_2d) if isinstance(record.box_2d, str) and record.box_2d.startswith("[") else None
        rows.append([record.unit, int(record.position), int(record.table), int(record.row), record.variable,
                     clean(record.value), clean(round(record.value_std, 4) if pd.notna(record.value_std) else None),
                     text_of(record.unit_std), text_of(record.period), text_of(record.scope), text_of(record.stand_id),
                     record.check_status, text_of(record.raw), box])
    index = {}
    for i, row in enumerate(rows):
        index.setdefault((row[0], row[4], row[8], row[9]), i)
    return {"cols": ["u", "p", "t", "r", "var", "v", "vs", "us", "per", "sc", "st", "cs", "raw", "box"], "rows": rows}, index


def series_points(observations: pd.DataFrame, obs_index: dict, unit: str, variables: dict, by_scope: bool) -> dict:
    frame = observations[(observations["unit"] == unit) & observations["variable"].isin(variables)
                         & (observations["row_type"] == "data")]
    periods = period_order([text_of(p) for p in frame["period"] if text_of(p)])
    groups = []
    keys = REVIERE if by_scope else list(variables)
    for key in keys:
        points = []
        for period in periods:
            if by_scope:
                subset = frame[(frame["scope"].map(revier_name) == key) & (frame["period"] == period)]
            else:
                subset = frame[(frame["variable"] == key) & (frame["period"] == period)]
            if subset.empty:
                points.append(None)
                continue
            value = subset["value_std"].fillna(subset["value"]).sum()
            refs = [obs_index.get((unit, v, period, text_of(s))) for v, s in zip(subset["variable"], subset["scope"])]
            statuses = sorted(set(subset["check_status"]))
            points.append({"v": round(float(value), 3), "refs": [r for r in refs if r is not None], "cs": statuses})
        groups.append({"key": key, "label": key if by_scope else variables[key], "points": points})
    return {"x": periods, "groups": groups}


def harvest_series(observations: pd.DataFrame, cells: pd.DataFrame, rows: pd.DataFrame, obs_index: dict) -> dict:
    series = series_points(observations, obs_index, "I-16",
                           {"harvest_main_use": "Hauptnutzung", "harvest_intermediate_use": "Zwischennutzung"}, False)
    notes = rows[(rows["unit"] == "I-16") & (rows["row_type"] == "note")]
    etat = overcut = None
    for note in notes.itertuples():
        value = cells[(cells["unit"] == "I-16") & (cells["row"] == note.row) & (cells["canonical"] == "total_use")]["value"]
        if not value.empty and pd.notna(value.iloc[0]):
            if "Etat" in text_of(note.label) or "Etat" in " ".join(map(str, cells[(cells["unit"] == "I-16") & (cells["row"] == note.row)]["raw"])):
                etat = float(value.iloc[0])
            elif "Mehr" in " ".join(map(str, cells[(cells["unit"] == "I-16") & (cells["row"] == note.row)]["raw"])):
                overcut = float(value.iloc[0])
    total = observations[(observations["unit"] == "I-16") & (observations["row_type"] == "sum")
                         & (observations["variable"] == "harvest_total")]["value_std"].max()
    series.update({"id": "harvest", "unit": "Ster", "source": "I-16",
                   "title": "Harvest of the complex, 1860/61–1877",
                   "note": "Klafter values until 1871 converted with the source's own factor 3.1325 Ster per Klafter.",
                   "etat": etat, "overcut": overcut, "total": clean(float(total)) if pd.notna(total) else None,
                   "years": len(series["x"])})
    return series


def scope_series(observations, obs_index, unit, variables, title, unit_label, series_id, note="") -> dict:
    series = series_points(observations, obs_index, unit, variables, True)
    variants = {"all": series["groups"]}
    for variable in variables:
        single = series_points(observations, obs_index, unit, {variable: variables[variable]}, True)
        variants[variable] = [{**group, "points": [single_point for single_point in group["points"]]}
                              for group in align_groups(single, series["x"])]
    series.update({"id": series_id, "title": title, "unit": unit_label, "source": unit, "note": note,
                   "variables": variables, "variants": variants})
    return series


def align_groups(series: dict, x: list) -> list:
    position = {period: i for i, period in enumerate(series["x"])}
    return [{**group, "points": [group["points"][position[period]] if period in position else None for period in x]}
            for group in series["groups"]]


def variable_series(observations, obs_index, unit, variables, title, unit_label, series_id, note="") -> dict:
    series = series_points(observations, obs_index, unit, variables, False)
    series.update({"id": series_id, "title": title, "unit": unit_label, "source": unit, "note": note})
    return series


def combine_groups(series: dict, combined: dict) -> dict:
    by_key = {group["key"]: group for group in series["groups"]}
    groups = []
    for key, (label, members) in combined.items():
        points = []
        for i in range(len(series["x"])):
            parts = [(by_key[m]["label"], by_key[m]["points"][i]) for m in members if by_key[m]["points"][i]]
            if not parts:
                points.append(None)
                continue
            points.append({"v": round(sum(p["v"] for _, p in parts), 3),
                           "refs": [r for _, p in parts for r in p["refs"]],
                           "cs": sorted({c for _, p in parts for c in p["cs"]}),
                           "parts": [[name, p["v"]] for name, p in parts]})
        groups.append({"key": key, "label": label, "points": points})
    return {**series, "groups": groups}


def stands_table(stands: pd.DataFrame, observations: pd.DataFrame, descriptions: list, obs_index_by_row: dict) -> list:
    by_stand = {}
    for i, row in enumerate(obs_index_by_row["rows"]):
        if row[10]:
            by_stand.setdefault(row[10], []).append(i)
    desc_by_stand = {}
    for d, item in enumerate(descriptions):
        if item["stand"]:
            desc_by_stand.setdefault(item["stand"], []).append(d)
    result = []
    for record in stands.itertuples():
        result.append({"id": record.stand_id, "rev": text_of(record.revier), "d": text_of(record.district_no),
                       "c": text_of(record.compartment_no), "s": text_of(record.subcompartment),
                       "names": text_of(record.district_names), "src": text_of(record.sources),
                       "obs": by_stand.get(record.stand_id, []), "desc": desc_by_stand.get(record.stand_id, [])})
    return result


def description_items(descriptions: pd.DataFrame, observations: pd.DataFrame) -> list:
    stand_by_row = {}
    for r in observations.itertuples():
        if text_of(r.stand_id):
            stand_by_row.setdefault((r.unit, int(r.position), int(r.table), int(r.row)), text_of(r.stand_id))
    items = []
    for d in descriptions.itertuples():
        if d.not_ecological is True or text_of(d.not_ecological).lower() == "true":
            continue
        key = (d.unit, int(d.position), int(d.table), int(d.row)) if pd.notna(d.position) else None
        parts = {}
        for part in ("site", "stand", "damage", "management", "culture", "non_timber_use", "events"):
            raw = text_of(getattr(d, part))
            if raw:
                try:
                    parts[part] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        items.append({"u": d.unit, "p": int(d.position) if pd.notna(d.position) else None,
                      "r": int(d.row) if pd.notna(d.row) else None, "stand": stand_by_row.get(key, ""),
                      "text": text_of(d.cell_text), "parts": parts, "ev": text_of(d.evidence)})
    return items


def statements_list(statements: pd.DataFrame) -> list:
    out = []
    for s in statements.itertuples():
        page = text_of(s.page)
        out.append({"u": s.unit, "cat": text_of(s.category), "subj": text_of(s.subject), "attr": text_of(s.attribute),
                    "val": text_of(s.value), "vu": text_of(getattr(s, "value_unit", "")), "time": text_of(s.time_text),
                    "edtf": text_of(s.time_edtf), "place": text_of(s.place_text), "rev": revier_name(s.revier),
                    "status": text_of(s.status), "quote": text_of(s.quote),
                    "p": int(page[1:]) if page[1:].isdigit() else None, "qc": text_of(s.quote_check),
                    "conf": text_of(s.confidence), "species": text_of(s.species)})
    return out


def events_list(events: pd.DataFrame) -> list:
    out = []
    for e in events.itertuples():
        page = text_of(e.page)
        out.append({"u": e.unit, "type": text_of(e.type), "date": text_of(e.date_text), "edtf": text_of(e.date_edtf),
                    "place": text_of(e.place_text), "rev": revier_name(e.revier), "extent": text_of(e.extent),
                    "quote": text_of(e.quote), "p": int(page[1:]) if page[1:].isdigit() else None,
                    "qc": text_of(e.quote_check)})
    return out


def maps_list(maps: pd.DataFrame, labels: pd.DataFrame) -> list:
    out = []
    for m in maps.itertuples():
        page_labels = labels[(labels["unit"] == m.unit) & (labels["position"] == m.position)]
        legend = json.loads(m.legend) if text_of(m.legend).startswith("[") else []
        out.append({"u": m.unit, "p": int(m.position), "title": text_of(m.title), "type": text_of(m.map_type),
                    "date": text_of(m.date_text), "scale": text_of(m.scale_text), "area": text_of(m.area),
                    "legend": [[text_of(i.get("symbol")), text_of(i.get("meaning"))] for i in legend],
                    "labels": [[text_of(l.text), text_of(l.label_class), int(l.x0), int(l.y0), int(l.x1), int(l.y1),
                                text_of(l.confidence)]
                               for l in page_labels.itertuples()],
                    "w": int(page_labels["image_width"].iloc[0]) if not page_labels.empty else 0,
                    "h": int(page_labels["image_height"].iloc[0]) if not page_labels.empty else 0})
    return out


def build_data(units_dir: Path, run_dir: Path, package: Path) -> dict:
    observations = read_csv(package, "observations")
    cells, rows, checks = read_csv(package, "table_cells"), read_csv(package, "table_rows"), read_csv(package, "table_checks")
    observations_all = observations.merge(rows[["unit", "position", "table", "row", "row_type"]].drop_duplicates(),
                                          on=["unit", "position", "table", "row"], how="left", suffixes=("", "_r"))
    if "row_type_r" in observations_all:
        observations_all["row_type"] = observations_all["row_type"].fillna(observations_all["row_type_r"])
    units, pages = units_and_pages(units_dir, checks)
    obs, obs_index = observation_table(observations)
    data_obs = observations_all[observations_all["page_role"].fillna("") != "copy"]
    labels = read_csv(package, "map_labels")
    labels = labels.rename(columns={"class": "label_class"})
    series = [
        harvest_series(data_obs, cells, rows, obs_index),
        scope_series(data_obs, obs_index, "I-20", {"grazing_cows": "Kühe", "grazing_oxen": "Ochsen",
                                                   "grazing_young_cattle": "Jungrinder"},
                     "Cattle grazing in the forest", "head of cattle", "grazing",
                     "Cows, oxen and young cattle driven into the state forest per year (Weidenschaft)."),
        scope_series(data_obs, obs_index, "I-20", {"litter_leaf": "Laubstreu", "litter_needle": "Nadelstreu",
                                                   "litter_other": "other litter"},
                     "Litter raking", "Fuder", "litter",
                     "Leaf, needle and other litter taken from the forest per year (Streunutzung), in Fuder of 144 c'."),
        combine_groups(variable_series(data_obs, obs_index, "I-18", {"conifer_planting_area": "planting",
                                                                     "conifer_sowing_area": "sowing",
                                                                     "broadleaf_planting_area": "planting",
                                                                     "broadleaf_sowing_area": "sowing"},
                                       "New cultures", "ha", "cultures",
                                       "Area of new plantings and sowings per year (Forstculturen, Titel 4); "
                                       "Tagwerk converted at 0.3407 ha."),
                       {"conifer": ("Nadelholz", ["conifer_planting_area", "conifer_sowing_area"]),
                        "broadleaf": ("Laubholz", ["broadleaf_planting_area", "broadleaf_sowing_area"])}),
        variable_series(data_obs, obs_index, "I-19", {"rafting_structures_cost_total": "Trift works"},
                        "Cost of the timber-floating works", "M", "trift",
                        "Total cash expense of the state for chutes, yards, stream clearing, bank protection and "
                        "splash dams (Gulden converted at 12/7 Mark)."),
    ]
    descriptions = description_items(read_csv(package, "stand_descriptions"), data_obs)
    statements = read_csv(package, "text_statements")
    events = read_csv(package, "text_events")
    maps = read_csv(package, "maps")
    status_counts = checks.groupby("status").size().to_dict() if not checks.empty else {}
    statement_items, event_items = statements_list(statements), events_list(events)
    reader, forms = reader_pages(units_dir, run_dir, checks, statement_items, event_items)
    return {
        "meta": {"title": "Ilzertrift-Komplex 1878/90", "built": date.today().isoformat(),
                 "pages": sum(u["pages"] for u in units), "units": len(units), "checks": status_counts,
                 "counts": {"cells": len(cells), "observations": len(obs["rows"]), "stands": len(read_csv(package, "stands")),
                            "statements": len(statements), "events": len(events), "map_labels": len(labels)}},
        "units": units, "pages": pages, "obs": obs, "series": series,
        "stands": stands_table(read_csv(package, "stands"), data_obs, descriptions, obs),
        "descs": descriptions, "statements": statement_items, "events": event_items,
        "maps": maps_list(maps, labels), "reader": reader, "forms": forms, "toc": table_of_contents(units_dir, units),
    }


STANDALONE_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   '</head>\n<body>\n')
STANDALONE_TAIL = "\n</body>\n</html>\n"


def build_explorer(units_dir: Path, run_dir: Path, package: Path, out_file: Path, image_base: str,
                   package_link: str = "", standalone: bool = True) -> Path:
    data = build_data(units_dir, run_dir, package)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__IMAGE_BASE__", image_base).replace("__PACKAGE_LINK__", package_link)
    html = html.replace("__DATA__", payload)
    if standalone:
        html = STANDALONE_HEAD + html + STANDALONE_TAIL
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
    return out_file
