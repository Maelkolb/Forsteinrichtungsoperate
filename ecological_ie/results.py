import json
import re
from pathlib import Path

import pandas as pd


def normalise(text) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def verify_quote(quote, text) -> str:
    if not quote:
        return "empty"
    if quote in text:
        return "exact"
    normal_text, normal_quote = normalise(text), normalise(quote)
    if normal_quote in normal_text:
        return "normalised"
    words = normal_quote.split()
    if len(words) >= 4 and " ".join(words[:6]) in normal_text:
        return "prefix"
    return "missing"


def joined(values) -> str:
    return "; ".join(values or [])


def text_rows(seq, record, text, tables):
    result = record["result"]
    tables["text_pages"].append({
        "seq": seq, "subtype": record["subtype"], "page_id": record["page_id"],
        "document_type": result.get("document_type"), "summary_en": result.get("summary_en"),
        "transcription_quality": result.get("transcription_quality"),
        "forest_offices": joined(result.get("forest_offices")), "dates": joined(result.get("dates_mentioned")),
        "n_findings": len(result.get("findings", []))})
    for i, finding in enumerate(result.get("findings", [])):
        tables["text_findings"].append({
            "seq": seq, "source_subtype": record["subtype"], "page_id": record["page_id"], "idx": i,
            "category": finding.get("category"), "finding_subtype": finding.get("subtype"),
            **{key: finding.get(key) for key in ("entity", "entity_en", "value", "unit", "date", "location",
                                                 "status", "confidence", "note")},
            "species": joined(finding.get("species")), "quote": finding.get("quote"),
            "quote_check": verify_quote(finding.get("quote"), text)})
    for species in result.get("tree_species", []):
        tables["text_species"].append({"seq": seq, "page_id": record["page_id"], **species,
                                       "quote_check": verify_quote(species.get("quote"), text)})
    for location in result.get("locations", []):
        tables["text_locations"].append({"seq": seq, "page_id": record["page_id"], **location,
                                         "quote_check": verify_quote(location.get("quote"), text)})


def species_label(species) -> str:
    share = species.get("share_value")
    return f"{species.get('name_de')}{'' if share is None else ' ' + str(share)}"


def table_rows(seq, record, text, tables):
    result = record["result"]
    tables["table_pages"].append({
        "seq": seq, "subtype": record["subtype"], "page_id": record["page_id"],
        "table_type": result.get("table_type"), "forest_office": result.get("forest_office"),
        "operating_class": result.get("operating_class"), "n_records": len(result.get("records", [])),
        "transcription_quality": result.get("transcription_quality"), "parsing_notes": result.get("parsing_notes"),
        "column_headers": " | ".join(result.get("column_headers", []) or [])})
    for i, rec in enumerate(result.get("records", [])):
        key = {"seq": seq, "subtype": record["subtype"], "page_id": record["page_id"], "rec": i,
               "record_kind": rec.get("record_kind"), "district_no": rec.get("district_no"),
               "district_name": rec.get("district_name"), "compartment_no": rec.get("compartment_no"),
               "subcompartment": rec.get("subcompartment")}
        row = dict(key, stand_name=rec.get("stand_name"), forest_office=rec.get("forest_office"),
                   year_or_period=rec.get("year_or_period"), area_value=rec.get("area_value"),
                   area_unit=rec.get("area_unit"), area_raw=rec.get("area_raw"))
        for name, value in (rec.get("site") or {}).items():
            row["site_" + name] = joined(value) if isinstance(value, list) else value
        stand = rec.get("stand") or {}
        for name, value in stand.items():
            if name == "species":
                row["stand_species"] = "; ".join(species_label(species) for species in value)
            else:
                row["stand_" + name] = value
        for name, value in (rec.get("management") or {}).items():
            if name == "history":
                tables["table_history"].extend(dict(key, **entry) for entry in value)
            else:
                row["mgmt_" + name] = value
        row["n_damage"] = len(rec.get("damage") or [])
        row["n_culture"] = len(rec.get("culture") or [])
        row["n_ntu"] = len(rec.get("non_timber_use") or [])
        row["red_annotations"] = joined(rec.get("red_annotations"))
        row["cross_references"] = joined(rec.get("cross_references"))
        row["uncertain"] = rec.get("uncertain")
        row["description_raw"] = rec.get("description_raw")
        row["description_check"] = (verify_quote(rec["description_raw"][:120], text)
                                    if rec.get("description_raw") else "")
        tables["table_records"].append(row)
        tables["table_species"].extend(dict(key, **species) for species in stand.get("species", []) or [])
        tables["table_damage"].extend(dict(key, **damage) for damage in rec.get("damage") or [])
        tables["table_culture"].extend(dict(key, **culture) for culture in rec.get("culture") or [])
        tables["table_non_timber_use"].extend(dict(key, **use) for use in rec.get("non_timber_use") or [])


TABLE_NAMES = ["text_pages", "text_findings", "text_species", "text_locations", "table_pages", "table_records",
               "table_species", "table_damage", "table_culture", "table_non_timber_use", "table_history",
               "image_corrections"]


def flatten(results: dict, pages_by_seq: dict) -> dict[str, pd.DataFrame]:
    tables = {name: [] for name in TABLE_NAMES}
    for seq, record in results.items():
        text = pages_by_seq[seq]["text"]
        if record["source_type"] == "text":
            text_rows(seq, record, text, tables)
        else:
            table_rows(seq, record, text, tables)
        for correction in record["result"].get("image_corrections", []) or []:
            tables["image_corrections"].append({"seq": seq, "page_id": record["page_id"], **correction})
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def write_tables(tables: dict[str, pd.DataFrame], results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name:22s} {len(frame):5d} rows")
    (out_dir / "all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def overview(tables: dict[str, pd.DataFrame]) -> str:
    parts = []
    findings, records = tables["text_findings"], tables["table_records"]
    if len(findings):
        parts += ["findings per subtype and category:", str(pd.crosstab(findings["source_subtype"], findings["category"])),
                  "quote verification:", str(findings["quote_check"].value_counts()),
                  "confidence:", str(findings["confidence"].value_counts()),
                  "most frequent entities:", str(findings["entity"].value_counts().head(25))]
    if len(records):
        parts += ["records per table page / kind:", str(pd.crosstab(records["subtype"], records["record_kind"])),
                  "description_raw verification:", str(records["description_check"].value_counts())]
        if len(tables["table_species"]):
            parts += ["species in tables:", str(tables["table_species"]["name_de"].value_counts().head(15))]
    if len(tables["image_corrections"]):
        parts += [f"image corrections: {len(tables['image_corrections'])}"]
    return "\n".join(parts)


def scoring_sheet(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    findings, records = tables["text_findings"], tables["table_records"]
    if len(findings):
        parts.append(findings[["seq", "page_id", "idx", "category", "entity", "value", "date", "location", "confidence",
                               "quote_check", "quote"]].assign(kind="text_finding", verdict="", comment=""))
    if len(records):
        columns = ["seq", "page_id", "rec", "record_kind", "district_no", "compartment_no", "subcompartment",
                   "area_value", "stand_species", "stand_age_raw", "uncertain", "description_check"]
        parts.append(records.reindex(columns=columns).rename(columns={"rec": "idx"})
                     .assign(kind="table_record", verdict="", comment=""))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
