import json
import time
from pathlib import Path

import pandas as pd

from .gemini import Gemini
from .prompts import GLOSSARY
from .schemas import RECORD_SCHEMA

PARTS = ["site", "stand", "damage", "management", "culture", "non_timber_use"]

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "ref": {"type": "string", "description": "the ref of the cell as given"},
            **{part: RECORD_SCHEMA["properties"][part] for part in PARTS},
            "events": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string"}, "date_text": {"type": "string"}, "extent": {"type": "string"},
                "quote": {"type": "string"}}, "required": ["type", "quote"]}},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "verbatim snippets of the cell supporting the extracted values"},
            "not_ecological": {"type": "boolean"}},
            "required": ["ref", "evidence"]}},
    },
    "required": ["items"],
}

DESCRIBE_PROMPT = '''You are an expert in historical forestry. Below are free-text cells from tables of the Waldstandsrevision
1878/90 of the Ilzertrift-Komplex (Bavarian Forest). Each cell describes a stand or area (Lage, Boden, Holzbestand,
Bemerkungen …). Decompose every cell into structured ecological attributes:
site (elevation, exposure, slope, bedrock, soil, moisture, ground vegetation, frost exposure), stand (species with
shares as 0-1 fractions, age, stocking, structure, health, regeneration, volume, increment, stem counts), damage,
management (cutting method, period, notes), culture (planting, sowing, drainage), non-timber use, and dated events.
Do not invent anything; leave fields empty when the cell does not say it. "evidence" holds verbatim snippets of the cell.
Mark administrative cells "not_ecological". Return one item per ref.

{glossary}

CELLS (ref | stand key | column | text):
{cells}
'''


def describe_cells(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return cells
    keys = [c for c in cells.columns if c.startswith("key_")]
    frame = cells.copy()
    frame["ref"] = (frame["unit"] + "/p" + frame["position"].astype(int).map("{:03d}".format) + "/t" +
                    frame["table"].astype(str) + "/r" + frame["row"].astype(str) + "/c" + frame["col"].astype(str))
    frame["stand_key"] = frame[keys].fillna("").astype(str).agg(" | ".join, axis=1) if keys else ""
    return frame


def run_describe(units: list[str], cells: pd.DataFrame, rows: pd.DataFrame, describe_columns: dict, gemini: Gemini,
                 out_dir: Path, batch: int = 25, force: bool = False) -> list[dict]:
    from .pipeline import run_parallel
    key_columns = [c for c in rows.columns if c.startswith("key_")]
    merged = cells.merge(rows[["unit", "position", "table", "row", *key_columns]], on=["unit", "position", "table", "row"],
                         how="left")
    merged = merged[merged.apply(lambda r: r["canonical"] in describe_columns.get((r["unit"], r["form"]), set()), axis=1)]
    merged = merged[(merged["row_type"] == "data") & merged["raw"].fillna("").str.strip().str.len().gt(3)]
    frame = describe_cells(merged)
    tasks = []
    for unit in units:
        unit_cells = frame[frame.unit == unit]
        for start in range(0, len(unit_cells), batch):
            chunk = unit_cells.iloc[start:start + batch]
            path = out_dir / unit / f"batch_{start // batch:03d}.json"
            tasks.append((f"{unit} batch {start // batch}",
                          lambda chunk=chunk, path=path: describe_batch(chunk, gemini, path, force)))
    results, _ = run_parallel(tasks, gemini.settings.workers, "description cells")
    return results


def describe_batch(chunk: pd.DataFrame, gemini: Gemini, path: Path, force: bool) -> dict:
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    listing = "\n".join(f"{row.ref} | {row.stand_key} | {row.header} | {row.raw}" for row in chunk.itertuples())
    started = time.time()
    data, usage, mode = gemini.extract(DESCRIBE_PROMPT.format(glossary=GLOSSARY, cells=listing), DESCRIBE_SCHEMA,
                                       thinking="low")
    record = {"refs": list(chunk.ref), "texts": dict(zip(chunk.ref, chunk.raw)), "usage": usage, "mode": mode,
              "elapsed_s": round(time.time() - started, 1), "result": data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record
