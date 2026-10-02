import json
import time
from pathlib import Path

from PIL import Image

from .gemini import Gemini
from .pages import clean_body, parse_md
from .prompts import GLOSSARY
from .schemas import IMAGE_CORRECTIONS
from .spec import KEY_NAMES, PagePlan, spec_summary_for_prompt

ROW_TYPES = ["data", "group_header", "sum", "carry_over", "heading", "note", "empty"]
OUTSIDE_KINDS = ["heading", "form_number", "page_number", "marginalia", "signature", "stamp", "note", "other"]

GRID_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {"type": "array", "items": {"type": "object", "properties": {
            "form_matches": {"type": "boolean", "description": "the table follows the given form"},
            "layout_notes": {"type": "string", "description": "deviations: missing/extra columns, two blocks side by side, rotated parts"},
            "columns": {"type": "array", "items": {"type": "object", "properties": {
                "header": {"type": "string", "description": "header path as printed, levels joined with ' › '"},
                "canonical": {"type": "string", "description": "id of the matching canonical column, or 'other'"},
                "x_range": {"type": "array", "items": {"type": "integer"},
                            "description": "[xmin, xmax] of the column in image 1, 0-1000"}},
                "required": ["header", "canonical", "x_range"]}},
            "rows": {"type": "array", "items": {"type": "object", "properties": {
                "row_type": {"type": "string", "enum": ROW_TYPES},
                "box_2d": {"type": "array", "items": {"type": "integer"},
                           "description": "[ymin, xmin, ymax, xmax] of the row in image 1, 0-1000"},
                "cells": {"type": "array", "items": {"type": "string"},
                          "description": "one string per column, same order and count as 'columns'; '' if empty"},
                "red": {"type": "array", "items": {"type": "object", "properties": {
                    "col": {"type": "integer"}, "text": {"type": "string"}}, "required": ["col", "text"]},
                    "description": "red-ink content per column index"},
                "uncertain": {"type": "array", "items": {"type": "integer"},
                              "description": "column indexes whose reading is uncertain"},
                "sets_key": {"type": "object", "properties": {
                    "key": {"type": "string", "enum": KEY_NAMES}, "value": {"type": "string"}},
                    "description": "for group_header rows: the key this heading sets, e.g. revier = Klingenbrunn"},
                "note": {"type": "string"}},
                "required": ["row_type", "box_2d", "cells"]}}},
            "required": ["columns", "rows"]}},
        "text_outside_tables": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": OUTSIDE_KINDS}, "text": {"type": "string"}},
            "required": ["kind", "text"]}},
        "image_corrections": IMAGE_CORRECTIONS,
        "transcription_quality": {"type": "string", "enum": ["good", "mixed", "poor"]},
    },
    "required": ["tables", "image_corrections", "transcription_quality"],
}

GRID_PROMPT = '''You are an expert in 19th-century Bavarian forestry records and in reading handwritten tables.
Reconstruct the table(s) on this page as a faithful grid.

INPUT
- Image 1 is the whole page. {crop_description}
- Below is an automatic HTR transcription of the page (HTML table). It is a draft: rows, numbers, column alignment and
  headers can be wrong or missing. The scan decides.
- The unit description and the canonical columns of the form tell you what the columns mean.

RULES
1. Columns: list the page's physical columns from left to right as they appear on the scan, down to the lowest header
   level (a value split into Tagw. | Dez., Hekt. | Ar, fl. | kr., M. | Pf. is TWO columns). Give the header path as
   printed (levels joined with " › "), its x_range in image 1 (0-1000) and map each column to a canonical column id,
   or "other" if none fits.
2. Rows: every row from top to bottom, including headings inside the table, group headers ("Revier Klingenbrunn",
   "III. Rachelhang"), sums (Summa, Summe, Zusammen), carry-overs (Übertrag, Transport, "Seite 2") and empty ruled rows
   only if they carry text. Give row_type and the row's box_2d in image 1 coordinates (0-1000).
3. Cells: exactly one string per column, in column order. Copy what is written: digits, decimal commas, fractions (½),
   dashes (—), ditto marks (" „ do. dto.), abbreviations and spelling as on the page. Do not compute, convert, complete
   or correct the historical writing. Empty cell = "".
4. Red ink: put red content into "red" (column index + text) and leave the cell for the black ink ("" if only red).
5. Mark doubtful readings in "uncertain". Never guess silently.
6. Rows that name a Revier, Distrikt, Abteilung, Betriebsklasse, Altersklasse or Sortiment for the rows below
   (group headers) fill "sets_key". Rows that change the unit for the rows below or carry a converted value
   ("Klafter.", "= Ster", "M. Pf.", "Hektar") fill "sets_key" with key "unit" (e.g. Ster, M, ha).
7. Where your reading differs from the transcription for a number, key or name, add an entry to "image_corrections".
8. Text outside the table (title, form number, page number, marginal notes, signatures, stamps) goes into
   "text_outside_tables".
9. If a page holds two tables or two blocks side by side that do not share rows, return them as separate tables.

{glossary}
{unit_context}

PAGE ID: {page_id} (page {position} of the unit, role: {role})

=== HTR TRANSCRIPTION (draft) ===
{transcript}
=== END TRANSCRIPTION ===
'''


def page_crops(image: Image.Image, overlap: float = 0.08) -> tuple[list[Image.Image], str]:
    width, height = image.size
    if width >= height:
        cut = int(width * (0.5 + overlap / 2)), int(width * (0.5 - overlap / 2))
        crops = [image.crop((0, 0, cut[0], height)), image.crop((cut[1], 0, width, height))]
        description = (f"Images 2 and 3 are zoomed crops of the left part (x 0–{50 + overlap * 50:.0f} %) and the right "
                       f"part (x {50 - overlap * 50:.0f}–100 %) of the same page, for reading small digits.")
    else:
        cut = int(height * (0.5 + overlap / 2)), int(height * (0.5 - overlap / 2))
        crops = [image.crop((0, 0, width, cut[0])), image.crop((0, cut[1], width, height))]
        description = (f"Images 2 and 3 are zoomed crops of the upper part (y 0–{50 + overlap * 50:.0f} %) and the lower "
                       f"part (y {50 - overlap * 50:.0f}–100 %) of the same page, for reading small digits.")
    return crops, description


def grid_prompt(plan: PagePlan, spec: dict, transcript: str, crop_description: str) -> str:
    return GRID_PROMPT.format(crop_description=crop_description, glossary=GLOSSARY,
                              unit_context=spec_summary_for_prompt(spec, plan.form), page_id=plan.page["pid"],
                              position=plan.position, role=plan.role, transcript=transcript)


def read_grid(plan: PagePlan, spec: dict, unit_dir: Path, gemini: Gemini, out_path: Path, force: bool = False,
              model: str | None = None, thinking: str | None = None, hint: str = "") -> dict:
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    _, body = parse_md((unit_dir / plan.page["transcript"]).read_text(encoding="utf-8"))
    transcript = clean_body(body, "table")
    images, crop_description = [], "There is no image of this page; rely on the transcription."
    if plan.page["image"]:
        page_image = Image.open(unit_dir / plan.page["image"])
        crops, crop_description = page_crops(page_image)
        images = [page_image, *crops]
    started = time.time()
    prompt = grid_prompt(plan, spec, transcript, crop_description)
    if hint:
        prompt += f"\nNOTE: {hint}\n"
    data, usage, mode = gemini.extract(prompt, GRID_SCHEMA, images=images, model=model, thinking=thinking)
    record = {"unit": spec["unit"], "position": plan.position, "page_id": plan.page["pid"], "form": plan.form,
              "role": plan.role, "image": plan.page["image"], "image_size": plan.page.get("image_size"),
              "usage": usage, "mode": mode, "thinking": thinking or gemini.settings.thinking_level, "hint": hint,
              "elapsed_s": round(time.time() - started, 1), "result": data}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record
