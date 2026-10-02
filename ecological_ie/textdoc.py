import json
import re
import time
from pathlib import Path

from .gemini import Gemini
from .pages import clean_body, parse_md
from .prompts import GLOSSARY
from .schemas import CATEGORIES
from .spec import PagePlan

CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}

PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {"type": "array", "items": {"type": "object", "properties": {
            "transcript_reads": {"type": "string", "description": "exact substring of the transcription (5-60 chars)"},
            "image_reads": {"type": "string", "description": "the same passage as written on the scan"},
            "kind": {"type": "string", "enum": ["word", "number", "name", "omission", "insertion", "punctuation", "other"]},
            "confidence": CONFIDENCE},
            "required": ["transcript_reads", "image_reads", "kind", "confidence"]}},
        "missing_text": {"type": "array", "items": {"type": "object", "properties": {
            "after": {"type": "string", "description": "exact transcription substring after which the text is missing"},
            "text": {"type": "string"}}, "required": ["after", "text"]}},
        "page_quality": {"type": "string", "enum": ["good", "mixed", "poor"]},
        "continues_from_previous_page": {"type": "boolean"},
        "continues_on_next_page": {"type": "boolean"},
    },
    "required": ["corrections", "page_quality"],
}

PROOF_PROMPT = '''You proofread an automatic HTR transcription of a handwritten (Kurrent) page from a 19th-century Bavarian
forest management plan against the scan of the page.

List only real misreadings: wrong words, names, numbers, dates, missing or invented words and lines. Keep the historical
spelling, abbreviations and line breaks of the original; do not modernise, do not correct the clerk's own errors, do not
touch Markdown formatting. "transcript_reads" must be copied exactly from the transcription below (long enough to be
unique, 5–60 characters); "image_reads" is the same passage as it is written on the scan. Text that is on the scan but
missing in the transcription goes into "missing_text". If the transcription is correct, return empty lists.

PAGE ID: {page_id}
=== TRANSCRIPTION ===
{text}
=== END ===
'''

STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "statements": {"type": "array", "items": {"type": "object", "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "subject": {"type": "string", "description": "normalised German headword (Fichte, Windwurf, Streunutzung, Hochwald …)"},
            "subject_en": {"type": "string"},
            "attribute": {"type": "string", "description": "what is stated about the subject (Anteil, Höhenlage, Schaden, Verjüngung …)"},
            "value": {"type": "string", "description": "value as written"},
            "unit": {"type": "string"},
            "species": {"type": "array", "items": {"type": "string"}},
            "time_text": {"type": "string"},
            "time_edtf": {"type": "string", "description": "EDTF date/interval: 1870-10-26, 1873/1874, 1861/1877, 187X"},
            "place_text": {"type": "string"},
            "revier": {"type": "string"},
            "district_no": {"type": "string"},
            "district_name": {"type": "string"},
            "compartment": {"type": "string", "description": "Abteilung / Unterabteilung, e.g. 4 b"},
            "status": {"type": "string", "enum": ["observed", "historical", "planned", "prescribed", "prohibited",
                                                   "recommended", "assessed", "unknown"]},
            "quote": {"type": "string", "description": "VERBATIM from the document text, max ~40 words"},
            "page": {"type": "string", "description": "page anchor id of the quote, e.g. p017"},
            "confidence": CONFIDENCE,
            "note": {"type": "string"}},
            "required": ["category", "subject", "status", "quote", "page", "confidence"]}},
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["windthrow", "snow_break", "ice_break", "bark_beetle", "other_insects",
                                                 "fire", "frost", "drought", "flood", "game_damage", "grazing_damage",
                                                 "fungal", "other"]},
            "date_text": {"type": "string"}, "date_edtf": {"type": "string"},
            "place_text": {"type": "string"}, "revier": {"type": "string"},
            "extent": {"type": "string", "description": "area, volume or description of the extent as written"},
            "quote": {"type": "string"}, "page": {"type": "string"}},
            "required": ["type", "quote", "page"]}},
        "segment_summary_en": {"type": "string"},
    },
    "required": ["statements", "events", "segment_summary_en"],
}

STATEMENT_PROMPT = '''You are an expert in historical forestry (Forsteinrichtung) and environmental history. Below is a complete
document from the 1878/90 Waldstandsrevision of the Ilzertrift-Komplex (Reviere Schönau, St. Oswald, Klingenbrunn,
Forstamt Schönberg, Bavarian Forest), proofread against the scans. Page anchors ⟦p017⟧ mark where each page begins.

{unit_context}

Extract ecological information as atomic statements: tree species and mixtures, stand structure, age and stocking,
site (elevation, exposure, soil, bedrock, moisture, Auen, Filze), ground vegetation, climate and weather, damage
events, regeneration and its success, silvicultural measures and rules, non-timber uses (grazing, litter, resin, peat),
wildlife, hydrology and land use (drainage, timber floating, roads), quantities with units.

RULES
1. One statement per distinct fact; do not merge different places, dates or species.
2. "quote" is copied verbatim from the document text (it must be findable), max ~40 words; "page" is the anchor of
   the page the quote is on.
3. Normalise subject and species to modern German headwords; keep value as written; dates as written + EDTF.
4. Place: give Revier, Distrikt, Abteilung where the text names them; otherwise place_text.
5. status: observed (state described), historical (past events), planned / prescribed / prohibited / recommended
   (rules and plans), assessed (estimates, yields).
6. Damage events (storm, snow, ice, beetles, fire, frost …) also go into "events" with date and extent.
7. Administrative matters (salaries, personnel, accounting) are not ecological unless they carry ecological facts.

{glossary}

=== DOCUMENT ===
{document}
=== END DOCUMENT ===
'''

SEGMENT_INSTRUCTION = '''TASK: extract statements and events ONLY from the pages {first} to {last} (anchors ⟦{first}⟧ … ⟦{last}⟧). Use
the rest of the document only as context (for subjects, places and dates that are introduced on other pages).'''


def page_anchor(position: int) -> str:
    return f"p{position:03d}"


def proofread_page(plan: PagePlan, unit_dir: Path, gemini: Gemini, out_path: Path, force: bool = False) -> dict:
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    _, body = parse_md((unit_dir / plan.page["transcript"]).read_text(encoding="utf-8"))
    text = clean_body(body, "text")
    images = [unit_dir / plan.page["image"]] if plan.page["image"] else []
    started = time.time()
    data, usage, mode = gemini.extract(PROOF_PROMPT.format(page_id=plan.page["pid"], text=text), PROOF_SCHEMA,
                                       images=images, thinking="low")
    corrected, applied = apply_corrections(text, data)
    record = {"position": plan.position, "page_id": plan.page["pid"], "usage": usage, "mode": mode,
              "elapsed_s": round(time.time() - started, 1), "has_image": bool(images), "result": data,
              "applied": applied, "text": text, "corrected": corrected}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def apply_corrections(text: str, proof: dict) -> tuple[str, list[dict]]:
    applied = []
    for correction in proof.get("corrections", []):
        source, target = correction.get("transcript_reads", ""), correction.get("image_reads", "")
        status = "applied" if source and text.count(source) == 1 else ("not_found" if source not in text else "ambiguous")
        if status == "applied":
            text = text.replace(source, target)
        applied.append({**correction, "status": status})
    for missing in proof.get("missing_text", []):
        anchor = missing.get("after", "")
        if anchor and text.count(anchor) == 1:
            text = text.replace(anchor, anchor + " " + missing["text"])
            applied.append({"transcript_reads": anchor, "image_reads": anchor + " " + missing["text"],
                            "kind": "omission", "confidence": "medium", "status": "applied"})
        else:
            applied.append({"transcript_reads": anchor, "image_reads": missing.get("text", ""), "kind": "omission",
                            "confidence": "medium", "status": "not_found"})
    return text, applied


def assemble(plans: list[PagePlan], proofs: dict) -> str:
    parts = []
    for plan in plans:
        proof = proofs.get(plan.position)
        if proof:
            parts.append(f"⟦{page_anchor(plan.position)}⟧ {plan.page['pid']}\n{proof['corrected']}")
    return "\n\n".join(parts)


def segments_for(plans: list[PagePlan], spec: dict, size: int = 4) -> list[tuple[int, int]]:
    positions = [plan.position for plan in plans]
    if spec.get("segments"):
        return [tuple(segment) for segment in spec["segments"] if segment[0] in positions or segment[1] in positions]
    return [(positions[i], positions[min(i + size, len(positions)) - 1]) for i in range(0, len(positions), size)]


def segment_positions(plans: list[PagePlan], segment: tuple[int, int]) -> list[int]:
    order = [plan.position for plan in plans]
    if segment[0] in order and segment[1] in order:
        start, end = order.index(segment[0]), order.index(segment[1])
        return order[min(start, end): max(start, end) + 1]
    return [p for p in order if segment[0] <= p <= segment[1]]


def extract_segment(document: str, spec: dict, segment: tuple[int, int], gemini: Gemini, out_path: Path,
                    force: bool = False) -> dict:
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    unit_context = f"UNIT: {spec.get('unit')} – {spec.get('title')}\n{(spec.get('summary') or '').strip()}"
    prompt = STATEMENT_PROMPT.format(unit_context=unit_context, glossary=GLOSSARY, document=document)
    instruction = SEGMENT_INSTRUCTION.format(first=page_anchor(segment[0]), last=page_anchor(segment[1]))
    started = time.time()
    data, usage, mode = gemini.extract(prompt + "\n" + instruction, STATEMENT_SCHEMA)
    record = {"segment": list(segment), "usage": usage, "mode": mode, "elapsed_s": round(time.time() - started, 1),
              "result": data}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


ANCHOR_LINE = re.compile(r"⟦p\d{3}⟧[^\n]*\n?")


def plain(text: str) -> str:
    text = ANCHOR_LINE.sub(" ", text or "")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"```\w*", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(\w)[-=¬]\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[*_#>~]+", " ", text)
    return text


def normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", plain(text)).strip().lower()


def quote_status(quote: str, document: str, page_text: str) -> str:
    if not quote:
        return "empty"
    if quote in page_text:
        return "exact_on_page"
    if normalise_space(quote) in normalise_space(page_text):
        return "normalised_on_page"
    if normalise_space(quote) in normalise_space(document):
        return "elsewhere_in_document"
    words = normalise_space(quote).split()
    if len(words) >= 5 and " ".join(words[:5]) in normalise_space(document):
        return "prefix"
    return "missing"
