"""Per-region transcription using Gemini for historical German administrative records.

Region-type → prompt routing:

  TableRegion   → HTML table (supports multi-row headers, merged cells, red-ink notes)
  GraphRegion   → structured description of axes, data points, curve
  TitleRegion   → exact transcription of heading text
  Text regions  → line-by-line Kurrent transcription with markup
  PageNumber    → skipped (already extracted during detection)
"""

import io
import logging
from typing import Any, Dict

from PIL import Image

from .config import PipelineConfig

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ────────────────────────────────────────────────────────────────────────────

# ── Tables ──────────────────────────────────────────────────────────────────
_TABLE_PROMPT = """\
You are transcribing a table from a 19th-century Bavarian / German \
administrative record (forest inventory, grazing ledger, land register, etc.).
The text is handwritten in German Kurrent (Kurrentschrift) or Sütterlin script.
Some entries use red ink for corrections, subtotals, or later annotations.

EXPECTED STRUCTURE
  Header rows : {header_rows} (rows containing column labels, potentially \
spanning multiple sub-header rows)
  Data rows   : approximately {data_rows}
  Columns     : approximately {cols}
  Red ink     : {has_red_ink}
  Totals row  : {has_totals_row}

OUTPUT FORMAT
Produce a single HTML <table> embedded in a Markdown code block tagged `html`.
Use <thead> for all header rows and <tbody> for data rows.
Use colspan / rowspan attributes wherever the original uses merged cells.
Rules:
  1. Transcribe EVERY cell exactly as written — preserve German words, \
abbreviations, and all numeric values character by character.
  2. Numbers: copy digits, commas, dots, and fractions exactly (e.g. "116 72", \
"85 00", "1/4").  Do NOT interpret or convert units.
  3. Red-ink content: wrap in <span class="red">…</span>.
  4. Uncertain characters: [?]  |  Illegible word: [illegible]
  5. Empty cells: leave the <td> empty.
  6. Do NOT add any text outside the HTML block.

Begin the response with:
```html
<table>
"""

# ── Graphs / charts ─────────────────────────────────────────────────────────
_GRAPH_PROMPT = """\
You are transcribing a graph or diagram page from a 19th-century German \
administrative/scientific record.  The image shows a chart drawn on graph paper.

Provide a structured description in Markdown with these exact sections:

## Graph Type
State the graph type (scatter plot, line/curve diagram, bar chart, other).

## Axes
Describe each axis: label as written (transcribe exactly), units (if visible), \
and approximate range of values.

## Data Points
List all legible individual data-point annotations visible on the graph \
(numbers, labels) as a Markdown table with columns: Label | Approximate X | \
Approximate Y.  If no labels are visible, note "unlabelled".

## Fitted Curve / Lines
Describe any drawn curves or straight lines: their shape \
(monotone decreasing, S-shaped, etc.) and approximate span.

## Additional Notes
Any legend, title, or marginal text transcribed exactly.

Transcribe all visible text character by character.  Mark uncertain readings \
with [?] and illegible text with [illegible].
"""

# ── Title / heading ──────────────────────────────────────────────────────────
_TITLE_PROMPT = """\
Transcribe the heading or title text in this image region exactly as written.
The text is handwritten in German Kurrent or Sütterlin script.

Rules:
  • Preserve spelling, punctuation, and capitalisation exactly.
  • Mark uncertain characters with [?].
  • Mark completely illegible words with [illegible].
  • Output ONLY the transcribed text, nothing else.
"""

# ── Running text (paragraphs, marginalia, footnotes) ────────────────────────
_TEXT_PROMPT = """\
Transcribe the German text in this image region exactly as written, \
character by character, line by line.

The handwriting is {script} ({region_type}).
This is a 19th-century Bavarian / German administrative or scientific document.

Rules:
  • Preserve line breaks exactly as they appear in the image.
  • German Kurrent / Sütterlin script: pay close attention to letters that \
look similar (e.g. n/u, e/i, r/n, ß/p, h/k, f/long-s).
  • Mark underlined words like this: <u>word</u>
  • Mark superscript (e.g. reference numbers) like this: <sup>3</sup>
  • Mark red-ink text like this: <span class="red">text</span>
  • Use [?] for uncertain characters and [illegible] for unreadable words.
  • Do NOT add interpretations, translations, or commentary.

Output ONLY the transcription.
"""

# ────────────────────────────────────────────────────────────────────────────
# Transcriber class
# ────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """Region-level transcription with Gemini."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    # ── public API ───────────────────────────────────────────────────────

    def transcribe_region(
        self,
        region_image: Image.Image,
        region: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transcribe a single cropped region image.

        Returns a dict with ``status``, ``text``, and optional metadata.
        """
        rtype = region["type"]

        # PageNumberRegion – number already extracted during detection
        if rtype == "PageNumberRegion":
            return {
                "status": "success",
                "text": str(region.get("page_number", "")),
                "skipped": True,
            }

        prompt = self._build_prompt(region)
        return self._call(region_image, prompt, rtype)

    # ── prompt routing ───────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(region: Dict) -> str:
        rtype = region["type"]

        if rtype == "TableRegion":
            rows = region.get("rows", "?")
            header_rows = region.get("header_rows", 1)
            cols = region.get("cols", "?")
            has_red_ink = "yes" if region.get("has_red_ink") else "no"
            has_totals = "yes" if region.get("has_totals_row") else "no"
            # Estimate data rows if we have both numbers
            if isinstance(rows, int) and isinstance(header_rows, int):
                data_rows = max(0, rows - header_rows)
            else:
                data_rows = "?"
            return _TABLE_PROMPT.format(
                header_rows=header_rows,
                data_rows=data_rows,
                cols=cols,
                has_red_ink=has_red_ink,
                has_totals_row=has_totals,
            )

        if rtype == "GraphRegion":
            return _GRAPH_PROMPT

        if rtype == "TitleRegion":
            return _TITLE_PROMPT

        if rtype == "ImageRegion":
            return _GRAPH_PROMPT  # reuse graph prompt for sketches/maps

        # ParagraphRegion, MarginaliaRegion, FootnoteRegion
        script = region.get("script", "kurrent")
        script_label = {
            "kurrent": "German Kurrent (Kurrentschrift)",
            "latin": "Latin cursive",
            "mixed": "mixed Kurrent and Latin cursive",
        }.get(script, "German Kurrent")
        return _TEXT_PROMPT.format(script=script_label, region_type=rtype)

    # ── Gemini call ──────────────────────────────────────────────────────

    def _call(
        self, image: Image.Image, prompt: str, rtype: str
    ) -> Dict[str, Any]:
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        try:
            resp = self.client.models.generate_content(
                model=self.cfg.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=self.cfg.transcription_temperature,
                    max_output_tokens=16384,   # tables can be very long
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.cfg.transcription_thinking
                    ),
                ),
            )
            text = resp.text.strip()
            return {"status": "success", "text": text}

        except Exception as exc:
            log.error("Transcription failed (%s): %s", rtype, exc)
            return {"status": "error", "error": str(exc), "text": ""}
