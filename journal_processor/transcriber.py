"""Per-region transcription using Gemini for historical German administrative records.

Region-type → prompt routing:

  TableRegion   → HTML table (supports multi-row headers, merged cells, red-ink notes)
  GraphRegion   → structured description of axes, data points, curve
  TitleRegion   → exact transcription of heading text
  Text regions  → line-by-line Kurrent transcription with markup
  PageNumber    → skipped (already extracted during detection)

Robustness:
  Each call goes through ``_call_with_retry`` which post-processes the
  raw model output to strip any reasoning that leaked outside the
  required format, detects truncation (``finish_reason == MAX_TOKENS``
  or a missing ``</table>``), and retries with a higher output-token
  budget and a stricter directive when needed.
"""

import io
import logging
import re
from typing import Any, Dict, Tuple

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
  6. Do NOT add ANY text outside the ```html block — no preamble, no thinking, \
no commentary, no summary.  Your entire response must be exactly one fenced \
code block, beginning with ```html and ending with ```.

Begin the response with:
```html
<table>
"""

_TABLE_RETRY_PREAMBLE = """\
IMPORTANT — your previous response was incomplete or contained text outside \
the required HTML block.  This time output ONLY a single complete \
```html ... ``` block containing one full <table>…</table>.  Do not write \
anything else — no thinking aloud, no row-by-row commentary, no summary.

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

Output ONLY the five sections above starting with "## Graph Type".  No \
preamble, no thinking aloud, no concluding remarks.
"""

_GRAPH_RETRY_PREAMBLE = """\
IMPORTANT — your previous response was incomplete or contained text outside \
the required structure.  This time output ONLY the five "## " sections \
(Graph Type, Axes, Data Points, Fitted Curve / Lines, Additional Notes), \
in order, with no preamble or commentary outside the sections.

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
# Response cleaning helpers
# ────────────────────────────────────────────────────────────────────────────

# Locate a fenced ```html block (closed or open-to-EOF).
_HTML_FENCE_RE       = re.compile(r"```html\s*\n(.*?)\n```", re.DOTALL)
_HTML_FENCE_OPEN_RE  = re.compile(r"```html\s*\n(.*)\Z",     re.DOTALL)


def _extract_table_html(text: str) -> Tuple[str, bool]:
    """Pull the HTML <table> out of a raw model response.

    Discards any prose before/after the fenced block (model "thinking
    aloud" leaks).  Returns ``(clean_text_in_fence, is_complete)`` where
    ``is_complete`` is True iff the table has a closing ``</table>`` tag.
    """
    # Closed fence first
    m = _HTML_FENCE_RE.search(text)
    if m:
        content = m.group(1).strip()
        return f"```html\n{content}\n```", "</table>" in content

    # Open-to-EOF fence (truncated)
    m = _HTML_FENCE_OPEN_RE.search(text)
    if m:
        content = m.group(1).strip()
        return f"```html\n{content}\n```", "</table>" in content

    # No fence at all — try to find a bare <table>
    table_start = text.find("<table")
    if table_start != -1:
        content = text[table_start:].strip()
        return f"```html\n{content}\n```", "</table>" in content

    # Nothing usable
    return text.strip(), False


# A graph response should contain these five headings, in order.
_GRAPH_HEADINGS = (
    "Graph Type", "Axes", "Data Points", "Fitted Curve", "Additional Notes",
)


def _extract_graph_md(text: str) -> Tuple[str, bool]:
    """Strip any prose before the first ``## `` heading.

    Returns ``(clean_text, is_complete)`` where ``is_complete`` requires
    at least three of the expected five sections to be present (some
    pages legitimately lack one or two).
    """
    first_h2 = text.find("## ")
    clean = (text[first_h2:] if first_h2 != -1 else text).strip()
    headings_found = sum(1 for h in _GRAPH_HEADINGS if f"## {h}" in clean)
    return clean, headings_found >= 3


def _extract_text(text: str) -> Tuple[str, bool]:
    """Lightweight cleanup for paragraph / title / marginalia regions.

    These prompts ask for plain text; the model only rarely leaks
    reasoning here, so we just strip whitespace and report complete.
    """
    return text.strip(), True


def _finish_reason(resp: Any) -> str:
    """Best-effort read of the SDK's stop reason — string-ified for logging."""
    try:
        reason = resp.candidates[0].finish_reason
        return getattr(reason, "name", str(reason)).upper()
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────────────────
# Transcriber class
# ────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """Region-level transcription with Gemini, with retry on truncation."""

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

        Returns a dict with ``status``, ``text``, optional ``warning``
        ("truncated") and ``attempts`` (how many calls it took).
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
        return self._call_with_retry(region_image, prompt, rtype)

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

    # ── retry-aware call ─────────────────────────────────────────────────

    def _call_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        rtype: str,
    ) -> Dict[str, Any]:
        max_attempts = max(1, self.cfg.transcription_max_attempts)
        token_budget = self.cfg.transcription_max_output_tokens
        retry_budget = self.cfg.transcription_max_output_tokens_retry

        last_clean = ""
        last_complete = False
        last_finish = ""

        for attempt in range(1, max_attempts + 1):
            budget = token_budget if attempt == 1 else retry_budget
            resp = self._call_once(image, prompt, rtype, max_output_tokens=budget)
            if resp["status"] != "success":
                # API/transport error — bubble up
                return {**resp, "attempts": attempt}

            raw = resp["text"]
            finish = resp.get("finish_reason", "")
            clean, complete = self._postprocess(raw, rtype)

            last_clean, last_complete, last_finish = clean, complete, finish
            truncated = (finish == "MAX_TOKENS") or not complete

            if not truncated:
                result = {"status": "success", "text": clean, "attempts": attempt}
                if attempt > 1:
                    log.info("Region %s recovered on retry %d", rtype, attempt)
                return result

            if attempt < max_attempts:
                log.warning(
                    "Region %s output looks %s (finish=%s) — retrying with %d tokens",
                    rtype,
                    "truncated" if not complete else "incomplete",
                    finish or "unknown",
                    retry_budget,
                )
                # Strengthen the directive for the retry
                prompt = self._strengthen_prompt(prompt, rtype)

        # Out of attempts — return best-effort with a warning flag
        log.warning(
            "Region %s still truncated after %d attempt(s) — keeping partial output",
            rtype, max_attempts,
        )
        return {
            "status": "success",
            "text": last_clean,
            "attempts": max_attempts,
            "warning": "truncated",
            "finish_reason": last_finish,
        }

    @staticmethod
    def _postprocess(raw: str, rtype: str) -> Tuple[str, bool]:
        if rtype == "TableRegion":
            return _extract_table_html(raw)
        if rtype in ("GraphRegion", "ImageRegion"):
            return _extract_graph_md(raw)
        return _extract_text(raw)

    @staticmethod
    def _strengthen_prompt(prompt: str, rtype: str) -> str:
        if rtype == "TableRegion":
            return _TABLE_RETRY_PREAMBLE + prompt
        if rtype in ("GraphRegion", "ImageRegion"):
            return _GRAPH_RETRY_PREAMBLE + prompt
        return prompt

    # ── single Gemini call ───────────────────────────────────────────────

    def _call_once(
        self,
        image: Image.Image,
        prompt: str,
        rtype: str,
        *,
        max_output_tokens: int,
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
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.cfg.transcription_thinking
                    ),
                ),
            )
            return {
                "status": "success",
                "text": (resp.text or "").strip(),
                "finish_reason": _finish_reason(resp),
            }

        except Exception as exc:
            log.error("Transcription failed (%s): %s", rtype, exc)
            return {"status": "error", "error": str(exc), "text": ""}
