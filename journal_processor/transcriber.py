"""Per-region transcription using Gemini for historical German administrative records.

Region-type → prompt routing:

  TableRegion   → HTML table (supports multi-row headers, merged cells, red-ink notes)
  GraphRegion   → structured description of axes, data points, curve
  TitleRegion   → exact transcription of heading text
  Text regions  → line-by-line Kurrent transcription with markup
  PageNumber    → skipped (already extracted during detection)

Single-pass full-page modes (used by doc_type="table", "graph", "map"):

  table-mode  → entire page → HTML <table> + any surrounding text as Markdown
  graph-mode  → entire page → structured Markdown for tree-height curve graphs
  map-mode    → entire page → structured map metadata (type, title, geoident,
                scale, date, …) + all visible text as Markdown
"""

import io
import logging
import time
from typing import Any, Dict, Optional

from PIL import Image

from .config import PipelineConfig

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Prompt templates (per-region)
# ────────────────────────────────────────────────────────────────────────────

# ── Tables (per-region, used by mixed/text doc_types) ───────────────────────
_TABLE_PROMPT = """\
You are transcribing a table from a 19th-century Bavarian / German \
administrative record (forest inventory, grazing ledger, land register, etc.).
The text is handwritten in German Kurrent (Kurrentschrift).

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

# ── Graphs / charts (per-region) ────────────────────────────────────────────
_GRAPH_PROMPT = """\
You are transcribing a graph or diagram page from a 19th-century German \
forestry record.  The image shows a chart drawn on graph paper, measuring the height of trees.

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
This is a 19th-century Bavarian / German administrative document.

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
# Prompt templates (full-page single-pass modes)
# ────────────────────────────────────────────────────────────────────────────

# ── Full-page table (doc_type="table") ──────────────────────────────────────
_FULL_PAGE_TABLE_PROMPT = """\
You are processing a single page from a 19th-century Bavarian / German \
administrative record (forestry, land management, agriculture, grazing register).

The page is dominated by ONE large complex table. There may also be additional \
text on the page outside the table — a title or heading above it, a caption, \
footnotes below it, or marginal notes beside it.

The text is handwritten in German Kurrent (Kurrentschrift) or Sütterlin script.

OUTPUT FORMAT
Produce a single Markdown document with these parts, in reading order:

  1. Any title or heading text appearing ABOVE the table → as a Markdown \
heading (`## Title`) or paragraph.

  2. The main table → as ONE HTML <table> wrapped in a fenced code block:

     ```html
     <table>
       <thead>...</thead>
       <tbody>...</tbody>
     </table>
     ```

  3. Any text appearing BELOW the table or in the margins (footnotes, \
captions, marginalia, signatures) → as Markdown paragraphs after the HTML \
block. Mark a marginal note with `> *[Marginalie]*` on its own line followed \
by the quoted text.

If there is NO additional text outside the table, output ONLY the fenced \
HTML block.

TABLE RULES
  • Use <thead> for ALL header rows (including stacked sub-headers) and \
<tbody> for data rows.
  • Use `colspan` / `rowspan` for merged cells exactly as drawn.
  • Transcribe EVERY cell exactly as written — preserve German words, \
abbreviations, and numeric values character by character.
  • Numbers: copy digits, commas, dots, fractions exactly (e.g. "116 72", \
"85 00", "1/4"). Do NOT interpret or convert units.
  • Red-ink content: wrap in <span class="red">…</span>.
  • Uncertain characters: [?]   |   Illegible word: [illegible]
  • Empty cells: leave the <td> empty.

TEXT RULES (for content outside the table)
  • Preserve line breaks as they appear in the image.
  • Mark underlined words: <u>word</u>
  • Mark superscript numbers: <sup>3</sup>
  • Mark red-ink text: <span class="red">text</span>
  • Mark uncertain characters: [?]   |   illegible words: [illegible]
  • Do NOT add interpretations, translations, or commentary.

Output ONLY the Markdown — no preamble, no commentary, no explanations.
"""


# ── Full-page tree-height graph (doc_type="graph") ─────────────────────────
# IMPORTANT: read the GRID-READING METHODOLOGY section near the bottom of the
# prompt. Locating scatter points is the part the model finds hardest, so the
# prompt now teaches an explicit "bracket between two grid lines / two
# tabulated heights, then interpolate" method. This is paired at runtime with
# ``cfg.graph_media_resolution = "ultra_high"`` so the model gets enough
# visual detail to read each circle's position.
_FULL_PAGE_GRAPH_PROMPT = """\
You are processing a single page from a 19th-century German forestry record.
The page contains ONE tree-height measurement graph (Baumhöhenkurve), drawn \
on graph paper.

IMPORTANT ORIENTATION NOTE: These pages are often scanned sideways. Before \
processing, visually determine the correct orientation. The axis with numeric \
labels usually represents the independent variable (e.g., Age or Diameter) \
and the handwritten floating numbers are the dependent variable (Height).

Such pages typically contain:
  • A title / heading naming the tree species (e.g. "Fichten", "Tannen").
  • A coordinate system. One axis usually has a fine numeric ruler.
  • A scatter of small circular points representing measured trees.
  • A single smooth fitted curve drawn through the scatter points.
  • A sequence of handwritten numerical values (e.g., "35.0", "45.0") written \
along the curve. These are the TABULATED HEIGHTS, and they specifically align \
with the major grid lines of the numerically labeled axis.
  • Possibly: a legend, marginal notes, or page numbers.

OUTPUT FORMAT (Markdown)
Produce a single Markdown document with the following sections, in this exact \
order. Omit a section only if it genuinely does not apply.

## Calibration and Axes
First, analyze the visible axes to establish the scale:
- **Primary Labeled Axis:** (Label/Units, Range from visible ticks, e.g., 0 to 46).
- **Secondary Axis:** (Label/Units, Range if visible).
- **Grid Scale:** State explicitly what one major grid square and one minor \
grid square represent on the primary labeled axis.

## Title / Heading
Transcribe the title text exactly as written, including any species name and \
yield-class notation (e.g., I.b.q.a.).

## Tabulated Heights & Point Labels
The handwritten values in the body of the graph correspond to specific points \
on the curve. Map each visible handwritten value to its corresponding \
coordinate on the Primary Labeled Axis by following the grid line.
Do not skip values, do not round, do not interpolate.

| Primary Axis Coordinate | Tabulated Height Value |
|---|---|
| (e.g., 4) | 35.0 |
| (e.g., 5) | 45.0 |

## Scatter Data Points
Approximate the coordinates of the plotted scatter points (the small circles). \
Use the primary labeled axis for one coordinate. For the secondary coordinate, \
estimate the value based on the nearest Tabulated Height label.
If the points are extremely dense, list a representative subset (at least 10 \
points spanning the graph) and add a note giving the estimated total count.

| # | Primary Axis (Approx) | Estimated Height (Y) |
|---|---|---|
| 1 | ... | ... |

## Fitted Curve
Describe the drawn curve specifically referencing the grid:
- **Start Point:** Where does the curve begin? (approximate coordinates).
- **End Point:** Where does the curve end? (approximate coordinates).
- **Trajectory:** Describe the shape (e.g., monotone increasing, decelerating, \
linear early on then flattening out).
- **Relationship to Scatter:** Does the curve perfectly bisect the scatter \
points, or does it sit above/below certain clusters?

## Notes
Any legend, marginal text, dates, signatures, page numbers, or other writing \
on the page — transcribed exactly.

GRID-READING METHODOLOGY (apply this when filling in "Scatter Data Points")
This is the part that requires the most care. For EACH scatter point:

  STEP 1 — Bracket on the Primary Labeled Axis.
    Identify the two MAJOR grid lines on the labeled axis that immediately \
bracket the point (one before, one after). The Primary-Axis coordinate lies \
between these two known values. Estimate the fractional position of the \
point between them in eighths or quarters (e.g. "halfway = +0.5 of one major \
square", "a quarter past = +0.25"). If the point sits ON a major grid line, \
record that exact value and note "(on grid line)".

  STEP 2 — Bracket on the Secondary (Height) Axis.
    The Tabulated Heights you wrote in the previous section are your reference \
markers along this axis: each one sits exactly on a major grid line of the \
Primary axis. Find the two Tabulated-Height markers that immediately bracket \
the scatter point (one above, one below). Estimate the fractional position of \
the point between them.

  STEP 3 — Combine and report.
    Combine STEP 1 and STEP 2 into a single (Primary, Height) pair. Round each \
coordinate to one decimal place. If you are uncertain about a particular point, \
mark it with a trailing [?]; if a circle is so faint or overlapped that you \
cannot read it, do NOT invent coordinates — list it as "[illegible]" instead.

  WORKED EXAMPLE
    A circle sits roughly halfway between gridline 7 and gridline 8 on the \
Primary axis, and roughly two-thirds of the way from the "35.0" marker up \
toward the "38.0" marker. Then:
      Primary  ≈ 7.5
      Height   ≈ 35.0 + 2/3 × (38.0 − 35.0) = 37.0
    Report this as: ``| n | 7.5 | 37.0 |``.

RULES
  • Transcribe all visible text and numbers character by character.
  • Mark uncertain readings with [?] and illegible text with [illegible].
  • Preserve original spelling and notation — do NOT convert units or \
modernise spelling.
  • Output ONLY the Markdown — no preamble, no commentary outside the sections.
"""


# ── Full-page map (doc_type="map") ──────────────────────────────────────────
_FULL_PAGE_MAP_PROMPT = """\
You are processing a single page from a 19th-century Bavarian / German \
forestry / land-management archive. The page is a MAP — typically a hand-drawn \
or printed cadastral, forestry, parcel, sketch, or topographic map, with \
handwritten labels in German Kurrent / Sütterlin script and / or printed \
German.

Your job is to extract a small set of structured METADATA fields together \
with ALL legible text. The output is a single Markdown document with the \
sections below in this exact order. Omit a section only if it genuinely \
does not apply.

## Map Type
A short description of the map type — for example: cadastral map (Flurkarte), \
forestry map (Forstkarte), parcel map (Lageplan), topographic map, sketch map, \
or stand map (Bestandskarte). Note whether it is hand-drawn or printed and \
whether it is monochrome or coloured.

## Map Title
Transcribe the map's title or principal heading text exactly as written. If \
no explicit title is present, write "(no explicit title)".

## Geographic Identification (Geoident)
The approximate location depicted on the map. List, in order from broadest \
to narrowest, every administrative or geographic identifier visible on the \
map: country / state (Bayern, Sachsen, …), district (Bezirk, Landgericht), \
forest district (Forstamt, Revier, Distrikt), municipality (Gemeinde, \
Pfarrei), and any individual place names (towns, hamlets, forests, fields, \
parcels, mountains, rivers). If the map shows a numbered parcel or stand \
("Abteilung 12"), include the number.

## Scale
Transcribe the scale exactly as written, including the German wording \
("Maßstab 1 : 5000", "1 zu 25 000", etc.). If only a graphical scale bar \
is visible without a numerical ratio, describe it (e.g. "graphical bar \
labelled 0–500 Schritt"). If no scale at all is present, write "(no scale \
visible)".

## Date
Transcribe any visible date exactly as written. This may be a year, a full \
date ("den 12. März 1873"), a survey year, or a revision date — list ALL \
of them with a short label of where they appear (e.g. "(below cartouche): \
1873", "(revision stamp): 1891"). If no date is visible, write \
"(no date visible)".

## Compass / Orientation
If a north arrow, compass rose, or written orientation indicator ("Norden", \
"N") is visible, report which way is north (e.g. "north arrow points to \
top-right"). If no orientation is indicated, write "(not indicated)".

## Transcribed Text
List EVERY legible text element on the map, one per line, exactly as written. \
This includes place names, parcel numbers, field names, owner names, legend \
entries, road / river labels, distance numbers, abbreviations, and any \
marginal annotations. Group spatially-related labels with a short prefix in \
parentheses if helpful (e.g. "(top-left, label of forest stand): \
Eichenwald"). Mark uncertain characters with [?] and illegible words with \
[illegible].

## Notes
Any signatures, surveyor names, marginal notes, page numbers, official \
stamps, seals, or other writing not already covered — transcribed exactly.

RULES
  • Transcribe all visible text and numbers character by character.
  • Preserve original spelling and notation; do NOT modernise or translate.
  • Mark uncertain readings with [?] and illegible text with [illegible].
  • Mark red-ink text with <span class="red">text</span>.
  • Output ONLY the Markdown — no preamble, no commentary outside the sections.
"""


# ────────────────────────────────────────────────────────────────────────────
# Helper: safely extract text from a Gemini response
# ────────────────────────────────────────────────────────────────────────────

def _extract_response_text(resp: Any) -> str:
    """Return the text payload of a Gemini response, robust to ``None``.

    The ``response.text`` property of the google-genai SDK can return ``None``
    when the model produced thought parts but no visible text part (this can
    happen with Gemini 3.1 Pro under heavy thinking budgets, where the model
    consumes its output budget on internal reasoning before emitting any
    user-facing tokens). To avoid an ``AttributeError`` on ``.strip()`` we
    fall back to walking ``candidates[*].content.parts`` ourselves.
    """
    text = getattr(resp, "text", None)
    if text:
        return text

    # Fallback: walk the candidate parts directly.
    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            collected = []
            for part in parts:
                # Skip thought-only parts (those carry a thoughtSignature but
                # no real text). Real text parts have a non-empty ``.text``.
                t = getattr(part, "text", None)
                if t:
                    collected.append(t)
            if collected:
                return "".join(collected)
    except Exception:  # pragma: no cover — purely defensive
        pass

    return ""


def _finish_reason(resp: Any) -> str:
    """Return the finish_reason of the first candidate as a string, or ''."""
    try:
        cand = (resp.candidates or [None])[0]
        if cand is None:
            return ""
        fr = getattr(cand, "finish_reason", None)
        if fr is None:
            return ""
        # google-genai SDK uses an Enum here; .name gives a clean string
        return getattr(fr, "name", str(fr))
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────────────────
# Helper: build an image Part with optional media_resolution override
# ────────────────────────────────────────────────────────────────────────────

# Allowed values for the media_resolution config knobs. We accept the short
# form ("high") or the full enum form ("media_resolution_high").
_MEDIA_RES_ALIASES = {
    "low":                          "media_resolution_low",
    "medium":                       "media_resolution_medium",
    "high":                         "media_resolution_high",
    "ultra_high":                   "media_resolution_ultra_high",
    "media_resolution_low":         "media_resolution_low",
    "media_resolution_medium":      "media_resolution_medium",
    "media_resolution_high":        "media_resolution_high",
    "media_resolution_ultra_high":  "media_resolution_ultra_high",
}


def _make_image_part(img_bytes: bytes, mime: str, level: str):
    """Build a ``types.Part`` for an image, with optional media_resolution.

    ``level`` may be empty (use SDK default) or one of:
      "low" / "medium" / "high" / "ultra_high"
    or the equivalent ``media_resolution_*`` form. Anything else falls back
    to the default. The Gemini 3 ``media_resolution`` parameter is only
    available in the v1alpha API version, which the Pipeline already
    selects.
    """
    from google.genai import types

    if not level:
        return types.Part.from_bytes(data=img_bytes, mime_type=mime)

    canonical = _MEDIA_RES_ALIASES.get(level)
    if not canonical:
        log.warning(
            "Unknown media_resolution %r — falling back to SDK default", level,
        )
        return types.Part.from_bytes(data=img_bytes, mime_type=mime)

    try:
        return types.Part(
            inline_data=types.Blob(mime_type=mime, data=img_bytes),
            media_resolution=types.PartMediaResolution(level=canonical),
        )
    except Exception as exc:  # pragma: no cover — defensive against SDK skew
        log.warning(
            "Could not attach media_resolution=%s to image part (%s); "
            "using SDK default.",
            canonical, exc,
        )
        return types.Part.from_bytes(data=img_bytes, mime_type=mime)


# ────────────────────────────────────────────────────────────────────────────
# Helper: Pro-only debug dump of every Gemini response
# ────────────────────────────────────────────────────────────────────────────

# We treat this exact model id as "Pro" for debug-dump purposes. Other model
# ids (Flash, Flash-Lite, etc.) are unaffected.
_PRO_MODEL_ID = "gemini-3.1-pro-preview"


def _debug_dump_pro_response(
    resp: Any,
    *,
    label: str,
    model_id: str,
    enabled: bool,
    attempt: int = 1,
    total_attempts: int = 1,
) -> None:
    """When running on Gemini 3.1 Pro, dump the full response to the log.

    Triggered for every call (successful or empty), so a user staring at an
    empty .md file can scroll up in the log and see exactly what the model
    actually returned — finish_reason, usage metadata, and every part on
    the candidate, including hidden ``thought=True`` parts.

    No-op for any other model id, or if ``cfg.pro_debug`` is False.
    """
    if not enabled or model_id != _PRO_MODEL_ID:
        return

    finish = _finish_reason(resp) or "?"
    usage = getattr(resp, "usage_metadata", None)

    log.info(
        "─── PRO DEBUG · %s · attempt %d/%d · finish=%s ───",
        label, attempt, total_attempts, finish,
    )
    if usage is not None:
        # usage_metadata exposes prompt_token_count / candidates_token_count /
        # thoughts_token_count / total_token_count on Gemini 3.
        try:
            log.info(
                "  usage: prompt=%s  candidates=%s  thoughts=%s  total=%s",
                getattr(usage, "prompt_token_count", "?"),
                getattr(usage, "candidates_token_count", "?"),
                getattr(usage, "thoughts_token_count", "?"),
                getattr(usage, "total_token_count", "?"),
            )
        except Exception:  # pragma: no cover
            log.info("  usage: %r", usage)

    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        log.info("  (no candidates on response)")
        return

    for ci, cand in enumerate(candidates):
        content = getattr(cand, "content", None)
        if content is None:
            log.info("  candidate[%d]: no content", ci)
            continue
        parts = getattr(content, "parts", None) or []
        log.info("  candidate[%d]: %d part(s)", ci, len(parts))
        for pi, part in enumerate(parts):
            text = getattr(part, "text", None) or ""
            is_thought = bool(getattr(part, "thought", False))
            sig = getattr(part, "thought_signature", None)
            sig_len = len(sig) if isinstance(sig, (bytes, str)) else 0
            kind = "THOUGHT" if is_thought else ("TEXT" if text else "EMPTY")
            # Truncate long bodies so the log stays readable; the full text
            # for successful calls is also written to the .md output anyway.
            preview = text if len(text) <= 800 else text[:800] + "…[truncated]"
            log.info(
                "    part[%d] %-7s sig=%dB  body=%r",
                pi, kind, sig_len, preview,
            )
    log.info("─── /PRO DEBUG · %s ───", label)


# ────────────────────────────────────────────────────────────────────────────
# Transcriber class
# ────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """Region-level and full-page transcription with Gemini."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    # ── public API: per-region (used by doc_type="text"/"mixed") ─────────

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
        return self._call(
            region_image,
            prompt,
            label=rtype,
            thinking_level=self.cfg.transcription_thinking,
            retries=self.cfg.transcription_retries,
            media_resolution=self.cfg.region_media_resolution,
        )

    # ── public API: full-page (used by doc_type="table"/"graph"/"map") ───

    def transcribe_full_page(
        self,
        page_image: Image.Image,
        mode: str,
    ) -> Dict[str, Any]:
        """Run a single-pass full-page transcription.

        ``mode`` must be ``"table"``, ``"graph"``, or ``"map"``. The graph
        mode uses ``cfg.graph_media_resolution`` (default ``ultra_high``) so
        the model gets enough visual detail to read individual scatter
        points; the other two use ``cfg.full_page_media_resolution``
        (default ``high``).
        """
        if mode == "table":
            prompt = _FULL_PAGE_TABLE_PROMPT
            label = "FullPageTable"
            media_res = self.cfg.full_page_media_resolution
        elif mode == "graph":
            prompt = _FULL_PAGE_GRAPH_PROMPT
            label = "FullPageGraph"
            media_res = self.cfg.graph_media_resolution
        elif mode == "map":
            prompt = _FULL_PAGE_MAP_PROMPT
            label = "FullPageMap"
            media_res = self.cfg.full_page_media_resolution
        else:  # pragma: no cover — guarded by config validator
            raise ValueError(
                f"transcribe_full_page: unsupported mode {mode!r} "
                "(expected 'table', 'graph', or 'map')"
            )

        return self._call(
            page_image,
            prompt,
            label=label,
            thinking_level=self.cfg.full_page_thinking,
            retries=self.cfg.full_page_retries,
            media_resolution=media_res,
        )

    # ── prompt routing for per-region transcription ──────────────────────

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
        self,
        image: Image.Image,
        prompt: str,
        label: str,
        thinking_level: str,
        retries: int = 1,
        media_resolution: str = "",
    ) -> Dict[str, Any]:
        """Send one image+prompt request to Gemini and return a result dict.

        Retries on empty/None response — this happens with Gemini 3.1 Pro
        when high-budget thinking exhausts the output token allowance before
        any visible text is emitted.

        ``media_resolution`` is one of "" / "low" / "medium" / "high" /
        "ultra_high" (without the ``media_resolution_`` prefix). An empty
        string means "use the SDK default".
        """
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        last_err: Optional[str] = None
        # We may escalate media_resolution mid-loop on retries. Track the
        # currently-active value separately from the originally-configured
        # one so the escalation log lines stay accurate.
        current_media_res = media_resolution
        for attempt in range(max(1, retries + 1)):
            image_part = _make_image_part(img_bytes, "image/png", current_media_res)
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model_id,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        temperature=self.cfg.transcription_temperature,
                        max_output_tokens=self.cfg.transcription_max_output_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_level
                        ),
                    ),
                )
            except Exception as exc:
                last_err = str(exc)
                log.error(
                    "Transcription request failed (%s, attempt %d/%d): %s",
                    label, attempt + 1, retries + 1, exc,
                )
                # Brief back-off before retrying transient API errors
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return {"status": "error", "error": last_err, "text": ""}

            # Pro-only: dump every response (success OR empty) for debugging.
            _debug_dump_pro_response(
                resp,
                label=label,
                model_id=self.cfg.model_id,
                enabled=self.cfg.pro_debug,
                attempt=attempt + 1,
                total_attempts=retries + 1,
            )

            text = _extract_response_text(resp)
            if text:
                return {"status": "success", "text": text.strip()}

            # Empty response — log diagnostics and (maybe) retry.
            finish = _finish_reason(resp) or "UNKNOWN"
            last_err = (
                f"empty response (finish_reason={finish}); the model returned "
                "no visible text — likely thought-only output exhausted the "
                "token budget"
            )
            log.warning(
                "Transcription returned empty text (%s, attempt %d/%d, "
                "finish_reason=%s)",
                label, attempt + 1, retries + 1, finish,
            )
            if attempt < retries:
                # Drop thinking one level on retry to leave more budget for
                # actual output. high → medium → low. Pro doesn't support
                # "minimal", so "low" is our floor on Pro.
                thinking_level = {
                    "high": "medium",
                    "medium": "low",
                    "low": "low",
                    "minimal": "minimal",
                }.get(thinking_level, "medium")
                # On the FINAL retry, also escalate media_resolution to
                # ultra_high (if not already there) — this gives the model
                # maximum visual signal as a last-ditch attempt and
                # demonstrably helps Pro stop guessing "[illegible]" on
                # crops it could have read with more pixels.
                will_be_final = (attempt + 1) == retries
                escalated_media = current_media_res
                if (
                    will_be_final
                    and current_media_res
                    and current_media_res not in ("ultra_high",
                                                  "media_resolution_ultra_high")
                ):
                    escalated_media = "ultra_high"
                log.info(
                    "Retrying %s with thinking_level=%s%s",
                    label, thinking_level,
                    (f", media_resolution={escalated_media}"
                     if escalated_media != current_media_res else ""),
                )
                current_media_res = escalated_media
                continue

        return {"status": "error", "error": last_err or "empty response", "text": ""}
