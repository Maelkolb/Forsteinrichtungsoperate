"""Region detection using Gemini for historical German administrative documents.

Detects layout regions in single-page scans of 19th-century handwritten records
(forest inventory tables, grazing ledgers, tree-height graphs, etc.) and returns
bounding boxes with rich metadata for the transcription stage.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import REGION_TYPES, PipelineConfig
from .utils import MIME_BY_EXT, clean_llm_json

log = logging.getLogger(__name__)


# ── Prompt ───────────────────────────────────────────────────────────────────

DETECTION_PROMPT = """\
You are a document-layout analyst specialising in 19th-century Bavarian and German
handwritten administrative records (forestry, land management, agriculture).

SOURCE TYPE
These scans come from official registers, ledgers, and survey books, typically
featuring:
  • Dense multi-column tables with 2–4 rows of hierarchical headers written in
    German Kurrent (Kurrentschrift) or Sütterlin script
  • Mixed black ink (main entries) and red ink (corrections, totals, annotations)
  • Numeric data: integer counts, decimal values, currency amounts
  • Occasional stand-alone document titles or section headings above tables
  • Scatter-plot / curve diagrams drawn on graph paper (tree-height curves, etc.)
  • Marginal annotations or footnotes added later in a different hand

TASK
Identify every distinct content region on this page and return precise bounding
boxes in the JSON schema below.

KEY RULES
1. Maximum {max_regions} regions per page.  Merge spatially adjacent blocks of
   the same type into one region rather than creating many small ones.
2. ALWAYS treat the full table — including all header rows — as ONE TableRegion.
   Do NOT separate table headers from table body.
3. Tight bounding boxes; no overlaps; reading order = top-to-bottom,
   left-to-right.
4. A stand-alone title line above a table is a separate TitleRegion.
5. Marginal notes written sideways or in a clearly different hand are
   MarginaliaRegion even if adjacent to a table.

COORDINATE SYSTEM
Normalised 0–1000 scale.  (0, 0) = top-left, (1000, 1000) = bottom-right.

bbox format — use EXACTLY these four keys, no others:
  {{"x": <int 0-1000>, "y": <int 0-1000>,
    "width": <int 1-1000>, "height": <int 1-1000>}}

Do NOT use corner notation. Do NOT emit "x1", "y1", "x2", "y2", "xmin",
"ymin", "xmax", "ymax", or any other key. Do NOT emit two keys with the
same name (e.g. two "x" keys, two "y" keys). The four keys above are
mandatory and exhaustive.

Example of a CORRECT bbox: {{"x": 100, "y": 250, "width": 800, "height": 400}}

REGION TYPES (use exactly these names):
TitleRegion · TableRegion · GraphRegion · ParagraphRegion ·
MarginaliaRegion · FootnoteRegion · PageNumberRegion

Note: Hand-drawn maps, parcel diagrams, and similar map-like figures should
NOT appear here — they are processed via a dedicated single-pass workflow
(``--doc-type map``). On a regular text/mixed page you will normally see
only the seven region types above.

EXTRA METADATA (always include where applicable):
• TitleRegion      → "title_text": <transcribed title string>
• TableRegion      → "rows": <int>, "cols": <int>,
                     "header_rows": <number of header/sub-header rows, int>,
                     "has_red_ink": <true|false>,
                     "has_totals_row": <true|false>
• GraphRegion      → "graph_type": <"scatter"|"curve"|"bar"|"other">,
                     "has_fitted_curve": <true|false>
• ParagraphRegion  → "line_count": <int>, "script": <"kurrent"|"latin"|"mixed">
• MarginaliaRegion → "line_count": <int>
• FootnoteRegion   → "line_count": <int>
• PageNumberRegion → "page_number": <int or string>

OUTPUT (JSON only, no commentary, no markdown fences):
{{"regions": [
  {{"id": "r1", "type": "…", "bbox": {{"x":…,"y":…,"width":…,"height":…}},
    "reading_order": 1, …metadata…}},
  …
], "total_regions": N}}
"""


# Strict JSON schema for the detection response. Pro on the looser prompt
# was emitting bboxes in corner notation with duplicate keys (e.g.
# `{"x": 395, "y": 28, "x1": 526, "y": 55}`), which after json.loads
# collapses to `{"x": 395, "y": 55, "x1": 526}` — losing y_min entirely
# and leaving width/height unset, so the downstream parser fell back to
# its 100×50 default and every region came out the same size.
#
# Passing this schema via ``response_json_schema`` constrains Gemini's
# token-by-token generation to a structure where:
#   • the bbox object cannot have additional keys (no "x1", "y1", …)
#   • all four of x, y, width, height are required
# which makes the duplicate-key failure mode literally impossible to
# emit. Tested working in the v1alpha API used by this pipeline.
def _build_detection_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":   {"type": "string"},
                        "type": {"type": "string", "enum": list(REGION_TYPES)},
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "x":      {"type": "integer",
                                           "minimum": 0, "maximum": 1000},
                                "y":      {"type": "integer",
                                           "minimum": 0, "maximum": 1000},
                                "width":  {"type": "integer",
                                           "minimum": 1, "maximum": 1000},
                                "height": {"type": "integer",
                                           "minimum": 1, "maximum": 1000},
                            },
                            "required": ["x", "y", "width", "height"],
                            "additionalProperties": False,
                        },
                        "reading_order": {"type": "integer"},
                        # ── optional, type-specific metadata fields ──
                        "title_text":       {"type": "string"},
                        "rows":             {"type": "integer"},
                        "cols":             {"type": "integer"},
                        "header_rows":      {"type": "integer"},
                        "has_red_ink":      {"type": "boolean"},
                        "has_totals_row":   {"type": "boolean"},
                        "graph_type":       {"type": "string"},
                        "has_fitted_curve": {"type": "boolean"},
                        "line_count":       {"type": "integer"},
                        "script":           {"type": "string"},
                        "page_number":      {"anyOf": [
                            {"type": "string"}, {"type": "integer"},
                        ]},
                    },
                    "required": ["type", "bbox", "reading_order"],
                },
            },
            "total_regions": {"type": "integer"},
        },
        "required": ["regions"],
    }


DETECTION_SCHEMA: Dict[str, Any] = _build_detection_schema()


class RegionDetector:
    """Detect and classify layout regions with Gemini."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    # ── margin helper ────────────────────────────────────────────────────

    @staticmethod
    def _add_margin(
        bbox: Dict, img_w: int, img_h: int, margin_frac: float
    ) -> Dict[str, int]:
        mx = int(img_w * margin_frac)
        my = int(img_h * margin_frac)
        x = max(0, bbox["x"] - mx)
        y = max(0, bbox["y"] - my)
        w = min(bbox["width"] + 2 * mx, img_w - x)
        h = min(bbox["height"] + 2 * my, img_h - y)
        return {"x": x, "y": y, "width": w, "height": h}

    # ── main entry point ─────────────────────────────────────────────────

    def detect(self, image_path: Path) -> Dict[str, Any]:
        """Run region detection on a single page image.

        Returns a dict with keys:
            status, image_path, image_dimensions, regions, total_regions,
            reading_order, region_types_detected
        """
        from google.genai import types

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        ext = image_path.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "image/png")
        img_bytes = image_path.read_bytes()

        prompt = DETECTION_PROMPT.format(max_regions=self.cfg.max_regions)

        raw = ""
        last_exc = None
        thinking_level = self.cfg.detection_thinking
        for attempt in range(self.cfg.detection_retries + 1):
            try:
                # Local import to avoid pulling Transcriber at module load time.
                from .transcriber import (
                    _make_image_part,
                    _debug_dump_pro_response,
                )
                image_part = _make_image_part(
                    img_bytes, mime, self.cfg.detection_media_resolution,
                )
                resp = self.client.models.generate_content(
                    model=self.cfg.model_id,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        temperature=self.cfg.detection_temperature,
                        max_output_tokens=self.cfg.transcription_max_output_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_level
                        ),
                        # Force the bbox schema so Pro can't emit corner
                        # notation with duplicate "x"/"y" keys (see notes
                        # next to DETECTION_SCHEMA).
                        response_mime_type="application/json",
                        response_json_schema=DETECTION_SCHEMA,
                    ),
                )
                # Pro-only: dump full response for debugging.
                _debug_dump_pro_response(
                    resp,
                    label="RegionDetect",
                    model_id=self.cfg.model_id,
                    enabled=self.cfg.pro_debug,
                    attempt=attempt + 1,
                    total_attempts=self.cfg.detection_retries + 1,
                )
                # response.text can be None on Gemini 3.1 Pro when high-budget
                # thinking exhausts the output allowance before any visible
                # token is emitted — fall back to walking candidate parts.
                from .transcriber import _extract_response_text  # local import to avoid cycle
                resp_text = _extract_response_text(resp)
                if not resp_text:
                    last_exc = RuntimeError(
                        "empty detection response — model returned no visible text"
                    )
                    log.warning(
                        "Detection returned empty response for %s (attempt %d/%d)",
                        image_path.name, attempt + 1, self.cfg.detection_retries + 1,
                    )
                    # Drop thinking one notch on retry so more budget goes to output.
                    thinking_level = {
                        "high": "medium", "medium": "low", "low": "low",
                    }.get(thinking_level, "medium")
                    if attempt < self.cfg.detection_retries:
                        continue
                    return self._error(image_path, str(last_exc), "")
                raw = clean_llm_json(resp_text)
                data = json.loads(raw)
                break  # success
            except json.JSONDecodeError as exc:
                last_exc = exc
                if attempt < self.cfg.detection_retries:
                    log.warning(
                        "JSON parse failed for %s (attempt %d/%d), retrying…",
                        image_path.name, attempt + 1, self.cfg.detection_retries + 1,
                    )
                    continue
                log.error(
                    "JSON parse error for %s: %s\nRaw: %s",
                    image_path.name, exc, raw[:400],
                )
                return self._error(image_path, f"JSON parse: {exc}", raw)
            except Exception as exc:
                log.error("Detection failed for %s: %s", image_path.name, exc)
                return self._error(image_path, str(exc), traceback.format_exc())

        regions = self._validate(data.get("regions", []), w, h)
        return {
            "status": "success",
            "image_path": str(image_path),
            "image_dimensions": {"width": w, "height": h},
            "regions": regions,
            "total_regions": len(regions),
            "reading_order": [r["id"] for r in regions],
            "region_types_detected": sorted(set(r["type"] for r in regions)),
        }

    # ── validation / normalisation ───────────────────────────────────────

    @staticmethod
    def _normalise_bbox(bbox: Any) -> Optional[Dict[str, float]]:
        """Coerce any plausible bbox shape into ``{x, y, width, height}``.

        Designed to recover from formats other than the canonical
        ``{x, y, width, height}`` schema we ask for, including:
          • Gemini's native list form ``[ymin, xmin, ymax, xmax]``
          • Corner notation ``{x, y, x1, y1}``
          • Corner notation ``{x_min, y_min, x_max, y_max}`` /
            ``{xmin, ymin, xmax, ymax}``
          • The malformed-duplicate-key form Pro produced earlier:
            ``{"x": 395, "y": 28, "x1": 526, "y": 55}`` collapses through
            ``json.loads`` to ``{"x": 395, "x1": 526, "y": 55}``, which
            this function recovers as a degenerate-but-usable bbox via
            the corner-notation branch (with the lost y_min defaulted
            from the surviving ``y``).
        Returns None for inputs that are clearly unusable.
        """
        # ── list-form: [ymin, xmin, ymax, xmax] (Gemini's native bbox) ──
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                a, b, c, d = (float(v) for v in bbox)
            except (TypeError, ValueError):
                return None
            # Gemini emits y first, x second
            ymin, xmin, ymax, xmax = a, b, c, d
            if xmax > xmin and ymax > ymin:
                return {"x": xmin, "y": ymin,
                        "width": xmax - xmin, "height": ymax - ymin}
            return None

        if not isinstance(bbox, dict):
            return None

        def f(*names: str) -> Optional[float]:
            for n in names:
                if n in bbox:
                    try:
                        return float(bbox[n])
                    except (TypeError, ValueError):
                        return None
            return None

        x = f("x", "left", "x_min", "xmin", "x1")
        y = f("y", "top",  "y_min", "ymin", "y1")
        w = f("width",  "w")
        h = f("height", "h")
        x_max = f("x_max", "xmax", "x2", "right",  "x1")  # x1 also valid as x_max
        y_max = f("y_max", "ymax", "y2", "bottom", "y1")  # y1 also valid as y_max

        # Direct (preferred) form
        if x is not None and y is not None and w is not None and h is not None:
            return {"x": x, "y": y, "width": w, "height": h}

        # Corner form (recovered)
        if (x is not None and y is not None
                and x_max is not None and y_max is not None
                and x_max > x and y_max > y):
            return {"x": x, "y": y,
                    "width":  x_max - x,
                    "height": y_max - y}
        return None

    def _validate(
        self, raw_regions: List[Dict], img_w: int, img_h: int
    ) -> List[Dict]:
        out: List[Dict] = []
        n_recovered = 0
        n_dropped = 0
        for i, r in enumerate(raw_regions[: self.cfg.max_regions]):
            rtype = self._normalise_type(r.get("type", "TableRegion"))
            raw_bbox = r.get("bbox")
            bbox = self._normalise_bbox(raw_bbox)
            if bbox is None:
                # Loud failure rather than a silent 100×50 default — the
                # old code masked the duplicate-key bug by inventing a
                # plausible-looking-but-wrong box for every malformed
                # region.
                log.warning(
                    "Region %d (%s): unusable bbox %r — dropping.",
                    i, rtype, raw_bbox,
                )
                n_dropped += 1
                continue

            # If the model supplied corner notation we accepted via the
            # fallback, log it once at INFO so the user knows their model
            # is drifting from the requested schema.
            if (isinstance(raw_bbox, dict)
                    and ("width" not in raw_bbox or "height" not in raw_bbox)):
                n_recovered += 1

            # normalised 0-1000 → pixel coordinates
            nx, ny = bbox["x"], bbox["y"]
            nw, nh = bbox["width"], bbox["height"]
            px = max(0, min(int(nx * img_w / 1000), img_w - 1))
            py = max(0, min(int(ny * img_h / 1000), img_h - 1))
            pw = max(10, min(int(nw * img_w / 1000), img_w - px))
            ph = max(10, min(int(nh * img_h / 1000), img_h - py))
            pixel_bbox = self._add_margin(
                {"x": px, "y": py, "width": pw, "height": ph},
                img_w, img_h, self.cfg.region_margin_frac,
            )

            entry: Dict[str, Any] = {
                "id": "",          # set after sorting
                "type": rtype,
                "bbox": pixel_bbox,
                "reading_order": int(r.get("reading_order", i + 1)),
            }

            # Carry through all type-specific metadata that was returned
            _passthrough = [
                "title_text",
                "rows", "cols", "header_rows", "has_red_ink", "has_totals_row",
                "graph_type", "has_fitted_curve",
                "line_count", "script",
                "page_number",
            ]
            for key in _passthrough:
                if key in r:
                    entry[key] = r[key]

            out.append(entry)

        if n_recovered:
            log.info(
                "Recovered %d region bbox(es) from non-canonical schema "
                "(corner notation or duplicate keys). The model is drifting "
                "from the requested {x,y,width,height} schema.",
                n_recovered,
            )
        if n_dropped:
            log.warning(
                "Dropped %d region(s) with unparseable bboxes — see warnings "
                "above. Consider re-running with --doc-type table or graph if "
                "the page is dominated by one element.",
                n_dropped,
            )

        # deterministic reading order
        out.sort(key=lambda r: r["reading_order"])
        for idx, region in enumerate(out):
            region["id"] = f"r{idx + 1:02d}"
            region["reading_order"] = idx + 1

        return out

    @staticmethod
    def _normalise_type(raw: str) -> str:
        """Map a potentially misspelled type to the canonical name."""
        if raw in REGION_TYPES:
            return raw
        low = raw.lower()
        for valid in REGION_TYPES:
            if low in valid.lower() or valid.lower() in low:
                return valid
        # Reasonable fallback for unlabelled text blocks
        return "ParagraphRegion"

    # ── error helper ─────────────────────────────────────────────────────

    @staticmethod
    def _error(path: Path, msg: str, detail: str = "") -> Dict[str, Any]:
        return {
            "status": "error",
            "image_path": str(path),
            "error": msg,
            "detail": detail[:500],
            "regions": [],
            "total_regions": 0,
        }
