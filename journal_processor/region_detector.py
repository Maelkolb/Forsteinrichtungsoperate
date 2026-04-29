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
bbox format: {{"x": left, "y": top, "width": w, "height": h}}

REGION TYPES (use exactly these names):
TitleRegion · TableRegion · GraphRegion · ParagraphRegion ·
MarginaliaRegion · FootnoteRegion · PageNumberRegion · ImageRegion

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
        for attempt in range(self.cfg.detection_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model_id,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.cfg.detection_temperature,
                        max_output_tokens=16384,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.cfg.detection_thinking
                        ),
                    ),
                )
                raw = clean_llm_json(resp.text)
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

    def _validate(
        self, raw_regions: List[Dict], img_w: int, img_h: int
    ) -> List[Dict]:
        out: List[Dict] = []
        for i, r in enumerate(raw_regions[: self.cfg.max_regions]):
            rtype = self._normalise_type(r.get("type", "TableRegion"))
            bbox = r.get("bbox", {})

            # normalised 0-1000 → pixel coordinates
            nx, ny = float(bbox.get("x", 0)), float(bbox.get("y", 0))
            nw, nh = float(bbox.get("width", 100)), float(bbox.get("height", 50))
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
