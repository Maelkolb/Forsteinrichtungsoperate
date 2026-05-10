"""Configuration for the historical document processing pipeline.

Tuned for 19th-century German handwritten administrative / forestry records:
  • Complex multi-column tables (Hochwald inventory, Weidenutzung, etc.)
  • Kurrent / Sütterlin script, mixed black + red ink
  • Graph / scatter-plot pages on graph paper
  • Single-page scans (no double-page splitting needed by default)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Region taxonomy
# ---------------------------------------------------------------------------
# NOTE: ``ImageRegion`` was removed in v4 — the detector tended to mis-classify
# graphs / sketches / dense text blocks as ``ImageRegion`` and the prompt was
# description-first, which gave the user descriptions instead of transcriptions.
# Maps now have their own dedicated full-page workflow (``--doc-type map``).
REGION_TYPES: List[str] = [
    "TitleRegion",          # stand-alone heading / document title
    "TableRegion",          # tabular data (most common content type)
    "GraphRegion",          # charts, scatter plots, curve diagrams
    "ParagraphRegion",      # free prose / narrative text
    "MarginaliaRegion",     # side annotations in a different hand
    "FootnoteRegion",       # bottom-of-page notes
    "PageNumberRegion",     # printed or stamped page number
]

# Region types that contain transcribable running text
TEXT_REGION_TYPES = {"ParagraphRegion", "FootnoteRegion", "TitleRegion"}

# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------
# Source-document categories that determine which pipeline branch is used.
#
#   "text"   → text-heavy pages (free narrative, annotation pages).
#              Uses the full layout pipeline (region detection + per-region
#              transcription) — same flow as "mixed" but a clearer label
#              when the user knows the page is mostly running text.
#
#   "table"  → pages dominated by ONE big complex table, possibly with
#              surrounding text (titles, footnotes). Skips region detection
#              and uses a single-pass full-page call to the model.
#              Output: HTML <table> + any extra text outside the table.
#
#   "graph"  → pages with one big graph (typically a tree-height curve on
#              graph paper). Skips region detection and uses a single-pass
#              full-page call tailored to forestry height-curve graphs.
#
#   "map"    → pages dominated by ONE map (cadastral / forestry / parcel /
#              topographic / sketch). Skips region detection and uses a
#              single-pass full-page call tailored to maps. Extracts a
#              structured set of metadata fields (type, title, geoident,
#              scale, date, …) plus all visible text.
#
#   "mixed"  → diverse / unknown documents. Uses the full layout pipeline
#              with region detection (the original behaviour).
DOC_TYPES: List[str] = ["text", "table", "graph", "map", "mixed"]
DEFAULT_DOC_TYPE: str = "mixed"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable knobs live here."""

    # I/O paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")

    # Model
    model_id: str = MODEL_ID

    # Source layout
    single_page: bool = True            # True  → no double-page split (most historical sources)
                                        # False → split down the centre (double-page spreads)
    split_overlap_px: int = 10

    # Document type — selects the processing branch.
    # "mixed" (default) and "text" use region detection. "table" and "graph"
    # use a single-pass full-page call without region detection.
    doc_type: str = DEFAULT_DOC_TYPE

    # Pre-processing
    deskew: bool = False                # optional deskew (enable for skewed scans)
    enhance_contrast: bool = True       # CLAHE-style contrast boost – ON by default for aged paper
    sharpen: bool = True                # gentle unsharp-mask to improve ink legibility

    # Region detection
    max_regions: int = 12               # historical pages often have more distinct regions
    region_margin_frac: float = 0.003   # very tight margin – avoid grabbing neighbour cells
    detection_temperature: float = 1.0  # Gemini 3 docs: keep at default 1.0 to avoid degradation
    detection_thinking: str = "high"    # more reasoning for complex multi-column layouts
    detection_retries: int = 3          # extra retry budget for tricky pages

    # Transcription
    transcription_temperature: float = 1.0   # Gemini 3 docs: keep at default 1.0
    transcription_thinking: str = "low"      # PERCEPTION TASK — keep low.
                                             # Pro at "medium" / "high" spends
                                             # its budget debating Kurrent
                                             # ambiguities and either emits 0
                                             # parts or a tiny fragment ("Zur",
                                             # "Lindenberg un", "[illegible]").
                                             # At "low" Pro typically does not
                                             # think at all (thoughts=None) and
                                             # just transcribes — exactly the
                                             # behaviour we want from Flash.
    transcription_retries: int = 2           # extra attempts on empty/failed responses
                                             # (per-region calls); each retry drops
                                             # thinking_level one notch to free up
                                             # output budget.

    # Full-page (single-pass) modes used by doc_type="table" and "graph".
    # These get a slightly larger thinking budget because the model has to
    # reason about an entire page in one call.
    full_page_thinking: str = "high"
    full_page_retries: int = 2

    # Output token budget for transcription / full-page calls.
    # Gemini 3 supports up to 64k output tokens (65536). High thinking can
    # consume a large share of that budget before producing visible text,
    # so we now request the full ceiling: this leaves the most room for
    # the visible transcription after dynamic thinking. Particularly
    # important for Gemini 3.1 Pro on the "text" / "mixed" workflows,
    # where empty responses were observed when the budget was tight.
    transcription_max_output_tokens: int = 65536

    # ── Media resolution (Gemini 3 v1alpha) ─────────────────────────────
    # Controls the maximum number of input tokens the model may allocate to
    # each image. Higher values give the model more visual detail at the
    # cost of input tokens.
    #   "media_resolution_low"        →  280 tokens / image
    #   "media_resolution_medium"     →  560 tokens / image
    #   "media_resolution_high"       → 1120 tokens / image (recommended)
    #   "media_resolution_ultra_high" → max detail (graphs, fine markings)
    # If empty string, the SDK default is used.
    #
    # NOTE: Region crops are small images of dense Kurrent text. Without an
    # explicit setting the SDK picks a default that's apparently too low
    # for handwriting, which on Pro pushes the model into "I'm uncertain,
    # let me think more" loops that then fail. "high" is the docs'
    # recommended setting for image analysis and is what makes Pro behave
    # like Flash on transcription.
    region_media_resolution: str = "high"                # per-region transcription
    full_page_media_resolution: str = "high"             # full-page table / map
    graph_media_resolution: str = "ultra_high"           # graph (smallest features)
    detection_media_resolution: str = "high"             # layout detection

    # ── Debug ──────────────────────────────────────────────────────────
    # When the model_id is "gemini-3.1-pro-preview", dump the full Gemini
    # response (finish_reason, usage, every part — text and thought) to
    # the log after every call. Helpful for diagnosing empty responses and
    # other Pro-specific behaviour. Other model_ids are unaffected.
    pro_debug: bool = True

    # GLM-OCR (disabled – Gemini handles all regions)
    use_glm_ocr: bool = False
    glm_ocr_base_model: str = "zai-org/GLM-OCR"
    glm_ocr_lora_path: str = ""
    glm_ocr_max_new_tokens: int = 2048

    # Output formats – only Markdown enabled
    output_md: bool = True
    output_pagexml: bool = False
    output_sharegpt: bool = False

    # Concurrency
    workers: int = 4

    def __post_init__(self) -> None:
        if self.doc_type not in DOC_TYPES:
            raise ValueError(
                f"Invalid doc_type {self.doc_type!r}. "
                f"Must be one of: {', '.join(DOC_TYPES)}"
            )

    @property
    def uses_region_detection(self) -> bool:
        """True for doc_types that go through the layout pipeline."""
        return self.doc_type in {"text", "mixed"}

    def ensure_dirs(self) -> None:
        """Create output sub-directories."""
        for sub in ("pages", "regions", "md"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
