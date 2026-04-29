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
REGION_TYPES: List[str] = [
    "TitleRegion",          # stand-alone heading / document title
    "TableRegion",          # tabular data (most common content type)
    "GraphRegion",          # charts, scatter plots, curve diagrams
    "ParagraphRegion",      # free prose / narrative text
    "MarginaliaRegion",     # side annotations in a different hand
    "FootnoteRegion",       # bottom-of-page notes
    "PageNumberRegion",     # printed or stamped page number
    "ImageRegion",          # drawings, maps, sketches
]

# Region types that contain transcribable running text
TEXT_REGION_TYPES = {"ParagraphRegion", "FootnoteRegion", "MarginaliaRegion", "TitleRegion"}

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
    transcription_thinking: str = "medium"   # enough reasoning for complex table cells
    transcription_max_attempts: int = 2      # retry once on truncated / leaked output
    transcription_max_output_tokens: int = 32768          # first attempt
    transcription_max_output_tokens_retry: int = 65536    # bigger budget on retry

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

    def ensure_dirs(self) -> None:
        """Create output sub-directories."""
        for sub in ("pages", "regions", "md"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
