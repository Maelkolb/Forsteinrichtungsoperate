"""Prepare page images for the processing pipeline.

Two modes, controlled by ``cfg.single_page``:

  single_page=True  (default)
      Each input file is one page.  Images are simply copied (converted to
      PNG) into the pages directory without any cropping.

  single_page=False
      Input files are double-page spreads.  Each is split down the centre
      into left and right half-pages.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .config import PipelineConfig
from .utils import natural_sort_key

log = logging.getLogger(__name__)

_SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# ── Single-page passthrough ──────────────────────────────────────────────────

def copy_single_page(image_path: Path, output_dir: Path) -> Path:
    """Convert one scan to PNG and place it in *output_dir*."""
    img = Image.open(image_path).convert("RGB")
    out_path = output_dir / f"{image_path.stem}.png"
    img.save(out_path, "PNG")
    log.debug("Copied %s → %s", image_path.name, out_path.name)
    return out_path


# ── Double-page splitting ────────────────────────────────────────────────────

def split_double_page(
    image_path: Path,
    output_dir: Path,
    overlap_px: int = 10,
) -> Tuple[Path, Path]:
    """Crop a double-page spread down the centre.

    Returns ``(left_page_path, right_page_path)``.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mid = w // 2

    left = img.crop((0, 0, mid + overlap_px, h))
    right = img.crop((mid - overlap_px, 0, w, h))

    stem = image_path.stem
    left_path  = output_dir / f"{stem}_L.png"
    right_path = output_dir / f"{stem}_R.png"

    left.save(left_path, "PNG")
    right.save(right_path, "PNG")
    log.info("Split %s → %s, %s", image_path.name, left_path.name, right_path.name)
    return left_path, right_path


# ── Main entry point ─────────────────────────────────────────────────────────

def split_all(cfg: PipelineConfig) -> List[Path]:
    """Prepare all images in *cfg.input_dir* and return sorted page paths."""
    pages_dir = cfg.output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    scans = sorted(
        [p for p in cfg.input_dir.iterdir() if p.suffix.lower() in _SUPPORTED],
        key=natural_sort_key,
    )

    if not scans:
        log.warning("No images found in %s", cfg.input_dir)
        return []

    page_paths: List[Path] = []

    if cfg.single_page:
        for scan in scans:
            page_paths.append(copy_single_page(scan, pages_dir))
        log.info("Prepared %d single-page scan(s)", len(page_paths))
    else:
        for scan in scans:
            left, right = split_double_page(scan, pages_dir, cfg.split_overlap_px)
            page_paths.extend([left, right])
        log.info(
            "Split %d double-page scan(s) → %d pages",
            len(scans), len(page_paths),
        )

    return sorted(page_paths, key=natural_sort_key)
