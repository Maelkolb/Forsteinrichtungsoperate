#!/usr/bin/env python3
"""Build a single self-contained HTML viewer for pipeline output.

Pairs each Markdown file in ``<output>/md/`` with the matching scan in
``<output>/pages/``, compresses and embeds the scans as base64, renders the
Markdown (including the HTML tables produced for ``TableRegion`` and the
text-table descriptions for ``GraphRegion``), and writes one self-contained
HTML file you can open in any browser or hand off as an archive.

When the pipeline was run with a layout-detection workflow (``--doc-type
mixed`` or ``--doc-type text``), the regions JSON in ``<output>/regions/``
is also picked up and rendered as a coloured SVG overlay on top of the
scan, with a toggle button in the header to show/hide all overlays.

Usage examples:

    # Default (writes to <output>/viewer.html):
    python scripts/build_viewer.py output/

    # Custom output path:
    python scripts/build_viewer.py output/ -o reports/2024-batch.html

    # Tweak image compression:
    python scripts/build_viewer.py output/ --max-width 1400 --quality 75

    # External images (one image file alongside the HTML, not embedded) —
    # useful if you have hundreds of pages and a 200 MB single file
    # would be unwieldy:
    python scripts/build_viewer.py output/ --no-embed

Requires:  Pillow, markdown
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("This script requires `markdown` — install with: pip install markdown")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("This script requires `Pillow` — install with: pip install Pillow")

log = logging.getLogger("build_viewer")

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
_IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

# YAML-ish front matter at the top of every md file
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

# Fenced ```html ... ``` blocks (TableRegion output is wrapped in these).
_FENCED_HTML_RE = re.compile(r"```html\s*\n(.*?)\n```", re.DOTALL)

# Pipe-tables that follow prose without an intervening blank line.
# python-markdown's `tables` extension is strict — it needs that blank line.
# This re-insertion is a defensive fix; it leaves correct tables alone.
_PIPE_TABLE_FIX_RE = re.compile(
    r"(?P<prose>^(?!\s*$|\||#|>|-|\*|\d+\.).+\S.*\n)"   # a prose line
    r"(?P<table>\|[^\n]*\|\s*\n\|[\s:|\-]+\|)",         # header + separator
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class RegionBox:
    """One bounding box read from the regions JSON."""
    region_id: str
    region_type: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class PageEntry:
    """One page: its md, its scan, parsed metadata and rendered html."""
    md_path: Path
    image_path: Optional[Path]
    page_id: str
    page_number: str
    regions_summary: str
    md_html: str
    image_src: Optional[str]   # data: URI or relative path, depending on --embed
    # Overlay support — populated only for layout-mode pages (mixed / text)
    overlay_svg: str = ""             # rendered SVG, or "" if no regions
    overlay_region_count: int = 0     # number of boxes drawn (for stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _natural_sort_key(s: str) -> List:
    """Sort '_2_' before '_10_'."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", s)]


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Extract a ``key: value`` block from the file head.

    Returns (metadata-dict, body-text). If no frontmatter is found, the
    metadata dict is empty and the body is the original text.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[m.end():]


def _close_unmatched_table(html_block: str) -> str:
    """Inject ``</table>`` if the fenced block contains an unclosed table.

    The pipeline occasionally produces tables truncated mid-row. Without
    this fix the trailing markdown (e.g. marginalia) gets absorbed into
    the open HTML block and rendered as literal text.
    """
    if "<table" in html_block and "</table>" not in html_block:
        return html_block.rstrip() + "\n</table>"
    return html_block


def render_markdown(md_body: str) -> str:
    """Pipeline-flavoured Markdown → HTML.

    Pre-processing:
      1. If the document has an unclosed ``` ```html ... ``` ``` fence
         (upstream truncation), close it. The closing fence is inserted
         before any obviously-markdown content (blockquote/heading)
         that follows a blank line, so trailing marginalia survives.
      2. Replace fenced html blocks with their raw content; auto-close
         any unclosed <table> inside them.
      3. Ensure pipe tables that follow prose have the blank line the
         `tables` extension expects.
    """
    # 1 — close any unclosed ```html fence in a way that preserves
    #     trailing markdown (marginalia is sometimes inside the fence).
    md_body = _close_unclosed_html_fence(md_body)

    # 2 — substitute fenced html with raw html (closing any open <table>)
    md_body = _FENCED_HTML_RE.sub(
        lambda m: _close_unmatched_table(m.group(1)), md_body,
    )

    # 3 — pipe-table blank-line fix
    md_body = _PIPE_TABLE_FIX_RE.sub(
        lambda m: f"{m.group('prose')}\n{m.group('table')}",
        md_body,
    )

    html = markdown.markdown(
        md_body,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )
    # Wrap each <table> in a scrollable container so wide colspan tables
    # scroll horizontally instead of squeezing every column into one char.
    return re.sub(
        r"(<table\b[\s\S]*?</table>)",
        r'<div class="table-wrap">\1</div>',
        html,
    )


# Lines that strongly suggest the html fence has accidentally swallowed
# trailing markdown — used to find a sensible spot to close the fence.
_MD_OUTSIDE_FENCE_RE = re.compile(r"^(>|#{1,6}\s|---|\*\s|-\s|\d+\.\s)")


def _close_unclosed_html_fence(text: str) -> str:
    """Append a closing ``` for an unmatched ```html fence.

    Best-effort: if obvious markdown (blockquote/heading/etc.) appears
    after a blank line inside the open fence, close right before it so
    the markdown gets parsed normally.
    """
    if len(re.findall(r"^```", text, re.MULTILINE)) % 2 == 0:
        return text   # all fences paired

    lines = text.splitlines(keepends=True)
    open_idx = None
    for i, line in enumerate(lines):
        if line.startswith("```html"):
            open_idx = i
        elif line.startswith("```") and open_idx is not None:
            open_idx = None   # this fence was actually closed
    if open_idx is None:
        # odd fence count but not the html one — just append a close
        return text.rstrip() + "\n```\n"

    # walk from open_idx forward, looking for a blank line followed by
    # an obviously-markdown line. close the fence at that blank line.
    for i in range(open_idx + 1, len(lines) - 1):
        if lines[i].strip() == "" and _MD_OUTSIDE_FENCE_RE.match(lines[i + 1]):
            lines.insert(i, "```\n")
            return "".join(lines)
    # no obvious split point — close at end of document
    return text.rstrip() + "\n```\n"


def compress_image(
    src: Path,
    *,
    max_width: int,
    quality: int,
) -> bytes:
    """Open *src*, downscale if wider than *max_width*, return JPEG bytes."""
    img = Image.open(src)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def find_image_for(md_path: Path, pages_dir: Path) -> Optional[Path]:
    """Look for ``<pages_dir>/<md_stem>.<ext>`` for any supported ext."""
    stem = md_path.stem
    for ext in _IMG_EXTENSIONS:
        candidate = pages_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Regions overlay
# ---------------------------------------------------------------------------
# Distinct, accessible colours per region type. The first six are taken from
# a colour-blind-safe palette (Wong, 2011, doi:10.1038/nmeth.1618). The
# remaining ones are tuned for visual contrast against tan/cream paper.
_REGION_COLOURS: Dict[str, str] = {
    "TitleRegion":      "#0072B2",   # blue
    "TableRegion":      "#009E73",   # green
    "GraphRegion":      "#E69F00",   # orange
    "ParagraphRegion":  "#56B4E9",   # sky blue
    "MarginaliaRegion": "#CC79A7",   # magenta / purple
    "FootnoteRegion":   "#F0E442",   # yellow
    "PageNumberRegion": "#D55E00",   # vermillion
    "FullPage":         "#888888",   # grey (full-page modes — not normally drawn)
}
_DEFAULT_COLOUR = "#444444"


def find_regions_json_for(md_path: Path, regions_dir: Path) -> Optional[Path]:
    """Look for ``<regions_dir>/<md_stem>.json``."""
    candidate = regions_dir / f"{md_path.stem}.json"
    return candidate if candidate.is_file() else None


def load_region_boxes(regions_json: Path) -> List[RegionBox]:
    """Parse a per-page regions JSON and return the layout boxes.

    Returns an empty list for full-page outputs (which carry a single
    ``FullPage`` record with no bbox). Defensive against malformed JSON
    and missing fields — in either case we just return [] so the page
    still renders without an overlay.
    """
    try:
        data = json.loads(regions_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read regions JSON %s: %s", regions_json, exc)
        return []
    if not isinstance(data, list):
        return []

    out: List[RegionBox] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type", "")
        bbox = r.get("bbox")
        if rtype == "FullPage" or not isinstance(bbox, dict):
            # full-page-mode entries have no bbox — skip overlay for them
            continue
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["width"])
            h = int(bbox["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append(RegionBox(
            region_id=str(r.get("id", "")),
            region_type=str(rtype),
            x=x, y=y, width=w, height=h,
        ))
    return out


def build_overlay_svg(
    boxes: List[RegionBox],
    image_size: Tuple[int, int],
) -> str:
    """Build an SVG fragment that overlays bounding boxes on a page image.

    The viewBox is set to the original (pre-compression) image dimensions
    so the overlay scales correctly when CSS shrinks the image to fit the
    container — bbox coords from the regions JSON are pixel-coords in the
    original image.
    """
    if not boxes:
        return ""

    img_w, img_h = image_size
    parts = [
        f'<svg class="region-overlay" viewBox="0 0 {img_w} {img_h}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    ]
    # rect-then-label per region (label after rect so it draws on top)
    # Use a tiny outer halo + main stroke so the boxes stay legible even
    # over busy ink lines.
    for box in boxes:
        colour = _REGION_COLOURS.get(box.region_type, _DEFAULT_COLOUR)
        # halo (white) for legibility
        parts.append(
            f'<rect class="rg-halo" x="{box.x}" y="{box.y}" '
            f'width="{box.width}" height="{box.height}"/>'
        )
        parts.append(
            f'<rect class="rg-box rg-{escape(box.region_type, quote=True)}" '
            f'x="{box.x}" y="{box.y}" '
            f'width="{box.width}" height="{box.height}" '
            f'stroke="{colour}"/>'
        )

    # Labels in a second pass so they sit on top of every rect
    for box in boxes:
        colour = _REGION_COLOURS.get(box.region_type, _DEFAULT_COLOUR)
        # Label position: top-left of the box, slightly inset.
        # Background pill behind the text for legibility.
        label_text = (
            f"{escape(box.region_id)}·{escape(box.region_type)}"
            if box.region_id else escape(box.region_type)
        )
        # Font size scales with the image so labels stay readable at any zoom.
        font_size = max(10, min(img_w, img_h) // 70)
        pad_x = font_size // 2
        pad_y = font_size // 3
        # Use approximate width estimate (0.6 × font_size per char) — close
        # enough for a tag pill.
        text_w = int(len(label_text) * font_size * 0.55)
        label_w = text_w + 2 * pad_x
        label_h = font_size + 2 * pad_y
        # Place label at the top of the box, slightly above the corner.
        lx = box.x
        ly = max(0, box.y - label_h - 2)
        # If the box is at the very top of the page, push label inside it.
        if ly < 2:
            ly = box.y + 2
        parts.append(
            f'<rect class="rg-label-bg" x="{lx}" y="{ly}" '
            f'width="{label_w}" height="{label_h}" fill="{colour}" />'
        )
        parts.append(
            f'<text class="rg-label" x="{lx + pad_x}" '
            f'y="{ly + label_h - pad_y - 1}" '
            f'font-size="{font_size}" fill="#fff">{label_text}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Return original (W, H) of the image, or None on failure."""
    try:
        with Image.open(image_path) as im:
            return im.size
    except Exception as exc:
        log.warning("Could not read image dimensions of %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
def build_pages(
    md_dir: Path,
    pages_dir: Path,
    regions_dir: Optional[Path],
    *,
    embed: bool,
    assets_dir: Optional[Path],
    max_width: int,
    quality: int,
) -> List[PageEntry]:
    md_files = sorted(md_dir.glob("*.md"), key=lambda p: _natural_sort_key(p.name))
    if not md_files:
        log.warning("No .md files found in %s", md_dir)
        return []

    pages: List[PageEntry] = []
    for i, md_path in enumerate(md_files, 1):
        text = md_path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        body_html = render_markdown(body)

        img_path = find_image_for(md_path, pages_dir)
        image_src: Optional[str] = None
        if img_path is not None:
            try:
                jpeg_bytes = compress_image(img_path, max_width=max_width, quality=quality)
                if embed:
                    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                    image_src = f"data:image/jpeg;base64,{b64}"
                else:
                    assert assets_dir is not None
                    out_img = assets_dir / f"{md_path.stem}.jpg"
                    out_img.write_bytes(jpeg_bytes)
                    image_src = f"{assets_dir.name}/{out_img.name}"
            except Exception as exc:
                log.error("Could not process image %s: %s", img_path, exc)
                image_src = None
        else:
            log.info("[%d/%d] %s — no matching image in %s",
                     i, len(md_files), md_path.name, pages_dir)

        # ── Regions overlay (only for layout-mode pages with an image) ──
        overlay_svg = ""
        n_boxes = 0
        if regions_dir is not None and img_path is not None:
            regions_json = find_regions_json_for(md_path, regions_dir)
            if regions_json is not None:
                boxes = load_region_boxes(regions_json)
                if boxes:
                    img_size = get_image_dimensions(img_path)
                    if img_size is not None:
                        overlay_svg = build_overlay_svg(boxes, img_size)
                        n_boxes = len(boxes)

        pages.append(PageEntry(
            md_path=md_path,
            image_path=img_path,
            page_id=meta.get("page_id", md_path.stem),
            page_number=meta.get("page_number", ""),
            regions_summary=meta.get("regions", ""),
            md_html=body_html,
            image_src=image_src,
            overlay_svg=overlay_svg,
            overlay_region_count=n_boxes,
        ))
        log.info("[%d/%d] %s ✓%s", i, len(md_files), md_path.name,
                 f" ({n_boxes} regions)" if n_boxes else "")

    return pages


# ---------------------------------------------------------------------------
# HTML emission
# ---------------------------------------------------------------------------
_CSS = r"""
:root {
  --bg: #f8f6f1;
  --bg-sidebar: #edeae3;
  --bg-card: #ffffff;
  --fg: #2a2520;
  --fg-muted: #6b6359;
  --accent: #8b3a2f;
  --border: #d8d2c5;
  --border-soft: #e8e3d8;
  --radius: 4px;
  --shadow: 0 1px 2px rgba(0,0,0,.04), 0 4px 12px rgba(0,0,0,.04);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  font-size: 15px;
  line-height: 1.55;
}
header.top {
  position: sticky; top: 0; z-index: 5;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex; align-items: baseline; gap: 16px;
}
header.top h1 {
  margin: 0; font-size: 17px; font-weight: 600; letter-spacing: .01em;
}
header.top .meta {
  color: var(--fg-muted); font-size: 13px;
}
.layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: calc(100vh - 50px);
}
aside.sidebar {
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  padding: 14px 0;
  position: sticky; top: 50px;
  height: calc(100vh - 50px);
  overflow-y: auto;
  align-self: start;
}
aside.sidebar .filter {
  margin: 0 14px 10px;
}
aside.sidebar input[type=search] {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
  font: inherit; font-size: 13px;
  color: var(--fg);
}
aside.sidebar input[type=search]:focus {
  outline: 2px solid var(--accent); outline-offset: -1px;
  border-color: var(--accent);
}
ol.page-list {
  list-style: none; padding: 0; margin: 0;
}
ol.page-list li a {
  display: flex; gap: 10px; align-items: baseline;
  padding: 7px 14px;
  text-decoration: none;
  color: var(--fg);
  border-left: 3px solid transparent;
  font-size: 13px;
}
ol.page-list li a:hover { background: rgba(139,58,47,.05); }
ol.page-list li a.active {
  background: rgba(139,58,47,.10);
  border-left-color: var(--accent);
  color: var(--accent);
  font-weight: 500;
}
ol.page-list .num {
  color: var(--fg-muted);
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  min-width: 28px;
}
ol.page-list .name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  word-break: break-all;
}
main {
  padding: 24px;
  max-width: 1600px;
}
article.page {
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  margin-bottom: 28px;
  box-shadow: var(--shadow);
  scroll-margin-top: 70px;
}
article.page > header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border-soft);
  display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: baseline;
}
article.page > header h2 {
  margin: 0; font-size: 15px; font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
article.page > header .pagenum {
  background: var(--bg-sidebar);
  color: var(--fg-muted);
  padding: 2px 8px; border-radius: 999px;
  font-size: 12px;
}
article.page > header .regions {
  color: var(--fg-muted); font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.page-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 0;
}
.scan {
  background: #ece7dc;
  border-right: 1px solid var(--border-soft);
  padding: 16px;
  display: flex; align-items: flex-start; justify-content: center;
  position: sticky; top: 60px;
  align-self: start;
  max-height: calc(100vh - 80px);
  overflow: auto;
}
.scan .frame {
  /* Tight wrapper around the image so the SVG overlay matches the
     displayed image bounds exactly. */
  position: relative;
  display: inline-block;
  max-width: 100%;
}
.scan img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
  background: #fff;
  cursor: zoom-in;
  display: block;
}
.scan .placeholder {
  color: var(--fg-muted); font-style: italic; padding: 40px 12px;
  text-align: center; font-size: 13px;
}

/* Region overlay (layout-mode pages only). The SVG is layered exactly
   over the displayed image; viewBox carries the original-image pixel
   dimensions so bbox coords from the regions JSON align correctly. */
.region-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  /* same border as .scan img so the overlay exactly tracks the image
     content area */
  border: 1px solid transparent;
}
.region-overlay .rg-halo {
  fill: none;
  stroke: rgba(255, 255, 255, .85);
  stroke-width: 6;
  stroke-linejoin: round;
}
.region-overlay .rg-box {
  fill: none;
  stroke-width: 3;
  stroke-linejoin: round;
}
.region-overlay .rg-label {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-weight: 600;
  letter-spacing: .02em;
  paint-order: stroke;
  stroke: rgba(0,0,0,.25);
  stroke-width: 1;
}
.region-overlay .rg-label-bg {
  rx: 2; ry: 2;
}
/* Hidden when the user toggles overlays off via the header button */
body.hide-regions .region-overlay { display: none; }

/* Header toggle button */
header.top button.regions-toggle {
  margin-left: auto;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 4px 12px;
  font: inherit;
  font-size: 13px;
  color: var(--fg);
  cursor: pointer;
  white-space: nowrap;
}
header.top button.regions-toggle:hover {
  background: var(--bg-sidebar);
}
header.top button.regions-toggle[disabled] {
  opacity: .4;
  cursor: default;
}

/* Compact region-type legend in the header */
header.top .region-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  font-size: 12px;
  color: var(--fg-muted);
  margin-left: 14px;
}
header.top .region-legend .swatch {
  display: inline-block;
  width: 9px; height: 9px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}
body.hide-regions header.top .region-legend { opacity: .4; }
.transcription {
  padding: 20px 24px;
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 15px;
  line-height: 1.65;
  overflow-wrap: anywhere;
}
.transcription h1, .transcription h2, .transcription h3, .transcription h4 {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-weight: 600;
  margin: 1.4em 0 .5em;
  line-height: 1.3;
}
.transcription h2 { font-size: 17px; color: var(--accent); }
.transcription h3 { font-size: 15px; }
.transcription p { margin: .6em 0; }
.transcription blockquote {
  margin: .9em 0;
  padding: .35em .9em;
  border-left: 3px solid var(--accent);
  background: rgba(139,58,47,.04);
  color: var(--fg-muted);
  font-size: 14px;
  font-style: normal;
}
.transcription blockquote em:first-child { color: var(--accent); font-weight: 500; }
.transcription u { text-decoration-color: var(--accent); text-underline-offset: 2px; }
.transcription sup, .transcription sub { line-height: 0; font-size: 0.75em; }
.transcription .red { color: var(--accent); }
.transcription .table-wrap {
  margin: 1em 0;
  overflow-x: auto;
  border: 1px solid var(--border);
  background: #fff;
}
.transcription .table-wrap table {
  border-collapse: collapse;
  font-size: 12.5px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  width: max-content;
  min-width: 100%;
}
.transcription .table-wrap th,
.transcription .table-wrap td {
  border: 1px solid var(--border);
  padding: 4px 8px;
  vertical-align: top;
  text-align: left;
}
.transcription .table-wrap thead th {
  background: var(--bg-sidebar);
  font-weight: 600;
  white-space: nowrap;
}
.transcription .table-wrap tbody tr:nth-child(even) td {
  background: rgba(0,0,0,.015);
}
.transcription pre, .transcription code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  background: var(--bg-sidebar);
  padding: 1px 5px;
  border-radius: 3px;
}
.transcription pre {
  padding: 10px 12px; overflow-x: auto;
}
.transcription hr {
  border: 0; border-top: 1px solid var(--border-soft); margin: 1.5em 0;
}

/* Lightbox */
.lightbox {
  position: fixed; inset: 0;
  background: rgba(20,18,15,.92);
  display: none;
  align-items: center; justify-content: center;
  z-index: 100;
  cursor: zoom-out;
  padding: 20px;
}
.lightbox.open { display: flex; }
.lightbox img {
  max-width: 100%; max-height: 100%;
  box-shadow: 0 8px 40px rgba(0,0,0,.5);
}

/* Narrow screens */
@media (max-width: 1000px) {
  .layout { grid-template-columns: 1fr; }
  aside.sidebar {
    position: static; height: auto; max-height: 240px;
    border-right: 0; border-bottom: 1px solid var(--border);
  }
  .page-body { grid-template-columns: 1fr; }
  .scan { position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--border-soft); }
}

/* Print: one page per page */
@media print {
  header.top, aside.sidebar, .lightbox { display: none !important; }
  .layout { display: block; }
  main { padding: 0; max-width: none; }
  article.page { page-break-after: always; box-shadow: none; border: 0; margin: 0; }
  .scan { position: static; max-height: none; }
  .scan img { cursor: default; }
}
"""

_JS = r"""
(function(){
  // --- Sidebar filter ---
  var input = document.getElementById('filter');
  var items = Array.from(document.querySelectorAll('ol.page-list li'));
  if (input) {
    input.addEventListener('input', function(){
      var q = input.value.toLowerCase().trim();
      items.forEach(function(li){
        li.style.display = (!q || li.dataset.search.indexOf(q) !== -1) ? '' : 'none';
      });
    });
  }
  // --- Regions overlay toggle ---
  var rgBtn = document.getElementById('regions-toggle');
  var anyOverlay = document.querySelector('.region-overlay');
  if (rgBtn) {
    if (!anyOverlay) {
      rgBtn.disabled = true;
      rgBtn.textContent = 'No regions';
      rgBtn.title = 'No layout regions on these pages — full-page mode was used.';
    } else {
      var setLabel = function(){
        rgBtn.textContent = document.body.classList.contains('hide-regions')
          ? 'Show regions' : 'Hide regions';
      };
      setLabel();
      rgBtn.addEventListener('click', function(){
        document.body.classList.toggle('hide-regions');
        setLabel();
      });
    }
  }
  // --- Highlight active page on scroll ---
  var links = {};
  document.querySelectorAll('ol.page-list li a').forEach(function(a){
    links[a.getAttribute('href').slice(1)] = a;
  });
  var observer = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if (e.isIntersecting && links[e.target.id]) {
        Object.values(links).forEach(function(l){ l.classList.remove('active'); });
        links[e.target.id].classList.add('active');
        // Keep active item visible in sidebar
        var li = links[e.target.id].parentElement;
        if (li && li.scrollIntoView) {
          var sb = document.querySelector('aside.sidebar');
          var liRect = li.getBoundingClientRect();
          var sbRect = sb.getBoundingClientRect();
          if (liRect.top < sbRect.top || liRect.bottom > sbRect.bottom) {
            li.scrollIntoView({block: 'nearest'});
          }
        }
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  document.querySelectorAll('article.page').forEach(function(a){ observer.observe(a); });

  // --- Lightbox ---
  var lb = document.getElementById('lightbox');
  var lbImg = lb.querySelector('img');
  document.querySelectorAll('.scan img').forEach(function(img){
    img.addEventListener('click', function(){
      lbImg.src = img.src;
      lb.classList.add('open');
    });
  });
  lb.addEventListener('click', function(){ lb.classList.remove('open'); lbImg.src = ''; });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape') { lb.classList.remove('open'); lbImg.src = ''; }
  });
})();
"""


def _slug(s: str) -> str:
    """A safe HTML id from any string."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-") or "page"


def _build_legend_html(present_types: List[str]) -> str:
    """Compact swatch legend showing only the region types actually present."""
    if not present_types:
        return ""
    items = []
    # Render in canonical REGION_TYPES order (whatever shows up in present)
    canonical_order = list(_REGION_COLOURS.keys())
    ordered = [t for t in canonical_order if t in present_types]
    # any unknown extra types last
    for t in present_types:
        if t not in ordered:
            ordered.append(t)
    for rtype in ordered:
        colour = _REGION_COLOURS.get(rtype, _DEFAULT_COLOUR)
        items.append(
            f'<span class="legend-item">'
            f'<span class="swatch" style="background:{colour}"></span>'
            f'{escape(rtype)}</span>'
        )
    return f'<div class="region-legend">{"".join(items)}</div>'


def build_html(pages: List[PageEntry], *, title: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Collect every region type that has been drawn somewhere, so we can
    # render a concise legend matching only what the user is looking at.
    present_types: List[str] = []
    seen = set()
    for p in pages:
        if not p.overlay_svg:
            continue
        for m in re.finditer(r'class="rg-box rg-([^"]+)"', p.overlay_svg):
            t = m.group(1)
            if t not in seen:
                seen.add(t)
                present_types.append(t)

    has_any_overlay = any(p.overlay_svg for p in pages)
    pages_with_overlay = sum(1 for p in pages if p.overlay_svg)
    total_boxes = sum(p.overlay_region_count for p in pages)
    legend_html = _build_legend_html(present_types)
    toggle_btn = (
        '<button id="regions-toggle" class="regions-toggle" type="button">'
        'Hide regions</button>'
    )

    # Sidebar
    sidebar_items = []
    for i, p in enumerate(pages, 1):
        anchor = _slug(p.md_path.stem)
        search_blob = f"{p.page_id} {p.page_number} {p.md_path.stem}".lower()
        sidebar_items.append(
            f'<li data-search="{escape(search_blob, quote=True)}">'
            f'<a href="#{anchor}">'
            f'<span class="num">{i}</span>'
            f'<span class="name">{escape(p.page_id)}</span>'
            f'</a></li>'
        )
    sidebar_html = "\n".join(sidebar_items)

    # Pages
    article_blocks = []
    for p in pages:
        anchor = _slug(p.md_path.stem)
        if p.image_src:
            img_tag = (
                f'<img src="{escape(p.image_src, quote=True)}" '
                f'alt="Scan of {escape(p.page_id)}" loading="lazy">'
            )
            # Wrap the img in a tight frame so the SVG overlay (if any)
            # tracks the displayed image bounds. The frame is always
            # present so the layout doesn't shift when overlays are toggled.
            scan = f'<div class="frame">{img_tag}{p.overlay_svg}</div>'
        else:
            scan = '<div class="placeholder">no scan available for this page</div>'

        page_num_badge = (
            f'<span class="pagenum">p.&nbsp;{escape(p.page_number)}</span>'
            if p.page_number else ""
        )
        regions_html = (
            f'<span class="regions">{escape(p.regions_summary)}</span>'
            if p.regions_summary else ""
        )

        article_blocks.append(f"""\
<article class="page" id="{anchor}">
  <header>
    <h2>{escape(p.page_id)}</h2>
    {page_num_badge}
    {regions_html}
  </header>
  <div class="page-body">
    <div class="scan">{scan}</div>
    <div class="transcription">{p.md_html}</div>
  </div>
</article>""")
    articles_html = "\n".join(article_blocks)

    if has_any_overlay:
        overlay_meta = (
            f' · {pages_with_overlay} page{"s" if pages_with_overlay != 1 else ""}'
            f' with {total_boxes} layout region{"s" if total_boxes != 1 else ""}'
        )
    else:
        overlay_meta = ""

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header class="top">
  <h1>{escape(title)}</h1>
  <span class="meta">{len(pages)} page{'s' if len(pages) != 1 else ''} · generated {timestamp}{overlay_meta}</span>
  {legend_html}
  {toggle_btn}
</header>
<div class="layout">
  <aside class="sidebar">
    <div class="filter"><input id="filter" type="search" placeholder="Filter pages…" autocomplete="off"></div>
    <ol class="page-list">
{sidebar_html}
    </ol>
  </aside>
  <main>
{articles_html}
  </main>
</div>
<div class="lightbox" id="lightbox" role="dialog" aria-label="Enlarged scan"><img alt=""></div>
<script>{_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML viewer for the markdown + scan output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_dir", type=Path,
                        help="Pipeline output directory (must contain md/ and pages/).")
    parser.add_argument("-o", "--viewer-path", type=Path, default=None,
                        help="Where to write the HTML file. "
                             "Default: <output_dir>/viewer.html")
    parser.add_argument("--title", default="Forsteinrichtungsoperate — Viewer",
                        help="Document title shown in the header and tab.")
    parser.add_argument("--max-width", type=int, default=1600,
                        help="Resize images so their longest edge is at most this many pixels.")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality (1-95) used for the embedded scans.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Write images to a sibling assets folder instead of "
                             "base64-embedding them. Use for very large datasets.")
    parser.add_argument("--md-dir", type=Path, default=None,
                        help="Override the markdown directory. Default: <output_dir>/md")
    parser.add_argument("--pages-dir", type=Path, default=None,
                        help="Override the page-image directory. Default: <output_dir>/pages")
    parser.add_argument("--regions-dir", type=Path, default=None,
                        help="Override the regions JSON directory. Default: "
                             "<output_dir>/regions. Used to draw layout overlays "
                             "for layout-mode pages (--doc-type mixed/text). "
                             "If the directory is missing, overlays are simply omitted.")
    parser.add_argument("--no-regions", action="store_true",
                        help="Disable layout overlays even if regions JSON is "
                             "available. Useful if you want a clean printout.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    md_dir = args.md_dir or (args.output_dir / "md")
    pages_dir = args.pages_dir or (args.output_dir / "pages")
    if not md_dir.is_dir():
        sys.exit(f"error: markdown directory not found: {md_dir}")
    if not pages_dir.is_dir():
        log.warning("pages directory not found: %s — viewer will have no scans", pages_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)

    # Regions directory is optional — silently skipped when absent or when
    # the user passes --no-regions.
    regions_dir: Optional[Path] = None
    if not args.no_regions:
        regions_dir = args.regions_dir or (args.output_dir / "regions")
        if not regions_dir.is_dir():
            log.info(
                "regions directory not found at %s — layout overlays disabled",
                regions_dir,
            )
            regions_dir = None

    viewer_path = args.viewer_path or (args.output_dir / "viewer.html")
    viewer_path.parent.mkdir(parents=True, exist_ok=True)

    embed = not args.no_embed
    assets_dir: Optional[Path] = None
    if not embed:
        assets_dir = viewer_path.parent / f"{viewer_path.stem}_assets"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        assets_dir.mkdir(parents=True)

    log.info("Building viewer from %s", args.output_dir)
    pages = build_pages(
        md_dir=md_dir,
        pages_dir=pages_dir,
        regions_dir=regions_dir,
        embed=embed,
        assets_dir=assets_dir,
        max_width=args.max_width,
        quality=max(1, min(95, args.quality)),
    )
    if not pages:
        sys.exit("error: no pages to render")

    html = build_html(pages, title=args.title)
    viewer_path.write_text(html, encoding="utf-8")

    size_mb = viewer_path.stat().st_size / (1024 * 1024)
    n_with_image = sum(1 for p in pages if p.image_src)
    n_with_overlay = sum(1 for p in pages if p.overlay_svg)
    n_boxes = sum(p.overlay_region_count for p in pages)
    log.info(
        "Wrote %s  (%.1f MB, %d pages, %d with scans, %d with %d region overlays)",
        viewer_path, size_mb, len(pages), n_with_image, n_with_overlay, n_boxes,
    )


if __name__ == "__main__":
    main()
