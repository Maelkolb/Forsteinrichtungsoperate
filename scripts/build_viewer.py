#!/usr/bin/env python3
"""Build a single self-contained HTML viewer for pipeline output.

Pairs each Markdown file in ``<output>/md/`` with the matching scan in
``<output>/pages/``, compresses and embeds the scans as base64, renders the
Markdown (including the HTML tables produced for ``TableRegion`` and the
text-table descriptions for ``GraphRegion``), and writes one self-contained
HTML file you can open in any browser or hand off as an archive.

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
import math
import re
import shutil
import sys
from dataclasses import dataclass
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

# Reasoning sometimes leaks into TableRegion output between the metadata
# comment and the ```html fence. Drop it.
_LEAKED_TABLE_PROSE_RE = re.compile(
    r"(<!--\s*TableRegion[^>]*-->\s*\n)"
    r"((?:(?!```html\s*\n)[^\n]*\n)+)"
    r"(```html\s*\n)",
    re.MULTILINE,
)

# Same problem for GraphRegion: prose between the metadata comment and
# the first ``## `` heading.
_LEAKED_GRAPH_PROSE_RE = re.compile(
    r"(<!--\s*GraphRegion[^>]*-->\s*\n)"
    r"((?:(?!##\s)[^\n]*\n)+)"
    r"(##\s)",
    re.MULTILINE,
)

# A "Data Points" section followed by a pipe table — we'll parse this
# at MD level and inject a chart placeholder before the table.
_DATA_POINTS_SECTION_RE = re.compile(
    r"(##\s+Data\s+Points\b[^\n]*\n)"   # the heading
    r"(.*?)"                            # any prose between heading and table
    r"(\|[^\n]*\|\s*\n\|[\s:|\-]+\|[\s\S]*?)"   # the pipe table
    r"(?=\n\s*##\s|\Z)",                # until next ## or EOF
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


def _strip_leaked_reasoning(md_body: str) -> str:
    """Remove model "thinking aloud" leaks from existing pipeline output.

    Two patterns are scrubbed:
      * Prose between ``<!-- TableRegion ... -->`` and the ``` ```html `` `` fence.
      * Prose between ``<!-- GraphRegion ... -->`` and the first ``## `` heading.

    No-op on clean output (no prose between the markers).
    """
    def _repl_table(m: "re.Match[str]") -> str:
        if m.group(2).strip():
            return m.group(1) + "\n" + m.group(3)
        return m.group(0)

    def _repl_graph(m: "re.Match[str]") -> str:
        if m.group(2).strip():
            return m.group(1) + "\n" + m.group(3)
        return m.group(0)

    md_body = _LEAKED_TABLE_PROSE_RE.sub(_repl_table, md_body)
    md_body = _LEAKED_GRAPH_PROSE_RE.sub(_repl_graph, md_body)
    return md_body


def _parse_data_points(table_md: str) -> List[Dict[str, object]]:
    """Parse a Markdown pipe table into ``{label, x, y}`` records.

    The table is expected to have at least three columns; the first is
    treated as a label and the next two as numeric X / Y. Rows that fail
    to parse are skipped silently.
    """
    points: List[Dict[str, object]] = []
    for raw_line in table_md.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        # Skip header separator and a likely header row
        if re.match(r"^\|[\s:|\-]+\|\s*$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        # Heuristic: skip the header (first row with non-numeric x/y)
        try:
            x = float(re.sub(r"[^\d.\-eE]", "", cells[1]))
            y = float(re.sub(r"[^\d.\-eE]", "", cells[2]))
        except ValueError:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        points.append({"label": cells[0], "x": x, "y": y})
    return points


def _inject_data_point_charts(md_body: str) -> str:
    """Insert a chart placeholder before any data-points pipe table.

    The placeholder is an HTML ``<div class="chart-host">`` carrying the
    parsed points as JSON; runtime JS turns each one into an SVG scatter
    plot. The original Markdown table is preserved below the chart for
    readers who want the raw numbers.
    """
    def _repl(m: "re.Match[str]") -> str:
        heading = m.group(1)
        between = m.group(2)
        table = m.group(3)
        points = _parse_data_points(table)
        if len(points) < 3:
            return m.group(0)   # not enough data to chart
        payload = json.dumps(points, ensure_ascii=False)
        # HTML attribute-safe (single-quoted attribute → escape any ' in payload)
        payload = payload.replace("'", "&#39;")
        chart_html = f'<div class="chart-host" data-points=\'{payload}\'></div>\n\n'
        return f"{heading}{between}{chart_html}{table}"

    return _DATA_POINTS_SECTION_RE.sub(_repl, md_body)


def render_markdown(md_body: str) -> str:
    """Pipeline-flavoured Markdown → HTML.

    Pre-processing:
      1. Strip any reasoning the model leaked into existing pipeline output
         (between the region metadata comment and the actual content).
      2. Inject a chart placeholder before every ``Data Points`` table so
         the runtime JS can render a scatter plot alongside the raw numbers.
      3. If the document has an unclosed ``` ```html ... ``` ``` fence
         (upstream truncation), close it. The closing fence is inserted
         before any obviously-markdown content (blockquote/heading)
         that follows a blank line, so trailing marginalia survives.
      4. Replace fenced html blocks with their raw content; auto-close
         any unclosed <table> inside them.
      5. Ensure pipe tables that follow prose have the blank line the
         `tables` extension expects.
    """
    # 1 — drop leaked reasoning from already-generated md
    md_body = _strip_leaked_reasoning(md_body)

    # 2 — pre-mark Data Points tables for chart rendering
    md_body = _inject_data_point_charts(md_body)

    # 3 — close any unclosed ```html fence in a way that preserves
    #     trailing markdown (marginalia is sometimes inside the fence).
    md_body = _close_unclosed_html_fence(md_body)

    # 4 — substitute fenced html with raw html (closing any open <table>)
    md_body = _FENCED_HTML_RE.sub(
        lambda m: _close_unmatched_table(m.group(1)), md_body,
    )

    # 5 — pipe-table blank-line fix
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
# Page assembly
# ---------------------------------------------------------------------------
def build_pages(
    md_dir: Path,
    pages_dir: Path,
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

        pages.append(PageEntry(
            md_path=md_path,
            image_path=img_path,
            page_id=meta.get("page_id", md_path.stem),
            page_number=meta.get("page_number", ""),
            regions_summary=meta.get("regions", ""),
            md_html=body_html,
            image_src=image_src,
        ))
        log.info("[%d/%d] %s ✓", i, len(md_files), md_path.name)

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

/* Chart host (rendered SVG scatter plots from data-points tables) */
.transcription .chart-host {
  margin: 1em 0;
  padding: 14px;
  background: #fff;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}
.transcription .chart-host svg {
  display: block;
  width: 100%;
  height: auto;
  max-width: 720px;
  margin: 0 auto;
}
.transcription .chart-host .axis line,
.transcription .chart-host .axis path {
  stroke: var(--fg-muted);
  stroke-width: 1;
  fill: none;
}
.transcription .chart-host .axis text {
  fill: var(--fg-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}
.transcription .chart-host .axis-title {
  fill: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 11px;
  font-weight: 500;
}
.transcription .chart-host .point {
  fill: var(--accent);
  fill-opacity: 0.7;
  stroke: var(--accent);
  stroke-width: 0.5;
}
.transcription .chart-host .grid line {
  stroke: var(--border-soft);
  stroke-width: 1;
  shape-rendering: crispEdges;
}
.transcription .chart-host .chart-caption {
  text-align: center;
  font-size: 12px;
  color: var(--fg-muted);
  margin-top: 6px;
  font-style: italic;
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
  // --- Render chart-host placeholders as SVG scatter plots ---
  function nice(n) {
    // Round to a "nice" tick step
    var exp = Math.pow(10, Math.floor(Math.log10(Math.abs(n) || 1)));
    var f = n / exp;
    var nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
    return nf * exp;
  }
  function ticks(min, max, count) {
    var step = nice((max - min) / Math.max(1, count));
    var t0 = Math.floor(min / step) * step;
    var out = [];
    for (var v = t0; v <= max + step * 0.001; v += step) {
      out.push(Math.round(v / step) * step);
    }
    return out;
  }
  function fmt(v) {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10)  return v.toFixed(1);
    return v.toFixed(2);
  }
  function renderChart(host) {
    var pts;
    try { pts = JSON.parse(host.dataset.points); } catch (e) { return; }
    if (!pts || pts.length < 3) return;

    var W = 640, H = 360;
    var pad = { l: 48, r: 16, t: 12, b: 36 };
    var iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;

    var xs = pts.map(function(p){ return p.x; });
    var ys = pts.map(function(p){ return p.y; });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    // pad ranges by 5%
    var dx = (xmax - xmin) || 1, dy = (ymax - ymin) || 1;
    xmin -= dx * 0.05; xmax += dx * 0.05;
    ymin -= dy * 0.05; ymax += dy * 0.05;

    function sx(x) { return pad.l + (x - xmin) / (xmax - xmin) * iw; }
    function sy(y) { return pad.t + ih - (y - ymin) / (ymax - ymin) * ih; }

    var xt = ticks(xmin, xmax, 6), yt = ticks(ymin, ymax, 6);

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Scatter plot of " + pts.length + " data points");

    var grid = document.createElementNS(svg.namespaceURI, "g");
    grid.setAttribute("class", "grid");
    xt.forEach(function(t){
      var x = sx(t);
      var ln = document.createElementNS(svg.namespaceURI, "line");
      ln.setAttribute("x1", x); ln.setAttribute("x2", x);
      ln.setAttribute("y1", pad.t); ln.setAttribute("y2", pad.t + ih);
      grid.appendChild(ln);
    });
    yt.forEach(function(t){
      var y = sy(t);
      var ln = document.createElementNS(svg.namespaceURI, "line");
      ln.setAttribute("x1", pad.l); ln.setAttribute("x2", pad.l + iw);
      ln.setAttribute("y1", y); ln.setAttribute("y2", y);
      grid.appendChild(ln);
    });
    svg.appendChild(grid);

    var ax = document.createElementNS(svg.namespaceURI, "g");
    ax.setAttribute("class", "axis");
    // x axis
    var xax = document.createElementNS(svg.namespaceURI, "line");
    xax.setAttribute("x1", pad.l); xax.setAttribute("x2", pad.l + iw);
    xax.setAttribute("y1", pad.t + ih); xax.setAttribute("y2", pad.t + ih);
    ax.appendChild(xax);
    xt.forEach(function(t){
      var x = sx(t);
      var lbl = document.createElementNS(svg.namespaceURI, "text");
      lbl.setAttribute("x", x); lbl.setAttribute("y", pad.t + ih + 14);
      lbl.setAttribute("text-anchor", "middle");
      lbl.textContent = fmt(t);
      ax.appendChild(lbl);
    });
    // y axis
    var yax = document.createElementNS(svg.namespaceURI, "line");
    yax.setAttribute("x1", pad.l); yax.setAttribute("x2", pad.l);
    yax.setAttribute("y1", pad.t); yax.setAttribute("y2", pad.t + ih);
    ax.appendChild(yax);
    yt.forEach(function(t){
      var y = sy(t);
      var lbl = document.createElementNS(svg.namespaceURI, "text");
      lbl.setAttribute("x", pad.l - 6); lbl.setAttribute("y", y + 3);
      lbl.setAttribute("text-anchor", "end");
      lbl.textContent = fmt(t);
      ax.appendChild(lbl);
    });
    svg.appendChild(ax);

    var pg = document.createElementNS(svg.namespaceURI, "g");
    pts.forEach(function(p){
      var c = document.createElementNS(svg.namespaceURI, "circle");
      c.setAttribute("class", "point");
      c.setAttribute("cx", sx(p.x));
      c.setAttribute("cy", sy(p.y));
      c.setAttribute("r", 3);
      var t = document.createElementNS(svg.namespaceURI, "title");
      t.textContent = (p.label ? p.label + " — " : "") + "(" + fmt(p.x) + ", " + fmt(p.y) + ")";
      c.appendChild(t);
      pg.appendChild(c);
    });
    svg.appendChild(pg);

    host.appendChild(svg);
    var cap = document.createElement("div");
    cap.className = "chart-caption";
    cap.textContent = "Reconstructed from " + pts.length + " transcribed data points (hover for label).";
    host.appendChild(cap);
  }
  document.querySelectorAll(".chart-host").forEach(renderChart);

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


def build_html(pages: List[PageEntry], *, title: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

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
            scan = f'<img src="{escape(p.image_src, quote=True)}" alt="Scan of {escape(p.page_id)}" loading="lazy">'
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
  <span class="meta">{len(pages)} page{'s' if len(pages) != 1 else ''} · generated {timestamp}</span>
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
    log.info("Wrote %s  (%.1f MB, %d pages, %d with scans)",
             viewer_path, size_mb, len(pages), n_with_image)


if __name__ == "__main__":
    main()
