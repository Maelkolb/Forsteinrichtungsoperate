"""Generate a clean Markdown reconstruction of each page.

Two entry points:

  ``generate_md(page_id, regions, output_dir)``
      Region-based reconstruction (used by ``doc_type="text"`` and
      ``doc_type="mixed"``). Iterates the regions in reading order and
      assembles a Markdown document.

  ``generate_full_page_md(page_id, doc_type, text, output_dir)``
      Single-pass output (used by ``doc_type="table"`` and ``doc_type="graph"``).
      Writes the model's response verbatim under a YAML front matter block.

Layout rules for the region-based mode:

  TitleRegion      → bold heading (## level)
  TableRegion      → pass-through the HTML table produced by the transcriber,
                     wrapped in a fenced ```html block if not already fenced
  GraphRegion      → pass-through the structured Markdown description
  ParagraphRegion  → plain text block
  MarginaliaRegion → block-quote with italic [Marginalie] tag
  FootnoteRegion   → footnote-style block
  PageNumberRegion → used only in the file-level YAML front matter
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_page_number(regions: List[Dict]) -> str:
    for r in regions:
        if r["type"] == "PageNumberRegion":
            pn = r.get("page_number") or r.get("transcription", {}).get("text", "")
            if pn:
                return str(pn)
    return ""


def _ensure_html_fence(text: str) -> str:
    """Wrap a bare HTML table in a fenced code block if not already wrapped."""
    stripped = text.strip()
    if stripped.startswith("```"):
        return stripped  # already fenced
    if stripped.lower().startswith("<table"):
        return f"```html\n{stripped}\n```"
    return stripped


def _region_to_md(r: Dict[str, Any]) -> Optional[str]:
    """Convert one region to its Markdown representation."""
    rtype = r["type"]
    trans = r.get("transcription", {})
    text = trans.get("text", "").strip()

    if not text:
        return None

    if rtype == "PageNumberRegion":
        return None   # handled in front matter

    if rtype == "TitleRegion":
        # Prefer the title extracted during detection if transcription is empty
        title = text or r.get("title_text", "")
        return f"## {title}" if title else None

    if rtype == "TableRegion":
        # Add a metadata comment so readers know table provenance
        rows = r.get("rows", "?")
        cols = r.get("cols", "?")
        header_rows = r.get("header_rows", "?")
        has_red = "yes" if r.get("has_red_ink") else "no"
        meta = (
            f"<!-- TableRegion: {rows} rows × {cols} cols, "
            f"{header_rows} header row(s), red ink: {has_red} -->"
        )
        table_block = _ensure_html_fence(text)
        return f"{meta}\n{table_block}"

    if rtype == "GraphRegion":
        gtype = r.get("graph_type", "unknown")
        return f"<!-- GraphRegion: {gtype} -->\n{text}"

    if rtype == "MarginaliaRegion":
        # Indent each line as a block-quote
        quoted = "\n".join(f"> {line}" for line in text.splitlines())
        return f"> *[Marginalie]*\n{quoted}"

    if rtype == "FootnoteRegion":
        first_line, *rest = text.splitlines()
        body = "\n".join(f"    {l}" for l in rest)
        return f"[^fn]: {first_line}\n{body}" if rest else f"[^fn]: {first_line}"

    # ParagraphRegion and anything else
    return text


# ── main entry point: region-based reconstruction ────────────────────────────

def generate_md(
    page_id: str,
    regions: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a Markdown file for one processed page.

    The file starts with a minimal YAML front matter block, then each region
    in reading order.
    """
    page_num = _find_page_number(regions)

    # ── YAML front matter ───────────────────────────────────────────────
    front: List[str] = ["---"]
    front.append(f"page_id: {page_id}")
    if page_num:
        front.append(f"page_number: {page_num}")
    region_summary = ", ".join(
        f"{rtype}×{sum(1 for r in regions if r['type'] == rtype)}"
        for rtype in dict.fromkeys(r["type"] for r in regions)
    )
    front.append(f"regions: \"{region_summary}\"")
    front.append("---")
    front.append("")

    # ── Body ────────────────────────────────────────────────────────────
    body: List[str] = []
    for r in sorted(regions, key=lambda r: r["reading_order"]):
        block = _region_to_md(r)
        if block:
            body.append(block)
            body.append("")   # blank line between regions

    lines = front + body
    md_path = output_dir / f"{page_id}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.debug("Wrote %s (%d regions)", md_path.name, len(regions))
    return md_path


# ── main entry point: single-pass full-page output ──────────────────────────

# Regex used to spot a page-number-looking string at the very top of a graph
# transcription. We keep it permissive — only used to optionally surface the
# page number into the YAML front matter.
_PAGE_NUM_RE = re.compile(r"^\s*(?:page|seite|pg|p\.?)\s*[:.\-]?\s*(\d+)\s*$",
                          re.IGNORECASE | re.MULTILINE)


def generate_full_page_md(
    page_id: str,
    doc_type: str,
    text: str,
    output_dir: Path,
) -> Path:
    """Write a Markdown file for a single-pass (full-page) transcription.

    ``doc_type`` is the source category, one of ``"table"`` or ``"graph"``.
    The model's response (``text``) is appended verbatim after a small YAML
    front matter block. We deliberately keep the model output untouched so
    that any HTML <table> or structured Markdown produced upstream survives
    intact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to surface a page number if the model happened to include one.
    page_num = ""
    m = _PAGE_NUM_RE.search(text or "")
    if m:
        page_num = m.group(1)

    front: List[str] = ["---"]
    front.append(f"page_id: {page_id}")
    front.append(f"doc_type: {doc_type}")
    front.append("processing_mode: full_page")
    if page_num:
        front.append(f"page_number: {page_num}")
    front.append("---")
    front.append("")

    body = (text or "").strip()
    if not body:
        # Make absolutely sure we never write a totally empty file — the
        # front matter alone would still be valid Markdown but unhelpful.
        body = "*(empty transcription — model returned no text)*"

    md_path = output_dir / f"{page_id}.md"
    md_path.write_text("\n".join(front) + body + "\n", encoding="utf-8")
    log.debug("Wrote %s (full-page %s)", md_path.name, doc_type)
    return md_path
