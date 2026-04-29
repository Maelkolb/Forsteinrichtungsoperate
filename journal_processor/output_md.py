"""Generate a clean Markdown reconstruction of each page.

Layout rules for historical administrative documents:

  TitleRegion      → bold heading (## level)
  TableRegion      → pass-through the HTML table produced by the transcriber,
                     wrapped in a fenced ```html block if not already fenced
  GraphRegion      → pass-through the structured Markdown description
  ParagraphRegion  → plain text block
  MarginaliaRegion → block-quote with italic [Marginalie] tag
  FootnoteRegion   → footnote-style block
  PageNumberRegion → used only in the file-level YAML front matter
  ImageRegion      → description block
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

    if rtype == "ImageRegion":
        return f"*[Abbildung]* {text}"

    # ParagraphRegion and anything else
    return text


# ── main entry point ─────────────────────────────────────────────────────────

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
