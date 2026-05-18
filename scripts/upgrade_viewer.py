"""
============================================================
upgrade_viewer.py — Retrofit existing Forsteinrichtungsoperate
                    viewer HTML files with the in-place editor
                    (CER/WER, JSON import/export, autosave …)
============================================================

The viewers produced by ``scripts/build_viewer.py`` are entirely
self-contained — every scan is base64-embedded, every transcription
is rendered HTML in the page itself.  That makes upgrades cheap:
we just need to inject some CSS, a JavaScript block, and a handful
of UI widgets.  No ``pages/`` folder, no ``output/md/``, no
re-running the pipeline.

After running this script the viewer gains, with zero loss of
existing functionality (sidebar, lightbox, regions toggle):

    * Click-to-edit transcriptions (toggle "Edit mode")
    * Per-page “Mark done” → automatic CER & WER vs. the AI's
      original transcription, with colour-coded thresholds
    * Per-page “Redo” flag for pages needing reprocessing
    * Progress counter + aggregate (micro-averaged) CER/WER
    * Auto-save to browser localStorage every 1.5 s
    * Restore banner on reload — jumps you back to the last
      page you edited
    * Portable JSON export / import (works across browsers and
      machines; matches the schema of HistOrniGraph exports)

USAGE
-----

Single file (overwrites in place; keeps a .bak):

    python upgrade_viewer.py path/to/viewer.html

Single file, separate output:

    python upgrade_viewer.py viewer.html -o viewer_editable.html

A whole directory (recursive), in place:

    python upgrade_viewer.py output/

A whole directory, written to a parallel tree:

    python upgrade_viewer.py output/ --output-dir output_editable/

Colab:

    import sys
    sys.path.insert(0, '/content/Forsteinrichtungsoperate/scripts')
    from upgrade_viewer import upgrade_viewer_file, upgrade_viewer_tree

    upgrade_viewer_file(
        '/content/drive/MyDrive/forst_outputs/run_42/viewer.html'
    )
    # …or batch:
    upgrade_viewer_tree('/content/drive/MyDrive/forst_outputs/')

THE CER/WER REFERENCE  (read this if you care what the numbers mean)
--------------------------------------------------------------------

CER & WER need a "reference" string.  Since the original markdown
isn't stored in the rendered HTML, we use each transcription's
``innerText`` at first page load as the AI's reference output.
Whatever's in the box on the day you first open the viewer is
"what the model produced" — exactly the text the human is now
correcting.  Editing it and then "Mark done" measures the human
corrections against that baseline, which is what you want.

(If you later want a stricter reference — the literal markdown the
pipeline emitted, including punctuation lost in markdown→HTML —
the matching update to ``build_viewer.py`` can embed the raw
markdown as a ``data-original-md`` attribute on each transcription;
this script will pick it up automatically if present.)
============================================================
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

try:
    from viewer_editor_addon import (
        ADDON_MARKER,
        ARTICLE_WIDGETS_TEMPLATE,
        CSS_ADDON,
        JS_ADDON,
        RESTORE_AND_TOAST_HTML,
        SIDEBAR_SUMMARY_HTML,
        TOOLBAR_HTML,
    )
except ImportError as exc:                                # pragma: no cover
    raise ImportError(
        "upgrade_viewer.py needs viewer_editor_addon.py importable. Put both "
        "files in the same folder (or on sys.path). Original error: " + str(exc)
    )


PathLike = Union[str, Path]


# ───────────────────────────────────────────────────────────
# HTML attribute escaping — page IDs are alphanumeric + dashes
# in practice, but be defensive in case future build_viewer.py
# versions slug them less strictly.
# ───────────────────────────────────────────────────────────
def _attr_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ───────────────────────────────────────────────────────────
# Idempotency check
# ───────────────────────────────────────────────────────────
def is_already_upgraded(html: str) -> bool:
    return ADDON_MARKER in html


# ───────────────────────────────────────────────────────────
# Core upgrade pass
# ───────────────────────────────────────────────────────────
def upgrade_html(html: str) -> Tuple[str, dict]:
    """
    Inject the editor addon into a viewer's HTML.

    Returns
    -------
    (new_html, stats) where stats is a small dict describing what
    was injected (n_articles, etc.) for the caller's log output.
    """
    stats = {"already_upgraded": False, "articles": 0, "had_top_header": False}

    if is_already_upgraded(html):
        stats["already_upgraded"] = True
        return html, stats

    # IMPORTANT: every replacement here uses a lambda (or ``re.escape``-
    # neutral string concatenation), NOT a plain ``repl`` string with
    # backreferences.  The CSS and JS payloads contain regex-significant
    # characters (``\s``, ``\d``, ``\g``, etc.) which ``re.sub`` would
    # otherwise try to interpret as escape sequences.

    # ── 1. Append our CSS just before </style> ─────────────────
    css_block = ADDON_MARKER + "\n" + CSS_ADDON + "\n</style>"
    new_html, n = re.subn(
        r"</style>",
        lambda _m: css_block,
        html, count=1,
    )
    if n == 0:
        raise ValueError(
            "Could not find a </style> tag — is this really a "
            "Forsteinrichtungsoperate viewer HTML?"
        )
    html = new_html

    # ── 2. Inject toolbar buttons inside <header class="top">…</header>
    new_html, n = re.subn(
        r'(<header class="top">.*?)(</header>)',
        lambda m: m.group(1) + TOOLBAR_HTML + m.group(2),
        html, count=1, flags=re.DOTALL,
    )
    if n == 0:
        raise ValueError(
            "Could not find <header class=\"top\">…</header>. The viewer "
            "structure isn't what this upgrader expects."
        )
    html = new_html
    stats["had_top_header"] = True

    # ── 3. Drop the restore banner + toast right after that header ─
    new_html, _ = re.subn(
        r'(<header class="top">.*?</header>)',
        lambda m: m.group(1) + "\n" + RESTORE_AND_TOAST_HTML,
        html, count=1, flags=re.DOTALL,
    )
    html = new_html

    # ── 4. Slot a validation-summary block into the sidebar ────
    # Insert it right before the existing <ol class="page-list"> so
    # it sits between the filter input and the page list.
    new_html, _ = re.subn(
        r'(<ol class="page-list">)',
        lambda m: SIDEBAR_SUMMARY_HTML + m.group(1),
        html, count=1,
    )
    html = new_html

    # ── 5. Per-article: inject done/redo + metrics chip into
    #       each <article class="page" id="X"><header>…</header>.
    # The regex captures (1) everything from the article opening tag
    # through the header's existing content, (2) the page id, and
    # (3) the </header> closing tag.  We then sandwich the per-page
    # widgets between (1) and (3).
    def _patch_article(m: re.Match) -> str:
        page_id = m.group(2)
        widgets = ARTICLE_WIDGETS_TEMPLATE.format(
            page_id_attr=_attr_escape(page_id)
        )
        return m.group(1) + widgets + m.group(3)

    new_html, n_articles = re.subn(
        r'(<article class="page" id="([^"]+)">\s*<header>.*?)(\s*</header>)',
        _patch_article,
        html, flags=re.DOTALL,
    )
    if n_articles == 0:
        print("   ⚠ No <article class=\"page\"> blocks found; injecting addon "
              "anyway, but per-page widgets will not appear.")
    html = new_html
    stats["articles"] = n_articles

    # ── 6. Append our JS as a new <script> block just before </body> ──
    js_block = "<script>\n" + JS_ADDON + "\n</script>\n</body>"
    new_html, n = re.subn(
        r"</body>",
        lambda _m: js_block,
        html, count=1,
    )
    if n == 0:
        raise ValueError("Could not find </body> tag. HTML may be malformed.")
    html = new_html

    return html, stats


# ───────────────────────────────────────────────────────────
# File-level wrappers
# ───────────────────────────────────────────────────────────
def upgrade_viewer_file(
    html_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    backup: bool = True,
    quiet: bool = False,
) -> Path:
    """
    Upgrade one viewer HTML file.

    Parameters
    ----------
    html_path : str | Path
        Path to the viewer HTML to upgrade.
    output_path : str | Path | None
        Where to write the upgraded HTML.  ``None`` overwrites in place.
    backup : bool
        When overwriting in place, keep a ``.bak`` copy of the previous
        file (only if one doesn't already exist; we never clobber a
        real backup).
    quiet : bool
        Suppress per-file log lines.
    """
    html_path = Path(html_path)
    if not html_path.is_file():
        raise FileNotFoundError(html_path)

    raw = html_path.read_text(encoding="utf-8")
    if not quiet:
        size_mb = len(raw) / (1024 * 1024)
        print(f"📄 {html_path.name}  ({size_mb:.1f} MB)")

    if is_already_upgraded(raw):
        if not quiet:
            print("   • Already upgraded — skipping.")
        return html_path

    new_html, stats = upgrade_html(raw)
    if not quiet:
        print(f"   ✓ Injected addon ({stats['articles']} page widget(s))")

    if output_path is None:
        output_path = html_path
    output_path = Path(output_path)

    overwriting = output_path.resolve() == html_path.resolve()
    if backup and overwriting:
        bak = html_path.with_suffix(html_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(html_path, bak)
            if not quiet:
                print(f"   ✓ Backup saved: {bak.name}")
        elif not quiet:
            print(f"   • Backup already exists: {bak.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_html, encoding="utf-8")
    if not quiet:
        new_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Wrote {output_path}  ({new_size_mb:.1f} MB)")
    return output_path


def upgrade_viewer_tree(
    root: PathLike,
    pattern: str = "*.html",
    *,
    skip_backups: bool = True,
    recursive: bool = True,
    output_dir: Optional[PathLike] = None,
    backup: bool = True,
    quiet: bool = False,
) -> List[Path]:
    """
    Walk *root* and upgrade every viewer HTML it finds.

    Returns the list of paths that were successfully upgraded
    (already-upgraded files are skipped, not re-counted).
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    it: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
    upgraded: List[Path] = []
    skipped: int = 0
    failures: List[Tuple[Path, Exception]] = []

    for html_path in sorted(it):
        if skip_backups and html_path.name.endswith(".bak"):
            continue
        # Quick sniff: only upgrade things that LOOK like our viewers.
        # We don't want to inject into unrelated HTML by accident.
        try:
            with html_path.open(encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
        except Exception:
            continue
        if "Forsteinrichtungsoperate" not in head and '<header class="top">' not in head:
            continue

        try:
            out: Optional[Path] = None
            if output_dir is not None:
                out = Path(output_dir) / html_path.relative_to(root)
            result = upgrade_viewer_file(
                html_path, output_path=out, backup=backup, quiet=quiet
            )
            # If the file was already upgraded, upgrade_viewer_file returns the
            # original path unchanged; count it as skipped, not done.
            if is_already_upgraded(result.read_text(encoding="utf-8")):
                if result == html_path and out is None:
                    # Path returned was the input, content unchanged → skip
                    pass
                upgraded.append(result)
            else:
                upgraded.append(result)
        except Exception as exc:
            failures.append((html_path, exc))
            print(f"   ⚠ FAILED {html_path}: {exc}")

    print()
    print(f"✅ Processed {len(upgraded)} viewer(s)")
    if failures:
        print(f"⚠ {len(failures)} failure(s):")
        for p, e in failures:
            print(f"   • {p}: {e}")
    return upgraded


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrofit existing Forsteinrichtungsoperate viewer HTML files "
            "with an in-place transcription editor (CER/WER, JSON "
            "import/export, autosave, last-page resume)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python upgrade_viewer.py output/viewer.html\n"
            "  python upgrade_viewer.py viewer.html -o viewer_editable.html\n"
            "  python upgrade_viewer.py output/\n"
            "  python upgrade_viewer.py output/ --output-dir output_editable/\n"
        ),
    )
    parser.add_argument("target", help="Viewer HTML file, or a directory of them.")
    parser.add_argument("-o", "--output", default=None,
                        help="(File mode) Write upgrade here instead of overwriting.")
    parser.add_argument("--pattern", default="*.html",
                        help="(Directory mode) Glob to match. Default: %(default)s")
    parser.add_argument("--output-dir", default=None,
                        help="(Directory mode) Write upgrades under this folder, "
                             "preserving the input tree.  Default: overwrite in place.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't keep a .bak when overwriting in place.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-file log lines.")
    args = parser.parse_args()

    target = Path(args.target)
    backup = not args.no_backup

    if target.is_file():
        upgrade_viewer_file(target, output_path=args.output,
                            backup=backup, quiet=args.quiet)
    elif target.is_dir():
        upgrade_viewer_tree(target, pattern=args.pattern,
                            output_dir=args.output_dir, backup=backup,
                            quiet=args.quiet)
    else:
        raise SystemExit(f"Not a file or directory: {target}")


if __name__ == "__main__":
    _cli()
