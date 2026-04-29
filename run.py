#!/usr/bin/env python3
"""Run the historical document processing pipeline.

Usage examples:

  # Single-page scans (default — most historical sources):
  python run.py --input /path/to/scans --output /path/to/output

  # Double-page spreads (split each image down the centre):
  python run.py --input /path/to/spreads --output /path/to/output --double-page

  # With deskew and verbose logging:
  python run.py -i scans/ -o out/ --deskew -v

Requires:
  - GOOGLE_API_KEY environment variable set
  - pip install google-genai Pillow
"""

import argparse
import logging
import sys
from pathlib import Path

from journal_processor import Pipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process historical German administrative document scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────
    parser.add_argument("--input",  "-i", required=True, type=Path,
                        help="Directory containing scan images.")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output directory (created if absent).")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Gemini model ID.")

    # ── Page layout ───────────────────────────────────────────────────────
    parser.add_argument("--double-page", action="store_true",
                        help="Input files are double-page spreads — split each "
                             "image down the centre. Default: single-page mode.")

    # ── Pre-processing ────────────────────────────────────────────────────
    parser.add_argument("--deskew", action="store_true",
                        help="Enable deskew (requires scipy).")
    parser.add_argument("--no-enhance-contrast", action="store_true",
                        help="Disable auto-contrast enhancement.")
    parser.add_argument("--no-sharpen", action="store_true",
                        help="Disable unsharp-mask sharpening.")

    # ── Detection ─────────────────────────────────────────────────────────
    parser.add_argument("--max-regions", type=int, default=12,
                        help="Maximum layout regions per page.")

    # ── Concurrency ───────────────────────────────────────────────────────
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel page-processing threads.")

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        model_id=args.model,
        single_page=not args.double_page,
        deskew=args.deskew,
        enhance_contrast=not args.no_enhance_contrast,
        sharpen=not args.no_sharpen,
        max_regions=args.max_regions,
        workers=args.workers,
        # Only Markdown output
        output_md=True,

        output_pagexml=False,
        output_sharegpt=False,
    )

    pipeline = Pipeline(cfg)
    summary = pipeline.run()

    if summary.get("errors"):
        print(
            f"\n⚠  Completed with {len(summary['errors'])} error(s). "
            "See output/summary.json for details.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
