"""Main processing pipeline.

Orchestrates:  prepare → preprocess → detect → transcribe → output (Markdown)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import PipelineConfig
from .splitter import split_all
from .preprocessor import preprocess_page
from .region_detector import RegionDetector
from .transcriber import Transcriber
from .output_md import generate_md

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end historical document processing pipeline."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self._init_client()

    # ── Gemini client setup ──────────────────────────────────────────────

    def _init_client(self) -> None:
        from google import genai

        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
        )
        self.detector = RegionDetector(self.client, self.cfg)
        self.transcriber = Transcriber(self.client, self.cfg)

    # ── Full run ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline. Returns a summary dict."""
        t0 = time.time()
        summary: Dict[str, Any] = {"pages_processed": 0, "errors": []}

        # 1 — Prepare pages (single passthrough or double-page split)
        log.info("=== Stage 1: Preparing pages (single_page=%s) ===", self.cfg.single_page)
        page_paths = split_all(self.cfg)
        if not page_paths:
            log.error("No pages produced — check input directory.")
            return summary

        # 2 — Pre-process
        log.info("=== Stage 2: Pre-processing %d page(s) ===", len(page_paths))
        for pp in page_paths:
            preprocess_page(pp, self.cfg)

        # 3+4+5 — Detect → Transcribe → Output
        log.info("=== Stages 3-5: Detect → Transcribe → Output ===")
        if self.cfg.workers > 1:
            self._run_parallel(page_paths, summary)
        else:
            self._run_sequential(page_paths, summary)

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["pages_processed"] = len(page_paths) - len(summary["errors"])

        # Write summary JSON
        summary_path = self.cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(
            "Done. %d page(s) in %.0fs (%d error(s)).",
            summary["pages_processed"], elapsed, len(summary["errors"]),
        )
        return summary

    # ── Sequential / parallel helpers ────────────────────────────────────

    def _run_sequential(self, page_paths: List[Path], summary: Dict) -> None:
        for idx, pp in enumerate(page_paths, 1):
            log.info("[%d/%d] %s", idx, len(page_paths), pp.name)
            try:
                self._process_page(pp)
            except Exception as exc:
                log.error("Failed %s: %s", pp.name, exc)
                summary["errors"].append({"page": pp.name, "error": str(exc)})

    def _run_parallel(self, page_paths: List[Path], summary: Dict) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {pool.submit(self._process_page, pp): pp for pp in page_paths}
            done = 0
            for future in as_completed(futures):
                done += 1
                pp = futures[future]
                try:
                    future.result()
                    log.info("[%d/%d] ✓ %s", done, len(page_paths), pp.name)
                except Exception as exc:
                    log.error("[%d/%d] ✗ %s: %s", done, len(page_paths), pp.name, exc)
                    summary["errors"].append({"page": pp.name, "error": str(exc)})

    # ── Single-page processing ────────────────────────────────────────────

    def _process_page(self, page_path: Path) -> None:
        pid = page_path.stem
        page_img = Image.open(page_path).convert("RGB")

        # --- Region detection ---
        det = self.detector.detect(page_path)
        if det["status"] != "success":
            raise RuntimeError(f"Detection failed: {det.get('error', 'unknown')}")

        regions = det["regions"]
        dims = det["image_dimensions"]

        # Save region crops for debugging
        regions_dir = self.cfg.output_dir / "regions" / pid
        regions_dir.mkdir(parents=True, exist_ok=True)

        # --- Transcription (per region) ---
        for r in regions:
            bbox = r["bbox"]
            crop = page_img.crop((
                bbox["x"], bbox["y"],
                bbox["x"] + bbox["width"],
                bbox["y"] + bbox["height"],
            ))
            crop_path = regions_dir / f"{r['id']}_{r['type']}.png"
            crop.save(crop_path, "PNG")

            r["transcription"] = self.transcriber.transcribe_region(crop, r)

        # --- Output: Markdown ---
        if self.cfg.output_md:
            generate_md(pid, regions, self.cfg.output_dir / "md")

        # --- Debug JSON (detection + transcription) ---
        serialisable = []
        for r in regions:
            sr = dict(r)
            if "transcription" in sr:
                sr["transcription"] = {
                    k: v for k, v in sr["transcription"].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
            serialisable.append(sr)
        page_json = self.cfg.output_dir / "regions" / f"{pid}.json"
        page_json.write_text(
            json.dumps(serialisable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
