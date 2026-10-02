import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from .gemini import Gemini, cost_usd
from .prompts import build_prompt
from .schemas import SCHEMAS

write_lock = threading.Lock()


def extract_page(page: dict, gemini: Gemini, raw_dir: Path, force: bool = False, use_image: bool = True) -> dict:
    out_path = raw_dir / f"{page['seq']}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    image = Path(page["image"]) if use_image and page.get("image") else None
    started = time.time()
    data, usage, mode = gemini.extract(build_prompt(page, with_image=image is not None),
                                       SCHEMAS[page["source_type"]], images=[image] if image else [])
    data.setdefault("page_id", page["page_id"])
    record = {
        "seq": page["seq"], "source_type": page["source_type"], "subtype": page["subtype"],
        "page_id": page["page_id"], "file": page["file"], "image": str(image) if image else "",
        "model": gemini.settings.model, "thinking_level": gemini.settings.thinking_level,
        "mode": mode, "usage": usage, "elapsed_s": round(time.time() - started, 1), "result": data,
    }
    with write_lock:
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def extract_pages(pages: list[dict], gemini: Gemini, raw_dir: Path, force: bool = False,
                  use_image: bool = True) -> tuple[dict, dict]:
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=gemini.settings.workers) as pool:
        futures = {pool.submit(extract_page, page, gemini, raw_dir, force, use_image): page for page in pages}
        for future in tqdm(as_completed(futures), total=len(futures), desc="extracting"):
            page = futures[future]
            try:
                results[page["seq"]] = future.result()
            except Exception as error:
                errors[page["seq"]] = str(error)
                print("ERROR", page["seq"], str(error)[:200])
    return dict(sorted(results.items())), errors


def usage_summary(results: dict, gemini: Gemini) -> str:
    total = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    for record in results.values():
        for key in total:
            total[key] += record["usage"].get(key, 0)
    return (f"tokens: {total} | est. cost ${cost_usd(total, gemini.settings):.3f} "
            f"(prices: {gemini.settings.price_input_per_m}/{gemini.settings.price_output_per_m} USD per 1M)")
