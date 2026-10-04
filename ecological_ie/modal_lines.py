"""Text-line detection (kraken blla) for every prepared page image, on Modal.

    uvx modal run ecological_ie/modal_lines.py                      # all pages without lines yet
    uvx modal run ecological_ie/modal_lines.py --units I-16,I-02    # some units
    uvx modal run ecological_ie/modal_lines.py --force              # everything again

Pages under 2000 px (the dump previews are 1200 px wide) are upscaled 2x before segmentation; polygons, baselines and boxes
are written back in the coordinates of the prepared image to work/runs/main/layout/lines/<unit>/pNNN.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

UPSCALE = 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("kraken>=7.1,<7.2", "pillow")
)
app = modal.App("forsteinrichtung-lines", image=image)


@app.function(cpu=8, memory=8192, timeout=3600, max_containers=10, retries=1)
def segment(pages: list[tuple[str, int, bytes]]) -> list[dict]:
    import io
    import time
    from importlib import resources

    import torch
    from kraken import blla
    from kraken.lib import vgsl
    from PIL import Image
    from shapely.geometry import Polygon

    torch.set_num_threads(8)
    model = vgsl.TorchVGSLModel.load_model(str(resources.files("kraken").joinpath("blla.mlmodel")))
    results = []
    for unit_id, position, data in pages:
        page = Image.open(io.BytesIO(data)).convert("RGB")
        width, height = page.size
        upscale = UPSCALE if max(width, height) < 2000 else 1
        started = time.time()
        work = page.resize((width * upscale, height * upscale), Image.LANCZOS) if upscale > 1 else page
        seg = blla.segment(work, model=model, device="cpu")
        lines = []
        for line in seg.lines:
            if not line.boundary or len(line.boundary) < 3:
                continue
            outline = Polygon([(x / upscale, y / upscale) for x, y in line.boundary])
            if not outline.is_valid:
                outline = outline.buffer(0)
            if outline.is_empty or outline.geom_type != "Polygon":
                continue
            simple = outline.simplify(1.5, preserve_topology=True)
            x0, y0, x1, y1 = outline.bounds
            lines.append({"box": [round(x0), round(y0), round(x1), round(y1)],
                          "polygon": [[round(x), round(y)] for x, y in simple.exterior.coords[:-1]],
                          "baseline": [[round(x / upscale), round(y / upscale)] for x, y in line.baseline or []]})
        print(f"{unit_id} p{position}: {len(lines)} lines in {time.time() - started:.0f} s")
        results.append({"unit": unit_id, "position": position, "image_size": [width, height], "lines": lines})
    return results


@app.local_entrypoint()
def main(units: str = "", force: bool = False, run: str = "work/runs/main"):
    root = Path(__file__).resolve().parents[1]
    out = root / run / "layout" / "lines"
    wanted = set(units.split(",")) if units else None
    todo = []
    for unit_file in sorted((root / "work" / "units").glob("*/unit.json")):
        unit = json.loads(unit_file.read_text(encoding="utf-8"))
        if wanted and unit["id"] not in wanted:
            continue
        for page in unit["pages"]:
            target = out / unit["id"] / f"p{page['position']:03d}.json"
            if page["image"] and page["kind"] != "Ka" and (force or not target.exists()):
                todo.append((unit["id"], page["position"], (unit_file.parent / page["image"]).read_bytes()))
    print(f"{len(todo)} pages")
    done = 0
    for start in range(0, len(todo), 10):
        group = [[page] for page in todo[start:start + 10]]
        for attempt in range(3):
            try:
                results = list(segment.map(group, order_outputs=False))
                break
            except Exception as error:
                print(f"pages {start}-{start + len(group)}: {error!r}, attempt {attempt + 1}")
        else:
            continue
        for result in results:
            for page in result:
                target = out / page["unit"] / f"p{page['position']:03d}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(page), encoding="utf-8")
                done += 1
        print(f"{done}/{len(todo)} pages")
