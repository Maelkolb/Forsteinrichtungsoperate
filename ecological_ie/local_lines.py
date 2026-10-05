"""Text-line detection (kraken blla) for every prepared page image on a local CPU or GPU – the same model, upscaling rule
and output format as modal_lines.py (work/runs/<run>/layout/lines/<unit>/pNNN.json), for machines without Modal.
Needs its own environment: pip install "kraken>=7.1,<7.2" pillow shapely (torch with CUDA for --device cuda:0).

    python ecological_ie/local_lines.py                                  # all pages without lines yet, CPU
    python ecological_ie/local_lines.py --device cuda:0 --procs 1 --force  # everything again on a GPU
    python ecological_ie/local_lines.py --units I-16,I-02 --procs 4        # some units

About 40 s per 2338 px page on 8 CPU threads, 3–11 s on an RTX 5090 (about 3.4 GB GPU memory per process).
"""
import argparse, io, json, os, time
from multiprocessing import Pool
from pathlib import Path

UPSCALE = 2
ROOT = Path(__file__).resolve().parents[1]


def init(threads, device="cpu"):
    global MODEL, DEVICE, blla, Image, Polygon
    import torch
    from importlib import resources
    from kraken import blla
    from kraken.lib import vgsl
    from PIL import Image
    from shapely.geometry import Polygon
    torch.set_num_threads(threads)
    MODEL = vgsl.TorchVGSLModel.load_model(str(resources.files("kraken").joinpath("blla.mlmodel")))
    DEVICE = device
    globals().update(blla=blla, Image=Image, Polygon=Polygon)


def segment(job):
    unit_id, position, path, target = job
    page = Image.open(path).convert("RGB")
    width, height = page.size
    upscale = UPSCALE if max(width, height) < 2000 else 1
    started = time.time()
    work = page.resize((width * upscale, height * upscale), Image.LANCZOS) if upscale > 1 else page
    seg = blla.segment(work, model=MODEL, device=DEVICE)
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
    result = {"unit": unit_id, "position": position, "image_size": [width, height], "lines": lines}
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(result), encoding="utf-8")
    return f"{unit_id} p{position}: {len(lines)} lines in {time.time() - started:.0f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run", default="work/runs/main")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = parser.parse_args()
    out = ROOT / args.run / "layout" / "lines"
    wanted = set(args.units.split(",")) if args.units else None
    todo = []
    for unit_file in sorted((ROOT / "work" / "units").glob("*/unit.json")):
        unit = json.loads(unit_file.read_text(encoding="utf-8"))
        if wanted and unit["id"] not in wanted:
            continue
        for page in unit["pages"]:
            target = out / unit["id"] / f"p{page['position']:03d}.json"
            if page["image"] and page["kind"] != "Ka" and (args.force or not target.exists()):
                todo.append((unit["id"], page["position"], str(unit_file.parent / page["image"]), str(target)))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} pages", flush=True)
    threads = max(1, (os.cpu_count() or 8) // args.procs)
    with Pool(args.procs, initializer=init, initargs=(threads, args.device)) as pool:
        for done, message in enumerate(pool.imap_unordered(segment, todo), 1):
            print(f"{done}/{len(todo)} {message}", flush=True)


if __name__ == "__main__":
    main()
