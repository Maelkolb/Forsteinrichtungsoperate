import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from .gemini import Gemini
from .spec import PagePlan

REGION_KINDS = ["map_body", "legend", "title_cartouche", "scale_bar", "north_arrow", "marginal_text", "inset", "other"]
LABEL_CLASSES = ["settlement", "water", "mountain", "forest_district", "compartment_number", "road", "trift_structure",
                 "field_name", "boundary", "administrative", "marginal_note", "legend_text", "other"]

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "map_type": {"type": "string"},
        "date_text": {"type": "string"}, "author": {"type": "string"},
        "scale_text": {"type": "string"}, "scale_bar": {"type": "string"},
        "orientation": {"type": "string", "description": "north arrow / orientation as shown"},
        "area": {"type": "string", "description": "area shown (Reviere, waters, villages)"},
        "legend": {"type": "array", "items": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "colour, line style or sign"},
            "meaning": {"type": "string"}}, "required": ["symbol", "meaning"]}},
        "regions": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": REGION_KINDS},
            "box_2d": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind", "box_2d"]}},
        "features": {"type": "array", "items": {"type": "string"},
                     "description": "kinds of features shown: rivers, Klausen, roads, compartment lines, stands by colour …"},
        "condition": {"type": "string", "description": "legibility, damage, folds"},
        "notes": {"type": "string"},
    },
    "required": ["title", "map_type", "legend", "regions", "features"],
}

TILE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the label exactly as written"},
            "class": {"type": "string", "enum": LABEL_CLASSES},
            "box_2d": {"type": "array", "items": {"type": "integer"}, "description": "[ymin, xmin, ymax, xmax] in this tile, 0-1000"},
            "ink": {"type": "string", "description": "black / red / blue / pencil / printed"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["text", "class", "box_2d", "confidence"]}},
        "symbols": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "description": "Klause, Brücke, Wehr, Grenzstein, Triftbau, building …"},
            "description": {"type": "string"},
            "box_2d": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind", "box_2d"]}},
    },
    "required": ["labels", "symbols"],
}

OVERVIEW_PROMPT = '''You analyse a historical map from the records of a Bavarian forest management plan (Waldstandsrevision
1878/90, Ilzertrift-Komplex: Reviere Schönau, St. Oswald, Klingenbrunn, Forstamt Schönberg).
{hints}
Describe the sheet: title and map type, date, author, scale statement and scale bar, orientation, the area shown,
every legend item (symbol or colour and its meaning, transcribed as written), the kinds of features shown and the
condition of the sheet. Give box_2d (0-1000, [ymin, xmin, ymax, xmax]) for the map body, legend, title cartouche, scale
bar, north arrow, insets and marginal texts.
EXISTING DESCRIPTION FROM THE TRANSCRIPTION PIPELINE (may contain errors):
{transcript}
'''

TILE_PROMPT = '''This image is one tile ({tile}) of a historical map (Ilzertrift-Komplex, Bavarian Forest, 1878/90).
Map title: {title}. Legend: {legend}

Transcribe EVERY label on this tile exactly as written: place names, waters, mountains, forest districts (Distrikt
names), compartment numbers and letters, roads, timber-floating structures (Klausen, Triftbauten), field names,
boundary marks and marginal notes. Give each label its class, its box_2d in THIS tile (0-1000, [ymin, xmin, ymax, xmax]),
the ink (printed, black, red, blue, pencil) and a confidence. Labels cut at the tile edge: transcribe the visible part
and give confidence "low". Also list map symbols (Klausen, bridges, weirs, buildings) with box_2d.
'''


def tile_grid(size: tuple[int, int], body: tuple[float, float, float, float], target: int = 450,
              overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = body
    width, height = right - left, bottom - top
    columns, rows = max(1, round(width / target)), max(1, round(height / target))
    tile_w, tile_h = width / columns, height / rows
    tiles = []
    for r in range(rows):
        for c in range(columns):
            x0 = max(0, left + c * tile_w - overlap * tile_w)
            y0 = max(0, top + r * tile_h - overlap * tile_h)
            x1 = min(size[0], left + (c + 1) * tile_w + overlap * tile_w)
            y1 = min(size[1], top + (r + 1) * tile_h + overlap * tile_h)
            tiles.append((int(x0), int(y0), int(x1), int(y1)))
    return tiles


def body_box(overview: dict, size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = size
    boxes = [r["box_2d"] for r in overview.get("regions", []) if r.get("kind") == "map_body" and len(r.get("box_2d", [])) == 4]
    if not boxes:
        return 0, 0, width, height
    ymin, xmin, ymax, xmax = boxes[0]
    return xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height


def upscale(image: Image.Image, longest: int = 2000) -> Image.Image:
    factor = longest / max(image.size)
    return image.resize((int(image.width * factor), int(image.height * factor)), Image.LANCZOS) if factor > 1 else image


def to_page(box: list[int], tile: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = tile
    ymin, xmin, ymax, xmax = box
    return [x0 + xmin / 1000 * (x1 - x0), y0 + ymin / 1000 * (y1 - y0), x0 + xmax / 1000 * (x1 - x0),
            y0 + ymax / 1000 * (y1 - y0)]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def similar(a: str, b: str) -> float:
    a, b = re.sub(r"\W+", "", a.lower()), re.sub(r"\W+", "", b.lower())
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


RANK = {"high": 3, "medium": 2, "low": 1}


def merge_labels(labels: list[dict]) -> list[dict]:
    merged = []
    for label in sorted(labels, key=lambda l: (-RANK.get(l.get("confidence"), 0), -len(l["text"]))):
        duplicate = next((m for m in merged if iou(m["box_px"], label["box_px"]) > 0.2 and similar(m["text"], label["text"]) > 0.6), None)
        if duplicate:
            duplicate["seen_in_tiles"].append(label["tile"])
        else:
            merged.append({**label, "seen_in_tiles": [label["tile"]]})
    return merged


def read_map(plan: PagePlan, spec: dict, unit_dir: Path, gemini: Gemini, out_dir: Path, force: bool = False,
             overview_model: str | None = None) -> dict:
    result_path = out_dir / "map.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    if not plan.page["image"]:
        return {}
    image = Image.open(unit_dir / plan.page["image"]).convert("RGB")
    transcript = (unit_dir / plan.page["transcript"]).read_text(encoding="utf-8")
    hints = "\n".join(f"{k}: {v}" for k, v in (spec.get("map") or {}).items())
    started = time.time()
    overview, usage, _ = gemini.extract(OVERVIEW_PROMPT.format(hints=hints, transcript=transcript[:6000]),
                                        OVERVIEW_SCHEMA, images=[image], model=overview_model, thinking="high")
    usages = [usage]
    tiles = tile_grid(image.size, body_box(overview, image.size))
    legend = "; ".join(f"{item['symbol']} = {item['meaning']}" for item in overview.get("legend", [])) or "none"
    labels, symbols = [], []
    for index, tile in enumerate(tiles):
        crop = upscale(image.crop(tile))
        prompt = TILE_PROMPT.format(tile=f"{index + 1} of {len(tiles)}", title=overview.get("title", ""), legend=legend)
        data, usage, _ = gemini.extract(prompt, TILE_SCHEMA, images=[crop], thinking="low")
        usages.append(usage)
        (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
        (out_dir / "tiles" / f"t{index:02d}.json").write_text(json.dumps({"tile": tile, "result": data},
                                                                        ensure_ascii=False, indent=1), encoding="utf-8")
        for label in data.get("labels", []):
            if len(label.get("box_2d", [])) == 4:
                labels.append({**label, "tile": index, "box_px": to_page(label["box_2d"], tile)})
        for symbol in data.get("symbols", []):
            if len(symbol.get("box_2d", [])) == 4:
                symbols.append({**symbol, "tile": index, "box_px": to_page(symbol["box_2d"], tile)})
    record = {"unit": spec["unit"], "position": plan.position, "page_id": plan.page["pid"],
              "image": plan.page["image"], "image_size": list(image.size), "overview": overview, "tiles": tiles,
              "labels": merge_labels(labels), "symbols": symbols, "usage": usages,
              "elapsed_s": round(time.time() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def labels_geojson(record: dict) -> dict:
    features = []
    for label in record.get("labels", []):
        x0, y0, x1, y1 = label["box_px"]
        features.append({"type": "Feature",
                         "geometry": {"type": "Polygon", "coordinates": [[[x0, -y0], [x1, -y0], [x1, -y1], [x0, -y1], [x0, -y0]]]},
                         "properties": {k: label.get(k) for k in ("text", "class", "ink", "confidence")}
                         | {"unit": record["unit"], "page_id": record["page_id"]}})
    return {"type": "FeatureCollection", "name": f"{record['unit']} labels (pixel coordinates, y negated)",
            "features": features}
