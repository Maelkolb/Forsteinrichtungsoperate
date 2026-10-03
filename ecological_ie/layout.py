import json
import math
from pathlib import Path

import numpy as np

from .dump import Dump
from .edition import plain_text
from .normalize import strip_markup
from .spec import load_spec, load_unit, page_plans

TEXT_REGIONS = {"ParagraphRegion", "TitleRegion", "MarginaliaRegion", "FootnoteRegion"}
ROW_SKIP, LEAD_BAND_SKIP, MAX_ROW_GAP, MAX_GAP_BANDS, MAX_SEGMENT = 1.0, 0.05, 4, 8, 60


def lines_path(run_dir: Path, unit_id: str, position: int) -> Path:
    return run_dir / "layout" / "lines" / unit_id / f"p{position:03d}.json"


def layout_path(run_dir: Path, unit_id: str, position: int) -> Path:
    return run_dir / "layout" / "pages" / unit_id / f"p{position:03d}.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_layout(run_dir: Path, unit_id: str, position: int) -> dict | None:
    return read_json(layout_path(run_dir, unit_id, position))


def normalised_box(box: list, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [round(y0 / height * 1000), round(x0 / width * 1000), round(y1 / height * 1000), round(x1 / width * 1000)]


def baseline_y(line: dict) -> float:
    points = line.get("baseline") or []
    if points:
        return float(np.median([y for _, y in points]))
    return line["box"][1] + 0.75 * (line["box"][3] - line["box"][1])


def visual_lines(lines: list[dict]) -> list[dict]:
    """Group detected line pieces that share a baseline height into bands (one table row or one text line)."""
    if not lines:
        return []
    typical = float(np.median([l["box"][3] - l["box"][1] for l in lines])) or 10.0
    bands = []
    for line in sorted(lines, key=baseline_y):
        y = baseline_y(line)
        if bands and abs(y - bands[-1]["base"]) < 0.45 * typical:
            band = bands[-1]
            band["lines"].append(line)
            band["base"] = float(np.mean([baseline_y(l) for l in band["lines"]]))
        else:
            bands.append({"lines": [line], "base": y})
    for band in bands:
        boxes = [l["box"] for l in band["lines"]]
        band["x0"], band["y0"] = min(b[0] for b in boxes), min(b[1] for b in boxes)
        band["x1"], band["y1"] = max(b[2] for b in boxes), max(b[3] for b in boxes)
        band["cy"] = (band["base"] * 2 + band["y0"]) / 3
    return bands


def covered_columns(band: dict, columns: list[tuple[float, float]]) -> set[int]:
    covered = set()
    for line in band["lines"]:
        x0, x1 = line["box"][0], line["box"][2]
        for c, (c0, c1) in enumerate(columns):
            overlap = max(0.0, min(x1, c1) - max(x0, c0))
            if overlap >= 0.35 * max(1.0, x1 - x0) or overlap >= 0.5 * max(1.0, c1 - c0):
                covered.add(c)
    return covered


def occupied_columns(row: dict) -> set[int]:
    return {c for c, cell in enumerate(row.get("cells", [])) if strip_markup(str(cell or "")).strip()}


def align_table(table: dict, lines: list[dict], width: int, height: int) -> list[dict] | None:
    rows = table.get("rows", [])
    boxed = [r for r in rows if len(r.get("box_2d") or []) == 4]
    columns = [((c.get("x_range") or [0, 0])[0] / 1000 * width, (c.get("x_range") or [0, 0])[1] / 1000 * width)
               for c in table.get("columns", [])]
    if len(boxed) < 2 or not columns or not lines:
        return None
    left, right = min(c[0] for c in columns), max(c[1] for c in columns)
    if right - left < 10:
        left, right = min(r["box_2d"][1] for r in boxed) / 1000 * width, max(r["box_2d"][3] for r in boxed) / 1000 * width
    top, bottom = min(r["box_2d"][0] for r in boxed) / 1000 * height, max(r["box_2d"][2] for r in boxed) / 1000 * height
    margin = 0.2 * (bottom - top) + 30
    inside = [l for l in lines if left - 10 <= (l["box"][0] + l["box"][2]) / 2 <= right + 10
              and top - margin <= (l["box"][1] + l["box"][3]) / 2 <= bottom + margin]
    bands = visual_lines(inside)
    if len(bands) < 2:
        return None
    for band in bands:
        band["cols"] = covered_columns(band, columns)
    n_cols = max(1, len(columns))
    expected = [((r["box_2d"][0] + r["box_2d"][2]) / 2000 * height) if len(r.get("box_2d") or []) == 4 else None for r in rows]
    occupied = [occupied_columns(r) for r in rows]
    spans = [r.get("row_type") in ("group_header", "heading", "note") and len(occupied[i]) <= 2 for i, r in enumerate(rows)]
    data_rows = [occupied[i] for i, r in enumerate(rows) if r.get("row_type", "data") == "data" and occupied[i]] or [o for o in occupied if o]
    key_columns = {c for c in range(n_cols) if sum(c in o for o in data_rows) >= 0.6 * len(data_rows)}
    gaps = [b2["cy"] - b1["cy"] for b1, b2 in zip(bands, bands[1:])]
    pitch = float(np.median(gaps)) if gaps else 30.0

    def row_like(j: int) -> bool:
        return bool(key_columns) and len(bands[j]["cols"] & key_columns) >= max(1, 0.5 * len(key_columns))

    def band_skip(j: int) -> float:
        return 0.9 if row_like(j) else 0.1 + 0.2 * len(bands[j]["cols"]) / n_cols

    usable = [i for i in range(len(rows)) if expected[i] is not None and (occupied[i] or spans[i])]
    if len(usable) < 2:
        return None
    n, m = len(usable), len(bands)
    expected_lines = {i: max(1, round((rows[i]["box_2d"][2] - rows[i]["box_2d"][0]) / 1000 * height / pitch)) for i in usable}
    expected_start = {i: rows[i]["box_2d"][0] / 1000 * height + 0.5 * pitch for i in usable}
    table_top = min(rows[i]["box_2d"][0] for i in usable) / 1000 * height - pitch
    table_bottom = max(rows[i]["box_2d"][2] for i in usable) / 1000 * height + pitch
    gap = [band_skip(j) if table_top <= band["cy"] <= table_bottom else LEAD_BAND_SKIP for j, band in enumerate(bands)]
    gap_before = np.concatenate([[0.0], np.cumsum(gap)])

    def segment_cost(i: int, first: int, last: int, covered: set[int]) -> float:
        if spans[i]:
            fit = 0.6 if covered else 0.0
        else:
            union = occupied[i] | covered
            fit = len(occupied[i] & covered) / len(union) if union else 0.0
        return 1.0 - fit + 0.4 * abs(math.log((last - first + 1) / expected_lines[i]))

    # each row gets a run of consecutive visual lines; runs are ordered, separated by at most a few skipped lines
    cost = np.full((n, m), np.inf)
    back = {}
    for a in range(n):
        i = usable[a]
        longest = min(MAX_SEGMENT, 2 * expected_lines[i] + 2)
        for last in range(m):
            covered = set()
            for first in range(last, max(-1, last - longest), -1):
                covered = covered | bands[first]["cols"]
                offset = bands[first]["cy"] - expected_start[i]
                best, choice = ROW_SKIP * a + gap_before[first] + 0.35 * min(abs(offset) / pitch, 3), None
                for pa in range(max(0, a - MAX_ROW_GAP), a):
                    for previous_last in range(max(0, first - MAX_GAP_BANDS - 1), first):
                        if not np.isfinite(cost[pa, previous_last]):
                            continue
                        previous_offset = back[(pa, previous_last)][2]
                        total = (cost[pa, previous_last] + 0.35 * min(abs(offset - previous_offset) / pitch, 2)
                                 + ROW_SKIP * (a - pa - 1) + gap_before[first] - gap_before[previous_last + 1])
                        if total < best:
                            best, choice = total, (pa, previous_last)
                total = best + segment_cost(i, first, last, covered)
                if total < cost[a, last]:
                    cost[a, last] = total
                    back[(a, last)] = (choice, first, offset)
    finals = [(cost[a, last] + ROW_SKIP * (n - 1 - a) + gap_before[m] - gap_before[last + 1], a, last)
              for a in range(n) for last in range(m) if np.isfinite(cost[a, last])]
    if not finals:
        return None
    _, a, last = min(finals)
    segments = {}
    state = (a, last)
    while state:
        choice, first, _ = back[state]
        segments[usable[state[0]]] = (first, state[1])
        state = choice
    if len(segments) < max(2, 0.5 * len(usable)):
        return None
    far = [i for i, (first, _) in segments.items()
           if not rows[i]["box_2d"][0] / 1000 * height - 3.5 * pitch <= bands[first]["cy"] <= rows[i]["box_2d"][2] / 1000 * height + 3.5 * pitch]
    if len(far) > 0.2 * len(segments):
        return None

    order = sorted(segments)
    centres = {i: (bands[segments[i][0]]["cy"], bands[segments[i][1]]["cy"]) for i in order}
    for i in range(len(rows)):
        if i in centres:
            continue
        before = [k for k in order if k < i]
        after = [k for k in order if k > i]
        if before and after:
            k0, k1 = before[-1], after[0]
            y = centres[k0][1] + (centres[k1][0] - centres[k0][1]) * (i - k0) / (k1 - k0)
        elif before:
            y = centres[before[-1]][1] + pitch * (i - before[-1])
        else:
            y = centres[after[0]][0] - pitch * (after[0] - i)
        centres[i] = (y, y)
    result = []
    for i in range(len(rows)):
        start, end = centres[i]
        top = (centres[i - 1][1] + start) / 2 if i > 0 else start - 0.55 * pitch
        bottom = (end + centres[i + 1][0]) / 2 if i + 1 < len(rows) else end + 0.5 * pitch
        result.append({"box": normalised_box([left, top, right, bottom], width, height), "how": "lines" if i in segments else "interpolated"})
    return result


def table_rows_layout(grid: dict, lines_page: dict | None) -> list:
    if not lines_page:
        return []
    width, height = lines_page["image_size"]
    return [align_table(table, lines_page["lines"], width, height) for table in grid["result"].get("tables", [])]


def apply_row_layout(grid: dict, layout: dict | None) -> dict:
    if not layout or not layout.get("rows"):
        return grid
    for table, aligned in zip(grid["result"].get("tables", []), layout["rows"]):
        if not aligned:
            continue
        for row, box in zip(table.get("rows", []), aligned):
            if box:
                row["box_gemini"] = row.get("box_2d")
                row["box_2d"] = box["box"]
                row["box_how"] = box["how"]
    return grid


def dump_regions(dump: Dump, page: dict) -> list[dict]:
    name = f"{page['run']}/regions/{page['pid']}.json"
    return json.loads(dump.read_text(name)) if dump.exists(name) else []


def region_crop(dump: Dump, page: dict, region: dict) -> bytes | None:
    name = f"{page['run']}/regions/{page['pid']}/{region['id']}_{region['type']}.png"
    return dump.read_bytes(name) if dump.exists(name) else None


def region_transform(dump: Dump, page: dict, regions: list[dict], image_path: Path) -> tuple[float, float, float] | None:
    """Scale and offset from the dump's region coordinates (original scan) to the prepared page image."""
    import cv2

    target = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    candidates = sorted([r for r in regions if r.get("bbox") and r["type"] in TEXT_REGIONS | {"TableRegion"}],
                        key=lambda r: r["bbox"]["width"] * r["bbox"]["height"], reverse=True)[:2]
    best = None
    for region in candidates:
        data = region_crop(dump, page, region)
        if not data:
            continue
        crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        box = region["bbox"]
        for scale in np.arange(0.30, 1.06, 0.01):
            w, h = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
            if w < 20 or h < 12 or w >= target.shape[1] or h >= target.shape[0]:
                continue
            small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            x, y = box["x"] * scale, box["y"] * scale
            pad = 40
            x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
            x1, y1 = min(target.shape[1], int(x) + w + pad), min(target.shape[0], int(y) + h + pad)
            window = target[y0:y1, x0:x1]
            if window.shape[0] < h or window.shape[1] < w:
                continue
            scores = cv2.matchTemplate(window, small, cv2.TM_CCOEFF_NORMED)
            _, score, _, at = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (score, scale, x0 + at[0] - box["x"] * scale, y0 + at[1] - box["y"] * scale)
        if best and best[0] > 0.6:
            break
    if not best or best[0] < 0.45:
        return None
    return best[1], best[2], best[3]


def prose_lines(text: str) -> list[tuple[int, int]]:
    """Index and length of every transcript line that is running text, leaving out embedded HTML tables."""
    found, inside_fence, inside_table = [], False, False
    for index, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if stripped.startswith("```"):
            inside_fence = not inside_fence
            continue
        if "<table" in stripped:
            inside_table = True
        if inside_fence or inside_table or stripped.startswith(("|", "<!--")):
            inside_table = inside_table and "</table>" not in stripped
            continue
        length = len(plain_text(stripped))
        if length:
            found.append((index, length))
    return found


def region_bands(lines: list[dict], regions: list[tuple[str, list[float]]], width: int, height: int) -> list[dict]:
    """Detected lines grouped into visual lines per text region, in the regions' reading order."""
    pad_x, pad_y = 0.015 * width, 0.01 * height
    owner = {}
    for k, line in enumerate(lines):
        cx, cy = (line["box"][0] + line["box"][2]) / 2, (line["box"][1] + line["box"][3]) / 2
        containing = [(abs((x1 - x0) * (y1 - y0)), r) for r, (kind, (x0, y0, x1, y1)) in enumerate(regions)
                      if x0 - pad_x <= cx <= x1 + pad_x and y0 - pad_y <= cy <= y1 + pad_y]
        if containing:
            owner[k] = min(containing)[1]
    bands = []
    for r, (kind, _) in enumerate(regions):
        if kind not in TEXT_REGIONS:
            continue
        for band in visual_lines([line for k, line in enumerate(lines) if owner.get(k) == r]):
            band["region"] = r
            band["width"] = sum(line["box"][2] - line["box"][0] for line in band["lines"])
            bands.append(band)
    return bands


def align_by_length(source: list[tuple[int, int]], bands: list[dict]) -> dict[int, list[int]]:
    """Monotonic alignment of transcript lines to visual lines by comparing character counts with line widths.
    A transcript line may cover two visual lines (a joined hyphenation) and two transcript lines may share one."""
    if not source or not bands:
        return {}
    rate = sum(length for _, length in source) / max(1.0, sum(band["width"] for band in bands))

    def fit(chars: int, width: float) -> float:
        return abs(math.log((chars + 4) / (rate * width + 4)))

    n, m = len(source), len(bands)
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []
            if i and j:
                options.append((cost[i - 1, j - 1] + fit(source[i - 1][1], bands[j - 1]["width"]), 1))
            if i and j > 1 and bands[j - 1]["region"] == bands[j - 2]["region"]:
                options.append((cost[i - 1, j - 2] + fit(source[i - 1][1], bands[j - 1]["width"] + bands[j - 2]["width"]) + 0.35, 2))
            if i > 1 and j:
                options.append((cost[i - 2, j - 1] + fit(source[i - 1][1] + source[i - 2][1], bands[j - 1]["width"]) + 0.35, 3))
            if i:
                options.append((cost[i - 1, j] + 0.9, 4))
            if j:
                options.append((cost[i, j - 1] + 0.7, 5))
            cost[i, j], move[i, j] = min(options)
    pairs, i, j = {}, n, m
    while i > 0 or j > 0:
        step = move[i, j]
        if step == 1:
            pairs[source[i - 1][0]] = [j - 1]
            i, j = i - 1, j - 1
        elif step == 2:
            pairs[source[i - 1][0]] = [j - 2, j - 1]
            i, j = i - 1, j - 2
        elif step == 3:
            pairs[source[i - 1][0]] = pairs[source[i - 2][0]] = [j - 1]
            i, j = i - 2, j - 1
        elif step == 4:
            i -= 1
        else:
            j -= 1
    return pairs


def text_layout(text: str, regions: list[dict], transform: tuple, lines_page: dict) -> dict:
    width, height = lines_page["image_size"]
    scale, dx, dy = transform
    placed = []
    for region in sorted([r for r in regions if r.get("bbox")], key=lambda r: r.get("reading_order", 0)):
        box = region["bbox"]
        x0, y0 = box["x"] * scale + dx, box["y"] * scale + dy
        placed.append((region["type"], [x0, y0, x0 + box["width"] * scale, y0 + box["height"] * scale]))
    bands = region_bands(lines_page["lines"], placed, width, height)
    pairs = align_by_length(prose_lines(text), bands)
    mapped = {str(index): [[[round(x / width * 1000), round(y / height * 1000)] for x, y in line["polygon"]]
                           for j in targets for line in bands[j]["lines"]]
              for index, targets in pairs.items()}
    return {"regions": [{"type": kind, "box": normalised_box(box, width, height)} for kind, box in placed], "lines": mapped}


def build_layout(units_dir: Path, run_dir: Path, dump_path: Path, sections=None) -> dict:
    dump = Dump(dump_path)
    summary = {"table_pages": 0, "rows_from_lines": 0, "rows_shifted": 0, "rows_kept": 0, "text_pages": 0,
               "text_lines_mapped": 0, "text_lines": 0, "no_transform": []}
    for unit_file in sorted(units_dir.glob("*/unit.json")):
        unit_dir = unit_file.parent
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec or (sections and unit["id"] not in sections):
            continue
        for plan in page_plans(unit, spec):
            lines_page = read_json(lines_path(run_dir, unit["id"], plan.position))
            if not lines_page:
                continue
            out = {"image_size": lines_page["image_size"]}
            grid_file = run_dir / "tables" / "grids" / unit["id"] / f"p{plan.position:03d}.json"
            proof_file = run_dir / "text" / unit["id"] / "proof" / f"p{plan.position:03d}.json"
            if plan.profile == "table" and grid_file.exists():
                rows = table_rows_layout(json.loads(grid_file.read_text(encoding="utf-8")), lines_page)
                out["rows"] = rows
                summary["table_pages"] += 1
                for table in rows:
                    for row in table or []:
                        if row:
                            summary["rows_from_lines" if row["how"] == "lines" else "rows_shifted"] += 1
                grid = json.loads(grid_file.read_text(encoding="utf-8"))
                summary["rows_kept"] += sum(len(t.get("rows", [])) for t, aligned in zip(grid["result"].get("tables", []), rows) if not aligned)
            elif plan.profile == "text" and proof_file.exists():
                regions = dump_regions(dump, plan.page)
                transform = region_transform(dump, plan.page, regions, unit_dir / plan.page["image"]) if regions else None
                summary["text_pages"] += 1
                if transform:
                    text = json.loads(proof_file.read_text(encoding="utf-8"))["corrected"]
                    out.update(text_layout(text, regions, transform, lines_page))
                    out["transform"] = [round(v, 4) for v in transform]
                    summary["text_lines_mapped"] += len(out["lines"])
                    summary["text_lines"] += sum(1 for t in text.split("\n") if t.strip())
                else:
                    summary["no_transform"].append(f"{unit['id']} p{plan.position}")
            else:
                continue
            target = layout_path(run_dir, unit["id"], plan.position)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(out), encoding="utf-8")
    return summary
