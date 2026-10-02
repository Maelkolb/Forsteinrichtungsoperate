import json
from pathlib import Path

import pandas as pd

from .gemini import Gemini
from .pipeline import run_parallel, unit_dirs
from .spec import load_spec, load_unit, page_plans


def text_dir(run_dir: Path, unit_id: str) -> Path:
    return run_dir / "text" / unit_id


def write_frames(frames: dict[str, pd.DataFrame], run_dir: Path):
    out_dir = run_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)


def run_text(units_dir: Path, run_dir: Path, gemini: Gemini, sections=None, force=False) -> dict:
    from .textdoc import assemble, extract_segment, proofread_page, segments_for
    units = [(d, load_unit(d), load_spec(d)) for d in unit_dirs(units_dir, sections)]
    units = [(d, u, s) for d, u, s in units if s]
    proof_tasks = []
    for unit_dir, unit, spec in units:
        for plan in page_plans(unit, spec):
            if plan.profile == "text":
                out = text_dir(run_dir, unit["id"]) / "proof" / f"p{plan.position:03d}.json"
                proof_tasks.append((f"{unit['id']} p{plan.position}",
                                    lambda plan=plan, unit_dir=unit_dir, out=out:
                                    proofread_page(plan, unit_dir, gemini, out, force)))
    run_parallel(proof_tasks, gemini.settings.workers, "proofreading")
    segment_tasks = []
    for unit_dir, unit, spec in units:
        plans = [plan for plan in page_plans(unit, spec) if plan.profile == "text"]
        if not plans:
            continue
        proofs = {}
        for plan in plans:
            path = text_dir(run_dir, unit["id"]) / "proof" / f"p{plan.position:03d}.json"
            if path.exists():
                proofs[plan.position] = json.loads(path.read_text(encoding="utf-8"))
        document = assemble(plans, proofs)
        (text_dir(run_dir, unit["id"]) / "document.txt").write_text(document, encoding="utf-8")
        for segment in segments_for(plans, spec):
            out = text_dir(run_dir, unit["id"]) / "statements" / f"s{segment[0]:03d}-{segment[1]:03d}.json"
            segment_tasks.append((f"{unit['id']} {segment}",
                                  lambda document=document, spec=spec, segment=segment, out=out:
                                  extract_segment(document, spec, segment, gemini, out, force)))
    results, errors = run_parallel(segment_tasks, gemini.settings.workers, "statements")
    return {"segments": len(results), "errors": errors}


def derive_text(units_dir: Path, run_dir: Path, sections=None) -> dict[str, pd.DataFrame]:
    from .textdoc import page_anchor, quote_status, segment_positions
    statements, events, corrections, pages = [], [], [], []
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        base = text_dir(run_dir, unit["id"])
        if not spec or not (base / "document.txt").exists():
            continue
        document = (base / "document.txt").read_text(encoding="utf-8")
        plans = [plan for plan in page_plans(unit, spec) if plan.profile == "text"]
        page_texts = {}
        for plan in plans:
            path = base / "proof" / f"p{plan.position:03d}.json"
            if not path.exists():
                continue
            proof = json.loads(path.read_text(encoding="utf-8"))
            page_texts[page_anchor(plan.position)] = proof["corrected"]
            pages.append({"unit": unit["id"], "position": plan.position, "page_id": plan.page["pid"],
                          "role": plan.role, "page_quality": proof["result"].get("page_quality"),
                          "n_corrections": len(proof["applied"]),
                          "n_applied": sum(c["status"] == "applied" for c in proof["applied"]),
                          "transcript": proof["text"], "corrected_text": proof["corrected"]})
            corrections += [{"unit": unit["id"], "position": plan.position, "page_id": plan.page["pid"], **c}
                            for c in proof["applied"]]
        for path in sorted((base / "statements").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            allowed = {page_anchor(p) for p in segment_positions(plans, tuple(record["segment"]))}
            segment = f"{record['segment'][0]}-{record['segment'][1]}"
            for item in record["result"].get("statements", []):
                page = item.get("page", "")
                item = {("value_unit" if key == "unit" else key): value for key, value in item.items()}
                statements.append({**item, "unit": unit["id"], "segment": segment,
                                   "species": "; ".join(item.get("species") or []),
                                   "page_in_segment": page in allowed,
                                   "quote_check": quote_status(item.get("quote", ""), document, page_texts.get(page, ""))})
            for item in record["result"].get("events", []):
                page = item.get("page", "")
                events.append({"unit": unit["id"], "segment": segment, **item,
                               "quote_check": quote_status(item.get("quote", ""), document, page_texts.get(page, ""))})
    frames = {"text_pages": pd.DataFrame(pages), "text_corrections": pd.DataFrame(corrections),
              "text_statements": pd.DataFrame(statements), "text_events": pd.DataFrame(events)}
    write_frames(frames, run_dir)
    return frames


def run_maps(units_dir: Path, run_dir: Path, gemini: Gemini, sections=None, force=False, overview_model=None) -> list:
    from .maps import read_map
    tasks = []
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec:
            continue
        for plan in page_plans(unit, spec):
            if plan.profile == "map":
                out = run_dir / "maps" / unit["id"] / f"p{plan.position:03d}"
                tasks.append((f"{unit['id']} p{plan.position}",
                              lambda plan=plan, spec=spec, unit_dir=unit_dir, out=out:
                              read_map(plan, spec, unit_dir, gemini, out, force, overview_model)))
    results, _ = run_parallel(tasks, gemini.settings.workers, "maps")
    return results


def derive_maps(run_dir: Path) -> dict[str, pd.DataFrame]:
    from .maps import labels_geojson
    maps, labels, symbols, features = [], [], [], []
    for path in sorted((run_dir / "maps").glob("*/p*/map.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        overview = record["overview"]
        maps.append({"unit": record["unit"], "position": record["position"], "page_id": record["page_id"],
                     **{k: overview.get(k) for k in ("title", "map_type", "date_text", "author", "scale_text",
                                                     "scale_bar", "orientation", "area", "condition", "notes")},
                     "legend": json.dumps(overview.get("legend", []), ensure_ascii=False),
                     "features": "; ".join(overview.get("features", [])), "n_labels": len(record["labels"]),
                     "n_tiles": len(record["tiles"])})
        width, height = record["image_size"]
        for label in record["labels"]:
            x0, y0, x1, y1 = label["box_px"]
            labels.append({"unit": record["unit"], "position": record["position"], "page_id": record["page_id"],
                           "text": label["text"], "class": label["class"], "ink": label.get("ink", ""),
                           "confidence": label["confidence"], "x0": round(x0), "y0": round(y0), "x1": round(x1),
                           "y1": round(y1), "image_width": width, "image_height": height,
                           "n_tiles_seen": len(label["seen_in_tiles"])})
        symbols += [{"unit": record["unit"], "position": record["position"], "kind": s.get("kind"),
                     "description": s.get("description", ""), "x0": round(s["box_px"][0]), "y0": round(s["box_px"][1]),
                     "x1": round(s["box_px"][2]), "y1": round(s["box_px"][3])} for s in record["symbols"]]
        features += labels_geojson(record)["features"]
    out_dir = run_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "map_labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                                           ensure_ascii=False), encoding="utf-8")
    frames = {"maps": pd.DataFrame(maps), "map_labels": pd.DataFrame(labels), "map_symbols": pd.DataFrame(symbols)}
    write_frames(frames, run_dir)
    return frames


def describe_columns(units_dir: Path, sections=None) -> dict:
    columns = {}
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        for name, form in (spec.get("forms") or {}).items():
            columns[(unit["id"], name)] = {c["id"] for c in form["columns"] if c.get("describe")}
    return columns


def run_describe_stage(units_dir: Path, run_dir: Path, gemini: Gemini, sections=None, force=False) -> list:
    from .describe import run_describe
    derived = run_dir / "derived"
    cells = pd.read_json(derived / "table_cells.jsonl", lines=True)
    rows = pd.read_json(derived / "table_rows.jsonl", lines=True)
    units = sections or sorted(cells["unit"].unique())
    return run_describe(units, cells, rows, describe_columns(units_dir, sections), gemini,
                        run_dir / "describe", force=force)


def derive_describe(run_dir: Path) -> pd.DataFrame:
    from .describe import PARTS
    items = []
    for path in sorted((run_dir / "describe").glob("*/batch_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for item in record["result"].get("items", []):
            ref = item.get("ref", "")
            unit, page, table, row, col = (ref.split("/") + [""] * 5)[:5]
            text = record["texts"].get(ref, "")
            evidence = item.get("evidence") or []
            items.append({"ref": ref, "unit": unit, "position": int(page[1:]) if page[1:].isdigit() else None,
                          "table": int(table[1:]) if table[1:].isdigit() else None,
                          "row": int(row[1:]) if row[1:].isdigit() else None,
                          "col": int(col[1:]) if col[1:].isdigit() else None, "cell_text": text,
                          **{part: json.dumps(item.get(part), ensure_ascii=False) if item.get(part) else ""
                             for part in [*PARTS, "events"]},
                          "evidence": " | ".join(evidence),
                          "evidence_found": all(e in text for e in evidence) if evidence else None,
                          "not_ecological": item.get("not_ecological", False)})
    frame = pd.DataFrame(items)
    write_frames({"stand_descriptions": frame}, run_dir)
    return frame
