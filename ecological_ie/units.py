import csv
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from .dump import Dump
from .images import ImageResolver
from .pages import clean_body, page_quality, parse_md
from .toc import Unit, build_units, load_annotations, load_toc_ui

PREVIOUS_TAIL_CHARS = 600


def slug(text: str, length: int = 40) -> str:
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("Ä", "Ae").replace("Ö", "Oe") \
        .replace("Ü", "Ue").replace("ß", "ss")
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:length].rstrip("-")


def unit_folder(unit: Unit) -> str:
    return f"{unit.id}_{slug(unit.title)}"


def page_stem(position: int, page) -> str:
    return f"p{position:03d}_{page.sig.replace(' ', '')}_{page.num:04d}_{page.kind}"


def prepare_unit(unit: Unit, dump: Dump, resolver: ImageResolver, out_dir: Path) -> dict:
    unit_dir = out_dir / unit_folder(unit)
    (unit_dir / "pages").mkdir(parents=True, exist_ok=True)
    (unit_dir / "images").mkdir(exist_ok=True)
    pages = []
    for position, page in enumerate(unit.pages, 1):
        stem = page_stem(position, page)
        raw = dump.read_text(page.md_path)
        meta, body = parse_md(raw)
        (unit_dir / "pages" / f"{stem}.md").write_text(raw, encoding="utf-8")
        image = resolver.find(page)
        image_file, image_size, image_source = "", None, "none"
        if image:
            data, image_size, image_source = image
            image_file = f"images/{stem}.jpg"
            (unit_dir / image_file).write_bytes(data)
        pages.append({"position": position, **asdict(page), "source_type": page.source_type,
                      "transcript": f"pages/{stem}.md", "image": image_file, "image_source": image_source,
                      "image_size": image_size, "processing_mode": meta.get("processing_mode", ""),
                      **page_quality(body)})
    record = {"id": unit.id, "heft": unit.heft, "nr": unit.nr, "title": unit.title, "raw": unit.raw,
              "note": unit.note, "done": unit.done, "folder": unit_folder(unit),
              "annotated_order_differs": unit.annotated_order_differs, "pages": pages}
    (unit_dir / "unit.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def overview_row(record: dict) -> dict:
    pages = record["pages"]
    kinds = {kind: sum(p["kind"] == kind for p in pages) for kind in ("Te", "Ta", "Ka")}
    return {"id": record["id"], "title": record["title"], "pages": len(pages),
            "text": kinds["Te"], "table": kinds["Ta"], "map": kinds["Ka"],
            "volumes": " + ".join(sorted({p["sig"] for p in pages})),
            "page_range": f"{min(p['num'] for p in pages)}–{max(p['num'] for p in pages)}",
            "categories": "; ".join(sorted({p["category"] for p in pages})),
            "chars": sum(p["chars"] for p in pages),
            "illegible": sum(p["illegible"] for p in pages), "uncertain": sum(p["uncertain"] for p in pages),
            "images": "; ".join(sorted({p["image_source"] for p in pages})),
            "annotated_order_differs": record["annotated_order_differs"], "folder": record["folder"]}


def prepare_units(dump_path: Path, toc_ui_path: Path, annotations_path: Path, out_dir: Path,
                  scans_dir: Path | None = None, order: str = "volume", max_side: int = 3072,
                  specs_dir: Path | None = None) -> list[dict]:
    toc_ui, annotations = load_toc_ui(toc_ui_path), load_annotations(annotations_path)
    if annotations.get("toc") != toc_ui["toc"]["pid"]:
        print(f"! annotation export belongs to TOC {annotations.get('toc')!r}, UI shows {toc_ui['toc']['pid']!r}")
    units, missing, unlinked = build_units(toc_ui, annotations, order)
    dump = Dump(dump_path)
    resolver = ImageResolver(dump, scans_dir, max_side)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for unit in units:
        records.append(prepare_unit(unit, dump, resolver, out_dir))
        spec_file = specs_dir / f"{unit.id}.yaml" if specs_dir else None
        if spec_file and spec_file.exists():
            shutil.copy(spec_file, out_dir / unit_folder(unit) / "spec.yaml")
        print(f"{unit.id:6s} {len(unit.pages):3d} pages  {unit.title[:70]}")
    rows = [overview_row(record) for record in records]
    with open(out_dir / "units.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {"toc": toc_ui["toc"]["pid"], "toc_title": next((s["title"] for s in toc_ui["toc"]["sections"]
                                                             if s["id"] == "kopf"), ""),
             "annotation_export": annotations.get("exported"), "order": order, "units": rows,
             "unlinked_sections": [{"id": s["id"], "title": s["title"]} for s in unlinked],
             "missing_documents": missing}
    (out_dir / "units.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(records)} units, {sum(len(r['pages']) for r in records)} pages → {out_dir}")
    if missing:
        print(f"! {len(missing)} annotated documents not in the UI index: {missing[:5]}")
    print(f"TOC sections without annotated pages: {', '.join(s['id'] for s in unlinked)}")
    return records


def load_unit_pages(units_dir: Path, sections=None, source_types=("text", "table")) -> list[dict]:
    pages = []
    for unit_file in sorted(units_dir.glob("*/unit.json")):
        unit = json.loads(unit_file.read_text(encoding="utf-8"))
        if sections and unit["id"] not in sections:
            continue
        unit_dir = unit_file.parent
        brief_file = unit_dir / "brief.md"
        brief = brief_file.read_text(encoding="utf-8") if brief_file.exists() else ""
        previous_text = ""
        for page in unit["pages"]:
            meta, body = parse_md((unit_dir / page["transcript"]).read_text(encoding="utf-8"))
            text = clean_body(body, page["source_type"])
            if page["source_type"] in source_types:
                pages.append({
                    "seq": f"{unit['id']}__{Path(page['transcript']).stem}", "unit": unit["id"],
                    "source_type": page["source_type"], "subtype": page["category"], "page_id": page["pid"],
                    "file": str(unit_dir / page["transcript"]), "text": text,
                    "image": str(unit_dir / page["image"]) if page["image"] else "",
                    "unit_title": f"{unit['heft']} {unit['nr']} {unit['title']}".strip(),
                    "position": f"{page['position']}/{len(unit['pages'])}", "brief": brief,
                    "previous_tail": previous_text[-PREVIOUS_TAIL_CHARS:] if page["source_type"] == "text" else ""})
            previous_text = text
    return pages
