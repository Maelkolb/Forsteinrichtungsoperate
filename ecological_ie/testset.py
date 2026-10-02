from pathlib import Path

import pandas as pd

from .pages import clean_body, parse_md

DEFAULT_TESTSET = Path(__file__).parent / "testset"


def load_testset(data_dir: Path = DEFAULT_TESTSET) -> list[dict]:
    manifest = pd.read_csv(data_dir / "manifest.csv")
    pages = []
    for _, row in manifest.iterrows():
        meta, body = parse_md((data_dir / row["file"]).read_text(encoding="utf-8"))
        pages.append({
            "seq": row["seq"], "source_type": row["source_type"], "subtype": row["subtype"],
            "page_id": meta.get("page_id", row["page_id"]), "file": row["file"], "meta": meta,
            "text": clean_body(body, row["source_type"]), "chars": len(body), "image": ""})
    return pages
