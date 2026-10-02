import base64
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_SCRIPT = re.compile(r'<script id="data" type="text/plain">\s*(.*?)\s*</script>', re.S)
SIGNATURE = re.compile(r"A (\d+)(?: ([IVX]+))?$")
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
SOURCE_TYPES = {"Te": "text", "Ta": "table", "Ka": "map"}


@dataclass
class PageRef:
    key: str
    sig: str
    num: int
    kind: str
    pid: str
    category: str
    branch: str
    run: str
    md_path: str
    doc_type: str
    annotated_index: int
    note: str = ""

    @property
    def source_type(self) -> str:
        return SOURCE_TYPES.get(self.kind, "text")


@dataclass
class Unit:
    id: str
    heft: str
    nr: str
    title: str
    raw: str
    note: str = ""
    done: bool = False
    pages: list[PageRef] = field(default_factory=list)

    @property
    def annotated_order_differs(self) -> bool:
        return [p.annotated_index for p in self.pages] != sorted(p.annotated_index for p in self.pages)


def load_toc_ui(path: Path) -> dict:
    match = DATA_SCRIPT.search(Path(path).read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{path}: no embedded data block – is this the Forsteinrichtung TOC UI?")
    return json.loads(gzip.decompress(base64.b64decode(match.group(1))))


def load_annotations(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("tool") != "forst_toc_ui":
        raise ValueError(f"{path}: not an export of the TOC UI (tool={data.get('tool')!r})")
    return data


def volume_order(sig: str) -> tuple[int, int]:
    match = SIGNATURE.search(sig)
    if not match:
        return 9999, 0
    return int(match.group(1)), ROMAN.get(match.group(2) or "", 0)


def page_ref(doc: dict, link: dict, index: int) -> PageRef:
    return PageRef(key=doc["k"], sig=doc["sig"], num=doc["num"], kind=doc["k"].split("|")[2], pid=doc["pid"],
                   category=doc["cat"], branch=doc["ber"], run=doc["run"], md_path=doc["path"],
                   doc_type=doc.get("dt", ""), annotated_index=index, note=link.get("n", ""))


def build_units(toc_ui: dict, annotations: dict, order: str = "volume") -> tuple[list[Unit], list[str], list[dict]]:
    docs = {doc["k"]: doc for doc in toc_ui["docs"]}
    state = annotations["state"]
    units, missing, unlinked = [], [], []
    for section in toc_ui["toc"]["sections"]:
        links = state["links"].get(section["id"], [])
        if not links:
            unlinked.append(section)
            continue
        unit = Unit(id=section["id"], heft=section["heft"], nr=section["nr"], title=section["title"],
                    raw=section["raw"], note=state.get("notes", {}).get(section["id"], ""),
                    done=bool(state.get("done", {}).get(section["id"])))
        for index, link in enumerate(links):
            if link["d"] in docs:
                unit.pages.append(page_ref(docs[link["d"]], link, index))
            else:
                missing.append(link["d"])
        if order == "volume":
            unit.pages.sort(key=lambda page: (volume_order(page.sig), page.num))
        units.append(unit)
    known = {section["id"] for section in toc_ui["toc"]["sections"]}
    for section_id in set(state["links"]) - known:
        missing += [f"{section_id}: {link['d']}" for link in state["links"][section_id]]
    return units, missing, unlinked
