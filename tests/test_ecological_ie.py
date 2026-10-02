import base64
import gzip
import io
import json
import unicodedata
import zipfile

from PIL import Image

from ecological_ie.dump import Dump
from ecological_ie.images import ImageResolver, page_key, viewer_images
from ecological_ie.pages import clean_body, compact_html, dehyphenate, page_quality, parse_md, truncate_marginalia_loops
from ecological_ie.prompts import build_prompt
from ecological_ie.results import flatten, verify_quote
from ecological_ie.testset import load_testset
from ecological_ie.toc import build_units, load_toc_ui, volume_order
from ecological_ie.units import load_unit_pages, prepare_units


def jpeg_bytes(size=(40, 60)):
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, "JPEG")
    return out.getvalue()


def test_parse_md_keeps_brackets_in_page_id():
    meta, body = parse_md('---\npage_id: "[144] StALa, KdForsten A 383 II"\ndoc_type: table\n---\n## Titel\n')
    assert meta == {"page_id": "[144] StALa, KdForsten A 383 II", "doc_type": "table"}
    assert body == "## Titel\n"


def test_dehyphenate_joins_only_lowercase_continuations():
    assert dehyphenate("Wind=\nrißflächen") == "Windrißflächen"
    assert dehyphenate("Bau-\nund Nutzholz") == "Bauund Nutzholz"
    assert dehyphenate("Bau-\nNutzholz") == "Bau-\nNutzholz"


def test_marginalia_loops_are_truncated():
    body = "\n".join(["Text"] + ["> loop"] * 30 + ["Ende"])
    lines = truncate_marginalia_loops(body).splitlines()
    assert lines.count("> loop") == 25
    assert "5 further lines" in lines[26]
    assert lines[-1] == "Ende"


def test_compact_html_puts_cells_on_one_line():
    text = "```html\n<table>\n  <tr>\n    <td>1</td>\n    <td>2</td>\n  </tr>\n</table>\n```"
    assert compact_html(text) == "```html\n<table><tr><td>1</td><td>2</td></tr>\n</table>\n```"


def test_page_quality_counts_markers():
    quality = page_quality("a [?] b [illegible] c [?]\n```html\n<table></table>\n```")
    assert (quality["uncertain"], quality["illegible"], quality["tables"]) == (2, 1, 1)


def test_verify_quote_levels():
    text = "Der Borkenkäfer trat 1873 in  großer Menge auf und zerstörte viele Bestände"
    assert verify_quote("trat 1873", text) == "exact"
    assert verify_quote("1873 in großer Menge", text) == "normalised"
    assert verify_quote("Der Borkenkäfer trat 1873 in großer Zahl", text) == "prefix"
    assert verify_quote("Windwurf", text) == "missing"
    assert verify_quote("", text) == "empty"


def test_prompt_mentions_scan_only_with_image():
    page = {"source_type": "table", "page_id": "p", "subtype": "Periodentabelle", "text": "<table></table>",
            "unit_title": "I. Heft 11 Periodentabelle", "position": "2/36", "brief": "Form F.N. 6"}
    with_image, without_image = build_prompt(page, with_image=True), build_prompt(page)
    assert "SCAN:" in with_image and "SCAN:" not in without_image
    assert "DOCUMENT UNIT: I. Heft 11 Periodentabelle (page 2/36)" in without_image
    assert "UNIT BRIEF:\nForm F.N. 6" in without_image


def test_flatten_text_and_table_results():
    pages = {"t": {"text": "Sturm vom 26. Oktober 1870 warf viel Holz"}, "b": {"text": "Lage. Südwestliches Gehäng"}}
    results = {
        "t": {"source_type": "text", "subtype": "Protokolle", "page_id": "[1] X A 384 II", "result": {
            "findings": [{"category": "damage_event", "entity": "Windwurf", "quote": "Sturm vom 26. Oktober 1870",
                          "confidence": "high"}],
            "tree_species": [], "locations": [],
            "image_corrections": [{"where": "line 3", "transcript_reads": "1876", "image_reads": "1870"}]}},
        "b": {"source_type": "table", "subtype": "Periodentabelle", "page_id": "[49] X A 383 II", "result": {
            "records": [{"record_kind": "stand", "district_no": "III", "uncertain": False,
                         "description_raw": "Lage. Südwestliches Gehäng",
                         "stand": {"species": [{"name_de": "Fichte", "share_value": 0.7}], "age_raw": "102"},
                         "damage": [{"type": "Windbruch", "quote": "Windbruch"}]}]}},
    }
    tables = flatten(results, pages)
    assert tables["text_findings"].loc[0, "quote_check"] == "exact"
    assert tables["table_records"].loc[0, "stand_species"] == "Fichte 0.7"
    assert tables["table_records"].loc[0, "description_check"] == "exact"
    assert len(tables["table_damage"]) == 1
    assert tables["image_corrections"].loc[0, "image_reads"] == "1870"


def test_load_testset_reads_all_forty_pages():
    pages = load_testset()
    assert len(pages) == 40
    assert {page["source_type"] for page in pages} == {"text", "table"}


def test_volume_order_and_page_key():
    assert sorted(["A 384 II", "A 383 II", "A 384 I", "A 385"], key=volume_order) == \
        ["A 383 II", "A 384 I", "A 384 II", "A 385"]
    assert page_key("[077] Reg NB, KdForten A 384 II") == ("A 384 II", 77)
    assert page_key("[005] RegNB, KdForsten A 379 I") == ("A 379 I", 5)


def toc_ui_html(data: dict) -> str:
    blob = base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()
    return f'<html><script id="data" type="text/plain">{blob}</script><script>app()</script></html>'


def corpus_fixture(tmp_path):
    run = "Tabellen/Materialertrag-Scans-in-schlecher-Qualität/run"
    pids = {12: "[012] StALa, KdForsten A 383 II", 3: "[003] StALa, KdForsten A 383 II"}
    docs = [{"k": f"A 383 II|{num:04d}|Ta", "pid": pid, "ber": "Tabellen", "cat": "Periodentabelle", "run": run,
             "sig": "A 383 II", "num": num, "dt": "table", "path": f"{run}/md/{pid}.md"} for num, pid in pids.items()]
    toc = {"pid": "[002] StALa, KdForsten A 383 I", "sections": [
        {"id": "kopf", "heft": "Kopf", "nr": "", "title": "Waldstandsrevision", "raw": ""},
        {"id": "I-11", "heft": "I. Heft", "nr": "11", "title": "Periodentabelle", "raw": "11) Periodentabelle"}]}
    (tmp_path / "toc.html").write_text(toc_ui_html({"toc": toc, "docs": docs}), encoding="utf-8")
    annotations = {"tool": "forst_toc_ui", "toc": toc["pid"], "exported": "2026-09-28",
                   "state": {"links": {"I-11": [{"d": "A 383 II|0012|Ta", "n": ""}, {"d": "A 383 II|0003|Ta", "n": ""}]},
                             "notes": {}, "done": {}}}
    (tmp_path / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    image = base64.b64encode(jpeg_bytes()).decode()
    viewer = f'<img src="data:image/jpeg;base64,{image}" alt="Scan of {pids[3]}" loading="lazy">'
    zip_path = tmp_path / "combined.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        nfd_run = unicodedata.normalize("NFD", run)
        archive.writestr(f"Forsteinrichtungsoperate_combined/{nfd_run}/viewer.html", viewer)
        for pid in pids.values():
            archive.writestr(f"Forsteinrichtungsoperate_combined/{nfd_run}/md/{pid}.md",
                             f'---\npage_id: "{pid}"\ndoc_type: table\n---\n```html\n<table>\n  <tr>\n    <td>{pid}</td>\n  </tr>\n</table>\n```\n')
    return zip_path, toc, docs, annotations


def test_build_units_sorts_by_volume_and_reports_unlinked(tmp_path):
    _, toc, docs, annotations = corpus_fixture(tmp_path)
    units, missing, unlinked = build_units({"toc": toc, "docs": docs}, annotations)
    assert [page.num for page in units[0].pages] == [3, 12]
    assert units[0].annotated_order_differs
    assert missing == [] and [section["id"] for section in unlinked] == ["kopf"]
    assert load_toc_ui(tmp_path / "toc.html")["toc"]["pid"] == toc["pid"]


def test_dump_reads_nfd_zip_with_nfc_paths(tmp_path):
    zip_path, _, docs, _ = corpus_fixture(tmp_path)
    dump = Dump(zip_path)
    assert dump.exists(docs[0]["path"])
    assert "[012]" in dump.read_text(docs[0]["path"])


def test_viewer_images_and_resolver_fallback(tmp_path):
    zip_path, _, docs, _ = corpus_fixture(tmp_path)
    dump = Dump(zip_path)
    images = viewer_images(dump.read_text(f"{docs[0]['run']}/viewer.html"))
    assert list(images) == ["[003] StALa, KdForsten A 383 II"]
    units, _, _ = build_units({"toc": load_toc_ui(tmp_path / "toc.html")["toc"], "docs": docs},
                              json.loads((tmp_path / "annotations.json").read_text()))
    resolver = ImageResolver(dump)
    assert resolver.find(units[0].pages[0])[2] == "viewer"
    assert resolver.find(units[0].pages[1]) is None


def test_scans_take_precedence_over_viewer_images(tmp_path):
    zip_path, _, docs, annotations = corpus_fixture(tmp_path)
    scans = tmp_path / "scans"
    scans.mkdir()
    Image.new("RGB", (4000, 3000), "white").save(scans / "[003] StALa, KdForsten A 383 II.png")
    units, _, _ = build_units({"toc": load_toc_ui(tmp_path / "toc.html")["toc"], "docs": docs}, annotations)
    data, size, source = ImageResolver(Dump(zip_path), scans, max_side=1000).find(units[0].pages[0])
    assert source == "scan" and size == (1000, 750) and data[:2] == b"\xff\xd8"


def test_prepare_and_load_unit_pages(tmp_path):
    zip_path, *_ = corpus_fixture(tmp_path)
    out = tmp_path / "units"
    prepare_units(zip_path, tmp_path / "toc.html", tmp_path / "annotations.json", out)
    unit = json.loads((out / "I-11_Periodentabelle" / "unit.json").read_text(encoding="utf-8"))
    assert [page["image_source"] for page in unit["pages"]] == ["viewer", "none"]
    (out / "I-11_Periodentabelle" / "brief.md").write_text("Form F.N. 6", encoding="utf-8")
    pages = load_unit_pages(out)
    assert [page["position"] for page in pages] == ["1/2", "2/2"]
    assert pages[0]["image"].endswith(".jpg") and pages[1]["image"] == ""
    assert pages[0]["brief"] == "Form F.N. 6"
    assert pages[0]["text"] == clean_body(parse_md((out / "I-11_Periodentabelle" / unit["pages"][0]["transcript"]).read_text(encoding="utf-8"))[1], "table")
