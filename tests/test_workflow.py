from ecological_ie.checks import check_unit, column_totals
from ecological_ie.html_tables import column_paths, parse_tables
from ecological_ie.maps import iou, merge_labels, tile_grid, to_page
from ecological_ie.normalize import KeyState, normalise_table, pair_value, parse_number
from ecological_ie.spec import page_plans, spec_problems
from ecological_ie.textdoc import apply_corrections, quote_status, segments_for

FORM = {"columns": [
    {"id": "year", "label": "Jahr", "type": "period"},
    {"id": "district", "label": "Distrikt", "type": "key", "key": "district_no"},
    {"id": "area", "label": "Tagw.", "type": "pair_major", "unit": "Tagwerk", "variable": "area"},
    {"id": "area_dec", "label": "Dez.", "type": "pair_minor", "of": "area", "mode": "digits"},
    {"id": "fee", "label": "fl.", "type": "pair_major", "unit": "fl"},
    {"id": "fee_kr", "label": "kr.", "type": "pair_minor", "of": "fee", "mode": "currency"},
]}
COLUMNS = [{"header": c["label"], "canonical": c["id"]} for c in FORM["columns"]]


def row(cells, row_type="data", **extra):
    return {"row_type": row_type, "box_2d": [0, 0, 10, 10], "cells": cells, **extra}


def normalise(rows, form=FORM):
    context = {"unit": "X", "position": 1, "page_id": "p", "table": 0, "form": "F", "data_index": 0}
    return normalise_table({"columns": COLUMNS, "rows": rows}, form, KeyState(), {}, context)


def test_parse_number_formats():
    assert parse_number("16151,74").value == 16151.74
    assert parse_number("1.234").value == 1234
    assert parse_number("3 ½").value == 3.5
    assert parse_number("3 25 1/2").flag == "ambiguous_space"
    assert parse_number("—").value == 0.0 and parse_number("—").flag == "dash"
    assert parse_number('"').flag == "ditto"
    assert parse_number("Summa").value is None
    assert parse_number("= Meter 139342").value == 139342
    assert parse_number("x7y").flag == "not_a_number"


def test_pair_modes():
    assert pair_value(parse_number("1"), "056", "digits").value == 1.056
    assert abs(pair_value(parse_number("209"), "20", "base60").value - 209.3333) < 1e-3
    assert pair_value(parse_number("3"), "5", "base100").value == 3.05
    assert abs(pair_value(parse_number("12"), "30", "currency", "fl").value - 12.5) < 1e-9
    assert pair_value(parse_number("12"), "30", "currency", "M").value == 12.3
    assert pair_value(parse_number("12"), "75", "base60").flag == "pair_minor_out_of_range"


def test_ditto_keys_and_pairs_are_normalised():
    rows, cells = normalise([
        row(["1861", "IV", "1", "056", "10", "20"]),
        row(["1862", '"', "2", "157", "", ""]),
        row(["Su", "", "3", "213", "10", "20"], "sum"),
    ])
    assert rows[1]["key_district_no"] == "IV"
    assert rows[1]["values"]["area"] == 2.157
    assert rows[0]["period"] == "1861"
    assert {c["canonical"] for c in cells} == {c["id"] for c in FORM["columns"]}


def test_column_total_passes_and_flags_misreading():
    rows, _ = normalise([row(["1861", "I", "1", "056", "", ""]), row(["1862", "I", "2", "157", "", ""]),
                         row(["Su", "", "3", "213", "", ""], "sum")])
    assert [r["status"] for r in column_totals(rows, FORM)] == ["ok"]
    rows, _ = normalise([row(["1861", "I", "1", "056", "", ""]), row(["1862", "I", "2", "157", "", ""]),
                         row(["Su", "", "3", "218", "", ""], "sum")])
    assert column_totals(rows, FORM)[0]["status"] == "mismatch"


def test_conversion_row_opens_next_section_and_unit_switches_currency():
    rows, _ = normalise([
        row(["1874", "I", "", "", "10", "30"]), row(["1875", "I", "", "", "5", "30"]),
        row(["Su", "", "", "", "16", "0"], "sum"),
        row(["= M.", "", "", "", "27", "43"], "carry_over", sets_key={"key": "unit", "value": "M"}),
        row(["1876", "I", "", "", "2", "57"]),
        row(["Su", "", "", "", "30", "00"], "sum"),
    ])
    results = [r for r in column_totals(rows, FORM) if r["column"] == "fee"]
    assert [r["status"] for r in results] == ["ok", "ok"]


def test_row_periods_fill_pages_without_year_column():
    form = {**FORM, "row_periods": ["1860/61", "1861/62"]}
    rows, _ = normalise([row(["", "I", "1", "0", "", ""]), row(["", "I", "1", "0", "", ""])], form)
    assert [r["period"] for r in rows] == ["1860/61", "1861/62"]


def test_carry_over_link_between_pages():
    first, _ = normalise([row(["1861", "I", "1", "5", "", ""]), row(["Übertrag", "", "1", "5", "", ""], "carry_over")])
    context = {"unit": "X", "position": 2, "page_id": "p2", "table": 0, "form": "F", "data_index": 0}
    second, _ = normalise_table({"columns": COLUMNS, "rows": [row(["Übertrag", "", "1", "6", "", ""], "carry_over")]},
                                FORM, KeyState(), {}, context)
    links = [r for r in check_unit([(FORM, first), (FORM, second)]) if r["rule"] == "carry_over_from_previous_page"]
    assert links and links[0]["status"] == "mismatch"


def test_column_paths_resolve_spans():
    html = """```html
<table><thead><tr><th rowspan="2">Jahr</th><th colspan="2">Fläche</th></tr><tr><th>Tagw.</th><th>Dez.</th></tr></thead>
<tbody><tr><td>1861</td><td>1</td><td>056</td></tr></tbody></table>
```"""
    assert column_paths(parse_tables(html)[0]) == ["Jahr", "Fläche › Tagw.", "Fläche › Dez."]


def test_spec_problems_and_page_plans():
    unit = {"pages": [{"position": 1, "kind": "Ta"}, {"position": 2, "kind": "Ta"}]}
    spec = {"unit": "X", "title": "T", "summary": "A summary that is long enough.", "default_profile": "table",
            "default_form": "F", "pages": {1: {"role": "title", "profile": "text"}},
            "forms": {"F": {"name": "n", "description": "d", "columns": FORM["columns"]}}}
    assert spec_problems(spec, unit) == []
    plans = page_plans(unit, spec)
    assert [(p.profile, p.form) for p in plans] == [("text", None), ("table", "F")]
    broken = {**spec, "forms": {"F": {"name": "n", "description": "d",
                                      "columns": [{"id": "a", "label": "a", "type": "pair_minor"}]}}}
    assert any("pair_minor" in problem for problem in spec_problems(broken, unit))


def test_text_corrections_and_quotes():
    text, applied = apply_corrections("Der Sturm vom 26. Oktober 1876 warf", {
        "corrections": [{"transcript_reads": "1876", "image_reads": "1870", "kind": "number", "confidence": "high"},
                        {"transcript_reads": "nicht da", "image_reads": "x", "kind": "word", "confidence": "low"}]})
    assert text == "Der Sturm vom 26. Oktober 1870 warf"
    assert [a["status"] for a in applied] == ["applied", "not_found"]
    assert quote_status("Sturm vom 26. Oktober", "doc", text) == "exact_on_page"
    assert quote_status("Hagel", "doc", text) == "missing"


def test_default_segments_cover_all_pages():
    class Plan:
        def __init__(self, position):
            self.position = position
    plans = [Plan(p) for p in range(1, 11)]
    assert segments_for(plans, {}) == [(1, 4), (5, 8), (9, 10)]


def test_map_tiles_and_label_merge():
    tiles = tile_grid((1400, 1400), (0, 0, 1400, 1400), target=700)
    assert len(tiles) == 4
    box = to_page([0, 0, 500, 500], (100, 100, 300, 300))
    assert box == [100, 100, 200, 200]
    labels = [{"text": "Rachelsee", "confidence": "high", "box_px": [0, 0, 10, 10], "tile": 0},
              {"text": "Rachelse", "confidence": "low", "box_px": [1, 1, 10, 10], "tile": 1}]
    merged = merge_labels(labels)
    assert len(merged) == 1 and merged[0]["seen_in_tiles"] == [0, 1]
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0


def test_second_reading_score_prefers_consistent_grid():
    from ecological_ie.consensus import needs_second_reading, page_score
    good = {"result": {"tables": [{"columns": COLUMNS, "rows": [
        row(["1861", "I", "1", "056", "10", "20"]), row(["1862", "I", "2", "157", "5", "10"]),
        row(["Su", "", "3", "213", "15", "30"], "sum")]}]}}
    shifted = {"result": {"tables": [{"columns": COLUMNS, "rows": [
        row(["1861", "I", "1", "056", "20", "10"]), row(["1862", "I", "2", "157", "10", "75"]),
        row(["Su", "", "3", "218", "30", "15"], "sum")]}]}}
    good_score, shifted_score = page_score(good, FORM), page_score(shifted, FORM)
    assert good_score["value"] > shifted_score["value"]
    assert shifted_score["out_of_range"] == 1 and shifted_score["mismatch"] >= 1
    assert not needs_second_reading(good_score)


def test_unit_heading_without_model_key_switches_unit():
    from ecological_ie.normalize import unit_in_label
    assert [unit_in_label(t) for t in ["= Ster", "Klafter.", "M. ₰", "Tgw.", "Su", "1872"]] == \
        ["Ster", "Klafter", "M", "Tagwerk", "", ""]
    rows, _ = normalise([row(["", "", "", "", "fl.", "kr."], "heading"), row(["1875", "I", "", "", "10", "30"]),
                         row(["= M. Pf.", "", "", "", "18", "00"], "carry_over"), row(["1876", "I", "", "", "2", "57"])])
    assert [r.get("key_unit") for r in rows] == ["fl", "fl", "M", "M"]
    assert rows[3]["values"]["fee"] == 2.57
