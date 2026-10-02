TOLERANCE = 1e-6
BAD_FLAGS = {"not_a_number", "ambiguous_space", "pair_not_a_number", "pair_ambiguous_space", "pair_minor_out_of_range"}


def numeric_columns(form: dict) -> list[str]:
    return [c["id"] for c in form.get("columns", []) if c["type"] in ("number", "pair_major")]


def value_of(row: dict, column: str) -> float:
    return row["values"].get(column) or 0.0


def unparseable(rows: list[dict], column: str) -> bool:
    return any(row["flags"].get(column) in BAD_FLAGS for row in rows)


def check_record(row: dict, column: str, rule: str, found: float, candidates: dict, rows_used: list[dict]) -> dict:
    matched = [name for name, expected in candidates.items() if abs(expected - found) <= TOLERANCE * max(1, abs(found))]
    status = "ok" if matched else ("unparseable" if unparseable(rows_used, column) else "mismatch")
    return {"unit": row["unit"], "position": row["position"], "page_id": row["page_id"], "table": row["table"],
            "row": row["row"], "column": column, "rule": rule, "found": found,
            "expected": {name: round(value, 6) for name, value in candidates.items()},
            "matched": matched, "status": status, "rows_used": [(r["position"], r["table"], r["row"]) for r in rows_used]}


def column_totals(rows: list[dict], form: dict) -> list[dict]:
    results = []
    for column in numeric_columns(form):
        section, sums, all_data, opening = [], [], [], None
        for row in rows:
            kind = row["row_type"]
            if kind == "data":
                section.append(row)
                all_data.append(row)
                continue
            found = row["values"].get(column)
            if kind not in ("sum", "carry_over") or found is None or row["flags"].get(column) in ("empty", "ditto"):
                continue
            if not section:
                opening = row
                continue
            opening_value = value_of(opening, column) if opening else 0.0
            candidates = {"section": sum(value_of(r, column) for r in section) + opening_value,
                          "all_rows": sum(value_of(r, column) for r in all_data) + opening_value}
            if sums:
                candidates["sum_of_sums"] = sum(value_of(r, column) for r in sums)
            used = section + ([opening] if opening else [])
            results.append(check_record(row, column, "column_total", found, candidates, used))
            sums.append(row)
            section, opening = [], None
    return results


def row_sums(rows: list[dict], form: dict) -> list[dict]:
    results = []
    for rule in form.get("row_sums", []):
        for row in rows:
            if row["row_type"] not in ("data", "sum", "carry_over"):
                continue
            found = row["values"].get(rule["target"])
            parts = [row["values"].get(column) for column in rule["of"]]
            if found is None or all(part is None for part in parts):
                continue
            expected = sum(part or 0.0 for part in parts)
            results.append(check_record(row, rule["target"], "row_sum " + "+".join(rule["of"]), found,
                                        {"row_sum": expected}, [row]))
    return results


def carry_over_links(tables: list[tuple[dict, list[dict]]]) -> list[dict]:
    results = []
    previous_closing = None
    for form, rows in tables:
        if not rows:
            continue
        first = next((r for r in rows if r["row_type"] in ("data", "carry_over", "sum")), None)
        if first and first["row_type"] == "carry_over" and previous_closing is not None:
            for column in numeric_columns(form):
                found, expected = first["values"].get(column), previous_closing["values"].get(column)
                if found is not None and expected is not None:
                    results.append(check_record(first, column, "carry_over_from_previous_page", found,
                                                {"previous_page": expected}, [previous_closing]))
        closing = [r for r in rows if r["row_type"] in ("sum", "carry_over")]
        previous_closing = closing[-1] if closing else None
    return results


def check_unit(tables: list[tuple[dict, list[dict]]]) -> list[dict]:
    results = []
    for form, rows in tables:
        results += column_totals(rows, form)
        results += row_sums(rows, form)
    return results + carry_over_links(tables)
