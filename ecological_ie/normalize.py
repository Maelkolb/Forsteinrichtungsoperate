import re
from dataclasses import dataclass, field

DITTO = {'"', "„", "“", "”", "''", ",,", "〃", "do", "do.", "dto", "dto.", "detto", "desgl", "desgl.", "dto:", "″"}
DASHES = {"—", "–", "-", "—.—", "-.-", ".", "·", "..", "...", "—.", "-.", "o", "0.0"}
FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875}
KEY_ORDER = ["revier", "betriebsklasse", "district_no", "district_name", "compartment_no", "subcompartment",
             "stand_name", "altersklasse", "sortiment"]


@dataclass
class Number:
    value: float | None
    flag: str = ""
    decimals: int = 0


def is_ditto(raw: str) -> bool:
    return raw.strip().lower() in DITTO


def strip_markup(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).replace(" ", " ").strip()


def parse_number(raw: str) -> Number:
    text = strip_markup(raw)
    if not text:
        return Number(None, "empty")
    if is_ditto(text):
        return Number(None, "ditto")
    if text in DASHES:
        return Number(0.0, "dash")
    unit_words = re.match(r"^=?\s*((?:[A-Za-zÄÖÜäöüß₰][A-Za-zÄÖÜäöüß.₰]*\s*)+)(?=[\d½¼¾]|$)", text)
    if unit_words:
        rest = text[unit_words.end():].strip()
        if not rest:
            return Number(None, "unit_marker")
        number = parse_number(rest)
        return Number(number.value, number.flag or "unit_in_cell", number.decimals)
    text = text.rstrip(".=-").replace("'", "").strip()
    fraction = 0.0
    for symbol, amount in FRACTIONS.items():
        if text.endswith(symbol):
            fraction, text = amount, text[: -len(symbol)].strip()
    match = re.fullmatch(r"(.*?)\s*(\d)/(\d{1,2})", text)
    if match and match.group(1).strip() and int(match.group(3)) > 0:
        fraction, text = int(match.group(2)) / int(match.group(3)), match.group(1).strip()
    if not text and fraction:
        return Number(fraction, "fraction", 2)
    if re.fullmatch(r"\d{1,3}(?:[. ]\d{3})*,\d+", text) or re.fullmatch(r"\d+,\d+", text):
        whole, decimals = text.replace(".", "").replace(" ", "").split(",")
        return Number(float(f"{whole}.{decimals}") + fraction, "", len(decimals))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        return Number(float(text.replace(".", "")) + fraction)
    if re.fullmatch(r"\d+\.\d+", text):
        return Number(float(text) + fraction, "", len(text.split(".")[1]))
    if re.fullmatch(r"\d+", text):
        return Number(float(text) + fraction, "fraction" if fraction else "")
    if re.fullmatch(r"\d+ \d+", text):
        return Number(None, "ambiguous_space")
    return Number(None, "not_a_number")


def currency_base(unit_key: str) -> int:
    return 60 if (unit_key or "fl").lower().strip(". ") in ("fl", "gulden", "fl kr", "fl. kr") else 100


def pair_value(major: Number, minor_raw: str, mode: str, unit_key: str = "") -> Number:
    if mode == "currency":
        mode = "base60" if currency_base(unit_key) == 60 else "base100"
    minor_text = strip_markup(minor_raw).rstrip(".")
    major_value = major.value or 0.0
    if not minor_text or minor_text in DASHES:
        return Number(major.value, major.flag)
    if not re.fullmatch(r"\d+", minor_text):
        minor = parse_number(minor_text)
        if minor.value is None:
            return Number(None, "pair_" + minor.flag)
        return Number(major_value + minor.value / (60 if mode == "base60" else 100), "pair")
    if mode == "digits":
        return Number(float(f"{int(major_value)}.{minor_text}"), "pair", len(minor_text))
    base = 60 if mode == "base60" else 100
    minor_value = int(minor_text)
    flag = "pair" if minor_value < base else "pair_minor_out_of_range"
    return Number(major_value + minor_value / base, flag, 2)


@dataclass
class KeyState:
    keys: dict = field(default_factory=dict)

    def update(self, new: dict):
        for name, value in new.items():
            if not value:
                continue
            if self.keys.get(name) != value and name in KEY_ORDER:
                for lower in KEY_ORDER[KEY_ORDER.index(name) + 1:]:
                    if lower not in new:
                        self.keys.pop(lower, None)
            self.keys[name] = value


def normalise_table(table: dict, form: dict, state: KeyState, previous: dict, context: dict) -> tuple[list, list]:
    columns = {column["id"]: column for column in form.get("columns", [])}
    page_columns = table.get("columns", [])
    rows_out, cells_out = [], []
    for row_index, row in enumerate(table.get("rows", [])):
        cells = row.get("cells", [])
        red = {item["col"]: item["text"] for item in row.get("red", []) if isinstance(item, dict)}
        uncertain = set(row.get("uncertain", []))
        row_type = row.get("row_type", "data")
        canonical_raw = {}
        for index, page_column in enumerate(page_columns):
            canonical = page_column.get("canonical") or "other"
            raw = cells[index] if index < len(cells) else ""
            effective = raw if strip_markup(raw) else red.get(index, "")
            if canonical != "other":
                canonical_raw.setdefault(canonical, []).append((index, raw, effective))
        if (row.get("sets_key") or {}).get("value"):
            state.update({row["sets_key"]["key"]: row["sets_key"]["value"]})
        values, row_keys = {}, {}
        for column_id, entries in canonical_raw.items():
            column = columns.get(column_id, {"type": "text"})
            effective = " ".join(e for _, _, e in entries if strip_markup(e)).strip()
            if row_type == "data" and is_ditto(effective):
                effective = previous.get(column_id, effective)
            if row_type == "data" and strip_markup(effective):
                previous[column_id] = effective
            if column["type"] == "key" and strip_markup(effective) and row_type in ("data", "sum"):
                row_keys[column.get("key", "other")] = strip_markup(effective)
            values[column_id] = effective
        if row_type == "data":
            state.update(row_keys)
        keys = dict(state.keys) if row_type == "data" else {**state.keys, **row_keys}
        period = next((strip_markup(values[c["id"]]) for c in columns.values()
                       if c["type"] in ("period", "year") and strip_markup(values.get(c["id"], ""))), "")
        period = re.sub(r"\s*[–-]\s*", "–", re.sub(r"\s+", " ", period)).strip()
        if not period and row_type == "data" and form.get("row_periods"):
            sequence = form["row_periods"]
            period = sequence[context["data_index"]] if context["data_index"] < len(sequence) else "?"
        if row_type == "data":
            context["data_index"] += 1
        normalised = {}
        for column_id, column in columns.items():
            if column_id not in values:
                continue
            raw = values[column_id]
            if column["type"] == "pair_minor":
                continue
            if column["type"] in ("number", "year") or column["type"] == "pair_major":
                number = parse_number(raw)
                if column["type"] == "pair_major":
                    minor = next((c for c in columns.values() if c["type"] == "pair_minor" and c.get("of") == column_id), None)
                    if minor and minor["id"] in values:
                        number = pair_value(number, values[minor["id"]], minor.get("mode", "digits"),
                                            keys.get("unit", ""))
                normalised[column_id] = number
        rows_out.append({**{k: v for k, v in context.items() if k != "data_index"}, "row": row_index,
                         "row_type": row_type, "box_2d": row.get("box_2d"), "period": period,
                         "note": row.get("note", ""), **{f"key_{k}": v for k, v in keys.items()},
                         "label": next((strip_markup(c) for c in cells if strip_markup(c)), ""),
                         "shape_ok": len(cells) == len(page_columns),
                         "values": {k: n.value for k, n in normalised.items()},
                         "decimals": {k: n.decimals for k, n in normalised.items()},
                         "flags": {k: n.flag for k, n in normalised.items() if n.flag}})
        for index, page_column in enumerate(page_columns):
            canonical = page_column.get("canonical") or "other"
            column = columns.get(canonical, {})
            number = normalised.get(canonical) if column.get("type") != "pair_minor" else None
            cells_out.append({**{k: v for k, v in context.items() if k != "data_index"}, "row": row_index,
                              "row_type": row_type, "col": index,
                              "header": page_column.get("header", ""), "canonical": canonical,
                              "raw": cells[index] if index < len(cells) else "", "red": red.get(index, ""),
                              "uncertain": index in uncertain,
                              "value": number.value if number else None, "measure_unit": column.get("unit", ""),
                              "flag": number.flag if number else "",
                              "variable": column.get("variable", ""), "period": column.get("period", ""),
                              "scope": column.get("scope", "")})
    return rows_out, cells_out
