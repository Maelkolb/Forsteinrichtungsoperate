import json
import re
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

from .html_tables import body_width, column_paths, parse_tables
from .pages import parse_md

PROFILES = ["text", "table", "map", "skip"]
ROLES = ["title", "table", "continuation", "text", "signature", "map", "empty", "copy", "other"]
COLUMN_TYPES = ["key", "label", "text", "number", "pair_major", "pair_minor", "year", "period"]
KEY_NAMES = ["revier", "district_no", "district_name", "compartment_no", "subcompartment", "stand_name",
             "betriebsklasse", "altersklasse", "sortiment", "unit", "other"]
PAIR_MODES = ["digits", "base100", "base60", "currency"]
KIND_PROFILE = {"Te": "text", "Ta": "table", "Ka": "map"}

COLUMN_SCHEMA = {
    "type": "object",
    "required": ["id", "label", "type"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]*$"},
        "label": {"type": "string"},
        "meaning": {"type": "string"},
        "type": {"enum": COLUMN_TYPES},
        "key": {"enum": KEY_NAMES},
        "unit": {"type": "string"},
        "of": {"type": "string"},
        "mode": {"enum": PAIR_MODES},
        "describe": {"type": "boolean"},
        "variable": {"type": "string", "pattern": r"^[a-z][a-z0-9_]*$"},
        "period": {"type": "string"},
        "scope": {"type": "string"},
    },
}

FORM_SCHEMA = {
    "type": "object",
    "required": ["name", "description", "columns"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "columns": {"type": "array", "minItems": 1, "items": COLUMN_SCHEMA},
        "row_sums": {"type": "array", "items": {
            "type": "object", "required": ["target", "of"], "additionalProperties": False,
            "properties": {"target": {"type": "string"}, "of": {"type": "array", "items": {"type": "string"}}}}},
        "row_periods": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "object", "additionalProperties": False, "properties": {
            "scope_keys": {"type": "array", "items": {"enum": KEY_NAMES}},
            "scope": {"type": "string"}}},
    },
}

SPEC_SCHEMA = {
    "type": "object",
    "required": ["unit", "title", "summary", "default_profile"],
    "additionalProperties": False,
    "properties": {
        "unit": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string", "minLength": 20},
        "reading_order": {"type": "array", "items": {"type": "integer"}},
        "default_profile": {"enum": PROFILES},
        "default_form": {"type": "string"},
        "pages": {"type": "object", "additionalProperties": {
            "type": "object", "additionalProperties": False, "properties": {
                "role": {"enum": ROLES}, "profile": {"enum": PROFILES}, "form": {"type": "string"},
                "note": {"type": "string"}}}},
        "forms": {"type": "object", "additionalProperties": FORM_SCHEMA},
        "segments": {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                                 "minItems": 2, "maxItems": 2}},
        "map": {"type": "object", "additionalProperties": {"type": "string"}},
        "notes": {"type": "string"},
    },
}


@dataclass
class PagePlan:
    position: int
    page: dict
    role: str
    profile: str
    form: str | None


def load_unit(unit_dir: Path) -> dict:
    return json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))


def load_spec(unit_dir: Path) -> dict:
    path = unit_dir / "spec.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def spec_problems(spec: dict, unit: dict) -> list[str]:
    problems = [f"{'/'.join(map(str, error.path)) or 'spec'}: {error.message}"
                for error in jsonschema.Draft202012Validator(SPEC_SCHEMA).iter_errors(spec)]
    if problems:
        return problems
    positions = {page["position"] for page in unit["pages"]}
    forms = spec.get("forms", {})
    if spec.get("reading_order") and sorted(spec["reading_order"]) != sorted(positions):
        problems.append("reading_order must list every page position exactly once")
    if spec.get("default_form") and spec["default_form"] not in forms:
        problems.append(f"default_form {spec['default_form']!r} is not defined")
    for position, override in (spec.get("pages") or {}).items():
        if int(position) not in positions:
            problems.append(f"pages.{position}: no such page")
        if override.get("form") and override["form"] not in forms:
            problems.append(f"pages.{position}: form {override['form']!r} is not defined")
    for name, form in forms.items():
        ids = [column["id"] for column in form["columns"]]
        if len(ids) != len(set(ids)):
            problems.append(f"forms.{name}: duplicate column ids")
        for column in form["columns"]:
            if column["type"] == "pair_minor" and column.get("of") not in ids:
                problems.append(f"forms.{name}.{column['id']}: pair_minor needs 'of' = a pair_major column")
            if column["type"] == "pair_minor" and not column.get("mode"):
                problems.append(f"forms.{name}.{column['id']}: pair_minor needs 'mode'")
            if column["type"] == "key" and not column.get("key"):
                problems.append(f"forms.{name}.{column['id']}: key column needs 'key'")
        for rule in form.get("row_sums", []):
            for column_id in [rule["target"], *rule["of"]]:
                if column_id not in ids:
                    problems.append(f"forms.{name}.row_sums: unknown column {column_id!r}")
    plans = page_plans(unit, spec)
    if any(plan.profile == "table" and plan.form is None for plan in plans):
        problems.append("table pages without form: set default_form or pages.N.form")
    return problems


def page_plans(unit: dict, spec: dict) -> list[PagePlan]:
    by_position = {page["position"]: page for page in unit["pages"]}
    order = spec.get("reading_order") or sorted(by_position)
    overrides = {int(k): v for k, v in (spec.get("pages") or {}).items()}
    plans = []
    for position in order:
        page = by_position[position]
        override = overrides.get(position, {})
        profile = override.get("profile") or (spec.get("default_profile") if spec else None) \
            or KIND_PROFILE.get(page["kind"], "text")
        role = override.get("role") or {"table": "table", "map": "map", "skip": "other"}.get(profile, "text")
        form = override.get("form") or spec.get("default_form") if profile == "table" else None
        plans.append(PagePlan(position, page, role, profile, form))
    return plans


def column_id(index: int) -> str:
    return f"c{index:02d}"


def draft_spec(unit_dir: Path) -> str:
    unit = load_unit(unit_dir)
    kinds = [page["kind"] for page in unit["pages"]]
    default_profile = KIND_PROFILE[max(set(kinds), key=kinds.count)]
    pages, header_groups = {}, {}
    for page in unit["pages"]:
        _, body = parse_md((unit_dir / page["transcript"]).read_text(encoding="utf-8"))
        tables = parse_tables(body)
        if page["kind"] == "Ta" and not tables:
            pages[page["position"]] = {"role": "title" if page["chars"] < 800 else "text", "profile": "text"}
        elif KIND_PROFILE[page["kind"]] != default_profile:
            pages[page["position"]] = {"profile": KIND_PROFILE[page["kind"]]}
        for table in tables[:1]:
            paths = column_paths(table)
            if paths:
                header_groups.setdefault(len(paths), []).append((page["position"], paths, body_width(table)))
    forms = {}
    for number, (width, members) in enumerate(sorted(header_groups.items(), key=lambda item: -len(item[1])), 1):
        name = f"F{number}"
        positions = [position for position, _, _ in members]
        forms[name] = {"name": "TODO", "description": f"TODO – header with {width} columns on pages {positions}",
                       "columns": [{"id": column_id(i), "label": path, "type": "TODO"}
                                   for i, path in enumerate(members[0][1], 1)]}
        if number > 1:
            for position in positions:
                pages.setdefault(position, {})["form"] = name
    spec = {"unit": unit["id"], "title": unit["title"], "summary": "TODO", "default_profile": default_profile}
    if forms:
        spec["default_form"] = "F1"
    if pages:
        spec["pages"] = dict(sorted(pages.items()))
    if forms:
        spec["forms"] = forms
    return yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=120)


def spec_summary_for_prompt(spec: dict, form_name: str | None = None) -> str:
    lines = [f"UNIT: {spec.get('unit')} – {spec.get('title')}", (spec.get("summary") or "").strip()]
    if form_name and form_name in (spec.get("forms") or {}):
        form = spec["forms"][form_name]
        lines += [f"FORM {form_name}: {form['name']}", form["description"].strip(), "CANONICAL COLUMNS (id: label – meaning):"]
        for column in form["columns"]:
            extra = ", ".join(f"{k}={column[k]}" for k in ("type", "unit", "period") if column.get(k))
            lines.append(f"- {column['id']}: {column['label']}" + (f" – {column['meaning']}" if column.get("meaning") else "")
                         + f" [{extra}]")
    return "\n".join(line for line in lines if line)


def slug_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
