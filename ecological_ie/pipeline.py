import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .checks import check_unit
from .gemini import Gemini
from .grid import read_grid
from .layout import apply_row_layout, load_layout
from .normalize import KeyState, normalise_table
from .recheck import mark_discrepancies
from .spec import load_spec, load_unit, page_plans


def unit_dirs(units_dir: Path, sections=None) -> list[Path]:
    dirs = sorted(path.parent for path in units_dir.glob("*/unit.json"))
    return [d for d in dirs if not sections or load_unit(d)["id"] in sections]


def run_parallel(tasks: list, workers: int, label: str) -> tuple[list, dict]:
    results, errors = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task): name for name, task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=label):
            try:
                results.append(future.result())
            except Exception as error:
                errors[futures[future]] = str(error)
                print("ERROR", futures[future], str(error)[:300])
    return results, errors


def grid_path(run_dir: Path, unit_id: str, position: int) -> Path:
    return run_dir / "tables" / "grids" / unit_id / f"p{position:03d}.json"


def run_tables(units_dir: Path, run_dir: Path, gemini: Gemini, sections=None, positions=None, force=False,
               model=None, thinking=None) -> tuple[list, dict]:
    tasks = []
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec:
            print(f"skip {unit['id']}: no spec.yaml")
            continue
        for plan in page_plans(unit, spec):
            if plan.profile != "table" or (positions and plan.position not in positions):
                continue
            out = grid_path(run_dir, unit["id"], plan.position)
            tasks.append((f"{unit['id']} p{plan.position}",
                          lambda plan=plan, spec=spec, unit_dir=unit_dir, out=out:
                          read_grid(plan, spec, unit_dir, gemini, out, force, model, thinking)))
    return run_parallel(tasks, gemini.settings.workers, "table grids")


def apply_overrides(grid: dict, overrides: dict) -> dict:
    for (table_index, row_index, col), raw in overrides.items():
        rows = grid["result"]["tables"][table_index]["rows"]
        if row_index < len(rows) and col < len(rows[row_index]["cells"]):
            rows[row_index]["cells"][col] = raw
    return grid


def load_overrides(run_dir: Path) -> dict:
    path = run_dir / "tables" / "overrides.json"
    overrides = {}
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            key = (item["unit"], item["position"])
            overrides.setdefault(key, {})[(item["table"], item["row"], item["col"])] = item["raw"]
    return overrides


def derive_tables(units_dir: Path, run_dir: Path, sections=None) -> dict[str, pd.DataFrame]:
    overrides = load_overrides(run_dir)
    all_rows, all_cells, all_checks, outside, corrections = [], [], [], [], []
    for unit_dir in unit_dirs(units_dir, sections):
        unit, spec = load_unit(unit_dir), load_spec(unit_dir)
        if not spec:
            continue
        state, previous, unit_tables, current_form = KeyState(), {}, [], None
        for plan in page_plans(unit, spec):
            path = grid_path(run_dir, unit["id"], plan.position)
            if plan.profile != "table" or not path.exists():
                continue
            if plan.form != current_form:
                state, previous, current_form = KeyState(), {}, plan.form
            grid = apply_overrides(json.loads(path.read_text(encoding="utf-8")),
                                   overrides.get((unit["id"], plan.position), {}))
            grid = apply_row_layout(grid, load_layout(run_dir, unit["id"], plan.position))
            form = spec.get("forms", {}).get(plan.form, {})
            result = grid["result"]
            for table_index, table in enumerate(result.get("tables", [])):
                context = {"unit": unit["id"], "position": plan.position, "page_id": plan.page["pid"],
                           "page_role": plan.role, "table": table_index, "form": plan.form, "data_index": 0}
                rows, cells = normalise_table(table, form, state, previous, context)
                all_rows += rows
                all_cells += cells
                unit_tables.append((form, rows))
            outside += [{"unit": unit["id"], "position": plan.position, "page_id": plan.page["pid"], **item}
                        for item in result.get("text_outside_tables", [])]
            corrections += [{"unit": unit["id"], "position": plan.position, "page_id": plan.page["pid"], **item}
                            for item in result.get("image_corrections", [])]
        all_checks += check_unit(unit_tables)
    frames = {"table_rows": pd.DataFrame(all_rows), "table_cells": pd.DataFrame(all_cells),
              "table_checks": mark_discrepancies(pd.DataFrame(all_checks), run_dir),
              "table_outside_text": pd.DataFrame(outside),
              "table_corrections": pd.DataFrame(corrections)}
    out_dir = run_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    return frames


def check_summary(checks: pd.DataFrame) -> pd.DataFrame:
    if checks.empty:
        return checks
    return checks.groupby(["unit", "status"]).size().unstack(fill_value=0)
