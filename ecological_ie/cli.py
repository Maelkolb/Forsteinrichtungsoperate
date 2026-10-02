import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_MODEL, Settings, find_api_key
from .corpus import enumerate_corpus, estimate
from .dump import Dump
from .extract import extract_pages, usage_summary
from .gemini import Gemini
from .results import flatten, overview, scoring_sheet, write_tables
from .review import write_review_viewer
from .spec import draft_spec, load_spec, load_unit, spec_problems
from .testset import DEFAULT_TESTSET, load_testset
from .units import load_unit_pages, prepare_units

STAGES = ["tables", "regrid", "recheck", "describe", "text", "maps"]


def gemini_from(args) -> Gemini:
    settings = Settings(model=args.model, thinking_level=args.thinking, workers=args.workers,
                        image_resolution=args.image_resolution)
    return Gemini(find_api_key(args.env_file), settings)


def finish(results: dict, pages: list[dict], out_dir: Path, gemini: Gemini):
    pages_by_seq = {page["seq"]: page for page in pages}
    tables = flatten(results, pages_by_seq)
    write_tables(tables, results, out_dir)
    print(overview(tables))
    scoring_sheet(tables).to_csv(out_dir / "scoring_sheet.csv", index=False, encoding="utf-8-sig")
    title = f"Forsteinrichtung – ecological IE review ({gemini.settings.model}, thinking={gemini.settings.thinking_level})"
    print("review viewer:", write_review_viewer(results, pages_by_seq, out_dir, title))


def run_pages(pages: list[dict], args, use_image: bool):
    gemini = gemini_from(args)
    results, errors = extract_pages(pages, gemini, args.out / "raw", force=args.force, use_image=use_image)
    print(f"\n{len(results)} pages ok, {len(errors)} errors")
    print(usage_summary(results, gemini))
    if results:
        finish(results, pages, args.out, gemini)


def run_testset(args):
    pages = load_testset(args.data)
    if args.subset:
        pages = [page for page in pages if page["seq"] in args.subset]
    run_pages(pages, args, use_image=False)


def run_prepare(args):
    prepare_units(args.dump, args.toc_ui, args.annotations, args.out, args.scans, args.order, args.max_side,
                  args.specs)


def run_extract(args):
    pages = load_unit_pages(args.units, args.section)
    if args.limit:
        pages = pages[:args.limit]
    print(f"{len(pages)} pages from {len({page['unit'] for page in pages})} units (map pages are not extracted here)")
    run_pages(pages, args, use_image=not args.no_image)


def run_corpus(args):
    corpus = enumerate_corpus(Dump(args.dump))
    print(estimate(corpus, Settings().price_input_per_m))
    if args.csv:
        corpus.to_csv(args.csv, index=False, encoding="utf-8-sig")


def selected_units(units_dir: Path, sections) -> list[Path]:
    dirs = sorted(path.parent for path in units_dir.glob("*/unit.json"))
    return [d for d in dirs if not sections or load_unit(d)["id"] in sections]


def run_draft_specs(args):
    for unit_dir in selected_units(args.units, args.section):
        target = unit_dir / "spec.draft.yaml"
        target.write_text(draft_spec(unit_dir), encoding="utf-8")
        print("draft:", target)


def run_spec_check(args):
    failed = 0
    for unit_dir in selected_units(args.units, args.section):
        unit = load_unit(unit_dir)
        if not (unit_dir / "spec.yaml").exists():
            print(f"{unit['id']:6s} no spec.yaml")
            failed += 1
            continue
        try:
            problems = spec_problems(load_spec(unit_dir), unit)
        except Exception as error:
            problems = [f"cannot read spec.yaml: {error}"]
        print(f"{unit['id']:6s} {'ok' if not problems else str(len(problems)) + ' problem(s)'}")
        for problem in problems:
            print("       -", problem)
        failed += bool(problems)
    raise SystemExit(1 if failed else 0)


def log_run(run_dir: Path, stage: str, args, gemini: Gemini):
    run_dir.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "stage": stage, "sections": args.section,
             "model": args.model, "thinking": args.thinking, "recheck_model": getattr(args, "recheck_model", None),
             "calls": len(gemini.ledger), "cost_usd": round(gemini.spent(), 4),
             "input_tokens": sum(e["input_tokens"] for e in gemini.ledger),
             "output_tokens": sum(e["output_tokens"] + e["thinking_tokens"] for e in gemini.ledger)}
    with open(run_dir / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"{stage}: {entry['calls']} calls, ${entry['cost_usd']:.2f}")


def run_stages(args):
    from .pipeline import check_summary, derive_tables, run_tables
    from .recheck import run_recheck
    from .stages import derive_describe, derive_maps, derive_text, run_describe_stage, run_maps, run_text
    stages = STAGES if "all" in args.stage else args.stage
    gemini = gemini_from(args)
    for stage in stages:
        gemini.ledger.clear()
        if stage == "tables":
            run_tables(args.units, args.run, gemini, args.section, args.positions, args.force)
            print(check_summary(derive_tables(args.units, args.run, args.section)["table_checks"]))
        elif stage == "regrid":
            from .consensus import run_second_readings
            results = run_second_readings(args.units, args.run, gemini, args.section, args.recheck_model, "high")
            print(f"second readings: {len(results)}, kept: {sum(r['winner'] == 'second' for r in results)}")
            print(check_summary(derive_tables(args.units, args.run, args.section)["table_checks"]))
        elif stage == "recheck":
            for round_number in range(1, args.rounds + 1):
                run_recheck(args.units, args.run, gemini, args.recheck_model, "high", round_number)
                print(check_summary(derive_tables(args.units, args.run, args.section)["table_checks"]))
        elif stage == "describe":
            run_describe_stage(args.units, args.run, gemini, args.section, args.force)
            print(f"{len(derive_describe(args.run))} description cells decomposed")
        elif stage == "text":
            print(run_text(args.units, args.run, gemini, args.section, args.force))
            frames = derive_text(args.units, args.run, args.section)
            print({name: len(frame) for name, frame in frames.items()})
        elif stage == "maps":
            run_maps(args.units, args.run, gemini, args.section, args.force, args.recheck_model)
            print({name: len(frame) for name, frame in derive_maps(args.run).items()})
        log_run(args.run, stage, args, gemini)


def run_derive(args):
    from .pipeline import check_summary, derive_tables
    from .stages import derive_describe, derive_maps, derive_text
    print(check_summary(derive_tables(args.units, args.run)["table_checks"]))
    print({name: len(frame) for name, frame in derive_text(args.units, args.run).items()})
    print({name: len(frame) for name, frame in derive_maps(args.run).items()})
    if (args.run / "describe").exists():
        print(f"stand_descriptions: {len(derive_describe(args.run))}")


def run_publish(args):
    from .publish import publish
    ledger = []
    if (args.run / "runs.jsonl").exists():
        ledger = [{"cost_usd": json.loads(line)["cost_usd"]}
                  for line in (args.run / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    print("package:", publish(args.units, args.run, args.out or args.run / "package",
                              "Waldstandsrevision Ilzertrift-Komplex 1878/90 – structured data", ledger))


def run_review(args):
    from .viewer import write_review_site
    print("review site:", write_review_site(args.units, args.run, args.out, args.copy_images))


def run_export_specs(args):
    args.to.mkdir(parents=True, exist_ok=True)
    for unit_dir in selected_units(args.units, None):
        if (unit_dir / "spec.yaml").exists():
            target = args.to / f"{load_unit(unit_dir)['id']}.yaml"
            target.write_text((unit_dir / "spec.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            print("spec:", target)


def add_model_options(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default="medium", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-resolution", default="ultra_high", choices=["low", "medium", "high", "ultra_high"])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--force", action="store_true", help="ignore cached results")


def add_unit_options(parser, with_sections=True):
    parser.add_argument("--units", type=Path, default=Path("work/units"))
    parser.add_argument("--run", type=Path, default=Path("work/runs/main"))
    if with_sections:
        parser.add_argument("--section", nargs="*", help="unit ids, e.g. I-11 II-06")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ecological_ie",
                                     description="Structured extraction from the Forsteinrichtungsoperate.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="collect the pages annotated in the TOC UI into document units")
    prepare.add_argument("--dump", type=Path, required=True, help="Forsteinrichtungsoperate_combined folder or zip")
    prepare.add_argument("--toc-ui", type=Path, required=True, help="Forsteinrichtung_TOC_UI.html")
    prepare.add_argument("--annotations", type=Path, required=True, help="JSON export of the TOC UI")
    prepare.add_argument("--out", type=Path, default=Path("work/units"))
    prepare.add_argument("--scans", type=Path, help="folder with full-resolution scans named like the page ids")
    prepare.add_argument("--order", choices=["volume", "annotated"], default="volume")
    prepare.add_argument("--max-side", type=int, default=3072, help="longest image side when re-encoding scans")
    prepare.add_argument("--specs", type=Path, help="folder with <unit-id>.yaml specs to copy into the units")
    prepare.set_defaults(func=run_prepare)

    draft = commands.add_parser("draft-specs", help="write spec.draft.yaml from the transcripts' table headers")
    add_unit_options(draft)
    draft.set_defaults(func=run_draft_specs)

    check = commands.add_parser("spec-check", help="validate the units' spec.yaml files")
    add_unit_options(check)
    check.set_defaults(func=run_spec_check)

    run = commands.add_parser("run", help="run extraction stages: " + ", ".join(STAGES) + " or all")
    run.add_argument("stage", nargs="+", choices=[*STAGES, "all"])
    add_unit_options(run)
    run.add_argument("--positions", nargs="*", type=int, help="only these page positions (tables)")
    run.add_argument("--recheck-model", default=DEFAULT_MODEL, help="model for re-checks and map overviews")
    run.add_argument("--rounds", type=int, default=2, help="re-check rounds")
    add_model_options(run)
    run.set_defaults(func=run_stages)

    derive = commands.add_parser("derive", help="rebuild all derived tables from cached model outputs")
    add_unit_options(derive, with_sections=False)
    derive.set_defaults(func=run_derive)

    publish = commands.add_parser("publish", help="write the Frictionless data package")
    add_unit_options(publish, with_sections=False)
    publish.add_argument("--out", type=Path)
    publish.set_defaults(func=run_publish)

    review = commands.add_parser("review", help="write the static review site (scan + extraction per page)")
    add_unit_options(review, with_sections=False)
    review.add_argument("--out", type=Path, help="site folder (default: <run>/review)")
    review.add_argument("--copy-images", action="store_true", help="copy the page images into the site folder")
    review.set_defaults(func=run_review)

    export = commands.add_parser("export-specs", help="copy the units' spec.yaml files into a folder (e.g. the repo)")
    export.add_argument("--units", type=Path, default=Path("work/units"))
    export.add_argument("--to", type=Path, required=True)
    export.set_defaults(func=run_export_specs)

    testset = commands.add_parser("testset", help="proof of concept: 40-page test set, transcripts only")
    testset.add_argument("--data", type=Path, default=DEFAULT_TESTSET)
    testset.add_argument("--out", type=Path, default=Path("ie_work/output"))
    testset.add_argument("--subset", nargs="*", help="seq ids, e.g. text_01 table_03")
    add_model_options(testset)
    testset.set_defaults(func=run_testset)

    extract = commands.add_parser("extract", help="baseline: generic page-level extraction of prepared units")
    extract.add_argument("--units", type=Path, default=Path("work/units"))
    extract.add_argument("--section", nargs="*")
    extract.add_argument("--out", type=Path, default=Path("work/extraction"))
    extract.add_argument("--limit", type=int, help="only the first N pages (smoke test)")
    extract.add_argument("--no-image", action="store_true", help="send the transcript only")
    add_model_options(extract)
    extract.set_defaults(func=run_extract)

    corpus = commands.add_parser("corpus", help="enumerate the full corpus and estimate the token volume")
    corpus.add_argument("--dump", type=Path, required=True)
    corpus.add_argument("--csv", type=Path)
    corpus.set_defaults(func=run_corpus)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
