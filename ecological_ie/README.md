# Structured extraction from the Forsteinrichtungsoperate

`ecological_ie` turns the Gemini transcriptions and page scans of the Forsteinrichtungsoperate into a verifiable,
publishable data set. The current target is the 24 TOC items of the Waldstandsrevision 1878/90 of the
Ilzertrift-Komplex that are linked to pages in the TOC UI. The workflow and its reasoning are in [`PLAN.md`](PLAN.md);
the format of the unit specs is in [`SPEC.md`](SPEC.md).

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="..."      # or GOOGLE_API_KEY, a .env file (--env-file), or the Colab secret LST_Gemini
```

## Workflow

```bash
# 0 collect the annotated pages into units (transcript + image per page)
python -m ecological_ie prepare --dump Forsteinrichtungsoperate_Gemini_combined.zip \
    --toc-ui Forsteinrichtung_TOC_UI.html --annotations Inhaltsverzeichnis-Zuordnung.json \
    --out work/units [--scans FOLDER_WITH_SCANS]

# 1 unit specs: drafts from the transcript headers, then spec.yaml written/reviewed by Claude Code and you
python -m ecological_ie draft-specs
python -m ecological_ie spec-check

# 2–4 extraction, validation and targeted re-reads (results cached per page in work/runs/main)
python -m ecological_ie run tables recheck describe text maps      # or: run all
python -m ecological_ie run tables --section I-11 --positions 3 4  # a single unit / pages

# 5 rebuild derived tables, publish the data package, write the review site and the explorer
python -m ecological_ie derive
python -m ecological_ie publish
python -m ecological_ie review --out work/release/review_site --copy-images
python -m ecological_ie explore --package work/runs/main/package --out work/release/explorer/index.html \
    --image-base ../review_site/images/
```

The explorer (`explorer.py`, `explorer_template.html`) is one HTML file with the data embedded: overview, harvest and
use 1860/61–1877, stands, disturbances, statements, maps and sources. Every value opens a drawer with the reading, its
check status and the page scan. `--fragment` leaves out the html/head/body wrapper for hosts that add their own.

Model options for `run`: `--model` (default `gemini-3.8-flash`), `--thinking`, `--recheck-model`, `--rounds`,
`--workers`, `--image-resolution` (default `ultra_high`), `--force`. Every stage appends model, token counts and cost
to `work/runs/<run>/runs.jsonl`.

| stage | what Gemini gets | what comes back |
|---|---|---|
| `tables` | scan + two zoomed halves + transcript + form description and canonical columns | faithful grid: page columns mapped to canonical ids, rows with type and box, cells as written, red ink, corrections |
| `recheck` | crop of the rows behind each failed sum + column zoom + current readings | re-read cells; stored as overrides with before/after |
| `describe` | batches of free-text description cells | site, stand, damage, management, culture and use attributes with evidence |
| `text` | page scan + transcript (proofreading), then the whole proofread unit as context per segment | corrections; atomic statements and events with verbatim quotes and page anchors |
| `maps` | whole sheet (overview), then overlapping zoomed tiles | title, legend, regions; every label with class and box in sheet coordinates |

Code does the rest: number formats, ditto marks, value pairs (`Tagw | Dez`, `fl | kr`, `M | Pf`), unit and currency
headings, key propagation, column totals, row sums, carry-overs, stand keys, tidy observations.

## Outputs (`work/runs/<run>/`)

* `derived/*.jsonl` – all tables of the faithful and interpreted layer
* `package/` – Frictionless data package (`datapackage.json`, `data/*.csv`, `map_labels.geojson`, README with codebook)
* `review/index.html` – one page per unit: scan with row or label boxes next to the reconstructed grid (failed checks
  red, re-read cells yellow) or the proofread text with its corrections and statements

## Other commands

* `testset` – the earlier proof of concept: 40 transcript pages, generic text/table schemas
* `extract` – the same generic schemas on prepared units, with images (baseline)
* `corpus` – enumerate the whole corpus and estimate its token volume

Tests: `python -m pytest tests`.

## Modules

| module | content |
|---|---|
| `dump.py`, `toc.py`, `images.py`, `units.py` | dump access (folder or zip, NFC paths), TOC UI and annotations, page images, unit preparation |
| `spec.py`, `html_tables.py` | spec schema, validation, page plans, drafts from transcript headers |
| `grid.py`, `normalize.py`, `checks.py`, `recheck.py`, `pipeline.py` | table grids, normalisation, arithmetic checks, re-reads, stage runner |
| `textdoc.py`, `maps.py`, `describe.py`, `stages.py` | text, map and description stages |
| `publish.py`, `viewer.py` | data package, review site |
| `gemini.py`, `config.py` | Gemini client (structured output, image parts, retries, cost ledger), models and prices |
| `pages.py`, `schemas.py`, `prompts.py`, `extract.py`, `results.py`, `review.py`, `testset.py`, `corpus.py` | proof-of-concept path |
