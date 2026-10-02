# Workflow: structured extraction of the Ilzertrift-Komplex Waldstandsrevision (1878/90)

Scope: the 24 TOC items of `[002] StALa, KdForsten A 383 I` that are linked to pages in the TOC UI (333 pages:
88 text, 241 table, 4 maps; volumes A 383 I/II, A 384 I/II, A 385). Goal: a faithful, verifiable and publishable data
set of everything the Operat says about the forest of the Reviere Schönau, St. Oswald and Klingenbrunn.

## Principles

1. **Two layers.** A faithful layer reproduces the source (every table cell as written, corrected text, map labels
   with position). An interpreted layer derives ecological data (stands, species, site, events, series). Each value of
   the second layer points to cells or quotes of the first, and each of those to a page and an image region.
2. **Models read, code calculates.** Gemini reads (scan + transcript) and interprets free text. Unit conversion, ditto
   marks, keys, sums and joins are done in code, with the rules written down in the unit specs.
3. **The source's own redundancy is the test.** Tables carry sums, carry-overs and totals; text repeats numbers from the
   tables. Mismatches trigger a targeted re-read of the image region. A mismatch that survives the re-read is recorded
   as a discrepancy *of the source*, never "fixed".
4. **The scan decides, the transcript is the draft.** Every page is read with its image. Readings that differ from the
   transcript are kept as corrections (transcript vs scan), so the HTR layer stays auditable.
5. **Claude Code orchestrates.** It writes and reviews the unit specs, runs the stages, reads validation reports,
   decides re-checks and writes the synthesis. Specs, prompts, model ids and runs are versioned so the data set can be
   regenerated.

## Models

| task | model | why |
|---|---|---|
| table grids, proofreading, map tiles | `gemini-3.8-flash` (stable; $0.75 / $3.75 per M tokens until 2026-12-31) | newest Flash, cheaper than 3.5 Flash ($1.50 / $9.00) |
| re-reads of failed checks, map overview | `gemini-3.1-pro-preview` ($2 / $12) or 3.8 Flash with high thinking | chosen by a pilot comparison on the hardest pages |
| specs, QA, integration, synthesis | Claude Code (subagents per unit group) | reads transcripts and scans, writes YAML specs, reviews issues |

Images go in per part at `ultra_high` (2240 tokens). Because the token budget per image is fixed, dense tables and maps
are additionally sent as **zoomed crops** (bands or tiles, each with its own budget). That raises the effective resolution
more than a larger scan would.

## Stage 0 · Units (done)

`prepare` collects the annotated pages into `work/units/<id>_<title>/` (transcript, image, `unit.json`).

**Images.** The zip holds 1200 px JPEGs. The scans at work are 200 dpi exports of at most 1652 × 2338 px; the local copies in
`Kategorien/` (A 379–A 382) show that format. So the work scans are only 1.2–1.4× larger. They should replace the
zip images when available (`prepare --scans`), and they are needed for the 12 pages without any image. An A/B
pilot on same-type pages from A 379–A 382, where both versions exist locally, measures the difference. For the maps
neither version is enough to read toponyms reliably; archival masters (≥ 400 dpi) should be requested for the four
annotated maps.

## Stage 1 · Unit specs (Claude Code)

`draft-spec` writes `spec.yaml` per unit from the transcripts: page roles, and for table pages the header structure of
each form (colspan/rowspan of `<thead>` resolved into column paths, pages with the same header grouped). Claude Code
subagents then complete each spec from transcripts and scans:

* reading order, page roles (title, table, continuation, text, signature, skip), profile per page
* per table form: column ids, meaning, type (`key`, `text`, `int`, `decimal`, `pair_major`/`pair_minor` with
  `digits`/`base` rule, `year`, `period`), unit, `describe: true` for free-text description cells, row sums
  (`target = a + b`), the series variables (scope column, period column, variable name and unit per column)
* for text units: segments (sections, §§) and themes
* for maps: expected map type and legend

`spec-check` validates the specs against a JSON Schema. Specs are reviewed by you before the full run.

## Stage 2 · Extraction (Gemini)

**Tables: faithful grid.** One call per page with the form's column model, transcript HTML and the scan (plus two
overlapping half-page crops for wide forms). The JSON schema is generated from the spec, so every row returns exactly the
form's columns:
`row_type` (data / group header / sum / carry-over / heading / note), `box_2d` of the row, one string per column
as read, red-ink and uncertain cells, and corrections against the transcript. Layout deviations from the form are
reported, not forced.

**Tables: normalisation (code).** Ditto marks, number formats (`27,87`, `½`, `—`), pairs (`Tagw | Dez`,
`Hekt | Ar`, `M | Pf`, `fl | kr` with base 60), key propagation (Distrikt and Abteilung given once for several rows),
units. The output is `cells` (long format, one row per cell) and `rows`.

**Tables: description cells.** `describe` columns (Beschreibung, Bemerkungen, Boden/Lage/Holzbestand) are sent in batches
per unit to a text-only pass. It decomposes them into site, species shares, age, stocking, structure, damage,
regeneration, management and use, each with a verbatim evidence span.

**Text: proofread, assemble, extract.**
1. *Proofreading:* page image + transcript → list of corrections (exact transcript substring → scan reading); code
   applies them and keeps both versions.
2. *Assembly:* the unit becomes one document in reading order, with page anchors `⟦p017⟧` and hyphenation across page
   breaks resolved.
3. *Extraction:* one call per segment (about 3–5 pages). The whole unit document comes first (implicitly cached
   prefix), then the instruction for the segment, so cross-page context is never lost. The output is atomic
   statements: category, subject, attribute, value + unit, time (as written + EDTF), place (as written + keys
   Revier / Distrikt / Abteilung / toponym), status (observed, historical, planned, prescribed, prohibited), verbatim
   quote + page anchor, confidence. Events (storm, beetle, snow, fire) are a separate list.

**Maps: multi-scale reading.**
1. *Overview* (whole sheet): title, type, date, author, scale statement and bar, orientation, legend items (symbol or
   colour → meaning), boxes of map body, legend, cartouche.
2. *Tiles* (overlapping grid over the map body, each tile at `ultra_high`): every label with `box_2d`, transcription,
   class (settlement, water, mountain, forest district, Distrikt/Abteilung number, road, Trift structure, marginal note)
   and confidence; marked symbols (Klausen, Triftbauten red = planned / blue = existing).
3. *Merge* (code): tile boxes → sheet coordinates, duplicates in overlaps merged.
4. *Links:* labels matched to the stand register (Distrikt names and numbers) and to toponyms in text and tables.
5. *Georeferencing:* the test map (Wegbaukarte 1890) carries sheet numbers along its border (54–61, XXXIV …,
   "N. 59 O."), which looks like the grid of the Bavarian cadastral sheets (Positionsblätter, Soldner system). If
   confirmed, the detected grid lines plus the sheet numbers give ground control points for the whole sheet without
   gazetteer matching. Otherwise settlements, lakes and peaks serve as control points (Wikidata/GeoNames coordinates,
   checked in QGIS).
6. *Colour classes* (next step, code): stand colours clustered and matched to the legend (e.g. "In die I. Periode
   eingereihte Bestände", "Nachhiebsflächen") give polygons and area shares per class; after georeferencing they can
   be compared with the stand register.

## Stage 3 · Validation (code)

* tables: sum rows against the data rows above them (and sums of sums), declared row sums, carry-overs across pages,
  minor units in range, years monotonic, keys well-formed
* text: every quote found in the corrected text; corrections applied uniquely
* across units: the same stand in I-02, I-11, II-01, II-06, I-13, I-14 (area, name); yearly harvest in I-16 vs II-10
  vs I-15; II-04 vs II-05; numbers quoted in I-00/I-21 vs the tables
* output: `issues.csv` with location, rule, expected/found values

## Stage 4 · Re-check (Gemini, targeted)

Pages with structural problems (rows with a wrong cell count, Kreuzer/Pfennig ≥ 60/100, many unreadable numbers or
failed sums) first get a second independent reading with high thinking and a note on what went wrong; the reading that
passes more of the page's own checks is kept (`regrid`). Then, for each failed arithmetic check, a generous crop over
the rows involved plus a zoom on the column, the current readings and the expected value go to the model, which re-reads
each cell; rows are found by their label, not by box. If the re-read resolves the mismatch, the cells are updated
(status `ok_after_recheck`, overrides with before/after). If the re-read confirms the reading, the status is
`reading_confirmed`: probably an error or rounding in the source, to be confirmed by a person. If cells changed but the
check still fails, `mismatch_after_recheck`. At most two rounds; the rest goes into the manual review list.

## Stage 5 · Integration and publication

Data package (Frictionless `datapackage.json` with field schemas, units and foreign keys; CSV + GeoJSON; codebook
generated from the specs):

| resource | content |
|---|---|
| `units`, `pages` | TOC items, pages, image references |
| `table_rows`, `table_cells` | faithful layer: every cell raw + normalised, row box, flags, corrections |
| `observations` | tidy series: scope (Revier, Komplex, Betriebsklasse, Altersklasse), period, variable, value, unit |
| `stands`, `stand_attributes` | stand register with normalised key and attributes per source |
| `statements`, `events` | text layer: atomic statements and events with quotes |
| `map_labels`, `maps` | labels with sheet coordinates (later georeferenced), legend, map metadata |
| `corrections`, `issues` | transcript vs scan readings; validation results and source discrepancies |
| `runs` | model ids, prompt versions, spec hashes, dates, token costs |

Each row of the interpreted layer has `unit_id`, `page_id`, row/cell or quote reference and the image region.
Claude Code writes `office_picture.md`, a synthesis by Revier and theme with citations. A static review site shows each
page's scan with row boxes and labels next to the extracted data and the open issues.

## Pilot results (2026-10-02)

| test | result |
|---|---|
| grid reading, I-16 + I-20 (3 dense pages) | checks passed: 3.8 Flash low 32/75, medium 38/75, 3.1 Pro 39/75; cost per page $0.03 / $0.08 / $0.33 → **3.8 Flash, medium** |
| re-check, same pages | I-16: 24/26 → 26/26 after one round (both models misread the same sum cell; the re-read on a zoomed crop fixed it). I-20: 14 → 25 of 41; 3.8 Flash (high) better than Pro. Rows are found by label and current reading; Gemini's row boxes drift by up to two rows on long tables, so they are only approximate provenance |
| check logic | sections must be tracked per column (a sum may cover only some columns) and a sum with no data rows above it is an opening balance (conversion rows "= Ster", "= M. Pf.") |
| resolution A/B, 3 Wirtschaftsplan pages A 379 IV | the zip images of that volume are already 1600 px (scan 1652 px): no measurable difference. Two readings agree on 58–76 % of cells, so the arithmetic checks and re-reads matter more than image size |
| map stage, Wegbaukarte 1890 (A 381 I p. 32, 1652 px) | title, date, 8 legend classes, 112 labels (Distrikt names, villages, sheet numbers) with well-placed boxes for $0.08; small compartment labels need finer tiles (default now about 450 px) |

## Cost (at gemini-3.8-flash prices)

About $15–20 for the table grids, $2–4 for proofreading and text extraction, $2–4 for maps, $3–6 for re-checks and
$2–3 for description cells: roughly $25–35 for the full run, plus about $3–5 for the pilots. The Batch API would halve
this when time does not matter.

## Open points

* Work scans (200 dpi) for all units, and the 12 pages without image: A 384 I 379–382; A 384 II 112, 114, 116, 117, 133, 197–199.
* Higher-resolution masters of the four maps.
* Nine TOC items without links (I-03, I-06, I-10, I-WHB, II-02, II-03, II-07–09).
* Licence and publication venue for the data package (e.g. Zenodo).
