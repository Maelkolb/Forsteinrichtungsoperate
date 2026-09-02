# Ecological information extraction (IE) from the transcriptions

`Forsteinrichtung_Ecological_IE_Gemini.ipynb` is a Colab proof of concept that extracts the most
important **ecological information** from the Markdown transcriptions produced by the pipeline in
this repository (`run.py --doc-type text|table`). It works on the `md/*.md` files only, never on the scans.

Model: **Gemini 3.5 Flash** (`gemini-3.5-flash`, structured JSON output, `thinking_level`).

## Test set

`testset/` holds 20 text pages and 20 table pages (plus `manifest.csv`) selected from the
`Forsteinrichtungsoperate_combined` output for ecological richness and format diversity:

* text: Erörternde Darstellung / Waldstandsrevision (10), Protokolle (4), Schreiben (3), gedruckter/typed text (3)
* tables: Periodentabelle (5), Bestandsauszählungen (3), Altersklassen (2), Kulturplan (2), Wirtschaftsplan (2),
  Probeflächenaufnahmen, Streunutzungsplan, Forstnebennutzungen, Wirtschaftsbuch, Betriebsplan, Übersicht der Forstverbesserungen

Zip the folder (`zip -r Forsteinrichtung_IE_testset.zip testset`) and upload it in the notebook, or point the
notebook at a Drive / local copy.

## What the workflow does

1. **Setup** – installs `google-genai`, reads the API key from the Colab secret `LST_Gemini` (fallback:
   `GEMINI_API_KEY`), creates the client, checks that `gemini-3.5-flash` is visible.
2. **Data + pre-processing** – loads `manifest.csv`, parses the YAML front matter of every md file, cuts
   marginalia blockquotes that degenerated into repetition loops, joins line-end hyphenation (`-`/`=`) in text
   pages so that quotes are contiguous, and collapses the indentation of the HTML tables (30–40 % fewer tokens).
   The cleaned text is what the model sees and what quotes are verified against.
3. **Schemas and prompts** – two JSON schemas and two prompts (see below), both with a glossary of local
   terms and abbreviations (Fi/Ta/Bu, Auen, Filze, Hochwald, Duftbruch, WZ, Tagwerk, Klafter, …).
4. **Extraction** – one Gemini call per page with `response_json_schema`; retries with back-off; automatic
   fallback to plain JSON mode with the schema embedded in the prompt if the API rejects the schema;
   results cached as one JSON per page in `output/raw/` (re-runs skip finished pages); 4 parallel workers;
   token usage and cost estimate.
5. **Results** – flat CSVs (`text_findings`, `text_species`, `text_locations`, `table_records`,
   `table_species`, `table_damage`, `table_culture`, `table_non_timber_use`, `table_history`),
   verification that every quote / description appears verbatim in the input, a standalone
   `review_viewer.html` (source text with highlighted quotes next to the extracted items) and a
   `scoring_sheet.csv` for manual evaluation; everything zipped for download.
6. **Scaling notes** – priority and skip lists for the table categories of the full corpus and a cell that
   enumerates the corpus and estimates the token volume.

## How Gemini is used for the IE

* **Text pages** → prompt asks for a flat list of *findings*, each with a `category` from a fixed enum
  (`tree_species`, `stand_structure`, `site_conditions`, `ground_vegetation`, `climate_weather`,
  `damage_event`, `regeneration`, `silvicultural_measure`, `non_timber_use`, `wildlife`,
  `land_use_hydrology`, `quantity`, `other_ecological`), a normalised German `entity`, `value`/`unit`,
  `date`, `location`, `status` (planned / executed / prohibited / observed / ended), `confidence` and a
  **verbatim quote** (≤ 40 words). Page level: document type, summary, forest offices, dates, a list of
  locations (Distrikt / Abteilung / Unterabteilung / toponym with area) and a list of tree species with
  role and share as a 0–1 fraction (8/10 → 0.8, WZ 0,7 → 0.7).
* **Tables** → prompt asks the model to (a) identify the form and list the flattened column headers,
  (b) resolve rowspan / ditto / group-header context, (c) merge values split over unit columns
  (Hektar | Ar → 18.60), (d) copy the free-text description cell verbatim and decompose it into
  `site` (Lage, Boden), `stand` (species with shares, age, stocking, structure, health, regeneration,
  volume, increment, stem counts), `damage`, `management` (planned cut, period, transition code, harvest
  history), `culture` (planting/sowing/drainage with species and quantities) and `non_timber_use`
  (Streu, Weide, Torf, Harz), (e) mark carry-over/summary rows, red-ink corrections and uncertain rows.
* The schemas are passed as `response_json_schema`, so the output is guaranteed-valid JSON that flattens
  directly into tables; the verbatim quotes make every extracted item auditable against the transcription
  and, via the pipeline's region JSON, against the scan.
