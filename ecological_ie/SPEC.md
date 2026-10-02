# Unit specs (`spec.yaml`)

One `spec.yaml` per unit folder in `work/units/<id>_<title>/` tells the extraction what the pages are and what the
columns of each table form mean. Gemini gets the summary, the form description and the canonical columns as context;
the code uses types, units, pairs, keys, row sums and variables to normalise, check and publish the data.
Validate with `python -m ecological_ie spec-check --section <id>`. Worked examples: `I-16` (one simple table) and
`I-20` (two forms, three Revier blocks, currency switch, page without a year column).

## Top level

| field | content |
|---|---|
| `unit`, `title` | as in `unit.json` |
| `summary` | 2–5 sentences in English: what the unit is (Beilage no., form, period, Reviere/Forstamt), what one row is, units used, anything unusual (unit or currency changes, double rows, red ink, continuation pages) |
| `default_profile` | `table`, `text`, `map` or `skip` |
| `default_form` | form id used by table pages without override |
| `reading_order` | optional: list of all page positions in reading order, only if it differs from the position order |
| `pages` | optional overrides by position: `role` (title, table, continuation, text, signature, map, empty, other), `profile`, `form`, `note` |
| `forms` | table forms (below) |
| `segments` | text units only, optional: `[[first, last], …]` page positions forming one section (≈ 3–6 pages, break at headings or §§) |
| `map` | map units only: free key/value hints, e.g. `expected: "Triftkarte with planned (red) and existing (blue) structures"` |
| `notes` | anything a reviewer should know (doubtful headers, missing pages, duplicates) |

Pages with a different structure get their own form; a title page with only text gets `profile: text, role: title`;
blank or irrelevant pages get `profile: skip`.

## Forms

```yaml
forms:
  wp7:
    name: "Form No. VII – Spezieller Wirthschaftsplan"
    description: |
      What one row is, how keys appear (Distrikt given once for several rows, group headings "Revier Klingenbrunn"),
      which rows are sums/carry-overs ("Seite 2", "Übertrag"), unit headings, two-line rows, red ink, anything that
      helps reading the page correctly.
    columns:
      - {id: district_no, label: "Distrikt No.", type: key, key: district_no}
      - {id: area, label: "Fläche › Tagw.", type: pair_major, unit: Tagwerk, variable: area}
      - {id: area_dec, label: "Fläche › Dez.", type: pair_minor, of: area, mode: digits}
      - {id: species, label: "Vorherrschende Holzart", type: text}
      - {id: age, label: "Alter", type: number, unit: Jahre, variable: age}
      - {id: remarks, label: "Bemerkungen", type: text, describe: true}
    row_sums:
      - {target: total, of: [part_a, part_b]}
    row_periods: ["1860/61", "1861/62"]      # only for pages without a period column
    observations: {scope_keys: [revier], scope: "free text if the whole table has one scope"}
```

**Columns describe the form as it is printed on the scan**, left to right, down to the lowest header level. The HTR
headers in the transcripts are often garbled ("Jaucher" for Tagwerk); read the scan. Rules:

* `id`: short English snake_case, unique in the form. Repeated blocks get prefixes: `schoenau_area`,
  `p1_cut_area` (period I), `p2_cut_area` …
* `label`: German header path as printed, levels joined with ` › `.
* `meaning`: optional English explanation when the header is not self-explanatory.
* `type`:
  * `key` + `key:` one of `revier`, `district_no`, `district_name`, `compartment_no`, `subcompartment`, `stand_name`,
    `betriebsklasse`, `altersklasse`, `sortiment`, `other`
  * `label` – row label that is not a key (e.g. "Seite 2", names of sums)
  * `text` – free text; add `describe: true` for ecological description cells (Beschreibung, Lage/Boden/Holzbestand,
    Bemerkungen with stand information) so they get the ecological decomposition
  * `number` – one number per cell (decimal comma allowed)
  * `pair_major` / `pair_minor` – a value split over two columns: `Tagw | Dez`, `Hekt | Ar`, `fl | kr`, `M | Pf`.
    The minor column has `of: <major id>` and `mode`: `digits` (decimal digits as written: 1 | 056 → 1.056; Dezimal,
    Ar), `base100` (Pf), `base60` (kr), `currency` (kr/Pf decided by the unit heading: Gulden → 60, Mark → 100)
  * `year`, `period` – the row's year or period (1860/61)
* `unit`: unit of the values (Tagwerk, ha, Klafter, Ster, Fuder, fl, M, Stück, Jahre, Kubikmeter …); if it changes
  in the table, give the first one and explain the change in the description (unit headings set it per row).
* `variable`: English snake_case name for numbers that become data observations (`harvest_main_use`, `area`,
  `cut_area`, `stem_count`, `volume`, `increment`, `grazing_oxen`, `cost`). Leave it out for pure bookkeeping numbers
  (row numbers, page references).
* `period`: for columns that belong to one period or year ("I. Periode 1855/64").
* `scope`: for column blocks belonging to one Revier or class ("Schönau").

`row_sums`: arithmetic the form promises inside a row (`Gesamt = Haupt + Zwischen`). Column sums (Summa, Übertrag) are
checked automatically.

## Review checklist

1. Every page position has the right profile and form; reading order checked against the text flow / page numbers.
2. Columns match the scan of at least two pages of each form, including the minor columns of pairs.
3. Types, units and pair modes are right (check one sum on the scan: 1 | 056 + 2 | 157 = 3 | 213 → digits).
4. Every ecologically relevant number has a `variable`, every description column `describe: true`.
5. `spec-check` reports no problems.
