GLOSSARY = '''
GLOSSARY (local usage in the Bavarian Forest Forsteinrichtungsoperate, 1830s-1920s):
- Distrikt (Roman numeral + name, e.g. "VIII Sagwasserhang"), Abteilung (number), Unterabteilung (lit. a, b, c) = stand address.
- Holzarten & abbreviations: Fi/F = Fichte (Picea abies), Ta/T = Tanne (Abies alba), Bu/Bü/B/Rothbuche = Buche (Fagus sylvatica),
  Ki/Föhre/Kiefer (Pinus sylvestris), Lä/Lär/Lerche = Lärche (Larix decidua), Ah = Ahorn (Acer pseudoplatanus), Esch = Esche,
  Bi/Be = Birke, Er = Erle, Ul/Elbe = Ulme, Ei = Eiche, Zirbe (Pinus cembra), Spirke/Latsche (Pinus mugo), Vogelbeere, Aspe,
  Douglasie, Strobe (Pinus strobus), Sitkafichte, Höhenkiefer.
- Mixture notation: "8/10 Fi u. Ta, 2/10 Bu" = fractions of area; "50 % der Masse"; "0,7 Fi 0,2 Ta 0,1 Bu" = shares (Bestockung);
  "WZ 0,7 Fi 0,3 BuTa" = Wirtschaftsziel (target mixture). Bestockung(sgrad)/Schluß 0,4-1,0 = stocking density.
- Auen = waterlogged spruce flats on plateaus (NOT riparian); Filze = raised bogs; Hochwald = the high-elevation spruce zone
  (a local Betriebsklasse above ~1150 m); Urwald = primeval/unmanaged old forest; Plenter/Plänterwald = selection forest.
- Hiebsarten: Kahlschlag/Kahlhieb, Plenterhieb, Femelschlag, Saumschlag, Saumfemelschlag, Dunkelschlag, Sammelschlag, Angriffshieb,
  Nachhieb, Räumung(shieb), Durchforstung, Läuterung, Vorbereitungshieb. Übergangsbestimmungen codes: N = Nachhieb, P = Pflanzung, D = Durchforstung.
- Schäden: Windwurf/Windbruch/Windriß (storm), Borkenkäfer/Käferfraß (bark beetle), Schneebruch/Schneedruck (snow), Duftbruch (rime/ice),
  Wildverbiß (game browsing), Schälschaden (bark stripping), Rotfäule (root rot), Frost/Spätfrost, Verunkrautung (weed competition).
- Nebennutzungen: Weide/Hut/Tagweide (grazing), Streunutzung/Laub-, Nadel-, Moosstreu (litter raking), Harz (resin), Torf (peat),
  Ameiseneier, Glashütten/Glasfabrik (wood for glassworks), Kohlen (charcoal), Mast, Lohe/Borke.
- Units: ha + Ar (Hektar|Ar columns -> 18.60 ha), Tagwerk (Tgw) + Dezimale (1 Tgw = 0.3407 ha), Fmtr/fm (Festmeter = m³),
  Ster, Klafter (Klftr, Str), Kubikfuß, Ruthen, Pfd (pound), Metzen, fl./kr. (Gulden/Kreuzer), M/Pf (Mark/Pfennig).
- HTR markers: [?] uncertain reading, [illegible], [...] gap, "..." lost line start; <span class="red"> = later red-ink correction.
'''

TEXT_PROMPT = '''You are an expert in historical forestry (Forsteinrichtung) and ecological history, reading Gemini HTR transcriptions
of 19th/early-20th-century German (Kurrent) forest-management documents from the Bavarian Forest (Forstamt St. Oswald, Klingenbrunn,
Spiegelau, Riedlhütte, Schönau/Waldhäuser).

TASK: Extract ALL ecologically relevant information from the page below into the JSON structure requested.
Focus on: tree species and mixture shares, stand structure and age, site conditions (elevation, exposure, slope, soil, bedrock,
moisture, Auen/Filze), ground vegetation, climate/weather, damage events (wind, snow/ice, insects, game, rot, fire) with dates,
regeneration (natural/sowing/planting) and its success, silvicultural measures, non-timber uses (grazing, litter, resin, peat, glassworks),
wildlife, hydrology/land-use changes, and quantities with units (area, volume, growth). Include statements about the past, the present
state and plans/prohibitions, and mark them with "status".

RULES:
1. One finding per distinct statement. Do not merge different stands, dates or species into one finding.
2. "quote" must be copied VERBATIM from the page text (same spelling, including HTR errors), at most ~40 words. Never paraphrase inside quote.
3. Normalise "entity" and species names to modern German headwords (Fichte, Tanne, Buche …), keep "as_written"/"value" as in the text.
4. Convert shares to fractions in share_value (8/10 -> 0.8, 50 % -> 0.5, WZ 0,7 -> 0.7). Do not invent numbers.
5. Administrative content (salaries, accounting, personnel, road costs, signatures) is NOT ecological: skip it, unless it carries
   ecological facts (e.g. wood delivered to a glassworks, grazing rights of villages).
6. Text in blockquotes ("> *[Marginalie]*") are marginal notes: extract from them too, but add "marginalia" in "note".
7. HTML tables inside the text hold stand lists: extract each stand into "locations" (with area) and, if they contain ecological
   attributes (age, species, treatment), also into "findings".
8. The page usually begins and ends mid-sentence. Extract what is on the page; if the subject is unclear because it was on the previous
   page, still extract and set confidence "low" with a note.
9. If the page contains no ecological information, return an empty "findings" list and describe the page in "summary_en".

''' + GLOSSARY + '''
PAGE ID: {page_id}
SOURCE SUBTYPE: {subtype}
{context}
=== PAGE TEXT (Markdown) ===
{text}
=== END PAGE TEXT ===
{image_instructions}'''

TABLE_PROMPT = '''You are an expert in historical forestry (Forsteinrichtung) and ecological history, reading Gemini HTR transcriptions
of 19th/early-20th-century German forest-management TABLES from the Bavarian Forest (Forstamt St. Oswald, Klingenbrunn, Spiegelau,
Riedlhütte). The table is given as an HTML <table> (with colspan/rowspan, <span class="red"> for red-ink corrections) plus header text
lines above and marginalia below.

TASK: Convert the table into a list of RECORDS – one per stand row (Distrikt / Abteilung / Unterabteilung), sample plot, year row,
or Forstamt total – and extract the ecological content of each record into the JSON structure requested.

RULES:
1. Identify the form first (table_type) from the headers, and list the flattened column headers in "column_headers" so the mapping
   is auditable. Note continuation pages, side-by-side duplicated column blocks (two stands per page) and header/body column drift
   in "parsing_notes".
2. Resolve rowspan/ditto context: a Distrikt or Abteilung given once (rowspan, first row, group header row like "III. Rachelhang")
   applies to the following rows until a new one appears. Ditto marks (", „, ”, ,,, dto, detto, desgleichen) repeat the cell above.
   Cross-references such as "wie bei IV. 8. b" go into "cross_references" – do not resolve them.
3. Numbers split across unit columns (Hektar | Ar; Tagw | Dez; M | Pf) are ONE value: 18 | 60 -> 18.60. Numbers split inside a cell
   ("186 27", "2 03") likewise. Use the decimal point in JSON. Keep the original in the *_raw / raw fields. Do not invent numbers.
4. Free-text description cells ("Beschreibung a) der Lage b) des Bodens c) des Holzbestandes d) der Bewirthschaftung e) Kulturen",
   "Bemerkungen", "Boden / Lage / Holzbestand") are the richest source: copy them verbatim into description_raw and decompose them into
   site (a, b), stand (c), management (d), culture (e), damage and non_timber_use. If a description continues in the next row of the
   same stand, attach it to that stand.
5. Species: normalise names (Fi -> Fichte, Bu/Bü -> Buche, Ta -> Tanne, Lerche -> Lärche …), keep as_written, shares as fractions
   (0,7 -> 0.7). "0,4 bis 0,7 d. Schst" = stocking density 0.4-0.7, not a species share.
6. Rows that are sums, carry-overs (Übertrag, Transport, Summa, Seite 17), or totals per Forstamt get record_kind summary_row /
   carry_over / forest_office_total and only the numeric values that are present.
7. Red-ink content (<span class="red">) is a later correction: put it into red_annotations of the row, and use it for the fields only
   if the black value is missing.
8. Set "uncertain": true for rows with [?], [illegible], or where the alignment of cells to rows is ambiguous.
9. Purely financial columns (Kostenanschlag, Geld) may be captured in culture.cost_* but need no further attention.

''' + GLOSSARY + '''
PAGE ID: {page_id}
TABLE CATEGORY (from folder name): {subtype}
{context}
=== PAGE (Markdown with HTML table) ===
{text}
=== END PAGE ===
{image_instructions}'''

IMAGE_INSTRUCTIONS = '''
SCAN: the scan of this page is attached. The page text above is an automatic HTR transcription and can contain misreadings
(digits, names, species abbreviations, Distrikt/Abteilung keys, shifted or merged columns, missed rows).
- Take the content from the transcription, but check it against the scan, above all numbers, keys and column alignment.
- Where the scan clearly shows something else, use the scan reading in the extracted fields and add an entry to
  "image_corrections" (where, transcript_reads, image_reads, confidence).
- Content that is visible in the scan but missing in the transcription may be extracted; add it to "image_corrections" with
  transcript_reads "" and give it confidence "medium" or "low".
- "quote" and "description_raw" stay VERBATIM copies of the transcription text, so that they can be verified automatically.
'''

PROMPTS = {"text": TEXT_PROMPT, "table": TABLE_PROMPT}


def context_block(unit_title: str = "", position: str = "", brief: str = "", previous_tail: str = "") -> str:
    lines = []
    if unit_title:
        lines.append(f"DOCUMENT UNIT: {unit_title}" + (f" (page {position})" if position else ""))
    if brief:
        lines.append("UNIT BRIEF:\n" + brief.strip())
    if previous_tail:
        lines.append("END OF THE PREVIOUS PAGE (context only, do not extract from it):\n" + previous_tail.strip())
    return ("\n" + "\n\n".join(lines) + "\n") if lines else ""


def build_prompt(page: dict, with_image: bool = False) -> str:
    template = PROMPTS[page["source_type"]]
    context = context_block(page.get("unit_title", ""), page.get("position", ""),
                            page.get("brief", ""), page.get("previous_tail", ""))
    return template.format(page_id=page["page_id"], subtype=page["subtype"], text=page["text"],
                           context=context, image_instructions=IMAGE_INSTRUCTIONS if with_image else "")
