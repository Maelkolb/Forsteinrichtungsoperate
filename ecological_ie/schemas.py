CATEGORIES = [
    "tree_species",
    "stand_structure",
    "site_conditions",
    "ground_vegetation",
    "climate_weather",
    "damage_event",
    "regeneration",
    "silvicultural_measure",
    "non_timber_use",
    "wildlife",
    "land_use_hydrology",
    "quantity",
    "other_ecological",
]

CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}


def string(description=""):
    return {"type": "string", "description": description}


def number(description=""):
    return {"type": "number", "description": description}


def array(item):
    return {"type": "array", "items": item}


IMAGE_CORRECTIONS = array({"type": "object", "properties": {
    "where": string("row / cell / line the correction applies to"),
    "transcript_reads": string("what the transcription says"),
    "image_reads": string("what the scan shows"),
    "confidence": CONFIDENCE},
    "required": ["where", "transcript_reads", "image_reads"]})

TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "page_id": string(),
        "document_type": {"type": "string", "enum": [
            "Eroerternde_Darstellung", "Grundlagenprotokoll", "Wirtschaftsregeln", "Protokoll", "Schreiben",
            "Bericht", "Inhaltsverzeichnis", "Tabelle", "other"]},
        "summary_en": string("1-2 sentences: what the page is about (English)"),
        "forest_offices": array(string("Forstamt / Revier names mentioned, e.g. Klingenbrunn, St. Oswald, Riedlhütte")),
        "dates_mentioned": array(string("ISO-like dates or years/periods as written, e.g. 1870-10-26, 1873/74, 1878/90")),
        "locations": array({"type": "object", "properties": {
            "district_no": string("Distrikt number (Roman) if given"), "district_name": string(),
            "compartment_no": string("Abteilung No"), "subcompartment": string("Unterabteilung lit."),
            "toponym": string("place / stand name as written"),
            "area_value": number(), "area_unit": string("ha | Tagwerk | Dezimale | other"),
            "quote": string("verbatim snippet")}, "required": ["toponym", "quote"]}),
        "tree_species": array({"type": "object", "properties": {
            "name_de": string("normalised German name: Fichte, Tanne, Buche, Kiefer, Lärche, Ahorn, Esche, Birke, Erle, Ulme, Eiche, Zirbe, Douglasie, Strobe, Spirke, Latsche, Vogelbeere, Aspe …"),
            "name_latin": string(), "as_written": string("form in the text, e.g. Fi, Ta, Bu, Rothbuche"),
            "role": {"type": "string", "enum": ["dominant", "admixed", "understorey", "single_trees", "exotic_trial", "absent_or_declining", "unspecified"]},
            "share_value": number("0-1 fraction if a share is given (8/10 -> 0.8; 50% -> 0.5; WZ 0,7 -> 0.7)"),
            "share_basis": string("area | mass | WZ (Wirtschaftsziel) | stem_count | unspecified"),
            "quote": string()}, "required": ["name_de", "role", "quote"]}),
        "findings": array({"type": "object", "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "subtype": string("free short label, e.g. windthrow, bark_beetle, elevation_limit, snow_cover, grazing_right, Femelschlag"),
            "entity": string("normalised German headword, e.g. Windwurf, Borkenkäfer, Fichte, Streunutzung, Gneis"),
            "entity_en": string("English gloss"),
            "species": array(string("normalised species names involved, if any")),
            "value": string("quantity / qualitative value as written, e.g. '1180 m', '8/10', 'Anfang Oktober bis Ende Mai'"),
            "unit": string(), "date": string("date/year/period the statement refers to, if any"),
            "location": string("Distrikt/Abteilung/toponym/Lage the statement refers to, if any"),
            "status": string("planned | executed | prohibited | recommended | observed | historical | ended – if applicable"),
            "quote": string("VERBATIM snippet from the text, max ~40 words, copied exactly"),
            "confidence": CONFIDENCE,
            "note": string("uncertainty, HTR problems, interpretation")},
            "required": ["category", "entity", "quote", "confidence"]}),
        "image_corrections": IMAGE_CORRECTIONS,
        "transcription_quality": {"type": "string", "enum": ["good", "mixed", "poor"]},
        "continues_previous_page": {"type": "boolean"},
        "ends_mid_sentence": {"type": "boolean"},
    },
    "required": ["page_id", "document_type", "summary_en", "findings", "tree_species", "locations", "transcription_quality"],
}

SPECIES_ITEM = {"type": "object", "properties": {
    "name_de": string("normalised: Fichte, Tanne, Buche, …"), "as_written": string(),
    "share_value": number("0-1 fraction if given"), "count": number("stem count if given"),
    "volume_value": number(), "volume_unit": string(),
    "role": string("dominant | admixed | understorey | single_trees | unspecified"),
    "note": string()}, "required": ["name_de"]}

RECORD_SCHEMA = {"type": "object", "properties": {
    "record_kind": {"type": "string", "enum": ["stand", "sample_plot", "summary_row", "carry_over", "forest_office_total", "year_row", "other"]},
    "district_no": string("Distrikt No, Roman numeral as written"), "district_name": string(),
    "compartment_no": string("Abteilung No"), "subcompartment": string("Unterabteilung lit."),
    "stand_name": string("local name if given"), "forest_office": string("Forstamt / Revier if given for this row"),
    "year_or_period": string("year, period label (I. Periode 18--) or year range this row refers to"),
    "area_value": number("area as one decimal number (Hektar+Ar -> 18.60; Tagw+Dez -> 12.35)"),
    "area_unit": string("ha | Tagwerk | other"), "area_raw": string("cells as written"),
    "site": {"type": "object", "properties": {
        "elevation_min_m": number(), "elevation_max_m": number(), "exposition": string(), "slope": string(),
        "bedrock": string("Grundgestein"), "soil": string("Bodenart, Gründigkeit, Steinigkeit"),
        "moisture": string("frisch | trocken | nass | Moor | …"), "ground_vegetation": array(string()),
        "frost_exposure": string("Frostlage / ungeschützt / geschützt"), "site_class": string("Bonitäts-/Standortsklasse")}},
    "stand": {"type": "object", "properties": {
        "species": array(SPECIES_ITEM),
        "age_min": number(), "age_max": number(), "age_mean": number(), "age_raw": string(), "age_class": string(),
        "stocking_min": number("Bestockungsgrad 0-1"), "stocking_max": number(),
        "structure": string("Schluss, Stammklassen, Stockwerke, Urwald, Plenter …"),
        "health": string("gutwüchsig | rückgängig | …"), "regeneration": string("Verjüngung: art, erwartet/vorhanden"),
        "volume_per_unit": number(), "volume_total": number(), "volume_unit": string("Fmtr | fm | Ster | Klafter …"),
        "increment_value": number(), "increment_unit": string(),
        "stem_count_main": number(), "stem_count_secondary": number(), "stem_count_unit": string("pro ha | pro Tagwerk | absolute")}},
    "damage": array({"type": "object", "properties": {
        "type": string("Windbruch | Windwurf | Borkenkäfer | Schneebruch | Duftbruch | Schälschaden | Wildverbiss | Frost | Rotfäule | Weide | other"),
        "extent": string(), "year": string(), "quote": string()}, "required": ["type", "quote"]}),
    "management": {"type": "object", "properties": {
        "planned_cut_type": string("Saumschlag | Femelschlag | Kahlschlag | Plenterhieb | Angriffshieb | Durchforstung | Nachhieb | Abtrieb | Läuterung …"),
        "period": string(), "cut_area_value": number(), "cut_area_unit": string(), "cut_yield_value": number(), "cut_yield_unit": string(),
        "transition_code": string("Uebergangsbestimmungen Gattung/Jahr, e.g. P 70, D 160, N 280"),
        "notes": string("Bemerkungen über das Betriebsverfahren, verbatim-ish"),
        "history": array({"type": "object", "properties": {
            "year": string(), "cut_type": string(), "area_value": number(), "timber_value": number(), "fuelwood_value": number(),
            "total_value": number(), "unit": string()}, "required": ["year"]})}},
    "culture": array({"type": "object", "properties": {
        "type": string("Pflanzung | Saat | Lückenpflanzung | Aufforstung | Entwässerung | Schlagpflege | Unkrautreinigung | other"),
        "species": string(), "quantity": number(), "unit": string("Stück | kg | Pfd | Ruthen | Metzen …"),
        "area_value": number(), "area_unit": string(), "year": string(), "cost_value": number(), "cost_unit": string(), "raw": string()},
        "required": ["type"]}),
    "non_timber_use": array({"type": "object", "properties": {
        "type": string("Streunutzung | Weide | Torf | Harz | Mast | Borke | Erden/Steine | other"),
        "species_group": string("Nadel | Laub | Moos | Farn …"), "area_value": number(), "area_unit": string(),
        "quantity": number(), "unit": string("Fuhren | Karren | Ster | Stück Vieh …"), "year": string(), "raw": string()},
        "required": ["type"]}),
    "description_raw": string("verbatim content of the free-text description cell(s) (Beschreibung a-e, Bemerkungen, Boden/Lage/Holzbestand), joined"),
    "red_annotations": array(string("content of <span class='red'> cells for this row")),
    "cross_references": array(string("e.g. 'wie bei IV. 8. b', 'Seite 17'")),
    "uncertain": {"type": "boolean", "description": "row contains [?], [illegible] or ambiguous cell alignment"},
}, "required": ["record_kind", "uncertain"]}

TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_id": string(),
        "table_type": string("recognised form, e.g. Periodentabelle, Wirtschaftsplan, Kulturplan, Bestandsauszählung, Probeflächenaufnahme, Altersklassentabelle, Streunutzungsplan, Wirtschaftsbuch, Betriebsplan"),
        "forest_office": string("Forstamt / Revier from header lines"),
        "operating_class": string("Betriebsklasse / Umtrieb from header, e.g. 'Hochwald in 280 jhr. Umtriebe'"),
        "column_headers": array(string("flattened header path per data column, in order")),
        "period_labels": array(string()),
        "records": array(RECORD_SCHEMA),
        "marginalia": array(string()),
        "image_corrections": IMAGE_CORRECTIONS,
        "parsing_notes": string("column drift, two side-by-side blocks, continuation from previous page, etc."),
        "transcription_quality": {"type": "string", "enum": ["good", "mixed", "poor"]},
    },
    "required": ["page_id", "table_type", "records", "transcription_quality"],
}

SCHEMAS = {"text": TEXT_SCHEMA, "table": TABLE_SCHEMA}
