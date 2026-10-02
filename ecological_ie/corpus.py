import pandas as pd

from .dump import Dump

TABLE_PRIORITY = ["Periodentabelle", "Wirtschaftsplan", "Probeflaechenaufnahmen", "Kulturplan", "Bestandsauszaehlungen",
                  "Altersklassen", "Streunutzungsplan", "Streunutzungsplan-Weideschaft_und_Streunutzung",
                  "Forstnebennutzungen", "Uebersicht-der-Forstverbesserungen", "Materialvorraete", "Betriebsplan",
                  "Entwurf-des-generellen-Betriebsplans", "Angriffshiebe", "Vermessungstabelle", "Wirtschaftsbuch",
                  "nicht-gegeordnet"]
TABLE_SKIP = {"Kostenanschlaege", "Tarif-für-die-Holzpreise", "Kreisflaechen-fuer-gegebene-Durchmesser", "Kubirung",
              "Berechnung-von-Walzen-Stuecken", "Nivellements_Tabellen", "Wegbauplan", "Spezieller_Triftbauplan",
              "Controle_Hauptbuch", "Zusammenstellung_des_Taxations_Solls_und_Habens", "Abschluss-der-Wirtschaftsbuecher",
              "Faellungsnachweisungen", "Aenderungen_am-staendigen_und_unstaendigen_Detail", "Gliederungen",
              "Materialertrag-Scans-in-schlecher-Qualität", "Materialergebnisse", "Massenberechnungen",
              "Materialberechnungen", "Streunutzungsplan-Uebersicht_der_an_den_Kartensteinen_vorgenommenen_Aenderungen",
              "Zusammenstellung-der-Inproduktiven-Flächen", "Verschiedenes"}


def page_id_of(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def enumerate_corpus(dump: Dump, min_chars: int = 300) -> pd.DataFrame:
    rows = []
    for path in dump.glob("Textseiten/*/md/*.md"):
        size = dump.size(path)
        if size >= min_chars:
            subtype = path.split("/")[1].replace("Textseiten-", "")
            rows.append({"seq": f"text_{len(rows):05d}", "source_type": "text", "subtype": subtype,
                         "page_id": page_id_of(path), "file": path, "chars": size})
    for path in dump.glob("Tabellen/*/md/*.md"):
        category, size = path.split("/")[1], dump.size(path)
        if category in TABLE_SKIP or size < min_chars:
            continue
        rows.append({"seq": f"table_{len(rows):05d}", "source_type": "table", "subtype": category,
                     "page_id": page_id_of(path), "file": path, "chars": size,
                     "priority": TABLE_PRIORITY.index(category) if category in TABLE_PRIORITY else 99})
    return pd.DataFrame(rows)


def estimate(corpus: pd.DataFrame, price_input_per_m: float) -> str:
    tokens = corpus["chars"].sum() / 3.2 + len(corpus) * 1800
    summary = (corpus.groupby(["source_type", "subtype"]).agg(pages=("file", "count"), chars=("chars", "sum"))
               .sort_values(["source_type", "pages"], ascending=[True, False]))
    return (f"{summary}\n\n{len(corpus)} pages · ≈{tokens / 1e6:.1f} M input tokens · "
            f"≈${tokens / 1e6 * price_input_per_m:.0f} input cost (output/thinking extra)")
