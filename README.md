# Forsteinrichtungsoperate — Historical Document Pipeline

Pipeline for transcribing 19th-century Bavarian/German handwritten forestry
administrative records (*Forsteinrichtungsoperate*) using Gemini. Handles complex
multi-column tables, Kurrent/Sütterlin script, mixed black/red ink, and
graph-paper diagrams.

## Supported source types

| Source | Region types detected |
|---|---|
| Forest inventory ledgers (*Hochwald*, *Altersklassen*) | TitleRegion, TableRegion |
| Grazing registers (*Weidenutzung*, *Weidenutzungsschaft*) | TitleRegion, TableRegion |
| Tree-height / growth curves (*Baumhöhe*) | GraphRegion |
| Free narrative or annotation pages | ParagraphRegion, MarginaliaRegion |

## Quick start

```bash
pip install google-genai Pillow numpy
export GOOGLE_API_KEY="your-key"

# Single-page scans, mixed content (default — full layout pipeline):
python run.py --input scans/ --output output/

# Pages dominated by one big complex table (single-pass, no region detection):
python run.py -i scans/ -o out/ --doc-type table

# Tree-height graph pages (single-pass, tailored to forestry curves):
python run.py -i scans/ -o out/ --doc-type graph

# Map pages (single-pass, extracts type/title/geoident/scale/date):
python run.py -i scans/ -o out/ --doc-type map

# Text-heavy pages (full layout pipeline, transcribes all running text):
python run.py -i scans/ -o out/ --doc-type text

# Double-page spreads:
python run.py --input spreads/ --output output/ --double-page

# With deskew and verbose logging on the newest model:
python run.py -i scans/ -o out/ --deskew -v --model gemini-3.1-pro-preview
```

## Document types (`--doc-type`)

The `--doc-type` flag selects which pipeline branch is used.

| Value | Pipeline | Best for |
|---|---|---|
| `mixed` (default) | Region detection → per-region transcription | Diverse / unknown content |
| `text` | Region detection → per-region transcription | Pages dominated by running narrative text |
| `table` | Single-pass full-page call (no detection) | Pages with one big complex table (with optional surrounding text) |
| `graph` | Single-pass full-page call (no detection) | Tree-height curve graphs on graph paper |
| `map`   | Single-pass full-page call (no detection) | Maps — extracts a structured set of fields (type, title, geoident, scale, date, compass, transcribed text, notes) |

For `table`, `graph`, and `map`, the entire page is sent to the model in one
request, which is faster and avoids region-detection mistakes when the page
already has a single dominant element.

## Output

Each page produces one Markdown file in `output/md/<page_id>.md`:

```
---
page_id: scan_001
page_number: 42
regions: "TitleRegion×1, TableRegion×1"
---

## Hochwaldbestand 1875

<!-- TableRegion: 14 rows × 8 cols, 3 header rows, red ink: yes -->
```html
<table>
  <thead>
    <tr><th colspan="4">Schönau</th><th colspan="4">St. Oswald</th></tr>
    ...
  </thead>
  <tbody>
    ...
  </tbody>
</table>
` ` `
```

Tables are output as HTML `<table>` (inside a fenced block) to preserve
merged header cells (`colspan`/`rowspan`).

Region crops and a debug JSON are saved under `output/regions/`.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | — | Input directory (required) |
| `--output` / `-o` | — | Output directory (required) |
| `--doc-type` | `mixed` | Source category: `text` / `table` / `graph` / `map` / `mixed` |
| `--double-page` | off | Split each image down the centre |
| `--deskew` | off | Deskew pages (requires `scipy`) |
| `--no-enhance-contrast` | off | Disable auto-contrast |
| `--no-sharpen` | off | Disable unsharp-mask sharpening |
| `--max-regions` | 12 | Max layout regions per page (text/mixed only) |
| `--workers` / `-w` | 4 | Parallel threads |
| `--model` | `gemini-3-flash-preview` | Gemini model ID |
| `--verbose` / `-v` | off | Debug logging |

## Google Colab

Open `colab_pipeline.ipynb` in Colab. It will:
1. Clone this repo
2. Install dependencies
3. Let you upload scan images
4. Run the pipeline and download results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maelkolb/Forsteinrichtungsoperate/blob/main/colab_pipeline.ipynb)


