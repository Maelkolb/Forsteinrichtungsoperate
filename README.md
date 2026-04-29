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

# Single-page scans (default):
python run.py --input scans/ --output output/

# Double-page spreads:
python run.py --input spreads/ --output output/ --double-page

# With deskew and verbose logging:
python run.py -i scans/ -o out/ --deskew -v
```

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
merged header cells (`colspan`/`rowspan`). Red-ink content is wrapped in
`<span class="red">…</span>`.

Region crops and a debug JSON are saved under `output/regions/`.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | — | Input directory (required) |
| `--output` / `-o` | — | Output directory (required) |
| `--double-page` | off | Split each image down the centre |
| `--deskew` | off | Deskew pages (requires `scipy`) |
| `--no-enhance-contrast` | off | Disable auto-contrast |
| `--no-sharpen` | off | Disable unsharp-mask sharpening |
| `--max-regions` | 12 | Max layout regions per page |
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

## Requirements

```
google-genai
Pillow
numpy
scipy          # optional — only needed with --deskew
```
