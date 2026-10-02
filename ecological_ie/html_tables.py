import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

TABLE_BLOCK = re.compile(r"```html\s*(.*?)```|(<table\b.*?</table>)", re.S)


@dataclass
class Cell:
    text: str
    colspan: int = 1
    rowspan: int = 1
    header: bool = False
    red: bool = False


@dataclass
class Table:
    head: list[list[Cell]] = field(default_factory=list)
    body: list[list[Cell]] = field(default_factory=list)


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[Table] = []
        self.section = "body"
        self.row = None
        self.cell = None
        self.red_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.tables.append(Table())
        elif tag == "thead":
            self.section = "head"
        elif tag in ("tbody", "tfoot"):
            self.section = "body"
        elif tag == "tr" and self.tables:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = Cell("", int(attrs.get("colspan") or 1), int(attrs.get("rowspan") or 1), header=tag == "th")
        elif tag == "br" and self.cell is not None:
            self.cell.text += " "
        elif tag == "span" and "red" in (attrs.get("class") or "") and self.cell is not None:
            self.cell.red = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cell.text = re.sub(r"\s+", " ", self.cell.text).strip()
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            table = self.tables[-1]
            (table.head if self.section == "head" else table.body).append(self.row)
            self.row = None
        elif tag == "thead":
            self.section = "body"

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.text += data


def parse_tables(markdown: str) -> list[Table]:
    parser = TableParser()
    for match in TABLE_BLOCK.finditer(markdown):
        parser.feed(match.group(1) or match.group(2))
    return parser.tables


def expand(rows: list[list[Cell]]) -> list[list[str]]:
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    grid.setdefault((r + dr, c + dc), cell.text)
            c += cell.colspan
    if not grid:
        return []
    height = max(r for r, _ in grid) + 1
    width = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(width)] for r in range(height)]


def column_paths(table: Table) -> list[str]:
    grid = expand(table.head)
    if not grid:
        return []
    paths = []
    for c in range(len(grid[0])):
        parts = []
        for row in grid:
            text = row[c] if c < len(row) else ""
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        paths.append(" › ".join(parts))
    return paths


def body_width(table: Table) -> int:
    return max((sum(cell.colspan for cell in row) for row in table.body), default=0)
