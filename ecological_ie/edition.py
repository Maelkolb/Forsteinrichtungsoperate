import html
import re
from html.parser import HTMLParser

import markdown

from .normalize import parse_number, strip_markup

ALLOWED = {"p", "br", "h1", "h2", "h3", "h4", "h5", "aside", "blockquote", "em", "strong", "i", "b", "u", "s", "del",
           "strike", "ins", "sup", "sub", "small", "span", "mark", "table", "thead", "tbody", "tr", "th", "td", "hr",
           "ul", "ol", "li", "div"}
ALLOWED_ATTRS = {"class", "colspan", "rowspan", "title", "data-q", "data-htr"}
VOID = {"br", "hr"}
CLASS_OK = re.compile(r"^[a-z0-9 _-]+$")


class Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED:
            return
        kept = []
        for name, value in attrs:
            if name in ALLOWED_ATTRS and value is not None and (name != "class" or CLASS_OK.match(value)):
                if name in ("colspan", "rowspan") and not value.isdigit():
                    continue
                kept.append(f' {name}="{html.escape(value, quote=True)}"')
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in ALLOWED and tag in self.stack:
            while self.stack:
                open_tag = self.stack.pop()
                self.out.append(f"</{open_tag}>")
                if open_tag == tag:
                    break

    def handle_data(self, data):
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")

    def result(self) -> str:
        return "".join(self.out) + "".join(f"</{tag}>" for tag in reversed(self.stack))


def sanitize(markup: str) -> str:
    parser = Sanitizer()
    parser.feed(markup)
    parser.close()
    return parser.result()


def editorial(text: str) -> str:
    text = re.sub(r"\[crossed out:\s*([^\]\n]+)\]", r"<del>\1</del>", text)
    text = re.sub(r"\[(?:Stempel|stamp):\s*([^\]\n]+)\]", r'<span class="ed">Stempel: \1</span>', text, flags=re.I)
    text = re.sub(r"\[(?:signature|Unterschrift|unleserliche Unterschrift|Unleserliche Unterschrift)\]",
                  r'<span class="ed">Unterschrift</span>', text)
    text = re.sub(r"\[(?:illegible|unleserlich)[^\]\n]*\]", r'<span class="ed">unleserlich</span>', text, flags=re.I)
    text = re.sub(r"\[\?\]", r'<span class="unc" title="unsichere Lesung">?</span>', text)
    text = re.sub(r"\[([^\]\n\^]{1,40})\?\]", r'<span class="unc-w" title="unsichere Lesung">\1</span>', text)
    text = re.sub(r"\[Marginalie\]", "", text)
    text = re.sub(r"\[([^\]\n\^]{1,40})\]", r'<span class="sup-text">[\1]</span>', text)
    return text


def marginalia(text: str) -> str:
    out, block = [], []

    def flush():
        lines = [re.sub(r"^>\s?", "", line) for line in block]
        lines = [line for line in lines if line.strip() not in ("*[Marginalie]*", "[Marginalie]", "*Marginalie*", "")]
        if lines:
            out.append('<span class="marg">' + "<br>".join(lines) + "</span>")
        block.clear()

    for line in text.split("\n"):
        if line.startswith(">"):
            block.append(line)
        else:
            if block:
                flush()
            out.append(line)
    if block:
        flush()
    return "\n".join(out)


BLOCK_START = re.compile(r"(\||```|<(table|div|aside|p|h\d|ul|ol|blockquote|hr)\b)")


def line_breaks(text: str) -> str:
    blocks = re.split(r"(\n\s*\n)", text)
    result = []
    for block in blocks:
        lines = block.split("\n")
        headings = []
        while lines and lines[0].lstrip().startswith("#"):
            headings.append(lines.pop(0))
        if headings:
            result.append("\n".join(headings) + ("\n\n" if lines else ""))
            block = "\n".join(lines)
        stripped = block.lstrip()
        if not stripped or BLOCK_START.match(stripped) or "\n" not in block.strip("\n"):
            result.append(block)
            continue
        joined = lines[0]
        for previous, line in zip(lines, lines[1:]):
            if re.search(r"\w[-=¬]\s*$", previous) and re.match(r"\s*[a-zäöüß]", line):
                joined = re.sub(r"[-=¬]\s*$", '<span class="hy">-</span>', joined) + '<span class="lb hy"></span>' + line.lstrip()
            else:
                joined += ' <span class="lb"></span>' + line
        result.append(joined)
    return "".join(result)


def wrap_first(text: str, needle: str, before: str, after: str) -> str:
    if not needle or len(needle) < 2 or any(c in needle for c in "<>[]*#`|\n"):
        return text
    index = text.find(needle)
    while index >= 0:
        head = text[:index]
        inside_tag = head.rfind("<") > head.rfind(">")
        inside_mark = head.count("<mark") > head.count("</mark>")
        if not inside_tag and not inside_mark:
            return head + before + needle + after + text[index + len(needle):]
        index = text.find(needle, index + 1)
    return text


def plain_with_map(text: str) -> tuple[str, list[int]]:
    chars, origin, i = [], [], 0
    while i < len(text):
        if text[i] == "<":
            end = text.find(">", i)
            if end > 0:
                i = end + 1
                continue
        hyphen = re.match(r"[-=¬][ \t]*\n[ \t]*(?=[a-zäöüß])", text[i:])
        if hyphen and i > 0 and text[i - 1].isalpha():
            i += hyphen.end()
            continue
        char = text[i]
        if char.isspace() or char in "*_#>`":
            if chars and chars[-1] != " ":
                chars.append(" ")
                origin.append(i)
        else:
            chars.append(char)
            origin.append(i)
        i += 1
    return "".join(chars), origin


def normalise_quote(quote: str) -> str:
    quote = re.sub(r"<[^>]+>", "", quote or "")
    quote = re.sub(r"(\w)[-=¬]\s+(?=[a-zäöüß])", r"\1", quote)
    return re.sub(r"[\s*_#>`]+", " ", quote).strip()


def mark_quotes(text: str, quotes: list[tuple[str, str]]) -> str:
    plain, origin = plain_with_map(text)
    lowered = plain.lower()
    spans = []
    for index, quote in quotes:
        needle = normalise_quote(quote)
        if len(needle) < 12:
            continue
        start = plain.find(needle)
        if start < 0:
            start = lowered.find(needle.lower())
        if start < 0:
            words = needle.split(" ")
            prefix = " ".join(words[:6])
            start = lowered.find(prefix.lower()) if len(words) >= 6 else -1
            needle = prefix
        if start < 0:
            continue
        a, b = origin[start], origin[min(start + len(needle), len(origin)) - 1] + 1
        spans.append((a, b, index))
    spans.sort(key=lambda s: s[0], reverse=True)
    taken = []
    for a, b, index in spans:
        segment = text[a:b]
        overlaps = any(not (b <= x or a >= y) for x, y in taken)
        if overlaps:
            text = text[:a] + f'<span class="qa" data-q="{index}"></span>' + text[a:]
        elif "<" in segment or ">" in segment or "\n\n" in segment:
            text = (text[:a] + f'<span class="qa" data-q="{index}"></span>' + segment
                    + f'<span class="qz" data-q="{index}"></span>' + text[b:])
            taken.append((a, b))
        else:
            text = text[:a] + f'<span class="q" data-q="{index}">' + segment + "</span>" + text[b:]
            taken.append((a, b))
    return text


def render_transcript(text: str, corrections: list[dict] | None = None, quotes: list[tuple[int, str]] | None = None) -> str:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)
    text = re.sub(r"```html\s*\n?(.*?)```", lambda m: "\n\n" + m.group(1).strip() + "\n\n", text, flags=re.S)
    for correction in corrections or []:
        if correction.get("status") == "applied" and correction.get("transcript_reads") != correction.get("image_reads"):
            text = wrap_first(text, correction.get("image_reads", ""),
                              f'<mark class="corr" data-htr="{html.escape(correction.get("transcript_reads", ""), quote=True)}">', "</mark>")
    text = mark_quotes(text, quotes or [])
    text = marginalia(text)
    text = editorial(text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    text = line_breaks(text)
    rendered = markdown.markdown(text, output_format="html")
    return sanitize(rendered)


def header_matrix(paths: list[str]) -> list[list[tuple[str, int, int]]]:
    split = [[part.strip() for part in path.split("›")] if path else [""] for path in paths]
    depth = max((len(parts) for parts in split), default=1)
    rows = []
    for level in range(depth):
        row, col = [], 0
        while col < len(split):
            parts = split[col]
            if level >= len(parts):
                col += 1
                continue
            span = 1
            while (col + span < len(split) and len(split[col + span]) > level
                   and split[col + span][:level + 1] == parts[:level + 1] and level < len(parts) - 1):
                span += 1
            rowspan = depth - level if level == len(parts) - 1 else 1
            row.append((parts[level], span, rowspan))
            col += span
        rows.append(row)
    return rows


NUMERIC = re.compile(r"^[\s\d.,½¼¾⅓⅔/\-—–]+$")


def cell_html(raw: str, red: str, classes: list[str], title: str) -> str:
    text = html.escape(strip_markup(raw or ""))
    if red:
        text += (" " if text else "") + f'<span class="red">{html.escape(red)}</span>'
    if NUMERIC.match(strip_markup(raw or "") or "x") and (raw or "").strip():
        classes = classes + ["n"]
    attributes = f' class="{" ".join(classes)}"' if classes else ""
    if title:
        attributes += f' title="{html.escape(title, quote=True)}"'
    return f"<td{attributes}>{text}</td>"


def render_grid(grid: dict, form: dict, bad_cells: set, reread: dict, table_offset: int = 0) -> str:
    columns_meta = {c["id"]: c for c in form.get("columns", [])}
    parts = []
    for t, table in enumerate(grid["result"].get("tables", [])):
        columns = table.get("columns", [])
        if not columns:
            continue
        head = []
        for row in header_matrix([c.get("header", "") for c in columns]):
            cells = "".join(f'<th colspan="{span}" rowspan="{rowspan}">{html.escape(text)}</th>' if (span > 1 or rowspan > 1)
                            else f"<th>{html.escape(text)}</th>" for text, span, rowspan in row)
            head.append(f"<tr>{cells}</tr>")
        canon = "".join(
            f'<th class="canon" title="{html.escape(columns_meta.get(c.get("canonical"), {}).get("label", c.get("canonical", "")), quote=True)}">'
            f'{html.escape(c.get("canonical", "") if c.get("canonical") != "other" else "")}</th>' for c in columns)
        body = []
        for r, row in enumerate(table.get("rows", [])):
            kind = row.get("row_type", "data")
            cells = row.get("cells", [])
            box = row.get("box_2d") or []
            data_box = f' data-b="{",".join(str(int(v)) for v in box)}"' if len(box) == 4 else ""
            if kind in ("group_header", "heading", "note") and sum(1 for c in cells if strip_markup(c)) <= 2:
                text = " ".join(strip_markup(c) for c in cells if strip_markup(c))
                red = " ".join(item.get("text", "") for item in row.get("red", []) if isinstance(item, dict))
                body.append(f'<tr class="{kind}" data-r="{r}"{data_box}><td colspan="{len(columns)}">{html.escape(text)}'
                            + (f' <span class="red">{html.escape(red)}</span>' if red else "") + "</td></tr>")
                continue
            red = {item["col"]: item["text"] for item in row.get("red", []) if isinstance(item, dict) and "col" in item}
            uncertain = set(row.get("uncertain", []))
            tds = []
            for c in range(len(columns)):
                classes, title = [], ""
                canonical = columns[c].get("canonical")
                if (t + table_offset, r, canonical) in bad_cells:
                    classes.append("bad")
                    title = "sum differs"
                if (t + table_offset, r, c) in reread:
                    classes.append("rr")
                    title = f"first reading: {reread[(t + table_offset, r, c)] or 'empty'}"
                if c in uncertain:
                    classes.append("unc-c")
                tds.append(cell_html(cells[c] if c < len(cells) else "", red.get(c, ""), classes, title))
            body.append(f'<tr class="{kind}" data-r="{r}"{data_box}>{"".join(tds)}</tr>')
        parts.append(f'<div class="grid-wrap"><table class="grid"><thead>{"".join(head)}<tr class="canon-row">{canon}</tr></thead>'
                     f'<tbody>{"".join(body)}</tbody></table></div>')
    return "".join(parts)


def is_number(raw: str) -> bool:
    return parse_number(raw).value is not None
