import re

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
LINE_END_HYPHEN = re.compile(r"(\w)[-=]\n(?=[a-zäöüß])")
ILLEGIBLE = re.compile(r"\[(?:illegible|unleserlich)[^\]]*\]", re.I)
UNCERTAIN = re.compile(r"\[\?\]")


def parse_md(raw: str) -> tuple[dict, str]:
    meta, body = {}, raw
    match = FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"')
        body = raw[match.end():]
    return meta, body


def truncate_marginalia_loops(body: str, max_lines: int = 25) -> str:
    out, run = [], []

    def flush():
        if len(run) > max_lines:
            out.extend(run[:max_lines])
            out.append(f"> *[marginalia truncated: {len(run) - max_lines} further lines]*")
        else:
            out.extend(run)
        run.clear()

    for line in body.splitlines():
        if line.startswith(">"):
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def dehyphenate(text: str) -> str:
    return LINE_END_HYPHEN.sub(r"\1", text)


def compact_html(text: str) -> str:
    def compact(match):
        block = re.sub(r"\n\s+", "\n", match.group(0))
        return re.sub(r"\n(?=</?(?:td|th|tr|thead|tbody)\b)", "", block)

    return re.sub(r"```html.*?```", compact, text, flags=re.S)


def clean_body(body: str, source_type: str) -> str:
    body = truncate_marginalia_loops(body)
    body = dehyphenate(body) if source_type == "text" else compact_html(body)
    return body.strip()


def page_quality(body: str) -> dict:
    return {
        "chars": len(body),
        "tables": body.count("<table"),
        "illegible": len(ILLEGIBLE.findall(body)),
        "uncertain": len(UNCERTAIN.findall(body)),
        "marginalia_truncated": "[marginalia truncated" in truncate_marginalia_loops(body),
    }
