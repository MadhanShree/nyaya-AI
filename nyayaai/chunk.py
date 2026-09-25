"""Split a document into clause-sized chunks so answers can cite "Clause 8"."""

import re
from dataclasses import dataclass

MAX_CHARS = 1500
# "Clause 8", "Section 8", "8." or "8)" at line start; "8.1" stays inside clause 8.
HEADING = re.compile(r"^\s*(?:(?:clause|section|article)\s+(\d+)|(\d{1,2})[.)](?:\s|$))", re.I)


@dataclass(frozen=True)
class Chunk:
    id: str
    label: str
    page: int
    text: str


def _split(text: str) -> list[str]:
    parts, cur = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if cur and len(cur) + len(sentence) > MAX_CHARS:
            parts.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
    parts.append(cur)
    return [p[i : i + MAX_CHARS] for p in parts for i in range(0, len(p), MAX_CHARS)]


def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    sections = [["Preamble", pages[0][0], []]]
    for page, text in pages:
        for line in filter(None, (ln.strip() for ln in text.splitlines())):
            m = HEADING.match(line)
            if m:
                sections.append([f"Clause {m.group(1) or m.group(2)}", page, []])
            sections[-1][2].append(line)
    if sum(1 for s in sections if s[0] != "Preamble") < 2:  # no clause structure found
        sections = [[f"Page {p}", p, [para]] for p, t in pages for para in re.split(r"\n\s*\n", t) if para.strip()]
    out: list[Chunk] = []
    for label, page, lines in sections:
        text = "\n".join(lines).strip()
        for n, part in enumerate(_split(text) if text else []):
            out.append(Chunk(f"c{len(out) + 1}", label if n == 0 else f"{label} (cont.)", page, part))
    return out
