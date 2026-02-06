import re
from dataclasses import dataclass
from typing import List


# Default cite format expected in outputs:
#   [doc_id ¶start-end]
_CITE_RE = re.compile(r"\[(?P<doc_id>[A-Za-z0-9_\-]+)\s+¶(?P<start>\d+)(?:-(?P<end>\d+))?\]")

# Excerpt cite format expected in memo outputs:
#   DOC_ID[EXCERPT_ID]
_EXCERPT_CITE_RE = re.compile(
    r"(?P<doc_id>[A-Za-z0-9_\-]+)\[(?P<excerpt_id>[A-Za-z0-9._:\-#]+)\]"
)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    start: int
    end: int
    raw: str


@dataclass(frozen=True)
class ExcerptCitation:
    doc_id: str
    excerpt_id: str
    raw: str


def extract_citations(text: str) -> List[Citation]:
    cites: List[Citation] = []
    for m in _CITE_RE.finditer(text):
        doc_id = m.group('doc_id')
        start = int(m.group('start'))
        end = int(m.group('end') or start)
        cites.append(Citation(doc_id=doc_id, start=start, end=end, raw=m.group(0)))
    return cites


def extract_excerpt_citations(text: str) -> List[ExcerptCitation]:
    citations: List[ExcerptCitation] = []
    for match in _EXCERPT_CITE_RE.finditer(text):
        doc_id = match.group("doc_id")
        excerpt_id = match.group("excerpt_id")
        raw = f"{doc_id}[{excerpt_id}]"
        citations.append(ExcerptCitation(doc_id=doc_id, excerpt_id=excerpt_id, raw=raw))
    return citations
