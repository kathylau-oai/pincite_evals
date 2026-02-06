import re
from dataclasses import dataclass
from typing import List


# Excerpt cite format expected in memo outputs:
#   DOC_ID[P<page>.B<block>(#<hash>)?]
_EXCERPT_CITE_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])"
    r"(?P<doc_id>[A-Za-z0-9_\-]+)"
    r"\["
    r"(?P<excerpt_id>P\d{3}\.B\d{2}(?:#[A-Za-z0-9]{4,16})?)"
    r"\]"
    r"(?![A-Za-z0-9_\-])"
)


@dataclass(frozen=True)
class ExcerptCitation:
    doc_id: str
    excerpt_id: str
    raw: str


def extract_excerpt_citations(text: str) -> List[ExcerptCitation]:
    citations: List[ExcerptCitation] = []
    for match in _EXCERPT_CITE_RE.finditer(text):
        doc_id = match.group("doc_id")
        excerpt_id = match.group("excerpt_id")
        raw = f"{doc_id}[{excerpt_id}]"
        citations.append(ExcerptCitation(doc_id=doc_id, excerpt_id=excerpt_id, raw=raw))
    return citations
