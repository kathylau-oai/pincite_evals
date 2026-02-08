import re
from dataclasses import dataclass
from typing import List


# Excerpt cite format expected in memo outputs:
#   DOC###.P###.B##
_EXCERPT_CITE_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])"
    r"(?P<doc_id>DOC\d{3})"
    r"\."
    r"(?P<page_id>P\d{3})"
    r"\."
    r"(?P<block_id>B\d{2})"
    r"(?![A-Za-z0-9_\-#])"
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
        page_id = match.group("page_id")
        block_id = match.group("block_id")
        excerpt_id = f"{page_id}.{block_id}"
        raw = f"{doc_id}.{page_id}.{block_id}"
        citations.append(ExcerptCitation(doc_id=doc_id, excerpt_id=excerpt_id, raw=raw))
    return citations
