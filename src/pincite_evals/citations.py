from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# Default cite format expected in outputs:
#   [doc_id ¶start-end]
_CITE_RE = re.compile(r"\[(?P<doc_id>[A-Za-z0-9_\-]+)\s+¶(?P<start>\d+)(?:-(?P<end>\d+))?\]")


@dataclass(frozen=True)
class Citation:
    doc_id: str
    start: int
    end: int
    raw: str


def extract_citations(text: str) -> List[Citation]:
    cites: List[Citation] = []
    for m in _CITE_RE.finditer(text):
        doc_id = m.group('doc_id')
        start = int(m.group('start'))
        end = int(m.group('end') or start)
        cites.append(Citation(doc_id=doc_id, start=start, end=end, raw=m.group(0)))
    return cites
