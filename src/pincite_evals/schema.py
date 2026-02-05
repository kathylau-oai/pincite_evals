from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple


class Authority(BaseModel):
    doc_id: str
    title: str
    court: str
    year: int
    source_url: Optional[str] = None


class PrecedenceEdge(BaseModel):
    lower: str
    higher: str
    relationship: str  # limited_by, overruled_by, persuasive_only, etc.


class Packet(BaseModel):
    packet_id: str
    issue_area: Optional[str] = None
    jurisdiction: Dict[str, Any] = Field(default_factory=dict)
    authorities: List[Authority]
    precedence: List[PrecedenceEdge] = Field(default_factory=list)


class DatasetConstraints(BaseModel):
    allowed_sources: List[str]
    require_citations: bool = True
    cite_format: str = "[{doc_id} ¶{start}-{end}]"
    required_sections: List[str] = Field(default_factory=list)
    max_words: Optional[int] = None


class DatasetItem(BaseModel):
    id: str
    packet_id: str
    task: str
    prompt: str
    constraints: DatasetConstraints
    targets: Dict[str, Any] = Field(default_factory=dict)
