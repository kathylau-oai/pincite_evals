import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


CANONICAL_CITATION_TOKEN_PATTERN = re.compile(r"^DOC\d{3}\[P\d{3}\.B\d{2}\]$")
XML_CITATION_TOKEN_PATTERN = re.compile(r"^DOC\d{3}\.P\d{3}\.B\d{2}$")
VALID_ERROR_MODES = {"A", "C", "D"}


def normalize_citation_token(citation_token: str) -> str:
    text_value = citation_token.strip()
    if CANONICAL_CITATION_TOKEN_PATTERN.match(text_value):
        return text_value
    if XML_CITATION_TOKEN_PATTERN.match(text_value):
        doc_id, page_number, block_number = text_value.split(".")
        return f"{doc_id}[{page_number}.{block_number}]"
    raise ValueError(f"Invalid citation token '{citation_token}'. Expected DOC_ID[P###.B##] or DOC_ID.P###.B##.")


def format_citation_token_as_block_id(citation_token: str) -> str:
    normalized_citation_token = normalize_citation_token(citation_token)
    doc_id, excerpt_id = normalized_citation_token.split("[", maxsplit=1)
    excerpt_id = excerpt_id.rstrip("]")
    return f"{doc_id}.{excerpt_id}"


def extract_doc_id_from_citation_token(citation_token: str) -> str:
    normalized_citation_token = normalize_citation_token(citation_token)
    return normalized_citation_token.split("[", maxsplit=1)[0]


class GradingContract(BaseModel):
    expected_citation_groups: list[list[str]] = Field(default_factory=list)
    citation_integrity_trigger_note: str | None = None
    citation_integrity_cautions: list[str] = Field(default_factory=list)
    overextension_trigger_note: str | None = None
    overextension_cautions: list[str] = Field(default_factory=list)
    precedence_trigger_note: str | None = None
    precedence_cautions: list[str] = Field(default_factory=list)

    @field_validator("expected_citation_groups")
    @classmethod
    def validate_expected_citations(cls, value: list[list[str]]) -> list[list[str]]:
        if not value:
            raise ValueError("expected_citation_groups must be non-empty.")

        for group_index, citation_group in enumerate(value):
            if not citation_group:
                raise ValueError(f"expected_citation_groups[{group_index}] must be non-empty.")
            cleaned_group: list[str] = []
            for citation_token in citation_group:
                cleaned_group.append(normalize_citation_token(citation_token))
            value[group_index] = cleaned_group
        return value


class SyntheticItem(BaseModel):
    schema_version: Literal["v1"] = "v1"
    item_id: str
    packet_id: str
    target_error_mode: Literal["A", "C", "D"]
    query_id: str
    as_of_date: str
    prompt: str
    scenario_facts: list[str] = Field(default_factory=list)
    grading_contract: GradingContract

    @field_validator("item_id", "packet_id", "query_id", "as_of_date", "prompt")
    @classmethod
    def validate_non_empty_text_fields(cls, value: str) -> str:
        text_value = value.strip()
        if not text_value:
            raise ValueError("Text field must be non-empty.")
        return text_value

    @field_validator("scenario_facts")
    @classmethod
    def validate_scenario_facts(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("scenario_facts must be non-empty.")
        cleaned_facts = [fact.strip() for fact in value if fact.strip()]
        if not cleaned_facts:
            raise ValueError("scenario_facts must contain at least one non-empty value.")
        return cleaned_facts
