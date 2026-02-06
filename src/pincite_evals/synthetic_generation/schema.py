import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


CITATION_TOKEN_PATTERN = re.compile(r"^DOC\d{3}\[P\d{3}\.B\d{2}\]$")
VALID_ERROR_MODES = {"A", "C", "D"}


class GradingContract(BaseModel):
    expected_citation_groups: list[list[str]] = Field(default_factory=list)
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
            for citation_token in citation_group:
                if CITATION_TOKEN_PATTERN.match(citation_token) is None:
                    raise ValueError(
                        f"Invalid citation token '{citation_token}'. Expected format DOC_ID[P###.B##]."
                    )
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
