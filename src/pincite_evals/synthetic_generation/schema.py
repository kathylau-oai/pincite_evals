import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# Citation tokens must be dotted packet block IDs:
#   DOC###.P###.B##
#
# We intentionally do not accept bracket citations (DOC###\[P###.B##\]) to keep the
# pipeline aligned with packet rendering (<BLOCK id="DOC###.P###.B##"> ...).
DOTTED_CITATION_TOKEN_PATTERN = re.compile(r"^DOC\d{3}\.P\d{3}\.B\d{2}$")


def normalize_citation_token(citation_token: str) -> str:
    text_value = citation_token.strip()
    if DOTTED_CITATION_TOKEN_PATTERN.match(text_value):
        return text_value
    raise ValueError(
        f"Invalid citation token '{citation_token}'. Expected DOC###.P###.B## (dotted packet block ID)."
    )


def format_citation_token_as_block_id(citation_token: str) -> str:
    # Citation tokens are already block IDs in dotted format.
    return normalize_citation_token(citation_token)


def extract_doc_id_from_citation_token(citation_token: str) -> str:
    normalized_citation_token = normalize_citation_token(citation_token)
    return normalized_citation_token.split(".", maxsplit=1)[0]


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
        normalized_groups: list[list[str]] = []
        for group_index, citation_group in enumerate(value):
            if not citation_group:
                raise ValueError(f"expected_citation_groups[{group_index}] must be non-empty.")
            normalized_groups.append([normalize_citation_token(citation_token) for citation_token in citation_group])
        return normalized_groups


class SyntheticItem(BaseModel):
    schema_version: Literal["v1"] = "v1"
    item_id: str
    packet_id: str
    target_error_mode: Literal["A", "C", "D"]
    query_id: str
    as_of_date: str
    user_query: str
    scenario_facts: list[str] = Field(default_factory=list)
    grading_contract: GradingContract

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_prompt_field(cls, value: object) -> object:
        if isinstance(value, dict):
            user_query = value.get("user_query")
            legacy_prompt = value.get("prompt")
            if (not isinstance(user_query, str) or not user_query.strip()) and isinstance(legacy_prompt, str):
                migrated_value = dict(value)
                migrated_value["user_query"] = legacy_prompt
                return migrated_value
        return value

    @field_validator("item_id", "packet_id", "query_id", "as_of_date", "user_query")
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

    @model_validator(mode="after")
    def validate_mode_specific_expected_citations(self) -> "SyntheticItem":
        if self.target_error_mode in {"C", "D"} and not self.grading_contract.expected_citation_groups:
            raise ValueError("expected_citation_groups must be non-empty for target_error_mode C or D.")
        return self
