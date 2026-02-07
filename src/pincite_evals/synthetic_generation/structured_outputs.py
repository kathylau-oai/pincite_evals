from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class StructuredGradingContract(BaseModel):
    expected_citation_groups: list[list[str]] = Field(default_factory=list)
    citation_integrity_trigger_note: str | None = None
    citation_integrity_cautions: list[str] = Field(default_factory=list)
    overextension_trigger_note: str | None = None
    overextension_cautions: list[str] = Field(default_factory=list)
    precedence_trigger_note: str | None = None
    precedence_cautions: list[str] = Field(default_factory=list)


class GeneratedSyntheticItemOutput(BaseModel):
    schema_version: str | None = None
    item_id: str | None = None
    packet_id: str | None = None
    target_error_mode: str | None = None
    query_id: str | None = None
    as_of_date: str | None = None
    user_query: str | None = None
    scenario_facts: list[str] = Field(default_factory=list)
    grading_contract: StructuredGradingContract = Field(default_factory=StructuredGradingContract)

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

    @field_validator("user_query", mode="before")
    @classmethod
    def normalize_user_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value if text_value else None


class VerifierOutput(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str
    risk_flags: list[str] = Field(default_factory=list)
    suggested_fix: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: str) -> str:
        return str(value).strip().lower()
