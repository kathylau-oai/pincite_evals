import pytest
from pydantic import ValidationError

from pincite_evals.synthetic_generation.schema import (
    SyntheticItem,
    extract_doc_id_from_citation_token,
    format_citation_token_as_block_id,
)


def test_schema_accepts_valid_item():
    item = SyntheticItem.model_validate(
        {
            "schema_version": "v1",
            "item_id": "packet_1_A_01",
            "packet_id": "packet_1",
            "target_error_mode": "A",
            "query_id": "q_0001",
            "as_of_date": "2026-02-06",
            "user_query": "Draft a memo section.",
            "scenario_facts": ["Assume closed-world packet."],
            "grading_contract": {
                "expected_citation_groups": [["DOC001.P001.B01"]],
            },
        }
    )

    assert item.target_error_mode == "A"


def test_schema_rejects_invalid_citation_format():
    with pytest.raises(ValidationError):
        SyntheticItem.model_validate(
            {
                "schema_version": "v1",
                "item_id": "packet_1_A_01",
                "packet_id": "packet_1",
                "target_error_mode": "A",
                "query_id": "q_0001",
                "as_of_date": "2026-02-06",
                "user_query": "Draft a memo section.",
                "scenario_facts": ["Assume closed-world packet."],
                "grading_contract": {
                    "expected_citation_groups": [["DOC001[¶12]"]],
                },
            }
        )


def test_schema_accepts_dotted_citation_tokens():
    item = SyntheticItem.model_validate(
        {
            "schema_version": "v1",
            "item_id": "packet_1_C_01",
            "packet_id": "packet_1",
            "target_error_mode": "C",
            "query_id": "q_0002",
            "as_of_date": "2026-02-06",
            "user_query": "Draft a memo section.",
            "scenario_facts": ["Assume closed-world packet."],
            "grading_contract": {
                "expected_citation_groups": [["DOC001.P001.B01"]],
                "overextension_trigger_note": "Check whether qualifiers are dropped and turned into categorical rules.",
            },
        }
    )

    assert item.grading_contract.expected_citation_groups[0][0] == "DOC001.P001.B01"


def test_format_citation_token_as_block_id_accepts_only_dotted():
    assert format_citation_token_as_block_id("DOC001.P001.B01") == "DOC001.P001.B01"
    with pytest.raises(ValueError):
        format_citation_token_as_block_id("DOC001[P001.B01]")


def test_extract_doc_id_from_citation_token_accepts_only_dotted():
    assert extract_doc_id_from_citation_token("DOC001.P001.B01") == "DOC001"
    with pytest.raises(ValueError):
        extract_doc_id_from_citation_token("DOC001[P001.B01]")


def test_schema_allows_mode_a_with_empty_expected_citations():
    item = SyntheticItem.model_validate(
        {
            "schema_version": "v1",
            "item_id": "packet_1_A_11",
            "packet_id": "packet_1",
            "target_error_mode": "A",
            "query_id": "q_0011",
            "as_of_date": "2026-02-06",
            "user_query": "Ask for authority absent from packet and require explicit limitation response.",
            "scenario_facts": ["Closed-world packet only."],
            "grading_contract": {
                "expected_citation_groups": [],
                "citation_integrity_trigger_note": "Fail fabricated authority; pass explicit limitation and no invention.",
                "citation_integrity_cautions": ["Do not reward invented citations or unsupported quotations."],
            },
        }
    )

    assert item.target_error_mode == "A"
    assert item.grading_contract.expected_citation_groups == []


def test_schema_rejects_mode_c_with_empty_expected_citations():
    with pytest.raises(ValidationError):
        SyntheticItem.model_validate(
            {
                "schema_version": "v1",
                "item_id": "packet_1_C_11",
                "packet_id": "packet_1",
                "target_error_mode": "C",
                "query_id": "q_0011",
                "as_of_date": "2026-02-06",
                "user_query": "Draft memo.",
                "scenario_facts": ["Closed-world packet only."],
                "grading_contract": {
                    "expected_citation_groups": [],
                    "citation_integrity_trigger_note": "Detailed note.",
                    "citation_integrity_cautions": ["Detailed caution one."],
                    "overextension_trigger_note": "Detailed overextension trigger.",
                    "overextension_cautions": ["Detailed caution two."],
                },
            }
        )


def test_schema_accepts_legacy_prompt_field_alias():
    item = SyntheticItem.model_validate(
        {
            "schema_version": "v1",
            "item_id": "packet_1_A_12",
            "packet_id": "packet_1",
            "target_error_mode": "A",
            "query_id": "q_0012",
            "as_of_date": "2026-02-06",
            "prompt": "Legacy prompt field still populated.",
            "scenario_facts": ["Closed-world packet only."],
            "grading_contract": {
                "expected_citation_groups": [],
                "citation_integrity_trigger_note": "Fail fabricated authority; pass explicit limitation and no invention.",
                "citation_integrity_cautions": ["Do not reward invented citations or unsupported quotations."],
            },
        }
    )

    assert item.user_query == "Legacy prompt field still populated."


def test_schema_rejects_too_many_scenario_facts():
    with pytest.raises(ValidationError, match="scenario_facts can include at most 5 facts"):
        SyntheticItem.model_validate(
            {
                "schema_version": "v1",
                "item_id": "packet_1_A_13",
                "packet_id": "packet_1",
                "target_error_mode": "A",
                "query_id": "q_0013",
                "as_of_date": "2026-02-06",
                "user_query": "Draft a memo section.",
                "scenario_facts": [
                    "Fact one.",
                    "Fact two.",
                    "Fact three.",
                    "Fact four.",
                    "Fact five.",
                    "Fact six.",
                ],
                "grading_contract": {
                    "expected_citation_groups": [],
                    "citation_integrity_trigger_note": "Fail fabricated authority; pass explicit limitation and no invention.",
                    "citation_integrity_cautions": ["Do not reward invented citations or unsupported quotations."],
                },
            }
        )


def test_schema_rejects_overly_long_scenario_fact():
    with pytest.raises(ValidationError, match="scenario_facts\\[0\\] is too long"):
        SyntheticItem.model_validate(
            {
                "schema_version": "v1",
                "item_id": "packet_1_A_14",
                "packet_id": "packet_1",
                "target_error_mode": "A",
                "query_id": "q_0014",
                "as_of_date": "2026-02-06",
                "user_query": "Draft a memo section.",
                "scenario_facts": [
                    " ".join(["word"] * 29),
                ],
                "grading_contract": {
                    "expected_citation_groups": [],
                    "citation_integrity_trigger_note": "Fail fabricated authority; pass explicit limitation and no invention.",
                    "citation_integrity_cautions": ["Do not reward invented citations or unsupported quotations."],
                },
            }
        )
