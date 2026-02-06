import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.schema import (  # noqa: E402
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
            "prompt": "Draft a memo section.",
            "scenario_facts": ["Assume closed-world packet."],
            "grading_contract": {
                "expected_citation_groups": [["DOC001[P001.B01]"]],
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
                "prompt": "Draft a memo section.",
                "scenario_facts": ["Assume closed-world packet."],
                "grading_contract": {
                    "expected_citation_groups": [["DOC001[¶12]"]],
                },
            }
        )


def test_schema_accepts_xml_citation_and_normalizes():
    item = SyntheticItem.model_validate(
        {
            "schema_version": "v1",
            "item_id": "packet_1_C_01",
            "packet_id": "packet_1",
            "target_error_mode": "C",
            "query_id": "q_0002",
            "as_of_date": "2026-02-06",
            "prompt": "Draft a memo section.",
            "scenario_facts": ["Assume closed-world packet."],
            "grading_contract": {
                "expected_citation_groups": [["DOC001.P001.B01"]],
                "overextension_trigger_note": "Check whether qualifiers are dropped and turned into categorical rules.",
            },
        }
    )

    assert item.grading_contract.expected_citation_groups[0][0] == "DOC001[P001.B01]"


def test_format_citation_token_as_block_id_converts_canonical_and_xml():
    assert format_citation_token_as_block_id("DOC001[P001.B01]") == "DOC001.P001.B01"
    assert format_citation_token_as_block_id("DOC001.P001.B01") == "DOC001.P001.B01"


def test_extract_doc_id_from_citation_token_handles_both_formats():
    assert extract_doc_id_from_citation_token("DOC001[P001.B01]") == "DOC001"
    assert extract_doc_id_from_citation_token("DOC001.P001.B01") == "DOC001"
