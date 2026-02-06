import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.pipeline import _deterministic_validation  # noqa: E402


def _detailed_integrity_note() -> str:
    return (
        "A passing answer must tie each legal proposition to cited packet text and avoid invented rules, "
        "invented quotations, and invented authority status labels. The grader should fail any claim that cannot "
        "be traced to the cited packet blocks and should require explicit uncertainty statements when packet support "
        "is incomplete rather than allowing confident unsupported conclusions."
    )


def _detailed_cautions(prefix: str) -> list[str]:
    return [
        f"{prefix} caution one requires explicit comparison between claimed rule scope and cited packet language.",
        f"{prefix} caution two prohibits adding doctrinal elements that are not stated in cited packet blocks.",
        f"{prefix} caution three requires the memo to acknowledge uncertainty instead of inventing legal certainty.",
    ]


def test_missing_citation_integrity_note_fails_deterministic_validation():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_A_01",
        "packet_id": "packet_1",
        "target_error_mode": "A",
        "query_id": "q_0001",
        "as_of_date": "2026-02-06",
        "prompt": "Draft an internal memo section.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [["DOC001[P001.B01]"]],
            "citation_integrity_trigger_note": None,
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
        },
    }

    deterministic_pass, reasons, _ = _deterministic_validation(item_payload, {"DOC001[P001.B01]"})

    assert deterministic_pass is False
    assert "missing_citation_integrity_criteria" in reasons


def test_missing_mode_specific_trigger_note_fails_for_overextension():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_C_01",
        "packet_id": "packet_1",
        "target_error_mode": "C",
        "query_id": "q_0002",
        "as_of_date": "2026-02-06",
        "prompt": "Draft an internal memo section.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [["DOC001[P001.B01]"]],
            "citation_integrity_trigger_note": _detailed_integrity_note(),
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
            "overextension_trigger_note": None,
            "overextension_cautions": _detailed_cautions("Overextension"),
        },
    }

    deterministic_pass, reasons, _ = _deterministic_validation(item_payload, {"DOC001[P001.B01]"})

    assert deterministic_pass is False
    assert "missing_overextension_criteria" in reasons


def test_missing_mode_specific_trigger_note_fails_for_precedence():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_D_01",
        "packet_id": "packet_1",
        "target_error_mode": "D",
        "query_id": "q_0003",
        "as_of_date": "2026-02-06",
        "prompt": "Draft an internal memo section.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [["DOC001[P001.B01]"]],
            "citation_integrity_trigger_note": _detailed_integrity_note(),
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
            "precedence_trigger_note": None,
            "precedence_cautions": _detailed_cautions("Precedence"),
        },
    }

    deterministic_pass, reasons, _ = _deterministic_validation(item_payload, {"DOC001[P001.B01]"})

    assert deterministic_pass is False
    assert "missing_precedence_criteria" in reasons


def test_missing_mode_specific_cautions_fail_for_precedence():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_D_02",
        "packet_id": "packet_1",
        "target_error_mode": "D",
        "query_id": "q_0004",
        "as_of_date": "2026-02-06",
        "prompt": "Draft an internal memo section.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [["DOC001[P001.B01]"]],
            "citation_integrity_trigger_note": _detailed_integrity_note(),
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
            "precedence_trigger_note": "Precedence rationale is present.",
            "precedence_cautions": [],
        },
    }

    deterministic_pass, reasons, _ = _deterministic_validation(item_payload, {"DOC001[P001.B01]"})

    assert deterministic_pass is False
    assert "missing_precedence_cautions" in reasons


def test_mode_a_allows_empty_expected_citations_when_other_checks_present():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_A_11",
        "packet_id": "packet_1",
        "target_error_mode": "A",
        "query_id": "q_0011",
        "as_of_date": "2026-02-06",
        "prompt": "Ask for absent authority and require explicit no-support answer.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [],
            "citation_integrity_trigger_note": _detailed_integrity_note(),
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
        },
    }

    deterministic_pass, reasons, details = _deterministic_validation(item_payload, {"DOC001[P001.B01]"})

    assert deterministic_pass is True
    assert reasons == []
    assert details["expected_citation_count"] == 0


def test_mode_c_requires_non_empty_expected_citations():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_C_11",
        "packet_id": "packet_1",
        "target_error_mode": "C",
        "query_id": "q_0011",
        "as_of_date": "2026-02-06",
        "prompt": "Draft an internal memo section.",
        "scenario_facts": ["Use packet-only authorities."],
        "grading_contract": {
            "expected_citation_groups": [],
            "citation_integrity_trigger_note": _detailed_integrity_note(),
            "citation_integrity_cautions": _detailed_cautions("Integrity"),
            "overextension_trigger_note": "Detailed overextension rationale for pass/fail.",
            "overextension_cautions": _detailed_cautions("Overextension"),
        },
    }

    with pytest.raises(ValidationError):
        _deterministic_validation(item_payload, {"DOC001[P001.B01]"})
