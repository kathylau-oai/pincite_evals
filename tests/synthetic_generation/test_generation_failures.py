import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.config import load_config  # noqa: E402
from pincite_evals.synthetic_generation.pipeline import _generate_one_item, _load_mode_prompts, _load_verifier_prompts  # noqa: E402
from pincite_evals.synthetic_generation.structured_outputs import GeneratedSyntheticItemOutput  # noqa: E402


def _make_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
packet_id: packet_1
output_root: {(tmp_path / 'results').as_posix()}
dataset_root: {(tmp_path / 'datasets').as_posix()}
dry_run: false
parallelism:
  max_retries: 3
""".strip(),
        encoding="utf-8",
    )
    return load_config(config_path)


def test_generate_one_item_drops_candidate_after_retries(monkeypatch, tmp_path):
    config = _make_config(tmp_path)

    def always_invalid_parse(*args, **kwargs):
        try:
            GeneratedSyntheticItemOutput.model_validate({"scenario_facts": 123})
        except ValidationError as validation_error:
            raise validation_error
        raise AssertionError("Expected ValidationError for malformed generated output payload.")

    monkeypatch.setattr(
        "pincite_evals.synthetic_generation.pipeline._call_openai_parse",
        always_invalid_parse,
    )

    candidate, metric = _generate_one_item(
        mode_name="overextension",
        request_id="req_test_0001",
        default_citation_token="DOC001[P001.B01]",
        packet_corpus_text='<DOCUMENT id="DOC001"><BLOCK id="DOC001.P001.B01">Example</BLOCK></DOCUMENT>\n',
        config=config,
        item_index=1,
        generation_traces_dir=tmp_path / "traces",
        openai_client=object(),
    )

    assert candidate is None
    assert metric["status"].startswith("generation_failed_after_3_attempts:invalid_structured_output")


def test_load_verifier_prompts_renders_item_citations_as_block_ids():
    item_payload = {
        "schema_version": "v1",
        "item_id": "packet_1_A_01",
        "packet_id": "packet_1",
        "target_error_mode": "A",
        "query_id": "q_fak_0001",
        "as_of_date": "2026-02-06",
        "user_query": "Draft a memo section.",
        "scenario_facts": ["Use packet-only sources."],
        "grading_contract": {
            "expected_citation_groups": [["DOC001[P001.B01]", "DOC002.P002.B03"]],
            "citation_integrity_trigger_note": "Detailed note.",
            "citation_integrity_cautions": ["Detailed caution."],
            "overextension_trigger_note": None,
            "overextension_cautions": [],
            "precedence_trigger_note": None,
            "precedence_cautions": [],
        },
    }

    _, user_prompt = _load_verifier_prompts(
        item_payload=item_payload,
        packet_corpus_text='<DOCUMENT id="DOC001"><BLOCK id="DOC001.P001.B01">Example</BLOCK></DOCUMENT>\n',
    )

    assert "DOC001.P001.B01" in user_prompt
    assert "DOC002.P002.B03" in user_prompt
    assert "DOC001[P001.B01]" not in user_prompt


def test_load_mode_prompts_adds_lawyer_realistic_query_style_guidance():
    _, user_prompt = _load_mode_prompts(
        mode_name="overextension",
        packet_id="packet_1",
        as_of_date="2026-02-06",
        item_index=1,
        packet_corpus_text='<DOCUMENT id="DOC001"><BLOCK id="DOC001.P001.B01">Example</BLOCK></DOCUMENT>\n',
    )

    assert "Lawyer-realistic query style guide" in user_prompt
    assert "Need a quick memo for the partner: can we remove this case to federal court under CAFA?" in user_prompt
    assert "Can you write a concise memo on Article III standing risks for this privacy class action" in user_prompt
