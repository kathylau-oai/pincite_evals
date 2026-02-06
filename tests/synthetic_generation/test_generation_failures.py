import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.synthetic_generation.config import load_config  # noqa: E402
from pincite_evals.synthetic_generation.pipeline import _generate_one_item  # noqa: E402
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
