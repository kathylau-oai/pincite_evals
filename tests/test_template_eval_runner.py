import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pincite_evals.template_eval_runner import (  # noqa: E402
    ModelConfig,
    _build_response_request,
    _compute_distribution_stats,
    _prepare_dataset,
    _template_grade,
)


def test_build_response_request_includes_temperature_when_reasoning_none():
    model_config = ModelConfig(
        name="baseline",
        model="gpt-5.2",
        reasoning_effort="none",
        temperature=0.2,
        system_prompt="You are a test assistant.",
    )

    request = _build_response_request(model_config, "Test prompt")

    assert request["model"] == "gpt-5.2"
    assert request["reasoning"] == {"effort": "none"}
    assert request["temperature"] == 0.2


def test_build_response_request_omits_temperature_when_reasoning_enabled():
    model_config = ModelConfig(
        name="reasoning_config",
        model="gpt-5.2",
        reasoning_effort="high",
        temperature=0.7,
        system_prompt="You are a test assistant.",
    )

    request = _build_response_request(model_config, "Test prompt")

    assert request["reasoning"] == {"effort": "high"}
    assert "temperature" not in request


def test_template_grade_exact_match_and_not_graded_cases():
    pass_grade = _template_grade("Final answer", "Final answer")
    assert pass_grade["grade_label"] == "pass"
    assert pass_grade["grade_passed"] is True
    assert pass_grade["grade_score"] == 1.0

    fail_grade = _template_grade("Different answer", "Expected answer")
    assert fail_grade["grade_label"] == "fail"
    assert fail_grade["grade_passed"] is False
    assert fail_grade["grade_score"] == 0.0

    skipped_grade = _template_grade("Any answer", None)
    assert skipped_grade["grade_label"] == "not_graded"
    assert skipped_grade["grade_passed"] is None
    assert skipped_grade["grade_score"] is None


def test_distribution_stats_computes_expected_values():
    values = pd.Series([1, 2, 3, 4, 5])
    stats = _compute_distribution_stats(values, "latency")

    assert stats["latency_count"] == 5
    assert stats["latency_avg"] == 3.0
    assert stats["latency_p50"] == 3.0
    assert stats["latency_p90"] == 4.6
    assert stats["latency_p95"] == 4.8
    assert stats["latency_p99"] == 4.96


def test_prepare_dataset_creates_id_and_expected_output_columns(tmp_path):
    input_csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"user_query": ["p1", "p2"]}).to_csv(input_csv_path, index=False)

    args = Namespace(
        input_csv=str(input_csv_path),
        user_query_column="user_query",
        id_column="item_id",
        expected_output_column="expected_output",
        max_samples=None,
    )

    dataset = _prepare_dataset(args)

    assert "item_id" in dataset.columns
    assert "expected_output" in dataset.columns
    assert "user_query" in dataset.columns
    assert dataset["item_id"].tolist() == ["row_0", "row_1"]


def test_prepare_dataset_accepts_legacy_prompt_column(tmp_path):
    input_csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"prompt": ["p1"]}).to_csv(input_csv_path, index=False)

    args = Namespace(
        input_csv=str(input_csv_path),
        user_query_column="user_query",
        id_column="item_id",
        expected_output_column="expected_output",
        max_samples=None,
    )

    dataset = _prepare_dataset(args)
    assert dataset["user_query"].tolist() == ["p1"]
