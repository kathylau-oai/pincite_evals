import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


@dataclass
class ModelConfig:
    name: str
    model: str
    reasoning_effort: str
    temperature: float
    system_prompt: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Template evaluation runner that supports parallel execution across "
            "multiple model configurations using the OpenAI Responses API."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--output-root", default="results", help="Root directory for run outputs.")
    parser.add_argument("--experiment-name", default="template_eval", help="Name for this experiment run.")

    parser.add_argument("--id-column", default="item_id", help="CSV column for unique item IDs.")
    parser.add_argument("--prompt-column", default="prompt", help="CSV column that contains model prompts.")
    parser.add_argument(
        "--expected-output-column",
        default="expected_output",
        help="Optional CSV column for expected/reference output used by template grading.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of rows to evaluate (useful for quick tests).",
    )

    parser.add_argument(
        "--model-config-file",
        default=None,
        help="Path to a JSON file containing a list of model configuration objects.",
    )
    parser.add_argument(
        "--model-config-json",
        action="append",
        default=[],
        help="Inline JSON object for a model config. Repeat this flag for multiple configs.",
    )

    parser.add_argument("--config-name", default="default", help="Default single-config name.")
    parser.add_argument("--model", default="gpt-5.2", help="Default single-config model.")
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="Default single-config reasoning effort (none, low, medium, high).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Default single-config temperature.")
    parser.add_argument(
        "--system-prompt",
        default="You are a careful legal drafting assistant.",
        help="Default single-config system prompt.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional file path for the default single-config system prompt.",
    )
    parser.add_argument(
        "--max-model-workers",
        type=int,
        default=4,
        help="Max parallel workers across model configurations.",
    )
    parser.add_argument(
        "--max-item-workers",
        type=int,
        default=8,
        help="Max parallel workers per model configuration across dataset rows.",
    )

    parser.add_argument("--retry-attempts", type=int, default=4, help="Max retries for retriable API errors.")
    parser.add_argument(
        "--retry-min-wait-seconds",
        type=float,
        default=1.0,
        help="Minimum backoff for retries.",
    )
    parser.add_argument(
        "--retry-max-wait-seconds",
        type=float,
        default=20.0,
        help="Maximum backoff for retries.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout per request to the OpenAI API.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and produce deterministic placeholder outputs for pipeline validation.",
    )
    return parser.parse_args()


def _load_system_prompt(*, system_prompt: str, system_prompt_file: Optional[str]) -> str:
    if system_prompt_file is None:
        return system_prompt
    return Path(system_prompt_file).read_text(encoding="utf-8").strip()


def _model_config_from_dict(config_data: Dict[str, Any], fallback_prompt: str) -> ModelConfig:
    config_name = str(config_data.get("name", "unnamed_config")).strip()
    model_name = str(config_data.get("model", "gpt-5.2")).strip()
    reasoning_effort = str(config_data.get("reasoning_effort", "none")).strip()
    temperature_value = float(config_data.get("temperature", 0.0))
    system_prompt = str(config_data.get("system_prompt", fallback_prompt)).strip()

    if not config_name:
        raise ValueError("Each model config must have a non-empty name.")
    if not model_name:
        raise ValueError(f"Model config '{config_name}' has an empty model field.")

    return ModelConfig(
        name=config_name,
        model=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature_value,
        system_prompt=system_prompt,
    )


def _load_model_configs(args: argparse.Namespace, default_prompt: str) -> List[ModelConfig]:
    config_objects: List[Dict[str, Any]] = []

    if args.model_config_file:
        config_file_path = Path(args.model_config_file)
        file_data = json.loads(config_file_path.read_text(encoding="utf-8"))
        if isinstance(file_data, dict) and "model_configs" in file_data:
            file_data = file_data["model_configs"]
        if not isinstance(file_data, list):
            raise ValueError("--model-config-file JSON must be a list or contain a 'model_configs' list.")
        for model_config in file_data:
            if not isinstance(model_config, dict):
                raise ValueError("Each model config in --model-config-file must be a JSON object.")
            config_objects.append(model_config)

    for config_json in args.model_config_json:
        inline_data = json.loads(config_json)
        if not isinstance(inline_data, dict):
            raise ValueError("Each --model-config-json value must be a JSON object.")
        config_objects.append(inline_data)

    if not config_objects:
        config_objects.append(
            {
                "name": args.config_name,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "temperature": args.temperature,
                "system_prompt": default_prompt,
            }
        )

    model_configs = [_model_config_from_dict(config, default_prompt) for config in config_objects]

    unique_names = set()
    for model_config in model_configs:
        if model_config.name in unique_names:
            raise ValueError(f"Duplicate model config name detected: {model_config.name}")
        unique_names.add(model_config.name)

    return model_configs


def _sanitize_name(raw_value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value)
    safe_value = safe_value.strip("._")
    return safe_value or "item"


def _safe_float(raw_value: Any) -> Optional[float]:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _compute_distribution_stats(values: pd.Series, prefix: str) -> Dict[str, Any]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_avg": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }

    return {
        f"{prefix}_count": int(numeric_values.shape[0]),
        f"{prefix}_avg": float(numeric_values.mean()),
        f"{prefix}_p50": float(numeric_values.quantile(0.50)),
        f"{prefix}_p90": float(numeric_values.quantile(0.90)),
        f"{prefix}_p95": float(numeric_values.quantile(0.95)),
        f"{prefix}_p99": float(numeric_values.quantile(0.99)),
    }


def _build_input_profile(dataset: pd.DataFrame, id_column: str, prompt_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    column_profile_rows: List[Dict[str, Any]] = []
    for column_name in dataset.columns:
        series = dataset[column_name]
        column_profile_rows.append(
            {
                "column": column_name,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=False)),
            }
        )

    summary_rows: List[Dict[str, Any]] = [
        {"metric": "row_count", "value": int(dataset.shape[0])},
        {"metric": "column_count", "value": int(dataset.shape[1])},
    ]

    if id_column in dataset.columns:
        duplicate_count = int(dataset[id_column].duplicated(keep=False).sum())
        summary_rows.append({"metric": "duplicate_id_count", "value": duplicate_count})

    if prompt_column in dataset.columns:
        prompt_series = dataset[prompt_column].astype(str)
        empty_prompt_count = int(prompt_series.str.strip().eq("").sum())
        summary_rows.append({"metric": "empty_prompt_count", "value": empty_prompt_count})

        prompt_length_series = prompt_series.str.len()
        prompt_length_stats = _compute_distribution_stats(prompt_length_series, "prompt_length_chars")
        for key, value in prompt_length_stats.items():
            summary_rows.append({"metric": key, "value": value})

    return pd.DataFrame(summary_rows), pd.DataFrame(column_profile_rows)


def _build_response_request(model_config: ModelConfig, prompt_text: str) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": model_config.model,
        "input": [
            {"role": "system", "content": model_config.system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "reasoning": {"effort": model_config.reasoning_effort},
    }
    if model_config.reasoning_effort == "none":
        request["temperature"] = model_config.temperature
    return request


def _template_grade(model_output: str, expected_output: Any) -> Dict[str, Any]:
    if expected_output is None or (isinstance(expected_output, float) and pd.isna(expected_output)):
        return {
            "grade_label": "not_graded",
            "grade_passed": None,
            "grade_score": None,
            "grade_notes": "No expected output provided in dataset; template grading skipped.",
        }

    expected_text = str(expected_output).strip()
    if not expected_text:
        return {
            "grade_label": "not_graded",
            "grade_passed": None,
            "grade_score": None,
            "grade_notes": "Expected output is empty; template grading skipped.",
        }

    predicted_text = model_output.strip()
    is_exact_match = predicted_text == expected_text
    return {
        "grade_label": "pass" if is_exact_match else "fail",
        "grade_passed": bool(is_exact_match),
        "grade_score": 1.0 if is_exact_match else 0.0,
        "grade_notes": "Template exact-match grading against expected output.",
    }


def _build_retrying_caller(*, client: OpenAI, args: argparse.Namespace):
    @retry(
        reraise=True,
        stop=stop_after_attempt(args.retry_attempts),
        wait=wait_exponential(
            min=args.retry_min_wait_seconds,
            max=args.retry_max_wait_seconds,
            multiplier=args.retry_min_wait_seconds,
        ),
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        ),
    )
    def _call(request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        delta_timestamps: List[float] = []
        output_fragments: List[str] = []

        with client.responses.stream(**request, timeout=args.request_timeout_seconds) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    now = time.perf_counter()
                    delta_timestamps.append(now)
                    output_fragments.append(event.delta)
            final_response = stream.get_final_response()

        end_time = time.perf_counter()

        output_text = "".join(output_fragments)
        if not output_text:
            output_text = final_response.output_text or ""

        incomplete_reason: Optional[str] = None
        if final_response.incomplete_details is not None:
            incomplete_reason = final_response.incomplete_details.reason

        if final_response.status == "incomplete" and not output_text.strip():
            raise RuntimeError(
                "LLM response was incomplete and returned empty output_text. "
                "Review response.incomplete_details and rerun."
            )

        inter_token_latencies: List[float] = []
        for index in range(1, len(delta_timestamps)):
            inter_token_latencies.append(delta_timestamps[index] - delta_timestamps[index - 1])

        usage_dict = None
        if final_response.usage is not None:
            usage_dict = final_response.usage.model_dump(mode="json")

        return {
            "response_id": final_response.id,
            "status": final_response.status,
            "incomplete_reason": incomplete_reason,
            "output_text": output_text,
            "usage": usage_dict,
            "latency_seconds": end_time - start_time,
            "ttft_seconds": (delta_timestamps[0] - start_time) if delta_timestamps else None,
            "inter_token_latencies": inter_token_latencies,
            "raw_response": final_response.model_dump(mode="json"),
        }

    return _call


def _extract_usage_fields(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    output_token_details = usage.get("output_tokens_details")
    reasoning_tokens = None
    if isinstance(output_token_details, dict):
        reasoning_tokens = output_token_details.get("reasoning_tokens")

    return {
        "input_tokens": _safe_float(usage.get("input_tokens")),
        "output_tokens": _safe_float(usage.get("output_tokens")),
        "reasoning_tokens": _safe_float(reasoning_tokens),
        "total_tokens": _safe_float(usage.get("total_tokens")),
    }


def _evaluate_single_row(
    *,
    row: pd.Series,
    source_row_index: int,
    model_config: ModelConfig,
    call_model,
    raw_response_dir: Path,
    prompt_column: str,
    id_column: str,
    expected_output_column: str,
    dry_run: bool,
) -> Tuple[Dict[str, Any], List[float]]:
    item_id = str(row[id_column])
    prompt_text = str(row[prompt_column])
    expected_output = row[expected_output_column]

    request = _build_response_request(model_config, prompt_text)

    if dry_run:
        output_text = f"[DRY_RUN::{model_config.name}] {prompt_text[:80]}"
        grade_data = _template_grade(output_text, expected_output)
        result_row = {
            "source_row_index": source_row_index,
            "model_config": model_config.name,
            "item_id": item_id,
            "prompt": prompt_text,
            "expected_output": expected_output,
            "model_output": output_text,
            "response_id": None,
            "response_status": "completed",
            "incomplete_reason": None,
            "request_error": None,
            "latency_seconds": 0.0,
            "ttft_seconds": None,
            "inter_token_latency_avg_seconds": None,
            "inter_token_event_count": 0,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            **grade_data,
        }
        return result_row, []

    try:
        model_result = call_model(request)
    except (OpenAIError, RuntimeError, TypeError, ValueError) as error:
        failed_row = {
            "source_row_index": source_row_index,
            "model_config": model_config.name,
            "item_id": item_id,
            "prompt": prompt_text,
            "expected_output": expected_output,
            "model_output": "",
            "response_id": None,
            "response_status": "error",
            "incomplete_reason": None,
            "request_error": str(error),
            "latency_seconds": None,
            "ttft_seconds": None,
            "inter_token_latency_avg_seconds": None,
            "inter_token_event_count": 0,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            "grade_label": "not_graded",
            "grade_passed": None,
            "grade_score": None,
            "grade_notes": "Request failed before grading.",
        }
        return failed_row, []

    usage_fields = _extract_usage_fields(model_result["usage"])
    inter_token_latencies = model_result["inter_token_latencies"]

    grade_data = _template_grade(model_result["output_text"], expected_output)

    file_stem = _sanitize_name(f"{source_row_index}_{item_id}")
    raw_path = raw_response_dir / f"{file_stem}.json"
    raw_payload = {
        "request": request,
        "result": {
            "response_id": model_result["response_id"],
            "status": model_result["status"],
            "incomplete_reason": model_result["incomplete_reason"],
            "output_text": model_result["output_text"],
            "latency_seconds": model_result["latency_seconds"],
            "ttft_seconds": model_result["ttft_seconds"],
            "inter_token_latencies": inter_token_latencies,
            "usage": model_result["usage"],
        },
        "raw_response": model_result["raw_response"],
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    result_row = {
        "source_row_index": source_row_index,
        "model_config": model_config.name,
        "item_id": item_id,
        "prompt": prompt_text,
        "expected_output": expected_output,
        "model_output": model_result["output_text"],
        "response_id": model_result["response_id"],
        "response_status": model_result["status"],
        "incomplete_reason": model_result["incomplete_reason"],
        "request_error": None,
        "latency_seconds": model_result["latency_seconds"],
        "ttft_seconds": model_result["ttft_seconds"],
        "inter_token_latency_avg_seconds": (
            float(pd.Series(inter_token_latencies).mean()) if inter_token_latencies else None
        ),
        "inter_token_event_count": len(inter_token_latencies),
        "input_tokens": usage_fields["input_tokens"],
        "output_tokens": usage_fields["output_tokens"],
        "reasoning_tokens": usage_fields["reasoning_tokens"],
        "total_tokens": usage_fields["total_tokens"],
        **grade_data,
    }
    return result_row, inter_token_latencies


def _evaluate_model_config(
    *,
    dataset: pd.DataFrame,
    model_config: ModelConfig,
    args: argparse.Namespace,
    run_dir: Path,
    progress_position: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_response_dir = run_dir / "raw_responses" / _sanitize_name(model_config.name)
    raw_response_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        client = None
        call_model = None
    else:
        client = OpenAI()
        call_model = _build_retrying_caller(client=client, args=args)

    row_results: List[Dict[str, Any]] = []
    inter_token_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_item_workers)) as item_pool:
        future_to_index = {}
        for source_row_index, row in dataset.iterrows():
            future = item_pool.submit(
                _evaluate_single_row,
                row=row,
                source_row_index=int(source_row_index),
                model_config=model_config,
                call_model=call_model,
                raw_response_dir=raw_response_dir,
                prompt_column=args.prompt_column,
                id_column=args.id_column,
                expected_output_column=args.expected_output_column,
                dry_run=args.dry_run,
            )
            future_to_index[future] = int(source_row_index)

        progress_bar = tqdm(
            total=len(future_to_index),
            desc=f"{model_config.name}",
            position=progress_position,
            leave=True,
        )
        for future in as_completed(future_to_index):
            row_result, inter_token_latencies = future.result()
            row_results.append(row_result)
            for inter_token_latency in inter_token_latencies:
                inter_token_rows.append(
                    {
                        "model_config": model_config.name,
                        "source_row_index": row_result["source_row_index"],
                        "inter_token_latency_seconds": inter_token_latency,
                    }
                )
            progress_bar.update(1)
        progress_bar.close()

    row_result_frame = pd.DataFrame(row_results).sort_values(
        by=["source_row_index", "item_id"], kind="mergesort"
    )

    quality_summary = _summarize_quality_metrics(row_result_frame)
    latency_summary = _summarize_latency_metrics(row_result_frame, inter_token_rows)
    token_summary = _summarize_token_metrics(row_result_frame)

    summary_row = {
        "model_config": model_config.name,
        **quality_summary,
        **latency_summary,
        **token_summary,
    }
    return row_result_frame, [summary_row], inter_token_rows


def _summarize_quality_metrics(result_frame: pd.DataFrame) -> Dict[str, Any]:
    graded_mask = result_frame["grade_passed"].notna()
    passed_mask = result_frame["grade_passed"] == True

    graded_count = int(graded_mask.sum())
    passed_count = int((graded_mask & passed_mask).sum())
    pass_rate = float(passed_count / graded_count) if graded_count else None

    return {
        "item_count": int(result_frame.shape[0]),
        "graded_item_count": graded_count,
        "graded_pass_count": passed_count,
        "graded_pass_rate": pass_rate,
        "not_graded_count": int((result_frame["grade_label"] == "not_graded").sum()),
        "error_count": int((result_frame["response_status"] == "error").sum()),
        "incomplete_count": int((result_frame["response_status"] == "incomplete").sum()),
    }


def _summarize_latency_metrics(result_frame: pd.DataFrame, inter_token_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latency_stats = _compute_distribution_stats(result_frame["latency_seconds"], "latency_seconds")
    ttft_stats = _compute_distribution_stats(result_frame["ttft_seconds"], "ttft_seconds")

    inter_token_series = pd.Series([row["inter_token_latency_seconds"] for row in inter_token_rows])
    inter_token_stats = _compute_distribution_stats(inter_token_series, "inter_token_latency_seconds")

    return {
        **latency_stats,
        **ttft_stats,
        **inter_token_stats,
    }


def _summarize_token_metrics(result_frame: pd.DataFrame) -> Dict[str, Any]:
    input_token_stats = _compute_distribution_stats(result_frame["input_tokens"], "input_tokens")
    output_token_stats = _compute_distribution_stats(result_frame["output_tokens"], "output_tokens")
    reasoning_token_stats = _compute_distribution_stats(result_frame["reasoning_tokens"], "reasoning_tokens")
    total_token_stats = _compute_distribution_stats(result_frame["total_tokens"], "total_tokens")

    return {
        **input_token_stats,
        **output_token_stats,
        **reasoning_token_stats,
        **total_token_stats,
    }


def _reshape_metric_table(
    summary_frame: pd.DataFrame,
    metric_prefixes: List[str],
    metric_group: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, summary_row in summary_frame.iterrows():
        model_config = summary_row["model_config"]
        for metric_prefix in metric_prefixes:
            metric_data = {
                "model_config": model_config,
                "metric_group": metric_group,
                "metric_name": metric_prefix,
                "count": summary_row.get(f"{metric_prefix}_count"),
                "avg": summary_row.get(f"{metric_prefix}_avg"),
                "p50": summary_row.get(f"{metric_prefix}_p50"),
                "p90": summary_row.get(f"{metric_prefix}_p90"),
                "p95": summary_row.get(f"{metric_prefix}_p95"),
                "p99": summary_row.get(f"{metric_prefix}_p99"),
            }
            rows.append(metric_data)
    return pd.DataFrame(rows)


def _prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    dataset = pd.read_csv(args.input_csv)

    if args.prompt_column not in dataset.columns:
        raise ValueError(f"Prompt column '{args.prompt_column}' does not exist in {args.input_csv}.")

    if args.id_column not in dataset.columns:
        dataset[args.id_column] = [f"row_{index}" for index in range(dataset.shape[0])]

    if args.expected_output_column not in dataset.columns:
        dataset[args.expected_output_column] = pd.NA

    dataset = dataset.copy()
    dataset[args.id_column] = dataset[args.id_column].astype(str)
    dataset[args.prompt_column] = dataset[args.prompt_column].fillna("").astype(str)

    if args.max_samples is not None:
        dataset = dataset.head(args.max_samples).copy()

    dataset.reset_index(drop=True, inplace=True)
    return dataset


def _save_run_configuration(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    model_configs: List[ModelConfig],
) -> None:
    config_payload = {
        "created_at": datetime.now().isoformat(),
        "cli_args": vars(args),
        "model_configs": [asdict(config) for config in model_configs],
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()

    default_prompt = _load_system_prompt(
        system_prompt=args.system_prompt,
        system_prompt_file=args.system_prompt_file,
    )

    model_configs = _load_model_configs(args, default_prompt)
    model_worker_count = min(max(1, args.max_model_workers), max(1, len(model_configs)))

    dataset = _prepare_dataset(args)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{_sanitize_name(args.experiment_name)}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_summary_frame, input_column_profile_frame = _build_input_profile(
        dataset,
        id_column=args.id_column,
        prompt_column=args.prompt_column,
    )
    input_summary_frame.to_csv(run_dir / "input_summary.csv", index=False)
    input_column_profile_frame.to_csv(run_dir / "input_column_profile.csv", index=False)

    _save_run_configuration(run_dir=run_dir, args=args, model_configs=model_configs)

    model_to_position = {config.name: index for index, config in enumerate(model_configs)}

    all_result_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    all_inter_token_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=model_worker_count) as model_pool:
        future_to_name = {}
        for model_config in model_configs:
            future = model_pool.submit(
                _evaluate_model_config,
                dataset=dataset,
                model_config=model_config,
                args=args,
                run_dir=run_dir,
                progress_position=model_to_position[model_config.name],
            )
            future_to_name[future] = model_config.name

        for future in as_completed(future_to_name):
            result_frame, summary_row_list, inter_token_rows = future.result()
            all_result_frames.append(result_frame)
            summary_rows.extend(summary_row_list)
            all_inter_token_rows.extend(inter_token_rows)

    all_results_frame = pd.concat(all_result_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values(by="model_config", kind="mergesort")

    all_results_frame.to_csv(run_dir / "predictions_and_grades.csv", index=False)
    summary_frame.to_csv(run_dir / "metrics_summary.csv", index=False)

    latency_metrics_frame = _reshape_metric_table(
        summary_frame,
        ["latency_seconds", "ttft_seconds", "inter_token_latency_seconds"],
        "latency",
    )
    token_metrics_frame = _reshape_metric_table(
        summary_frame,
        ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"],
        "tokens",
    )

    latency_metrics_frame.to_csv(run_dir / "latency_metrics.csv", index=False)
    token_metrics_frame.to_csv(run_dir / "token_metrics.csv", index=False)

    pd.DataFrame(all_inter_token_rows).to_csv(run_dir / "inter_token_latency_events.csv", index=False)

    print(f"Run complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
