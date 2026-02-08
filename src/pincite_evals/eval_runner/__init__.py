import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
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

from pincite_evals.citations import extract_excerpt_citations
from pincite_evals.graders import (
    CitationFidelityLLMJudgeGrader,
    CitationOverextensionLLMJudgeGrader,
    ExpectedCitationPresenceGrader,
    PrecedenceLLMJudgeGrader,
)
from pincite_evals.openai_model_capabilities import supports_reasoning_effort
from pincite_evals.prompt_templates import render_template_file
from pincite_evals.synthetic_generation.schema import normalize_citation_token


REQUIRED_SYNTHETIC_BASE_COLUMNS = [
    "item_id",
    "packet_id",
    "target_error_mode",
    "query_id",
    "as_of_date",
    "scenario_facts",
    "grading_contract",
]
USER_QUERY_COLUMN = "user_query"
PRIORITY_SERVICE_TIER = "priority"

BLOCK_PATTERN = re.compile(
    r'<BLOCK id="(?P<block_id>DOC\d{3}\.P\d{3}\.B\d{2})">\s*(?P<block_text>.*?)\s*</BLOCK>',
    re.DOTALL,
)

DEFAULT_DRAFTING_SYSTEM_PROMPT = (
    "You are a careful legal drafting assistant writing an internal litigation memo in a closed-world setting. "
    "Use only the packet authorities provided in the user message. Do not rely on outside law, cases, statutes, "
    "or treatises. Use pinpoint citations as dotted packet block IDs in this exact format: DOC###.P###.B##. "
    "Use the exact `<BLOCK id=\"DOC###.P###.B##\">` identifiers from the packet; do not use bracket notation. "
    "If authority is missing from the packet, say so explicitly instead of fabricating support."
)

MODE_TO_EXTRA_GRADER = {
    "A": "citation_fidelity_llm_judge",
    "C": "citation_overextension_llm_judge",
    "D": "precedence_llm_judge",
}


@dataclass
class ModelConfig:
    name: str
    model: str
    reasoning_effort: str
    temperature: float
    system_prompt: str


@dataclass(frozen=True)
class PacketResources:
    packet_corpus_text: str
    block_text_by_token: Dict[str, str]
    document_count: int
    block_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Packet-aware eval runner for legal memo drafting. Runs model inference in parallel "
            "across model configs and rows, then runs graders in parallel."
        )
    )

    parser.add_argument("--input-csv", default=None, help="Optional single CSV path. If omitted, --input-glob is used.")
    parser.add_argument(
        "--input-glob",
        default="data/datasets/packet_*/synthetic_items.csv",
        help="Glob for synthetic item CSV files when --input-csv is not provided.",
    )
    parser.add_argument("--packet-root", default="data/case_law_packets", help="Root folder for packet corpora.")

    parser.add_argument("--output-root", default="results/experiments", help="Root directory for run outputs.")
    parser.add_argument(
        "--run-id", default=None, help="Optional run ID. Outputs are written to <output-root>/<run_id>."
    )
    parser.add_argument("--experiment-name", default="template_eval", help="Name used when auto-generating run IDs.")
    parser.add_argument(
        "--artifact-level",
        default="standard",
        choices=["standard", "debug"],
        help="Artifact verbosity. standard writes compact final/analysis outputs; debug also writes raw request traces.",
    )

    parser.add_argument("--id-column", default="item_id", help="CSV column for unique item IDs.")
    parser.add_argument(
        "--expected-output-column",
        default="expected_output",
        help="Optional CSV column for exact-match template grading (mostly unused for synthetic items).",
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
        help="Default single-config reasoning effort (none, low, medium, high, xhigh).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Default single-config temperature.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_DRAFTING_SYSTEM_PROMPT,
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

    parser.add_argument("--max-grader-workers", type=int, default=12, help="Max parallel workers for graders.")
    parser.add_argument("--grader-model", default="gpt-5.1", help="Model for LLM-based graders.")
    parser.add_argument(
        "--grader-reasoning-effort",
        default="none",
        help="Reasoning effort for grader LLM calls (should remain none).",
    )
    parser.add_argument("--grader-temperature", type=float, default=0.0, help="Temperature for graders.")

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
    return render_template_file(Path(system_prompt_file), {})


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


def _estimate_inter_token_latency_seconds(
    *, latency_seconds: Optional[float], output_tokens: Optional[float]
) -> Optional[float]:
    if latency_seconds is None or output_tokens is None:
        return None
    if output_tokens <= 0:
        return None
    return float(latency_seconds / output_tokens)


def _build_input_profile(dataset: pd.DataFrame, id_column: str, prompt_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    column_profile_rows: List[Dict[str, Any]] = []
    for column_name in dataset.columns:
        series = dataset[column_name]
        try:
            unique_count = int(series.nunique(dropna=False))
        except TypeError:
            # Some profiling columns store lists/dicts; use a stable JSON string for uniqueness counting.
            unique_count = int(
                series.apply(lambda value: json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)).nunique(dropna=False)
            )
        column_profile_rows.append(
            {
                "column": column_name,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "n_unique": unique_count,
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
        summary_rows.append({"metric": "empty_user_query_count", "value": empty_prompt_count})

        prompt_length_series = prompt_series.str.len()
        prompt_length_stats = _compute_distribution_stats(prompt_length_series, "user_query_length_chars")
        for key, value in prompt_length_stats.items():
            summary_rows.append({"metric": key, "value": value})

    if "packet_id" in dataset.columns:
        summary_rows.append({"metric": "packet_count", "value": int(dataset["packet_id"].astype(str).nunique())})

    if "scenario_facts_parse_error" in dataset.columns:
        summary_rows.append(
            {
                "metric": "scenario_facts_parse_error_count",
                "value": int(dataset["scenario_facts_parse_error"].notna().sum()),
            }
        )

    if "grading_contract_parse_error" in dataset.columns:
        summary_rows.append(
            {
                "metric": "grading_contract_parse_error_count",
                "value": int(dataset["grading_contract_parse_error"].notna().sum()),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(column_profile_rows)


def _build_response_request(model_config: ModelConfig, prompt_text: str) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": model_config.model,
        "service_tier": PRIORITY_SERVICE_TIER,
        "input": [
            {"role": "system", "content": model_config.system_prompt},
            {"role": "user", "content": prompt_text},
        ],
    }

    if supports_reasoning_effort(model_config.model):
        request["reasoning"] = {"effort": model_config.reasoning_effort}
        if model_config.reasoning_effort == "none":
            request["temperature"] = model_config.temperature
    else:
        # Many non-reasoning models (e.g. gpt-4o, gpt-4.1) reject `reasoning.effort`.
        # In that case, we always fall back to temperature-based control.
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
        first_output_delta_timestamp: Optional[float] = None
        output_fragments: List[str] = []

        with client.responses.stream(**request, timeout=args.request_timeout_seconds) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    now = time.perf_counter()
                    if first_output_delta_timestamp is None:
                        first_output_delta_timestamp = now
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
            "ttft_seconds": (
                first_output_delta_timestamp - start_time
                if first_output_delta_timestamp is not None
                else None
            ),
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


def _parse_serialized_list(raw_value: Any) -> Tuple[List[str], Optional[str]]:
    if isinstance(raw_value, list):
        cleaned_items = [str(item).strip() for item in raw_value if str(item).strip()]
        return cleaned_items, None

    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return [], None

    text_value = str(raw_value).strip()
    if not text_value:
        return [], None

    try:
        parsed = ast.literal_eval(text_value)
    except (ValueError, SyntaxError) as error:
        return [], str(error)

    if isinstance(parsed, list):
        cleaned_items = [str(item).strip() for item in parsed if str(item).strip()]
        return cleaned_items, None

    return [text_value], "Expected a serialized list but got a different type."


def _parse_serialized_dict(raw_value: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(raw_value, dict):
        return raw_value, None

    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return {}, None

    text_value = str(raw_value).strip()
    if not text_value:
        return {}, None

    try:
        parsed = ast.literal_eval(text_value)
    except (ValueError, SyntaxError) as error:
        return {}, str(error)

    if isinstance(parsed, dict):
        return parsed, None

    return {}, "Expected a serialized dict but got a different type."


def _normalize_expected_citation_groups(raw_groups: Any) -> List[List[str]]:
    if not isinstance(raw_groups, list):
        return []

    normalized_groups: List[List[str]] = []
    for raw_group in raw_groups:
        if isinstance(raw_group, str):
            group_candidates = [raw_group]
        elif isinstance(raw_group, list):
            group_candidates = raw_group
        else:
            continue

        normalized_group: List[str] = []
        for raw_token in group_candidates:
            token_text = str(raw_token).strip()
            if not token_text:
                continue
            try:
                normalized_group.append(normalize_citation_token(token_text))
            except ValueError:
                continue

        if normalized_group:
            normalized_groups.append(normalized_group)

    return normalized_groups


def _flatten_expected_tokens(expected_groups: List[List[str]]) -> List[str]:
    ordered_tokens: List[str] = []
    seen_tokens: set[str] = set()
    for citation_group in expected_groups:
        for citation_token in citation_group:
            if citation_token in seen_tokens:
                continue
            seen_tokens.add(citation_token)
            ordered_tokens.append(citation_token)
    return ordered_tokens


def _parse_structured_field(raw_value: Any) -> Any:
    if isinstance(raw_value, (dict, list)):
        return raw_value

    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None

    text_value = str(raw_value).strip()
    if not text_value:
        return None

    try:
        return json.loads(text_value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text_value)
    except (ValueError, SyntaxError):
        return None


def _load_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.input_csv:
        input_path = Path(args.input_csv)
        if not input_path.exists():
            raise ValueError(f"Input CSV not found: {input_path}")
        return [input_path]

    globbed_paths = sorted(Path(".").glob(args.input_glob))
    if not globbed_paths:
        raise ValueError(f"No CSV files matched --input-glob pattern: {args.input_glob}")
    return globbed_paths


def _prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    args_dict = vars(args)
    input_paths: List[Path]

    input_csv_value = args_dict.get("input_csv")
    input_glob_value = args_dict.get("input_glob")

    if input_csv_value is not None and input_glob_value is None:
        input_path = Path(str(input_csv_value))
        if not input_path.exists():
            raise ValueError(f"Input CSV not found: {input_path}")
        input_paths = [input_path]
    else:
        input_paths = _load_input_paths(args)

    dataset_frames: List[pd.DataFrame] = []
    for input_path in input_paths:
        frame = pd.read_csv(input_path)
        frame["source_dataset_path"] = str(input_path)
        dataset_frames.append(frame)

    dataset = pd.concat(dataset_frames, ignore_index=True)

    prompt_column_name = USER_QUERY_COLUMN
    id_column_name = str(args_dict.get("id_column", "item_id"))
    expected_output_column_name = str(args_dict.get("expected_output_column", "expected_output"))

    if prompt_column_name not in dataset.columns:
        raise ValueError(f"User query column '{prompt_column_name}' does not exist in the input dataset.")

    if id_column_name not in dataset.columns:
        dataset[id_column_name] = [f"row_{index}" for index in range(dataset.shape[0])]

    if expected_output_column_name not in dataset.columns:
        dataset[expected_output_column_name] = pd.NA

    dataset = dataset.copy()
    dataset[id_column_name] = dataset[id_column_name].astype(str)
    dataset[prompt_column_name] = dataset[prompt_column_name].fillna("").astype(str)

    if "packet_id" not in dataset.columns:
        dataset["packet_id"] = ""

    scenario_parse_results = dataset.get("scenario_facts", pd.Series([None] * dataset.shape[0])).apply(_parse_serialized_list)
    dataset["scenario_facts_parsed"] = scenario_parse_results.apply(lambda parsed: parsed[0])
    dataset["scenario_facts_parse_error"] = scenario_parse_results.apply(lambda parsed: parsed[1])

    grading_contract_results = dataset.get("grading_contract", pd.Series([None] * dataset.shape[0])).apply(_parse_serialized_dict)
    dataset["grading_contract_parsed"] = grading_contract_results.apply(lambda parsed: parsed[0])
    dataset["grading_contract_parse_error"] = grading_contract_results.apply(lambda parsed: parsed[1])

    normalized_groups: List[List[List[str]]] = []
    normalized_contracts: List[Dict[str, Any]] = []
    for raw_contract in dataset["grading_contract_parsed"].tolist():
        contract_copy = dict(raw_contract) if isinstance(raw_contract, dict) else {}
        expected_groups = _normalize_expected_citation_groups(contract_copy.get("expected_citation_groups", []))
        contract_copy["expected_citation_groups"] = expected_groups
        normalized_groups.append(expected_groups)
        normalized_contracts.append(contract_copy)

    dataset["expected_citation_groups"] = normalized_groups
    dataset["grading_contract_parsed"] = normalized_contracts

    if args.max_samples is not None:
        dataset = dataset.head(args.max_samples).copy()

    dataset.reset_index(drop=True, inplace=True)
    return dataset


def _validate_required_synthetic_columns(dataset: pd.DataFrame) -> None:
    missing_columns = [column_name for column_name in REQUIRED_SYNTHETIC_BASE_COLUMNS if column_name not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Synthetic dataset missing required columns: {', '.join(missing_columns)}")

    if USER_QUERY_COLUMN not in dataset.columns:
        raise ValueError(f"Synthetic dataset is missing user query column '{USER_QUERY_COLUMN}'.")



def _extract_block_text_map(annotated_text: str) -> Dict[str, str]:
    block_text_by_token: Dict[str, str] = {}
    for block_match in BLOCK_PATTERN.finditer(annotated_text):
        block_id = block_match.group("block_id")
        block_text = block_match.group("block_text").strip()
        try:
            citation_token = normalize_citation_token(block_id)
        except ValueError:
            continue

        if citation_token not in block_text_by_token:
            block_text_by_token[citation_token] = block_text

    return block_text_by_token


def _load_packet_resources(packet_id: str, packet_root: Path) -> PacketResources:
    packet_folder = packet_root / packet_id
    manifest_path = packet_folder / "packet_manifest.csv"
    if not manifest_path.exists():
        raise ValueError(f"packet_manifest.csv not found for packet '{packet_id}': {manifest_path}")

    manifest_frame = pd.read_csv(manifest_path)
    if "doc_id" not in manifest_frame.columns or "source_order" not in manifest_frame.columns:
        raise ValueError(f"packet_manifest.csv for '{packet_id}' is missing required columns (doc_id, source_order).")

    manifest_sorted = manifest_frame.sort_values("source_order")
    if "parse_status" in manifest_sorted.columns:
        manifest_sorted = manifest_sorted[manifest_sorted["parse_status"] == "parsed"]

    if manifest_sorted.empty:
        raise ValueError(f"Packet '{packet_id}' has no parsed documents in packet_manifest.csv.")

    packet_lines: List[str] = []
    merged_block_map: Dict[str, str] = {}

    for _, manifest_row in manifest_sorted.iterrows():
        doc_id = str(manifest_row["doc_id"]).strip()
        source_filename = str(manifest_row.get("source_filename", "")).strip()
        annotated_path = packet_folder / "text" / f"{doc_id}.annotated.txt"

        if not annotated_path.exists():
            raise ValueError(f"Missing annotated text for packet '{packet_id}', doc '{doc_id}': {annotated_path}")

        annotated_text = annotated_path.read_text(encoding="utf-8").strip()

        packet_lines.append(f'<DOCUMENT id="{doc_id}" source_filename="{source_filename}">')
        packet_lines.append(annotated_text)
        packet_lines.append("</DOCUMENT>")
        packet_lines.append("")

        block_map = _extract_block_text_map(annotated_text)
        for citation_token, block_text in block_map.items():
            if citation_token not in merged_block_map:
                merged_block_map[citation_token] = block_text

    packet_corpus_text = "\n".join(packet_lines).strip() + "\n"

    return PacketResources(
        packet_corpus_text=packet_corpus_text,
        block_text_by_token=merged_block_map,
        document_count=int(manifest_sorted.shape[0]),
        block_count=int(len(merged_block_map)),
    )


def _build_packet_resource_map(dataset: pd.DataFrame, packet_root: Path) -> Dict[str, PacketResources]:
    packet_resource_map: Dict[str, PacketResources] = {}
    packet_ids = sorted(
        {
            str(packet_id).strip()
            for packet_id in dataset["packet_id"].tolist()
            if str(packet_id).strip()
        }
    )

    for packet_id in packet_ids:
        packet_resource_map[packet_id] = _load_packet_resources(packet_id, packet_root)

    return packet_resource_map


def _build_drafting_user_prompt(
    *,
    source_user_query: str,
    scenario_facts: List[str],
    packet_corpus_text: str,
) -> str:
    scenario_lines = [f"- {fact}" for fact in scenario_facts if str(fact).strip()]
    if not scenario_lines:
        scenario_lines = ["- No additional scenario facts were provided."]

    guidance = [
        "Output requirements:",
        "1) Draft as an internal legal memo.",
        "2) Use only packet authorities and cite using dotted packet block IDs: DOC###.P###.B##.",
        "3) If authority is missing, say so clearly and do not fabricate citations.",
        "4) Keep legal claims faithful to source scope and qualifiers.",
    ]

    sections = [
        "Drafting task:",
        source_user_query.strip(),
        "",
        "Scenario facts:",
        "\n".join(scenario_lines),
        "",
        "\n".join(guidance),
        "",
        "Annotated packet corpus:",
        packet_corpus_text.strip(),
    ]
    return "\n".join(sections).strip()


def _evaluate_single_row(
    *,
    row: pd.Series,
    source_row_index: int,
    model_config: ModelConfig,
    call_model,
    raw_response_dir: Optional[Path],
    prompt_column: str,
    id_column: str,
    expected_output_column: str,
    dry_run: bool,
) -> Tuple[Dict[str, Any], List[float]]:
    item_id = str(row[id_column])
    source_user_query = str(row[prompt_column])
    expected_output = row[expected_output_column]

    packet_corpus_text = str(row.get("packet_corpus_text", ""))
    scenario_facts = row.get("scenario_facts_parsed", [])
    if not isinstance(scenario_facts, list):
        scenario_facts = []

    rendered_user_prompt = _build_drafting_user_prompt(
        source_user_query=source_user_query,
        scenario_facts=[str(fact) for fact in scenario_facts],
        packet_corpus_text=packet_corpus_text,
    )

    request = _build_response_request(model_config, rendered_user_prompt)

    # Most output columns are identical across dry-run, error, and success paths.
    # Keep them in one place so we don't drift schemas across branches.
    base_row = {
        "source_row_index": source_row_index,
        "model_config": model_config.name,
        "item_id": item_id,
        "packet_id": str(row.get("packet_id", "")),
        "target_error_mode": str(row.get("target_error_mode", "")),
        "query_id": str(row.get("query_id", "")),
        "as_of_date": str(row.get("as_of_date", "")),
        "source_dataset_path": str(row.get("source_dataset_path", "")),
        "source_user_query": source_user_query,
        "rendered_user_prompt": rendered_user_prompt,
        "scenario_facts_json": json.dumps(scenario_facts, ensure_ascii=True),
        "grading_contract_json": json.dumps(row.get("grading_contract_parsed", {}), ensure_ascii=True),
        "expected_citation_groups_json": json.dumps(row.get("expected_citation_groups", []), ensure_ascii=True),
        "expected_output": expected_output,
    }

    if dry_run:
        output_text = (
            f"[DRY_RUN::{model_config.name}] Memo draft placeholder for {item_id}. "
            "No API call was made."
        )
        grade_data = _template_grade(output_text, expected_output)
        result_row = {
            **base_row,
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
            **base_row,
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
    inter_token_latency_estimate_seconds = _estimate_inter_token_latency_seconds(
        latency_seconds=model_result["latency_seconds"],
        output_tokens=usage_fields["output_tokens"],
    )
    inter_token_latencies = (
        [inter_token_latency_estimate_seconds]
        if inter_token_latency_estimate_seconds is not None
        else []
    )
    output_token_count = (
        int(usage_fields["output_tokens"])
        if usage_fields["output_tokens"] is not None and usage_fields["output_tokens"] > 0
        else 0
    )

    grade_data = _template_grade(model_result["output_text"], expected_output)

    if raw_response_dir is not None:
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
        **base_row,
        "model_output": model_result["output_text"],
        "response_id": model_result["response_id"],
        "response_status": model_result["status"],
        "incomplete_reason": model_result["incomplete_reason"],
        "request_error": None,
        "latency_seconds": model_result["latency_seconds"],
        "ttft_seconds": model_result["ttft_seconds"],
        "inter_token_latency_avg_seconds": inter_token_latency_estimate_seconds,
        # This denominator is used for the e2e/token estimate above.
        "inter_token_event_count": output_token_count,
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
    raw_response_dir: Optional[Path] = None
    if args.artifact_level == "debug":
        raw_response_dir = run_dir / "debug" / "raw_responses" / _sanitize_name(model_config.name)
        raw_response_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
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
                prompt_column=USER_QUERY_COLUMN,
                id_column=args.id_column,
                expected_output_column=args.expected_output_column,
                dry_run=args.dry_run,
            )
            future_to_index[future] = int(source_row_index)

        progress_bar = tqdm(
            total=len(future_to_index),
            desc=f"model:{model_config.name}",
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


def _extract_predicted_citation_tokens(output_text: str) -> List[str]:
    ordered_tokens: List[str] = []
    seen_tokens: set[str] = set()

    for citation in extract_excerpt_citations(output_text):
        citation_token = citation.raw
        try:
            citation_token = normalize_citation_token(citation_token)
        except ValueError:
            continue

        if citation_token in seen_tokens:
            continue

        seen_tokens.add(citation_token)
        ordered_tokens.append(citation_token)

    return ordered_tokens


def _build_citation_fidelity_items(
    *,
    predicted_citation_tokens: List[str],
    expected_citation_groups: List[List[str]],
    block_text_by_token: Dict[str, str],
) -> List[Dict[str, Any]]:
    expected_tokens = set(_flatten_expected_tokens(expected_citation_groups))
    fidelity_items: List[Dict[str, Any]] = []

    for citation_token in predicted_citation_tokens:
        fidelity_items.append(
            {
                "citation_token": citation_token,
                "exists_in_packet": citation_token in block_text_by_token,
                "expected_for_item": citation_token in expected_tokens,
                "canonical_excerpt": block_text_by_token.get(citation_token),
            }
        )

    return fidelity_items


def _build_grader_context(row: Dict[str, Any], block_text_by_packet: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    grading_contract = row.get("grading_contract_parsed")
    if not isinstance(grading_contract, dict):
        parsed_contract = _parse_structured_field(row.get("grading_contract_json"))
        grading_contract = parsed_contract if isinstance(parsed_contract, dict) else {}

    scenario_facts = row.get("scenario_facts_parsed")
    if not isinstance(scenario_facts, list):
        parsed_scenario_facts = _parse_structured_field(row.get("scenario_facts_json"))
        scenario_facts = parsed_scenario_facts if isinstance(parsed_scenario_facts, list) else []

    expected_citation_groups_raw = row.get("expected_citation_groups")
    if isinstance(expected_citation_groups_raw, list):
        expected_citation_groups = _normalize_expected_citation_groups(expected_citation_groups_raw)
    else:
        parsed_expected_groups = _parse_structured_field(row.get("expected_citation_groups_json"))
        if isinstance(parsed_expected_groups, list):
            expected_citation_groups = _normalize_expected_citation_groups(parsed_expected_groups)
        else:
            expected_citation_groups = _normalize_expected_citation_groups(
                grading_contract.get("expected_citation_groups", [])
            )

    model_output = str(row.get("model_output", ""))
    predicted_tokens = _extract_predicted_citation_tokens(model_output)
    target_error_mode = str(row.get("target_error_mode", "")).strip().upper()

    packet_id = str(row.get("packet_id", ""))
    block_map = block_text_by_packet.get(packet_id, {})

    # Mode A items intentionally omit required citation groups when missing authority is the point of the test.
    allow_unexpected_citations_when_no_expected_groups = (
        target_error_mode == "A" and len(expected_citation_groups) == 0
    )

    context: Dict[str, Any] = {
        "expected_citation_groups": expected_citation_groups,
        "allow_unexpected_citations_when_no_expected_groups": allow_unexpected_citations_when_no_expected_groups,
        "test_case_context": {
            "item_id": str(row.get("item_id", "")),
            "packet_id": packet_id,
            "target_error_mode": str(row.get("target_error_mode", "")),
            "query_id": str(row.get("query_id", "")),
            "as_of_date": str(row.get("as_of_date", "")),
            "scenario_facts": scenario_facts,
        },
        "citation_fidelity_note": grading_contract.get("citation_integrity_trigger_note"),
        "citation_fidelity_items": _build_citation_fidelity_items(
            predicted_citation_tokens=predicted_tokens,
            expected_citation_groups=expected_citation_groups,
            block_text_by_token=block_map,
        ),
        "overextension_trigger_note": grading_contract.get("overextension_trigger_note"),
        "overextension_cautions": grading_contract.get("overextension_cautions", []),
        "precedence_trigger_note": grading_contract.get("precedence_trigger_note"),
        "precedence_cautions": grading_contract.get("precedence_cautions", []),
        "authority_graph": grading_contract.get("authority_graph"),
    }
    return context


def _select_graders_for_mode(target_error_mode: str) -> List[str]:
    selected_graders = ["expected_citation_presence"]
    mode_token = str(target_error_mode).strip().upper()
    extra_grader = MODE_TO_EXTRA_GRADER.get(mode_token)
    if extra_grader:
        selected_graders.append(extra_grader)
    return selected_graders


def _build_grader_registry(args: argparse.Namespace) -> Dict[str, Any]:
    grader_registry: Dict[str, Any] = {
        "expected_citation_presence": ExpectedCitationPresenceGrader(),
    }

    if args.dry_run:
        return grader_registry

    shared_client = OpenAI()
    grader_registry["citation_fidelity_llm_judge"] = CitationFidelityLLMJudgeGrader(
        model=args.grader_model,
        reasoning_effort=args.grader_reasoning_effort,
        temperature=args.grader_temperature,
        client=shared_client,
    )
    grader_registry["citation_overextension_llm_judge"] = CitationOverextensionLLMJudgeGrader(
        model=args.grader_model,
        reasoning_effort=args.grader_reasoning_effort,
        temperature=args.grader_temperature,
        client=shared_client,
    )
    grader_registry["precedence_llm_judge"] = PrecedenceLLMJudgeGrader(
        model=args.grader_model,
        reasoning_effort=args.grader_reasoning_effort,
        temperature=args.grader_temperature,
        client=shared_client,
    )
    return grader_registry


def _extract_score_from_grader_details(details: Dict[str, Any]) -> Optional[float]:
    if "score" in details:
        return _safe_float(details.get("score"))
    if "overall_score" in details:
        return _safe_float(details.get("overall_score"))
    return None


def _extract_label_from_grader_details(details: Dict[str, Any]) -> Optional[str]:
    for key in ["label", "grade_label"]:
        if key in details and str(details[key]).strip():
            return str(details[key]).strip()
    return None


def _run_single_grader(
    *,
    prediction_row: Dict[str, Any],
    grader_name: str,
    grader_registry: Dict[str, Any],
    block_text_by_packet: Dict[str, Dict[str, str]],
    dry_run: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start_time = time.perf_counter()

    base_row = {
        "model_config": str(prediction_row.get("model_config", "")),
        "source_row_index": int(prediction_row.get("source_row_index", -1)),
        "item_id": str(prediction_row.get("item_id", "")),
        "packet_id": str(prediction_row.get("packet_id", "")),
        "target_error_mode": str(prediction_row.get("target_error_mode", "")),
        "query_id": str(prediction_row.get("query_id", "")),
        "as_of_date": str(prediction_row.get("as_of_date", "")),
        "grader_name": grader_name,
    }

    if str(prediction_row.get("response_status", "")) == "error":
        return (
            {
                **base_row,
                "grader_status": "skipped_model_error",
                "grader_passed": None,
                "grader_score": None,
                "grader_label": None,
                "grader_error": "Model output unavailable due to model request error.",
                "grader_latency_seconds": float(time.perf_counter() - start_time),
                "grader_input_tokens": None,
                "grader_output_tokens": None,
                "grader_reasoning_tokens": None,
                "grader_total_tokens": None,
                "grader_details_json": json.dumps({}, ensure_ascii=True),
            },
            {},
        )

    if grader_name != "expected_citation_presence" and dry_run:
        return (
            {
                **base_row,
                "grader_status": "skipped_dry_run",
                "grader_passed": None,
                "grader_score": None,
                "grader_label": None,
                "grader_error": None,
                "grader_latency_seconds": float(time.perf_counter() - start_time),
                "grader_input_tokens": None,
                "grader_output_tokens": None,
                "grader_reasoning_tokens": None,
                "grader_total_tokens": None,
                "grader_details_json": json.dumps({}, ensure_ascii=True),
            },
            {},
        )

    grader = grader_registry[grader_name]
    context = _build_grader_context(prediction_row, block_text_by_packet)

    if grader_name == "citation_fidelity_llm_judge" and not context["citation_fidelity_items"]:
        details = {
            "score": 1.0,
            "label": "no_citations_predicted",
            "reason": "No citation tokens were present in model output; no hallucinated citation detected.",
            "passed": True,
            "usage": None,
        }
        latency_seconds = float(time.perf_counter() - start_time)
        return (
            {
                **base_row,
                "grader_status": "completed",
                "grader_passed": True,
                "grader_score": 1.0,
                "grader_label": "no_citations_predicted",
                "grader_error": None,
                "grader_latency_seconds": latency_seconds,
                "grader_input_tokens": None,
                "grader_output_tokens": None,
                "grader_reasoning_tokens": None,
                "grader_total_tokens": None,
                "grader_details_json": json.dumps(details, ensure_ascii=True),
            },
            {
                "user_query": prediction_row.get("source_user_query", prediction_row.get("source_prompt")),
                "model_output": prediction_row.get("model_output"),
                "context": context,
                "details": details,
            },
        )

    try:
        grade_result = grader.grade(
            prompt=str(prediction_row.get("source_user_query", prediction_row.get("source_prompt", ""))),
            output=str(prediction_row.get("model_output", "")),
            context=context,
        )
    except (OpenAIError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        latency_seconds = float(time.perf_counter() - start_time)
        return (
            {
                **base_row,
                "grader_status": "error",
                "grader_passed": None,
                "grader_score": None,
                "grader_label": None,
                "grader_error": str(error),
                "grader_latency_seconds": latency_seconds,
                "grader_input_tokens": None,
                "grader_output_tokens": None,
                "grader_reasoning_tokens": None,
                "grader_total_tokens": None,
                "grader_details_json": json.dumps({}, ensure_ascii=True),
            },
            {
                "user_query": prediction_row.get("source_user_query", prediction_row.get("source_prompt")),
                "model_output": prediction_row.get("model_output"),
                "context": context,
                "error": str(error),
            },
        )

    latency_seconds = float(time.perf_counter() - start_time)
    details = grade_result.details if isinstance(grade_result.details, dict) else {}

    usage_fields = _extract_usage_fields(details.get("usage"))

    grader_row = {
        **base_row,
        "grader_status": "completed",
        "grader_passed": bool(grade_result.passed),
        "grader_score": _extract_score_from_grader_details(details),
        "grader_label": _extract_label_from_grader_details(details),
        "grader_error": None,
        "grader_latency_seconds": latency_seconds,
        "grader_input_tokens": usage_fields["input_tokens"],
        "grader_output_tokens": usage_fields["output_tokens"],
        "grader_reasoning_tokens": usage_fields["reasoning_tokens"],
        "grader_total_tokens": usage_fields["total_tokens"],
        "grader_details_json": json.dumps(details, ensure_ascii=True),
    }

    raw_payload = {
        "user_query": prediction_row.get("source_user_query", prediction_row.get("source_prompt")),
        "model_output": prediction_row.get("model_output"),
        "context": context,
        "details": details,
    }
    return grader_row, raw_payload


def _run_graders(
    *,
    predictions_frame: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    block_text_by_packet: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    grader_registry = _build_grader_registry(args)

    raw_grader_dir: Optional[Path] = None
    if args.artifact_level == "debug":
        raw_grader_dir = run_dir / "debug" / "raw_grader_responses"
        raw_grader_dir.mkdir(parents=True, exist_ok=True)

    prediction_records = predictions_frame.to_dict(orient="records")

    grader_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_grader_workers)) as grader_pool:
        futures = {}
        for prediction_row in prediction_records:
            selected_graders = _select_graders_for_mode(str(prediction_row.get("target_error_mode", "")))
            for grader_name in selected_graders:
                future = grader_pool.submit(
                    _run_single_grader,
                    prediction_row=prediction_row,
                    grader_name=grader_name,
                    grader_registry=grader_registry,
                    block_text_by_packet=block_text_by_packet,
                    dry_run=args.dry_run,
                )
                futures[future] = {
                    "model_config": str(prediction_row.get("model_config", "")),
                    "item_id": str(prediction_row.get("item_id", "")),
                    "grader_name": grader_name,
                    "source_row_index": int(prediction_row.get("source_row_index", -1)),
                }

        progress_bar = tqdm(total=len(futures), desc="graders", leave=True)
        for future in as_completed(futures):
            record_context = futures[future]
            grader_row, raw_payload = future.result()
            grader_rows.append(grader_row)

            if raw_grader_dir is not None:
                raw_file_stem = _sanitize_name(
                    f"{record_context['source_row_index']}_{record_context['item_id']}_{record_context['model_config']}_{record_context['grader_name']}"
                )
                raw_path = raw_grader_dir / f"{raw_file_stem}.json"
                raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
            progress_bar.update(1)
        progress_bar.close()

    if not grader_rows:
        return pd.DataFrame()

    grader_frame = pd.DataFrame(grader_rows).sort_values(
        by=["model_config", "source_row_index", "grader_name"], kind="mergesort"
    )
    return grader_frame


def _summarize_grader_metrics(grader_frame: pd.DataFrame) -> pd.DataFrame:
    if grader_frame.empty:
        return pd.DataFrame(
            [
                {
                    "model_config": "none",
                    "grader_name": "none",
                    "target_error_mode": "none",
                    "grader_request_count": 0,
                    "grader_completed_count": 0,
                }
            ]
        )

    summary_rows: List[Dict[str, Any]] = []
    grouped = grader_frame.groupby(["model_config", "grader_name", "target_error_mode"], dropna=False)

    for (model_config, grader_name, target_error_mode), rows in grouped:
        completed_rows = rows[rows["grader_status"] == "completed"]
        completed_passed = completed_rows[completed_rows["grader_passed"] == True]

        pass_rate = None
        if not completed_rows.empty:
            pass_rate = float(completed_passed.shape[0] / completed_rows.shape[0])

        summary_row = {
            "model_config": model_config,
            "grader_name": grader_name,
            "target_error_mode": target_error_mode,
            "grader_request_count": int(rows.shape[0]),
            "grader_completed_count": int(completed_rows.shape[0]),
            "grader_pass_count": int(completed_passed.shape[0]),
            "grader_pass_rate": pass_rate,
            "grader_error_count": int((rows["grader_status"] == "error").sum()),
            "grader_skipped_count": int(rows["grader_status"].str.startswith("skipped").sum()),
        }
        summary_row.update(_compute_distribution_stats(rows["grader_latency_seconds"], "grader_latency_seconds"))
        summary_row.update(_compute_distribution_stats(rows["grader_input_tokens"], "grader_input_tokens"))
        summary_row.update(_compute_distribution_stats(rows["grader_output_tokens"], "grader_output_tokens"))
        summary_row.update(_compute_distribution_stats(rows["grader_reasoning_tokens"], "grader_reasoning_tokens"))
        summary_row.update(_compute_distribution_stats(rows["grader_total_tokens"], "grader_total_tokens"))
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def _build_slice_metrics(grader_frame: pd.DataFrame) -> pd.DataFrame:
    if grader_frame.empty:
        return pd.DataFrame(
            columns=[
                "packet_id",
                "target_error_mode",
                "model_config",
                "grader_name",
                "completed_count",
                "pass_count",
                "pass_rate",
            ]
        )

    rows: List[Dict[str, Any]] = []
    grouped = grader_frame.groupby(["packet_id", "target_error_mode", "model_config", "grader_name"], dropna=False)

    for (packet_id, target_error_mode, model_config, grader_name), group_rows in grouped:
        completed_rows = group_rows[group_rows["grader_status"] == "completed"]
        pass_count = int((completed_rows["grader_passed"] == True).sum())
        completed_count = int(completed_rows.shape[0])
        pass_rate = float(pass_count / completed_count) if completed_count else None

        rows.append(
            {
                "packet_id": packet_id,
                "target_error_mode": target_error_mode,
                "model_config": model_config,
                "grader_name": grader_name,
                "completed_count": completed_count,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
            }
        )

    return pd.DataFrame(rows)


def _build_predictions_and_grades(predictions_frame: pd.DataFrame, grader_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty:
        return predictions_frame

    if grader_frame.empty:
        output = predictions_frame.copy()
        output["overall_required_graders_passed"] = None
        return output

    key_columns = ["model_config", "source_row_index", "item_id"]

    grader_status_summary = (
        grader_frame.groupby(key_columns, dropna=False)
        .apply(
            lambda group_rows: bool(
                (group_rows["grader_status"] == "completed").all() and (group_rows["grader_passed"] == True).all()
            )
        )
        .reset_index(name="overall_required_graders_passed")
    )

    grader_pass_wide = (
        grader_frame.pivot_table(
            index=key_columns,
            columns="grader_name",
            values="grader_passed",
            aggfunc="first",
        )
        .rename(columns=lambda grader_name: f"grader_{grader_name}_passed")
        .reset_index()
    )

    grader_score_wide = (
        grader_frame.pivot_table(
            index=key_columns,
            columns="grader_name",
            values="grader_score",
            aggfunc="first",
        )
        .rename(columns=lambda grader_name: f"grader_{grader_name}_score")
        .reset_index()
    )

    merged = predictions_frame.merge(grader_status_summary, on=key_columns, how="left")
    merged = merged.merge(grader_pass_wide, on=key_columns, how="left")
    merged = merged.merge(grader_score_wide, on=key_columns, how="left")
    return merged


def _build_error_table(predictions_frame: pd.DataFrame, grader_frame: pd.DataFrame) -> pd.DataFrame:
    model_error_rows: List[Dict[str, Any]] = []
    prediction_errors = predictions_frame[predictions_frame["response_status"] != "completed"]
    for _, row in prediction_errors.iterrows():
        model_error_rows.append(
            {
                "stage": "model",
                "model_config": row.get("model_config"),
                "source_row_index": row.get("source_row_index"),
                "item_id": row.get("item_id"),
                "packet_id": row.get("packet_id"),
                "target_error_mode": row.get("target_error_mode"),
                "component_name": "drafting_model",
                "status": row.get("response_status"),
                "error_message": row.get("request_error"),
                "incomplete_reason": row.get("incomplete_reason"),
            }
        )

    grader_error_rows: List[Dict[str, Any]] = []
    if not grader_frame.empty:
        grader_issues = grader_frame[grader_frame["grader_status"] != "completed"]
        for _, row in grader_issues.iterrows():
            grader_error_rows.append(
                {
                    "stage": "grader",
                    "model_config": row.get("model_config"),
                    "source_row_index": row.get("source_row_index"),
                    "item_id": row.get("item_id"),
                    "packet_id": row.get("packet_id"),
                    "target_error_mode": row.get("target_error_mode"),
                    "component_name": row.get("grader_name"),
                    "status": row.get("grader_status"),
                    "error_message": row.get("grader_error"),
                    "incomplete_reason": None,
                }
            )

    if not model_error_rows and not grader_error_rows:
        return pd.DataFrame(
            columns=[
                "stage",
                "model_config",
                "source_row_index",
                "item_id",
                "packet_id",
                "target_error_mode",
                "component_name",
                "status",
                "error_message",
                "incomplete_reason",
            ]
        )

    error_frame = pd.DataFrame(model_error_rows + grader_error_rows)
    return error_frame.sort_values(
        by=["stage", "source_row_index", "item_id", "component_name"], kind="mergesort"
    ).reset_index(drop=True)


def _aggregate_grader_performance(grader_frame: pd.DataFrame) -> pd.DataFrame:
    if grader_frame.empty:
        return pd.DataFrame(columns=["model_config", "grader_name", "completed_count", "pass_count", "pass_rate"])

    completed = grader_frame[grader_frame["grader_status"] == "completed"]
    if completed.empty:
        return pd.DataFrame(columns=["model_config", "grader_name", "completed_count", "pass_count", "pass_rate"])

    grouped = (
        completed.groupby(["model_config", "grader_name"], dropna=False)["grader_passed"]
        .agg(completed_count="count", pass_count="sum")
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["completed_count"]
    return grouped.sort_values(by=["model_config", "grader_name"], kind="mergesort")


def _save_metric_plots(
    *,
    charts_dir: Path,
    summary_frame: pd.DataFrame,
    grader_performance_frame: pd.DataFrame,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    pass_rate_frame = summary_frame[["model_config", "graded_pass_rate"]].copy()
    pass_rate_frame["graded_pass_rate"] = pass_rate_frame["graded_pass_rate"].fillna(0.0)
    pass_rate_figure = px.bar(
        pass_rate_frame,
        x="model_config",
        y="graded_pass_rate",
        text=pass_rate_frame["graded_pass_rate"].map(lambda value: f"{value:.1%}"),
        title="Template Graded Pass Rate by Model",
        labels={"model_config": "Model Config", "graded_pass_rate": "Pass Rate"},
    )
    pass_rate_figure.update_layout(yaxis=dict(range=[0, 1]), template="plotly_white")
    pass_rate_figure.write_image(charts_dir / "metrics_pass_rate_by_model.png", width=1100, height=700, scale=2)

    latency_frame = summary_frame[
        ["model_config", "latency_seconds_p50", "latency_seconds_p90", "latency_seconds_p95"]
    ].melt(
        id_vars="model_config",
        var_name="latency_metric",
        value_name="seconds",
    )
    latency_figure = px.bar(
        latency_frame,
        x="model_config",
        y="seconds",
        color="latency_metric",
        barmode="group",
        title="Latency Metrics by Model",
        labels={"model_config": "Model Config", "seconds": "Seconds", "latency_metric": "Metric"},
    )
    latency_figure.update_layout(template="plotly_white")
    latency_figure.write_image(charts_dir / "metrics_latency_by_model.png", width=1200, height=700, scale=2)

    token_frame = summary_frame[
        ["model_config", "output_tokens_avg", "output_tokens_p90", "total_tokens_avg"]
    ].melt(
        id_vars="model_config",
        var_name="token_metric",
        value_name="tokens",
    )
    token_figure = px.bar(
        token_frame,
        x="model_config",
        y="tokens",
        color="token_metric",
        barmode="group",
        title="Token Metrics by Model",
        labels={"model_config": "Model Config", "tokens": "Tokens", "token_metric": "Metric"},
    )
    token_figure.update_layout(template="plotly_white")
    token_figure.write_image(charts_dir / "metrics_tokens_by_model.png", width=1200, height=700, scale=2)

    if grader_performance_frame.empty:
        return

    grader_plot = grader_performance_frame.copy()
    grader_plot["pass_rate"] = grader_plot["pass_rate"].fillna(0.0)
    grader_figure = px.bar(
        grader_plot,
        x="grader_name",
        y="pass_rate",
        color="model_config",
        barmode="group",
        text=grader_plot["pass_rate"].map(lambda value: f"{value:.1%}"),
        title="Grader Performance Pass Rate",
        labels={"grader_name": "Grader", "pass_rate": "Pass Rate", "model_config": "Model Config"},
    )
    grader_figure.update_layout(yaxis=dict(range=[0, 1]), template="plotly_white")
    grader_figure.write_image(charts_dir / "grader_performance_by_grader.png", width=1400, height=800, scale=2)


def _write_failure_mode_report(
    *,
    analysis_dir: Path,
    predictions_and_grades_frame: pd.DataFrame,
    grader_performance_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    total_items = int(predictions_and_grades_frame.shape[0])
    completed_items = int((predictions_and_grades_frame["response_status"] == "completed").sum())

    overall_pass_series = predictions_and_grades_frame["overall_required_graders_passed"].dropna()
    overall_pass_rate = float(overall_pass_series.mean()) if not overall_pass_series.empty else None

    mode_summary = (
        predictions_and_grades_frame.groupby("target_error_mode", dropna=False)["overall_required_graders_passed"]
        .agg(item_count="count", pass_count="sum", pass_rate="mean")
        .reset_index()
        .sort_values("target_error_mode", kind="mergesort")
    )

    report_lines = [
        "# Failure Mode Report",
        "",
        f"- Total items: {total_items}",
        f"- Completed model responses: {completed_items}/{total_items}",
        (
            f"- Overall required-grader pass rate: {overall_pass_rate:.1%}"
            if overall_pass_rate is not None
            else "- Overall required-grader pass rate: n/a"
        ),
        f"- Total non-completed/error rows: {int(error_frame.shape[0])}",
        "",
        "## Pass Rate by Error Mode",
    ]

    for _, row in mode_summary.iterrows():
        pass_rate_text = "n/a" if pd.isna(row["pass_rate"]) else f"{float(row['pass_rate']):.1%}"
        report_lines.append(
            f"- Mode {row['target_error_mode']}: {int(row['pass_count'])}/{int(row['item_count'])} ({pass_rate_text})"
        )

    report_lines.append("")
    report_lines.append("## Grader Pass Rates")
    if grader_performance_frame.empty:
        report_lines.append("- No completed grader rows.")
    else:
        for _, row in grader_performance_frame.iterrows():
            report_lines.append(
                f"- `{row['model_config']}` / `{row['grader_name']}`: "
                f"{int(row['pass_count'])}/{int(row['completed_count'])} ({float(row['pass_rate']):.1%})"
            )

    report_lines.append("")
    report_lines.append("## Error Summary")
    if error_frame.empty:
        report_lines.append("- No model or grader errors.")
    else:
        stage_counts = error_frame.groupby("stage", dropna=False).size().to_dict()
        for stage_name, stage_count in stage_counts.items():
            report_lines.append(f"- {stage_name}: {int(stage_count)}")
        top_rows = error_frame.head(10)
        report_lines.append("- First 10 error rows:")
        for _, row in top_rows.iterrows():
            report_lines.append(
                f"  - [{row['stage']}] item={row['item_id']} component={row['component_name']} status={row['status']} "
                f"error={row['error_message']}"
            )

    (analysis_dir / "failure_mode_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _write_manifest(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    model_configs: List[ModelConfig],
    final_dir: Path,
    analysis_dir: Path,
    predictions_and_grades_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
) -> None:
    completed_items = int((predictions_and_grades_frame["response_status"] == "completed").sum())
    overall_required_pass = predictions_and_grades_frame["overall_required_graders_passed"].dropna()
    payload = {
        "created_at": datetime.now().isoformat(),
        "run_id": run_dir.name,
        "artifact_level": args.artifact_level,
        "cli_args": vars(args),
        "model_configs": [asdict(config) for config in model_configs],
        "summary": {
            "item_count": int(predictions_and_grades_frame.shape[0]),
            "completed_item_count": completed_items,
            "non_completed_item_count": int(predictions_and_grades_frame.shape[0] - completed_items),
            "overall_required_graders_pass_rate": (
                float(overall_required_pass.mean()) if not overall_required_pass.empty else None
            ),
            "error_row_count": int(error_frame.shape[0]),
        },
        "artifacts": {
            "predictions_with_grades_csv": str((final_dir / "predictions_with_grades.csv").relative_to(run_dir)),
            "metrics_summary_csv": str((final_dir / "metrics_summary.csv").relative_to(run_dir)),
            "grader_metrics_summary_csv": str((final_dir / "grader_metrics_summary.csv").relative_to(run_dir)),
            "errors_csv": str((final_dir / "errors.csv").relative_to(run_dir)),
            "failure_mode_report_md": str((analysis_dir / "failure_mode_report.md").relative_to(run_dir)),
            "charts_dir": str((analysis_dir / "charts").relative_to(run_dir)),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return _sanitize_name(args.run_id)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _sanitize_name(f"{args.experiment_name}_{run_timestamp}")


def main() -> None:
    args = _parse_args()

    default_prompt = _load_system_prompt(
        system_prompt=args.system_prompt,
        system_prompt_file=args.system_prompt_file,
    )

    model_configs = _load_model_configs(args, default_prompt)
    model_worker_count = min(max(1, args.max_model_workers), max(1, len(model_configs)))

    dataset = _prepare_dataset(args)
    _validate_required_synthetic_columns(dataset)

    run_id = _build_run_id(args)
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final_dir = run_dir / "final"
    analysis_dir = run_dir / "analysis"
    charts_dir = analysis_dir / "charts"
    final_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    debug_dir: Optional[Path] = None
    if args.artifact_level == "debug":
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    input_summary_frame, input_column_profile_frame = _build_input_profile(
        dataset,
        id_column=args.id_column,
        prompt_column=USER_QUERY_COLUMN,
    )
    if debug_dir is not None:
        input_summary_frame.to_csv(debug_dir / "input_summary.csv", index=False)
        input_column_profile_frame.to_csv(debug_dir / "input_column_profile.csv", index=False)

    packet_resource_map = _build_packet_resource_map(dataset, Path(args.packet_root))

    packet_corpus_by_packet = {
        packet_id: packet_resources.packet_corpus_text
        for packet_id, packet_resources in packet_resource_map.items()
    }

    block_text_by_packet = {
        packet_id: packet_resources.block_text_by_token
        for packet_id, packet_resources in packet_resource_map.items()
    }

    dataset = dataset.copy()
    dataset["packet_corpus_text"] = dataset["packet_id"].map(packet_corpus_by_packet)
    if dataset["packet_corpus_text"].isna().any():
        missing_packet_rows = dataset[dataset["packet_corpus_text"].isna()]
        missing_packets = sorted({str(packet_id) for packet_id in missing_packet_rows["packet_id"].tolist()})
        raise ValueError(f"Missing packet corpus text for packet IDs: {', '.join(missing_packets)}")

    packet_sanity_rows: List[Dict[str, Any]] = []
    for packet_id, packet_resources in packet_resource_map.items():
        packet_sanity_rows.append(
            {
                "packet_id": packet_id,
                "document_count": packet_resources.document_count,
                "block_count": packet_resources.block_count,
                "corpus_char_length": len(packet_resources.packet_corpus_text),
            }
        )
    if debug_dir is not None:
        pd.DataFrame(packet_sanity_rows).to_csv(debug_dir / "packet_input_sanity.csv", index=False)

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

    if debug_dir is not None:
        all_results_frame.to_csv(debug_dir / "predictions.csv", index=False)
        pd.DataFrame(all_inter_token_rows).to_csv(debug_dir / "inter_token_latency_events.csv", index=False)

    grader_frame = _run_graders(
        predictions_frame=all_results_frame,
        args=args,
        run_dir=run_dir,
        block_text_by_packet=block_text_by_packet,
    )
    if debug_dir is not None:
        grader_frame.to_csv(debug_dir / "grader_results.csv", index=False)

    grader_summary_frame = _summarize_grader_metrics(grader_frame)

    predictions_and_grades_frame = _build_predictions_and_grades(all_results_frame, grader_frame)
    error_frame = _build_error_table(all_results_frame, grader_frame)
    grader_performance_frame = _aggregate_grader_performance(grader_frame)

    predictions_and_grades_frame.to_csv(final_dir / "predictions_with_grades.csv", index=False)
    summary_frame.to_csv(final_dir / "metrics_summary.csv", index=False)
    grader_summary_frame.to_csv(final_dir / "grader_metrics_summary.csv", index=False)
    error_frame.to_csv(final_dir / "errors.csv", index=False)

    _save_metric_plots(
        charts_dir=charts_dir,
        summary_frame=summary_frame,
        grader_performance_frame=grader_performance_frame,
    )
    _write_failure_mode_report(
        analysis_dir=analysis_dir,
        predictions_and_grades_frame=predictions_and_grades_frame,
        grader_performance_frame=grader_performance_frame,
        error_frame=error_frame,
    )
    _write_manifest(
        run_dir=run_dir,
        args=args,
        model_configs=model_configs,
        final_dir=final_dir,
        analysis_dir=analysis_dir,
        predictions_and_grades_frame=predictions_and_grades_frame,
        error_frame=error_frame,
    )

    if debug_dir is not None:
        _build_slice_metrics(grader_frame).to_csv(debug_dir / "slice_metrics.csv", index=False)
        _reshape_metric_table(
            summary_frame,
            ["latency_seconds", "ttft_seconds", "inter_token_latency_seconds"],
            "latency",
        ).to_csv(debug_dir / "latency_metrics.csv", index=False)
        _reshape_metric_table(
            summary_frame,
            ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"],
            "tokens",
        ).to_csv(debug_dir / "token_metrics.csv", index=False)
        predictions_and_grades_frame.to_csv(debug_dir / "predictions_and_grades.csv", index=False)

    print(f"Run complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
