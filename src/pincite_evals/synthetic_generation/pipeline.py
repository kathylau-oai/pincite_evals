import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import SyntheticGenerationConfig
from .schema import SyntheticItem, normalize_citation_token
from .structured_outputs import GeneratedSyntheticItemOutput, VerifierOutput


MODE_TO_ERROR = {
    "overextension": "C",
    "precedence": "D",
    "fake_citations": "A",
}
ERROR_TO_MODE = {value: key for key, value in MODE_TO_ERROR.items()}
PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
ParsedModelType = TypeVar("ParsedModelType", bound=BaseModel)


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    metadata_dir: Path
    target_bank_dir: Path
    generation_dir: Path
    generation_candidates_dir: Path
    generation_metrics_dir: Path
    generation_traces_dir: Path
    validation_dir: Path
    validation_metrics_dir: Path
    validation_traces_dir: Path
    selection_dir: Path
    summary_dir: Path


@dataclass(frozen=True)
class PipelineContext:
    config: SyntheticGenerationConfig
    packet_root: Path
    manifest_csv: Path
    blocks_dir: Path
    packet_information_md: Path
    run_paths: RunPaths


@dataclass(frozen=True)
class TargetBankResult:
    target_bank: pd.DataFrame
    packet_corpus_text: str
    sanity_summary: pd.DataFrame


@dataclass(frozen=True)
class GenerationResult:
    candidates_by_mode: dict[str, list[dict[str, Any]]]
    request_metrics: pd.DataFrame


@dataclass(frozen=True)
class ValidationResult:
    accepted_items: list[dict[str, Any]]
    deterministic_pass_items: list[dict[str, Any]]
    deterministic_checks: pd.DataFrame
    llm_reviews: list[dict[str, Any]]
    rejection_log: pd.DataFrame
    request_metrics: pd.DataFrame


@dataclass(frozen=True)
class SelectionResult:
    selected_items: list[dict[str, Any]]
    selection_table: pd.DataFrame
    selection_report_markdown: str


def _compute_distribution_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return {
            f"{prefix}_avg": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }

    return {
        f"{prefix}_avg": float(numeric_values.mean()),
        f"{prefix}_p50": float(numeric_values.quantile(0.50)),
        f"{prefix}_p90": float(numeric_values.quantile(0.90)),
        f"{prefix}_p95": float(numeric_values.quantile(0.95)),
        f"{prefix}_p99": float(numeric_values.quantile(0.99)),
    }


def summarize_request_metrics(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    if metrics_dataframe.empty:
        return pd.DataFrame([{"stage": "none", "request_count": 0, "completed_count": 0, "items_per_minute": None}])

    summary_rows: list[dict[str, Any]] = []
    for stage_name, stage_rows in metrics_dataframe.groupby("stage"):
        completed_rows = stage_rows[stage_rows["status"] == "completed"]
        total_latency_seconds = pd.to_numeric(completed_rows["latency_seconds"], errors="coerce").dropna().sum()
        completed_count = int(completed_rows.shape[0])
        items_per_minute = None
        if total_latency_seconds > 0 and completed_count > 0:
            items_per_minute = float((completed_count / total_latency_seconds) * 60.0)

        summary_row = {
            "stage": stage_name,
            "request_count": int(stage_rows.shape[0]),
            "completed_count": completed_count,
            "items_per_minute": items_per_minute,
        }
        summary_row.update(_compute_distribution_stats(stage_rows["latency_seconds"], "latency_seconds"))
        summary_row.update(_compute_distribution_stats(stage_rows["input_tokens"], "input_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["output_tokens"], "output_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["reasoning_tokens"], "reasoning_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["total_tokens"], "total_tokens"))
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def _augment_with_latency_milliseconds(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    if metrics_dataframe.empty:
        return metrics_dataframe

    augmented_dataframe = metrics_dataframe.copy()
    latency_series = pd.to_numeric(augmented_dataframe["latency_seconds"], errors="coerce")
    augmented_dataframe["latency_milliseconds"] = latency_series * 1000.0
    return augmented_dataframe


def _build_datapoint_timing_table(
    generation_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    generation_columns = [
        "item_id",
        "mode_name",
        "request_id",
        "status",
        "latency_seconds",
        "latency_milliseconds",
    ]
    validation_columns = [
        "item_id",
        "status",
        "latency_seconds",
        "latency_milliseconds",
        "verdict",
    ]
    generation_slice = generation_metrics.reindex(columns=generation_columns).rename(
        columns={
            "status": "generation_status",
            "latency_seconds": "generation_latency_seconds",
            "latency_milliseconds": "generation_latency_milliseconds",
        }
    )
    validation_slice = validation_metrics.reindex(columns=validation_columns).rename(
        columns={
            "status": "validation_status",
            "latency_seconds": "validation_latency_seconds",
            "latency_milliseconds": "validation_latency_milliseconds",
            "verdict": "validation_verdict",
        }
    )
    merged = generation_slice.merge(validation_slice, on="item_id", how="left")
    merged["end_to_end_latency_seconds"] = (
        pd.to_numeric(merged["generation_latency_seconds"], errors="coerce")
        + pd.to_numeric(merged["validation_latency_seconds"], errors="coerce")
    )
    merged["end_to_end_latency_milliseconds"] = merged["end_to_end_latency_seconds"] * 1000.0
    if merged.empty:
        return merged
    return merged.sort_values(["mode_name", "item_id"]).reset_index(drop=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalize_text(raw_text: str) -> str:
    return " ".join(raw_text.split())

def _citation_token_to_xml_block_id(citation_token: str) -> str:
    canonical_token = normalize_citation_token(citation_token)
    token_match = re.fullmatch(r"(DOC\d{3})\[P(\d{3})\.B(\d{2})\]", canonical_token)
    if token_match is None:
        raise ValueError(f"Unable to convert citation token into XML block id: {citation_token}")
    doc_id, page_number, block_number = token_match.groups()
    return f"{doc_id}.P{page_number}.B{block_number}"


def _load_prompt_template(prompt_relative_path: str) -> str:
    prompt_path = PROMPTS_ROOT / prompt_relative_path
    if not prompt_path.exists():
        raise ValueError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _render_prompt_template(prompt_text: str, replacements: dict[str, str]) -> str:
    rendered_prompt = prompt_text
    for token_name, token_value in replacements.items():
        rendered_prompt = rendered_prompt.replace(token_name, token_value)
    return rendered_prompt


def _load_mode_prompts(
    mode_name: str,
    packet_id: str,
    as_of_date: str,
    item_index: int,
    packet_corpus_text: str,
) -> tuple[str, str]:
    replacements = {
        "__PACKET_ID__": packet_id,
        "__AS_OF_DATE__": as_of_date,
        "__ITEM_INDEX__": str(item_index),
        "__PACKET_CORPUS__": packet_corpus_text,
    }
    system_prompt = _render_prompt_template(_load_prompt_template(f"{mode_name}/system.txt"), replacements)
    user_prompt = _render_prompt_template(_load_prompt_template(f"{mode_name}/user.txt"), replacements)
    return system_prompt, user_prompt


def _load_verifier_prompts(item_payload: dict[str, Any]) -> tuple[str, str]:
    system_prompt = _load_prompt_template("verifier/system.txt")
    user_prompt = _render_prompt_template(
        _load_prompt_template("verifier/user.txt"),
        {"__ITEM_JSON__": json.dumps(item_payload, ensure_ascii=True)},
    )
    return system_prompt, user_prompt


def _extract_reasoning_tokens(usage: Any) -> int | None:
    if usage is None or usage.output_tokens_details is None:
        return None
    return usage.output_tokens_details.reasoning_tokens


def _create_run_paths(output_root: Path, packet_id: str, run_id: str | None) -> RunPaths:
    resolved_run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / packet_id / resolved_run_id

    metadata_dir = run_root / "metadata"
    target_bank_dir = run_root / "target_bank"
    generation_dir = run_root / "generation"
    generation_candidates_dir = generation_dir / "candidates"
    generation_metrics_dir = generation_dir / "metrics"
    generation_traces_dir = generation_dir / "traces"
    validation_dir = run_root / "validation"
    validation_metrics_dir = validation_dir / "metrics"
    validation_traces_dir = validation_dir / "traces"
    selection_dir = run_root / "selection"
    summary_dir = run_root / "summary"

    for folder in [
        metadata_dir,
        target_bank_dir,
        generation_dir,
        generation_candidates_dir,
        generation_metrics_dir,
        generation_traces_dir,
        validation_dir,
        validation_metrics_dir,
        validation_traces_dir,
        selection_dir,
        summary_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_root=run_root,
        metadata_dir=metadata_dir,
        target_bank_dir=target_bank_dir,
        generation_dir=generation_dir,
        generation_candidates_dir=generation_candidates_dir,
        generation_metrics_dir=generation_metrics_dir,
        generation_traces_dir=generation_traces_dir,
        validation_dir=validation_dir,
        validation_metrics_dir=validation_metrics_dir,
        validation_traces_dir=validation_traces_dir,
        selection_dir=selection_dir,
        summary_dir=summary_dir,
    )


def _load_packet_manifest(manifest_csv: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_csv)


def _load_packet_blocks(blocks_dir: Path) -> pd.DataFrame:
    block_paths = sorted(blocks_dir.glob("*.blocks.csv"))
    if not block_paths:
        raise ValueError(f"No .blocks.csv files found in {blocks_dir}")
    frames = [pd.read_csv(path) for path in block_paths]
    return pd.concat(frames, ignore_index=True)


def _prepare_packet_inputs(
    packet_manifest: pd.DataFrame,
    packet_blocks: pd.DataFrame,
    max_documents: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "source_order" not in packet_manifest.columns or "doc_id" not in packet_manifest.columns:
        raise ValueError("packet_manifest.csv must include source_order and doc_id columns.")

    manifest_sorted = packet_manifest.sort_values("source_order")
    if "parse_status" in manifest_sorted.columns:
        manifest_sorted = manifest_sorted[manifest_sorted["parse_status"] == "parsed"]
    if manifest_sorted.empty:
        raise ValueError("No parsed documents are available for synthetic generation.")

    source_documents = manifest_sorted.head(max_documents).copy()
    source_documents["doc_id"] = source_documents["doc_id"].astype(str)
    source_document_ids = set(source_documents["doc_id"].tolist())

    selected_blocks = packet_blocks[packet_blocks["doc_id"].astype(str).isin(source_document_ids)].copy()
    if selected_blocks.empty:
        raise ValueError("No packet blocks found for selected source documents.")

    selected_blocks["doc_id"] = selected_blocks["doc_id"].astype(str)
    selected_blocks["citation_token"] = selected_blocks["citation_token"].apply(lambda token: normalize_citation_token(str(token)))
    selected_blocks = selected_blocks.sort_values(["doc_id", "page_number", "block_number"]).reset_index(drop=True)

    return selected_blocks, source_documents


def _build_packet_corpus(packet_blocks: pd.DataFrame, source_documents: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, source_row in source_documents.sort_values("source_order").iterrows():
        doc_id = str(source_row["doc_id"])
        source_filename = str(source_row.get("source_filename", ""))
        lines.append(f'<DOCUMENT id="{doc_id}" source_filename="{source_filename}">')

        document_blocks = packet_blocks[packet_blocks["doc_id"] == doc_id].sort_values(["page_number", "block_number"])
        for _, block_row in document_blocks.iterrows():
            block_id = _citation_token_to_xml_block_id(str(block_row["citation_token"]))
            lines.append(f'<BLOCK id="{block_id}">')
            lines.append(str(block_row["text"]))
            lines.append("</BLOCK>")
        lines.append("</DOCUMENT>")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _target_bank_sanity(target_bank: pd.DataFrame) -> pd.DataFrame:
    def has_invalid_citation_token(citation_token: Any) -> bool:
        try:
            normalize_citation_token(str(citation_token))
            return False
        except ValueError:
            return True

    invalid_tokens = target_bank["citation_token"].fillna("").apply(has_invalid_citation_token)
    document_count = int(target_bank["doc_id"].astype(str).nunique()) if "doc_id" in target_bank.columns else 0
    avg_blocks_per_document = None
    if document_count > 0:
        avg_blocks_per_document = float(target_bank.shape[0] / document_count)
    duplicate_subset = [column_name for column_name in ["doc_id", "page_number", "block_number"] if column_name in target_bank.columns]
    duplicate_count = int(target_bank.duplicated(subset=duplicate_subset).sum()) if duplicate_subset else int(target_bank.duplicated().sum())
    missing_text_count = int(target_bank["text"].isna().sum()) if "text" in target_bank.columns else 0
    rows = [
        {"metric": "row_count", "value": int(target_bank.shape[0])},
        {"metric": "column_count", "value": int(target_bank.shape[1])},
        {"metric": "duplicate_rows", "value": duplicate_count},
        {"metric": "missing_block_text", "value": missing_text_count},
        {"metric": "invalid_citation_tokens", "value": int(invalid_tokens.sum())},
        {"metric": "document_count", "value": document_count},
        {"metric": "avg_blocks_per_document", "value": avg_blocks_per_document},
    ]
    return pd.DataFrame(rows)


def _build_generation_request(
    model: str,
    reasoning_effort: str,
    temperature: float | None,
    service_tier: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "service_tier": service_tier,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        "reasoning": {"effort": reasoning_effort},
    }
    if reasoning_effort == "none" and temperature is not None:
        request["temperature"] = temperature
    return request


@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
)
def _call_openai_parse(client: OpenAI, request: dict[str, Any], text_format: type[ParsedModelType]) -> Any:
    return client.responses.parse(text_format=text_format, **request)


def _normalize_generated_item(
    parsed_candidate: dict[str, Any],
    *,
    mode_name: str,
    citation_token: str,
    packet_id: str,
    as_of_date: str,
) -> dict[str, Any]:
    normalized = dict(parsed_candidate)

    normalized["schema_version"] = "v1"
    normalized["packet_id"] = packet_id
    normalized["as_of_date"] = as_of_date

    raw_mode_value = str(normalized.get("target_error_mode", "")).strip()
    if raw_mode_value in {"A", "C", "D"}:
        normalized["target_error_mode"] = raw_mode_value
    elif raw_mode_value.lower() in MODE_TO_ERROR:
        normalized["target_error_mode"] = MODE_TO_ERROR[raw_mode_value.lower()]
    else:
        normalized["target_error_mode"] = MODE_TO_ERROR[mode_name]

    scenario_facts = normalized.get("scenario_facts")
    if isinstance(scenario_facts, str):
        normalized["scenario_facts"] = [scenario_facts]
    elif not isinstance(scenario_facts, list):
        normalized["scenario_facts"] = ["Assume closed-world packet constraints."]

    grading_contract = normalized.get("grading_contract")
    if not isinstance(grading_contract, dict):
        grading_contract = {}
    normalized["grading_contract"] = grading_contract

    citation_groups = grading_contract.get("expected_citation_groups")
    if isinstance(citation_groups, list):
        cleaned_groups: list[list[str]] = []
        for group in citation_groups:
            if isinstance(group, list):
                cleaned = [str(token).strip() for token in group if str(token).strip()]
                if cleaned:
                    cleaned_groups.append(cleaned)
            elif isinstance(group, str) and group.strip():
                cleaned_groups.append([group.strip()])
        if cleaned_groups:
            grading_contract["expected_citation_groups"] = cleaned_groups
        else:
            grading_contract["expected_citation_groups"] = [[citation_token]]
    else:
        grading_contract["expected_citation_groups"] = [[citation_token]]

    over_note = grading_contract.get("overextension_trigger_note")
    if mode_name == "overextension":
        if not isinstance(over_note, str) or not over_note.strip():
            grading_contract["overextension_trigger_note"] = (
                "Check whether qualified source language is overstated into categorical claims."
            )
    else:
        if "overextension_trigger_note" not in grading_contract:
            grading_contract["overextension_trigger_note"] = None

    precedence_note = grading_contract.get("precedence_trigger_note")
    if mode_name == "precedence":
        if not isinstance(precedence_note, str) or not precedence_note.strip():
            grading_contract["precedence_trigger_note"] = (
                "Check whether controlling authority is used over superseded or non-binding sources."
            )
    else:
        if "precedence_trigger_note" not in grading_contract:
            grading_contract["precedence_trigger_note"] = None

    over_cautions = grading_contract.get("overextension_cautions")
    if not isinstance(over_cautions, list):
        grading_contract["overextension_cautions"] = []

    precedence_cautions = grading_contract.get("precedence_cautions")
    if not isinstance(precedence_cautions, list):
        grading_contract["precedence_cautions"] = []

    if not isinstance(normalized.get("prompt"), str) or not str(normalized.get("prompt", "")).strip():
        normalized["prompt"] = (
            "Draft an internal legal memo section using only packet authorities and cite packet tokens only."
        )

    if not isinstance(normalized.get("item_id"), str):
        normalized["item_id"] = ""
    if not isinstance(normalized.get("query_id"), str):
        normalized["query_id"] = ""

    return normalized


def _default_candidate_from_mode(
    mode_name: str,
    packet_id: str,
    as_of_date: str,
    item_index: int,
    default_citation_token: str,
) -> dict[str, Any]:
    mode_letter = MODE_TO_ERROR[mode_name]
    citation_token = normalize_citation_token(default_citation_token)

    over_note = None
    precedence_note = None
    if mode_name == "overextension":
        over_note = "Checks whether qualified source language is overgeneralized into categorical claims."
    if mode_name == "precedence":
        precedence_note = "Checks whether the answer follows controlling authority and rejects superseded authority."

    candidate = {
        "schema_version": "v1",
        "item_id": f"{packet_id}_{mode_letter}_{item_index:02d}",
        "packet_id": packet_id,
        "target_error_mode": mode_letter,
        "query_id": f"q_{mode_name[:3]}_{item_index:04d}",
        "as_of_date": as_of_date,
        "prompt": (
            "Draft an internal legal memo section using only packet authorities. "
            f"Ground analysis in {citation_token} and avoid unsupported categorical statements."
        ),
        "scenario_facts": [
            "Assume a closed-world packet with no external authorities.",
            f"Primary support token: {citation_token}.",
            "Address uncertainty explicitly and avoid fabricated citations.",
        ],
        "grading_contract": {
            "expected_citation_groups": [[citation_token]],
            "overextension_trigger_note": over_note,
            "overextension_cautions": ["Do not reward dropped qualifiers or modal inflation."] if over_note else [],
            "precedence_trigger_note": precedence_note,
            "precedence_cautions": ["Do not treat vacated or persuasive-only authority as controlling."] if precedence_note else [],
        },
    }

    return SyntheticItem.model_validate(candidate).model_dump()


def _generate_one_item(
    *,
    mode_name: str,
    request_id: str,
    default_citation_token: str,
    packet_corpus_text: str,
    config: SyntheticGenerationConfig,
    item_index: int,
    generation_traces_dir: Path,
    openai_client: OpenAI | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_time = time.perf_counter()
    mode_letter = MODE_TO_ERROR[mode_name]
    item_id = f"{config.packet_id}_{mode_letter}_{item_index:02d}"
    query_id = f"q_{mode_name[:3]}_{item_index:04d}"

    if config.dry_run:
        candidate = _default_candidate_from_mode(
            mode_name,
            config.packet_id,
            config.as_of_date,
            item_index,
            default_citation_token,
        )
        latency = time.perf_counter() - start_time
        return candidate, {
            "stage": "generation",
            "item_id": item_id,
            "item_index": item_index,
            "mode_name": mode_name,
            "request_id": request_id,
            "status": "completed",
            "latency_seconds": latency,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    if openai_client is None:
        raise ValueError("OpenAI client is required when dry_run is false.")

    system_prompt, user_prompt = _load_mode_prompts(
        mode_name=mode_name,
        packet_id=config.packet_id,
        as_of_date=config.as_of_date,
        item_index=item_index,
        packet_corpus_text=packet_corpus_text,
    )
    request = _build_generation_request(
        model=config.generation_model,
        reasoning_effort=config.generation_reasoning_effort,
        temperature=config.generation_temperature,
        service_tier=config.service_tier,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        response = _call_openai_parse(openai_client, request, GeneratedSyntheticItemOutput)
    except ValidationError:
        latency = time.perf_counter() - start_time
        validated_candidate = _default_candidate_from_mode(
            mode_name,
            config.packet_id,
            config.as_of_date,
            item_index,
            default_citation_token,
        )
        return validated_candidate, {
            "stage": "generation",
            "item_id": item_id,
            "item_index": item_index,
            "mode_name": mode_name,
            "request_id": request_id,
            "status": "invalid_structured_output_fallback_default_candidate",
            "latency_seconds": latency,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    latency = time.perf_counter() - start_time

    trace_path = generation_traces_dir / f"{mode_name}_{request_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    generation_status = response.status
    try:
        if response.output_parsed is None:
            raise ValueError("Missing parsed structured output.")
        parsed_candidate = response.output_parsed.model_dump(mode="python", exclude_none=True)
        normalized_candidate = _normalize_generated_item(
            parsed_candidate,
            mode_name=mode_name,
            citation_token=normalize_citation_token(default_citation_token),
            packet_id=config.packet_id,
            as_of_date=config.as_of_date,
        )
        validated_candidate = SyntheticItem.model_validate(normalized_candidate).model_dump()
    except (ValueError, ValidationError):
        # Fallback keeps the run moving when model output cannot be validated.
        validated_candidate = _default_candidate_from_mode(
            mode_name,
            config.packet_id,
            config.as_of_date,
            item_index,
            default_citation_token,
        )
        generation_status = "fallback_default_candidate"

    validated_candidate["item_id"] = item_id
    validated_candidate["query_id"] = query_id

    usage = response.usage
    return validated_candidate, {
        "stage": "generation",
        "item_id": item_id,
        "item_index": item_index,
        "mode_name": mode_name,
        "request_id": request_id,
        "status": generation_status,
        "latency_seconds": latency,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "reasoning_tokens": _extract_reasoning_tokens(usage),
        "total_tokens": usage.total_tokens if usage else None,
    }


def _is_natural_language_criteria(criteria_note: str | None) -> bool:
    if not isinstance(criteria_note, str):
        return False
    cleaned_note = criteria_note.strip()
    if not cleaned_note:
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned_note) if word]
    return len(words) >= 5


def _deterministic_validation(item_payload: dict[str, Any], citation_universe: set[str]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    item = SyntheticItem.model_validate(item_payload)

    expected_citations = [
        citation_token
        for citation_group in item.grading_contract.expected_citation_groups
        for citation_token in citation_group
    ]
    if not expected_citations:
        reasons.append("missing_expected_citations")

    for citation_token in expected_citations:
        if citation_token not in citation_universe:
            reasons.append("expected_citation_outside_packet")

    if item.target_error_mode == "C" and not _is_natural_language_criteria(item.grading_contract.overextension_trigger_note):
        reasons.append("missing_overextension_criteria")

    if item.target_error_mode == "D" and not _is_natural_language_criteria(item.grading_contract.precedence_trigger_note):
        reasons.append("missing_precedence_criteria")

    deduped_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for reason in reasons:
        if reason in seen_reasons:
            continue
        deduped_reasons.append(reason)
        seen_reasons.add(reason)

    return len(deduped_reasons) == 0, deduped_reasons, {
        "expected_citation_count": len(expected_citations),
        "criteria_checks_passed": not any(
            reason in {"missing_overextension_criteria", "missing_precedence_criteria"} for reason in deduped_reasons
        ),
    }


def _build_validation_request(config: SyntheticGenerationConfig, item_payload: dict[str, Any]) -> dict[str, Any]:
    system_prompt, user_prompt = _load_verifier_prompts(item_payload)
    return {
        "model": config.validation_model,
        "service_tier": config.service_tier,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        "reasoning": {"effort": config.validation_reasoning_effort},
    }


def _validate_one_item(
    *,
    config: SyntheticGenerationConfig,
    item_payload: dict[str, Any],
    validation_traces_dir: Path,
    openai_client: OpenAI | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_time = time.perf_counter()

    if config.dry_run:
        return {
            "item_id": item_payload["item_id"],
            "verdict": "pass",
            "reason": "Dry-run verifier pass.",
            "risk_flags": [],
            "suggested_fix": "",
            "full_response": {"dry_run": True},
        }, {
            "stage": "validation",
            "item_id": item_payload["item_id"],
            "status": "completed",
            "verdict": "pass",
            "latency_seconds": time.perf_counter() - start_time,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    if openai_client is None:
        raise ValueError("OpenAI client is required when dry_run is false.")

    request = _build_validation_request(config, item_payload)
    try:
        response = _call_openai_parse(openai_client, request, VerifierOutput)
    except ValidationError:
        latency = time.perf_counter() - start_time
        return {
            "item_id": item_payload["item_id"],
            "verdict": "fail",
            "reason": "Verifier response failed structured-output parsing.",
            "risk_flags": ["invalid_verifier_output"],
            "suggested_fix": "Ensure verifier returns strict JSON object matching schema.",
            "full_response": {"parse_error": True},
        }, {
            "stage": "validation",
            "item_id": item_payload["item_id"],
            "status": "invalid_structured_output",
            "verdict": "fail",
            "latency_seconds": latency,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    latency = time.perf_counter() - start_time

    parsed_output = response.output_parsed
    if parsed_output is None:
        verdict = "fail"
        parsed_review = {
            "reason": "Verifier response was not valid JSON.",
            "risk_flags": ["invalid_verifier_output"],
            "suggested_fix": "Ensure verifier returns strict JSON object.",
        }
    else:
        parsed_review = parsed_output.model_dump(mode="python")
        verdict = parsed_output.verdict

    review = {
        "item_id": item_payload["item_id"],
        "verdict": verdict,
        "reason": str(parsed_review.get("reason", "")).strip(),
        "risk_flags": parsed_review.get("risk_flags", []),
        "suggested_fix": str(parsed_review.get("suggested_fix", "")).strip(),
        "full_response": response.model_dump(),
    }

    trace_path = validation_traces_dir / f"{item_payload['item_id']}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    usage = response.usage
    metrics = {
        "stage": "validation",
        "item_id": item_payload["item_id"],
        "status": response.status,
        "verdict": verdict,
        "latency_seconds": latency,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "reasoning_tokens": _extract_reasoning_tokens(usage),
        "total_tokens": usage.total_tokens if usage else None,
    }
    return review, metrics


def _quality_score(item_payload: dict[str, Any]) -> float:
    prompt_score = min(len(item_payload["prompt"].split()), 200) / 200.0
    fact_score = min(len(item_payload.get("scenario_facts", [])), 5) / 5.0
    citation_group_score = min(len(item_payload["grading_contract"]["expected_citation_groups"]), 3) / 3.0
    return prompt_score + fact_score + citation_group_score


def _trap_signature(prompt_text: str) -> str:
    words = [word.strip(".,;:()[]{}").lower() for word in prompt_text.split()]
    return " ".join(words[:14])


def _primary_doc_id(item_payload: dict[str, Any]) -> str:
    first_citation = item_payload["grading_contract"]["expected_citation_groups"][0][0]
    return first_citation.split("[")[0]


def build_items_for_selection(
    accepted_items: list[dict[str, Any]],
    deterministic_pass_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_modes = {"overextension", "precedence", "fake_citations"}
    selected_items = list(accepted_items)
    selected_item_ids = {item["item_id"] for item in selected_items}
    selected_modes = {ERROR_TO_MODE[item["target_error_mode"]] for item in selected_items}

    missing_modes = required_modes - selected_modes
    if not missing_modes:
        return selected_items

    for item in deterministic_pass_items:
        mode_name = ERROR_TO_MODE[item["target_error_mode"]]
        if mode_name not in missing_modes:
            continue
        if item["item_id"] in selected_item_ids:
            continue
        selected_items.append(item)
        selected_item_ids.add(item["item_id"])
        selected_modes.add(mode_name)
        missing_modes = required_modes - selected_modes
        if not missing_modes:
            break

    return selected_items


def _select_mode_items(mode_rows: pd.DataFrame, keep_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_signatures: set[str] = set()
    used_docs: set[str] = set()

    for _, row in mode_rows.sort_values("quality_score", ascending=False).iterrows():
        signature = str(row["trap_signature"])
        doc_id = str(row["primary_doc_id"])

        if signature in used_signatures:
            continue
        if doc_id in used_docs and len(used_docs) < keep_count:
            continue

        selected.append(row["payload"])
        used_signatures.add(signature)
        used_docs.add(doc_id)
        if len(selected) == keep_count:
            break

    if len(selected) < keep_count:
        for _, row in mode_rows.sort_values("quality_score", ascending=False).iterrows():
            if row["payload"] in selected:
                continue
            selected.append(row["payload"])
            if len(selected) == keep_count:
                break

    return selected


class SyntheticGenerationPipeline:
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config

    def bootstrap(self, run_id: str | None = None) -> PipelineContext:
        packet_root = self.config.packet_root / self.config.packet_id
        manifest_csv = packet_root / "packet_manifest.csv"
        blocks_dir = packet_root / "blocks"
        packet_information_md = packet_root / "packet_information.md"

        if not packet_root.exists():
            raise ValueError(f"Packet folder not found: {packet_root}")
        if not manifest_csv.exists():
            raise ValueError(f"Packet manifest not found: {manifest_csv}")

        run_paths = _create_run_paths(self.config.output_root, self.config.packet_id, run_id)
        (run_paths.metadata_dir / "config_snapshot.md").write_text(
            "\n".join(
                [
                    f"packet_id: {self.config.packet_id}",
                    f"generation_model: {self.config.generation_model}",
                    f"generation_reasoning_effort: {self.config.generation_reasoning_effort}",
                    f"validation_model: {self.config.validation_model}",
                    f"validation_reasoning_effort: {self.config.validation_reasoning_effort}",
                    f"service_tier: {self.config.service_tier}",
                    f"request_timeout_seconds: {self.config.request_timeout_seconds}",
                    f"dry_run: {self.config.dry_run}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_paths.metadata_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "packet_id": self.config.packet_id,
                    "run_root": str(run_paths.run_root),
                    "created_at_utc": _utc_now_iso(),
                    "layout_version": "v2",
                    "paths": _json_safe(run_paths.__dict__),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return PipelineContext(
            config=self.config,
            packet_root=packet_root,
            manifest_csv=manifest_csv,
            blocks_dir=blocks_dir,
            packet_information_md=packet_information_md,
            run_paths=run_paths,
        )

    def run_target_bank(self, context: PipelineContext) -> TargetBankResult:
        manifest = _load_packet_manifest(context.manifest_csv)
        blocks = _load_packet_blocks(context.blocks_dir)
        target_bank, source_documents = _prepare_packet_inputs(manifest, blocks, max_documents=8)
        packet_corpus_text = _build_packet_corpus(target_bank, source_documents)
        sanity = _target_bank_sanity(target_bank)

        target_bank.to_csv(context.run_paths.target_bank_dir / "target_bank.csv", index=False)
        source_documents.to_csv(context.run_paths.target_bank_dir / "source_documents.csv", index=False)
        (context.run_paths.target_bank_dir / "packet_corpus.txt").write_text(packet_corpus_text, encoding="utf-8")
        sanity.to_csv(context.run_paths.target_bank_dir / "sanity_checks.csv", index=False)
        return TargetBankResult(target_bank=target_bank, packet_corpus_text=packet_corpus_text, sanity_summary=sanity)

    def run_generation(
        self,
        *,
        context: PipelineContext,
        target_bank: pd.DataFrame,
        packet_corpus_text: str,
        openai_client: OpenAI | None = None,
    ) -> GenerationResult:
        per_mode_counts = {
            "overextension": context.config.generate_count.overextension,
            "precedence": context.config.generate_count.precedence,
            "fake_citations": context.config.generate_count.fake_citations,
        }
        if target_bank.empty:
            raise ValueError("No packet blocks available for generation.")
        default_citation_token = normalize_citation_token(str(target_bank.iloc[0]["citation_token"]))

        candidates_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in per_mode_counts}
        metrics_rows: list[dict[str, Any]] = []

        def generate_mode(mode_name: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
            mode_candidates: list[dict[str, Any]] = []
            mode_metrics: list[dict[str, Any]] = []
            item_count = per_mode_counts[mode_name]
            with ThreadPoolExecutor(max_workers=min(context.config.parallelism.generation_workers, item_count)) as item_pool:
                futures: dict[Any, dict[str, Any]] = {}
                for item_index in range(1, item_count + 1):
                    request_id = f"req_{mode_name[:3]}_{item_index:04d}"
                    futures[
                        item_pool.submit(
                            _generate_one_item,
                            mode_name=mode_name,
                            request_id=request_id,
                            default_citation_token=default_citation_token,
                            packet_corpus_text=packet_corpus_text,
                            config=context.config,
                            item_index=item_index,
                            generation_traces_dir=context.run_paths.generation_traces_dir,
                            openai_client=openai_client,
                        )
                    ] = {"request_id": request_id, "item_index": item_index}

                for future in as_completed(futures):
                    request_info = futures[future]
                    request_id = str(request_info["request_id"])
                    item_index = int(request_info["item_index"])
                    try:
                        candidate, metric = future.result()
                    except OpenAIError as error:
                        fallback_candidate = _default_candidate_from_mode(
                            mode_name=mode_name,
                            packet_id=context.config.packet_id,
                            as_of_date=context.config.as_of_date,
                            item_index=item_index,
                            default_citation_token=default_citation_token,
                        )
                        mode_candidates.append(fallback_candidate)
                        mode_metrics.append(
                            {
                                "stage": "generation",
                                "item_id": fallback_candidate["item_id"],
                                "item_index": item_index,
                                "mode_name": mode_name,
                                "request_id": request_id,
                                "status": "error_fallback_default_candidate",
                                "latency_seconds": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "reasoning_tokens": None,
                                "total_tokens": None,
                                "error_type": type(error).__name__,
                            }
                        )
                        continue

                    mode_candidates.append(candidate)
                    mode_metrics.append(metric)

            return mode_name, mode_candidates, mode_metrics

        with ThreadPoolExecutor(max_workers=min(context.config.parallelism.mode_workers, len(per_mode_counts))) as mode_pool:
            mode_futures = [mode_pool.submit(generate_mode, mode_name) for mode_name in per_mode_counts]
            for mode_future in as_completed(mode_futures):
                mode_name, mode_candidates, mode_metrics = mode_future.result()
                candidates_by_mode[mode_name] = mode_candidates
                metrics_rows.extend(mode_metrics)

        metrics_df = pd.DataFrame(metrics_rows)
        for mode_name, candidates in candidates_by_mode.items():
            with (context.run_paths.generation_candidates_dir / f"{mode_name}_candidates.jsonl").open(
                "w",
                encoding="utf-8",
            ) as file_handle:
                for candidate in candidates:
                    file_handle.write(json.dumps(candidate, ensure_ascii=True) + "\n")

        augmented_metrics_df = _augment_with_latency_milliseconds(metrics_df)
        augmented_metrics_df.to_csv(context.run_paths.generation_metrics_dir / "request_metrics.csv", index=False)
        augmented_metrics_df.to_csv(context.run_paths.generation_metrics_dir / "datapoint_timings.csv", index=False)
        return GenerationResult(candidates_by_mode=candidates_by_mode, request_metrics=augmented_metrics_df)

    def run_validation(
        self,
        *,
        context: PipelineContext,
        target_bank: pd.DataFrame,
        candidates: list[dict[str, Any]],
        openai_client: OpenAI | None = None,
    ) -> ValidationResult:
        citation_universe = {normalize_citation_token(str(token)) for token in target_bank["citation_token"].dropna().tolist()}

        deterministic_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = []
        items_for_llm: list[dict[str, Any]] = []

        for item in candidates:
            deterministic_pass, reasons, details = _deterministic_validation(item, citation_universe)
            deterministic_rows.append(
                {
                    "item_id": item["item_id"],
                    "deterministic_pass": deterministic_pass,
                    "reason_codes": "|".join(reasons),
                    "expected_citation_count": details["expected_citation_count"],
                    "criteria_checks_passed": details["criteria_checks_passed"],
                }
            )
            if deterministic_pass:
                items_for_llm.append(item)
            else:
                rejection_rows.append(
                    {
                        "item_id": item["item_id"],
                        "rejection_stage": "deterministic",
                        "rejection_reason": "|".join(reasons),
                    }
                )

        llm_reviews: list[dict[str, Any]] = []
        accepted_items: list[dict[str, Any]] = []
        metrics_rows: list[dict[str, Any]] = []

        if items_for_llm:
            with ThreadPoolExecutor(max_workers=min(context.config.parallelism.validation_workers, len(items_for_llm))) as pool:
                futures = {
                    pool.submit(
                        _validate_one_item,
                        config=context.config,
                        item_payload=item,
                        validation_traces_dir=context.run_paths.validation_traces_dir,
                        openai_client=openai_client,
                    ): item
                    for item in items_for_llm
                }

                for future in as_completed(futures):
                    item = futures[future]
                    item_id = item["item_id"]
                    try:
                        review, metrics = future.result()
                    except OpenAIError as error:
                        rejection_rows.append(
                            {
                                "item_id": item_id,
                                "rejection_stage": "llm_verifier",
                                "rejection_reason": type(error).__name__,
                            }
                        )
                        metrics_rows.append(
                            {
                                "stage": "validation",
                                "item_id": item_id,
                                "status": "error",
                                "verdict": "fail",
                                "latency_seconds": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "reasoning_tokens": None,
                                "total_tokens": None,
                            }
                        )
                        continue

                    llm_reviews.append(review)
                    metrics_rows.append(metrics)
                    if review["verdict"] == "pass":
                        accepted_items.append(item)
                    else:
                        rejection_rows.append(
                            {
                                "item_id": item_id,
                                "rejection_stage": "llm_verifier",
                                "rejection_reason": review["reason"],
                            }
                        )

        deterministic_df = pd.DataFrame(deterministic_rows)
        rejection_df = pd.DataFrame(rejection_rows)
        metrics_df = pd.DataFrame(metrics_rows)

        deterministic_df.to_csv(context.run_paths.validation_dir / "deterministic_checks.csv", index=False)
        with (context.run_paths.validation_dir / "llm_consensus_reviews.jsonl").open("w", encoding="utf-8") as file_handle:
            for review in llm_reviews:
                file_handle.write(json.dumps(review, ensure_ascii=True) + "\n")
        rejection_df.to_csv(context.run_paths.validation_dir / "rejection_log.csv", index=False)
        augmented_metrics_df = _augment_with_latency_milliseconds(metrics_df)
        augmented_metrics_df.to_csv(context.run_paths.validation_metrics_dir / "request_metrics.csv", index=False)
        augmented_metrics_df.to_csv(context.run_paths.validation_metrics_dir / "datapoint_timings.csv", index=False)

        return ValidationResult(
            accepted_items=accepted_items,
            deterministic_pass_items=items_for_llm,
            deterministic_checks=deterministic_df,
            llm_reviews=llm_reviews,
            rejection_log=rejection_df,
            request_metrics=augmented_metrics_df,
        )

    def run_selection(self, *, context: PipelineContext, accepted_items: list[dict[str, Any]]) -> SelectionResult:
        if not accepted_items:
            raise ValueError("No accepted items available for selection.")

        table_rows: list[dict[str, Any]] = []
        for item in accepted_items:
            table_rows.append(
                {
                    "item_id": item["item_id"],
                    "mode_name": ERROR_TO_MODE[item["target_error_mode"]],
                    "target_error_mode": item["target_error_mode"],
                    "primary_doc_id": _primary_doc_id(item),
                    "trap_signature": _trap_signature(item["prompt"]),
                    "quality_score": _quality_score(item),
                    "payload": item,
                }
            )

        item_table = pd.DataFrame(table_rows)

        keep_counts = {
            "overextension": context.config.final_keep_count.overextension,
            "precedence": context.config.final_keep_count.precedence,
            "fake_citations": context.config.final_keep_count.fake_citations,
        }

        selected_items: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []

        for mode_name, keep_count in keep_counts.items():
            mode_rows = item_table[item_table["mode_name"] == mode_name]
            if mode_rows.empty:
                raise ValueError(f"No accepted items for mode {mode_name}")

            mode_selected = _select_mode_items(mode_rows, keep_count)
            if len(mode_selected) < keep_count:
                raise ValueError(f"Insufficient items for mode {mode_name}: {len(mode_selected)} < {keep_count}")

            selected_items.extend(mode_selected)
            for item in mode_selected:
                selection_rows.append(
                    {
                        "item_id": item["item_id"],
                        "mode_name": mode_name,
                        "target_error_mode": item["target_error_mode"],
                        "primary_doc_id": _primary_doc_id(item),
                    }
                )

        selection_table = pd.DataFrame(selection_rows).sort_values(["mode_name", "item_id"]).reset_index(drop=True)

        report_lines = [
            "# Selection Report",
            "",
            f"Total selected items: {len(selected_items)}",
            "",
            "## Per-mode counts",
        ]
        for mode_name, count in selection_table.groupby("mode_name")["item_id"].count().to_dict().items():
            report_lines.append(f"- {mode_name}: {count}")
        report_lines.append("")
        report_lines.append("## Selected items")
        for _, row in selection_table.iterrows():
            report_lines.append(f"- {row['item_id']} ({row['mode_name']}, {row['primary_doc_id']})")
        report_text = "\n".join(report_lines) + "\n"

        with (context.run_paths.selection_dir / "selected_items.jsonl").open("w", encoding="utf-8") as file_handle:
            for item in selected_items:
                file_handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        selection_table.to_csv(context.run_paths.selection_dir / "selected_items.csv", index=False)
        (context.run_paths.selection_dir / "selection_report.md").write_text(report_text, encoding="utf-8")

        return SelectionResult(
            selected_items=selected_items,
            selection_table=selection_table,
            selection_report_markdown=report_text,
        )

    def export_canonical_dataset(self, selected_items: list[dict[str, Any]]) -> Path:
        dataset_dir = self.config.dataset_root / self.config.packet_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        with (dataset_dir / "synthetic_items.jsonl").open("w", encoding="utf-8") as file_handle:
            for item in selected_items:
                file_handle.write(json.dumps(item, ensure_ascii=True) + "\n")

        pd.DataFrame(selected_items).to_csv(dataset_dir / "synthetic_items.csv", index=False)
        return dataset_dir

    def run_all(self, context: PipelineContext, openai_client: OpenAI | None = None) -> dict[str, Any]:
        run_start_time = time.perf_counter()

        target_bank_start_time = time.perf_counter()
        target_bank_result = self.run_target_bank(context)
        target_bank_duration_seconds = time.perf_counter() - target_bank_start_time

        generation_start_time = time.perf_counter()
        generation_result = self.run_generation(
            context=context,
            target_bank=target_bank_result.target_bank,
            packet_corpus_text=target_bank_result.packet_corpus_text,
            openai_client=openai_client,
        )
        generation_duration_seconds = time.perf_counter() - generation_start_time

        all_candidates: list[dict[str, Any]] = []
        for mode_items in generation_result.candidates_by_mode.values():
            all_candidates.extend(mode_items)

        validation_start_time = time.perf_counter()
        validation_result = self.run_validation(
            context=context,
            target_bank=target_bank_result.target_bank,
            candidates=all_candidates,
            openai_client=openai_client,
        )
        validation_duration_seconds = time.perf_counter() - validation_start_time

        items_for_selection = build_items_for_selection(
            accepted_items=validation_result.accepted_items,
            deterministic_pass_items=validation_result.deterministic_pass_items,
        )
        selection_fallback_used = len(items_for_selection) > len(validation_result.accepted_items)

        selection_start_time = time.perf_counter()
        selection_result = self.run_selection(context=context, accepted_items=items_for_selection)
        selection_duration_seconds = time.perf_counter() - selection_start_time

        export_start_time = time.perf_counter()
        dataset_dir = self.export_canonical_dataset(selection_result.selected_items)
        export_duration_seconds = time.perf_counter() - export_start_time

        metrics_frames = [frame for frame in [generation_result.request_metrics, validation_result.request_metrics] if not frame.empty]
        combined_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        metrics_summary = summarize_request_metrics(combined_metrics)
        metrics_summary.to_csv(context.run_paths.summary_dir / "stage_metrics_summary.csv", index=False)

        datapoint_timings = _build_datapoint_timing_table(
            generation_metrics=generation_result.request_metrics,
            validation_metrics=validation_result.request_metrics,
        )
        datapoint_timings.to_csv(context.run_paths.summary_dir / "datapoint_timings.csv", index=False)

        total_run_duration_seconds = time.perf_counter() - run_start_time
        run_summary = {
            "target_bank_rows": int(target_bank_result.target_bank.shape[0]),
            "generated_counts": {mode: len(items) for mode, items in generation_result.candidates_by_mode.items()},
            "accepted_items": len(validation_result.accepted_items),
            "selection_fallback_used": selection_fallback_used,
            "selected_items": len(selection_result.selected_items),
            "run_root": str(context.run_paths.run_root),
            "dataset_dir": str(dataset_dir),
            "stage_durations_seconds": {
                "target_bank": target_bank_duration_seconds,
                "generation": generation_duration_seconds,
                "validation": validation_duration_seconds,
                "selection": selection_duration_seconds,
                "dataset_export": export_duration_seconds,
                "run_total": total_run_duration_seconds,
            },
            "completed_at_utc": _utc_now_iso(),
        }
        (context.run_paths.summary_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (context.run_paths.metadata_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "packet_id": self.config.packet_id,
                    "run_root": str(context.run_paths.run_root),
                    "completed_at_utc": _utc_now_iso(),
                    "layout_version": "v2",
                    "paths": _json_safe(context.run_paths.__dict__),
                    "summary_file": str(context.run_paths.summary_dir / "run_summary.json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return run_summary


def load_candidates_from_run(candidates_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for mode_name in ["overextension", "precedence", "fake_citations"]:
        mode_file = candidates_dir / f"{mode_name}_candidates.jsonl"
        if not mode_file.exists():
            continue
        with mode_file.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))
    return candidates


def build_openai_client(config: SyntheticGenerationConfig) -> OpenAI:
    return OpenAI(timeout=config.request_timeout_seconds)
