import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


VALID_REASONING_EFFORTS = {"none", "low", "medium", "high"}
VALID_ERROR_MODES = {"A", "C", "D"}
VALID_SERVICE_TIERS = {"auto", "default", "flex", "scale", "priority"}


@dataclass(frozen=True)
class ModeCountConfig:
    overextension: int
    precedence: int
    fake_citations: int


@dataclass(frozen=True)
class ParallelismConfig:
    mode_workers: int
    generation_workers: int
    validation_workers: int
    max_in_flight_requests: int
    max_retries: int


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    packet_id: str
    packet_root: Path
    output_root: Path
    dataset_root: Path
    generation_model: str
    generation_reasoning_effort: str
    generation_temperature: float | None
    validation_model: str
    validation_reasoning_effort: str
    validation_temperature: float | None
    selection_model: str
    selection_reasoning_effort: str
    service_tier: str
    generate_count: ModeCountConfig
    final_keep_count: ModeCountConfig
    quality_thresholds: dict[str, Any]
    parallelism: ParallelismConfig
    as_of_date: str
    request_timeout_seconds: float
    dry_run: bool


DEFAULT_CONFIG = {
    "packet_root": "data/case_law_packets",
    "output_root": "results/synthetic_generation",
    "dataset_root": "data/datasets",
    "generation_model": "gpt-5.2",
    "generation_reasoning_effort": "medium",
    "generation_temperature": None,
    "validation_model": "gpt-5.2",
    "validation_reasoning_effort": "medium",
    "validation_temperature": None,
    "selection_model": "gpt-5.2",
    "selection_reasoning_effort": "medium",
    "service_tier": "priority",
    "generate_count": {
        "overextension": 10,
        "precedence": 10,
        "fake_citations": 10,
    },
    "final_keep_count": {
        "overextension": 3,
        "precedence": 3,
        "fake_citations": 3,
    },
    "quality_thresholds": {
        "min_expected_citation_groups": 1,
    },
    "parallelism": {
        "mode_workers": 3,
        "generation_workers": 32,
        "validation_workers": 32,
        "max_in_flight_requests": 64,
        "max_retries": 4,
    },
    "request_timeout_seconds": 900.0,
    "dry_run": False,
}


def _load_dict_from_file(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(raw_text)
    elif suffix == ".json":
        loaded = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a JSON/YAML object.")
    return loaded


def _merge_config(user_config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    for key, value in user_config.items():
        if key in {
            "generate_count",
            "final_keep_count",
            "parallelism",
            "quality_thresholds",
        }:
            base_section = dict(DEFAULT_CONFIG[key])
            if not isinstance(value, dict):
                raise ValueError(f"Config field '{key}' must be an object.")
            base_section.update(value)
            merged[key] = base_section
        else:
            merged[key] = value
    return merged


def _validate_reasoning_and_temperature(
    stage_name: str, reasoning_effort: str, temperature: float | None
) -> None:
    if reasoning_effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"{stage_name}_reasoning_effort must be one of {sorted(VALID_REASONING_EFFORTS)}, got: {reasoning_effort}"
        )

    if reasoning_effort != "none" and temperature is not None:
        raise ValueError(
            f"{stage_name}_temperature must be omitted when {stage_name}_reasoning_effort is '{reasoning_effort}'."
        )


def _validate_service_tier(service_tier: str) -> None:
    if service_tier not in VALID_SERVICE_TIERS:
        raise ValueError(
            f"service_tier must be one of {sorted(VALID_SERVICE_TIERS)}, got: {service_tier}"
        )


def load_config(config_path: Path) -> SyntheticGenerationConfig:
    user_config = _load_dict_from_file(config_path)
    merged = _merge_config(user_config)

    packet_id = str(merged.get("packet_id", "")).strip()
    if not packet_id:
        raise ValueError("Config must provide a non-empty packet_id.")

    generation_reasoning_effort = str(merged["generation_reasoning_effort"]).strip()
    validation_reasoning_effort = str(merged["validation_reasoning_effort"]).strip()
    selection_reasoning_effort = str(merged["selection_reasoning_effort"]).strip()
    service_tier = str(merged["service_tier"]).strip()

    _validate_reasoning_and_temperature(
        "generation",
        generation_reasoning_effort,
        merged.get("generation_temperature"),
    )
    _validate_reasoning_and_temperature(
        "validation",
        validation_reasoning_effort,
        merged.get("validation_temperature"),
    )
    _validate_reasoning_and_temperature(
        "selection",
        selection_reasoning_effort,
        None,
    )
    _validate_service_tier(service_tier)

    generate_count = ModeCountConfig(
        overextension=int(merged["generate_count"]["overextension"]),
        precedence=int(merged["generate_count"]["precedence"]),
        fake_citations=int(merged["generate_count"]["fake_citations"]),
    )
    final_keep_count = ModeCountConfig(
        overextension=int(merged["final_keep_count"]["overextension"]),
        precedence=int(merged["final_keep_count"]["precedence"]),
        fake_citations=int(merged["final_keep_count"]["fake_citations"]),
    )

    for mode_name, mode_value in {
        "generate_count.overextension": generate_count.overextension,
        "generate_count.precedence": generate_count.precedence,
        "generate_count.fake_citations": generate_count.fake_citations,
        "final_keep_count.overextension": final_keep_count.overextension,
        "final_keep_count.precedence": final_keep_count.precedence,
        "final_keep_count.fake_citations": final_keep_count.fake_citations,
    }.items():
        if mode_value <= 0:
            raise ValueError(f"{mode_name} must be > 0.")

    parallelism = ParallelismConfig(
        mode_workers=int(merged["parallelism"]["mode_workers"]),
        generation_workers=int(merged["parallelism"]["generation_workers"]),
        validation_workers=int(merged["parallelism"]["validation_workers"]),
        max_in_flight_requests=int(merged["parallelism"]["max_in_flight_requests"]),
        max_retries=int(merged["parallelism"]["max_retries"]),
    )

    if (
        parallelism.mode_workers <= 0
        or parallelism.generation_workers <= 0
        or parallelism.validation_workers <= 0
    ):
        raise ValueError("parallelism worker counts must be > 0.")

    as_of_date_value = str(merged.get("as_of_date", date.today().isoformat())).strip()

    return SyntheticGenerationConfig(
        packet_id=packet_id,
        packet_root=Path(merged["packet_root"]),
        output_root=Path(merged["output_root"]),
        dataset_root=Path(merged["dataset_root"]),
        generation_model=str(merged["generation_model"]),
        generation_reasoning_effort=generation_reasoning_effort,
        generation_temperature=(
            float(merged["generation_temperature"])
            if merged.get("generation_temperature") is not None
            else None
        ),
        validation_model=str(merged["validation_model"]),
        validation_reasoning_effort=validation_reasoning_effort,
        validation_temperature=(
            float(merged["validation_temperature"])
            if merged.get("validation_temperature") is not None
            else None
        ),
        selection_model=str(merged["selection_model"]),
        selection_reasoning_effort=selection_reasoning_effort,
        service_tier=service_tier,
        generate_count=generate_count,
        final_keep_count=final_keep_count,
        quality_thresholds=dict(merged["quality_thresholds"]),
        parallelism=parallelism,
        as_of_date=as_of_date_value,
        request_timeout_seconds=float(merged["request_timeout_seconds"]),
        dry_run=bool(merged["dry_run"]),
    )
