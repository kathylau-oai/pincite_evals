import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import (
    SyntheticGenerationPipeline,
    build_openai_client,
    load_candidates_from_run,
    summarize_request_metrics,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    parser.add_argument("--run-id", default=None, help="Optional run id. If omitted, timestamp-based run id is used.")


def _cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    pipeline = SyntheticGenerationPipeline(config)
    context = pipeline.bootstrap(run_id=args.run_id)

    openai_client = None if config.dry_run else build_openai_client(config)
    generation_result = pipeline.run_generation(
        context=context,
        openai_client=openai_client,
    )

    metrics_summary = summarize_request_metrics(generation_result.request_metrics)
    metrics_summary.to_csv(context.run_paths.summary_dir / "stage_metrics_summary.csv", index=False)

    print(f"run_root={context.run_paths.run_root}")
    print(
        "generated_counts="
        + ",".join([f"{mode}:{len(items)}" for mode, items in generation_result.candidates_by_mode.items()])
    )


def _cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    pipeline = SyntheticGenerationPipeline(config)
    context = pipeline.bootstrap(run_id=args.run_id)

    candidates = load_candidates_from_run(context.run_paths.generation_candidates_dir)
    if not candidates:
        raise ValueError(f"No candidates found in {context.run_paths.generation_candidates_dir}")

    openai_client = None if config.dry_run else build_openai_client(config)
    validation_result = pipeline.run_validation(
        context=context,
        candidates=candidates,
        openai_client=openai_client,
    )
    dataset_dir = pipeline.export_canonical_dataset(validation_result.accepted_items)

    metrics_frames = []
    generation_metrics = context.run_paths.generation_metrics_dir / "request_metrics.csv"
    validation_metrics = context.run_paths.validation_metrics_dir / "request_metrics.csv"

    if generation_metrics.exists():
        metrics_frames.append(pd.read_csv(generation_metrics))
    if validation_metrics.exists():
        metrics_frames.append(pd.read_csv(validation_metrics))

    combined_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    metrics_summary = summarize_request_metrics(combined_metrics)
    metrics_summary.to_csv(context.run_paths.summary_dir / "stage_metrics_summary.csv", index=False)

    print(f"run_root={context.run_paths.run_root}")
    print(f"accepted_items={len(validation_result.accepted_items)}")
    print(f"dataset_dir={dataset_dir}")


def _cmd_run_all(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    pipeline = SyntheticGenerationPipeline(config)
    context = pipeline.bootstrap(run_id=args.run_id)

    openai_client = None if config.dry_run else build_openai_client(config)
    summary = pipeline.run_all(context=context, openai_client=openai_client)

    print(f"run_root={summary['run_root']}")
    print(f"packet_block_rows={summary['packet_block_rows']}")
    print(f"packet_document_count={summary['packet_document_count']}")
    print(
        "generated_counts="
        + ",".join([f"{mode}:{count}" for mode, count in summary["generated_counts"].items()])
    )
    print(f"accepted_items={summary['accepted_items']}")
    print(f"dataset_dir={summary['dataset_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic generation pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Run candidate generation from packet full text corpus.")
    _add_common_args(generate_parser)
    generate_parser.set_defaults(handler=_cmd_generate)

    validate_parser = subparsers.add_parser("validate", help="Run validation using existing run candidates.")
    _add_common_args(validate_parser)
    validate_parser.set_defaults(handler=_cmd_validate)

    run_all_parser = subparsers.add_parser("run-all", help="Run end-to-end generation + validation.")
    _add_common_args(run_all_parser)
    run_all_parser.set_defaults(handler=_cmd_run_all)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
