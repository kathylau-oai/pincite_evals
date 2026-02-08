#!/usr/bin/env bash
set -euo pipefail

run_timestamp=""
skip_generation="0"

for argument in "$@"; do
  if [[ "$argument" == "--skip-generation" ]]; then
    skip_generation="1"
    continue
  fi

  if [[ -z "$run_timestamp" ]]; then
    run_timestamp="$argument"
  else
    echo "Unexpected argument: $argument" >&2
    echo "Usage: bash skills/synthetic-generation-audit/scripts/run_and_analyze.sh [run_timestamp] [--skip-generation]" >&2
    exit 1
  fi
done

if [[ -z "$run_timestamp" ]]; then
  run_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ "$skip_generation" == "0" ]]; then
  echo "Running synthetic generation for run_timestamp=$run_timestamp"
  bash src/pincite_evals/synthetic_generation/run_all_packets.sh "$run_timestamp"
else
  echo "Skipping generation and analyzing existing run_timestamp=$run_timestamp"
fi

echo "Preparing audit evidence for run_timestamp=$run_timestamp"
uv run python skills/synthetic-generation-audit/scripts/analyze_synthetic_generation_run.py \
  --run-timestamp "$run_timestamp"

echo "Done. Evidence outputs are under results/synthetic_generation_audit/$run_timestamp"
