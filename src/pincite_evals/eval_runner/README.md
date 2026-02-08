# Eval Runner

This package contains the main evaluation runner used to:

- Load one or more synthetic-item CSV datasets
- Run drafting inference with one or more model configs in parallel
- Collect latency and token metrics (including percentile stats)
- Run graders and export final combined prediction + grading artifacts

## Main entrypoint

- Module: `pincite_evals.eval_runner`
- Function: `main()`
- CLI script: `pincite-eval`

## Quick run

```bash
uv run pincite-eval \
  --input-csv data/datasets/packet_001/synthetic_items.csv \
  --experiment-name packet001_smoke \
  --max-samples 10 \
  --dry-run
```

```bash
uv run pincite-eval --input-glob 'data/datasets/packet_*/synthetic_items.csv' --experiment-name gpt52_none_full --model gpt-5.2 --reasoning-effort none --temperature 0.0 --max-model-workers 32 --max-item-workers 64 --max-grader-workers 32


uv run pincite-eval --input-glob 'data/datasets/packet_*/synthetic_items.csv' --experiment-name gpt4o_none_full --model gpt-4o --temperature 0.0 --max-model-workers 32 --max-item-workers 64 --max-grader-workers 32
```

## Multi-model run

```bash
uv run pincite-eval \
  --input-glob "data/datasets/packet_*/synthetic_items.csv" \
  --model-config-file docs/template_model_configs.json \
  --experiment-name multi_model_eval
```

Outputs are written under `results/<run_id>/`.
