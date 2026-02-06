# Synthetic Generation Subproject

This subproject builds adversarial legal memo eval items for modes `A`, `C`, and `D`:

- `A`: fake citations
- `C`: overextension
- `D`: precedence / authority hierarchy

Mode `B` (wrong span) is intentionally not generated here; it is measured by graders later.

## What it does

1. Reads a packet from `data/case_law_packets/<packet_id>/`.
2. Builds a deterministic target bank from block-level text.
3. Generates candidates in parallel (default: `gpt-5.2` with high reasoning, or dry-run mode).
4. Validates with deterministic checks + single LLM pass/fail verifier.
5. Selects a diverse final set per mode.
6. Writes canonical dataset files under `data/datasets/<packet_id>/`.

## Main files

- `config.py`: typed config loader and defaults.
- `schema.py`: item schema and citation token validation.
- `pipeline.py`: end-to-end logic (target bank, generation, validation, selection, metrics).
- `cli.py`: CLI entrypoint (`generate`, `validate`, `run-all`).

## CLI usage

```bash
pincite-synth run-all --config path/to/config.yaml --run-id my_run
```

Or:

```bash
pincite-synth generate --config path/to/config.yaml --run-id my_run
pincite-synth validate --config path/to/config.yaml --run-id my_run
```

## Outputs

- Run artifacts: `results/synthetic_generation/<packet_id>/<run_id>/`
- Final dataset: `data/datasets/<packet_id>/synthetic_items.jsonl` and `.csv`
- Metrics summary: `results/synthetic_generation/<packet_id>/<run_id>/traces/metrics.csv`
