# Synthetic Generation Subproject

This subproject builds adversarial legal memo eval items for modes `A`, `C`, and `D`:

- `A`: fake citations under missing-authority pressure (requested authority is absent from packet)
- `C`: overextension
- `D`: precedence / authority hierarchy

Mode `B` (wrong span) is intentionally not generated here; it is measured by graders later.

## What it does

1. Reads a packet from `data/case_law_packets/<packet_id>/`.
2. Loads up to 8 parsed docs from `text/DOC###.annotated.txt` and builds one packet corpus directly from annotated text.
3. Generates candidates in parallel per mode by giving the model the full packet corpus and letting it choose trap opportunities (no deterministic mode target-bank seeding).
4. Validates with lightweight deterministic checks + single LLM pass/fail verifier.
5. Selects a diverse final set per mode.
6. Writes canonical dataset files under `data/datasets/<packet_id>/`.

Citation format compatibility: generation and validation accept both `DOC001[P001.B01]` and XML-style `DOC001.P001.B01` citation IDs and normalize to canonical token form in saved items.

Mode `A` design note: fake-citation prompts should ask for authority that is not present in the packet corpus. For this mode, `expected_citation_groups` may be empty or minimal when grading is based on explicit refusal to fabricate.

Generation/verifier requests do not set `max_output_tokens`; outputs are controlled by prompt constraints and structured-output schema parsing.

## Main files

- `config.py`: typed config loader and defaults.
- `schema.py`: item schema and citation token validation.
- `structured_outputs.py`: structured output model classes used by Responses API parsing.
- `pipeline.py`: end-to-end logic (packet corpus prep, generation, validation, selection, metrics).
- `cli.py`: CLI entrypoint (`generate`, `validate`, `run-all`).
- `prompts/`: editable prompt templates per generation mode plus verifier prompts.

## CLI usage

```bash
pincite-synth run-all --config path/to/config.yaml --run-id my_run
```

Or:

```bash
pincite-synth generate --config path/to/config.yaml --run-id my_run
pincite-synth validate --config path/to/config.yaml --run-id my_run
```

Run the pipeline on all packets (from repo root):

```bash
./src/pincite_evals/synthetic_generation/run_all_packets.sh
```

## Outputs

- Run artifacts: `results/synthetic_generation/<packet_id>/<run_id>/`
- Final dataset: `data/datasets/<packet_id>/synthetic_items.jsonl` and `.csv`
- Stage summary metrics: `results/synthetic_generation/<packet_id>/<run_id>/summary/stage_metrics_summary.csv`
- Per-datapoint timing table: `results/synthetic_generation/<packet_id>/<run_id>/summary/datapoint_timings.csv`

Run folder layout:

- `metadata/`: config snapshot + run manifest
- `metadata/packet_input_sanity.csv`: packet block input quality checks
- `generation/candidates/`: per-mode candidate JSONL files
- `generation/metrics/`: request metrics + generation datapoint timings
- `generation/traces/`: raw generation Responses API payloads
- `validation/`: deterministic checks, LLM reviews, rejection log
- `validation/llm_consensus_reviews.csv`: flat LLM verifier results table (one row per LLM-reviewed datapoint)
- `validation/validation_datapoints.csv`: full validation review table for all generated datapoints (deterministic + LLM + rejection + latency/tokens)
- `validation/metrics/`: request metrics + validation datapoint timings
- `validation/traces/`: raw validation Responses API payloads
- `selection/`: selected items and selection report
- `summary/`: cross-stage summaries (`run_summary.json`, stage metrics, datapoint timings)

## Prompt layout

- `prompts/overextension/system.txt`
- `prompts/overextension/user.txt`
- `prompts/precedence/system.txt`
- `prompts/precedence/user.txt`
- `prompts/fake_citations/system.txt`
- `prompts/fake_citations/user.txt`
- `prompts/verifier/system.txt`
- `prompts/verifier/user.txt`
