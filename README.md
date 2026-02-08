# Pincite Evals

Pincite Evals is a minimal scaffold for implementing an eval that tests legal citation accuracy in memo-style drafting. The current scope is US case law and short litigation memos. Motions and evidence packets are out of scope until SME guidance is finalized.

The planned evaluation targets four core failure modes:

- Fabricated citations
- Wrong source span
- Overextended claims
- Precedent / overruling issues

## Repo layout

```
├─ data/
│  ├─ case_law_packets/        # closed-world case law packets + metadata
│  ├─ datasets/                # JSONL drafting tasks tied to packets
│  └─ deep_research_prompts/   # packet-curation prompt assets
├─ docs/
│  ├─ project_overview.md      # core goals, scope, and eval strategy
│  ├─ packet_design.md
│  ├─ data_schema.md           # placeholder until SME confirmation
│  └─ grading.md               # early guidance; subject to change
├─ src/pincite_evals/          # library code (eval runner, graders, synthetic generation)
└─ tests/
```

## Quickstart

### Setup

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# Create/update .venv from the project lockfile and install dev deps
uv sync --extra dev

# Run tests inside the uv-managed environment
uv run pytest
```

Dependency updates are managed with `uv` commands (for example `uv add ...` and `uv lock`).

Create a `.env` file:

```bash
OPENAI_API_KEY=...  # required to run model and (optionally) LLM-based graders
```

### Eval Runner

Run the general-purpose eval runner with a small dry-run:

```bash
uv run pincite-eval \
  --input-csv data/template_eval_input.csv \
  --experiment-name template_eval_smoke \
  --dry-run \
  --max-samples 10 \
  --max-model-workers 2 \
  --max-item-workers 4
```

Run with multiple model configs in parallel:

```bash
uv run pincite-eval \
  --input-csv data/template_eval_input.csv \
  --model-config-file docs/template_model_configs.json \
  --experiment-name template_eval_multi_config
```

By default, the runner writes under `results/experiments/<run_id>/` with a compact structure:

- `manifest.json`
- `final/predictions_with_grades.csv`
- `final/metrics_summary.csv`
- `final/grader_metrics_summary.csv`
- `final/errors.csv`
- `analysis/failure_mode_report.md`
- `analysis/charts/` (metric plots + grader performance bar plot)

Use `--artifact-level debug` to additionally save verbose debug traces under `debug/`.

### Synthetic Generation Pipeline

Run packet-level synthetic generation (modes `A/C/D`) via the package CLI:

```bash
uv run pincite-synth run-all --config path/to/config.yaml --run-id my_run
```

Run artifacts are written under:

- `results/synthetic_generation/<packet_id>/<run_id>/`

Key outputs include:

- `summary/stage_metrics_summary.csv`
- `summary/datapoint_timings.csv`
- `validation/llm_consensus_reviews.csv`
- `validation/validation_datapoints.csv`
- `validation/accepted_items.jsonl`
- canonical dataset export in `data/datasets/<packet_id>/synthetic_items.{jsonl,csv}`

## Disclaimer

This repository is for evaluation engineering and is not legal advice.

All case law included in packets should come from public, citable sources and be used consistently with any applicable licenses and court rules.
