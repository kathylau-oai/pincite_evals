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
│  ├─ packets/                 # closed-world case law packets + metadata
│  ├─ datasets/                # JSONL drafting tasks tied to packets
│  └─ prompts/                 # prompt templates and system instructions
├─ docs/
│  ├─ project_overview.md      # core goals, scope, and eval strategy
│  ├─ packet_design.md
│  ├─ data_schema.md           # placeholder until SME confirmation
│  └─ grading.md               # early guidance; subject to change
├─ graders/                    # grader implementations + interfaces
├─ src/pincite_evals/          # library code
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

### Template Eval Runner

Run the general-purpose template eval runner with a small dry-run:

```bash
uv run pincite-template-eval \
  --input-csv data/template_eval_input.csv \
  --experiment-name template_eval_smoke \
  --dry-run \
  --max-samples 10 \
  --max-model-workers 2 \
  --max-item-workers 4
```

Run with multiple model configs in parallel:

```bash
uv run pincite-template-eval \
  --input-csv data/template_eval_input.csv \
  --model-config-file docs/template_model_configs.json \
  --experiment-name template_eval_multi_config
```

The runner writes all outputs under `results/<experiment_name>_<timestamp>/`, including:

- `predictions_and_grades.csv`
- `metrics_summary.csv`
- `latency_metrics.csv`
- `token_metrics.csv`
- `raw_responses/` (per-item raw API payloads for auditability)

## Disclaimer

This repository is for evaluation engineering and is not legal advice.

All case law included in packets should come from public, citable sources and be used consistently with any applicable licenses and court rules.
