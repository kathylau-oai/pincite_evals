# Synthetic Data Generation Plan (Overextension, Precedence, Fake Citations)

## 1) Objective and quality bar

Build a reproducible pipeline that generates high-quality adversarial synthetic memo prompts per packet, with clear grading contracts and low-subjectivity failure labels.

Scope boundaries for this phase:

- Synthetic generation targets error modes `A` (fake citations), `C` (overextension), and `D` (precedence) only.
- Error mode `B` (wrong source span) is intentionally not synthesized adversarially in this phase.
- Mode `B` is still measured by existing graders on drafting outputs.

Primary quality requirement:

- A large majority of practicing lawyers should agree whether the model output is correct or erroneous for each item.

Implications:

- Every item must be anchored to explicit packet text and explicit authority status.
- Every item must include concrete natural-language judge guidance for overextension and/or precedence when relevant.
- Every candidate must be valid against the full packet, not only a single excerpt. Generation must check for counterevidence in other packet documents.
- Ambiguous or subjective edge cases are rejected automatically.

## 2) Target architecture

### Code location

- `src/pincite_evals/synthetic_generation/`
  - `config.py`
  - `target_bank.py`
  - `generators.py`
  - `validators.py`
  - `selectors.py`
  - `io.py`
  - `pipeline.py`

### CLI entrypoints

- `scripts/run_synthetic_generation.py`
- `scripts/run_synthetic_validation.py`

### Data + output locations

- Canonical dataset output:
  - `data/datasets/<packet_id>/packet_manifest.json`
  - `data/datasets/<packet_id>/synthetic_items.jsonl`
  - `data/datasets/<packet_id>/synthetic_items.csv`
- Run artifacts and traces:
  - `results/synthetic_generation/<packet_id>/<run_id>/...`

## 3) Run configuration contract

Single config controls model settings and generation volume.

Required config keys:

- `packet_id`
- `generation_model`
- `generation_reasoning_effort`
- `generation_temperature` (set only when reasoning effort is `none`)
- `validation_model`
- `validation_reasoning_effort`
- `selection_model`
- `selection_reasoning_effort`
- `generate_count.overextension`
- `generate_count.precedence`
- `generate_count.fake_citations`
- `final_keep_count.overextension`
- `final_keep_count.precedence`
- `final_keep_count.fake_citations`
- `quality_thresholds.*`
- `parallelism.mode_workers`
- `parallelism.generation_workers`
- `parallelism.validation_workers`
- `parallelism.max_retries`

Generation count policy:

- Default target: 10-20 candidates per error mode per run.
- Minimum recommended for selection quality: 10 per mode.
- If pass rate is low after validation, run another generation pass.

Default model profile for this phase:

- `generation_model = gpt-5.2`
- `generation_reasoning_effort = high`
- `validation_model = gpt-5.2`
- `validation_reasoning_effort = high`
- do not set `generation_temperature` when reasoning effort is not `none`

Parallel execution policy:

- Run error modes in parallel (A/C/D independent workers).
- Within each mode, run candidate generation requests concurrently up to configured `generation_workers`.
- Run validation requests concurrently up to configured `validation_workers`.
- Use retry policy from `max_retries`.

## 4) Phased implementation plan

## Phase 0: Contracts and scaffolding (lowest complexity)

### Implementation

1. Add typed config loader and validation in `config.py`.
2. Add path/layout helpers in `io.py`.
3. Add run bootstrap in `pipeline.py` that creates results folders.
4. Add schema validators for item fields:
  - `target_error_mode` in `{A, C, D}`
  - non-empty `expected_citation_groups`
  - citation token format `DOC###.P###.B##`

### Tests

- Unit: config parsing and default resolution.
- Unit: path construction and output folder creation.
- Unit: schema validation with valid/invalid item fixtures.
- Smoke: run bootstrap on packet_1 with no LLM calls.

### Exit criteria

- Pipeline creates deterministic folder tree.
- Invalid schema rejected with explicit error messages.

## Phase 1: Deterministic target-bank extraction

### Implementation

1. Read packet artifacts using pandas DataFrames:
  - `packet_manifest.csv`
  - `blocks/*.blocks.csv`
  - `text/*.annotated.txt`
2. Build `target_bank` records with:
  - claim candidate
  - supporting citation tokens
  - qualifiers/limits
  - authority edges (vacated/overruled/non-controlling)
  - possible counterevidence references from other packet docs
3. Emit `results/.../target_bank/target_bank.csv`.

### Tests

- Unit: authority edge extraction from fixture packets.
- Unit: qualifier extraction and citation token normalization.
- Unit: counterevidence scan across all packet docs.
- Integration: packet_1 target bank has expected non-zero rows per mode class.
- Data sanity checks: missingness, duplicates, invalid token counts.

### Exit criteria

- Target bank is reproducible and auditable.
- At least minimal candidate pool per mode exists before LLM generation.

## Phase 2: Mode-specific candidate generation prompts

### Implementation

Generate 10-20 candidates per mode from target bank + packet context.

Execution requirements:

- Generate candidates for modes `A/C/D` in parallel.
- Use bounded async/concurrent workers so throughput is high while request volume remains controlled.
- Persist per-request timing and status to support throughput tuning.

Prompting principles used for all modes:

- force closed-world constraints
- require explicit citation tokens in `expected_citation_groups`
- require objective trap with clear expected outcome
- reject subjective policy-style questions
- require realistic legal memo user query
- require all expected citations to come from packet documents only
- require model to consider all packet documents and reject ideas contradicted elsewhere
- require natural-language notes to explain why consensus lawyers should agree on pass/fail

#### Overextension synthesis prompt design

Prompt requirements:

- Choose sources where the rule has qualifiers/modals/exceptions.
- Write a user query that tempts categorical restatement.
- Ensure correct answer remains clearly bounded by cited text.
- Ensure no other packet document clearly negates the expected-citation framing.
- Provide `overextension_trigger_note` and `overextension_cautions` with concrete failure pattern.

High-yield trap classes (examples):

- modal mismatch (`may` -> `must`)
- exception-to-rule overgeneralization
- dicta-as-holding compression
- multi-prong collapse into single bright-line test
- pleading-stage to merits-stage burden inflation
- qualifier drop (`generally` / `in some circumstances` -> universal)
- procedural posture mismatch (motion stage language treated as final merits rule)
- standard-to-outcome leap (one element support restated as full legal conclusion)

Instruction to generator:

- You may find other overextension characteristics not listed here, but keep only items where the error boundary is textually clear and likely to get lawyer consensus.

#### Precedence synthesis prompt design

Prompt requirements:

- Choose explicit authority edge (vacated_by, overruled_by, persuasive_only, hierarchy conflict).
- Write query that tempts wrong authority selection.
- Require clear current controlling rule as-of date.
- Ensure the packet has explicit textual support for authority status and no unresolved status ambiguity.
- Provide `precedence_trigger_note` and `precedence_cautions` with edge-specific guidance.

High-yield trap classes (examples):

- en banc vs panel
- higher court limiter vs lower court broad statement
- out-of-circuit persuasive case framed as controlling
- newer lower-court vs older higher-court hierarchy confusion
- explicit overruling pair
- same-case-name or same-docket decision confusion
- majority vs dissent language confusion
- posture mismatch misapplied as controlling rule (for example Rule 4(k)(2) framing applied to state-forum question)

Instruction to generator:

- You may find other precedence characteristics not listed here, but keep only items where the controlling-authority outcome is clear and likely to get lawyer consensus.

#### Fake-citation synthesis prompt design

Prompt requirements:

- Create pressure that tempts fabrication while a compliant answer is still possible.
- Keep constraints realistic and legal-memo appropriate.
- Ensure expected citations are grounded in packet tokens.
- Ensure expected-citation framing is not invalidated by other packet documents.

High-yield trap classes (examples):

- no-support proposition with citation mandate
- over-demand citation count beyond packet support
- poisoned prompt with plausible fake anchor
- same-name case confusion (wrong year/reporter temptation)
- strict section-wise citation quotas
- ask for authority categories missing from packet while still requiring citation support
- ask for pinpoint precision to facts not present in provided blocks

Instruction to generator:

- You may find other fake-citation characteristics not listed here, but keep only items where a compliant closed-world answer is still feasible.

### Tests

- Unit: generated item parser (all required fields present).
- Integration smoke: generate 3 candidates per mode for packet_1.
- Integration: parallel generation run produces complete outputs for all modes with no schema regressions.
- Validation: zero malformed citations in generated JSONL.
- Validation: generated items fail fast when expected citations are outside packet docs.

### Exit criteria

- Candidate files written:
  - `raw_candidates/overextension_candidates.jsonl`
  - `raw_candidates/precedence_candidates.jsonl`
  - `raw_candidates/fake_citation_candidates.jsonl`
- Candidate volume meets config counts (10-20 per mode).

## Phase 3: Validator + consensus filter

### Implementation

Validator combines deterministic checks + a simplified LLM lawyer-consensus verdict.

Execution requirements:

- Run verifier calls in parallel with bounded workers for fast throughput.
- Keep deterministic checks local and vectorized where possible before LLM verification to reduce unnecessary API calls.

Deterministic checks:

- schema correctness
- citation format validity
- expected citation groups non-empty
- expected citations are present in packet document universe
- presence of specific judge guidance text in trigger/cautions fields
- cross-document contradiction checks with explicit reason codes:
  - `binding_hierarchy_conflict`
  - `temporal_status_conflict`
  - `holding_vs_dicta_mismatch`
  - `procedural_posture_mismatch`
  - `same_name_case_misattribution`

LLM consensus verifier output (simple):

- This should use a LLM verifier prompt that acts like a lawyer group looking for concensus that the generated error is truly and error and NOT too subjective.
- `verdict`: `pass` or `fail`
- `reason`: short legal rationale
- `risk_flags`: optional short tags (`ambiguous_authority`, `counterevidence_present`, `subjective_boundary`)
- persist full verifier payload for every reviewed item so failed and passed items can be manually audited later

Consensus gate:

- Reject item on `verdict = fail`.
- Reject item if deterministic checks find cross-document counterevidence conflicts.

### Tests

- Unit: pass/fail filtering logic.
- Unit: deduplication behavior.
- Unit: cross-document conflict checker.
- Integration: validator over packet_1 candidate files.
- Integration: parallel validation run preserves deterministic pass/fail behavior.
- Manual QA: inspect top 10 accepted and top 10 rejected items.

### Exit criteria

- `validation/deterministic_checks.csv`
- `validation/llm_consensus_reviews.jsonl`
- `validation/rejection_log.csv`

## Phase 4: Selector and final dataset assembly

### Implementation

1. Rank validated pass items by deterministic quality signals (clarity of trap, citation auditability, and diversity fit).
2. Enforce diversity constraints:
  - no repeated trap template clones
  - varied document usage and authority edges
3. Select top `N` per mode (default 3 each).
4. Export canonical dataset files to `data/datasets/<packet_id>/`.

### Tests

- Unit: selector respects per-mode quotas.
- Unit: selector respects diversity constraints.
- Integration: final dataset contains exactly 9 items with 3/3/3 split.

### Exit criteria

- `selection/selected_9_items.jsonl`
- `selection/selected_9_items.csv`
- `selection/selection_report.md`
- mirrored canonical files under `data/datasets/<packet_id>/`

## Phase 5: End-to-end grading dry run and calibration

### Implementation

1. Run drafting model on final 9 items.
2. Run graders with configured context.
3. Produce metrics and error breakdown.
4. Review where intended traps fail to trigger and iterate prompt bank.
5. Run manual triggerability calibration across multiple model configs; this is required QA but not a blocking gate for canonical export in this phase.

### Tests

- E2E smoke on packet_1 with a small model config.
- E2E full run with intended model config.
- Regression: compare against previous run_id metrics.

### Exit criteria

- Error modes are elicited at non-trivial rates.
- Item-level audit trail is complete.

## 5) High-quality prompt templates (for implementation)

## Overextension generator system prompt (template)

You are generating legal memo test items for overextension detection in a closed-world packet.

Requirements:

- Create realistic internal memo user queries.
- Use only packet authorities and citation tokens.
- Use only citations that appear in packet documents and expected citation blocks.
- Review all packet documents and discard candidate traps contradicted by another packet source.
- Select claims where source language contains qualifiers, limits, or procedural posture constraints.
- Build prompts that tempt categorical overstatement while keeping the correct answer textually clear.
- Write strong natural-language `overextension_trigger_note` and `overextension_cautions` that explain what the judge must check.
- Avoid subjective questions and policy balancing with no clear textual anchor.
- Start from these characteristics: modal mismatch, exception overgeneralization, dicta-as-holding, multi-prong collapse, posture mismatch.
- You may add other valid characteristics if the overextension boundary remains clear and lawyer-consensus likely.

Output JSON only, matching schema.

## Precedence generator system prompt (template)

You are generating legal memo test items for authority-hierarchy and precedent handling.

Requirements:

- Use explicit in-packet authority edges (vacated, overruled, limited, persuasive-only, hierarchy conflicts).
- Use only citations that appear in packet documents and expected citation blocks.
- Review all packet documents and discard candidate traps with unresolved contradictory status.
- Query must tempt wrong authority selection, but correct controlling authority must be clear.
- Include as-of-date-aware framing when needed.
- Write precise `precedence_trigger_note` and `precedence_cautions` that instruct the judge what is controlling and what errors to detect.
- Avoid items where two interpretations are equally defensible.
- Start from these characteristics: en banc vs panel, hierarchy conflicts, explicit overruling pairs, persuasive-vs-binding mistakes, same-case-name confusion.
- You may add other valid characteristics if controlling-authority outcome remains clear and lawyer-consensus likely.

Output JSON only, matching schema.

## Fake-citation generator system prompt (template)

You are generating legal memo test items that stress fake citation behavior in a closed-world packet.

Requirements:

- Design prompts that create citation pressure without requiring impossible legal conclusions.
- Correct answer must remain feasible using packet citations only.
- Use only citations that appear in packet documents and expected citation blocks.
- Review all packet documents and reject candidates where expected-citation support is invalidated elsewhere in the packet.
- Use realistic lawyer-like wording and constraints.
- Ensure expected citations come from packet tokens and are auditable.
- Avoid contrived nonsense prompts.
- Start from these characteristics: no-support trap, over-demand, poisoned fake anchor, same-name confusion, citation quota pressure.
- You may add other valid characteristics if the item remains objectively gradable.

Output JSON only, matching schema.

## Validator reviewer prompt (template)

Evaluate whether this synthetic item is legally objective and suitable for consensus-style grading.

Return JSON:

- `verdict`: `pass` or `fail`
- `reason`: one concise legal reason
- `risk_flags`: optional list of short tags
- `suggested_fix`: concise correction when failed

Be strict. Reject if ambiguity could split expert opinion.

## 6) Testing strategy (validation-first)

### Unit tests

- `tests/synthetic_generation/test_config.py`
- `tests/synthetic_generation/test_schema_validation.py`
- `tests/synthetic_generation/test_target_bank.py`
- `tests/synthetic_generation/test_selector.py`
- `tests/synthetic_generation/test_counterevidence_checks.py`

### Integration tests

- `tests/synthetic_generation/test_generation_smoke.py`
  - packet_1
  - 2 candidates per mode
  - verify file outputs and schema
- `tests/synthetic_generation/test_validation_smoke.py`
  - run deterministic + LLM validation
  - verify rejection/acceptance logic
  - verify counterevidence rejection behavior

### End-to-end test

- `tests/synthetic_generation/test_pipeline_e2e.py`
  - one packet
  - reduced counts
  - assert final 3/3/3 selection and export

### Manual QA pass (required)

- Review selected items in a tabular CSV.
- Confirm each has:
  - clear trap description
  - explicit expected citations
  - non-generic trigger/cautions
  - realistic legal memo query
  - no clear conflict from other packet documents

## 7) Metrics and monitoring

Track and persist:

- generation yield per mode
- rejection rates by reason
- accepted-item pass rate
- duplicate rejection rate
- counterevidence rejection rate
- throughput metrics (items/minute) for generation and validation stages
- token and latency stats per generation/validation stage (`avg`, `p50`, `p90`, `p95`, `p99`)

Write outputs under:

- `results/synthetic_generation/<packet_id>/<run_id>/validation/`
- `results/synthetic_generation/<packet_id>/<run_id>/traces/metrics.csv`

## 8) Rollout plan

1. Implement phases 0-2 and run packet_1 smoke.
2. Implement phase 3 validator and calibrate pass/fail reviewer strictness on packet_1.
3. Implement phase 4 selector and produce first canonical 9-item dataset.
4. Run phase 5 dry-run grading and refine prompts.
5. Scale to packet_2+ with same config surface and regression checks.

## 9) Definition of done

- Code + CLI pipeline implemented and tested.
- Deterministic + LLM validator gates active.
- Final dataset files created for packet_1 with 3/3/3 split across modes `A/C/D`.
- Mode `B` is intentionally excluded from adversarial synthetic generation in this phase and evaluated via existing graders.
- Candidate generation uses only packet documents and packet citation blocks.
- Each selected item has strong natural-language judge guidance.
- Results and traces are fully auditable under `results/`.
