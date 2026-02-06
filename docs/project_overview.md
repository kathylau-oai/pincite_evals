# Project Overview

## Core idea

Pincite Evals measures citation overconfidence in legal drafting.  
The evaluation tests whether a model can draft memo-style legal analysis while citing only a provided closed-world packet of US case law.

The central question is: can the model make legally sound claims without fabricating authority, mis-citing text, overextending holdings, or relying on bad precedent?

## Current scope

- Draft type: legal memos (not motions)
- Domain: US case law
- Context source: closed-world packet of case law opinions
- Packet size target: 6-8 documents (often 8)

We start with memos because they are simpler to evaluate and do not require evidence packets.

## What we are trying to evaluate

The dataset is intentionally designed to trigger four key error modes:


| ID  | Error mode                                  | Problem description                                                                                 | Trigger focus                                        |
| --- | ------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| A   | Fabricated citations                        | The prompt asks for authority intentionally absent from the packet, and the model may invent support. | Detect hallucinated authorities under missing-evidence pressure. |
| B   | Wrong source span                           | The model cites nearby text that does not support the claim.                                        | Test paragraph/span-level precision.                 |
| C   | Partially supported but overextended claims | The cited text supports part of the claim, but the final statement overreaches the cited authority. | Test scope control and faithful interpretation.      |
| D   | Precedent / overruling issues               | The cited text supports the claim, but the authority is overruled, superseded, or non-controlling.  | Test legal hierarchy and authority status awareness. |


## Evaluation strategy

We use a mixed grading stack:

- Deterministic graders for structure, allowed-citation checks, and citation parsing/existence.
- LLM-judge graders for harder legal judgment tasks, especially:
  - overextension (claim scope vs supported holding)
  - precedent validity and controlling-authority behavior

This combination gives high precision on objective checks and flexibility on nuanced legal reasoning checks.

Check [graders.md](http://graders.md) to get a better sense.

## How data is created

Case law packets are curated with adversarial intent, not only topical relevance.  
Authorities are selected using iterative deep-research prompting to find opinions with characteristics that can reliably trigger the target error modes.

Each packet typically includes:

- one or more controlling authorities
- at least one overruled/limited/superseded authority
- near-neighbor span traps (adjacent text with similar language but different legal meaning)
- tempting but non-controlling authority
- factually similar but distinguishable authority

From packet metadata, we generate synthetic memo prompts designed to induce specific failures while still being realistic drafting tasks.
For mode `A`, prompts intentionally request famous or specific authorities that are not present in the 6-8 packet documents, so the correct behavior is to acknowledge the gap and avoid fabricated citations.

## Key design decisions


| Consideration            | Decision                                    | Rationale                                                                            |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------ |
| Memos vs motions         | Start with memos                            | Easier setup and cleaner evaluation without evidence packets.                        |
| Context injection        | Inject full case law documents              | Simple initial implementation and realistic agentic behavior over full opinions.     |
| Alternative context mode | Do not start with retrieved chunks          | Chunk-only setups are deferred; full-doc setup is primary for this phase.            |
| Parsing/chunking         | Open-source PDF parser + paragraph chunking | Sufficient for span-level citation checks and close to expected production behavior. |


## Risks and mitigations


| Failure point                                  | Mitigation                                                                                    |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Limited SME bandwidth for full manual curation | Use deep research workflows and verifier loops to generate and validate adversarial examples. |
| Lack of motion-style evidence packets          | Keep initial scope on memos and iterate with technical review feedback.                       |


## High-level workflow

1. Curate case law packets (6-8 docs) with explicit adversarial roles.
2. Extract packet metadata (span traps, qualifiers, precedence edges, controlling hierarchy).
3. Generate synthetic memo prompts targeting specific error modes.
4. Run deterministic + LLM-based graders.
5. Measure signs of life and error-mode separability.
6. Scale dataset and evaluation depth once early quality is confirmed.

## Success criteria (early phase)

- The eval reliably elicits all four error modes.
- Graders can distinguish failure types with low ambiguity.
- Results show meaningful separation across model/prompt variants.
- Outputs are easy to audit from citation text back to packet evidence.
