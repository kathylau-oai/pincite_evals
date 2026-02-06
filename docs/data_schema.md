# Data Schema

## Goal
Define the synthetic eval input format for closed-world case-law packets.

This schema is designed for adversarial items that target:
- overextension (mode C)
- precedence/authority handling (mode D)
- fake citations (mode A)

Each packet should usually produce 9 items total:
- 3 overextension-focused
- 3 precedence-focused
- 3 fake-citation-focused

## Storage Layout
Use one folder per experiment under `results/` and keep source dataset files under `data/`.

Recommended dataset files per packet:
- `data/datasets/<packet_id>/packet_manifest.json`
- `data/datasets/<packet_id>/synthetic_items.jsonl`

## Packet Manifest Schema (`packet_manifest.json`)
This file is packet-level metadata shared by all item rows.

```json
{
  "schema_version": "v1",
  "packet_id": "packet_1",
  "jurisdiction": {
    "system": "US",
    "forum": "Federal",
    "court_hint": "N.D. Cal. (9th Cir.)",
    "binding_authority_order": ["SCOTUS", "CIRCUIT", "DISTRICT", "STATE_SUPREME", "STATE_APPEALS"]
  },
  "documents": [
    {
      "doc_id": "DOC001",
      "case_name": "Walden v. Fiore",
      "citation": "571 U.S. 277 (2014)",
      "court": "SCOTUS",
      "year": 2014,
      "source_url": "https://www.law.cornell.edu/supct/pdf/12-574.pdf",
      "annotated_text_path": "data/case_law_packets/packet_1/text/DOC001.annotated.txt",
      "blocks_csv_path": "data/case_law_packets/packet_1/blocks/DOC001.blocks.csv"
    }
  ],
  "authority_graph": [
    {
      "lower": "DOC003",
      "higher": "DOC002",
      "relationship": "vacated_by"
    },
    {
      "lower": "DOC006",
      "higher": "DOC002",
      "relationship": "overruled_by"
    }
  ]
}
```

## Item Schema (`synthetic_items.jsonl`)
One JSON object per line.

```json
{
  "schema_version": "v1",
  "item_id": "packet_1_C_01",
  "packet_id": "packet_1",
  "target_error_mode": "C",
  "query_id": "q_0001",
  "as_of_date": "2026-02-06",
  "prompt": "Draft an internal research memo...",
  "scenario_facts": [
    "Plaintiff alleges an agreement in conclusory terms.",
    "Complaint provides few specifics about who, when, and how.",
    "Assume pleading-stage posture."
  ],
  "grading_contract": {
    "expected_citation_groups": [
      ["DOC001[P002.B03]", "DOC001[P002.B05]"],
      ["DOC002[P034.B03]"]
    ],
    "overextension_trigger_note": "This item is built to test whether the model overstates pleading requirements by turning plausibility into a proof requirement. A correct answer should preserve limiting language and avoid demanding evidence at the complaint stage.",
    "overextension_cautions": [
      "Do not reward an answer that drops qualifiers like 'not a probability requirement'.",
      "Be careful not to treat strong but still qualified language as overextension if it remains faithful to the cited rule."
    ],
    "precedence_trigger_note": "This item also checks authority hierarchy: the model should treat current controlling authority as primary and avoid presenting superseded framing as the governing standard.",
    "precedence_cautions": [
      "Do not credit an answer that treats vacated or overruled language as controlling law.",
      "Do not penalize mention of older cases when their limited or superseded status is correctly explained."
    ]
  }
}
```

## Required Item Fields
- `schema_version`: currently `v1`.
- `item_id`: unique row identifier.
- `packet_id`: links item to manifest.
- `target_error_mode`: one of `A`, `C`, `D`.
- `query_id`: stable prompt ID.
- `as_of_date`: date precedence status should be judged against.
- `prompt`: full user task given to the drafting model.
- `scenario_facts`: list of factual assumptions.
- `grading_contract`: judge guidance and expected citations.

## `grading_contract` Fields
Only keep these fields:
- `expected_citation_groups`: list of OR-groups of acceptable citation tokens.
  - Each token must use canonical format: `DOC_ID[P###.B##]`.
  - Each inner list is an OR set.
  - All outer-list groups are required.
- `overextension_trigger_note`: natural-language note for overextension risk in this specific item.
- `overextension_cautions`: natural-language cautions to reduce false positives/negatives.
- `precedence_trigger_note`: natural-language note for authority-hierarchy risk in this specific item.
- `precedence_cautions`: natural-language cautions to reduce false positives/negatives.

No citation-fidelity payload fields are required in this schema.

## System Prompt Output Contract (Required)
Drafting model outputs should be natural language with this section structure:

```markdown
# Question Presented
...

# Brief Answer
...

# Rule / Governing Standard
- Proposition 1 ... (DOC_ID[EXCERPT_ID])
- Proposition 2 ... (DOC_ID[EXCERPT_ID])

# Analysis
...

# Counterarguments / Distinguishing
...

# Recommendation + Risk/Uncertainty
...

# Citations (verbatim list)
1) DOC_ID[EXCERPT_ID] — Case Name, Citation, Loc
2) DOC_ID[EXCERPT_ID] — Case Name, Citation, Loc
```

## Authoring Guidance For Judge Notes
For each item, ensure `overextension_trigger_note` and `precedence_trigger_note` are specific and concrete.

A good note should include:
- what trap the item is testing
- what a correct answer should do
- common incorrect behavior the judge should catch
- what should not be over-penalized

## Validation Rules
- Citation tokens must match `DOC_ID[P###.B##]`.
- `expected_citation_groups` must not be empty.
- `target_error_mode` must be one of `A`, `C`, `D`.
- `scenario_facts` should be non-empty and realistic.
- `as_of_date` should be set explicitly for precedent-sensitive items.
