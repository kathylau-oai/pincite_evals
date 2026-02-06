# Packet design

A "packet" is the closed-world context for an eval. The guiding goal is not "find cases about X".
It is "find opinions whose structure creates measurable opportunities for models to misstep".

## Recommended roles inside one packet

Use 6 to 8 opinions with explicit roles:

- Controlling authority (highest precedence in the set)
- A case overruled, superseded, or materially limited by a later higher authority
- A "near neighbor" span trap (adjacent paragraphs with similar keywords where only one supports the proposition)
- A rhetorically tempting but non-controlling case
- A factually similar but distinguishable case
- At least one multi-part standard (factors or balancing test)

## Authoring guidance

- Source URLs are stored in `pdf_list.txt` inside each packet folder.
- `prepare_packets/prepare_packets.py` downloads and parses each packet end-to-end.
- Every authority gets a deterministic `doc_id` in URL order: `DOC001`, `DOC002`, ...

## Folder structure (current)

Each packet is self-contained under:

```text
data/case_law_packets/<packet_id>/
  pdf_list.txt
  packet_information.md                  # optional human notes
  pdfs/
    DOC001.pdf
    DOC002.pdf
  blocks/
    DOC001.blocks.csv
    DOC002.blocks.csv
  text/
    DOC001.clean.txt
    DOC001.annotated.txt
  packet_manifest.csv
  parsing_quality_summary.csv
  processing_report.json
```

## Citation/block format (current)

- Citable ID format is block-only: `DOC_ID[P<page>.B<block>]`.
- Legacy paragraph style (`¶`) is not used.
- Hash suffixes are not used.
- Annotated text uses XML block wrappers:
  - `<BLOCK id="DOC001.P012.B07">`
  - `... block text ...`
  - `</BLOCK>`

## Parsing pipeline (current)

1. Read URLs from `pdf_list.txt`.
2. Download PDFs to `pdfs/` (retry + PDF signature validation).
3. Parse layout-aware text with PyMuPDF (`page.get_text("dict", sort=True)`).
4. Build paragraph-like blocks:
   - merge tiny adjacent fragments
   - split long blocks into citable chunks
5. Assign deterministic excerpt IDs by page/block index (`P012.B07`).
6. Write artifacts:
   - block table CSV in `blocks/`
   - plain text and annotated text in `text/` (annotated text uses `BLOCK` tags with `id="DOC###.P###.B##"`)
7. Record packet-level manifests and QA metrics.

## Block table schema (key fields)

`*.blocks.csv` includes:

- `packet_id`, `doc_id`, `source_url`
- `page_number`, `block_number`, `excerpt_id`, `citation_token`
- `word_count`, `token_count`
- `x0`, `y0`, `x1`, `y1` (bbox)
- `text`

## QA metrics we store

Per PDF (`packet_manifest.csv`, `parsing_quality_summary.csv`):

- `page_count`
- `pdf_type` (`native`, `mixed`, `scanned`)
- `block_count`
- `total_words_pdf`
- `total_tokens_pdf`
- `avg_block_words`
- `avg_tokens_per_block`
- `p50_block_words`, `p90_block_words`
- `tiny_block_count_lt_15`
- `large_block_count_gt_160`

Per packet (`processing_report.json`):

- parse/download success counts
- native/mixed/scanned document counts
- average block metrics
- total words/tokens across parsed docs

## Validation workflow

- Start with a smoke test:
  - `python prepare_packets/prepare_packets.py --packets packet_1 packet_2 --max-docs-per-packet 1 --overwrite-downloads`
- Then run full packets:
  - `python prepare_packets/prepare_packets.py --packets packet_1 packet_2 --overwrite-downloads`
