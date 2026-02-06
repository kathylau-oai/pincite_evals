# prepare_packets

This folder contains scripts to download packet PDFs and convert them into citable block-level text.

## Script

- `prepare_packets.py`: end-to-end pipeline
  - reads `pdf_list.txt` in each packet folder
  - downloads PDFs into `pdfs/`
  - parses PDFs with PyMuPDF into block-level text
  - assigns citable IDs like `DOC001[P012.B07]`
  - writes `blocks/*.blocks.csv`
  - writes `text/*.clean.txt` and `text/*.annotated.txt`, where annotated blocks use:
    - `<BLOCK id="DOC001.P012.B07">`
    - `... block text ...`
    - `</BLOCK>`
  - writes packet-level reports (`packet_manifest.csv`, `parsing_quality_summary.csv`, `processing_report.json`)

## Usage

Smoke test (first document in packet 1 and packet 2):

```bash
python prepare_packets/prepare_packets.py --packets packet_1 packet_2 --max-docs-per-packet 1 --overwrite-downloads
```

Full run for packet 1 and packet 2:

```bash
python prepare_packets/prepare_packets.py --packets packet_1 packet_2 --overwrite-downloads
```

Run all packets:

```bash
python prepare_packets/prepare_packets.py --overwrite-downloads
```
