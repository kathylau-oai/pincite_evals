#!/usr/bin/env python3
import argparse
import json
import re
import ssl
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import pymupdf
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

PDF_SIGNATURE = b"%PDF"
TOKENIZER = tiktoken.get_encoding("o200k_base")


def build_ssl_context() -> ssl.SSLContext:
    ssl_context = ssl.create_default_context()
    try:
        import certifi

        ssl_context.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        # Fallback to system trust store when certifi is unavailable.
        pass
    return ssl_context


SSL_CONTEXT = build_ssl_context()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((HTTPError, URLError, TimeoutError, OSError, ValueError, ssl.SSLError)),
    reraise=True,
)
def download_pdf_bytes(source_url: str) -> bytes:
    """Download a PDF and fail fast if the payload is not a PDF file."""
    request = Request(source_url, headers={"User-Agent": "pincite-evals/prepare-packets"})
    with urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        payload = response.read()

    if not payload.startswith(PDF_SIGNATURE):
        raise ValueError(f"Downloaded payload is not a PDF: {source_url}")
    return payload


def parse_pdf_list(packet_dir: Path, max_docs: int | None) -> pd.DataFrame:
    pdf_list_path = packet_dir / "pdf_list.txt"
    if not pdf_list_path.exists():
        raise FileNotFoundError(f"Missing pdf_list.txt in {packet_dir}")

    source_urls: List[str] = []
    for line in pdf_list_path.read_text(encoding="utf-8").splitlines():
        cleaned_line = line.strip()
        if not cleaned_line or cleaned_line.startswith("#"):
            continue
        source_urls.append(cleaned_line)

    if max_docs is not None:
        source_urls = source_urls[:max_docs]

    rows: List[Dict[str, object]] = []
    for index, source_url in enumerate(source_urls, start=1):
        parsed_url = urlparse(source_url)
        source_filename = Path(parsed_url.path).name or f"source_{index:03d}.pdf"
        doc_id = f"DOC{index:03d}"
        rows.append(
            {
                "packet_id": packet_dir.name,
                "doc_id": doc_id,
                "source_url": source_url,
                "source_filename": source_filename,
                "source_order": index,
            }
        )

    return pd.DataFrame(rows)


def validate_pdf_file(pdf_path: Path) -> bool:
    with pdf_path.open("rb") as file_handle:
        header = file_handle.read(4)
    return header.startswith(PDF_SIGNATURE)


def normalize_block_text(block_text: str) -> str:
    text_without_soft_hyphen = block_text.replace("\u00ad", "")
    dehyphenated_text = re.sub(r"(?<=\\w)-\\s*\\n\\s*(?=\\w)", "", text_without_soft_hyphen)
    single_line_text = re.sub(r"\\s*\\n\\s*", " ", dehyphenated_text)
    normalized_text = re.sub(r"\\s+", " ", single_line_text).strip()
    return normalized_text


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def extract_text_blocks_from_page(page_dict: Dict[str, object]) -> List[Dict[str, object]]:
    extracted_blocks: List[Dict[str, object]] = []
    raw_blocks = page_dict.get("blocks", [])

    for block in raw_blocks:
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        line_text_parts: List[str] = []
        for line in block.get("lines", []):
            span_text_parts: List[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", ""))
                if span_text.strip():
                    span_text_parts.append(span_text)
            if span_text_parts:
                line_text_parts.append(" ".join(span_text_parts))

        normalized_text = normalize_block_text("\n".join(line_text_parts))
        block_word_count = count_words(normalized_text)
        if block_word_count == 0:
            continue

        extracted_blocks.append(
            {
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
                "text": normalized_text,
                "word_count": block_word_count,
            }
        )

    extracted_blocks.sort(key=lambda block: (block["y0"], block["x0"]))
    return extracted_blocks


def merge_small_adjacent_blocks(
    page_blocks: List[Dict[str, object]],
    min_words: int = 25,
    max_vertical_gap: float = 28.0,
    same_column_tolerance: float = 80.0,
) -> List[Dict[str, object]]:
    if not page_blocks:
        return []

    merged_blocks: List[Dict[str, object]] = [page_blocks[0].copy()]

    for current_block in page_blocks[1:]:
        previous_block = merged_blocks[-1]
        vertical_gap = float(current_block["y0"]) - float(previous_block["y1"])
        same_column = abs(float(current_block["x0"]) - float(previous_block["x0"])) <= same_column_tolerance

        previous_short = int(previous_block["word_count"]) < min_words
        current_short = int(current_block["word_count"]) < min_words
        previous_text = str(previous_block["text"]).rstrip()
        sentence_continues = previous_text and previous_text[-1] not in {".", "!", "?", ":", ";"}

        should_merge = same_column and vertical_gap <= max_vertical_gap and (
            previous_short or current_short or sentence_continues
        )

        if should_merge:
            merged_text = normalize_block_text(f"{previous_block['text']} {current_block['text']}")
            previous_block["text"] = merged_text
            previous_block["word_count"] = count_words(merged_text)
            previous_block["x0"] = min(float(previous_block["x0"]), float(current_block["x0"]))
            previous_block["y0"] = min(float(previous_block["y0"]), float(current_block["y0"]))
            previous_block["x1"] = max(float(previous_block["x1"]), float(current_block["x1"]))
            previous_block["y1"] = max(float(previous_block["y1"]), float(current_block["y1"]))
        else:
            merged_blocks.append(current_block.copy())

    return merged_blocks


def split_long_blocks(page_blocks: List[Dict[str, object]], max_words: int = 120) -> List[Dict[str, object]]:
    split_blocks: List[Dict[str, object]] = []

    for block in page_blocks:
        block_text = str(block["text"])
        if int(block["word_count"]) <= max_words:
            split_blocks.append(block)
            continue

        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?;])\\s+", block_text) if part.strip()]
        if len(sentence_parts) <= 1:
            words = block_text.split()
            chunk_size = 100
            for chunk_start in range(0, len(words), chunk_size):
                chunk_words = words[chunk_start : chunk_start + chunk_size]
                chunk_text = normalize_block_text(" ".join(chunk_words))
                split_blocks.append(
                    {
                        "x0": block["x0"],
                        "y0": block["y0"],
                        "x1": block["x1"],
                        "y1": block["y1"],
                        "text": chunk_text,
                        "word_count": count_words(chunk_text),
                    }
                )
            continue

        current_chunk_parts: List[str] = []
        current_chunk_word_count = 0

        for sentence in sentence_parts:
            sentence_word_count = count_words(sentence)
            if current_chunk_parts and current_chunk_word_count + sentence_word_count > max_words:
                merged_chunk_text = normalize_block_text(" ".join(current_chunk_parts))
                split_blocks.append(
                    {
                        "x0": block["x0"],
                        "y0": block["y0"],
                        "x1": block["x1"],
                        "y1": block["y1"],
                        "text": merged_chunk_text,
                        "word_count": count_words(merged_chunk_text),
                    }
                )
                current_chunk_parts = [sentence]
                current_chunk_word_count = sentence_word_count
            else:
                current_chunk_parts.append(sentence)
                current_chunk_word_count += sentence_word_count

        if current_chunk_parts:
            merged_chunk_text = normalize_block_text(" ".join(current_chunk_parts))
            split_blocks.append(
                {
                    "x0": block["x0"],
                    "y0": block["y0"],
                    "x1": block["x1"],
                    "y1": block["y1"],
                    "text": merged_chunk_text,
                    "word_count": count_words(merged_chunk_text),
                }
            )

    normalized_split_blocks: List[Dict[str, object]] = []
    for block in split_blocks:
        if int(block["word_count"]) <= max_words:
            normalized_split_blocks.append(block)
            continue

        words = str(block["text"]).split()
        chunk_size = 100
        for chunk_start in range(0, len(words), chunk_size):
            chunk_words = words[chunk_start : chunk_start + chunk_size]
            chunk_text = normalize_block_text(" ".join(chunk_words))
            normalized_split_blocks.append(
                {
                    "x0": block["x0"],
                    "y0": block["y0"],
                    "x1": block["x1"],
                    "y1": block["y1"],
                    "text": chunk_text,
                    "word_count": count_words(chunk_text),
                }
            )

    return normalized_split_blocks


def build_excerpt_id(page_number: int, block_number: int) -> str:
    return f"P{page_number:03d}.B{block_number:02d}"


def render_clean_text(blocks_df: pd.DataFrame) -> str:
    lines: List[str] = []
    for page_number in sorted(blocks_df["page_number"].unique()):
        lines.append(f"[Page {int(page_number)}]")
        page_blocks = blocks_df[blocks_df["page_number"] == page_number]
        for _, row in page_blocks.iterrows():
            lines.append(str(row["text"]))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_annotation_block_id(citation_token: str) -> str:
    # Convert canonical citation token format into a tag-friendly XML block id.
    token_match = re.fullmatch(r"(DOC\d{3})\[P(\d{3})\.B(\d{2})\]", citation_token)
    if token_match is None:
        raise ValueError(f"Invalid citation token for annotation rendering: {citation_token}")

    doc_id, page_number, block_number = token_match.groups()
    return f"{doc_id}.P{page_number}.B{block_number}"


def render_annotated_text(blocks_df: pd.DataFrame) -> str:
    lines: List[str] = []
    for page_number in sorted(blocks_df["page_number"].unique()):
        lines.append(f"[Page {int(page_number)}]")
        page_blocks = blocks_df[blocks_df["page_number"] == page_number]
        for _, row in page_blocks.iterrows():
            block_id = format_annotation_block_id(str(row["citation_token"]))
            lines.append(f'<BLOCK id="{block_id}">')
            lines.append(str(row["text"]))
            lines.append("</BLOCK>")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_pdf_to_blocks(
    packet_id: str,
    doc_id: str,
    source_url: str,
    pdf_path: Path,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    with pymupdf.open(pdf_path) as document:
        block_rows: List[Dict[str, object]] = []
        scanned_like_page_count = 0
        page_count = int(document.page_count)

        for page_index in range(page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)
            page_dict = page.get_text("dict", sort=True)
            page_words = page.get_text("words", sort=True)
            page_word_count = len(page_words)
            image_block_count = sum(1 for block in page_dict.get("blocks", []) if block.get("type") == 1)

            raw_page_blocks = extract_text_blocks_from_page(page_dict)
            merged_page_blocks = merge_small_adjacent_blocks(raw_page_blocks)
            final_page_blocks = split_long_blocks(merged_page_blocks)

            if page_word_count < 20 and image_block_count > 0 and not raw_page_blocks:
                scanned_like_page_count += 1

            for block_number, block in enumerate(final_page_blocks, start=1):
                block_text = str(block["text"])
                excerpt_id = build_excerpt_id(page_number, block_number)
                citation_token = f"{doc_id}[{excerpt_id}]"
                block_rows.append(
                    {
                        "packet_id": packet_id,
                        "doc_id": doc_id,
                        "source_url": source_url,
                        "page_number": page_number,
                        "block_number": block_number,
                        "excerpt_id": excerpt_id,
                        "citation_token": citation_token,
                        "word_count": int(block["word_count"]),
                        "token_count": count_tokens(block_text),
                        "x0": round(float(block["x0"]), 2),
                        "y0": round(float(block["y0"]), 2),
                        "x1": round(float(block["x1"]), 2),
                        "y1": round(float(block["y1"]), 2),
                        "text": block_text,
                    }
                )

    blocks_df = pd.DataFrame(block_rows)
    scanned_ratio = (scanned_like_page_count / page_count) if page_count else 0.0

    if scanned_ratio >= 0.8:
        pdf_type = "scanned"
    elif scanned_ratio > 0:
        pdf_type = "mixed"
    else:
        pdf_type = "native"

    quality_summary: Dict[str, object] = {
        "page_count": page_count,
        "scanned_like_page_count": scanned_like_page_count,
        "scanned_ratio": round(scanned_ratio, 4),
        "pdf_type": pdf_type,
        "block_count": int(len(blocks_df)),
        "total_words_pdf": int(blocks_df["word_count"].sum()) if not blocks_df.empty else 0,
        "total_tokens_pdf": int(blocks_df["token_count"].sum()) if not blocks_df.empty else 0,
        "avg_block_words": round(float(blocks_df["word_count"].mean()), 2) if not blocks_df.empty else 0.0,
        "avg_tokens_per_block": round(float(blocks_df["token_count"].mean()), 2) if not blocks_df.empty else 0.0,
        "p50_block_words": round(float(blocks_df["word_count"].quantile(0.5)), 2) if not blocks_df.empty else 0.0,
        "p90_block_words": round(float(blocks_df["word_count"].quantile(0.9)), 2) if not blocks_df.empty else 0.0,
        "tiny_block_count_lt_15": int((blocks_df["word_count"] < 15).sum()) if not blocks_df.empty else 0,
        "large_block_count_gt_160": int((blocks_df["word_count"] > 160).sum()) if not blocks_df.empty else 0,
    }
    return blocks_df, quality_summary


def process_packet(
    packet_dir: Path,
    max_docs: int | None,
    overwrite_downloads: bool,
) -> Dict[str, object]:
    packet_id = packet_dir.name
    print(f"\\n=== Processing {packet_id} ===")

    pdfs_dir = packet_dir / "pdfs"
    blocks_dir = packet_dir / "blocks"
    text_dir = packet_dir / "text"

    pdfs_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    source_df = parse_pdf_list(packet_dir, max_docs)
    if source_df.empty:
        print(f"No PDF URLs found for {packet_id}. Skipping.")
        return {
            "packet_id": packet_id,
            "documents_total": 0,
            "documents_parsed": 0,
            "documents_failed": 0,
        }

    manifest_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []

    for _, source_row in source_df.iterrows():
        doc_id = str(source_row["doc_id"])
        source_url = str(source_row["source_url"])
        pdf_path = pdfs_dir / f"{doc_id}.pdf"

        download_status = "existing"
        download_error = ""

        should_download = overwrite_downloads or (not pdf_path.exists())
        if should_download:
            try:
                payload = download_pdf_bytes(source_url)
                pdf_path.write_bytes(payload)
                download_status = "downloaded"
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                download_status = "failed"
                download_error = str(error)

        if download_status != "failed" and pdf_path.exists() and not validate_pdf_file(pdf_path):
            download_status = "failed"
            download_error = "File exists but does not have a valid PDF signature"

        parse_status = "skipped"
        parse_error = ""

        blocks_csv_path = blocks_dir / f"{doc_id}.blocks.csv"
        clean_text_path = text_dir / f"{doc_id}.clean.txt"
        annotated_text_path = text_dir / f"{doc_id}.annotated.txt"

        quality_summary: Dict[str, object] = {
            "page_count": 0,
            "scanned_like_page_count": 0,
            "scanned_ratio": 0.0,
            "pdf_type": "unknown",
            "block_count": 0,
            "total_words_pdf": 0,
            "total_tokens_pdf": 0,
            "avg_block_words": 0.0,
            "avg_tokens_per_block": 0.0,
            "p50_block_words": 0.0,
            "p90_block_words": 0.0,
            "tiny_block_count_lt_15": 0,
            "large_block_count_gt_160": 0,
        }

        if download_status != "failed":
            try:
                blocks_df, quality_summary = parse_pdf_to_blocks(packet_id, doc_id, source_url, pdf_path)
                blocks_df.to_csv(blocks_csv_path, index=False)

                clean_text = render_clean_text(blocks_df)
                annotated_text = render_annotated_text(blocks_df)
                clean_text_path.write_text(clean_text, encoding="utf-8")
                annotated_text_path.write_text(annotated_text, encoding="utf-8")
                parse_status = "parsed"
            except (RuntimeError, ValueError, OSError, pymupdf.FileDataError) as error:
                parse_status = "failed"
                parse_error = str(error)

        manifest_rows.append(
            {
                "packet_id": packet_id,
                "doc_id": doc_id,
                "source_order": int(source_row["source_order"]),
                "source_url": source_url,
                "source_filename": str(source_row["source_filename"]),
                "pdf_path": str(pdf_path),
                "download_status": download_status,
                "download_error": download_error,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "blocks_csv_path": str(blocks_csv_path),
                "clean_text_path": str(clean_text_path),
                "annotated_text_path": str(annotated_text_path),
                **quality_summary,
            }
        )

        quality_rows.append(
            {
                "packet_id": packet_id,
                "doc_id": doc_id,
                **quality_summary,
            }
        )

        print(
            f"{doc_id}: download={download_status}, parse={parse_status}, "
            f"pdf_type={quality_summary['pdf_type']}, blocks={quality_summary['block_count']}"
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(by=["source_order"])
    quality_df = pd.DataFrame(quality_rows).sort_values(by=["doc_id"])

    manifest_path = packet_dir / "packet_manifest.csv"
    quality_summary_path = packet_dir / "parsing_quality_summary.csv"

    manifest_df.to_csv(manifest_path, index=False)
    quality_df.to_csv(quality_summary_path, index=False)

    parsed_docs_df = manifest_df[manifest_df["parse_status"] == "parsed"]
    packet_report = {
        "packet_id": packet_id,
        "documents_total": int(len(manifest_df)),
        "documents_parsed": int(len(parsed_docs_df)),
        "documents_failed": int((manifest_df["parse_status"] == "failed").sum()),
        "download_failed": int((manifest_df["download_status"] == "failed").sum()),
        "native_docs": int((manifest_df["pdf_type"] == "native").sum()),
        "mixed_docs": int((manifest_df["pdf_type"] == "mixed").sum()),
        "scanned_docs": int((manifest_df["pdf_type"] == "scanned").sum()),
        "avg_blocks_per_doc": round(float(parsed_docs_df["block_count"].mean()), 2) if not parsed_docs_df.empty else 0.0,
        "avg_block_words": round(float(parsed_docs_df["avg_block_words"].mean()), 2) if not parsed_docs_df.empty else 0.0,
        "avg_tokens_per_block": round(float(parsed_docs_df["avg_tokens_per_block"].mean()), 2) if not parsed_docs_df.empty else 0.0,
        "total_words_all_docs": int(parsed_docs_df["total_words_pdf"].sum()) if not parsed_docs_df.empty else 0,
        "total_tokens_all_docs": int(parsed_docs_df["total_tokens_pdf"].sum()) if not parsed_docs_df.empty else 0,
        "manifest_path": str(manifest_path),
        "quality_summary_path": str(quality_summary_path),
    }

    report_path = packet_dir / "processing_report.json"
    report_path.write_text(json.dumps(packet_report, indent=2), encoding="utf-8")

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved quality summary: {quality_summary_path}")
    print(f"Saved report: {report_path}")

    return packet_report


def discover_packet_dirs(base_dir: Path, selected_packet_ids: List[str] | None) -> List[Path]:
    if selected_packet_ids:
        packet_dirs = [base_dir / packet_id for packet_id in selected_packet_ids]
    else:
        packet_dirs = sorted(path for path in base_dir.glob("packet_*") if path.is_dir())

    existing_packet_dirs = [packet_dir for packet_dir in packet_dirs if packet_dir.exists()]
    return existing_packet_dirs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and parse packet PDFs into citable block text.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/case_law_packets"),
        help="Base directory containing packet_* folders.",
    )
    parser.add_argument(
        "--packets",
        nargs="*",
        default=None,
        help="Optional packet IDs (for example: packet_1 packet_2). Defaults to all packet_* folders.",
    )
    parser.add_argument(
        "--max-docs-per-packet",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Re-download PDFs even if they already exist.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    packet_dirs = discover_packet_dirs(arguments.base_dir, arguments.packets)
    if not packet_dirs:
        raise FileNotFoundError(f"No packet folders found in {arguments.base_dir}")

    packet_reports: List[Dict[str, object]] = []
    for packet_dir in packet_dirs:
        packet_report = process_packet(
            packet_dir=packet_dir,
            max_docs=arguments.max_docs_per_packet,
            overwrite_downloads=arguments.overwrite_downloads,
        )
        packet_reports.append(packet_report)

    run_report = pd.DataFrame(packet_reports)
    run_report_path = arguments.base_dir / "prepare_packets_run_report.csv"
    run_report.to_csv(run_report_path, index=False)

    print("\\n=== Run Summary ===")
    print(run_report.to_string(index=False))
    print(f"Saved run summary: {run_report_path}")


if __name__ == "__main__":
    main()
