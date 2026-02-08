import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
PREPARE_PACKETS_PATH = ROOT_DIR / "prepare_packets" / "prepare_packets.py"
SPEC = importlib.util.spec_from_file_location("prepare_packets_script", PREPARE_PACKETS_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_packets_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_packets_script)

format_annotation_block_id = prepare_packets_script.format_annotation_block_id
render_annotated_text = prepare_packets_script.render_annotated_text


def test_format_annotation_block_id_accepts_dotted_token() -> None:
    assert format_annotation_block_id("DOC001.P001.B01") == "DOC001.P001.B01"


def test_format_annotation_block_id_rejects_invalid_token() -> None:
    with pytest.raises(ValueError, match="Invalid citation token"):
        format_annotation_block_id("DOC001[P001.B01]")


def test_render_annotated_text_uses_block_xml_tags() -> None:
    blocks_dataframe = pd.DataFrame(
        [
            {"page_number": 1, "citation_token": "DOC001.P001.B01", "text": "First block text."},
            {"page_number": 1, "citation_token": "DOC001.P001.B02", "text": "Second block text."},
            {"page_number": 2, "citation_token": "DOC001.P002.B01", "text": "Third block text."},
        ]
    )

    annotated_text = render_annotated_text(blocks_dataframe)

    expected_text = (
        "[Page 1]\n"
        "<BLOCK id=\"DOC001.P001.B01\">\n"
        "First block text.\n"
        "</BLOCK>\n"
        "<BLOCK id=\"DOC001.P001.B02\">\n"
        "Second block text.\n"
        "</BLOCK>\n"
        "\n"
        "[Page 2]\n"
        "<BLOCK id=\"DOC001.P002.B01\">\n"
        "Third block text.\n"
        "</BLOCK>\n"
    )
    assert annotated_text == expected_text
