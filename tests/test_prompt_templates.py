from pathlib import Path

import pytest
from jinja2 import UndefinedError

from pincite_evals.prompt_templates import load_template_text, render_template_text


def test_render_template_text_injects_variables():
    template_text = "Packet {{ packet_id }} item {{ item_index }}"
    rendered = render_template_text(
        template_text,
        {"packet_id": "packet_1", "item_index": "3"},
    )
    assert rendered == "Packet packet_1 item 3"


def test_render_template_text_raises_for_missing_variable():
    with pytest.raises(UndefinedError):
        render_template_text("Packet {{ packet_id }}", {})


def test_load_template_text_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(ValueError, match="Prompt template not found"):
        load_template_text(tmp_path / "missing_prompt.txt")
