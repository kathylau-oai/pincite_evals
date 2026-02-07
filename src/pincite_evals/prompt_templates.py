from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, StrictUndefined


# Use strict undefined variables so template drift fails fast during tests/runs.
PROMPT_TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    undefined=StrictUndefined,
)


def load_template_text(template_path: Path) -> str:
    if not template_path.exists():
        raise ValueError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()


def render_template_text(template_text: str, template_variables: Mapping[str, Any]) -> str:
    template = PROMPT_TEMPLATE_ENVIRONMENT.from_string(template_text)
    return template.render(**dict(template_variables)).strip()


def render_template_file(template_path: Path, template_variables: Mapping[str, Any]) -> str:
    template_text = load_template_text(template_path)
    return render_template_text(template_text, template_variables)
