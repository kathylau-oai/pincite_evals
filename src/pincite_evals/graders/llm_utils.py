import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from pincite_evals.openai_model_capabilities import supports_reasoning_effort
from pincite_evals.prompt_templates import load_template_text, render_template_text

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class LLMJudgeConfig:
    model: str = "gpt-5.2"
    reasoning_effort: str = "none"
    temperature: float = 0.0
    service_tier: str = "priority"


def load_prompt_template(file_name: str) -> str:
    return load_template_text(PROMPTS_DIR / file_name)


def render_prompt_template(prompt_text: str, template_variables: Dict[str, Any]) -> str:
    return render_template_text(prompt_text, template_variables)


def parse_json_object(output_text: str) -> Dict[str, Any]:
    stripped_text = output_text.strip()
    if not stripped_text:
        raise ValueError("LLM output was empty; expected a JSON object.")

    try:
        parsed = json.loads(stripped_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped_text, re.DOTALL)
        if match is None:
            raise ValueError("Could not find JSON object in LLM output.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON was not an object.")
    return parsed


def call_llm_judge(
    *,
    client: OpenAI,
    config: LLMJudgeConfig,
    system_prompt: str,
    user_prompt: str,
    response_schema_name: str | None = None,
    response_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if (response_schema_name is None) != (response_schema is None):
        raise ValueError("response_schema_name and response_schema must be set together.")

    request: Dict[str, Any] = {
        "model": config.model,
        "service_tier": config.service_tier,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if supports_reasoning_effort(config.model):
        request["reasoning"] = {"effort": config.reasoning_effort}
    if response_schema_name is not None and response_schema is not None:
        # Structured outputs ensure judge responses follow the exact JSON contract.
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": response_schema_name,
                "schema": response_schema,
                "strict": True,
            }
        }
    if supports_reasoning_effort(config.model):
        if config.reasoning_effort == "none":
            request["temperature"] = config.temperature
    else:
        # Non-reasoning models reject `reasoning.effort` — fall back to temperature.
        request["temperature"] = config.temperature

    response = client.responses.create(**request)
    output_text = response.output_text or ""

    incomplete_reason: Optional[str] = None
    if response.incomplete_details is not None:
        incomplete_reason = response.incomplete_details.reason
    if response.status == "incomplete" and not output_text.strip():
        raise RuntimeError(
            "LLM response was incomplete and had empty output_text."
        )

    parsed = parse_json_object(output_text)

    usage = None
    if response.usage is not None:
        usage = response.usage.model_dump(mode="json")

    return {
        "response_id": response.id,
        "status": response.status,
        "incomplete_reason": incomplete_reason,
        "usage": usage,
        "raw_output_text": output_text,
        "parsed": parsed,
        "request": request,
    }
