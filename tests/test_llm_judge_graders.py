import json
import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from graders.citation_fidelity_llm_judge import CitationFidelityLLMJudgeGrader
from graders.citation_overextension_llm_judge import CitationOverextensionLLMJudgeGrader
from graders.precedence_llm_judge import PrecedenceLLMJudgeGrader


class FakeUsage:
    def model_dump(self, mode: str = "json"):
        return {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


class FakeIncompleteDetails:
    def __init__(self, reason: str):
        self.reason = reason


class FakeResponse:
    def __init__(self, *, output_text: str, status: str = "completed", incomplete_reason: str | None = None):
        self.id = "resp_test_123"
        self.status = status
        self.output_text = output_text
        self.usage = FakeUsage()
        self.incomplete_details = None
        if incomplete_reason is not None:
            self.incomplete_details = FakeIncompleteDetails(incomplete_reason)


class FakeResponsesAPI:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: FakeResponse):
        self.responses = FakeResponsesAPI(response)


def test_citation_fidelity_llm_judge_fails_hallucinated_label():
    response_payload = {
        "overall_score": 0.92,
        "passed": True,
        "summary": "One citation was hallucinated.",
        "item_results": [
            {
                "citation_token": "DOC1[P001.B01#AAAA]",
                "label": "hallucinated",
                "score": 0.0,
                "passed": False,
                "reason": "Block does not support claim.",
                "evidence_excerpt": "",
            }
        ],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationFidelityLLMJudgeGrader(client=fake_client)

    result = grader.grade(
        prompt="p",
        output="o",
        context={
            "citation_fidelity_items": [
                {"citation_token": "DOC1[P001.B01#AAAA]"}
            ]
        },
    )
    assert result.passed is False
    assert result.details["blocked_count"] == 1
    assert fake_client.responses.last_request["model"] == "gpt-5.2"
    assert fake_client.responses.last_request["reasoning"] == {"effort": "none"}
    assert "temperature" in fake_client.responses.last_request


def test_overextension_llm_judge_passes_clean_label():
    response_payload = {
        "label": "no_overextension",
        "score": 0.93,
        "passed": True,
        "reason": "Claims remained within citation bounds.",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationOverextensionLLMJudgeGrader(client=fake_client)

    result = grader.grade(prompt="p", output="o", context={"overextension_trigger_note": "trap note"})
    assert result.passed is True
    assert result.details["label"] == "no_overextension"


def test_precedence_llm_judge_fails_precedence_error():
    response_payload = {
        "label": "precedence_error",
        "score": 0.25,
        "passed": False,
        "reason": "Non-controlling authority treated as controlling.",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = PrecedenceLLMJudgeGrader(client=fake_client)

    result = grader.grade(prompt="p", output="o", context={"precedence_trigger_note": "note"})
    assert result.passed is False
    assert result.details["label"] == "precedence_error"
