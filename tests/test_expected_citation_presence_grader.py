from pincite_evals.graders.expected_citation_presence import ExpectedCitationPresenceGrader


def test_expected_citation_presence_grader_perfect_match():
    grader = ExpectedCitationPresenceGrader()
    output = (
        "Rule text (DOC001.P001.B01). "
        "Second rule text (DOC002.P002.B03)."
    )
    context = {
        "expected_citations": [
            "DOC001.P001.B01",
            "DOC002.P002.B03",
        ]
    }

    result = grader.grade(prompt="p", output=output, context=context)
    assert result.passed is True
    assert result.details["score"] == 1.0
    assert result.details["missing_groups"] == []
    assert result.details["unexpected_predictions"] == []


def test_expected_citation_presence_grader_missing_and_unexpected():
    grader = ExpectedCitationPresenceGrader()
    output = "Text (DOC001.P001.B01) and extra (DOC009.P004.B09)."
    context = {
        "expected_citations": [
            "DOC001.P001.B01",
            "DOC002.P002.B03",
        ]
    }

    result = grader.grade(prompt="p", output=output, context=context)
    assert result.passed is False
    assert result.details["missing_groups"] == [["DOC002.P002.B03"]]
    assert result.details["unexpected_predictions"] == ["DOC009.P004.B09"]


def test_expected_citation_presence_with_alternative_group():
    grader = ExpectedCitationPresenceGrader()
    output = "Text (DOC001.P001.B02)."
    context = {
        "expected_citation_groups": [
            ["DOC001.P001.B01", "DOC001.P001.B02"],
        ]
    }

    result = grader.grade(prompt="p", output=output, context=context)
    assert result.passed is True
    assert result.details["score"] == 1.0
