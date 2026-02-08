from typing import Any, Dict, List, Set

from pincite_evals.citations import extract_excerpt_citations

from .base import GradeResult, Grader


class ExpectedCitationPresenceGrader(Grader):
    name = "expected_citation_presence"

    def _normalize_expected_groups(self, context: Dict[str, Any]) -> List[List[str]]:
        groups = context.get("expected_citation_groups")
        if groups is None:
            expected_citations = context.get("expected_citations", [])
            groups = [[citation] for citation in expected_citations]

        normalized_groups: List[List[str]] = []
        for group in groups:
            if isinstance(group, str):
                cleaned = group.strip()
                if cleaned:
                    normalized_groups.append([cleaned])
                continue

            group_tokens: List[str] = []
            for token in group:
                cleaned_token = str(token).strip()
                if cleaned_token:
                    group_tokens.append(cleaned_token)
            if group_tokens:
                normalized_groups.append(group_tokens)
        return normalized_groups

    def grade(self, *, prompt: str, output: str, context: Dict[str, Any]) -> GradeResult:
        expected_groups = self._normalize_expected_groups(context)
        allow_unexpected_when_no_expected = bool(
            context.get("allow_unexpected_citations_when_no_expected_groups", False)
        )
        expected_token_set: Set[str] = set()
        for group in expected_groups:
            expected_token_set.update(group)

        parsed_citations = extract_excerpt_citations(output)
        predicted_citations = sorted({citation.raw for citation in parsed_citations})
        predicted_set = set(predicted_citations)

        matched_groups: List[List[str]] = []
        missing_groups: List[List[str]] = []
        for group in expected_groups:
            if any(token in predicted_set for token in group):
                matched_groups.append(group)
            else:
                missing_groups.append(group)

        matched_predictions = sorted(token for token in predicted_set if token in expected_token_set)
        unexpected_predictions = sorted(token for token in predicted_set if token not in expected_token_set)

        if expected_groups:
            recall = len(matched_groups) / len(expected_groups)
        else:
            recall = 1.0

        if predicted_citations:
            precision = len(matched_predictions) / len(predicted_citations)
        else:
            precision = 1.0

        pass_threshold = float(context.get("pass_threshold", 1.0))

        if not expected_groups and allow_unexpected_when_no_expected:
            # In this mode, citation presence is not the gating metric when required groups are intentionally empty.
            score = 1.0
            passed = score >= pass_threshold
        else:
            score = (0.7 * recall) + (0.3 * precision)
            passed = (len(missing_groups) == 0) and (len(unexpected_predictions) == 0) and (score >= pass_threshold)

        details = {
            "score": round(score, 6),
            "recall": round(recall, 6),
            "precision": round(precision, 6),
            "pass_threshold": pass_threshold,
            "allow_unexpected_citations_when_no_expected_groups": allow_unexpected_when_no_expected,
            "expected_citation_groups": expected_groups,
            "predicted_citations": predicted_citations,
            "matched_groups": matched_groups,
            "missing_groups": missing_groups,
            "matched_predictions": matched_predictions,
            "unexpected_predictions": unexpected_predictions,
            "num_expected_groups": len(expected_groups),
            "num_predicted_citations": len(predicted_citations),
        }
        return GradeResult(name=self.name, passed=passed, details=details)
