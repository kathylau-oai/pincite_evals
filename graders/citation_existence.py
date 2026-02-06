from typing import Any, Dict

from .base import GradeResult, Grader
from pincite_evals.citations import extract_excerpt_citations


class CitationExistenceGrader(Grader):
    name = "citation_existence"

    def grade(self, *, prompt: str, output: str, context: Dict[str, Any]) -> GradeResult:
        allowed = set(context.get('allowed_sources', []))
        cites = extract_excerpt_citations(output)
        bad = [c.raw for c in cites if c.doc_id not in allowed]
        passed = (len(cites) > 0) and (len(bad) == 0)
        return GradeResult(self.name, passed, {
            'num_citations': len(cites),
            'bad_citations': bad,
        })
