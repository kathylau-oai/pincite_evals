from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GradeResult:
    name: str
    passed: bool
    details: Dict[str, Any]


class Grader:
    name: str

    def grade(self, *, prompt: str, output: str, context: Dict[str, Any]) -> GradeResult:
        raise NotImplementedError
