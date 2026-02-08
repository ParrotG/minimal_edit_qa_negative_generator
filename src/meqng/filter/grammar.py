from __future__ import annotations

from dataclasses import dataclass

from .base import FilterDecision


@dataclass
class GrammarFilter:
    """
    Optional grammar filter.

    The dependency is optional and only required when this filter is enabled.
    """

    name: str = "grammar"
    language: str = "en-US"
    max_extra_issues: int = 1

    def __post_init__(self) -> None:
        try:
            import language_tool_python
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "language_tool_python is not installed. Install optional dependency group first (e.g. pip install -e '.[grammar]')."
            ) from exc
        self.tool = language_tool_python.LanguageTool(self.language)

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        chosen_issues = len(self.tool.check(chosen))
        candidate_issues = len(self.tool.check(candidate))
        keep = candidate_issues <= chosen_issues + self.max_extra_issues
        return FilterDecision(
            keep=keep,
            reason="ok" if keep else "too_many_grammar_issues",
            meta={
                "chosen_issues": chosen_issues,
                "candidate_issues": candidate_issues,
                "max_extra_issues": self.max_extra_issues,
            },
        )
