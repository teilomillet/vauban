"""Typed models for config validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.config._types import TomlDict
    from vauban.types import PipelineConfig


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One rendered validation issue."""

    severity: str
    message: str
    fix: str | None = None

    def render(self) -> str:
        """Render one issue using the existing warning format."""
        full = f"[{self.severity}] {self.message}"
        if self.fix is not None:
            full += f" — fix: {self.fix}"
        return full


@dataclass(slots=True)
class ValidationCollector:
    """Collector for structured validation issues."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        message: str,
        *,
        fix: str | None = None,
    ) -> None:
        """Append one typed issue."""
        self.issues.append(
            ValidationIssue(
                severity=severity,
                message=message,
                fix=fix,
            ),
        )

    def extend(self, issues: list[ValidationIssue]) -> None:
        """Append prebuilt issues."""
        self.issues.extend(issues)

    def render(self) -> list[str]:
        """Render collected issues into the legacy string form."""
        return [issue.render() for issue in self.issues]


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Shared validation context."""

    config_path: Path
    raw: TomlDict
    config: PipelineConfig


class ValidationRule(Protocol):
    """Validation rule callable protocol."""

    def __call__(
        self,
        context: ValidationContext,
        collector: ValidationCollector,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ValidationRuleSpec:
    """Typed registration for one validation rule."""

    name: str
    order: int
    rule: ValidationRule
