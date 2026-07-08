"""Pydantic data models for The Crucible Arena.

These are for structured output from the LLM: the flaw registry,
critic outputs, sieve results, resolution logs, and referee scorecards.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "moderate", "minor"] # Severity of Each Flaw can be any of the 3

CriticAxis = Literal[
    "technical_architect",
    "security_compliance",
    "cynical_user",
    "finance_business",
] #These are the four critic nodes we have in our workflow


class Flaw(BaseModel):
    """A single vulnerability against the spec."""

    flaw_id: str = Field(description="Unique, stable identifier, e.g. 'F007'.")
    source: CriticAxis = Field(description="Which critic axis raised this flaw.")
    description: str = Field(description="Concise description of the flaw.")
    severity: Severity
    round_raised: int = Field(description="Round number this flaw was first raised.")
    resolved_round: int | None = Field(
        default=None,
        description="Round number this flaw was resolved, or None if still open.",
    )
    is_regression: bool = Field(
        default=False,
        description="True if this flaw is a previously-resolved issue reintroduced.",
    )
    regression_of: str | None = Field(
        default=None,
        description="flaw_id of the original flaw this regresses, if applicable.",
    )


class RawCritique(BaseModel):
    """Raw output from a single critic node."""

    axis: CriticAxis
    findings: list["RawFinding"]
    summary: str = Field(description="One paragraph overall assessment.")


class RawFinding(BaseModel):
    """A single raw finding from a critic, pre-deduplication."""

    description: str
    severity: Severity


RawCritique.model_rebuild()


def merge_critiques(
    left: dict[CriticAxis, RawCritique], right: dict[CriticAxis, RawCritique]
) -> dict[CriticAxis, RawCritique]:
    """LangGraph reducer: merges concurrent critic writes into one dict.

    The 4 critic nodes run in parallel within the same superstep and each
    return {axis: critique}. LangGraph needs a reducer to combine those
    partial updates into a single dict instead of the last write clobbering
    the others.

    A special sentinel value of None (not a dict) signals a full reset —
    used by advance_round_node to clear stale critiques between rounds.
    """
    if right is None:
        return {}
    merged = dict(left)
    merged.update(right)
    return merged


class SievedFlawDraft(BaseModel):
    """LLM-produced draft of a deduplicated flaw, before ID assignment.

    The sieve LLM never invents flaw_ids for new flaws — it only ever
    references existing flaw_ids from the registry it was given, to flag
    regressions. Code assigns real IDs afterward.
    """

    primary_source: CriticAxis = Field(
        description="The critic axis whose wording best captures this issue."
    )
    description: str
    severity: Severity
    duplicate_of_sources: list[CriticAxis] = Field(
        default_factory=list,
        description="Other critic axes that independently raised the same issue.",
    )
    regression_of: str | None = Field(
        default=None,
        description=(
            "If this flaw reintroduces a previously-resolved flaw from the "
            "registry, the exact flaw_id of that original flaw. Otherwise null."
        ),
    )


class SieveResult(BaseModel):
    """Full output of the Data Sieve for a round."""

    sieved_flaws: list[SievedFlawDraft]


class SievedFlaw(BaseModel):
    """A sieved flaw after code has assigned it a stable flaw_id."""

    flaw_id: str
    source: CriticAxis
    description: str
    severity: Severity
    is_regression: bool = False
    regression_of: str | None = None
    duplicate_of_sources: list[CriticAxis] = Field(default_factory=list)


class PatchStep(BaseModel):
    """One step of the sequential micro-patching pipeline."""

    axis: CriticAxis
    resolved_flaw_ids: list[str] = Field(
        description="flaw_ids this patch step claims to resolve."
    )
    change_summary: str = Field(
        description="Concrete description of the spec changes made in this step."
    )
    updated_spec: str = Field(description="Full specification text after this patch.")


class RoundScorecard(BaseModel):
    """Referee's 4-critic axis scoring for a single round."""

    round_number: int
    technical_architect: int = Field(ge=1, le=10)
    security_compliance: int = Field(ge=1, le=10)
    cynical_user: int = Field(ge=1, le=10)
    finance_business: int = Field(ge=1, le=10)
    tradeoff_notes: str = Field(
        description="Notes on any tradeoffs validated or rejected this round."
    )

    def as_dict(self) -> dict[str, int]:
        return {
            "technical_architect": self.technical_architect,
            "security_compliance": self.security_compliance,
            "cynical_user": self.cynical_user,
            "finance_business": self.finance_business,
        }


class RefereeVerdict(BaseModel):
    """Structured referee output."""

    scorecard: RoundScorecard
    tradeoffs_validated: list[str] = Field(default_factory=list)
    tradeoffs_rejected: list[str] = Field(default_factory=list)


class ArenaState(BaseModel):
    """The full, immutable-per-round running state of the arena.

    This entire object is the system's memory. There is no chat history —
    only this structured state, overwritten in place each round.
    """

    idea: str
    specification: str
    round_number: int = 0
    flaw_registry: list[Flaw] = Field(default_factory=list)
    scorecards: list[RoundScorecard] = Field(default_factory=list)
    resolution_log: list["ResolutionEntry"] = Field(default_factory=list)
    next_flaw_seq: int = 1

    # Scratch space for the current round only. The 4 critic nodes fan out
    # and write here concurrently; merge_critiques combines their partial
    # updates. Cleared by advance_round_node at the start of each new round.
    raw_critiques: Annotated[dict[CriticAxis, RawCritique], merge_critiques] = Field(
        default_factory=dict
    )
    sieved_flaws: list["SievedFlaw"] = Field(default_factory=list)

    def open_critical_count(self) -> int:
        return sum(
            1
            for f in self.flaw_registry
            if f.severity == "critical" and f.resolved_round is None
        )

    def score_plateaued(self, delta_threshold: int = 1) -> bool:
        """True if the last two scorecards differ by <= delta_threshold on all axes."""
        if len(self.scorecards) < 2:
            return False
        prev, curr = self.scorecards[-2], self.scorecards[-1]
        pd, cd = prev.as_dict(), curr.as_dict()
        return all(abs(cd[k] - pd[k]) <= delta_threshold for k in cd)


class ResolutionEntry(BaseModel):
    """Traceability record mapping a technical change back to a flaw_id."""

    round_number: int
    flaw_id: str
    axis: CriticAxis
    change_summary: str


ArenaState.model_rebuild()
