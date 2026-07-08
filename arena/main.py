"""LangGraph orchestrator, Sieve logic, export, and CLI.

This module owns everything that isn't a raw agent call or a data schema:
- Node functions wrapping each agent call, wired into a LangGraph StateGraph.
- Formatting registry/findings text for the sieve and referee agent calls.
- Assigning flaw_ids and writing sieved flaws into the permanent registry.
- The conditional edge implementing dynamic early-stopping.
- The final Markdown portfolio report.
- The CLI entry point.

Graph shape (one round):

    START -> initial_score -> [skip: END] or fan-out to all 4 critics
    critics: critic_technical, critic_security, critic_user, critic_finance
          -> sieve -> patch_technical -> patch_security -> patch_user
          -> patch_finance -> referee -> [loop back | END]

The 4 critic nodes fan out from START (or from the referee's loop-back) and
fan in to sieve, replacing the old asyncio.gather call. The patch nodes
replace the old sequential pipeline loop. The conditional edge after referee
replaces the old Python `for round in range(1, 5)` + early-stop check.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from arena.agents import (
    run_critic,
    run_patch_step,
    run_referee_agent,
    run_sieve_agent,
    synthesize_initial_spec,
)
from arena.state import (
    ArenaState,
    CriticAxis,
    Flaw,
    ResolutionEntry,
    SievedFlaw,
)

MAX_ROUNDS = 4
PLATEAU_DELTA = 1
MIN_SCORE_TO_STOP = 8   # all axes must reach this before early-stop is allowed
INITIAL_SCORE_PASS_THRESHOLD = 8  # if raw spec scores >= this on all axes, skip arena
DEFAULT_OUTPUT_PATH = "crucible_report.md"

AXIS_ORDER: list[CriticAxis] = [
    "technical_architect",
    "security_compliance",
    "cynical_user",
    "finance_business",
]


def _label(axis: str) -> str:
    """'technical_architect' -> 'Technical Architect'."""
    return axis.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Graph nodes: critics (fan-out / fan-in)
# ---------------------------------------------------------------------------


async def critic_node(axis: CriticAxis, state: ArenaState) -> dict:
    """One critic's node function. Returns a partial update to raw_critiques.

    Four of these run concurrently in the same LangGraph superstep (one per
    axis) since they all have an edge from the same source node(s). Their
    outputs are combined by the merge_critiques reducer on ArenaState.
    """
    critique = await run_critic(axis, state.specification, idea=state.idea)
    return {"raw_critiques": {axis: critique}}


async def critic_technical_node(state: ArenaState) -> dict:
    return await critic_node("technical_architect", state)


async def critic_security_node(state: ArenaState) -> dict:
    return await critic_node("security_compliance", state)


async def critic_user_node(state: ArenaState) -> dict:
    return await critic_node("cynical_user", state)


async def critic_finance_node(state: ArenaState) -> dict:
    return await critic_node("finance_business", state)


# ---------------------------------------------------------------------------
# Graph node: Data Sieve (fan-in point)
# ---------------------------------------------------------------------------


def _format_resolved_registry(state: ArenaState) -> str:
    resolved = [f for f in state.flaw_registry if f.resolved_round is not None]
    if not resolved:
        return "(none yet — this is an early round)"
    return "\n".join(
        f"- {f.flaw_id} [{f.source}] (resolved round {f.resolved_round}): "
        f"{f.description}"
        for f in resolved
    )


def _format_raw_findings(state: ArenaState) -> str:
    blocks = []
    for axis in AXIS_ORDER:
        critique = state.raw_critiques.get(axis)
        if critique is None or not critique.findings:
            blocks.append(f"[{axis}] (no findings)")
            continue
        finding_lines = "\n".join(
            f"  - ({finding.severity}) {finding.description}"
            for finding in critique.findings
        )
        blocks.append(f"[{axis}]\n{finding_lines}")
    return "\n\n".join(blocks)


async def sieve_node(state: ArenaState) -> dict:
    """Deduplicate this round's critiques, detect regressions, and register
    the result into the permanent flaw registry.

    Runs once all 4 critic nodes have completed (LangGraph waits for every
    incoming edge before running a node).
    """
    result = await run_sieve_agent(
        resolved_registry_text=_format_resolved_registry(state),
        raw_findings_text=_format_raw_findings(state),
    )

    resolved_ids = {
        f.flaw_id for f in state.flaw_registry if f.resolved_round is not None
    }

    sieved: list[SievedFlaw] = []
    # Assign globally unique IDs by continuing from the current registry
    # sequence. We can't rely on state.new_flaw_id() mutating in-place because
    # LangGraph only persists state through the returned dict — mutations to
    # the state object inside a node are lost. So we track seq locally and
    # return the updated value.
    next_seq = state.next_flaw_seq
    for draft in result.sieved_flaws:
        is_regression = bool(
            draft.regression_of and draft.regression_of in resolved_ids
        )
        flaw_id = f"F{next_seq:03d}"
        next_seq += 1
        sieved.append(
            SievedFlaw(
                flaw_id=flaw_id,
                source=draft.primary_source,
                description=draft.description,
                severity="critical" if is_regression else draft.severity,
                is_regression=is_regression,
                regression_of=draft.regression_of if is_regression else None,
                duplicate_of_sources=draft.duplicate_of_sources,
            )
        )

    new_flaws = [
        Flaw(
            flaw_id=sf.flaw_id,
            source=sf.source,
            description=sf.description,
            severity=sf.severity,
            round_raised=state.round_number,
            is_regression=sf.is_regression,
            regression_of=sf.regression_of,
        )
        for sf in sieved
    ]

    return {
        "sieved_flaws": sieved,
        "flaw_registry": state.flaw_registry + new_flaws,
        "next_flaw_seq": next_seq,
    }


# ---------------------------------------------------------------------------
# Graph nodes: sequential micro-patching pipeline
# ---------------------------------------------------------------------------


def _flaws_for_axis(state: ArenaState, axis: CriticAxis) -> list[SievedFlaw]:
    return [f for f in state.sieved_flaws if f.source == axis]


async def patch_node(axis: CriticAxis, state: ArenaState) -> dict:
    """One stage of the assembly line. No-ops if this axis has no flaws.

    Each patch node reads state.specification as left by the *previous*
    patch node (LangGraph runs these strictly in sequence via add_edge
    chaining, so there's no concurrency to worry about here).
    """
    axis_flaws = _flaws_for_axis(state, axis)
    if not axis_flaws:
        return {}

    step = await run_patch_step(axis, axis_flaws, state.specification)

    resolved_this_round = state.round_number
    updated_registry = [
        f.model_copy(update={"resolved_round": resolved_this_round})
        if f.flaw_id in step.resolved_flaw_ids and f.resolved_round is None
        else f
        for f in state.flaw_registry
    ]
    new_entries = [
        ResolutionEntry(
            round_number=state.round_number,
            flaw_id=flaw_id,
            axis=axis,
            change_summary=step.change_summary,
        )
        for flaw_id in step.resolved_flaw_ids
    ]

    return {
        "specification": step.updated_spec,
        "flaw_registry": updated_registry,
        "resolution_log": state.resolution_log + new_entries,
    }


async def patch_technical_node(state: ArenaState) -> dict:
    return await patch_node("technical_architect", state)


async def patch_security_node(state: ArenaState) -> dict:
    return await patch_node("security_compliance", state)


async def patch_user_node(state: ArenaState) -> dict:
    return await patch_node("cynical_user", state)


async def patch_finance_node(state: ArenaState) -> dict:
    return await patch_node("finance_business", state)


# ---------------------------------------------------------------------------
# Graph node: Referee
# ---------------------------------------------------------------------------


def _format_resolution_log(state: ArenaState) -> str:
    entries = [e for e in state.resolution_log if e.round_number == state.round_number]
    if not entries:
        return "(no changes made this round)"
    return "\n".join(
        f"- [{e.axis}] resolved {e.flaw_id}: {e.change_summary}" for e in entries
    )


async def referee_node(state: ArenaState) -> dict:
    verdict = await run_referee_agent(
        state.specification,
        state.round_number,
        _format_resolution_log(state),
    )
    print(
        f"\n=== Round {state.round_number} complete ===\n"
        f"  Technical Architect:    {verdict.scorecard.technical_architect}/10\n"
        f"  Security & Compliance:  {verdict.scorecard.security_compliance}/10\n"
        f"  Cynical User:           {verdict.scorecard.cynical_user}/10\n"
        f"  Finance & Business:     {verdict.scorecard.finance_business}/10\n"
    )
    return {"scorecards": state.scorecards + [verdict.scorecard]}


# ---------------------------------------------------------------------------
# Graph node + conditional edge: round advance / dynamic early-stopping
# ---------------------------------------------------------------------------


async def advance_round_node(state: ArenaState) -> dict:
    """Bumps the round counter and clears per-round scratch space.

    Clearing raw_critiques is critical: without it, round N+1's critics see
    round N's raw findings still in state (via the merge_critiques reducer),
    which poisons the sieve's regression detection and inflates flaw counts.
    """
    next_round = state.round_number + 1
    print(f"\n--- Starting Round {next_round} ---")
    return {
        "round_number": next_round,
        "raw_critiques": None,  # sentinel: merge_critiques resets to {} on None
        "sieved_flaws": [],
    }


def route_after_referee(state: ArenaState) -> str:
    """Conditional edge: loop back to the critics, or stop.

    Stops early only when ALL of these are true:
    - No open critical flaws remain
    - All axis scores are >= MIN_SCORE_TO_STOP (not just plateaued at any level)
    - Scores have plateaued (<=PLATEAU_DELTA point delta) over the last 2 rounds
    Otherwise loops, capped at MAX_ROUNDS.
    """
    if state.round_number >= MAX_ROUNDS:
        return "stop"
    if state.open_critical_count() > 0:
        return "loop"
    # Check minimum score threshold — don't stop if any axis is still weak
    if state.scorecards:
        last = state.scorecards[-1]
        scores = last.as_dict()
        if any(s < MIN_SCORE_TO_STOP for s in scores.values()):
            return "loop"
    if state.score_plateaued(PLATEAU_DELTA):
        return "stop"
    return "loop"


# ---------------------------------------------------------------------------
# Graph node: Initial spec scorer (pre-arena gate)
# ---------------------------------------------------------------------------


async def initial_score_node(state: ArenaState) -> dict:
    """Score the raw PM spec before any critics touch it (round 0).

    This is a fast gate: if the initial spec is already strong on all axes
    (all scores >= INITIAL_SCORE_PASS_THRESHOLD), there is no point running
    the full critic/patch/referee pipeline. The route_after_initial_score
    conditional edge will short-circuit to END in that case.

    Uses the same referee agent but with round_number=0 and an empty
    resolution log so the referee knows this is a baseline assessment.
    """
    print("  [initial referee] Scoring raw specification...")
    verdict = await run_referee_agent(
        state.specification,
        round_number=0,
        resolution_log_text="(initial spec — no changes made yet)",
    )
    # Use round_number=0 to mark this as the baseline scorecard
    verdict.scorecard.round_number = 0
    sc = verdict.scorecard
    scores = sc.as_dict()
    min_score = min(scores.values())
    print(
        f"\n=== Initial Spec Score (Round 0) ===\n"
        f"  Technical Architect:    {sc.technical_architect}/10\n"
        f"  Security & Compliance:  {sc.security_compliance}/10\n"
        f"  Cynical User:           {sc.cynical_user}/10\n"
        f"  Finance & Business:     {sc.finance_business}/10\n"
        + (
            f"  → All axes >= {INITIAL_SCORE_PASS_THRESHOLD}. "
            f"Spec passes gate — skipping arena.\n"
            if min_score >= INITIAL_SCORE_PASS_THRESHOLD
            else f"  → Weakest axis: {min_score}/10. Entering arena.\n"
            + "\n--- Starting Round 1 ---"
        )
    )
    return {"scorecards": state.scorecards + [sc]}


def route_after_initial_score(state: ArenaState) -> str:
    """If the raw spec is already strong enough, skip the full arena."""
    if not state.scorecards:
        return "enter_arena"
    last = state.scorecards[-1]
    scores = last.as_dict()
    if all(s >= INITIAL_SCORE_PASS_THRESHOLD for s in scores.values()):
        return "skip"
    return "enter_arena"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Wire all nodes into the compiled LangGraph StateGraph."""
    graph = StateGraph(ArenaState)

    graph.add_node("initial_score", initial_score_node)
    graph.add_node("critic_technical", critic_technical_node)
    graph.add_node("critic_security", critic_security_node)
    graph.add_node("critic_user", critic_user_node)
    graph.add_node("critic_finance", critic_finance_node)
    graph.add_node("sieve", sieve_node)
    graph.add_node("patch_technical", patch_technical_node)
    graph.add_node("patch_security", patch_security_node)
    graph.add_node("patch_user", patch_user_node)
    graph.add_node("patch_finance", patch_finance_node)
    graph.add_node("referee", referee_node)
    graph.add_node("advance_round", advance_round_node)

    critic_nodes = [
        "critic_technical",
        "critic_security",
        "critic_user",
        "critic_finance",
    ]

    # START -> initial_score (baseline gate before critics)
    graph.add_edge(START, "initial_score")

    # initial_score: if the raw spec is already strong, skip straight to END.
    # Otherwise fan out to all 4 critics. A conditional edge can only name
    # one destination per branch value, so we route through a single
    # "enter_arena" no-op node that then fans out via normal edges.
    graph.add_node("enter_arena", lambda state: {})
    graph.add_conditional_edges(
        "initial_score",
        route_after_initial_score,
        {"enter_arena": "enter_arena", "skip": END},
    )
    for node in critic_nodes:
        graph.add_edge("enter_arena", node)
        # Fan-in: all 4 critics -> sieve. LangGraph waits for all of them.
        graph.add_edge(node, "sieve")

    # Sequential assembly line, fixed axis order.
    graph.add_edge("sieve", "patch_technical")
    graph.add_edge("patch_technical", "patch_security")
    graph.add_edge("patch_security", "patch_user")
    graph.add_edge("patch_user", "patch_finance")
    graph.add_edge("patch_finance", "referee")

    # Conditional loop: referee -> advance_round -> critics again, or END.
    graph.add_conditional_edges(
        "referee", route_after_referee, {"loop": "advance_round", "stop": END}
    )
    for node in critic_nodes:
        graph.add_edge("advance_round", node)

    return graph.compile()


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------


async def run_arena(idea: str) -> ArenaState:
    """Run the full Crucible Arena graph and return the final state."""
    print("Synthesizing initial specification and entering the arena...\n")
    initial_spec = await synthesize_initial_spec(idea)
    state = ArenaState(idea=idea, specification=initial_spec, round_number=1)

    graph = build_graph()
    final_state = await graph.ainvoke(state, config={"recursion_limit": 100})
    return ArenaState.model_validate(final_state)


# ---------------------------------------------------------------------------
# Portfolio Artifact Exporter
# ---------------------------------------------------------------------------


def _score_table(state: ArenaState) -> str:
    if not state.scorecards:
        return "_No scorecards recorded._"
    header = (
        "| Round | Technical Architect | Security & Compliance | Cynical User | "
        "Finance & Business |\n"
        "|---|---|---|---|---|"
    )
    rows = [
        f"| {'Initial' if sc.round_number == 0 else sc.round_number} "
        f"| {sc.technical_architect} | {sc.security_compliance} "
        f"| {sc.cynical_user} | {sc.finance_business} |"
        for sc in state.scorecards
    ]
    return "\n".join([header, *rows])


def _resolution_timeline(state: ArenaState) -> str:
    """Build a deduplicated resolution timeline grouped by patch step.

    Each patch step resolves a batch of flaws with a single change_summary.
    Rather than repeating the same summary once per flaw_id (which makes the
    report bloated and hard to read), we group entries by (round, axis,
    change_summary) and list all resolved flaw_ids together.
    """
    if not state.resolution_log:
        return "_No resolutions recorded._"

    # Group entries by (round_number, axis, change_summary)
    groups: dict[tuple, list[str]] = defaultdict(list)
    regression_flaw_ids: set[str] = {
        f.flaw_id for f in state.flaw_registry if f.is_regression
    }
    flaw_rounds: dict[str, int] = {f.flaw_id: f.round_raised for f in state.flaw_registry}

    for entry in state.resolution_log:
        key = (entry.round_number, entry.axis, entry.change_summary)
        groups[key].append(entry.flaw_id)

    lines = []
    for (round_num, axis, summary), flaw_ids in groups.items():
        regression_ids = [fid for fid in flaw_ids if fid in regression_flaw_ids]
        regression_tag = f" ⚠️ REGRESSIONS: {', '.join(regression_ids)}" if regression_ids else ""
        raised_info = ", ".join(
            f"{fid}(r{flaw_rounds.get(fid, '?')})" for fid in flaw_ids
        )
        lines.append(
            f"- **Round {round_num} · {_label(axis)}**{regression_tag}\n"
            f"  Resolved: {raised_info}\n"
            f"  Change: {summary}"
        )
    return "\n\n".join(lines)


def _open_flaws_section(state: ArenaState) -> str:
    open_flaws = [f for f in state.flaw_registry if f.resolved_round is None]
    if not open_flaws:
        return "_All flaws were resolved. The specification is clean._"
    return "\n".join(
        f"- **{f.flaw_id}** [{f.severity}] ({_label(f.source)}, "
        f"raised round {f.round_raised}): {f.description}"
        for f in open_flaws
    )


def build_report(state: ArenaState) -> str:
    """Build the full Markdown portfolio report from final arena state."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_flaws = len(state.flaw_registry)
    resolved_flaws = sum(
        1 for f in state.flaw_registry if f.resolved_round is not None
    )
    regressions = sum(1 for f in state.flaw_registry if f.is_regression)

    return f"""# The Crucible Arena — Final Report

Generated: {generated_at}
Rounds completed: {state.round_number}

## Original Idea

> {state.idea}

## Optimization Trajectory

{_score_table(state)}

## Flaw Registry Summary

- Total flaws raised: {total_flaws}
- Flaws resolved: {resolved_flaws}
- Regressions caught: {regressions}
- Open critical flaws remaining: {state.open_critical_count()}

## Resolution Traceability Timeline

{_resolution_timeline(state)}

## Remaining Open Flaws

{_open_flaws_section(state)}

## Final Battle-Hardened Specification

{state.specification}
"""


def write_report(state: ArenaState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_report(state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _run_cli(idea: str) -> None:
    state = await run_arena(idea)
    write_report(state, DEFAULT_OUTPUT_PATH)
    print(f"\nArena complete after {state.round_number} round(s).")
    print(f"Report written to {DEFAULT_OUTPUT_PATH}")


def main() -> None:
    idea = input("Enter your one-paragraph product idea:\n> ").strip()
    if not idea:
        print("No idea provided. Exiting.")
        return

    try:
        asyncio.run(_run_cli(idea))
    except RuntimeError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
