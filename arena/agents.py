"""System instructions and structured agent calls.

Prompts were written with the help of LLMs.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from ddgs import DDGS
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from arena.state import (
    CriticAxis,
    PatchStep,
    RawCritique,
    RefereeVerdict,
    SievedFlaw,
    SieveResult,
)

T = TypeVar("T", bound=BaseModel)

MODEL = os.environ.get("CRUCIBLE_MODEL", "gpt-4o-mini")
GUARDRAILS_PATH = Path(__file__).parent / "data" / "tech_guardrails.txt"
SEARCH_MAX_RESULTS = 5



# Cached per temperature since we'd otherwise create a new client per call.
@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.4) -> BaseChatModel:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(model=MODEL, temperature=temperature, api_key=api_key)



async def _invoke_structured(chain: Runnable, payload: dict, schema_name: str) -> T:
    """Invoke a chain ending in .with_structured_output() and validate the result.

    Some providers return None instead of raising when they can't parse the
    model's output into the target schema — this turns that into a clear error.
    """
    result = await chain.ainvoke(payload)
    if result is None:
        raise RuntimeError(
            f"Model returned no parsable {schema_name} output. "
            f"Confirm {MODEL} supports structured/tool-calling output."
        )
    return result


# Loads data/tech_guardrails.txt for critics to reference as real-world benchmarks.
@lru_cache(maxsize=1)
def load_guardrails() -> str:
    if not GUARDRAILS_PATH.exists():
        return ""
    return GUARDRAILS_PATH.read_text(encoding="utf-8")



# Web search so critics ground findings in real data instead of hallucinating.
def _search_web(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=SEARCH_MAX_RESULTS))
    except Exception:
        return ""
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()[:200]
        href = r.get("href", "")
        lines.append(f"- [{title}]({href})\n  {body}")
    return "\n".join(lines)


async def _search_web_async(query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_web, query)

#build the websearch queries, one templated list per critic axis
SEARCH_QUERY_TEMPLATES: dict[CriticAxis, list[str]] = {
    "technical_architect": [
        "{snippet} build complexity integration challenges developer",
        "{snippet} platform API restrictions technical limitations 2024 2025",
    ],
    "security_compliance": [
        "{snippet} Google Play Store policy restrictions banned",
        "{snippet} GDPR data privacy compliance requirements",
        "{snippet} app security vulnerabilities data breach risks",
    ],
    "cynical_user": [
        "{snippet} best apps alternatives 2025 comparison",
        "{snippet} app user complaints reddit why switched uninstalled",
        "why users quit {snippet} apps switching cost barriers",
    ],
    "finance_business": [
        "{snippet} app business model revenue unit economics CAC LTV",
        "{snippet} startup go to market strategy user acquisition 2024 2025",
    ],
}


def _build_search_queries(idea: str, axis: CriticAxis) -> list[str]:
    snippet = idea[:80]
    return [t.format(snippet=snippet) for t in SEARCH_QUERY_TEMPLATES.get(axis, [])]

async def _gather_search_context(idea: str, axis: CriticAxis) -> str:
    queries = _build_search_queries(idea, axis)
    results = await asyncio.gather(
        *[_search_web_async(q) for q in queries],
        return_exceptions=True,
    )
    blocks = []
    for query, result in zip(queries, results):
        if isinstance(result, Exception) or not result:
            continue
        blocks.append(f'Search: "{query}"\n{result}')
    return "\n\n".join(blocks) if blocks else "(no web search results available)"


#_______________________Optimistic Product Manager ____________________________________________________

PM_SYSTEM_PROMPT = """\
You are an Optimistic Product Manager. You take a raw, one-paragraph product \
idea and expand it into a formal, detailed, and genuinely exciting initial \
product specification. You are optimistic and thorough, not critical — that \
is someone else's job later.

Write the specification in clean Markdown with these sections:
1. Product Name & One-Line Pitch
2. Problem Statement
3. Target Users
4. Core Features (bulleted, specific)
5. Proposed Architecture (high-level: client, backend, data stores, any \
   third-party APIs/services you propose using)
6. User Flow (step by step, from first open to core value delivered)
7. Monetization / Business Model
8. Success Metrics

Be specific and concrete rather than vague. Invent reasonable technical and \
business details where the user did not specify them, since this is a first \
draft that will be stress-tested and revised later. Output only the Markdown \
document, no preamble.
"""

PM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PM_SYSTEM_PROMPT),
    ("human", "Here is the raw product idea:\n\n{idea}\n\n"
              "Produce the full initial product specification document."),
])


async def synthesize_initial_spec(idea: str) -> str:
    print("  [PM] Drafting initial specification...")
    chain = PM_PROMPT | get_llm(temperature=0.4)
    response = await chain.ainvoke({"idea": idea})
    return response.content

#_______________________________ Critics ____________________________________________________

CRITIC_SYSTEM_PROMPTS: dict[CriticAxis, str] = {
    "technical_architect": (
        "You are a Senior Technical Architect and Engineering Lead reviewing a "
        "product specification. Your domain covers two equally important questions:\n\n"
        "1. CAN THIS BE BUILT? Assess real-world build complexity: Are the "
        "proposed integrations achievable with available APIs and platforms? "
        "Are there platform restrictions that block the approach (e.g. OS "
        "limitations, app store technical policies, hardware constraints)? "
        "Does the proposed stack match the features being built? Would a "
        "realistic engineering team be able to ship this without exotic expertise?\n\n"
        "2. CAN IT SCALE AND OPERATE RELIABLY? Infrastructure scaling limits, "
        "single points of failure, data consistency across devices/services, "
        "network protocol choices, latency requirements, and operational "
        "reliability at realistic user loads.\n\n"
        "Before flagging a flaw, read the spec carefully — if it already "
        "addresses the concern, do not raise it. Use the search data to cite "
        "real platform restrictions, known integration challenges, or benchmarks. "
        "Ignore UX, legal/compliance, and business model concerns.\n\n"
        "Severity: critical = blocks shipping or causes production failure; "
        "moderate = significant rework needed or reliability risk; minor = polish."
    ),
    "security_compliance": (
        "You are a Security Engineer and Legal/Compliance Auditor reviewing a "
        "product specification. Your domain covers:\n\n"
        "SECURITY: Authentication and authorization design, PII data handling, "
        "encryption at rest and in transit, attack surface (XSS, injection, "
        "MITM, unauthorized access), secrets management, third-party data "
        "sharing risks, and what happens if the product is breached.\n\n"
        "COMPLIANCE & LEGAL: Data privacy laws (GDPR, CCPA — where does user "
        "data live and who can access it?), platform policies (Google Play, "
        "Apple App Store, Microsoft Store restrictions that could get the app "
        "rejected or banned), platform API terms of service (are the proposed "
        "integrations allowed under the platform's ToS?), age-related laws "
        "(COPPA if children might use this), and any domain-specific regulations.\n\n"
        "Use the search data to check real platform policies and legal requirements. "
        "A product that violates Google Play policy or GDPR at the spec stage "
        "is a showstopper — flag it as critical. Ignore infrastructure scaling, "
        "UX, and business model concerns.\n\n"
        "Severity: critical = product cannot ship or faces legal liability; "
        "moderate = requires significant design changes to comply; minor = best-practice gap."
    ),
    "cynical_user": (
        "You are a real person evaluating this product spec honestly. "
        "Think through three questions:\n\n"
        "1. WOULD I USE THIS? Walk through the actual user flow. At which "
        "specific step would you lose interest or abandon the app? Be honest.\n\n"
        "2. WHAT WOULD I USE INSTEAD? Use the search results to name specific "
        "existing apps. For each key feature, ask whether a named competitor "
        "already does it better. If the spec explains a clear switching cost "
        "story or differentiation, acknowledge it — don't invent a problem "
        "that the spec already addresses.\n\n"
        "3. IS THIS PRACTICAL? Would a non-technical user realistically set "
        "this up? Are the permissions and setup steps reasonable? Does the "
        "product solve a real problem people actually have?\n\n"
        "Focus on user experience, practicality, and competitive differentiation. "
        "Ignore infrastructure, legal, and financial concerns. Be honest rather "
        "than artificially harsh — if something is genuinely good, say so.\n\n"
        "Severity: critical = users would not adopt or would quickly uninstall; "
        "moderate = meaningful friction that hurts retention; minor = polish."
    ),
    "finance_business": (
        "You are a CFO and Head of Growth reviewing the financial and business "
        "viability of this product spec. Your domain covers two areas:\n\n"
        "UNIT ECONOMICS: CAC vs LTV math, third-party API and infrastructure "
        "costs at realistic user volumes, payment processing fees, path to "
        "break-even, and whether the monetization model can actually sustain "
        "the product. Be specific with dollar estimates.\n\n"
        "GO-TO-MARKET REALISM: Is the target market reachable with the described "
        "strategy? Is the CAC assumption realistic for the acquisition channel? "
        "Does the product have a distribution story — how do the first 1,000 "
        "users actually find it? Is the competitive landscape so crowded that "
        "the go-to-market needs a clear wedge?\n\n"
        "Read the spec carefully before flagging a concern — if it already "
        "addresses it with specific numbers or strategy, don't raise it. Only "
        "flag genuine gaps. Use the search data for market sizing and competitor "
        "pricing benchmarks. Ignore infrastructure and UX concerns.\n\n"
        "Severity: critical = business model is fundamentally unviable; "
        "moderate = margins at risk or GTM strategy needs major rethink; "
        "minor = optimization opportunity."
    ),
}

CRITIC_USER_TEMPLATE = """\
FACTUAL GUARDRAILS (real-world benchmarks — reference where relevant):
{guardrails}

REAL-WORLD RESEARCH (use this to ground your findings in reality):
{search_context}

Read the specification below carefully before writing any findings. If the spec \
already addresses a concern, do not raise it as a flaw. Only flag issues that \
are genuinely present in the current spec text.

Where the search results contain relevant data (competitor names, pricing, \
benchmarks, user complaints), use that data to support your findings. Do not \
manufacture findings just to reference the search data.

Return a list of distinct findings (each with a concrete description and a \
severity of critical/moderate/minor) plus a one-paragraph verdict on whether \
this spec is viable in your domain as written.

--- SPECIFICATION ---
{specification}
--- END SPECIFICATION ---
"""


def _critic_prompt(axis: CriticAxis) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPTS[axis]),
        ("human", CRITIC_USER_TEMPLATE),
    ])


async def run_critic(axis: CriticAxis, specification: str, idea: str = "") -> RawCritique:
    print(f"  [{axis}] Searching for context...")
    search_context = await _gather_search_context(idea or specification[:100], axis)

    print(f"  [{axis}] Running critique...")
    llm = get_llm(temperature=0.4).with_structured_output(RawCritique)
    chain = _critic_prompt(axis) | llm
    critique: RawCritique = await _invoke_structured(
        chain,
        {
            "guardrails": load_guardrails(),
            "search_context": search_context,
            "specification": specification,
        },
        "RawCritique",
    )
    critique.axis = axis
    return critique


#_______________________________ Data Sieve ________________________________________________________

SIEVE_SYSTEM_PROMPT = """\
You are the Data Sieve: a precise deduplication and regression-detection \
engine for a product-critique pipeline. You receive raw findings from four \
independent critics (who cannot see each other's work) plus a registry of \
previously RESOLVED flaws from earlier rounds.

Your job, purely as data processing, not critique:
1. Merge findings from different critics that describe the SAME underlying \
   issue into one entry. Record which axes independently raised it in \
   duplicate_of_sources, and pick the clearest description.
2. For every merged/unique finding, check it against the RESOLVED FLAW \
   REGISTRY. A finding is a regression ONLY IF the exact same root-cause \
   problem was already fixed in a prior round and has now been reintroduced \
   — i.e. a patch actively removed or reversed a previous fix. \
   A finding that adds MORE DETAIL to an already-patched area is NOT a \
   regression — it is a new finding. Only set regression_of when the \
   identical problem is back, not when a critic is drilling deeper into \
   the same component.
3. Do not invent new findings. Do not drop findings. Every distinct issue \
   from the raw critiques must appear exactly once in your output.
"""

SIEVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SIEVE_SYSTEM_PROMPT),
    ("human",
     "RESOLVED FLAW REGISTRY (previously fixed issues — watch for "
     "regressions):\n{resolved_registry}\n\n"
     "RAW CRITIC FINDINGS THIS ROUND:\n{raw_findings}\n\n"
     "Return the deduplicated, regression-checked list of sieved flaws."),
])


async def run_sieve_agent(
    resolved_registry_text: str, raw_findings_text: str
) -> SieveResult:
    print("  [sieve] Deduplicating and checking regressions...")
    llm = get_llm(temperature=0.1).with_structured_output(SieveResult)
    chain = SIEVE_PROMPT | llm
    return await _invoke_structured(
        chain,
        {
            "resolved_registry": resolved_registry_text,
            "raw_findings": raw_findings_text,
        },
        "SieveResult",
    )


#___________________________________Patching Agent_________________________________________

PATCH_SYSTEM_PROMPT = """\
You are a focused specification editor. You will be given a product \
specification and a list of flaws from ONE specific domain only. Your sole \
job is to revise the specification to resolve those flaws — do not touch \
unrelated sections, do not attempt to fix issues outside the given list, and \
do not remove content that isn't related to these flaws.

Make concrete, specific edits (add architecture details, change technical \
choices, adjust pricing/monetization, restructure a flow, etc.) — do not \
just add vague caveats. If resolving a flaw requires a tradeoff (e.g. adding \
cost to fix a reliability issue), make the tradeoff explicitly and note it in \
the spec text itself.

CONSISTENCY CHECK: Before applying any change, scan the existing specification \
for architectural decisions already in place. Do NOT introduce contradictions. \
If the spec already says "multi-region active-active deployment", do not switch \
to "single-region". If the spec already says "Redis caching with LRU eviction", \
do not propose a different caching strategy — extend the existing one instead. \
Contradicting a prior decision is worse than leaving a flaw open.

Return the change_summary describing exactly what you changed and why, the \
list of flaw_ids you resolved, and the FULL updated specification text \
(the entire document, not just a diff).
"""

PATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PATCH_SYSTEM_PROMPT),
    ("human",
     "DOMAIN: {axis}\n\n"
     "FLAWS TO RESOLVE (only these — ignore all others):\n{flaws}\n\n"
     "CURRENT SPECIFICATION:\n{specification}"),
])


def _format_flaws(flaws: list[SievedFlaw]) -> str:
    return "\n".join(f"- {f.flaw_id} ({f.severity}): {f.description}" for f in flaws)


async def run_patch_step(
    axis: CriticAxis, flaws: list[SievedFlaw], specification: str
) -> PatchStep:
    print(f"  [patch/{axis}] Resolving {len(flaws)} flaw(s)...")
    llm = get_llm(temperature=0.3).with_structured_output(PatchStep)
    chain = PATCH_PROMPT | llm
    return await _invoke_structured(
        chain,
        {
            "axis": axis,
            "flaws": _format_flaws(flaws),
            "specification": specification,
        },
        "PatchStep",
    )


#________________________________ Referee (not biased like fifa)_____________________________________

REFEREE_SYSTEM_PROMPT = """\
You are the Referee: an impartial senior reviewer scoring a product \
specification stress-test. You score the CURRENT specification from 1-10 \
across four axes. Use this scale strictly — scores are relative to a \
production-ready, fully-validated specification:

  1-3  Broken or missing entirely. Would not survive a real design review.
  4-5  Significant structural gaps. Major rework needed before this is viable.
  6    Covers the basics but has concrete exploitable weaknesses. Needs work.
  7    Solid foundation with real but addressable gaps. Acceptable, not great.
  8    Genuinely strong. Minor issues only, none of which would cause failure.
  9    Near-production quality. Very few specs reach this legitimately.
  10   Reserved for exhaustive, fully-validated, battle-tested specifications.

Expected score range by round — enforce this calibration:
- Round 0 (raw PM draft, no patches): scores of 3-5 are typical. A 6 \
  requires unusually concrete detail. Scores of 7+ are almost always inflation.
- Round 1-2 (partially patched): 5-7 is typical. An 8 requires the spec to \
  have explicitly addressed the relevant risks with working, specific solutions.
- Round 3+ (heavily patched): 7-8 is achievable. A 9 requires near-complete \
  coverage — cite what specifically earns it.

Axes:
- technical_architect: buildability, integration complexity, scaling, reliability
- security_compliance: data privacy, legal compliance, platform policies, security
- cynical_user: real-world usability, competitive differentiation, adoption friction
- finance_business: unit economics, burn rate, go-to-market realism, path to break-even

Where a fix in one domain introduced a tradeoff in another, note whether it \
is justified and list it under tradeoffs_validated or tradeoffs_rejected.
"""

REFEREE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REFEREE_SYSTEM_PROMPT),
    ("human",
     "ROUND NUMBER: {round_number}\n\n"
     "RESOLUTION LOG THIS ROUND:\n{resolution_log}\n\n"
     "CURRENT SPECIFICATION:\n{specification}"),
])


async def run_referee_agent(
    specification: str, round_number: int, resolution_log_text: str
) -> RefereeVerdict:
    print("  [referee] Scoring round...")
    llm = get_llm(temperature=0.2).with_structured_output(RefereeVerdict)
    chain = REFEREE_PROMPT | llm
    verdict: RefereeVerdict = await _invoke_structured(
        chain,
        {
            "round_number": round_number,
            "resolution_log": resolution_log_text,
            "specification": specification,
        },
        "RefereeVerdict",
    )
    verdict.scorecard.round_number = round_number
    return verdict
