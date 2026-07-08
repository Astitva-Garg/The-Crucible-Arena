# The Crucible Arena

An automated, multi-agent adversarial simulation engine, built on **LangGraph** and **LangChain**, that stress-tests a raw product idea. A one-paragraph idea is expanded into a full specification, then forced through repeated rounds against four uncompromising domain critics until it converges on a battle-hardened blueprint (or hits the round limit).

## Structure

```
arena/
├── data/
│   └── tech_guardrails.txt   # Factual benchmarks the Critics ground their findings in
├── state.py                  # Pydantic schemas & the ArenaState graph-state model
├── agents.py                 # LangChain prompts + with_structured_output chains for every agent
└── main.py                   # LangGraph StateGraph wiring, Sieve logic, exporter, CLI
main.py                        # Thin top-level shim -> arena.main.main()
```

- **state.py** — `ArenaState` is a Pydantic model used directly as the LangGraph `state_schema`. No chat history; state fields are updated in place each round. `raw_critiques` uses an `Annotated[..., merge_critiques]` reducer so the four critic nodes can write concurrently without clobbering each other.
- **agents.py** — every prompt as a LangChain `ChatPromptTemplate`, piped into a shared chat model (`ChatOpenAI` or `ChatNVIDIA`, picked via `CRUCIBLE_PROVIDER`) through `.with_structured_output(PydanticModel)`. Covers the Optimistic Product Manager, the four critics, the per-axis patch agent, and the Referee.
- **main.py** — the LangGraph graph: node functions wrapping each agent call, the fan-out/fan-in critic dragnet, the sequential patch chain, the conditional loop-or-stop edge, plus the sieve's flaw-ID bookkeeping, the Markdown report exporter, and the CLI.

## Graph shape

```
        ┌─► critic_technical ──┐
START ──┼─► critic_user ───────┼──► sieve ──► patch_technical ──► patch_user
        ├─► critic_finance ────┤                 ──► patch_finance ──► patch_code_efficiency
        └─► critic_code_eff ──┘                                            │
                                                                            ▼
                                                                        referee
                                                                            │
                                                        ┌───── loop ────────┴──── stop ─────┐
                                                        ▼                                    ▼
                                                  advance_round ──► critics again           END
```

- **Fan-out / fan-in**: the 4 critic nodes all branch from `START` (or from `advance_round` on subsequent loops) and all feed into `sieve`. LangGraph runs them concurrently within one superstep and waits for all 4 before running `sieve` — this replaces a manual `asyncio.gather`.
- **Sequential assembly line**: `sieve → patch_technical → patch_user → patch_finance → patch_code_efficiency → referee` are plain chained edges, each patch node only touching its own domain's flaws.
- **Conditional loop**: `route_after_referee` inspects `ArenaState` after each round and returns `"loop"` (back to the critics via `advance_round`) or `"stop"` (to `END`) — implementing dynamic early-stopping (zero open critical flaws + ≤1 point score plateau over 2 rounds) with a hard cap at round 4.

## Setup

```
uv sync
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Optionally set `CRUCIBLE_MODEL` (defaults to `gpt-4.1`).

### Using NVIDIA NIM or OpenRouter instead of OpenAI

`arena/agents.py` picks the chat model provider at import time via
`CRUCIBLE_PROVIDER` (`openai`, `nvidia`, or `openrouter`; default `openai`).
All three branches build a LangChain `BaseChatModel` — `ChatOpenAI` or
`ChatNVIDIA` — and every prompt/chain/structured-output call downstream is
provider-agnostic.

**NVIDIA NIM:**

1. Create a free account at [build.nvidia.com](https://build.nvidia.com/), open any model page, and click "Get API Key". Copy the key (starts with `nvapi-`).
2. In `.env`, set:
   ```
   CRUCIBLE_PROVIDER=nvidia
   NVIDIA_API_KEY=nvapi-your-key-here
   CRUCIBLE_MODEL=meta/llama-3.1-70b-instruct
   ```
   Pick a model that supports tool calling (structured output relies on it).
   `ChatNVIDIA.get_available_models()` lists every catalog model along with a
   `.supports_tools` flag — use that to verify before committing to one, and
   double check the model isn't deprecated/end-of-life. `meta/llama-3.1-70b-instruct`
   is a known-good starting point.
3. Run as usual with `uv run main.py`.

Note: NVIDIA's shared free-tier inference endpoints can be slow and
inconsistent under load (observed end-to-end latency for the same call
ranging from ~30s to ~340s in testing). `agents.py` retries transient
timeouts a few times with backoff, but if you're hitting this often,
OpenRouter or OpenAI are more consistent.

**OpenRouter:**

OpenRouter exposes an OpenAI-compatible API in front of many underlying
models, so it's implemented via `ChatOpenAI` pointed at OpenRouter's
`base_url` rather than a dedicated client.

1. Create an account at [openrouter.ai](https://openrouter.ai/) and generate an API key (starts with `sk-or-`).
2. In `.env`, set:
   ```
   CRUCIBLE_PROVIDER=openrouter
   OPENROUTER_API_KEY=sk-or-your-key-here
   CRUCIBLE_MODEL=openai/gpt-4.1
   ```
   `CRUCIBLE_MODEL` must be an OpenRouter model slug (`vendor/model-name`) —
   see [openrouter.ai/models](https://openrouter.ai/models) for the catalog,
   and pick one that supports tool calling (structured output relies on it).
3. Run as usual with `uv run main.py`.

Set `CRUCIBLE_PROVIDER=openai` (or omit it, since that's the default) to use OpenAI directly with `OPENAI_API_KEY`.

## Usage

```
uv run main.py
```

You'll be prompted for your one-paragraph product idea. The final report is written to `crucible_report.md`.
