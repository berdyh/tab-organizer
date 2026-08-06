# SPEC — Provider routing: no silent selection, always announce

Status: **PARTIALLY IMPLEMENTED**. Written 2026-08-05.
Config side landed in `6493974`. Service side (ai-engine):

| | State |
|---|---|
| R1 no implicit provider | **Implemented.** `AI_PROVIDER` / `EMBEDDING_PROVIDER` have no defaults, in `llm_client.py`, `docker-compose.yml`, and `.env.example`. |
| R2 no silent embedding fallback | **Implemented.** Deleted; raises `embedding_provider_cannot_embed` with a catalog-derived list. |
| R3 honour `requires_explicit_opt_in` | **Invariant only, as specified.** Stated in `LLMClient.__init__`'s docstring and frozen by a test; the general mechanism stays TS. |
| R4 startup log + `/health` | **Implemented.** `provider.active` per role; `providers` block on `/health`. |
| R4 UI badge | **Deferred to TS** (plan decision 16). |
| R5 per-response attribution | **Deferred to the wk10 cutover.** No columns added to the Python store. |
| R6 `cli.py configure-provider` | **Implemented.** See `scripts/MODULE.md`. Provider selection refuses to proceed non-interactively without an explicit flag; `host-ai` and `check-provider` fail closed the same way. |

R2 also covers dimensions: `EMBEDDING_DIMENSIONS` is catalog-derived, a missing
catalog entry raises `embedding_dimensions_unknown` rather than inferring 1536,
and an env override contradicting the model raises
`embedding_dimensions_mismatch`. Announcing an inferred width would have made
the R4 surface launder the drift it exists to expose.

Gates: `tests/unit/test_provider_routing.py`, `tests/unit/test_cli_*.py`.

### Amendment, 2026-08-06 — R1 had a hole: "preferred" was not enforced downstream

R1 stops the *service* choosing a provider. It said nothing about the
*catalog* pointing a chosen provider at the metered copy of a model the user
already pays for. `codex_cli.default_models.llm` was `gpt-5.6-luna` and
`openrouter.default_models.llm` was `openai/gpt-5.6-luna` — same weights, two
ids, two cost models, no link between the entries. Choosing openrouter (a
deliberate, consented choice under R3) therefore billed for something already
bought, and nothing on screen said so.

Two changes close it, both data-first:

1. **Routes are explicit.** Each `models:` entry is one provider route: the key
   is the exact wire id, `provider:` names the provider, `model_family:` is the
   identity shared by every route to the same weights, and `cost_model` is
   resolved from the provider (`get_model_cost_model`). `describe_model()` /
   `get_family_routes()` / `format_model_description()` make the route visible,
   and every human-facing menu renders through the last of these — so a listing
   can never show a bare model name with no provider or price attached. The
   merged one-entry-per-model design was rejected: the key IS the wire value,
   and merging would push `dimensions` into a per-provider sub-map, adding a
   second way to resolve the wrong vector width.
2. **`routing.metered_duplicate_policy: never_default`.** A metered route may
   not be `recommended: true`, a `default_models` value, or a `use_cases`
   model when its family has a subscription route or it declares
   `superseded_by:`. Registered is fine — openrouter stays the deliberate
   smoke-test path (`use_cases.smoke_test`) — but nothing arrives there on its
   own.

`gemini_cli` joins `llm_preference_order` as a fourth subscription CLI. It is
**specified and implemented but NOT verified end-to-end**: the `gemini` CLI on
the development host is installed (0.54.0) and unauthenticated, so its live
`requires_provider_credentials` test skips and no generation through the
adapter has ever been observed. `antigravity`, the originally proposed Gemini
route, is a GUI IDE with no headless mode and cannot back a provider at all.

One measured behaviour is worth stating here because it generalises to any
future CLI adapter: **a `--version` preflight can be a false positive for
authentication.** Logged out, `gemini --version` exits 0 while `gemini -p` blocks
forever on an interactive browser-login prompt that ignores EOF and survives
having no controlling terminal. An availability check that only proves the
binary runs will advertise such a provider and hang every request to the
timeout.

**The first version of this line was false and is worth keeping as a warning.**
It read "Every one of them was confirmed to fail against the reintroduced old
behaviour before being kept." An adversarial review then found two changes in
`ec4cc0a` that no test gated at all — the `LLMClient.PROVIDERS` capability
mirror (flipping it back left 92/92 green) and `switch_provider`'s embedding
guard — plus two filters in `configure-provider` that were each untested
because the other happened to agree. The claim was written in good faith by
authors who had genuinely watched their other tests fail. Assertions about
test coverage are themselves untested, which is why this repo requires the
mutation to be run and its output recorded rather than asserted.

`PROVIDERS` is now frozen by a catalog-agreement test covering every provider
and capability. `switch_provider`'s guard is shadowed by
`get_provider_runtime_state`, which raises the byte-identical error three lines
later, so no test can gate the line; the behaviour is frozen instead and the
shadowing is recorded in `services/ai-engine/MODULE.md`.

The contract already exists as data in `config/ai_models.yaml` under `routing:`.
This document says what the code must do to honour it. Read
`config/ai_models.yaml` first — it is the source of truth for preference order,
opt-in requirements, and per-model tiers. This file only describes behaviour.

---

## The requirement, in one paragraph

A user must never discover after the fact which provider answered their request.
Subscription CLIs (`claude_code`, `codex_cli`) are preferred because they spend
a subscription already paid for. If they are unavailable the service must **stop
and say so** rather than reaching for a metered or local provider on the user's
behalf. Every other provider — OpenRouter, Ollama, OpenAI, Gemini — requires a
deliberate choice, and whichever one is active must be visible at startup, in
`GET /health`, and in the UI.

Why this is worth code rather than documentation: silent substitution is the
exact failure mode that let a completely broken embedding path look healthy for
months (WI0-B1). The system reported success while doing nothing. A provider
that silently changes is the same class of defect, one layer up — the answers
just get quietly worse, or quietly cost money.

---

## Behaviour before this change (kept for the record; R1/R2/R4 are now fixed)

| Location | Today | Problem |
|---|---|---|
| `services/ai-engine/app/core/llm_client.py:115` | `os.getenv("AI_PROVIDER") or "openrouter"` | Hardcoded metered default. An unset env var silently spends money. |
| `services/ai-engine/app/core/llm_client.py:141` | `os.getenv("EMBEDDING_PROVIDER") or "openrouter"` | Hardcoded metered default for the embedding role too. (This row used to add "and openrouter cannot embed at all — this default can only fail". That was false; see the correction below. The default was still wrong, for the money reason above.) |
| `services/ai-engine/app/core/llm_client.py:148-152` | If the provider can't embed, silently swap to `defaults.provider` | The silent fallback. Nothing logs it; nothing surfaces it. |
| everywhere | Nothing announces the active provider | The user cannot tell who answered. |

`scripts/init.py` already prompts correctly (`prompt_choice`, and it derives the
supported-embedding list from the catalog rather than hardcoding it). It is not
the problem — do not rewrite it. The gap is entirely at runtime.

---

## Required behaviour

### R1 — No implicit provider, ever

Remove both hardcoded `"openrouter"` defaults. If `AI_PROVIDER` (or
`EMBEDDING_PROVIDER`) is unset, the service must not choose. It fails closed
with a structured error and reports `degraded` on `/health`:

```
{
  "code": "provider_not_selected",
  "cause": "AI_PROVIDER is not set. This service does not pick a provider for you.",
  "fix": "Run ./scripts/cli.py configure-provider, or set AI_PROVIDER explicitly.
          Preferred: claude_code or codex_cli (uses your subscription).
          Metered: openrouter, openai, gemini (requires an API key and consent).
          Local: ollama (free, requires models pulled first)."
}
```

Match the `{code, cause, fix}` shape already used by `CDPConnectionError`
(`services/browser-engine/app/tabs/cdp.py`) and `CredentialStoreError`
(`services/browser-engine/app/auth/queue.py`).

### R2 — Delete the silent embedding fallback

`llm_client.py:148-152` currently rewrites the provider when the chosen one
cannot embed. Delete that. Raise instead, naming the providers that can:

```
code: "embedding_provider_cannot_embed"
cause: "EMBEDDING_PROVIDER='claude_code' serves no embedding models."
fix:   "Choose one of: ollama, openrouter, openai, gemini.
        That list comes from config/ai_models.yaml, so it is current
        by construction."
```

Derive the "can embed" list from the catalog (`is_provider_supported(p,
"embeddings")`), never hardcode it. That is what made `init.py` self-correct
when the catalog was fixed.

The `fix` string names **no provider of its own**, and that is a requirement,
not a style preference. It used to end with "Note openrouter serves NO
embedding models -- verified 2026-08-04". See the correction below for why that
line is gone.

#### Correction, 2026-08-05 — "openrouter cannot embed" was false

Recorded rather than deleted, because the repo has now been bitten four times
by a documented-but-false invariant, and the durable lesson is the method, not
the fact.

**The claim.** On 2026-08-04 `config/ai_models.yaml` was given
`openrouter.supports.embeddings: false`, annotated "VERIFIED FALSE ... OpenRouter
serves NO embedding models". It propagated into this spec, `CLAUDE.md`,
`docs/MODULE_INDEX.md`, `docs/AI_CONFIG.md`, `.env.example`, `README.md`, two
module cards, `scripts/cli.py`, the `LLMClient.PROVIDERS` mirror, a
user-facing error string, and four tests.

**Why it was believed.** The check was real and honestly reported. It fetched
`GET https://openrouter.ai/api/v1/models`, scanned all 340 entries, found
output modalities of only `[audio, image, text]` and no id matching "embed",
and concluded embeddings were unavailable.

**Why the check could not work.** That listing describes the
**chat-completions** surface. `POST /v1/embeddings` is a *separate* surface it
never enumerates. Absence from the listing was therefore not evidence of
absence — the method had no way to observe the thing it was ruling on.

**What replaced it.** A live call to the endpoint itself, 2026-08-05:

| Request | Result |
|---|---|
| `text-embedding-3-small` | HTTP 200, 1536-d, $0.02/Mtok |
| `text-embedding-3-large` | HTTP 200, 3072-d, $0.13/Mtok |
| `openai/text-embedding-3-small` | HTTP 200, 1536-d |
| `{"dimensions": 768}` override | HTTP 200, 768-d |
| `text-embedding-004`, `nomic-embed-text` | HTTP 400, "Model does not exist" |

**The rule this yields.** A capability claim is only as strong as the method
behind it. Asking a *listing* what a provider offers is weak evidence and must
never be recorded as "verified"; *calling the endpoint* that serves the
capability is authoritative. Annotating the weak result as verified is what
made it durable — every later reader treated it as settled and copied it
onward.

**What caught it, and what did not.** Nothing did, for a day: every downstream
check compared the claim against another copy of the claim. When the catalog
was corrected, `scripts/init.py` and the CLI's `configure-provider` filters
self-corrected with no edit (they ask the catalog), while the `PROVIDERS`
mirror, the embedding adapter map, six documents and four tests each had to be
repaired by hand. `tests/unit/test_provider_routing.py` now carries two tests
for this specific blind spot: one asserting `supports.embeddings` agrees with
the adapter map, and one (`requires_provider_credentials`) asserting it agrees
with the live endpoint — the only kind of test that could have caught the
original error, since the original error was a false belief about a remote
service.

### R3 — Honour `requires_explicit_opt_in`

`ai_models.yaml` marks `ollama`, `openrouter`, `openai`, `gemini`, `anthropic`
and `deepseek` as requiring a deliberate choice. Setting `AI_PROVIDER=openrouter`
in `.env` **is** that deliberate choice — the env var is the record of consent,
so do not prompt again at runtime (a service cannot prompt anyway).

What R3 forbids is the service arriving at one of these *without* an env var
saying so. In practice R1 already guarantees that; R3 exists so the invariant is
stated rather than emergent, and so a future "try the next provider" feature
cannot be added without confronting it.

Ollama is explicitly included. Free and local is not the same as harmless:
routing to a small local model silently changes answer quality, and the user
should know it happened.

### R4 — Announce the active provider

Three surfaces, all required:

**Startup** — one structured log line per provider role, using the existing
`log_event` from `services/observability.py`:

```
{"event": "provider.active", "role": "llm", "provider": "claude_code",
 "model": "sonnet", "cost_model": "subscription", "tier": "mid"}
{"event": "provider.active", "role": "embedding", "provider": "ollama",
 "model": "nomic-embed-text", "dimensions": 768, "cost_model": "free_local"}
```

`cost_model` and `tier` come from the catalog. Never log API keys.

**`GET /health`** — extend the existing payload (it already reports
`degraded`; see `services/ai-engine/app/main.py:197`):

```json
{
  "status": "healthy",
  "providers": {
    "llm":       {"provider": "claude_code", "model": "sonnet", "cost_model": "subscription"},
    "embedding": {"provider": "ollama", "model": "nomic-embed-text", "cost_model": "free_local"}
  }
}
```

**UI** — a persistent badge showing the active LLM provider and its cost model,
not buried in a settings page. `ui_options.show_active_provider_badge` is
already set to `true` in the catalog. Metered providers should be visually
distinct from subscription and local ones: the point is that a user glancing at
the screen can tell whether this conversation is costing money.

### R5 — Per-response attribution

Every generated artifact that reaches storage — cluster labels, facets,
summaries — records the provider and model that produced it. The plan already
requires this for replayability ("every agent-produced row stamped with agent
id, model, prompt version, run timestamp"), so R5 is that requirement extended
to the provider dimension. Cheap to add while touching this code; expensive to
retrofit once rows exist without it.

### R6 — `cli.py configure-provider`

The service cannot ask a question. The CLI can, and this is where the asking
belongs:

1. Probe availability: is the `claude` binary on PATH and authenticated? the
   `codex` binary? is Ollama reachable, and which models are pulled? which API
   keys are present in `.env`?
2. Present only what is actually usable, in `routing.llm_preference_order`,
   annotated with cost model. Do not offer a provider whose binary is missing.
3. Write the choice to `.env` (`AI_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`,
   `EMBEDDING_MODEL`), leaving `EMBEDDING_DIMENSIONS` **blank** so it resolves
   from the catalog and cannot drift — this is existing, correct behaviour
   (`scripts/init.py`), keep it.
4. If nothing is usable, say exactly what to install or set. Never write a
   provider the probe could not verify.

Also: if the user picks Ollama, check the model is actually pulled and offer to
pull it. WI0-B1 found the Ollama container running with **zero models**, which
is indistinguishable from working until the first request fails.

---

## Where this should be built — read before starting

Split the work by lifespan. The Python data plane is scheduled for deletion at
the wk10 cutover; `scripts/cli.py` is tooling and survives.

| Item | Build in | Why |
|---|---|---|
| R6 (`cli.py configure-provider`) | **Python, now** | Tooling. Survives the migration. Delivers most of the user-facing value on its own. |
| R1, R2 (fail closed) | **Python, now** | ~15 lines. A live correctness and trust issue, and removing a fallback is subtractive — it cannot rot. |
| R4 startup + `/health` | **Python, now** | ~20 lines against the existing `log_event` and health payload. |
| R4 UI badge | **TS, at facade** | The Streamlit UI is replaced in wk1-6. Building it twice is exactly what plan decision 16 forbids. |
| R5 per-response attribution | **TS, at cutover** | Needs the content-addressed schema, which is built once in TS. Do NOT add columns to the Python store for it. |
| R3 as a general mechanism | **TS** | Only matters once there is a router. The Python side gets the invariant for free from R1. |

The guiding rule from the plan (decision 21): fix in Python only what the
migration will not reach, and never build the same thing twice.

---

## Acceptance

- With no `AI_PROVIDER` set, the service starts, reports `degraded`, and names
  the fix. It does **not** answer requests using a provider nobody chose.
- With `EMBEDDING_PROVIDER` set to any provider that cannot embed, the error
  names the providers that can, derived from the catalog.
- `provider.active` appears in the startup log for both roles.
- `/health` reports the active provider and model for both roles.
- `cli.py configure-provider` offers only verified-available providers and
  refuses to write an unverified one.
- Every new behaviour has a test proving it gates: reintroduce the fallback,
  watch the test fail, restore. This repo has had four separate cases of a test
  that could not fail; do not add a fifth.
