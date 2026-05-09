# llama.cpp Responses Compatibility Roadmap for Codex + llama-swap

## Goal

Define the shortest reliable path to the best possible Codex compatibility for this stack:

1. Codex client
2. `llama-swap` bridge
3. `llama.cpp` upstream server

The target is not generic OpenAI parity at any cost. The target is the Codex-facing contract that keeps:

- final answers renderable
- tool calls executable
- plan mode stable
- streamed turns complete and resumable
- continuations coherent after tool output

This roadmap uses local evidence from:

- `docs/responses-passthrough-investigation.md`
- `tmp/responses_passthrough_investigation/compatibility_matrix.md`
- `proxy/proxymanager.go`
- `proxy/proxymanager_bridge_xml_test.go`
- `/home/admmin/llama/cuda/llama.cpp/tools/server/server-chat.cpp`
- `/home/admmin/llama/cuda/llama.cpp/tools/server/server-task.cpp`
- `/home/admmin/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/responses`

## Executive Summary

`llama.cpp` implements a useful but incomplete Responses layer. It currently behaves like a Responses-to-chat adapter plus a partial SSE emitter, not a full Responses state machine.

The architectural decision for this repo should be:

- `llama-swap` owns the Codex-facing Responses contract
- `llama.cpp` is an inference backend and optional upstream helper
- model-specific behavior stays limited to parsing and normalization
- vLLM is the best local reference for how a fuller Responses implementation organizes request validation, response storage, retrieval, streaming lifecycle, and tool/event normalization

In other words, we should not make native passthrough the primary goal. We should build a bridge-owned Responses implementation that can opportunistically reuse upstream pieces when they are correct and useful.

## What the Three Systems Actually Do

### 1. `llama.cpp`

The local `llama.cpp` server does support `POST /v1/responses`, but the implementation is intentionally thin:

- request handling converts Responses input into Chat Completions input in `tools/server/server-chat.cpp`
- `previous_response_id` is explicitly rejected
- `max_output_tokens` is mapped to `max_tokens`
- tool definitions are limited to `type="function"`
- streaming emits native Responses-style events from chat diffs in `tools/server/server-task.cpp`

This means the native server is strongest at:

- simple text turns
- array input normalization
- reasoning text emission
- function-call argument streaming
- tool-output passback when the caller reinserts `function_call_output`

It is weakest at:

- response lifecycle persistence and retrieval
- strict response-state transitions
- Codex-specific tool orchestration semantics
- exact SSE event contract completeness

### 2. `llama-swap`

`llama-swap` is already doing meaningful contract repair for Codex. The bridge currently adds or normalizes:

- top-level `output_text`
- `requires_action` for actionable tool turns
- tool-phase output shaping
- commentary / plan-friendly response shaping
- `response.reasoning_summary_text.*` event lanes
- `response.function_call_arguments.done`
- bridge-specific continuation and fallback behavior

For this stack, those are not cosmetic. They are active compatibility layers, and they are the beginnings of a bridge-owned Responses implementation.

### 3. vLLM

The installed vLLM package has a dedicated Responses subsystem instead of a thin adapter:

- typed request/response models in `responses/protocol.py`
- explicit router support for create, retrieve, and cancel in `responses/api_router.py`
- response storage, background execution, retrieval, and cancel support in `responses/serving.py`
- parser-aware and harmony-aware response assembly in `responses/serving.py`
- complete streaming event builders, including `response.function_call_arguments.done`, in `responses/streaming_events.py`

vLLM is not a drop-in template for `llama.cpp`, but it is the best local reference for:

- which lifecycle pieces belong in a real Responses implementation
- which state transitions should be explicit
- how streaming and non-stream paths can share a typed contract

## Architecture Decision

The key question is whether “Qwen understands the Responses API” should influence the system design.

The answer is: only a little.

The model does not implement:

- HTTP endpoints
- response storage
- retrieval
- cancellation
- `previous_response_id`
- `requires_action`
- top-level `output_text`
- SSE event ordering
- terminal argument events like `response.function_call_arguments.done`

Those are harness concerns.

The model only influences whether it can:

- follow tool-use prompting
- emit parseable function/tool intent
- separate final text from tool intent well enough
- produce stable planning/reasoning patterns

So for this stack:

- **Qwen capability** matters for tool-use and structured-output quality
- **`llama.cpp` capability** matters for inference and whatever partial Responses helpers it exposes
- **`llama-swap` capability** must own the actual Codex Responses contract

That makes the safest architecture:

1. `llama-swap` is the source of truth for Responses semantics
2. `llama.cpp` is a backend generator, not the contract authority
3. model-family adapters in `llama-swap` handle parsing, cleanup, and normalization only

## Proven Current Gaps

The live matrix in `tmp/responses_passthrough_investigation/compatibility_matrix.md` shows three categories:

### A. Bridge-only today

These request classes should stay on the translation path until the bridge grows a native-aware replacement path:

- `previous_response_id`
- `max_output_tokens`
- `tool_choice=auto`
- `tool_choice=required`
- mixed prose + tool output turns
- `apply_patch`-intent turns
- streamed tool turns

### B. Native usable as helper input

These are the best request classes for bridge-owned Responses built from mostly-native upstream output:

- basic text turns
- `instructions` turns
- array-form `input`
- `reasoning` option variants
- `include` variants
- `tool_choice=none`
- tool-output continuation replies
- plan-only replies
- streamed non-tool replies

### C. Native not exposed

These are missing at the native `llama.cpp` layer and must stay bridge-owned if Codex needs them:

- response retrieval by id
- response collection listing
- response cancellation
- stored/background lifecycle

## First-Wrong-Stage Analysis

The most important mismatches are not random. They fail at specific stages.

| Surface | First wrong stage | Why it matters |
|---|---|---|
| `previous_response_id` | upstream request normalization in `llama.cpp` | native rejects the field before generation starts |
| missing top-level `output_text` | bridge response normalization if passthrough is enabled | Codex and current bridge tests rely on a stable final text field |
| tool turn ends `completed` instead of `requires_action` | upstream response semantics | Codex loses the actionable tool phase |
| mixed prose + tool output in one native turn | upstream response semantics | Codex tool router and bridge invariants expect cleaner separation |
| missing `response.function_call_arguments.done` | upstream SSE contract | Codex stream consumers and bridge tests rely on tool argument termination |
| `reasoning_text` lane names differ from bridge `reasoning_summary_text` lane | bridge/native stream contract boundary | Codex UI behavior is tied to current bridge naming and sequencing |
| `GET /v1/responses/{id}` missing in `llama.cpp` | upstream API surface | replay/retrieval cannot be delegated upstream |
| invalid JSON status mismatch | native error normalization | caller-visible failure semantics diverge |

## 1-by-1 Comparison

| Contract area | `llama.cpp` today | `llama-swap` today | vLLM reference | Roadmap target |
|---|---|---|---|---|
| create response | yes, via chat conversion | yes | yes | bridge-owned response assembly over upstream generation |
| retrieve response | no | no persistent native retrieval passthrough | yes | optional bridge store if Codex needs it |
| cancel response | no | no | yes | defer unless background mode is introduced |
| `previous_response_id` | rejected | bridge can continue conversation via rewritten history | supported when stored response exists | keep bridge-owned continuation |
| top-level `output_text` | absent in tested native replies | synthesized consistently | response output is structured; full typed response object exists | bridge must keep emitting `output_text` |
| tool turn status | `completed` | normalized to `requires_action` when needed | status machinery exists but even vLLM notes some WIP areas | bridge must own Codex tool-phase status |
| tool item lifecycle | partial | normalized | explicit add/delta/done builders | bridge should mirror vLLM completeness where native is missing |
| reasoning stream lane | `response.reasoning_text.*` | `response.reasoning_summary_text.*` plus commentary-aware shaping | native OpenAI-style `reasoning_text` and `reasoning_part.*` | bridge keeps Codex-facing lane semantics |
| function argument stream done | missing in observed native stream | emitted | emitted | bridge should synthesize when absent |
| mixed text + tool output | possible | bridge separates or rewrites for Codex | parser/serving paths can structure output items cleanly | bridge must continue splitting mixed tool turns |
| passback via `function_call_output` | works | works | works | preserve |
| background/store lifecycle | no | no | yes | out of scope for first compatibility pass |
| built-in tool loop | no bridge-owned native execution | bridge/client-driven | vLLM can do server-owned tool loops with tool server | do not copy into `llama-swap` now |

## What We Should Learn From vLLM

### Adopt

These ideas are directly useful for `llama-swap` even though the upstream engine is different:

1. **Typed Responses contract as an internal interface**
   - Treat create, full-response assembly, stream assembly, retrieval, and error normalization as separate surfaces.
2. **Dedicated streaming event builders**
   - Centralize event synthesis instead of scattering per-case rewrites.
3. **Tool delta and done separation**
   - Always model tool call start, argument delta, argument done, item done, and final response status separately.
4. **Request validation before generation**
   - Validate impossible combinations early and return stable error shapes.
5. **Continuation as stateful response assembly**
   - Keep continuation logic explicit instead of pretending raw `previous_response_id` passthrough exists upstream.

### Do not copy blindly

These vLLM features are useful references but should not drive the first `llama-swap` roadmap:

1. built-in background response execution
2. response storage with indefinite in-memory retention
3. server-owned built-in tool execution loop
4. full harmony-specific stack
5. broad OpenAI Responses parity unrelated to Codex

The point is not to turn `llama-swap` into vLLM. The point is to steal the right contract ideas while keeping the bridge small, reliable, and clearly responsible for the Codex contract.

## Design Principles for `llama-swap`

1. **Codex contract beats generic purity**
   - If Codex needs a shim, the shim is correct.
2. **Prefer bridge-owned assembly over passthrough**
   - Reuse native upstream output when helpful, but do not delegate contract ownership.
3. **Keep bridge ownership where upstream lacks state**
   - Tool-phase state, continuations, and plan-mode shaping remain bridge territory.
4. **Normalize once, test everywhere**
   - One canonical normalized Responses shape should feed both non-stream and stream paths.
5. **Separate native gaps from bridge choices**
   - Track which differences are required for Codex versus accidental legacy behavior.

## Recommended Implementation Plan

### Phase 0: Lock the Oracle

Before changing routing, promote the current Codex contract into explicit acceptance tests.

Work:

- codify the current oracle from `proxy/proxymanager_bridge_xml_test.go`
- extract a compact set of invariants for:
  - plain final answers
  - plan-only answers
  - actionable tool turns
  - tool passback
  - streamed final answers
  - streamed tool turns
- keep the existing probe harness as the external A/B evidence source

Success criteria:

- every future routing experiment is judged against one stable oracle

### Phase 1: Build a Bridge-Owned Native-Input Normalizer for Non-Tool Turns

Goal: assemble Codex-safe Responses from native upstream output for non-tool turns without depending on native Responses semantics.

Work:

- add a request classifier inside `llama-swap` for:
  - non-tool plain text
  - plan-only
  - continuation answer after `function_call_output`
  - streamed non-tool text
- add a native-response normalizer that can:
  - synthesize top-level `output_text`
  - keep output item ordering stable
  - normalize native reasoning lane output into bridge-compatible shape when needed
  - normalize native HTTP errors into stable bridge errors
- treat native Responses payloads as raw material, not authoritative final objects

Do not do in this phase:

- delegate actionable tool turns
- delegate `previous_response_id`
- rely on native response retrieval

Success criteria:

- `post_basic`
- `instructions_text`
- `array_input`
- `reasoning_*`
- `include_field`
- `tool_choice_none`
- `function_call_output_passback`
- `plan_only`
- `stream_basic`
- `stream_plan_only`

all become bridge-approved under a bridge-owned normalization route.

### Phase 2: Build a Bridge-Owned Tool-Turn Assembly Path

Goal: make tool turns consumable by Codex using bridge-owned semantics even when upstream emits partial or inconsistent Responses shapes.

Work:

- detect native tool outputs that come back as `completed`
- rewrite tool turns into bridge-owned actionable phases:
  - set `status` to `requires_action`
  - preserve tool call ids
  - synthesize empty or safe `output_text`
  - split mixed prose + tool output when both appear
- for streaming:
  - synthesize `response.function_call_arguments.done` when native omits it
  - preserve deterministic event order
  - maintain a final tool-phase status consistent with Codex expectations

Important constraint:

`llama-swap` should continue to own the Codex tool contract even if the upstream server emits a nearly-correct function call. The final consumer is Codex, not a generic SDK.

Success criteria:

- `tool_choice_auto`
- `tool_choice_required`
- `mixed_text_tool`
- `apply_patch_intent`
- `stream_tool`

move from `bridge-only` to `bridge-owned assembly from native-compatible upstream output`.

### Phase 3: Keep Continuation Bridge-Owned, but Make It Explicit

Goal: stop treating native `previous_response_id` as a future default path.

Work:

- define a bridge-owned continuation state model
- classify all continuation requests into:
  - upstream-native-safe replay from explicit history
  - bridge-only replay from stored translated history
- document that native `previous_response_id` passthrough is not a prerequisite for Codex compatibility

Optional work:

- add an internal response-store abstraction inside `llama-swap` only if Codex workflows truly need retrieval semantics later

Success criteria:

- no Codex tool loop depends on native `previous_response_id`

### Phase 4: Rationalize the Reasoning and Commentary Contract

Goal: reduce accidental bridge complexity while preserving Codex UI behavior.

Work:

- compare where Codex truly requires:
  - `response.reasoning_summary_text.*`
  - commentary lane visibility
  - plan-only `<proposed_plan>` wrappers
- keep the current bridge event names where UI behavior depends on them
- only collapse toward native `reasoning_text` if Codex proves it renders equivalently for local models

Success criteria:

- no empty assistant render
- no hidden plan text
- no reasoning-only response that drops the visible answer

### Phase 5: Decide the Final Upstream Reuse Boundary

After Phases 1 through 4, define request routing explicitly.

Recommended policy:

- **bridge-owned Responses assembly**
  - default for all Codex-facing requests
- **upstream native reuse**
  - allowed only where upstream output reduces work without owning semantics
  - simple non-tool turns
  - plan-only turns
  - post-tool answer turns
  - optionally tool turns once Phase 2 is complete
- **bridge-only fallback**
  - any request needing continuation state not reproducible from upstream inputs alone
  - any request needing strict Codex plan/tool repair
  - any future high-risk or unknown request class

The router should be gated by:

- a request classifier
- a feature flag
- an escape hatch to force bridge translation

## Concrete Work Items

### Shim layer

- create one native-response normalization module instead of scattering special cases in `proxymanager.go`
- give it separate entry points for:
  - full non-stream normalization
  - stream event normalization
  - tool-turn normalization
  - error normalization

### Tests

- extend the existing probe harness into a regression fixture source
- add table-driven tests for every request class in the live matrix
- keep separate assertions for:
  - top-level fields
  - output item types and order
  - final status
  - SSE event names and order
  - continuation behavior

### Documentation

- keep `docs/responses-passthrough-investigation.md` as the probe/how-to document
- use this roadmap as the implementation planning document
- update the roadmap after each phase with:
  - request classes unlocked
  - request classes still bridge-only
  - new native mismatches discovered

## Acceptance Matrix

`llama-swap` should not switch a request class to native-first until all of these are true:

1. Codex renders the final answer correctly.
2. Tool calls remain actionable and executable.
3. Stream completion semantics stay deterministic.
4. Plan mode does not leak execution or lose the `<proposed_plan>` structure.
5. Post-tool continuation remains coherent.
6. Error responses stay stable for the client.

If any one of those fails, the class stays bridge-owned.

## Suggested Order of Execution

1. land Phase 0 test/oracle cleanup
2. implement Phase 1 non-tool passthrough shim
3. rerun the A/B harness and update the matrix
4. implement Phase 2 tool normalization
5. rerun the A/B harness and update the matrix
6. decide whether Phase 3 needs a bridge response store
7. finalize routing policy and feature flag defaults

## Recommendation

The best path for this repo is not:

- “trust native `llama.cpp` now”
- or “keep translating everything forever”

The best path is:

1. make `llama-swap` the source of truth for Codex-facing Responses semantics
2. use native `llama.cpp` output only as an upstream generation/helper surface
3. keep model-specific code focused on parsing, cleanup, and normalization
4. borrow vLLM’s separation of concerns for validation, streaming lifecycle, and response assembly
5. unlock upstream reuse one request class at a time behind tests

That gives us the highest compatibility for the current Codex + `llama-swap` + `llama.cpp` setup without pretending the upstream Responses implementation or the model itself owns the API contract.
