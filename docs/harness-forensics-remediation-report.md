# Harness Forensics and Remediation Report

## Summary

This report compares three harness layers with a focus on transport, orchestration, tool calling, stream handling, continuation control, and correction logic:

1. Pi core RPC/session harness plus `pi-vscode`
2. OpenCode backend/server harness plus the VS Code plugin
3. `llama-swap` Responses/chat bridge, stream adapters, parser/repair logic, and tool normalization

The goal is not to compare model quality. The goal is to identify which harness contracts Pi and OpenCode enforce that `llama-swap` does not, which current `llama-swap` protections are too narrow or too prompt-specific, and what should be added or corrected so `llama-swap` behaves like a first-class harness adapter instead of a growing Codex/Qwen recovery layer.

## Evidence Base

### External sources

- Pi RPC protocol: https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/rpc.md
- Pi extensions and tool/event model: https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md
- Pi VS Code bridge tools: local `pi-vscode` readme and bridge script
- OpenCode README: https://github.com/opencode-ai/opencode/blob/main/README.md
- OpenCode VS Code plugin changelog and bridge code: local `paviko.opencode-ux-plus` extension

### Local `llama-swap` sources

- `proxy/proxymanager.go`
- `proxy/tool_call_parser.go`
- `proxy/chat_sse_event.go`
- `proxy/sse_monitor.go`
- `proxy/proxymanager_test.go`
- `proxy/proxymanager_bridge_xml_test.go`
- `proxy/chat_sse_event_test.go`

## Exact Integration Types Proven from Code

This section corrects the earlier high-level shorthand. Each system has more than one integration type.

### Pi

Pi uses two distinct integration layers:

1. Pi chat participant to Pi agent:
   - transport type: local child-process RPC over newline-delimited JSON on stdin/stdout
   - proof:
     - `pi-vscode` spawns the Pi binary with `stdio: ["pipe","pipe","pipe"]`
     - commands are written as `JSON.stringify(command) + "\n"`
     - stdout is read line-by-line and each line is `JSON.parse(...)`
   - code:
     - local file `pi0.pi-vscode/dist/extension.cjs:1077-1162`

2. Pi agent to VS Code bridge:
   - transport type: local HTTP JSON RPC
   - proof:
     - `pi-vscode` creates an HTTP server that accepts `POST /rpc`
     - authorization uses `x-pi-vscode-authorization`
     - the bundled bridge tool code calls `fetch(${bridgeUrl}/rpc)` with JSON `{method, params}`
   - code:
     - local file `pi0.pi-vscode/dist/extension.cjs:726-799`
     - local file `pi0.pi-vscode/bridge/pi-vscode-bridge.js:19-35`

3. Pi tool integration model:
   - type: in-process extension tool registration inside the Pi runtime
   - proof:
     - bundled bridge script calls `pi.registerTool(...)`
     - tool implementations return Pi-native tool result envelopes
   - code:
     - local file `pi0.pi-vscode/bridge/pi-vscode-bridge.js:219-260`
     - local file `pi0.pi-vscode/bridge/pi-vscode-bridge.js:792`

So the exact Pi integration type is:
- JSONL process RPC for agent conversation
- HTTP JSON RPC for IDE bridge callbacks
- in-runtime tool registration for VS Code capabilities

### OpenCode

OpenCode also uses two distinct integration layers:

1. VS Code plugin to OpenCode backend:
   - transport type: spawned local HTTP server
   - proof:
     - the backend launcher builds `opencode serve`
     - the SDK server helper also spawns `opencode serve --hostname=... --port=...`
     - connection info is parsed from stdout line `opencode server listening on ...`
   - code:
     - local file `paviko.opencode-ux-plus/out/backend/BackendLauncher.js:220-235`
     - local file `paviko.opencode-ux-plus/out/backend/BackendLauncher.js:262-320`
     - local file `@opencode-ai/sdk/dist/v2/server.js:2-65`

2. OpenCode web UI to VS Code plugin bridge:
   - transport type: local HTTP plus SSE, session-scoped
   - proof:
     - `IdeBridgeServer` exposes `/idebridge/{sessionId}/events` as `text/event-stream`
     - the same server accepts `/idebridge/{sessionId}/send` for POSTed bridge actions
     - sessions carry token auth and SSE clients per session
   - code:
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:145-181`
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:183-240`

3. OpenCode capability model:
   - type: backend server capabilities plus IDE bridge message handlers
   - proof:
     - SDK exposes `createOpencodeServer()` and `createOpencodeClient()`
     - plugin changelog explicitly adds question tool, retry, variants, changed-files, command execution, and HTTP+SSE bridge changes as backend/plugin features
   - code:
     - local file `@opencode-ai/sdk/dist/v2/server.js:2-94`
     - local file `paviko.opencode-ux-plus/changelog.md`

So the exact OpenCode integration type is:
- local spawned HTTP backend server for core agent/runtime
- HTTP plus SSE session bridge for IDE/webview interaction
- backend capability model rather than prompt-only tool exposure

### `llama-swap`

`llama-swap` currently exposes one broad integration layer that combines several responsibilities:

1. Client-facing API:
   - transport type: OpenAI-style HTTP Responses API and chat completions API
   - proof:
     - request translation starts in `translateResponsesToChatCompletionsRequest()`
     - streaming emission is implemented in `writeResponsesStream()` and `writeResponsesStreamFromChatSSE()`
   - code:
     - local file `proxy/proxymanager.go:2249-2465`
     - local file `proxy/proxymanager.go:6542-6930`
     - local file `proxy/proxymanager.go:9529-10357`

2. Upstream-facing integration:
   - transport type: translated chat-completions request/response plus SSE adaptation
   - proof:
     - Responses input is normalized, rewritten, tool-pruned, and translated into chat payloads
     - upstream chat-completion outputs and chunks are then normalized back into Responses output items and SSE events
   - code:
     - local file `proxy/proxymanager.go:2249-2465`
     - local file `proxy/proxymanager.go:5719-6077`
     - local file `proxy/proxymanager.go:9529-10357`

3. Tool integration model:
   - type: bridge-side synthetic tool normalization and repair
   - proof:
     - tools are not registered inside a native agent runtime
     - instead they are translated, pruned, recovered from malformed text, and re-emitted as Responses items
   - code:
     - local file `proxy/tool_call_parser.go`
     - local file `proxy/proxymanager.go:2291-2465`
     - local file `proxy/proxymanager.go:5719-6077`

So the exact `llama-swap` integration type is:
- OpenAI-style HTTP API façade on the client side
- translated chat-completions/SSE adapter on the upstream side
- bridge-level tool contract synthesis rather than native runtime capability registration

## Exact Implementation Techniques Proven from Code

This section compares the implementation style, not just the public transport label.

### Pi implementation technique

Pi is implemented as a real agent process with a strict event protocol.

1. The VS Code extension starts Pi as a child process with explicit stdio pipes.
   - proof:
     - `spawn(options.piPath, ..., { stdio: ["pipe","pipe","pipe"] })`
   - code:
     - local file `pi0.pi-vscode/dist/extension.cjs:1077-1085`

2. The extension writes structured commands, not prompt glue.
   - proof:
     - `child.stdin.write(\`${JSON.stringify(command)}\n\`)`
   - code:
     - local file `pi0.pi-vscode/dist/extension.cjs:1100-1102`

3. The extension parses stdout as framed JSON events line-by-line.
   - proof:
     - stdout is buffered until newline
     - each line is parsed with `JSON.parse(line)`
     - event `type` then dispatches handling such as `extension_ui_request`, `message_update`, and failure `response`
   - code:
     - local file `pi0.pi-vscode/dist/extension.cjs:1103-1160`

4. IDE capabilities are registered as runtime tools, not reconstructed later from model text.
   - proof:
     - the bridge bundle calls `pi.registerTool(toolDefinition)` for each tool
   - code:
     - local file `pi0.pi-vscode/bridge/pi-vscode-bridge.js:792`

Pi’s core technique is therefore:
- process-hosted agent runtime
- strict JSONL command/event framing
- typed event dispatch
- native runtime tool registration

### OpenCode implementation technique

OpenCode is implemented as a server-backed system with a separate IDE bridge.

1. The VS Code extension launches the backend as a real subprocess.
   - proof:
     - `spawn(args[0], args.slice(1), { stdio: ["pipe","pipe","pipe"], shell, ... })`
   - code:
     - local file `paviko.opencode-ux-plus/out/backend/BackendLauncher.js:266-275`

2. The backend connection is discovered from backend stdout, not hardcoded into prompt logic.
   - proof:
     - launcher scans stdout lines for `opencode server listening on (https?://...)`
     - parses that URL and derives `uiBase`
   - code:
     - local file `paviko.opencode-ux-plus/out/backend/BackendLauncher.js:289-320`

3. The IDE bridge is a separate session server with explicit route handling.
   - proof:
     - session objects are created with `sessionId` and `token`
     - requests are routed by `/idebridge/{sessionId}/{action}`
     - unauthorized sessions are rejected before action dispatch
   - code:
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:85-99`
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:127-143`

4. Streaming is explicit SSE with route-specific action handlers.
   - proof:
     - `handleSSE()` writes `Content-Type: text/event-stream`
     - `handleSend()` parses JSON `{type,id,payload}` and dispatches typed actions like `openFile` and `openUrl`
   - code:
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:157-175`
     - local file `paviko.opencode-ux-plus/out/ui/IdeBridgeServer.js:176-205`

OpenCode’s core technique is therefore:
- spawned backend server
- server-discovered connection bootstrap
- session/token-scoped IDE bridge
- explicit SSE plus typed action dispatch

### `llama-swap` implementation technique

`llama-swap` is implemented as a translation bridge rather than a native agent runtime.

1. The bridge normalizes one contract into another inside the same large translation unit.
   - proof:
     - `translateResponsesToChatCompletionsRequest()` rewrites Responses input into chat-completions payloads
   - code:
     - local file `proxy/proxymanager.go:2244`

2. The bridge reconstructs streamed Responses events from upstream chat outputs.
   - proof:
     - `writeResponsesStream()` builds Responses SSE from a completed normalized payload
     - `writeResponsesStreamFromChatSSE()` builds Responses SSE incrementally from upstream chat SSE
   - code:
     - local file `proxy/proxymanager.go:6542`
     - local file `proxy/proxymanager.go:9592`

3. Tool behavior is enforced by bridge-side normalization, validation, and repair rather than runtime registration.
   - proof:
     - tool-call parsing and recovery live in `proxy/tool_call_parser.go`
     - request pruning, tool rewriting, malformed-call recovery, and continuation steering all live in bridge code
   - code:
     - local file `proxy/tool_call_parser.go`
     - local file `proxy/proxymanager.go`

`llama-swap`’s core technique is therefore:
- facade API plus translator
- bridge-side event synthesis
- bridge-side tool repair and recovery
- orchestration policy embedded in translation paths

### Implementation-technique conclusion

Pi and OpenCode both put orchestration in a native runtime boundary:

- Pi: process RPC runtime
- OpenCode: backend server plus session bridge

`llama-swap` instead puts orchestration inside translation code:

- request rewriting
- tool schema shaping
- malformed-output repair
- continuation forcing
- stream synthesis

That is the clearest implementation difference proven by the code. It also explains why `llama-swap` currently accumulates Codex/Qwen-specific heuristics in `proxymanager.go`: it is acting as adapter, validator, parser, and orchestrator at the same time.

## Tool Inventory

### Pi

Core built-ins and harness capabilities:

- Built-in file and shell tools documented by Pi: `read`, `write`, `edit`, `bash`
- Strict RPC command/response/event protocol over JSONL
- Streamed tool execution lifecycle events: `tool_execution_start`, `tool_execution_update`, `tool_execution_end`
- Turn lifecycle events: `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`
- Queue and retry events: `queue_update`, `auto_retry_start`, `auto_retry_end`, `compaction_start`, `compaction_end`
- Extension interception points for `tool_call`, `tool_result`, `user_bash`, session lifecycle, prompt shaping, and shutdown

`pi-vscode` bridge tools:

- Inspection: `vscode_get_editor_state`, `vscode_get_selection`, `vscode_get_latest_selection`, `vscode_get_diagnostics`, `vscode_get_open_editors`, `vscode_get_workspace_folders`, `vscode_get_document_symbols`, `vscode_get_definitions`, `vscode_get_type_definitions`, `vscode_get_implementations`, `vscode_get_declarations`, `vscode_get_hover`, `vscode_get_workspace_symbols`, `vscode_get_references`, `vscode_get_code_actions`, `vscode_get_notifications`
- Actions: `vscode_open_file`, `vscode_check_document_dirty`, `vscode_save_document`, `vscode_execute_code_action`, `vscode_apply_workspace_edit`, `vscode_format_document`, `vscode_format_range`, `vscode_clear_notifications`, `vscode_show_notification`

Important Pi harness properties:

- Strict framing contract
- Typed event stream with explicit tool progress
- Built-in queueing and retry visibility
- Extension-level interception before and after tool execution
- Session/control helpers instead of prompt-only steering
- Two concrete integration types in code:
  - JSONL process RPC for the agent session
  - HTTP JSON RPC for the VS Code bridge

### OpenCode

Core backend/server and plugin surfaces visible from installed packages and docs:

- Server mode via `opencode serve`
- Client/server architecture with generated SDKs
- SSE-based server communication
- MCP support with both stdio and SSE transports
- Permission system for MCP tools
- Built-in agent modes including read-only plan-style behavior
- Plugin-supported question tool, retry, file refresh, changed-files tracking, model variants, and command-prefix execution

OpenCode VS Code bridge/plugin surfaces:

- HTTP plus SSE bridge between web UI and IDE
- Session-scoped IDE bridge server
- Path insertion and paste-path context plumbing
- File open, reload, clipboard, UI state, and persisted model/settings state
- Retry button and in-chat session error handling

Important OpenCode harness properties:

- Native backend server contract rather than only prompt/response translation
- Tool and agent behavior exposed as server capabilities
- Explicit permission model
- Session-aware SSE bridge and multi-instance handling
- Subagent/task semantics are part of the backend system, not bolted on in the client
- Two concrete integration types in code:
  - spawned local HTTP backend server for the agent runtime
  - HTTP plus SSE IDE bridge for webview/plugin coordination

### `llama-swap`

Normalized or explicitly handled tools in current bridge code:

- `shell`
- `apply_patch`
- `web_search`, `web_search_preview`
- `file_search`
- `code_interpreter`
- `image_generation`
- `computer`
- `request_user_input`
- `update_plan`
- `spawn_agent`, `send_input`, `resume_agent`, `wait_agent`, `close_agent`
- MCP browser and namespaced tool passthrough via function-call names
- `multi_tool_use.parallel` unpacking

Parser-recovered or repaired shapes:

- Native `tool_calls` / `function_call`
- Qwen XML wrappers such as `<tool_call>`, `<function=...>`, `<tools>...`
- Tagged envelopes such as `<shell_commands>`, `<apply_patch>`, `<tool_use>`
- Function-style pseudo-calls such as `apply_patch(...)`, `update_plan(...)`, `spawn_agent(...)`
- Reasoning-derived fallback recovery for some tools, especially `apply_patch`

## Stage Comparison Matrix

| Stage | Pi | OpenCode | `llama-swap` current state | Main gap |
| --- | --- | --- | --- | --- |
| 1. Incoming client contract | JSONL child-process RPC plus HTTP JSON bridge RPC | spawned local HTTP server plus HTTP/SSE IDE bridge | Mostly OpenAI Responses/chat request intake | No first-class Pi or OpenCode contract adapters |
| 2. Request normalization | Session and extension APIs shape requests structurally | Server owns request/session semantics | Heavy in-function request rewriting in `translateResponsesToChatCompletionsRequest()` | Too much policy is embedded as prompt surgery |
| 3. Upstream translation | Provider/client layers are native to Pi | Backend is native server | Responses -> chat translation is large and special-case heavy | Translation layer is acting as orchestration layer |
| 4. Tool schema exposure | Built-in tools plus typed extension tools | Backend and MCP expose tools as server capabilities | Tool exposure is reconstructed and pruned per request | No canonical per-tool schema/validation registry |
| 5. Upstream output decoding | Native event stream with typed deltas | Native SSE/session events | Decoder relies on mixed native tool calls plus text repair parsers | Too dependent on malformed-output recovery |
| 6. Response normalization | Native agent message/tool result model | Native server model | Chat completion -> Responses normalization is doing repair, policy, and validation at once | Missing clean adapter boundary |
| 7. Stream event emission | Explicit message/tool progress events | Explicit SSE session/event transport | SSE bridge emits Responses-style events and monitor summaries | No generalized lifecycle model across transports |
| 8. Client continuation compatibility | Queue + idle semantics are explicit | Session/server continuity is explicit | Continuation control is partly prompt-injected and partly tool-choice-forced | State machine is implicit and narrow |

## Findings by `llama-swap` Stage

### F1. No first-class harness adapters

- Expected:
  - Pi and OpenCode each define a native harness contract at the transport and session level.
  - Translation between client and provider is separated from orchestration policy.
- Actual in `llama-swap`:
  - `proxy/proxymanager.go` combines request normalization, plan-mode policy, tool exposure, continuation forcing, malformed-output repair, and transport translation in one path.
- First diverging stage:
  - 1. incoming client contract
- Proof artifact:
  - `translateResponsesToChatCompletionsRequest()` and `translateChatCompletionToResponsesResponse()` in `proxy/proxymanager.go`
- Minimal fix target:
  - Extract adapter boundaries from `proxy/proxymanager.go` into dedicated harness adapter modules

### F2. Tool handling is schema-light and parser-heavy

- Expected:
  - Pi extensions define typed tool schemas and the harness exposes pre/post tool interception.
  - OpenCode exposes server-side capabilities and permissioned tool access.
- Actual in `llama-swap`:
  - Tool handling is partly normalized structurally, but much correctness depends on parser recovery in `proxy/tool_call_parser.go` and post-hoc sanitization in response translation.
- First diverging stage:
  - 4. tool schema exposure
- Proof artifact:
  - `normalizeResponsesToolsMap()` in `proxy/proxymanager.go`
  - parser fallbacks throughout `proxy/tool_call_parser.go`
- Minimal fix target:
  - Add a per-tool validator/canonicalizer registry used before exposure and again before execution/result emission

### F3. Continuation control is too prompt-specific

- Expected:
  - Pi exposes queue, idle, shutdown, compaction, and retry semantics as harness state.
  - OpenCode exposes session-level control and agent modes as backend behavior.
- Actual in `llama-swap`:
  - Continuation logic relies on prompt injection and forced `tool_choice` rules, especially around plan mode, `request_user_input`, and `apply_patch`.
- First diverging stage:
  - 8. client continuation compatibility
- Proof artifact:
  - `forcePlanQuestionContinuation`, `forcePlanReturnAfterQuestions`, and apply-patch-specific forcing in `proxy/proxymanager.go`
- Minimal fix target:
  - Introduce a canonical continuation state machine and drive tool choice from state, not prompt-specific heuristics

### F4. Stream lifecycle is transport-specific but not canonically modeled

- Expected:
  - Pi emits explicit tool and message lifecycle events.
  - OpenCode uses server/SSE events as the primary contract.
- Actual in `llama-swap`:
  - `writeResponsesStream` paths and monitors understand several event shapes, but lifecycle semantics are inferred from OpenAI-like events and patch-specific safeguards.
- First diverging stage:
  - 7. stream event emission
- Proof artifact:
  - `proxy/sse_monitor.go`
  - `proxy/chat_sse_event.go`
  - live-stream tool validation logic in `proxy/proxymanager.go`
- Minimal fix target:
  - Define a transport-neutral internal lifecycle: `exposed`, `pending`, `args_delta`, `args_done`, `validated`, `rejected`, `executed`, `output_returned`

### F5. Output repair is too coupled to Qwen/Codex failure patterns

- Expected:
  - Harness corrections should be reusable policy layers, not a growing set of format-specific rescues.
- Actual in `llama-swap`:
  - `tool_call_parser.go` and response translation contain many Qwen/XML/tag/function-style recoveries and reasoning-derived fallbacks.
- First diverging stage:
  - 5. upstream output decoding
- Proof artifact:
  - `parseQwenXMLToolCalls()`, `parseQwenToolsEnvelopeCalls()`, tagged-envelope and function-style fallbacks in `proxy/tool_call_parser.go`
  - reasoning-derived recovery in `translateChatCompletionToResponsesResponse()` in `proxy/proxymanager.go`
- Minimal fix target:
  - Split generic malformed-output recovery from model-family adapters; keep Qwen/Codex-specific recovery behind explicit adapter selection

### F6. Permission and mutation boundaries are only partially represented

- Expected:
  - Pi and OpenCode both encode stronger tool/runtime boundaries: OpenCode has explicit permission controls and read-only agent modes; Pi exposes tool operations and extension controls structurally.
- Actual in `llama-swap`:
  - There is useful tool tiering and plan-mode tool removal, but no generalized policy engine for read-only vs mutating vs approval-required actions across harnesses.
- First diverging stage:
  - 2. request normalization
- Proof artifact:
  - tier mapping and plan-mode pruning in `proxy/proxymanager.go`
- Minimal fix target:
  - Replace special-case plan pruning with a policy layer that applies capability tiers per harness mode

### F7. Agent orchestration is exposed but not deeply normalized

- Expected:
  - OpenCode treats agent/task behavior as backend semantics.
  - Pi exposes queue and session controls in the harness itself.
- Actual in `llama-swap`:
  - orchestration tools are passed through and some names are normalized, but there is not yet a robust orchestration-state contract comparable to native harnesses
- First diverging stage:
  - 4. tool schema exposure
- Proof artifact:
  - orchestrator tool support and prompt guidance in `proxy/proxymanager.go`
  - function-style recovery candidates in `proxy/tool_call_parser.go`
- Minimal fix target:
  - define explicit orchestration-tool schemas and continuation semantics instead of relying on the generic function-call lane

## What `llama-swap` Should Add

### 1. Formal harness adapters

Add explicit adapters instead of continuing to grow shared heuristics:

- `codex_responses_adapter`
- `pi_rpc_adapter`
- `opencode_server_adapter`

Each adapter should own:

- transport framing
- request intake shape
- event/lifecycle translation
- continuation semantics expected by that client

### 2. Schema-first tool validation

Introduce a registry per tool with:

- canonical input schema
- argument repair policy
- rejection policy
- user-visible validation warning text
- output/result normalization rules

`apply_patch`, `shell`, and `request_user_input` should become the first three canonical implementations, then expand to orchestration and MCP tools.

### 3. Canonical continuation state

Add a transport-neutral state machine:

- `pre_tool`
- `tool_running`
- `tool_completed_awaiting_followup`
- `final_answer_required`

Use this state to drive:

- `tool_choice`
- tool pruning
- plan/question behavior
- final-answer forcing
- loop prevention

### 4. Canonical stream lifecycle

Normalize all transports to one internal lifecycle:

- `exposed`
- `pending`
- `args_delta`
- `args_done`
- `validated`
- `rejected`
- `executed`
- `output_returned`

Then map that lifecycle outward to:

- Responses SSE
- Pi JSONL RPC events
- OpenCode SSE/session events
- local monitor and UI events

### 5. Broader loop protection

Generalize beyond plan-mode and `apply_patch`:

- repeated same-call detection
- empty-argument retries
- reasoning-only no-op turn detection
- duplicate continuation forcing suppression
- orphan stream delta detection
- tool-output-satisfied finalization rules for all tools, not just `apply_patch`

## What `llama-swap` Should Correct

### Correct now

- Stop using prompt injection as the main continuation controller
- Stop letting model-family parser recovery stand in for primary tool-call decoding
- Stop keeping harness-specific behavior buried inside `proxy/proxymanager.go`

### Keep, but move behind cleaner abstractions

- `apply_patch` content/diff repair
- malformed shell validation
- plan/question continuation forcing
- MCP name normalization
- commentary/final split handling
- stream orphan detection

## Recommended File-Level Direction

If kept constrained, the next implementation pass should focus on:

- `proxy/proxymanager.go`
  - shrink policy/orchestration logic and delegate to adapters plus registries
- `proxy/tool_call_parser.go`
  - separate generic repair from model-specific adapters
- `proxy/chat_sse_event.go`
  - map transport events to canonical lifecycle entries
- `proxy/sse_monitor.go`
  - monitor canonical lifecycle states instead of raw event-name fragments

If this grows further, extract new files instead of expanding `proxymanager.go` again:

- `proxy/harness_adapter.go`
- `proxy/harness_adapter_codex.go`
- `proxy/harness_adapter_pi.go`
- `proxy/harness_adapter_opencode.go`
- `proxy/tool_contracts.go`
- `proxy/continuation_state.go`

## Test Requirements for the Next Implementation

### Request and adapter tests

- Pi JSONL request/response framing
- OpenCode server/session request shaping
- existing Responses/chat translation regression coverage

### Tool contract tests

- all supported tools normalize to canonical schemas
- unsupported tools fail explicitly
- repaired arguments are marked as repaired
- rejected arguments surface stable validation messages

### Stream and lifecycle tests

- pending -> args_delta -> args_done -> validated -> executed -> output_returned
- orphan delta and orphan done handling
- commentary/final separation remains correct
- Pi and OpenCode transport mapping do not regress Responses clients

### Continuation and loop tests

- post-tool continuation without extra loop
- question-tool continuation
- final-answer forcing after successful tool satisfaction
- repeated identical tool-call suppression
- reasoning-only dead-turn detection

## Bottom Line

Pi and OpenCode both act like native harnesses: they define transport, lifecycle, session control, tool boundaries, and stream semantics as first-class contracts. `llama-swap` currently behaves more like a powerful repair proxy for OpenAI-style traffic, with many useful protections, but too many of those protections are embedded as request rewrites, prompt steering, and model-family parser recovery.

The highest-value change is not one more parser fallback. It is introducing explicit harness adapters plus canonical tool and continuation state. That is the shortest path from "Codex bridge with recoveries" to "general-purpose harness layer for local coding agents."
