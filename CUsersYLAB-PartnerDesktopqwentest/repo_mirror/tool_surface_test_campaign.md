# Llama-Swap Tool-Surface Reliability Campaign

## Summary

Build a repeatable, phase-based test campaign that validates the local Codex + `llama-swap` tool surface one tool family at a time, using the two new skills as fixed workflow roles:

- `llama-swap-auto-repair-loop`
  - owns execution of the campaign
  - runs tests
  - collects artifacts
  - performs the try -> test -> fix -> rebuild -> retest loop
- `llama-swap-contract-forensics`
  - owns ambiguity resolution
  - determines the first wrong stage when a failure is not already localized
  - hands back one concrete fix target before new patches are made

Defaults chosen for this campaign:

- Scope: `Core-first`
- Failure gate: `Stop and fix immediately`
- Test depth: target `5` tests per tool family where practical, allow `3` only when the surface is too small
- Validation rule: a tool family is complete only when all planned tests pass and one repeat run is stable
- Client integrity rule: every tested family must also confirm that CLI-side streamed items and IDs are complete and understandable to Codex
- Agent constraint rule: treat agent and subagent behavior as a `single-GPU serialized workflow` first, not a concurrency-first workflow
- Agent performance rule: gate on `correctness first`, record latency separately, and optimize speed only after behavior is reliable
- Reasoning-call rule: execute only native tool-call items by default; reasoning-derived execution is allowed only as a narrow malformed-Qwen recovery path

The campaign should test both directions:

1. `Codex -> bridge -> model -> bridge -> Codex` for normal tool use
2. `Qwen-style output -> bridge -> Codex` for malformed, partial, or non-native response formats that Codex may reject

## Skill Workflow

For every tool family, the workflow is fixed:

1. Use `llama-swap-auto-repair-loop` to run the planned tests and collect standard artifacts.
2. If the failure target is obvious from the first failing run, stay in `llama-swap-auto-repair-loop` and perform one minimal repair iteration.
3. If the failure target is not obvious, pause patching and switch to `llama-swap-contract-forensics`.
4. `llama-swap-contract-forensics` must identify:
   - the first wrong stage
   - the strongest proof artifact
   - the failure class
   - the minimal fix target
5. Return to `llama-swap-auto-repair-loop` only after the likely patch surface is concrete enough for one minimal repair iteration.
6. Do not advance to the next tool family until the current one is re-tested and stable.

Use the forensic skill immediately when:

- captures, trace logs, and session events disagree
- bridge output looks correct but Codex executes something else or nothing
- MCP exposure or agent orchestration behavior is unclear
- the next patch would otherwise rely on prompt wording instead of a proven contract mismatch

## Tool Families and Order

### 1. Execution Core

Start here because this is the base for almost every other workflow.

Tool families:

- `shell`
- `exec_command`
- `write_stdin`

Required checks:

- single command execution
- working directory handling
- streaming and yield behavior
- multi-turn shell session continuation
- rejection of bad shell-shape outputs or malformed command payloads

Skill use:

- use `llama-swap-auto-repair-loop` for the normal test matrix
- switch to `llama-swap-contract-forensics` if shell output appears in captures but not in CLI events, or if command/session ownership is ambiguous

Failure loop trigger:

- wrong command shape
- output not streamed back correctly
- session continuation broken
- shell result returned in a Codex-incompatible output item

### 2. Runtime Policy and Safety Controls

Run this immediately after basic execution because the rest of the campaign depends on the client and bridge honoring runtime controls correctly.

Policy surfaces:

- `sandbox_permissions`
- `approval_policy`
- read-only vs write-allowed execution expectations
- command rejection and escalation behavior

Required checks:

- command allowed under current baseline policy
- write attempt rejected under read-only or restricted sandbox case
- command requiring elevated or forbidden behavior is blocked exactly as expected
- no silent downgrade from denied command into partial execution
- policy-related failure is surfaced back to Codex in a stable, understandable shape

Skill use:

- use `llama-swap-auto-repair-loop` for the policy test matrix
- switch to `llama-swap-contract-forensics` if the policy appears different between request config, bridge logs, and actual client execution

Failure loop trigger:

- sandbox policy ignored
- command runs when it should be blocked
- command is blocked when baseline config should allow it
- rejection shape is malformed or missing from session events

Current known limitation:

- isolated WSL `codex exec` campaign runs may coerce requested `approval_policy = "on-request"` to active `never` before tool execution
- classify that at the Codex client/runtime-context stage unless evidence shows a later bridge rewrite
- keep sandbox enforcement tests in the `exec` path
- validate true interactive approval prompting only on a client surface that actually supports approval prompts
- treat `/tmp` writability under `workspace-write` according to the active developer permission instructions for that session, even if the abbreviated `turn_context` writable-roots field is narrower
- version-specific note: the installed WSL Codex `0.123.0` binary on this machine rejects `codex exec --approval-policy ...` as an unexpected argument, so source-level docs about that flag must be verified against the live binary before use

### 3. Streaming and Client-Side Event Integrity

Run this before file mutation and advanced tools so we know the CLI client is receiving a complete event stream.

Streaming surfaces:

- item IDs
- `item.started` / `item.completed`
- reasoning summaries
- command and file-change event ordering
- tool argument streaming where applicable

Required checks:

- every emitted item has a stable ID
- no missing `started` or `completed` counterpart for executed items
- reasoning summaries appear in the CLI when reasoning exists
- no tool phase is present in trace/capture but absent from CLI events
- final CLI transcript is consistent with the bridge output and session log

Skill use:

- use `llama-swap-auto-repair-loop` for the expected stream matrix
- switch to `llama-swap-contract-forensics` if captures, SSE behavior, and session events diverge

Failure loop trigger:

- missing IDs
- missing intermediate reasoning or tool-status events
- capture shows tool activity but CLI stream drops it
- CLI final state is correct but intermediate event chain is incomplete or reordered in a confusing way

Current known nuance:

- on the isolated WSL `codex exec` surface, one logical Codex turn can span multiple `/v1/responses` capture IDs
- evaluate streaming integrity against the full capture chain plus the CLI event log, not from a single capture in isolation
- expected pattern is:
  - initial captured SSE exposes tool intent as `function_call` items plus `response.function_call_arguments.*`
  - local Codex CLI then synthesizes `command_execution` or `file_change` events during tool execution
  - later continuation captures return final `message` items ending with `response.output_item.done` and then `response.completed`

### 4. File Mutation Core

Only proceed once execution, policy, and streaming are stable.

Tool families:

- `apply_patch`
- `createfile`
- `updatefile`
- `deletefile`
- alias normalization such as `create_file`, `update_file`, `delete_file`, `createfile`, `updatefile`, `deletefile`

Required checks:

- create new file
- update existing file
- delete existing file
- native `apply_patch` is the default for ordinary file-mutation tasks
- shell-first then patch then verify only when the prompt explicitly asks for ordered mixed behavior like "first use shell ... then use apply_patch ... then verify"
- complex rewrite and multi-file patch behavior
- malformed patch normalization from Qwen-side outputs

Skill use:

- use `llama-swap-auto-repair-loop` for the full create/update/delete and complex rewrite matrix
- switch to `llama-swap-contract-forensics` when first-pass correctness fails but final state later becomes correct, or when continuation-state misalignment is suspected

Failure loop trigger:

- any fallback to shell file writes
- second corrective patch after an initially successful patch when first-pass correctness was expected
- invalid operation shape
- diff/content ambiguity not repaired correctly
- explicit ordered mixed-workflow prompts losing their requested shell -> apply_patch -> verify sequence

### 5. Search and Read Core

Only proceed once execution and patching are stable.

Tool families:

- `websearch`
- `filesearch`
- `read_file`
- `list_files`
- shell `rg` / `rg --files` behaviors where the prompt expects search-style tool use

Required checks:

- direct query handling
- correct result shape
- indexed search vs direct shell fallback separation
- no accidental remapping between search tools and shell
- enumerate the actually exposed native search/read tools on the active client surface before treating a missing tool as a llama-swap regression

Current WSL `codex exec` control-surface note:
- native `web_search` is exposed and executes server-side
- `filesearch`, `list_files`, and `read_file` are not currently exposed as native tools on this surface
- when those absent tools are requested, the model may fall back to `shell` with `rg`, `ls`, or `cat`
- do not classify that fallback as a bridge regression unless captures prove the native tool was actually present and then mishandled

Skill use:

- use `llama-swap-auto-repair-loop` for the search/read matrix
- switch to `llama-swap-contract-forensics` if the model emits one search tool while the client appears to execute another, if server-side search is present in captures but missing from expected bridge output, or if a supposedly native search tool is not actually present in the request tool list

Failure loop trigger:

- empty or malformed result envelopes
- shell being used where a native search tool should be exposed
- native search tool being emitted in a way Codex cannot execute

### 6. Planning and Interaction Core

Stabilize non-mutating workflow behavior before advanced delegation.

Tool families:

- `update_plan`
- `request_user_input`

Required checks:

- plan state transitions
- plan-only mode non-execution
- question rendering and answer return path
- no file mutation or tool execution in plan mode unless explicitly allowed by the client contract

Current WSL `codex exec` control-surface note:
- `update_plan` is exposed and currently renders into CLI as a `todo_list`
- `request_user_input` is advertised in the request tool list, but the client router rejects it in Default mode with `request_user_input is unavailable in Default mode`
- treat that rejection as a client-surface limitation unless captures show llama-swap malformed the tool call first

Skill use:

- use `llama-swap-auto-repair-loop` for the plan/question matrix
- switch to `llama-swap-contract-forensics` if plan-mode requests are being hijacked into tool execution or if rendered interaction events differ from what the bridge emits

Failure loop trigger:

- plan mode acts instead of planning
- missing or malformed user-input return path
- wrong streamed item types for reasoning, plan, or question events

### 7. Agent and Subagent Orchestration

Test this after the core native workflow is stable, and treat it as a special single-GPU context-swap subsystem.

Tool families:

- `spawn_agent`
- `send_input`
- `resume_agent`
- `wait_agent`
- `close_agent`

Single-GPU assumptions:

- only one agent or subagent should be expected to actively consume model context at a time in the first campaign
- parent-child delegation should be tested as serialized handoff, not as true parallel throughput
- latency from context swapping is expected and should be measured, but not treated as a functional failure by itself

Required checks:

- child agent creation
- child agent receives task-local context correctly
- serialized parent -> child -> parent handoff works
- `wait_agent` correctly blocks until the delegated turn completes or times out
- `resume_agent` returns to the correct state
- `close_agent` reliably terminates the child session
- no context bleed between parent and child
- no duplicate or conflicting tool phases caused by context swapping

Required first-wave tests:

- single child spawn with no overlap
- child receives one bounded task and returns
- parent waits, receives result, and finishes
- child paused then resumed
- child closed, then rejected on further input

Deferred until later:

- overlapping multi-child delegation
- true parallel subagent stress
- throughput-based optimization across simultaneous agents

Skill use:

- use `llama-swap-auto-repair-loop` for serialized handoff tests and timing capture
- switch to `llama-swap-contract-forensics` if parent and child logs disagree, if session ownership is unclear, or if context-swapping appears to reorder outputs or events

Failure loop trigger:

- unsupported-call behavior
- wrong child/session identifiers
- context bleed between agents
- handoff result arrives in the wrong session
- continuation resumes the wrong agent
- bridge or client confuses orchestration events with normal tool execution

Pass rule for this phase:

- correctness first
- latency recorded for each spawn, wait, resume, and close path
- slow context swap is acceptable if behavior is correct and stable
- known surface caveat from the first control-model batch:
  - verify agent-tool correctness against parent/child rollout logs, not only the CLI event log
  - on WSL `codex exec` `0.123.0`, the raw CLI event stream can render a real `resume_agent` step as a generic `wait` collaboration item even when the session rollout proves `resume_agent` was actually emitted
  - the campaign harness now reconciles `collab_tool_call` names against the authoritative rollout and preserves the original CLI label as `rendered_tool`, so `*_events.jsonl` should be treated as canonicalized evidence rather than a byte-for-byte dump of the raw rendered label
- known prompt-shaping caveat from the first control-model batch:
  - loose multi-child prompts may terminate after the second spawn without emitting the second `wait_agent`
  - use an explicit serialized prompt for canonical validation:
    - spawn child 1
    - wait child 1
    - spawn child 2
    - wait child 2
    - close child 1
    - close child 2
    - final sentinel reply
- known improvement after the first repair loop:
  - the bridge now injects serialized orchestration guidance and forces `parallel_tool_calls=false` when prompts clearly describe agent/subagent workflows
  - this improved the loose two-child control prompt from early termination to a full successful serialized chain
  - keep validating the loose prompt class anyway, because resume-label rendering and timeout behavior remain partially client-side
- additional confirmed Phase 7 behaviors after retest:
  - `wait_agent` short-timeout followed by retry-to-completion works on this surface
  - `send_input` to a closed child without resume fails cleanly with `not_found`
  - a real `resume_agent` step can now be recovered into saved campaign artifacts as `tool:"resume_agent"` with `rendered_tool:"wait"`
  - treat these as positive functional checks even if the raw CLI collaboration-tool labels are imperfect

### 8. MCP Resource Layer

Treat as a distinct subsystem because the current evidence suggests client-side availability can differ from bridge exposure.

Tool families:

- `list_mcp_resources`
- `list_mcp_resource_templates`
- `read_mcp_resource`

Required checks:

- registered server visibility
- pagination or cursor handling
- unknown-server behavior
- separation between MCP registration problems and bridge formatting problems

Skill use:

- use `llama-swap-auto-repair-loop` for the MCP resource matrix
- switch to `llama-swap-contract-forensics` as soon as MCP registration, tool exposure, and actual session availability do not line up

Failure loop trigger:

- resources unavailable despite registration
- unsupported server or URI behavior
- bridge emits correct shapes but Codex session cannot access MCP state
- first confirmed control-model result on WSL `codex exec`:
  - registered MCP servers can be surfaced as native `mcp_tool_call` items in-session
  - the Playwright MCP server currently returns `-32601 Method not found` for:
    - `list_mcp_resources`
    - `list_mcp_resource_templates`
  - treat this as an MCP-server capability mismatch, not a llama-swap bridge-format failure

### 9. Playwright MCP and Computer Layer

Keep separate from MCP resources because the action surface behaves differently.

Tool families:

- `mcp__playwright__browser_*`
- `computer`
- `view_image`

Required checks:

- real MCP browser tool exposure
- `file://` navigation handling
- snapshot / evaluate / screenshot flow
- fallback behavior when MCP is unavailable
- distinction between `computer` tool and actual MCP Playwright execution

Skill use:

- use `llama-swap-auto-repair-loop` for the concrete browser and image interaction matrix
- switch to `llama-swap-contract-forensics` if registered MCP servers do not appear as callable session tools, or if the client silently replaces MCP behavior with shell or `computer` fallbacks

Failure loop trigger:

- MCP tools are registered but not surfaced to the session
- namespace-root or leaf-tool mismatch
- shell fallback masks missing MCP execution
- client exposes only `computer` while the prompt expects MCP browser tools

Current known WSL `codex exec 0.123.0` finding:

- Phase 9 browser leaf-tool testing progressed from silent tool loss to explicit router rejection
- after the compatibility patch, browser calls persist as `function_call` items like `mcp__playwright__browser_navigate`
- the remaining failure is client-side:
  - `function_call_output` returns `unsupported call: mcp__playwright__browser_navigate`
- treat this as a Codex client/router limitation on this surface before blaming `llama-swap` again

Windows priority note for Playwright validation:

- prefer the Windows Codex client surface over WSL for final Playwright MCP validation
- on Windows, browser tools execute correctly when Codex is launched from a real local Windows working directory
- if launched from a UNC/WSL host context, Playwright MCP may fail with:
  - `EPERM: operation not permitted, mkdir 'C:\Windows\.playwright-mcp'`
- so the campaign should record the launch working directory as part of every Windows Playwright repro

### 10. Optional Extended Surface

Plan but defer until the first nine families are stable.

Tool families:

- `notebook_read`
- `notebook_edit_cell`
- `memory_read`
- `memory_write`

These should remain out of the first pass unless logs or captures prove they are currently exposed by the local Codex path.

Current reconnaissance status:

- no evidence yet in the active WSL or Windows campaign session artifacts that `notebook_read`, `notebook_edit_cell`, `memory_read`, or `memory_write` are exposed
- Phase 10 therefore stays in exposure-confirmation mode first
- do not open a bridge repair branch for these tools unless a real session proves they are surfaced or surfaced-but-rejected
- current supporting evidence:
  - WSL campaign session searches returned no occurrences of those tool names
  - Windows campaign session searches returned no occurrences of those tool names
  - the nearby `list_files` request already showed the expected absence pattern on this surface:
    - the model explicitly noted it did not have a `list_files` tool and fell back to `shell`

Skill use:

- use `llama-swap-contract-forensics` first to confirm actual exposure
- only then add them to `llama-swap-auto-repair-loop` as normal tested families

## Per-Tool Test Method

For each tool family:

1. Define `3-5` canonical prompts.
2. Define exact expected artifacts:
   - session events
   - capture shape
   - bridge trace summary
   - file state or returned payload
3. Define expected policy behavior where relevant:
   - allowed
   - blocked
   - requires escalation
4. Define expected streaming behavior:
   - expected item IDs
   - expected started/completed transitions
   - expected reasoning or tool-progress visibility
5. For agent and subagent cases, also define:
   - expected session ownership
   - expected serialized handoff order
   - acceptable latency as observational data
6. Run the minimal test set with `llama-swap-auto-repair-loop`.
7. Stop on the first failing test.
8. If the fix surface is obvious, continue in `llama-swap-auto-repair-loop`.
9. If the fix surface is ambiguous, switch to `llama-swap-contract-forensics`.
10. Return to `llama-swap-auto-repair-loop` for exactly one minimal repair iteration.
11. Do not continue to the next tool family until:
   - all tests for the current family pass
   - one repeat run also passes
   - the result is logged into the repair record and skill knowledge

Per-test evidence should always include:

- latest `/api/metrics` rows
- relevant `/api/captures/:id`
- matching session `rollout-*.jsonl`
- `tmp/*_events.jsonl`
- `tmp/*_last.txt`
- `tmp/*_stderr.txt` when relevant
- `/tmp/llama-swap-apply-patch-trace.log`

For agent and subagent tests, also record:

- parent session log
- child session log if separately created
- timing for spawn, wait, resume, and close
- whether only one session was actively consuming model work at a time

## Reverse-Direction Qwen Contract Tests

For every tool family where Qwen formatting can break Codex, add a second test group for malformed or non-native model output.

Priority reverse-direction cases:

- `<think>...</think>` leakage before tool output
- XML-style tool calls instead of JSON-native calls
- malformed JSON arguments such as missing colons or broken quotes
- sequential tool emissions that should be coalesced into one Codex-understandable turn
- alias spellings and schema drift in `apply_patch` operations
- reasoning content leaking into visible response or tool payloads
- reasoning-derived tool execution occurring outside the approved malformed-output recovery signatures
- agent orchestration outputs that arrive in generic tool-call form instead of the client-understood orchestration shape

These reverse-direction tests should not be run as generic prose tests. They should be attached to the tool family they threaten:

- execution core: malformed shell or command call shapes
- file mutation core: malformed `apply_patch` operation bodies
- planning core: plan-tag and reasoning leakage
- execution contract: native tool-call items vs guarded reasoning-derived repair behavior
- streaming integrity: missing item IDs, missing reasoning deltas, missing output item completion
- agent orchestration: session or agent identity drift
- MCP layer: namespace vs callable-tool drift

Use `llama-swap-contract-forensics` by default for reverse-direction failures, because these are often first-wrong-stage classification problems before they are patching problems.

## Cross-Cutting Acceptance Checks

Every phase must include these cross-checks before being marked complete:

- sandbox behavior matched the planned policy for that phase
- approval behavior matched the planned policy for that phase
- all expected stream items appeared in the CLI
- no event ID or intermediate phase was missing from the CLI compared with captures and trace logs
- no hidden fallback path made the final answer look correct while bypassing the intended tool contract
- no reasoning-derived tool call executed unless the raw malformed output matched an approved recovery signature such as Qwen XML/tool envelope or validated fragmented `apply_patch`
- for agent/subagent phases, serialized single-GPU handoff behaved correctly even when slower than normal tool calls
- if a failure required deep blame isolation, the forensic skill was used and its conclusion was recorded before patching
- if `approval_policy` differs between requested config and active session metadata, classify that at the client/runtime stage before attributing it to llama-swap

## Test Infrastructure and Deliverables

Create and maintain a reusable campaign structure:

- one checklist or matrix of tool families
- `3-5` prompts per family
- expected artifact rules per prompt
- expected sandbox and approval outcome per prompt where relevant
- expected stream-event sequence per prompt
- expected parent/child session sequence for agent tests
- pass/fail log for each family
- one field showing whether `llama-swap-auto-repair-loop` alone was enough or whether `llama-swap-contract-forensics` was required
- a known-good baseline section for the active control model
- a second section for “model emits what Codex cannot parse”

Use the existing repair workflow as the enforcement engine:

- `llama-swap-auto-repair-loop` owns execution, patching, rebuilds, and retests
- `llama-swap-contract-forensics` owns first-wrong-stage isolation when evidence is ambiguous
- after each repaired family, update both the repair log and the skills with only confirmed, re-tested findings

## Assumptions and Defaults

- Primary control model remains `gpt-5.2` backed by `Qwen3.6-35B-A3B-UD-Q8_K_XL` until another model is intentionally substituted
- WSL local Codex plus `llama-swap` remains the primary fix environment
- Windows Codex remains a later confirmation surface, not the first repair surface
- MCP and agent orchestration are included in the campaign, but only after the core native tools are stable
- Notebook and memory tools are planned as optional unless local evidence proves they are currently exposed
- “Working as expected” for policy tests means behavior matches the active test config exactly, not a generic OpenAI-hosted default
- In the first agent campaign, reliability of serialized delegation matters more than raw parallelism because the current hardware path is a single-GPU context-swapping system
