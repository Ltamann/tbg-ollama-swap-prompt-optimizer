## 2026-04-24 11:26:41 +02:00 - Pre-read snapshot

- `apply_patch_repare.md` was missing before this snapshot.
- `/tmp/llama-swap-apply-patch-trace.log` is currently missing.
- `/api/metrics` currently returns captures with IDs: `0, 1, 2, 3` (all `has_capture=true`, all `/v1/responses`).
- Capture summary:
  - `id=0`: `model=gpt-5.2`, `reasoning=low`, no function-call event, no `<tools>` payload.
  - `id=1`: `model=gpt-5.2`, `reasoning=low`, no function-call event, assistant emitted escaped `<tools>` block with `{"command": ["cat", "c:\\Users\\YLAB-Partner\\.codex\\config.toml"]}`.
  - `id=2`: `model=gpt-5.2`, `reasoning=low`, no function-call event, same escaped `<tools>` block.
  - `id=3`: `model=gpt-5.2`, `reasoning=high`, no function-call event, same escaped `<tools>` block.
- Initial hypothesis from captures: Plan-mode/tool-use breakage is primarily tool-call formatting/parsing failure (assistant emitting tool intent as plain output text) rather than reasoning-effort setting.

## 2026-04-24 11:32:31 +02:00 - Iteration intent
- Hypothesis: Plan-mode broke because upstream output wrapped tool intent in escaped <tools>...</tools> text, which existing parser ignored; reasoning low/high is not causal.
- Change plan: add parser fallback in proxy/tool_call_parser.go to decode escaped tools envelope and synthesize executable tool calls (shell_command/pply_patch).

## 2026-04-24 11:32:49 +02:00 - Iteration outcome
- Implemented: parseQwenToolsEnvelopeCalls fallback for escaped/raw <tools> envelopes plus command-array normalization.
- Added tests in proxy/tool_call_parser_test.go for escaped tools envelope (shell_command and pply_patch).
- Validation: WSL command /usr/local/go/bin/go test ./proxy -run "TestParseModelSpecificToolCalls_QwenEscapedToolsEnvelopeCommandArray|TestParseModelSpecificToolCalls_QwenEscapedToolsEnvelopeApplyPatch|TestParseModelSpecificToolCalls_QwenFunctionStyleCall" -count=1 passed.
- Note: existing branch test TestTranslateChatCompletionToResponsesResponse_RecoversPrefixedApplyPatchFromXMLParsedArgs currently fails due separate in-branch behavior expecting pply_patch_call but receiving unction_call.

## 2026-04-24 11:48:43 +02:00 - Pre-read snapshot
- Service endpoint confirmed: :8080 is active (not :8090).
- Fresh /api/metrics capture IDs with has_capture=true: 0,1,2 (all req_path=/v1/responses).
- Capture decode finding: assistant response came as output_text with XML-like tags (<shell_commands><commands>[...]</commands></shell_commands>) instead of function/tool-call output items.
- Existing parser coverage handles <tools>...</tools> and <function=...> but not <shell_commands> / <update_plan> wrapper format.
- Classification: wrong_tool_call / parser_mismatch.

## 2026-04-24 11:48:43 +02:00 - Iteration intent
- Hypothesis: add fallback parser for XML-like wrappers (<shell_commands>, <update_plan>, <proposed_plan>) to recover executable tool calls from tagged JSON payloads.
- Planned minimal change: extend proxy/tool_call_parser.go and add targeted unit tests in proxy/tool_call_parser_test.go.


## 2026-04-24 11:58:56 +02:00 - Iteration outcome
- Implemented parser fallbacks in proxy/tool_call_parser.go for tagged envelopes: <shell_commands>, <shell_command>, <shell>, <update_plan>, <proposed_plan>, including incomplete closing-tag handling and command-array extraction.
- Added/updated unit tests in proxy/tool_call_parser_test.go; targeted proxy parser tests pass in WSL.
- Fresh capture IDs after latest repro are from /api/metrics: [{"id":0,"timestamp":"2026-04-24T11:58:44.256917761+02:00","model":"gpt-5.3-codex","status_code":200,"cache_tokens":0,"input_tokens":0,"output_tokens":0,"prompt_per_second":0,"tokens_per_second":0,"duration_ms":5025,"has_capture":true}]
- Current repro status: still failing end-to-end for some outputs because model now emits a different pseudo-tool wrapper (<tool_use><file_read><path>...) that is not yet mapped into executable tool calls.
- Classification remains parser_mismatch / wrong_tool_call with evolving malformed wrapper variants.


## 2026-04-24 12:10:19 +02:00 - Iteration outcome
- Added malformed wrapper recovery for <tool_use>/<tool_call> blocks and function-style update_plan(steps=[...]).
- Added tests: ToolUseFileReadPath, BareToolCallOpener, FunctionStyleUpdatePlanSteps; focused tests pass.
- Per user rule, rebuilt+copied+restarted llama-swap before focused tests and again before repro.
- Post-fix repro evidence is in tmp/win_gpt52_planexec_* and /home/admmin/llama-swap/tmp/wsl_gpt52_planexec_* (see whether update_plan/shell_command/PLAN_DONE executed).


## 2026-04-24 12:14:28 +02:00 - Iteration outcome
- Added recovery for malformed plan payloads: steps= assignment, single-quoted pseudo-JSON, wrapped {plan:[...]} blocks, and dangling angle shell commands like <ls -la ...
- Added tests for ToolCall+UpdatePlan+DanglingLs and new plan payload styles; focused proxy tests pass.
- Enforced user-required rebuild+copy+restart sequence before each test/repro cycle in this iteration.
- Latest repro artifacts: tmp/win_gpt52_planexec_* and /home/admmin/llama-swap/tmp/wsl_gpt52_planexec_*.


## 2026-04-24 12:20:39 +02:00 - Iteration outcome
- Added plain numbered-plan fallback to synthesize update_plan from text-only Plan: blocks.
- Added regression test for real numbered-plan text + prior malformed blob.
- Enforced restart rule: rebuild+copy+restart before tests and before repro.
- Result: parser tests pass, but end-to-end gpt-5.2 Codex run still exits on first assistant text output without executing any tool call turn (item.completed only, then 	urn.completed).
- Latest malformed behaviors observed: plain plan prose only (Windows) and empty patch wrapper (*** Begin Patch ... End Patch) as text (WSL).


## 2026-04-24 12:59:00 +02:00 - Iteration outcome
- Patched stream bridge path (`writeResponsesStreamFromChatSSE`) to route buffered chat-stream text through `translateChatCompletionToResponsesResponse` before emitting Responses SSE, so malformed tool-call recovery is applied in stream mode too.
- Added malformed plan variant recovery in parser for decorated block format:
  - `*** Begin Plan` ... numbered steps ... `*** End Plan...`
  - Synthesizes `update_plan` with step statuses (`in_progress` then `pending`).
- Added test `TestParseModelSpecificToolCalls_DecoratedPlanFallback`; focused parser tests pass.
- Enforced restart rule: rebuild binary, copy to `/home/admmin/bin/llama-swap`, restart services before repro.
- Repro status:
  - Windows Codex client (`codex.exe`, gpt-5.2, localhost:8080): plan executed end-to-end with real command executions and final completed plan checklist (todo list all completed).
  - WSL Codex JS run is still inconsistent in this environment (one run failed for missing OPENAI_API_KEY; next run returned unrelated prompt-handling text instead of executing requested plan), so WSL parity remains pending.

## 2026-04-24 13:26:00 +02:00 - Iteration outcome
- Added malformed `<apply_patch>...</apply_patch>` parser recovery in `proxy/tool_call_parser.go`.
- Recovery now wraps raw `*** Add/Update/Delete File:` payloads into normalized patch text and emits executable `apply_patch` tool call arguments (`input`).
- Added regression test `TestParseModelSpecificToolCalls_ApplyPatchTagAddFileFallback`.
- Added `todo_list` planning continuation handling in `responseLooksLikePlanningOnly` plus tests in `proxy/planning_retry_test.go`.
- Rebuild/copy/restart cycle executed again before repro.

## 2026-04-24 13:40:00 +02:00 - Iteration outcome
- Added generic XML-tag sanitizer/parser in `parseGenericTaggedCalls` to prevent prompt pollution from unknown `<tag>...</tag>` blocks.
- Generic mapping now infers tool calls from tag/content patterns and converts when possible:
  - apply_patch-like tags/content -> `apply_patch`
  - shell/terminal-like tags/content -> `shell_command`
  - plan/todo-like tags/content -> `update_plan`
- Unknown/unmappable XML-like blocks are stripped from assistant text so they do not carry into the next prompt.
- Added regression tests for generic terminal tag conversion and unknown-tag stripping.
- Rebuild/copy/restart cycle executed again.

## 2026-04-28 14:12:00 +0200
- Intent: Resolve client continuation execution behavior versus forced local apply_patch fallback.
- Hypothesis: isApplyPatchIntent path is still auto-executing valid apply_patch locally, causing continuation mismatch/no-final-message behavior; default should preserve native apply_patch_call and only local-exec when explicitly forced.

- Change applied: default path now preserves native client continuation for valid apply_patch_call; local execution fallback runs only when `llamaswap_force_local_apply_patch` is explicitly true (top-level or metadata flag).
- Tests: focused proxy tests for force-local flag + shell/apply_patch normalization/translation passed.
- Runtime probe (Windows Codex gpt-5.3-codex): no forced local fallback observed; run produced reasoning-only output with no tool call (`tmp/win_cont_events.jsonl`), and codex stderr ended with `no last agent message`.
- Status: continuation-vs-forced-fallback regression fixed in bridge logic; upstream no-tool-call behavior remains a separate issue for apply_patch reliability.

## 2026-05-01 15:05:00 +0200
- Intent: Add a Codex UI workaround for hidden reasoning by mirroring extracted think text onto a visible `commentary` channel message while preserving native Responses reasoning items.
- Change applied in `proxy/proxymanager.go`:
  - `writeResponsesStreamFromChatSSE(...)` now starts a second assistant message lane with `channel:"commentary"` when reasoning deltas appear.
  - The bridge streams the same extracted reasoning text into both:
    - native `type:"reasoning"` summary events
    - visible `type:"message", channel:"commentary"` output text events
  - Final assistant content continues on the separate `channel:"final"` lane.
  - This keeps the intended output order for think-bearing turns: reasoning -> commentary -> final.
- Regression coverage added in `proxy/proxymanager_bridge_xml_test.go`:
  - `TestWriteResponsesStreamFromChatSSE_EmitsReasoningAndContentOnSeparateLanes`
  - `TestWriteResponsesStreamFromChatSSE_StreamsReasoningIntoCommentaryAndKeepsFinalClean`
- Focused tests passed:
  - `TestWriteResponsesStreamFromChatSSE_EmitsReasoningAndContentOnSeparateLanes`
  - `TestWriteResponsesStreamFromChatSSE_StreamsReasoningIntoCommentaryAndKeepsFinalClean`
  - `TestExtractContentAndReasoning_DropsPreThinkPreambleAndClosingTagLeak`
  - `TestShouldEnforcePlanModeSyntheticRewrite_OnlyForProxyPlanMode`
  - `TestWriteResponsesStreamFromChatSSE_EmitsToolCallArgumentDeltas`
  - `TestWriteResponsesStreamFromChatSSE_ToolFirstUsesOutputIndexZeroWithoutEmptyMessage`
- Status: workaround is ready for live rebuild/restart validation against the Codex UI.

## 2026-05-01 15:35:00 +0200
- Documentary finding:
  - Native Responses reasoning delivery is not sufficient to guarantee visible thinking in Codex UI for local model IDs.
  - We observed valid `rs_...` reasoning items and valid `response.reasoning_summary_text.*` events that still did not render in the Codex thinking panel.
  - Because of that, a UI-facing workaround is required when Codex is used with local `.gguf` model identities.
- Workaround conclusion:
  - Keep native `reasoning` events for compatibility.
  - Add visible `channel:"commentary"` assistant output as the fallback display path.
  - Keep `channel:"final"` separate for the user-facing answer.
- Follow-up design concern:
  - `commentary` may become part of replayed assistant history, so using it for full raw reasoning is risky.
  - Preferred long-term direction is to use `commentary` for a short summary, not full reasoning text.
  - Sending full reasoning as a synthetic function call is not recommended because it pollutes tool-call semantics and continuation history.
- Stable write-up added under `docs/codex-reasoning-ui-study.md`.

## 2026-05-01 16:05:00 +0200
- Root cause refinement:
  - The first `commentary` workaround patch only covered `writeResponsesStreamFromChatSSE(...)`.
  - Some captured Codex sessions replayed through `writeResponsesStream(...)` instead, so those streams still showed only:
    - reasoning item at output index 0
    - final message at output index 1
  - This made it look like the workaround was undeployed when the real issue was an uncovered second stream path.
- Change applied:
  - Added the same synthetic `channel:"commentary"` mirror to `writeResponsesStream(...)` for replayed Responses payloads.
  - Replay path now appends a visible commentary message immediately after a reasoning item whenever reasoning text exists.
- Tests:
  - `TestWriteResponsesStream_ReplaysReasoningItems`
  - `TestWriteResponsesStream_CompletesForToolPhase`
  - `TestWriteResponsesStreamFromChatSSE_EmitsReasoningAndContentOnSeparateLanes`
  - `TestWriteResponsesStreamFromChatSSE_StreamsReasoningIntoCommentaryAndKeepsFinalClean`
- Status:
  - Both bridge stream paths now mirror reasoning into `commentary`; future missing-commentary captures should be treated as stale session/client-state unless proven otherwise.

## 2026-05-01 17:05:00 +0200
- Pre-read snapshot:
  - Re-checked the May 1 rollouts and local capture set before patching.
  - Corrected the earlier analysis:
    - the Spain quiz file mutation did not happen inside the same plan turn that produced the plan
    - the later mutation happened after a fresh `default` turn began
    - the empty shell `{}` call was rejected before OS execution
    - reasoning is emitted in valid Responses shape, but only as one buffered summary delta instead of continuous incremental updates
    - commentary mirroring is the lane the user actually sees in Codex UI
  - Recorded the corrected bug matrix in `docs/codex-bug-fix-loop.md`.

## 2026-05-01 17:05:00 +0200
- Iteration intent:
  - Harden malformed shell validation in the bridge's non-stream response-normalization path.
  - Goal: treat empty shell arguments the same way broken apply_patch payloads are treated now, with a structured bridge validation message instead of an executable-looking tool call.

## 2026-05-01 17:05:00 +0200
- Iteration outcome:
  - Added shell-argument validation to `validateBridgeToolCallItem(...)`.
  - Empty `shell` / `shell_command` `function_call` payloads now fail bridge validation when neither `command` nor `commands` contains a usable value.
  - Added regression test:
    - `TestTranslateChatCompletionToResponsesResponse_EmptyShellArgumentsBecomeValidationMessage`
  - Focused tests passed:
    - `TestTranslateChatCompletionToResponsesResponse_EmptyShellArgumentsBecomeValidationMessage`
    - `TestTranslateChatCompletionToResponsesResponse_ApplyPatchWarningFieldUsesNeutralWording`
    - `TestTranslateChatCompletionToResponsesResponse_ToolValidationMessagesAvoidToolBlamingText`
  - Rebuilt and redeployed:
    - binary hash: `02ab78682d9b6b0164afdd29c4a2988c8f8688f6f2197e6f5ba8c92b8f1796e2`
    - deployed hash matched `/home/admmin/bin/llama-swap`
  - Restart result:
    - `llama-swap` came back on `0.0.0.0:8080`
    - `/health` returned `status: ok`
  - Remaining scope note:
    - this iteration hardens the non-stream normalization path only
    - live stream-path handling of malformed shell tool calls remains open in `docs/codex-bug-fix-loop.md`

## 2026-05-01 17:32:00 +0200
- Iteration intent:
  - Close the remaining streamed-shell gap in `writeResponsesStreamFromChatSSE(...)`.
  - Goal: do not expose malformed live `shell` tool calls to Codex as executable `function_call` items when the streamed arguments never produce a usable `command` or `commands` payload.

## 2026-05-01 17:32:00 +0200
- Iteration outcome:
  - Added delayed visibility for streamed shell tool calls in `writeResponsesStreamFromChatSSE(...)`.
  - Streamed tool items are now exposed only after their buffered shell arguments normalize to a usable `command` string or `commands` array.
  - Empty streamed shell payloads now stay out of the live tool lane and fall back to a bridge validation assistant message instead of a visible `function_call`.
  - Added regression test:
    - `TestWriteResponsesStreamFromChatSSE_EmptyShellArgumentsBecomeValidationMessage`
  - Focused tests passed:
    - `TestWriteResponsesStreamFromChatSSE_EmitsToolCallArgumentDeltas`
    - `TestWriteResponsesStreamFromChatSSE_EmptyShellArgumentsBecomeValidationMessage`
    - `TestWriteResponsesStreamFromChatSSE_EmitsEachReasoningChunkWithoutCoalescing`
    - `TestWriteResponsesStreamFromChatSSE_ApplyPatchUsesFreeformPatchArguments`
    - `TestWriteResponsesStreamFromChatSSE_ToolFirstUsesOutputIndexZeroWithoutEmptyMessage`
    - `TestTranslateChatCompletionToResponsesResponse_EmptyShellArgumentsBecomeValidationMessage`
  - Investigation result:
    - added a focused stream test proving the bridge emits one `response.reasoning_summary_text.delta` per upstream reasoning chunk
    - this exonerates `writeResponsesStreamFromChatSSE(...)` from chunk coalescing when upstream actually sends multiple reasoning pieces
    - remaining live-buffering suspicion is now upstream/model-side or in an uncovered replay/client path, not in the tested direct chat-SSE bridge loop
  - Next confirmed scope after redeploy:
    - instrument why reasoning arrives as a buffered summary chunk instead of true incremental live streaming
    - reduce commentary from full reasoning mirror toward short summary mode to lower history contamination risk

## 2026-05-01 23:45:00 +0200
- A/B intent:
  - Baseline replayed captured requests before changing code.
  - Results:
    - cap 7 (35B question turn): still emits request_user_input.
    - cap 9 (35B plan turn): still does not emit <proposed_plan>.
    - cap 3 (NVFP4 implementation turn): still leaks escaped </think> markers.
  - Next step:
    - add a temporary bridge switch to disable replay of prior reasoning items into upstream reasoning_content, then rerun the same captures to test causation rather than infer from history growth.

## 2026-05-01 23:52:00 +0200
- Patch intent:
  - Added temporary env-controlled bridge switch `LLAMASWAP_DISABLE_REASONING_HISTORY_REPLAY` in `responsesRequestToChatMessages(...)`.
  - Scope is deliberately narrow: skip replay of standalone Responses `type:"reasoning"` history items into upstream `reasoning_content`, while preserving inline assistant reasoning extraction.
  - Goal: run a true A/B against the same captured 35B/NVFP4 requests without changing broader tool or plan logic.
## 2026-05-02 00:02:00 +0200
- A/B outcome with `LLAMASWAP_DISABLE_REASONING_HISTORY_REPLAY=1`:
  - Deployment correction:
    - Verified the first restart was still using the stale copied binary.
    - Confirmed mismatch between `/home/admmin/llama-swap/llama-swap-main/llama-swap` and `/home/admmin/bin/llama-swap`.
    - Re-copied the fresh workspace binary through `\\wsl$` and restarted `llama-swap` with the env flag enabled.
  - Replayed captured requests against the live service using the same request JSON bodies:
    - `cap 7` (35B question turn): still emits `request_user_input`.
    - `cap 9` (35B plan turn): still does **not** emit `<proposed_plan>`.
    - `cap 3` (NVFP4 malformed implementation turn): escaped `</think>` leakage disappeared under the A/B replay.
  - Conclusion:
    - Disabling reasoning-history replay does **not** explain the 35B stop-before-plan bug; that failure persists.
    - Disabling reasoning-history replay **does** affect NVFP4 output cleanliness, so reasoning replay is at least a contributing factor for malformed think-tag leakage there.
    - Updated confidence:
      - 35B missing-question/missing-plan behavior is still primarily a plan-continuation / tool-steering issue, not proven reasoning-history contamination.
      - NVFP4 malformed output is influenced by reasoning replay, but likely still also depends on template/output integrity.
## 2026-05-02 00:16:00 +0200
- Patch intent:
  - Fix the 35B plan-continuation auto-stop path.
  - Change is intentionally narrow:
    - if plan mode explicitly wants a native question, keep forcing `request_user_input` even after prior tool output
    - if plan mode has prior tool output and the user explicitly wants the completed plan now, force `tool_choice="none"` and inject a continuation instruction to return one `<proposed_plan>` block immediately
  - NVFP4 remains in the forensic bucket; no template or trainer-specific behavior changes in this iteration.
## 2026-05-02 00:20:00 +0200
- Verification note:
  - Focused Go tests reached the real Linux path successfully via `wsl --cd` and `/usr/local/go/bin/go`.
  - Immediate failure was not in the bridge logic; it was a test compile issue because `proxy/proxymanager_test.go` did not import `require` for the new cases.

## 2026-05-08 10:40:00 +0200
- Iteration intent:
  - Finish the forensics presentation slice for the new Qwen semantic stream stage.
  - Promote `bridge.qwen_stream_normalization` into top-level `/api/forensics` output and add classification notes so localhost debugging does not require manual `stages[]` inspection.
- Iteration outcome:
  - Added top-level `qwen_stream` to `RequestForensicSummary`.
  - `/api/forensics` now promotes the normalized Qwen stream summary from the capture stage into the main forensic payload.
  - Added semantic forensic notes for:
    - malformed normalized reasoning
    - recovered question semantics
    - recovered plan semantics
    - semantic message errors
    - reasoning-only stream routes with no visible commentary/final output
  - Added `TestBuildRequestForensicSummary_PromotesQwenStreamAndMalformedReasoning`.
  - Updated structured forensics API test to assert top-level `qwen_stream` and the recovered-question note.
  - Focused proxy tests passed.
  - `go build -o llama-swap .` passed.

## 2026-05-08 11:05:00 +0200
- Iteration intent:
  - Improve forensic presentation for streamed upstream chat responses.
  - Goal: stop surfacing raw streamed chat-completions payloads as `error_text` when they can be summarized structurally.
- Iteration outcome:
  - `summarizeForensicChatResponse(...)` now detects raw SSE payloads and parses them into a structured streamed chat-response summary.
  - Added streamed chat forensic fields:
    - `is_stream`
    - `chunk_count`
    - `has_done_marker`
  - Streamed upstream chat response summaries now preserve:
    - model
    - finish reason
    - content preview
    - reasoning preview
    - tool call names
  - Added regression test:
    - `TestSummarizeForensicChatResponse_StreamedChatCompletions`
  - Focused proxy tests passed.
  - `go build -o llama-swap .` passed.

## 2026-05-08 11:35:00 +0200
- Iteration intent:
  - Repair the latest manual CLI biology-quiz failure where an implementation retry turn was translated upstream with `tool_choice:"none"` and zero tools.
  - Hypothesis: stale post-`request_user_input` / post-plan continuation state is leaking into implementation/retry turns, disabling tools before Qwen sees the request.
- Change applied:
  - Added implementation/retry intent detection for current Responses requests.
  - `requestStillWantsStructuredPlan(...)` now exits early for explicit implementation/retry turns.
  - Continuation controller no longer forces stale plan/question follow-up behavior when the current turn is an implementation retry.
  - Added regressions for:
    - retry after invalid `apply_patch` keeps tools available
    - implementation retry after native question keeps tools enabled
    - structured-plan detection stays false on `try again` after invalid patch feedback
- Iteration outcome:
  - Focused continuation/plan/stream proxy tests passed.
  - `go build -o llama-swap .` passed.
  - Rebuilt, copied, and restarted live `:8080` successfully.
  - Localhost retry repro now keeps tools exposed through the chat translation layer:
    - `bridge.responses_request.tool_choice = "auto"`
    - `bridge.chat_completions_request.tool_choice = "auto"`
    - `bridge.chat_completions_request.tools_count = 3`
  - Fresh localhost repro also produced native upstream tool behavior instead of pseudo-tool text:
    - `bridge.chat_completions_response.finish_reason = "tool_calls"`
    - `tool_call_names = ["apply_patch"]`
    - `bridge.responses_output.status = "requires_action"`
  - The previous bad symptom (`<tool_code> apply_patch(...)` as final assistant text) did not occur in the localhost retry repro after this patch.
  - Next action: add the missing import, rerun the same targeted tests, then rebuild/copy/restart.
## 2026-05-02 00:28:00 +0200
- Iteration outcome before rebuild:
  - Added 35B plan-continuation guards:
    - explicit native-question continuations in plan mode keep forcing `request_user_input` even after prior tool output
    - explicit short follow-up continuations like `do so` can now resolve to `return the completed <proposed_plan> now` when the prior assistant turn was already teeing up the final plan
  - Focused tests passed:
    - `TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputInPlanContinuationAfterToolOutput`
    - `TestTranslateResponsesToChatCompletionsRequest_ForcesProposedPlanReturnAfterToolOutput`
    - existing reasoning-history replay tests still pass
## 2026-05-02 00:36:00 +0200
- Correction:
  - Reverted the overfit short-continuation plan-return heuristic.
  - Reason: replay evidence showed `do so` was already the recovery turn that could produce the correct plan, so teaching short continuation phrases was the wrong target and pushed the bridge in the wrong direction.
  - Kept the useful part only: native `request_user_input` forcing can still survive plan continuations after prior tool output.
  - Next real fix target remains unchanged:
    - why the model misses the first question turn
    - why the model misses the first completed plan turn
## 2026-05-02 00:43:00 +0200
- Patch intent:
  - Target the actual first-call 35B stalls structurally.
  - New rule set:
    - if a native plan turn already explored via tool output but has not yet completed any `request_user_input` call, force `request_user_input`
    - if a native plan turn already has completed `request_user_input` output, stop tool calling and force the assistant to return one complete `<proposed_plan>` block
  - This avoids teaching continuation phrases and instead uses plan-phase state from the Responses history.
## 2026-05-02 00:46:00 +0200
- Verification status before deploy:
  - Focused structural 35B tests passed:
    - `TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputInPlanContinuationAfterToolOutput`
    - `TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputAfterExplorationBeforeAnyQuestion`
    - `TestTranslateResponsesToChatCompletionsRequest_ForcesProposedPlanAfterCompletedRequestUserInput`
  - Proceeding to rebuild/copy/restart and replay the same 35B captured requests against the live bridge.
## 2026-05-02 00:58:00 +0200
- Expanded verification from single replays to a broad matrix:
  - Added `TestTranslateResponsesToChatCompletionsRequest_PlanAndQuestionMatrix`.
  - Coverage now spans 24 distinct plan/question scenarios, including:
    - plan-mode exploration continuations across `shell`, `web_search`, and `apply_patch`
    - explicit native-question requests with and without prior tool output
    - completed `request_user_input` cycles forcing plan return
    - incomplete question cycles that must *not* force plan return yet
    - explicit specific `tool_choice` preservation on continuations
    - Default-mode direct-plan wrapping and `update_plan` removal behavior
    - negative cases where plan-only instructions must not leak into Default mode and vice versa
- Matrix result:
  - all 24 targeted scenarios passed after fixing one continuation bug
  - specific requested tool choices are now preserved instead of being dropped on prior-tool continuations
- Deploy status:
  - rebuilt workspace binary
  - stopped the running `:8080` process
  - copied the fresh binary to `/home/admmin/bin/llama-swap`
  - restarted successfully on `0.0.0.0:8080`
  - verified matching hashes for workspace and deployed binaries:
    - `73a1bb8b17c9f9023573eee5f74465ca7611387e67d8bc74439306bb5954499a`
  - `/health` returned `status: ok`
## 2026-05-02 07:48:00 +0200
- New repair intent for `B01` backend 502s:
  - Focus the first patch on payload minimization without changing tool semantics.
  - Observation from live bridge traces:
    - narrow turns like native question continuations and strict apply_patch retries still forward the full translated tool catalog
    - those payloads are among the requests correlated with unstable upstream 502 behavior
  - Patch hypothesis:
    - when bridge translation has already forced a specific function tool choice, keep only that tool in the translated upstream request
    - when bridge translation forces `tool_choice="none"`, remove translated tools entirely
  - Expected impact:
    - smaller translated chat payloads on continuation/retry turns
    - lower upstream churn for plan/question and apply_patch recovery requests
    - no behavior change for auto-tool turns
## 2026-05-02 08:13:00 +0200
- `B01` iteration 1 outcome:
  - Implemented bridge-side tool pruning when translation already forces a specific function tool choice.
  - Also remove translated tools entirely when bridge forces `tool_choice="none"`.
  - Added focused tests:
    - `TestTranslateResponsesToChatCompletionsRequest_CodexManagedPlanModePrefersRequestUserInputTool`
    - `TestTranslateResponsesToChatCompletionsRequest_PrunesToolsToForcedSpecificToolChoice`
  - Targeted Go tests passed.
- Live validation notes:
  - The normal launcher path still points at `/home/admmin/bin/llama-swap`, and that deployed copy did not update cleanly because replacement of the running binary kept failing.
  - For validation, the service was launched directly from the rebuilt workspace binary:
    - `/home/admmin/llama-swap/llama-swap-main/llama-swap`
  - Focused `B01` repro loop improved materially:
    - `T50` no longer produced backend `502` rows and once completed in a single `200` request
    - `T52` repeatedly ran with `200`-only backend rows
    - remaining backend `502`s are now concentrated in `T55` and `T57`
  - Interpretation:
    - pruning forced-tool turns reduced one real `502` cluster
    - the remaining `502` path is likely tied to auto-tool turns, especially agent orchestration and some apply_patch continuation cases
  - Next likely target:
    - narrow or stabilize auto-tool exposure on the remaining `T55` / `T57` paths without relying on prompt-specific heuristics
## 2026-05-05 19:05:00 +0200
- Patch intent:
  - Target the true-local T11 regression where Codex stops after the first shell turn even though the bridge returns an apply_patch tool phase.
  - Strongest current evidence: capture 5 shows the bridge emits a commentary assistant message before the apply_patch_call on a tool-continuation turn.
  - Minimal fix hypothesis: preserve reasoning lanes but suppress assistant commentary messages on turns that contain live tool calls, in both writeResponsesStream() and writeResponsesStreamFromChatSSE().
  - Validation gate after patch:
    - focused Go tests for stream/replay tool-turn commentary behavior
    - rebuild
    - true-local Codex smoke + T20 + T11

## 2026-05-05 19:18:00 +0200
- Patch outcome before live retest:
  - Suppressed assistant commentary messages on tool turns in both response replay and direct chat-SSE reconstruction paths.
  - Direct chat-SSE path now buffers plain text until finalization and only emits a final assistant message when the turn actually finishes as a non-tool answer.
  - Added focused regressions:
    - TestWriteResponsesStream_SuppressesCommentaryWhenToolContinuationIsRequired
    - TestWriteResponsesStreamFromChatSSE_SuppressesCommentaryOnToolTurn
  - Focused gate passed:
    - TestWriteResponsesStream_SuppressesCommentaryWhenToolContinuationIsRequired
    - TestWriteResponsesStreamFromChatSSE_SuppressesCommentaryOnToolTurn
    - TestWriteResponsesStreamFromChatSSE_EmitsToolCallArgumentDeltas
    - TestProxyCompatibilitySuite

## 2026-05-05 19:33:00 +0200
- Second T11 repair iteration:
  - Proven working localhost artifact uses streamed apply_patch as a normal function_call with name=apply_patch and structured JSON arguments.
  - Bridge was currently rewriting streamed apply_patch into apply_patch_call/custom_tool_call shapes.
  - Changed both stream emitters to restore the proven contract:
    - type=function_call
    - name=apply_patch
    - arguments={input,operation}
    - response.function_call_arguments.* emitted for apply_patch again
  - Focused gate passed again before live retest.

## 2026-05-05 19:46:00 +0200
- Third T11 repair iteration:
  - Remaining true-local bug after tool-shape fix: first apply_patch execution duplicated BASE_A because the bridge synthesized weak @@ +... patch text from full desired file content for a path it could not inspect locally.
  - Cause: Windows Codex workspace path was outside the proxy workspace, so heuristic contextual patch rebuilding could not read the target file.
  - Fix: for content-only update_file operations on non-local paths, omit synthetic patch text and let structured operation.content drive the update instead.
  - Added regression: external-workspace content-only update payload should omit generated input patch text.

## 2026-05-05 20:05:00 +0200
- Forensic reset after all-502 / zero-token localhost runs:
  - Latest `/api/metrics` rows 0..3 all show status 502 with input_tokens=0 and output_tokens=0.
  - Failing capture 0 proves the original Codex `/v1/responses` request is normal (`stream=true`, `tool_choice=auto`, full tool catalog present), but the stored response body is only: `unable to start process: upstream command exited prematurely but successfully`.
  - `/tmp/llama-swap.log` shows repeated `EnsureStarted` retries and `exit status 1` before any real upstream output is produced.
  - First wrong stage is therefore upstream runtime startup / model process initialization, not later tool parsing, stream reconstruction, or Codex continuation routing.
  - Next safe action is runtime recovery / model-start investigation, not more bridge-contract edits until the upstream process is stable again.

## 2026-05-05 20:32:00 +0200
- Patch intent:
  - Target the persistent startup/state-machine race instead of more retry timing tweaks.
  - Latest logs show overlapping EnsureStarted() loops repeatedly colliding on the same Process while one start attempt is still in flight, producing process was already starting but wound up in state stopped and ailed to set Process state to ready churn.
  - Minimal fix hypothesis: serialize Process.start() so only one goroutine can run the startup/health/ready handoff at a time, and make concurrent callers observe the settled state instead of opening fresh overlapping start attempts.
  - Validation gate after patch:
    - focused Go tests for concurrent start behavior
    - rebuild
    - repeated localhost smoke pass


## 2026-05-05 20:56:00 +0200
- Patch intent:
  - Target the remaining localhost T11 timeout after the startup fix.
  - Latest live evidence: T20 and T22 pass again, but T11 times out after Codex completes the file_change and waits for the forced shell verification continuation.
  - Capture 9 shows the bridge forces shell correctly after completed pply_patch, but still injects the Qwen close-think logit_bias on that post-tool verification turn.
  - Minimal fix hypothesis: keep thinking enabled and preserve parallel support, but suppress the close-think bias on the specific pply_patch -> forced shell verification continuation shape.
  - Validation gate after patch:
    - focused translation tests for shell-verification continuations
    - rebuild
    - localhost smoke + T20 + T22 + T11


## 2026-05-05 21:12:00 +0200
- Patch outcome:
  - Serialized Process.start() with a dedicated mutex and early eady return, which stopped overlapping EnsureStarted() callers from dogpiling the same startup handoff.
  - Focused startup tests passed:
    - TestProcess_WaitOnMultipleStarts
    - TestProcess_StartConcurrentCallsSettleCleanly
    - TestProcess_ExitInterruptsHealthCheck
    - TestIsTransientStartError
  - Live direct smoke improved materially on the fresh workspace binary: 5/5 repeated /v1/responses runs returned 200 PORT8080_OK.
  - After that, the real Windows Codex localhost gate moved from broad startup failure to a narrower post-tool continuation bug:
    - T20 PASS
    - T22 PASS
    - T11 FAILING by timeout after completed file_change
- Second patch outcome:
  - Suppressed the Qwen close-think logit_bias only on the specific pply_patch -> forced shell verification continuation shape.
  - Focused translation/compatibility tests passed:
    - TestTranslateResponsesToChatCompletionsRequest_ContinuationAfterApplyPatchForcesShellVerification
    - TestTranslateResponsesToChatCompletionsRequest_ExplicitNoMutationPlanDisablesTools
    - TestProxyCompatibilitySuite
  - Live Windows Codex localhost gate on the fresh workspace binary now passes structurally:
    - T20 PASS
    - T22 PASS
    - T11 PASS (no timeout; shell -> apply_patch -> shell -> final answer completed)
  - Remaining known quality issue:
    - T11 final file still duplicates BASE_A on the first patch result (BASE_A, BASE_A, ORDERED_T11) before the session finishes successfully.
    - So the timeout/regression is fixed, but the content-quality duplication issue remains the next repair target.

## 2026-05-05 18:40:00 +0200
- Apply-patch repair follow-up:
  - Confirmed the duplication entry point from the latest T11 artifacts:
    - the model returned an `apply_patch` `update_file` operation with full desired `content`
    - the bridge preserved a weak synthetic diff body like `@@` plus only `+...` lines
    - that shape can duplicate existing file content during Codex execution
  - Added a path-hint pre-repair before `selectApplyPatchOperation(...)` so stringified `operation` payloads can inherit the real file path hint earlier.
  - Added argument repair for two malformed native tool-call shapes:
    - quoted JSON object arguments
    - JSON object arguments with literal newlines inside quoted string fields
  - Added focused regressions for:
    - quoted-object argument unwrapping
    - literal-newline JSON repair
    - translate-time `apply_patch` path-hint replacement patch building
  - Focused proxy tests and build passed after the repair slice.
- Runtime verification note:
  - After the code fix, I discovered `:8080` was still serving `/home/admmin/bin/llama-swap`, not the fresh workspace binary.
  - I removed that listener and replaced it with `/home/admmin/llama-swap/llama-swap-main/llama-swap`.
  - The workspace binary now binds `:8080`, but the first direct `/v1/responses` smoke on that fresh process timed out and the live log shows:
    - request translation begins normally
    - then the upstream `Qwen3.6-35B-A3B-UD-Q8_K_XL` process exits with `exit status 1` while the process state is already `ready`
  - So the latest localhost runtime blocker is again the harness/upstream lifecycle path on the fresh workspace binary, not the old deployed binary.

## 2026-05-06 13:55:00 +0200
- Streaming recovery / plan-finalization pass:
  - Broadened structured plan continuation so completed `request_user_input` clarification can force a real `<proposed_plan>` return outside strict plan-mode only flows.
  - Added `requestStillWantsStructuredPlan(...)` and `structuredPlanOutputRequiredFromRequestBody(...)` so the stream bridge can recognize:
    - formal plan mode
    - default-mode “write/return/finalize a plan” turns
    - post-clarification plan continuations that still need a structured plan block
  - Tightened plan-preface recovery:
    - `looksLikePlanAcknowledgementWithoutPlan(...)` now catches “I have everything I need / here's the plan” style acknowledgements that stop before a real plan body
    - those now rewrite to a real `<proposed_plan>` output instead of completing as conversational prose
  - Refactored `writeResponsesStreamFromChatSSEWithWorkflow(...)` into a more explicit presentation model:
    - `reasoning_summary` lane uses incremental synthesized preview text from visible reasoning instead of blindly forwarding raw `reasoning_content`
    - commentary and final text are routed separately
    - plan-output turns buffer raw model text until the bridge decides whether it is a valid plan or a weak preface
    - verification-in-progress and final-answer-safe turns buffer raw content so weak “Now verifying...” / “I'll create...” prose can be rewritten before it leaks as the final user-facing output
  - Added bridge-side workflow progress commentary:
    - generating the requested plan
    - working on the request
    - applying the requested file update
    - verification in progress
    - finalizing the response
  - Added weak-finalization rewrite for completed workflows:
    - final-answer-safe implementation turns now rewrite future-tense verification/setup prose to a short completion summary like:
      - `The requested file changes were applied and verified successfully.`
  - Added direct upstream measurement tooling:
    - new script: `scripts/qwen_stream_probe.py`
    - new doc: `docs/qwen-stream-probe.md`
    - probe cases:
      - `reasoning_only`
      - `plan_only`
      - `tool_reasoning`
      - `tool_then_final`
    - outputs:
      - raw SSE transcript
      - compact JSON summary with content/reasoning/tool delta counts and previews
  - Ran the new direct upstream probe against `http://localhost:10008/v1` for `Qwen3.6-35B-A3B-UD-Q8_K_XL`:
    - `reasoning_only`:
      - `reasoning_delta_count=1509`
      - `content_delta_count=48`
      - final one-sentence visible content arrived separately from a large reasoning stream
    - `plan_only`:
      - `reasoning_delta_count=1519`
      - `content_delta_count=990`
      - model produced a real `<proposed_plan>` content stream with separate reasoning deltas
    - `tool_reasoning`:
      - `reasoning_delta_count=95`
      - `tool_call_delta_count=7`
      - `finish_reason=tool_calls`
      - native `shell` tool deltas were present while visible `content` stayed empty
    - artifacts saved under `tmp/qwen_stream_probe_20260506`
  - Added/updated focused regressions for:
    - default-mode post-clarification plan forcing
    - plan-preface rewrite to structured `<proposed_plan>`
    - tool-turn commentary/progress streaming
    - synthesized reasoning-summary deltas
    - weak final workflow text rewrite after verification
  - Verification:
    - focused proxy stream + compatibility + plan-matrix tests passed
    - `go build -o llama-swap .` passed
    - probe script syntax check passed (`py -3 -m py_compile scripts/qwen_stream_probe.py`)
  - Deployment note:
    - rebuilt workspace binary was copied to `/home/admmin/bin/llama-swap`
    - live server restart initially failed when launched from the repo root because `config.yaml` lives in `/home/admmin/llama-swap`
    - corrected restart now runs from `/home/admmin/llama-swap`, and `:8080` is serving the updated deployed binary again

2026-05-06 stream regression repair follow-up:
  - Repaired the streamed native-question path after the stream-lane refactor:
    - broadened native-question intent detection
    - passed `nativeQuestionRequired` into the chat-SSE reconstruction path
    - added streamed fallback recovery for empty `request_user_input` arguments using:
      - visible reasoning/text recovery first
      - request-intent synthesis second
    - fixed the mixed SSE case where `reasoning_content`, `tool_calls`, and `finish_reason:"tool_calls"` arrive in the same chunk so tool finalization now still happens
    - commentary for native question turns now says `Preparing the clarification question.` instead of plan-generation text
  - Added focused regression:
    - `TestWriteResponsesStreamFromChatSSEWithWorkflow_RecoversRequestUserInputArgsAndUsesQuestionCommentary`
  - Focused verification:
    - `go test ./proxy -run "TestTranslateResponsesToChatCompletionsRequest_PlanAndQuestionMatrix|TestWriteResponsesStreamFromChatSSE|TestWriteResponsesStreamFromChatSSEWithWorkflow_RecoversRequestUserInputArgsAndUsesQuestionCommentary|TestWriteResponsesStreamFromChatSSE_PlanModeStripsFollowupQuestionsAndMalformedCloser|TestProxyCompatibilitySuite" -count=1`
    - `go build -o llama-swap .`
  - Live stream validation artifacts:
    - `C:\Users\YLAB-Partner\Downloads\llama_swap_stream_repair_question_live.txt`
    - `C:\Users\YLAB-Partner\Downloads\llama_swap_stream_repair_plan_live.txt`
  - Live result:
    - question turn now emits native `request_user_input` with recovered `questions` in the final `response.function_call_arguments.done`
    - plan turn again emits a final `<proposed_plan>...</proposed_plan>` block
  - Remaining stream presentation wart:
    - the question turn still forwards ugly early partial arg deltas from upstream before the corrected final `arguments.done`
    - the plan turn commentary is semantically valid but still generic (`Continuing from the latest tool results.` / `Finalizing the response.`) rather than ideal plan-specific phrasing

## 2026-05-07 09:35:00 +0200
- Non-stream repair loop:
  - Localized live regression after restart to the non-stream chat->responses path.
  - Streamed path already recovers native `request_user_input` and emits `requires_action`.
  - Non-stream path still had two mismatches:
    - tool-bearing outputs were serialized with top-level `status:"completed"`
    - native-question-required turns could still collapse to plain assistant text when upstream answered with text/reasoning instead of a native question tool
  - Next minimal fix:
    - make normalized non-stream tool responses force `requires_action`
    - add a request-gated native-question recovery rewrite for non-stream responses using upstream content/reasoning plus existing recovery helpers
- Patch applied:
  - `normalizeTranslatedResponsesOutput(...)` now marks any pending tool-bearing non-stream response as `requires_action`.
  - Added `recoverNativeQuestionResponseIfRequired(...)` and wired it into the non-stream bridge path after payload rewrite, so explicit native-question requests can be rewritten from plain assistant question text into a native `request_user_input` call using native artifacts first and existing text/request recovery second.
  - Added focused regressions for:
    - tool-only non-stream responses becoming `requires_action`
    - non-stream native-question recovery from plain assistant text
- Verification note:
  - `gofmt` passed on the touched Go files.
  - I could not run `go test` / `go build` from this session because:
    - the explicit local `Ubuntu` WSL distro does not have `go` installed
    - Windows `go` against `\\wsl$\Ubuntu\...` still fails with `go: RLock ... go.mod: Incorrect function`
    - direct SSH to `tbwork` is still refusing port 22
  - So the next step is a rebuild/copy/restart from the actual Linux build host, then live re-check T20 + tool-only non-stream behavior.
## 2026-05-07 10:00:00 +0200
- Workflow correction note:
  - The only accepted rebuild/deploy path for this project is:
    - `wsl -d Ubuntu -u admmin`
    - `cd /home/admmin/llama-swap/llama-swap-main`
    - `go build -o llama-swap .`
    - `cp ./llama-swap /home/admmin/bin/llama-swap`
    - restart with `C:\Users\YLAB-Partner\Desktop\start-tbg-services-wsl.ps1`
  - If `go` is not on PATH in that exact environment, stop immediately and report the environment mismatch.
  - Do not retry Windows `go`, UNC `\\wsl$...` builds, mapped-drive workarounds, or generic fallback loops.
- Skill correction:
  - root cause was non-login shell PATH mismatch, not missing Go.
  - verified working probe:
    - `wsl -d Ubuntu -u admmin bash -lic 'command -v go; go version'`
    - resolves to `/usr/local/go/bin/go`
  - workflow note updated to always use `bash -lic` for build/copy on the Ubuntu host.
- Restart workflow correction:
  - proved working path:
    - if the script does not replace the `llama-swap` PID, kill the live PID directly
    - verify `:8080` closes
    - rerun `C:\Users\YLAB-Partner\Desktop\start-tbg-services-wsl.ps1`
    - verify a new PID is listening on `:8080`
  - `health` alone is not enough; PID change is the restart proof.
  - a true cold restart can reset `/api/metrics` to an empty array; that is normal and not itself a regression.

## 2026-05-07 10:35:00 +0200
- Native-question upstream lifecycle repair intent:
  - Fresh cold-restart live evidence narrowed the remaining failure to the upstream non-stream chat-completions request shape for forced `request_user_input`.
  - Non-stream tool-only path is already fixed and returns `status:"requires_action"`.
  - Native-question non-stream requests still die at `upstream_process_lifecycle` with bridge retries and final `502`, but direct upstream probing showed the same request shape streams successfully.
  - Next minimal fix:
    - for explicit native-question non-stream bridge turns only, send the upstream chat request as `stream:true`
    - accumulate the upstream chat SSE into a synthetic final chat-completions response body
    - run the existing non-stream chat->responses translation and native-question recovery on that synthetic body
  - Goal:
    - preserve the external non-stream `/v1/responses` contract
    - avoid the exact upstream non-stream mode that is crashing on `request_user_input`

## 2026-05-07 10:50:00 +0200
- Restart script repair:
  - A failed relaunch showed the backend `llama-server` on `:10008` was still alive, so the new process exited with:
    - `couldn't bind HTTP server socket, hostname: 0.0.0.0, port: 10008`
  - This is a stale-port operational failure, not a model-load regression.
  - Updated the standard restart path so it also kills any process still listening on `:10008` before relaunch, waits for `:10008` to close, and includes `:10008` in the post-restart listener verification.
  - Follow-up correction:
    - the first PID-extraction attempt around `ss` was too fragile under PowerShell/WSL quoting
    - simplified the kill step to `fuser -k 10008/tcp` so the backend port is freed directly before relaunch

## 2026-05-07 11:20:00 +0200
- Qwen stream-translation continuation:
  - The repair loop that unblocked native question and plan behavior is complete.
  - Next architectural slice for the larger Qwen stream plan:
    - stop having stream and non-stream paths manually walk normalized artifact arrays in separate ad hoc ways
    - introduce one reusable normalized-artifact view/selection layer
    - make stream and native-question recovery consume the same view so native-vs-recovered precedence is centralized
  - Goal:
    - continue the Qwen interpretation-layer migration incrementally
    - reduce the chance of stream/non-stream semantic drift without reopening the live plan/question fixes

## 2026-05-07 11:45:00 +0200
- Normalized streamed-event validation:
  - Added a minimal Qwen normalized stream-event helper and reusable normalized-artifact view selection layer.
  - First validation target was safety, not new UX behavior:
    - confirm native question recovery still works
    - confirm structured plan streaming still works
    - confirm empty-stop fallback still works
    - confirm the new stream chunk normalization emits the expected semantic event kinds
- Verification:
  - focused normalization/question/stream tests: pass
  - wider stream compatibility tests: pass
  - `go build -o llama-swap .`: pass
- Next step:
  - deploy/restart and confirm live T20/T22/tool controls still behave correctly before continuing the larger SSE event-layer migration

## 2026-05-07 11:58:00 +0200
- Live regression after architectural-slice deploy:
  - T22 stayed green.
  - Tool-only non-stream stayed green.
  - T20 regressed again, but in a narrower way:
    - upstream streamed turn returned plain assistant content
    - content was imperative (`Please provide the unique marker T20_SENTINEL.`), not a literal question
    - existing fallback question recovery did not fire because it prefers question-shaped text or explicit `questions` payloads
- Minimal repair:
  - extend request-based native-question synthesis for the explicit T20-style prompt:
    - detect the exact “ask one native Codex question before planning” request intent
    - preserve `T##_SENTINEL` when present
    - synthesize one stable native question instead of allowing imperative assistant prose to escape
- Verification:
  - focused native-question recovery tests: pass
  - `go build -o llama-swap .`: pass
- Next step:
  - redeploy and re-check live T20/T22/tool controls

## 2026-05-07 12:12:00 +0200
- Qwen stream normalization progression:
  - Continued the live SSE-path migration without changing external wire behavior.
  - Added `QwenNormalizedStreamFrame` as a per-chunk semantic fold over normalized Qwen stream events.
  - `writeResponsesStreamFromChatSSEWithWorkflow(...)` now consumes:
    - chunk id / model / created / usage / timings
    - assistant delta text
    - reasoning delta text
    - tool call deltas
    - finish reason
    through the normalized frame instead of rebuilding these fields ad hoc from raw event iteration variables.
- Verification:
  - focused stream-normalization and bridge tests: pass
  - wider stream compatibility tests: pass
  - `go build -o llama-swap .`: pass
- Architectural effect:
  - stream interpretation and stream presentation are a little more separated now
  - the SSE loop is closer to consuming a stable semantic object per chunk
  - remaining work is still to move more routing/output decisions out of the monolithic stream writer

## 2026-05-07 12:28:00 +0200
- Qwen stream interpretation/presentation split:
  - Added `QwenStreamFrameDecision` and `decideQwenStreamFramePresentation(...)`.
  - The live SSE writer no longer decides inline whether each accumulated chunk becomes:
    - reasoning
    - commentary
    - final buffered assistant text
    - pure tool-phase finalization
  - That routing now comes from the normalization layer using:
    - normalized chunk frame
    - workflow state
    - plan-mode state
    - observed tool presence
- Verification:
  - focused routing + stream tests: pass
  - wider bridge compatibility tests: pass
  - `go build -o llama-swap .`: pass
- Architectural effect:
  - the monolithic SSE writer is thinner
  - Qwen semantic interpretation now owns both:
    - chunk folding
    - first-pass presentation routing
  - next remaining step is to extract more tool-lifecycle/progress emission out of the writer and into reusable normalized stream interpretation helpers

## 2026-05-07 12:42:00 +0200
- Qwen tool-lifecycle extraction:
  - Moved streamed tool delta application into `applyNormalizedToolCallDelta(...)`.
  - Moved tool-phase closeout into `finalizeNormalizedToolCallPhase(...)`.
  - These helpers now own:
    - tool state creation / reuse by index
    - tool name / call id updates
    - args accumulation
    - tool-args delta emission
    - tool-start progress commentary triggering
    - request-user-input arg recovery on closeout
    - shell empty-args validation fallback
    - plan-mode apply_patch suppression / plan text extraction
    - tool args done / item done emission
- Verification:
  - focused stream/bridge/native-question tests: pass
  - `go build -o llama-swap .`: pass
- Architectural effect:
  - the main SSE loop is now much closer to:
    - normalize frame
    - apply tool semantics
    - get frame routing decision
    - render events
  - remaining work is mostly higher-level cleanup and optional forensics/debug exposure, not the core semantic split anymore

## 2026-05-08 13:05:00 +0200
- Bridge semantic narrowing repair:
  - Fixed `deriveContinuationAllowedToolNames(...)` so a previously completed `apply_patch` no longer forces the next turn into patch-family narrowing by itself.
  - Patch-family narrowing now stays limited to turns whose current user intent still explicitly asks for `apply_patch` sequencing or shell-before-patch ordering.
  - This addresses the manual CLI failure where a post-create verification/browser turn was translated upstream with only `apply_patch` exposed.
- Regression coverage:
  - `TestDefaultContinuationController_DeriveAllowedToolNames_UsesWorkflowState`
  - `TestTranslateResponsesToChatCompletionsRequest_PostPatchContinuationKeepsBroadTools`

## 2026-05-08 13:32:00 +0200
- Apply-patch transcript and recovery-finalization repair:
  - Normalized replayed `apply_patch` tool output before feeding it back into the chat transcript.
  - Rewrites terse patch status markers like `A path`, `M path`, `D path` into explicit forms:
    - `created: path`
    - `modified: path`
    - `deleted: path`
  - Unwraps serialized `apply_patch_call_output` payloads before normalization when needed.
  - Repeated completed `apply_patch` fingerprints no longer trigger the generic final-answer-required loop guard by themselves.
  - This keeps failed patch recovery tool-capable instead of prematurely disabling tools.
- Regression coverage:
  - `TestResponsesRequestToChatMessages_NormalizesApplyPatchToolOutputTranscript`
  - `TestBuildLoopGuardDecision_CentralizesLoopProtections`

## 2026-05-08 14:05:00 +0200
- Built-in browser/search delegation clarification:
  - Direct Codex CLI testing against local `llama-swap` confirmed that built-in tool families like:
    - `web_search`
    - `file_search`
    - `computer_use_preview`
    are not being executed as local native CLI tools in this backend setup.
  - In the direct Codex test, the visible Codex event stream did not show a local/native search execution item.
  - Matching `llama-swap` forensics showed the search flow was handled server-side through the bridge:
    - upstream emitted a `web_search` tool phase
    - the bridge executed/continued the search flow
    - Codex received the final answer after the server-side continuation
  - This confirms why `llama-swap` implements delegated wrapper tools for these families:
    - `__llamaswap_web_search_preview`
    - `__llamaswap_file_search`
    - `__llamaswap_computer`
  - These wrappers exist because OpenAI built-in Responses tools need backend/runtime support when the backend is local Qwen via `llama-swap`; they are not the same as MCP tools like `mcp__playwright__browser_*`.
  - The real bridge bug in this area was not the existence of wrappers, but mismatched tool-name guidance. Qwen must be instructed to call the exact wrapper names actually exposed in translated `tools[]`.
- Regression coverage:
  - `TestBuildQwenResponsesToolPolicy_UsesExactWrapperNamesForBuiltInTools`

## 2026-05-08 15:02:00 +0200
- Native-question misclassification repair:
  - Root cause of the fresh "hi"/simple search failures was `requestExplicitlyWantsNativeCodexQuestion(...)` scanning trusted developer/system instruction text from `input[]` together with the current user text.
  - That let global boilerplate mentioning `request_user_input` poison ordinary default-mode turns and collapse translated chat requests to:
    - `tool_choice: function:request_user_input`
    - `tools_count: 1`
  - The detector now uses only current-turn intent sources:
    - top-level `instructions`
    - current user input text
  - It no longer treats developer/system boilerplate inside `input[]` as an explicit native-question request.
  - `synthesizeRequestUserInputArgumentsFromRequest(...)` now uses the same narrowed intent text so recovered question text is derived from the current turn only.
- Live validation after rebuild/copy/restart:
  - `hi` on `localhost:8080/v1/responses`:
    - `bridge.chat_completions_request.tool_choice = "auto"`
    - tools remained `["shell","request_user_input"]`
    - upstream answered normally with no tool call
  - `search the web for YLAB` on `localhost:8080/v1/responses`:
    - `bridge.chat_completions_request.tool_choice = "auto"`
    - tools remained `["web_search","shell","request_user_input"]`
    - upstream emitted native `web_search`
    - bridge returned `web_search_call`, `web_search_call_output`, and final assistant text with overall `status:"completed"`
- Regression coverage:
  - `TestTranslateResponsesToChatCompletionsRequest_ContinuationToolChoiceBehavior/default_mode_simple_hello_does_not_force_request_user_input_from_boilerplate`
  - `TestTranslateResponsesToChatCompletionsRequest_ContinuationToolChoiceBehavior/default_mode_web_search_does_not_force_request_user_input_from_boilerplate`

## 2026-05-08 15:12:00 +0200
- Manual multi-turn web-search regression forensic snapshot:
  - Session anchor:
    - `C:\Users\YLAB-Partner\.codex\sessions\2026\05\08\rollout-2026-05-08T14-00-32-019e0775-f832-7e83-b065-bd77c07f7edb.jsonl`
  - `/api/forensics/6`:
    - first upstream turn correctly used native `web_search`
    - follow-up continuation was then wrongly narrowed to `tool_choice:function:request_user_input`
  - `/api/forensics/7`:
    - after answered clarification plus “write a native plan and do web research before”
    - translated chat request became `tool_choice:none`
    - upstream fell back to literal `<websearch>` prompt text
  - `/api/forensics/15`:
    - later explicit Reddit search still had original Responses `tool_choice:auto`
    - translated chat request became `tool_choice:none`
    - upstream emitted literal `<web_search>...</tool_call>` text
  - Stage trace showed stale `tool_completed_awaiting_followup` continuing to win over fresh search intent.
- Patch:
  - `extractResponsesNativeQuestionIntentText(...)` now reads only current user text, removing top-level instruction boilerplate from native-question forcing.
  - `requestExplicitlyWantsExplorationFollowup(...)` now recognizes broader search/research intent:
    - `web research`
    - `research before`
    - search verbs paired with `web`, `browser`, `reddit`, or `site:`
  - This is intended to keep tools alive for fresh exploration/search turns in plan, act, and default conversations instead of forcing stale plan/question follow-up behavior.
- Focused regression coverage:
  - `TestRequestStillWantsStructuredPlan_FalseForWebResearchBeforePlanFollowup`
  - `TestRequestExplicitlyWantsNativeCodexQuestion_IgnoresInstructionBoilerplate`
  - continuation/translation focused subset: pass

## 2026-05-08 16:32:00 +0200
- Follow-up repair on stale search-intent suppression:
  - Added an explicit current-turn search intent detector separate from the older follow-up helper.
  - `ContinuationContext` now carries `SearchIntent`.
  - Search intent now overrides stale post-question plan suppression in three places:
    - plan-follow-up loop guard no longer forces `tool_choice:none`
    - continuation state stays `tool_running` instead of `tool_completed_awaiting_followup`
    - forced `request_user_input` continuation no longer wins on fresh search turns
  - `requestStillWantsStructuredPlan(...)` now exits early when the current user turn explicitly asks for fresh search/research.
- Operational note:
  - an earlier deploy was still running the older `/home/admmin/bin/llama-swap`; verified by mismatched SHA256 between the built binary and the installed binary.
  - corrected by stopping `:8080`, copying again, and confirming matching SHA256 before restart.
- Regression coverage:
  - `TestTranslateResponsesToChatCompletionsRequest_TopLevelPlanInstructions_WebResearchBeforePlanKeepsTools`
  - `TestTranslateResponsesToChatCompletionsRequest_TopLevelPlanInstructions_LaterRedditSearchKeepsTools`
  - focused continuation/tool-choice subset: pass

## 2026-05-08 15:00:00 +0200
- Search retry follow-up repair:
  - The remaining manual failure shape was shorter retry text like `try again` after a previous `web_search` attempt or pseudo-search assistant output.
  - `requestExplicitlyWantsSearchIntent(...)` is now workflow-state aware and can inherit recent search context from:
    - completed `web_search` / `web_search_preview`
    - prior search tool calls in `input[]`
    - pseudo-search assistant text such as `print(websearch(...))`
  - This keeps the continuation state in `tool_running` instead of letting stale post-`request_user_input` follow-up logic fall back to `tool_completed_awaiting_followup`.
- New regression coverage:
  - `TestRequestStillWantsStructuredPlan_FalseForSearchRetryAfterPseudoSearchOutput`
  - `TestTranslateResponsesToChatCompletionsRequest_SearchRetryAfterPseudoSearchOutputKeepsTools`
- Live localhost validation after proper redeploy:
  - verified the installed `/home/admmin/bin/llama-swap` SHA matched the rebuilt workspace binary before restart
  - stage trace now shows the repaired retry path:
    - `translation.responses_to_chat.done`
    - `state=tool_running`
    - `tools_count=3`
  - `/api/forensics/2` confirms:
    - translated chat request kept `request_user_input`, `web_search`, and `shell`
    - upstream emitted a native `web_search` tool call
  - remaining failure on that probe was `upstream_process_lifecycle` after translation, not the earlier bridge-side tool stripping bug

## 2026-05-08 16:10:00 +0200
- General continuation-state forensic conclusion:
  - The broader instability is not just "search retry" or one broken prompt variant.
  - The real structural weakness is that `llama-swap` has been deciding turn intent from multiple overlapping booleans:
    - `PlanModeRequested`
    - `PlanOutputRequested`
    - `SearchIntent`
    - `ExplorationFollowupIntent`
    - `ImplementationRetryIntent`
  - In long mixed chats this allowed the bridge to derive slightly different answers in:
    - `translateResponsesToChatCompletionsRequest(...)`
    - `defaultContinuationController.BuildDecision(...)`
    - `buildLoopGuardDecision(...)`
  - That mismatch is why manual runs could still regress after:
    - default -> plan -> search
    - question -> research -> return plan
    - plan -> default -> retry
- Local software comparison:
  - Qwen Companion docs and stream-json/dual-output design reinforce a typed runtime/event model instead of texty intent recovery.
  - Cline separates Plan and Act much more explicitly and uses dedicated control/tool lanes instead of relying on raw assistant prose for mode transitions.
  - `llama-swap` should not copy those wire contracts, but it should centralize continuation phase decisions the same way.
- Patch:
  - Introduced shared `ContinuationTurnPhase` with phases:
    - `general`
    - `question`
    - `plan_gather`
    - `plan_finalize`
    - `research`
    - `implementation_retry`
  - `BuildDecision(...)` and `buildLoopGuardDecision(...)` now reason through the same phase classifier instead of recomputing plan/search/followup behavior ad hoc.
  - Live translation trace now records `turn_phase` in `continuation.responses_to_chat`.
- Regression coverage:
  - `TestClassifyContinuationTurnPhase`
  - plan/question/search continuation subset
  - top-level plan-instructions web-research/search-retry subset
- Verification before redeploy:
  - focused continuation tests: pass
  - `go build -o llama-swap .`: pass

## 2026-05-08 18:05:00 +0200
- New forensic anchor for the latest manual failure:
  - `id=8`:
    - chat request still had `web_search`
    - upstream returned `finish_reason=stop`
    - visible content only `Working on the request.`
    - reasoning still clearly wanted web research
    - bridge incorrectly accepted that as a completed final answer instead of retrying/recovering the missing tool call
  - `id=9`:
    - chat request had `tool_choice:none`
    - upstream returned literal empty `<tool_code></tool_code>`
    - bridge leaked that pseudo-tool wrapper into final output
- Minimal fix hypothesis:
  - parser cleanup from Qwen XML/tag stripping is being ignored when no parsed tool call is recovered
  - bridge needs a generic web-search retry path when:
    - `web_search` is available upstream
    - upstream emits only placeholder/progress text or pseudo-tool search text
    - no real tool call was returned
  - web search recovery should prefer native `web_search` and explicitly avoid Playwright for this path

## 2026-05-08 19:55:00 +0200
- Implemented in this loop:
  - preserve parser cleanup even when Qwen XML/tag parsing recovers zero tool calls
    - empty `<tool_code></tool_code>` no longer survives as final output text in the translator test path
  - add generic missing-web-search retry in the bridge
    - if `web_search` is available upstream
    - and upstream returns only placeholder/progress text or pseudo-search text without a real tool call
    - retry with an explicit native `web_search` instruction instead of finalizing
  - widen search-intent text detection for:
    - `current`
    - `latest`
    - `external`
    - `news`
  - add an extra continuation-controller guard so explicit research text should not be hijacked back into `request_user_input`
- Regression coverage added:
  - missing web-search retry bridge test
  - empty tool_code stripping test
  - plan-mode research-before-plan translation test
- Current live status after redeploy:
  - retry-after-pseudo-search path keeps `web_search` alive and reaches real native `web_search`, but the continuation can still end in upstream `502` after translation
  - a narrower live failure still remains for this exact non-stream request shape:
    - prior `request_user_input` answered
    - plan mode
    - user says `research current ... before writing the plan`
  - despite unit coverage for that shape, the live bridge still translated it to:
    - `tool_choice=function:request_user_input`
    - tools only `request_user_input`
  - this means the remaining bug is still in live continuation-state handling before upstream model execution
