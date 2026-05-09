# Plan Mode Native Question Flow Note

## Date
2026-04-30

## Final Status
- Native Codex plan flow is working again.
- Native structured question flow is working again.
- The bridge now returns a real `request_user_input` function call with populated `questions` arguments on the repaired plan-question path.
- User-confirmed outcome: plan and question flow are working now.

## Final Behavior
In Codex-managed plan turns, the bridge now:
- preserves `request_user_input`
- preserves `update_plan`
- strips `apply_patch`
- preserves non-mutating exploration tools such as `shell`
- forces `tool_choice` to `request_user_input` when the prompt explicitly asks for one native Codex question before planning

In proxy-enforced raw plan turns, the bridge remains stricter:
- strips `apply_patch`
- strips `request_user_input`
- strips `update_plan`
- requires direct-text planning behavior

## What Was Actually Broken
There were two distinct issues in the repair sequence:

1. Native question exposure regressions
- Codex-managed plan turns temporarily lost the native plan/question surface or over-hardened into the wrong tool contract.

2. Malformed upstream native tool arguments
- the local Qwen path could emit a real `request_user_input` tool call with `arguments:"{}"`
- the reasoning text still contained a structurally recoverable question payload

The second issue was the last real blocker for native UI question flow.

## Final Bridge Fixes
1. Codex-managed plan tool preservation
- keep `request_user_input`
- keep `update_plan`
- strip `apply_patch`

2. Native question steering
- when a prompt explicitly asks for one native Codex question first, translated `tool_choice` is forced to:

```json
{"type":"function","function":{"name":"request_user_input"}}
```

3. Empty-argument recovery for `request_user_input`
- when the upstream model returns a native `request_user_input` call with empty arguments, the bridge now recovers arguments only from explicit reasoning-side structures:
  - JSON object containing `"questions": [...]`
  - `questions: ["..."]`
  - anchored `Question: "..."` lines

This recovery is narrow and contract-safe:
- it only applies when the native function call already exists
- it does not invent tool calls from arbitrary prose

## Tests Passed
Focused bridge tests passed:

```powershell
go test ./proxy -run "TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromReasoning|TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromQuestionLine|TestTranslateResponsesToChatCompletionsRequest_CodexManagedPlanModePrefersRequestUserInputTool" -count=1
```

Relevant regressions now cover:
- empty native args recovered from `Args: {"questions": [...]}` in reasoning
- empty native args recovered from `Question: "..."` line in reasoning
- Codex-managed plan mode preferring the native `request_user_input` tool

## Live Verified Result
Validated binary hash after rebuild/redeploy:

```text
b7ee656635b9de885da5a79bf79ac01613f217d30865c5a1c4b30fb136229993
```

Fresh live proof:
- `/tmp/bridge_plan_probe5_resp.json`

The repaired response contains:

```json
{
  "type": "function_call",
  "name": "request_user_input",
  "arguments": "{\"questions\":[\"What task or project do you need a plan for?\"]}"
}
```

This is the bridge-side condition needed for Codex to surface the native structured question interaction.

## Related Evidence
- `/tmp/bridge_plan_probe5_resp.json`
- `/tmp/llama-swap.log`
- `/tmp/llama-swap-apply-patch-trace.log`
- `apply_patch_repare.md`

## Conclusion
The native plan/question regression is closed on the bridge side and validated live.
