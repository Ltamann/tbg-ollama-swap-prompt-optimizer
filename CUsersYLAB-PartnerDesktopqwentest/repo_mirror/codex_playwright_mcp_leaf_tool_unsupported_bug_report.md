# Codex WSL `codex exec` Playwright MCP Leaf Tool Unsupported

## Summary

On WSL `codex exec 0.123.0` with local `llama-swap`, Playwright browser leaf tools can now be emitted and persisted correctly as namespaced `function_call` items, but the Codex router still rejects them as unsupported:

```text
unsupported call: mcp__playwright__browser_navigate
```

This is now the first wrong stage for the remaining Phase 9 browser-tool failure.

## Environment

- Surface: WSL `codex exec`
- Codex CLI version: `0.123.0`
- Model alias: `gpt-5.2`
- Local backend: `llama-swap` on `http://localhost:8080/v1`
- MCP-enabled Codex home:
  - [config.toml](\\wsl$\Ubuntu\home\admmin\llama-swap\.codex-phase8-mcp\config.toml)
- Playwright MCP server registered in that home

## Symptom

Browser tasks do not complete even though the model emits the correct Playwright leaf tool call.

Navigation repro final artifact:
- [p9_nav_fix2_last.txt](\\wsl$\Ubuntu\home\admmin\llama-swap\tmp\p9_nav_fix2_last.txt)

Snapshot repro final artifact:
- [p9_snapshot_fix2_last.txt](\\wsl$\Ubuntu\home\admmin\llama-swap\tmp\p9_snapshot_fix2_last.txt)

Both runs explain that the Playwright browser tool is not available after the router returns `unsupported call`.

## Proven progress before this bug

Earlier in the same investigation, raw browser `mcp_tool_call` items were visible in `/v1/responses` captures but disappeared before session rollout persistence.

After the latest bridge compatibility patch, the browser tool call now survives into the rollout as a real `function_call`:

```json
{"type":"function_call","name":"mcp__playwright__browser_navigate","arguments":"{\"url\":\"https://example.com\"}"}
```

So this is no longer a pure `llama-swap` parsing/stream-loss issue.

## Strongest proof

### Navigation rollout

- [rollout-2026-04-29T14-43-34-019dd944-247e-78f0-be57-3622c5cff14a.jsonl](\\wsl$\Ubuntu\home\admmin\llama-swap\.codex-phase8-mcp\sessions\2026\04\29\rollout-2026-04-29T14-43-34-019dd944-247e-78f0-be57-3622c5cff14a.jsonl)

Key lines:
- `response_item` `function_call` name `mcp__playwright__browser_navigate`
- immediate `response_item` `function_call_output`
- output text:
  - `unsupported call: mcp__playwright__browser_navigate`

### Snapshot rollout

- [rollout-2026-04-29T14-43-34-019dd944-247e-7bc0-af03-35d33593066d.jsonl](\\wsl$\Ubuntu\home\admmin\llama-swap\.codex-phase8-mcp\sessions\2026\04\29\rollout-2026-04-29T14-43-34-019dd944-247e-7bc0-af03-35d33593066d.jsonl)

Key lines:
- two repeated `function_call` attempts:
  - `mcp__playwright__browser_navigate`
- two matching `function_call_output` errors:
  - `unsupported call: mcp__playwright__browser_navigate`
- later fallback `function_call`:
  - `list_mcp_resources`
- which fails separately because Playwright does not implement `resources/list`

### CLI-facing event logs

- [p9_nav_fix2_events.jsonl](\\wsl$\Ubuntu\home\admmin\llama-swap\tmp\p9_nav_fix2_events.jsonl)
- [p9_snapshot_fix2_events.jsonl](\\wsl$\Ubuntu\home\admmin\llama-swap\tmp\p9_snapshot_fix2_events.jsonl)

These logs now show the browser-phase failure explicitly instead of silently collapsing after reasoning.

## Why earlier stages are exonerated

1. MCP server registration exists in the active WSL Codex home.
2. The model emits the intended browser leaf tool name.
3. The bridge now preserves that tool call into the rollout as a proper `function_call`.
4. The router itself returns the failure message as tool output.

That means the failure is no longer:
- missing model tool intent
- missing bridge stream item
- lost continuation item before persistence

## Current first wrong stage

Codex router / executable tool registry for Playwright browser leaf tools on WSL `codex exec 0.123.0`.

## Failure class

Client-side tool execution limitation or registry mismatch.

## Minimal fix target

Codex client support for namespaced Playwright leaf tools on this surface, specifically the router path that decides whether:

- `mcp__playwright__browser_navigate`
- `mcp__playwright__browser_snapshot`

are supported executable calls.

## Notes

This is separate from the Playwright MCP resource-method limitation:

- `list_mcp_resources`
- `list_mcp_resource_templates`

still fail with `Mcp error: -32601: Method not found` because the Playwright MCP server does not implement those resource endpoints.

The browser leaf-tool issue is different:
- the router says the call itself is unsupported before MCP execution succeeds.
