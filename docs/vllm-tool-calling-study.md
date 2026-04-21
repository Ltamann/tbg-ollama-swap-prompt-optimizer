# vLLM Tool Calling Study For llama-swap

## Concise Architecture Summary
vLLM splits tool calling into two mechanisms:
1. Structured decoding path for named function calling and `tool_choice="required"`.
2. Parser extraction path for `tool_choice="auto"` (model emits text, parser extracts tool calls).

In both cases, the caller executes the tool and performs passback by appending a `tool`-role message in the next turn.

Sources:
- https://docs.vllm.ai/en/stable/features/tool_calling/
- https://docs.vllm.ai/en/stable/examples/online_serving/openai_responses_client_with_mcp_tools/
- https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai
- https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers
- https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/responses
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/tool_server.py

## vLLM Architecture Summary
Request lifecycle:
1. Client sends chat/responses request with `messages`, `tools`, and `tool_choice`.
2. Server shapes prompt with chat template (`tools`, prior assistant tool calls, and `tool` role result messages must be representable by template).
3. Model generates.
4. For named/required modes, structured decoding enforces schema-conforming tool-call JSON.
5. For auto mode, parser extracts tool calls from raw model text.
6. API returns structured `tool_calls`.
7. Caller executes tools externally (unless using an explicit server-owned tool loop).
8. Caller appends tool result with `role="tool"` and matching `tool_call_id`.
9. Next model turn consumes the appended tool result and continues.

Key behavior from docs:
- `auto`, `required`, `none`, named function are supported.
- `required`/named use constrained structured outputs.
- `auto` depends on parser extraction and may produce malformed args.
- `strict` is accepted but not enforced in vLLM currently.
- Auto tool choice requires flags like:
  - `--enable-auto-tool-choice`
  - `--tool-call-parser <name>`
  - optional `--tool-parser-plugin`
  - chat template support for tool roles and prior tool calls.

## Exact Techniques To Replicate
1. Keep `tool_choice` semantics explicit and mode-dependent.
2. Use parser abstraction and registry, model-family selected.
3. Keep chat-template/tool-role passback compatible with OpenAI shape.
4. Keep streaming and non-stream extraction code paths separate.
5. Treat MCP/tool-server integration as explicit registration/routing, not implicit execution.

## Gap Analysis
| Area | vLLM behavior | llama-swap current | Gap |
|---|---|---|---|
| `tool_choice` modes | explicit semantics (`auto`,`required`,`none`,named) | forwarded/normalized in bridge paths | needs stricter validation matrix docs/tests |
| Parser architecture | pluggable parser registry + plugins | previously hardcoded Qwen XML extraction in translation | now moved to parser interface + registry (phase 1) |
| Chat template integration | template must encode tools, prior tool calls, tool results | upstream model/template-dependent, llama-swap proxy is mostly transport/normalization | document boundary; no template compiler in llama-swap |
| Passback loop | caller executes tool + reinjects `tool` message | supported in request normalization path | needs explicit round-trip tests/examples |
| Streaming extraction | parser-specific incremental extraction in vLLM | responses bridge emits SSE from final translated payload | phase 2 needed for incremental extraction |
| MCP tool servers | explicit tool server process/registration | no first-class built-in loop | add optional hooks, keep default external execution |

## Recommended Architecture For llama-swap
1. Keep llama-swap as protocol bridge and normalizer by default.
2. Expose structured tool calls consistently.
3. Maintain explicit passback-ready message transformations.
4. Add parser registry (`ToolCallParser`) for model-family parsing in bridge translation.
5. Keep optional extension points for server-owned tool execution later.

## Phased Implementation Plan
Phase 1 (implemented in this patch):
1. Parser abstraction (`ToolCallParser`, registry, model-match).
2. Qwen XML parser implementation in dedicated parser module.
3. Non-stream extraction wired through parser registry in responses bridge translation.
4. Tests for parser selection and `tool_choice` passthrough compatibility.

Phase 2:
1. Streaming extraction/assembly for tool deltas before terminal chunk.
2. Additional model-family parsers (e.g., llama json, hermes-like patterns).
3. Config-driven parser selection override.
4. Optional MCP/tool-server adapter hooks (without forcing built-in execution loop).

## Risk List
1. Auto mode parser false positives/false negatives on mixed content.
2. Model output drift between checkpoints can break parser assumptions.
3. Streaming clients may rely on chunk order/shape; delta extraction must be backward compatible.
4. Tool argument coercion may hide malformed model output unless surfaced.

## Test Plan
1. Non-stream:
   - tool call extracted into structured output item
   - malformed/partial XML parameter closure behavior
   - no-tools normal chat unchanged
2. Request normalization:
   - `tool_choice` passthrough for named/required/none/auto
   - tool-role passback mapping remains stable
3. Round trip:
   - first turn returns tool call
   - second turn includes tool result with `tool_call_id`
   - model gets coherent next-turn context
4. Streaming (phase 2):
   - partial tool-call assembly
   - boundary detection
   - compatibility with existing SSE consumers

## Adoption Notes
Current support in this phase:
1. Parser architecture is model-family aware.
2. Qwen XML family parser is implemented.
3. Passback-ready shape is preserved through existing message normalization.

Current limitations:
1. No constrained decoding inside llama-swap; this remains upstream-engine capability.
2. Streaming tool-call extraction is not yet incremental.
3. Non-Qwen families require additional parser implementations.
