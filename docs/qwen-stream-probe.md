# Qwen Stream Probe

Use [scripts/qwen_stream_probe.py](/home/admmin/llama-swap/llama-swap-main/scripts/qwen_stream_probe.py) to measure what the upstream OpenAI-compatible `chat/completions` server actually streams before `llama-swap` reconstructs Responses/Codex events.

## Cases

- `reasoning_only`: text-only thinking turn
- `plan_only`: direct `<proposed_plan>` turn with no tools
- `tool_reasoning`: thinking plus native tool call
- `tool_then_final`: tool call followed by final explanation

## Example

```bash
python3 scripts/qwen_stream_probe.py \
  --base-url http://localhost:10008/v1 \
  --model Qwen3.6-35B-A3B-UD-Q8_K_XL \
  --output-dir tmp/qwen_stream_probe \
  --timeout-s 120
```

## Outputs

For each case, the probe writes:

- `CASE.sse.txt`: raw upstream SSE lines
- `CASE.summary.json`: compact capability summary

The summary records:

- `chunk_count`
- `content_delta_count`
- `reasoning_delta_count`
- `tool_call_delta_count`
- `tool_names`
- `finish_reasons`
- `content_preview`
- `reasoning_preview`

Use these probe outputs as the source of truth for:

- whether `reasoning_content` is incremental or one large block
- whether visible `content` streams progressively
- whether native `tool_calls` appear
- whether tool arguments arrive incrementally
