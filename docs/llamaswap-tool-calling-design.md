# llama-swap Tool Calling Design (vLLM-aligned)

## Interfaces
```go
type ParsedToolCall struct {
  CallID    string
  Name      string
  Arguments map[string]any
}

type ToolCallParser interface {
  Name() string
  MatchesModel(modelName string) bool
  ParseAssistantOutput(content string) ([]ParsedToolCall, string)
}
```

Registry:
- Default registry in `proxy/tool_call_parser.go`.
- First parser: Qwen XML (`qwen_xml` semantics).
- Selection: first parser whose `MatchesModel(model)` returns true.

## Data Structures
1. Request path:
   - `/v1/responses` normalized to chat-completions-compatible payload.
   - Preserves `tools`, `tool_choice`, and passback-ready history.
2. Response path:
   - Upstream chat completion message parsed into responses `output[]`.
   - Tool calls map to `shell_call`/`apply_patch_call`/`function_call` etc.
3. Passback:
   - Existing normalization maps `function_call_output` + `*_call_output` to `role="tool"` with `tool_call_id`.

## Endpoint Changes
No external endpoint shape change in this phase.
Internal behavior change:
- Parser extraction is now modular via registry, not hardcoded in `proxymanager.go`.

## Streaming Event Format Strategy
Current:
- `writeResponsesStream()` emits OpenAI responses-style event envelopes from final translated payload.

Phase 2 target:
1. Parse incremental upstream chat deltas.
2. Assemble tool call fragments by call index/id.
3. Emit deterministic `response.output_item.*` events for partial and done states.
4. Preserve current event order for backward compatibility.

## Chat-Template Integration Strategy
llama-swap remains template-agnostic by design:
1. Upstream model server/template is source of formatting truth.
2. llama-swap preserves tool-call history and `tool` role passback fields.
3. Model-family parser is used only to recover structured tool calls from assistant text when upstream does not supply `tool_calls`.

Compatibility boundary:
- If upstream template does not support tool-role or prior tool calls, loop quality degrades; llama-swap cannot fix template semantics alone.

## Parser Abstraction Strategy
1. Core registry and parser contract in `proxy/tool_call_parser.go`.
2. Parsers are isolated and model-family bound.
3. Adding a parser is additive:
   - implement `ToolCallParser`
   - register in default registry.
4. Optional future config override:
   - `model.tool_parser: qwen_xml|...`
   - fallback to model-name matching.

## Examples
### 1) `tool_choice=auto`
```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3-Coder-30B",
    "messages":[{"role":"user","content":"show cwd using shell"}],
    "tools":[{"type":"function","function":{"name":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],
    "tool_choice":"auto"
  }'
```

### 2) `tool_choice=required`
```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3-Coder-30B",
    "messages":[{"role":"user","content":"call weather tool"}],
    "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}],
    "tool_choice":"required"
  }'
```

### 3) Named function calling
```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3-Coder-30B",
    "messages":[{"role":"user","content":"weather"}],
    "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}],
    "tool_choice":{"type":"function","function":{"name":"get_weather"}}
  }'
```

### 4) Second-turn passback (`tool` role)
```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="dummy")
tools = [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]

first = client.chat.completions.create(
    model="Qwen3-Coder-30B",
    messages=[{"role":"user","content":"what is weather in Madrid?"}],
    tools=tools,
    tool_choice="auto",
)

call = first.choices[0].message.tool_calls[0]
result = {"temp_c": 21, "condition": "sunny"}  # external executor output

second = client.chat.completions.create(
    model="Qwen3-Coder-30B",
    messages=[
        {"role":"user","content":"what is weather in Madrid?"},
        {"role":"assistant","tool_calls":[call.model_dump()]},
        {"role":"tool","tool_call_id":call.id,"content":json.dumps(result)},
    ],
)
print(second.choices[0].message.content)
```

## Test Evidence (Phase 1)
Run:
```bash
cd /home/admmin/llama-swap/llama-swap-main
/usr/local/go/bin/go test ./proxy/... -count=1
```

Relevant tests:
- `proxy/proxymanager_bridge_xml_test.go`
- `proxy/tool_call_parser_test.go`

## Limitations
1. Schema-constrained decoding (`required`/named guarantees) depends on upstream model server capability, not llama-swap proxy.
2. Incremental streaming tool extraction is pending phase 2.
3. Only Qwen XML parser is included in this phase.
