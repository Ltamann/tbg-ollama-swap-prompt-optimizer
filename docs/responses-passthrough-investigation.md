# Responses Passthrough Investigation

Use [scripts/investigate_responses_passthrough.py](/home/admmin/llama-swap/llama-swap-main/scripts/investigate_responses_passthrough.py) to evaluate whether native `llama.cpp` Responses traffic can bypass the current `llama-swap` Responses bridge for the current Codex stack.

The script does not change routing. It probes both targets with the same fixed request matrix and writes a compatibility report you can review before touching `proxymanager.go`.

## What It Tests

- Endpoint and validation surface:
  - `POST /v1/responses`
  - `GET /v1/responses`
  - `GET /v1/responses/{id}`
  - `previous_response_id`
  - invalid JSON
  - invalid tool definitions
- Non-stream behavior:
  - plain text replies
  - `instructions`
  - array-form `input`
  - `reasoning.effort`
  - `reasoning.summary`
  - `include`
  - `max_output_tokens`
- Tool behavior:
  - `tool_choice: none`
  - `tool_choice: auto`
  - `tool_choice: required`
  - mixed prose plus tool output
  - `function_call_output` continuation
  - `apply_patch`-intent prompt
- Stream behavior:
  - plain final-answer turn
  - tool turn
  - plan-only turn

## Codex Oracle

The probe evaluates native output against the current bridge contract, not against a generic OpenAI-compatible baseline.

The current oracle treats these as critical:

- tool turns must remain actionable and end in `requires_action`
- tool turns must not mix final prose and tool output in one response
- streamed tool turns must include the full tool-arguments lifecycle
- plan turns must preserve a single `<proposed_plan>...</proposed_plan>` block
- final-answer turns should preserve the final-answer shape that Codex already sees through `llama-swap`

Cases are classified as:

- `pass`: native output already matches the current bridge contract closely enough for direct passthrough
- `needs shim`: native output is close but still needs a lightweight normalization step
- `bridge-only`: native output still needs the existing translation/repair path

## Example

```bash
python3 scripts/investigate_responses_passthrough.py \
  --native-base-url http://127.0.0.1:10008/v1 \
  --bridge-base-url http://127.0.0.1:8080/v1 \
  --model Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --output-dir tmp/responses_passthrough_investigation \
  --startup-wait-s 180 \
  --timeout-s 120
```

If `llama-swap` expects a routed alias instead of the native upstream model id, pass `--bridge-model ...` explicitly. Otherwise the script auto-detects the first bridge model from `/v1/models`.

Use `--startup-wait-s` when the native server is still loading a large model and may answer early probes with `503 Loading model`.

## Outputs

- `compatibility_matrix.md`: human-readable report with recommendations
- `compatibility_matrix.json`: machine-readable summary
- `native/CASE.request.json`: request payload
- `native/CASE.response.txt`: raw response body
- `native/CASE.summary.json`: normalized summary
- `bridge/CASE.request.json`: request payload
- `bridge/CASE.response.txt`: raw response body
- `bridge/CASE.summary.json`: normalized summary

Review the markdown report first. If a case is still `bridge-only`, keep it on the translated path until the exact mismatch is fixed or explicitly shimmable.
