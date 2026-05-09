# Responses Passthrough Investigation

- Generated: 2026-05-09T09:53:43+02:00
- Native target: `http://127.0.0.1:10008/v1`
- Bridge target: `http://127.0.0.1:8080/v1`
- Output directory: `tmp/responses_passthrough_investigation`

## Codex Oracle
- Text turns must complete with a message-shaped final answer that Codex can render directly.
- Tool turns must preserve the bridge tool phase: actionable tool item, `requires_action`, and no mixed prose+tool payload.
- Streamed tool turns must include complete lifecycle events, including terminal function-arguments events.
- Plan turns must preserve a single `<proposed_plan>...</proposed_plan>` block with no execution leakage.
- Native passthrough stays blocked wherever bridge-only contract repair is still required.

## Compatibility Matrix
| case | category | native | bridge | recommendation | notes |
|---|---|---|---|---|---|
| `post_basic` | protocol | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `get_collection` | protocol | `error=not_found_error; code=404` | `http=404` | `needs shim` | collection GET is not implemented natively (HTTP 404) |
| `get_by_id` | protocol | `error=not_found_error; code=404` | `http=404` | `needs shim` | response retrieval by id is not implemented natively (HTTP 404) |
| `previous_response_id` | protocol | `error=invalid_request_error; code=400` | `status=completed; output_types=reasoning,message; output_text=True` | `bridge-only` | previous_response_id is rejected natively (HTTP 400) |
| `invalid_json` | protocol | `error=server_error; code=500` | `error=; code=None` | `needs shim` | http status differs (native 500 vs bridge 400) |
| `invalid_tool_type` | protocol | `error=invalid_request_error; code=400` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | http status differs (native 400 vs bridge 200) |
| `instructions_text` | nonstream | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `array_input` | nonstream | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `reasoning_effort` | nonstream | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=message; output_text=True` | `needs shim` | top-level output_text is missing; output item types differ from bridge; top-level output_text presence differs from bridge |
| `reasoning_summary` | nonstream | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `include_field` | nonstream | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `max_output_tokens` | nonstream | `status=completed; output_types=reasoning; output_text=False` | `status=completed; output_types=message; output_text=True` | `bridge-only` | text response has no assistant message |
| `tool_choice_none` | tool | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | top-level output_text is missing; top-level output_text presence differs from bridge |
| `tool_choice_auto` | tool | `status=completed; output_types=reasoning,function_call; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `bridge-only` | tool turn does not end in requires_action; tool item status differs from bridge tool phase |
| `tool_choice_required` | tool | `status=completed; output_types=reasoning,function_call; output_text=False` | `status=requires_action; output_types=reasoning,function_call; output_text=True` | `bridge-only` | tool turn does not end in requires_action; tool item status differs from bridge tool phase |
| `mixed_text_tool` | tool | `status=completed; output_types=reasoning,message,function_call; output_text=False` | `status=requires_action; output_types=reasoning,message,function_call; output_text=True` | `bridge-only` | tool turn does not end in requires_action; tool turn mixes prose and tool output; tool item status differs from bridge tool phase |
| `function_call_output_passback` | tool | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | post-tool continuation omits top-level output_text; top-level output_text presence differs from bridge |
| `plan_only` | codex | `status=completed; output_types=reasoning,message; output_text=False` | `status=completed; output_types=reasoning,message; output_text=True` | `needs shim` | plan response omits top-level output_text; top-level output_text presence differs from bridge |
| `apply_patch_intent` | codex | `status=completed; output_types=reasoning,function_call; output_text=False` | `status=completed; output_types=message; output_text=True` | `bridge-only` | tool turn does not end in requires_action; tool item status differs from bridge tool phase |
| `stream_basic` | stream | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_text.delta,response.content_part.added,response.output_text.delta,response.output_item.done,response.output_text.done,response.content_part.done,response.completed; status=completed; output_types=reasoning,message` | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_summary_part.added,response.reasoning_summary_text.delta,response.content_part.added,response.output_text.delta,response.reasoning_summary_text.done,response.reasoning_summary_part.done,response.output_item.done,response.output_text.done,response.content_part.done,response.completed; status=completed; output_types=reasoning,message,message` | `needs shim` | completed streamed response omits top-level output_text; stream reasoning lane name differs from bridge expectation; stream event order differs from bridge; reasoning event names differ from bridge |
| `stream_tool` | stream | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_text.delta,response.function_call_arguments.delta,response.output_item.done,response.completed; status=completed; output_types=reasoning,function_call` | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_summary_part.added,response.reasoning_summary_text.delta,response.content_part.added,response.output_text.delta,response.reasoning_summary_text.done,response.reasoning_summary_part.done,response.output_item.done,response.output_text.done,response.content_part.done,response.completed; status=completed; output_types=reasoning,message,message` | `bridge-only` | tool stream does not end in requires_action; tool stream omits response.function_call_arguments.done |
| `stream_plan_only` | stream | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_text.delta,response.content_part.added,response.output_text.delta,response.output_item.done,response.output_text.done,response.content_part.done,response.completed; status=completed; output_types=reasoning,message` | `events=response.created,response.in_progress,response.output_item.added,response.reasoning_summary_part.added,response.reasoning_summary_text.delta,response.content_part.added,response.output_text.delta,response.reasoning_summary_text.done,response.reasoning_summary_part.done,response.output_item.done,response.output_text.done,response.content_part.done,response.completed; status=completed; output_types=reasoning,message,message` | `needs shim` | streamed plan reasoning lane differs from bridge expectation; completed streamed plan response omits top-level output_text; stream event order differs from bridge; reasoning event names differ from bridge |

## Key Mismatches
### Bridge-Only
- `previous_response_id`: previous_response_id is rejected natively (HTTP 400)
- `max_output_tokens`: text response has no assistant message
- `tool_choice_auto`: tool turn does not end in requires_action; tool item status differs from bridge tool phase
- `tool_choice_required`: tool turn does not end in requires_action; tool item status differs from bridge tool phase
- `mixed_text_tool`: tool turn does not end in requires_action; tool turn mixes prose and tool output; tool item status differs from bridge tool phase
- `apply_patch_intent`: tool turn does not end in requires_action; tool item status differs from bridge tool phase
- `stream_tool`: tool stream does not end in requires_action; tool stream omits response.function_call_arguments.done

### Needs Shim
- `post_basic`: top-level output_text is missing; top-level output_text presence differs from bridge
- `get_collection`: collection GET is not implemented natively (HTTP 404)
- `get_by_id`: response retrieval by id is not implemented natively (HTTP 404)
- `invalid_json`: http status differs (native 500 vs bridge 400)
- `invalid_tool_type`: http status differs (native 400 vs bridge 200)
- `instructions_text`: top-level output_text is missing; top-level output_text presence differs from bridge
- `array_input`: top-level output_text is missing; top-level output_text presence differs from bridge
- `reasoning_effort`: top-level output_text is missing; output item types differ from bridge; top-level output_text presence differs from bridge
- `reasoning_summary`: top-level output_text is missing; top-level output_text presence differs from bridge
- `include_field`: top-level output_text is missing; top-level output_text presence differs from bridge
- `tool_choice_none`: top-level output_text is missing; top-level output_text presence differs from bridge
- `function_call_output_passback`: post-tool continuation omits top-level output_text; top-level output_text presence differs from bridge
- `plan_only`: plan response omits top-level output_text; top-level output_text presence differs from bridge
- `stream_basic`: completed streamed response omits top-level output_text; stream reasoning lane name differs from bridge expectation; stream event order differs from bridge; reasoning event names differ from bridge
- `stream_plan_only`: streamed plan reasoning lane differs from bridge expectation; completed streamed plan response omits top-level output_text; stream event order differs from bridge; reasoning event names differ from bridge

## Routing Recommendation
- Native passthrough safe now: `none`
- Native passthrough only with shim: `post_basic, get_collection, get_by_id, invalid_json, invalid_tool_type, instructions_text, array_input, reasoning_effort, reasoning_summary, include_field, tool_choice_none, function_call_output_passback, plan_only, stream_basic, stream_plan_only`
- Keep bridge translation: `previous_response_id, max_output_tokens, tool_choice_auto, tool_choice_required, mixed_text_tool, apply_patch_intent, stream_tool`

## Artifacts
- `TARGET/CASE.request.json`: request payload used for the probe
- `TARGET/CASE.response.txt`: raw HTTP body
- `TARGET/CASE.summary.json`: normalized summary
- `compatibility_matrix.json`: machine-readable investigation results
