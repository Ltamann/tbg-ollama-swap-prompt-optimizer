# Codex Reasoning UI Study

## Finding

In local Codex plus `llama-swap` testing, native Responses reasoning events can be structurally correct and still not render in the Codex UI thinking panel.

Observed good native event shape:

- `response.output_item.added` with `type:"reasoning"`
- `id` shaped like `rs_<completion>_<index>`
- `response.reasoning_summary_part.added`
- `response.reasoning_summary_text.delta`
- `response.reasoning_summary_text.done`
- `response.output_item.done` for the reasoning item

Even when that sequence is correct, Codex UI may still hide the reasoning panel for local model IDs such as `.gguf` model names. In practice this means native reasoning delivery alone is not enough for reliable user-visible thinking in Codex.

## Confirmed Workaround

For responses that contain extracted think content, the bridge should emit three lanes:

1. Native Responses reasoning item
2. Assistant message with `channel:"commentary"`
3. Assistant message with `channel:"final"`

This gives two benefits:

- Native clients that do understand reasoning keep working
- Codex UI gets a visible intermediary message even when it does not render the native reasoning lane

Recommended output order:

1. `reasoning`
2. `commentary`
3. `final`

## Important Caveat: Commentary Becomes History

`commentary` is rendered visibly, but it is still an assistant message lane. That means it may become part of conversation history and may be replayed back to the model on continuation turns.

This creates a real risk:

- if full raw reasoning is copied into `commentary`, the model may later read its own full chain-of-thought-like text
- that can bloat prompt history
- that can cause self-conditioning on internal reasoning text instead of on the user-facing result
- that can accidentally turn the workaround into a persistent memory channel

Because of that, `commentary` should be treated as a UI visibility workaround, not as a safe storage lane for full reasoning.

## Recommendation

Use `commentary` for a short reasoning summary, not for full reasoning text.

Recommended policy:

- Native `reasoning` lane: keep full extracted reasoning summary/events as currently emitted
- `commentary` lane: emit a short user-safe summary of the thinking, ideally one to three sentences
- `final` lane: emit the clean answer only

Example:

- `reasoning`: full native reasoning summary stream
- `commentary`: "Thinking through the game scope, choosing a small web-based format, and keeping the feature set simple."
- `final`: actual answer

This reduces both prompt pollution and accidental leakage while still making the UI feel alive.

## Why Not Send Full Reasoning as a Function Call

Using a synthetic function call to carry full reasoning is not recommended.

Reasons:

1. It changes semantics
   - function calls are supposed to request tool execution, not carry hidden UI-only content

2. It pollutes tool history
   - synthetic tool calls would appear in the tool-call transcript and may confuse continuation logic

3. It risks breaking clients
   - clients may try to execute or validate the fake call
   - tool-call handling is stricter than message rendering

4. It is not a true reasoning lane
   - even if a client renders function calls in a collapsible block, that is still not the same contract as native reasoning

So while a function call may visually collapse in some clients, it is the wrong protocol layer for reasoning transport.

## Best Practical Approach

If Codex UI does not render native reasoning for local model IDs:

1. Keep native `reasoning` events for compatibility and future-proofing
2. Add a short `commentary` summary for immediate visible UX
3. Do not mirror full raw reasoning into `commentary` by default
4. Do not encode reasoning as a fake function call

## Optional Future Improvement

The bridge can support a policy knob such as:

- `reasoning_visibility_mode: native_only`
- `reasoning_visibility_mode: commentary_summary`
- `reasoning_visibility_mode: commentary_full`

Recommended default:

- `commentary_summary`

That keeps the workaround available while making the history/prompt tradeoff explicit.
