# Codex Local Bug Fix Loop

## Current Snapshot

This document tracks the current Codex -> bridge -> llama-swap investigation so confirmed bugs stay separate from earlier false positives.

## Evidence Set

Primary rollouts:

1. `C:\Users\YLAB-Partner\.codex\sessions\2026\05\01\rollout-2026-05-01T08-19-49-019de231-855d-73f0-9545-66cb9692952d.jsonl`
2. `C:\Users\YLAB-Partner\.codex\sessions\2026\05\01\rollout-2026-05-01T08-15-17-019de22d-5f3e-7dd1-9406-0c16d27fbbb6.jsonl`

Primary captures:

1. `C:\Users\YLAB-Partner\AppData\Local\Temp\cap7.json`
2. `C:\Users\YLAB-Partner\AppData\Local\Temp\cap9.json`
3. `C:\Users\YLAB-Partner\AppData\Local\Temp\cap12.json`
4. `C:\Users\YLAB-Partner\AppData\Local\Temp\cap16_resp.json`

## Corrected Bug Matrix

### B1: Plan/default transition semantics

- Status: confirmed behavior, product intent still needs clarification
- Corrected finding:
  - the plan turn produced a `<proposed_plan>`
  - a later turn started in `default`
  - file mutation happened in that later default-mode turn
  - current evidence does not prove that the same plan turn violated plan-mode rules

### B2: malformed shell tool-call validation

- Status: non-stream path fixed, live stream path fixed
- Corrected finding:
  - the observed empty shell `{}` call was rejected before OS execution
  - validation behavior still needs to be explicit and consistent in the bridge
- Fix target:
  - `proxy/proxymanager.go`
 - Latest repair status:
   - non-stream normalization now converts empty shell calls into bridge validation messages
   - live SSE streaming now delays shell tool visibility until buffered arguments normalize to a usable command payload
   - malformed streamed shell calls no longer surface to Codex as executable `function_call` items

### B3: reasoning is not live-streamed incrementally

- Status: confirmed
- Corrected finding:
  - reasoning is emitted in valid Responses event shape
  - in the investigated runs it appears as one buffered summary delta, not a continuous stream during execution

### B4: reasoning/commentary duplication

- Status: confirmed, currently intentional workaround
- Corrected finding:
  - the user-visible comment is coming from `commentary`
  - native reasoning is still emitted separately
  - duplication is expected until native reasoning UI becomes reliable

### B5: native reasoning panel suppression

- Status: confirmed symptom
- Corrected finding:
  - native reasoning items are structurally correct
  - Codex UI still prefers the commentary workaround for visibility in these local-model runs

### B6: reasoning-history contamination

- Status: risk, not proven
- Corrected finding:
  - full reasoning in commentary may pollute replayed assistant history
  - the current captures do not yet prove that replay path

## Repair Order

1. Harden malformed tool-call validation
2. Trace live reasoning emission timing
3. Reduce commentary risk without losing visible UI feedback
4. Verify history replay behavior
5. Revisit native reasoning-panel activation only after the bridge contract is stable
