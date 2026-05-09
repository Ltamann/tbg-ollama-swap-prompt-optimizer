#!/usr/bin/env python3
"""Direct upstream Qwen/llama.cpp streaming probe utility.

This script sends OpenAI-compatible chat.completions requests directly to an
upstream server and saves raw SSE output plus a compact capability summary.
It is intended to measure what the model/runtime actually streams before
llama-swap reconstructs Responses/Codex-style events.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ProbeCase:
    name: str
    body: dict[str, Any]


def build_cases(model: str) -> dict[str, ProbeCase]:
    base_messages = lambda text: [{"role": "user", "content": text}]
    return {
        "reasoning_only": ProbeCase(
            "reasoning_only",
            {
                "model": model,
                "stream": True,
                "messages": base_messages(
                    "Think carefully and explain how you would compare two local harnesses. "
                    "Do not call tools. Keep the final visible answer to one sentence."
                ),
            },
        ),
        "plan_only": ProbeCase(
            "plan_only",
            {
                "model": model,
                "stream": True,
                "messages": base_messages(
                    "Write a plan directly in <proposed_plan>...</proposed_plan> tags for a small language-learning game. "
                    "Do not call tools."
                ),
            },
        ),
        "tool_reasoning": ProbeCase(
            "tool_reasoning",
            {
                "model": model,
                "stream": True,
                "messages": base_messages(
                    "Use the shell tool to inspect the current working directory. "
                    "Think first, then call the tool."
                ),
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "description": "Run a read-only shell command.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                },
                                "required": ["command"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        ),
        "tool_then_final": ProbeCase(
            "tool_then_final",
            {
                "model": model,
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "First use the shell tool to echo FINAL_PROBE_OK, then provide a short final explanation."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "description": "Run a read-only shell command.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                },
                                "required": ["command"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        ),
    }


def iter_sse_lines(response) -> list[str]:
    lines: list[str] = []
    for raw in response:
        text = raw.decode("utf-8", errors="replace")
        lines.append(text)
    return lines


def summarize_sse(lines: list[str]) -> dict[str, Any]:
    chunks = 0
    content_deltas = 0
    reasoning_deltas = 0
    tool_call_deltas = 0
    finish_reasons: list[str] = []
    content_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    tool_names: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        chunks += 1
        choice = (((obj.get("choices") or [{}])[0]) or {})
        delta = (choice.get("delta") or {})
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        tool_calls = delta.get("tool_calls") or []
        finish_reason = choice.get("finish_reason")
        if content:
            content_deltas += 1
            content_fragments.append(content)
        if reasoning:
            reasoning_deltas += 1
            reasoning_fragments.append(reasoning)
        if tool_calls:
            tool_call_deltas += 1
            for call in tool_calls:
                name = (((call or {}).get("function") or {}).get("name") or "").strip()
                if name:
                    tool_names.append(name)
        if finish_reason:
            finish_reasons.append(str(finish_reason))

    return {
        "chunk_count": chunks,
        "content_delta_count": content_deltas,
        "reasoning_delta_count": reasoning_deltas,
        "tool_call_delta_count": tool_call_deltas,
        "tool_names": tool_names,
        "finish_reasons": finish_reasons,
        "content_preview": "".join(content_fragments)[:400],
        "reasoning_preview": "".join(reasoning_fragments)[:400],
    }


def run_probe(base_url: str, case: ProbeCase, output_dir: str, timeout_s: int) -> int:
    url = base_url.rstrip("/") + "/chat/completions"
    request_body = json.dumps(case.body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            lines = iter_sse_lines(response)
            duration_ms = int((time.time() - started) * 1000)
            summary = summarize_sse(lines)
            summary["case"] = case.name
            summary["http_status"] = response.status
            summary["duration_ms"] = duration_ms
            with open(os.path.join(output_dir, f"{case.name}.sse.txt"), "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            with open(os.path.join(output_dir, f"{case.name}.summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        with open(os.path.join(output_dir, f"{case.name}.error.txt"), "w", encoding="utf-8") as fh:
            fh.write(body)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe direct Qwen/llama.cpp streaming behavior.")
    parser.add_argument("--base-url", required=True, help="Upstream base URL, e.g. http://localhost:10008/v1")
    parser.add_argument("--model", required=True, help="Model name to send in chat.completions requests")
    parser.add_argument(
        "--case",
        action="append",
        choices=["reasoning_only", "plan_only", "tool_reasoning", "tool_then_final"],
        help="Run one or more named cases; default is all cases",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save raw SSE and summaries")
    parser.add_argument("--timeout-s", type=int, default=90, help="Request timeout in seconds")
    args = parser.parse_args()

    cases = build_cases(args.model)
    selected = args.case or list(cases.keys())
    failures = 0
    for name in selected:
        failures += run_probe(args.base_url, cases[name], args.output_dir, args.timeout_s)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
