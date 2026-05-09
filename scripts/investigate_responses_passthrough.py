#!/usr/bin/env python3
"""Probe native llama.cpp Responses compatibility against the current bridge."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


SEVERITY_ORDER = {"pass": 0, "needs shim": 1, "bridge-only": 2}


@dataclass(frozen=True)
class ProbeCase:
    name: str
    category: str
    description: str
    method: str
    path_template: str
    stream: bool
    oracle_kind: str
    request_body: Any | None = None
    request_body_raw: str | None = None
    dynamic_response_id_from: str | None = None


@dataclass(frozen=True)
class Target:
    name: str
    base_url: str
    model: str = ""


def fixed_tool_schema(tool_name: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool_name,
            "description": f"{tool_name} test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
                "required": ["value"],
            },
        }
    ]


def apply_patch_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a structured file patch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "diff": {"type": "string"},
                        },
                        "required": ["type", "path"],
                    }
                },
                "required": ["operation"],
            },
        }
    ]


def build_cases(model: str) -> list[ProbeCase]:
    return [
        ProbeCase(
            name="post_basic",
            category="protocol",
            description="Basic non-stream text response",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={"model": model, "input": "Reply with exactly: BASIC_OK"},
        ),
        ProbeCase(
            name="get_collection",
            category="protocol",
            description="Collection GET support",
            method="GET",
            path_template="/responses",
            stream=False,
            oracle_kind="protocol_collection_get",
        ),
        ProbeCase(
            name="get_by_id",
            category="protocol",
            description="Response retrieval by id",
            method="GET",
            path_template="/responses/{response_id}",
            stream=False,
            oracle_kind="protocol_get_by_id",
            dynamic_response_id_from="post_basic",
        ),
        ProbeCase(
            name="previous_response_id",
            category="protocol",
            description="Continuation via previous_response_id",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="continuation_link",
            request_body={
                "model": model,
                "input": "Reply with exactly: PREV_OK",
                "previous_response_id": "resp_probe_previous",
            },
        ),
        ProbeCase(
            name="invalid_json",
            category="protocol",
            description="Invalid JSON request validation",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="protocol_invalid_json",
            request_body_raw='{"model":"broken"',
        ),
        ProbeCase(
            name="invalid_tool_type",
            category="protocol",
            description="Invalid tool type validation",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="protocol_invalid_tool",
            request_body={"model": model, "input": "hi", "tools": [{"type": "computer_use_preview"}]},
        ),
        ProbeCase(
            name="instructions_text",
            category="nonstream",
            description="Instructions plus plain input",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "instructions": "You are a terse assistant.",
                "input": "Reply with exactly: INSTR_OK",
            },
        ),
        ProbeCase(
            name="array_input",
            category="nonstream",
            description="Array-form input with message/input_text",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Reply with exactly: ARRAY_OK"}],
                    }
                ],
            },
        ),
        ProbeCase(
            name="reasoning_effort",
            category="nonstream",
            description="Reasoning effort request",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": "Reply with exactly: THINK_OK",
                "reasoning": {"effort": "low"},
            },
        ),
        ProbeCase(
            name="reasoning_summary",
            category="nonstream",
            description="Reasoning summary request",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": "Reply with exactly: SUMMARY_OK",
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        ),
        ProbeCase(
            name="include_field",
            category="nonstream",
            description="Include field request",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": "Reply with exactly: INCLUDE_OK",
                "include": ["reasoning.encrypted_content"],
            },
        ),
        ProbeCase(
            name="max_output_tokens",
            category="nonstream",
            description="max_output_tokens handling",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": "Reply with exactly: TOKENS_OK",
                "max_output_tokens": 32,
            },
        ),
        ProbeCase(
            name="tool_choice_none",
            category="tool",
            description="tool_choice=none with tool present",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="text_nonstream",
            request_body={
                "model": model,
                "input": "Reply with exactly: NO_TOOL_OK",
                "tools": fixed_tool_schema("echo_tool"),
                "tool_choice": "none",
            },
        ),
        ProbeCase(
            name="tool_choice_auto",
            category="tool",
            description="tool_choice=auto single tool call",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="tool_nonstream",
            request_body={
                "model": model,
                "input": "Use the tool to answer. If you can call it, do so.",
                "tools": fixed_tool_schema("echo_tool"),
                "tool_choice": "auto",
            },
        ),
        ProbeCase(
            name="tool_choice_required",
            category="tool",
            description="tool_choice=required single tool call",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="tool_nonstream",
            request_body={
                "model": model,
                "input": "Use the tool.",
                "tools": fixed_tool_schema("echo_tool"),
                "tool_choice": "required",
            },
        ),
        ProbeCase(
            name="mixed_text_tool",
            category="tool",
            description="Mixed prose and tool call in one turn",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="tool_nonstream",
            request_body={
                "model": model,
                "input": "First say PRELUDE and then call the tool echo_tool with value mixed.",
                "tools": fixed_tool_schema("echo_tool"),
                "tool_choice": "auto",
            },
        ),
        ProbeCase(
            name="function_call_output_passback",
            category="tool",
            description="Passback using function_call_output",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="continuation_final_nonstream",
            request_body={
                "model": model,
                "input": [
                    {"type": "function_call_output", "call_id": "call_123", "output": '{"ok":true}'},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Given the tool output, reply with exactly: PASSBACK_OK",
                            }
                        ],
                    },
                ],
            },
        ),
        ProbeCase(
            name="plan_only",
            category="codex",
            description="Plan-only prompt with proposed plan wrapper",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="plan_nonstream",
            request_body={
                "model": model,
                "input": (
                    "Return exactly one complete <proposed_plan> block for a small math game. "
                    "Do not call tools."
                ),
            },
        ),
        ProbeCase(
            name="apply_patch_intent",
            category="codex",
            description="Apply-patch intent with native function schema",
            method="POST",
            path_template="/responses",
            stream=False,
            oracle_kind="tool_nonstream",
            request_body={
                "model": model,
                "input": "Use apply_patch to create hello.txt containing HELLO.",
                "tools": apply_patch_tool_schema(),
                "tool_choice": "auto",
            },
        ),
        ProbeCase(
            name="stream_basic",
            category="stream",
            description="Plain streamed final answer",
            method="POST",
            path_template="/responses",
            stream=True,
            oracle_kind="stream_text",
            request_body={"model": model, "input": "Reply with exactly: STREAM_OK", "stream": True},
        ),
        ProbeCase(
            name="stream_tool",
            category="stream",
            description="Streamed tool call turn",
            method="POST",
            path_template="/responses",
            stream=True,
            oracle_kind="stream_tool",
            request_body={
                "model": model,
                "input": "Use the tool to answer. If you can call it, do so.",
                "stream": True,
                "tools": fixed_tool_schema("echo_tool"),
                "tool_choice": "auto",
            },
        ),
        ProbeCase(
            name="stream_plan_only",
            category="stream",
            description="Streamed plan-only turn",
            method="POST",
            path_template="/responses",
            stream=True,
            oracle_kind="stream_plan",
            request_body={
                "model": model,
                "stream": True,
                "input": (
                    "Return exactly one complete <proposed_plan> block for a small language-learning game. "
                    "Do not call tools."
                ),
            },
        ),
    ]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def dump_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def safe_json_loads(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def detect_first_model_id(base_url: str, timeout_s: int) -> str:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                model_id = str(item.get("id", "")).strip()
                if model_id:
                    return model_id
    models = payload.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                model_id = str(item.get("id") or item.get("model") or item.get("name") or "").strip()
                if model_id:
                    return model_id
    raise RuntimeError(f"could not detect model id from {url}")


def target_request_body(target: Target, case: ProbeCase) -> Any | None:
    if case.request_body is None:
        return None
    if not isinstance(case.request_body, dict):
        return case.request_body
    body = json.loads(json.dumps(case.request_body))
    if target.model and "model" in body:
        body["model"] = target.model
    return body


def make_request(target: Target, case: ProbeCase, response_ids: dict[str, str]) -> tuple[str, bytes | None]:
    path = case.path_template
    if case.dynamic_response_id_from:
        response_id = response_ids.get(case.dynamic_response_id_from, "") or "resp_probe_missing"
        path = path.format(response_id=response_id)
    url = target.base_url.rstrip("/") + path
    if case.request_body_raw is not None:
        body_bytes = case.request_body_raw.encode("utf-8")
    else:
        request_body = target_request_body(target, case)
        body_bytes = json.dumps(request_body).encode("utf-8") if request_body is not None else None
    return url, body_bytes


def send_http_request(
    target: Target,
    case: ProbeCase,
    timeout_s: int,
    response_ids: dict[str, str],
) -> dict[str, Any]:
    url, body_bytes = make_request(target, case, response_ids)
    headers = {"Accept": "text/event-stream" if case.stream else "application/json"}
    if case.method != "GET":
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=body_bytes, headers=headers, method=case.method)
    started = time.time()

    status_code = 0
    response_headers: dict[str, str] = {}
    raw_body = ""
    error_text = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status_code = response.status
            response_headers = dict(response.headers.items())
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status_code = error.code
        response_headers = dict(error.headers.items()) if error.headers else {}
        raw_body = error.read().decode("utf-8", errors="replace")
        error_text = str(error)
    except TimeoutError as error:
        error_text = f"timeout: {error}"
    except urllib.error.URLError as error:
        error_text = str(error)
    except Exception as error:
        error_text = str(error)
    duration_ms = int((time.time() - started) * 1000)
    return {
        "url": url,
        "status_code": status_code,
        "headers": response_headers,
        "body": raw_body,
        "error_text": error_text,
        "duration_ms": duration_ms,
    }


def wait_for_target_ready(target: Target, timeout_s: int) -> None:
    if timeout_s <= 0:
        return
    deadline = time.time() + timeout_s
    request = urllib.request.Request(
        url=target.base_url.rstrip("/") + "/responses",
        data=json.dumps({"model": target.model, "input": "Reply with exactly: READY_OK"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status == 200:
                    return
                payload = response.read().decode("utf-8", errors="replace")
                last_error = payload
        except urllib.error.HTTPError as error:
            payload = error.read().decode("utf-8", errors="replace")
            last_error = payload or str(error)
            if error.code == 503 and "Loading model" in last_error:
                time.sleep(5)
                continue
            return
        except Exception as error:
            last_error = str(error)
            time.sleep(2)
            continue
    raise RuntimeError(f"target {target.name} did not become ready within {timeout_s}s: {last_error}")


def summarize_output_items(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    if not isinstance(output, list):
        output = []

    output_types: list[str] = []
    tool_items: list[dict[str, Any]] = []
    message_items: list[dict[str, Any]] = []
    reasoning_items: list[dict[str, Any]] = []
    first_message_text = ""
    final_message_text = ""
    plan_wrapper_present = False

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip()
        if item_type:
            output_types.append(item_type)
        if item_type == "message":
            message_items.append(item)
            content = item.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and str(part.get("type", "")).strip() == "output_text":
                        text = str(part.get("text", ""))
                        if text:
                            text_parts.append(text)
                if text_parts:
                    joined = "".join(text_parts).strip()
                    if joined:
                        if not first_message_text:
                            first_message_text = joined
                        final_message_text = joined
        elif item_type == "reasoning":
            reasoning_items.append(item)
        elif item_type.endswith("_call") or item_type == "function_call":
            tool_items.append(item)

    candidate_plan_text = final_message_text or first_message_text
    if "<proposed_plan>" in candidate_plan_text and "</proposed_plan>" in candidate_plan_text:
        plan_wrapper_present = True

    return {
        "response_id": str(response.get("id", "")).strip(),
        "response_status": str(response.get("status", "")).strip(),
        "object": str(response.get("object", "")).strip(),
        "output_types": output_types,
        "tool_item_count": len(tool_items),
        "message_item_count": len(message_items),
        "reasoning_item_count": len(reasoning_items),
        "mixed_message_tool": bool(tool_items and message_items),
        "top_level_output_text_present": "output_text" in response,
        "usage_present": "usage" in response,
        "timings_present": "timings" in response,
        "plan_wrapper_present": plan_wrapper_present,
        "first_message_text": first_message_text,
        "final_message_text": final_message_text,
        "tool_item_statuses": [str(item.get("status", "")).strip() for item in tool_items],
        "tool_item_names": [str(item.get("name", "")).strip() for item in tool_items],
    }


def parse_sse(body: str) -> dict[str, Any]:
    event_names: list[str] = []
    event_counts: dict[str, int] = {}
    reasoning_event_names: list[str] = []
    current_event = ""
    completed_response: dict[str, Any] | None = None
    output_item_done_types: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("event: "):
            current_event = stripped[7:]
            event_names.append(current_event)
            event_counts[current_event] = event_counts.get(current_event, 0) + 1
            if "reasoning" in current_event and current_event not in reasoning_event_names:
                reasoning_event_names.append(current_event)
            continue
        if not stripped.startswith("data: "):
            continue
        data = stripped[6:]
        if not data or data == "[DONE]":
            continue
        payload = safe_json_loads(data)
        if not isinstance(payload, dict):
            continue
        if current_event == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                completed_response = response
        elif current_event == "response.output_item.done":
            item = payload.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip()
                if item_type:
                    output_item_done_types.append(item_type)

    deduped_event_order: list[str] = []
    for name in event_names:
        if name not in deduped_event_order:
            deduped_event_order.append(name)

    completed_summary = summarize_output_items(completed_response or {})
    return {
        "raw_event_count": len(event_names),
        "event_order": deduped_event_order,
        "event_counts": event_counts,
        "reasoning_event_names": reasoning_event_names,
        "has_created": "response.created" in event_counts,
        "has_in_progress": "response.in_progress" in event_counts,
        "has_output_text_done": "response.output_text.done" in event_counts,
        "has_function_arguments_done": "response.function_call_arguments.done" in event_counts,
        "has_completed": "response.completed" in event_counts,
        "output_item_done_types": output_item_done_types,
        "completed_response": completed_summary,
    }


def summarize_target_case(case: ProbeCase, http_result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "http_status": http_result["status_code"],
        "duration_ms": http_result["duration_ms"],
        "error_text": http_result["error_text"],
        "stream": case.stream,
    }
    body = http_result["body"]
    if case.stream:
        summary["stream_summary"] = parse_sse(body)
    else:
        payload = safe_json_loads(body)
        if isinstance(payload, dict):
            if "output" in payload or payload.get("object") == "response":
                summary["response_summary"] = summarize_output_items(payload)
            else:
                raw_error = payload.get("error")
                error_obj = raw_error if isinstance(raw_error, dict) else {}
                summary["error_summary"] = {
                    "error_type": str(error_obj.get("type", "")).strip(),
                    "error_code": error_obj.get("code"),
                    "message": (
                        str(error_obj.get("message", "")).strip()
                        if isinstance(raw_error, dict)
                        else str(raw_error or "").strip()
                    ),
                }
        else:
            summary["non_json_body"] = body[:800]
    return summary


def compare_against_bridge(case: ProbeCase, native_summary: dict[str, Any], bridge_summary: dict[str, Any] | None) -> list[str]:
    if not bridge_summary:
        return []
    diffs: list[str] = []
    if native_summary.get("http_status") != bridge_summary.get("http_status"):
        diffs.append(
            f"http status differs (native {native_summary.get('http_status')} vs bridge {bridge_summary.get('http_status')})"
        )

    if case.stream:
        native_stream = native_summary.get("stream_summary", {})
        bridge_stream = bridge_summary.get("stream_summary", {})
        if native_stream.get("event_order") != bridge_stream.get("event_order"):
            diffs.append("stream event order differs from bridge")
        if native_stream.get("reasoning_event_names") != bridge_stream.get("reasoning_event_names"):
            diffs.append("reasoning event names differ from bridge")
        native_completed = native_stream.get("completed_response", {})
        bridge_completed = bridge_stream.get("completed_response", {})
        if native_completed.get("response_status") != bridge_completed.get("response_status"):
            diffs.append(
                "stream completed response status differs "
                f"(native {native_completed.get('response_status')} vs bridge {bridge_completed.get('response_status')})"
            )
    else:
        native_response = native_summary.get("response_summary", {})
        bridge_response = bridge_summary.get("response_summary", {})
        if native_response and bridge_response:
            if native_response.get("response_status") != bridge_response.get("response_status"):
                diffs.append(
                    "response status differs "
                    f"(native {native_response.get('response_status')} vs bridge {bridge_response.get('response_status')})"
                )
            if native_response.get("output_types") != bridge_response.get("output_types"):
                diffs.append("output item types differ from bridge")
            if native_response.get("top_level_output_text_present") != bridge_response.get(
                "top_level_output_text_present"
            ):
                diffs.append("top-level output_text presence differs from bridge")
    return diffs


def pick_classification(issues: list[tuple[str, str]]) -> tuple[str, list[str]]:
    if not issues:
        return "pass", []
    highest = max(issues, key=lambda item: SEVERITY_ORDER[item[0]])[0]
    reasons = [reason for severity, reason in issues if severity == highest]
    return highest, reasons


def evaluate_native_case(case: ProbeCase, native_summary: dict[str, Any], bridge_summary: dict[str, Any] | None) -> dict[str, Any]:
    issues: list[tuple[str, str]] = []
    http_status = int(native_summary.get("http_status", 0) or 0)
    bridge_diffs = compare_against_bridge(case, native_summary, bridge_summary)

    if case.oracle_kind == "protocol_collection_get":
        if http_status == 200:
            issues.append(("needs shim", "native collection GET works but must still be checked against bridge semantics"))
        else:
            issues.append(("needs shim", f"collection GET is not implemented natively (HTTP {http_status})"))
    elif case.oracle_kind == "protocol_get_by_id":
        if http_status != 200:
            issues.append(("needs shim", f"response retrieval by id is not implemented natively (HTTP {http_status})"))
    elif case.oracle_kind == "continuation_link":
        if http_status != 200:
            issues.append(("bridge-only", f"previous_response_id is rejected natively (HTTP {http_status})"))
    elif case.oracle_kind in {"protocol_invalid_json", "protocol_invalid_tool"}:
        if http_status < 400:
            issues.append(("needs shim", "invalid request did not return an error"))
    elif case.stream:
        stream_summary = native_summary.get("stream_summary", {})
        completed = stream_summary.get("completed_response", {})
        event_order = stream_summary.get("event_order", [])
        if http_status != 200:
            issues.append(("bridge-only", f"stream request failed with HTTP {http_status}"))
        if not stream_summary.get("has_created") or not stream_summary.get("has_in_progress") or not stream_summary.get("has_completed"):
            issues.append(("bridge-only", "stream lifecycle is incomplete"))

        if case.oracle_kind == "stream_text":
            if completed.get("tool_item_count"):
                issues.append(("bridge-only", "plain text stream unexpectedly emitted tool items"))
            if not completed.get("message_item_count"):
                issues.append(("bridge-only", "plain text stream did not finish with a message item"))
            if not completed.get("top_level_output_text_present"):
                issues.append(("needs shim", "completed streamed response omits top-level output_text"))
            if "response.reasoning_summary_text.delta" not in event_order:
                issues.append(("needs shim", "stream reasoning lane name differs from bridge expectation"))
        elif case.oracle_kind == "stream_tool":
            if completed.get("tool_item_count", 0) == 0:
                issues.append(("bridge-only", "tool stream did not produce a tool item"))
            if completed.get("response_status") != "requires_action":
                issues.append(("bridge-only", "tool stream does not end in requires_action"))
            if completed.get("mixed_message_tool"):
                issues.append(("bridge-only", "tool stream mixes prose and tool output in one turn"))
            if not stream_summary.get("has_function_arguments_done"):
                issues.append(("bridge-only", "tool stream omits response.function_call_arguments.done"))
        elif case.oracle_kind == "stream_plan":
            if not completed.get("plan_wrapper_present"):
                issues.append(("bridge-only", "streamed plan turn is missing the proposed plan wrapper"))
            if completed.get("tool_item_count"):
                issues.append(("bridge-only", "streamed plan turn emitted tool items"))
            if "response.reasoning_summary_text.delta" not in event_order:
                issues.append(("needs shim", "streamed plan reasoning lane differs from bridge expectation"))
            if not completed.get("top_level_output_text_present"):
                issues.append(("needs shim", "completed streamed plan response omits top-level output_text"))
    else:
        response_summary = native_summary.get("response_summary", {})
        if http_status != 200:
            issues.append(("bridge-only", f"request failed with HTTP {http_status}"))
        if case.oracle_kind == "text_nonstream":
            if response_summary.get("response_status") != "completed":
                issues.append(("bridge-only", "text response is not completed"))
            if response_summary.get("tool_item_count"):
                issues.append(("bridge-only", "text response unexpectedly emitted tool items"))
            if not response_summary.get("message_item_count"):
                issues.append(("bridge-only", "text response has no assistant message"))
            if not response_summary.get("top_level_output_text_present"):
                issues.append(("needs shim", "top-level output_text is missing"))
        elif case.oracle_kind == "tool_nonstream":
            if response_summary.get("tool_item_count", 0) == 0:
                issues.append(("bridge-only", "tool turn did not emit a tool item"))
            if response_summary.get("response_status") != "requires_action":
                issues.append(("bridge-only", "tool turn does not end in requires_action"))
            if response_summary.get("mixed_message_tool"):
                issues.append(("bridge-only", "tool turn mixes prose and tool output"))
            if response_summary.get("tool_item_statuses") and any(
                status != "in_progress" for status in response_summary.get("tool_item_statuses", [])
            ):
                issues.append(("bridge-only", "tool item status differs from bridge tool phase"))
        elif case.oracle_kind == "continuation_final_nonstream":
            if response_summary.get("response_status") != "completed":
                issues.append(("bridge-only", "post-tool continuation is not completed"))
            if response_summary.get("tool_item_count"):
                issues.append(("bridge-only", "post-tool continuation still contains tool items"))
            if not response_summary.get("message_item_count"):
                issues.append(("bridge-only", "post-tool continuation has no final assistant message"))
            if not response_summary.get("top_level_output_text_present"):
                issues.append(("needs shim", "post-tool continuation omits top-level output_text"))
        elif case.oracle_kind == "plan_nonstream":
            if response_summary.get("tool_item_count"):
                issues.append(("bridge-only", "plan turn emitted tool items"))
            if not response_summary.get("plan_wrapper_present"):
                issues.append(("bridge-only", "plan turn is missing the proposed plan wrapper"))
            if not response_summary.get("top_level_output_text_present"):
                issues.append(("needs shim", "plan response omits top-level output_text"))

    for diff in bridge_diffs:
        issues.append(("needs shim", diff))

    classification, reasons = pick_classification(issues)
    return {
        "classification": classification,
        "reasons": reasons,
        "bridge_differences": bridge_diffs,
    }


def render_case_highlights(case: ProbeCase, summary: dict[str, Any]) -> str:
    if case.stream:
        stream_summary = summary.get("stream_summary", {})
        completed = stream_summary.get("completed_response", {})
        event_order = stream_summary.get("event_order", [])
        return (
            f"events={','.join(event_order)}; "
            f"status={completed.get('response_status', '')}; "
            f"output_types={','.join(completed.get('output_types', []))}"
        )
    response_summary = summary.get("response_summary", {})
    error_summary = summary.get("error_summary", {})
    if response_summary:
        return (
            f"status={response_summary.get('response_status', '')}; "
            f"output_types={','.join(response_summary.get('output_types', []))}; "
            f"output_text={response_summary.get('top_level_output_text_present')}"
        )
    if error_summary:
        return f"error={error_summary.get('error_type', '')}; code={error_summary.get('error_code')}"
    return f"http={summary.get('http_status')}"


def render_report(
    output_dir: str,
    targets: list[Target],
    cases: list[ProbeCase],
    summaries: dict[str, dict[str, dict[str, Any]]],
    evaluations: dict[str, dict[str, Any]],
) -> str:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    native = next((target for target in targets if target.name == "native"), None)
    bridge = next((target for target in targets if target.name == "bridge"), None)

    lines: list[str] = []
    lines.append("# Responses Passthrough Investigation")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    if native:
        lines.append(f"- Native target: `{native.base_url}`")
    if bridge:
        lines.append(f"- Bridge target: `{bridge.base_url}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("")
    lines.append("## Codex Oracle")
    lines.append("- Text turns must complete with a message-shaped final answer that Codex can render directly.")
    lines.append("- Tool turns must preserve the bridge tool phase: actionable tool item, `requires_action`, and no mixed prose+tool payload.")
    lines.append("- Streamed tool turns must include complete lifecycle events, including terminal function-arguments events.")
    lines.append("- Plan turns must preserve a single `<proposed_plan>...</proposed_plan>` block with no execution leakage.")
    lines.append("- Native passthrough stays blocked wherever bridge-only contract repair is still required.")
    lines.append("")
    lines.append("## Compatibility Matrix")
    lines.append("| case | category | native | bridge | recommendation | notes |")
    lines.append("|---|---|---|---|---|---|")
    for case in cases:
        native_summary = summaries.get("native", {}).get(case.name, {})
        bridge_summary = summaries.get("bridge", {}).get(case.name, {})
        evaluation = evaluations.get(case.name, {})
        notes = "; ".join(evaluation.get("reasons", [])) or "passes current oracle"
        lines.append(
            f"| `{case.name}` | {case.category} | `{render_case_highlights(case, native_summary)}` | "
            f"`{render_case_highlights(case, bridge_summary)}` | `{evaluation.get('classification', '')}` | {notes} |"
        )
    lines.append("")
    lines.append("## Key Mismatches")
    grouped: dict[str, list[str]] = {"bridge-only": [], "needs shim": []}
    for case in cases:
        evaluation = evaluations.get(case.name, {})
        classification = evaluation.get("classification", "")
        if classification in grouped:
            grouped[classification].append(
                f"`{case.name}`: " + "; ".join(evaluation.get("reasons", []))
            )
    if grouped["bridge-only"]:
        lines.append("### Bridge-Only")
        for line in grouped["bridge-only"]:
            lines.append(f"- {line}")
        lines.append("")
    if grouped["needs shim"]:
        lines.append("### Needs Shim")
        for line in grouped["needs shim"]:
            lines.append(f"- {line}")
        lines.append("")

    lines.append("## Routing Recommendation")
    safe = [case.name for case in cases if evaluations.get(case.name, {}).get("classification") == "pass"]
    shim = [case.name for case in cases if evaluations.get(case.name, {}).get("classification") == "needs shim"]
    blocked = [case.name for case in cases if evaluations.get(case.name, {}).get("classification") == "bridge-only"]
    lines.append(f"- Native passthrough safe now: `{', '.join(safe) if safe else 'none'}`")
    lines.append(f"- Native passthrough only with shim: `{', '.join(shim) if shim else 'none'}`")
    lines.append(f"- Keep bridge translation: `{', '.join(blocked) if blocked else 'none'}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- `TARGET/CASE.request.json`: request payload used for the probe")
    lines.append("- `TARGET/CASE.response.txt`: raw HTTP body")
    lines.append("- `TARGET/CASE.summary.json`: normalized summary")
    lines.append("- `compatibility_matrix.json`: machine-readable investigation results")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Investigate native Responses passthrough safety against llama-swap.")
    parser.add_argument(
        "--native-base-url",
        default="http://127.0.0.1:10008/v1",
        help="Native llama.cpp base URL ending in /v1",
    )
    parser.add_argument(
        "--bridge-base-url",
        default="http://127.0.0.1:8080/v1",
        help="llama-swap bridge base URL ending in /v1",
    )
    parser.add_argument("--model", required=True, help="Native llama.cpp model identifier to send in requests")
    parser.add_argument(
        "--bridge-model",
        default="",
        help="Optional explicit bridge model identifier; defaults to the first id returned by /v1/models",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for raw artifacts and reports")
    parser.add_argument("--timeout-s", type=int, default=90, help="Per-request timeout in seconds")
    parser.add_argument(
        "--startup-wait-s",
        type=int,
        default=0,
        help="Optional per-target readiness wait before running the matrix; useful when a llama.cpp server is still loading",
    )
    args = parser.parse_args()

    bridge_model = args.bridge_model or detect_first_model_id(args.bridge_base_url, args.timeout_s)
    targets = [
        Target(name="native", base_url=args.native_base_url, model=args.model),
        Target(name="bridge", base_url=args.bridge_base_url, model=bridge_model),
    ]
    cases = build_cases(args.model)
    ensure_dir(args.output_dir)

    for target in targets:
        wait_for_target_ready(target, args.startup_wait_s)

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    response_ids: dict[str, dict[str, str]] = {target.name: {} for target in targets}

    for target in targets:
        target_dir = os.path.join(args.output_dir, target.name)
        ensure_dir(target_dir)
        summaries[target.name] = {}
        for case in cases:
            request_path = os.path.join(target_dir, f"{case.name}.request.json")
            request_dump = {"method": case.method, "path_template": case.path_template, "target_model": target.model}
            resolved_body = target_request_body(target, case)
            if resolved_body is not None:
                request_dump["body"] = resolved_body
            if case.request_body_raw is not None:
                request_dump["body_raw"] = case.request_body_raw
            if case.dynamic_response_id_from:
                request_dump["dynamic_response_id_from"] = case.dynamic_response_id_from
                request_dump["resolved_response_id"] = response_ids[target.name].get(case.dynamic_response_id_from, "")
            dump_json(request_path, request_dump)

            http_result = send_http_request(target, case, args.timeout_s, response_ids[target.name])
            dump_text(os.path.join(target_dir, f"{case.name}.response.txt"), http_result["body"])
            dump_json(os.path.join(target_dir, f"{case.name}.headers.json"), http_result["headers"])

            summary = summarize_target_case(case, http_result)
            summaries[target.name][case.name] = summary
            dump_json(os.path.join(target_dir, f"{case.name}.summary.json"), summary)

            if not case.stream:
                response_summary = summary.get("response_summary", {})
                response_id = str(response_summary.get("response_id", "")).strip()
                if response_id:
                    response_ids[target.name][case.name] = response_id
            else:
                stream_summary = summary.get("stream_summary", {})
                completed_response = stream_summary.get("completed_response", {})
                response_id = str(completed_response.get("response_id", "")).strip()
                if response_id:
                    response_ids[target.name][case.name] = response_id

    evaluations: dict[str, dict[str, Any]] = {}
    for case in cases:
        evaluations[case.name] = evaluate_native_case(
            case,
            summaries.get("native", {}).get(case.name, {}),
            summaries.get("bridge", {}).get(case.name),
        )

    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": [target.__dict__ for target in targets],
        "cases": [case.__dict__ for case in cases],
        "summaries": summaries,
        "evaluations": evaluations,
    }
    dump_json(os.path.join(args.output_dir, "compatibility_matrix.json"), matrix)
    report = render_report(args.output_dir, targets, cases, summaries, evaluations)
    dump_text(os.path.join(args.output_dir, "compatibility_matrix.md"), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
