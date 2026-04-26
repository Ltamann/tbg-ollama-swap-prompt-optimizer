#!/usr/bin/env python3
import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NativeSummary:
    prompt_line: str
    ran_command_count: int
    success_count: int
    asked_questions_count: int
    has_plan_block: bool
    has_implement_plan: bool
    has_deleted_file: bool
    has_review_step: bool


@dataclass
class CaptureCheck:
    capture_id: int
    timestamp: str
    status_code: int
    model: str
    tools_count: int
    output_types: List[str]
    tool_call_count: int
    status: str
    user_excerpt: str


def http_get_json(url: str, timeout_sec: int = 15) -> Any:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def decode_b64(s: str) -> str:
    if not s:
        return ""
    return base64.b64decode(s).decode("utf-8", errors="replace")


def parse_response_completed_sse(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
            return payload.get("response")
        except json.JSONDecodeError:
            return None

    lines = raw.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == "event: response.completed":
            for inner in range(index + 1, min(index + 8, len(lines))):
                line = lines[inner].strip()
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        payload = json.loads(data)
                        return payload.get("response")
                    except json.JSONDecodeError:
                        return None
    return None


def summarize_native_file(path: str) -> NativeSummary:
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prompt_line = ""
    for line in lines:
        if "math game" in line.lower() and "plan" in line.lower():
            prompt_line = line
            break

    return NativeSummary(
        prompt_line=prompt_line,
        ran_command_count=text.count("Ran command"),
        success_count=text.count("Success"),
        asked_questions_count=text.count("Asked 2 questions"),
        has_plan_block="Plan: 1st-Grade Math Game" in text,
        has_implement_plan="Implement plan" in text,
        has_deleted_file="Deleted file" in text,
        has_review_step=re.search(r"(?m)^Review\s*$", text) is not None,
    )


def extract_last_user_excerpt(req: Dict[str, Any], max_len: int = 120) -> str:
    items = req.get("input")
    if not isinstance(items, list):
        return ""
    user_messages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "user":
            user_messages.append(item)
    if not user_messages:
        return ""
    last = user_messages[-1]
    content = last.get("content")
    chunks: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_text":
                text = str(part.get("text", "")).strip()
                if text:
                    chunks.append(text)
    value = " ".join(chunks).strip()
    if len(value) > max_len:
        return value[:max_len] + "..."
    return value


def classify_capture(
    metric: Dict[str, Any],
    req: Dict[str, Any],
    response: Optional[Dict[str, Any]],
) -> CaptureCheck:
    output = response.get("output") if isinstance(response, dict) else []
    if not isinstance(output, list):
        output = []

    output_types: List[str] = []
    tool_call_count = 0
    message_count = 0
    for item in output:
        item_type = ""
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).strip()
        if item_type:
            output_types.append(item_type)
        if item_type == "message":
            message_count += 1
        if item_type == "function_call" or item_type.endswith("_call"):
            tool_call_count += 1

    tools_count = 0
    tools = req.get("tools")
    if isinstance(tools, list):
        tools_count = len(tools)

    status = "ok"
    status_code = int(metric.get("status_code", 0) or 0)
    if status_code != 200:
        status = f"http_{status_code}"
    elif response is None:
        status = "missing_response_completed"
    elif tools_count > 0 and tool_call_count == 0 and message_count > 0:
        status = "tools_requested_but_message_only"
    elif message_count > 0 and tool_call_count > 0:
        status = "mixed_message_tool_call"

    return CaptureCheck(
        capture_id=int(metric.get("id", -1)),
        timestamp=str(metric.get("timestamp", "")),
        status_code=status_code,
        model=str(metric.get("model", "")),
        tools_count=tools_count,
        output_types=output_types,
        tool_call_count=tool_call_count,
        status=status,
        user_excerpt=extract_last_user_excerpt(req),
    )


def collect_checks(base_url: str, limit: int) -> Tuple[List[CaptureCheck], int]:
    metrics = http_get_json(f"{base_url.rstrip('/')}/api/metrics")
    if not isinstance(metrics, list):
        raise RuntimeError("`/api/metrics` did not return a JSON array.")

    selected = [row for row in metrics if isinstance(row, dict) and row.get("has_capture")]
    selected = sorted(selected, key=lambda row: int(row.get("id", 0)), reverse=True)[:limit]

    checks: List[CaptureCheck] = []
    missing_count = 0
    for metric in selected:
        capture_id = int(metric.get("id", -1))
        try:
            capture = http_get_json(f"{base_url.rstrip('/')}/api/captures/{capture_id}")
        except urllib.error.HTTPError as err:
            if err.code == 404:
                missing_count += 1
                continue
            raise

        if not isinstance(capture, dict):
            continue
        if capture.get("req_path") != "/v1/responses":
            continue

        req_raw = decode_b64(str(capture.get("req_body", "")))
        resp_raw = decode_b64(str(capture.get("resp_body", "")))
        try:
            req = json.loads(req_raw) if req_raw else {}
        except json.JSONDecodeError:
            req = {}
        response = parse_response_completed_sse(resp_raw)
        checks.append(classify_capture(metric=metric, req=req, response=response))

    checks = sorted(checks, key=lambda row: row.capture_id)
    return checks, missing_count


def render_report(native: NativeSummary, checks: List[CaptureCheck], missing_count: int) -> str:
    total = len(checks)
    status_counts: Dict[str, int] = {}
    for row in checks:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    lines: List[str] = []
    lines.append("# Native vs Bridge Output-Shape Report")
    lines.append("")
    lines.append("## Native Baseline")
    lines.append(f"- Prompt sample: `{native.prompt_line}`")
    lines.append(f"- `Ran command` count: {native.ran_command_count}")
    lines.append(f"- `Success` count: {native.success_count}")
    lines.append(f"- Has plan block: {native.has_plan_block}")
    lines.append(f"- Has implement step: {native.has_implement_plan}")
    lines.append(f"- Has file edit step (`Deleted file` marker): {native.has_deleted_file}")
    lines.append(f"- Has review step: {native.has_review_step}")
    lines.append("")
    lines.append("## Bridge Capture Summary")
    lines.append(f"- Checked captures: {total}")
    lines.append(f"- Missing/evicted captures skipped: {missing_count}")
    for key in sorted(status_counts.keys()):
        lines.append(f"- `{key}`: {status_counts[key]}")
    lines.append("")
    lines.append("## Flagged Captures")
    lines.append("| id | status | model | tools | tool_calls | output_types | user_excerpt |")
    lines.append("|---:|---|---|---:|---:|---|---|")
    for row in checks:
        if row.status == "ok":
            continue
        types = ",".join(row.output_types)
        lines.append(
            f"| {row.capture_id} | {row.status} | {row.model} | {row.tools_count} | "
            f"{row.tool_call_count} | `{types}` | `{row.user_excerpt}` |"
        )
    if all(row.status == "ok" for row in checks):
        lines.append("| - | - | - | - | - | - | All checked captures were `ok`. |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Native transcript shape is action-first (plan -> implement -> edit/review markers present).")
    lines.append("- Any `tools_requested_but_message_only` indicates a non-actionable turn for the CLI.")
    lines.append("- Any `mixed_message_tool_call` indicates combined prose + tool phase, which can confuse plan/tool rendering in some clients.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare native Codex transcript shape against recent llama-swap /v1/responses captures."
    )
    parser.add_argument("--native-file", required=True, help="Path to the native Codex output transcript file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-swap base URL.")
    parser.add_argument("--limit", type=int, default=80, help="Max retained capture rows to inspect.")
    parser.add_argument("--out", default="", help="Optional output report path. Prints to stdout if omitted.")
    args = parser.parse_args()

    native = summarize_native_file(args.native_file)
    checks, missing_count = collect_checks(args.base_url, args.limit)
    report = render_report(native, checks, missing_count)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            file.write(report)
        print(f"Report written: {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
