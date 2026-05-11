package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestParseChatEvent_ResponsesSSETimelineIncludesToolCalls(t *testing.T) {
	req := []byte(`{"model":"gpt-5.3-codex","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"run ls"}]}]}`)
	resp := []byte("event: response.output_item.added\n" +
		`data: {"item":{"type":"shell_call","id":"fc_1","call_id":"call_1","action":{"commands":["ls"]},"status":"completed"}}` + "\n\n" +
		"event: response.function_call_arguments.done\n" +
		`data: {"name":"shell","call_id":"call_1","arguments":"{\"commands\":[\"ls\"]}"}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"delta":"done"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{"status":"completed"}}` + "\n\n" +
		"data: [DONE]\n")

	evt := parseChatEvent("/v1/responses", req, resp, TokenMetrics{ID: 1, Model: "gpt-5.3-codex", StatusCode: 200})
	require.NotNil(t, evt)
	require.NotEmpty(t, evt.Timeline)

	var hasToolCall bool
	var hasToolArgs bool
	for _, entry := range evt.Timeline {
		if entry.Kind == "tool_call" && entry.ToolName == "shell" {
			hasToolCall = true
		}
		if entry.Kind == "tool_args" && entry.CallID == "call_1" {
			hasToolArgs = true
		}
	}
	assert.True(t, hasToolCall)
	assert.True(t, hasToolArgs)
	require.NotNil(t, evt.AssistantResponse)
	assert.Contains(t, evt.AssistantResponse.Content, "done")
}

func TestParseChatEvent_ResponsesJSONTimelineToolOutputPreview(t *testing.T) {
	req := []byte(`{"model":"gpt-5.3-codex","input":"hello"}`)
	resp := []byte(`{
		"id":"resp_1",
		"status":"completed",
		"output":[
			{"type":"shell_call_output","call_id":"call_1","output":"line1\nline2\nline3\nline4"}
		]
	}`)

	evt := parseChatEvent("/v1/responses", req, resp, TokenMetrics{ID: 2, Model: "gpt-5.3-codex", StatusCode: 200})
	require.NotNil(t, evt)
	require.Len(t, evt.Timeline, 1)
	assert.Equal(t, "tool_output", evt.Timeline[0].Kind)
	assert.Equal(t, "line1\nline2\nline3", evt.Timeline[0].OutputPreview)
	assert.True(t, evt.Timeline[0].Truncated)
}

func TestParseChatEvent_ResponsesTimelineFlagsCompletedUnresolvedToolCall(t *testing.T) {
	req := []byte(`{"model":"gpt-5.3-codex","input":"hello"}`)
	resp := []byte(`{
		"id":"resp_2",
		"status":"completed",
		"output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Proceeding with tool execution."}]},
			{"type":"shell_call","call_id":"call_1","action":{"command":"pwd"},"status":"completed"}
		]
	}`)

	evt := parseChatEvent("/v1/responses", req, resp, TokenMetrics{ID: 3, Model: "gpt-5.3-codex", StatusCode: 200})
	require.NotNil(t, evt)

	var found bool
	for _, entry := range evt.Timeline {
		if entry.Kind == "error" && entry.Status == "protocol_incomplete_tool_phase" {
			found = true
			assert.Contains(t, entry.Content, "unresolved tool call")
		}
	}
	assert.True(t, found)
}

func TestParseChatEvent_ResponsesSSETimelineFlagsOrphanToolArgEvents(t *testing.T) {
	req := []byte(`{"model":"gpt-5.3-codex","input":"run tool"}`)
	resp := []byte("event: response.function_call_arguments.delta\n" +
		`data: {"item_id":"fc_missing","call_id":"call_missing","delta":"{\"command\":\"pwd\"}"}` + "\n\n" +
		"event: response.function_call_arguments.done\n" +
		`data: {"item_id":"fc_missing","name":"shell","call_id":"call_missing","arguments":"{\"command\":\"pwd\"}"}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"response":{"status":"completed"}}` + "\n\n" +
		"data: [DONE]\n")

	evt := parseChatEvent("/v1/responses", req, resp, TokenMetrics{ID: 4, Model: "gpt-5.3-codex", StatusCode: 200})
	require.NotNil(t, evt)

	var orphanDelta bool
	var orphanDone bool
	for _, entry := range evt.Timeline {
		if entry.Kind == "error" && entry.Status == "tool_args_orphan_delta" {
			orphanDelta = true
		}
		if entry.Kind == "error" && entry.Status == "tool_args_orphan_done" {
			orphanDone = true
		}
	}

	assert.True(t, orphanDelta)
	assert.True(t, orphanDone)
}

func TestParseAssistantResponse_StripsLeadingOrphanThinkCloser(t *testing.T) {
	resp := gjson.Parse(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"role":"assistant",
					"content":"",
					"reasoning_content":"</think>\n\nWIN_P9_FILE_LOCALCWD_DONE"
				}
			}
		]
	}`)

	ar := parseAssistantResponse(resp)
	require.NotNil(t, ar)
	assert.Equal(t, "WIN_P9_FILE_LOCALCWD_DONE", ar.Content)
	assert.Equal(t, "", ar.ReasoningContent)
}

func TestParseAssistantResponseFromBody_StripsLeadingOrphanThinkCloserFromSSE(t *testing.T) {
	resp := []byte("data: " +
		`{"choices":[{"index":0,"delta":{"reasoning_content":"</think>\n\n"},"finish_reason":null}]}` + "\n\n" +
		"data: " +
		`{"choices":[{"index":0,"delta":{"reasoning_content":"<tool_call>\n<function=exec_command>\n<parameter=cmd>\npwd\n</parameter>\n</function>\n</tool_call>"},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n")

	ar := parseAssistantResponseFromBody(resp)
	require.NotNil(t, ar)
	assert.NotContains(t, ar.Content, "</think>")
	assert.NotContains(t, ar.ReasoningContent, "</think>")
	assert.Contains(t, ar.Content, "<tool_call>")
}
