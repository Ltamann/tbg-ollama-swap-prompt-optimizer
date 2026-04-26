package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestParseQwenXMLToolCalls_ParsesFunctionAndParameters(t *testing.T) {
	text := `I will run a check first.

<tool_call>
<function=shell>
<parameter=command>
pwd
</parameter>
</function>
</tool_call>`

	calls, remaining := parseQwenXMLToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell", calls[0].Name)
	assert.Equal(t, "pwd", calls[0].Arguments["command"])
	assert.Equal(t, "I will run a check first.", remaining)
}

func TestParseQwenXMLToolCalls_ToleratesMissingParameterClose(t *testing.T) {
	text := `<tool_call>
<function=apply_patch>
<parameter=path>
README.md
<parameter=diff>
@@
-old
+new
</function>
</tool_call>`

	calls, _ := parseQwenXMLToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "apply_patch", calls[0].Name)
	assert.Equal(t, "README.md", calls[0].Arguments["path"])
	assert.Contains(t, calls[0].Arguments["diff"], "@@")
}

func TestTranslateChatCompletionToResponsesResponse_ConvertsQwenXMLToolCall(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "Planning done.\n\n<tool_call>\n<function=shell>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
	    },
	    "finish_reason": "stop"
	  }],
	  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	msg := output[0].(map[string]any)
	assert.Equal(t, "message", msg["type"])
	assert.Equal(t, "Planning done.", gjson.GetBytes(out, "output.0.content.0.text").String())

	call := output[1].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "shell", call["name"])
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	assert.Equal(t, "pwd", args["command"])
}

func TestTranslateChatCompletionToResponsesResponse_EmptyApplyPatchArgsBecomeAssistantMessage(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "tool_calls": [{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{}}"}
	      }]
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 1)
	call := output[0].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "apply_patch", call["name"])
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	_, hasOperation := args["operation"]
	assert.True(t, hasOperation)
}

func TestTranslateChatCompletionToResponsesResponse_EmptyApplyPatchArgsUsesPatchFromText(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "*** Begin Patch\n*** Update File: README.md\n@@\n-old\n+new\n*** End Patch\n",
	      "tool_calls": [{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{}"}
	      }]
	    },
	    "finish_reason": "tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	msg := output[0].(map[string]any)
	assert.Equal(t, "message", msg["type"])
	call := output[1].(map[string]any)
	assert.Equal(t, "apply_patch_call", call["type"])
	operation, ok := call["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", operation["type"])
	assert.Equal(t, "README.md", operation["path"])
	assert.Equal(t, "@@\n-old\n+new", operation["diff"])
}

func TestTranslateChatCompletionToResponsesResponse_SynthesizesApplyPatchCallFromPlainPatchText(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "I will apply this patch now.\n*** Begin Patch\n*** Update File: README.md\n@@\n-old\n+new\n*** End Patch\n"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	msg := output[0].(map[string]any)
	assert.Equal(t, "message", msg["type"])
	call := output[1].(map[string]any)
	assert.Equal(t, "apply_patch_call", call["type"])
	operation, ok := call["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", operation["type"])
	assert.Equal(t, "README.md", operation["path"])
	assert.Equal(t, "@@\n-old\n+new", operation["diff"])
}

func TestTranslateChatCompletionToResponsesResponse_SynthesizesApplyPatchCallFromFragmentedPatchBlocks(t *testing.T) {
	content := "I'll patch now.\n```diff\n*** Begin Patch\n*** Add File: demo-frag.txt\n+hello\n```\ntext in between\n```diff\n+world\n*** End Patch\n```"
	body := []byte(fmt.Sprintf(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": %q
	    },
	    "finish_reason": "stop"
	  }]
	}`, content))

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 1)

	call := output[0].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "apply_patch", call["name"])
	operation, ok := call["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "create_file", operation["type"])
	assert.Equal(t, "demo-frag.txt", operation["path"])
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchInputArgumentMappedToOperation(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "tool_calls": [{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"input\":\"*** Begin Patch\\n*** Update File: README.md\\n@@\\n-old\\n+new\\n*** End Patch\\n\"}"}
	      }]
	    },
	    "finish_reason": "tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 1)

	call := output[0].(map[string]any)
	assert.Equal(t, "apply_patch_call", call["type"])
	operation, ok := call["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", operation["type"])
	assert.Equal(t, "README.md", operation["path"])
	assert.Equal(t, "@@\n-old\n+new", operation["diff"])
}

func TestConvertApplyPatchTextToOperation_HandlesBackslashLineContinuations(t *testing.T) {
	raw := "*** Begin Patch\\\n*** Add File: demo.txt\\\n+hello\\\n*** End Patch\\\n"
	op, ok := convertApplyPatchTextToOperation(raw)
	require.True(t, ok)
	assert.Equal(t, "create_file", op["type"])
	assert.Equal(t, "demo.txt", op["path"])
	assert.Equal(t, "+hello", op["diff"])
}

func TestTranslateChatCompletionToResponsesResponse_RecoversPrefixedApplyPatchFromXMLParsedArgs(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Update File: README.md\n@@\n-old\n+new\n*** End Patch\n</parameter>\n</function>\n</tool_call>",
	      "tool_calls": [{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"__llamaswap_apply_patch","arguments":"{}"}
	      }]
	    },
	    "finish_reason": "tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 1)

	call := output[0].(map[string]any)
	assert.Equal(t, "apply_patch_call", call["type"])
	operation, ok := call["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", operation["type"])
	assert.Equal(t, "README.md", operation["path"])
	assert.Equal(t, "@@\n-old\n+new", operation["diff"])
}

func TestShouldForceStrictApplyPatchRetry(t *testing.T) {
	assert.False(t, shouldForceStrictApplyPatchRetry(""))
	assert.False(t, shouldForceStrictApplyPatchRetry("wrong_tool_call"))
	assert.False(t, shouldForceStrictApplyPatchRetry(" Wrong_Tool_Call "))
	assert.True(t, shouldForceStrictApplyPatchRetry("no_tool_call"))
	assert.True(t, shouldForceStrictApplyPatchRetry("planning_only"))
	assert.True(t, shouldForceStrictApplyPatchRetry("empty_operation"))
	assert.True(t, shouldForceStrictApplyPatchRetry("invalid_diff"))
}

func TestExtractResponsesRequestMode_DetectsPlanModeFromCollaborationInstruction(t *testing.T) {
	req := map[string]any{
		"instructions": "You are a coding agent.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<collaboration_mode># Plan Mode (Conversational)\nReturn plans only.</collaboration_mode>",
					},
				},
			},
		},
	}
	assert.Equal(t, "plan", extractResponsesRequestMode(req))
}

func TestExtractResponsesRequestMode_DoesNotInferPlanFromDefaultCollaborationModeText(t *testing.T) {
	req := map[string]any{
		"instructions": "<collaboration_mode># Collaboration Mode: Default\nKnown mode names are Default and Plan.</collaboration_mode>",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Please continue in default mode.",
					},
				},
			},
		},
	}
	assert.Equal(t, "", extractResponsesRequestMode(req))
}

func TestExtractResponsesRequestMode_LatestCollaborationModeBlockWins(t *testing.T) {
	req := map[string]any{
		"instructions": "You are a coding agent.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<collaboration_mode># Plan Mode (Conversational)\nReturn plans only.</collaboration_mode>",
					},
				},
			},
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<collaboration_mode># Collaboration Mode: Default\nYou are now in Default mode.</collaboration_mode>",
					},
				},
			},
		},
	}

	assert.Equal(t, "", extractResponsesRequestMode(req))
	assert.False(t, requestLooksLikePlanMode(req))
}

func TestEnforcePlanModeResponse_WrapsMissingProposedPlanTag(t *testing.T) {
	body := []byte(`{
		"id":"resp_test",
		"status":"requires_action",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"# Build Plan\n1. Scope\n2. Test"}]
			},
			{
				"id":"call_1",
				"type":"function_call",
				"name":"shell",
				"call_id":"call_1",
				"arguments":"{\"command\":\"pwd\"}"
			}
		],
		"output_text":"# Build Plan\n1. Scope\n2. Test"
	}`)

	out := enforcePlanModeResponse(body, true)
	assert.True(t, gjson.GetBytes(out, "output.0.content.0.text").Exists())
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "<proposed_plan>")
	assert.Contains(t, text, "</proposed_plan>")
	assert.Equal(t, "completed", gjson.GetBytes(out, "status").String())
	assert.Equal(t, 1, len(gjson.GetBytes(out, "output").Array()))
}

func TestRequestMapContainsAnyToolOutput(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{"type": "message"},
			map[string]any{"type": "function_call_output"},
		},
	}
	assert.True(t, requestMapContainsAnyToolOutput(req))

	req2 := map[string]any{
		"input": []any{
			map[string]any{"type": "message"},
			map[string]any{"type": "function_call"},
		},
	}
	assert.False(t, requestMapContainsAnyToolOutput(req2))
}

func TestTranslateChatCompletionToResponsesResponse_KeepsAssistantTextWithToolCall(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "I will run the tool now",
	      "tool_calls": [{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"shell_command","arguments":"{\"command\":\"pwd\"}"}
	      }]
	    },
	    "finish_reason": "tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	msg := output[0].(map[string]any)
	assert.Equal(t, "message", msg["type"])
	assert.Equal(t, "I will run the tool now", gjson.GetBytes(out, "output.0.content.0.text").String())

	call := output[1].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "I will run the tool now", strings.TrimSpace(resp["output_text"].(string)))
}

func TestTranslateChatCompletionToResponsesResponse_ReasoningStaysOutOfOutputText(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "final answer",
	      "reasoning_content": "thinking chain"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "final answer", gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, "final answer", gjson.GetBytes(out, "output.1.content.0.text").String())
	assert.Equal(t, "reasoning", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "thinking chain", gjson.GetBytes(out, "output.0.summary.0.text").String())
	assert.NotContains(t, gjson.GetBytes(out, "output_text").String(), "<think>")
}

func TestTranslateChatCompletionToResponsesResponse_StripsThinkFromMergedContent(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-merged",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "<think>internal notes</think>\nfinal answer"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "final answer", gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, "final answer", gjson.GetBytes(out, "output.1.content.0.text").String())
	assert.Equal(t, "reasoning", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "internal notes", gjson.GetBytes(out, "output.0.summary.0.text").String())
	assert.NotContains(t, gjson.GetBytes(out, "output_text").String(), "<think>")
}

func TestResponsesRequestToChatMessages_ReasoningItemUsesDedicatedField(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "reasoning",
				"summary": []any{
					map[string]any{
						"type": "summary_text",
						"text": "internal reasoning",
					},
				},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	require.Len(t, messages, 1)
	assert.Equal(t, "assistant", messages[0]["role"])
	assert.Equal(t, "", messages[0]["content"])
	assert.Equal(t, "internal reasoning", messages[0]["reasoning_content"])
}

func TestResponsesRequestToChatMessages_StripsLegacyThinkFromAssistantContent(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "<think>internal chain</think>\nfinal answer",
					},
				},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	require.Len(t, messages, 1)
	assert.Equal(t, "assistant", messages[0]["role"])
	assert.Equal(t, "final answer", messages[0]["content"])
	assert.Equal(t, "internal chain", messages[0]["reasoning_content"])
}

func TestNormalizeTranslatedResponsesOutput_AddsToolMetadataForSpecializedToolCalls(t *testing.T) {
	resp := map[string]any{
		"id": "resp_test",
		"output": []any{
			map[string]any{
				"type":    "shell_call",
				"call_id": "call_shell",
				"action": map[string]any{
					"command": "pwd",
				},
			},
		},
		"output_text": "stale planning text",
	}

	normalizeTranslatedResponsesOutput(resp)
	output := resp["output"].([]any)
	call := output[0].(map[string]any)
	assert.Equal(t, "shell", call["name"])
	assert.Equal(t, `{"command":"pwd","commands":["pwd"]}`, call["arguments"])
	assert.Equal(t, "", strings.TrimSpace(resp["output_text"].(string)))
}

func TestNormalizeTranslatedResponsesOutput_NormalizesShellCommandAndCommands(t *testing.T) {
	resp := map[string]any{
		"id": "resp_test",
		"output": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell",
				"arguments": `{"commands":["pwd","ls -la"]}`,
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output := resp["output"].([]any)
	call := output[0].(map[string]any)
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	commands, ok := args["commands"].([]any)
	require.True(t, ok)
	require.Len(t, commands, 2)
	assert.Equal(t, "pwd", fmt.Sprintf("%v", commands[0]))
}

func TestNormalizeTranslatedResponsesOutput_NormalizesReasoningItem(t *testing.T) {
	resp := map[string]any{
		"id": "resp_test",
		"output": []any{
			map[string]any{
				"type": "reasoning",
				"summary": []any{
					map[string]any{
						"type": "summary_text",
						"text": "  keep me  ",
					},
				},
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output := resp["output"].([]any)
	require.Len(t, output, 1)
	item := output[0].(map[string]any)
	assert.Equal(t, "reasoning", item["type"])
	assert.Equal(t, "keep me", gjson.Get(mustJSONString(item), "summary.0.text").String())
	assert.NotEmpty(t, strings.TrimSpace(fmt.Sprintf("%v", item["id"])))
	assert.Equal(t, "", strings.TrimSpace(fmt.Sprintf("%v", resp["output_text"])))
}

func TestTranslateChatCompletionToResponsesResponse_UnpacksMultiToolUseParallel(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-test",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "tool_calls": [{
	        "id":"call_parallel",
	        "type":"function",
	        "function":{
	          "name":"multi_tool_use.parallel",
	          "arguments":"{\"tool_uses\":[{\"recipient_name\":\"functions.shell_command\",\"parameters\":{\"command\":\"pwd\"}},{\"recipient_name\":\"functions.update_plan\",\"parameters\":{\"plan\":[{\"step\":\"Check logs\",\"status\":\"in_progress\"}]}}]}"
	        }
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)
	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	first := output[0].(map[string]any)
	assert.Equal(t, "function_call", first["type"])
	assert.Equal(t, "shell", first["name"])
	firstArgs := parseToolArgsMapString(fmt.Sprintf("%v", first["arguments"]))
	assert.Equal(t, "pwd", firstArgs["command"])

	second := output[1].(map[string]any)
	assert.Equal(t, "function_call", second["type"])
	assert.Equal(t, "update_plan", second["name"])
}

func TestTranslateChatCompletionToResponsesResponse_MapsUsageTokenDetails(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-usage",
	  "model":"qwen-test",
	  "choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],
	  "usage":{
	    "prompt_tokens":42,
	    "completion_tokens":18,
	    "total_tokens":60,
	    "completion_tokens_details":{"reasoning_tokens":11},
	    "prompt_tokens_details":{"cached_tokens":7}
	  }
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)
	assert.Equal(t, int64(11), gjson.GetBytes(out, "usage.output_tokens_details.reasoning_tokens").Int())
	assert.Equal(t, int64(7), gjson.GetBytes(out, "usage.input_tokens_details.cached_tokens").Int())
}

func TestSanitizeResponsesInputToolArguments_PrefixRuleInjectsSystemMessage(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"arguments": `{"command":"pwd","prefix_rule":"OK:","justification":"why"}`,
			},
		},
	}

	changed := sanitizeResponsesInputToolArguments(req)
	require.True(t, changed)
	input := req["input"].([]any)
	require.GreaterOrEqual(t, len(input), 2)
	sysMsg := input[0].(map[string]any)
	assert.Equal(t, "message", sysMsg["type"])
	assert.Equal(t, "system", sysMsg["role"])
	assert.Contains(t, extractResponsesInputText(sysMsg["content"]), "Safety constraint:")
	call := input[1].(map[string]any)
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	_, hasPrefixRule := args["prefix_rule"]
	_, hasJustification := args["justification"]
	assert.False(t, hasPrefixRule)
	assert.False(t, hasJustification)
}

func TestBodyLooksLikeArchitectureUnsupported_DetectsDeltaSignals(t *testing.T) {
	assert.True(t, bodyLooksLikeArchitectureUnsupported([]byte("unknown arch: gated_delta")))
	assert.True(t, bodyLooksLikeArchitectureUnsupported([]byte("not implemented: delta op")))
	assert.False(t, bodyLooksLikeArchitectureUnsupported([]byte("generic timeout")))
}

func TestTranslateChatCompletionToResponsesResponse_ToolOnlyTurnIsInProgress(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-tool-only",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "in_progress", gjson.GetBytes(out, "status").String())
	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "apply_patch", gjson.GetBytes(out, "output.0.name").String())
}

func TestRequestInputMentionsApplyPatch_IgnoresDeveloperOnlyMentions(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Always use apply_patch for edits."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "hi"},
				},
			},
		},
	}
	assert.False(t, requestInputMentionsApplyPatch(req))
}

func TestRequestInputMentionsApplyPatch_UsesLatestUserMessage(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "hello"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "please apply_patch this file"},
				},
			},
		},
	}
	assert.True(t, requestInputMentionsApplyPatch(req))
}

func TestRequestInputMentionsApplyPatch_RecognizesStrictToolCallPhrase(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "retry with a strikt tool call"},
				},
			},
		},
	}
	assert.True(t, requestInputMentionsApplyPatch(req))
}

func TestRequestInputMentionsApplyPatch_DoesNotTreatGenericEditTextAsStrictApplyPatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "please edit file README.md and verify with shell first"},
				},
			},
		},
	}
	assert.False(t, requestInputMentionsApplyPatch(req))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_UsesToolOutputPath(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "make an apply_patch change after checking the file first"},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_1",
				"output":  "\\\\wsl$\\Ubuntu\\home\\admmin\\llama-swap\\cline_test.md",
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, `\\wsl$\Ubuntu\home\admmin\llama-swap\cline_test.md`, extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_IgnoresVersionLikeFragments(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":   "function_call_output",
				"output": "version 0.4 loaded successfully",
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestCanonicalizeLocalApplyPatchPath_MapsWindowsQwenTestPath(t *testing.T) {
	got, err := canonicalizeLocalApplyPatchPathForHost(`C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, "linux")
	require.NoError(t, err)
	assert.Equal(t, "/mnt/c/Users/YLAB-Partner/Downloads/qwentest/cline_test.md", got)
}

func TestCanonicalizeLocalApplyPatchPathForHost_WindowsAbsoluteAllowed(t *testing.T) {
	got, err := canonicalizeLocalApplyPatchPathForHost(`C:\Users\YLAB-Partner\.codex\config.toml`, "windows")
	require.NoError(t, err)
	assert.Equal(t, "C:/Users/YLAB-Partner/.codex/config.toml", filepath.ToSlash(got))
}

func TestSelectApplyPatchOperation_RecoversStringifiedInputEnvelope(t *testing.T) {
	args := map[string]any{
		"input": `{"operation":{"type":"update_file","path":"C:\\Users\\YLAB-Partner\\Downloads\\qwentest\\cline_test.md","content":"WINDOWS_LOOP_OK"}}`,
	}

	opAny := selectApplyPatchOperation(args)
	require.True(t, applyPatchOperationPayloadValid(opAny))

	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", op["type"]))))
	assert.Equal(t, `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, strings.TrimSpace(fmt.Sprintf("%v", op["path"])))
	assert.Equal(t, "WINDOWS_LOOP_OK", strings.TrimSpace(fmt.Sprintf("%v", op["content"])))
}

func TestApplyPatchOperationPayloadValid_RejectsNilSentinels(t *testing.T) {
	assert.False(t, applyPatchOperationPayloadValid(map[string]any{
		"type": "update_file",
		"path": `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`,
	}))

	assert.False(t, applyPatchOperationPayloadValid(map[string]any{
		"type":    "update_file",
		"path":    `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`,
		"content": nil,
	}))

	assert.False(t, applyPatchOperationPayloadValid(map[string]any{
		"type":    "update_file",
		"path":    `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`,
		"content": "<nil>",
	}))

	assert.False(t, applyPatchOperationPayloadValid(map[string]any{
		"type": "update_file",
		"path": nil,
		"diff": "@@ -1 +1 @@\n-old\n+new\n",
	}))

	assert.False(t, applyPatchOperationPayloadValid(map[string]any{
		"type": "update_file",
		"path": `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`,
		"diff": "WSL_LOOP_OK\n\nWINDOWS_LOOP_OK\n+WSL_LOOP_POSTFIX\n",
	}))

	assert.True(t, applyPatchOperationPayloadValid(map[string]any{
		"type": "update_file",
		"path": `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`,
		"diff": "@@ -1,1 +1,2 @@\n line1\n+line2\n",
	}))
}

func TestExecuteApplyPatchOperationLocallyWithDisplay_UsesPreferredDisplayPath(t *testing.T) {
	op := map[string]any{
		"type": "delete_file",
		"path": filepath.Join(t.TempDir(), "does-not-exist.txt"),
	}

	summary, err := executeApplyPatchOperationLocallyWithDisplay(op, `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`)
	require.NoError(t, err)
	assert.Equal(t, `deleted C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, summary)
}

func TestBuildSyntheticApplyPatchCompletedResponse_PreservesApplyPatchCall(t *testing.T) {
	baseResp := []byte(`{
		"id":"resp_apply_patch",
		"object":"response",
		"created_at":1776903081,
		"model":"gpt-5.3-codex",
		"status":"in_progress",
		"output":[
			{
				"id":"apc_1",
				"type":"apply_patch_call",
				"call_id":"call_1",
				"operation":{"type":"update_file","path":"C:\\Users\\YLAB-Partner\\Downloads\\qwentest\\cline_test.md","content":"PATCH_OK"}
			}
		]
	}`)

	call := findFirstApplyPatchCallItem(baseResp)
	require.NotNil(t, call)

	got := buildSyntheticApplyPatchCompletedResponse(baseResp, call, "call_1", `update_file C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`)
	assert.Equal(t, "resp_apply_patch", gjson.GetBytes(got, "id").String())
	assert.Equal(t, "gpt-5.3-codex", gjson.GetBytes(got, "model").String())
	assert.Equal(t, "completed", gjson.GetBytes(got, "status").String())
	assert.Equal(t, "apply_patch_call", gjson.GetBytes(got, "output.0.type").String())
	assert.Equal(t, "call_1", gjson.GetBytes(got, "output.0.call_id").String())
	assert.Equal(t, "completed", gjson.GetBytes(got, "output.0.status").String())
	assert.Equal(t, "update_file", gjson.GetBytes(got, "output.0.operation.type").String())
	assert.Equal(t, `C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, gjson.GetBytes(got, "output.0.operation.path").String())
	assert.Equal(t, "PATCH_OK", gjson.GetBytes(got, "output.0.operation.content").String())
	assert.Equal(t, "apply_patch_call_output", gjson.GetBytes(got, "output.1.type").String())
	assert.Equal(t, "call_1", gjson.GetBytes(got, "output.1.call_id").String())
	assert.Equal(t, `update_file C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, gjson.GetBytes(got, "output.1.output").String())
	assert.Equal(t, "message", gjson.GetBytes(got, "output.2.type").String())
	assert.Equal(t, "assistant", gjson.GetBytes(got, "output.2.role").String())
	assert.Equal(t, `apply_patch completed via bridge local fallback: update_file C:\Users\YLAB-Partner\Downloads\qwentest\cline_test.md`, gjson.GetBytes(got, "output.2.content.0.text").String())
}

func TestBuildResponsesBridgeHandler_PreservesApplyPatchToolPhaseForClientContinuation(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":"Use apply_patch to update README.md.",
		"stream":false,
		"tools":[{"type":"apply_patch"}]
	}`)

	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-tool-only",
			"model":"qwen-test",
			"choices":[{
				"message":{
					"role":"assistant",
					"tool_calls":[{
						"id":"call_1",
						"type":"function",
						"function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}
					}]
				},
				"finish_reason":"tool_calls"
			}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "in_progress", gjson.GetBytes(rec.Body.Bytes(), "status").String())
	assert.Equal(t, "function_call", gjson.GetBytes(rec.Body.Bytes(), "output.0.type").String())
	assert.False(t, strings.Contains(rec.Body.String(), "apply_patch completed via bridge local fallback"))
}

func TestWriteResponsesStream_EmitsApplyPatchCallBeforeFallbackCompletion(t *testing.T) {
	responseJSON := buildSyntheticApplyPatchCompletedResponse([]byte(`{
		"id":"resp_apply_patch",
		"object":"response",
		"created_at":1776903081,
		"model":"gpt-5.3-codex",
		"status":"in_progress",
		"output":[
			{
				"id":"apc_1",
				"type":"apply_patch_call",
				"call_id":"call_1",
				"operation":{"type":"update_file","path":"README.md","content":"PATCH_OK"}
			}
		]
	}`), map[string]any{
		"id":      "apc_1",
		"type":    "apply_patch_call",
		"call_id": "call_1",
		"operation": map[string]any{
			"type":    "update_file",
			"path":    "README.md",
			"content": "PATCH_OK",
		},
	}, "call_1", "update_file README.md")

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON)
	body := rec.Body.String()

	assert.Contains(t, body, `"type":"apply_patch_call"`)
	assert.Contains(t, body, `event: response.function_call_arguments.done`)
	assert.Contains(t, body, `"name":"apply_patch"`)
	assert.Contains(t, body, `"type":"apply_patch_call_output"`)
	assert.Contains(t, body, `apply_patch completed via bridge local fallback: update_file README.md`)
}

func TestValidateBridgeToolCallItem_ApplyPatchAcceptsTopLevelInput(t *testing.T) {
	ok, warning := validateBridgeToolCallItem(map[string]any{
		"type":  "apply_patch_call",
		"input": "*** Begin Patch\n*** Add File: ok.txt\n+ok\n*** End Patch\n",
	})
	assert.True(t, ok)
	assert.Equal(t, "", warning)
}

func TestValidateBridgeToolCallItem_ApplyPatchRejectsIncompletePatchFragment(t *testing.T) {
	ok, warning := validateBridgeToolCallItem(map[string]any{
		"type":  "apply_patch_call",
		"input": "*** End Patch",
	})
	assert.False(t, ok)
	assert.Contains(t, warning, "apply_patch call was not executed")
}

func TestBuildResponsesBridgeHandler_RetriesTransientUpstreamFailures(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{"model":"gpt-5.3-codex","input":"hello","stream":false}`)

	var attempts atomic.Int32
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		n := attempts.Add(1)
		if n < 3 {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":"upstream unavailable"}`))
			return nil
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-ok",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, int32(3), attempts.Load())
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"status":"completed"`)
}

func TestBuildResponsesBridgeHandler_DoesNotSynthesizeToolCompletionOnUpstream502(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"input":[
			{"type":"function_call_output","call_id":"call_1","output":"{\"ok\":true}"}
		]
	}`)

	var attempts atomic.Int32
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		attempts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":"upstream unavailable"}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, int32(3), attempts.Load())
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	assert.Contains(t, rec.Body.String(), "upstream unavailable")
	assert.NotContains(t, rec.Body.String(), "A prior tool step appears to have completed")
}

func TestBuildResponsesBridgeHandler_RetriesOnInvalidToolArgsInSameCycle(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{"model":"gpt-5.3-codex","input":"run patch","stream":false}`)

	var attempts atomic.Int32
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		n := attempts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if n == 1 {
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-first",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{}}"}}]},"finish_reason":"tool_calls"}]
			}`))
			return nil
		}
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-second",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"retry succeeded with concrete args"},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, int32(2), attempts.Load())
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "retry succeeded with concrete args")
	assert.NotContains(t, rec.Body.String(), "call was not executed because arguments were empty")
}

func TestBuildGatewayRetryHandler_Retries502ForNonStreamChat(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{"model":"gpt-5.3-codex","messages":[{"role":"user","content":"hello"}],"stream":false}`)
	var attempts atomic.Int32
	base := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		n := attempts.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":"upstream timeout"}`))
			return nil
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
		return nil
	}

	handler := pm.buildGatewayRetryHandler("gpt-5.3-codex", body, base)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)
	assert.Equal(t, int32(2), attempts.Load())
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"content":"ok"`)
}

func TestWriteResponsesStream_CompletesForToolPhase(t *testing.T) {
	responseJSON := []byte(`{
		"id":"resp_test",
		"object":"response",
		"created_at":1776794896,
		"status":"completed",
		"model":"qwen-test",
		"output":[
			{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Proceeding with tool execution."}]},
			{"id":"fc_call_1","type":"function_call","name":"shell_command","call_id":"call_1","arguments":"{\"command\":\"pwd\"}","status":"completed"}
		]
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON)
	body := rec.Body.String()
	assert.Contains(t, body, "event: response.output_item.added")
	assert.Contains(t, body, "event: response.completed")
	assert.Contains(t, body, "data: [DONE]")
}

func TestBuildResponsesBridgeHandler_KeepsMixedMessageAndToolCallNonTerminal(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{"model":"gpt-5.3-codex","input":"run check","stream":false,"tools":[{"type":"function","name":"shell_command"}]}`)

	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-mixed",
			"model":"qwen-test",
			"choices":[{"message":{
				"role":"assistant",
				"content":"Proceeding with systematic tool testing.",
				"tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell_command","arguments":"{\"command\":\"pwd\"}"}}]
			},"finish_reason":"tool_calls"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"status":"completed"`)
	assert.Contains(t, rec.Body.String(), `"type":"function_call"`)
	assert.Contains(t, rec.Body.String(), `"Proceeding with systematic tool testing."`)
}

func TestBuildResponsesBridgeHandler_RecoversEmptyPostToolOutput(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"input":[
			{"type":"function_call_output","call_id":"call_1","output":"{\"cwd\":\"/tmp\"}"}
		]
	}`)

	var attempts atomic.Int32
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		n := attempts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if n == 1 {
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-empty",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}]
			}`))
			return nil
		}
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-recovered",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"Tool result analyzed. Final answer."},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, int32(2), attempts.Load())
	assert.Contains(t, rec.Body.String(), "Tool result analyzed. Final answer.")
	assert.Contains(t, rec.Body.String(), `"status":"completed"`)
}

func TestTranslateResponsesToChatCompletionsRequest_PlanModeDisablesToolsAndKeepsStream(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"mode":"plan",
		"stream":true,
		"tools":[{"type":"apply_patch"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a chat app"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(out, "stream").Bool())
	assert.Equal(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.False(t, gjson.GetBytes(out, "tools").Exists())
	assert.Contains(t, string(out), "Planning mode is active")
	assert.Contains(t, string(out), "Build a chat app")
}

func TestTranslateResponsesToChatCompletionsRequest_CodexManagedPlanModeLeavesProxyEnforcementOff(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"mode":"plan",
		"stream":true,
		"tools":[{"type":"apply_patch"}],
		"input":[
			{"type":"message","role":"system","content":[{"type":"input_text","text":"<collaborationmode>Plan Mode\nConversational work in 3 phases...</collaborationmode>"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Plan the migration"}]}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(out, "stream").Bool())
	assert.True(t, gjson.GetBytes(out, "tools").Exists())
	assert.NotContains(t, string(out), "Planning mode is active. Do NOT execute tasks")
}

func TestTranslateResponsesToChatCompletionsRequest_SlashModeAndReasoningCommands(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"tools":[{"type":"apply_patch"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"/mode plan\n/reasoning high\nCreate a REST API in FastAPI"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.False(t, gjson.GetBytes(out, "temperature").Exists())
	assert.False(t, gjson.GetBytes(out, "top_p").Exists())
	assert.Equal(t, true, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget").Exists())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget_message").Exists())
	assert.NotContains(t, string(out), "/mode plan")
	assert.NotContains(t, string(out), "/reasoning high")
	assert.Contains(t, string(out), "Create a REST API in FastAPI")
}

func TestTranslateResponsesToChatCompletionsRequest_ReasoningEffortMapsReasoningControls(t *testing.T) {
	lowBody := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"low","summary":"auto"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain transformers"}]}]
	}`)

	lowOut, err := translateResponsesToChatCompletionsRequest(lowBody)
	require.NoError(t, err)

	assert.Equal(t, false, gjson.GetBytes(lowOut, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(lowOut, "reasoning_budget").Exists())
	assert.False(t, gjson.GetBytes(lowOut, "reasoning_budget_message").Exists())
	assert.False(t, gjson.GetBytes(lowOut, "logit_bias").Exists())
	assert.False(t, gjson.GetBytes(lowOut, "grammar").Exists())

	mediumBody := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"medium","summary":"auto"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain transformers"}]}]
	}`)

	mediumOut, err := translateResponsesToChatCompletionsRequest(mediumBody)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(mediumOut, "chat_template_kwargs.enable_thinking").Bool())
	assert.InEpsilon(t, 11.8, gjson.GetBytes(mediumOut, "logit_bias.248069").Float(), 0.0001)
	assert.Contains(t, gjson.GetBytes(mediumOut, "grammar").String(), "<[248069]>")

	highBody := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"high","summary":"auto"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain transformers"}]}]
	}`)

	highOut, err := translateResponsesToChatCompletionsRequest(highBody)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(highOut, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(highOut, "logit_bias").Exists())
	assert.False(t, gjson.GetBytes(highOut, "grammar").Exists())

	extraHighBody := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"xhigh","summary":"auto"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain transformers"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(extraHighBody)
	require.NoError(t, err)

	assert.False(t, gjson.GetBytes(out, "temperature").Exists())
	assert.False(t, gjson.GetBytes(out, "top_p").Exists())
	assert.Equal(t, true, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget").Exists())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget_message").Exists())
	assert.False(t, gjson.GetBytes(out, "logit_bias").Exists())
	assert.False(t, gjson.GetBytes(out, "grammar").Exists())
}

func TestTranslateResponsesToChatCompletionsRequest_ReasoningRespectsExplicitSampling(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"temperature":0.11,
		"top_p":0.33,
		"reasoning":{"effort":"low"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain attention"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.InEpsilon(t, 0.11, gjson.GetBytes(out, "temperature").Float(), 0.0001)
	assert.InEpsilon(t, 0.33, gjson.GetBytes(out, "top_p").Float(), 0.0001)
	assert.Equal(t, false, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget").Exists())
	assert.False(t, gjson.GetBytes(out, "reasoning_budget_message").Exists())
}

func TestTranslateResponsesToChatCompletionsRequest_PreservesLogitBiasAndGrammar(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"logit_bias":{"248069":12.5},
		"grammar":"root ::= pre <[248069]> post\npre ::= !<[248069]>*\npost ::= !<[248069]>*",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello world"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, 12.5, gjson.GetBytes(out, "logit_bias.248069").Float())
	assert.Contains(t, gjson.GetBytes(out, "grammar").String(), "<[248069]>")
}

func TestTranslateResponsesToChatCompletionsRequest_MediumReasoningRespectsExplicitBiasAndGrammar(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"medium","summary":"auto"},
		"logit_bias":{"248069":12.5},
		"grammar":"root ::= custom",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello world"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Bool())
	assert.Equal(t, 12.5, gjson.GetBytes(out, "logit_bias.248069").Float())
	assert.Equal(t, "root ::= custom", gjson.GetBytes(out, "grammar").String())
}

func TestTranslateResponsesToChatCompletionsRequest_ToolsStripGrammarConstraints(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"effort":"medium","summary":"auto"},
		"tools":[{"type":"function","name":"shell_command","parameters":{"type":"object"}}],
		"grammar":"root ::= custom",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"run diagnostics"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.True(t, gjson.GetBytes(out, "tools").Exists())
	assert.False(t, gjson.GetBytes(out, "grammar").Exists())
	assert.Equal(t, true, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Bool())
}

func TestNormalizeResponsesRequest_AcceptsApplyPatchAliases(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"tools":[
			{"type":"applypatch"},
			{"type":"custom","name":"applypatch"},
			{"type":"apply_patch"}
		],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"patch file"}]}]
	}`)

	out, adapted, unsupported, err := normalizeResponsesRequest(body)
	require.NoError(t, err)
	require.Empty(t, unsupported)
	assert.Contains(t, adapted, "apply_patch")
	tools := gjson.GetBytes(out, "tools").Array()
	require.Len(t, tools, 3)
	for _, tool := range tools {
		assert.Equal(t, "function", tool.Get("type").String())
		assert.Equal(t, "__llamaswap_apply_patch", tool.Get("name").String())
	}
}

func TestNormalizeResponsesInputItem_AcceptsApplyPatchCallAlias(t *testing.T) {
	item := map[string]any{
		"type":    "applypatch_call",
		"call_id": "call_1",
		"operation": map[string]any{
			"type": "update_file",
			"path": "README.md",
			"diff": "@@\n-a\n+b",
		},
	}

	mapped, changed := normalizeResponsesInputItem(item)
	require.True(t, changed)
	m, ok := mapped.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call", m["type"])
	assert.Equal(t, "__llamaswap_apply_patch", m["name"])
}

func TestTranslateResponsesToChatCompletionsRequest_NoReasoningEffortLeavesThinkingDefault(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain attention"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.False(t, gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Exists())
	assert.NotContains(t, string(out), "Reasoning style:")
}

func TestShouldUseNativeResponsesBridgeStream_DisabledForPlanMode(t *testing.T) {
	reqTopLevel := map[string]any{
		"mode": "plan",
	}
	assert.False(t, shouldUseNativeResponsesBridgeStream(reqTopLevel))

	reqSlash := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "/mode plan\nBuild a chat app",
					},
				},
			},
		},
	}
	assert.False(t, shouldUseNativeResponsesBridgeStream(reqSlash))
}

func TestShouldUseNativeResponsesBridgeStream_DisabledWhenToolsPresent(t *testing.T) {
	req := map[string]any{
		"stream": true,
		"tools": []any{
			map[string]any{
				"type": "function",
				"name": "shell_command",
			},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Check python and continue",
					},
				},
			},
		},
	}
	assert.False(t, shouldUseNativeResponsesBridgeStream(req))
}

func TestEnforcePlanModeResponse_RewritesExecutionAndDropsToolCalls(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_bad",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Built ChatFlow and saved to /tmp/chat-app.html. Open the file to use it."}]
			},
			{
				"id":"fc_1",
				"type":"function_call",
				"name":"apply_patch",
				"call_id":"call_1",
				"arguments":"{\"operation\":{\"type\":\"create_file\"}}"
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	assert.Contains(t, gjson.GetBytes(out, "output.0.content.0.text").String(), "Planning mode is active")
	assert.False(t, gjson.GetBytes(out, "output.1").Exists())
	assert.Equal(t, "completed", gjson.GetBytes(out, "status").String())
}

func TestEnforcePlanModeResponse_PreservesValidPlanText(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_ok",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"1. Define requirements and constraints.\n2. Design architecture and interfaces.\n3. Implement in milestones.\n4. Validate with tests.\n5. Review risks and rollout."}]
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "1. Define requirements and constraints.")
	assert.NotContains(t, text, "Planning mode is active. Here is a structured plan only:")
}

func TestEnforcePlanModeResponse_PreservesPlanTextWithDiffPatchWords(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_patch_words",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"1. Review the diff to scope changes.\n2. Plan the patch sequencing by module.\n3. Validate rollout and rollback steps."}]
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "Review the diff")
	assert.Contains(t, text, "Plan the patch sequencing")
	assert.NotContains(t, text, "Planning mode is active. Here is a structured plan only:")
}

func TestEnforcePlanModeResponse_RewritesPhaseOneExplorationWithShellTags(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_phase1_shell",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Let me first check if there is existing project context.\n<shell_commands><commands>[{\"command\":\"Get-ChildItem -Force\"}]</commands></shell_commands>"}]
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "Planning mode is active. Here is a structured plan only:")
}

func TestIsCodexManagedPlanMode_DetectsCollaborationModeTag(t *testing.T) {
	msgs := []map[string]any{
		{"role": "system", "content": "...<collaborationmode>Plan Mode\nConversational work in 3 phases...</collaborationmode>..."},
	}
	assert.True(t, isCodexManagedPlanMode(msgs))
}

func TestIsCodexManagedPlanMode_IgnoresUserMessage(t *testing.T) {
	msgs := []map[string]any{
		{"role": "user", "content": "<collaborationmode>Plan Mode</collaborationmode>"},
	}
	assert.False(t, isCodexManagedPlanMode(msgs))
}

func TestIsCodexManagedPlanMode_FalseForPlainSystemPrompt(t *testing.T) {
	msgs := []map[string]any{
		{"role": "system", "content": "You are a helpful assistant."},
	}
	assert.False(t, isCodexManagedPlanMode(msgs))
}

func TestIsCodexManagedPlanMode_RequiresBothTagAndPlanMode(t *testing.T) {
	msgs := []map[string]any{
		{"role": "system", "content": "<collaborationmode>Default</collaborationmode>"},
	}
	assert.False(t, isCodexManagedPlanMode(msgs))
}

func TestIsCodexManagedPlanMode_UsesLatestCollaborationModeBlock(t *testing.T) {
	msgs := []map[string]any{
		{"role": "developer", "content": "<collaboration_mode># Plan Mode (Conversational)\nReturn plans only.</collaboration_mode>"},
		{"role": "developer", "content": "<collaboration_mode># Collaboration Mode: Default\nYou are now in Default mode.</collaboration_mode>"},
	}
	assert.False(t, isCodexManagedPlanMode(msgs))
}

func TestEnforcePlanModeResponse_DoesNotFallbackForEmptyWhenUpstreamNotNormal(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_empty_error",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":""}]
			}
		],
		"output_text":""
	}`)

	out := enforcePlanModeResponse(body, false)
	assert.Equal(t, "", gjson.GetBytes(out, "output.0.content.0.text").String())
	assert.NotContains(t, string(out), "Planning mode is active. Here is a structured plan only:")
}

func TestEnforcePlanModeResponse_EmitsLengthDiagnosticForEmptyLengthFinish(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_empty_length",
		"object":"response",
		"choices":[{"finish_reason":"length"}],
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":""}]
			}
		],
		"output_text":""
	}`)

	out := enforcePlanModeResponse(body, false)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, `finish_reason: "length"`)
	assert.Contains(t, text, "Retry with a higher output token limit")
	assert.Contains(t, text, "<proposed_plan>")
}

func TestBuildResponsesBridgeHandler_ForwardsNativeStreamWhenSafe(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain distributed systems"}]}]
	}`)

	var upstreamPath string
	var upstreamReqBody []byte
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		upstreamPath = r.URL.Path
		upstreamReqBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.2", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	req.Header.Set("Accept", "text/event-stream")
	rec := httptest.NewRecorder()
	err := handler("gpt-5.2", rec, req)
	require.NoError(t, err)

	assert.Equal(t, "/v1/chat/completions", upstreamPath)
	assert.Equal(t, true, gjson.GetBytes(upstreamReqBody, "stream").Bool())
	assert.Equal(t, "none", gjson.GetBytes(upstreamReqBody, "tool_choice").String())
	assert.False(t, gjson.GetBytes(upstreamReqBody, "tools").Exists())
	assert.Contains(t, rec.Header().Get("Content-Type"), "text/event-stream")
	assert.Contains(t, rec.Body.String(), "event: response.created")
	assert.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	assert.Contains(t, rec.Body.String(), "event: response.completed")
	assert.Contains(t, rec.Body.String(), "Hello")
	assert.NotContains(t, rec.Body.String(), "chat.completion.chunk")
}

func TestBuildResponsesBridgeHandler_StreamRetriesBeforeEmittingResponsesSSE(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":true,
		"tools":[{"type":"apply_patch"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"patch file now"}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":"upstream unavailable"}`))
			return nil
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Recovered stream\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	req.Header.Set("Accept", "text/event-stream")
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 2, attempts)
	assert.Contains(t, rec.Body.String(), "event: response.created")
	assert.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	assert.Contains(t, rec.Body.String(), "Recovered stream")
	assert.Contains(t, rec.Body.String(), "event: response.completed")
	assert.NotContains(t, rec.Body.String(), "upstream unavailable")
}

func TestWriteResponsesStreamFromChatSSE_EmitsReasoningAndContentOnSeparateLanes(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"thinking step 1"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"final answer"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.output_text.delta")
	assert.Contains(t, body, "event: response.completed")
	assert.Contains(t, body, "event: response.reasoning_summary_text.delta")
	assert.Contains(t, body, "event: response.reasoning_summary_text.done")
	assert.Contains(t, body, "final answer")
	assert.Contains(t, body, "thinking step 1")
}

func TestWriteResponsesStream_ReplaysReasoningItems(t *testing.T) {
	responseJSON := []byte(`{
	  "id":"resp_reasoning_stream",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"completed",
	  "output":[
	    {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"thinking one"}]},
	    {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer one"}]}
	  ],
	  "output_text":"answer one"
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.reasoning_summary_text.delta")
	assert.Contains(t, body, "event: response.reasoning_summary_text.done")
	assert.Contains(t, body, "thinking one")
	assert.Contains(t, body, "answer one")
}

func TestWriteResponsesStreamFromChatSSE_EmitsToolCallArgumentDeltas(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Running tool...","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"pwd\"}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.function_call_arguments.delta")
	assert.Contains(t, body, "event: response.function_call_arguments.done")
	assert.Contains(t, body, `"name":"shell"`)
	assert.Contains(t, body, `"status":"requires_action"`)
	assert.Contains(t, body, `"channel":"commentary"`)
}

func TestWriteResponsesStreamFromChatSSE_PlanModeWrapsOutputAsProposedPlan(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"1. Scope\n2. Risks\n3. Validation"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `\u003cproposed_plan\u003e`)
	assert.Contains(t, body, `\u003c/proposed_plan\u003e`)
	assert.Contains(t, body, "event: response.output_text.delta")
	assert.Contains(t, body, "event: response.completed")
}

func TestWriteResponsesStreamFromChatSSE_PlanModeLengthEmitsDiagnostic(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `finish_reason: \"length\"`)
	assert.Contains(t, body, "Retry with a higher output token limit")
	assert.Contains(t, body, "event: response.output_text.delta")
	assert.Contains(t, body, "event: response.completed")
}

func TestRequestLooksLikePlanMode_DetectsSlashAndFlagSignals(t *testing.T) {
	reqSlash := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "/mode plan\noutline migration"},
				},
			},
		},
	}
	assert.True(t, requestLooksLikePlanMode(reqSlash))

	reqFlag := map[string]any{
		"instructions": "codex run --plan",
		"input":        []any{},
	}
	assert.True(t, requestLooksLikePlanMode(reqFlag))
}
