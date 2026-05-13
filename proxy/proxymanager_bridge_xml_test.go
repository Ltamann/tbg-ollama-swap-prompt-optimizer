package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
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
	commands := normalizeShellCommandsValue(calls[0].Arguments["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "pwd", commands[0])
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
	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "pwd", commands[0])
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
	assert.Equal(t, "apply_patch_call", call["type"])
	operation, hasOperation := call["operation"].(map[string]any)
	assert.True(t, hasOperation)
	assert.NotNil(t, operation)
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
	require.GreaterOrEqual(t, len(output), 1)
	var operation map[string]any
	for _, raw := range output {
		call := raw.(map[string]any)
		if fmt.Sprintf("%v", call["type"]) != "apply_patch_call" {
			continue
		}
		op, ok := call["operation"].(map[string]any)
		if ok {
			operation = op
			break
		}
	}
	require.NotNil(t, operation)
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
	assert.True(t, shouldForceStrictApplyPatchRetry("wrong_tool_call"))
	assert.True(t, shouldForceStrictApplyPatchRetry(" Wrong_Tool_Call "))
	assert.True(t, shouldForceStrictApplyPatchRetry("wrong_tool_call_oversized_shell_write"))
	assert.True(t, shouldForceStrictApplyPatchRetry("no_tool_call"))
	assert.False(t, shouldForceStrictApplyPatchRetry("planning_only"))
	assert.True(t, shouldForceStrictApplyPatchRetry("empty_operation"))
	assert.True(t, shouldForceStrictApplyPatchRetry("invalid_diff"))
}

func TestBuildResponsesApplyPatchFunctionTool_UsesPreferredDescription(t *testing.T) {
	tool := buildResponsesApplyPatchFunctionTool()
	assert.Equal(t, applyPatchPreferredToolDescription, tool["description"])
}

func TestNormalizeBridgeChatTools_ApplyPatchDescriptionUsesPreferredWording(t *testing.T) {
	normalized := normalizeBridgeChatTools([]any{
		map[string]any{"type": "apply_patch"},
	})
	require.Len(t, normalized, 1)
	item, ok := normalized[0].(map[string]any)
	require.True(t, ok)
	fn, ok := item["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, applyPatchPreferredToolDescription, fn["description"])
}

func TestNormalizeBridgeChatTools_PreservesMCPTool(t *testing.T) {
	normalized := normalizeBridgeChatTools([]any{
		map[string]any{
			"type":         "mcp",
			"server_label": "docs",
			"server_url":   "https://example.test/mcp",
		},
	})
	require.Len(t, normalized, 1)
	item, ok := normalized[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "mcp", item["type"])
	assert.Equal(t, "docs", item["server_label"])
	assert.Equal(t, "https://example.test/mcp", item["server_url"])
}

func TestNormalizeResponsesToolsMap_PreservesMCPTool(t *testing.T) {
	data := map[string]any{
		"tools": []any{
			map[string]any{
				"type":         "mcp",
				"server_label": "docs",
				"server_url":   "https://example.test/mcp",
			},
		},
	}

	adapted, unsupported, changed := normalizeResponsesToolsMap(data)
	assert.Empty(t, adapted)
	assert.Empty(t, unsupported)
	assert.False(t, changed)
	assert.Equal(t, "mcp", gjson.Get(mustJSONString(data), "tools.0.type").String())
	assert.Equal(t, "docs", gjson.Get(mustJSONString(data), "tools.0.server_label").String())
}

func TestAppendApplyPatchTailConstraintToUserTurn_AppendsToLatestUserMessage(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "first"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "middle"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "last user text"},
				},
			},
		},
	}
	ok := appendApplyPatchTailConstraintToUserTurn(req, applyPatchTailConstraintText)
	require.True(t, ok)
	assert.Contains(t, extractResponsesInputText(req["input"]), applyPatchTailConstraintText)
}

func TestExtractExactFinalReplyHint(t *testing.T) {
	assert.Equal(t, "PORT8080_OK", extractExactFinalReplyHint("Reply exactly: PORT8080_OK"))
	assert.Equal(t, "PATCH27UD_DONE", extractExactFinalReplyHint("Use apply_patch to update file /tmp/x by appending one line: Y. Then reply PATCH27UD_DONE"))
	assert.Equal(t, "BUILD_DONE", extractExactFinalReplyHint("Then respond BUILD_DONE"))
	assert.Equal(t, "", extractExactFinalReplyHint("Please continue and explain what you did."))
}

func TestTranslateResponsesToChatCompletionsRequest_PostToolExactReplyHintAddsContinuationInstruction(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.3-codex",
		"tools": []any{
			map[string]any{"type": "apply_patch"},
		},
		"tool_choice": "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use apply_patch to update file /tmp/demo.txt by appending one line: PATCH27UD_OK. Then reply PATCH27UD_DONE"},
				},
			},
			map[string]any{
				"type":    "apply_patch_call_output",
				"call_id": "call_1",
				"output":  `{"ok":true,"summary":"updated /tmp/demo.txt"}`,
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var translated map[string]any
	require.NoError(t, json.Unmarshal(out, &translated))
	rawMessages, ok := translated["messages"].([]any)
	require.True(t, ok)
	joined := mustJSONString(rawMessages)
	assert.Contains(t, joined, `reply with exactly: \"PATCH27UD_DONE\"`)
	assert.Contains(t, joined, `if the task is already complete from prior tool results, do not call more tools`)
}

func TestEnforceExactFinalReplyHint_CollapsesFinalAssistantText(t *testing.T) {
	body := []byte(`{
	  "id":"resp_1",
	  "status":"completed",
	  "output":[
	    {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"File updated successfully. PATCH27UD_DONE"}]}
	  ],
	  "output_text":"File updated successfully. PATCH27UD_DONE"
	}`)

	got := enforceExactFinalReplyHint(body, "PATCH27UD_DONE", false)
	assert.Equal(t, "PATCH27UD_DONE", gjson.GetBytes(got, "output_text").String())
	assert.Equal(t, "PATCH27UD_DONE", gjson.GetBytes(got, "output.0.content.0.text").String())
}

func TestEnforceExactFinalReplyHint_DoesNotRewriteToolCallResponses(t *testing.T) {
	body := []byte(`{
	  "id":"resp_1",
	  "status":"requires_action",
	  "output":[
	    {"id":"fc_1","type":"apply_patch_call","call_id":"call_1","operation":{"type":"update_file","path":"x","content":"y"}}
	  ]
	}`)

	got := enforceExactFinalReplyHint(body, "PATCH27UD_DONE", false)
	assert.JSONEq(t, string(body), string(got))
}

func TestEnforceExactFinalReplyHint_ForcesCompletedWorkflowReply(t *testing.T) {
	body := []byte(`{
	  "id":"resp_1",
	  "status":"completed",
	  "output":[
	    {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Now verifying the final content."}]}
	  ],
	  "output_text":"Now verifying the final content."
	}`)

	got := enforceExactFinalReplyHint(body, "T11_SENTINEL", true)
	assert.Equal(t, "T11_SENTINEL", gjson.GetBytes(got, "output_text").String())
	assert.Equal(t, "T11_SENTINEL", gjson.GetBytes(got, "output.0.content.0.text").String())
}

func TestBuildApplyPatchRetryFailedResponse_UsesPreferredWording(t *testing.T) {
	baseResp := []byte(`{"id":"resp_1","model":"qwen-test"}`)
	got := buildApplyPatchRetryFailedResponse(baseResp, "wrong_tool_call", `{"commands":["Set-Content README.md hi"]}`)
	text := gjson.GetBytes(got, "output.0.content.0.text").String()
	assert.Contains(t, text, applyPatchRetryPreferredFailureText)
	assert.NotContains(t, text, "apply_patch retry failed")
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchWarningFieldUsesNeutralWording(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-warning",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":"{}"}}]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	warning := gjson.GetBytes(out, "output.0.bridgevalidationwarning").String()
	assert.Contains(t, warning, applyPatchValidationWarningPrefix)
	assert.NotContains(t, warning, "apply_patch call was not executed because")
}

func TestTranslateChatCompletionToResponsesResponse_ToolValidationMessagesAvoidToolBlamingText(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-tool-validation",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{"id":"call_1","type":"function","function":{"name":"","arguments":"{}"}}]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	output := gjson.GetBytes(out, "output").Array()
	require.NotEmpty(t, output)
	for _, item := range output {
		warning := item.Get("bridgevalidationwarning").String()
		assert.NotContains(t, warning, "apply_patch call was not executed because")
		itemID := item.Get("id").String()
		for _, part := range item.Get("content").Array() {
			text := part.Get("text").String()
			if strings.Contains(itemID, "_tool_validation_") {
				assert.NotContains(t, text, "apply_patch call was not executed because")
			}
		}
	}
}

func TestTranslateChatCompletionToResponsesResponse_EmptyShellArgumentsBecomeValidationMessage(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-empty-shell",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{}"}}]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	output := gjson.GetBytes(out, "output").Array()
	require.Len(t, output, 1)
	assert.Equal(t, "message", output[0].Get("type").String())
	text := output[0].Get("content.0.text").String()
	assert.Contains(t, text, shellValidationWarningPrefix)
	assert.Contains(t, text, "Provide a non-empty `command` string or `commands` array and retry.")
}

func TestResponsesRequestContractMode_DetectsPlanModeFromCollaborationInstruction(t *testing.T) {
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
	assert.Equal(t, "", extractResponsesRequestMode(req))
	assert.Equal(t, "plan", responsesRequestContractMode(req))
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
	assert.Equal(t, "default", responsesRequestContractMode(req))
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

func TestEnforcePlanModeResponse_RecoversPlanFromBlockedApplyPatchContent(t *testing.T) {
	body := []byte(`{
		"id":"resp_test",
		"status":"requires_action",
		"output":[
			{
				"id":"call_1",
				"type":"apply_patch_call",
				"call_id":"call_1",
				"operation":{
					"type":"create_file",
					"path":"c:\\Users\\YLAB-Partner\\Downloads\\qwentest\\plan.md",
					"content":"# Plan\n\n1. Scope\n2. Risks\n3. Validation\n\n</parameter></function> </tool_call> <tool_call>update_plan> <explanation>bad</explanation>"
				}
			}
		],
		"output_text":""
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "<proposed_plan>")
	assert.Contains(t, text, "1. Scope")
	assert.NotContains(t, text, "</parameter>")
	assert.NotContains(t, text, "<tool_call>")
	assert.Equal(t, "completed", gjson.GetBytes(out, "status").String())
	assert.NotContains(t, string(out), `"apply_patch_call"`)
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

func TestRequestContainsAnyToolOutput(t *testing.T) {
	assert.True(t, requestContainsAnyToolOutput([]byte(`{
		"input": [
			{"type":"message"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"}
		]
	}`)))

	assert.False(t, requestContainsAnyToolOutput([]byte(`{
		"input": [
			{"type":"message"},
			{"type":"function_call","name":"shell"}
		]
	}`)))
}

func TestRequestMapContainsApplyPatchToolOutput(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{"type": "function_call", "call_id": "call_1", "name": "apply_patch"},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "ok"},
		},
	}
	assert.True(t, requestMapContainsApplyPatchToolOutput(req))

	req2 := map[string]any{
		"input": []any{
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": `{"type":"apply_patch_call_output","output":"ok"}`},
		},
	}
	assert.True(t, requestMapContainsApplyPatchToolOutput(req2))

	req3 := map[string]any{
		"input": []any{
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "ok"},
		},
	}
	assert.False(t, requestMapContainsApplyPatchToolOutput(req3))
}

func TestRequestContainsApplyPatchToolOutput(t *testing.T) {
	assert.True(t, requestContainsApplyPatchToolOutput([]byte(`{
		"input": [
			{"type":"function_call","call_id":"call_1","name":"apply_patch"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"}
		]
	}`)))

	assert.True(t, requestContainsApplyPatchToolOutput([]byte(`{
		"input": [
			{"type":"function_call_output","call_id":"call_1","output":{"type":"apply_patch_call_output","output":"ok"}}
		]
	}`)))

	assert.False(t, requestContainsApplyPatchToolOutput([]byte(`{
		"input": [
			{"type":"function_call","call_id":"call_1","name":"shell"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"}
		]
	}`)))
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

func TestTranslateChatCompletionToResponsesResponse_PromotesReasoningWhenContentEmpty(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-only",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "only reasoning"
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

	assert.Equal(t, "only reasoning", gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, "message", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "only reasoning", gjson.GetBytes(out, "output.0.content.0.text").String())
}

func TestBuildReasoningCommentaryPreview_TruncatesAndStripsTags(t *testing.T) {
	input := "<think>\nThis is a long internal reasoning chain with extra whitespace. " +
		strings.Repeat("detail ", 60) + "\n</think>"

	got := buildReasoningCommentaryPreview(input)
	assert.NotContains(t, got, "<think>")
	assert.NotContains(t, got, "</think>")
	assert.Less(t, len(got), len(input))
	assert.Contains(t, got, "This is a long internal reasoning chain")
	assert.Contains(t, got, "...")
}

func TestTranslateChatCompletionToResponsesResponse_PromotesReasoningPreviewWhenContentEmptyLong(t *testing.T) {
	longReasoning := "I am thinking through the task carefully. " + strings.Repeat("step detail ", 40) + "final hidden thought"
	body := []byte(`{
	  "id": "chatcmpl-reasoning-only-long",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "` + longReasoning + `"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	preview := buildReasoningCommentaryPreview(longReasoning)
	assert.Equal(t, preview, gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, preview, gjson.GetBytes(out, "output.0.content.0.text").String())
	assert.NotContains(t, gjson.GetBytes(out, "output_text").String(), "final hidden thought")
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

func TestTranslateChatCompletionToResponsesResponse_StripsThinkingFromMergedContent(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-merged-thinking",
	  "model": "qwen-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "<thinking>internal notes</thinking>\nfinal answer"
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
	assert.NotContains(t, gjson.GetBytes(out, "output_text").String(), "<thinking>")
}

func TestStripLeadingReasoningDirective(t *testing.T) {
	assert.Equal(t, "PORT8080_OK", stripLeadingReasoningDirective("/no_think\nPORT8080_OK"))
	assert.Equal(t, "hello", stripLeadingReasoningDirective("/think\nhello"))
	assert.Equal(t, "PORT8080_OK", stripLeadingReasoningDirective("<thinking>hidden</thinking>\nPORT8080_OK"))
	assert.Equal(t, "PORT8080_OK", stripLeadingReasoningDirective("</think>\n\nPORT8080_OK"))
	assert.Equal(t, "", stripLeadingReasoningDirective("</think>\n\n</think>\n\n</think>"))
	assert.Equal(t, "PORT8080_OK", stripLeadingReasoningDirective("PORT8080_OK"))
}

func TestExtractContentAndReasoning_DropsPreThinkPreambleAndClosingTagLeak(t *testing.T) {
	raw := "Here's a plan for a small math game:\n\n<think>The user wants me to write a plan first.</think>\n\nHere's a plan for a small, fun math game:\n\n## Math Blitz"
	content, reasoning := extractContentAndReasoning(raw)
	assert.Equal(t, "The user wants me to write a plan first.", reasoning)
	assert.Equal(t, "Here's a plan for a small, fun math game:\n\n## Math Blitz", content)
	assert.NotContains(t, content, "</think>")
	assert.NotContains(t, content, "Here's a plan for a small math game:")
}

func TestTranslateChatCompletionToResponsesResponse_ParsesReasoningToolCallWithoutLeakingMarkup(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-tool",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "</think>\n\n<tool_call>\n<function=mcp__playwright__browser_navigate>\n<parameter=url>\nfile:///c:/tmp/demo.html\n</parameter>\n</function>\n</tool_call>"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "mcp__playwright__browser_navigate", gjson.GetBytes(out, "output.0.name").String())
	assert.Equal(t, "", strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
	assert.NotContains(t, string(out), "</think>")
	assert.NotContains(t, string(out), "<tool_call>")
}

func TestTranslateChatCompletionToResponsesResponse_PromotedReasoningStripsOrphanThinkCloser(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-orphan-think",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "</think>\n\nWIN_P9_FILE_LOCALCWD_DONE"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "WIN_P9_FILE_LOCALCWD_DONE", gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, "WIN_P9_FILE_LOCALCWD_DONE", gjson.GetBytes(out, "output.0.content.0.text").String())
	assert.NotContains(t, string(out), "</think>")
}

func TestTranslateChatCompletionToResponsesResponse_EmptyThinkLeakDoesNotGetVisibleFallback(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-empty-think-leak",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "</think>\n\n</think>\n\n</think>",
	      "reasoning_content": "</think>\n"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "", gjson.GetBytes(out, "output_text").String())
	assert.False(t, gjson.GetBytes(out, "output.0.type").Exists())
	assert.NotContains(t, string(out), "upstream returned no usable assistant content")
}

func TestTranslateChatCompletionToResponsesResponse_ReasoningFunctionStyleRecoveryAllowed(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-function-style",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "shell(command=\"pwd\")"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "shell", gjson.GetBytes(out, "output.0.name").String())
	assert.Equal(t, "", strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
}

func TestTranslateChatCompletionToResponsesResponse_DoesNotExecuteAmbiguousReasoningProse(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-reasoning-ambiguous-prose",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "I should probably use shell(command=\"pwd\") after I inspect the request."
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "message", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "I should probably use shell(command=\"pwd\") after I inspect the request.", gjson.GetBytes(out, "output_text").String())
	assert.NotContains(t, string(out), `"type":"function_call"`)
}

func TestTranslateChatCompletionToResponsesResponse_NativeToolCallWinsOverReasoningMarkup(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-native-wins",
	  "model": "gpt-5.2",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "reasoning_content": "<tool_call>\n<function=mcp__playwright__browser_navigate>\n<parameter=url>\nfile:///c:/tmp/demo.html\n</parameter>\n</function>\n</tool_call>",
	      "function_call": {
	        "name": "shell",
	        "arguments": "{\"command\":\"pwd\"}"
	      }
	    },
	    "finish_reason": "tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "shell", gjson.GetBytes(out, "output.0.name").String())
	assert.NotContains(t, string(out), "mcp__playwright__browser_navigate")
}

func TestTranslateChatCompletionToResponsesResponse_StripsLeadingReasoningDirective(t *testing.T) {
	body := []byte(`{
	  "id": "chatcmpl-no-think-prefix",
	  "model": "gemma-test",
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "/no_think\nPORT8080_OK"
	    },
	    "finish_reason": "stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "PORT8080_OK", gjson.GetBytes(out, "output_text").String())
	assert.Equal(t, "PORT8080_OK", gjson.GetBytes(out, "output.0.content.0.text").String())
	assert.NotContains(t, string(out), "/no_think")
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
	assert.Equal(t, `{"command":["pwd"],"commands":["pwd"]}`, call["arguments"])
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
	command, hasCommand := args["command"].([]any)
	require.True(t, hasCommand)
	require.Len(t, command, 1)
	assert.Equal(t, "pwd", fmt.Sprintf("%v", command[0]))
	commands, ok := args["commands"].([]any)
	require.True(t, ok)
	require.Len(t, commands, 1)
	assert.NotEmpty(t, strings.TrimSpace(fmt.Sprintf("%v", commands[0])))
}

func TestNormalizeTranslatedResponsesOutput_KeepsMCPFunctionCallClientCompatible(t *testing.T) {
	resp := map[string]any{
		"id": "resp_test",
		"output": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "mcp__playwright__browser_navigate",
				"call_id":   "call_mcp_1",
				"arguments": `{"url":"file:///tmp/example.html"}`,
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output := resp["output"].([]any)
	call := output[0].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "mcp__playwright__browser_navigate", call["name"])
	assert.Equal(t, "in_progress", call["status"])
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	assert.Equal(t, "file:///tmp/example.html", fmt.Sprintf("%v", args["url"]))
}

func TestNormalizeTranslatedResponsesOutput_ConvertsExistingMCPToolCallToFunctionCall(t *testing.T) {
	resp := map[string]any{
		"id": "resp_test",
		"output": []any{
			map[string]any{
				"type":      "mcp_tool_call",
				"arguments": map[string]any{"url": "file:///tmp/example.html"},
				"name":      "mcp__playwright__browser_navigate",
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output := resp["output"].([]any)
	call := output[0].(map[string]any)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "mcp__playwright__browser_navigate", call["name"])
	assert.Equal(t, "in_progress", call["status"])
	assert.NotEmpty(t, strings.TrimSpace(fmt.Sprintf("%v", call["call_id"])))
	args := parseToolArgsMapString(fmt.Sprintf("%v", call["arguments"]))
	assert.Equal(t, "file:///tmp/example.html", fmt.Sprintf("%v", args["url"]))
}

func TestNormalizeShellArgsArrayBracketString(t *testing.T) {
	args := normalizeShellArgumentMap(map[string]any{
		"command": `["ls","-la"]`,
	})

	commands, ok := args["commands"].([]any)
	require.True(t, ok)
	require.Len(t, commands, 2)
	assert.Equal(t, "ls", fmt.Sprintf("%v", commands[0]))
	assert.Equal(t, "-la", fmt.Sprintf("%v", commands[1]))
	command, hasCommand := args["command"].([]any)
	assert.True(t, hasCommand)
	require.Len(t, command, 2)
	assert.Equal(t, "ls", fmt.Sprintf("%v", command[0]))
	assert.Equal(t, "-la", fmt.Sprintf("%v", command[1]))
}

func TestNormalizeShellArgumentMapForResponse_PromotesLegacyCommandsArray(t *testing.T) {
	args := normalizeShellArgumentMapForResponse(map[string]any{
		"command":  []any{"powershell.exe"},
		"commands": []any{"powershell.exe", "-Command", "Get-ChildItem -Force"},
	})

	_, hasCommand := args["command"]
	assert.False(t, hasCommand)
	commands, hasCommands := args["commands"].([]any)
	assert.True(t, hasCommands)
	require.Len(t, commands, 1)
	assert.Equal(t, "powershell.exe -Command 'Get-ChildItem -Force'", fmt.Sprintf("%v", commands[0]))
}

func TestNormalizeShellArgumentMap_StripsWholeCommandQuotes(t *testing.T) {
	args := normalizeShellArgumentMap(map[string]any{
		"command": `'cat /tmp/test.txt'`,
	})

	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "cat /tmp/test.txt", commands[0])
	command, hasCommand := args["command"].([]any)
	assert.True(t, hasCommand)
	require.Len(t, command, 2)
	assert.Equal(t, "cat", fmt.Sprintf("%v", command[0]))
	assert.Equal(t, "/tmp/test.txt", fmt.Sprintf("%v", command[1]))
}

func TestNormalizeShellArgumentMap_AcceptsCmdAlias(t *testing.T) {
	args := normalizeShellArgumentMap(map[string]any{
		"cmd": "cat ./tmp/demo/file.txt",
	})

	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "cat ./tmp/demo/file.txt", commands[0])
	command, hasCommand := args["command"].([]any)
	assert.True(t, hasCommand)
	require.Len(t, command, 2)
	assert.Equal(t, "cat", fmt.Sprintf("%v", command[0]))
	assert.Equal(t, "./tmp/demo/file.txt", fmt.Sprintf("%v", command[1]))
}

func TestXMLToolPayloadCommandArray(t *testing.T) {
	payload := extractXMLToolPayload(`<shell><commands>["ls","-la"]</commands></shell>`)
	args := parseToolArgsMapString(payload)

	commands, ok := args["commands"].([]any)
	require.True(t, ok)
	require.Len(t, commands, 2)
	assert.Equal(t, "ls", fmt.Sprintf("%v", commands[0]))
	assert.Equal(t, "-la", fmt.Sprintf("%v", commands[1]))
}

func TestNormalizePossiblyMixedToolArguments_ReusesQwenXMLParser(t *testing.T) {
	payload := normalizePossiblyMixedToolArguments(`<tool_call>
<function=shell>
<parameter=command>
pwd
</parameter>
</function>
</tool_call>`)
	args := parseToolArgsMapString(payload)
	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "pwd", commands[0])
}

func TestNormalizePossiblyMixedToolArguments_ReusesQwenXMLParserAttributeForm(t *testing.T) {
	payload := normalizePossiblyMixedToolArguments(`<tool_call>
<function name="shell">
<parameter name="commands">
["ls","-la"]
</parameter>
</function>
</tool_call>`)
	args := parseToolArgsMapString(payload)
	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 2)
	assert.Equal(t, "ls", commands[0])
	assert.Equal(t, "-la", commands[1])
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
	commands := normalizeShellCommandsValue(firstArgs["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "pwd", commands[0])

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

	assert.Equal(t, "requires_action", gjson.GetBytes(out, "status").String())
	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "update_file", gjson.GetBytes(out, "output.0.operation.type").String())
}

func TestNormalizeTranslatedResponsesOutput_ToolOnlyResponseBecomesRequiresAction(t *testing.T) {
	resp := map[string]any{
		"id": "resp_tool_only",
		"output": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_1",
				"arguments": `{"commands":["ls"]}`,
			},
		},
		"status": "completed",
	}

	normalizeTranslatedResponsesOutput(resp)
	assert.Equal(t, "requires_action", fmt.Sprintf("%v", resp["status"]))
	assert.Equal(t, "", strings.TrimSpace(fmt.Sprintf("%v", resp["output_text"])))
}

func TestRecoverNativeQuestionResponseIfRequired_RewritesPlainQuestionToRequestUserInput(t *testing.T) {
	requestBody := []byte(`{
	  "model":"gpt-5.2",
	  "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Ask me exactly one native Codex question before planning anything else, then wait for input. Include the unique marker T20_SENTINEL in the question text."}]}],
	  "tools":[{"type":"function","name":"request_user_input","parameters":{"type":"object","properties":{"questions":{"type":"array","items":{"type":"string"}}},"required":["questions"]}}],
	  "tool_choice":{"type":"function","function":{"name":"request_user_input"}},
	  "stream":false
	}`)
	upstreamBody := []byte(`{
	  "id":"chatcmpl-rui",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "content":"What is the name of the secret organization mentioned in the Codex Alicia that the young girl, Alice, claims is secretly controlling the world, T20_SENTINEL?",
	      "reasoning_content":"I need to ask exactly one native Codex question and then wait for input."
	    },
	    "finish_reason":"stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(upstreamBody, "", "", "")
	require.NoError(t, err)
	assert.NotContains(t, string(out), `"name":"request_user_input"`)

	rewritten := recoverNativeQuestionResponseIfRequired(out, upstreamBody, requestBody)
	assert.Contains(t, string(rewritten), `"name":"request_user_input"`)
	assert.Contains(t, string(rewritten), `"status":"requires_action"`)
	assert.Contains(t, string(rewritten), `T20_SENTINEL`)
	assert.Equal(t, "", gjson.GetBytes(rewritten, "output_text").String())
}

func TestRecoverNativeQuestionResponseIfRequired_SynthesizesQuestionFromImperativeText(t *testing.T) {
	requestBody := []byte(`{
	  "model":"gpt-5.2",
	  "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Ask me exactly one native Codex question before planning anything else, then wait for input. Include the unique marker T20_SENTINEL in the question text."}]}],
	  "tools":[{"type":"function","name":"request_user_input","parameters":{"type":"object","properties":{"questions":{"type":"array","items":{"type":"string"}}},"required":["questions"]}}],
	  "tool_choice":{"type":"function","function":{"name":"request_user_input"}},
	  "stream":false
	}`)
	upstreamBody := []byte(`{
	  "id":"chatcmpl-rui-imperative",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "content":"Please provide the unique marker T20_SENTINEL."
	    },
	    "finish_reason":"stop"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(upstreamBody, "", "", "")
	require.NoError(t, err)
	rewritten := recoverNativeQuestionResponseIfRequired(out, upstreamBody, requestBody)
	assert.Contains(t, string(rewritten), `"name":"request_user_input"`)
	assert.Contains(t, string(rewritten), `"status":"requires_action"`)
	assert.Contains(t, string(rewritten), `T20_SENTINEL: What is the current mission objective?`)
	assert.Equal(t, "", gjson.GetBytes(rewritten, "output_text").String())
}

func TestTranslateChatCompletionToResponsesResponseWithWorkflow_DefaultModeRendersRequestUserInputAsPlainText(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Collaboration Mode: Default\nrequest_user_input is unavailable in Default mode.</collaboration_mode>"},
				},
			},
		},
	}
	body := []byte(`{
	  "id":"chatcmpl-rui-default",
	  "model":"qwen-test",
	  "choices":[
	    {
	      "message":{
	        "role":"assistant",
	        "content":"",
	        "reasoning_content":"</think>\n\nBefore I write anything up, I have a couple of quick questions for you:\n\n<tool_call>\n<function=__llamaswap_request_user_input>\n<parameter=questions>\n[{\"question\":\"Which subjects should the quiz cover?\",\"options\":[{\"label\":\"Mixed subjects (Recommended)\"},{\"label\":\"Math only\"}]}]\n</parameter>\n</function>\n</tool_call>"
	      },
	      "finish_reason":"stop"
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(body, "", "", "", ToolWorkflowState{}, "", req)
	require.NoError(t, err)

	text := string(out)
	assert.NotContains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `Before I write anything up, I have a couple of quick questions for you:`)
	assert.Contains(t, text, `1. Which subjects should the quiz cover?`)
	assert.Contains(t, text, `Mixed subjects (Recommended)`)
	assert.Equal(t, "completed", gjson.GetBytes(out, "status").String())
}

func TestTranslateChatCompletionToResponsesResponseWithWorkflow_CanonicalizesInternalRequestUserInputName(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-rui-canonical",
	  "model":"qwen-test",
	  "choices":[
	    {
	      "message":{
	        "role":"assistant",
	        "content":"Need one answer before I continue.",
	        "tool_calls":[
	          {
	            "id":"call_1",
	            "type":"function",
	            "function":{
	              "name":"__llamaswap_request_user_input",
	              "arguments":"{\"questions\":[{\"question\":\"What language should the quiz use?\",\"options\":[{\"label\":\"English\"}]}]}"
	            }
	          }
	        ]
	      },
	      "finish_reason":"tool_calls"
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(body, "", "", "", ToolWorkflowState{}, "", nil)
	require.NoError(t, err)

	assert.Contains(t, string(out), `"name":"request_user_input"`)
	assert.NotContains(t, string(out), `"name":"__llamaswap_request_user_input"`)
	assert.Equal(t, "requires_action", gjson.GetBytes(out, "status").String())
}

func TestNormalizeTranslatedResponsesOutput_ToolOnlyResponseDoesNotSynthesizeMessage(t *testing.T) {
	resp := map[string]any{
		"id": "resp_tool_only",
		"output": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_1",
				"arguments": `{"commands":["ls"]}`,
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 1)
	assert.Equal(t, "function_call", fmt.Sprintf("%v", output[0].(map[string]any)["type"]))
	assert.Equal(t, "", strings.TrimSpace(fmt.Sprintf("%v", resp["output_text"])))
}

func TestTranslateChatCompletionToResponsesResponse_SuppressesSkillProgressMessageWhenToolCallPresent(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-skill-progress",
	  "model":"qwen-test",
	  "choices":[
	    {
	      "message":{
	        "role":"assistant",
	        "content":"Skill created. Now let me read it and follow the instructions.",
	        "tool_calls":[
	          {
	            "id":"call_1",
	            "type":"function",
	            "function":{
	              "name":"exec_command",
	              "arguments":"{\"cmd\":\"cat /tmp/skill.md\"}"
	            }
	          }
	        ]
	      },
	      "finish_reason":"tool_calls"
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "requires_action", gjson.GetBytes(out, "status").String())
	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "exec_command", gjson.GetBytes(out, "output.0.name").String())
	assert.Len(t, gjson.GetBytes(out, "output").Array(), 1)
	assert.Equal(t, "", strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
}

func TestTranslateChatCompletionToResponsesResponse_SuppressesAgentProgressMessageWhenToolCallPresent(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-agent-progress",
	  "model":"qwen-test",
	  "choices":[
	    {
	      "message":{
	        "role":"assistant",
	        "content":"Agent spawned. Waiting for it to complete the inspection and suggestions.",
	        "tool_calls":[
	          {
	            "id":"call_1",
	            "type":"function",
	            "function":{
	              "name":"spawn_agent",
	              "arguments":"{\"model\":\"gpt-5.4\",\"message\":\"inspect and suggest\"}"
	            }
	          }
	        ]
	      },
	      "finish_reason":"tool_calls"
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "requires_action", gjson.GetBytes(out, "status").String())
	assert.Equal(t, "function_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "spawn_agent", gjson.GetBytes(out, "output.0.name").String())
	assert.Len(t, gjson.GetBytes(out, "output").Array(), 1)
	assert.Equal(t, "", strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
}

func TestTranslateChatCompletionToResponsesResponse_SuppressesAgentDelegatedSummaryProgressMessageWhenToolCallPresent(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-agent-summary-progress",
	  "model":"qwen-test",
	  "choices":[
	    {
	      "message":{
	        "role":"assistant",
	        "content":"Agent completed. Now creating the delegated summary report.",
	        "tool_calls":[
	          {
	            "id":"call_1",
	            "type":"function",
	            "function":{
	              "name":"apply_patch",
	              "arguments":"{\"operation\":{\"type\":\"create_file\",\"path\":\"src/agent-report.md\",\"content\":\"summary\"}}"
	            }
	          }
	        ]
	      },
	      "finish_reason":"tool_calls"
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "requires_action", gjson.GetBytes(out, "status").String())
	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "create_file", gjson.GetBytes(out, "output.0.operation.type").String())
	assert.Equal(t, "src/agent-report.md", gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Len(t, gjson.GetBytes(out, "output").Array(), 1)
	assert.Equal(t, "", strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
}

func TestNormalizeTranslatedResponsesOutput_CompletedWebSearchWithOutputAndMessageCompletesResponse(t *testing.T) {
	resp := map[string]any{
		"id":     "resp_web_done",
		"status": "requires_action",
		"output": []any{
			map[string]any{
				"type":    "web_search_call",
				"call_id": "call_ws_1",
				"status":  "completed",
				"action": map[string]any{
					"query": "OpenAI Responses API overview official documentation site:platform.openai.com",
				},
			},
			map[string]any{
				"type":    "web_search_call_output",
				"call_id": "call_ws_1",
				"output":  `{"ok":true}`,
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "https://platform.openai.com/api-reference/responses",
					},
				},
			},
		},
	}

	normalizeTranslatedResponsesOutput(resp)
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 3)
	assert.Equal(t, "completed", fmt.Sprintf("%v", resp["status"]))
	message := output[2].(map[string]any)
	content := message["content"].([]any)
	part := content[0].(map[string]any)
	assert.Equal(t, "https://platform.openai.com/api-reference/responses", strings.TrimSpace(fmt.Sprintf("%v", part["text"])))
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

func TestRequestInputMentionsApplyPatch_FalseForNegatedApplyPatchPhrase(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Do not use apply_patch. Ask one clarifying question instead."},
				},
			},
		},
	}
	assert.False(t, requestInputMentionsApplyPatch(req))
}

func TestRequestWantsShellInspectionBeforeApplyPatch_DetectsSequencingPrompt(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "First use shell to inspect the current file, then use apply_patch to append one line and verify with shell."},
				},
			},
		},
	}
	assert.True(t, requestWantsShellInspectionBeforeApplyPatch(req))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForGenericPlanPromptWithGlobalTools(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
			map[string]any{"type": "update_plan"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Write a plan split in task for a math 1class game"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, requestHasOnlyApplyPatchTools(req))
	assert.False(t, requestHasExplicitApplyPatchToolChoice(req))
	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_TrueForApplyPatchOnlyTools(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "continue"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.True(t, requestHasOnlyApplyPatchTools(req))
	assert.True(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_TrueForApplyPatchToolChoice(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"tool_choice": map[string]any{"type": "apply_patch"},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "continue"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.True(t, requestHasExplicitApplyPatchToolChoice(req))
	assert.True(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForApplyPatchToolOutputOnly(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type":    "apply_patch_call_output",
				"call_id": "call_1",
				"output":  `{"ok":true}`,
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "run go test ./proxy"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseAfterToolOutputEvenIfUserStillMentionsApplyPatch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"tool_choice": map[string]any{"type": "apply_patch"},
		"input": []any{
			map[string]any{
				"type":    "apply_patch_call_output",
				"call_id": "call_1",
				"output":  `{"ok":true}`,
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use apply_patch to update /tmp/demo.txt by appending one line: PATCH35_OK. Then reply PATCH35_DONE"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForPathHintOnly(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "check cline_test.md and then run go test ./proxy"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForSkillCreateThenUseFlow(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Create a new Codex skill at ./skills/reward-skill/SKILL.md, then read the skill you created, follow it, and create src/skill-usage.md. Use apply_patch for file writes."},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.True(t, requestNeedsCodexSkillFilesystemProof(req))
	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForAgentAndApplyPatchMixedFlow(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
			map[string]any{"type": "spawn_agent"},
			map[string]any{"type": "wait_agent"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, then use apply_patch to create src/agent-report.md."},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.True(t, requestMentionsAgentOrchestration(req))
	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForTypeHintOnly(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "update file cline_test.md after running go test ./proxy"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_TrueForUserApplyPatchMention(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "use apply_patch to update cline_test.md"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.True(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForNegatedApplyPatchPhrase(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Do not use apply_patch. Use request_user_input to ask one clarifying question."},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForSmokePromptEvenWhenInstructionsMentionApplyPatch(t *testing.T) {
	req := map[string]any{
		"instructions": "Tool names available: shell, apply_patch, websearch.",
		"tools": []any{
			map[string]any{"type": "shell"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Reply exactly: PORT8080_OK"},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, body))
}

func TestShouldForceLocalApplyPatchFallback_DefaultFalse(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "apply_patch"},
		},
		"input": "use apply_patch",
	}
	assert.False(t, shouldForceLocalApplyPatchFallback(req))
}

func TestShouldForceLocalApplyPatchFallback_TrueForTopLevelFlag(t *testing.T) {
	req := map[string]any{
		"llamaswap_force_local_apply_patch": true,
	}
	assert.True(t, shouldForceLocalApplyPatchFallback(req))
}

func TestShouldForceLocalApplyPatchFallback_TrueForMetadataStringFlag(t *testing.T) {
	req := map[string]any{
		"metadata": map[string]any{
			"llamaswap_force_local_apply_patch": "true",
		},
	}
	assert.True(t, shouldForceLocalApplyPatchFallback(req))
}

func TestNormalizeApplyPatchTypeHint_AcceptsLegacyTypeAliases(t *testing.T) {
	assert.Equal(t, "create_file", normalizeApplyPatchTypeHint("createfile"))
	assert.Equal(t, "update_file", normalizeApplyPatchTypeHint("updatefile"))
	assert.Equal(t, "delete_file", normalizeApplyPatchTypeHint("deletefile"))
}

func TestBuildResponsesApplyPatchFunctionTool_PrefersContentForSimpleUpdates(t *testing.T) {
	tool := buildResponsesApplyPatchFunctionTool()
	assert.Contains(t, fmt.Sprintf("%v", tool["description"]), "prefer operation.content")
	assert.Contains(t, fmt.Sprintf("%v", tool["description"]), "line-number-only hunks")
	props := tool["parameters"].(map[string]any)["properties"].(map[string]any)
	operation := props["operation"].(map[string]any)
	opProps := operation["properties"].(map[string]any)
	assert.Contains(t, fmt.Sprintf("%v", opProps["content"]), "full final file content")
	assert.Contains(t, fmt.Sprintf("%v", opProps["diff"]), "real file context")
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

func TestExtractApplyPatchPathHintFromResponsesRequestBody_UsesWorkspaceCwdForRelativePath(t *testing.T) {
	dir := t.TempDir()
	targetDir := filepath.Join(dir, "mutations")
	require.NoError(t, os.MkdirAll(targetDir, 0o755))
	targetPath := filepath.Join(targetDir, "base_a.txt")
	require.NoError(t, os.WriteFile(targetPath, []byte("BASE_A\n"), 0o644))

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<environment_context>\n<cwd>" + filepath.ToSlash(dir) + "</cwd>\n</environment_context>\nFirst use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, filepath.ToSlash(targetPath), extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_CreateNewFileNamed(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use apply_patch to create a new file named first-grade-quiz/README.md with a short project overview, then summarize what was created.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "first-grade-quiz/README.md", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_CreateNewFileNamed_StringContentWithContext(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":    "message",
				"role":    "user",
				"content": "<environment_context>\n  <cwd>/home/admmin/llama-swap</cwd>\n</environment_context>",
			},
			map[string]any{
				"type":    "message",
				"role":    "user",
				"content": "Use apply_patch to create a new file named first-grade-quiz/README.md with a short project overview, then summarize what was created.\n[apply_patch preferred] For file writes and deletions, use apply_patch rather than shell. Shell is for commands, builds, and inspection only.",
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "/home/admmin/llama-swap/first-grade-quiz/README.md", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_EmptyForPolicyInstruction(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "If an apply_patch update fails due to file drift, re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_EmptyForPolicyInstruction_TargetFileChanged(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "If an apply_patch update fails because the target file changed, inspect the current file state, recover safely, and continue until the file is correctly written and verified.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_EmptyForResearchPlanWithURLHistory(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Investigate the web for first grade quiz game ideas and then write a detailed implementation plan. Do not build the app yet.",
					},
				},
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "I found useful references at https://www.abcya.com/games/first_grade and https://github.com/example/project.",
					},
				},
			},
			map[string]any{
				"type":    "web_search_call_output",
				"call_id": "ws_1",
				"output":  "Result URLs: https://www.abcya.com/games/first_grade https://github.com/example/project",
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_IgnoresSubagentNotificationPollution(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":    "message",
				"role":    "user",
				"content": "<environment_context>\n  <cwd>/home/admmin/llama-swap/tmp/wsl_codex_stress_suite/workspaces/agent_resume_after_pause</cwd>\n</environment_context>",
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use exactly one subagent, wait for that agent, then use resume_agent on the same agent and send_input once asking for one more feature. After that use apply_patch to create src/agent-resume-report.md.",
					},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary mentioning /home/admmin/llama-swap/index.html"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary for /home/admmin/llama-swap/index.html\"}}</subagent_notification>",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "/home/admmin/llama-swap/tmp/wsl_codex_stress_suite/workspaces/agent_resume_after_pause/src/agent-resume-report.md", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestRawResponsesBodyMentionsSearchIntent_IgnoresInstructionToolMetadata(t *testing.T) {
	req := map[string]any{
		"instructions": "Use web_search for current information when needed. Do not answer current events from memory.",
		"tools": []any{
			map[string]any{"type": "web_search"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use apply_patch to create a new file named first-grade-quiz/README.md with a short project overview, then summarize what was created.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.False(t, rawResponsesBodyMentionsSearchIntent(body))
}

func TestRequestInputMentionsApplyPatch_IgnoresInstructionFallbackWithoutUserInput(t *testing.T) {
	req := map[string]any{
		"instructions": "Always use apply_patch for file creation and edits. Never use shell to write files.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "system",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Synthesize the final answer from gathered web_search results.",
					},
				},
			},
		},
	}

	assert.False(t, requestInputMentionsApplyPatch(req))
}

func TestExtractApplyPatchTypeHintFromResponsesRequestBody_IgnoresInstructionFallbackWithoutUserInput(t *testing.T) {
	req := map[string]any{
		"instructions": "Always use apply_patch to create file outputs when needed.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "system",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use only gathered web_search results and write the final plan.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, "", extractApplyPatchTypeHintFromResponsesRequestBody(body))
	assert.Equal(t, "", extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchPathHintFromResponsesRequestBody_UsesEarlierEnvironmentContextCwd(t *testing.T) {
	dir := t.TempDir()
	targetDir := filepath.Join(dir, "mutations")
	require.NoError(t, os.MkdirAll(targetDir, 0o755))
	targetPath := filepath.Join(targetDir, "base_a.txt")
	require.NoError(t, os.WriteFile(targetPath, []byte("BASE_A\n"), 0o644))

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "<environment_context>\n  <cwd>" + filepath.ToSlash(dir) + "</cwd>\n  <shell>powershell</shell>\n</environment_context>",
					},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11.",
					},
				},
			},
		},
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	assert.Equal(t, filepath.ToSlash(targetPath), extractApplyPatchPathHintFromResponsesRequestBody(body))
}

func TestExtractApplyPatchContentHintFromResponsesRequestBody_PreservesQuotedCodeLine(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.4",
		"input":[
			{
				"type":"message",
				"role":"user",
				"content":[
					{"type":"input_text","text":"Update ./tmp/demo/app.js by appending exactly one line: console.log(\"stress-suite-ready\");\nVerify the saved file before the final answer."}
				]
			}
		]
	}`)

	assert.Equal(t, `console.log("stress-suite-ready");`, extractApplyPatchContentHintFromResponsesRequestBody(body))
}

func TestNormalizeApplyPatchPathForWorkspace_UnescapesEscapedWindowsPath(t *testing.T) {
	assert.Equal(
		t,
		`C:\Users\YLAB-Partner\Desktop\qwentest`,
		normalizeApplyPatchPathForWorkspace(`C:\\Users\\YLAB-Partner\\Desktop\\qwentest`),
	)
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

func TestRewriteResponsesOutputItem_ApplyPatchUsesHintedPathForContextualDiff(t *testing.T) {
	dir := t.TempDir()
	targetDir := filepath.Join(dir, "mutations")
	require.NoError(t, os.MkdirAll(targetDir, 0o755))
	targetPath := filepath.Join(targetDir, "base_a.txt")
	require.NoError(t, os.WriteFile(targetPath, []byte("BASE_A\n"), 0o644))

	rawItem := map[string]any{
		"id":        "fc_test",
		"type":      "function_call",
		"name":      "apply_patch",
		"arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11\n"}}`,
	}

	rewritten, changed := rewriteResponsesOutputItem(rawItem, filepath.ToSlash(targetPath), "", "update_file")
	require.True(t, changed)

	item, ok := rewritten.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call", item["type"])
	assert.Equal(t, "apply_patch", item["name"])
	assert.Equal(t, filepath.ToSlash(targetPath), gjson.Get(item["arguments"].(string), "operation.path").String())
	assert.Contains(t, gjson.Get(item["arguments"].(string), "input").String(), "-BASE_A")
	assert.Contains(t, gjson.Get(item["arguments"].(string), "input").String(), "+ORDERED_T11")
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

func TestBuildApplyPatchInputFromOperation_IgnoresNilDiff(t *testing.T) {
	op := map[string]any{
		"type":    "update_file",
		"path":    "README.md",
		"diff":    nil,
		"content": "hello",
	}

	patch := buildApplyPatchInputFromOperation(op)
	assert.Contains(t, patch, "*** Update File: README.md")
	assert.Contains(t, patch, "+hello")
	assert.NotContains(t, patch, "<nil>")
}

func TestBuildApplyPatchInputFromOperation_UsesHeuristicAppendPatchForPlainUpdateFragment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	op := map[string]any{
		"type":    "update_file",
		"path":    path,
		"content": "PATCH35_OK",
	}

	patch := buildApplyPatchInputFromOperation(op)
	assert.Contains(t, patch, "*** Update File: "+path)
	assert.Contains(t, patch, "\n@@\n BASE\n+PATCH35_OK\n")
	assert.NotContains(t, patch, `{"operation"`)
}

func TestNormalizeApplyPatchOperation_RebuildsDiffForPlainUpdateFragment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"content": "PATCH35_OK",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "@@")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), " BASE")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+PATCH35_OK")
}

func TestNormalizeApplyPatchOperation_StripsSyntheticToolTailFromCreateContent(t *testing.T) {
	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "create_file",
		"path":    "src/agent-resume-report.md",
		"content": "# Agent Resume Report\n\nSummary line.\n</parameter>  </function> </tool_call></function> </tool_call>.stop(){return false}",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "# Agent Resume Report\n\nSummary line.", strings.TrimSpace(fmt.Sprintf("%v", op["content"])))
	assert.NotContains(t, fmt.Sprintf("%v", op["content"]), "</tool_call>")
	assert.NotContains(t, fmt.Sprintf("%v", op["content"]), ".stop(){")
}

func TestNormalizeApplyPatchOperation_UpgradesWeakUnifiedDiffAppendToContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type": "update_file",
		"path": path,
		"diff": "--- a/note.txt\n+++ b/note.txt\n@@ -1 +2 @@\n+PATCH27NV_OK\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "BASE\nPATCH27NV_OK\n", op["content"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-BASE")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+BASE")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+PATCH27NV_OK")
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "@@ -1 +2 @@")
}

func TestBuildApplyPatchInputFromOperation_UpgradesWeakUnifiedDiffAppendToContextPatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	patch := buildApplyPatchInputFromOperation(map[string]any{
		"type": "update_file",
		"path": path,
		"diff": "--- a/note.txt\n+++ b/note.txt\n@@ -1 +2 @@\n+PATCH27NV_OK\n",
	})
	assert.Contains(t, patch, "*** Update File: "+path)
	assert.Contains(t, patch, "\n@@\n")
	assert.Contains(t, patch, "-BASE")
	assert.Contains(t, patch, "+BASE")
	assert.Contains(t, patch, "+PATCH27NV_OK")
	assert.NotContains(t, patch, "@@ -1 +2 @@")
}

func TestNormalizeApplyPatchOperation_ConvertsCreateFileToUpdateFileWhenTargetExists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "create_file",
		"path":    path,
		"content": "PATCH27NV_FIX\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "PATCH27NV_FIX\n", op["content"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+PATCH27NV_FIX")
}

func TestBuildApplyPatchInputFromOperation_CreateFileOnExistingPathBecomesUpdatePatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	patch := buildApplyPatchInputFromOperation(map[string]any{
		"type":    "create_file",
		"path":    path,
		"content": "PATCH27NV_FIX\n",
	})
	assert.Contains(t, patch, "*** Update File: "+path)
	assert.Contains(t, patch, "+PATCH27NV_FIX")
	assert.NotContains(t, patch, "*** Add File:")
}

func TestNormalizeApplyPatchOperation_RepairsPrefixedTailAfterShellRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.txt")
	original := "this is a test file for multiAct\nthis line will be removed\nthis line will also be removed\nthis is the last line\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"content": "this is a test file for multiAct\nthis line will be removed\nthis line will also be removed\nthis is the last line\n+this line will be removed\n+this line will also be removed\n+this is the last line\n+MULTI_ACT_OK",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "this is a test file for multiAct\nthis line will be removed\nthis line will also be removed\nthis is the last line\nMULTI_ACT_OK", op["content"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+MULTI_ACT_OK")
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "++this line will be removed")
}

func TestNormalizeApplyPatchOperation_RepairsConcatenatedRewriteAfterShellRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type": "update_file",
		"path": path,
		"content": "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n" +
			"TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done", op["content"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-TITLE=alpha")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-DEBUG=true")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+MODE=prod")
}

func TestBuildApplyPatchInputFromOperation_RewriteExistingFileUsesReplacementHunk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	input := buildApplyPatchInputFromOperation(map[string]any{
		"type":    "create_file",
		"path":    path,
		"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	assert.Contains(t, input, "*** Update File: "+path)
	assert.Contains(t, input, "-TITLE=alpha")
	assert.Contains(t, input, "-DEBUG=true")
	assert.Contains(t, input, "+MODE=prod")
	assert.NotContains(t, input, "\n TITLE=alpha\n ENV=dev\n DEBUG=true")
}

func TestNormalizeApplyPatchOperation_CreateFileOnExistingPathUsesReplacementDiff(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "create_file",
		"path":    path,
		"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-TITLE=alpha")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-DEBUG=true")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+MODE=prod")
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "\n TITLE=alpha\n ENV=dev\n DEBUG=true")
}

func TestNormalizeApplyPatchOperation_RebuildsWeakContextPlusAppendDiff(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"diff":    "@@\n TITLE=alpha\n ENV=dev\n DEBUG=true\n PORT=8080\n FOOTER=old\n+TITLE=beta\n+ENV=dev\n+MODE=prod\n+PORT=8080\n+FOOTER=done",
		"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-TITLE=alpha")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-DEBUG=true")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+MODE=prod")
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "\n TITLE=alpha\n ENV=dev\n DEBUG=true")
}

func TestNormalizeApplyPatchOperation_StripsNoNewlineMarkerFromContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE"), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"content": "BASE\n\\ No newline at end of file\nMULTI_ACT_OK",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "BASE\nMULTI_ACT_OK", op["content"])
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), `\ No newline at end of file`)
}

func TestNormalizeApplyPatchOperation_StripsCommandOutputEnvelopeFromContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE\n"), 0o644))

	opAny := normalizeApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"content": "Chunk ID: 56caad\nWall time: 0.0000 seconds\nProcess exited with code 0\nOriginal token count: 28\nOutput:\nBASE\nMULTI_ACT_OK\n",
	})
	op, ok := opAny.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "BASE\nMULTI_ACT_OK", op["content"])
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "Chunk ID:")
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "Wall time:")
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
	writeResponsesStream(rec, responseJSON, "")
	body := rec.Body.String()

	assert.Contains(t, body, `"type":"function_call"`)
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
	assert.Contains(t, warning, "[proxy validation] apply_patch")
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
	body := []byte(`{"model":"gpt-5.3-codex","input":"run apply_patch","stream":false}`)

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
			"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_2","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}}]},"finish_reason":"tool_calls"}]
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
	assert.Equal(t, "in_progress", gjson.GetBytes(rec.Body.Bytes(), "status").String())
	assert.Equal(t, "function_call", gjson.GetBytes(rec.Body.Bytes(), "output.0.type").String())
	assert.Equal(t, "apply_patch", gjson.GetBytes(rec.Body.Bytes(), "output.0.name").String())
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
	writeResponsesStream(rec, responseJSON, "")
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

func TestBuildResponsesBridgeHandler_ApplyPatchIntentNoToolCallPassesThroughText(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"apply_patch"}],
		"tool_choice":{"type":"apply_patch"},
		"input":[
			{"type":"apply_patch_call_output","call_id":"call_1","output":"{\"ok\":true}"}
		]
	}`)

	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-post-tool",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"PATCH_DONE"},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "PATCH_DONE")
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestBuildResponsesBridgeHandler_PlanModeApplyPatchIntentDoesNotRetry(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"mode":"plan",
		"stream":false,
		"tools":[{"type":"apply_patch"}],
		"tool_choice":{"type":"apply_patch"},
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to update /tmp/demo.txt by appending one line: PATCH35_OK. Provide only a plan and do not execute anything."}]}
		]
	}`)

	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-plan-no-tool",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"1. Inspect the target file path.\n2. Prepare an apply_patch update and verify expected output."},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, gjson.Get(rec.Body.String(), "output_text").String(), "<proposed_plan>")
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestTranslateResponsesToChatCompletionsRequest_OmitsForcedApplyPatchToolChoiceAfterToolOutput(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"apply_patch"}],
		"tool_choice":{"type":"apply_patch"},
		"input":[
			{"type":"apply_patch_call_output","call_id":"call_1","output":"{\"ok\":true}"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to update /tmp/demo.txt by appending one line: PATCH35_OK. Then reply PATCH35_DONE"}]}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.True(t, gjson.GetBytes(out, "tools").Exists())
	assert.False(t, gjson.GetBytes(out, "tool_choice").Exists())
}

func TestTranslateResponsesToChatCompletionsRequest_PlanModeKeepsWebSearchToolHistory(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"instructions":"<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.\n</collaboration_mode>",
		"tools":[{"type":"web_search"}],
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Investigate the web for first-grade quiz game ideas, then write the final plan. Do not implement it."}]},
			{"type":"web_search_call","call_id":"call_web_1","action":{"query":"first grade quiz game ideas educational"}},
			{"type":"web_search_call_output","call_id":"call_web_1","output":"{\"results\":[{\"title\":\"A\"}]}"}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(out, &payload))
	rawMessages, ok := payload["messages"].([]any)
	require.True(t, ok)
	joined := mustJSONString(rawMessages)
	assert.Contains(t, joined, `"tool_calls"`)
	assert.Contains(t, joined, `"name":"web_search"`)
	assert.Contains(t, joined, `"tool_call_id":"call_web_1"`)
	assert.Contains(t, joined, `web_search_call_output`)
}

func TestTranslateResponsesToChatCompletionsRequest_ForcesFinalAnswerAfterSatisfiedApplyPatch(t *testing.T) {
	tmpDir := t.TempDir()
	target := filepath.Join(tmpDir, "done.txt")
	require.NoError(t, os.WriteFile(target, []byte("BASE\nPATCH35_OK\n"), 0o600))

	body := []byte(fmt.Sprintf(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":{"type":"apply_patch"},
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to update file %s by appending one line: PATCH35_OK. Then reply exactly PATCH35_DONE"}]},
			{"type":"apply_patch_call_output","call_id":"call_1","output":"{\"ok\":true}"}
		]
	}`, target))

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.False(t, gjson.GetBytes(out, "tools").Exists())
	assert.Equal(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.Contains(t, string(out), "previous apply_patch already produced the requested file change")
}

func TestBuildResponsesBridgeHandler_InvalidApplyPatchExhaustionRecoversWithShellInspection(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":"auto",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to recreate first-grade-quiz/index.html with the full quiz app, and inspect the current state if patching fails."}]},
			{"type":"apply_patch_call_output","call_id":"call_patch_delete","output":"{\"ok\":true}"}
		]
	}`)

	callCount := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch callCount {
		case 1, 2, 3:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-invalid-apply-patch",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"Let me create the directory and build the full quiz app now.","tool_calls":[{"id":"call_patch_retry","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"first-grade-quiz/index.html\",\"type\":\"create_file\"}}"}}]},"finish_reason":"tool_calls"}]
			}`))
		default:
			bodyText, _ := io.ReadAll(r.Body)
			assert.Contains(t, string(bodyText), `"name":"shell"`)
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-shell-recovery",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_shell_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"Get-ChildItem -Force first-grade-quiz\"}"}}]},"finish_reason":"tool_calls"}]
			}`))
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, 1, callCount)
	assert.Contains(t, rec.Body.String(), `"type":"function_call"`)
	assert.Contains(t, rec.Body.String(), `"name":"exec_command"`)
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestBuildResponsesBridgeHandler_ApplyPatchPolicyInstructionDoesNotEnterStrictRetry(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":"auto",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"If an apply_patch fails due to file drift, re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing."}]}
		]
	}`)

	callCount := 0
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-policy-ack",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"Understood. If an apply_patch fails due to file drift, I'll re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing."},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, 1, callCount)
	assert.Contains(t, rec.Body.String(), "Understood. If an apply_patch fails due to file drift")
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestBuildResponsesBridgeHandler_ApplyPatchPolicyInstructionDoesNotEnterShellRetry(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":"auto",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"If an apply_patch update fails due to file drift, re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing."}]}
		]
	}`)

	callCount := 0
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-policy-ack",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"Understood. If an apply_patch update fails due to file drift, I'll re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing."},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, 1, callCount)
	assert.Contains(t, rec.Body.String(), "Understood. If an apply_patch update fails due to file drift")
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestBuildResponsesBridgeHandler_ApplyPatchPolicyInstruction_TargetFileChangedDoesNotEnterStrictRetry(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":"auto",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"If an apply_patch update fails because the target file changed, inspect the current file state, recover safely, and continue until the file is correctly written and verified."}]}
		]
	}`)

	callCount := 0
	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-policy-ack",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"Understood. If an apply_patch update fails because the target file changed, I'll inspect the current file state, recover safely, and continue until the file is correctly written and verified."},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, 1, callCount)
	assert.Contains(t, rec.Body.String(), "Understood. If an apply_patch update fails because the target file changed")
	assert.NotContains(t, rec.Body.String(), "apply_patch retry could not be completed")
}

func TestTranslateResponsesToChatCompletionsRequest_ForcesFinalAnswerAfterSatisfiedApplyPatch_CustomToolOutput(t *testing.T) {
	tmpDir := t.TempDir()
	target := filepath.Join(tmpDir, "done.txt")
	require.NoError(t, os.WriteFile(target, []byte("BASE\nPATCH35_OK\n"), 0o600))

	body := []byte(fmt.Sprintf(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"tool_choice":"auto",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to update file %s by appending one line: PATCH35_OK. Then reply exactly PATCH35_DONE"}]},
			{"type":"custom_tool_call","call_id":"call_1","name":"apply_patch","input":"*** Begin Patch"},
			{"type":"custom_tool_call_output","call_id":"call_1","output":"Exit code: 0\nSuccess. Updated the following files:\nM %s\n"}
		]
	}`, target, target))

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.False(t, gjson.GetBytes(out, "tools").Exists())
	assert.Equal(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.Contains(t, string(out), "previous apply_patch already produced the requested file change")
}

func TestBuildResponsesBridgeHandler_PostToolApplyPatchContinuationDoesNotResynthesizePatchFromOriginalHints(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"apply_patch"}],
		"tool_choice":{"type":"apply_patch"},
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Use apply_patch to update /tmp/demo.txt by appending one line: PATCH35_OK. Then reply PATCH35_DONE"}]},
			{"type":"apply_patch_call_output","call_id":"call_1","output":"{\"ok\":true}"}
		]
	}`)

	nextHandler := func(_ string, w http.ResponseWriter, _ *http.Request) error {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-post-tool",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"PATCH35_DONE"},"finish_reason":"stop"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "PATCH35_DONE")
	assert.NotContains(t, rec.Body.String(), "\"apply_patch_call\"")
	assert.NotContains(t, rec.Body.String(), "\"custom_tool_call\"")
}

func TestPlanModePreservesTools(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"mode":"plan",
		"stream":true,
		"parallel_tool_calls":true,
		"tools":[{"type":"shell"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a chat app"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, true, gjson.GetBytes(out, "stream").Bool())
	assert.Equal(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.True(t, gjson.GetBytes(out, "tools").Exists())
	assert.True(t, gjson.GetBytes(out, "parallel_tool_calls").Bool())
	assert.Equal(t, "shell", gjson.GetBytes(out, "tools.0.function.name").String())
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
	assert.False(t, gjson.GetBytes(out, "tools").Exists())
	assert.NotContains(t, string(out), "Planning mode is active. Do NOT execute tasks")
	assert.NotContains(t, string(out), `"name":"apply_patch"`)
}

func TestTranslateResponsesToChatCompletionsRequest_PlanModeInvestigateTheWebAfterAnsweredQuestionKeepsSearchTools(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"parallel_tool_calls":true,
		"tool_choice":"auto",
		"tools":[
			{"type":"function","function":{"name":"request_user_input","parameters":{"type":"object"}}},
			{"type":"web_search"}
		],
		"input":[
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.</collaboration_mode>"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."}]},
			{"type":"function_call","name":"request_user_input","call_id":"call_q_1","arguments":"{\"questions\":[\"How should the game be built?\"]}"},
			{"type":"function_call_output","call_id":"call_q_1","output":"{\"answers\":{\"quiz_format\":{\"answers\":[\"HTML/JS single file (Recommended)\"]}}}"}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.NotEqual(t, "none", gjson.GetBytes(out, "tool_choice").String())
	assert.Contains(t, string(out), `"name":"web_search"`)
	assert.NotContains(t, string(out), `Return exactly one complete <proposed_plan> block now`)
}

func TestTranslateResponsesToChatCompletionsRequest_SlashModePlanWebResearchKeepsSearchTools(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"parallel_tool_calls":true,
		"tool_choice":"auto",
		"tools":[
			{"type":"web_search"},
			{"type":"function","function":{"name":"shell"}}
		],
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"/mode plan\nInvestigate the web for a first grade quiz game, define a plan. Use web search and return the final visible answer as exactly one <proposed_plan>...</proposed_plan> block."}]}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, "auto", gjson.GetBytes(out, "tool_choice").String())
	assert.Contains(t, string(out), `"name":"web_search"`)
	assert.Contains(t, string(out), `"name":"shell"`)
	assert.NotContains(t, string(out), `"tool_choice":"none"`)
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
		"reasoning":{"effort":"low"},
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
		"reasoning":{"effort":"medium"},
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
	assert.False(t, gjson.GetBytes(out, "logit_bias.248069").Exists())
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

func TestTranslateResponsesToChatCompletionsRequest_ReasoningSummaryAutoEnablesThinking(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"reasoning":{"summary":"auto"},
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain attention"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "true", gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Raw)
}

func TestTranslateResponsesToChatCompletionsRequest_StripsUnsupportedReasoningEncryptedInclude(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"include":["reasoning.encrypted_content","foo.bar"],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Explain attention"}]}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var translated map[string]any
	require.NoError(t, json.Unmarshal(out, &translated))
	_, hasInclude := translated["include"]
	assert.False(t, hasInclude)
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

func TestEnforcePlanModeResponse_StripsThinkingTagFromVisiblePlanText(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_thinking",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"<thinking>private notes</thinking>\n1. Inspect the file.\n2. Draft the update.\n3. Validate results."}]
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.NotContains(t, text, "<thinking>")
	assert.Contains(t, text, "<proposed_plan>")
	assert.Contains(t, text, "1. Inspect the file.")
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
	assert.Contains(t, text, "Let me first check if there is existing project context.")
	assert.NotContains(t, text, "Planning mode is active. Here is a structured plan only:")
}

func TestEnforcePlanModeResponse_DoesNotInjectGenericPlanForPlaceholderReply(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_placeholder",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Working on the request."}]
			}
		]
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Equal(t, "Working on the request.", text)
	assert.NotContains(t, text, "Planning mode is active. Here is a structured plan only:")
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

func TestEnforcePlanModeResponse_DoesNotSynthesizePlanFromReasoningSummaryOnly(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_reasoning",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"rs_1",
				"type":"reasoning",
				"summary":[
					{
						"type":"summary_text",
						"text":"The user has answered:\n- Tech: HTML + JavaScript (Recommended)\n- Features: Classic quiz (Recommended)\n- Style: Colorful & Fun\n\nLet me create a single HTML file with:\n- Multiple choice answers\n- Score tracking\n- Progress bar"
					}
				]
			},
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Got it — HTML + JavaScript, classic quiz format, colorful and fun. Let me build it now."}]
			}
		],
		"output_text":"Got it — HTML + JavaScript, classic quiz format, colorful and fun. Let me build it now."
	}`)

	out := enforcePlanModeResponse(body, true)
	assert.JSONEq(t, string(body), string(out))
}

func TestEnforcePlanModeResponse_PrefersVisiblePlanBlockOverReasoningSummary(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_native_visible",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"rs_1",
				"type":"reasoning",
				"summary":[
					{
						"type":"summary_text",
						"text":"The user has answered the questions. Now I have clear direction:\n1. Plain HTML/CSS/JS\n2. Local JSON files for quiz storage\n3. Solo play first"
					}
				]
			},
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Perfect—plain HTML/CSS/JS, local JSON quizzes, and solo play first.\n\n<proposed_plan>\n## First-Grade Quiz Game\n1. Build the UI\n2. Load local JSON quizzes\n3. Validate solo play flow\n</proposed_plan>"}]
			}
		],
		"output_text":"Perfect—plain HTML/CSS/JS, local JSON quizzes, and solo play first.\n\n<proposed_plan>\n## First-Grade Quiz Game\n1. Build the UI\n2. Load local JSON quizzes\n3. Validate solo play flow\n</proposed_plan>"
	}`)

	out := enforcePlanModeResponse(body, true)
	text := gjson.GetBytes(out, "output.0.content.0.text").String()
	assert.Contains(t, text, "## First-Grade Quiz Game")
	assert.Contains(t, text, "Load local JSON quizzes")
	assert.NotContains(t, text, "The user has answered the questions. Now I have clear direction")
}

func TestEnforcePlanModeResponse_DoesNotWrapResearchSummaryWithoutVisiblePlanBlock(t *testing.T) {
	body := []byte(`{
		"id":"resp_plan_research_summary",
		"object":"response",
		"status":"completed",
		"output":[
			{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"I've gathered great research from several quiz platforms.\n\nKey findings from the web:\n- Big buttons\n- Bright colors\n- Short questions"}]
			}
		],
		"output_text":"I've gathered great research from several quiz platforms.\n\nKey findings from the web:\n- Big buttons\n- Bright colors\n- Short questions"
	}`)

	out := enforcePlanModeResponse(body, true)
	assert.JSONEq(t, string(body), string(out))
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

func TestBuildResponsesBridgeHandler_ForwardsNativeStreamWhenSafe_CapturesAllBridgeStages(t *testing.T) {
	mm := newMetricsMonitor(testLogger, 10, 5)
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Reply exactly STREAM_TEST_OK"}]}]
	}`)

	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"},\"finish_reason\":null}]}\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"STREAM_TEST_OK\"},\"finish_reason\":null}]}\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.2", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	err := mm.wrapHandler("gpt-5.2", ginCtx.Writer, req, handler)
	require.NoError(t, err)

	metrics := mm.getMetrics()
	if assert.Len(t, metrics, 1) {
		capture := mm.getCaptureByID(metrics[0].ID)
		if assert.NotNil(t, capture) {
			if assert.Len(t, capture.Stages, 5) {
				assert.Equal(t, "bridge.responses_request", capture.Stages[0].Name)
				assert.Equal(t, "bridge.chat_completions_request", capture.Stages[1].Name)
				assert.Equal(t, "bridge.qwen_stream_normalization", capture.Stages[2].Name)
				assert.Equal(t, "bridge.chat_completions_response", capture.Stages[3].Name)
				assert.Equal(t, "bridge.responses_output", capture.Stages[4].Name)
			}
			assert.Contains(t, string(capture.Stages[2].Payload), `"frames"`)
			assert.Contains(t, string(capture.Stages[3].Payload), "chat.completion.chunk")
			assert.Contains(t, string(capture.Stages[4].Payload), "response.created")
			assert.Contains(t, string(capture.Stages[4].Payload), "response.completed")
		}
	}
}

func TestBuildResponsesBridgeHandler_StreamApplyPatchIntentUsesEvaluatedPath(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":true,
		"tools":[{"type":"apply_patch"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"patch file now"}]}]
	}`)

	attempts := 0
	var upstreamBodies [][]byte
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		upstreamBodies = append(upstreamBodies, raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if attempts == 1 {
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-first",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"commands\":[\"Set-Content README.md hi\"]}"}}]},"finish_reason":"tool_calls"}]
			}`))
			return nil
		}
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-second",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_2","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}}]},"finish_reason":"tool_calls"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	req.Header.Set("Accept", "text/event-stream")
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 2, attempts)
	require.Len(t, upstreamBodies, 2)
	assert.Equal(t, false, gjson.GetBytes(upstreamBodies[0], "stream").Bool())
	assert.Equal(t, false, gjson.GetBytes(upstreamBodies[1], "stream").Bool())
	assert.NotContains(t, string(upstreamBodies[0]), "Strict apply_patch recovery mode")
	assert.Contains(t, string(upstreamBodies[0]), applyPatchTailConstraintText)
	assert.Contains(t, string(upstreamBodies[1]), "Strict apply_patch recovery mode")
	assert.Contains(t, rec.Body.String(), "event: response.created")
	assert.Contains(t, rec.Body.String(), `"type":"custom_tool_call"`)
	assert.Contains(t, rec.Body.String(), `"name":"apply_patch"`)
	assert.Contains(t, rec.Body.String(), "event: response.completed")
	assert.NotContains(t, rec.Body.String(), "chat.completion.chunk")
}

func TestBuildResponsesBridgeHandler_StreamShellFirstPromptDoesNotEnterStrictApplyPatchRetry(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"First use shell to inspect the current file at /tmp/demo.txt. Then use apply_patch to append one new line: PATCH_OK. Then verify with shell and reply only DONE."}]}]
	}`)

	attempts := 0
	var upstreamBodies [][]byte
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		upstreamBodies = append(upstreamBodies, raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-shellfirst",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"/tmp/demo.txt\",\"content\":\"PATCH_OK\\n\"}}"}}]},"finish_reason":"tool_calls"}]
		}`))
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.2", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	req.Header.Set("Accept", "text/event-stream")
	rec := httptest.NewRecorder()
	err := handler("gpt-5.2", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 1, attempts)
	require.Len(t, upstreamBodies, 1)
	assert.NotContains(t, string(upstreamBodies[0]), "Strict apply_patch recovery mode")
	assert.Contains(t, rec.Body.String(), `"name":"exec_command"`)
	assert.Contains(t, rec.Body.String(), `/tmp/demo.txt`)
	assert.NotContains(t, rec.Body.String(), `"name":"apply_patch"`)
}

func TestBuildResponsesBridgeHandler_StreamWebSearchResolvesLocally(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "OpenAI latest news", r.URL.Query().Get("q"))
		assert.Equal(t, "json", r.URL.Query().Get("format"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"results":[
				{"title":"OpenAI update","url":"https://openai.com/news","content":"Latest update text"}
			]
		}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":true,
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Find current OpenAI news"}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if attempts == 1 {
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-1",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_web_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"OpenAI latest news\"}"}}]},"finish_reason":"tool_calls"}]
			}`))
			return nil
		}
		assert.Contains(t, string(raw), "web_search_call_output")
		assert.Contains(t, string(raw), "Latest update text")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-web-2",
			"model":"qwen-test",
			"choices":[{"message":{"role":"assistant","content":"FINAL_WEB_OK"},"finish_reason":"stop"}]
		}`))
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
	assert.Contains(t, rec.Body.String(), "event: response.completed")
	assert.Contains(t, rec.Body.String(), "FINAL_WEB_OK")
	assert.Contains(t, rec.Body.String(), "web_search_call_output")
}

func TestBuildResponsesBridgeHandler_RetriesMissingWebSearchCallBeforeFinalizing(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"title":"Recovered Search Result","url":"https://example.com"}]}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Research current OpenAI responses API news"}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			assert.Contains(t, bodyText, `"name":"web_search"`)
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-miss-1",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"Working on the request.","reasoning_content":"I should search the web for current OpenAI responses API information before answering."},"finish_reason":"stop"}]
			}`))
		case 2:
			assert.Contains(t, bodyText, "emit exactly one real web_search tool call next")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-miss-2",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_web_retry_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"Research current OpenAI responses API news\"}"}}]},"finish_reason":"tool_calls"}]
			}`))
		default:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-miss-3",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"FINAL_WEB_RETRY_OK"},"finish_reason":"stop"}]
			}`))
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 3, attempts)
	assert.Contains(t, rec.Body.String(), "FINAL_WEB_RETRY_OK")
	assert.Contains(t, rec.Body.String(), "web_search_call_output")
}

func TestBuildResponsesBridgeHandler_RetriesInvalidApplyPatchOperationBeforeSurfacingWarning(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"apply_patch"},{"type":"shell"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Create quiz/index.html with apply_patch."}]}]
	}`)

	attempts := 0
	var upstreamBodies [][]byte
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		upstreamBodies = append(upstreamBodies, raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-ap-1",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_ap_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"quiz/index.html\",\"type\":\"create_file\"}}"}}]},"finish_reason":"tool_calls"}]
			}`))
		default:
			assert.Contains(t, string(raw), "Strict apply_patch recovery mode")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-ap-2",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_ap_2","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"create_file\",\"path\":\"quiz/index.html\",\"content\":\"<html></html>\"}}"}}]},"finish_reason":"tool_calls"}]
			}`))
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 2, attempts)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"path":"quiz/index.html"`)
	assert.Contains(t, rec.Body.String(), `"type":"create_file"`)
	assert.NotContains(t, rec.Body.String(), "apply_patch call was not executed because operation was invalid")
}

func TestBuildResponsesBridgeHandler_RetriesMissingBrowserOpenCallBeforeFinalizing(t *testing.T) {
	pm := &ProxyManager{}
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"tools":[{"type":"shell"},{"type":"apply_patch"}],
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Create a quiz app, verify it, and open it in the default browser."}]},
			{"type":"function_call_output","call_id":"call_patch_1","output":"Patch applied."},
			{"type":"function_call_output","call_id":"call_shell_1","output":"index.html style.css script.js"}
		]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-browser-1",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"Let's open it in your default browser!","reasoning_content":"All three files are created successfully. Let me open it in the browser so the user can see it working."},"finish_reason":"stop"}]
			}`))
		default:
			assert.Contains(t, bodyText, "open the finished local artifact in the user's default browser")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-browser-2",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_shell_open_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"powershell.exe -Command Start-Process quiz/index.html\"}"}}]},"finish_reason":"tool_calls"}]
			}`))
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 2, attempts)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"name":"shell"`)
	assert.NotContains(t, rec.Body.String(), `"output_text":"Let's open it in your default browser!"`)
}

func TestTranslateChatCompletionToResponsesResponse_StripsEmptyToolCodeMarkup(t *testing.T) {
	body := []byte(`{
		"id":"chatcmpl-empty-tool-code",
		"model":"qwen-test",
		"choices":[{
			"message":{
				"role":"assistant",
				"content":"<tool_code>\n</tool_code>",
				"reasoning_content":"I still need to search the web before finalizing."
			},
			"finish_reason":"stop"
		}]
	}`)

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(body, "", "", "", ToolWorkflowState{}, "", nil)
	require.NoError(t, err)

	assert.NotContains(t, string(out), "<tool_code>")
	assert.NotContains(t, gjson.GetBytes(out, "output_text").String(), "<tool_code>")
	assert.NotEmpty(t, strings.TrimSpace(gjson.GetBytes(out, "output_text").String()))
}

func TestBuildResponsesBridgeHandler_RetriesDeferredSearchPromiseWithoutToolCall(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"title":"A","url":"https://example.com/a","content":"alpha"}]}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"stream":false,
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Research first-grade quiz game ideas and mechanics"}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-deferred-1",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"Good! I now have more context. Let me also search for first-grade specific quiz game ideas and mechanics."},"finish_reason":"stop"}]
			}`))
		case 2:
			assert.Contains(t, bodyText, "emit exactly one real web_search tool call next")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-deferred-2",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_web_retry_2","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"first-grade quiz game ideas and mechanics\"}"}}]},"finish_reason":"tool_calls"}]
			}`))
		default:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-web-deferred-3",
				"model":"qwen-test",
				"choices":[{"message":{"role":"assistant","content":"FINAL_DEFERRED_SEARCH_OK"},"finish_reason":"stop"}]
			}`))
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.3-codex", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.3-codex", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 3, attempts)
	assert.Contains(t, rec.Body.String(), "FINAL_DEFERRED_SEARCH_OK")
	assert.Contains(t, rec.Body.String(), "web_search_call_output")
}

func TestBuildResponsesBridgeHandler_RepeatedWebSearchContinuationsFinalizeLastSearchWithoutBridge500(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"title":"A","url":"https://example.com/a","content":"alpha"}]}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"instructions":"<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.\n</collaboration_mode>",
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Investigate the web for first-grade quiz game ideas, then write the final plan. Do not implement it."}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game ideas educational\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_2\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game mechanics interactive learning games kids\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 3:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-3\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_3\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz reward systems stars badges classroom\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-3\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 4:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-4\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"FINAL_WEB_PLAN_OK\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-loop-4\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected extra attempt %d", attempts)
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 4, attempts)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"type":"web_search_call_output"`)
	assert.Contains(t, rec.Body.String(), `"call_id":"call_web_3"`)
	assert.Contains(t, rec.Body.String(), "FINAL_WEB_PLAN_OK")
	assert.NotContains(t, rec.Body.String(), "responses bridge failed after retries")
}

func TestBuildResponsesBridgeHandler_FinalizesMultiplePendingWebSearchCallsFromLastTurn(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"title":"A","url":"https://example.com/a","content":"alpha"}]}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"instructions":"<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.\n</collaboration_mode>",
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Investigate the web for first-grade quiz game ideas, then write the final plan. Do not implement it."}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game ideas educational\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_2\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game mechanics interactive learning games kids\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 3:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-3\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_3\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz reward systems stars badges classroom\\\"}\"}},{\"index\":1,\"id\":\"call_web_4\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"ABCya first grade quiz game features how it works\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-3\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 4:
			assert.Contains(t, bodyText, "web_search_call_output")
			assert.Contains(t, bodyText, "call_web_3")
			assert.Contains(t, bodyText, "call_web_4")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-4\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"FINAL_MULTI_WEB_PLAN_OK\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-multi-4\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected extra attempt %d", attempts)
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 4, attempts)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"type":"web_search_call_output"`)
	assert.Contains(t, rec.Body.String(), `"call_id":"call_web_3"`)
	assert.Contains(t, rec.Body.String(), `"call_id":"call_web_4"`)
	assert.Contains(t, rec.Body.String(), "FINAL_MULTI_WEB_PLAN_OK")
	assert.NotContains(t, rec.Body.String(), `"status":"requires_action"`)
	assert.NotContains(t, rec.Body.String(), "responses bridge failed after retries")
}

func TestBuildResponsesBridgeHandler_ForcedWebSearchSynthesisRecoversMalformedPseudoSearch(t *testing.T) {
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"title":"A","url":"https://example.com/a","content":"alpha"}]}`))
	}))
	defer searchServer.Close()

	pm := &ProxyManager{}
	pm.setWebSearchFallbackSettings(true, "searxng", searchServer.URL+"/search")
	body := []byte(`{
		"model":"gpt-5.4",
		"stream":false,
		"instructions":"<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.\n</collaboration_mode>",
		"tools":[{"type":"web_search"}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Investigate the web for first-grade quiz game ideas, then write the final plan. Do not implement it."}]}]
	}`)

	attempts := 0
	nextHandler := func(_ string, w http.ResponseWriter, r *http.Request) error {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		bodyText := string(raw)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		switch attempts {
		case 1:
			assert.Contains(t, bodyText, `"name":"web_search"`)
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-1\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game ideas educational\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-1\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-2\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_2\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"first grade quiz game mechanics kids rewards\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-2\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 3:
			assert.Contains(t, bodyText, "web_search_call_output")
			assert.Contains(t, bodyText, `"name":"web_search"`)
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-3\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_web_3\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"ABCya first grade quiz game features how it works\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-3\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 4:
			assert.Contains(t, bodyText, `"tool_choice":"none"`)
			assert.Contains(t, bodyText, "call_web_3")
			assert.Contains(t, bodyText, "web_search_call_output")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-4\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"<tool_call>\\n<function=web_search>\\n<parameter=query>\\nSplashLearn first grade quiz game mechanics rewards system\\n</parameter>\\n</function>\\n</tool_call>\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-4\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 5:
			assert.Contains(t, bodyText, `"tool_choice":"none"`)
			assert.Contains(t, bodyText, "web_search_call_output")
			assert.Contains(t, bodyText, "SplashLearn first grade quiz game mechanics rewards system")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-5\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"FINAL_RECOVERED_WEB_PLAN_OK\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-web-pseudo-5\",\"object\":\"chat.completion.chunk\",\"model\":\"Abiray-Qwen3.6-27B-NVFP4.gguf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected extra attempt %d", attempts)
		}
		return nil
	}

	handler := pm.buildResponsesBridgeHandler("gpt-5.4", body, nextHandler)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	err := handler("gpt-5.4", rec, req)
	require.NoError(t, err)

	assert.Equal(t, 5, attempts)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"type":"web_search_call_output"`)
	assert.Contains(t, rec.Body.String(), "SplashLearn first grade quiz game mechanics rewards system")
	assert.Contains(t, rec.Body.String(), "FINAL_RECOVERED_WEB_PLAN_OK")
	assert.NotContains(t, rec.Body.String(), "responses bridge failed after retries")
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
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "detailed")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.output_text.delta")
	assert.Contains(t, body, "event: response.completed")
	assert.Contains(t, body, "event: response.output_item.added")
	assert.Contains(t, body, `"type":"reasoning"`)
	assert.Contains(t, body, `"status":"in_progress"`)
	assert.Contains(t, body, "event: response.output_item.done")
	assert.Contains(t, body, `"status":"completed"`)
	assert.Contains(t, body, "event: response.reasoning_summary_part.added")
	assert.Contains(t, body, "event: response.reasoning_summary_part.done")
	assert.Contains(t, body, "event: response.reasoning_summary_text.delta")
	assert.Contains(t, body, "event: response.reasoning_summary_text.done")
	assert.Contains(t, body, `"summary_index":0`)
	assert.Contains(t, body, `"reasoning":{"summary":"detailed"}`)
	assert.Contains(t, body, `"summary_mode":"detailed"`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, "final answer")
	assert.Contains(t, body, "thinking step 1")
}

func TestWriteResponsesStreamFromChatSSE_StreamsReasoningIntoCommentaryAndKeepsFinalClean(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-workaround","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"The user wants a plan."},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-workaround","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Here's a plan for a small, fun math game:"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "detailed")
	require.NoError(t, err)

	body := rec.Body.String()
	preview := buildReasoningCommentaryPreview("The user wants a plan.")
	assert.Contains(t, body, `"output_index":0`)
	assert.Contains(t, body, `"type":"reasoning"`)
	assert.Contains(t, body, `"output_index":1`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, `"output_index":2`)
	assert.Contains(t, body, `"channel":"final"`)
	assert.Contains(t, body, `"reasoning":{"summary":"detailed"}`)
	assert.Contains(t, body, `"summary_mode":"detailed"`)
	assert.Contains(t, body, preview)
	assert.Contains(t, body, `Here's a plan for a small, fun math game:`)
	assert.NotContains(t, body, `</think>`)
}

func TestWriteResponsesStreamFromChatSSE_CommentaryUsesReasoningPreview(t *testing.T) {
	reasoning := "The user wants a plan. " + strings.Repeat("detail ", 50) + "tail marker"
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-workaround-preview","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"` + reasoning + `"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-workaround-preview","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Short final answer."},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "detailed")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"delta":"`+reasoning+`"`)
	assert.Contains(t, body, `"text":"`+reasoning+`"`)
	assert.NotContains(t, body, `"channel":"commentary","content":[{"type":"output_text","text":"`+reasoning+`"}]`)
	assert.Contains(t, body, `"text":"Working on the request."`)
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
	writeResponsesStream(rec, responseJSON, "detailed")

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.reasoning_summary_part.added")
	assert.Contains(t, body, "event: response.reasoning_summary_part.done")
	assert.Contains(t, body, "event: response.reasoning_summary_text.delta")
	assert.Contains(t, body, "event: response.reasoning_summary_text.done")
	assert.Contains(t, body, `"summary_index":0`)
	assert.Contains(t, body, `"status":"in_progress"`)
	assert.Contains(t, body, `"status":"completed"`)
	assert.Contains(t, body, `"reasoning":{"summary":"detailed"}`)
	assert.Contains(t, body, `"summary_mode":"detailed"`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, "thinking one")
	assert.Contains(t, body, "answer one")
}

func TestWriteResponsesStream_ReplaysReasoningItemsWithShortCommentaryPreview(t *testing.T) {
	reasoning := "thinking one " + strings.Repeat("detail ", 50) + "tail marker"
	preview := buildReasoningCommentaryPreview(reasoning)
	responseJSON := []byte(`{
	  "id":"resp_reasoning_stream_preview",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"completed",
	  "output":[
	    {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"` + reasoning + `"}]},
	    {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer one"}]}
	  ],
	  "output_text":"answer one"
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "detailed")

	body := rec.Body.String()
	assert.Contains(t, body, preview)
	assert.NotContains(t, body, `"channel":"commentary","content":[{"type":"output_text","text":"`+reasoning+`"}]`)
}

func TestWriteResponsesStream_EmitsProgressCommentaryWhenToolContinuationIsRequired(t *testing.T) {
	responseJSON := []byte(`{
	  "id":"resp_reasoning_tool_stream",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"requires_action",
	  "output":[
	    {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"Inspect file then patch it"}]},
	    {"id":"fc_1","type":"apply_patch_call","call_id":"call_1","operation":{"type":"update_file","path":"README.md","content":"PATCH_OK"}}
	  ]
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "detailed")

	body := rec.Body.String()
	assert.Contains(t, body, `"type":"reasoning"`)
	assert.Contains(t, body, `"type":"function_call"`)
	assert.Contains(t, body, `"name":"apply_patch"`)
	assert.Contains(t, body, `"status":"requires_action"`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, `Applying the requested file update.`)
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
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.function_call_arguments.delta")
	assert.Contains(t, body, "event: response.function_call_arguments.done")
	assert.Contains(t, body, `"name":"shell"`)
	assert.Contains(t, body, `"status":"requires_action"`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, "Running tool...")
}

func TestWriteResponsesStreamFromChatSSEWithWorkflow_RecoversRequestUserInputArgsAndUsesQuestionCommentary(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-rui-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"I should ask how they want this built.","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"request_user_input","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	originalReq := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use request_user_input in native question format and ask how they would like this built."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "ask the user how they would like this game built and which approach they prefer"},
				},
			},
		},
	}

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSEWithWorkflow(rec, strings.NewReader(upstream), false, true, "", ToolWorkflowState{}, "", originalReq)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"name":"request_user_input"`)
	assert.Contains(t, body, `How would you like this built, and which approach do you prefer?`)
	assert.Contains(t, body, `Preparing the clarification question.`)
	assert.NotContains(t, body, `Generating the requested plan`)
	assert.NotContains(t, body, `"delta":"{}"`)
	assert.NotContains(t, body, `"delta":"{\"raw\":\"{\"}"`)
}

func TestWriteResponsesStreamFromChatSSE_SuppressesCommentaryOnToolTurn(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-tool-reasoning","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"Inspecting the file before patching.","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "detailed")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"type":"reasoning"`)
	assert.Contains(t, body, `"type":"function_call"`)
	assert.Contains(t, body, `"name":"apply_patch"`)
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, "Applying the requested file update.")
}

func TestWriteResponsesStreamFromChatSSE_EmptyShellArgumentsBecomeValidationMessage(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-empty-shell","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.NotContains(t, body, "event: response.function_call_arguments.done")
	assert.NotContains(t, body, `"status":"requires_action"`)
	assert.NotContains(t, body, `"name":"shell"`)
	assert.Contains(t, body, shellValidationWarningPrefix)
	assert.Contains(t, body, "Provide a non-empty `command` string or `commands` array and retry.")
}

func TestWriteResponsesStreamFromChatSSE_EmitsEachReasoningChunkWithoutCoalescing(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-reasoning-chunks","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"first chunk "},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-reasoning-chunks","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"second chunk"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-reasoning-chunks","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"final answer"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "detailed")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"delta":"first chunk "`)
	assert.Contains(t, body, `"delta":"second chunk"`)
	assert.Equal(t, 2, strings.Count(body, "event: response.reasoning_summary_text.delta"))
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, `"text":"Working on the request."`)
	assert.Contains(t, body, "final answer")
}

func TestBuildIncrementalReasoningSummaryDelta_DoesNotReplayWholeSummaryOnRewrite(t *testing.T) {
	previous := "The user wants a plan for building a single, colorful HTML file that serves as a learning game for German and Spanish vocabulary. I need to structure this plan into a format that can be used ..."
	reasoning := "The user wants a plan for building a single, colorful HTML file that serves as a learning game for German and Spanish vocabulary. I need to structure this plan into a format that can be used to generate the code later, while keeping the explanation concise and practical for the requested task."

	summary, delta := buildIncrementalReasoningSummaryDelta(reasoning, previous)
	assert.NotEmpty(t, summary)
	assert.Empty(t, delta)
	assert.NotEqual(t, previous, summary)
}

func TestWriteResponsesStreamFromChatSSE_PreservesTimingsAndBackfillsUsage(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-metrics","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-metrics","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":3,"prompt_n":16,"prompt_ms":383.737,"prompt_per_second":41.69522355154702,"predicted_n":2,"predicted_ms":11.5,"predicted_per_second":17.818020910087416}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"type":"response.completed"`)
	assert.Contains(t, body, `"timings":{"cache_n":3,"predicted_ms":11.5,"predicted_n":2,"predicted_per_second":17.818020910087416,"prompt_ms":383.737,"prompt_n":16,"prompt_per_second":41.69522355154702}`)
	assert.Contains(t, body, `"usage":{"input_tokens":16,"input_tokens_details":{"cached_tokens":3},"output_tokens":2,"total_tokens":18}`)
}

func TestWriteResponsesStreamFromChatSSE_EmptyStopTurnDoesNotGetVisibleFallback(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-empty-stop","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.NotContains(t, body, "upstream returned no usable assistant content")
	assert.Contains(t, body, `"status":"completed"`)
	assert.Contains(t, body, `event: response.completed`)
}

func TestWriteResponsesStreamFromChatSSE_ApplyPatchUsesFreeformPatchArguments(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-ap","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"README.md\",\"content\":\"PATCH_OK\"}}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `event: response.function_call_arguments.done`)
	assert.Contains(t, body, `*** Begin Patch\n*** Update File: README.md`)
	assert.Contains(t, body, `"type":"function_call"`)
	assert.Contains(t, body, `"name":"apply_patch"`)
	assert.Contains(t, body, `+PATCH_OK`)
	assert.Contains(t, body, `\"operation\"`)
}

func TestWriteResponsesStreamFromChatSSE_ToolFirstUsesOutputIndexZeroWithoutEmptyMessage(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-tool-first","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"commands\":[\""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-tool-first","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"pwd\"]}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "event: response.created")
	assert.Contains(t, body, `"output_index":0`)
	assert.Contains(t, body, `"type":"function_call"`)
	assert.Contains(t, body, `"status":"requires_action"`)
	assert.NotContains(t, body, `"type":"message","role":"assistant"`)
	assert.NotContains(t, body, `event: response.content_part.added`)
	assert.NotContains(t, body, `event: response.output_text.done`)
}

func TestWriteResponsesStreamFromChatSSE_PlanModeWrapsOutputAsProposedPlan(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"1. Scope\n2. Risks\n3. Validation"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
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
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `finish_reason: \"length\"`)
	assert.Contains(t, body, "Retry with a higher output token limit")
	assert.Contains(t, body, "event: response.output_text.delta")
	assert.Contains(t, body, "event: response.completed")
}

func TestWriteResponsesStreamFromChatSSE_PlanModeBlocksApplyPatchAndRecoversPlanText(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-ap","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"c:\\\\Users\\\\YLAB-Partner\\\\Downloads\\\\qwentest\\\\plan.md\",\"type\":\"create_file\",\"content\":\"# Plan\\n\\n1. Scope\\n2. Risks\\n3. Validation\\n\\n</parameter></function> </tool_call> <tool_call>update_plan> <explanation>bad</explanation>\"}}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.NotContains(t, body, `"apply_patch_call"`)
	assert.NotContains(t, body, `"name":"apply_patch"`)
	assert.Contains(t, body, `\u003cproposed_plan\u003e`)
	assert.Contains(t, body, "1. Scope")
	assert.NotContains(t, body, "</parameter>")
	assert.NotContains(t, body, "<tool_call>")
	assert.Contains(t, body, `"status":"completed"`)
}

func TestWriteResponsesStreamFromChatSSE_DefaultPlanPrefaceStaysPlainWithoutVisiblePlanBlock(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-preface","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Great — I have everything I need. Here's the plan:"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	workflowState := ToolWorkflowState{
		HasToolOutput:      true,
		CompletedToolNames: []string{"request_user_input"},
	}
	err := writeResponsesStreamFromChatSSEWithWorkflow(rec, strings.NewReader(upstream), true, false, "", workflowState, "", nil)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "Great — I have everything I need. Here's the plan:")
	assert.NotContains(t, body, `\u003cproposed_plan\u003e`)
}

func TestWriteResponsesStreamFromChatSSE_PlanModeStripsFollowupQuestionsAndMalformedCloser(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-followup","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<proposed_plan>\n1. Scope\n2. Risks\n3. Validation\n\n---\n\nWant me to build this next?\nIf so, do you have any preferences?\n</parameter>"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `\u003cproposed_plan\u003e`)
	assert.Contains(t, body, `\u003c/proposed_plan\u003e`)
	assert.Contains(t, body, "1. Scope")
	assert.Contains(t, body, "2. Risks")
	assert.NotContains(t, body, "Want me to build this next?")
	assert.NotContains(t, body, "preferences?")
	assert.NotContains(t, body, "</parameter>")
}

func TestWriteResponsesStreamFromChatSSE_PlanModeUsesReasoningToBuildSpecificPlan(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-reasoning","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"The user has answered:\n- Tech: HTML + JavaScript (Recommended)\n- Features: Classic quiz (Recommended)\n- Style: Colorful & Fun\n\nLet me create a single HTML file with:\n- Multiple choice answers\n- Score tracking\n- Progress bar"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-plan-reasoning","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Got it — HTML + JavaScript, classic quiz format, colorful and fun. Let me build it now."},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `\u003cproposed_plan\u003e`)
	assert.Contains(t, body, "HTML + JavaScript")
	assert.Contains(t, body, "Score tracking")
	assert.NotContains(t, body, "Define scope and constraints")
}

func TestWriteResponsesStreamFromChatSSE_PlanModePrefersVisiblePlanOverReasoningSummary(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-visible","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"The user has answered the questions. Now I have clear direction:\n1. Plain HTML/CSS/JS\n2. Local JSON files for quiz storage\n3. Solo play first"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-plan-visible","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Perfect—plain HTML/CSS/JS, local JSON quizzes, and solo play first.\n\n<proposed_plan>\n## First-Grade Quiz Game\n1. Build the UI\n2. Load local JSON quizzes\n3. Validate solo play flow\n</proposed_plan>"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "## First-Grade Quiz Game")
	assert.Contains(t, body, "Load local JSON quizzes")
	assert.Contains(t, body, `"output_text":"\u003cproposed_plan\u003e\n## First-Grade Quiz Game`)
}

func TestWriteResponsesStreamFromChatSSE_PlanModeDoesNotWrapResearchSummaryWithoutVisiblePlanBlock(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-research","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"I've gathered great research from several quiz platforms.\n\nKey findings from the web:\n- Big buttons\n- Bright colors\n- Short questions"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "Key findings from the web:")
	assert.NotContains(t, body, `\u003cproposed_plan\u003e`)
}

func TestWriteResponsesStreamFromChatSSE_PlanModeSuppressesQuestionShapedShellRecovery(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-plan-shell","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"The user has answered:\n- Tech: HTML + JavaScript (Recommended)\n- Features: Classic quiz (Recommended)\n- Style: Colorful & Fun\n\nLet me create a single HTML file with:\n- Multiple choice answers\n- Score tracking\n- Progress bar"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-plan-shell","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":[\"What\",\"does\",\"synthesis\",\"gas\",\"consist\",\"of?\"]}"}}]},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), true, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `\u003cproposed_plan\u003e`)
	assert.Contains(t, body, "HTML + JavaScript")
	assert.NotContains(t, body, `event: response.function_call_arguments.done`)
	assert.NotContains(t, body, `"name":"shell"`)
	assert.Contains(t, body, `"status":"completed"`)
}

func TestWriteResponsesStreamFromChatSSE_PreservesWeakFinalWorkflowTextAfterVerification(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-weak-final","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Now verifying the final file content:"},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	workflowState := ToolWorkflowState{
		HasToolOutput:           true,
		CompletedToolNames:      []string{"shell", "apply_patch", "shell"},
		LatestCompletedToolName: "shell",
		ApplyPatchSatisfied:     true,
		VerificationExpected:    true,
		VerificationCompleted:   true,
		FinalAnswerSafe:         true,
	}
	err := writeResponsesStreamFromChatSSEWithWorkflow(rec, strings.NewReader(upstream), false, false, "", workflowState, "", nil)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "Now verifying the final file content:")
	assert.NotContains(t, body, "The requested file changes were applied and verified successfully.")
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
	assert.False(t, requestLooksLikePlanMode(reqSlash))
	assert.Equal(t, "plan", extractResponsesRequestMode(reqSlash))

	reqFlag := map[string]any{
		"instructions": "codex run --plan",
		"input":        []any{},
	}
	assert.True(t, requestLooksLikePlanMode(reqFlag))
}

func TestRequestLooksLikePlanMode_DoesNotInferPlanFromNaturalLanguageOnly(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch to update a file. Return only the plan.",
					},
				},
			},
		},
	}

	assert.False(t, requestLooksLikePlanMode(req))
	assert.Equal(t, "", extractResponsesRequestMode(req))
	assert.Equal(t, "default", responsesRequestContractMode(req))
}

func TestRequestLooksLikePlanMode_IgnoresUserOnlyPlanText(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Plan mode only. Do not execute tools. Return only the plan.",
					},
				},
			},
		},
	}

	assert.False(t, requestLooksLikePlanMode(req))
	assert.Equal(t, "", extractResponsesRequestMode(req))
}

func TestExtractResponsesRequestModeFromBody_DoesNotInferPlanFromNaturalLanguageOnly(t *testing.T) {
	body := []byte(`{
	  "model": "gpt-5.2",
	  "instructions": "You are Codex.",
	  "input": [{
	    "type": "message",
	    "role": "developer",
	    "content": [{
	      "type": "input_text",
	      "text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch. Return only the plan."
	    }]
	  }]
	}`)

	assert.Equal(t, "default", extractResponsesRequestModeFromBody(body))
	assert.False(t, rawResponsesBodyLooksLikePlanMode(body))
}

func TestRawResponsesBodyLooksLikePlanMode_UsesExplicitContractMarkersOnly(t *testing.T) {
	modeBody := []byte(`{"mode":"plan","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`)
	collabBody := []byte(`{"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.</collaboration_mode>"}]}]}`)
	naturalLanguageBody := []byte(`{"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"Plan mode only. Return only the plan."}]}]}`)

	assert.True(t, rawResponsesBodyLooksLikePlanMode(modeBody))
	assert.True(t, rawResponsesBodyLooksLikePlanMode(collabBody))
	assert.False(t, rawResponsesBodyLooksLikePlanMode(naturalLanguageBody))
}

func TestExtractResponsesRequestModeFromBody_IgnoresUserOnlyPlanNaturalLanguage(t *testing.T) {
	body := []byte(`{
	  "model": "gpt-5.2",
	  "instructions": "You are Codex.",
	  "input": [{
	    "type": "message",
	    "role": "user",
	    "content": [{
	      "type": "input_text",
	      "text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch. Return only the plan."
	    }]
	  }]
	}`)

	assert.Equal(t, "default", extractResponsesRequestModeFromBody(body))
}

func TestShouldEnableStrictApplyPatchIntent_DisabledForPlanOnlyPrompt(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch to append a line and then verify it with shell. Return only the plan.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "apply_patch"},
		},
		"tool_choice": map[string]any{"type": "apply_patch"},
	}

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, nil))
}

func TestMaybeForceRetryToolChoice_DoesNotNarrowPlanOnlyPrompt(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch. Return only the plan.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	maybeForceRetryToolChoice(req, `name="apply_patch"`)

	tools, ok := req["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 2)
	_, hasToolChoice := req["tool_choice"]
	assert.False(t, hasToolChoice)
}

func TestBuildApplyPatchRetryLoopDecision(t *testing.T) {
	t.Run("normalizes empty reason to no_tool_call", func(t *testing.T) {
		decision := buildApplyPatchRetryLoopDecision("", "")
		assert.Equal(t, "no_tool_call", decision.ReasonCode)
		assert.True(t, decision.ForceStrict)
		assert.Contains(t, decision.FailureText, "`no_tool_call`")
	})

	t.Run("prefers apply patch warning for wrong tool call", func(t *testing.T) {
		decision := buildApplyPatchRetryLoopDecision("wrong_tool_call", "sampled")
		assert.True(t, decision.ForceStrict)
		assert.Contains(t, decision.FailureText, applyPatchRetryPreferredFailureText)
		assert.Contains(t, decision.FailureText, "Sampled arguments: sampled")
	})

	t.Run("does not force strict on unrelated reason", func(t *testing.T) {
		decision := buildApplyPatchRetryLoopDecision("local_exec_error", "")
		assert.False(t, decision.ForceStrict)
		assert.Contains(t, decision.FailureText, "`local_exec_error`")
	})
}

func TestAppendApplyPatchFirstAttemptConstraint_DoesNotMutatePlanOnlyPrompt(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Plan mode only. Do not execute tools. Explain how you would use apply_patch and return only the plan.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	appendApplyPatchFirstAttemptConstraint(req)

	tools, ok := req["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 2)
	_, hasToolChoice := req["tool_choice"]
	assert.False(t, hasToolChoice)
	assert.NotContains(t, mustJSONString(req), applyPatchTailConstraintText)
}

func TestAppendApplyPatchFirstAttemptConstraint_PrefersShellInspectionBeforePatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First use shell to inspect the current file at /tmp/demo.txt, then use apply_patch to append one line and verify with shell.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	appendApplyPatchFirstAttemptConstraint(req)

	tools, ok := req["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Equal(t, "shell", fmt.Sprintf("%v", tools[0].(map[string]any)["name"]))
	_, hasToolChoice := req["tool_choice"]
	assert.False(t, hasToolChoice)
	assert.Contains(t, mustJSONString(req), "first tool call must be a shell read/inspection")
	assert.NotContains(t, mustJSONString(req), applyPatchTailConstraintText)
}

func TestPlanModeBlocksMutatingToolCall_ApplyPatchAndRunCommand(t *testing.T) {
	applyPatchResp := []byte(`{
	  "output":[
	    {"type":"function_call","name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"a.txt\"}}"}
	  ]
	}`)
	assert.True(t, planModeBlocksMutatingToolCall(applyPatchResp))

	runCommandResp := []byte(`{
	  "output":[
	    {"type":"function_call","name":"run_command","arguments":"{\"command\":\"touch out.txt\"}"}
	  ]
	}`)
	assert.True(t, planModeBlocksMutatingToolCall(runCommandResp))

	readOnlyResp := []byte(`{
	  "output":[
	    {"type":"function_call","name":"request_user_input","arguments":"{\"questions\":[]}"}
	  ]
	}`)
	assert.False(t, planModeBlocksMutatingToolCall(readOnlyResp))
}

func TestShouldEnforcePlanModeSyntheticRewrite_UsesPlanContractUntilCompliantFinalOutput(t *testing.T) {
	mutatingResp := []byte(`{
	  "output":[
	    {"type":"function_call","name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"a.txt\"}}"}
	  ]
	}`)
	plainTextResp := []byte(`{
	  "output":[
	    {"type":"message","content":[{"type":"output_text","text":"Got it — let me build it now."}]}
	  ]
	}`)
	planTextResp := []byte(`{
	  "output":[
	    {"type":"message","content":[{"type":"output_text","text":"<proposed_plan>\n1. Scope\n</proposed_plan>"}]}
	  ]
	}`)

	assert.False(t, shouldEnforcePlanModeSyntheticRewrite(true, false, mutatingResp))
	assert.True(t, shouldEnforcePlanModeSyntheticRewrite(true, false, plainTextResp))
	assert.False(t, shouldEnforcePlanModeSyntheticRewrite(true, false, planTextResp))
	assert.True(t, shouldEnforcePlanModeSyntheticRewrite(true, true, mutatingResp))
	assert.False(t, shouldEnforcePlanModeSyntheticRewrite(false, false, mutatingResp))
}

func TestTranslateResponsesToChatCompletionsRequest_CodexManagedPlanModeKeepsPlanInteractionTools(t *testing.T) {
	tools := []any{
		map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "update_plan"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "request_user_input"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "shell"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "list_mcp_resources"}},
	}

	filtered := removeMutatingPlanModeTools(tools, false)
	text := mustJSONString(filtered)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"update_plan"`)
	assert.Contains(t, text, `"name":"shell"`)
	assert.Contains(t, text, `"name":"list_mcp_resources"`)
}

func TestTranslateResponsesToChatCompletionsRequest_ProxyPlanModeStripsPlanInteractionTools(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "mode":"plan",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Please create a plan only."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, "do not call request_user_input or update_plan")
	assert.Contains(t, text, `"tool_choice":"none"`)
}

func TestRemoveMutatingPlanModeTools_StripsNamedMutatingToolsByTier(t *testing.T) {
	tools := []any{
		map[string]any{"type": "function", "name": "write_file"},
		map[string]any{"type": "function", "name": "shell_exec"},
		map[string]any{"type": "function", "name": "request_user_input"},
		map[string]any{"type": "function", "name": "shell"},
	}

	filtered := removeMutatingPlanModeTools(tools, true)
	serialized := mustJSONString(filtered)
	assert.NotContains(t, serialized, `"name":"write_file"`)
	assert.NotContains(t, serialized, `"name":"shell_exec"`)
	assert.Contains(t, serialized, `"name":"request_user_input"`)
	assert.Contains(t, serialized, `"name":"shell"`)
}

func TestRemoveMutatingPlanModeTools_StripsUnknownToolsFailClosed(t *testing.T) {
	tools := []any{
		map[string]any{"type": "function", "name": "mcp__filesystem__write_file"},
		map[string]any{"type": "function", "name": "shell"},
	}

	filtered := removeMutatingPlanModeTools(tools, true)
	serialized := mustJSONString(filtered)
	assert.NotContains(t, serialized, `"name":"mcp__filesystem__write_file"`)
	assert.Contains(t, serialized, `"name":"shell"`)
}

func TestRemoveMutatingPlanModeTools_CodexManagedPlanKeepsUnknownNativeTools(t *testing.T) {
	tools := []any{
		map[string]any{"type": "function", "name": "request_user_input"},
		map[string]any{"type": "function", "name": "list_mcp_resources"},
		map[string]any{"type": "function", "name": "view_image"},
		map[string]any{"type": "function", "name": "shell"},
		map[string]any{"type": "function", "name": "apply_patch"},
	}

	filtered := removeMutatingPlanModeTools(tools, false)
	serialized := mustJSONString(filtered)
	assert.Contains(t, serialized, `"name":"request_user_input"`)
	assert.Contains(t, serialized, `"name":"list_mcp_resources"`)
	assert.Contains(t, serialized, `"name":"view_image"`)
	assert.Contains(t, serialized, `"name":"shell"`)
	assert.NotContains(t, serialized, `"name":"apply_patch"`)
}

func TestTranslateResponsesToChatCompletionsRequest_DefaultModeKeepsRequestUserInput(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Ask one short clarifying question before coding."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.NotContains(t, text, "In Default mode, do not call request_user_input")
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_InjectsFileSearchWhenExplicitlyRequested(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.4",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use file_search to find the reward marker, then use apply_patch to create src/file-search-report.md."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"name":"file_search"`)
	assert.Contains(t, text, `"name":"apply_patch"`)
	assert.Contains(t, text, `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_ExplicitFileSearchForcesFileSearchOnlyFirstTurn(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.4",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use file_search to find the reward marker, then use apply_patch to create src/file-search-report.md. Do not use shell for the search step."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"name":"file_search"`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.Contains(t, text, `"tool_choice":{"function":{"name":"file_search"},"type":"function"}`)
}

func TestTranslateResponsesToChatCompletionsRequest_ExplicitPlaywrightBrowserTaskKeepsOnlyPlaywrightNamespace(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.4",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use Playwright browser tools to validate the local page and take a mobile screenshot."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {
	      "type":"namespace",
	      "name":"mcp__playwright__",
	      "tools":[
	        {"type":"function","name":"browser_navigate","parameters":{"type":"object"}},
	        {"type":"function","name":"browser_snapshot","parameters":{"type":"object"}},
	        {"type":"function","name":"browser_click","parameters":{"type":"object"}}
	      ]
	    }
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"name":"mcp__playwright__browser_navigate"`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.Contains(t, text, `"tool_choice":{"function":{"name":"mcp__playwright__browser_navigate"},"type":"function"}`)
}

func TestTranslateResponsesToChatCompletionsRequest_DefaultPlanRequestPrefersReturnedPlanOverUpdatePlan(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "input":[
	    {
	      "type":"message",
	      "role":"developer",
	      "content":[{"type":"input_text","text":"<collaboration_mode># Collaboration Mode: Default\nrequest_user_input is unavailable in Default mode.</collaboration_mode>"}]
	    },
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Plan a small HTML math game for children before building anything."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, "Do not call update_plan to present the user-facing plan")
	assert.NotContains(t, text, `"name":"update_plan"`)
	assert.NotContains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_PlanModeKeepsRequestUserInputButStripsUpdatePlan(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "input":[
	    {
	      "type":"message",
	      "role":"developer",
	      "content":[{"type":"input_text","text":"<collaboration_mode># Plan Mode (Conversational)\nUse request_user_input to ask questions that materially affect the plan.</collaboration_mode>"}]
	    },
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Research first-grade quiz topics, ask me a few questions, then write a plan."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.NotContains(t, text, `"name":"update_plan"`)
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_DefaultImplementTurnStillKeepsUpdatePlan(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "input":[
	    {
	      "type":"message",
	      "role":"developer",
	      "content":[{"type":"input_text","text":"<collaboration_mode># Collaboration Mode: Default\nrequest_user_input is unavailable in Default mode.</collaboration_mode>"}]
	    },
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Create hello.txt containing exactly HELLO using apply_patch. Do not use shell for writing."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"custom","name":"apply_patch"},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.NotContains(t, text, "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>")
	assert.Contains(t, text, `"name":"update_plan"`)
	assert.Contains(t, text, `"name":"apply_patch"`)
}

func TestTranslateResponsesToChatCompletionsRequest_LatestDefaultTagDoesNotStripApplyPatch(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "instructions":"You are Codex.",
	  "input":[
	    {
	      "type":"message",
	      "role":"developer",
	      "content":[{"type":"input_text","text":"<collaboration_mode># Plan Mode (Conversational)\nPlan first.</collaboration_mode>"}]
	    },
	    {
	      "type":"message",
	      "role":"developer",
	      "content":[{"type":"input_text","text":"<collaboration_mode># Collaboration Mode: Default\nYou are now in Default mode.</collaboration_mode>"}]
	    },
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use apply_patch to create index.html."}]
	    }
	  ],
	  "tools":[
	    {"type":"custom","name":"shell"},
	    {"type":"custom","name":"apply_patch"},
	    {"type":"function","function":{"name":"request_user_input"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var req map[string]any
	require.NoError(t, json.Unmarshal(body, &req))
	assert.Equal(t, "", extractResponsesRequestMode(req))
	assert.Contains(t, string(out), `"name":"apply_patch"`)
	assert.Contains(t, string(out), `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_CodexManagedPlanModeDoesNotForceRequestUserInputTool(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "instructions":"<collaboration_mode># Plan Mode (Conversational)\nUse Codex plan mode.</collaboration_mode>",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Ask exactly one short clarifying question in the native Codex question format before drafting the plan."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.NotContains(t, text, "use the request_user_input tool instead of writing the questions as plain assistant text")
	assert.NotContains(t, text, "Return a native function call named request_user_input")
	assert.NotContains(t, text, "Arguments must contain a questions array with exactly one short question")
	assert.Contains(t, text, `"tool_choice":"auto"`)
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"update_plan"`)
	assert.Contains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
}

func TestTranslateResponsesToChatCompletionsRequest_T20NativeQuestionPromptDoesNotForceRequestUserInput(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "instructions":"<collaboration_mode># Plan Mode (Conversational)\nUse Codex plan mode.</collaboration_mode>",
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Ask me exactly one native Codex question before planning anything else, then wait for input. Include the unique marker T20_SENTINEL in the question text."}]
	    }
	  ],
	  "tools":[
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"update_plan"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":"auto"
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":"auto"`)
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.Contains(t, text, `"name":"update_plan"`)
	assert.Contains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.Equal(t, "", gjson.GetBytes(out, "chat_template_kwargs.enable_thinking").Raw)
}

func TestTranslateResponsesToChatCompletionsRequest_PrunesToolsToForcedSpecificToolChoice(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"request_user_input"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":{"type":"function","function":{"name":"request_user_input"}},
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Ask one native clarifying question."}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":{"function":{"name":"request_user_input"},"type":"function"}`)
	assert.Contains(t, text, `"name":"request_user_input"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.NotContains(t, text, `"name":"shell"`)
}

func TestTranslateResponsesToChatCompletionsRequest_PrunesStrictApplyPatchIntentTools(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"spawn_agent"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use apply_patch to update /tmp/demo.txt by appending PATCH_OK. Then reply exactly DONE."}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"name":"apply_patch"`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"spawn_agent"`)
}

func TestTranslateResponsesToChatCompletionsRequest_PrunesExplicitAgentOrchestrationTools(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"spawn_agent"}},
	    {"type":"function","function":{"name":"send_input"}},
	    {"type":"function","function":{"name":"resume_agent"}},
	    {"type":"function","function":{"name":"wait_agent"}},
	    {"type":"function","function":{"name":"close_agent"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"mcp__playwright__browser_navigate"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Variation C: spawn one child, wait for completion, and if resume is needed use it explicitly before ending with exactly AGENT_DONE."}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"parallel_tool_calls":false`)
	assert.Contains(t, text, `"name":"spawn_agent"`)
	assert.Contains(t, text, `"name":"wait_agent"`)
	assert.Contains(t, text, `"name":"resume_agent"`)
	assert.Contains(t, text, `"name":"send_input"`)
	assert.Contains(t, text, `"name":"close_agent"`)
	toolNames := gjson.GetBytes(out, "tools.#.function.name").Array()
	serialized := make([]string, 0, len(toolNames))
	for _, name := range toolNames {
		serialized = append(serialized, name.String())
	}
	assert.NotContains(t, serialized, "apply_patch")
	assert.NotContains(t, serialized, "shell")
	assert.NotContains(t, serialized, "mcp__playwright__browser_navigate")
}

func TestTranslateResponsesToChatCompletionsRequest_KeepsAgentPruningAfterPriorToolOutput(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"spawn_agent"}},
	    {"type":"function","function":{"name":"send_input"}},
	    {"type":"function","function":{"name":"resume_agent"}},
	    {"type":"function","function":{"name":"wait_agent"}},
	    {"type":"function","function":{"name":"close_agent"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"mcp__playwright__browser_navigate"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Variation C: spawn one child, wait for completion, and if resume is needed use it explicitly before ending with exactly AGENT_DONE."}]
	    },
	    {
	      "type":"function_call",
	      "name":"shell",
	      "arguments":"{\"command\":[\"rg\",\"--files\"]}",
	      "call_id":"call_1"
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_1",
	      "output":"hello.txt"
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"parallel_tool_calls":false`)
	assert.Contains(t, text, `"name":"spawn_agent"`)
	assert.Contains(t, text, `"name":"wait_agent"`)
	assert.Contains(t, text, `"name":"resume_agent"`)
	assert.Contains(t, text, `"name":"send_input"`)
	assert.Contains(t, text, `"name":"close_agent"`)
	toolNames := gjson.GetBytes(out, "tools.#.function.name").Array()
	serialized := make([]string, 0, len(toolNames))
	for _, name := range toolNames {
		serialized = append(serialized, name.String())
	}
	assert.NotContains(t, serialized, "apply_patch")
	assert.NotContains(t, serialized, "shell")
	assert.NotContains(t, serialized, "mcp__playwright__browser_navigate")
}

func TestTranslateResponsesToChatCompletionsRequest_StripsToolsForTextOnlyExactReply(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"spawn_agent"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Reply exactly MT00_READY"}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":"none"`)
	assert.Contains(t, text, `"parallel_tool_calls":false`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.NotContains(t, text, `"name":"spawn_agent"`)
}

func TestTranslateResponsesToChatCompletionsRequest_StripsToolsForAcknowledgeAndReplyPrompt(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"spawn_agent"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Acknowledge receipt and reply with \"Child agent done.\""}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":"none"`)
	assert.Contains(t, text, `"parallel_tool_calls":false`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.NotContains(t, text, `"name":"spawn_agent"`)
}

func TestTranslateResponsesToChatCompletionsRequest_ContinuationKeepsExecutionSubset(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"spawn_agent"}},
	    {"type":"function","function":{"name":"wait_agent"}},
	    {"type":"function","function":{"name":"mcp__playwright__browser_navigate"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"First use shell to inspect demo.txt. Then use apply_patch to append one line: PATCH_OK. Then verify with shell and reply exactly DONE."}]
	    },
	    {
	      "type":"function_call",
	      "name":"shell",
	      "arguments":"{\"commands\":[\"Get-Content demo.txt\"]}",
	      "call_id":"call_1"
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_1",
	      "output":"BASE"
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	toolNames := gjson.GetBytes(out, "tools.#.function.name").Array()
	serialized := make([]string, 0, len(toolNames))
	for _, name := range toolNames {
		serialized = append(serialized, name.String())
	}
	assert.Contains(t, serialized, "shell")
	assert.Contains(t, serialized, "apply_patch")
	assert.NotContains(t, serialized, "spawn_agent")
	assert.NotContains(t, serialized, "wait_agent")
	assert.NotContains(t, serialized, "mcp__playwright__browser_navigate")
}

func TestTranslateResponsesToChatCompletionsRequest_ContinuationAfterApplyPatchDoesNotReopenFullCatalog(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"spawn_agent"}},
	    {"type":"function","function":{"name":"mcp__playwright__browser_navigate"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Use apply_patch to update demo.txt by appending one line PATCH_OK, then verify with shell and reply exactly DONE."}]
	    },
	    {
	      "type":"function_call",
	      "name":"apply_patch",
	      "arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"demo.txt\",\"content\":\"BASE\\nPATCH_OK\\n\"}}",
	      "call_id":"call_1"
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_1",
	      "output":"Success"
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	toolNames := gjson.GetBytes(out, "tools.#.function.name").Array()
	serialized := make([]string, 0, len(toolNames))
	for _, name := range toolNames {
		serialized = append(serialized, name.String())
	}
	assert.Contains(t, serialized, "shell")
	assert.Contains(t, serialized, "apply_patch")
	assert.NotContains(t, serialized, "spawn_agent")
	assert.NotContains(t, serialized, "mcp__playwright__browser_navigate")
}

func TestTranslateResponsesToChatCompletionsRequest_ContinuationAfterApplyPatchForcesShellVerification(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"spawn_agent"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"First use shell to inspect demo.txt, then use apply_patch to append one line and verify with shell, then reply exactly DONE."}]
	    },
	    {
	      "type":"function_call",
	      "name":"shell",
	      "arguments":"{\"command\":[\"powershell.exe\",\"-Command\",\"Get-Content demo.txt\"]}",
	      "call_id":"call_1"
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_1",
	      "output":"BASE"
	    },
	    {
	      "type":"function_call",
	      "name":"apply_patch",
	      "arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"demo.txt\",\"content\":\"BASE\\nPATCH_OK\\n\"}}",
	      "call_id":"call_2"
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_2",
	      "output":"Success"
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":{"function":{"name":"shell"},"type":"function"}`)
	assert.Contains(t, text, "explicitly required verification with shell")
	assert.Contains(t, text, `"name":"shell"`)
	assert.NotContains(t, text, `"name":"spawn_agent"`)
	assert.NotContains(t, text, `"logit_bias"`)
}

func TestTranslateResponsesToChatCompletionsRequest_ExplicitNoMutationPlanDisablesTools(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}},
	    {"type":"function","function":{"name":"apply_patch"}},
	    {"type":"function","function":{"name":"update_plan"}}
	  ],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"Stay in plan mode for a hypothetical patch to repo_mirror/config.yaml, do not edit anything, and explicitly say you are not executing. End with exactly T22_SENTINEL."}]
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"tool_choice":"none"`)
	assert.NotContains(t, text, `"name":"apply_patch"`)
	assert.NotContains(t, text, `"name":"shell"`)
	assert.Contains(t, text, "Do not execute tools")
	assert.NotContains(t, text, `"logit_bias"`)
	assert.NotContains(t, text, `"grammar"`)
}

func TestTranslateResponsesToChatCompletionsRequest_DropsEmptyAssistantContinuationMessages(t *testing.T) {
	body := []byte(`{
	  "model":"gpt-5.2",
	  "stream":false,
	  "tools":[
	    {"type":"function","function":{"name":"shell"}}
	  ],
	  "tool_choice":{"type":"function","function":{"name":"shell"}},
	  "parallel_tool_calls":true,
	  "input":[
	    {
	      "type":"message",
	      "role":"user",
	      "content":[{"type":"input_text","text":"First use shell to inspect demo.txt, then verify with shell and reply exactly DONE."}]
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_1",
	      "output":"BASE_A"
	    },
	    {
	      "type":"message",
	      "role":"assistant",
	      "content":[{"type":"output_text","text":""}]
	    },
	    {
	      "type":"function_call_output",
	      "call_id":"call_2",
	      "output":"Success. Updated the following files:\nM demo.txt"
	    }
	  ]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(out, &payload))
	messages, ok := payload["messages"].([]any)
	require.True(t, ok)
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		require.True(t, ok)
		role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["role"])))
		content := strings.TrimSpace(fmt.Sprintf("%v", msg["content"]))
		assert.False(t, role == "assistant" && content == "")
		assert.False(t, role == "assistant" && content == " ")
	}
}

func TestCollapsePreToolAssistantMessages_KeepsOnlyFirstAssistantMessageBeforeTool(t *testing.T) {
	output := []any{
		map[string]any{"type": "reasoning"},
		map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "first"}}},
		map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "second"}}},
		map[string]any{"type": "custom_tool_call", "name": "apply_patch"},
	}

	collapsed := collapsePreToolAssistantMessages(output)
	require.Len(t, collapsed, 3)
	assert.Equal(t, "reasoning", collapsed[0].(map[string]any)["type"])
	assert.Equal(t, "message", collapsed[1].(map[string]any)["type"])
	assert.Equal(t, "custom_tool_call", collapsed[2].(map[string]any)["type"])
	text := collapsed[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	assert.Equal(t, "first", text)
}

func TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromReasoning(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl_test_rui",
	  "model":"Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
	  "choices":[
	    {
	      "finish_reason":"tool_calls",
	      "index":0,
	      "message":{
	        "role":"assistant",
	        "content":"",
	        "reasoning_content":"I will use request_user_input.\nArgs: {\"questions\": [\"What task would you like me to plan?\"]}",
	        "tool_calls":[
	          {
	            "id":"call_123",
	            "type":"function",
	            "function":{
	              "name":"request_user_input",
	              "arguments":"{}"
	            }
	          }
	        ]
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, _ := resp["output"].([]any)
	require.Len(t, output, 2)

	call, ok := output[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "request_user_input", call["name"])
	assert.JSONEq(t, `{"questions":["What task would you like me to plan?"]}`, fmt.Sprintf("%v", call["arguments"]))
}

func TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromQuestionLine(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl_test_rui_question",
	  "model":"Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
	  "choices":[
	    {
	      "finish_reason":"tool_calls",
	      "index":0,
	      "message":{
	        "role":"assistant",
	        "content":"",
	        "reasoning_content":"I need to ask exactly one short clarifying question.\nQuestion: \"What would you like me to create a plan for?\"",
	        "tool_calls":[
	          {
	            "id":"call_456",
	            "type":"function",
	            "function":{
	              "name":"request_user_input",
	              "arguments":"{}"
	            }
	          }
	        ]
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, _ := resp["output"].([]any)
	require.Len(t, output, 2)

	call, ok := output[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call", call["type"])
	assert.Equal(t, "request_user_input", call["name"])
	assert.JSONEq(t, `{"questions":["What would you like me to create a plan for?"]}`, fmt.Sprintf("%v", call["arguments"]))
}

func TestTranslateResponsesToChatCompletionsRequest_DoesNotReAddApplyPatchForShellFirstPrompt(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.2",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First use shell to read /tmp/demo.txt. Then use apply_patch to append one line: PATCH_OK. Then verify with shell and reply exactly: DONE",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{
				"type": "shell",
				"name": "exec_command",
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var translated map[string]any
	require.NoError(t, json.Unmarshal(out, &translated))
	tools, ok := translated["tools"].([]any)
	require.True(t, ok)
	var names []string
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "shell")
	assert.NotContains(t, names, "apply_patch")
}

func TestAppendApplyPatchFirstAttemptConstraint_ShellFirstDeleteKeepsNativeDeleteGuidance(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First use shell to inspect /tmp/a.txt and /tmp/b.txt, then use apply_patch to delete file /tmp/a.txt and create file /tmp/c.txt.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	appendApplyPatchFirstAttemptConstraint(req)

	tools, ok := req["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Equal(t, "shell", fmt.Sprintf("%v", tools[0].(map[string]any)["name"]))
	body := mustJSONString(req)
	assert.Contains(t, body, "first tool call must be a shell read/inspection")
	assert.Contains(t, body, "operation.type=delete_file")
	assert.Contains(t, body, "not shell rm/del")
}

func TestCanonicalToolNameFromRecipient_CollaborationTools(t *testing.T) {
	assert.Equal(t, "spawn_agent", canonicalToolNameFromRecipient("functions.spawn_agent"))
	assert.Equal(t, "send_input", canonicalToolNameFromRecipient("functions.send_input"))
	assert.Equal(t, "resume_agent", canonicalToolNameFromRecipient("functions.resume_agent"))
	assert.Equal(t, "wait_agent", canonicalToolNameFromRecipient("functions.wait_agent"))
	assert.Equal(t, "close_agent", canonicalToolNameFromRecipient("functions.close_agent"))
}

func TestParseQwenFunctionStyleToolCalls_CollaborationTools(t *testing.T) {
	calls, remaining := parseQwenFunctionStyleToolCalls(`resume_agent(id="child-123")`)
	require.Len(t, calls, 1)
	assert.Equal(t, "resume_agent", calls[0].Name)
	assert.Equal(t, "child-123", fmt.Sprintf("%v", calls[0].Arguments["id"]))
	assert.Empty(t, remaining)
}

func TestAppendSerializedAgentOrchestrationInstruction_SetsSerialGuidance(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use spawn_agent twice to create two child agents. Wait for the first child to finish before interacting with the second. Then wait for the second child to finish. Close both.",
					},
				},
			},
		},
	}

	appendSerializedAgentOrchestrationInstruction(req)

	assert.Equal(t, false, req["parallel_tool_calls"])
	body := mustJSONString(req)
	assert.Contains(t, body, "Agent orchestration mode")
	assert.Contains(t, body, "Do not say you are waiting unless you actually emit wait_agent")
	assert.Contains(t, body, "keep the sequence strictly serialized")
}

func TestNormalizeBridgeChatTools_FlattensNamespaceTools(t *testing.T) {
	tools := []any{
		map[string]any{
			"type":        "namespace",
			"name":        "mcp__playwright__",
			"description": "Tools in the mcp__playwright__ namespace.",
			"tools": []any{
				map[string]any{
					"type":        "function",
					"name":        "browser_navigate",
					"description": "Navigate to a URL",
					"strict":      false,
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"url": map[string]any{"type": "string"},
						},
						"required": []string{"url"},
					},
				},
				map[string]any{
					"type":        "function",
					"name":        "browser_click",
					"description": "Click an element",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"target": map[string]any{"type": "string"},
						},
						"required": []string{"target"},
					},
				},
			},
		},
	}

	got := normalizeBridgeChatTools(tools)
	require.Len(t, got, 2)

	fn0, ok := got[0].(map[string]any)["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "mcp__playwright__browser_navigate", fn0["name"])
	assert.Equal(t, "Navigate to a URL", fn0["description"])

	fn1, ok := got[1].(map[string]any)["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "mcp__playwright__browser_click", fn1["name"])
	assert.Equal(t, "Click an element", fn1["description"])
}

func TestTranslateResponsesToChatCompletionsRequest_PrependsNamespaceToolInstructionAndFlattens(t *testing.T) {
	reqBody := []byte(`{
	  "model":"gpt-5.2",
	  "instructions":"You are Codex.",
	  "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Use Playwright to open a local page and tell me the title."}]}],
	  "tools":[
	    {
	      "type":"namespace",
	      "name":"mcp__playwright__",
	      "description":"Tools in the mcp__playwright__ namespace.",
	      "tools":[
	        {"type":"function","name":"browser_navigate","description":"Navigate to a URL","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}},
	        {"type":"function","name":"browser_snapshot","description":"Snapshot the page","parameters":{"type":"object","properties":{} }}
	      ]
	    }
	  ],
	  "stream":true
	}`)

	out, err := translateResponsesToChatCompletionsRequest(reqBody)
	require.NoError(t, err)

	body := string(out)
	assert.Contains(t, body, `mcp__playwright__browser_navigate`)
	assert.Contains(t, body, `mcp__playwright__browser_snapshot`)
	assert.Contains(t, body, `Never call \"mcp__playwright__\" directly`)
}

func TestTranslateResponsesToChatCompletionsRequest_AddsPlaywrightBrowserGuidance(t *testing.T) {
	reqBody := []byte(`{
	  "model":"gpt-5.2",
	  "instructions":"You are Codex.",
	  "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Use the Playwright MCP browser tool to navigate to https://example.com and then call browser_snapshot."}]}],
	  "tools":[
	    {
	      "type":"namespace",
	      "name":"mcp__playwright__",
	      "description":"Tools in the mcp__playwright__ namespace.",
	      "tools":[
	        {"type":"function","name":"browser_navigate","description":"Navigate to a URL","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}},
	        {"type":"function","name":"browser_snapshot","description":"Snapshot the page","parameters":{"type":"object","properties":{} }},
	        {"type":"function","name":"browser_click","description":"Click an element","parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}}
	      ]
	    }
	  ],
	  "stream":true
	}`)

	out, err := translateResponsesToChatCompletionsRequest(reqBody)
	require.NoError(t, err)

	body := string(out)
	assert.Contains(t, body, `For Playwright or browser tasks, emit the exact MCP browser tool calls instead of stopping after reasoning.`)
	assert.Contains(t, body, `mcp__playwright__browser_navigate`)
	assert.Contains(t, body, `mcp__playwright__browser_snapshot`)
	assert.Contains(t, body, `do not finish the turn until the requested browser tool calls have been emitted`)
}

func TestResponseContainsApplyPatchCall_DetectsCustomToolCall(t *testing.T) {
	body := []byte(`{
		"output":[
			{"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch"}
		]
	}`)
	assert.True(t, responseContainsApplyPatchCall(body))
}

func TestExtractApplyPatchPathFromResponse_DetectsCustomToolCallOperationPath(t *testing.T) {
	body := []byte(`{
		"output":[
			{"type":"custom_tool_call","name":"apply_patch","operation":{"type":"update_file","path":"/tmp/demo.txt","content":"PATCH_OK"}}
		]
	}`)
	assert.Equal(t, "/tmp/demo.txt", extractApplyPatchPathFromResponse(body))
}

func TestWriteResponsesStream_CustomApplyPatchCallRebuildsWeakInputFromOperation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	responseJSON := []byte(fmt.Sprintf(`{
	  "id":"resp_apply_patch_custom",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"requires_action",
	  "output":[
	    {"id":"fc_1","type":"custom_tool_call","call_id":"call_1","name":"apply_patch","input":"*** Begin Patch\n*** Update File: %s\n@@\n TITLE=alpha\n ENV=dev\n DEBUG=true\n PORT=8080\n FOOTER=old\n+TITLE=beta\n+ENV=dev\n+MODE=prod\n+PORT=8080\n+FOOTER=done\n*** End Patch","operation":{"type":"update_file","path":"%s","content":"TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n","diff":"@@\n TITLE=alpha\n ENV=dev\n DEBUG=true\n PORT=8080\n FOOTER=old\n+TITLE=beta\n+ENV=dev\n+MODE=prod\n+PORT=8080\n+FOOTER=done"}}
	  ]
	}`, path, path))

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "")

	body := rec.Body.String()
	assert.Contains(t, body, `"channel":"commentary"`)
	assert.Contains(t, body, "Applying the requested file update.")
	assert.Contains(t, body, "*** Update File: "+path)
	assert.Contains(t, body, "+TITLE=beta")
	assert.Contains(t, body, "+MODE=prod")
	assert.Contains(t, body, `"type":"function_call"`)
}

func TestWriteResponsesStream_EmptyShellArgumentsBecomeValidationMessage(t *testing.T) {
	responseJSON := []byte(`{
	  "id":"resp_empty_shell",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"requires_action",
	  "output":[
	    {"id":"fc_1","type":"shell_call","call_id":"call_1","action":{}}
	  ]
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "")

	body := rec.Body.String()
	assert.NotContains(t, body, `"name":"shell"`)
	assert.NotContains(t, body, `event: response.function_call_arguments.done`)
	assert.Contains(t, body, shellValidationWarningPrefix)
	assert.Contains(t, body, "Provide a non-empty `command` string or `commands` array and retry.")
	assert.Contains(t, body, `"status":"completed"`)
	assert.NotContains(t, body, `"status":"requires_action"`)
}

func TestWriteResponsesStream_EmptyRequestUserInputArgumentsBecomeValidationMessage(t *testing.T) {
	responseJSON := []byte(`{
	  "id":"resp_empty_rui",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"requires_action",
	  "output":[
	    {"id":"fc_1","type":"function_call","call_id":"call_1","name":"request_user_input","arguments":"{}"}
	  ]
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "")

	body := rec.Body.String()
	assert.NotContains(t, body, `"name":"request_user_input"`)
	assert.Contains(t, body, "request_user_input was not executed because `questions` was empty")
	assert.Contains(t, body, `"status":"completed"`)
	assert.NotContains(t, body, `"status":"requires_action"`)
}

func TestTranslateChatCompletionToResponsesResponse_RejectsEmptyParallelToolUses(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl_parallel_empty",
	  "model":"Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
	  "choices":[
	    {
	      "finish_reason":"tool_calls",
	      "index":0,
	      "message":{
	        "role":"assistant",
	        "content":"",
	        "tool_calls":[
	          {
	            "id":"call_parallel_1",
	            "type":"function",
	            "function":{
	              "name":"multi_tool_use.parallel",
	              "arguments":"{\"tool_uses\":[]}"
	            }
	          }
	        ]
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)
	text := string(out)
	assert.NotContains(t, text, `"name":"multi_tool_use.parallel"`)
	assert.Contains(t, text, "multi_tool_use.parallel was not executed because `tool_uses` was empty")
}

func TestWriteResponsesStream_NormalizesShellCallToResponseSafeShape(t *testing.T) {
	responseJSON := []byte(`{
	  "id":"resp_shell_safe",
	  "object":"response",
	  "created_at":123,
	  "model":"qwen-test",
	  "status":"requires_action",
	  "output":[
	    {
	      "id":"fc_1",
	      "type":"shell_call",
	      "call_id":"call_1",
	      "action":{"command":["powershell.exe","-Command","Get-Content mutations/base_a.txt"]}
	    }
	  ]
	}`)

	rec := httptest.NewRecorder()
	writeResponsesStream(rec, responseJSON, "")

	body := rec.Body.String()
	assert.Contains(t, body, `"name":"shell"`)
	assert.Contains(t, body, `event: response.function_call_arguments.done`)
	assert.Contains(t, body, `\"command\":[\"powershell.exe\",\"-Command\",\"Get-Content mutations/base_a.txt\"]`)
	assert.NotContains(t, body, `\"arguments\":\"{\\\"command\\\":[\\\"powershell.exe\\\"],\\\"commands\\\":`)
}

func TestWriteResponsesStreamFromChatSSE_ApplyPatchUsesContentDrivenReplacementHunk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	argsJSON := mustJSONString(map[string]any{
		"operation": map[string]any{
			"type":    "create_file",
			"path":    path,
			"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
		},
	})
	upstream := strings.Join([]string{
		fmt.Sprintf(`data: {"id":"chatcmpl-ap","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch_1","type":"function","function":{"name":"apply_patch","arguments":%q}}]},"finish_reason":"tool_calls"}]}`, argsJSON),
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSE(rec, strings.NewReader(upstream), false, "")
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, "*** Update File: "+path)
	assert.Contains(t, body, "-TITLE=alpha")
	assert.Contains(t, body, "-DEBUG=true")
	assert.Contains(t, body, "+MODE=prod")
	assert.NotContains(t, body, "\n TITLE=alpha\n ENV=dev\n DEBUG=true")
}

func TestTranslateChatCompletionToResponsesResponseWithWorkflow_RecoversVerificationShellCallFromVerificationProse(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-verify-prose",
	  "object":"chat.completion",
	  "created":123,
	  "model":"qwen-test",
	  "choices":[
	    {
	      "index":0,
	      "finish_reason":"stop",
	      "message":{
	        "role":"assistant",
	        "content":"Now verifying the final file content.",
	        "reasoning_content":"The patch was applied successfully. Now I need to verify the final content."
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(
		body,
		"",
		"",
		"",
		ToolWorkflowState{
			VerificationExpected:  true,
			VerificationCompleted: false,
		},
		`{"command":["powershell.exe","-Command","Get-Content mutations/base_a.txt"]}`,
		nil,
	)
	require.NoError(t, err)

	jsonText := string(out)
	assert.Contains(t, jsonText, `"type":"function_call"`)
	assert.Contains(t, jsonText, `"name":"shell"`)
	assert.Contains(t, jsonText, `Get-Content mutations/base_a.txt`)
	assert.NotContains(t, jsonText, `Now verifying the final file content.`)
}

func TestTranslateChatCompletionToResponsesResponseWithWorkflow_PreservesVisibleProposedPlanBlock(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-plan-visible",
	  "object":"chat.completion",
	  "created":123,
	  "model":"qwen-test",
	  "choices":[
	    {
	      "index":0,
	      "finish_reason":"stop",
	      "message":{
	        "role":"assistant",
	        "content":"Based on the research, here's the plan.\n\n<proposed_plan>\n1. Research current first-grade quiz patterns.\n2. Define kid-friendly mechanics.\n3. Build the MVP in plain HTML/CSS/JS.\n</proposed_plan>",
	        "reasoning_content":"I should provide the visible plan block exactly as requested."
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(body, "", "", "", ToolWorkflowState{}, "", nil)
	require.NoError(t, err)

	text := gjson.GetBytes(out, "output.1.content.0.text").String()
	assert.Contains(t, text, "<proposed_plan>")
	assert.Contains(t, text, "Build the MVP in plain HTML/CSS/JS.")
	assert.Equal(t, "completed", gjson.GetBytes(out, "status").String())
}

func TestTranslateChatCompletionToResponsesResponseWithWorkflow_RecoversCodexSkillProofShellCallFromFalseFoundClaim(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-skill-found-claim",
	  "object":"chat.completion",
	  "created":123,
	  "model":"qwen-test",
	  "choices":[
	    {
	      "index":0,
	      "finish_reason":"stop",
	      "message":{
	        "role":"assistant",
	        "content":"Found it — let me look for the actual list-workspace-files definition.",
	        "reasoning_content":"I found it."
	      }
	    }
	  ]
	}`)

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "update $list-workspace-files and find where the skill is stored"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":["powershell.exe","-Command","Get-ChildItem $env:USERPROFILE/.codex -Recurse -Filter *workspace*"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "C:\\Users\\YLAB-Partner\\.codex\\.tmp\\plugins\\plugins\\plugin-eval\\src\\core\\benchmark-workspace.js",
			},
		},
	}

	out, err := translateChatCompletionToResponsesResponseWithWorkflow(
		body,
		"",
		"",
		"",
		ToolWorkflowState{VerificationExpected: true, VerificationCompleted: false},
		`{"command":["powershell.exe","-Command","$root = if ($env:CODEX_HOME) { Join-Path $env:CODEX_HOME 'skills' } else { Join-Path $env:USERPROFILE '.codex\\skills' }; Get-ChildItem -Path $root -Recurse -Filter 'SKILL.md' -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match '(?i)[\\\\/](list-workspace-files)[\\\\/]SKILL\\.md$' -or $_.FullName -match '(?i)list-workspace-files' } | Select-Object FullName"]}`,
		req,
	)
	require.NoError(t, err)

	jsonText := string(out)
	assert.Contains(t, jsonText, `"type":"function_call"`)
	assert.Contains(t, jsonText, `"name":"shell"`)
	assert.Contains(t, jsonText, `list-workspace-files`)
	assert.NotContains(t, jsonText, `Found it`)
}

func TestTranslateChatCompletionToResponsesResponse_RepairsStringifiedOperationFromReasoningXML(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-applypatch-opstring",
	  "object":"chat.completion",
	  "created":123,
	  "model":"qwen-test",
	  "choices":[
	    {
	      "index":0,
	      "finish_reason":"stop",
	      "message":{
	        "role":"assistant",
	        "content":"",
	        "reasoning_content":"</think>\n\nNow I'll append ORDERED_T11 to the file:\n\n<tool_call>\n<function=apply_patch>\n<parameter=operation>\n{\"content\": \"BASE_A\\nORDERED_T11\", \"path\": \"C:\\\\Users\\\\YLAB-Partner\\\\Desktop\\\\qwentest\\\\mutations\\\\base_a.txt\"}\n</parameter>\n</function>\n</tool_call>"
	      }
	    }
	  ]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "C:/Users/YLAB-Partner/Desktop/qwentest/mutations/base_a.txt", "", "update_file")
	require.NoError(t, err)

	text := string(out)
	assert.Contains(t, text, `"type":"apply_patch_call"`)
	assert.Contains(t, text, `"type":"update_file"`)
	assert.Contains(t, text, `C:/Users/YLAB-Partner/Desktop/qwentest/mutations/base_a.txt`)
	assert.NotContains(t, text, `apply_patch call was not executed because operation was invalid`)
}

func TestWriteResponsesStreamFromChatSSEWithWorkflow_RecoversVerificationShellCallFromVerificationProse(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-verify-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Now verifying the final file content."},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	rec := httptest.NewRecorder()
	err := writeResponsesStreamFromChatSSEWithWorkflow(
		rec,
		strings.NewReader(upstream),
		false,
		false,
		"",
		ToolWorkflowState{
			VerificationExpected:  true,
			VerificationCompleted: false,
		},
		`{"command":["powershell.exe","-Command","Get-Content mutations/base_a.txt"]}`,
		nil,
	)
	require.NoError(t, err)

	body := rec.Body.String()
	assert.Contains(t, body, `"name":"shell"`)
	assert.Contains(t, body, `event: response.function_call_arguments.done`)
	assert.Contains(t, body, `Get-Content mutations/base_a.txt`)
	assert.NotContains(t, body, `Now verifying the final file content.`)
}

func TestBuildApplyPatchInputFromOperation_IgnoresWeakDiffWhenContentNeedsReplacement(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	input := buildApplyPatchInputFromOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"diff":    "@@\n TITLE=alpha\n ENV=dev\n DEBUG=true\n PORT=8080\n FOOTER=old\n+TITLE=beta\n+ENV=dev\n+MODE=prod\n+PORT=8080\n+FOOTER=done",
		"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	assert.Contains(t, input, "-TITLE=alpha")
	assert.Contains(t, input, "-DEBUG=true")
	assert.Contains(t, input, "+MODE=prod")
	assert.NotContains(t, input, "\n TITLE=alpha\n ENV=dev\n DEBUG=true")
}

func TestNormalizeApplyPatchOperation_RebuildsLooseDiffBodyAgainstExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	require.NoError(t, os.WriteFile(path, []byte("TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"), 0o644))

	op := normalizeApplyPatchOperation(map[string]any{
		"type": "update_file",
		"path": path,
		"diff": "TITLE=alpha\n+TITLE=beta\n ENV=dev\n-DEBUG=true\n+MODE=prod\n PORT=8080\n-FOOTER=old\n+FOOTER=done",
	}).(map[string]any)

	assert.Equal(t, "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done", strings.TrimSpace(fmt.Sprintf("%v", op["content"])))
	assert.NotContains(t, fmt.Sprintf("%v", op["diff"]), "++TITLE=beta")
	input := buildApplyPatchInputFromOperation(op)
	assert.Contains(t, input, "-TITLE=alpha")
	assert.Contains(t, input, "+TITLE=beta")
}

func TestBuildHeuristicUpdatePatchFromExistingFile_StructuredLineReplacementDoesNotAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.txt")
	require.NoError(t, os.WriteFile(path, []byte("PORT=3000\nDEBUG=true\n"), 0o644))

	patch := buildHeuristicUpdatePatchFromExistingFile(path, "PORT=8080")

	assert.Contains(t, patch, "-PORT=3000")
	assert.Contains(t, patch, "-DEBUG=true")
	assert.Contains(t, patch, "+PORT=8080")
	assert.NotContains(t, patch, "\n PORT=3000\n DEBUG=true\n+PORT=8080")
}

func TestBuildHeuristicUpdatePatchFromExistingFile_SingleLineReplacementDoesNotAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "single.txt")
	require.NoError(t, os.WriteFile(path, []byte("OLD_LINE\n"), 0o644))

	patch := buildHeuristicUpdatePatchFromExistingFile(path, "NEW_LINE")

	assert.Contains(t, patch, "-OLD_LINE")
	assert.Contains(t, patch, "+NEW_LINE")
	assert.NotContains(t, patch, "\n OLD_LINE\n+NEW_LINE")
}

func TestBuildHeuristicUpdatePatchFromExistingFile_FullContentRewriteUsesDeleteAndAdd(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	patch := buildHeuristicUpdatePatchFromExistingFile(path, "BASE_A\nORDERED_T11")

	assert.Contains(t, patch, "-BASE_A")
	assert.Contains(t, patch, "+BASE_A")
	assert.Contains(t, patch, "+ORDERED_T11")
}

func TestBuildHeuristicUpdatePatchFromContentOnly_FullContentRewriteUsesDeleteAndAdd(t *testing.T) {
	patch := buildHeuristicUpdatePatchFromContentOnly("mutations/base_a.txt", "BASE_A\nORDERED_T11")

	assert.Contains(t, patch, "*** Update File: mutations/base_a.txt")
	assert.Contains(t, patch, "-BASE_A")
	assert.Contains(t, patch, "+BASE_A")
	assert.Contains(t, patch, "+ORDERED_T11")
}

func TestBuildApplyPatchInputFromOperation_ContentOnlyFallbackUsesReplacementPatch(t *testing.T) {
	patch := buildApplyPatchInputFromOperation(map[string]any{
		"type":    "update_file",
		"path":    "mutations/base_a.txt",
		"content": "BASE_A\nORDERED_T11",
	})

	assert.Contains(t, patch, "*** Update File: mutations/base_a.txt")
	assert.Contains(t, patch, "-BASE_A")
	assert.Contains(t, patch, "+BASE_A")
	assert.Contains(t, patch, "+ORDERED_T11")
}

func TestRecoverContentFromWeakApplyPatchUpdate_StripsLeakedPlusFromAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "append.txt")
	require.NoError(t, os.WriteFile(path, []byte("LINE1\nLINE2\n"), 0o644))

	recovered, ok := recoverContentFromWeakApplyPatchUpdate(path, "@@\n LINE1\n LINE2\n++FM_OK")

	require.True(t, ok)
	assert.Equal(t, "LINE1\nLINE2\nFM_OK\n", recovered)
}

func TestRecoverContentFromWeakApplyPatchUpdate_AppendsPlainAddedLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	recovered, ok := recoverContentFromWeakApplyPatchUpdate(path, "@@ -1 +1,2 @@\n BASE_A\n+ORDERED_T11")

	require.True(t, ok)
	assert.Equal(t, "BASE_A\nORDERED_T11\n", recovered)
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchWeakAppendDiffUsesRecoveredContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-weak-append",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"mutations/base_a.txt\",\"diff\":\"@@ -1 +1,2 @@\\n BASE_A\\n+ORDERED_T11\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, path, "", "update_file")
	require.NoError(t, err)

	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Equal(t, "BASE_A\nORDERED_T11", gjson.GetBytes(out, "output.0.operation.content").String())
	assert.Equal(t, "", gjson.GetBytes(out, "output.0.operation.diff").String())
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "-BASE_A")
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "+BASE_A")
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "+ORDERED_T11")
}

func TestRepairApplyPatchOperationFromReasoningBlock_UsesFinalCodeFence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	repaired := repairApplyPatchOperationFromReasoningBlock(
		map[string]any{
			"type": "create_file",
			"path": path,
			"diff": "TITLE=alpha\n+TITLE=beta\n ENV=dev\n-DEBUG=true\n PORT=8080\n-FOOTER=old\n+FOOTER=done\n",
		},
		"",
		"Result should be:\n```text\nTITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n```",
	)
	op, ok := repaired.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done", op["content"])
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "+MODE=prod")
	assert.Contains(t, fmt.Sprintf("%v", op["diff"]), "-DEBUG=true")
}

func TestPreferContentDrivenApplyPatchOperation_RebuildsExistingFilePatchFromContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "complex.txt")
	original := "TITLE=alpha\nENV=dev\nDEBUG=true\nPORT=8080\nFOOTER=old\n"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o644))

	repaired := preferContentDrivenApplyPatchOperation(map[string]any{
		"type":    "update_file",
		"path":    path,
		"diff":    "@@\n TITLE=alpha\n ENV=dev\n DEBUG=true\n PORT=8080\n FOOTER=old\n+TITLE=beta\n+ENV=dev\n+MODE=prod\n+PORT=8080\n+FOOTER=done",
		"content": "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done\n",
	})
	op, ok := repaired.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", op["type"])
	assert.Equal(t, "TITLE=beta\nENV=dev\nMODE=prod\nPORT=8080\nFOOTER=done", strings.TrimSpace(fmt.Sprintf("%v", op["content"])))
	_, hasDiff := op["diff"]
	assert.False(t, hasDiff)
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchContentUsesPathHintForReplacementPatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-hint",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"update_file\",\"path\":\"mutations/base_a.txt\",\"content\":\"BASE_A\nORDERED_T11\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, path, "", "update_file")
	require.NoError(t, err)

	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Equal(t, "BASE_A\nORDERED_T11", gjson.GetBytes(out, "output.0.operation.content").String())
	assert.Equal(t, "", gjson.GetBytes(out, "output.0.operation.diff").String())
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "*** Update File:")
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "+BASE_A")
	assert.Contains(t, gjson.GetBytes(out, "output.0.input").String(), "+ORDERED_T11")
}

func TestRepairApplyPatchOperationPathWithHint_RewritesExistingRelativePathToAbsoluteHint(t *testing.T) {
	originalWD, err := os.Getwd()
	require.NoError(t, err)
	tmp := t.TempDir()
	require.NoError(t, os.Chdir(tmp))
	defer func() {
		_ = os.Chdir(originalWD)
	}()

	require.NoError(t, os.MkdirAll(filepath.Join(tmp, "mutations"), 0o755))
	relativePath := filepath.Join("mutations", "base_a.txt")
	absolutePath := filepath.Join(tmp, "mutations", "base_a.txt")
	require.NoError(t, os.WriteFile(absolutePath, []byte("BASE_A\n"), 0o644))

	repaired, changed := repairApplyPatchOperationPathWithHint(map[string]any{
		"type":    "update_file",
		"path":    filepath.ToSlash(relativePath),
		"content": "BASE_A\nORDERED_T11\n",
	}, absolutePath)
	require.True(t, changed)
	op, ok := repaired.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, filepath.ToSlash(absolutePath), filepath.ToSlash(fmt.Sprintf("%v", op["path"])))
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchContentUsesTypeHintWhenOperationTypeMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-missing-type",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"C:\\\\Users\\\\YLAB-Partner\\\\Desktop\\\\qwentest\\\\mutations\\\\base_a.txt\",\"content\":\"BASE_A\nORDERED_T11\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, path, "", "update_file")
	require.NoError(t, err)

	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "update_file", gjson.GetBytes(out, "output.0.operation.type").String())
	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Equal(t, "BASE_A\nORDERED_T11", gjson.GetBytes(out, "output.0.operation.content").String())
	assert.NotContains(t, string(out), "operation was invalid")
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchInfersUpdateTypeWhenOperationTypeMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base_a.txt")
	require.NoError(t, os.WriteFile(path, []byte("BASE_A\n"), 0o644))

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-infer-update-type",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"` + filepath.ToSlash(path) + `\",\"content\":\"BASE_A\nORDERED_T11\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "update_file", gjson.GetBytes(out, "output.0.operation.type").String())
	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Equal(t, "BASE_A\nORDERED_T11", gjson.GetBytes(out, "output.0.operation.content").String())
	assert.NotContains(t, string(out), "operation was invalid")
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchInfersCreateTypeWhenOperationTypeMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "new_skill.txt")

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-infer-create-type",
	  "model":"qwen-test",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"` + filepath.ToSlash(path) + `\",\"content\":\"HELLO_SKILL\"}}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "create_file", gjson.GetBytes(out, "output.0.operation.type").String())
	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.0.operation.path").String())
	assert.Equal(t, "HELLO_SKILL", gjson.GetBytes(out, "output.0.operation.content").String())
	assert.NotContains(t, string(out), "operation was invalid")
}

func TestTranslateChatCompletionToResponsesResponse_ApplyPatchRecoversFromReasoningWhenNativeArgsEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "probe.txt")

	body := []byte(`{
	  "id":"chatcmpl-apply-patch-reasoning-recovery",
	  "model":"Abiray-Qwen3.6-27B-NVFP4.gguf",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "reasoning_content":"I need to create the file with apply_patch.\\n\\napply_patch(input=\"` + filepath.ToSlash(path) + `\", operation={}, patch=\"*** Begin Patch\\n*** Add File: ` + filepath.ToSlash(path) + `\\n+HELLO_PROBE\\n*** End Patch\")",
	      "tool_calls":[{
	        "id":"call_patch_1",
	        "type":"function",
	        "function":{"name":"apply_patch","arguments":"{}"}
	      }]
	    },
	    "finish_reason":"tool_calls"
	  }]
	}`)

	out, err := translateChatCompletionToResponsesResponse(body, "", "", "")
	require.NoError(t, err)

	assert.Equal(t, "reasoning", gjson.GetBytes(out, "output.0.type").String())
	assert.Equal(t, "apply_patch_call", gjson.GetBytes(out, "output.1.type").String())
	assert.Equal(t, "create_file", gjson.GetBytes(out, "output.1.operation.type").String())
	assert.Equal(t, filepath.ToSlash(path), gjson.GetBytes(out, "output.1.operation.path").String())
	assert.Contains(t, gjson.GetBytes(out, "output.1.input").String(), "HELLO_PROBE")
	assert.NotContains(t, string(out), "operation was invalid")
}

func TestNormalizePossiblyMixedToolArguments_UnwrapsQuotedJSONObject(t *testing.T) {
	raw := `"{\"operation\":{\"type\":\"update_file\",\"path\":\"mutations/base_a.txt\",\"content\":\"BASE_A\\nORDERED_T11\"}}"`
	normalized := normalizePossiblyMixedToolArguments(raw)
	assert.JSONEq(t, `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`, normalized)
}

func TestNormalizePossiblyMixedToolArguments_RepairsLiteralNewlinesInJSONObjectStrings(t *testing.T) {
	raw := "{\"operation\":{\"type\":\"update_file\",\"path\":\"mutations/base_a.txt\",\"content\":\"BASE_A\nORDERED_T11\"}}"
	normalized := normalizePossiblyMixedToolArguments(raw)
	assert.JSONEq(t, `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`, normalized)
}
