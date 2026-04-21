package proxy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

	out, err := translateChatCompletionToResponsesResponse(body)
	require.NoError(t, err)

	var resp map[string]any
	require.NoError(t, json.Unmarshal(out, &resp))
	output, ok := resp["output"].([]any)
	require.True(t, ok)
	require.Len(t, output, 2)

	msg := output[0].(map[string]any)
	assert.Equal(t, "message", msg["type"])
	content := msg["content"].([]any)
	assert.Equal(t, "Planning done.", content[0].(map[string]any)["text"])

	call := output[1].(map[string]any)
	assert.Equal(t, "shell_call", call["type"])
	action := call["action"].(map[string]any)
	assert.Equal(t, "pwd", action["command"])
}

