package proxy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseModelSpecificToolCalls_UsesQwenParser(t *testing.T) {
	content := `<tool_call>
<function=shell>
<parameter=command>
pwd
</parameter>
</function>
</tool_call>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3-Coder-30B", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell", calls[0].Name)
	assert.Equal(t, "pwd", calls[0].Arguments["command"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_UnknownModelDoesNotParse(t *testing.T) {
	content := `<tool_call><function=shell><parameter=command>pwd</parameter></function></tool_call>`
	calls, remaining := parseModelSpecificToolCalls("llama-3.1", content)
	assert.Empty(t, calls)
	assert.Equal(t, content, remaining)
}

func TestTranslateResponsesToChatCompletionsRequest_PreservesToolChoiceModes(t *testing.T) {
	req := []byte(`{
		"model":"qwen",
		"input":"hello",
		"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}],
		"tool_choice":{"type":"function","function":{"name":"get_weather"}}
	}`)

	out, err := translateResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(out, &payload))
	toolChoice, ok := payload["tool_choice"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", toolChoice["type"])
	function, ok := toolChoice["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "get_weather", function["name"])
}
