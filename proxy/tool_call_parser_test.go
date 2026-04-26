package proxy

import (
	"encoding/json"
	"strings"
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

func TestParseModelSpecificToolCalls_CodexAliasUsesQwenParser(t *testing.T) {
	content := `<shell_commands><commands>[{"command":"pwd"}]</commands></shell_commands>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.3-codex", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, "pwd", calls[0].Arguments["command"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenBareFunctionBlock(t *testing.T) {
	content := `Planning complete.

<function=apply_patch>
<parameter=patch>
--- a/bridge_apply_patch_test.txt
+++ b/bridge_apply_patch_test.txt
@@ -0,0 +1 @@
+bridge-test-ok
</parameter>
</function>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "apply_patch", calls[0].Name)
	assert.Contains(t, calls[0].Arguments["patch"], "bridge-test-ok")
	assert.Equal(t, "Planning complete.", remaining)
}

func TestParseModelSpecificToolCalls_QwenFunctionStyleCall(t *testing.T) {
	content := `apply_patch(input="bridge_apply_patch_test.txt", operation={}, patch="--- a/bridge_apply_patch_test.txt\n+++ b/bridge_apply_patch_test.txt\n@@ -0,0 +1 @@\n+bridge-test-ok\n")`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "apply_patch", calls[0].Name)
	assert.Equal(t, "bridge_apply_patch_test.txt", calls[0].Arguments["input"])
	assert.Contains(t, calls[0].Arguments["patch"], "bridge-test-ok")
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenEscapedToolsEnvelopeCommandArray(t *testing.T) {
	content := `Planning first.
\u003ctools\u003e
{"command":["cat","c:\\Users\\YLAB-Partner\\.codex\\config.toml"]}
\u003c/tools\u003e`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, `cat c:\Users\YLAB-Partner\.codex\config.toml`, calls[0].Arguments["command"])
	assert.Equal(t, "Planning first.", remaining)
}

func TestParseModelSpecificToolCalls_QwenEscapedToolsEnvelopeApplyPatch(t *testing.T) {
	content := `\u003ctools\u003e
{"operation":{"type":"update_file","path":"README.md","content":"PATCH_OK"}}
\u003c/tools\u003e`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "apply_patch", calls[0].Name)
	operation, ok := calls[0].Arguments["operation"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "update_file", operation["type"])
	assert.Equal(t, "README.md", operation["path"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedShellCommands(t *testing.T) {
	content := `I will inspect first.
<shell_commands>
<commands>[{"command":"dir \"c:\\Users\\YLAB-Partner\\Downloads\\qwentest\"","description":"Check workspace contents"}]</commands>
</shell_commands>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, `dir "c:\Users\YLAB-Partner\Downloads\qwentest"`, calls[0].Arguments["command"])
	assert.Equal(t, "I will inspect first.", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedUpdatePlan(t *testing.T) {
	content := `<update_plan>
<steps>[{"content":"Initialize project","status":"in_progress"},{"content":"Set up DB schema","status":"pending"}]</steps>
</update_plan>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 2)
	step0 := plan[0]
	step1 := plan[1]
	assert.Equal(t, "Initialize project", step0["step"])
	assert.Equal(t, "in_progress", step0["status"])
	assert.Equal(t, "Set up DB schema", step1["step"])
	assert.Equal(t, "pending", step1["status"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedShellCommandsCatNoClose(t *testing.T) {
	content := `Starting check.
<shell_commands>
<cat> C:/Users/YLAB-Partner/.codex/skills/llama-swap-auto-repair-loop/SKILL.md </cat>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, `cat C:/Users/YLAB-Partner/.codex/skills/llama-swap-auto-repair-loop/SKILL.md`, calls[0].Arguments["command"])
	assert.Equal(t, "Starting check.", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedShellCommandTag(t *testing.T) {
	content := `Start.
<shell_command>
cat C:/Users/YLAB-Partner/.codex/skills/llama-swap-auto-repair-loop/SKILL.md
</shell_command>`

	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, `cat C:/Users/YLAB-Partner/.codex/skills/llama-swap-auto-repair-loop/SKILL.md`, calls[0].Arguments["command"])
	assert.Equal(t, "Start.", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedShellWrapperWithCommands(t *testing.T) {
	content := `<shell><commands>[{"command":"pwd"}]</commands></shell>`
	calls, remaining := parseModelSpecificToolCalls("Qwen3.6-35B-A3B-UD-Q8_K_XL", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, "pwd", calls[0].Arguments["command"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenToolUseFileReadPath(t *testing.T) {
	content := `<tool_use>
<file_read>
<path>C:/Users/YLAB-Partner/.codex/config.toml</path>
</file_read>
</tool_use>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, `cat "C:/Users/YLAB-Partner/.codex/config.toml"`, calls[0].Arguments["command"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenBareToolCallOpener(t *testing.T) {
	content := `Calling update first.
<tool_call>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	assert.Empty(t, calls)
	assert.Equal(t, "Calling update first.\n<tool_call>", remaining)
}

func TestParseModelSpecificToolCalls_QwenFunctionStyleUpdatePlanSteps(t *testing.T) {
	content := `update_plan(steps=[{"content":"Inspect workspace","status":"in_progress"},{"content":"Read config","status":"pending"}])`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 2)
	assert.Equal(t, "Inspect workspace", plan[0]["step"])
	assert.Equal(t, "in_progress", plan[0]["status"])
	assert.Equal(t, "Read config", plan[1]["step"])
	assert.Equal(t, "pending", plan[1]["status"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenFunctionStyleUpdatePlanPythonLike(t *testing.T) {
	content := `[update_plan(steps=[{'content': 'Inspect workspace structure', 'status': 'pending'}, {'content': 'Read and analyze config.toml', 'status': 'pending'}])]`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 2)
	assert.Equal(t, "Inspect workspace structure", plan[0]["step"])
	assert.Equal(t, "pending", plan[0]["status"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenTaggedUpdatePlanWithStepsAssignment(t *testing.T) {
	content := `<update_plan>
steps=[{"content":"Inspect workspace structure","status":"in_progress"},{"content":"Read and analyze config.toml","status":"pending"},{"content":"Summarize all findings","status":"pending"}]
</update_plan>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 3)
	assert.Equal(t, "Inspect workspace structure", plan[0]["step"])
	assert.Equal(t, "in_progress", plan[0]["status"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_QwenToolCallWithUpdatePlanJSONAndDanglingLs(t *testing.T) {
	content := `<tool_call>
<update_plan>
{"plan":[{"step":"Inspect workspace structure","status":"in_progress"},{"step":"Read config.toml","status":"pending"},{"step":"Summarize findings","status":"pending"}]}
</update_plan>
<ls -la /home/admmin/llama-swap`
	calls, _ := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 2)
	assert.Equal(t, "update_plan", calls[0].Name)
	assert.Equal(t, "shell_command", calls[1].Name)
	assert.Equal(t, "ls -la /home/admmin/llama-swap", calls[1].Arguments["command"])
}

func TestParseModelSpecificToolCalls_RealWSLMalformedPlanBlob(t *testing.T) {
	content := `The user wants me to:
1. Inspect the workspace
2. Read config.toml
3. Summarize findings

They want a 3-step plan, executed fully, with update_plan calls and PLAN_DONE at the end.

Let me start by creating the plan and then execute each step.
I'll inspect the workspace, read the config, and summarize everything. Let me start.

**Step 1: Inspect workspace structure.**

<tool_call>
<update_plan>
{"plan": [{"step": "Inspect workspace structure", "status": "in_progress"}, {"step": "Read config.toml", "status": "pending"}, {"step": "Summarize findings", "status": "pending"}]}
</update_plan>
<ls -la /home/admmin/llama-swap`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.NotEmpty(t, calls)
	assert.Equal(t, "update_plan", calls[0].Name)
	assert.NotContains(t, remaining, "<update_plan>")
}

func TestParseModelSpecificToolCalls_PlainNumberedPlanFallback(t *testing.T) {
	content := `I'll create a 3-step plan to inspect the workspace, read config.toml, and summarize findings. Let me start by setting up the plan and then execute each step.

### Plan:
1. Inspect workspace structure
2. Read and analyze config.toml
3. Summarize key findings`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 3)
	assert.Equal(t, "Inspect workspace structure", plan[0]["step"])
	assert.Equal(t, "in_progress", plan[0]["status"])
	assert.Equal(t, "", remaining)
}

func TestParseModelSpecificToolCalls_DecoratedPlanFallback(t *testing.T) {
	content := `I'll create a simple 3-step plan and execute it:

*** Begin Plan
1. Check current directory
2. List files in the current directory
3. Show git status
*** End PlanNow executing Step 1:`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	assert.Empty(t, strings.TrimSpace(remaining))
}

func TestParseModelSpecificToolCalls_ProposedPlanSummaryFallback(t *testing.T) {
	content := `Building it now — a colorful, self-contained math game in one HTML file.

<proposed_plan>
Summary: Create a single self-contained HTML game and verify it runs.
</proposed_plan>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 1)
	assert.Equal(t, "in_progress", plan[0]["status"])
	assert.Contains(t, plan[0]["step"], "single self-contained HTML game")
	assert.NotContains(t, remaining, "<proposed_plan>")
}

func TestParseModelSpecificToolCalls_UpdatePlanWithStrayParameterCloseTag(t *testing.T) {
	content := `Let me start implementing.
I'll build this step by step.

<update_plan>
{"steps": [{"content": "Verify Python 3 + tkinter available", "status": "in_progress"}, {"content": "Create math_quest.py with core game logic", "status": "pending"}, {"content": "Add kid-friendly UI with progress tracking", "status": "pending"}]}
</parameter>
</update_plan>`

	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.NotEmpty(t, calls)
	assert.Equal(t, "update_plan", calls[0].Name)
	planJSON, err := json.Marshal(calls[0].Arguments["plan"])
	require.NoError(t, err)
	var plan []map[string]any
	require.NoError(t, json.Unmarshal(planJSON, &plan))
	require.Len(t, plan, 3)
	assert.Equal(t, "Verify Python 3 + tkinter available", plan[0]["step"])
	assert.Equal(t, "in_progress", plan[0]["status"])
	assert.NotContains(t, remaining, "<update_plan>")
	assert.NotContains(t, remaining, "</parameter>")
}

func TestParseModelSpecificToolCalls_ApplyPatchTagAddFileFallback(t *testing.T) {
	content := `<apply_patch>
*** Add File: c:\Users\YLAB-Partner\Downloads\qwentest\math-game.html
+<!DOCTYPE html>
+<html><body>ok</body></html>
</apply_patch>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "apply_patch", calls[0].Name)
	input := strings.TrimSpace(calls[0].Arguments["input"].(string))
	assert.Contains(t, input, "*** Begin Patch")
	assert.Contains(t, input, "*** Add File: c:\\Users\\YLAB-Partner\\Downloads\\qwentest\\math-game.html")
	assert.Empty(t, strings.TrimSpace(remaining))
}

func TestParseModelSpecificToolCalls_GenericTerminalTagFallback(t *testing.T) {
	content := `<terminal>pwd</terminal>`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell_command", calls[0].Name)
	assert.Equal(t, "pwd", calls[0].Arguments["command"])
	assert.Empty(t, strings.TrimSpace(remaining))
}

func TestParseModelSpecificToolCalls_GenericUnknownTagStripped(t *testing.T) {
	content := `Before text.
<analysis>internal only</analysis>
After text.`
	calls, remaining := parseModelSpecificToolCalls("gpt-5.2", content)
	assert.Empty(t, calls)
	assert.NotContains(t, remaining, "<analysis>")
	assert.Contains(t, remaining, "Before text.")
	assert.Contains(t, remaining, "After text.")
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

func TestTranslateResponsesToChatCompletionsRequest_SystemMessagesAreLeading(t *testing.T) {
	req := []byte(`{
		"model":"gpt-5.3-codex",
		"instructions":"global system",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"user first"}]},
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev second"}]}
		]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(out, &payload))
	rawMessages, ok := payload["messages"].([]any)
	require.True(t, ok)
	require.Len(t, rawMessages, 2)

	first, ok := rawMessages[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "system", first["role"])
	content, _ := first["content"].(string)
	assert.Contains(t, content, "global system")
	assert.Contains(t, content, "dev second")

	second, ok := rawMessages[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "user", second["role"])
	assert.Equal(t, "user first", second["content"])
}

func TestTranslateResponsesToChatCompletionsRequest_ApplyPatchBridgeToolRequiresOperation(t *testing.T) {
	req := []byte(`{
		"model":"qwen",
		"input":"hello",
		"tools":[{"type":"apply_patch"}]
	}`)

	out, err := translateResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(out, &payload))
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)

	tool, ok := tools[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", tool["type"])
	fn, ok := tool["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "apply_patch", fn["name"])

	params, ok := fn["parameters"].(map[string]any)
	require.True(t, ok)
	required, ok := params["required"].([]any)
	require.True(t, ok)
	require.Len(t, required, 1)
	assert.Equal(t, "operation", required[0])
	_, hasAnyOf := params["anyOf"]
	assert.False(t, hasAnyOf)

	props, ok := params["properties"].(map[string]any)
	require.True(t, ok)
	_, hasInput := props["input"]
	assert.False(t, hasInput)
	_, hasPatch := props["patch"]
	assert.False(t, hasPatch)
	_, hasOperation := props["operation"]
	assert.True(t, hasOperation)
}
