package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	PROFILE_SPLIT_CHAR = ":"
)

const (
	llamaSwapResponseToolAdapterHeader   = "X-LlamaSwap-Responses-Tool-Adapter"
	llamaSwapApplyPatchPathHintHeader    = "X-LlamaSwap-Apply-Patch-Path-Hint"
	llamaSwapApplyPatchContentHintHeader = "X-LlamaSwap-Apply-Patch-Content-Hint"
	llamaSwapApplyPatchTypeHintHeader    = "X-LlamaSwap-Apply-Patch-Type-Hint"
	llamaSwapShellFunctionName           = "__llamaswap_shell"
	llamaSwapApplyPatchFunctionName      = "__llamaswap_apply_patch"
	llamaSwapWebSearchFunctionName       = "__llamaswap_web_search_preview"
	llamaSwapFileSearchFunctionName      = "__llamaswap_file_search"
	llamaSwapCodeInterpreterFunctionName = "__llamaswap_code_interpreter"
	llamaSwapImageGenerationFunctionName = "__llamaswap_image_generation"
	llamaSwapComputerFunctionName        = "__llamaswap_computer"
)

const applyPatchTraceLogPath = "/tmp/llama-swap-apply-patch-trace.log"

const (
	applyPatchPreferredToolDescription  = "Preferred method for file create, update, and delete. Use apply_patch instead of shell for writing file content or deleting files. For simple updates, prefer operation.content with the final file text. Use operation.diff only when you can provide exact patch context, not line-number-only hunks. For deletions, use operation.type=delete_file instead of shell rm/del."
	applyPatchTailConstraintText        = "[apply_patch preferred] For file writes and deletions, use apply_patch rather than shell. Shell is for commands, builds, and inspection only."
	applyPatchDeleteConstraintText      = "[native delete required] If the task deletes a file and apply_patch is available, use apply_patch with operation.type=delete_file. Do not use shell rm, del, unlink, or trash commands for that deletion."
	applyPatchRetryPreferredFailureText = "[proxy] apply_patch is available and preferred for file writes. The previous turn used shell to write file content. Retry with apply_patch."
	applyPatchValidationWarningPrefix   = "[proxy validation] apply_patch"
	shellValidationWarningPrefix        = "[proxy validation] shell"
	llamaSwapWebSearchURLVar            = "LLAMA_SWAP_WEB_SEARCH_URL"
	llamaSwapSearxngCommandVar          = "LLAMA_SWAP_SEARXNG_CMD"
	llamaSwapSearxngStopCommandVar      = "LLAMA_SWAP_SEARXNG_STOP_CMD"
	llamaSwapSearxngEnabledVar          = "LLAMA_SWAP_SEARXNG_ENABLED"
)

var applyPatchTraceMu sync.Mutex
var bridgeHTMLTagRegexp = regexp.MustCompile(`(?s)<[^>]+>`)
var planModeQuestionCountRegexp = regexp.MustCompile(`(?i)\b(\d+)\s+(?:[a-z-]+\s+){0,3}questions\b`)
var codexAvailableSkillsBlockRegexp = regexp.MustCompile(`(?is)### Available skills\s*(.*?)\s*(?:### How to use skills|</skills_instructions>|$)`)
var codexAvailableSkillLineRegexp = regexp.MustCompile(`(?m)^\s*-\s*([a-z0-9][a-z0-9._-]*)\s*:`)
var codexAvailableSkillPathRegexp = regexp.MustCompile(`(?im)^\s*-\s*[a-z0-9][a-z0-9._-]*\s*:.*?\(file:\s*([^)]+?/SKILL\.md)\s*\)\s*$`)
var codexSkillReferenceRegexp = regexp.MustCompile(`\$([A-Za-z0-9._-]+)`)

// stripTopLevelParam removes a top-level parameter from JSON body without affecting nested fields.
// This prevents accidentally removing nested fields like tools[].type when stripping "type".
func stripTopLevelParam(bodyBytes []byte, paramName string) ([]byte, error) {
	// Parse the JSON to work with it structurally
	var data map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return bodyBytes, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Only delete from the top level
	delete(data, paramName)

	// Marshal back to JSON
	result, err := json.Marshal(data)
	if err != nil {
		return bodyBytes, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return result, nil
}

// normalizeChatCompletionTools converts Responses-style top-level function tools
// into the nested chat.completions format expected by llama.cpp-compatible servers.

func normalizeChatCompletionTools(bodyBytes []byte) ([]byte, error) {
	var data map[string]any
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return bodyBytes, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	rawTools, ok := data["tools"].([]any)
	if !ok || len(rawTools) == 0 {
		return bodyBytes, nil
	}

	changed := false
	normalizedTools := make([]any, 0, len(rawTools))
	for _, rawTool := range rawTools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			normalizedTools = append(normalizedTools, rawTool)
			continue
		}

		if fn, ok := tool["function"].(map[string]any); ok {
			if name, _ := fn["name"].(string); strings.TrimSpace(name) != "" {
				normalizedTools = append(normalizedTools, map[string]any{
					"type":     "function",
					"function": fn,
				})
				continue
			}
		}

		name, _ := tool["name"].(string)
		name = strings.TrimSpace(name)
		if name == "" {
			normalizedTools = append(normalizedTools, rawTool)
			continue
		}

		fn := map[string]any{"name": name}
		if desc, ok := tool["description"]; ok {
			fn["description"] = desc
		}
		if params, ok := tool["parameters"]; ok {
			fn["parameters"] = params
		}
		if strict, ok := tool["strict"]; ok {
			fn["strict"] = strict
		}
		normalizedTools = append(normalizedTools, map[string]any{
			"type":     "function",
			"function": fn,
		})
		changed = true
	}

	if !changed {
		return bodyBytes, nil
	}
	data["tools"] = normalizedTools
	result, err := json.Marshal(data)
	if err != nil {
		return bodyBytes, fmt.Errorf("failed to marshal JSON: %w", err)
	}
	return result, nil
}

type grammarToolsConflictResult struct {
	removedGrammar            bool
	removedJSONSchemaResponse bool
}

func stripGrammarToolsConflictMap(data map[string]any) grammarToolsConflictResult {
	result := grammarToolsConflictResult{}
	if data == nil {
		return result
	}

	rawTools, ok := data["tools"].([]any)
	if !ok || len(rawTools) == 0 {
		return result
	}

	if _, hasGrammar := data["grammar"]; hasGrammar {
		delete(data, "grammar")
		result.removedGrammar = true
	}

	responseFormat, hasResponseFormat := data["response_format"].(map[string]any)
	if !hasResponseFormat {
		return result
	}
	respType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", responseFormat["type"])))
	if respType == "json_schema" {
		delete(data, "response_format")
		result.removedJSONSchemaResponse = true
	}
	return result
}

func stripGrammarToolsConflictJSON(bodyBytes []byte) ([]byte, grammarToolsConflictResult, error) {
	var data map[string]any
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return bodyBytes, grammarToolsConflictResult{}, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	result := stripGrammarToolsConflictMap(data)
	if !result.removedGrammar && !result.removedJSONSchemaResponse {
		return bodyBytes, result, nil
	}
	updated, err := json.Marshal(data)
	if err != nil {
		return bodyBytes, grammarToolsConflictResult{}, fmt.Errorf("failed to marshal JSON: %w", err)
	}
	return updated, result, nil
}

func normalizeResponsesRequest(bodyBytes []byte) ([]byte, []string, []string, error) {
	var data map[string]any
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return bodyBytes, nil, nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	adaptedTools, unsupportedTools, changedTools := normalizeResponsesToolsMap(data)
	changedInput := normalizeResponsesInputMap(data)
	changedSteering := injectQwenResponsesToolPolicy(data, adaptedTools)

	if !changedTools && !changedInput && !changedSteering {
		return bodyBytes, adaptedTools, unsupportedTools, nil
	}

	result, err := json.Marshal(data)
	if err != nil {
		return bodyBytes, nil, nil, fmt.Errorf("failed to marshal JSON: %w", err)
	}
	return result, adaptedTools, unsupportedTools, nil
}

func injectQwenResponsesToolPolicy(data map[string]any, adaptedTools []string) bool {
	modelName, _ := data["model"].(string)
	if !isQwenModelName(modelName) || len(adaptedTools) == 0 {
		return false
	}

	policy := buildQwenResponsesToolPolicy(adaptedTools)
	if policy == "" {
		return false
	}

	input, ok := data["input"].([]any)
	if !ok {
		return false
	}

	for idx, rawItem := range input {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType, _ := item["type"].(string)
		role, _ := item["role"].(string)
		if itemType != "message" || !strings.EqualFold(strings.TrimSpace(role), "system") {
			continue
		}

		rewritten := cloneMap(item)
		rewritten["content"] = appendPolicyToMessageContent(item["content"], policy)
		input[idx] = rewritten
		data["input"] = input
		return true
	}

	data["input"] = append([]any{map[string]any{
		"type":    "message",
		"role":    "system",
		"content": policy,
	}}, input...)
	return true
}

func extractCodexAvailableSkillsFromText(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	match := codexAvailableSkillsBlockRegexp.FindStringSubmatch(text)
	if len(match) < 2 {
		return nil
	}
	lines := codexAvailableSkillLineRegexp.FindAllStringSubmatch(match[1], -1)
	if len(lines) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		if len(line) < 2 {
			continue
		}
		name := strings.TrimSpace(line[1])
		if name == "" {
			continue
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func resolveCodexSkillsRoot() string {
	if codexHome := strings.TrimSpace(os.Getenv("CODEX_HOME")); codexHome != "" {
		return filepath.Join(codexHome, "skills")
	}
	homeDir, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(homeDir) == "" {
		return ""
	}
	return filepath.Join(homeDir, ".codex", "skills")
}

func extractCodexSkillRootsFromText(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	matches := codexAvailableSkillPathRegexp.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		skillFile := strings.TrimSpace(match[1])
		if skillFile == "" {
			continue
		}
		canonical, err := canonicalizeLocalApplyPatchPath(skillFile)
		if err != nil {
			continue
		}
		root := filepath.Dir(filepath.Dir(canonical))
		root = filepath.Clean(root)
		if root == "" {
			continue
		}
		if _, exists := seen[root]; exists {
			continue
		}
		seen[root] = struct{}{}
		out = append(out, root)
	}
	sort.Strings(out)
	return out
}

func listCodexSkillNamesOnDiskFromRoot(skillsRoot string) []string {
	skillsRoot = strings.TrimSpace(skillsRoot)
	if strings.TrimSpace(skillsRoot) == "" {
		return nil
	}
	entries, err := os.ReadDir(skillsRoot)
	if err != nil {
		return nil
	}
	out := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := strings.TrimSpace(entry.Name())
		if name == "" || strings.HasPrefix(name, ".") {
			continue
		}
		if info, err := os.Stat(filepath.Join(skillsRoot, name, "SKILL.md")); err == nil && !info.IsDir() {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

func detectCodexSkillSnapshotDrift(bodyBytes []byte) (missing []string, extra []string, ok bool) {
	var data map[string]any
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return nil, nil, false
	}
	input, _ := data["input"].([]any)
	snapshot := map[string]struct{}{}
	disk := map[string]struct{}{}
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		text := extractResponsesInputText(item["content"])
		for _, candidate := range extractCodexAvailableSkillsFromText(text) {
			snapshot[candidate] = struct{}{}
		}
		for _, root := range extractCodexSkillRootsFromText(text) {
			for _, name := range listCodexSkillNamesOnDiskFromRoot(root) {
				disk[name] = struct{}{}
			}
		}
	}
	if len(snapshot) == 0 {
		return nil, nil, false
	}
	if len(disk) == 0 {
		for _, name := range listCodexSkillNamesOnDiskFromRoot(resolveCodexSkillsRoot()) {
			disk[name] = struct{}{}
		}
		if len(disk) == 0 {
			return nil, nil, false
		}
	}
	for name := range disk {
		if _, exists := snapshot[name]; !exists {
			missing = append(missing, name)
		}
	}
	for name := range snapshot {
		if _, exists := disk[name]; !exists {
			extra = append(extra, name)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	return missing, extra, true
}

func isQwenModelName(modelName string) bool {
	modelName = strings.TrimSpace(strings.ToLower(modelName))
	return strings.Contains(modelName, "qwen")
}

func buildQwenResponsesToolPolicy(adaptedTools []string) string {
	toolSet := make(map[string]struct{}, len(adaptedTools))
	for _, tool := range adaptedTools {
		normalized := strings.TrimSpace(strings.ToLower(tool))
		switch normalized {
		case "web_search_preview", "websearchpreview", llamaSwapWebSearchFunctionName:
			normalized = "web_search"
		}
		toolSet[normalized] = struct{}{}
	}

	lines := make([]string, 0, 5)
	if hasAnyTool(toolSet, "apply_patch", "shell") {
		lines = append(lines,
			"Tool policy:",
			"- Use apply_patch for any file creation, deletion, or modification.",
			"- Do not use shell to edit files.",
			"- Use shell only to inspect files, run builds, tests, or commands.",
			"- When changing files, prefer the tool call over prose.",
		)
	}
	if hasAnyTool(toolSet, "web_search") {
		if len(lines) == 0 {
			lines = append(lines, "Tool policy:")
		}
		lines = append(lines,
			"- Use web_search for current or external information.",
			"- Do not answer current-events or live-information questions from memory when web_search is available.",
		)
	}
	if hasAnyTool(toolSet, "file_search") {
		if len(lines) == 0 {
			lines = append(lines, "Tool policy:")
		}
		lines = append(lines,
			"- Use file_search for indexed or local document search requests.",
		)
	}
	if hasAnyTool(toolSet, "computer") {
		if len(lines) == 0 {
			lines = append(lines, "Tool policy:")
		}
		lines = append(lines,
			fmt.Sprintf("- Use %s for UI automation requests.", llamaSwapComputerFunctionName),
			fmt.Sprintf("- If a request mentions computer_use_preview, emit %s with action plus optional x/y/text/button fields.", llamaSwapComputerFunctionName),
		)
	}

	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func hasAnyTool(toolSet map[string]struct{}, names ...string) bool {
	for _, name := range names {
		if _, ok := toolSet[name]; ok {
			return true
		}
	}
	return false
}

func appendPolicyToMessageContent(content any, policy string) any {
	switch typed := content.(type) {
	case string:
		existing := strings.TrimSpace(typed)
		if existing == "" {
			return policy
		}
		return existing + "\n\n" + policy
	case []any:
		rewritten := make([]any, 0, len(typed)+1)
		rewritten = append(rewritten, typed...)
		rewritten = append(rewritten, map[string]any{
			"type": "input_text",
			"text": policy,
		})
		return rewritten
	default:
		return policy
	}
}

func normalizeResponsesToolsMap(data map[string]any) ([]string, []string, bool) {
	rawTools, ok := data["tools"].([]any)
	if !ok || len(rawTools) == 0 {
		return nil, nil, false
	}

	adapted := make([]string, 0, 2)
	unsupported := make([]string, 0)
	changed := false
	normalizedTools := make([]any, 0, len(rawTools))

	for _, rawTool := range rawTools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			normalizedTools = append(normalizedTools, rawTool)
			continue
		}

		toolType, _ := tool["type"].(string)
		toolName, _ := tool["name"].(string)
		toolName = strings.TrimSpace(toolName)
		switch toolType {
		case "function":
			normalizedTools = append(normalizedTools, rawTool)
		case "mcp":
			normalizedTools = append(normalizedTools, cloneMap(tool))
		case "shell":
			normalizedTools = append(normalizedTools, buildResponsesShellFunctionTool())
			adapted = appendIfMissing(adapted, "shell")
			changed = true
		case "apply_patch", "applypatch":
			normalizedTools = append(normalizedTools, buildResponsesApplyPatchFunctionTool())
			adapted = appendIfMissing(adapted, "apply_patch")
			changed = true
		case "web_search_preview", "web_search":
			normalizedTools = append(normalizedTools, buildResponsesWebSearchFunctionTool())
			adapted = appendIfMissing(adapted, "web_search")
			changed = true
		case "file_search":
			normalizedTools = append(normalizedTools, buildResponsesFileSearchFunctionTool())
			adapted = appendIfMissing(adapted, toolType)
			changed = true
		case "code_interpreter":
			normalizedTools = append(normalizedTools, buildResponsesCodeInterpreterFunctionTool())
			adapted = appendIfMissing(adapted, toolType)
			changed = true
		case "image_generation":
			normalizedTools = append(normalizedTools, buildResponsesImageGenerationFunctionTool())
			adapted = appendIfMissing(adapted, toolType)
			changed = true
		case "computer", "computer_use_preview":
			normalizedTools = append(normalizedTools, buildResponsesComputerFunctionTool())
			adapted = appendIfMissing(adapted, "computer")
			changed = true
		default:
			if toolType == "custom" {
				switch toolName {
				case "apply_patch", "applypatch":
					normalizedTools = append(normalizedTools, buildResponsesApplyPatchFunctionTool())
					adapted = appendIfMissing(adapted, "apply_patch")
					changed = true
					continue
				case "shell":
					normalizedTools = append(normalizedTools, buildResponsesShellFunctionTool())
					adapted = appendIfMissing(adapted, toolName)
					changed = true
					continue
				}
			}

			// Unknown tool types are treated as custom function tools when possible.
			name := toolName
			if name == "" {
				unsupported = appendIfMissing(unsupported, toolType)
				changed = true
				continue
			}

			adapted = appendIfMissing(adapted, toolType)
			changed = true
			normalizedTools = append(normalizedTools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        name,
					"description": tool["description"],
					"parameters":  tool["parameters"],
					"strict":      tool["strict"],
				},
			})
		}
	}

	if changed {
		data["tools"] = normalizedTools
	}
	return adapted, unsupported, changed
}

func normalizeResponsesInputMap(data map[string]any) bool {
	input, exists := data["input"]
	if !exists {
		return false
	}

	inputItems, ok := input.([]any)
	if !ok {
		return false
	}

	changed := false
	normalized := make([]any, 0, len(inputItems))
	for _, item := range inputItems {
		mapped, itemChanged := normalizeResponsesInputItem(item)
		if itemChanged {
			changed = true
		}
		normalized = append(normalized, mapped)
	}

	reordered, reorderedChanged := moveSystemMessagesToFront(normalized)
	if reorderedChanged {
		normalized = reordered
		changed = true
	}

	merged, mergedChanged := mergeLeadingSystemMessages(normalized)
	if mergedChanged {
		normalized = merged
		changed = true
	}

	if changed {
		data["input"] = normalized
	}
	return changed
}

func normalizeResponsesInputItem(item any) (any, bool) {
	m, ok := item.(map[string]any)
	if !ok {
		return item, false
	}

	itemType, _ := m["type"].(string)
	switch itemType {
	case "message":
		role, _ := m["role"].(string)
		normalizedRole := normalizeResponsesMessageRole(role)
		if normalizedRole == role {
			return item, false
		}

		rewritten := cloneMap(m)
		rewritten["role"] = normalizedRole
		return rewritten, true
	case "shell_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapShellFunctionName,
			"arguments": mustJSONString(action),
		}, true
	case "apply_patch_call", "applypatch_call":
		payload := map[string]any{}
		if operation, ok := m["operation"]; ok {
			payload["operation"] = normalizeApplyPatchOperation(operation)
			if input := strings.TrimSpace(buildApplyPatchInputFromOperation(operation)); input != "" {
				payload["input"] = input
			}
		}
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapApplyPatchFunctionName,
			"arguments": mustJSONString(payload),
		}, true
	case "web_search_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      "web_search",
			"arguments": mustJSONString(action),
		}, true
	case "file_search_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapFileSearchFunctionName,
			"arguments": mustJSONString(action),
		}, true
	case "code_interpreter_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapCodeInterpreterFunctionName,
			"arguments": mustJSONString(action),
		}, true
	case "image_generation_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapImageGenerationFunctionName,
			"arguments": mustJSONString(action),
		}, true
	case "computer_call":
		action, _ := m["action"].(map[string]any)
		return map[string]any{
			"type":      "function_call",
			"call_id":   m["call_id"],
			"name":      llamaSwapComputerFunctionName,
			"arguments": mustJSONString(action),
		}, true
	case "shell_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "shell_call_output", "payload": m}),
		}, true
	case "apply_patch_call_output", "applypatch_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "apply_patch_call_output", "payload": m}),
		}, true
	case "web_search_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "web_search_call_output", "payload": m}),
		}, true
	case "file_search_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "file_search_call_output", "payload": m}),
		}, true
	case "code_interpreter_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "code_interpreter_call_output", "payload": m}),
		}, true
	case "image_generation_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "image_generation_call_output", "payload": m}),
		}, true
	case "computer_call_output":
		return map[string]any{
			"type":    "function_call_output",
			"call_id": m["call_id"],
			"output":  mustJSONString(map[string]any{"type": "computer_call_output", "payload": m}),
		}, true
	default:
		return item, false
	}
}

func normalizeResponsesMessageRole(role string) string {
	return role
}

func moveSystemMessagesToFront(items []any) ([]any, bool) {
	systemItems := make([]any, 0, len(items))
	otherItems := make([]any, 0, len(items))

	seenNonSystem := false
	changed := false

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			seenNonSystem = true
			otherItems = append(otherItems, item)
			continue
		}

		itemType, _ := m["type"].(string)
		role, _ := m["role"].(string)
		isSystemMessage := itemType == "message" && strings.EqualFold(strings.TrimSpace(role), "system")
		if isSystemMessage {
			if seenNonSystem {
				changed = true
			}
			systemItems = append(systemItems, item)
			continue
		}

		seenNonSystem = true
		otherItems = append(otherItems, item)
	}

	if !changed {
		return items, false
	}

	reordered := make([]any, 0, len(items))
	reordered = append(reordered, systemItems...)
	reordered = append(reordered, otherItems...)
	return reordered, true
}

func mergeLeadingSystemMessages(items []any) ([]any, bool) {
	if len(items) < 2 {
		return items, false
	}

	systemCount := 0
	mergedParts := make([]string, 0)
	mergedArrayParts := make([]any, 0)
	useArrayContent := false

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			break
		}

		itemType, _ := m["type"].(string)
		role, _ := m["role"].(string)
		if itemType != "message" || !strings.EqualFold(strings.TrimSpace(role), "system") {
			break
		}

		systemCount++
		switch typed := m["content"].(type) {
		case []any:
			useArrayContent = true
			for _, part := range typed {
				text := strings.TrimSpace(extractResponsesInputText(part))
				if text != "" {
					mergedParts = append(mergedParts, text)
					mergedArrayParts = append(mergedArrayParts, map[string]any{"type": "input_text", "text": text})
				}
			}
		case string:
			text := strings.TrimSpace(typed)
			if text != "" {
				mergedParts = append(mergedParts, text)
				if useArrayContent {
					mergedArrayParts = append(mergedArrayParts, map[string]any{"type": "input_text", "text": text})
				}
			}
		default:
			text := strings.TrimSpace(extractResponsesInputText(typed))
			if text != "" {
				mergedParts = append(mergedParts, text)
				if useArrayContent {
					mergedArrayParts = append(mergedArrayParts, map[string]any{"type": "input_text", "text": text})
				}
			}
		}
	}

	if systemCount < 2 {
		return items, false
	}

	firstMessage, ok := items[0].(map[string]any)
	if !ok {
		return items, false
	}

	mergedFirst := cloneMap(firstMessage)
	joined := strings.TrimSpace(strings.Join(mergedParts, "\n\n"))
	if useArrayContent {
		mergedFirst["content"] = mergedArrayParts
	} else {
		mergedFirst["content"] = joined
	}

	rewritten := make([]any, 0, len(items)-systemCount+1)
	rewritten = append(rewritten, mergedFirst)
	rewritten = append(rewritten, items[systemCount:]...)
	return rewritten, true
}

func normalizeChatMessagesForStrictTemplates(messages []map[string]any) []map[string]any {
	if len(messages) < 2 {
		return messages
	}

	systemLike := make([]map[string]any, 0, len(messages))
	other := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		trimmedRole := strings.TrimSpace(strings.ToLower(role))
		if trimmedRole == "system" || trimmedRole == "developer" {
			cloned := cloneMap(msg)
			// Strict templates expect at most one leading system message.
			cloned["role"] = "system"
			systemLike = append(systemLike, cloned)
			continue
		}
		other = append(other, msg)
	}

	if len(systemLike) == 0 || len(other) == 0 {
		return messages
	}

	ordered := make([]map[string]any, 0, len(messages))
	ordered = append(ordered, systemLike...)
	ordered = append(ordered, other...)

	if len(ordered) < 2 {
		return ordered
	}

	mergedTextParts := make([]string, 0, len(ordered))
	leadingSystemCount := 0
	for _, msg := range ordered {
		role, _ := msg["role"].(string)
		if !strings.EqualFold(strings.TrimSpace(role), "system") {
			break
		}
		leadingSystemCount++
		content, _ := msg["content"].(string)
		content = strings.TrimSpace(content)
		if content != "" {
			mergedTextParts = append(mergedTextParts, content)
		}
	}

	if leadingSystemCount < 2 {
		return ordered
	}

	first := cloneMap(ordered[0])
	first["content"] = strings.TrimSpace(strings.Join(mergedTextParts, "\n\n"))
	merged := make([]map[string]any, 0, len(ordered)-leadingSystemCount+1)
	merged = append(merged, first)
	merged = append(merged, ordered[leadingSystemCount:]...)
	return merged
}

func buildResponsesShellFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapShellFunctionName,
		"description": "Compatibility wrapper for the Responses API shell tool. Return commands to execute locally.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"commands": map[string]any{
					"type":        "array",
					"description": "Commands to execute in order.",
					"items": map[string]any{
						"type": "string",
					},
				},
				"timeout_ms": map[string]any{
					"type":        "integer",
					"description": "Timeout in milliseconds.",
				},
				"max_output_length": map[string]any{
					"type":        "integer",
					"description": "Maximum output length to capture.",
				},
			},
			"required": []string{"commands"},
		},
	}
}

func buildResponsesApplyPatchFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapApplyPatchFunctionName,
		"description": applyPatchPreferredToolDescription,
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"operation": buildApplyPatchOperationSchema(),
			},
			"required": []string{"operation"},
		},
	}
}

func buildApplyPatchOperationSchema() map[string]any {
	return map[string]any{
		"type":        "object",
		"description": "Single file operation. Use type create_file, update_file, or delete_file. Alias spellings createfile/updatefile/deletefile are also accepted on input.",
		"properties": map[string]any{
			"type": map[string]any{
				"type":        "string",
				"enum":        []string{"create_file", "update_file", "delete_file"},
				"description": "Operation type. Prefer create_file, update_file, or delete_file.",
			},
			"path": map[string]any{
				"type":        "string",
				"description": "Absolute or workspace-relative file path.",
			},
			"diff": map[string]any{
				"type":        "string",
				"description": "Exact patch text with real file context. Do not send line-number-only hunks without matching context.",
			},
			"content": map[string]any{
				"type":        "string",
				"description": "Preferred for simple create/update operations: provide the full final file content.",
			},
		},
		"required": []string{"type", "path"},
	}
}

func buildResponsesWebSearchFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        "web_search",
		"description": "Search the web and return search parameters for the caller to execute.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query.",
				},
				"domains": map[string]any{
					"type":        "array",
					"description": "Optional domains to constrain search results.",
					"items": map[string]any{
						"type": "string",
					},
				},
				"search_context_size": map[string]any{
					"type":        "string",
					"description": "Optional context hint such as low, medium, or high.",
				},
				"user_location": map[string]any{
					"type":                 "object",
					"description":          "Optional user location metadata.",
					"additionalProperties": true,
				},
			},
			"required": []string{"query"},
		},
	}
}

func requestIncludesWebSearchTool(req map[string]any) bool {
	if req == nil {
		return false
	}
	tools, _ := req["tools"].([]any)
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		names := []string{
			strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", tool["type"]))),
			strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", tool["name"]))),
		}
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", fn["name"]))))
		}
		for _, name := range names {
			switch name {
			case "web_search", "web_search_preview", "websearch", "websearchpreview", llamaSwapWebSearchFunctionName:
				return true
			}
		}
	}
	return false
}

func rawResponsesBodyIncludesWebSearchTool(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	if gjson.ValidBytes(body) {
		for _, tool := range gjson.GetBytes(body, "tools").Array() {
			candidates := []string{
				strings.TrimSpace(strings.ToLower(tool.Get("type").String())),
				strings.TrimSpace(strings.ToLower(tool.Get("name").String())),
				strings.TrimSpace(strings.ToLower(tool.Get("function.name").String())),
			}
			for _, candidate := range candidates {
				switch candidate {
				case "web_search", "web_search_preview", "websearch", "websearchpreview", llamaSwapWebSearchFunctionName:
					return true
				}
			}
		}
	}
	lower := strings.ToLower(string(body))
	return strings.Contains(lower, `"type":"web_search"`) || strings.Contains(lower, `"type": "web_search"`)
}

func rawResponsesBodyMentionsSearchIntent(body []byte) bool {
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if lower == "" {
		return false
	}
	return strings.Contains(lower, "research the web") ||
		strings.Contains(lower, "web research") ||
		strings.Contains(lower, "search the web") ||
		strings.Contains(lower, "web search") ||
		strings.Contains(lower, "look it up")
}

func chatRequestIncludesToolName(body []byte, names ...string) bool {
	if len(body) == 0 || !gjson.ValidBytes(body) || len(names) == 0 {
		return false
	}
	want := make(map[string]struct{}, len(names))
	for _, name := range names {
		name = strings.TrimSpace(strings.ToLower(name))
		if name != "" {
			want[name] = struct{}{}
		}
	}
	for _, tool := range gjson.GetBytes(body, "tools").Array() {
		candidates := []string{
			strings.TrimSpace(strings.ToLower(tool.Get("type").String())),
			strings.TrimSpace(strings.ToLower(tool.Get("name").String())),
			strings.TrimSpace(strings.ToLower(tool.Get("function.name").String())),
		}
		for _, candidate := range candidates {
			if _, ok := want[candidate]; ok {
				return true
			}
		}
	}
	return false
}

func cloneAnySlice(items []any) []any {
	if len(items) == 0 {
		return []any{}
	}
	cloned := make([]any, 0, len(items))
	for _, raw := range items {
		if m, ok := raw.(map[string]any); ok {
			cloned = append(cloned, cloneMap(m))
			continue
		}
		cloned = append(cloned, raw)
	}
	return cloned
}

func extractStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		out := make([]string, 0, len(typed))
		for _, entry := range typed {
			if trimmed := strings.TrimSpace(entry); trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out
	case []any:
		out := make([]string, 0, len(typed))
		for _, raw := range typed {
			if trimmed := strings.TrimSpace(fmt.Sprintf("%v", raw)); trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out
	default:
		return nil
	}
}

func normalizeMapValue(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	encoded := mustJSONBytes(value)
	if len(encoded) == 0 {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(encoded, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func sanitizeBridgeWebSearchQuery(query string, domains []string) string {
	query = strings.TrimSpace(query)
	if query == "" {
		return ""
	}
	if len(domains) == 0 {
		return query
	}
	extra := make([]string, 0, len(domains))
	for _, domain := range domains {
		domain = strings.TrimSpace(domain)
		if domain == "" {
			continue
		}
		extra = append(extra, "site:"+domain)
	}
	if len(extra) == 0 {
		return query
	}
	return strings.TrimSpace(query + " " + strings.Join(extra, " "))
}

func stripBridgeHTML(text string) string {
	if strings.TrimSpace(text) == "" {
		return ""
	}
	text = bridgeHTMLTagRegexp.ReplaceAllString(text, " ")
	text = html.UnescapeString(text)
	return strings.Join(strings.Fields(text), " ")
}

func decodeDuckDuckGoResultURL(rawURL string) string {
	rawURL = html.UnescapeString(strings.TrimSpace(rawURL))
	if rawURL == "" {
		return ""
	}
	parsed, err := url.Parse(rawURL)
	if err == nil {
		if uddg := strings.TrimSpace(parsed.Query().Get("uddg")); uddg != "" {
			if decoded, decodeErr := url.QueryUnescape(uddg); decodeErr == nil && strings.TrimSpace(decoded) != "" {
				return decoded
			}
			return uddg
		}
	}
	return rawURL
}

func executeDuckDuckGoHTMLSearch(ctx context.Context, effectiveQuery string) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://html.duckduckgo.com/html/?q="+url.QueryEscape(effectiveQuery), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "llama-swap/bridge-web-search")
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("duckduckgo search returned status %d", resp.StatusCode)
	}

	titleRe := regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	snippetRe := regexp.MustCompile(`(?is)<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>`)
	titleMatches := titleRe.FindAllStringSubmatchIndex(string(body), 6)
	results := make([]any, 0, len(titleMatches))
	for idx, match := range titleMatches {
		if len(match) < 6 {
			continue
		}
		title := stripBridgeHTML(string(body[match[4]:match[5]]))
		link := decodeDuckDuckGoResultURL(string(body[match[2]:match[3]]))
		if title == "" || link == "" {
			continue
		}
		snippet := ""
		searchEnd := len(body)
		if idx+1 < len(titleMatches) && len(titleMatches[idx+1]) >= 2 {
			searchEnd = titleMatches[idx+1][0]
		}
		segment := string(body[match[1]:searchEnd])
		if snippetMatch := snippetRe.FindStringSubmatch(segment); len(snippetMatch) >= 2 {
			snippet = stripBridgeHTML(snippetMatch[1])
		}
		results = append(results, map[string]any{
			"title":   title,
			"url":     link,
			"snippet": snippet,
		})
	}
	return map[string]any{
		"provider": "duckduckgo_html",
		"results":  results,
	}, nil
}

func executeConfiguredWebSearch(ctx context.Context, endpoint string, effectiveQuery string) (map[string]any, error) {
	parsed, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil {
		return nil, err
	}
	queryValues := parsed.Query()
	if strings.TrimSpace(queryValues.Get("q")) == "" {
		queryValues.Set("q", effectiveQuery)
	}
	if strings.TrimSpace(queryValues.Get("format")) == "" {
		queryValues.Set("format", "json")
	}
	parsed.RawQuery = queryValues.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "llama-swap/bridge-web-search")
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("configured web search returned status %d", resp.StatusCode)
	}

	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, err
	}
	rawResults, _ := decoded["results"].([]any)
	if len(rawResults) == 0 {
		if alt, ok := decoded["items"].([]any); ok {
			rawResults = alt
		}
	}
	results := make([]any, 0, len(rawResults))
	for _, raw := range rawResults {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		title := strings.TrimSpace(fmt.Sprintf("%v", item["title"]))
		if title == "" {
			title = strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
		}
		link := strings.TrimSpace(fmt.Sprintf("%v", item["url"]))
		if link == "" {
			link = strings.TrimSpace(fmt.Sprintf("%v", item["link"]))
		}
		snippet := strings.TrimSpace(fmt.Sprintf("%v", item["content"]))
		if snippet == "" {
			snippet = strings.TrimSpace(fmt.Sprintf("%v", item["snippet"]))
		}
		if title == "" && link == "" && snippet == "" {
			continue
		}
		results = append(results, map[string]any{
			"title":   title,
			"url":     link,
			"snippet": snippet,
		})
		if len(results) >= 6 {
			break
		}
	}
	return map[string]any{
		"provider": strings.TrimSpace(parsed.Host),
		"results":  results,
	}, nil
}

func executeBridgeWebSearch(ctx context.Context, action map[string]any) map[string]any {
	return executeBridgeWebSearchWithSettings(ctx, action, "duckduckgo_html", strings.TrimSpace(os.Getenv(llamaSwapWebSearchURLVar)))
}

func executeBridgeWebSearchWithSettings(ctx context.Context, action map[string]any, engine string, endpoint string) map[string]any {
	query := strings.TrimSpace(fmt.Sprintf("%v", action["query"]))
	domains := extractStringSlice(action["domains"])
	effectiveQuery := sanitizeBridgeWebSearchQuery(query, domains)
	payload := map[string]any{
		"ok":                  true,
		"query":               query,
		"effective_query":     effectiveQuery,
		"domains":             domains,
		"search_context_size": strings.TrimSpace(fmt.Sprintf("%v", action["search_context_size"])),
		"fetched_at":          time.Now().UTC().Format(time.RFC3339),
	}
	if userLocation, ok := action["user_location"].(map[string]any); ok && len(userLocation) > 0 {
		payload["user_location"] = userLocation
	}
	if effectiveQuery == "" {
		payload["ok"] = false
		payload["error"] = "missing web search query"
		payload["results"] = []any{}
		return payload
	}

	var (
		result map[string]any
		err    error
	)
	engine = strings.TrimSpace(strings.ToLower(engine))
	switch {
	case engine == "searxng" && strings.TrimSpace(endpoint) != "":
		result, err = executeConfiguredWebSearch(ctx, endpoint, effectiveQuery)
	case (engine == "" || engine == "duckduckgo_html") && strings.TrimSpace(endpoint) != "" && strings.TrimSpace(os.Getenv(llamaSwapWebSearchURLVar)) == strings.TrimSpace(endpoint):
		result, err = executeConfiguredWebSearch(ctx, endpoint, effectiveQuery)
	case engine == "" || engine == "duckduckgo_html":
		result, err = executeDuckDuckGoHTMLSearch(ctx, effectiveQuery)
	case engine == "searxng":
		err = fmt.Errorf("searxng fallback selected but no endpoint configured")
	default:
		err = fmt.Errorf("unsupported web search engine %q", engine)
	}
	if err != nil {
		payload["ok"] = false
		payload["error"] = err.Error()
		payload["results"] = []any{}
		payload["provider"] = "bridge_error"
		return payload
	}
	for key, value := range result {
		payload[key] = value
	}
	if _, ok := payload["results"]; !ok {
		payload["results"] = []any{}
	}
	return payload
}

func normalizeWebSearchFallbackEngine(engine string) string {
	switch strings.ToLower(strings.TrimSpace(engine)) {
	case "searxng":
		return "searxng"
	default:
		return "duckduckgo_html"
	}
}

func envBoolTrue(name string) bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func (pm *ProxyManager) getWebSearchFallbackSettings() (bool, string, string) {
	if pm == nil {
		return true, "duckduckgo_html", strings.TrimSpace(os.Getenv(llamaSwapWebSearchURLVar))
	}
	pm.Lock()
	defer pm.Unlock()
	engine := normalizeWebSearchFallbackEngine(pm.webSearchFallbackEngine)
	endpoint := strings.TrimSpace(pm.webSearchFallbackURL)
	if endpoint == "" {
		endpoint = strings.TrimSpace(os.Getenv(llamaSwapWebSearchURLVar))
	}
	return pm.webSearchFallbackEnabled, engine, endpoint
}

func (pm *ProxyManager) getManagedWebSearchSettings() (bool, string, string) {
	if pm == nil {
		return false, strings.TrimSpace(os.Getenv(llamaSwapSearxngCommandVar)), strings.TrimSpace(os.Getenv(llamaSwapSearxngStopCommandVar))
	}
	pm.Lock()
	defer pm.Unlock()
	return pm.webSearchManagedEnabled, strings.TrimSpace(pm.webSearchManagedCommand), strings.TrimSpace(pm.webSearchManagedStopCmd)
}

func (pm *ProxyManager) setWebSearchFallbackSettings(enabled bool, engine string, endpoint string) {
	if pm == nil {
		return
	}
	pm.Lock()
	defer pm.Unlock()
	pm.webSearchFallbackEnabled = enabled
	pm.webSearchFallbackEngine = normalizeWebSearchFallbackEngine(engine)
	pm.webSearchFallbackURL = strings.TrimSpace(endpoint)
}

func (pm *ProxyManager) setManagedWebSearchSettings(enabled bool, command string, stopCommand string) {
	if pm == nil {
		return
	}
	pm.Lock()
	pm.webSearchManagedEnabled = enabled
	pm.webSearchManagedCommand = strings.TrimSpace(command)
	pm.webSearchManagedStopCmd = strings.TrimSpace(stopCommand)
	service := pm.webSearchManagedService
	pm.Unlock()
	if service != nil {
		service.SetStopCommand(stopCommand)
	}
}

func (pm *ProxyManager) syncManagedWebSearchProcess() error {
	if pm == nil {
		return nil
	}
	enabled, command, stopCommand := pm.getManagedWebSearchSettings()
	service := pm.webSearchManagedService
	if service == nil {
		return nil
	}
	service.SetStopCommand(stopCommand)
	if !enabled || strings.TrimSpace(command) == "" {
		return service.Stop()
	}
	return service.Start(command)
}

func (pm *ProxyManager) managedWebSearchStatus() string {
	if pm == nil || pm.webSearchManagedService == nil {
		return "stopped"
	}
	return pm.webSearchManagedService.Status()
}

func (pm *ProxyManager) executeBridgeWebSearch(ctx context.Context, action map[string]any) map[string]any {
	enabled, engine, endpoint := pm.getWebSearchFallbackSettings()
	if !enabled {
		return map[string]any{
			"ok":       false,
			"provider": "disabled",
			"error":    "llama-swap local web-search fallback is disabled",
			"results":  []any{},
			"query":    strings.TrimSpace(fmt.Sprintf("%v", action["query"])),
		}
	}
	return executeBridgeWebSearchWithSettings(ctx, action, engine, endpoint)
}

func executeBridgeFileSearch(action map[string]any) map[string]any {
	query := strings.TrimSpace(fmt.Sprintf("%v", action["query"]))
	maxResults := 5
	switch typed := action["max_num_results"].(type) {
	case float64:
		if typed > 0 {
			maxResults = int(typed)
		}
	case int:
		if typed > 0 {
			maxResults = typed
		}
	}
	if maxResults <= 0 {
		maxResults = 5
	}
	searchRoot := strings.TrimSpace(cleanFallbackInput(action["root"], ""))
	if searchRoot == "" {
		searchRoot = strings.TrimSpace(cleanFallbackInput(action["search_root"], ""))
	}
	if searchRoot == "" {
		searchRoot = "."
	}
	for _, field := range strings.Fields(query) {
		candidatePath := strings.Trim(field, " \t\r\n\"'`,;:()[]{}")
		if candidatePath == "" {
			continue
		}
		if strings.HasPrefix(candidatePath, "./") || strings.HasPrefix(candidatePath, "/") {
			searchRoot = candidatePath
			query = strings.Replace(query, field, "", 1)
			break
		}
	}
	searchRoot = resolveBridgeSearchRoot(searchRoot)
	normalizedQuery := strings.ToLower(strings.TrimSpace(query))
	rootTermExclusions := buildFileSearchRootTermExclusions(searchRoot)
	lowerSearchRoot := strings.ToLower(filepath.ToSlash(searchRoot))
	terms := make([]string, 0, 4)
	for _, field := range strings.Fields(normalizedQuery) {
		trimmed := strings.Trim(field, " \t\r\n\"'`,.;:()[]{}")
		if trimmed == "" {
			continue
		}
		switch trimmed {
		case "inside", "within", "under", "search", "find", "the", "a", "an", "for", "file_search":
			continue
		}
		if strings.ContainsAny(trimmed, `/\`) {
			normalizedTermPath := strings.Trim(strings.ReplaceAll(trimmed, `\`, `/`), `/`)
			if normalizedTermPath != "" && strings.Contains(lowerSearchRoot, normalizedTermPath) {
				continue
			}
			pathParts := strings.FieldsFunc(normalizedTermPath, func(r rune) bool {
				return r == '/' || r == '\\' || r == '.' || r == '-' || r == '_'
			})
			if len(pathParts) > 0 {
				allExcluded := true
				for _, part := range pathParts {
					if !rootTermExclusions[strings.TrimSpace(part)] {
						allExcluded = false
						break
					}
				}
				if allExcluded {
					continue
				}
			}
		}
		if rootTermExclusions[trimmed] {
			continue
		}
		terms = append(terms, trimmed)
	}
	matches := make([]map[string]any, 0, maxResults)
	walkErr := filepath.Walk(searchRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil {
			return nil
		}
		if info.IsDir() {
			return nil
		}
		if len(matches) >= maxResults {
			return io.EOF
		}
		if info.Size() > 1<<20 {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil || !utf8.Valid(data) {
			return nil
		}
		content := string(data)
		lowerContent := strings.ToLower(content)
		matched := len(terms) == 0
		for _, term := range terms {
			if strings.Contains(lowerContent, term) {
				matched = true
				continue
			}
			matched = false
			break
		}
		if !matched {
			return nil
		}
		relPath := path
		if rel, relErr := filepath.Rel(".", path); relErr == nil {
			relPath = rel
		}
		excerpt := strings.TrimSpace(content)
		if len(excerpt) > 400 {
			excerpt = excerpt[:400]
		}
		matches = append(matches, map[string]any{
			"path":    filepath.ToSlash(relPath),
			"excerpt": excerpt,
		})
		return nil
	})
	if walkErr != nil && !errors.Is(walkErr, io.EOF) {
		return map[string]any{
			"ok":      false,
			"query":   strings.TrimSpace(fmt.Sprintf("%v", action["query"])),
			"matches": []any{},
			"error":   walkErr.Error(),
			"root":    filepath.ToSlash(searchRoot),
		}
	}
	return map[string]any{
		"ok":      true,
		"query":   strings.TrimSpace(fmt.Sprintf("%v", action["query"])),
		"matches": matches,
		"root":    filepath.ToSlash(searchRoot),
	}
}

func buildFileSearchRootTermExclusions(searchRoot string) map[string]bool {
	exclusions := map[string]bool{}
	base := strings.TrimSpace(strings.ToLower(filepath.Base(filepath.Clean(searchRoot))))
	if base != "" && base != "." && base != string(filepath.Separator) {
		exclusions[base] = true
	}
	for _, part := range strings.FieldsFunc(strings.ToLower(filepath.ToSlash(searchRoot)), func(r rune) bool {
		return r == '/' || r == '\\' || r == '.' || r == '-' || r == '_'
	}) {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		exclusions[part] = true
	}
	return exclusions
}

func extractFileSearchRootHintFromRequest(req map[string]any) string {
	if req == nil {
		return ""
	}
	return extractFileSearchRootHintFromText(
		extractResponsesExplicitUserInputText(req),
		extractResponsesAllExplicitUserInputText(req),
	)
}

func extractFileSearchRootHintFromRequestBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return extractFileSearchRootHintFromRequest(req)
}

func extractFileSearchRootHintFromText(text string, contextText string) string {
	candidates := []string{text}
	if trimmed := strings.TrimSpace(contextText); trimmed != "" && !strings.EqualFold(strings.TrimSpace(text), trimmed) {
		candidates = append(candidates, trimmed)
	}
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:inside|within|under|in)\s+([^\s"'` + "`" + `]+)`),
		regexp.MustCompile(`(?i)\b(?:workspace|directory|folder)\s+([^\s"'` + "`" + `]+)`),
	}
	for _, source := range candidates {
		source = strings.TrimSpace(source)
		if source == "" {
			continue
		}
		for _, re := range patterns {
			matches := re.FindAllStringSubmatch(source, -1)
			for _, match := range matches {
				if len(match) < 2 {
					continue
				}
				if resolved := resolveFileSearchRootCandidate(match[1], source); resolved != "" {
					return resolved
				}
			}
		}
		for _, field := range strings.Fields(source) {
			if resolved := resolveFileSearchRootCandidate(field, source); resolved != "" {
				return resolved
			}
		}
	}
	return ""
}

func resolveFileSearchRootCandidate(candidate string, requestText string) string {
	candidate = strings.TrimSpace(candidate)
	candidate = strings.Trim(candidate, "\"'`,;:!?)(")
	candidate = strings.TrimRight(candidate, ".")
	if candidate == "" {
		return ""
	}
	candidate = normalizeApplyPatchPathForWorkspace(candidate)
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		return candidate
	}
	workspaceRoot := strings.TrimSpace(extractWorkspaceRootHintFromResponsesText(requestText))
	if workspaceRoot == "" {
		return ""
	}
	joined := normalizeApplyPatchPathForWorkspace(filepath.Join(workspaceRoot, candidate))
	if info, err := os.Stat(joined); err == nil && info.IsDir() {
		return joined
	}
	return ""
}

func resolveBridgeSearchRoot(searchRoot string) string {
	searchRoot = strings.TrimSpace(searchRoot)
	if searchRoot == "" {
		searchRoot = "."
	}
	candidates := make([]string, 0, 6)
	addCandidate := func(base string) {
		base = strings.TrimSpace(base)
		if base == "" {
			return
		}
		candidate := searchRoot
		if filepath.IsAbs(searchRoot) {
			candidate = searchRoot
		} else {
			candidate = filepath.Join(base, searchRoot)
		}
		candidates = append(candidates, candidate)
	}
	if filepath.IsAbs(searchRoot) {
		candidates = append(candidates, searchRoot)
	} else {
		if cwd, err := os.Getwd(); err == nil {
			addCandidate(cwd)
		}
		addCandidate(os.Getenv("PWD"))
		if exePath, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exePath)
			addCandidate(exeDir)
			addCandidate(filepath.Join(exeDir, "..", "llama-swap"))
		}
	}
	for _, candidate := range candidates {
		if abs, err := filepath.Abs(candidate); err == nil {
			if info, statErr := os.Stat(abs); statErr == nil && info.IsDir() {
				return abs
			}
		}
	}
	if abs, err := filepath.Abs(searchRoot); err == nil {
		return abs
	}
	return searchRoot
}

func extractPendingBridgeWebSearchCalls(responseBody []byte) []map[string]any {
	if !gjson.ValidBytes(responseBody) {
		return nil
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return nil
	}
	output, _ := resp["output"].([]any)
	completed := map[string]bool{}
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) == "web_search_call_output" {
			completed[strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))] = true
		}
	}
	pending := make([]map[string]any, 0, 2)
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "web_search_call" {
			continue
		}
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		if completed[callID] {
			continue
		}
		pending = append(pending, cloneMap(item))
	}
	return pending
}

func extractPendingBridgeFileSearchCalls(responseBody []byte) []map[string]any {
	if !gjson.ValidBytes(responseBody) {
		return nil
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return nil
	}
	output, _ := resp["output"].([]any)
	completed := map[string]bool{}
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) == "file_search_call_output" {
			completed[strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))] = true
		}
	}
	pending := make([]map[string]any, 0, 2)
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "file_search_call" {
			continue
		}
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		if completed[callID] {
			continue
		}
		pending = append(pending, cloneMap(item))
	}
	return pending
}

func appendResponsesInputItems(body []byte, items ...map[string]any) ([]byte, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	input, _ := req["input"].([]any)
	if input == nil {
		input = []any{}
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		input = append(input, cloneMap(item))
	}
	req["input"] = input
	return json.Marshal(req)
}

func prependResponsesOutputItems(body []byte, items []any) ([]byte, error) {
	if len(items) == 0 {
		return body, nil
	}
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	output, _ := resp["output"].([]any)
	resp["output"] = append(cloneAnySlice(items), output...)
	normalizeTranslatedResponsesOutput(resp)
	return json.Marshal(resp)
}

func replacePendingWebSearchCallWithResolvedItems(body []byte, resolvedCall map[string]any, resolvedOutput map[string]any) ([]byte, error) {
	if !gjson.ValidBytes(body) {
		return body, nil
	}
	callID := strings.TrimSpace(fmt.Sprintf("%v", resolvedCall["call_id"]))
	if callID == "" {
		return body, nil
	}
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	output, _ := resp["output"].([]any)
	rewritten := make([]any, 0, len(output)+1)
	replaced := false
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			rewritten = append(rewritten, raw)
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) == "web_search_call" &&
			strings.TrimSpace(fmt.Sprintf("%v", item["call_id"])) == callID {
			rewritten = append(rewritten, cloneMap(resolvedCall), cloneMap(resolvedOutput))
			replaced = true
			continue
		}
		rewritten = append(rewritten, raw)
	}
	if !replaced {
		rewritten = append([]any{cloneMap(resolvedCall), cloneMap(resolvedOutput)}, rewritten...)
	}
	resp["output"] = rewritten
	normalizeTranslatedResponsesOutput(resp)
	return json.Marshal(resp)
}

func replacePendingWebSearchCallsWithResolvedItems(body []byte, resolvedCalls []map[string]any, resolvedOutputs []map[string]any) ([]byte, error) {
	if !gjson.ValidBytes(body) {
		return body, nil
	}
	if len(resolvedCalls) == 0 || len(resolvedCalls) != len(resolvedOutputs) {
		return body, nil
	}
	rewritten := body
	var err error
	for idx := range resolvedCalls {
		rewritten, err = replacePendingWebSearchCallWithResolvedItems(rewritten, resolvedCalls[idx], resolvedOutputs[idx])
		if err != nil {
			return nil, err
		}
	}
	return rewritten, nil
}

func replacePendingFileSearchCallWithResolvedItems(body []byte, resolvedCall map[string]any, resolvedOutput map[string]any) ([]byte, error) {
	if !gjson.ValidBytes(body) {
		return body, nil
	}
	callID := strings.TrimSpace(fmt.Sprintf("%v", resolvedCall["call_id"]))
	if callID == "" {
		return body, nil
	}
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	output, _ := resp["output"].([]any)
	rewritten := make([]any, 0, len(output)+1)
	replaced := false
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			rewritten = append(rewritten, raw)
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) == "file_search_call" &&
			strings.TrimSpace(fmt.Sprintf("%v", item["call_id"])) == callID {
			rewritten = append(rewritten, cloneMap(resolvedCall), cloneMap(resolvedOutput))
			replaced = true
			continue
		}
		rewritten = append(rewritten, raw)
	}
	if !replaced {
		rewritten = append([]any{cloneMap(resolvedCall), cloneMap(resolvedOutput)}, rewritten...)
	}
	resp["output"] = rewritten
	normalizeTranslatedResponsesOutput(resp)
	return json.Marshal(resp)
}

func replacePendingFileSearchCallsWithResolvedItems(body []byte, resolvedCalls []map[string]any, resolvedOutputs []map[string]any) ([]byte, error) {
	if !gjson.ValidBytes(body) {
		return body, nil
	}
	if len(resolvedCalls) == 0 || len(resolvedCalls) != len(resolvedOutputs) {
		return body, nil
	}
	rewritten := body
	var err error
	for idx := range resolvedCalls {
		rewritten, err = replacePendingFileSearchCallWithResolvedItems(rewritten, resolvedCalls[idx], resolvedOutputs[idx])
		if err != nil {
			return nil, err
		}
	}
	return rewritten, nil
}

func buildResponsesFileSearchFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapFileSearchFunctionName,
		"description": "Compatibility wrapper for the Responses API file_search tool. Return a file search request for the caller to execute.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":            map[string]any{"type": "string", "description": "The file search query."},
				"vector_store_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional vector store identifiers."},
				"max_num_results":  map[string]any{"type": "integer", "description": "Optional maximum result count."},
				"filters":          map[string]any{"type": "object", "additionalProperties": true, "description": "Optional structured filters."},
			},
			"required": []string{"query"},
		},
	}
}

func buildResponsesCodeInterpreterFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapCodeInterpreterFunctionName,
		"description": "Compatibility wrapper for the Responses API code_interpreter tool. Return an execution request for the caller to run.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"code":     map[string]any{"type": "string", "description": "Code to execute."},
				"language": map[string]any{"type": "string", "description": "Execution language, such as python."},
				"files":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional file references."},
				"args":     map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional execution arguments."},
			},
			"required": []string{"code"},
		},
	}
}

func buildResponsesImageGenerationFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapImageGenerationFunctionName,
		"description": "Compatibility wrapper for the Responses API image_generation tool. Return an image generation request for the caller to execute.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prompt":        map[string]any{"type": "string", "description": "The image generation prompt."},
				"size":          map[string]any{"type": "string", "description": "Optional image size."},
				"quality":       map[string]any{"type": "string", "description": "Optional image quality."},
				"background":    map[string]any{"type": "string", "description": "Optional background mode."},
				"output_format": map[string]any{"type": "string", "description": "Optional output format."},
			},
			"required": []string{"prompt"},
		},
	}
}

func buildResponsesComputerFunctionTool() map[string]any {
	return map[string]any{
		"type":        "function",
		"name":        llamaSwapComputerFunctionName,
		"description": "Compatibility wrapper for the Responses API computer tool. Return a desktop automation action for the caller to execute.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{"type": "string", "description": "Computer action such as click, type, keypress, scroll, or screenshot."},
				"x":      map[string]any{"type": "number", "description": "Optional X coordinate."},
				"y":      map[string]any{"type": "number", "description": "Optional Y coordinate."},
				"text":   map[string]any{"type": "string", "description": "Optional text payload."},
				"button": map[string]any{"type": "string", "description": "Optional mouse button."},
			},
			"required": []string{"action"},
		},
	}
}

type reasoningEffortProfile struct {
	instruction            string
	hasEnableThinking      bool
	enableThinking         bool
	hasCloseThinkBias      bool
	closeThinkBias         float64
	hasCloseThinkGuardRule bool
	closeThinkGuardRule    string
}

const (
	qwenCloseThinkTokenID          = "248069"
	qwenMediumReasoningCloseBias   = 11.8
	qwenCloseThinkSingleUseGrammar = "root ::= pre <[248069]> post\npre ::= !<[248069]>*\npost ::= !<[248069]>*"
)

func normalizeResponsesReasoningEffort(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "minimal", "low":
		return "low"
	case "medium", "med", "balanced", "normal", "default":
		return "medium"
	case "high":
		return "high"
	case "xhigh", "extra_high", "extrahigh", "very_high", "max":
		return "extrahigh"
	default:
		return ""
	}
}

func reasoningEffortProfileForEffort(effort string) (reasoningEffortProfile, bool) {
	switch normalizeResponsesReasoningEffort(effort) {
	case "low":
		return reasoningEffortProfile{
			instruction:       "Reasoning style: keep the response concise and direct.",
			hasEnableThinking: true,
			enableThinking:    false,
		}, true
	case "medium":
		return reasoningEffortProfile{
			instruction:            "Reasoning style: keep reasoning focused and concise before the final answer.",
			hasEnableThinking:      true,
			enableThinking:         true,
			hasCloseThinkBias:      true,
			closeThinkBias:         qwenMediumReasoningCloseBias,
			hasCloseThinkGuardRule: true,
			closeThinkGuardRule:    qwenCloseThinkSingleUseGrammar,
		}, true
	case "high":
		return reasoningEffortProfile{
			instruction:       "",
			hasEnableThinking: true,
			enableThinking:    true,
		}, true
	case "extrahigh":
		return reasoningEffortProfile{
			instruction:       "Reasoning style: explain the solution in clear multi-step detail.",
			hasEnableThinking: true,
			enableThinking:    true,
		}, true
	default:
		return reasoningEffortProfile{}, false
	}
}

func extractSlashDirectiveFromResponsesInput(req map[string]any, directive string) string {
	if req == nil {
		return ""
	}
	directive = strings.ToLower(strings.TrimSpace(directive))
	if directive == "" {
		return ""
	}
	input, _ := req["input"].([]any)
	lastValue := ""
	prefix := "/" + directive
	for _, raw := range input {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["type"]))) != "message" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["role"]))) != "user" {
			continue
		}
		text := extractResponsesInputText(msg["content"])
		for _, line := range strings.Split(text, "\n") {
			trimmed := strings.TrimSpace(strings.ToLower(line))
			if !strings.HasPrefix(trimmed, prefix) {
				continue
			}
			fields := strings.Fields(trimmed)
			if len(fields) >= 2 {
				lastValue = fields[1]
			}
		}
	}
	return strings.TrimSpace(lastValue)
}

func filterSlashDirectiveLines(text string) (string, bool) {
	if strings.TrimSpace(text) == "" {
		return text, false
	}
	lines := strings.Split(text, "\n")
	kept := make([]string, 0, len(lines))
	changed := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(strings.ToLower(line))
		if strings.HasPrefix(trimmed, "/mode ") || strings.HasPrefix(trimmed, "/reasoning ") {
			changed = true
			continue
		}
		kept = append(kept, line)
	}
	if !changed {
		return text, false
	}
	return strings.TrimSpace(strings.Join(kept, "\n")), true
}

func stripSlashDirectivesFromMessageContent(content any) (any, bool) {
	switch typed := content.(type) {
	case string:
		filtered, changed := filterSlashDirectiveLines(typed)
		if !changed {
			return content, false
		}
		return filtered, true
	case []any:
		changed := false
		rewritten := make([]any, 0, len(typed))
		for _, rawPart := range typed {
			part, ok := rawPart.(map[string]any)
			if !ok {
				rewritten = append(rewritten, rawPart)
				continue
			}
			updated := cloneMap(part)
			if text, exists := updated["text"]; exists {
				filtered, partChanged := filterSlashDirectiveLines(fmt.Sprintf("%v", text))
				if partChanged {
					updated["text"] = filtered
					changed = true
				}
			}
			rewritten = append(rewritten, updated)
		}
		if !changed {
			return content, false
		}
		return rewritten, true
	default:
		return content, false
	}
}

func stripSlashDirectivesFromResponsesInput(req map[string]any) bool {
	if req == nil {
		return false
	}
	input, ok := req["input"].([]any)
	if !ok || len(input) == 0 {
		return false
	}
	changed := false
	for idx, raw := range input {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["type"]))) != "message" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["role"]))) != "user" {
			continue
		}
		content, contentChanged := stripSlashDirectivesFromMessageContent(msg["content"])
		if !contentChanged {
			continue
		}
		updated := cloneMap(msg)
		updated["content"] = content
		input[idx] = updated
		changed = true
	}
	if changed {
		req["input"] = input
	}
	return changed
}

func extractResponsesRequestMode(req map[string]any) string {
	if req == nil {
		return ""
	}
	mode := ""
	if rawMode, hasMode := req["mode"]; hasMode && rawMode != nil {
		mode = strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", rawMode)))
	}
	if mode == "" {
		mode = extractSlashDirectiveFromResponsesInput(req, "mode")
	}
	return strings.ToLower(strings.TrimSpace(mode))
}

func responsesRequestContractMode(req map[string]any) string {
	if req == nil {
		return "default"
	}
	if extractResponsesRequestMode(req) == "plan" || isCodexManagedPlanModeFromResponsesRequest(req) {
		return "plan"
	}
	return "default"
}

func requestUserInputUnavailableInDefaultMode(req map[string]any) bool {
	if req == nil || responsesRequestContractMode(req) != "default" {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractTrustedResponsesInstructionText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "request_user_input is unavailable in default mode")
}

func requestLooksLikePlanMode(req map[string]any) bool {
	if req == nil {
		return false
	}

	safeCombined := strings.ToLower(strings.TrimSpace(extractTrustedResponsesInstructionText(req)))
	if safeCombined == "" {
		return false
	}

	if strings.Contains(safeCombined, "/mode plan") || strings.Contains(safeCombined, "mode: plan") {
		return true
	}
	if strings.Contains(safeCombined, " --plan") || strings.HasPrefix(strings.TrimSpace(safeCombined), "--plan") {
		return true
	}
	return false
}

func rawResponsesBodyLooksLikePlanMode(body []byte) bool {
	if gjson.ValidBytes(body) {
		var req map[string]any
		if err := json.Unmarshal(body, &req); err == nil {
			return responsesRequestContractMode(req) == "plan" || requestLooksLikePlanMode(req)
		}
	}
	lower := strings.ToLower(strings.TrimSpace(string(body)))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, `"mode":"plan"`) || strings.Contains(lower, `"mode": "plan"`) {
		return true
	}
	if strings.Contains(lower, "<collaboration_mode>") || strings.Contains(lower, `\u003ccollaboration_mode\u003e`) ||
		strings.Contains(lower, "<collaborationmode>") || strings.Contains(lower, `\u003ccollaborationmode\u003e`) {
		return strings.Contains(lower, "plan mode")
	}
	if strings.Contains(lower, "/mode plan") || strings.Contains(lower, "mode: plan") {
		return true
	}
	if strings.Contains(lower, " --plan") || strings.HasPrefix(lower, "--plan") {
		return true
	}
	return false
}

type ToolTier int

const (
	TierRead ToolTier = iota
	TierWrite
	TierDestructive
)

var toolTierRegistry = map[string]ToolTier{
	"request_user_input": TierRead,
	"update_plan":        TierRead,
	"read_file":          TierRead,
	"list_dir":           TierRead,
	"stat":               TierRead,
	"find_files":         TierRead,
	"grep":               TierRead,
	"apply_patch":        TierWrite,
	"write_file":         TierWrite,
	"create_file":        TierWrite,
	"patch_file":         TierWrite,
	"append_to_file":     TierWrite,
	"move_file":          TierWrite,
	"rename_file":        TierWrite,
	"run_command":        TierWrite,
	"shell_exec":         TierWrite,
	"execute_command":    TierWrite,
	"bash":               TierWrite,
	"http_post":          TierWrite,
	"http_put":           TierWrite,
	"http_patch":         TierWrite,
	"db_execute":         TierWrite,
	"kubectl_apply":      TierWrite,
	"terraform_apply":    TierWrite,
	"docker_build":       TierWrite,
	"delete_file":        TierDestructive,
	"git_push":           TierDestructive,
	"git_reset_hard":     TierDestructive,
	"git_push_force":     TierDestructive,
	"http_delete":        TierDestructive,
	"kubectl_delete":     TierDestructive,
	"terraform_destroy":  TierDestructive,
	"db_drop":            TierDestructive,
	"db_truncate":        TierDestructive,
}

func normalizeToolTierName(name string) string {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if normalized == "" {
		return ""
	}
	normalized = strings.TrimPrefix(normalized, "__llamaswap_")
	switch normalized {
	case "applypatch":
		return "apply_patch"
	case llamaSwapApplyPatchFunctionName:
		return "apply_patch"
	case "exec_command", "write_stdin", "shellcommand":
		return "shell"
	}
	if idx := strings.LastIndex(normalized, "__"); idx >= 0 && idx+2 < len(normalized) {
		normalized = normalized[idx+2:]
	}
	return normalized
}

func extractFunctionToolName(tool map[string]any) string {
	if tool == nil {
		return ""
	}
	if name, _ := tool["name"].(string); strings.TrimSpace(name) != "" {
		return strings.TrimSpace(name)
	}
	if fn, ok := tool["function"].(map[string]any); ok {
		if name, _ := fn["name"].(string); strings.TrimSpace(name) != "" {
			return strings.TrimSpace(name)
		}
	}
	return ""
}

func classifyTool(name string) ToolTier {
	normalized := normalizeToolTierName(name)
	if normalized == "" {
		return TierWrite
	}
	if tier, ok := toolTierRegistry[normalized]; ok {
		return tier
	}
	return TierWrite
}

func lookupToolTier(name string) (ToolTier, bool) {
	normalized := normalizeToolTierName(name)
	if normalized == "" {
		return TierWrite, false
	}
	tier, ok := toolTierRegistry[normalized]
	return tier, ok
}

func toolIsMutating(name string) bool {
	return classifyTool(name) >= TierWrite
}

func isCodexManagedPlanMode(messages []map[string]any) bool {
	lastBlock := ""
	hasTag := false
	for _, msg := range messages {
		role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", msg["role"])))
		if role != "system" && role != "developer" {
			continue
		}
		content := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", msg["content"])))
		if content == "" {
			continue
		}

		if block, ok := extractLastCollaborationModeTagBlock(content); ok {
			lastBlock = block
			hasTag = true
		}
	}
	if !hasTag || strings.TrimSpace(lastBlock) == "" {
		return false
	}
	mode := inferCollaborationModeFromTagBlock(lastBlock)
	return mode == "plan"
}

func isCodexManagedPlanModeFromResponsesRequest(req map[string]any) bool {
	if req == nil {
		return false
	}
	input, _ := req["input"].([]any)
	if len(input) == 0 {
		// Allow detection from top-level instructions when input is missing.
		instructions := strings.TrimSpace(extractResponsesInputText(req["instructions"]))
		if instructions == "" {
			return false
		}
		return isCodexManagedPlanMode([]map[string]any{{"role": "system", "content": instructions}})
	}

	systemMessages := make([]map[string]any, 0, len(input)+1)
	if instructions := strings.TrimSpace(extractResponsesInputText(req["instructions"])); instructions != "" {
		systemMessages = append(systemMessages, map[string]any{
			"role":    "system",
			"content": instructions,
		})
	}
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["type"])))
		role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"])))
		if itemType != "message" || (role != "system" && role != "developer") {
			continue
		}
		systemMessages = append(systemMessages, map[string]any{
			"role":    role,
			"content": extractResponsesInputText(item["content"]),
		})
	}
	return isCodexManagedPlanMode(systemMessages)
}

func isCodexManagedPlanModeFromResponsesBody(body []byte) bool {
	if !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return isCodexManagedPlanModeFromResponsesRequest(req)
}

func extractResponsesRequestReasoningEffort(req map[string]any) string {
	if req == nil {
		return ""
	}
	effort := ""
	switch typed := req["reasoning"].(type) {
	case string:
		effort = typed
	case map[string]any:
		if v := strings.TrimSpace(fmt.Sprintf("%v", typed["effort"])); v != "" {
			effort = v
		} else if v := strings.TrimSpace(fmt.Sprintf("%v", typed["level"])); v != "" {
			effort = v
		}
	}
	if slashEffort := extractSlashDirectiveFromResponsesInput(req, "reasoning"); strings.TrimSpace(slashEffort) != "" {
		effort = slashEffort
	}
	return normalizeResponsesReasoningEffort(effort)
}

func extractResponsesRequestReasoningSummary(req map[string]any) string {
	if req == nil {
		return ""
	}
	typed, ok := req["reasoning"].(map[string]any)
	if !ok {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", typed["summary"])))
}

func normalizeResponsesReasoningSummary(summary string) string {
	trimmed := strings.ToLower(strings.TrimSpace(summary))
	switch trimmed {
	case "none", "concise", "detailed", "auto":
		return trimmed
	default:
		return ""
	}
}

func extractResponsesRequestReasoningSummaryFromBody(body []byte) string {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return normalizeResponsesReasoningSummary(extractResponsesRequestReasoningSummary(req))
}

func reasoningSummaryRequestsThinking(req map[string]any) bool {
	switch extractResponsesRequestReasoningSummary(req) {
	case "auto", "detailed":
		return true
	default:
		return false
	}
}

func stripUnsupportedResponsesInclude(req map[string]any) {
	if req == nil {
		return
	}
	raw, ok := req["include"]
	if !ok {
		return
	}
	items, ok := raw.([]any)
	if !ok {
		delete(req, "include")
		return
	}
	filtered := make([]any, 0, len(items))
	for _, rawItem := range items {
		value := strings.TrimSpace(fmt.Sprintf("%v", rawItem))
		if value == "" || strings.EqualFold(value, "reasoning.encrypted_content") {
			continue
		}
		filtered = append(filtered, value)
	}
	if len(filtered) == 0 {
		delete(req, "include")
		return
	}
	req["include"] = filtered
}

func extractLastTagBlockLooseWithStart(text, tag string) (content string, blockStart int, ok bool) {
	lower := strings.ToLower(text)
	openTag := "<" + strings.ToLower(tag) + ">"
	openIdx := strings.LastIndex(lower, openTag)
	if openIdx < 0 {
		return "", 0, false
	}
	start := openIdx + len(openTag)
	closeTag := "</" + strings.ToLower(tag) + ">"
	closeRel := strings.Index(lower[start:], closeTag)
	if closeRel < 0 {
		return text[start:], openIdx, true
	}
	return text[start : start+closeRel], openIdx, true
}

func extractLastCollaborationModeTagBlock(text string) (string, bool) {
	blockA, startA, okA := extractLastTagBlockLooseWithStart(text, "collaboration_mode")
	blockB, startB, okB := extractLastTagBlockLooseWithStart(text, "collaborationmode")
	switch {
	case okA && okB:
		if startA >= startB {
			return blockA, true
		}
		return blockB, true
	case okA:
		return blockA, true
	case okB:
		return blockB, true
	default:
		return "", false
	}
}

func inferCollaborationModeFromTagBlock(block string) string {
	lower := strings.ToLower(strings.TrimSpace(block))
	if lower == "" {
		return ""
	}
	if strings.Contains(lower, "collaboration mode: default") || strings.Contains(lower, "default mode") {
		return "default"
	}
	if strings.Contains(lower, "collaboration mode: plan") || strings.Contains(lower, "plan mode") {
		return "plan"
	}
	return ""
}

func extractTrustedResponsesInstructionText(req map[string]any) string {
	if req == nil {
		return ""
	}
	parts := make([]string, 0, 8)
	if instructions := strings.TrimSpace(extractResponsesInputText(req["instructions"])); instructions != "" {
		parts = append(parts, instructions)
	}
	if inputArr, ok := req["input"].([]any); ok {
		for _, rawItem := range inputArr {
			item, ok := rawItem.(map[string]any)
			if !ok {
				continue
			}
			if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["type"]))) != "message" {
				continue
			}
			role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"])))
			if role != "system" && role != "developer" {
				continue
			}
			text := strings.TrimSpace(extractResponsesInputText(item["content"]))
			if text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func extractEffectiveCollaborationModeFromResponsesRequest(req map[string]any) string {
	safeCombined := extractTrustedResponsesInstructionText(req)
	if safeCombined == "" {
		return ""
	}

	lastBlock := ""
	if block, ok := extractLastCollaborationModeTagBlock(safeCombined); ok {
		lastBlock = block
	}
	if strings.TrimSpace(lastBlock) != "" {
		return inferCollaborationModeFromTagBlock(lastBlock)
	}
	// Non-tagged fallback (trusted roles only).
	defaultIdx := strings.LastIndex(strings.ToLower(safeCombined), "collaboration mode: default")
	planIdx := strings.LastIndex(strings.ToLower(safeCombined), "collaboration mode: plan")
	if defaultIdx > planIdx && defaultIdx >= 0 {
		return "default"
	}
	if planIdx > defaultIdx && planIdx >= 0 {
		return "plan"
	}
	return ""
}

func sanitizeResponsesInputToolArguments(req map[string]any) bool {
	if req == nil {
		return false
	}
	input, ok := req["input"].([]any)
	if !ok || len(input) == 0 {
		return false
	}
	changed := false
	pendingSystemInstructions := make([]string, 0, 2)
	for idx, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["type"])))
		switch itemType {
		case "function_call":
			toolName := strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
			argsRaw := normalizePossiblyMixedToolArguments(fmt.Sprintf("%v", item["arguments"]))
			args := map[string]any{}
			if err := json.Unmarshal([]byte(argsRaw), &args); err != nil {
				args = parseToolArgsMapString(argsRaw)
			}
			sanitized := sanitizeBridgeToolArgumentsWithContext(args, toolName, func(instr string) {
				pendingSystemInstructions = append(pendingSystemInstructions, instr)
			})
			item["arguments"] = mustJSONString(sanitized)
			input[idx] = item
			changed = true
		case "shell_call":
			action, _ := item["action"].(map[string]any)
			sanitized := sanitizeBridgeToolArgumentsWithContext(action, "shell", func(instr string) {
				pendingSystemInstructions = append(pendingSystemInstructions, instr)
			})
			item["action"] = sanitized
			input[idx] = item
			changed = true
		}
	}
	if changed {
		req["input"] = input
	}
	for _, instr := range pendingSystemInstructions {
		prependSystemInstructionOnce(req, instr)
		changed = true
	}
	return changed
}

func shouldUseNativeResponsesBridgeStream(req map[string]any) bool {
	if req == nil {
		return false
	}
	if responsesRequestContractMode(req) == "plan" {
		return false
	}
	// Native stream-forward mode strips tools and forces tool_choice=none.
	// Never use it when the request carries tools, otherwise tool-capable
	// turns degrade into plain-text pseudo tool calls.
	if requestHasTools(req) {
		return false
	}
	// Keep tool-phase recovery for apply_patch-heavy turns. For normal descriptive
	// prompts, native upstream SSE forwarding provides true live tokens.
	if proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req).HasToolOutput {
		return false
	}
	if requestInputMentionsApplyPatch(req) {
		return false
	}
	userText := strings.ToLower(extractResponsesUserInputText(req))
	return !strings.Contains(userText, "apply_patch")
}

func summarizeResponsesBridgeControls(body []byte) string {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		stream := gjson.GetBytes(body, "stream").Bool()
		mode := strings.TrimSpace(gjson.GetBytes(body, "mode").String())
		reasoning := strings.TrimSpace(gjson.GetBytes(body, "reasoning.effort").String())
		return fmt.Sprintf("mode=%q reasoning=%q stream=%t", mode, reasoning, stream)
	}
	stream := false
	switch typed := req["stream"].(type) {
	case bool:
		stream = typed
	case string:
		stream = strings.EqualFold(strings.TrimSpace(typed), "true")
	}
	mode := extractResponsesRequestMode(req)
	reasoning := extractResponsesRequestReasoningEffort(req)
	return fmt.Sprintf("mode=%q reasoning=%q stream=%t", mode, reasoning, stream)
}

func summarizeBridgeSamplingControls(chatReq []byte) string {
	temp := strings.TrimSpace(gjson.GetBytes(chatReq, "temperature").Raw)
	topP := strings.TrimSpace(gjson.GetBytes(chatReq, "top_p").Raw)
	enableThinkingRaw := strings.TrimSpace(gjson.GetBytes(chatReq, "chat_template_kwargs.enable_thinking").Raw)
	reasoningBudgetRaw := strings.TrimSpace(gjson.GetBytes(chatReq, "reasoning_budget").Raw)
	reasoningBudgetMessage := strings.TrimSpace(gjson.GetBytes(chatReq, "reasoning_budget_message").String())
	logitBiasRaw := strings.TrimSpace(gjson.GetBytes(chatReq, "logit_bias").Raw)
	grammarRaw := strings.TrimSpace(gjson.GetBytes(chatReq, "grammar").String())
	if temp == "" {
		temp = "<default>"
	}
	if topP == "" {
		topP = "<default>"
	}
	if enableThinkingRaw == "" {
		enableThinkingRaw = "<default>"
	}
	if reasoningBudgetRaw == "" {
		reasoningBudgetRaw = "<default>"
	}
	if reasoningBudgetMessage == "" {
		reasoningBudgetMessage = "<default>"
	}
	if logitBiasRaw == "" {
		logitBiasRaw = "<default>"
	}
	if grammarRaw == "" {
		grammarRaw = "<default>"
	} else {
		grammarRaw = "<set>"
	}
	return fmt.Sprintf(
		"temperature=%s top_p=%s enable_thinking=%s reasoning_budget=%s reasoning_budget_message=%q logit_bias=%s grammar=%s",
		temp,
		topP,
		enableThinkingRaw,
		reasoningBudgetRaw,
		reasoningBudgetMessage,
		logitBiasRaw,
		grammarRaw,
	)
}

func removeQwenCloseThinkControls(out map[string]any) {
	if out == nil {
		return
	}
	if existing, ok := out["logit_bias"].(map[string]any); ok {
		updated := cloneMap(existing)
		delete(updated, qwenCloseThinkTokenID)
		if len(updated) == 0 {
			delete(out, "logit_bias")
		} else {
			out["logit_bias"] = updated
		}
	}
	delete(out, "grammar")
}

func forceQwenThinkingDisabled(out map[string]any) {
	if out == nil {
		return
	}
	removeQwenCloseThinkControls(out)
	delete(out, "reasoning_budget")
	delete(out, "reasoning_budget_message")
	if existing, ok := out["chat_template_kwargs"].(map[string]any); ok {
		updated := cloneMap(existing)
		updated["enable_thinking"] = false
		out["chat_template_kwargs"] = updated
		return
	}
	out["chat_template_kwargs"] = map[string]any{
		"enable_thinking": false,
	}
}

func translateResponsesToChatCompletionsRequest(body []byte) ([]byte, error) {
	appendProxyStageTrace("translation.responses_to_chat.start", map[string]any{
		"body_len": len(body),
	})
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	// Bridge mode must also normalize input ordering/roles so strict chat templates
	// (e.g. Qwen variants) never see non-leading system messages.
	_ = normalizeResponsesInputMap(req)
	mode := extractResponsesRequestMode(req)
	codexManagedPlanMode := isCodexManagedPlanModeFromResponsesRequest(req)
	explicitNoMutationPlan := requestExplicitlyWantsNoMutationPlan(req)
	planModeRequested := mode == "plan" || codexManagedPlanMode
	proxyPlanEnforcement := mode == "plan" && !codexManagedPlanMode
	reasoningEffort := extractResponsesRequestReasoningEffort(req)
	reasoningSummaryEnabled := reasoningSummaryRequestsThinking(req)
	stripUnsupportedResponsesInclude(req)
	stripSlashDirectivesFromResponsesInput(req)
	sanitizeResponsesInputToolArguments(req)
	workflowState := proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
	if requestExplicitlyForbidsWebSearch(req) && !workflowState.HasToolOutput {
		removeNamedTools(req, "web_search")
	}
	if !workflowState.HasToolOutput && requestExplicitlyWantsFileSearch(req) && !responsesRequestExposesTool(req, "file_search") {
		appendResponsesTool(req, map[string]any{"type": "file_search"})
	}
	if !workflowState.HasToolOutput && requestExplicitlyWantsPlaywrightBrowserTools(req) && responsesRequestExposesTool(req, "mcp__playwright__") {
		appendStrictPlaywrightBrowserToolInstruction(req)
	}
	if !workflowState.HasToolOutput && requestExplicitlyWantsFileSearch(req) {
		appendStrictFileSearchOnlyInstruction(req)
	}
	if requestShouldForceApplyPatchAfterSingleSubagentWait(req) {
		removeRedundantSubagentNotificationMessages(req)
		workflowState = proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
		forceFunctionToolChoice(req, "apply_patch")
		req["parallel_tool_calls"] = false
	}
	if requestShouldForceSendInputAfterResume(req) {
		removeRedundantSubagentNotificationMessages(req)
		workflowState = proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
		forceFunctionToolChoice(req, "send_input")
		req["parallel_tool_calls"] = false
	}
	if explicitNoMutationPlan && !workflowState.HasToolOutput {
		planModeRequested = true
		proxyPlanEnforcement = true
	}

	if profile, ok := reasoningEffortProfileForEffort(reasoningEffort); ok && strings.TrimSpace(profile.instruction) != "" {
		prependSystemInstructionOnce(req, profile.instruction)
	}
	appendSerializedAgentOrchestrationInstruction(req)
	if requestNeedsCodexSkillFilesystemProof(req) {
		prependSystemInstructionOnce(req,
			"For Codex skill and SKILL.md tasks, do not claim a skill was found, created, installed, updated, modified, or exists unless a tool result proves it. "+
				"Acceptable proof is either a shell result that lists or reads the concrete SKILL.md path, or an apply_patch call that targets that SKILL.md path. "+
				"If proof is missing, use shell to inspect the relevant skills directory before making the claim.")
	}
	if requestLooksLikeTextOnlyReply(req) {
		req["parallel_tool_calls"] = false
		req["tool_choice"] = "none"
		delete(req, "tools")
	}
	if requestShouldFinalizeExactlyOneSubagentSummary(req, workflowState) {
		prependSystemInstructionOnce(req,
			"The delegated subagent work and file write are already complete. Do not call any more tools. Return a concise final summary only.")
		req["parallel_tool_calls"] = false
		req["tool_choice"] = "none"
		delete(req, "tools")
	}
	if requestMentionsAgentOrchestration(req) &&
		!requestInputMentionsApplyPatch(req) &&
		!requestNeedsCodexSkillFilesystemProof(req) {
		keepOnlyNamedTools(req, "spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent")
	}
	if requestShouldForceApplyPatchAfterSingleSubagentWait(req) {
		keepOnlyNamedTools(req, "apply_patch")
		forceFunctionToolChoice(req, "apply_patch")
		req["parallel_tool_calls"] = false
	}
	if requestShouldForceSendInputAfterResume(req) {
		keepOnlyNamedTools(req, "send_input")
		forceFunctionToolChoice(req, "send_input")
		req["parallel_tool_calls"] = false
	}
	if !planModeRequested && !workflowState.HasToolOutput {
		lowerUserText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
		if rawResponsesBodyIncludesWebSearchTool(body) &&
			(requestExplicitlyWantsSearchIntent(req, workflowState) ||
				textIndicatesSearchIntent(lowerUserText) ||
				textLooksLikeDeferredSearchPromise(lowerUserText) ||
				rawResponsesBodyMentionsSearchIntent(body) ||
				strings.Contains(lowerUserText, "research the web") ||
				strings.Contains(lowerUserText, "web research") ||
				strings.Contains(lowerUserText, "search the web")) &&
			!requestHistoryContainsResearchArtifacts(req, workflowState) {
			appendStrictWebSearchOnlyInstruction(req)
		} else if body, err := json.Marshal(req); err == nil &&
			shouldEnableStrictApplyPatchIntent(req, body) &&
			!requestWantsShellInspectionBeforeApplyPatch(req) {
			keepOnlyNamedTools(req, "apply_patch")
		}
	} else if !planModeRequested &&
		requestShouldStayWebSearchOnlyForResearchPlan(req, workflowState) &&
		requestHistoryContainsSearchArtifacts(req, workflowState) &&
		!responsesRequestHasForcedWebSearchSynthesis(body) {
		appendStrictWebSearchOnlyInstruction(req)
	}

	out := map[string]any{}
	normalizedTools := []any(nil)
	if toolsRaw, ok := req["tools"].([]any); ok {
		prependNamespaceToolInstructions(req, toolsRaw)
		prependPlaywrightMCPToolInstructions(req, toolsRaw)
		normalizedTools = normalizeBridgeChatTools(toolsRaw)
	}
	requestUserInputUnavailable := requestUserInputUnavailableInDefaultMode(req)
	if requestUserInputUnavailable {
		normalizedTools = removeNamedToolFromList(normalizedTools, "request_user_input")
	}
	if planModeRequested {
		normalizedTools = removeMutatingPlanModeTools(normalizedTools, proxyPlanEnforcement)
		normalizedTools = removePlanInteractionTools(normalizedTools)
	}
	if !planModeRequested && requestExplicitlyWantsReturnedPlan(req) {
		prependSystemInstructionOnce(req,
			"When the user asks for a plan in Default mode, return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>. "+
				"Do not call update_plan to present the user-facing plan.")
		prependSystemInstructionOnce(req,
			"In plan-return turns, keep the output focused on the current user task only. Do not ask follow-up questions, preference questions, or how-to-proceed questions in the plan response.")
		normalizedTools = removeNamedToolFromList(normalizedTools, "update_plan")
	}
	// Ensure apply_patch remains callable when request intent clearly asks for it,
	// even if upstream client omitted that tool from the translated set.
	// Exception: explicit shell-first mixed workflows intentionally narrow the
	// first turn to shell-only; do not re-add apply_patch during translation.
	reqBytesForIntent, _ := json.Marshal(req)
	lowerUserText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	researchMustHappenFirst := rawResponsesBodyIncludesWebSearchTool(body) &&
		!workflowState.HasToolOutput &&
		(requestExplicitlyWantsSearchIntent(req, workflowState) ||
			textIndicatesSearchIntent(lowerUserText) ||
			textLooksLikeDeferredSearchPromise(lowerUserText) ||
			strings.Contains(lowerUserText, "research the web") ||
			strings.Contains(lowerUserText, "web research") ||
			strings.Contains(lowerUserText, "search the web")) &&
		!requestHistoryContainsResearchArtifacts(req, workflowState)
	if !planModeRequested &&
		!researchMustHappenFirst &&
		shouldEnableStrictApplyPatchIntent(req, reqBytesForIntent) &&
		!requestWantsShellInspectionBeforeApplyPatch(req) &&
		!bridgeToolListHasFunction(normalizedTools, "apply_patch") {
		normalizedTools = append(normalizedTools, bridgeToolToChatTool("apply_patch", applyPatchPreferredToolDescription, map[string]any{
			"type": "object",
			"properties": map[string]any{
				"operation": buildApplyPatchOperationSchema(),
			},
			"required": []string{"operation"},
		}))
	}
	hasToolRequests := len(normalizedTools) > 0
	copyField := func(key string) {
		if v, ok := req[key]; ok {
			out[key] = v
		}
	}
	for _, key := range []string{
		// Removed: use loaded model, not request model
		"temperature",
		"top_p",
		"chat_template_kwargs",
		"reasoning_budget",
		"reasoning_budget_message",
		"logit_bias",
		"grammar",
		"presence_penalty",
		"frequency_penalty",
		"stop",
		"n",
		"metadata",
	} {
		copyField(key)
	}
	if profile, ok := reasoningEffortProfileForEffort(reasoningEffort); ok {
		if profile.hasEnableThinking || reasoningSummaryEnabled {
			if existing, hasExisting := out["chat_template_kwargs"].(map[string]any); hasExisting {
				updated := cloneMap(existing)
				updated["enable_thinking"] = profile.enableThinking || reasoningSummaryEnabled
				out["chat_template_kwargs"] = updated
			} else {
				out["chat_template_kwargs"] = map[string]any{
					"enable_thinking": profile.enableThinking || reasoningSummaryEnabled,
				}
			}
		}
		translatedToolChoice := strings.TrimSpace(fmt.Sprintf("%v", out["tool_choice"]))
		if translatedToolChoice == "" {
			translatedToolChoice = strings.TrimSpace(fmt.Sprintf("%v", req["tool_choice"]))
		}
		suppressCloseThinkGuard := hasToolRequests || strings.EqualFold(translatedToolChoice, "none")
		if profile.hasCloseThinkBias && !suppressCloseThinkGuard {
			if existing, hasExisting := out["logit_bias"].(map[string]any); hasExisting {
				if _, hasCloseTokenBias := existing[qwenCloseThinkTokenID]; !hasCloseTokenBias {
					updated := cloneMap(existing)
					updated[qwenCloseThinkTokenID] = profile.closeThinkBias
					out["logit_bias"] = updated
				}
			} else if _, hasExisting := out["logit_bias"]; !hasExisting {
				out["logit_bias"] = map[string]any{
					qwenCloseThinkTokenID: profile.closeThinkBias,
				}
			}
		}
		if profile.hasCloseThinkGuardRule && !suppressCloseThinkGuard {
			if _, hasGrammar := out["grammar"]; !hasGrammar && !hasToolRequests {
				out["grammar"] = profile.closeThinkGuardRule
			}
		}
	}
	if reasoningSummaryEnabled {
		if _, ok := out["chat_template_kwargs"]; !ok {
			out["chat_template_kwargs"] = map[string]any{
				"enable_thinking": true,
			}
		}
	}
	if v, ok := req["parallel_tool_calls"]; ok {
		out["parallel_tool_calls"] = v
	}

	hasCompletedRequestUserInputOutput := completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input")
	searchIntent := requestExplicitlyWantsSearchIntent(req, workflowState)
	explorationFollowupIntent := requestExplicitlyWantsExplorationFollowup(req, workflowState)
	implementationRetryIntent := requestExplicitlyWantsImplementationRetry(req, workflowState)
	nativeQuestionRequested := requestExplicitlyWantsNativeCodexQuestion(req)
	planOutputRequested := requestStillWantsStructuredPlan(req, workflowState)
	if codexManagedPlanMode && planOutputRequested {
		prependSystemInstructionOnce(req, "Codex Plan Mode is still active. Do not begin implementation, do not claim file changes, and do not switch to execution prose.")
		prependSystemInstructionOnce(req, "If the gathered research and user answers are sufficient, return exactly one <proposed_plan>...</proposed_plan> block now.")
	}
	continuationPhase := classifyContinuationTurnPhase(ContinuationContext{
		PlanModeRequested:         planModeRequested,
		PlanOutputRequested:       planOutputRequested,
		RequestUserInputAvailable: bridgeToolListHasFunction(normalizedTools, "request_user_input"),
		NativeQuestionRequested:   nativeQuestionRequested,
		ImplementationRetryIntent: implementationRetryIntent,
		SearchIntent:              searchIntent,
		ExplorationFollowupIntent: explorationFollowupIntent,
	})
	continuationDecision := proxyCompatibilityAdapters.Continuation.BuildDecision(req, ContinuationContext{
		PlanModeRequested:         planModeRequested,
		PlanOutputRequested:       planOutputRequested,
		ProxyPlanEnforcement:      proxyPlanEnforcement,
		RequestUserInputAvailable: bridgeToolListHasFunction(normalizedTools, "request_user_input"),
		NativeQuestionRequested:   nativeQuestionRequested,
		ImplementationRetryIntent: implementationRetryIntent,
		SearchIntent:              searchIntent,
		ExplorationFollowupIntent: explorationFollowupIntent,
		RequestedToolChoice:       normalizeBridgeToolChoice(req["tool_choice"]),
		ActiveToolChoice:          out["tool_choice"],
		TurnPhase:                 continuationPhase,
	})
	appendProxyStageTrace("continuation.responses_to_chat", map[string]any{
		"summary": summarizeContinuationState(
			continuationDecision.State,
			workflowState,
			continuationDecision.State == ContinuationStateFinalAnswerRequired,
			hasCompletedRequestUserInputOutput,
			workflowState.HasToolOutput,
		),
		"turn_phase": continuationPhase,
	})
	for _, instruction := range continuationDecision.Instructions {
		prependSystemInstructionOnce(req, instruction)
	}
	if continuationDecision.DisableTools {
		out["tool_choice"] = "none"
	} else {
		if requestedToolChoice := normalizeBridgeToolChoice(req["tool_choice"]); requestedToolChoice != nil {
			out["tool_choice"] = requestedToolChoice
		}
		if continuationDecision.ForceToolChoice != nil {
			out["tool_choice"] = continuationDecision.ForceToolChoice
		}
		if !toolChoiceTargetsSpecificTool(out["tool_choice"]) && len(continuationDecision.AllowedToolNames) > 0 {
			normalizedTools = filterBridgeToolsToAllowedNames(normalizedTools, continuationDecision.AllowedToolNames...)
		}
	}
	if forcedToolName := specificToolChoiceFunctionName(out["tool_choice"]); forcedToolName != "" {
		if requestUserInputUnavailable && strings.EqualFold(forcedToolName, "request_user_input") {
			out["tool_choice"] = "auto"
		} else {
			normalizedTools = filterBridgeToolsToFunctionName(normalizedTools, forcedToolName)
		}
	}
	if workflowState.VerificationExpected &&
		!workflowState.VerificationCompleted &&
		workflowState.LatestCompletedToolName == "apply_patch" &&
		specificToolChoiceFunctionName(out["tool_choice"]) == "shell" {
		removeQwenCloseThinkControls(out)
	}
	if requestExplicitlyWantsNativeCodexQuestion(req) &&
		specificToolChoiceFunctionName(out["tool_choice"]) == "request_user_input" {
		forceQwenThinkingDisabled(out)
	}
	if strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", out["tool_choice"])), "none") {
		normalizedTools = nil
		out["parallel_tool_calls"] = false
		removeQwenCloseThinkControls(out)
	}
	if len(normalizedTools) > 0 && continuationDecision.State != ContinuationStateFinalAnswerRequired {
		out["tools"] = normalizedTools
	} else {
		delete(out, "tools")
	}
	conflictResult := stripGrammarToolsConflictMap(out)
	if hasToolRequests && !conflictResult.removedGrammar {
		delete(out, "grammar")
	}

	if v, ok := req["max_output_tokens"]; ok {
		out["max_tokens"] = v
	} else if v, ok := req["max_tokens"]; ok {
		out["max_tokens"] = v
	}

	messages := responsesRequestToChatMessages(req)
	hasUserMessage := false
	for _, message := range messages {
		if strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", message["role"])), "user") {
			hasUserMessage = true
			break
		}
	}
	if !hasUserMessage {
		userText := extractResponsesInputText(req["input"])
		userText = strings.TrimSpace(cleanFallbackInput(req["input"], userText))
		if userText != "" {
			messages = append(messages, map[string]any{"role": "user", "content": userText})
		}
	}
	if len(messages) == 0 {
		messages = []map[string]any{{"role": "user", "content": " "}}
	}
	messages = filterEmptyAssistantChatMessages(messages)
	messages = normalizeChatMessagesForStrictTemplates(messages)
	for _, message := range messages {
		content, _ := message["content"].(string)
		if strings.TrimSpace(content) == "" {
			message["content"] = " "
		}
	}
	out["messages"] = messages
	streamRequested := false
	switch typed := req["stream"].(type) {
	case bool:
		streamRequested = typed
	case string:
		streamRequested = strings.EqualFold(strings.TrimSpace(typed), "true")
	}
	out["stream"] = streamRequested
	appendProxyStageTrace("translation.responses_to_chat.done", map[string]any{
		"mode":                   mode,
		"plan_mode_requested":    planModeRequested,
		"proxy_plan_enforcement": proxyPlanEnforcement,
		"tools_count":            len(normalizedTools),
		"stream":                 streamRequested,
		"force_final":            continuationDecision.State == ContinuationStateFinalAnswerRequired,
	})
	return json.Marshal(out)
}

func normalizeBridgeToolChoice(raw any) any {
	switch v := raw.(type) {
	case string:
		trimmed := strings.TrimSpace(strings.ToLower(v))
		if trimmed == "" {
			return nil
		}
		if trimmed == "auto" || trimmed == "none" || trimmed == "required" {
			return trimmed
		}
		return v
	case map[string]any:
		choiceType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", v["type"])))
		switch choiceType {
		case "", "auto", "none", "required":
			return choiceType
		case "function":
			return v
		case "shell", "apply_patch", "web_search", "web_search_preview", "file_search", "code_interpreter", "image_generation", "computer":
			name := choiceType
			if name == "web_search_preview" {
				name = "web_search"
			}
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": name,
				},
			}
		}
		if fn, ok := v["function"].(map[string]any); ok {
			if name := strings.TrimSpace(fmt.Sprintf("%v", fn["name"])); name != "" {
				return map[string]any{
					"type":     "function",
					"function": map[string]any{"name": name},
				}
			}
		}
		return v
	default:
		return raw
	}
}

func specificToolChoiceFunctionName(raw any) string {
	choice, ok := raw.(map[string]any)
	if !ok {
		return ""
	}
	if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", choice["type"])), "function") {
		return ""
	}
	fn, ok := choice["function"].(map[string]any)
	if !ok {
		return ""
	}
	return strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
}

func filterEmptyAssistantChatMessages(messages []map[string]any) []map[string]any {
	if len(messages) == 0 {
		return messages
	}
	filtered := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", message["role"])))
		content, _ := message["content"].(string)
		_, hasFunctionCall := message["function_call"]
		rawToolCalls, hasToolCalls := message["tool_calls"].([]any)
		if role == "assistant" && strings.TrimSpace(content) == "" &&
			(!hasToolCalls || len(rawToolCalls) == 0) &&
			!hasFunctionCall &&
			strings.TrimSpace(fmt.Sprintf("%v", message["reasoning_content"])) == "" {
			continue
		}
		filtered = append(filtered, message)
	}
	return filtered
}

func filterBridgeToolsToFunctionName(tools []any, name string) []any {
	name = normalizeToolTierName(name)
	if name == "" || len(tools) == 0 {
		return tools
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if toolDefinitionTargetsNormalizedName(tool, name) {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func normalizeBridgeChatTools(toolsRaw []any) []any {
	out := make([]any, 0, len(toolsRaw))
	for _, rawTool := range toolsRaw {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			continue
		}

		if fn, ok := tool["function"].(map[string]any); ok {
			if name, _ := fn["name"].(string); strings.TrimSpace(name) != "" {
				if strings.EqualFold(strings.TrimSpace(name), "multi_tool_use.parallel") {
					continue
				}
				out = append(out, map[string]any{
					"type":     "function",
					"function": fn,
				})
			}
			continue
		}

		toolType, _ := tool["type"].(string)
		toolType = strings.TrimSpace(toolType)
		toolName, _ := tool["name"].(string)
		toolName = strings.TrimSpace(toolName)
		if strings.EqualFold(toolType, "namespace") {
			namespaceTools, _ := tool["tools"].([]any)
			for _, childRaw := range namespaceTools {
				child, ok := childRaw.(map[string]any)
				if !ok {
					continue
				}
				childType := strings.TrimSpace(fmt.Sprintf("%v", child["type"]))
				if childType != "" && !strings.EqualFold(childType, "function") {
					continue
				}
				childName := strings.TrimSpace(fmt.Sprintf("%v", child["name"]))
				if childName == "" {
					if fn, ok := child["function"].(map[string]any); ok {
						childName = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
					}
				}
				if childName == "" {
					continue
				}
				fullName := joinToolNamespaceName(toolName, childName)
				fn := map[string]any{"name": fullName}
				if description, ok := child["description"].(string); ok && strings.TrimSpace(description) != "" {
					fn["description"] = description
				}
				if parameters, ok := child["parameters"]; ok {
					fn["parameters"] = parameters
				} else if childFn, ok := child["function"].(map[string]any); ok {
					if parameters, ok := childFn["parameters"]; ok {
						fn["parameters"] = parameters
					}
				}
				if strict, ok := child["strict"]; ok {
					fn["strict"] = strict
				} else if childFn, ok := child["function"].(map[string]any); ok {
					if strict, ok := childFn["strict"]; ok {
						fn["strict"] = strict
					}
				}
				out = append(out, map[string]any{"type": "function", "function": fn})
			}
			continue
		}
		// Codex custom tools often arrive as type="custom" with only a name.
		// Normalize known tool names into explicit schema branches.
		if strings.EqualFold(toolType, "custom") {
			switch strings.ToLower(toolName) {
			case "apply_patch", "applypatch":
				toolType = "apply_patch"
			case "shell", "shell_command":
				toolType = "shell"
			case "web_search", "web_search_preview":
				toolType = "web_search"
			case "file_search":
				toolType = "file_search"
			case "code_interpreter":
				toolType = "code_interpreter"
			case "image_generation":
				toolType = "image_generation"
			case "computer":
				toolType = "computer"
			}
		}
		switch toolType {
		case "", "function":
			name, _ := tool["name"].(string)
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			if strings.EqualFold(name, "multi_tool_use.parallel") {
				continue
			}
			fn := map[string]any{"name": name}
			if description, ok := tool["description"].(string); ok && strings.TrimSpace(description) != "" {
				fn["description"] = description
			}
			if parameters, ok := tool["parameters"]; ok {
				fn["parameters"] = parameters
			}
			if strict, ok := tool["strict"]; ok {
				fn["strict"] = strict
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
		case "shell":
			out = append(out, bridgeToolToChatTool("shell", "Run a shell command.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string"},
					"commands": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Commands to execute, one per array element.",
						"minItems":    1,
					},
				},
				"required": []string{"commands"},
			}))
		case "apply_patch":
			out = append(out, bridgeToolToChatTool("apply_patch", applyPatchPreferredToolDescription, map[string]any{
				"type": "object",
				"properties": map[string]any{
					"operation": buildApplyPatchOperationSchema(),
				},
				"required": []string{"operation"},
			}))
		case "web_search", "web_search_preview":
			out = append(out, bridgeToolToChatTool("web_search", "Search the web.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
				},
				"required": []string{"query"},
			}))
		case "file_search":
			out = append(out, bridgeToolToChatTool("file_search", "Search indexed files.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
				},
				"required": []string{"query"},
			}))
		case "code_interpreter":
			out = append(out, bridgeToolToChatTool("code_interpreter", "Run code in an interpreter.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"code": map[string]any{"type": "string"},
				},
				"required": []string{"code"},
			}))
		case "image_generation":
			out = append(out, bridgeToolToChatTool("image_generation", "Generate an image.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt": map[string]any{"type": "string"},
				},
				"required": []string{"prompt"},
			}))
		case "computer":
			out = append(out, bridgeToolToChatTool("computer", "Perform a computer action.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"action": map[string]any{"type": "string"},
				},
				"required": []string{"action"},
			}))
		case "mcp":
			out = append(out, cloneMap(tool))
		default:
			// Unknown tool types (e.g. Codex "custom") -> convert to function format
			name, _ := tool["name"].(string)
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			fn := map[string]any{"name": name}
			if description, ok := tool["description"].(string); ok && strings.TrimSpace(description) != "" {
				fn["description"] = description
			}
			if parameters, ok := tool["parameters"]; ok {
				fn["parameters"] = parameters
			}
			if strict, ok := tool["strict"]; ok {
				fn["strict"] = strict
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
		}
	}
	return out
}

func joinToolNamespaceName(namespace string, child string) string {
	namespace = strings.TrimSpace(namespace)
	child = strings.TrimSpace(child)
	if namespace == "" {
		return child
	}
	if child == "" {
		return namespace
	}
	if strings.HasSuffix(namespace, "__") {
		return namespace + child
	}
	return namespace + "__" + child
}

func buildMCPToolName(server string, tool string) string {
	server = strings.TrimSpace(server)
	tool = strings.TrimSpace(tool)
	if server == "" {
		return tool
	}
	return joinToolNamespaceName("mcp__"+server, tool)
}

func parseMCPToolName(name string) (server string, tool string, ok bool) {
	name = strings.TrimSpace(name)
	if !strings.HasPrefix(name, "mcp__") {
		return "", "", false
	}
	rest := strings.TrimPrefix(name, "mcp__")
	parts := strings.SplitN(rest, "__", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	server = strings.TrimSpace(parts[0])
	tool = strings.TrimSpace(parts[1])
	if server == "" || tool == "" {
		return "", "", false
	}
	return server, tool, true
}

func collectNamespaceCallableNames(toolsRaw []any) map[string][]string {
	out := map[string][]string{}
	for _, rawTool := range toolsRaw {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			continue
		}
		toolType := strings.TrimSpace(fmt.Sprintf("%v", tool["type"]))
		if !strings.EqualFold(toolType, "namespace") {
			continue
		}
		namespace := strings.TrimSpace(fmt.Sprintf("%v", tool["name"]))
		if namespace == "" {
			continue
		}
		children, _ := tool["tools"].([]any)
		names := make([]string, 0, len(children))
		for _, childRaw := range children {
			child, ok := childRaw.(map[string]any)
			if !ok {
				continue
			}
			childName := strings.TrimSpace(fmt.Sprintf("%v", child["name"]))
			if childName == "" {
				if fn, ok := child["function"].(map[string]any); ok {
					childName = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
				}
			}
			if childName == "" {
				continue
			}
			names = append(names, joinToolNamespaceName(namespace, childName))
		}
		if len(names) > 0 {
			out[namespace] = names
		}
	}
	return out
}

func prependNamespaceToolInstructions(req map[string]any, toolsRaw []any) {
	if req == nil || len(toolsRaw) == 0 {
		return
	}
	namespaces := collectNamespaceCallableNames(toolsRaw)
	if len(namespaces) == 0 {
		return
	}
	keys := make([]string, 0, len(namespaces))
	for namespace := range namespaces {
		keys = append(keys, namespace)
	}
	sort.Strings(keys)
	for _, namespace := range keys {
		callable := namespaces[namespace]
		if len(callable) == 0 {
			continue
		}
		preview := callable
		if len(preview) > 12 {
			preview = preview[:12]
		}
		instruction := fmt.Sprintf(
			"Namespace tools are not callable by their namespace root. Never call %q directly. Use one of these exact callable tool names instead: %s.",
			namespace,
			strings.Join(preview, ", "),
		)
		prependSystemInstructionOnce(req, instruction)
	}
}

func requestMentionsPlaywrightBrowserTask(req map[string]any) bool {
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		reqBytes, _ := json.Marshal(req)
		text = strings.ToLower(string(reqBytes))
	}
	return strings.Contains(text, "playwright") ||
		strings.Contains(text, "browser tool") ||
		strings.Contains(text, "browser_navigate") ||
		strings.Contains(text, "browser_snapshot") ||
		strings.Contains(text, "navigate to http") ||
		strings.Contains(text, "navigate to https") ||
		strings.Contains(text, "take a snapshot")
}

func prependPlaywrightMCPToolInstructions(req map[string]any, toolsRaw []any) {
	if !requestMentionsPlaywrightBrowserTask(req) {
		return
	}
	namespaces := collectNamespaceCallableNames(toolsRaw)
	callable := namespaces["mcp__playwright__"]
	if len(callable) == 0 {
		return
	}
	preview := make([]string, 0, len(callable))
	for _, name := range callable {
		if strings.HasPrefix(name, "mcp__playwright__browser_") {
			preview = append(preview, name)
		}
	}
	if len(preview) == 0 {
		preview = callable
	}
	if len(preview) > 8 {
		preview = preview[:8]
	}
	prependSystemInstructionOnce(req, fmt.Sprintf(
		"For Playwright or browser tasks, emit the exact MCP browser tool calls instead of stopping after reasoning. Use callable tool names like: %s.",
		strings.Join(preview, ", "),
	))
	prependSystemInstructionOnce(req, "For browser tasks, do not finish the turn until the requested browser tool calls have been emitted and any requested final text reply has been produced.")
}

func bridgeToolListHasFunction(tools []any, name string) bool {
	want := strings.ToLower(strings.TrimSpace(name))
	if want == "" {
		return false
	}
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if fn, ok := tool["function"].(map[string]any); ok {
			got := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
			if got == want {
				return true
			}
		}
	}
	return false
}

func parsedCallListHasName(calls []ParsedToolCall, name string) bool {
	want := strings.ToLower(strings.TrimSpace(name))
	if want == "" {
		return false
	}
	for _, c := range calls {
		if strings.ToLower(strings.TrimSpace(c.Name)) == want {
			return true
		}
	}
	return false
}

func bridgeToolToChatTool(name string, description string, parameters map[string]any) map[string]any {
	fn := map[string]any{"name": name}
	if strings.TrimSpace(description) != "" {
		fn["description"] = description
	}
	if len(parameters) > 0 {
		fn["parameters"] = parameters
	}
	return map[string]any{
		"type":     "function",
		"function": fn,
	}
}

func extractResponsesInputText(input any) string {
	parts := make([]string, 0)
	collectResponseText(input, &parts)
	return strings.Join(parts, "\n")
}

var thinkTagPrefixRegex = regexp.MustCompile(`(?is)^\s*(?:<think>\s*(.*?)\s*</think>|<thinking>\s*(.*?)\s*</thinking>)\s*`)
var leadingReasoningDirectiveRegex = regexp.MustCompile(`(?is)^\s*/(?:no_)?think\s*(?:\r?\n+|\z)`)
var leadingOrphanThinkCloseRegex = regexp.MustCompile(`(?is)^\s*(?:(?:</think>|</thinking>)\s*)+`)
var trailingOrphanThinkCloseRegex = regexp.MustCompile(`(?is)(?:\s*(?:</think>|</thinking>))+\s*$`)
var reasoningFunctionStyleRecoveryPrefixRegex = regexp.MustCompile(`(?is)^\s*(?:</think>\s*|</thinking>\s*)*(?:apply_patch|shell|shell_command|web_search|file_search|code_interpreter|image_generation|computer|update_plan|spawn_agent|send_input|resume_agent|wait_agent|close_agent)\s*\(`)

func splitThinkTaggedText(raw string) (leading string, reasoning string, trailing string, found bool) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", "", "", false
	}
	lower := strings.ToLower(trimmed)
	firstOpen := -1
	openTag := ""
	closeTag := ""
	switch {
	case strings.Contains(lower, "<think>"):
		firstOpen = strings.Index(lower, "<think>")
		openTag = "<think>"
		closeTag = "</think>"
	case strings.Contains(lower, "<thinking>"):
		firstOpen = strings.Index(lower, "<thinking>")
		openTag = "<thinking>"
		closeTag = "</thinking>"
	default:
		return trimmed, "", "", false
	}
	if firstOpen < 0 {
		return trimmed, "", "", false
	}

	leading = strings.TrimSpace(trimmed[:firstOpen])
	pos := firstOpen
	reasoningParts := make([]string, 0, 2)
	trailingParts := make([]string, 0, 2)
	for pos < len(trimmed) {
		remainingLower := strings.ToLower(trimmed[pos:])
		nextThink := strings.Index(remainingLower, "<think>")
		nextThinking := strings.Index(remainingLower, "<thinking>")
		var nextOpen int
		if nextThink >= 0 && (nextThinking < 0 || nextThink < nextThinking) {
			nextOpen = nextThink
			openTag = "<think>"
			closeTag = "</think>"
		} else if nextThinking >= 0 {
			nextOpen = nextThinking
			openTag = "<thinking>"
			closeTag = "</thinking>"
		} else {
			break
		}
		if nextOpen > 0 {
			outside := strings.TrimSpace(trimmed[pos : pos+nextOpen])
			if outside != "" {
				trailingParts = append(trailingParts, outside)
			}
		}
		tagStart := pos + nextOpen
		innerStart := tagStart + len(openTag)
		innerLower := strings.ToLower(trimmed[innerStart:])
		closeIndex := strings.Index(innerLower, closeTag)
		if closeIndex < 0 {
			// Keep unmatched text visible; only balanced tags are treated structurally.
			rest := strings.TrimSpace(trimmed[tagStart:])
			if rest != "" {
				trailingParts = append(trailingParts, rest)
			}
			break
		}
		innerEnd := innerStart + closeIndex
		innerText := strings.TrimSpace(trimmed[innerStart:innerEnd])
		if innerText != "" {
			reasoningParts = append(reasoningParts, innerText)
		}
		pos = innerEnd + len(closeTag)
		found = true
	}
	if pos < len(trimmed) {
		rest := strings.TrimSpace(trimmed[pos:])
		if rest != "" {
			trailingParts = append(trailingParts, rest)
		}
	}
	return strings.TrimSpace(leading), strings.TrimSpace(strings.Join(reasoningParts, "\n\n")), strings.TrimSpace(strings.Join(trailingParts, "\n\n")), found
}

func stripLeadingReasoningDirective(text string) string {
	if text == "" {
		return ""
	}
	stripped := leadingReasoningDirectiveRegex.ReplaceAllString(text, "")
	if match := thinkTagPrefixRegex.FindStringSubmatchIndex(stripped); match != nil && len(match) >= 2 {
		stripped = strings.TrimSpace(stripped[match[1]:])
	}
	stripped = leadingOrphanThinkCloseRegex.ReplaceAllString(stripped, "")
	stripped = strings.TrimSpace(stripped)
	if stripped == "" {
		if leadingOrphanThinkCloseRegex.MatchString(strings.TrimSpace(text)) {
			return ""
		}
		return strings.TrimSpace(text)
	}
	return stripped
}

func normalizeAssistantReasoningFields(content string, reasoning string) (string, string) {
	content = strings.TrimSpace(content)
	reasoning = strings.TrimSpace(reasoning)

	if extractedContent, extractedReasoning := extractContentAndReasoning(content); strings.TrimSpace(extractedReasoning) != "" {
		content = strings.TrimSpace(extractedContent)
		extractedReasoning = strings.TrimSpace(extractedReasoning)
		if reasoning == "" {
			reasoning = extractedReasoning
		} else if !strings.EqualFold(reasoning, extractedReasoning) {
			reasoning = strings.TrimSpace(extractedReasoning + "\n\n" + reasoning)
		}
	}

	if extractedContent, extractedReasoning := extractContentAndReasoning(reasoning); strings.TrimSpace(extractedReasoning) != "" {
		reasoning = strings.TrimSpace(extractedReasoning)
		extractedContent = strings.TrimSpace(extractedContent)
		if extractedContent != "" {
			if content == "" {
				content = extractedContent
			} else if !strings.EqualFold(content, extractedContent) {
				content = strings.TrimSpace(extractedContent + "\n" + content)
			}
		}
		return content, reasoning
	}

	if reasoning != "" {
		content = strings.TrimSpace(trailingOrphanThinkCloseRegex.ReplaceAllString(content, ""))
	}

	if reasoning == "" {
		return content, reasoning
	}

	if leadingOrphanThinkCloseRegex.MatchString(reasoning) {
		leakedContent := stripLeadingReasoningDirective(reasoning)
		if leakedContent != "" {
			if content == "" {
				content = leakedContent
			} else if !strings.EqualFold(content, leakedContent) {
				content = strings.TrimSpace(leakedContent + "\n" + content)
			}
		}
		return content, ""
	}

	return content, reasoning
}

func chatCompletionLooksLikeEmptyVisibleOutput(content string, reasoning string) bool {
	if strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) == "" {
		return false
	}
	normalizedContent, normalizedReasoning := normalizeAssistantReasoningFields(content, reasoning)
	normalizedContent = strings.TrimSpace(stripLeadingReasoningDirective(normalizedContent))
	normalizedReasoning = strings.TrimSpace(stripLeadingReasoningDirective(normalizedReasoning))
	return normalizedContent == "" && normalizedReasoning == ""
}

func normalizeChatCompletionReasoningBoundary(body []byte) []byte {
	var envelope map[string]any
	if err := json.Unmarshal(body, &envelope); err != nil {
		return body
	}
	choices, ok := envelope["choices"].([]any)
	if !ok || len(choices) == 0 {
		return body
	}

	changed := false
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		message, ok := choice["message"].(map[string]any)
		if !ok {
			continue
		}
		content := strings.TrimSpace(fmt.Sprintf("%v", message["content"]))
		reasoning := strings.TrimSpace(fmt.Sprintf("%v", message["reasoning_content"]))
		normalizedContent, normalizedReasoning := normalizeAssistantReasoningFields(content, reasoning)
		if normalizedContent != content {
			message["content"] = normalizedContent
			changed = true
		}
		if normalizedReasoning != reasoning {
			if normalizedReasoning == "" {
				delete(message, "reasoning_content")
			} else {
				message["reasoning_content"] = normalizedReasoning
			}
			changed = true
		}
	}

	if !changed {
		return body
	}
	normalized, err := json.Marshal(envelope)
	if err != nil {
		return body
	}
	return normalized
}

func extractContentAndReasoning(raw string) (content string, reasoning string) {
	leading, extractedReasoning, trailing, found := splitThinkTaggedText(raw)
	if !found {
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			return "", ""
		}
		return trimmed, ""
	}
	reasoning = strings.TrimSpace(extractedReasoning)
	if strings.TrimSpace(trailing) != "" {
		return strings.TrimSpace(trailing), reasoning
	}
	if strings.TrimSpace(leading) != "" {
		return strings.TrimSpace(leading), reasoning
	}
	if reasoning == "" {
		return "", ""
	}
	return "", reasoning
}

func shouldAllowReasoningDerivedToolRecovery(raw string, parsedCalls []ParsedToolCall) bool {
	if len(parsedCalls) == 0 {
		return false
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return false
	}
	lower := strings.ToLower(trimmed)
	for _, marker := range []string{
		"<tool_call>",
		"</tool_call>",
		"<function=",
		"<tools>",
		"</tools>",
		"<apply_patch>",
		"</apply_patch>",
		"<shell_commands>",
		"</shell_commands>",
		"<shell_command>",
		"</shell_command>",
		"<shell>",
		"</shell>",
		"<tool_use>",
		"</tool_use>",
	} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return reasoningFunctionStyleRecoveryPrefixRegex.MatchString(trimmed)
}

func nativeToolArgumentsEmptyForParsedReasoningCall(message gjson.Result, parsedCalls []ParsedToolCall) bool {
	if len(parsedCalls) == 0 {
		return false
	}
	names := map[string]struct{}{}
	for _, call := range parsedCalls {
		name := strings.ToLower(strings.TrimSpace(call.Name))
		if name != "" {
			names[name] = struct{}{}
		}
	}
	if len(names) == 0 {
		return false
	}
	argsEmpty := func(raw string) bool {
		trimmed := strings.TrimSpace(raw)
		return trimmed == "" || trimmed == "{}" || strings.EqualFold(trimmed, "null")
	}
	for _, toolCall := range message.Get("tool_calls").Array() {
		name := strings.TrimSpace(toolCall.Get("function.name").String())
		if name == "" {
			name = strings.TrimSpace(toolCall.Get("name").String())
		}
		if _, ok := names[strings.ToLower(name)]; !ok {
			continue
		}
		args := strings.TrimSpace(toolCall.Get("function.arguments").String())
		if args == "" {
			args = strings.TrimSpace(toolCall.Get("arguments").String())
		}
		if argsEmpty(args) {
			return true
		}
	}
	functionCall := message.Get("function_call")
	if functionCall.Exists() {
		name := strings.TrimSpace(functionCall.Get("name").String())
		if _, ok := names[strings.ToLower(name)]; ok {
			if argsEmpty(functionCall.Get("arguments").String()) {
				return true
			}
		}
	}
	return false
}

func collectResponseText(v any, out *[]string) {
	switch typed := v.(type) {
	case string:
		s := strings.TrimSpace(typed)
		if s != "" {
			*out = append(*out, s)
		}
	case []any:
		for _, item := range typed {
			collectResponseText(item, out)
		}
	case map[string]any:
		for _, key := range []string{"input_text", "text", "content", "value"} {
			if child, ok := typed[key]; ok {
				collectResponseText(child, out)
			}
		}
	}
}

func responsesRequestToChatMessages(req map[string]any) []map[string]any {
	out := make([]map[string]any, 0)
	if instructions, ok := req["instructions"].(string); ok && strings.TrimSpace(instructions) != "" {
		out = append(out, map[string]any{
			"role":    "system",
			"content": strings.TrimSpace(instructions),
		})
	}
	pendingReasoningText := ""
	toolNameByCallID := map[string]string{}
	disableReasoningHistoryReplay := false
	switch value := strings.TrimSpace(strings.ToLower(os.Getenv("LLAMASWAP_DISABLE_REASONING_HISTORY_REPLAY"))); value {
	case "1", "true", "yes", "on":
		disableReasoningHistoryReplay = true
	}

	convertOne := func(role string, content any) {
		if content == nil {
			return
		}
		normalizedRole := normalizeChatCompletionRole(role)
		text := extractResponsesInputText(content)
		text = strings.TrimSpace(cleanFallbackInput(content, text))
		reasoningText := ""
		if normalizedRole == "assistant" {
			var extractedReasoning string
			text, extractedReasoning = extractContentAndReasoning(text)
			reasoningText = strings.TrimSpace(extractedReasoning)
		}
		if text == "" && reasoningText == "" {
			return
		}
		message := map[string]any{
			"role":    normalizedRole,
			"content": text,
		}
		if normalizedRole == "assistant" && reasoningText == "" && pendingReasoningText != "" {
			reasoningText = pendingReasoningText
			pendingReasoningText = ""
		}
		if normalizedRole == "assistant" && reasoningText != "" {
			message["reasoning_content"] = reasoningText
		}
		out = append(out, message)
	}

	appendAssistantToolCall := func(name string, arguments any, callID string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		if strings.EqualFold(name, "shell") {
			if argMap, ok := arguments.(map[string]any); ok {
				arguments = normalizeShellArgumentMap(argMap)
			}
		}

		stringArgs := encodeAnyAsJSONString(arguments)
		stringArgs = normalizePossiblyMixedToolArguments(stringArgs)

		callID = strings.TrimSpace(callID)
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(out))
		}
		if normalizedName := normalizeToolTierName(name); normalizedName != "" {
			toolNameByCallID[callID] = normalizedName
		}

		message := map[string]any{
			"role":    "assistant",
			"content": "",
			"tool_calls": []any{
				map[string]any{
					"id":   callID,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": stringArgs,
					},
				},
			},
		}
		if pendingReasoningText != "" {
			message["reasoning_content"] = pendingReasoningText
			pendingReasoningText = ""
		}
		out = append(out, message)
	}

	appendToolResult := func(callID string, toolName string, output any) {
		callID = strings.TrimSpace(callID)
		if callID == "" {
			return
		}
		if strings.TrimSpace(toolName) == "" {
			toolName = toolNameByCallID[callID]
		}
		text := extractResponsesInputText(output)
		text = strings.TrimSpace(text)
		if text == "" {
			text = encodeAnyAsJSONString(output)
		}
		text = strings.TrimSpace(cleanFallbackInput(output, text))
		if normalizeToolTierName(toolName) == "apply_patch" {
			text = normalizeApplyPatchToolOutputText(text)
		}
		if text == "" {
			text = " "
		}
		out = append(out, map[string]any{
			"role":         "tool",
			"tool_call_id": callID,
			"content":      text,
		})
	}

	if inputArr, ok := req["input"].([]any); ok {
		for _, rawItem := range inputArr {
			item, ok := rawItem.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := item["type"].(string)
			if strings.TrimSpace(itemType) == "" {
				if roleRaw, hasRole := item["role"]; hasRole {
					if _, hasContent := item["content"]; hasContent {
						convertOne(fmt.Sprintf("%v", roleRaw), item["content"])
						continue
					}
				}
			}
			switch itemType {
			case "message":
				convertOne(fmt.Sprintf("%v", item["role"]), item["content"])
			case "reasoning":
				if disableReasoningHistoryReplay {
					continue
				}
				summary, _ := item["summary"].([]any)
				reasoningParts := make([]string, 0, len(summary))
				for _, rawSummary := range summary {
					summaryPart, ok := rawSummary.(map[string]any)
					if !ok {
						continue
					}
					summaryText := strings.TrimSpace(extractResponsesInputText(summaryPart))
					if summaryText != "" {
						reasoningParts = append(reasoningParts, summaryText)
					}
				}
				reasoningText := strings.TrimSpace(strings.Join(reasoningParts, "\n"))
				if reasoningText == "" {
					continue
				}
				if pendingReasoningText == "" {
					pendingReasoningText = reasoningText
				} else {
					pendingReasoningText = strings.TrimSpace(pendingReasoningText + "\n" + reasoningText)
				}
			case "function_call":
				appendAssistantToolCall(fmt.Sprintf("%v", item["name"]), item["arguments"], fmt.Sprintf("%v", item["call_id"]))
			case "function_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "", item["output"])
			case "shell_call":
				appendAssistantToolCall("shell", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "shell_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "shell", item["output"])
			case "apply_patch_call":
				payload := map[string]any{}
				if operation, ok := item["operation"]; ok && hasNonEmptyApplyPatchOperation(operation) {
					switch op := operation.(type) {
					case map[string]any:
						if rawType := strings.TrimSpace(fmt.Sprintf("%v", op["type"])); rawType != "" {
							payload["operation"] = normalizeApplyPatchOperation(op)
						} else if patch, ok := op["patch"].(string); ok && strings.TrimSpace(patch) != "" {
							payload["input"] = patch
						} else if input, ok := op["input"].(string); ok && strings.TrimSpace(input) != "" {
							payload["input"] = input
						} else if diff, ok := op["diff"].(string); ok && strings.TrimSpace(diff) != "" {
							payload["operation"] = normalizeApplyPatchOperation(op)
						} else {
							payload["operation"] = normalizeApplyPatchOperation(op)
						}
					case string:
						if converted, ok := convertApplyPatchTextToOperation(op); ok {
							payload["operation"] = converted
						} else if strings.TrimSpace(op) != "" {
							payload["input"] = strings.TrimSpace(op)
						}
					default:
						encoded := strings.TrimSpace(encodeAnyAsJSONString(operation))
						if encoded != "" && encoded != "{}" && encoded != "null" {
							payload["operation"] = operation
						}
					}
				}
				if len(payload) == 0 {
					continue
				}
				appendAssistantToolCall("apply_patch", payload, fmt.Sprintf("%v", item["call_id"]))
			case "custom_tool_call":
				callName := strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
				if callName == "" {
					continue
				}
				payload := map[string]any{}
				if input := strings.TrimSpace(cleanFallbackInput(item["input"], "")); input != "" {
					payload["input"] = input
				}
				if len(payload) == 0 {
					continue
				}
				appendAssistantToolCall(callName, payload, fmt.Sprintf("%v", item["call_id"]))
			case "apply_patch_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "apply_patch", item["output"])
			case "custom_tool_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), fmt.Sprintf("%v", item["name"]), item["output"])
			case "web_search_call":
				appendAssistantToolCall("web_search", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "web_search_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "web_search", item["output"])
			case "file_search_call":
				appendAssistantToolCall("file_search", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "file_search_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "file_search", item["output"])
			case "code_interpreter_call":
				appendAssistantToolCall("code_interpreter", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "code_interpreter_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "code_interpreter", item["output"])
			case "image_generation_call":
				appendAssistantToolCall("image_generation", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "image_generation_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "image_generation", item["output"])
			case "computer_call":
				appendAssistantToolCall("computer", item["action"], fmt.Sprintf("%v", item["call_id"]))
			case "computer_call_output":
				appendToolResult(fmt.Sprintf("%v", item["call_id"]), "computer", item["output"])
			default:
				convertOne("user", item)
			}
		}
	}
	if pendingReasoningText != "" {
		out = append(out, map[string]any{
			"role":              "assistant",
			"content":           "",
			"reasoning_content": pendingReasoningText,
		})
	}
	return out
}

func normalizeChatCompletionRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "developer", "user", "assistant", "tool":
		return strings.ToLower(strings.TrimSpace(role))
	default:
		return "user"
	}
}

// Add near the other small helper functions in proxymanager.go.
func splitNonEmptyLines(s string) []string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

func decodeShellCommandArrayString(raw string) ([]string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, false
	}
	var stringValues []string
	if err := json.Unmarshal([]byte(raw), &stringValues); err == nil {
		out := make([]string, 0, len(stringValues))
		for _, value := range stringValues {
			trimmed := strings.TrimSpace(value)
			if trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out, true
	}
	var anyValues []any
	if err := json.Unmarshal([]byte(raw), &anyValues); err == nil {
		out := make([]string, 0, len(anyValues))
		for _, value := range anyValues {
			trimmed := strings.TrimSpace(fmt.Sprintf("%v", value))
			if trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out, true
	}
	return nil, false
}

func normalizeSingleShellCommandString(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if len(trimmed) < 2 {
		return trimmed
	}
	if (trimmed[0] == '\'' && trimmed[len(trimmed)-1] == '\'') || (trimmed[0] == '"' && trimmed[len(trimmed)-1] == '"') {
		inner := strings.TrimSpace(trimmed[1 : len(trimmed)-1])
		if inner != "" && !strings.ContainsAny(inner, "\r\n") {
			return inner
		}
	}
	return trimmed
}

func tokenizeShellCommandString(raw string) []string {
	command := normalizeSingleShellCommandString(raw)
	if command == "" {
		return nil
	}
	tokens := make([]string, 0, 4)
	var current strings.Builder
	inSingle := false
	inDouble := false
	escaped := false

	flush := func() {
		if current.Len() == 0 {
			return
		}
		tokens = append(tokens, current.String())
		current.Reset()
	}

	for _, r := range command {
		switch {
		case escaped:
			current.WriteRune(r)
			escaped = false
		case r == '\\' && !inSingle:
			escaped = true
		case r == '\'' && !inDouble:
			inSingle = !inSingle
		case r == '"' && !inSingle:
			inDouble = !inDouble
		case (r == ' ' || r == '\t') && !inSingle && !inDouble:
			flush()
		default:
			current.WriteRune(r)
		}
	}
	if escaped {
		current.WriteRune('\\')
	}
	flush()
	if len(tokens) == 0 {
		return []string{command}
	}
	return tokens
}

func buildRouterShellCommand(commands []string) []any {
	if len(commands) == 0 {
		return nil
	}
	var tokens []string
	if len(commands) == 1 {
		tokens = tokenizeShellCommandString(commands[0])
	} else {
		allSingleToken := true
		for _, cmd := range commands {
			if len(tokenizeShellCommandString(cmd)) != 1 {
				allSingleToken = false
				break
			}
		}
		if allSingleToken {
			tokens = make([]string, 0, len(commands))
			for _, cmd := range commands {
				tokens = append(tokens, normalizeSingleShellCommandString(cmd))
			}
		} else {
			tokens = tokenizeShellCommandString(commands[0])
		}
	}
	out := make([]any, 0, len(tokens))
	for _, token := range tokens {
		trimmed := strings.TrimSpace(token)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func normalizeShellCommandsValue(raw any) []string {
	switch typed := raw.(type) {
	case nil:
		return nil
	case []any:
		out := make([]string, 0, len(typed))
		for _, entry := range typed {
			trimmed := normalizeSingleShellCommandString(fmt.Sprintf("%v", entry))
			if trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(typed))
		for _, entry := range typed {
			trimmed := normalizeSingleShellCommandString(entry)
			if trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out
	case string:
		trimmed := normalizeSingleShellCommandString(typed)
		if trimmed == "" {
			return nil
		}
		if decoded, ok := decodeShellCommandArrayString(trimmed); ok {
			return decoded
		}
		return splitNonEmptyLines(trimmed)
	default:
		trimmed := normalizeSingleShellCommandString(fmt.Sprintf("%v", typed))
		if trimmed == "" {
			return nil
		}
		if decoded, ok := decodeShellCommandArrayString(trimmed); ok {
			return decoded
		}
		return []string{trimmed}
	}
}

func normalizeShellArgumentMap(raw map[string]any) map[string]any {
	if raw == nil {
		return map[string]any{}
	}
	out := cloneMap(raw)
	commands := make([]string, 0, 2)
	if normalized := normalizeShellCommandsValue(out["commands"]); len(normalized) > 0 {
		commands = append(commands, normalized...)
	}
	if len(commands) == 0 {
		if normalized := normalizeShellCommandsValue(out["command"]); len(normalized) > 0 {
			commands = append(commands, normalized...)
		}
	}
	if len(commands) == 0 {
		if normalized := normalizeShellCommandsValue(out["cmd"]); len(normalized) > 0 {
			commands = append(commands, normalized...)
		}
	}
	if len(commands) > 0 {
		arr := make([]any, 0, len(commands))
		for _, cmd := range commands {
			arr = append(arr, cmd)
		}
		out["commands"] = arr
		// Router-facing canonical shape: `command` must be argv tokens, not a whole
		// multiword command string packed into one array slot.
		out["command"] = buildRouterShellCommand(commands)
	} else {
		delete(out, "commands")
		delete(out, "command")
		delete(out, "cmd")
	}
	return out
}

func shellTokenShouldRemainBare(token string) bool {
	switch strings.TrimSpace(token) {
	case "|", "||", "&", "&&", ";", ";;", "(", ")", "{", "}", ">", ">>", "<", "<<", "2>", "2>>", "1>", "1>>", "2>&1", "1>&2":
		return true
	default:
		return false
	}
}

func shellQuoteToken(token string) string {
	token = normalizeSingleShellCommandString(token)
	if token == "" {
		return "''"
	}
	if shellTokenShouldRemainBare(token) {
		return token
	}
	if !strings.ContainsAny(token, " \t\r\n'\"|&;(){}<>$`\\") {
		return token
	}
	return "'" + strings.ReplaceAll(token, "'", `'\''`) + "'"
}

func buildResponseShellCommand(commands []string) string {
	if len(commands) == 0 {
		return ""
	}
	if len(commands) == 1 {
		return normalizeSingleShellCommandString(commands[0])
	}
	parts := make([]string, 0, len(commands))
	for _, cmd := range commands {
		trimmed := strings.TrimSpace(cmd)
		if trimmed == "" {
			continue
		}
		parts = append(parts, shellQuoteToken(trimmed))
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

func normalizeShellArgumentMapForResponse(raw map[string]any) map[string]any {
	out := normalizeShellArgumentMap(raw)
	if len(out) == 0 {
		return out
	}
	legacyCommands := normalizeShellCommandsValue(out["commands"])
	if len(legacyCommands) > 0 {
		if responseCommand := buildResponseShellCommand(legacyCommands); responseCommand != "" {
			out["commands"] = []any{responseCommand}
		} else {
			delete(out, "commands")
		}
	} else {
		routerCommand := normalizeShellCommandsValue(out["command"])
		if len(routerCommand) > 0 {
			arr := make([]any, 0, len(routerCommand))
			for _, cmd := range routerCommand {
				cmd = normalizeSingleShellCommandString(cmd)
				if cmd != "" {
					arr = append(arr, cmd)
				}
			}
			if len(arr) > 0 {
				out["commands"] = arr
			} else {
				delete(out, "commands")
			}
		}
	}
	return out
}

func canonicalToolNameFromRecipient(recipient string) string {
	recipient = strings.TrimSpace(strings.ToLower(recipient))
	switch recipient {
	case "functions.shell_command":
		return "shell"
	case "functions.apply_patch":
		return "apply_patch"
	case "functions.update_plan":
		return "update_plan"
	case "functions.spawn_agent":
		return "spawn_agent"
	case "functions.send_input":
		return "send_input"
	case "functions.resume_agent":
		return "resume_agent"
	case "functions.wait_agent":
		return "wait_agent"
	case "functions.close_agent":
		return "close_agent"
	}
	if recipient == "" {
		return ""
	}
	parts := strings.Split(recipient, ".")
	last := strings.TrimSpace(parts[len(parts)-1])
	switch last {
	case "shell_command":
		return "shell"
	default:
		return last
	}
}

type unpackedParallelToolUse struct {
	name string
	args map[string]any
}

func unpackParallelToolUses(arguments string) []unpackedParallelToolUse {
	args := parseToolArgsMapString(arguments)
	toolUsesRaw, ok := args["tool_uses"].([]any)
	if !ok || len(toolUsesRaw) == 0 {
		return nil
	}
	out := make([]unpackedParallelToolUse, 0, len(toolUsesRaw))
	for _, raw := range toolUsesRaw {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		name := canonicalToolNameFromRecipient(fmt.Sprintf("%v", item["recipient_name"]))
		if name == "" {
			continue
		}
		params, _ := item["parameters"].(map[string]any)
		if name == "shell" {
			params = normalizeShellArgumentMap(params)
		}
		out = append(out, unpackedParallelToolUse{
			name: name,
			args: params,
		})
	}
	return out
}

func cleanFallbackInput(raw any, preferred string) string {
	if strings.TrimSpace(preferred) != "" {
		return preferred
	}
	if raw == nil {
		return ""
	}
	switch raw.(type) {
	case map[string]any, []any:
		if encoded := encodeAnyAsJSONString(raw); strings.TrimSpace(encoded) != "" {
			return encoded
		}
	}
	s := strings.TrimSpace(fmt.Sprintf("%v", raw))
	lower := strings.ToLower(s)
	if s == "" || lower == "<nil>" || lower == "null" || lower == "[]" || lower == "{}" {
		return ""
	}
	return s
}

func wrapReasoningForHistory(reasoning string) string {
	reasoning = strings.TrimSpace(reasoning)
	if reasoning == "" {
		return ""
	}
	lower := strings.ToLower(reasoning)
	if strings.Contains(lower, "<think>") && strings.Contains(lower, "</think>") {
		return reasoning
	}
	return "<think>\n" + reasoning + "\n</think>"
}

func encodeAnyAsJSONString(v any) string {
	switch typed := v.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return strings.TrimSpace(fmt.Sprintf("%v", v))
		}
		return strings.TrimSpace(string(b))
	}
}

func normalizePossiblyMixedToolArguments(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "{}"
	}
	if json.Valid([]byte(raw)) {
		if strings.HasPrefix(raw, `"`) {
			var inner string
			if err := json.Unmarshal([]byte(raw), &inner); err == nil {
				inner = strings.TrimSpace(inner)
				if inner != "" && json.Valid([]byte(inner)) {
					return inner
				}
			}
		}
		return raw
	}
	if repaired := extractJSONObject(raw); repaired != "" && json.Valid([]byte(repaired)) {
		return repaired
	}
	if repaired := extractJSONObject(raw); repaired != "" {
		if escaped := escapeLiteralNewlinesInJSONObjectStrings(repaired); escaped != "" && json.Valid([]byte(escaped)) {
			return escaped
		}
	}
	if repaired := extractQwenXMLToolArguments(raw); repaired != "" && json.Valid([]byte(repaired)) {
		return repaired
	}
	if repaired := extractXMLToolPayload(raw); repaired != "" && json.Valid([]byte(repaired)) {
		return repaired
	}
	return mustJSONString(map[string]any{"raw": raw})
}

func extractJSONObject(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		return strings.TrimSpace(s[start : end+1])
	}
	return ""
}

func escapeLiteralNewlinesInJSONObjectStrings(s string) string {
	if s == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(s) + 8)
	inString := false
	escaped := false
	for _, r := range s {
		switch {
		case escaped:
			b.WriteRune(r)
			escaped = false
		case r == '\\':
			b.WriteRune(r)
			escaped = true
		case r == '"':
			b.WriteRune(r)
			inString = !inString
		case inString && r == '\n':
			b.WriteString(`\n`)
		case inString && r == '\r':
			// Drop CR and let LF branch handle newline normalization.
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func extractQwenXMLToolArguments(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	lower := strings.ToLower(s)
	if !(strings.Contains(lower, "<tool_call") ||
		strings.Contains(lower, "<function=") ||
		strings.Contains(lower, "<parameter=") ||
		strings.Contains(lower, "<tools>") ||
		strings.Contains(lower, "<shell_commands") ||
		strings.Contains(lower, "<shell_command") ||
		strings.Contains(lower, "<tool_use") ||
		strings.Contains(lower, "<update_plan") ||
		strings.Contains(lower, "<apply_patch") ||
		strings.Contains(lower, "<terminal>")) {
		return ""
	}
	calls, _ := parseQwenXMLToolCalls(s)
	if len(calls) != 1 {
		return ""
	}
	return mustJSONString(calls[0].Arguments)
}

func extractXMLToolPayload(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	lower := strings.ToLower(s)
	if !strings.Contains(lower, "<") || !strings.Contains(lower, ">") {
		return ""
	}
	fields := map[string]string{}
	for _, tag := range []string{"command", "commands", "query", "path", "diff", "text", "input", "code", "language", "action", "button", "x", "y", "prompt", "size", "quality", "background", "output_format"} {
		open := "<" + tag + ">"
		close := "</" + tag + ">"
		start := strings.Index(lower, open)
		end := strings.Index(lower, close)
		if start >= 0 && end > start {
			value := strings.TrimSpace(s[start+len(open) : end])
			if value != "" {
				fields[tag] = value
			}
		}
	}
	if len(fields) == 0 {
		return ""
	}
	if commands, ok := fields["commands"]; ok {
		if decoded, parsed := decodeShellCommandArrayString(commands); parsed {
			return mustJSONString(map[string]any{"commands": decoded})
		}
		return mustJSONString(map[string]any{"commands": splitNonEmptyLines(commands)})
	}
	if command, ok := fields["command"]; ok {
		if decoded, parsed := decodeShellCommandArrayString(command); parsed {
			return mustJSONString(map[string]any{"commands": decoded})
		}
		command = strings.Trim(strings.TrimSpace(command), `[]"`)
		if command == "" {
			return mustJSONString(map[string]any{})
		}
		return mustJSONString(map[string]any{"commands": []string{command}})
	}
	converted := make(map[string]any, len(fields))
	for k, v := range fields {
		converted[k] = v
	}
	return mustJSONString(converted)
}

func parseToolArgsMapString(s string) map[string]any {
	s = normalizePossiblyMixedToolArguments(s)
	out := map[string]any{}
	if err := json.Unmarshal([]byte(s), &out); err == nil {
		return sanitizeBridgeToolArgumentsWithContext(out, "", nil)
	}
	return map[string]any{"raw": strings.TrimSpace(s)}
}

func parseJSONStringMap(s string) (map[string]any, bool) {
	s = strings.TrimSpace(s)
	if s == "" || !json.Valid([]byte(s)) {
		return nil, false
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(s), &out); err != nil || len(out) == 0 {
		return nil, false
	}
	return out, true
}

var requestUserInputQuestionsArrayRegexp = regexp.MustCompile(`(?is)\bquestions\s*:\s*\[((?:[^\[\]"]+|"[^"\\]*(?:\\.[^"\\]*)*")*)\]`)
var quotedJSONStringLiteralRegexp = regexp.MustCompile(`"((?:\\.|[^"\\])*)"`)
var requestUserInputQuestionLineRegexp = regexp.MustCompile(`(?im)\bquestion\s*:\s*"((?:\\.|[^"\\])*)"`)
var requestUserInputToolParameterRegexp = regexp.MustCompile(`(?is)<parameter=questions>\s*(\[[\s\S]*?\])\s*</parameter>`)

func recoverRequestUserInputArgumentsFromTextSources(texts ...string) (string, bool) {
	for _, raw := range texts {
		text := strings.TrimSpace(raw)
		if text == "" {
			continue
		}
		lower := strings.ToLower(text)
		if idx := strings.Index(lower, "\"questions\""); idx >= 0 {
			if start := strings.LastIndex(text[:idx], "{"); start >= 0 {
				if endRel := strings.Index(text[idx:], "}"); endRel > 0 {
					candidate := strings.TrimSpace(text[start : idx+endRel+1])
					if parsed, ok := parseJSONStringMap(candidate); ok {
						if questions, ok := parsed["questions"].([]any); ok && len(questions) > 0 {
							return mustJSONString(map[string]any{"questions": questions}), true
						}
					}
				}
			}
		}
		if match := requestUserInputQuestionsArrayRegexp.FindStringSubmatch(text); len(match) == 2 {
			quoted := quotedJSONStringLiteralRegexp.FindAllString(match[1], -1)
			questions := make([]string, 0, len(quoted))
			for _, q := range quoted {
				decoded, err := strconv.Unquote(q)
				if err != nil {
					continue
				}
				decoded = strings.TrimSpace(decoded)
				if decoded != "" {
					questions = append(questions, decoded)
				}
			}
			if len(questions) > 0 {
				return mustJSONString(map[string]any{"questions": questions}), true
			}
		}
		if matches := requestUserInputQuestionLineRegexp.FindAllStringSubmatch(text, -1); len(matches) > 0 {
			last := matches[len(matches)-1]
			if len(last) == 2 {
				decoded, err := strconv.Unquote(`"` + last[1] + `"`)
				if err == nil {
					decoded = strings.TrimSpace(decoded)
					if decoded != "" {
						return mustJSONString(map[string]any{"questions": []string{decoded}}), true
					}
				}
			}
		}
		if strings.Contains(text, "?") {
			candidate := strings.TrimSpace(text)
			candidate = strings.Trim(candidate, "\"'`")
			if strings.Count(candidate, "?") == 1 && strings.HasSuffix(candidate, "?") {
				return mustJSONString(map[string]any{"questions": []string{candidate}}), true
			}
		}
	}
	return "", false
}

func sanitizeBridgeToolArgumentsWithContext(args map[string]any, toolName string, systemInjector func(string)) map[string]any {
	if args == nil {
		return map[string]any{}
	}
	sanitized := cloneMap(args)
	droppedFields := make([]string, 0, 3)

	popField := func(keys ...string) (any, string, bool) {
		for _, k := range keys {
			if v, ok := sanitized[k]; ok {
				delete(sanitized, k)
				return v, k, true
			}
		}
		return nil, "", false
	}

	if rawRule, key, ok := popField("prefix_rule", "prefixrule"); ok {
		droppedFields = append(droppedFields, key)
		rule := strings.TrimSpace(fmt.Sprintf("%v", rawRule))
		if rule != "" && systemInjector != nil {
			systemInjector(fmt.Sprintf(
				"Safety constraint: the output of the next shell command must begin with: %q. If it does not, stop and report the mismatch before continuing.",
				rule,
			))
		}
	}
	if _, key, ok := popField("sandbox_permissions", "sandboxpermissions"); ok {
		droppedFields = append(droppedFields, key)
	}
	if _, key, ok := popField("justification"); ok {
		droppedFields = append(droppedFields, key)
	}
	if len(droppedFields) > 0 {
		appendApplyPatchTrace("sanitize.bridge_tool_args", map[string]any{
			"dropped_fields": droppedFields,
			"tool_name":      strings.TrimSpace(toolName),
		})
	}
	return sanitized
}

func sanitizeBridgeToolArguments(args map[string]any) map[string]any {
	return sanitizeBridgeToolArgumentsWithContext(args, "", nil)
}

func hasAnyNonEmptyValue(m map[string]any) bool {
	for _, v := range m {
		switch x := v.(type) {
		case string:
			if strings.TrimSpace(x) != "" {
				return true
			}
		case map[string]any:
			if hasAnyNonEmptyValue(x) {
				return true
			}
		case []any:
			for _, item := range x {
				switch typed := item.(type) {
				case string:
					if strings.TrimSpace(typed) != "" {
						return true
					}
				case map[string]any:
					if hasAnyNonEmptyValue(typed) {
						return true
					}
				case []any:
					if len(typed) > 0 {
						return true
					}
				case nil:
					continue
				default:
					return true
				}
			}
		case nil:
			continue
		default:
			return true
		}
	}
	return false
}

func isLikelyJSONSchemaObject(m map[string]any) bool {
	if len(m) == 0 {
		return false
	}
	schemaKeys := map[string]struct{}{
		"type":                 {},
		"properties":           {},
		"required":             {},
		"items":                {},
		"anyOf":                {},
		"oneOf":                {},
		"allOf":                {},
		"enum":                 {},
		"additionalProperties": {},
		"description":          {},
	}
	nonSchemaKeys := 0
	for k := range m {
		if _, ok := schemaKeys[k]; !ok {
			nonSchemaKeys++
		}
	}
	if nonSchemaKeys > 0 {
		return false
	}
	if rawType, ok := m["type"].(string); ok {
		switch strings.ToLower(strings.TrimSpace(rawType)) {
		case "string", "object", "array", "number", "integer", "boolean", "null":
			return true
		}
	}
	_, hasProperties := m["properties"]
	_, hasAnyOf := m["anyOf"]
	_, hasOneOf := m["oneOf"]
	_, hasAllOf := m["allOf"]
	return hasProperties || hasAnyOf || hasOneOf || hasAllOf
}

func hasNonEmptyApplyPatchMap(op map[string]any) bool {
	if len(op) == 0 || isLikelyJSONSchemaObject(op) {
		return false
	}

	if nestedOperation, ok := op["operation"]; ok {
		if hasNonEmptyApplyPatchOperation(nestedOperation) {
			return true
		}
	}

	rawType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", op["type"])))
	path := strings.TrimSpace(fmt.Sprintf("%v", op["path"]))
	diff := normalizeApplyPatchDiff(strings.TrimSpace(cleanFallbackInput(op["diff"], "")))
	patch := strings.TrimSpace(cleanFallbackInput(op["patch"], ""))
	content := strings.TrimSpace(cleanFallbackInput(op["content"], ""))
	input := strings.TrimSpace(cleanFallbackInput(op["input"], ""))
	switch rawType {
	case "create_file", "update_file":
		if path == "" {
			return false
		}
		return diff != "" || patch != "" || content != "" || input != ""
	case "delete_file":
		return path != ""
	}
	if path != "" && (diff != "" || patch != "" || content != "" || input != "") {
		return true
	}

	for _, key := range []string{"patch", "input", "raw", "content"} {
		if value, ok := op[key].(string); ok && strings.TrimSpace(value) != "" {
			if _, ok := convertApplyPatchTextToOperation(value); ok {
				return true
			}
		}
	}
	if diff, ok := op["diff"].(string); ok && strings.TrimSpace(diff) != "" {
		return true
	}

	return false
}

func looksLikeFilePathHint(s string) bool {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return false
	}
	if strings.Contains(trimmed, "\n") || strings.Contains(trimmed, "\r") {
		return false
	}
	if looksLikePatchText(trimmed) {
		return false
	}
	if strings.Contains(trimmed, "/") || strings.Contains(trimmed, "\\") {
		return true
	}
	return false
}

func looksLikeAbsoluteWindowsOrUNCPath(s string) bool {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return false
	}
	if strings.HasPrefix(trimmed, `\\`) || strings.HasPrefix(trimmed, `//`) {
		return true
	}
	if len(trimmed) >= 3 && ((trimmed[0] >= 'A' && trimmed[0] <= 'Z') || (trimmed[0] >= 'a' && trimmed[0] <= 'z')) &&
		trimmed[1] == ':' && (trimmed[2] == '\\' || trimmed[2] == '/') {
		return true
	}
	return false
}

func recoverApplyPatchOperationFromArgs(args map[string]any) (map[string]any, bool) {
	rawOp, ok := args["operation"].(map[string]any)
	if !ok {
		return nil, false
	}
	opAny := normalizeApplyPatchOperation(rawOp)
	op, ok := opAny.(map[string]any)
	if !ok {
		return nil, false
	}
	opType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", op["type"])))
	if opType != "create_file" && opType != "update_file" && opType != "delete_file" {
		return nil, false
	}
	if strings.TrimSpace(fmt.Sprintf("%v", op["path"])) != "" {
		return nil, false
	}

	patchArg := strings.TrimSpace(fmt.Sprintf("%v", args["patch"]))
	inputArg := strings.TrimSpace(fmt.Sprintf("%v", args["input"]))
	contentArg := strings.TrimSpace(fmt.Sprintf("%v", args["content"]))
	diffArg := normalizeApplyPatchDiff(strings.TrimSpace(fmt.Sprintf("%v", args["diff"])))

	path := ""
	if looksLikeFilePathHint(patchArg) {
		path = patchArg
	} else if looksLikeFilePathHint(inputArg) {
		path = inputArg
	}
	if path == "" {
		return nil, false
	}

	recovered := map[string]any{
		"type": opType,
		"path": normalizeApplyPatchPathForWorkspace(path),
	}
	if opType == "create_file" || opType == "update_file" {
		switch {
		case contentArg != "":
			recovered["content"] = contentArg
		case diffArg != "":
			recovered["diff"] = diffArg
		case inputArg != "" && !looksLikeFilePathHint(inputArg):
			recovered["content"] = inputArg
		case patchArg != "" && !looksLikeFilePathHint(patchArg):
			recovered["content"] = patchArg
		}
	}

	if !applyPatchOperationPayloadValid(recovered) {
		return nil, false
	}
	return recovered, true
}

func repairApplyPatchArgsOperationPathWithHint(args map[string]any, applyPatchPathHint string) map[string]any {
	if len(args) == 0 {
		return args
	}
	rawOp, ok := args["operation"].(map[string]any)
	rewrittenArgs := cloneMap(args)
	if !ok {
		for _, key := range []string{"raw", "input", "patch"} {
			raw, _ := args[key].(string)
			decoded, decodedOK := parseJSONStringMap(raw)
			if !decodedOK {
				continue
			}
			nestedOp, nestedOK := decoded["operation"].(map[string]any)
			if !nestedOK {
				continue
			}
			rawOp = nestedOp
			rewrittenArgs = decoded
			ok = true
			break
		}
		if !ok {
			return args
		}
	}
	pathHint := strings.TrimSpace(normalizeApplyPatchPathForWorkspace(applyPatchPathHint))
	if pathHint == "" || !applyPatchPathExistsLocally(pathHint) {
		return args
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(rawOp["type"], ""))
	if opType != "create_file" && opType != "update_file" && opType != "delete_file" {
		return args
	}
	opPath := strings.TrimSpace(cleanFallbackInput(rawOp["path"], ""))
	if opPath != "" {
		normalizedPath := normalizeApplyPatchPathForWorkspace(opPath)
		if applyPatchPathExistsLocally(normalizedPath) || strings.EqualFold(filepath.ToSlash(normalizedPath), filepath.ToSlash(pathHint)) {
			return args
		}
	}
	rewrittenOp := cloneMap(rawOp)
	rewrittenOp["path"] = pathHint
	if (opType == "create_file" || opType == "update_file") &&
		(strings.TrimSpace(cleanFallbackInput(rewrittenOp["content"], "")) != "" ||
			strings.TrimSpace(cleanFallbackInput(rewrittenOp["input"], "")) != "") {
		delete(rewrittenOp, "diff")
	}
	rewrittenArgs["operation"] = rewrittenOp
	return rewrittenArgs
}

func recoverApplyPatchOperationFromStringifiedArgs(args map[string]any) (map[string]any, bool) {
	for _, key := range []string{"input", "patch", "raw"} {
		raw, ok := args[key].(string)
		if !ok || strings.TrimSpace(raw) == "" {
			continue
		}
		decoded, ok := parseJSONStringMap(raw)
		if !ok {
			continue
		}
		if operation, ok := decoded["operation"]; ok {
			normalized := normalizeApplyPatchOperation(operation)
			if applyPatchOperationPayloadValid(normalized) {
				if op, ok := normalized.(map[string]any); ok {
					return op, true
				}
			}
		}
		if opAny := selectApplyPatchOperation(decoded); applyPatchOperationPayloadValid(opAny) {
			if op, ok := opAny.(map[string]any); ok {
				return op, true
			}
		}
	}
	return nil, false
}

func selectApplyPatchOperation(args map[string]any) any {
	if recovered, ok := recoverApplyPatchOperationFromStringifiedArgs(args); ok {
		return recovered
	}
	if operation, ok := args["operation"]; ok {
		if rawOperation, ok := operation.(string); ok && strings.TrimSpace(rawOperation) != "" {
			if decoded, decodedOK := parseJSONStringMap(rawOperation); decodedOK {
				normalized := normalizeApplyPatchOperation(decoded)
				if applyPatchOperationPayloadValid(normalized) {
					return normalized
				}
				return normalized
			}
			if converted, convertedOK := convertApplyPatchTextToOperation(rawOperation); convertedOK {
				return converted
			}
		}
		if hasNonEmptyApplyPatchOperation(operation) {
			normalized := normalizeApplyPatchOperation(operation)
			if applyPatchOperationPayloadValid(normalized) {
				return normalized
			}
			if recovered, ok := recoverApplyPatchOperationFromArgs(args); ok {
				return recovered
			}
			return normalized
		}
	}
	if patch, ok := args["patch"].(string); ok && strings.TrimSpace(patch) != "" {
		if operation, ok := convertApplyPatchTextToOperation(patch); ok {
			return operation
		}
	}
	if input, ok := args["input"].(string); ok && strings.TrimSpace(input) != "" {
		if operation, ok := convertApplyPatchTextToOperation(input); ok {
			return operation
		}
	}
	if recovered, ok := recoverApplyPatchOperationFromArgs(args); ok {
		return recovered
	}
	if raw, ok := args["raw"].(string); ok && strings.TrimSpace(raw) != "" {
		if operation, ok := convertApplyPatchTextToOperation(raw); ok {
			return operation
		}
	}
	if hasNonEmptyApplyPatchMap(args) {
		return normalizeApplyPatchOperation(args)
	}
	return map[string]any{}
}

func hasNonEmptyApplyPatchOperation(operation any) bool {
	switch op := operation.(type) {
	case nil:
		return false
	case string:
		trimmed := strings.TrimSpace(op)
		if trimmed == "" {
			return false
		}
		if _, ok := convertApplyPatchTextToOperation(trimmed); ok {
			return true
		}
		return false
	case map[string]any:
		return hasNonEmptyApplyPatchMap(op)
	case []any:
		return len(op) > 0
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", op)) != ""
	}
}

func normalizeApplyPatchText(text string) string {
	normalized := strings.ReplaceAll(text, "\r\n", "\n")
	// Recover patch blocks emitted with JSON-like line continuations (`\\\n`).
	normalized = strings.ReplaceAll(normalized, "\\\n", "\n")
	normalized = stripApplyPatchNoNewlineMarker(normalized)
	normalized = strings.TrimSpace(normalized)
	if normalized == "" {
		return ""
	}
	replacer := strings.NewReplacer(
		"*** Edit File:", "*** Update File:",
		"*** Patch File:", "*** Update File:",
	)
	normalized = replacer.Replace(normalized)

	if strings.Contains(normalized, "*** Update File:") ||
		strings.Contains(normalized, "*** Add File:") ||
		strings.Contains(normalized, "*** Delete File:") {
		if !strings.Contains(normalized, "*** Begin Patch") {
			normalized = "*** Begin Patch\n" + normalized
		}
		if !strings.Contains(normalized, "*** End Patch") {
			normalized += "\n*** End Patch"
		}
	}
	if !strings.HasSuffix(normalized, "\n") {
		normalized += "\n"
	}
	return normalized
}

func normalizeApplyPatchPathForWorkspace(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	path = strings.ReplaceAll(path, `\/`, `/`)
	path = strings.ReplaceAll(path, `\\`, `\`)

	// Preserve absolute Windows/UNC paths as provided by the model. The test
	// harness may execute apply_patch against a workspace outside llama-swap.
	if isLikelyAbsoluteApplyPatchPath(path) {
		return path
	}

	normalized := strings.ReplaceAll(path, "\\", "/")
	normalized = strings.TrimSpace(normalized)

	for _, marker := range []string{"/llama-swap-main/", "/llama-swap/"} {
		if idx := strings.LastIndex(strings.ToLower(normalized), marker); idx >= 0 {
			candidate := strings.TrimPrefix(normalized[idx+len(marker):], "/")
			if candidate != "" {
				candidate = filepath.ToSlash(candidate)
				if _, err := os.Stat(candidate); err == nil {
					return candidate
				}
			}
		}
	}

	if _, err := os.Stat(normalized); err == nil {
		return filepath.ToSlash(normalized)
	}

	return path
}

func extractWorkspaceRootHintFromResponsesText(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	matchers := []*regexp.Regexp{
		regexp.MustCompile(`(?is)<cwd>\s*([^<\r\n]+?)\s*</cwd>`),
		regexp.MustCompile("(?im)^\\s*cwd\\s*[:=]\\s*([^\\r\\n]+)\\s*$"),
	}
	for _, re := range matchers {
		match := re.FindStringSubmatch(text)
		if len(match) < 2 {
			continue
		}
		candidate := strings.TrimSpace(match[1])
		candidate = strings.Trim(candidate, "\"'`")
		if candidate == "" {
			continue
		}
		return normalizeApplyPatchPathForWorkspace(candidate)
	}
	return ""
}

func resolveApplyPatchPathHintCandidate(candidate string, requestText string) string {
	candidate = strings.TrimSpace(candidate)
	candidate = strings.Trim(candidate, ".,;:!?)(")
	if candidate == "" {
		return ""
	}
	normalizedCandidate := normalizeApplyPatchPathForWorkspace(candidate)
	if isLikelyAbsoluteApplyPatchPath(normalizedCandidate) || applyPatchPathExistsLocally(normalizedCandidate) {
		return normalizedCandidate
	}
	workspaceRoot := strings.TrimSpace(extractWorkspaceRootHintFromResponsesText(requestText))
	if workspaceRoot == "" {
		return normalizedCandidate
	}
	joinedRoot := strings.TrimRight(strings.ReplaceAll(workspaceRoot, "\\", "/"), "/")
	joinedCandidate := strings.TrimLeft(strings.ReplaceAll(candidate, "\\", "/"), "/")
	joined := normalizeApplyPatchPathForWorkspace(joinedRoot + "/" + joinedCandidate)
	if applyPatchPathExistsLocally(joined) {
		return joined
	}
	return normalizedCandidate
}

func isLikelyAbsoluteApplyPatchPath(path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	if strings.HasPrefix(path, `\\`) || strings.HasPrefix(path, `//`) {
		return true
	}
	if len(path) >= 3 {
		drive := path[0]
		if ((drive >= 'A' && drive <= 'Z') || (drive >= 'a' && drive <= 'z')) &&
			path[1] == ':' && (path[2] == '\\' || path[2] == '/') {
			return true
		}
	}
	return filepath.IsAbs(path)
}

func normalizeApplyPatchDiff(diff string) string {
	diff = strings.ReplaceAll(diff, "\r\n", "\n")
	diff = stripApplyPatchNoNewlineMarker(diff)
	diff = strings.TrimSpace(diff)
	if diff == "" {
		return ""
	}

	lines := strings.Split(diff, "\n")
	for idx, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "@@") {
			return strings.Join(lines[idx:], "\n")
		}
	}

	filtered := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "********** ") ||
			strings.HasPrefix(trimmed, "--- ") ||
			strings.HasPrefix(trimmed, "+++ ") {
			continue
		}
		filtered = append(filtered, line)
	}
	return strings.TrimSpace(strings.Join(filtered, "\n"))
}

func stripApplyPatchNoNewlineMarker(text string) string {
	if text == "" {
		return ""
	}
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	filtered := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == `\ No newline at end of file` {
			continue
		}
		filtered = append(filtered, line)
	}
	return strings.Join(filtered, "\n")
}

func stripCodexCommandOutputEnvelope(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	if strings.HasPrefix(trimmed, "Chunk ID:") {
		if idx := strings.Index(trimmed, "\nOutput:\n"); idx >= 0 {
			return strings.TrimSpace(trimmed[idx+len("\nOutput:\n"):])
		}
	}
	return trimmed
}

func extractApplyPatchDiffBody(patch string) string {
	normalized := normalizeApplyPatchText(patch)
	if normalized == "" {
		return ""
	}
	lines := strings.Split(normalized, "\n")
	directiveIndex := -1
	for idx, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "*** Update File:") || strings.HasPrefix(trimmed, "*** Add File:") {
			directiveIndex = idx
			break
		}
	}
	if directiveIndex == -1 {
		return normalizeApplyPatchDiff(normalized)
	}
	body := make([]string, 0, len(lines))
	for _, line := range lines[directiveIndex+1:] {
		trimmed := strings.TrimSpace(line)
		if trimmed == "*** End Patch" {
			break
		}
		body = append(body, line)
	}
	return strings.TrimSpace(strings.Join(body, "\n"))
}

func looksLikeLooseApplyPatchDiffBody(diff string) bool {
	diff = strings.ReplaceAll(strings.TrimSpace(diff), "\r\n", "\n")
	if diff == "" {
		return false
	}
	if strings.Contains(diff, "@@") || strings.Contains(diff, "*** Begin Patch") {
		return false
	}
	lines := strings.Split(diff, "\n")
	if len(lines) < 2 {
		return false
	}
	hasDelta := false
	hasContext := false
	for _, line := range lines {
		if line == "" {
			continue
		}
		switch line[0] {
		case '+', '-':
			hasDelta = true
		case ' ':
			hasContext = true
		default:
			hasContext = true
		}
	}
	return hasDelta && hasContext
}

func rebuildContentFromLooseApplyPatchDiff(path string, diff string) (string, bool) {
	path = strings.TrimSpace(normalizeApplyPatchPathForWorkspace(path))
	diff = strings.ReplaceAll(strings.TrimSpace(diff), "\r\n", "\n")
	if path == "" || diff == "" || !applyPatchPathExistsLocally(path) || !looksLikeLooseApplyPatchDiffBody(diff) {
		return "", false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	existingLines := strings.Split(strings.TrimRight(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n"), "\n")
	diffLines := strings.Split(diff, "\n")
	out := make([]string, 0, len(existingLines)+len(diffLines))
	existingIdx := 0
	for idx, line := range diffLines {
		if line == "" {
			out = append(out, "")
			if existingIdx < len(existingLines) && existingLines[existingIdx] == "" {
				existingIdx++
			}
			continue
		}
		switch line[0] {
		case '+':
			out = append(out, line[1:])
		case '-':
			target := line[1:]
			if existingIdx < len(existingLines) && existingLines[existingIdx] == target {
				existingIdx++
			}
		case ' ':
			target := line[1:]
			out = append(out, target)
			if existingIdx < len(existingLines) && existingLines[existingIdx] == target {
				existingIdx++
			}
		default:
			target := line
			if existingIdx < len(existingLines) && existingLines[existingIdx] == target {
				if idx+1 < len(diffLines) && strings.HasPrefix(diffLines[idx+1], "+") {
					existingIdx++
					continue
				}
				existingIdx++
			}
			out = append(out, target)
		}
	}
	if existingIdx < len(existingLines) {
		out = append(out, existingLines[existingIdx:]...)
	}
	rebuilt := strings.TrimSpace(strings.Join(out, "\n"))
	if rebuilt == "" {
		return "", false
	}
	return rebuilt, true
}

func buildApplyPatchDiffFromOperation(op map[string]any) string {
	if op == nil {
		return ""
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(op["type"], ""))
	path := strings.TrimSpace(cleanFallbackInput(op["path"], ""))
	diff := normalizeApplyPatchDiff(strings.TrimSpace(cleanFallbackInput(op["diff"], "")))
	content := strings.TrimSpace(cleanFallbackInput(op["content"], ""))
	if content == "" {
		content = strings.TrimSpace(cleanFallbackInput(op["input"], ""))
	}
	if diff != "" {
		if strings.Contains(diff, "*** Begin Patch") {
			if body := extractApplyPatchDiffBody(diff); body != "" {
				return body
			}
		}
		if looksLikePatchHunkOrDocument(diff) && !shouldRebuildWeakApplyPatchDiff(path, diff, content) {
			return diff
		}
	}
	switch opType {
	case "create_file":
		if content == "" {
			return diff
		}
		lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
		prefixed := make([]string, 0, len(lines))
		for _, line := range lines {
			prefixed = append(prefixed, "+"+line)
		}
		return strings.Join(prefixed, "\n")
	case "update_file":
		if patch := buildHeuristicUpdatePatchFromExistingFile(path, content); patch != "" {
			if body := extractApplyPatchDiffBody(patch); body != "" {
				return body
			}
		}
		if content != "" {
			lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
			var b strings.Builder
			b.WriteString("@@\n")
			for _, line := range lines {
				b.WriteString("+")
				b.WriteString(line)
				b.WriteString("\n")
			}
			return strings.TrimSpace(b.String())
		}
	}
	return diff
}

func shouldRebuildWeakApplyPatchDiff(path string, diff string, content string) bool {
	path = strings.TrimSpace(normalizeApplyPatchPathForWorkspace(path))
	diff = normalizeApplyPatchDiff(diff)
	content = strings.ReplaceAll(strings.TrimSpace(content), "\r\n", "\n")
	if path == "" || diff == "" || content == "" || !applyPatchPathExistsLocally(path) {
		return false
	}
	lines := strings.Split(diff, "\n")
	hasDelete := false
	for _, line := range lines {
		if strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "---") {
			hasDelete = true
			break
		}
	}
	if hasDelete {
		appendApplyPatchTrace("weak_diff_rebuild_probe", map[string]any{
			"path":         path,
			"result":       false,
			"reason":       "has_delete",
			"content_len":  len(content),
			"diff_excerpt": truncateBridgeDebugText(diff, 180),
		})
		return false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		appendApplyPatchTrace("weak_diff_rebuild_probe", map[string]any{
			"path":         path,
			"result":       false,
			"reason":       "read_failed",
			"content_len":  len(content),
			"diff_excerpt": truncateBridgeDebugText(diff, 180),
		})
		return false
	}
	existingLines := strings.Split(strings.TrimRight(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n"), "\n")
	contentLines := strings.Split(content, "\n")
	if len(existingLines) > 0 && len(contentLines) > len(existingLines) {
		prefixMatch := true
		for i := range existingLines {
			if i >= len(contentLines) || existingLines[i] != contentLines[i] {
				prefixMatch = false
				break
			}
		}
		if prefixMatch {
			appendApplyPatchTrace("weak_diff_rebuild_probe", map[string]any{
				"path":           path,
				"result":         true,
				"reason":         "existing_prefix_of_content",
				"content_len":    len(content),
				"existing_lines": len(existingLines),
				"content_lines":  len(contentLines),
				"diff_excerpt":   truncateBridgeDebugText(diff, 180),
			})
			return true
		}
	}
	if len(contentLines) < max(2, len(existingLines)/2) {
		appendApplyPatchTrace("weak_diff_rebuild_probe", map[string]any{
			"path":           path,
			"result":         false,
			"reason":         "content_too_short",
			"content_len":    len(content),
			"existing_lines": len(existingLines),
			"content_lines":  len(contentLines),
			"diff_excerpt":   truncateBridgeDebugText(diff, 180),
		})
		return false
	}
	shared := countSharedApplyPatchLines(existingLines, contentLines)
	result := shared >= 2
	appendApplyPatchTrace("weak_diff_rebuild_probe", map[string]any{
		"path":           path,
		"result":         result,
		"reason":         "shared_line_eval",
		"content_len":    len(content),
		"existing_lines": len(existingLines),
		"content_lines":  len(contentLines),
		"shared_lines":   shared,
		"diff_excerpt":   truncateBridgeDebugText(diff, 180),
	})
	return result
}

func normalizeApplyPatchOperation(operation any) any {
	switch typed := operation.(type) {
	case map[string]any:
		if nested, ok := typed["operation"]; ok {
			return normalizeApplyPatchOperation(nested)
		}
		for _, key := range []string{"patch", "input", "raw", "content"} {
			if s, ok := typed[key].(string); ok && strings.TrimSpace(s) != "" {
				if operation, ok := convertApplyPatchTextToOperation(s); ok {
					return operation
				}
			}
		}
		out := cloneMap(typed)
		opType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", out["type"])))
		if path, ok := out["path"].(string); ok && strings.TrimSpace(path) != "" {
			out["path"] = normalizeApplyPatchPathForWorkspace(path)
		}
		if opType == "create_file" {
			if path := strings.TrimSpace(cleanFallbackInput(out["path"], "")); path != "" && applyPatchPathExistsLocally(path) {
				out["type"] = "update_file"
				opType = "update_file"
			}
		}
		if s, ok := out["diff"].(string); ok && strings.TrimSpace(s) != "" {
			out["diff"] = normalizeApplyPatchDiff(s)
		}
		// Accept legacy `input` but emit canonical operation fields.
		if opType == "create_file" || opType == "update_file" {
			content := strings.TrimSpace(stripTrailingApplyPatchArtifact(stripCodexCommandOutputEnvelope(stripApplyPatchNoNewlineMarker(cleanFallbackInput(out["content"], "")))))
			input := strings.TrimSpace(stripTrailingApplyPatchArtifact(stripCodexCommandOutputEnvelope(stripApplyPatchNoNewlineMarker(cleanFallbackInput(out["input"], "")))))
			diff := strings.TrimSpace(cleanFallbackInput(out["diff"], ""))
			patch := strings.TrimSpace(cleanFallbackInput(out["patch"], ""))
			if content != "" {
				out["content"] = content
			}
			if input != "" {
				out["input"] = input
			}
			if opType == "update_file" && diff != "" && content == "" && input == "" && patch == "" {
				path := strings.TrimSpace(cleanFallbackInput(out["path"], ""))
				if rebuilt, ok := rebuildContentFromLooseApplyPatchDiff(path, diff); ok && strings.TrimSpace(rebuilt) != "" {
					out["content"] = rebuilt
					content = rebuilt
					delete(out, "diff")
					diff = ""
				}
			}
			// Recover weaker update diffs (e.g. "\n+line") by deriving content.
			if opType == "update_file" && diff != "" && content == "" && input == "" && patch == "" && !looksLikePatchHunkOrDocument(diff) {
				if derived, ok := deriveContentFromApplyPatchDiff(diff); ok && strings.TrimSpace(derived) != "" {
					content = strings.TrimSpace(derived)
					out["content"] = content
				}
			}
			if opType == "update_file" && diff != "" && content == "" && input == "" && patch == "" {
				path := strings.TrimSpace(cleanFallbackInput(out["path"], ""))
				if recovered, ok := recoverContentFromWeakApplyPatchUpdate(path, diff); ok && strings.TrimSpace(recovered) != "" {
					out["content"] = recovered
					content = recovered
					delete(out, "diff")
					diff = ""
				}
			}
			if opType == "update_file" && content != "" {
				path := strings.TrimSpace(cleanFallbackInput(out["path"], ""))
				if repaired, ok := repairApplyPatchContentWithConcatenatedRewrite(path, content); ok && strings.TrimSpace(repaired) != "" {
					out["content"] = repaired
					content = repaired
				}
				if repaired, ok := repairApplyPatchContentWithPrefixedTail(path, content); ok && strings.TrimSpace(repaired) != "" {
					out["content"] = repaired
					content = repaired
				}
				if diff != "" && shouldRebuildWeakApplyPatchDiff(path, diff, content) {
					delete(out, "diff")
					diff = ""
				}
			}
			if input != "" && content == "" {
				out["content"] = input
			}
		}
		if opType == "create_file" || opType == "update_file" {
			if strings.TrimSpace(cleanFallbackInput(out["content"], "")) != "" {
				// Content-first operations are safer for Codex-facing clients than
				// synthetic diff reconstruction, which can re-expand a full-file
				// rewrite into a weak append-only diff.
				delete(out, "diff")
			} else if rebuiltDiff := strings.TrimSpace(buildApplyPatchDiffFromOperation(out)); rebuiltDiff != "" {
				out["diff"] = rebuiltDiff
			}
		}
		delete(out, "input")
		delete(out, "patch")
		delete(out, "raw")
		return out
	case string:
		if operation, ok := convertApplyPatchTextToOperation(typed); ok {
			return operation
		}
		return normalizeApplyPatchText(typed)
	default:
		return operation
	}
}

func stripTrailingApplyPatchArtifact(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	if text == "" {
		return ""
	}
	lower := strings.ToLower(text)
	suspiciousMarkers := []string{
		"</parameter>",
		"</function>",
		"</tool_call>",
	}
	cutIdx := -1
	for _, marker := range suspiciousMarkers {
		idx := strings.Index(lower, marker)
		if idx < 0 {
			continue
		}
		if idx > 0 && (cutIdx == -1 || idx < cutIdx) {
			cutIdx = idx
		}
	}
	if cutIdx >= 0 {
		return strings.TrimSpace(text[:cutIdx])
	}
	return strings.TrimSpace(text)
}

func repairApplyPatchContentWithPrefixedTail(path string, content string) (string, bool) {
	path = strings.TrimSpace(normalizeApplyPatchPathForWorkspace(path))
	content = strings.ReplaceAll(content, "\r\n", "\n")
	if path == "" || content == "" || !applyPatchPathExistsLocally(path) {
		return "", false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	existing := strings.ReplaceAll(string(raw), "\r\n", "\n")
	existingTrimmed := strings.TrimRight(existing, "\n")
	contentTrimmed := strings.TrimRight(content, "\n")
	if existingTrimmed == "" || !strings.HasPrefix(contentTrimmed, existingTrimmed+"\n+") {
		return "", false
	}
	tail := strings.TrimPrefix(contentTrimmed, existingTrimmed+"\n")
	if strings.TrimSpace(tail) == "" {
		return "", false
	}
	tailLines := strings.Split(tail, "\n")
	stripped := make([]string, 0, len(tailLines))
	for _, line := range tailLines {
		if !strings.HasPrefix(line, "+") {
			return "", false
		}
		stripped = append(stripped, strings.TrimPrefix(line, "+"))
	}
	existingLines := strings.Split(existingTrimmed, "\n")
	overlap := longestApplyPatchSuffixPrefixOverlap(existingLines, stripped)
	newLines := stripped[overlap:]
	if len(newLines) == 0 {
		return "", false
	}
	desired := append(append([]string{}, existingLines...), newLines...)
	return strings.Join(desired, "\n"), true
}

func repairApplyPatchContentWithConcatenatedRewrite(path string, content string) (string, bool) {
	path = strings.TrimSpace(normalizeApplyPatchPathForWorkspace(path))
	content = strings.ReplaceAll(content, "\r\n", "\n")
	if path == "" || content == "" || !applyPatchPathExistsLocally(path) {
		return "", false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	existing := strings.ReplaceAll(string(raw), "\r\n", "\n")
	existingTrimmed := strings.TrimRight(existing, "\n")
	contentTrimmed := strings.TrimRight(content, "\n")
	if existingTrimmed == "" || !strings.HasPrefix(contentTrimmed, existingTrimmed+"\n") {
		return "", false
	}
	candidate := strings.TrimPrefix(contentTrimmed, existingTrimmed+"\n")
	if strings.TrimSpace(candidate) == "" {
		return "", false
	}
	if strings.Contains(candidate, "\n+") {
		return "", false
	}
	candidateLines := strings.Split(candidate, "\n")
	existingLines := strings.Split(existingTrimmed, "\n")
	if len(candidateLines) < 2 || len(candidateLines) < max(2, len(existingLines)/2) {
		return "", false
	}
	shared := countSharedApplyPatchLines(existingLines, candidateLines)
	if shared < 2 {
		return "", false
	}
	if candidate == existingTrimmed {
		return "", false
	}
	return candidate, true
}

func longestApplyPatchSuffixPrefixOverlap(existingLines []string, candidateLines []string) int {
	maxOverlap := len(candidateLines)
	if len(existingLines) < maxOverlap {
		maxOverlap = len(existingLines)
	}
	for overlap := maxOverlap; overlap > 0; overlap-- {
		matched := true
		start := len(existingLines) - overlap
		for i := 0; i < overlap; i++ {
			if existingLines[start+i] != candidateLines[i] {
				matched = false
				break
			}
		}
		if matched {
			return overlap
		}
	}
	return 0
}

func countSharedApplyPatchLines(existingLines []string, candidateLines []string) int {
	counts := make(map[string]int, len(existingLines))
	for _, line := range existingLines {
		counts[line]++
	}
	shared := 0
	for _, line := range candidateLines {
		if counts[line] > 0 {
			counts[line]--
			shared++
		}
	}
	return shared
}

func buildApplyPatchInputFromOperation(operation any) string {
	opAny := normalizeApplyPatchOperation(operation)
	op, ok := opAny.(map[string]any)
	if !ok {
		return ""
	}
	opType := strings.ToLower(strings.TrimSpace(cleanFallbackInput(op["type"], "")))
	path := strings.TrimSpace(cleanFallbackInput(op["path"], ""))
	if path == "" {
		return ""
	}
	switch opType {
	case "delete_file":
		return "*** Begin Patch\n*** Delete File: " + path + "\n*** End Patch\n"
	case "create_file":
		content, _ := op["content"].(string)
		content = strings.ReplaceAll(strings.TrimSpace(content), "\r\n", "\n")
		if content == "" {
			input, _ := op["input"].(string)
			content = strings.ReplaceAll(strings.TrimSpace(input), "\r\n", "\n")
		}
		if content == "" {
			diff := normalizeApplyPatchDiff(strings.TrimSpace(cleanFallbackInput(op["diff"], "")))
			if diff != "" {
				if strings.Contains(diff, "*** Begin Patch") {
					return diff
				}
				lines := strings.Split(strings.ReplaceAll(diff, "\r\n", "\n"), "\n")
				contentLines := make([]string, 0, len(lines))
				for _, line := range lines {
					if strings.HasPrefix(line, "+++") || strings.HasPrefix(line, "@@") {
						continue
					}
					if strings.HasPrefix(line, "+") {
						contentLines = append(contentLines, strings.TrimPrefix(line, "+"))
					}
				}
				if len(contentLines) > 0 {
					var b strings.Builder
					b.WriteString("*** Begin Patch\n")
					b.WriteString("*** Add File: ")
					b.WriteString(path)
					b.WriteString("\n")
					for _, c := range contentLines {
						b.WriteString("+")
						b.WriteString(c)
						b.WriteString("\n")
					}
					b.WriteString("*** End Patch\n")
					return b.String()
				}
				return ""
			}
			return ""
		}
		lines := strings.Split(content, "\n")
		var b strings.Builder
		b.WriteString("*** Begin Patch\n")
		b.WriteString("*** Add File: ")
		b.WriteString(path)
		b.WriteString("\n")
		for _, line := range lines {
			b.WriteString("+")
			b.WriteString(line)
			b.WriteString("\n")
		}
		b.WriteString("*** End Patch\n")
		return b.String()
	case "update_file":
		diff := normalizeApplyPatchDiff(strings.TrimSpace(cleanFallbackInput(op["diff"], "")))
		content := strings.TrimSpace(cleanFallbackInput(op["content"], ""))
		if content == "" {
			content = strings.TrimSpace(cleanFallbackInput(op["input"], ""))
		}
		if diff != "" && shouldRebuildWeakApplyPatchDiff(path, diff, content) {
			diff = ""
		}
		if diff != "" {
			if strings.Contains(diff, "*** Begin Patch") {
				return normalizeApplyPatchText(diff)
			}
			if !looksLikePatchHunkOrDocument(diff) {
				if patch := buildHeuristicUpdatePatchFromExistingFile(path, diff); patch != "" {
					return patch
				}
			}
			var b strings.Builder
			b.WriteString("*** Begin Patch\n")
			b.WriteString("*** Update File: ")
			b.WriteString(path)
			b.WriteString("\n")
			b.WriteString(diff)
			if !strings.HasSuffix(diff, "\n") {
				b.WriteString("\n")
			}
			b.WriteString("*** End Patch\n")
			return b.String()
		}
		// If only full replacement content is available, synthesize a minimal
		// hunk so Codex apply_patch receives valid patch text instead of an
		// empty/malformed payload that triggers retry loops.
		if content != "" {
			if patch := buildHeuristicUpdatePatchFromExistingFile(path, content); patch != "" {
				return patch
			}
			if patch := buildHeuristicUpdatePatchFromContentOnly(path, content); patch != "" {
				return patch
			}
			lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
			var b strings.Builder
			b.WriteString("*** Begin Patch\n")
			b.WriteString("*** Update File: ")
			b.WriteString(path)
			b.WriteString("\n@@\n")
			for _, line := range lines {
				b.WriteString("+")
				b.WriteString(line)
				b.WriteString("\n")
			}
			b.WriteString("*** End Patch\n")
			return b.String()
		}
		return ""
	default:
		return ""
	}
}

func buildHeuristicUpdatePatchFromExistingFile(path string, fragment string) string {
	fragment = strings.ReplaceAll(strings.TrimSpace(fragment), "\r\n", "\n")
	if fragment == "" || looksLikePatchHunkOrDocument(fragment) {
		return ""
	}
	targetPath, err := canonicalizeLocalApplyPatchPath(path)
	if err != nil {
		return ""
	}
	raw, err := os.ReadFile(targetPath)
	if err != nil {
		return ""
	}
	existing := strings.ReplaceAll(string(raw), "\r\n", "\n")
	appendMode := shouldTreatUpdateFragmentAsAppend(existing, fragment)
	existingLinesRaw := strings.Split(strings.TrimRight(existing, "\n"), "\n")
	fragmentLinesRaw := strings.Split(fragment, "\n")
	if len(fragmentLinesRaw) >= 2 {
		existingFirst := ""
		fragmentFirst := ""
		if len(existingLinesRaw) > 0 {
			existingFirst = strings.ToLower(strings.TrimSpace(existingLinesRaw[0]))
		}
		if len(fragmentLinesRaw) > 0 {
			fragmentFirst = strings.ToLower(strings.TrimSpace(fragmentLinesRaw[0]))
		}
		if (strings.HasPrefix(existingFirst, "<!doctype") && strings.HasPrefix(fragmentFirst, "<!doctype")) ||
			(strings.HasPrefix(existingFirst, "<html") && strings.HasPrefix(fragmentFirst, "<html")) ||
			(strings.HasPrefix(existingFirst, "{") && strings.HasPrefix(fragmentFirst, "{")) ||
			(strings.HasPrefix(existingFirst, "[") && strings.HasPrefix(fragmentFirst, "[")) {
			appendMode = false
		}
	}
	addedFragment := fragment
	existingTrimmed := strings.TrimRight(existing, "\n")
	if appendMode && existingTrimmed != "" && strings.HasPrefix(fragment, existingTrimmed+"\n") {
		addedFragment = strings.TrimPrefix(fragment, existingTrimmed+"\n")
	}
	var b strings.Builder
	b.WriteString("*** Begin Patch\n")
	b.WriteString("*** Update File: ")
	b.WriteString(path)
	b.WriteString("\n@@\n")
	if strings.TrimRight(existing, "\n") != "" {
		for _, line := range strings.Split(strings.TrimRight(existing, "\n"), "\n") {
			prefix := "-"
			if appendMode {
				prefix = " "
			}
			b.WriteString(prefix)
			b.WriteString(line)
			b.WriteString("\n")
		}
	}
	for _, line := range strings.Split(addedFragment, "\n") {
		b.WriteString("+")
		b.WriteString(line)
		b.WriteString("\n")
	}
	b.WriteString("*** End Patch\n")
	return b.String()
}

func buildHeuristicUpdatePatchFromContentOnly(path string, content string) string {
	content = strings.ReplaceAll(strings.TrimSpace(content), "\r\n", "\n")
	if path == "" || content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString("*** Begin Patch\n")
	b.WriteString("*** Update File: ")
	b.WriteString(path)
	b.WriteString("\n@@\n")
	b.WriteString("-")
	b.WriteString(lines[0])
	b.WriteString("\n")
	for _, line := range lines {
		b.WriteString("+")
		b.WriteString(line)
		b.WriteString("\n")
	}
	b.WriteString("*** End Patch\n")
	return b.String()
}

func extractStructuredLineKey(line string) string {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return ""
	}
	for _, sep := range []string{"=", ":"} {
		if idx := strings.Index(trimmed, sep); idx > 0 {
			key := strings.TrimSpace(trimmed[:idx])
			if key != "" {
				return strings.ToLower(key)
			}
		}
	}
	return ""
}

func fragmentConflictsWithExistingStructuredLines(existing string, fragment string) bool {
	existing = strings.ReplaceAll(existing, "\r\n", "\n")
	fragment = strings.ReplaceAll(strings.TrimSpace(fragment), "\r\n", "\n")
	if existing == "" || fragment == "" {
		return false
	}
	existingKeys := map[string]string{}
	for _, line := range strings.Split(strings.TrimRight(existing, "\n"), "\n") {
		key := extractStructuredLineKey(line)
		if key == "" {
			continue
		}
		existingKeys[key] = strings.TrimSpace(line)
	}
	if len(existingKeys) == 0 {
		return false
	}
	conflict := false
	for _, line := range strings.Split(fragment, "\n") {
		key := extractStructuredLineKey(line)
		if key == "" {
			continue
		}
		if existingLine, ok := existingKeys[key]; ok && existingLine != strings.TrimSpace(line) {
			conflict = true
			break
		}
	}
	return conflict
}

func shouldTreatUpdateFragmentAsAppend(existing string, fragment string) bool {
	fragment = strings.ReplaceAll(strings.TrimSpace(fragment), "\r\n", "\n")
	existing = strings.ReplaceAll(existing, "\r\n", "\n")
	if fragment == "" {
		return false
	}
	if fragmentConflictsWithExistingStructuredLines(existing, fragment) {
		return false
	}
	existingLines := strings.Split(strings.TrimRight(existing, "\n"), "\n")
	fragmentLines := strings.Split(fragment, "\n")
	// A one-line file updated to a different one-line value is much more
	// likely to be a replacement than an append.
	if len(existingLines) == 1 && len(fragmentLines) == 1 &&
		strings.TrimSpace(existingLines[0]) != "" &&
		strings.TrimSpace(existingLines[0]) != strings.TrimSpace(fragmentLines[0]) {
		return false
	}
	existingFirst := ""
	fragmentFirst := ""
	if len(existingLines) > 0 {
		existingFirst = strings.ToLower(strings.TrimSpace(existingLines[0]))
	}
	if len(fragmentLines) > 0 {
		fragmentFirst = strings.ToLower(strings.TrimSpace(fragmentLines[0]))
	}
	if len(fragmentLines) >= 2 {
		if (strings.HasPrefix(existingFirst, "<!doctype") && strings.HasPrefix(fragmentFirst, "<!doctype")) ||
			(strings.HasPrefix(existingFirst, "<html") && strings.HasPrefix(fragmentFirst, "<html")) ||
			(strings.HasPrefix(existingFirst, "{") && strings.HasPrefix(fragmentFirst, "{")) ||
			(strings.HasPrefix(existingFirst, "[") && strings.HasPrefix(fragmentFirst, "[")) {
			return false
		}
	}
	if !strings.Contains(fragment, "\n") {
		return true
	}
	if existing == "" {
		return false
	}
	if strings.Contains(existing, fragment) {
		return false
	}
	if len(fragmentLines) >= max(2, len(existingLines)/2) {
		if countSharedApplyPatchLines(existingLines, fragmentLines) >= 2 {
			return false
		}
	}
	return len(fragment) < len(existing)
}

func applyPatchPathExistsLocally(path string) bool {
	targetPath, err := canonicalizeLocalApplyPatchPath(path)
	if err != nil {
		return false
	}
	info, err := os.Stat(targetPath)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

func recoverContentFromWeakApplyPatchUpdate(path string, diff string) (string, bool) {
	path = strings.TrimSpace(path)
	diff = normalizeApplyPatchDiff(diff)
	if path == "" || diff == "" || !looksLikePatchHunkOrDocument(diff) {
		return "", false
	}

	targetPath, err := canonicalizeLocalApplyPatchPath(path)
	if err != nil {
		return "", false
	}
	raw, err := os.ReadFile(targetPath)
	if err != nil {
		return "", false
	}
	existing := strings.ReplaceAll(string(raw), "\r\n", "\n")

	lines := strings.Split(diff, "\n")
	inHunk := false
	added := make([]string, 0, len(lines))
	removed := make([]string, 0, len(lines))
	context := make([]string, 0, len(lines))
	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "@@"):
			inHunk = true
		case strings.HasPrefix(line, "---") || strings.HasPrefix(line, "+++"):
			continue
		case !inHunk:
			continue
		case strings.HasPrefix(line, "+"):
			added = append(added, strings.TrimPrefix(line, "+"))
		case strings.HasPrefix(line, "-"):
			removed = append(removed, strings.TrimPrefix(line, "-"))
		case strings.HasPrefix(line, " "):
			context = append(context, strings.TrimPrefix(line, " "))
		}
	}

	if len(added) == 0 {
		return "", false
	}

	addedText := strings.Join(added, "\n")
	removedText := strings.Join(removed, "\n")
	contextText := strings.Join(context, "\n")
	existingTrimmed := strings.TrimRight(existing, "\n")

	if len(removed) == 0 && len(context) > 0 && existingTrimmed == contextText {
		stripped := make([]string, 0, len(added))
		allPrefixed := len(added) > 0
		for _, line := range added {
			if !strings.HasPrefix(line, "+") {
				allPrefixed = false
				break
			}
			stripped = append(stripped, strings.TrimPrefix(line, "+"))
		}
		if allPrefixed && len(stripped) > 0 {
			out := existing
			if out != "" && !strings.HasSuffix(out, "\n") {
				out += "\n"
			}
			out += strings.Join(stripped, "\n")
			if strings.HasSuffix(existing, "\n") && !strings.HasSuffix(out, "\n") {
				out += "\n"
			}
			return out, true
		}
		out := existing
		if out != "" && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		out += addedText
		if strings.HasSuffix(existing, "\n") && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		return out, true
	}

	if len(removed) == 0 && shouldTreatUpdateFragmentAsAppend(existing, addedText) {
		out := existing
		if out != "" && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		out += addedText
		if strings.HasSuffix(existing, "\n") && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		return out, true
	}

	if removedText != "" && existingTrimmed == removedText {
		out := addedText
		if strings.HasSuffix(existing, "\n") && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		return out, true
	}

	return "", false
}

func convertApplyPatchTextToOperation(text string) (map[string]any, bool) {
	normalized := normalizeApplyPatchText(text)
	if strings.TrimSpace(normalized) == "" {
		return nil, false
	}
	lines := strings.Split(normalized, "\n")
	directiveIndex := -1
	directiveLine := ""
	for idx, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "*** Update File:") ||
			strings.HasPrefix(trimmed, "*** Add File:") ||
			strings.HasPrefix(trimmed, "*** Delete File:") {
			directiveIndex = idx
			directiveLine = trimmed
			break
		}
	}
	if directiveIndex == -1 {
		return nil, false
	}

	parseDirective := func(prefix string, opType string) (map[string]any, bool) {
		if !strings.HasPrefix(directiveLine, prefix) {
			return nil, false
		}
		path := normalizeApplyPatchPathForWorkspace(strings.TrimSpace(strings.TrimPrefix(directiveLine, prefix)))
		if path == "" {
			return nil, false
		}
		op := map[string]any{
			"type": opType,
			"path": path,
		}
		if opType == "delete_file" {
			return op, true
		}
		diffLines := make([]string, 0, len(lines))
		for _, line := range lines[directiveIndex+1:] {
			trimmed := strings.TrimSpace(line)
			if trimmed == "*** End Patch" {
				break
			}
			diffLines = append(diffLines, line)
		}
		diff := strings.TrimRight(strings.Join(diffLines, "\n"), "\n")
		if strings.TrimSpace(diff) == "" {
			return nil, false
		}
		op["diff"] = diff
		return op, true
	}

	if op, ok := parseDirective("*** Update File:", "update_file"); ok {
		return op, true
	}
	if op, ok := parseDirective("*** Add File:", "create_file"); ok {
		return op, true
	}
	if op, ok := parseDirective("*** Delete File:", "delete_file"); ok {
		return op, true
	}
	return nil, false
}

func extractApplyPatchBlock(text string) string {
	start := strings.Index(text, "*** Begin Patch")
	if start < 0 {
		return ""
	}
	rest := text[start:]
	endRel := strings.Index(rest, "*** End Patch")
	if endRel < 0 {
		return ""
	}
	end := start + endRel + len("*** End Patch")
	patch := strings.TrimSpace(text[start:end])
	if patch == "" {
		return ""
	}
	return patch + "\n"
}

func extractApplyPatchFromFragmentedText(text string) string {
	if p := extractApplyPatchBlock(text); p != "" {
		return p
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}

	candidates := make([]string, 0, 8)
	candidates = append(candidates, trimmed)

	fenceRe := regexp.MustCompile("(?s)```(?:diff|patch|text|[a-zA-Z0-9_-]+)?\\s*\\n(.*?)```")
	for _, m := range fenceRe.FindAllStringSubmatch(text, -1) {
		if len(m) < 2 {
			continue
		}
		body := strings.TrimSpace(m[1])
		if body != "" {
			candidates = append(candidates, body)
		}
	}

	applyTagRe := regexp.MustCompile(`(?is)<apply_patch[^>]*>(.*?)</apply_patch>`)
	for _, m := range applyTagRe.FindAllStringSubmatch(text, -1) {
		if len(m) < 2 {
			continue
		}
		body := strings.TrimSpace(m[1])
		if body != "" {
			candidates = append(candidates, body)
		}
	}

	seen := map[string]struct{}{}
	patchLike := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		if _, ok := seen[candidate]; ok {
			continue
		}
		seen[candidate] = struct{}{}
		lower := strings.ToLower(candidate)
		if strings.Contains(lower, "*** add file:") ||
			strings.Contains(lower, "*** update file:") ||
			strings.Contains(lower, "*** delete file:") ||
			strings.Contains(lower, "*** begin patch") ||
			strings.Contains(lower, "@@") {
			patchLike = append(patchLike, candidate)
		}
	}

	for _, candidate := range patchLike {
		normalized := normalizeApplyPatchText(candidate)
		if _, ok := convertApplyPatchTextToOperation(normalized); ok {
			return normalized
		}
	}

	if len(patchLike) > 1 {
		merged := strings.Join(patchLike, "\n")
		normalized := normalizeApplyPatchText(merged)
		if _, ok := convertApplyPatchTextToOperation(normalized); ok {
			return normalized
		}
	}

	return ""
}

func looksLikePatchText(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || trimmed == "{}" {
		return false
	}
	if strings.Contains(trimmed, "*** Begin Patch") && strings.Contains(trimmed, "*** End Patch") {
		return true
	}
	// Unified diff style markers.
	if strings.Contains(trimmed, "\n+++ ") || strings.Contains(trimmed, "\n--- ") ||
		strings.Contains(trimmed, "\n@@ ") || strings.Contains(trimmed, "\n@@ -") {
		return true
	}
	return false
}

func validateBridgeToolCallItemResult(item map[string]any) ToolValidationResult {
	return proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(item)
}

func validateBridgeToolCallItem(item map[string]any) (bool, string) {
	result := validateBridgeToolCallItemResult(item)
	return result.Valid, result.Warning
}

func shellToolArgumentsValid(raw map[string]any) bool {
	if len(raw) == 0 {
		return false
	}
	normalized := normalizeShellArgumentMap(raw)
	if len(normalizeShellCommandsValue(normalized["commands"])) > 0 {
		return true
	}
	if len(normalizeShellCommandsValue(normalized["command"])) > 0 {
		return true
	}
	return false
}

func shellToolArgumentsLookLikePlainQuestion(raw map[string]any) bool {
	if len(raw) == 0 {
		return false
	}
	normalized := normalizeShellArgumentMap(raw)
	commands := normalizeShellCommandsValue(normalized["commands"])
	if len(commands) == 0 {
		commands = normalizeShellCommandsValue(normalized["command"])
	}
	if len(commands) == 0 {
		return false
	}
	joined := strings.TrimSpace(strings.Join(commands, " "))
	if joined == "" {
		return false
	}
	lower := strings.ToLower(joined)
	questionLike := strings.HasSuffix(joined, "?") ||
		strings.HasPrefix(lower, "what ") ||
		strings.HasPrefix(lower, "why ") ||
		strings.HasPrefix(lower, "how ") ||
		strings.HasPrefix(lower, "which ") ||
		strings.HasPrefix(lower, "when ") ||
		strings.HasPrefix(lower, "where ")
	if !questionLike {
		return false
	}
	for _, marker := range []string{"|", ">", "<", ";", "&", "$", "/", "\\", "git ", "rg ", "ls", "pwd", "cat ", "cd ", ".ps1", ".sh"} {
		if strings.Contains(lower, marker) {
			return false
		}
	}
	return len(strings.Fields(joined)) >= 4
}

func truncateBridgeDebugText(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "...<truncated>"
}

func buildReasoningCommentaryPreview(reasoning string) string {
	reasoning = strings.TrimSpace(stripLeadingReasoningDirective(reasoning))
	if reasoning == "" {
		return ""
	}
	reasoning = bridgeHTMLTagRegexp.ReplaceAllString(reasoning, " ")
	reasoning = strings.Join(strings.Fields(reasoning), " ")
	if reasoning == "" {
		return ""
	}
	const maxLen = 220
	if len(reasoning) <= maxLen {
		return reasoning
	}
	trimmed := strings.TrimSpace(reasoning[:maxLen])
	lastBreak := strings.LastIndexAny(trimmed, ".!?;:, ")
	if lastBreak >= 80 {
		trimmed = strings.TrimSpace(trimmed[:lastBreak])
	}
	if trimmed == "" {
		trimmed = strings.TrimSpace(reasoning[:maxLen])
	}
	return trimmed + " ..."
}

func buildIncrementalReasoningSummaryDelta(reasoning string, previousSummary string) (string, string) {
	summary := buildReasoningCommentaryPreview(reasoning)
	if strings.TrimSpace(summary) == "" {
		return strings.TrimSpace(previousSummary), ""
	}
	previousSummary = strings.TrimSpace(previousSummary)
	if summary == previousSummary {
		return summary, ""
	}
	if previousSummary != "" && strings.HasPrefix(summary, previousSummary) {
		return summary, summary[len(previousSummary):]
	}
	if previousSummary != "" {
		// When the rolling preview gets re-trimmed, avoid replaying the whole
		// summary as a fresh delta. Keep the canonical final summary updated, but
		// let response.reasoning_summary_text.done carry the rewritten full text.
		return summary, ""
	}
	return summary, summary
}

func looksLikePlanAcknowledgementWithoutPlan(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(stripLeadingReasoningDirective(text)))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, "<proposed_plan>") && strings.Contains(lower, "</proposed_plan>") {
		return false
	}
	return strings.Contains(lower, "i have everything i need") ||
		strings.Contains(lower, "here's the plan:") ||
		strings.Contains(lower, "here is the plan:") ||
		strings.Contains(lower, "great - i have everything i need") ||
		strings.Contains(lower, "great — i have everything i need")
}

func buildWorkflowProgressCommentary(planOutputRequired bool, nativeQuestionRequired bool, workflowState ToolWorkflowState, latestToolName string, finalizing bool) string {
	switch {
	case nativeQuestionRequired:
		return ""
	case planOutputRequired && completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input"):
		return ""
	case planOutputRequired && finalizing:
		return ""
	case planOutputRequired:
		return ""
	case finalizing && workflowState.FinalAnswerSafe:
		return ""
	case strings.EqualFold(strings.TrimSpace(latestToolName), "apply_patch"):
		return ""
	case strings.EqualFold(strings.TrimSpace(latestToolName), "shell") && workflowState.VerificationExpected && !workflowState.VerificationCompleted:
		return ""
	case workflowState.VerificationExpected && !workflowState.VerificationCompleted:
		return ""
	case workflowState.HasToolOutput:
		return ""
	default:
		return ""
	}
}

func toolStartProgressCommentary(toolName string, workflowState ToolWorkflowState) string {
	switch strings.ToLower(strings.TrimSpace(toolName)) {
	case "apply_patch":
		return ""
	case "request_user_input":
		return ""
	case "shell":
		_ = workflowState
	}
	return ""
}

func replayToolProgressCommentary(toolName string) string {
	_ = toolName
	return ""
}

func buildResponsesUsageFromChatUsage(usage gjson.Result) map[string]any {
	if !usage.Exists() {
		return nil
	}
	usageMap := map[string]any{
		"input_tokens":  usage.Get("prompt_tokens").Int(),
		"output_tokens": usage.Get("completion_tokens").Int(),
		"total_tokens":  usage.Get("total_tokens").Int(),
	}
	reasoningTokens := usage.Get("completion_tokens_details.reasoning_tokens").Int()
	if reasoningTokens > 0 {
		usageMap["output_tokens_details"] = map[string]any{
			"reasoning_tokens": reasoningTokens,
		}
	}
	cachedTokens := usage.Get("prompt_tokens_details.cached_tokens").Int()
	if cachedTokens > 0 {
		usageMap["input_tokens_details"] = map[string]any{
			"cached_tokens": cachedTokens,
		}
	}
	return usageMap
}

func buildResponsesTimingFromChatTiming(timings gjson.Result) map[string]any {
	if !timings.Exists() {
		return nil
	}
	timingMap := map[string]any{
		"prompt_n":             timings.Get("prompt_n").Int(),
		"predicted_n":          timings.Get("predicted_n").Int(),
		"prompt_per_second":    timings.Get("prompt_per_second").Float(),
		"predicted_per_second": timings.Get("predicted_per_second").Float(),
		"prompt_ms":            timings.Get("prompt_ms").Float(),
		"predicted_ms":         timings.Get("predicted_ms").Float(),
	}
	if cacheN := timings.Get("cache_n"); cacheN.Exists() {
		timingMap["cache_n"] = cacheN.Int()
	}
	return timingMap
}

func buildResponsesUsageFromTimingMap(timingMap map[string]any) map[string]any {
	if len(timingMap) == 0 {
		return nil
	}
	inputTokens, _ := timingMap["prompt_n"].(int64)
	outputTokens, _ := timingMap["predicted_n"].(int64)
	if inputTokens <= 0 && outputTokens <= 0 {
		return nil
	}
	usageMap := map[string]any{
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
		"total_tokens":  inputTokens + outputTokens,
	}
	if cacheN, ok := timingMap["cache_n"].(int64); ok && cacheN > 0 {
		usageMap["input_tokens_details"] = map[string]any{
			"cached_tokens": cacheN,
		}
	}
	return usageMap
}

func summarizeChatCompletionToolCalls(body []byte) string {
	var parts []string
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	if model != "" {
		parts = append(parts, "model="+model)
	}
	message := gjson.GetBytes(body, "choices.0.message")
	parts = append(parts, fmt.Sprintf("content_len=%d", len(strings.TrimSpace(message.Get("content").String()))))
	parts = append(parts, fmt.Sprintf("reasoning_len=%d", len(strings.TrimSpace(message.Get("reasoning_content").String()))))

	toolCalls := message.Get("tool_calls").Array()
	parts = append(parts, fmt.Sprintf("tool_calls=%d", len(toolCalls)))
	for i, tc := range toolCalls {
		name := strings.TrimSpace(tc.Get("function.name").String())
		args := truncateBridgeDebugText(tc.Get("function.arguments").String(), 200)
		parts = append(parts, fmt.Sprintf("tc[%d].name=%q", i, name))
		parts = append(parts, fmt.Sprintf("tc[%d].args=%q", i, args))
	}

	fnName := strings.TrimSpace(message.Get("function_call.name").String())
	if fnName != "" {
		fnArgs := truncateBridgeDebugText(message.Get("function_call.arguments").String(), 200)
		parts = append(parts, fmt.Sprintf("function_call.name=%q", fnName))
		parts = append(parts, fmt.Sprintf("function_call.args=%q", fnArgs))
	}

	return strings.Join(parts, " ")
}

func appendApplyPatchTrace(stage string, fields map[string]any) {
	if strings.TrimSpace(stage) == "" {
		return
	}
	record := map[string]any{
		"ts":    time.Now().UTC().Format(time.RFC3339Nano),
		"stage": stage,
	}
	for k, v := range fields {
		record[k] = v
	}
	line, err := json.Marshal(record)
	if err != nil {
		return
	}
	applyPatchTraceMu.Lock()
	defer applyPatchTraceMu.Unlock()
	f, err := os.OpenFile(applyPatchTraceLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}

func summarizeApplyPatchResponsesRequest(body []byte) string {
	tools := gjson.GetBytes(body, "tools").Array()
	applyPatchTools := 0
	toolNames := make([]string, 0, len(tools))
	for _, t := range tools {
		name := strings.TrimSpace(t.Get("name").String())
		if name == "" {
			name = strings.TrimSpace(t.Get("function.name").String())
		}
		if cleaned := strings.TrimSpace(cleanFallbackInput(name, "")); cleaned != "" {
			name = cleaned
			toolNames = append(toolNames, name)
		}
		if strings.EqualFold(name, "apply_patch") {
			applyPatchTools++
		}
	}
	inputRaw := gjson.GetBytes(body, "input").String()
	inputLower := strings.ToLower(inputRaw)
	return fmt.Sprintf(
		"model=%q tools=%d apply_patch_tools=%d tool_names=%q tool_choice=%q input_mentions_apply_patch=%t input_has_begin_patch=%t",
		strings.TrimSpace(gjson.GetBytes(body, "model").String()),
		len(tools),
		applyPatchTools,
		truncateBridgeDebugText(strings.Join(toolNames, ","), 120),
		truncateBridgeDebugText(strings.TrimSpace(gjson.GetBytes(body, "tool_choice").Raw), 120),
		strings.Contains(inputLower, "apply_patch"),
		strings.Contains(inputRaw, "*** Begin Patch"),
	)
}

func summarizeApplyPatchResponsesOutput(body []byte) string {
	output := gjson.GetBytes(body, "output").Array()
	applyPatchCalls := 0
	warnings := 0
	for _, item := range output {
		itemType := strings.TrimSpace(item.Get("type").String())
		if itemType == "apply_patch_call" {
			applyPatchCalls++
		}
		if itemType == "message" {
			content := item.Get("content").Array()
			for _, part := range content {
				text := strings.ToLower(strings.TrimSpace(part.Get("text").String()))
				if strings.Contains(text, strings.ToLower(applyPatchValidationWarningPrefix)) {
					warnings++
				}
			}
		}
	}
	return fmt.Sprintf("output_items=%d apply_patch_calls=%d empty_arg_warnings=%d", len(output), applyPatchCalls, warnings)
}

func summarizeApplyPatchChatCompletionRequest(body []byte) string {
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	toolChoice := truncateBridgeDebugText(strings.TrimSpace(gjson.GetBytes(body, "tool_choice").Raw), 140)
	parallel := strings.TrimSpace(gjson.GetBytes(body, "parallel_tool_calls").Raw)
	if parallel == "" {
		parallel = "unset"
	}
	tools := gjson.GetBytes(body, "tools").Array()
	applyPatchFound := false
	applyPatchRequired := "n/a"
	applyPatchProps := "n/a"
	toolNames := make([]string, 0, len(tools))
	for _, t := range tools {
		fn := t.Get("function")
		name := strings.TrimSpace(fn.Get("name").String())
		if name == "" {
			name = strings.TrimSpace(t.Get("name").String())
		}
		if cleaned := strings.TrimSpace(cleanFallbackInput(name, "")); cleaned != "" {
			name = cleaned
			toolNames = append(toolNames, name)
		}
		if strings.EqualFold(name, "apply_patch") || strings.EqualFold(name, llamaSwapApplyPatchFunctionName) {
			applyPatchFound = true
			requiredRaw := strings.TrimSpace(fn.Get("parameters.required").Raw)
			if requiredRaw == "" {
				requiredRaw = strings.TrimSpace(t.Get("parameters.required").Raw)
			}
			if requiredRaw != "" {
				applyPatchRequired = truncateBridgeDebugText(requiredRaw, 120)
			}
			propsRaw := strings.TrimSpace(fn.Get("parameters.properties").Raw)
			if propsRaw == "" {
				propsRaw = strings.TrimSpace(t.Get("parameters.properties").Raw)
			}
			if propsRaw != "" {
				applyPatchProps = truncateBridgeDebugText(propsRaw, 140)
			}
		}
	}
	return fmt.Sprintf(
		"model=%q tools=%d tool_names=%q tool_choice=%q parallel_tool_calls=%s apply_patch_found=%t apply_patch_required=%q apply_patch_props=%q",
		model,
		len(tools),
		truncateBridgeDebugText(strings.Join(toolNames, ","), 160),
		toolChoice,
		parallel,
		applyPatchFound,
		applyPatchRequired,
		applyPatchProps,
	)
}

func summarizeParsedToolCallForensics(label string, calls []ParsedToolCall) string {
	if len(calls) == 0 {
		return label + "=0"
	}
	parts := []string{fmt.Sprintf("%s=%d", label, len(calls))}
	for idx, call := range calls {
		name := strings.TrimSpace(call.Name)
		argsJSON := mustJSONString(call.Arguments)
		parts = append(parts,
			fmt.Sprintf("%s[%d].name=%q", label, idx, name),
			fmt.Sprintf("%s[%d].has_input=%t", label, idx, strings.TrimSpace(fmt.Sprintf("%v", call.Arguments["input"])) != ""),
			fmt.Sprintf("%s[%d].has_patch=%t", label, idx, strings.TrimSpace(fmt.Sprintf("%v", call.Arguments["patch"])) != ""),
			fmt.Sprintf("%s[%d].has_raw=%t", label, idx, strings.TrimSpace(fmt.Sprintf("%v", call.Arguments["raw"])) != ""),
			fmt.Sprintf("%s[%d].looks_like_patch=%t", label, idx, looksLikePatchText(argsJSON)),
		)
	}
	return strings.Join(parts, " ")
}

func summarizeApplyPatchBridgeForensics(body []byte) string {
	message := gjson.GetBytes(body, "choices.0.message")
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	content := message.Get("content").String()
	reasoning := message.Get("reasoning_content").String()
	toolCalls := message.Get("tool_calls").Array()

	parts := []string{
		"model=" + model,
		fmt.Sprintf("content_has_patch=%t", extractApplyPatchBlock(content) != ""),
		fmt.Sprintf("reasoning_has_patch=%t", extractApplyPatchBlock(reasoning) != ""),
		fmt.Sprintf("content_mentions_apply_patch=%t", strings.Contains(strings.ToLower(content), "apply_patch")),
		fmt.Sprintf("reasoning_mentions_apply_patch=%t", strings.Contains(strings.ToLower(reasoning), "apply_patch")),
	}

	parsedTextCalls, _ := proxyCompatibilityAdapters.ToolRepair.ParseAssistantOutput(model, content)
	parsedReasoningCalls, _ := proxyCompatibilityAdapters.ToolRepair.ParseAssistantOutput(model, reasoning)
	parts = append(parts, summarizeParsedToolCallForensics("parsed_text_calls", parsedTextCalls))
	parts = append(parts, summarizeParsedToolCallForensics("parsed_reasoning_calls", parsedReasoningCalls))

	parts = append(parts, fmt.Sprintf("native_tool_calls=%d", len(toolCalls)))
	for idx, tc := range toolCalls {
		name := strings.TrimSpace(tc.Get("function.name").String())
		rawArgs := strings.TrimSpace(tc.Get("function.arguments").String())
		normalizedArgs := normalizePossiblyMixedToolArguments(rawArgs)
		parsedArgs := parseToolArgsMapString(normalizedArgs)
		parts = append(parts,
			fmt.Sprintf("native[%d].name=%q", idx, name),
			fmt.Sprintf("native[%d].raw_len=%d", idx, len(rawArgs)),
			fmt.Sprintf("native[%d].raw_is_empty=%t", idx, rawArgs == "" || rawArgs == "{}"),
			fmt.Sprintf("native[%d].raw_looks_like_patch=%t", idx, looksLikePatchText(rawArgs)),
			fmt.Sprintf("native[%d].normalized=%q", idx, truncateBridgeDebugText(normalizedArgs, 160)),
			fmt.Sprintf("native[%d].parsed_has_input=%t", idx, strings.TrimSpace(fmt.Sprintf("%v", parsedArgs["input"])) != ""),
			fmt.Sprintf("native[%d].parsed_has_patch=%t", idx, strings.TrimSpace(fmt.Sprintf("%v", parsedArgs["patch"])) != ""),
			fmt.Sprintf("native[%d].parsed_has_raw=%t", idx, strings.TrimSpace(fmt.Sprintf("%v", parsedArgs["raw"])) != ""),
		)
	}
	return strings.Join(parts, " ")
}

func normalizeApplyPatchTypeHint(hint string) string {
	switch strings.ToLower(strings.TrimSpace(hint)) {
	case "create_file", "createfile":
		return "create_file"
	case "update_file", "updatefile":
		return "update_file"
	case "delete_file", "deletefile":
		return "delete_file"
	default:
		return ""
	}
}

func enrichApplyPatchOperationWithHints(operation any, applyPatchPathHint string, applyPatchContentHint string, applyPatchTypeHint string) any {
	op, ok := normalizeApplyPatchOperation(operation).(map[string]any)
	if !ok {
		return operation
	}
	rewritten := cloneMap(op)
	if strings.TrimSpace(cleanFallbackInput(rewritten["path"], "")) == "" {
		if pathHint := strings.TrimSpace(normalizeApplyPatchPathForWorkspace(applyPatchPathHint)); pathHint != "" {
			rewritten["path"] = pathHint
		}
	}
	if normalizeApplyPatchTypeHint(cleanFallbackInput(rewritten["type"], "")) == "" {
		if typeHint := normalizeApplyPatchTypeHint(applyPatchTypeHint); typeHint != "" {
			rewritten["type"] = typeHint
		}
	}
	if normalizeApplyPatchTypeHint(cleanFallbackInput(rewritten["type"], "")) == "" {
		path := strings.TrimSpace(cleanFallbackInput(rewritten["path"], ""))
		hasContentPayload := strings.TrimSpace(cleanFallbackInput(rewritten["content"], "")) != "" ||
			strings.TrimSpace(cleanFallbackInput(rewritten["diff"], "")) != ""
		if path != "" && hasContentPayload {
			if applyPatchPathExistsLocally(path) {
				rewritten["type"] = "update_file"
			} else {
				rewritten["type"] = "create_file"
			}
		}
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(rewritten["type"], ""))
	if (opType == "create_file" || opType == "update_file") &&
		strings.TrimSpace(cleanFallbackInput(rewritten["content"], "")) == "" &&
		strings.TrimSpace(cleanFallbackInput(rewritten["diff"], "")) == "" {
		if contentHint := strings.TrimSpace(applyPatchContentHint); contentHint != "" {
			rewritten["content"] = contentHint
		}
	}
	return normalizeApplyPatchOperation(rewritten)
}

func extractLikelyFinalFileContentBlock(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	start := strings.LastIndex(text, "```")
	if start == -1 {
		return ""
	}
	prefix := text[:start]
	open := strings.LastIndex(prefix, "```")
	if open == -1 {
		return ""
	}
	block := prefix[open+3:]
	if nl := strings.Index(block, "\n"); nl != -1 {
		lang := strings.TrimSpace(block[:nl])
		if lang != "" && !strings.Contains(lang, "=") && !strings.Contains(lang, "/") && !strings.Contains(lang, "\\") {
			block = block[nl+1:]
		}
	}
	block = strings.TrimSpace(block)
	if block == "" || !strings.Contains(block, "\n") || looksLikePatchText(block) {
		return ""
	}
	return block
}

func repairApplyPatchOperationFromReasoningBlock(operation any, text string, reasoningText string) any {
	op, ok := normalizeApplyPatchOperation(operation).(map[string]any)
	if !ok {
		return operation
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(op["type"], ""))
	path := strings.TrimSpace(cleanFallbackInput(op["path"], ""))
	if path == "" || !applyPatchPathExistsLocally(path) || (opType != "create_file" && opType != "update_file") {
		return op
	}
	content := strings.TrimSpace(cleanFallbackInput(op["content"], ""))
	diff := strings.TrimSpace(cleanFallbackInput(op["diff"], ""))
	block := extractLikelyFinalFileContentBlock(reasoningText)
	if block == "" {
		block = extractLikelyFinalFileContentBlock(text)
	}
	if block == "" {
		return op
	}
	needsRepair := opType == "create_file" || content == "" || (!looksLikePatchHunkOrDocument(diff) && strings.Contains(diff, "+")) || strings.Contains(content, "+") || strings.Contains(content, "-")
	if !needsRepair {
		return op
	}
	op["type"] = "update_file"
	op["path"] = normalizeApplyPatchPathForWorkspace(path)
	op["content"] = block
	delete(op, "diff")
	return normalizeApplyPatchOperation(op)
}

func preferContentDrivenApplyPatchOperation(operation any) any {
	op, ok := normalizeApplyPatchOperation(operation).(map[string]any)
	if !ok {
		return operation
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(op["type"], ""))
	path := strings.TrimSpace(cleanFallbackInput(op["path"], ""))
	content := strings.TrimSpace(cleanFallbackInput(op["content"], ""))
	if content == "" {
		return op
	}
	if path == "" || !applyPatchPathExistsLocally(path) {
		return op
	}
	if opType != "update_file" && opType != "create_file" {
		return op
	}
	rewritten := cloneMap(op)
	delete(rewritten, "diff")
	return normalizeApplyPatchOperation(rewritten)
}

func translateChatCompletionToResponsesResponse(body []byte, applyPatchPathHint string, applyPatchContentHint string, applyPatchTypeHint string) ([]byte, error) {
	return translateChatCompletionToResponsesResponseWithWorkflow(body, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint, ToolWorkflowState{}, "", nil)
}

func recoverNativeQuestionResponseIfRequired(responseBody []byte, upstreamBody []byte, requestBody []byte) []byte {
	if !nativeQuestionOutputRequiredFromRequestBody(requestBody) || !gjson.ValidBytes(responseBody) || !gjson.ValidBytes(upstreamBody) {
		return responseBody
	}
	if responseContainsNamedFunctionCall(responseBody, "request_user_input") {
		return forceResponseStatus(responseBody, "requires_action")
	}

	message := gjson.GetBytes(upstreamBody, "choices.0.message")
	if !message.Exists() {
		return responseBody
	}
	contentText := strings.TrimSpace(message.Get("content").String())
	reasoningText := strings.TrimSpace(message.Get("reasoning_content").String())
	contentText, extractedReasoning := extractContentAndReasoning(contentText)
	contentText = strings.TrimSpace(stripLeadingReasoningDirective(contentText))
	if strings.TrimSpace(reasoningText) == "" {
		reasoningText = strings.TrimSpace(extractedReasoning)
	}

	normalizedArtifacts, _ := normalizeQwenAssistantMessageFromBody(upstreamBody, false, true)
	normalizedView := buildNormalizedArtifactView(normalizedArtifacts)
	recoveredArgs := ""
	if normalizedView.PreferredQuestion != nil {
		parsedArgs, _ := normalizedView.PreferredQuestion.Payload["parsed_args"].(map[string]any)
		if hasNonEmptyQuestionList(parsedArgs["questions"]) {
			recoveredArgs = mustJSONString(parsedArgs)
		}
		rawArgs := strings.TrimSpace(cleanFallbackInput(normalizedView.PreferredQuestion.Payload["raw_args"], ""))
		if strings.TrimSpace(recoveredArgs) == "" && rawArgs != "" {
			recoveredArgs = rawArgs
		}
	}
	if strings.TrimSpace(recoveredArgs) == "" {
		if recovered, ok := recoverRequestUserInputArgumentsFromTextSources(reasoningText, contentText); ok {
			recoveredArgs = recovered
		}
	}
	if strings.TrimSpace(recoveredArgs) == "" {
		var req map[string]any
		if err := json.Unmarshal(requestBody, &req); err == nil {
			if synthesized, ok := synthesizeRequestUserInputArgumentsFromRequest(req); ok {
				recoveredArgs = synthesized
			}
		}
	}
	if strings.TrimSpace(recoveredArgs) == "" {
		return responseBody
	}

	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}
	respID := strings.TrimSpace(fmt.Sprintf("%v", resp["id"]))
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
		resp["id"] = respID
	}

	output := make([]any, 0, 2)
	if normalizedView.PreferredThinking != nil {
		text := strings.TrimSpace(cleanFallbackInput(normalizedView.PreferredThinking.Payload["text"], ""))
		if text == "" {
			goto doneReasoning
		}
		if wellFormed, _ := normalizedView.PreferredThinking.Payload["well_formed"].(bool); !wellFormed {
			goto doneReasoning
		}
		output = append(output, map[string]any{
			"id":   fmt.Sprintf("rs_%s_native_question", respID),
			"type": "reasoning",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": text,
				},
			},
		})
	}
doneReasoning:
	output = append(output, map[string]any{
		"id":        fmt.Sprintf("fc_%s_native_question", respID),
		"type":      "function_call",
		"call_id":   fmt.Sprintf("call_%s_native_question", respID),
		"name":      "request_user_input",
		"arguments": normalizePossiblyMixedToolArguments(recoveredArgs),
		"status":    "in_progress",
	})
	resp["output"] = output
	resp["output_text"] = ""
	resp["status"] = "requires_action"
	out, err := json.Marshal(resp)
	if err != nil {
		return responseBody
	}
	return out
}

func translateChatCompletionToResponsesResponseWithWorkflow(body []byte, applyPatchPathHint string, applyPatchContentHint string, applyPatchTypeHint string, workflowState ToolWorkflowState, verificationShellArgs string, originalReq map[string]any) ([]byte, error) {
	body = normalizeChatCompletionReasoningBoundary(body)
	appendProxyStageTrace("translation.chat_to_responses.start", map[string]any{
		"body_len":         len(body),
		"apply_patch_hint": strings.TrimSpace(applyPatchPathHint) != "",
	})
	message := gjson.GetBytes(body, "choices.0.message")
	if !message.Exists() || strings.TrimSpace(message.Raw) == "" || message.Type == gjson.Null {
		return nil, fmt.Errorf("chat completion missing choices[0].message")
	}

	id := strings.TrimSpace(gjson.GetBytes(body, "id").String())
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	rawContent := message.Get("content").String()
	rawReasoningText := message.Get("reasoning_content").String()
	text := rawContent
	reasoningText := rawReasoningText
	text, extractedReasoning := extractContentAndReasoning(text)
	text = stripLeadingReasoningDirective(text)
	if strings.TrimSpace(reasoningText) == "" {
		reasoningText = extractedReasoning
	}
	requestUserInputUnavailable := requestUserInputUnavailableInDefaultMode(originalReq)
	normalizedArtifacts, _ := normalizeQwenAssistantMessageFromBody(body, false, false)
	normalizedView := buildNormalizedArtifactView(normalizedArtifacts)
	if requestUserInputUnavailable && normalizedView.PreferredQuestion != nil {
		recoveredQuestionArgs := ""
		if parsedArgs, _ := normalizedView.PreferredQuestion.Payload["parsed_args"].(map[string]any); hasNonEmptyQuestionList(parsedArgs["questions"]) {
			recoveredQuestionArgs = mustJSONString(parsedArgs)
		}
		if strings.TrimSpace(recoveredQuestionArgs) == "" {
			recoveredQuestionArgs = strings.TrimSpace(cleanFallbackInput(normalizedView.PreferredQuestion.Payload["raw_args"], ""))
		}
		if strings.TrimSpace(recoveredQuestionArgs) == "" {
			if recovered, ok := recoverRequestUserInputArgumentsFromTextSources(reasoningText, text); ok {
				recoveredQuestionArgs = recovered
			}
		}
		if strings.TrimSpace(recoveredQuestionArgs) == "" {
			recoveredQuestionArgs = extractRequestUserInputArgumentsFromToolMarkup(reasoningText)
		}
		text = mergeAssistantQuestionText(text, renderRequestUserInputQuestionsAsPlainText(recoveredQuestionArgs))
	}
	output := make([]any, 0, 2)

	var appendCall func(callID string, name string, arguments string, index int)
	appendCall = func(callID string, name string, arguments string, index int) {
		name = canonicalizeResponsesOutputToolName(name)
		if requestUserInputUnavailable && strings.EqualFold(name, "request_user_input") {
			return
		}
		callID = strings.TrimSpace(callID)
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), index)
		}
		item := map[string]any{
			"id":      fmt.Sprintf("fc_%s", callID),
			"call_id": callID,
			"status":  "in_progress",
		}

		arguments = normalizePossiblyMixedToolArguments(arguments)
		switchName := strings.TrimPrefix(strings.TrimSpace(name), "__llamaswap_")
		switch switchName {
		case "shell":
			action := normalizeShellArgumentMapForResponse(parseToolArgsMapString(arguments))
			item["type"] = "function_call"
			item["name"] = "shell"
			item["arguments"] = mustJSONString(action)
		case "multi_tool_use.parallel":
			unpacked := unpackParallelToolUses(arguments)
			if len(unpacked) == 0 {
				item["type"] = "function_call"
				item["name"] = "multi_tool_use.parallel"
				item["arguments"] = mustJSONString(parseToolArgsMapString(arguments))
				result := validateBridgeToolCallItemResult(item)
				if !result.Valid {
					warning := result.Warning
					output = append(output, map[string]any{
						"id":   fmt.Sprintf("msg_%s_tool_validation_%d", id, index),
						"type": "message",
						"role": "assistant",
						"content": []any{
							map[string]any{
								"type": "output_text",
								"text": warning,
							},
						},
					})
					return
				}
				if result.NormalizedItem != nil {
					item = result.NormalizedItem
				}
				output = append(output, item)
				return
			}
			for nestedIdx, use := range unpacked {
				appendCall("", use.name, mustJSONString(use.args), index+nestedIdx+1)
			}
			return
		case "apply_patch":
			item["type"] = "apply_patch_call"
			item["status"] = "in_progress"
			args := parseToolArgsMapString(arguments)
			args = repairApplyPatchArgsOperationPathWithHint(args, applyPatchPathHint)
			operation := selectApplyPatchOperation(args)
			operation = enrichApplyPatchOperationWithHints(operation, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint)
			if repaired, repairedOK := repairApplyPatchOperationPathWithHint(operation, applyPatchPathHint); repairedOK {
				operation = repaired
			}
			if !applyPatchOperationPayloadValid(operation) {
				if recovered, ok := recoverApplyPatchOperationFromArgs(args); ok && applyPatchOperationPayloadValid(recovered) {
					operation = recovered
				}
			}
			operation = enrichApplyPatchOperationWithHints(operation, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint)
			if !applyPatchOperationPayloadValid(operation) && strings.TrimSpace(applyPatchPathHint) != "" {
				fallbackType := normalizeApplyPatchTypeHint(applyPatchTypeHint)
				if fallbackType == "" {
					fallbackType = "create_file"
				}
				fallback := map[string]any{
					"type": fallbackType,
					"path": normalizeApplyPatchPathForWorkspace(applyPatchPathHint),
				}
				if strings.TrimSpace(applyPatchContentHint) != "" {
					fallback["content"] = strings.TrimSpace(applyPatchContentHint)
				} else if patch := strings.TrimSpace(fmt.Sprintf("%v", args["patch"])); patch != "" && !looksLikeFilePathHint(patch) {
					fallback["content"] = patch
				} else if input := strings.TrimSpace(fmt.Sprintf("%v", args["input"])); input != "" && !looksLikeFilePathHint(input) {
					fallback["content"] = input
				}
				if applyPatchOperationPayloadValid(fallback) {
					operation = fallback
				}
			}
			operation = normalizeApplyPatchOperation(operation)
			if repaired, repairedOK := repairApplyPatchOperationPathWithHint(operation, applyPatchPathHint); repairedOK {
				operation = repaired
			}
			operation = repairApplyPatchOperationFromReasoningBlock(operation, text, reasoningText)
			if repaired, repairedOK := repairApplyPatchOperationPathWithHint(operation, applyPatchPathHint); repairedOK {
				operation = repaired
			}
			operation = preferContentDrivenApplyPatchOperation(operation)
			operation = normalizeApplyPatchOperation(operation)
			if !hasNonEmptyApplyPatchOperation(operation) {
				if extracted := extractApplyPatchFromFragmentedText(text); extracted != "" {
					operation = map[string]any{"patch": extracted}
				} else if extracted := extractApplyPatchFromFragmentedText(reasoningText); extracted != "" {
					operation = map[string]any{"patch": extracted}
				} else if extracted := extractApplyPatchFromFragmentedText(arguments); extracted != "" {
					operation = map[string]any{"patch": extracted}
				} else if looksLikePatchText(arguments) {
					operation = map[string]any{"patch": strings.TrimSpace(arguments)}
				}
				if !hasNonEmptyApplyPatchOperation(operation) {
					diagnosticParts := []string{
						fmt.Sprintf("name=%q", strings.TrimSpace(name)),
						fmt.Sprintf("normalized_args=%q", truncateBridgeDebugText(arguments, 180)),
						fmt.Sprintf("parsed_args=%q", truncateBridgeDebugText(mustJSONString(args), 180)),
						fmt.Sprintf("msg_patch=%t", extractApplyPatchBlock(text) != ""),
						fmt.Sprintf("reasoning_patch=%t", extractApplyPatchBlock(reasoningText) != ""),
						fmt.Sprintf("args_patch=%t", extractApplyPatchBlock(arguments) != ""),
					}
					item["_bridge_diagnostic"] = strings.Join(diagnosticParts, " ")
					appendApplyPatchTrace("translate.apply_patch_empty_after_recovery", map[string]any{
						"name":       strings.TrimSpace(name),
						"diagnostic": item["_bridge_diagnostic"],
					})
				}
			}
			item["operation"] = operation
			if input := strings.TrimSpace(buildApplyPatchInputFromOperation(operation)); input != "" {
				item["input"] = input
			}
			delete(item, "name")
			delete(item, "arguments")
		case "web_search", "web_search_preview":
			item["type"] = "web_search_call"
			item["action"] = parseToolArgsMapString(arguments)
		case "file_search":
			item["type"] = "file_search_call"
			item["action"] = parseToolArgsMapString(arguments)
		case "code_interpreter":
			item["type"] = "code_interpreter_call"
			item["action"] = parseToolArgsMapString(arguments)
		case "image_generation":
			item["type"] = "image_generation_call"
			item["action"] = parseToolArgsMapString(arguments)
		case "computer":
			item["type"] = "computer_call"
			item["action"] = parseToolArgsMapString(arguments)
		default:
			if server, toolName, ok := parseMCPToolName(strings.TrimSpace(name)); ok {
				// Keep MCP leaf tools as function_call with the fully qualified
				// namespace name. WSL codex exec 0.123.0 persists and continues
				// reliably with this shape, while raw mcp_tool_call items can be
				// dropped before they reach the local rollout/event surfaces.
				item["type"] = "function_call"
				item["name"] = buildMCPToolName(server, toolName)
				item["arguments"] = mustJSONString(parseToolArgsMapString(arguments))
			} else {
				item["type"] = "function_call"
				item["name"] = strings.TrimSpace(name)
				cleanArgs := parseToolArgsMapString(arguments)
				item["arguments"] = mustJSONString(cleanArgs)
			}
		}
		result := validateBridgeToolCallItemResult(item)
		if !result.Valid {
			warning := result.Warning
			sanitized := false
			switch strings.TrimSpace(fmt.Sprintf("%v", item["type"])) {
			case "apply_patch_call":
				item["type"] = "apply_patch_call"
				item["operation"] = normalizeApplyPatchOperation(item["operation"])
				delete(item, "name")
				delete(item, "arguments")
				sanitized = true
			case "shell_call":
				item["type"] = "function_call"
				item["name"] = "shell"
				item["arguments"] = mustJSONString(item["action"])
				sanitized = true
			}
			if sanitized {
				item["bridgevalidationwarning"] = warning
				item["_bridge_validation_warning"] = warning
				output = append(output, item)
				return
			}
			output = append(output, map[string]any{
				"id":   fmt.Sprintf("msg_%s_tool_validation_%d", id, index),
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": warning,
					},
				},
			})
			return
		}
		if result.NormalizedItem != nil {
			item = result.NormalizedItem
		}
		output = append(output, item)
	}

	toolCalls := message.Get("tool_calls")
	functionCall := message.Get("function_call")
	hasToolCalls := toolCalls.IsArray() && len(toolCalls.Array()) > 0
	hasFunctionCall := functionCall.Exists() && strings.TrimSpace(functionCall.Get("name").String()) != ""
	parsedToolCalls, cleanedText := proxyCompatibilityAdapters.ToolRepair.ParseAssistantOutput(model, text)
	parsedReasoningToolCalls := []ParsedToolCall{}
	cleanedReasoningText := ""
	if len(parsedToolCalls) == 0 && strings.TrimSpace(reasoningText) != "" {
		parsedReasoningToolCalls, cleanedReasoningText = proxyCompatibilityAdapters.ToolRepair.ParseAssistantOutput(model, reasoningText)
		allowReasoningRecovery := shouldAllowReasoningDerivedToolRecovery(reasoningText, parsedReasoningToolCalls)
		if !allowReasoningRecovery && nativeToolArgumentsEmptyForParsedReasoningCall(message, parsedReasoningToolCalls) {
			allowReasoningRecovery = true
		}
		if len(parsedReasoningToolCalls) > 0 && !allowReasoningRecovery {
			appendApplyPatchTrace("translate.reasoning_recovery_rejected", map[string]any{
				"model":          model,
				"reasoning_len":  len(strings.TrimSpace(reasoningText)),
				"parsed_calls":   len(parsedReasoningToolCalls),
				"reasoning_head": truncateBridgeDebugText(strings.TrimSpace(reasoningText), 220),
			})
			parsedReasoningToolCalls = nil
			cleanedReasoningText = ""
		}
	}
	if requestUserInputUnavailable {
		recoveredQuestionArgs := ""
		for _, call := range parsedToolCalls {
			if strings.EqualFold(canonicalizeResponsesOutputToolName(call.Name), "request_user_input") {
				recoveredQuestionArgs = extractRequestUserInputArguments(mustJSONString(call.Arguments))
				if recoveredQuestionArgs != "" {
					break
				}
			}
		}
		if recoveredQuestionArgs == "" {
			for _, call := range parsedReasoningToolCalls {
				if strings.EqualFold(canonicalizeResponsesOutputToolName(call.Name), "request_user_input") {
					recoveredQuestionArgs = extractRequestUserInputArguments(mustJSONString(call.Arguments))
					if recoveredQuestionArgs != "" {
						break
					}
				}
			}
		}
		if recoveredQuestionArgs == "" && normalizedView.PreferredQuestion != nil {
			if parsedArgs, _ := normalizedView.PreferredQuestion.Payload["parsed_args"].(map[string]any); hasNonEmptyQuestionList(parsedArgs["questions"]) {
				recoveredQuestionArgs = mustJSONString(parsedArgs)
			}
			if strings.TrimSpace(recoveredQuestionArgs) == "" {
				recoveredQuestionArgs = extractRequestUserInputArguments(strings.TrimSpace(cleanFallbackInput(normalizedView.PreferredQuestion.Payload["raw_args"], "")))
			}
		}
		if recoveredQuestionArgs == "" {
			if recovered, ok := recoverRequestUserInputArgumentsFromTextSources(reasoningText, text); ok {
				recoveredQuestionArgs = extractRequestUserInputArguments(recovered)
			}
		}
		if recoveredQuestionArgs == "" {
			recoveredQuestionArgs = extractRequestUserInputArgumentsFromToolMarkup(reasoningText)
		}
		text = mergeAssistantQuestionText(text, renderRequestUserInputQuestionsAsPlainText(recoveredQuestionArgs))
	}
	appendApplyPatchTrace("translate.tool_parse_summary", map[string]any{
		"model":                  model,
		"has_tool_calls":         hasToolCalls,
		"has_function_call":      hasFunctionCall,
		"parsed_text_calls":      len(parsedToolCalls),
		"parsed_reasoning_calls": len(parsedReasoningToolCalls),
		"text_len":               len(strings.TrimSpace(text)),
		"reasoning_len":          len(strings.TrimSpace(reasoningText)),
	})
	appendProxyStageTrace("repair.chat_to_responses.tool_parse", map[string]any{
		"model":                  model,
		"native_tool_calls":      hasToolCalls,
		"native_function_call":   hasFunctionCall,
		"parsed_text_calls":      len(parsedToolCalls),
		"parsed_reasoning_calls": len(parsedReasoningToolCalls),
	})
	if len(parsedToolCalls) > 0 {
		text = cleanedText
	}
	if len(parsedToolCalls) == 0 && strings.TrimSpace(cleanedText) != strings.TrimSpace(text) {
		originalVisiblePlan := extractVisibleProposedPlanBlock(text)
		cleanedVisiblePlan := extractVisibleProposedPlanBlock(cleanedText)
		if !(originalVisiblePlan != "" && cleanedVisiblePlan == "") {
			text = cleanedText
		}
	}
	if len(parsedReasoningToolCalls) > 0 {
		reasoningText = cleanedReasoningText
	}
	if len(parsedReasoningToolCalls) == 0 && strings.TrimSpace(cleanedReasoningText) != strings.TrimSpace(reasoningText) {
		reasoningText = cleanedReasoningText
	}
	text = strings.TrimSpace(stripLeadingReasoningDirective(text))
	reasoningText = strings.TrimSpace(stripLeadingReasoningDirective(reasoningText))
	if textLooksLikeEmptyToolArtifact(text) {
		text = ""
	}
	if textLooksLikeEmptyToolArtifact(reasoningText) {
		reasoningText = ""
	}
	if requestUserInputUnavailable {
		recoveredQuestionArgs := ""
		for _, call := range parsedToolCalls {
			if strings.EqualFold(canonicalizeResponsesOutputToolName(call.Name), "request_user_input") {
				recoveredQuestionArgs = extractRequestUserInputArguments(mustJSONString(call.Arguments))
				if recoveredQuestionArgs != "" {
					break
				}
			}
		}
		if recoveredQuestionArgs == "" {
			for _, call := range parsedReasoningToolCalls {
				if strings.EqualFold(canonicalizeResponsesOutputToolName(call.Name), "request_user_input") {
					recoveredQuestionArgs = extractRequestUserInputArguments(mustJSONString(call.Arguments))
					if recoveredQuestionArgs != "" {
						break
					}
				}
			}
		}
		if recoveredQuestionArgs == "" {
			recoveredQuestionArgs = extractRequestUserInputArgumentsFromToolMarkup(reasoningText)
		}
		text = mergeAssistantQuestionText(text, renderRequestUserInputQuestionsAsPlainText(recoveredQuestionArgs))
	}
	// Completion-style outputs may include a valid patch block in plain text
	// without emitting a native tool call. Synthesize an apply_patch tool call
	// so the Responses API client can execute it.
	if !hasToolCalls && !hasFunctionCall && len(parsedToolCalls) == 0 && len(parsedReasoningToolCalls) == 0 {
		if extracted := extractApplyPatchFromFragmentedText(text); extracted != "" {
			parsedToolCalls = []ParsedToolCall{
				{
					Name:      "apply_patch",
					Arguments: map[string]any{"input": extracted},
				},
			}
		} else if extracted := extractApplyPatchFromFragmentedText(reasoningText); extracted != "" {
			parsedReasoningToolCalls = []ParsedToolCall{
				{
					Name:      "apply_patch",
					Arguments: map[string]any{"input": extracted},
				},
			}
		} else if strings.TrimSpace(applyPatchPathHint) != "" &&
			strings.TrimSpace(applyPatchContentHint) != "" &&
			(strings.Contains(strings.ToLower(text), "apply_patch") || strings.Contains(strings.ToLower(reasoningText), "apply_patch")) {
			opType := normalizeApplyPatchTypeHint(applyPatchTypeHint)
			if opType == "" {
				opType = "update_file"
			}
			parsedReasoningToolCalls = []ParsedToolCall{
				{
					Name: "apply_patch",
					Arguments: map[string]any{
						"operation": map[string]any{
							"type":    opType,
							"path":    normalizeApplyPatchPathForWorkspace(applyPatchPathHint),
							"content": strings.TrimSpace(applyPatchContentHint),
						},
					},
				},
			}
		}
	}
	// If apply_patch intent is explicit and we still have no executable apply_patch
	// call candidate, synthesize one from request hints as a deterministic fallback.
	if !hasToolCalls && !hasFunctionCall &&
		!parsedCallListHasName(parsedToolCalls, "apply_patch") &&
		!parsedCallListHasName(parsedReasoningToolCalls, "apply_patch") &&
		strings.TrimSpace(applyPatchPathHint) != "" &&
		strings.TrimSpace(applyPatchContentHint) != "" {
		opType := normalizeApplyPatchTypeHint(applyPatchTypeHint)
		if opType == "" {
			opType = "update_file"
		}
		parsedToolCalls = append(parsedToolCalls, ParsedToolCall{
			Name: "apply_patch",
			Arguments: map[string]any{
				"operation": map[string]any{
					"type":    opType,
					"path":    normalizeApplyPatchPathForWorkspace(applyPatchPathHint),
					"content": strings.TrimSpace(applyPatchContentHint),
				},
			},
		})
	}

	// Some upstreams emit reasoning-only outputs (empty `content`, populated `reasoning_content`).
	// Codex/VScode clients currently prioritize `output_text`, so produce a non-empty assistant
	// message when no tool call exists and only reasoning was provided.
	if strings.TrimSpace(text) == "" && strings.TrimSpace(reasoningText) != "" &&
		!hasToolCalls && !hasFunctionCall && len(parsedToolCalls) == 0 && len(parsedReasoningToolCalls) == 0 {
		appendApplyPatchTrace("translate.promote_reasoning_to_text", map[string]any{
			"model": model,
		})
		text = proxyCompatibilityAdapters.Reasoning.BuildCommentaryPreview(reasoningText)
		if strings.TrimSpace(text) == "" {
			text = reasoningText
		}
		reasoningText = ""
	}
	if !hasToolCalls && !hasFunctionCall && len(parsedToolCalls) == 0 && len(parsedReasoningToolCalls) == 0 &&
		shouldRecoverVerificationShellCall(workflowState, verificationShellArgs, text, reasoningText) {
		parsedToolCalls = []ParsedToolCall{
			{
				Name:      "shell",
				Arguments: parseToolArgsMapString(verificationShellArgs),
			},
		}
		text = ""
		reasoningText = ""
	}
	if !hasToolCalls && !hasFunctionCall && len(parsedToolCalls) == 0 && len(parsedReasoningToolCalls) == 0 &&
		shouldRecoverCodexSkillProofShellCall(originalReq, verificationShellArgs, text, reasoningText) {
		parsedToolCalls = []ParsedToolCall{
			{
				Name:      "shell",
				Arguments: parseToolArgsMapString(verificationShellArgs),
			},
		}
		text = ""
		reasoningText = ""
	}
	parsedByName := map[string][]ParsedToolCall{}
	for _, call := range parsedToolCalls {
		key := strings.ToLower(strings.TrimSpace(call.Name))
		parsedByName[key] = append(parsedByName[key], call)
	}
	for _, call := range parsedReasoningToolCalls {
		key := strings.ToLower(strings.TrimSpace(call.Name))
		parsedByName[key] = append(parsedByName[key], call)
	}
	popParsedArgs := func(name string) (string, bool) {
		key := strings.ToLower(strings.TrimSpace(name))
		key = strings.TrimPrefix(key, "__llamaswap_")
		queue := parsedByName[key]
		if len(queue) == 0 {
			return "", false
		}
		call := queue[0]
		parsedByName[key] = queue[1:]
		return mustJSONString(call.Arguments), true
	}
	isLikelyEmptyToolArgs := func(args string) bool {
		normalized := strings.TrimSpace(normalizePossiblyMixedToolArguments(args))
		if normalized == "" || normalized == "{}" {
			return true
		}
		parsed := parseToolArgsMapString(normalized)
		return !hasAnyNonEmptyValue(parsed)
	}

	// Preserve any assistant text the upstream emitted, even when tool calls are present.
	// Codex clients can handle mixed message + tool call output, and keeping the message
	// improves user-visible progress in streaming UIs.
	if (hasToolCalls || hasFunctionCall || len(parsedToolCalls) > 0 || len(parsedReasoningToolCalls) > 0) &&
		shouldSuppressVisibleProgressMessageDuringToolPhase(text) {
		text = ""
	}
	if reasoningText != "" {
		output = append(output, map[string]any{
			"id":   fmt.Sprintf("rs_%s_0", id),
			"type": "reasoning",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": reasoningText,
				},
			},
		})
	}
	if strings.TrimSpace(text) != "" {
		channel := "final"
		if hasToolCalls || hasFunctionCall || len(parsedToolCalls) > 0 || len(parsedReasoningToolCalls) > 0 {
			channel = "commentary"
		}
		output = append(output, map[string]any{
			"id":      "msg_" + id,
			"type":    "message",
			"role":    "assistant",
			"channel": channel,
			"content": []any{
				map[string]any{
					"type": "output_text",
					"text": text,
				},
			},
		})
	}

	if hasToolCalls {
		idx := 0
		toolCalls.ForEach(func(_, tc gjson.Result) bool {
			name := tc.Get("function.name").String()
			args := tc.Get("function.arguments").String()
			if strings.EqualFold(strings.TrimSpace(name), "request_user_input") && isLikelyEmptyToolArgs(args) {
				if recovered, ok := recoverRequestUserInputArgumentsFromTextSources(reasoningText, text); ok {
					args = recovered
				}
			}
			if isLikelyEmptyToolArgs(args) {
				if recovered, ok := popParsedArgs(name); ok && !isLikelyEmptyToolArgs(recovered) {
					args = recovered
				}
			}
			appendCall(tc.Get("id").String(), name, args, idx)
			idx++
			return true
		})
	} else if hasFunctionCall {
		name := functionCall.Get("name").String()
		args := functionCall.Get("arguments").String()
		if strings.EqualFold(strings.TrimSpace(name), "request_user_input") && isLikelyEmptyToolArgs(args) {
			if recovered, ok := recoverRequestUserInputArgumentsFromTextSources(reasoningText, text); ok {
				args = recovered
			}
		}
		if isLikelyEmptyToolArgs(args) {
			if recovered, ok := popParsedArgs(name); ok && !isLikelyEmptyToolArgs(recovered) {
				args = recovered
			}
		}
		appendCall("", name, args, 0)
	} else if len(parsedToolCalls) > 0 || len(parsedReasoningToolCalls) > 0 {
		fallbackCalls := parsedToolCalls
		if len(fallbackCalls) == 0 {
			fallbackCalls = parsedReasoningToolCalls
		}
		for idx, call := range fallbackCalls {
			appendCall(call.CallID, call.Name, mustJSONString(call.Arguments), idx)
		}
	}

	resp := map[string]any{
		"id":          "resp_" + id,
		"object":      "response",
		"created_at":  time.Now().Unix(),
		"status":      "completed",
		"model":       model,
		"output":      output,
		"output_text": stripLeadingReasoningDirective(text),
	}
	if created := gjson.GetBytes(body, "created").Int(); created > 0 {
		resp["created_at"] = created
	}
	if usageMap := buildResponsesUsageFromChatUsage(gjson.GetBytes(body, "usage")); usageMap != nil {
		resp["usage"] = usageMap
	}
	normalizeTranslatedResponsesOutput(resp)
	appendApplyPatchTrace("translate.tool_response_summary", map[string]any{
		"model":           model,
		"output_items":    len(output),
		"response_status": strings.TrimSpace(fmt.Sprintf("%v", resp["status"])),
	})
	appendProxyStageTrace("translation.chat_to_responses.done", map[string]any{
		"model":           model,
		"output_items":    len(output),
		"response_status": strings.TrimSpace(fmt.Sprintf("%v", resp["status"])),
	})
	return json.Marshal(resp)
}

func normalizeTranslatedResponsesOutput(resp map[string]any) {
	if resp == nil {
		return
	}
	output, ok := resp["output"].([]any)
	if !ok || len(output) == 0 {
		if _, exists := resp["output"]; !exists {
			resp["output"] = []any{}
		}
		if _, exists := resp["output_text"]; !exists {
			resp["output_text"] = ""
		}
		return
	}

	respID := strings.TrimSpace(fmt.Sprintf("%v", resp["id"]))
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
		resp["id"] = respID
	}

	hasMessageText := false
	hasToolCall := false
	hasPendingToolCall := false
	normalizedOutput := make([]any, 0, len(output))
	for idx, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		if itemType == "" {
			continue
		}

		if strings.TrimSpace(fmt.Sprintf("%v", item["id"])) == "" {
			switch {
			case strings.HasSuffix(itemType, "_call") || itemType == "function_call":
				item["id"] = fmt.Sprintf("fc_%s_%d", respID, idx)
			case itemType == "reasoning":
				item["id"] = fmt.Sprintf("rs_%s_%d", respID, idx)
			default:
				item["id"] = fmt.Sprintf("msg_%s_%d", respID, idx)
			}
		}

		switch itemType {
		case "reasoning":
			summary, _ := item["summary"].([]any)
			normalizedSummary := make([]any, 0, len(summary))
			for _, summaryRaw := range summary {
				summaryItem, ok := summaryRaw.(map[string]any)
				if !ok {
					continue
				}
				summaryText := strings.TrimSpace(extractResponsesInputText(summaryItem))
				if summaryText == "" {
					continue
				}
				normalizedSummary = append(normalizedSummary, map[string]any{
					"type": "summary_text",
					"text": summaryText,
				})
			}
			if len(normalizedSummary) == 0 {
				summaryText := strings.TrimSpace(extractResponsesInputText(item["summary"]))
				if summaryText != "" {
					normalizedSummary = append(normalizedSummary, map[string]any{
						"type": "summary_text",
						"text": summaryText,
					})
				}
			}
			item["summary"] = normalizedSummary
		case "message":
			content, _ := item["content"].([]any)
			normalizedContent := make([]any, 0, len(content))
			for _, partRaw := range content {
				part, ok := partRaw.(map[string]any)
				if !ok {
					continue
				}
				partType := strings.TrimSpace(fmt.Sprintf("%v", part["type"]))
				if partType == "" {
					partType = "output_text"
				}
				partText := strings.TrimSpace(stripLeadingReasoningDirective(fmt.Sprintf("%v", part["text"])))
				if partText != "" {
					hasMessageText = true
				}
				normalizedContent = append(normalizedContent, map[string]any{
					"type": partType,
					"text": partText,
				})
			}
			item["content"] = normalizedContent
		case "function_call":
			if cleanFallbackInput(item["call_id"], "") == "" {
				item["call_id"] = fmt.Sprintf("call_%s_%d", respID, idx)
			}
			name := strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
			if name == "" {
				item["name"] = "__invalid_function"
				name = "__invalid_function"
			}
			args := normalizePossiblyMixedToolArguments(fmt.Sprintf("%v", item["arguments"]))
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			switch {
			case strings.EqualFold(name, "shell"):
				parsed := parseToolArgsMapString(args)
				args = mustJSONString(normalizeShellArgumentMapForResponse(parsed))
				item["arguments"] = args
				hasToolCall = true
				if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
					hasPendingToolCall = true
				}
			case func() bool {
				_, _, ok := parseMCPToolName(name)
				return ok
			}():
				server, toolName, _ := parseMCPToolName(name)
				item["type"] = "function_call"
				item["name"] = buildMCPToolName(server, toolName)
				item["arguments"] = mustJSONString(parseToolArgsMapString(args))
				if cleanFallbackInput(item["status"], "") == "" {
					item["status"] = "in_progress"
				}
				hasToolCall = true
				if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
					hasPendingToolCall = true
				}
			case strings.EqualFold(name, "apply_patch"), strings.EqualFold(name, llamaSwapApplyPatchFunctionName):
				parsed := parseToolArgsMapString(args)
				op := selectApplyPatchOperation(parsed)
				op = normalizeApplyPatchOperation(op)
				item["type"] = "apply_patch_call"
				if !hasNonEmptyApplyPatchOperation(op) || !applyPatchOperationPayloadValid(op) {
					item["operation"] = op
					delete(item, "name")
					delete(item, "arguments")
					item["bridgevalidationwarning"] = applyPatchValidationWarningPrefix + " operation was invalid; emitting sanitized apply_patch_call payload."
					item["_bridge_validation_warning"] = item["bridgevalidationwarning"]
					hasToolCall = true
					if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
						hasPendingToolCall = true
					}
					break
				}
				item["operation"] = op
				delete(item, "name")
				delete(item, "arguments")
				hasToolCall = true
				if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
					hasPendingToolCall = true
				}
			default:
				item["arguments"] = args
				hasToolCall = true
				if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
					hasPendingToolCall = true
				}
			}
		case "apply_patch_call":
			op := normalizeApplyPatchOperation(item["operation"])
			if !hasNonEmptyApplyPatchOperation(op) || !applyPatchOperationPayloadValid(op) {
				item["type"] = "apply_patch_call"
				item["operation"] = op
				delete(item, "name")
				delete(item, "arguments")
				item["bridgevalidationwarning"] = applyPatchValidationWarningPrefix + " operation was invalid; emitting sanitized apply_patch_call payload."
				item["_bridge_validation_warning"] = item["bridgevalidationwarning"]
				hasToolCall = true
				if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
					hasPendingToolCall = true
				}
				break
			}
			hasToolCall = true
			if strings.TrimSpace(fmt.Sprintf("%v", item["call_id"])) == "" {
				item["call_id"] = fmt.Sprintf("call_%s_%d", respID, idx)
			}
			item["operation"] = op
			if input := strings.TrimSpace(buildApplyPatchInputFromOperation(op)); input != "" {
				item["input"] = input
			}
			if strings.TrimSpace(fmt.Sprintf("%v", item["status"])) == "" {
				item["status"] = "in_progress"
			}
			if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
				hasPendingToolCall = true
			}
		case "mcp_tool_call":
			hasToolCall = true
			if cleanFallbackInput(item["call_id"], "") == "" {
				item["call_id"] = fmt.Sprintf("call_%s_%d", respID, idx)
			}
			server := cleanFallbackInput(item["server"], "")
			toolName := cleanFallbackInput(item["tool"], "")
			if server == "" || toolName == "" {
				if parsedServer, parsedTool, ok := parseMCPToolName(cleanFallbackInput(item["name"], "")); ok {
					server = parsedServer
					toolName = parsedTool
				}
			}
			item["type"] = "function_call"
			item["name"] = buildMCPToolName(server, toolName)
			item["arguments"] = mustJSONString(parseToolArgsMapString(mustJSONString(normalizeMapValue(item["arguments"]))))
			delete(item, "server")
			delete(item, "tool")
			if cleanFallbackInput(item["status"], "") == "" {
				item["status"] = "in_progress"
			}
			if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
				hasPendingToolCall = true
			}
		case "shell_call", "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "computer_call":
			hasToolCall = true
			if strings.TrimSpace(fmt.Sprintf("%v", item["call_id"])) == "" {
				item["call_id"] = fmt.Sprintf("call_%s_%d", respID, idx)
			}
			name := strings.TrimSuffix(itemType, "_call")
			if itemType == "shell_call" {
				item["type"] = "function_call"
				item["name"] = "shell"
				if actionMap, ok := item["action"].(map[string]any); ok {
					item["arguments"] = mustJSONString(normalizeShellArgumentMapForResponse(actionMap))
				} else {
					item["arguments"] = mustJSONString(map[string]any{})
				}
			} else {
				item["name"] = name
				item["arguments"] = mustJSONString(item["action"])
			}
			if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["status"])), "completed") {
				hasPendingToolCall = true
			}
		}
		output[idx] = item
		normalizedOutput = append(normalizedOutput, item)
	}
	resp["output"] = normalizedOutput
	if hasPendingToolCall {
		resp["status"] = "requires_action"
	} else if hasToolCall {
		resp["status"] = "completed"
	}
	if hasToolCall && !hasMessageText {
		resp["output_text"] = ""
		return
	}
	if !hasMessageText {
		resp["output_text"] = ""
		return
	}
	resp["output_text"] = strings.TrimSpace(stripLeadingReasoningDirective(fmt.Sprintf("%v", resp["output_text"])))
}

func enforceExactFinalReplyHint(responseBody []byte, hint string, force bool) []byte {
	hint = strings.TrimSpace(hint)
	if hint == "" || !gjson.ValidBytes(responseBody) || responseContainsToolCall(responseBody) {
		return responseBody
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}
	output, _ := resp["output"].([]any)
	found := false
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok || strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "message" {
			continue
		}
		content, _ := item["content"].([]any)
		for _, partRaw := range content {
			part, ok := partRaw.(map[string]any)
			if !ok {
				continue
			}
			text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
			if strings.Contains(text, hint) {
				found = true
				break
			}
		}
		if found {
			break
		}
	}
	if !found && !force {
		return responseBody
	}
	respID := strings.TrimSpace(fmt.Sprintf("%v", resp["id"]))
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
		resp["id"] = respID
	}
	resp["output"] = []any{
		map[string]any{
			"id":   fmt.Sprintf("msg_%s_exact_reply", respID),
			"type": "message",
			"role": "assistant",
			"content": []any{
				map[string]any{
					"type": "output_text",
					"text": hint,
				},
			},
		},
	}
	resp["output_text"] = hint
	resp["status"] = "completed"
	out, err := json.Marshal(resp)
	if err != nil {
		return responseBody
	}
	return out
}

func collapsePreToolAssistantMessages(output []any) []any {
	if len(output) < 3 {
		return output
	}
	firstToolIndex := -1
	for idx, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		if itemType == "function_call" ||
			itemType == "shell_call" ||
			itemType == "apply_patch_call" ||
			itemType == "mcp_tool_call" ||
			itemType == "custom_tool_call" ||
			itemType == "web_search_call" ||
			itemType == "file_search_call" ||
			itemType == "code_interpreter_call" ||
			itemType == "image_generation_call" ||
			itemType == "computer_call" {
			firstToolIndex = idx
			break
		}
	}
	if firstToolIndex <= 1 {
		return output
	}
	seenAssistantMessage := false
	collapsed := make([]any, 0, len(output))
	for idx, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			collapsed = append(collapsed, raw)
			continue
		}
		if idx < firstToolIndex &&
			strings.TrimSpace(fmt.Sprintf("%v", item["type"])) == "message" &&
			strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["role"])), "assistant") {
			if seenAssistantMessage {
				continue
			}
			seenAssistantMessage = true
		}
		collapsed = append(collapsed, raw)
	}
	return collapsed
}

func writeResponsesStream(w http.ResponseWriter, responseJSON []byte, requestedReasoningSummary string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		_, _ = w.Write(responseJSON)
		return
	}

	needsContinuation := responseNeedsToolContinuation(responseJSON)
	if needsContinuation {
		// Even when the response contains a tool call and requires a follow-up turn,
		// the Responses SSE stream must still end with a `response.completed` event.
		// Codex CLI treats streams that end at `[DONE]` without `response.completed`
		// as a disconnect and will repeatedly reconnect.
		responseJSON = forceResponseStatus(responseJSON, "requires_action")
	}

	sequence := 0
	writeEvent := func(eventType string, payload map[string]any) {
		if _, ok := payload["type"]; !ok {
			payload["type"] = eventType
		}
		payload["sequence_number"] = sequence
		sequence++
		data, _ := json.Marshal(payload)
		_, _ = w.Write([]byte("event: " + eventType + "\n"))
		_, _ = w.Write([]byte("data: " + string(data) + "\n\n"))
		flusher.Flush()
	}
	respID := strings.TrimSpace(gjson.GetBytes(responseJSON, "id").String())
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
	}
	writeReasoningSummaryPartAdded := func(itemID string, outputIndex int) {
		writeEvent("response.reasoning_summary_part.added", map[string]any{
			"response_id":   respID,
			"item_id":       itemID,
			"output_index":  outputIndex,
			"summary_index": 0,
			"part": map[string]any{
				"type": "summary_text",
				"text": "",
			},
		})
	}
	writeReasoningSummaryPartDone := func(itemID string, outputIndex int, text string) {
		writeEvent("response.reasoning_summary_part.done", map[string]any{
			"response_id":   respID,
			"item_id":       itemID,
			"output_index":  outputIndex,
			"summary_index": 0,
			"part": map[string]any{
				"type": "summary_text",
				"text": text,
			},
		})
	}
	createdAt := gjson.GetBytes(responseJSON, "created_at").Int()
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	model := strings.TrimSpace(gjson.GetBytes(responseJSON, "model").String())
	responseSkeleton := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": createdAt,
		"model":      model,
		"status":     "in_progress",
		"output":     []any{},
	}
	if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
		responseSkeleton["reasoning"] = map[string]any{"summary": summary}
	}

	writeEvent("response.created", map[string]any{"response": responseSkeleton})
	writeEvent("response.in_progress", map[string]any{"response": responseSkeleton})

	var full map[string]any
	_ = json.Unmarshal(responseJSON, &full)
	if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
		reasoning, _ := full["reasoning"].(map[string]any)
		if reasoning == nil {
			reasoning = map[string]any{}
		} else {
			reasoning = cloneMap(reasoning)
		}
		reasoning["summary"] = summary
		full["reasoning"] = reasoning
	}
	if needsContinuation {
		full["status"] = "requires_action"
	}
	buildToolValidationMessageItem := func(outputIndex int, warning string) map[string]any {
		return map[string]any{
			"id":   fmt.Sprintf("msg_%s_tool_validation_%d", respID, outputIndex),
			"type": "message",
			"role": "assistant",
			"content": []any{
				map[string]any{
					"type": "output_text",
					"text": warning,
				},
			},
		}
	}
	output := gjson.GetBytes(responseJSON, "output").Array()
	normalizedOutput := make([]any, 0, len(output))
	hasValidToolCall := false
	streamAdapter := proxyCompatibilityAdapters.Stream
	emitReasoningCommentary := true
	toolProgressCommentary := ""
	commentaryAlreadyPresent := false
	for _, itemResult := range output {
		item := map[string]any{}
		if err := json.Unmarshal([]byte(itemResult.Raw), &item); err != nil {
			continue
		}
		item = streamAdapter.NormalizeResponseOutputItem(item)
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		if itemType == "message" && strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["channel"])), "commentary") {
			commentaryAlreadyPresent = true
		}
		if toolView, isToolItem := streamAdapter.BuildResponseToolItemView(item); isToolItem {
			if ok, _ := validateBridgeToolCallItem(item); ok && strings.TrimSpace(toolView.ToolName) != "" {
				emitReasoningCommentary = false
				if toolProgressCommentary == "" {
					toolProgressCommentary = replayToolProgressCommentary(toolView.ToolName)
				}
				break
			}
		}
	}
	emitCommentaryReplayItem := func(outputIndex int, commentaryText string) map[string]any {
		commentaryItemID := fmt.Sprintf("msg_%s_%d", respID, outputIndex)
		commentaryItem := map[string]any{
			"id":      commentaryItemID,
			"type":    "message",
			"role":    "assistant",
			"channel": "commentary",
			"content": []any{
				map[string]any{"type": "output_text", "text": commentaryText},
			},
		}
		writeEvent("response.output_item.added", map[string]any{
			"response_id":  respID,
			"output_index": outputIndex,
			"item": map[string]any{
				"id":      commentaryItemID,
				"type":    "message",
				"role":    "assistant",
				"channel": "commentary",
				"content": []any{
					map[string]any{"type": "output_text", "text": ""},
				},
			},
		})
		writeEvent("response.content_part.added", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryItemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": ""},
		})
		writeEvent("response.output_text.delta", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryItemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"delta":         commentaryText,
		})
		writeEvent("response.output_text.done", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryItemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"text":          commentaryText,
		})
		writeEvent("response.content_part.done", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryItemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": commentaryText},
		})
		writeEvent("response.output_item.done", map[string]any{
			"response_id":  respID,
			"output_index": outputIndex,
			"item":         commentaryItem,
		})
		return commentaryItem
	}
	for _, itemResult := range output {
		item := map[string]any{}
		if err := json.Unmarshal([]byte(itemResult.Raw), &item); err != nil {
			continue
		}
		item = streamAdapter.NormalizeResponseOutputItem(item)
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		toolView, isToolItem := streamAdapter.BuildResponseToolItemView(item)
		if isToolItem {
			result := validateBridgeToolCallItemResult(item)
			if !result.Valid {
				warning := result.Warning
				item = buildToolValidationMessageItem(len(normalizedOutput), warning)
				toolView = ResponseToolItemView{}
				isToolItem = false
			} else if result.NormalizedItem != nil {
				item = result.NormalizedItem
				itemType = strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
				toolView, _ = streamAdapter.BuildResponseToolItemView(item)
			}
		}
		addedItem := item
		itemType = strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		if isToolItem {
			if !commentaryAlreadyPresent && strings.TrimSpace(toolProgressCommentary) != "" {
				commentaryItem := emitCommentaryReplayItem(len(normalizedOutput), toolProgressCommentary)
				normalizedOutput = append(normalizedOutput, commentaryItem)
				commentaryAlreadyPresent = true
			}
			hasValidToolCall = true
			if _, ok := addedItem["id"]; !ok {
				addedItem = cloneMap(item)
			}
			addedItem["status"] = "in_progress"
		} else if itemType == "reasoning" {
			if _, ok := addedItem["id"]; !ok {
				addedItem = cloneMap(item)
			}
			addedItem["status"] = "in_progress"
		}
		outputIndex := len(normalizedOutput)
		writeEvent("response.output_item.added", map[string]any{
			"response_id":  respID,
			"output_index": outputIndex,
			"item":         addedItem,
		})
		if isToolItem {
			for _, event := range streamAdapter.BuildResponseToolArgumentEvents(toolView, respID, outputIndex) {
				switch event.Kind {
				case StreamEventToolArgsDelta:
					writeEvent("response.function_call_arguments.delta", map[string]any{
						"response_id":  respID,
						"item_id":      event.ItemID,
						"output_index": event.OutputIndex,
						"delta":        event.Payload["delta"],
					})
				case StreamEventToolArgsDone:
					writeEvent("response.function_call_arguments.done", map[string]any{
						"response_id":  respID,
						"item_id":      event.ItemID,
						"output_index": event.OutputIndex,
						"name":         event.ToolName,
						"call_id":      event.CallID,
						"arguments":    event.Payload["arguments"],
					})
				}
			}
		}
		if itemType == "message" {
			if content, ok := item["content"].([]any); ok {
				for contentIndex, rawPart := range content {
					part, ok := rawPart.(map[string]any)
					if !ok {
						continue
					}
					writeEvent("response.content_part.added", map[string]any{
						"response_id":   respID,
						"item_id":       item["id"],
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"part":          map[string]any{"type": part["type"], "text": ""},
					})
					writeEvent("response.output_text.delta", map[string]any{
						"response_id":   respID,
						"item_id":       item["id"],
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"delta":         fmt.Sprintf("%v", part["text"]),
					})
					writeEvent("response.output_text.done", map[string]any{
						"response_id":   respID,
						"item_id":       item["id"],
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"text":          fmt.Sprintf("%v", part["text"]),
					})
					writeEvent("response.content_part.done", map[string]any{
						"response_id":   respID,
						"item_id":       item["id"],
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"part":          part,
					})
				}
			}
		}
		if itemType == "reasoning" {
			reasoningText := strings.TrimSpace(extractResponsesInputText(item["summary"]))
			if reasoningText != "" {
				writeReasoningSummaryPartAdded(fmt.Sprintf("%v", item["id"]), outputIndex)
				writeEvent("response.reasoning_summary_text.delta", map[string]any{
					"response_id":   respID,
					"item_id":       item["id"],
					"output_index":  outputIndex,
					"summary_index": 0,
					"delta":         reasoningText,
				})
				writeEvent("response.reasoning_summary_text.done", map[string]any{
					"response_id":   respID,
					"item_id":       item["id"],
					"output_index":  outputIndex,
					"summary_index": 0,
					"text":          reasoningText,
				})
				writeReasoningSummaryPartDone(fmt.Sprintf("%v", item["id"]), outputIndex, reasoningText)
			}
		}
		doneItem := item
		if isToolItem {
			if _, ok := doneItem["id"]; !ok {
				doneItem = cloneMap(item)
			}
			if !needsContinuation {
				doneItem["status"] = "completed"
			} else {
				doneItem["status"] = "in_progress"
			}
		} else if itemType == "reasoning" {
			if _, ok := doneItem["id"]; !ok {
				doneItem = cloneMap(item)
			}
			doneItem["status"] = "completed"
			if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
				doneItem["summary_mode"] = summary
			}
		}
		writeEvent("response.output_item.done", map[string]any{
			"response_id":  respID,
			"output_index": outputIndex,
			"item":         doneItem,
		})
		normalizedOutput = append(normalizedOutput, doneItem)
		if itemType == "reasoning" && emitReasoningCommentary {
			reasoningText := strings.TrimSpace(extractResponsesInputText(item["summary"]))
			commentaryText := proxyCompatibilityAdapters.Reasoning.BuildCommentaryPreview(reasoningText)
			if commentaryText != "" {
				commentaryItem := emitCommentaryReplayItem(len(normalizedOutput), commentaryText)
				normalizedOutput = append(normalizedOutput, commentaryItem)
				commentaryAlreadyPresent = true
			}
		}
	}
	if len(normalizedOutput) > 0 {
		normalizedOutput = collapsePreToolAssistantMessages(normalizedOutput)
		full["output"] = normalizedOutput
	}
	if needsContinuation && !hasValidToolCall {
		full["status"] = "completed"
	}

	// Always finish the SSE stream with a `response.completed` event. If a tool
	// continuation is required, `response.status` will be `requires_action`.
	if event, ok := streamAdapter.BuildResponseCompletedEvent(full); ok {
		writeCanonicalResponseStreamEvent(writeEvent, event)
	} else {
		writeEvent("response.completed", map[string]any{"response": full})
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

func extractInvalidToolArgFeedback(responseBody []byte) (string, bool) {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		warning := strings.TrimSpace(item.Get("bridgevalidationwarning").String())
		if strings.Contains(strings.ToLower(warning), strings.ToLower(applyPatchValidationWarningPrefix)) {
			return warning, true
		}
		if strings.TrimSpace(item.Get("type").String()) != "message" {
			continue
		}
		content := item.Get("content").Array()
		for _, part := range content {
			text := strings.TrimSpace(part.Get("text").String())
			lower := strings.ToLower(text)
			if strings.Contains(lower, strings.ToLower(applyPatchValidationWarningPrefix)) ||
				strings.Contains(lower, "apply_patch call was not executed because") {
				return text, true
			}
		}
	}
	return "", false
}

func requestIncludesApplyPatchTool(req map[string]any) bool {
	if req == nil {
		return false
	}
	tools, _ := req["tools"].([]any)
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		toolType := strings.TrimSpace(fmt.Sprintf("%v", tool["type"]))
		if strings.EqualFold(toolType, "apply_patch") {
			return true
		}
		name := strings.TrimSpace(fmt.Sprintf("%v", tool["name"]))
		if name == "" {
			if fn, ok := tool["function"].(map[string]any); ok {
				name = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
			}
		}
		if strings.EqualFold(name, "apply_patch") || strings.EqualFold(name, llamaSwapApplyPatchFunctionName) {
			return true
		}
	}
	return false
}

func canonicalStreamEventTypeName(kind StreamEventKind) string {
	switch kind {
	case StreamEventToolItemAdded:
		return "response.output_item.added"
	case StreamEventToolArgsDelta:
		return "response.function_call_arguments.delta"
	case StreamEventToolArgsDone:
		return "response.function_call_arguments.done"
	case StreamEventToolOutputReturned:
		return "response.output_item.done"
	case StreamEventResponseCompleted:
		return "response.completed"
	default:
		return ""
	}
}

func writeCanonicalResponseStreamEvent(writeEvent func(string, map[string]any), event StreamEvent) {
	eventType := canonicalStreamEventTypeName(event.Kind)
	if eventType == "" {
		return
	}
	payload := map[string]any{}
	if event.Kind != StreamEventResponseCompleted && event.ResponseID != "" {
		payload["response_id"] = event.ResponseID
	}
	if event.ItemID != "" {
		payload["item_id"] = event.ItemID
	}
	if event.OutputIndex >= 0 {
		payload["output_index"] = event.OutputIndex
	}
	if event.ToolName != "" && event.Kind == StreamEventToolArgsDone {
		payload["name"] = event.ToolName
	}
	if event.CallID != "" && event.Kind == StreamEventToolArgsDone {
		payload["call_id"] = event.CallID
	}
	for k, v := range event.Payload {
		payload[k] = v
	}
	writeEvent(eventType, payload)
}

func toolDefinitionTargetsNormalizedName(tool map[string]any, want string) bool {
	if tool == nil {
		return false
	}
	want = normalizeToolTierName(want)
	if want == "" {
		return false
	}
	if normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", tool["type"]))) == want {
		return true
	}
	if normalizeToolTierName(extractFunctionToolName(tool)) == want {
		return true
	}
	return false
}

func toolChoiceTargetsNormalizedName(raw any, want string) bool {
	want = normalizeToolTierName(want)
	if want == "" {
		return false
	}
	normalized := normalizeBridgeToolChoice(raw)
	switch typed := normalized.(type) {
	case string:
		return normalizeToolTierName(strings.TrimSpace(typed)) == want
	case map[string]any:
		if normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", typed["type"]))) == want {
			return true
		}
		if fn, ok := typed["function"].(map[string]any); ok {
			return normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))) == want
		}
	}
	return false
}

func requestHasOnlyApplyPatchTools(req map[string]any) bool {
	if req == nil {
		return false
	}
	tools, _ := req["tools"].([]any)
	if len(tools) == 0 {
		return false
	}
	sawApplyPatch := false
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			return false
		}
		if !toolDefinitionTargetsNormalizedName(tool, "apply_patch") {
			return false
		}
		sawApplyPatch = true
	}
	return sawApplyPatch
}

func requestHasExplicitApplyPatchToolChoice(req map[string]any) bool {
	if req == nil {
		return false
	}
	raw, ok := req["tool_choice"]
	if !ok || raw == nil {
		return false
	}
	return toolChoiceTargetsNormalizedName(raw, "apply_patch")
}

func shouldEnableStrictApplyPatchIntent(req map[string]any, requestBody []byte) bool {
	if req == nil {
		return false
	}
	if responsesRequestContractMode(req) == "plan" || requestExplicitlyWantsReturnedPlan(req) || requestExplicitlyWantsNoMutationPlan(req) {
		return false
	}
	// Post-tool continuation turns should not be forced back into initial
	// apply_patch recovery mode. At that point the model must be free to either
	// finish or choose another tool without inherited patch-only steering.
	if requestMapContainsAnyToolOutput(req) || requestContainsAnyToolOutput(requestBody) {
		return false
	}
	if requestHasExplicitApplyPatchToolChoice(req) {
		return true
	}
	if requestHasOnlyApplyPatchTools(req) {
		return true
	}
	if requestMentionsAgentOrchestration(req) || requestNeedsCodexSkillFilesystemProof(req) {
		return false
	}
	lowerInput := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if requestMentionsPrePatchNonPatchTool(req) {
		return false
	}
	if requestInputExplicitlyForbidsApplyPatch(lowerInput) {
		return false
	}
	if requestLooksLikeApplyPatchPolicyInstruction(lowerInput) {
		return false
	}
	if requestInputMentionsApplyPatch(req) {
		return true
	}
	// Last resort: if request body contains an explicit apply_patch request phrase
	// from user-authored input text, enable strict mode. Do not scan the full
	// request body or top-level instructions here; those contain tool metadata
	// and model instructions that mention apply_patch on every turn.
	if strings.Contains(lowerInput, "use apply_patch") || strings.Contains(lowerInput, "retry with apply_patch") {
		return true
	}
	_ = requestBody
	return false
}

func requestLooksLikeApplyPatchPolicyInstruction(input string) bool {
	input = strings.ToLower(strings.TrimSpace(input))
	if input == "" {
		return false
	}
	hasPolicyLead := strings.Contains(input, "apply_patch") &&
		strings.Contains(input, "fails") &&
		(strings.Contains(input, "if ") || strings.Contains(input, "when "))
	if !hasPolicyLead {
		return false
	}
	return strings.Contains(input, "file drift") ||
		strings.Contains(input, "target file changed") ||
		strings.Contains(input, "re-read the current file") ||
		strings.Contains(input, "inspect the current file state") ||
		strings.Contains(input, "reconcile the patch") ||
		strings.Contains(input, "recover safely") ||
		strings.Contains(input, "latest contents") ||
		strings.Contains(input, "written and verified") ||
		strings.Contains(input, "verify the final file state") ||
		strings.Contains(input, "before continuing")
}

func requestLooksLikeApplyPatchPolicyInstructionAny(inputs ...string) bool {
	for _, input := range inputs {
		if requestLooksLikeApplyPatchPolicyInstruction(input) {
			return true
		}
	}
	return false
}

func requestExplicitlyWantsNativeCodexQuestion(req map[string]any) bool {
	if req == nil {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(
		extractResponsesNativeQuestionIntentText(req),
	))
	if lowerText == "" {
		return false
	}
	if strings.Contains(lowerText, "do not ask") || strings.Contains(lowerText, "no questions") {
		return false
	}
	wantsQuestion := strings.Contains(lowerText, "ask exactly one short clarifying question") ||
		strings.Contains(lowerText, "ask one short clarifying question") ||
		strings.Contains(lowerText, "ask a short clarifying question") ||
		strings.Contains(lowerText, "ask exactly one short question") ||
		strings.Contains(lowerText, "ask exactly one native codex question") ||
		strings.Contains(lowerText, "ask me exactly one native codex question") ||
		strings.Contains(lowerText, "ask one native codex question") ||
		strings.Contains(lowerText, "ask the user how they would like") ||
		strings.Contains(lowerText, "ask how they would like") ||
		strings.Contains(lowerText, "ask how the user would like") ||
		strings.Contains(lowerText, "ask which approach they prefer") ||
		strings.Contains(lowerText, "ask which option they prefer") ||
		strings.Contains(lowerText, "ask which build approach they prefer")
	wantsNativeFormat := strings.Contains(lowerText, "native codex question format") ||
		strings.Contains(lowerText, "native codex question") ||
		strings.Contains(lowerText, "request_user_input") ||
		strings.Contains(lowerText, "codex question dialog") ||
		strings.Contains(lowerText, "native question")
	return wantsQuestion && wantsNativeFormat
}

func extractResponsesNativeQuestionIntentText(req map[string]any) string {
	if req == nil {
		return ""
	}
	return strings.TrimSpace(extractResponsesUserInputText(req))
}

func requestExplicitlyWantsReturnedPlan(req map[string]any) bool {
	if req == nil {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return false
	}
	hasPlanIntent := strings.HasPrefix(lowerText, "plan ") ||
		strings.Contains(lowerText, "write a plan") ||
		strings.Contains(lowerText, "return a plan") ||
		strings.Contains(lowerText, "finalize the plan") ||
		strings.Contains(lowerText, "finalizing the plan") ||
		strings.Contains(lowerText, "<proposed_plan>")
	if !hasPlanIntent {
		return false
	}
	return !requestExplicitlyWantsImplementationRetry(req, ToolWorkflowState{})
}

func requestHistoryContainsApplyPatchValidationFeedback(req map[string]any) bool {
	if req == nil {
		return false
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "message" {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["role"])) != "assistant" {
			continue
		}
		for _, text := range extractMessageTextParts(item["content"]) {
			lower := strings.ToLower(strings.TrimSpace(text))
			if strings.Contains(lower, strings.ToLower(applyPatchValidationWarningPrefix)) ||
				strings.Contains(lower, "apply_patch call was not executed because") {
				return true
			}
		}
	}
	return false
}

func requestExplicitlyWantsImplementationRetry(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return false
	}
	if strings.Contains(lowerText, "please implement this plan") ||
		strings.Contains(lowerText, "implement this plan") ||
		strings.Contains(lowerText, "start implementing") ||
		strings.Contains(lowerText, "using apply_patch") ||
		strings.Contains(lowerText, "do not use shell for writing") ||
		strings.Contains(lowerText, "create the complete quiz file") ||
		strings.Contains(lowerText, "build this out") {
		return true
	}
	if !(strings.Contains(lowerText, "try again") ||
		strings.Contains(lowerText, "retry") ||
		strings.Contains(lowerText, "fix it") ||
		strings.Contains(lowerText, "try the patch again")) {
		return false
	}
	if requestInputMentionsApplyPatch(req) ||
		completedToolNamesContain(workflowState.CompletedToolNames, "apply_patch") ||
		requestHistoryContainsApplyPatchValidationFeedback(req) {
		return true
	}
	return false
}

func searchToolNameMatches(name string) bool {
	lowerName := strings.ToLower(strings.TrimSpace(name))
	switch lowerName {
	case "web_search", "web_search_preview", "__llamaswap_web_search_preview", "websearch":
		return true
	default:
		return false
	}
}

func requestTextLooksLikeRetry(lowerText string) bool {
	return strings.Contains(lowerText, "try again") ||
		strings.Contains(lowerText, "retry") ||
		strings.Contains(lowerText, "fix it") ||
		strings.Contains(lowerText, "do it again")
}

func textLooksLikeSearchArtifact(text string) bool {
	lowerText := strings.ToLower(strings.TrimSpace(text))
	if lowerText == "" {
		return false
	}
	return strings.Contains(lowerText, "<websearch") ||
		strings.Contains(lowerText, "<web_search") ||
		strings.Contains(lowerText, "print(websearch") ||
		strings.Contains(lowerText, "print(web_search") ||
		strings.Contains(lowerText, "websearch(") ||
		strings.Contains(lowerText, "web_search(") ||
		strings.Contains(lowerText, "search the web") ||
		strings.Contains(lowerText, "web research") ||
		(strings.Contains(lowerText, "site:") && strings.Contains(lowerText, "reddit"))
}

func textLooksLikeEmptyToolArtifact(text string) bool {
	compact := strings.ToLower(strings.TrimSpace(text))
	if compact == "" {
		return false
	}
	compact = strings.NewReplacer("\n", "", "\r", "", "\t", "", " ", "").Replace(compact)
	if strings.HasPrefix(compact, "<tool_call") || strings.HasPrefix(compact, "<tool_use") || strings.HasPrefix(compact, "<tool_code") {
		return true
	}
	switch compact {
	case "<tool_code>", "<tool_call>", "<tool_use>", "<tool_code></tool_code>", "<tool_call></tool_call>", "<tool_use></tool_use>":
		return true
	default:
		return false
	}
}

func textLooksLikeDeferredToolPromise(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}
	promisePhrases := []string{
		"let me put together a detailed plan",
		"let me put together the detailed plan",
		"let me create a comprehensive plan",
		"let me write a detailed plan",
		"now i have a few questions to finalize the plan",
		"let me build this out",
		"let me create the directory",
		"let me create the full file",
		"let me create the quiz directory",
		"let me create both files now",
		"now let me create",
		"now let me build",
		"good, old file is gone. now let me create",
		"now applying the patch",
		"now applying the update",
		"now applying the update with apply_patch",
		"now i'll apply the patch",
		"now i will apply the patch",
		"fixing that now",
		"i'll fix that now",
		"i will fix that now",
		"updating that now",
		"now updating with apply_patch",
		"now i'll update the file with apply_patch",
		"now i'll update the file with `apply_patch`",
		"now i will update the file with apply_patch",
		"update it with apply_patch",
		"let me read",
		"let me first read",
		"let me check",
		"let me also check",
		"let me first check",
		"let me inspect",
		"let me first inspect",
		"let me start by",
		"let me start by examining",
		"let me verify",
		"let me first verify",
		"let me use a shell command",
		"let me use shell",
		"let me search",
		"let me get a bit more detail",
		"let me look",
		"let me also look",
		"let me first look",
		"i'll read",
		"i will read",
		"i'll check",
		"i will check",
		"i'll inspect",
		"i will inspect",
		"i'll verify",
		"i will verify",
		"i'll create it",
		"i will create it",
		"i need to create it",
		"now following its instructions",
		"follow its instructions",
		"good, the skill reads correctly. now i need to follow it",
		"skill created. now let me read it and follow the instructions",
		"agent spawned. waiting for it to complete",
		"now i need to wait for it",
		"now i need to wait for the agent",
		"let me wait for the agent",
		"good, i've spawned the agent. now i need to wait for it to complete its work",
		"i need to spawn an agent first",
		"now verifying",
		"verifying the content",
		"verifying the file content",
		"verifying the saved content",
		"verifying the saved file",
	}
	for _, phrase := range promisePhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func shouldSuppressVisibleProgressMessageDuringToolPhase(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}
	if textLooksLikeWeakProgressPlaceholder(lower) || textLooksLikeDeferredToolPromise(lower) || textLooksLikeDeferredSearchPromise(lower) {
		return true
	}
	specificPhrases := []string{
		"skill confirmed. now following its instructions",
		"skill created. now let me read it and follow the instructions",
		"good, the skill reads correctly. now i need to follow it",
		"agent spawned. waiting for it to complete the inspection and suggestions",
		"agent completed. now creating the delegated summary report",
		"good, i've spawned the agent. now i need to wait for it to complete its work",
		"i need to spawn an agent first. let me do that",
		"let me inspect the workspace directory first",
	}
	for _, phrase := range specificPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func textLooksLikeWeakProgressPlaceholder(text string) bool {
	switch strings.ToLower(strings.TrimSpace(text)) {
	case "", "working on the request.", "working on the request", "continuing from the latest tool results.", "continuing from the latest tool results", "running tool...", "running tool…":
		return true
	default:
		return false
	}
}

func textLooksLikeTruncatedAgentContinuation(text string, req map[string]any, workflowState ToolWorkflowState) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}
	if utf8.RuneCountInString(trimmed) > 2 {
		return false
	}
	if !requestMentionsAgentOrchestration(req) && !requestMentionsExactlyOneSubagent(req) {
		return false
	}
	if requestHistoryContainsCompletedTool(req, "apply_patch") {
		return false
	}
	hasAgentProgress := requestHistoryContainsCompletedTool(req, "spawn_agent") ||
		requestHistoryContainsCompletedTool(req, "wait_agent") ||
		requestHistoryContainsCompletedTool(req, "resume_agent") ||
		requestHistoryContainsCompletedTool(req, "send_input") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "spawn_agent") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "wait_agent") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "resume_agent") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "send_input")
	if !hasAgentProgress {
		return false
	}
	for _, r := range trimmed {
		return unicode.IsLetter(r)
	}
	return false
}

func textLooksLikeDeferredSearchPromise(text string) bool {
	lowerText := strings.ToLower(strings.TrimSpace(text))
	if lowerText == "" {
		return false
	}
	promisePhrases := []string{
		"let me search",
		"let me also search",
		"i will search",
		"i'll search",
		"let me research",
		"i will research",
		"i'll research",
		"let me browse",
		"let me investigate",
		"let me look up",
	}
	for _, phrase := range promisePhrases {
		if strings.Contains(lowerText, phrase) {
			return true
		}
	}
	return false
}

func textIndicatesSearchIntent(text string) bool {
	lowerText := strings.ToLower(strings.TrimSpace(text))
	if lowerText == "" {
		return false
	}
	hasSearchVerb := strings.Contains(lowerText, "search") ||
		strings.Contains(lowerText, "research") ||
		strings.Contains(lowerText, "investigate") ||
		strings.Contains(lowerText, "explore") ||
		strings.Contains(lowerText, "browse") ||
		strings.Contains(lowerText, "look up") ||
		strings.Contains(lowerText, "find ")
	hasSearchTarget := strings.Contains(lowerText, "web") ||
		strings.Contains(lowerText, "browser") ||
		strings.Contains(lowerText, "reddit") ||
		strings.Contains(lowerText, "site:") ||
		strings.Contains(lowerText, "current") ||
		strings.Contains(lowerText, "latest")
	return textLooksLikeSearchArtifact(lowerText) ||
		strings.Contains(lowerText, "search the web") ||
		strings.Contains(lowerText, "research the web") ||
		strings.Contains(lowerText, "investigate the web") ||
		strings.Contains(lowerText, "explore the web") ||
		strings.Contains(lowerText, "browse the web") ||
		strings.Contains(lowerText, "web research") ||
		strings.Contains(lowerText, "do research before") ||
		strings.Contains(lowerText, "research before") ||
		(hasSearchVerb && hasSearchTarget)
}

func requestHistoryContainsSearchArtifacts(req map[string]any, workflowState ToolWorkflowState) bool {
	if completedToolNamesContain(workflowState.CompletedToolNames, "web_search") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "web_search_preview") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "__llamaswap_web_search_preview") ||
		completedToolNamesContain(workflowState.CompletedToolNames, "websearch") {
		return true
	}
	if req == nil {
		return false
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		if searchToolNameMatches(strings.TrimSpace(fmt.Sprintf("%v", item["name"]))) {
			return true
		}
		switch strings.TrimSpace(fmt.Sprintf("%v", item["type"])) {
		case "message":
			role := strings.TrimSpace(fmt.Sprintf("%v", item["role"]))
			// User prompts can describe a desired web-search step without meaning
			// search already happened. Only treat assistant/tool-visible history as
			// completed research artifacts.
			if role != "assistant" && role != "tool" {
				continue
			}
			for _, text := range extractMessageTextParts(item["content"]) {
				if textLooksLikeSearchArtifact(text) {
					return true
				}
			}
		case "function_call_output":
			if textLooksLikeSearchArtifact(fmt.Sprintf("%v", item["output"])) {
				return true
			}
		}
	}
	return false
}

func requestHistoryContainsResearchArtifacts(req map[string]any, workflowState ToolWorkflowState) bool {
	if requestHistoryContainsSearchArtifacts(req, workflowState) {
		return true
	}
	for _, name := range workflowState.CompletedToolNames {
		normalized := normalizeToolTierName(name)
		if strings.HasPrefix(normalized, "browser_") {
			return true
		}
	}
	return false
}

func requestStillNeedsWebSearchBeforeApplyPatch(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil || !requestIncludesWebSearchTool(req) {
		return false
	}
	userText := extractResponsesUserInputText(req)
	if !requestExplicitlyWantsSearchIntent(req, workflowState) &&
		!textIndicatesSearchIntent(userText) &&
		!textLooksLikeDeferredSearchPromise(userText) {
		return false
	}
	if requestHistoryContainsSearchArtifacts(req, workflowState) || requestHistoryContainsResearchArtifacts(req, workflowState) {
		return false
	}
	return true
}

func requestShouldStayWebSearchOnlyForResearchPlan(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil || !requestIncludesWebSearchTool(req) || requestExplicitlyForbidsWebSearch(req) {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return false
	}
	if !requestExplicitlyWantsSearchIntent(req, workflowState) {
		return false
	}
	hasPlanIntent := strings.Contains(lowerText, "implementation plan") ||
		strings.Contains(lowerText, "detailed plan") ||
		strings.Contains(lowerText, "write a plan") ||
		strings.Contains(lowerText, "write the plan") ||
		strings.Contains(lowerText, "then write a plan") ||
		strings.Contains(lowerText, "then write the plan") ||
		strings.Contains(lowerText, "plan for it") ||
		strings.Contains(lowerText, " plan")
	if !hasPlanIntent {
		return false
	}
	hasNoBuildConstraint := requestExplicitlyWantsNoMutationPlan(req) ||
		strings.Contains(lowerText, "do not build") ||
		strings.Contains(lowerText, "don't build") ||
		strings.Contains(lowerText, "do not implement") ||
		strings.Contains(lowerText, "don't implement") ||
		strings.Contains(lowerText, "do not write code") ||
		strings.Contains(lowerText, "don't write code") ||
		strings.Contains(lowerText, "do not build the app") ||
		strings.Contains(lowerText, "don't build the app") ||
		strings.Contains(lowerText, "not build the app yet")
	if !hasNoBuildConstraint {
		return false
	}
	if requestInputMentionsApplyPatch(req) || requestWantsShellInspectionBeforeApplyPatch(req) {
		return false
	}
	return true
}

func requestedWebSearchContinuationsBeforeMutation(req map[string]any) int {
	if req == nil || !requestIncludesWebSearchTool(req) {
		return 0
	}
	if requestShouldStayWebSearchOnlyForResearchPlan(req, ToolWorkflowState{}) || requestExplicitlyWantsNoMutationPlan(req) {
		return 0
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return 0
	}
	mentionsWritePhase := strings.Contains(lowerText, "apply_patch") ||
		strings.Contains(lowerText, "create ") ||
		strings.Contains(lowerText, "write ") ||
		strings.Contains(lowerText, "save ") ||
		strings.Contains(lowerText, "verify ")
	if !mentionsWritePhase {
		return 0
	}
	switch {
	case strings.Contains(lowerText, "at least twice"),
		strings.Contains(lowerText, "twice with different quer"),
		strings.Contains(lowerText, "two different quer"),
		strings.Contains(lowerText, "2 different quer"):
		return 2
	case strings.Contains(lowerText, "at least three"),
		strings.Contains(lowerText, "three different quer"),
		strings.Contains(lowerText, "3 different quer"):
		return 3
	default:
		return 0
	}
}

func requestExplicitlyForbidsWebSearch(req map[string]any) bool {
	if req == nil {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(
		extractTrustedResponsesInstructionText(req) + "\n" + extractResponsesUserInputText(req),
	))
	if lowerText == "" {
		return false
	}
	return strings.Contains(lowerText, "do not use web search") ||
		strings.Contains(lowerText, "don't use web search") ||
		strings.Contains(lowerText, "without web search") ||
		strings.Contains(lowerText, "no web search") ||
		strings.Contains(lowerText, "do not search the web") ||
		strings.Contains(lowerText, "don't search the web")
}

func requestExplicitlyWantsExplorationFollowup(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if !(completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input") || workflowState.HasToolOutput) {
		return false
	}
	return requestExplicitlyWantsSearchIntent(req, workflowState)
}

func requestExplicitlyWantsSearchIntent(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if requestHasBoolFlag(req, "llamaswap_force_search_intent") {
		return true
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return false
	}
	hasAnsweredRequestUserInput := completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input")
	hasResearchHistory := requestHistoryContainsResearchArtifacts(req, workflowState)
	if requestTextLooksLikeRetry(lowerText) && hasResearchHistory {
		return true
	}
	if responsesRequestContractMode(req) == "plan" && hasResearchHistory {
		return false
	}
	if isCodexManagedPlanModeFromResponsesRequest(req) &&
		hasAnsweredRequestUserInput &&
		hasResearchHistory {
		return false
	}
	if textIndicatesSearchIntent(lowerText) {
		return true
	}
	hasSearchVerb := strings.Contains(lowerText, "search") ||
		strings.Contains(lowerText, "research") ||
		strings.Contains(lowerText, "look up") ||
		strings.Contains(lowerText, "find ")
	hasSearchTarget := strings.Contains(lowerText, "web") ||
		strings.Contains(lowerText, "browser") ||
		strings.Contains(lowerText, "reddit") ||
		strings.Contains(lowerText, "site:") ||
		strings.Contains(lowerText, "current") ||
		strings.Contains(lowerText, "latest") ||
		strings.Contains(lowerText, "external") ||
		strings.Contains(lowerText, "news")
	if strings.Contains(lowerText, "do a web_search") ||
		strings.Contains(lowerText, "web_search") ||
		strings.Contains(lowerText, "web search") ||
		strings.Contains(lowerText, "web research") ||
		strings.Contains(lowerText, "research the web") ||
		strings.Contains(lowerText, "do research before") ||
		strings.Contains(lowerText, "research before") ||
		strings.Contains(lowerText, "search the web") ||
		strings.Contains(lowerText, "look it up") ||
		strings.Contains(lowerText, "use the browser") ||
		strings.Contains(lowerText, "open it in the browser") ||
		strings.Contains(lowerText, "browser search") ||
		(hasSearchVerb && hasSearchTarget) {
		return true
	}
	return false
}

func requestStillWantsStructuredPlan(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if requestExplicitlyWantsImplementationRetry(req, workflowState) {
		return false
	}
	if requestExplicitlyWantsSearchIntent(req, workflowState) {
		return false
	}
	if requestExplicitlyWantsExplorationFollowup(req, workflowState) {
		return false
	}
	if requestExplicitlyWantsReturnedPlan(req) {
		return true
	}
	if !completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input") {
		return false
	}
	if isCodexManagedPlanModeFromResponsesRequest(req) {
		return true
	}
	lowerText := strings.ToLower(strings.TrimSpace(
		extractTrustedResponsesInstructionText(req) + "\n" + extractResponsesUserInputText(req),
	))
	if lowerText == "" {
		return false
	}
	return strings.Contains(lowerText, "<proposed_plan>") ||
		strings.Contains(lowerText, "write a plan") ||
		strings.Contains(lowerText, "return a plan") ||
		strings.Contains(lowerText, "finalize the plan") ||
		strings.Contains(lowerText, "plan directly as assistant text")
}

func structuredPlanOutputRequiredFromRequestBody(body []byte) bool {
	if !gjson.ValidBytes(body) {
		return rawResponsesBodyLooksLikePlanMode(body)
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return rawResponsesBodyLooksLikePlanMode(body)
	}
	if responsesRequestContractMode(req) == "plan" {
		return true
	}
	return requestStillWantsStructuredPlan(req, buildToolWorkflowState(req))
}

func nativeQuestionOutputRequiredFromRequestBody(body []byte) bool {
	if !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return requestExplicitlyWantsNativeCodexQuestion(req)
}

func synthesizeRequestUserInputArgumentsFromRequest(req map[string]any) (string, bool) {
	if req == nil || !requestExplicitlyWantsNativeCodexQuestion(req) {
		return "", false
	}
	fullText := strings.TrimSpace(
		extractResponsesNativeQuestionIntentText(req),
	)
	lowerText := strings.ToLower(fullText)
	if lowerText == "" {
		return "", false
	}
	if strings.Contains(lowerText, "ask me exactly one native codex question before planning anything else") ||
		strings.Contains(lowerText, "ask exactly one native codex question before planning anything else") {
		marker := ""
		if match := regexp.MustCompile(`\bT[0-9]+_SENTINEL\b`).FindString(fullText); strings.TrimSpace(match) != "" {
			marker = strings.TrimSpace(match)
		}
		question := "What is the current mission objective?"
		if marker != "" {
			question = marker + ": " + question
		}
		return mustJSONString(map[string]any{"questions": []string{question}}), true
	}
	switch {
	case strings.Contains(lowerText, "how they would like this built") ||
		strings.Contains(lowerText, "how you would like this built") ||
		strings.Contains(lowerText, "how the user would like this built") ||
		strings.Contains(lowerText, "which build approach they prefer"):
		return `{"questions":["How would you like this built, and which approach do you prefer?"]}`, true
	case strings.Contains(lowerText, "which approach they prefer") ||
		strings.Contains(lowerText, "which option they prefer"):
		return `{"questions":["Which approach do you prefer?"]}`, true
	}
	return "", false
}

func streamedRequestUserInputArgumentsFallback(reasoningText string, text string, req map[string]any) (string, bool) {
	if recovered, ok := proxyCompatibilityAdapters.ToolRepair.RecoverRequestUserInputArguments(reasoningText, text); ok {
		return recovered, true
	}
	return synthesizeRequestUserInputArgumentsFromRequest(req)
}

func requestExplicitlyWantsNoMutationPlan(req map[string]any) bool {
	if req == nil {
		return false
	}
	lowerText := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if lowerText == "" {
		return false
	}
	hasPlanIntent := strings.Contains(lowerText, "stay in plan mode") ||
		strings.Contains(lowerText, "plan mode") ||
		strings.Contains(lowerText, "hypothetical patch") ||
		strings.Contains(lowerText, "do not edit anything") ||
		strings.Contains(lowerText, "do not mutate") ||
		strings.Contains(lowerText, "do not execute")
	if !hasPlanIntent {
		return false
	}
	hasNoMutationConstraint := strings.Contains(lowerText, "do not edit") ||
		strings.Contains(lowerText, "do not mutate") ||
		strings.Contains(lowerText, "do not execute") ||
		strings.Contains(lowerText, "not executing")
	return hasNoMutationConstraint
}

func buildToolWorkflowState(req map[string]any) ToolWorkflowState {
	state := ToolWorkflowState{}
	if req == nil {
		return state
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return state
	}

	callNamesByID := map[string]string{}
	callFingerprintsByID := map[string]string{}
	pendingByID := map[string]string{}
	completedSequence := make([]string, 0, 4)
	appendCompleted := func(name string, fingerprint string) {
		name = normalizeToolTierName(name)
		if name == "" {
			return
		}
		state.HasToolOutput = true
		completedSequence = append(completedSequence, name)
		state.PreviousCompletedToolFingerprint = state.LatestCompletedToolFingerprint
		state.LatestCompletedToolName = name
		state.LatestCompletedToolFingerprint = fingerprint
		state.RepeatedLatestToolFingerprint = strings.TrimSpace(fingerprint) != "" &&
			strings.TrimSpace(fingerprint) == strings.TrimSpace(state.PreviousCompletedToolFingerprint)
	}
	appendPendingCall := func(callID string, name string, fingerprint string) {
		callID = strings.TrimSpace(callID)
		name = normalizeToolTierName(name)
		if callID == "" || name == "" {
			return
		}
		callNamesByID[callID] = name
		callFingerprintsByID[callID] = strings.TrimSpace(fingerprint)
		pendingByID[callID] = name
	}
	markCompletedCall := func(callID string) {
		callID = strings.TrimSpace(callID)
		if callID == "" {
			return
		}
		fingerprint := strings.TrimSpace(callFingerprintsByID[callID])
		if name := normalizeToolTierName(pendingByID[callID]); name != "" {
			delete(pendingByID, callID)
			appendCompleted(name, fingerprint)
			return
		}
		if name := normalizeToolTierName(callNamesByID[callID]); name != "" {
			appendCompleted(name, fingerprint)
		}
	}

	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		switch itemType {
		case "function_call", "custom_tool_call":
			appendPendingCall(callID, fmt.Sprintf("%v", item["name"]), buildToolCallFingerprint(fmt.Sprintf("%v", item["name"]), item["arguments"]))
		case "apply_patch_call":
			appendPendingCall(callID, "apply_patch", buildToolCallFingerprint("apply_patch", item["operation"]))
		case "shell_call":
			appendPendingCall(callID, "shell", buildToolCallFingerprint("shell", item["action"]))
		case "function_call_output", "custom_tool_call_output":
			state.HasToolOutput = true
			markCompletedCall(callID)
		case "apply_patch_call_output":
			state.HasToolOutput = true
			if callID != "" {
				markCompletedCall(callID)
			} else {
				appendCompleted("apply_patch", buildToolCallFingerprint("apply_patch", item["output"]))
			}
		case "shell_call_output":
			state.HasToolOutput = true
			if callID != "" {
				markCompletedCall(callID)
			} else {
				appendCompleted("shell", buildToolCallFingerprint("shell", item["output"]))
			}
		}
	}

	state.CompletedToolNames = completedSequence
	if len(pendingByID) > 0 {
		state.PendingToolNames = make([]string, 0, len(pendingByID))
		for _, name := range pendingByID {
			if normalized := normalizeToolTierName(name); normalized != "" {
				state.PendingToolNames = append(state.PendingToolNames, normalized)
			}
		}
		sort.Strings(state.PendingToolNames)
	}

	state.VerificationExpected = requestExplicitlyWantsShellVerification(req) &&
		(requestInputMentionsApplyPatch(req) ||
			requestWantsShellInspectionBeforeApplyPatch(req) ||
			completedToolNamesContain(completedSequence, "apply_patch"))
	if !state.VerificationExpected && requestNeedsCodexSkillFilesystemProof(req) {
		state.VerificationExpected = true
	}

	if state.VerificationExpected {
		seenApplyPatch := false
		for _, name := range completedSequence {
			switch normalizeToolTierName(name) {
			case "apply_patch":
				seenApplyPatch = true
			case "shell":
				if seenApplyPatch {
					state.VerificationCompleted = true
				}
			}
		}
		if !state.VerificationCompleted && requestHistoryShowsCodexSkillFilesystemProof(req) {
			state.VerificationCompleted = true
		}
	}

	state.ApplyPatchSatisfied = shouldForceFinalAnswerAfterSatisfiedApplyPatch(req)
	state.FinalAnswerSafe = len(state.PendingToolNames) == 0 && (!state.VerificationExpected || state.VerificationCompleted)
	return state
}

func buildToolCallFingerprint(name string, payload any) string {
	name = normalizeToolTierName(name)
	if name == "" {
		return ""
	}
	switch typed := payload.(type) {
	case string:
		trimmed := strings.TrimSpace(typed)
		if name == "apply_patch" {
			trimmed = normalizeApplyPatchToolOutputText(trimmed)
		}
		if trimmed == "" {
			return name
		}
		return name + "|" + trimmed
	case map[string]any:
		normalized, _ := json.Marshal(normalizeMapValue(typed))
		trimmed := strings.TrimSpace(string(normalized))
		if trimmed == "" || trimmed == "{}" {
			return name
		}
		return name + "|" + trimmed
	default:
		normalized, _ := json.Marshal(normalizeMapValue(payload))
		trimmed := strings.TrimSpace(string(normalized))
		if trimmed == "" || trimmed == "null" {
			return name
		}
		return name + "|" + trimmed
	}
}

func normalizeApplyPatchToolOutputText(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if gjson.Valid(text) {
		kind := strings.TrimSpace(gjson.Get(text, "type").String())
		if strings.EqualFold(kind, "apply_patch_call_output") {
			payload := gjson.Get(text, "payload")
			if payload.Exists() {
				if extracted := strings.TrimSpace(payload.Get("output").String()); extracted != "" {
					text = extracted
				}
			} else if extracted := strings.TrimSpace(gjson.Get(text, "output").String()); extracted != "" {
				text = extracted
			}
		}
	}
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return text
	}
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) < 3 || trimmed[1] != ' ' {
			continue
		}
		path := strings.TrimSpace(trimmed[2:])
		if path == "" {
			continue
		}
		switch trimmed[0] {
		case 'A':
			lines[i] = "created: " + path
		case 'M':
			lines[i] = "modified: " + path
		case 'D':
			lines[i] = "deleted: " + path
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func extractLatestCompletedShellArguments(req map[string]any) string {
	if req == nil {
		return ""
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return ""
	}
	callArgsByID := map[string]string{}
	latest := ""
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		switch itemType {
		case "function_call", "custom_tool_call":
			if normalizeToolTierName(fmt.Sprintf("%v", item["name"])) != "shell" || callID == "" {
				continue
			}
			args := normalizeShellArgumentMapForResponse(parseToolArgsMapString(fmt.Sprintf("%v", item["arguments"])))
			if shellToolArgumentsValid(args) {
				callArgsByID[callID] = mustJSONString(args)
			}
		case "shell_call":
			if callID == "" {
				continue
			}
			action := normalizeMapValue(item["action"])
			if len(action) == 0 {
				continue
			}
			args := normalizeShellArgumentMapForResponse(action)
			if shellToolArgumentsValid(args) {
				callArgsByID[callID] = mustJSONString(args)
			}
		case "function_call_output", "custom_tool_call_output", "shell_call_output":
			if callID == "" {
				continue
			}
			if args := strings.TrimSpace(callArgsByID[callID]); args != "" {
				latest = args
			}
		}
	}
	return strings.TrimSpace(latest)
}

func requestHistoryContainsCompletedTool(req map[string]any, target string) bool {
	if req == nil {
		return false
	}
	target = normalizeToolTierName(target)
	if target == "" {
		return false
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return false
	}
	callNamesByID := map[string]string{}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		switch itemType {
		case "function_call", "custom_tool_call":
			if callID != "" {
				callNamesByID[callID] = normalizeToolTierName(fmt.Sprintf("%v", item["name"]))
			}
		case "apply_patch_call":
			if callID != "" {
				callNamesByID[callID] = "apply_patch"
			}
		case "shell_call":
			if callID != "" {
				callNamesByID[callID] = "shell"
			}
		case "function_call_output", "custom_tool_call_output", "shell_call_output", "apply_patch_call_output":
			if normalizeToolTierName(callNamesByID[callID]) == target {
				return true
			}
		}
	}
	return false
}

func shouldRecoverVerificationShellCall(workflowState ToolWorkflowState, verificationShellArgs string, text string, reasoning string) bool {
	if !workflowState.VerificationExpected || workflowState.VerificationCompleted {
		return false
	}
	if strings.TrimSpace(verificationShellArgs) == "" {
		return false
	}
	combined := strings.ToLower(strings.TrimSpace(text + "\n" + reasoning))
	if combined == "" {
		return false
	}
	return strings.Contains(combined, "verify") ||
		strings.Contains(combined, "verification") ||
		strings.Contains(combined, "inspect the updated file") ||
		strings.Contains(combined, "inspect the final file")
}

func extractWorkflowRecoveryContextFromRequestBody(body []byte) (ToolWorkflowState, string) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ToolWorkflowState{}, ""
	}
	workflowState := buildToolWorkflowState(req)
	verificationShellArgs := extractLatestCompletedShellArguments(req)
	if strings.TrimSpace(verificationShellArgs) == "" && requestNeedsCodexSkillFilesystemProof(req) {
		verificationShellArgs = buildCodexSkillProofShellArguments(req)
	}
	return workflowState, verificationShellArgs
}

func completedToolNamesContain(names []string, want string) bool {
	want = normalizeToolTierName(want)
	for _, name := range names {
		if normalizeToolTierName(name) == want {
			return true
		}
	}
	return false
}

func allCompletedToolNamesWithinSet(names []string, allowed ...string) bool {
	if len(names) == 0 {
		return false
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, name := range allowed {
		normalized := normalizeToolTierName(name)
		if normalized != "" {
			allowedSet[normalized] = struct{}{}
		}
	}
	if len(allowedSet) == 0 {
		return false
	}
	for _, name := range names {
		if _, ok := allowedSet[normalizeToolTierName(name)]; !ok {
			return false
		}
	}
	return true
}

func filterBridgeToolsToAllowedNames(tools []any, allowedNames ...string) []any {
	if len(tools) == 0 || len(allowedNames) == 0 {
		return tools
	}
	allowedSet := make(map[string]struct{}, len(allowedNames))
	for _, name := range allowedNames {
		normalized := normalizeToolTierName(name)
		if normalized != "" {
			allowedSet[normalized] = struct{}{}
		}
	}
	if len(allowedSet) == 0 {
		return tools
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		name := normalizeToolTierName(extractFunctionToolName(tool))
		if name == "" {
			name = normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", tool["type"])))
		}
		if _, ok := allowedSet[name]; ok {
			filtered = append(filtered, tool)
		}
	}
	if len(filtered) == 0 {
		return tools
	}
	return filtered
}

func deriveContinuationAllowedToolNames(req map[string]any) []string {
	if req == nil {
		return nil
	}
	workflowState := buildToolWorkflowState(req)
	if !workflowState.HasToolOutput {
		return nil
	}
	completed := workflowState.CompletedToolNames
	if requestNeedsCodexSkillFilesystemProof(req) {
		if requestHistoryContainsCompletedTool(req, "apply_patch") && !requestHistoryContainsCompletedTool(req, "shell") {
			return []string{"shell"}
		}
		return nil
	}
	if requestShouldForceApplyPatchAfterSingleSubagentWait(req) {
		return []string{"apply_patch"}
	}
	if requestShouldForceSendInputAfterResume(req) {
		return []string{"send_input"}
	}
	if requestMentionsExactlyOneSubagent(req) &&
		requestHistoryContainsCompletedTool(req, "close_agent") {
		if requestInputMentionsApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch") {
			return []string{"apply_patch", "shell"}
		}
		if responsesRequestExposesTool(req, "exec_command") || responsesRequestExposesTool(req, "shell") {
			return []string{"shell"}
		}
	}
	if requestExplicitlyWantsFileSearch(req) {
		if requestHistoryContainsCompletedTool(req, "file_search") {
			if requestInputMentionsApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch") {
				return []string{"apply_patch"}
			}
		}
	}
	if requestExplicitlyWantsPlaywrightBrowserTools(req) {
		return []string{"mcp__playwright__"}
	}
	if requestMentionsExactlyOneSubagent(req) &&
		!requestHistoryContainsCompletedTool(req, "close_agent") &&
		requestHistoryContainsCompletedTool(req, "wait_agent") &&
		requestMentionsPostWaitAgentFollowup(req) {
		return []string{"spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent"}
	}
	if requestMentionsExactlyOneSubagent(req) &&
		requestHistoryContainsCompletedTool(req, "wait_agent") {
		if requestInputMentionsApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch") {
			return []string{"apply_patch", "shell"}
		}
		if responsesRequestExposesTool(req, "exec_command") || responsesRequestExposesTool(req, "shell") {
			return []string{"shell"}
		}
	}
	if requestMentionsAgentOrchestration(req) &&
		!requestMentionsPostWaitAgentFollowup(req) &&
		(requestInputMentionsApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch")) &&
		requestHistoryContainsCompletedTool(req, "wait_agent") {
		return []string{"apply_patch", "shell"}
	}
	if requestMentionsAgentOrchestration(req) && requestInputMentionsApplyPatch(req) {
		return nil
	}
	if requestMentionsAgentOrchestration(req) || allCompletedToolNamesWithinSet(completed, "spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent") {
		return []string{"spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent"}
	}
	hasResearchHistory := requestHistoryContainsResearchArtifacts(req, workflowState)
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	wantsShell := strings.Contains(text, "shell") || strings.Contains(text, "verify with shell") || strings.Contains(text, "use shell")
	// A prior apply_patch completion is workflow history, not proof that the
	// next turn should be narrowed to patch-only semantics. Keep patch-family
	// narrowing for turns that still explicitly ask for patch sequencing.
	wantsPatchFamily := requestInputMentionsApplyPatch(req) || requestWantsShellInspectionBeforeApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch")
	if workflowState.VerificationExpected && completedToolNamesContain(completed, "apply_patch") && !workflowState.VerificationCompleted {
		return []string{"shell"}
	}
	// Local web research continuations should not collapse into patch-only tool
	// exposure just because apply_patch is mentioned somewhere in the original
	// request. Until a patch step has actually started, preserve broad tools so
	// the model can continue or finish research naturally.
	if hasResearchHistory && !completedToolNamesContain(completed, "apply_patch") {
		return nil
	}
	if wantsPatchFamily {
		allowed := make([]string, 0, 2)
		if completedToolNamesContain(completed, "shell") || wantsShell || requestWantsShellInspectionBeforeApplyPatch(req) {
			allowed = append(allowed, "shell")
		}
		if requestInputMentionsApplyPatch(req) || responsesRequestExposesTool(req, "apply_patch") {
			allowed = append(allowed, "apply_patch")
		}
		if len(allowed) > 0 {
			return allowed
		}
	}
	return nil
}

func responsesRequestExposesTool(req map[string]any, target string) bool {
	if req == nil {
		return false
	}
	target = normalizeToolTierName(target)
	tools, ok := req["tools"].([]any)
	if !ok {
		return false
	}
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			continue
		}
		name := normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", tool["name"])))
		if name == "" {
			if fn, ok := tool["function"].(map[string]any); ok {
				name = normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
			}
		}
		if name == "" {
			name = normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", tool["type"])))
		}
		if name == target {
			return true
		}
	}
	return false
}

func toolChoiceTargetsSpecificTool(raw any) bool {
	normalized := normalizeBridgeToolChoice(raw)
	switch typed := normalized.(type) {
	case nil:
		return false
	case string:
		trimmed := strings.TrimSpace(strings.ToLower(typed))
		return trimmed != "" && trimmed != "auto" && trimmed != "none" && trimmed != "required"
	case map[string]any:
		choiceType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", typed["type"])))
		return choiceType == "function"
	default:
		return false
	}
}

func requestHasBoolFlag(req map[string]any, key string) bool {
	if req == nil {
		return false
	}
	parseFlag := func(v any) bool {
		switch typed := v.(type) {
		case bool:
			return typed
		case string:
			lower := strings.TrimSpace(strings.ToLower(typed))
			return lower == "1" || lower == "true" || lower == "yes" || lower == "on"
		default:
			return false
		}
	}
	if parseFlag(req[key]) {
		return true
	}
	if meta, ok := req["metadata"].(map[string]any); ok {
		if parseFlag(meta[key]) {
			return true
		}
	}
	return false
}

func shouldForceLocalApplyPatchFallback(req map[string]any) bool {
	return requestHasBoolFlag(req, "llamaswap_force_local_apply_patch")
}

func extractMessageTextParts(content any) []string {
	switch typed := content.(type) {
	case string:
		if text := strings.TrimSpace(typed); text != "" {
			return []string{text}
		}
	case []any:
		parts := make([]string, 0, len(typed))
		for _, partRaw := range typed {
			part, ok := partRaw.(map[string]any)
			if !ok {
				continue
			}
			partType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", part["type"])))
			if partType != "input_text" && partType != "output_text" && partType != "text" && partType != "" {
				continue
			}
			if text := strings.TrimSpace(fmt.Sprintf("%v", part["text"])); text != "" {
				parts = append(parts, text)
			}
		}
		return parts
	default:
		if text := strings.TrimSpace(extractResponsesInputText(content)); text != "" {
			return []string{text}
		}
	}
	return nil
}

func extractResponsesUserInputText(req map[string]any) string {
	if req == nil {
		return ""
	}
	if raw, ok := req["input"].(string); ok {
		text := strings.TrimSpace(raw)
		if text != "" {
			return text
		}
	}

	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return ""
	}

	lastUserText := ""
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		if itemType != "message" {
			continue
		}
		role := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"])))
		if role != "user" {
			continue
		}
		if parts := extractMessageTextParts(item["content"]); len(parts) > 0 {
			lastUserText = strings.TrimSpace(strings.Join(parts, "\n"))
		} else {
			lastUserText = ""
		}
	}
	if strings.TrimSpace(lastUserText) != "" {
		return lastUserText
	}
	return strings.TrimSpace(cleanFallbackInput(req["instructions"], ""))
}

func extractResponsesExplicitUserInputText(req map[string]any) string {
	if req == nil {
		return ""
	}
	if raw, ok := req["input"].(string); ok {
		return strings.TrimSpace(raw)
	}

	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return ""
	}

	lastUserText := ""
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "message" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"]))) != "user" {
			continue
		}
		if parts := extractMessageTextParts(item["content"]); len(parts) > 0 {
			candidate := strings.TrimSpace(strings.Join(parts, "\n"))
			if responsesUserTextIsPureSubagentNotification(candidate) {
				continue
			}
			lastUserText = candidate
		} else {
			lastUserText = ""
		}
	}
	return strings.TrimSpace(lastUserText)
}

func extractResponsesAllExplicitUserInputText(req map[string]any) string {
	if req == nil {
		return ""
	}
	if raw, ok := req["input"].(string); ok {
		return strings.TrimSpace(raw)
	}

	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return ""
	}

	parts := make([]string, 0, len(items))
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "message" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"]))) != "user" {
			continue
		}
		messageParts := extractMessageTextParts(item["content"])
		if len(messageParts) == 0 {
			continue
		}
		text := strings.TrimSpace(strings.Join(messageParts, "\n"))
		if text == "" || responsesUserTextIsPureSubagentNotification(text) {
			continue
		}
		parts = append(parts, text)
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func responsesUserTextIsPureSubagentNotification(text string) bool {
	trimmed := strings.TrimSpace(text)
	return strings.HasPrefix(trimmed, "<subagent_notification>") && strings.HasSuffix(trimmed, "</subagent_notification>")
}

func extractResponsesAllUserInputText(req map[string]any) string {
	if req == nil {
		return ""
	}
	if raw, ok := req["input"].(string); ok {
		return strings.TrimSpace(raw)
	}

	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return strings.TrimSpace(cleanFallbackInput(req["instructions"], ""))
	}

	parts := make([]string, 0, len(items))
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", item["type"])) != "message" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["role"]))) != "user" {
			continue
		}
		switch typed := item["content"].(type) {
		case string:
			text := strings.TrimSpace(typed)
			if text != "" {
				parts = append(parts, text)
			}
		case []any:
			for _, partRaw := range typed {
				part, ok := partRaw.(map[string]any)
				if !ok {
					continue
				}
				partType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", part["type"])))
				if partType != "input_text" && partType != "output_text" && partType != "text" {
					continue
				}
				text := strings.TrimSpace(cleanFallbackInput(part["text"], ""))
				if text != "" {
					parts = append(parts, text)
				}
			}
		}
	}
	if len(parts) > 0 {
		return strings.TrimSpace(strings.Join(parts, "\n"))
	}
	return strings.TrimSpace(cleanFallbackInput(req["instructions"], ""))
}

var applyPatchPathHintRegexps = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bcreate\s+(?:a\s+new\s+)?file\s+named\s+([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bcreate\s+(?:a\s+new\s+)?file\s+at\s+([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bcreate\s+file\s+([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bupdate\s+(?:file\s+)?([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bmodify\s+(?:file\s+)?([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bedit\s+(?:file\s+)?([^\s"'` + "`" + `]+)`),
	regexp.MustCompile(`(?i)\bto\s+file\s+([^\s"'` + "`" + `]+)`),
}

var applyPatchContentHintRegexps = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bone\s+line:\s*([^\r\n]+)`),
	regexp.MustCompile(`(?i)\bcontaining\s+exactly\s+one\s+line:\s*([^\r\n]+)`),
	regexp.MustCompile(`(?i)\bappend(?:ing)?\s+exactly\s+one\s+line\s+['"]([^'"]+)['"]`),
	regexp.MustCompile(`(?i)\bappend(?:ing)?\s+one\s+line\s+['"]([^'"]+)['"]`),
}

var genericPathCandidateRegexps = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(\\\\[^\\/\r\n]+\\[^\\/\r\n]+(?:\\[^\\/\r\n]+)+\.[A-Za-z0-9]{1,16})`),
	regexp.MustCompile(`(?i)([A-Za-z]:\\(?:[^\\/:*?"<>\r\n]+\\)*[^\\/:*?"<>\r\n]+\.[A-Za-z0-9]{1,16})`),
	regexp.MustCompile(`(?i)(/mnt/[A-Za-z]/(?:[^\s"'` + "`" + `]+/)*[^\s"'` + "`" + `]+\.[A-Za-z0-9]{1,16})`),
	regexp.MustCompile(`(?i)(/(?:[^\s"'` + "`" + `]+/)*[^\s"'` + "`" + `]+\.[A-Za-z0-9]{1,16})`),
	regexp.MustCompile(`(?i)\b((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,16})\b`),
	regexp.MustCompile(`(?i)\b([A-Za-z_][A-Za-z0-9_.-]*\.[A-Za-z0-9]{1,16})\b`),
}

func extractApplyPatchPathHintFromToolOutputs(req map[string]any) string {
	if req == nil {
		return ""
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return ""
	}
	for i := len(items) - 1; i >= 0; i-- {
		item, ok := items[i].(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		if !strings.HasSuffix(itemType, "_call_output") {
			continue
		}
		candidates := []string{
			strings.TrimSpace(extractResponsesInputText(item["output"])),
			strings.TrimSpace(extractResponsesInputText(item)),
			strings.TrimSpace(encodeAnyAsJSONString(item["output"])),
		}
		for _, candidateText := range candidates {
			if candidateText == "" {
				continue
			}
			for _, re := range genericPathCandidateRegexps {
				match := re.FindStringSubmatch(candidateText)
				if len(match) < 2 {
					continue
				}
				candidate := strings.TrimSpace(match[1])
				candidate = strings.Trim(candidate, ".,;:!?)(")
				if candidate == "" {
					continue
				}
				return normalizeApplyPatchPathForWorkspace(candidate)
			}
		}
	}
	return ""
}

func extractApplyPatchPathHintFromResponsesRequestBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	text := strings.TrimSpace(extractResponsesExplicitUserInputText(req))
	contextText := strings.TrimSpace(extractResponsesAllExplicitUserInputText(req))
	if contextText == "" {
		contextText = strings.TrimSpace(extractResponsesAllUserInputText(req))
	}
	if contextText == "" {
		contextText = text
	}
	if text == "" && contextText == "" {
		return ""
	}
	if requestLooksLikeApplyPatchPolicyInstructionAny(text, contextText) {
		return ""
	}
	workflowState := proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
	hasPatchIntent := requestInputMentionsApplyPatch(req) ||
		requestWantsShellInspectionBeforeApplyPatch(req) ||
		requestHistoryContainsApplyPatchValidationFeedback(req) ||
		extractApplyPatchTypeHintFromResponsesRequestBody(body) != "" ||
		completedToolNamesContain(workflowState.CompletedToolNames, "apply_patch") ||
		completedToolNamesContain(workflowState.PendingToolNames, "apply_patch")
	if !hasPatchIntent {
		return ""
	}
	if candidate := extractApplyPatchNamedOrAtPathHint(text, contextText); candidate != "" {
		return candidate
	}
	if candidate := extractExplicitPathCandidateHint(text, contextText); candidate != "" {
		return candidate
	}
	searchTexts := []string{text}
	if trimmed := strings.TrimSpace(contextText); trimmed != "" && !strings.EqualFold(strings.TrimSpace(text), trimmed) {
		searchTexts = append(searchTexts, trimmed)
	}
	for _, re := range applyPatchPathHintRegexps {
		var match []string
		for _, searchText := range searchTexts {
			if strings.TrimSpace(searchText) == "" {
				continue
			}
			match = re.FindStringSubmatch(searchText)
			if len(match) >= 2 {
				break
			}
		}
		if len(match) < 2 {
			continue
		}
		if candidate := resolveApplyPatchPathHintCandidate(match[1], contextText); candidate != "" {
			return candidate
		}
	}
	bestCandidate := ""
	searchText := text
	if searchText == "" {
		searchText = contextText
	}
	for _, re := range genericPathCandidateRegexps {
		matches := re.FindAllStringSubmatch(searchText, -1)
		for _, match := range matches {
			if len(match) < 2 {
				continue
			}
			candidate := resolveApplyPatchPathHintCandidate(match[1], contextText)
			if len(candidate) > len(bestCandidate) {
				bestCandidate = candidate
			}
		}
	}
	if bestCandidate != "" {
		return bestCandidate
	}
	if fallback := extractApplyPatchPathHintFromToolOutputs(req); fallback != "" {
		return fallback
	}
	return ""
}

func extractExplicitPathCandidateHint(text string, contextText string) string {
	bestCandidate := ""
	searchText := text
	if strings.TrimSpace(searchText) == "" {
		searchText = contextText
	}
	for _, re := range genericPathCandidateRegexps {
		matches := re.FindAllStringSubmatch(searchText, -1)
		for _, match := range matches {
			if len(match) < 2 {
				continue
			}
			candidate := resolveApplyPatchPathHintCandidate(match[1], contextText)
			if len(candidate) > len(bestCandidate) {
				bestCandidate = candidate
			}
		}
	}
	return bestCandidate
}

func extractApplyPatchNamedOrAtPathHint(text string, contextText string) string {
	for _, source := range []string{text, contextText} {
		lower := strings.ToLower(source)
		phrases := []string{
			"file named ",
			"file at ",
		}
		for _, phrase := range phrases {
			idx := strings.Index(lower, phrase)
			if idx < 0 {
				continue
			}
			start := idx + len(phrase)
			if start >= len(source) {
				continue
			}
			rest := strings.TrimSpace(source[start:])
			if rest == "" {
				continue
			}
			end := len(rest)
			for i, r := range rest {
				if unicode.IsSpace(r) || r == '"' || r == '\'' || r == '`' || r == ',' || r == ';' {
					end = i
					break
				}
			}
			candidate := strings.TrimSpace(rest[:end])
			candidate = strings.Trim(candidate, ".,;:!?)(")
			if candidate == "" {
				continue
			}
			if resolved := resolveApplyPatchPathHintCandidate(candidate, contextText); resolved != "" {
				return resolved
			}
		}
	}
	return ""
}

func repairApplyPatchOperationPathWithHint(operation any, applyPatchPathHint string) (any, bool) {
	op, ok := normalizeApplyPatchOperation(operation).(map[string]any)
	if !ok {
		return operation, false
	}
	pathHint := strings.TrimSpace(normalizeApplyPatchPathForWorkspace(applyPatchPathHint))
	if pathHint == "" || !applyPatchPathExistsLocally(pathHint) {
		return op, false
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(op["type"], ""))
	if opType != "create_file" && opType != "update_file" && opType != "delete_file" {
		return op, false
	}
	opPath := strings.TrimSpace(cleanFallbackInput(op["path"], ""))
	if opPath != "" {
		normalizedPath := normalizeApplyPatchPathForWorkspace(opPath)
		pathLooksAbsolute := looksLikeAbsoluteWindowsOrUNCPath(opPath) || filepath.IsAbs(opPath)
		if strings.EqualFold(filepath.ToSlash(normalizedPath), filepath.ToSlash(pathHint)) {
			return op, false
		}
		if pathLooksAbsolute && applyPatchPathExistsLocally(normalizedPath) {
			return op, false
		}
	}
	rewritten := cloneMap(op)
	rewritten["path"] = pathHint
	return rewritten, true
}

func extractApplyPatchContentHintFromResponsesRequestBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	text := strings.TrimSpace(extractResponsesExplicitUserInputText(req))
	if text == "" {
		return ""
	}
	for _, re := range applyPatchContentHintRegexps {
		match := re.FindStringSubmatch(text)
		if len(match) < 2 {
			continue
		}
		content := strings.TrimSpace(match[1])
		if idx := strings.Index(strings.ToLower(content), ". then "); idx >= 0 {
			content = strings.TrimSpace(content[:idx])
		}
		if idx := strings.Index(strings.ToLower(content), " then "); idx >= 0 {
			content = strings.TrimSpace(content[:idx])
		}
		if content == "" {
			continue
		}
		return content
	}
	return ""
}

func extractApplyPatchTypeHintFromResponsesRequestBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesExplicitUserInputText(req)))
	if text == "" {
		return ""
	}
	if strings.Contains(text, "delete file") || strings.Contains(text, "remove file") {
		return "delete_file"
	}
	if strings.Contains(text, "update file") || strings.Contains(text, "modify file") || strings.Contains(text, "edit file") {
		return "update_file"
	}
	if strings.Contains(text, "append") {
		return "update_file"
	}
	if strings.Contains(text, "create file") || strings.Contains(text, "add file") || strings.Contains(text, "new file") {
		return "create_file"
	}
	return ""
}

func requestInputMentionsApplyPatch(req map[string]any) bool {
	input := strings.ToLower(extractResponsesAllUserInputText(req))
	if input == "" {
		return false
	}
	if requestInputExplicitlyForbidsApplyPatch(input) && !strings.Contains(input, "*** begin patch") {
		return false
	}
	if strings.Contains(input, "apply_patch") || strings.Contains(input, "*** begin patch") {
		return true
	}
	if strings.Contains(input, "strict tool call") || strings.Contains(input, "strict tool-only") || strings.Contains(input, "strikt tool call") {
		return true
	}
	return false
}

func extractRequestedCodexSkillName(req map[string]any) string {
	if req == nil {
		return ""
	}
	text := strings.TrimSpace(extractResponsesUserInputText(req))
	if text == "" {
		return ""
	}
	matches := codexSkillReferenceRegexp.FindAllStringSubmatch(text, -1)
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		name := strings.TrimSpace(match[1])
		if name != "" {
			return name
		}
	}
	return ""
}

func requestNeedsCodexSkillFilesystemProof(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	mentionsSkill := strings.Contains(text, "skill.md") ||
		strings.Contains(text, ".codex/skills") ||
		strings.Contains(text, ".codex\\skills") ||
		strings.Contains(text, "codex skill") ||
		strings.Contains(text, "skills directory") ||
		strings.Contains(text, "available skills") ||
		strings.Contains(text, "installed skills") ||
		strings.Contains(text, "skill definition") ||
		strings.Contains(text, "track down where that skill") ||
		strings.Contains(text, "skill is actually stored") ||
		strings.Contains(text, "find the skill") ||
		strings.Contains(text, "create a skill") ||
		strings.Contains(text, "update a skill") ||
		strings.Contains(text, "modify a skill") ||
		strings.Contains(text, "change the skill") ||
		strings.Contains(text, "write skill.md") ||
		extractRequestedCodexSkillName(req) != ""
	if !mentionsSkill {
		return false
	}
	return strings.Contains(text, "find") ||
		strings.Contains(text, "found") ||
		strings.Contains(text, "locate") ||
		strings.Contains(text, "stored") ||
		strings.Contains(text, "definition") ||
		strings.Contains(text, "create") ||
		strings.Contains(text, "created") ||
		strings.Contains(text, "write") ||
		strings.Contains(text, "install") ||
		strings.Contains(text, "installed") ||
		strings.Contains(text, "update") ||
		strings.Contains(text, "modify") ||
		strings.Contains(text, "change") ||
		strings.Contains(text, "edit") ||
		strings.Contains(text, "register")
}

func buildCodexSkillProofShellArguments(req map[string]any) string {
	if !requestNeedsCodexSkillFilesystemProof(req) {
		return ""
	}
	targetSkill := strings.TrimSpace(extractRequestedCodexSkillName(req))
	rootExpr := `$root = if ($env:CODEX_HOME) { Join-Path $env:CODEX_HOME 'skills' } else { Join-Path $env:USERPROFILE '.codex\skills' }`
	var command string
	if targetSkill != "" {
		safeName := regexp.QuoteMeta(targetSkill)
		command = rootExpr + "; " +
			`Get-ChildItem -Path $root -Recurse -Filter 'SKILL.md' -ErrorAction SilentlyContinue | ` +
			fmt.Sprintf(`Where-Object { $_.FullName -match '(?i)[\\/](%s)[\\/]SKILL\.md$' -or $_.FullName -match '(?i)%s' } | `, safeName, safeName) +
			`Select-Object FullName`
	} else {
		command = rootExpr + "; Get-ChildItem -Path $root -Recurse -Filter 'SKILL.md' -ErrorAction SilentlyContinue | Select-Object FullName"
	}
	return mustJSONString(map[string]any{
		"command": []string{"powershell.exe", "-Command", command},
	})
}

func textMentionsCodexSkillProof(raw string, targetSkill string) bool {
	lower := strings.ToLower(strings.TrimSpace(raw))
	if lower == "" {
		return false
	}
	hasSkillPath := strings.Contains(lower, "skill.md") ||
		strings.Contains(lower, ".codex\\skills") ||
		strings.Contains(lower, ".codex/skills") ||
		strings.Contains(lower, "/skills/")
	if !hasSkillPath {
		return false
	}
	if strings.TrimSpace(targetSkill) == "" {
		return true
	}
	return strings.Contains(lower, strings.ToLower(strings.TrimSpace(targetSkill)))
}

func requestHistoryShowsCodexSkillFilesystemProof(req map[string]any) bool {
	if req == nil {
		return false
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return false
	}
	targetSkill := strings.TrimSpace(extractRequestedCodexSkillName(req))
	callNamesByID := map[string]string{}
	callArgsByID := map[string]string{}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		switch itemType {
		case "function_call", "custom_tool_call":
			name := normalizeToolTierName(fmt.Sprintf("%v", item["name"]))
			callNamesByID[callID] = name
			callArgsByID[callID] = strings.TrimSpace(fmt.Sprintf("%v", item["arguments"]))
			if name == "apply_patch" && textMentionsCodexSkillProof(callArgsByID[callID], targetSkill) {
				return true
			}
		case "apply_patch_call":
			if textMentionsCodexSkillProof(mustJSONString(item["operation"]), targetSkill) {
				return true
			}
		case "shell_call":
			callNamesByID[callID] = "shell"
			callArgsByID[callID] = mustJSONString(item["action"])
		case "function_call_output", "custom_tool_call_output", "shell_call_output":
			if normalizeToolTierName(callNamesByID[callID]) != "shell" {
				continue
			}
			if textMentionsCodexSkillProof(fmt.Sprintf("%v", item["output"]), targetSkill) {
				return true
			}
		case "apply_patch_call_output":
			if textMentionsCodexSkillProof(fmt.Sprintf("%v", item["output"]), targetSkill) {
				return true
			}
		}
	}
	return false
}

func assistantTextMakesUnverifiedCodexSkillClaim(text string, reasoning string) bool {
	combined := strings.ToLower(strings.TrimSpace(text + "\n" + reasoning))
	if combined == "" {
		return false
	}
	claimMarkers := []string{
		"found it",
		"i found it",
		"i found the skill",
		"the skill was created",
		"i created the skill",
		"created the skill",
		"skill is installed",
		"skill exists",
		"updated the skill",
		"modified the skill",
		"wrote skill.md",
		"wrote the skill",
		"set it up",
	}
	for _, marker := range claimMarkers {
		if strings.Contains(combined, marker) {
			return true
		}
	}
	return false
}

func shouldRecoverCodexSkillProofShellCall(req map[string]any, verificationShellArgs string, text string, reasoning string) bool {
	if !requestNeedsCodexSkillFilesystemProof(req) {
		return false
	}
	if strings.TrimSpace(verificationShellArgs) == "" {
		return false
	}
	if requestHistoryShowsCodexSkillFilesystemProof(req) {
		return false
	}
	return assistantTextMakesUnverifiedCodexSkillClaim(text, reasoning)
}

func requestInputExplicitlyForbidsApplyPatch(input string) bool {
	input = strings.ToLower(strings.TrimSpace(input))
	if input == "" {
		return false
	}
	phrases := []string{
		"do not use apply_patch",
		"don't use apply_patch",
		"dont use apply_patch",
		"do not call apply_patch",
		"don't call apply_patch",
		"dont call apply_patch",
		"without apply_patch",
		"instead of apply_patch",
		"rather than apply_patch",
	}
	for _, phrase := range phrases {
		if strings.Contains(input, phrase) {
			return true
		}
	}
	return false
}

func requestInputMentionsApplyPatchDelete(req map[string]any) bool {
	input := strings.ToLower(extractResponsesExplicitUserInputText(req))
	if input == "" {
		return false
	}
	if !(strings.Contains(input, "apply_patch") || strings.Contains(input, "native apply_patch")) {
		return false
	}
	return strings.Contains(input, "delete file") ||
		strings.Contains(input, "delete ") ||
		strings.Contains(input, "remove file") ||
		strings.Contains(input, "delete /") ||
		strings.Contains(input, "delete \\")
}

func appendAssistantMessageToResponsesOutput(responseBody []byte, text string) []byte {
	text = strings.TrimSpace(text)
	if text == "" || !gjson.ValidBytes(responseBody) {
		return responseBody
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}
	output, _ := resp["output"].([]any)
	output = append(output, map[string]any{
		"id":   fmt.Sprintf("msg_%d_apply_patch_feedback", time.Now().UnixNano()),
		"type": "message",
		"role": "assistant",
		"content": []any{
			map[string]any{
				"type": "output_text",
				"text": text,
			},
		},
	})
	resp["output"] = output
	if out, err := json.Marshal(resp); err == nil {
		return out
	}
	return responseBody
}

func forceShellInspectionResponse(responseBody []byte, targetPath string) []byte {
	targetPath = strings.TrimSpace(normalizeApplyPatchPathForWorkspace(targetPath))
	if targetPath == "" || !gjson.ValidBytes(responseBody) {
		return responseBody
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}
	respID := strings.TrimSpace(fmt.Sprintf("%v", resp["id"]))
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
		resp["id"] = respID
	}
	callID := fmt.Sprintf("call_shell_inspect_%d", time.Now().UnixNano())
	item := map[string]any{
		"id":        "fc_" + callID,
		"type":      "function_call",
		"call_id":   callID,
		"name":      "exec_command",
		"status":    "in_progress",
		"arguments": mustJSONString(map[string]any{"cmd": "cat " + targetPath}),
	}
	resp["output"] = []any{item}
	resp["status"] = "requires_action"
	out, err := json.Marshal(resp)
	if err != nil {
		return responseBody
	}
	return out
}

func synthesizeApplyPatchNoActionFeedback(responseBody []byte) []byte {
	feedback := applyPatchValidationWarningPrefix + " no executable tool call was returned. The previous response only contained planning text or no actionable tool arguments. Retry immediately with a non-empty `operation` object, or provide the final answer if no patch is needed."
	return appendAssistantMessageToResponsesOutput(responseBody, feedback)
}

func responseHasNonEmptyAssistantMessage(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		if strings.TrimSpace(item.Get("type").String()) != "message" {
			continue
		}
		content := item.Get("content").Array()
		for _, part := range content {
			if strings.TrimSpace(part.Get("text").String()) != "" {
				return true
			}
		}
	}
	return false
}

func responseContainsToolCall(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.TrimSpace(item.Get("type").String())
		if itemType == "function_call" || strings.HasSuffix(itemType, "_call") {
			return true
		}
	}
	return false
}

func responseContainsNamedFunctionCall(responseBody []byte, name string) bool {
	name = strings.TrimSpace(strings.ToLower(name))
	if name == "" {
		return false
	}
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.TrimSpace(item.Get("type").String())
		if itemType != "function_call" {
			continue
		}
		if strings.TrimSpace(strings.ToLower(item.Get("name").String())) == name {
			return true
		}
	}
	return false
}

func responseContainsApplyPatchCall(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
		name := strings.ToLower(strings.TrimSpace(item.Get("name").String()))
		args := strings.ToLower(item.Get("arguments").String())
		if itemType == "apply_patch_call" {
			return true
		}
		if itemType == "function_call" && name == "apply_patch" {
			return true
		}
		if itemType == "custom_tool_call" && name == "apply_patch" {
			return true
		}
		if strings.Contains(args, "apply_patch") {
			return true
		}
	}
	return false
}

func responseContainsMutatingToolCall(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
		name := strings.TrimSpace(item.Get("name").String())
		switch itemType {
		case "apply_patch_call":
			return true
		case "function_call", "custom_tool_call":
			if toolIsMutating(name) {
				return true
			}
		}
	}
	return false
}

func extractApplyPatchPathFromResponse(responseBody []byte) string {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
		name := strings.ToLower(strings.TrimSpace(item.Get("name").String()))
		if itemType != "apply_patch_call" && !(itemType == "function_call" && name == "apply_patch") && !(itemType == "custom_tool_call" && name == "apply_patch") {
			continue
		}
		path := strings.TrimSpace(item.Get("operation.path").String())
		if path != "" {
			return normalizeApplyPatchPathForWorkspace(path)
		}
		args := parseToolArgsMapString(item.Get("arguments").String())
		if op, ok := normalizeApplyPatchOperation(selectApplyPatchOperation(args)).(map[string]any); ok {
			if candidate := strings.TrimSpace(cleanFallbackInput(op["path"], "")); candidate != "" {
				return normalizeApplyPatchPathForWorkspace(candidate)
			}
		}
	}
	return ""
}

func responseContainsToolOutput(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.TrimSpace(item.Get("type").String())
		if strings.HasSuffix(itemType, "_call_output") {
			return true
		}
	}
	return false
}

func requestContainsApplyPatchToolOutput(body []byte) bool {
	if !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return requestMapContainsApplyPatchToolOutput(req)
}

func requestContainsAnyToolOutput(body []byte) bool {
	if !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return requestMapContainsAnyToolOutput(req)
}

func requestMapContainsAnyToolOutput(req map[string]any) bool {
	input, _ := req["input"].([]any)
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		if strings.HasSuffix(itemType, "_call_output") {
			return true
		}
	}
	return false
}

func requestMapContainsNamedToolOutput(req map[string]any, names ...string) bool {
	if req == nil || len(names) == 0 {
		return false
	}
	allowed := map[string]struct{}{}
	for _, name := range names {
		name = strings.ToLower(strings.TrimSpace(name))
		if name != "" {
			allowed[name] = struct{}{}
		}
	}
	if len(allowed) == 0 {
		return false
	}
	callIDs := map[string]struct{}{}
	input, _ := req["input"].([]any)
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"]))) != "function_call" {
			continue
		}
		name := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["name"])))
		if _, ok := allowed[name]; !ok {
			continue
		}
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		if callID != "" {
			callIDs[callID] = struct{}{}
		}
	}
	if len(callIDs) == 0 {
		return false
	}
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"]))) != "function_call_output" {
			continue
		}
		callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
		if _, ok := callIDs[callID]; ok {
			return true
		}
	}
	return false
}

func requestMapContainsApplyPatchToolOutput(req map[string]any) bool {
	if req == nil {
		return false
	}
	if workflowState := buildToolWorkflowState(req); completedToolNamesContain(workflowState.CompletedToolNames, "apply_patch") {
		return true
	}
	input, _ := req["input"].([]any)
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", item["type"])))
		if itemType == "apply_patch_call_output" || itemType == "applypatch_call_output" {
			return true
		}
		if itemType != "function_call_output" {
			continue
		}
		output := strings.ToLower(strings.TrimSpace(encodeAnyAsJSONString(item["output"])))
		if strings.Contains(output, "apply_patch_call_output") || strings.Contains(output, "apply_patch") {
			return true
		}
	}
	return false
}

func requestContainsNamedToolOutput(body []byte, names ...string) bool {
	if !gjson.ValidBytes(body) {
		return false
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return requestMapContainsNamedToolOutput(req, names...)
}

func bridgeResponsesMaxAttempts(req []byte, shouldResolveWebSearchLocally bool) int {
	maxBridgeAttempts := 3
	maxWebSearchContinuations := 3
	if shouldResolveWebSearchLocally {
		maxBridgeAttempts = maxWebSearchContinuations + 2
	}
	if requestContainsNamedToolOutput(req, "wait_agent", "spawn_agent", "send_input", "close_agent") && maxBridgeAttempts > 2 {
		return 2
	}
	return maxBridgeAttempts
}

func bridgeResponsesAttemptTimeout(req []byte, shouldResolveWebSearchLocally bool) time.Duration {
	if requestContainsNamedToolOutput(req, "wait_agent", "spawn_agent", "send_input", "close_agent") {
		return 45 * time.Second
	}
	if requestContainsAnyToolOutput(req) && !shouldResolveWebSearchLocally {
		return 60 * time.Second
	}
	return 0
}

func shouldForceFinalAnswerAfterSatisfiedApplyPatch(req map[string]any) bool {
	if req == nil || !requestMapContainsAnyToolOutput(req) {
		return false
	}
	text := strings.TrimSpace(extractResponsesUserInputText(req))
	if text == "" || !strings.Contains(strings.ToLower(text), "append") || extractExactFinalReplyHintFromRequest(req) == "" {
		return false
	}
	body, err := json.Marshal(req)
	if err != nil {
		return false
	}
	pathHint := strings.TrimSpace(extractApplyPatchPathHintFromResponsesRequestBody(body))
	contentHint := strings.TrimSpace(extractApplyPatchContentHintFromResponsesRequestBody(body))
	typeHint := normalizeApplyPatchTypeHint(extractApplyPatchTypeHintFromResponsesRequestBody(body))
	if pathHint == "" || contentHint == "" || typeHint != "update_file" || !applyPatchPathExistsLocally(pathHint) {
		return false
	}
	raw, err := os.ReadFile(pathHint)
	if err != nil {
		return false
	}
	current := strings.ReplaceAll(string(raw), "\r\n", "\n")
	return strings.Contains(current, contentHint)
}

func buildPostApplyPatchTransientFallbackResponse() []byte {
	now := time.Now().Unix()
	resp := map[string]any{
		"id":         fmt.Sprintf("resp_apply_patch_fallback_%d", now),
		"object":     "response",
		"created_at": now,
		"status":     "completed",
		"output": []any{
			map[string]any{
				"id":   fmt.Sprintf("msg_apply_patch_fallback_%d", now),
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "apply_patch appears to have completed, but the upstream process disconnected while finalizing the follow-up step. Continuing as completed to avoid duplicate patch execution.",
					},
				},
			},
		},
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return []byte(`{"id":"resp_apply_patch_fallback","object":"response","status":"completed","output":[{"id":"msg_apply_patch_fallback","type":"message","role":"assistant","content":[{"type":"output_text","text":"apply_patch completed."}]}]}`)
	}
	return out
}

func buildPostToolTransientFallbackResponse() []byte {
	now := time.Now().Unix()
	resp := map[string]any{
		"id":         fmt.Sprintf("resp_tool_fallback_%d", now),
		"object":     "response",
		"created_at": now,
		"status":     "completed",
		"output": []any{
			map[string]any{
				"id":   fmt.Sprintf("msg_tool_fallback_%d", now),
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "A prior tool step appears to have completed, but the upstream process disconnected while finalizing follow-up output. Continuing as completed to avoid duplicate tool execution.",
					},
				},
			},
		},
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return []byte(`{"id":"resp_tool_fallback","object":"response","status":"completed","output":[{"id":"msg_tool_fallback","type":"message","role":"assistant","content":[{"type":"output_text","text":"Tool step completed."}]}]}`)
	}
	return out
}

func responseNeedsToolContinuation(responseBody []byte) bool {
	if !responseContainsToolCall(responseBody) {
		return false
	}
	return !responseContainsToolOutput(responseBody)
}

func responseIsTerminal(responseBody []byte) bool {
	if responseNeedsToolContinuation(responseBody) {
		return false
	}
	return responseHasNonEmptyAssistantMessage(responseBody) || !responseContainsToolCall(responseBody)
}

func classifyResponsesProtocolState(responseBody []byte) string {
	if responseNeedsToolContinuation(responseBody) {
		if responseHasNonEmptyAssistantMessage(responseBody) {
			return "non_terminal_mixed_message_tool_call"
		}
		return "protocol_incomplete_tool_phase"
	}
	if responseContainsToolOutput(responseBody) && !responseHasNonEmptyAssistantMessage(responseBody) {
		return "empty_post_tool_recovery"
	}
	return ""
}

func forceResponseStatus(responseBody []byte, status string) []byte {
	if strings.TrimSpace(status) == "" || !gjson.ValidBytes(responseBody) {
		return responseBody
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}
	resp["status"] = status
	if out, err := json.Marshal(resp); err == nil {
		return out
	}
	return responseBody
}

func responseLooksLikePlanningOnly(responseBody []byte) bool {
	if responseContainsToolCall(responseBody) {
		return false
	}
	hasTodoList := false
	var textParts []string
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.TrimSpace(strings.ToLower(item.Get("type").String()))
		if itemType == "todo_list" {
			hasTodoList = true
			continue
		}
		if itemType != "message" {
			continue
		}
		item.Get("content").ForEach(func(_, part gjson.Result) bool {
			t := strings.TrimSpace(part.Get("text").String())
			if t != "" {
				textParts = append(textParts, strings.ToLower(t))
			}
			return true
		})
	}
	if hasTodoList {
		return true
	}
	if len(textParts) == 0 {
		return false
	}
	full := strings.Join(textParts, "\n")
	if (strings.Contains(full, "i'll") || strings.Contains(full, "i will") || strings.Contains(full, "let me") || strings.Contains(full, "now i'll")) &&
		(strings.Contains(full, "apply_patch") || strings.Contains(full, "tool")) {
		return true
	}
	planHints := []string{
		"i'll", "i will", "let me", "starting with", "proceeding to", "now i'll",
		"moving to", "running tests", "test", "plan", "next", "subtask", "batch",
	}
	score := 0
	for _, hint := range planHints {
		if strings.Contains(full, hint) {
			score++
		}
	}
	return score >= 2
}

func extractResponsesRequestModeFromBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		if rawResponsesBodyLooksLikePlanMode(body) {
			return "plan"
		}
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		if rawResponsesBodyLooksLikePlanMode(body) {
			return "plan"
		}
		return ""
	}
	mode := responsesRequestContractMode(req)
	if mode == "" && rawResponsesBodyLooksLikePlanMode(body) {
		return "plan"
	}
	return mode
}

func shouldRewritePlanModeResponseText(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return true
	}
	if looksLikePlanAcknowledgementWithoutPlan(trimmed) {
		return true
	}
	lower := strings.ToLower(trimmed)
	if strings.Contains(lower, "<proposed_plan>") && strings.Contains(lower, "</proposed_plan>") {
		return false
	}
	planHints := 0
	if strings.Contains(lower, "1.") {
		planHints++
	}
	if strings.Contains(lower, "2.") {
		planHints++
	}
	if strings.Contains(lower, "assumption") || strings.Contains(lower, "risk") {
		planHints++
	}
	if strings.Contains(lower, "validation") || strings.Contains(lower, "test plan") {
		planHints++
	}
	if planHints >= 2 &&
		!strings.Contains(lower, "```diff") &&
		!strings.Contains(lower, "```patch") &&
		!strings.Contains(lower, "*** begin patch") &&
		!strings.Contains(lower, "*** end patch") &&
		!strings.Contains(lower, "saved to") &&
		!strings.Contains(lower, "open the file") {
		return false
	}
	badHints := []string{
		"```diff",
		"```patch",
		"*** begin patch",
		"*** end patch",
		"apply_patch",
		"shell_command",
		"i built",
		"i created",
		"saved to",
		"open the file",
		"written to",
		"file_change",
	}
	for _, hint := range badHints {
		if strings.Contains(lower, hint) {
			return true
		}
	}
	return planHints < 2
}

func stripTrailingPlanFollowup(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lower := strings.ToLower(text)
	cutMarkers := []string{
		"\n---",
		"\nwant me to",
		"\nwould you like me to",
		"\nif so,",
		"\nwhich approach do you prefer",
		"\nwhich option do you prefer",
		"\ndo you have any preferences",
		"\ntarget directory:",
	}
	cutIdx := -1
	for _, marker := range cutMarkers {
		if idx := strings.Index(lower, marker); idx >= 0 && (cutIdx == -1 || idx < cutIdx) {
			cutIdx = idx
		}
	}
	if cutIdx >= 0 {
		text = text[:cutIdx]
	}
	return strings.TrimSpace(text)
}

func sanitizeStructuredPlanText(text string) string {
	text = sanitizePlanModeCandidateText(text)
	text = stripTrailingPlanFollowup(text)
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	lower := strings.ToLower(trimmed)
	openIdx := strings.Index(lower, "<proposed_plan>")
	if openIdx >= 0 {
		closeIdx := strings.Index(lower, "</proposed_plan>")
		if closeIdx > openIdx {
			inner := strings.TrimSpace(trimmed[openIdx+len("<proposed_plan>") : closeIdx])
			if inner == "" {
				return ""
			}
			return "<proposed_plan>\n" + inner + "\n</proposed_plan>"
		}
		inner := strings.TrimSpace(trimmed[openIdx+len("<proposed_plan>"):])
		if inner == "" {
			return ""
		}
		return "<proposed_plan>\n" + inner + "\n</proposed_plan>"
	}
	return trimmed
}

func extractVisibleProposedPlanBlock(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	lower := strings.ToLower(trimmed)
	openIdx := strings.Index(lower, "<proposed_plan>")
	if openIdx < 0 {
		return ""
	}
	afterOpen := openIdx + len("<proposed_plan>")
	closeRel := strings.Index(lower[afterOpen:], "</proposed_plan>")
	if closeRel < 0 {
		return ""
	}
	closeIdx := afterOpen + closeRel
	block := strings.TrimSpace(trimmed[openIdx : closeIdx+len("</proposed_plan>")])
	return sanitizeStructuredPlanText(block)
}

func planModeLengthDiagnosticText() string {
	return "<proposed_plan>\n" +
		"The model stopped early because it hit the max output limit (finish_reason: \"length\") before completing the plan.\n" +
		"Retry with a higher output token limit or a shorter prompt/context.\n" +
		"</proposed_plan>"
}

func extractPlanModeBulletsAfterMarker(text string, markers ...string) []string {
	if strings.TrimSpace(text) == "" || len(markers) == 0 {
		return nil
	}
	lower := strings.ToLower(text)
	for _, marker := range markers {
		marker = strings.ToLower(strings.TrimSpace(marker))
		if marker == "" {
			continue
		}
		idx := strings.Index(lower, marker)
		if idx < 0 {
			continue
		}
		segment := strings.TrimLeft(text[idx+len(marker):], " \t:\n")
		lines := strings.Split(segment, "\n")
		bullets := make([]string, 0, 8)
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				if len(bullets) > 0 {
					break
				}
				continue
			}
			if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
				bullets = append(bullets, strings.TrimSpace(trimmed[2:]))
				continue
			}
			if len(bullets) > 0 {
				break
			}
		}
		if len(bullets) > 0 {
			return bullets
		}
	}
	return nil
}

func appendUniquePlanItem(items *[]string, value string) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	for _, existing := range *items {
		if strings.EqualFold(strings.TrimSpace(existing), value) {
			return
		}
	}
	*items = append(*items, value)
}

func extractPlanModePreferenceValue(bullets []string, key string) string {
	key = strings.ToLower(strings.TrimSpace(key))
	if key == "" {
		return ""
	}
	for _, bullet := range bullets {
		trimmed := strings.TrimSpace(bullet)
		lower := strings.ToLower(trimmed)
		if !strings.HasPrefix(lower, key) {
			continue
		}
		value := strings.TrimSpace(trimmed[len(key):])
		value = strings.TrimLeft(value, ": ")
		value = strings.TrimSuffix(value, "(Recommended)")
		return strings.TrimSpace(value)
	}
	return ""
}

func countPlanModeQuestionItems(text string) int {
	count := 0
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		dot := strings.Index(trimmed, ".")
		if dot <= 0 {
			continue
		}
		validDigits := true
		for _, ch := range trimmed[:dot] {
			if ch < '0' || ch > '9' {
				validDigits = false
				break
			}
		}
		if validDigits {
			count++
		}
	}
	return count
}

func extractPlanModeQuestionCount(text string) int {
	if count := countPlanModeQuestionItems(text); count > 0 {
		return count
	}
	matches := planModeQuestionCountRegexp.FindStringSubmatch(text)
	if len(matches) < 2 {
		return 0
	}
	count, err := strconv.Atoi(matches[1])
	if err != nil || count <= 0 {
		return 0
	}
	return count
}

func buildStructuredPlanFromResearchNotes(text string) string {
	text = sanitizePlanModeCandidateText(text)
	if strings.TrimSpace(text) == "" {
		return ""
	}
	lower := strings.ToLower(text)
	if strings.Contains(lower, "<proposed_plan>") && strings.Contains(lower, "</proposed_plan>") {
		return text
	}
	preferences := extractPlanModeBulletsAfterMarker(text, "the user has answered:", "user has answered:")
	buildFeatures := extractPlanModeBulletsAfterMarker(
		text,
		"let me create a single html file with:",
		"let me create a single html + javascript file with:",
		"let me create with:",
		"let me build with:",
		"i will build with:",
		"i'll build with:",
	)
	if len(buildFeatures) == 0 {
		for _, feature := range []string{
			"multiple choice answers",
			"score tracking",
			"progress bar",
			"explanation section",
			"explanations after each answer",
			"smooth animations",
			"confetti",
			"colorful, fun design",
		} {
			if strings.Contains(lower, feature) {
				appendUniquePlanItem(&buildFeatures, feature)
			}
		}
	}
	questionCount := extractPlanModeQuestionCount(text)
	tech := extractPlanModePreferenceValue(preferences, "tech")
	format := extractPlanModePreferenceValue(preferences, "features")
	style := extractPlanModePreferenceValue(preferences, "style")
	if tech == "" && strings.Contains(lower, "html + javascript") {
		tech = "HTML + JavaScript"
	} else if tech == "" && (strings.Contains(lower, "single html file") || strings.Contains(lower, "embedded css + js") || strings.Contains(lower, "embedded css + javascript")) {
		tech = "HTML + JavaScript"
	}
	if format == "" && strings.Contains(lower, "classic quiz") {
		format = "Classic quiz"
	} else if format == "" && strings.Contains(lower, "multiple-choice questions") {
		format = "Classic multiple-choice quiz"
	}
	if style == "" && strings.Contains(lower, "colorful") && strings.Contains(lower, "fun") {
		style = "Colorful & Fun"
	}
	if questionCount == 0 && len(preferences) == 0 && len(buildFeatures) == 0 {
		return ""
	}

	summaryBullets := make([]string, 0, 3)
	implementationBullets := make([]string, 0, 5)
	validationBullets := make([]string, 0, 3)
	assumptionBullets := make([]string, 0, 4)

	appendUniquePlanItem(&summaryBullets, "Keep the turn in Plan Mode and convert the gathered research into a decision-complete implementation plan instead of starting file changes.")
	if questionCount > 0 {
		appendUniquePlanItem(&summaryBullets, fmt.Sprintf("Use the %d researched chemistry questions as the initial quiz content, preserving answer choices and short explanations.", questionCount))
	} else {
		appendUniquePlanItem(&summaryBullets, "Use the already researched chemistry question set as the initial quiz content rather than re-running the research step.")
	}

	switch {
	case tech != "" && format != "" && style != "":
		appendUniquePlanItem(&summaryBullets, fmt.Sprintf("Build the first version as a %s %s in `%s` with a `%s` presentation.", style, format, tech, style))
	case tech != "" && format != "":
		appendUniquePlanItem(&summaryBullets, fmt.Sprintf("Build the first version as a `%s` %s implementation.", tech, format))
	case tech != "":
		appendUniquePlanItem(&summaryBullets, fmt.Sprintf("Keep the first version in `%s` so it stays easy to run and hand off.", tech))
	}

	appendUniquePlanItem(&implementationBullets, "Normalize each researched question into a quiz data shape with prompt, answer options, correct answer, and explanation text.")
	if tech != "" {
		appendUniquePlanItem(&implementationBullets, fmt.Sprintf("Create the quiz shell in `%s`, keeping the page standalone and easy to open locally.", tech))
	} else {
		appendUniquePlanItem(&implementationBullets, "Create a standalone quiz shell that can be opened and tested locally without extra setup.")
	}
	if len(buildFeatures) > 0 {
		appendUniquePlanItem(&implementationBullets, "Implement the core quiz flow and UI behaviors called out in the research notes:")
		for _, feature := range buildFeatures {
			appendUniquePlanItem(&implementationBullets, "- "+feature)
		}
	} else {
		appendUniquePlanItem(&implementationBullets, "Implement the core quiz flow with question progression, answer selection, score tracking, and result feedback.")
	}
	appendUniquePlanItem(&implementationBullets, "Finish with restart/results behavior and lightweight UI polish that fits the chosen style without expanding scope beyond the quiz itself.")

	appendUniquePlanItem(&validationBullets, "Verify every researched question renders with the intended answer choices and explanation text.")
	appendUniquePlanItem(&validationBullets, "Verify score, progress, and answer feedback stay correct across a full quiz run.")
	if tech != "" {
		appendUniquePlanItem(&validationBullets, fmt.Sprintf("Verify the `%s` delivery approach works directly in the target runtime without missing assets or broken interactions.", tech))
	}

	if tech != "" {
		appendUniquePlanItem(&assumptionBullets, "Tech: "+tech)
	}
	if format != "" {
		appendUniquePlanItem(&assumptionBullets, "Format: "+format)
	}
	if style != "" {
		appendUniquePlanItem(&assumptionBullets, "Style: "+style)
	}
	if questionCount > 0 {
		appendUniquePlanItem(&assumptionBullets, fmt.Sprintf("Content: use the %d-question research set already gathered in this turn as the initial quiz bank.", questionCount))
	}

	lines := []string{"# Chemistry Quiz Plan", "", "## Summary"}
	for _, bullet := range summaryBullets {
		lines = append(lines, "- "+bullet)
	}
	lines = append(lines, "", "## Implementation")
	stepNumber := 1
	for idx, bullet := range implementationBullets {
		if strings.HasPrefix(bullet, "- ") {
			lines = append(lines, bullet)
			continue
		}
		_ = idx
		lines = append(lines, fmt.Sprintf("%d. %s", stepNumber, bullet))
		stepNumber++
	}
	lines = append(lines, "", "## Validation")
	for _, bullet := range validationBullets {
		lines = append(lines, "- "+bullet)
	}
	if len(assumptionBullets) > 0 {
		lines = append(lines, "", "## Assumptions")
		for _, bullet := range assumptionBullets {
			lines = append(lines, "- "+bullet)
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func ensureProposedPlanWrapper(text string) string {
	trimmed := extractVisibleProposedPlanBlock(text)
	if trimmed == "" {
		trimmed = sanitizeStructuredPlanText(text)
	}
	if trimmed == "" {
		return ""
	}
	lower := strings.ToLower(trimmed)
	if strings.Contains(lower, "<proposed_plan>") && strings.Contains(lower, "</proposed_plan>") {
		return trimmed
	}
	return "<proposed_plan>\n" + trimmed + "\n</proposed_plan>"
}

func removeMutatingPlanModeTools(tools []any, failClosed bool) []any {
	if len(tools) == 0 {
		return tools
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		trimmedName := extractFunctionToolName(tool)
		// Plan turns must never expose mutating tools. Generic `shell` remains
		// available for read/inspection flows and is guarded separately at
		// command-level; named write/destructive tools are stripped here.
		if strings.EqualFold(trimmedName, "shell") {
			filtered = append(filtered, raw)
			continue
		}
		tier, known := lookupToolTier(trimmedName)
		// Codex-managed Plan Mode should preserve unknown native tools so the
		// non-mutating question/exploration surface does not collapse to zero.
		// Proxy-enforced raw plan mode remains fail-closed for unknown tools.
		if !known && !failClosed {
			filtered = append(filtered, raw)
			continue
		}
		if tier >= TierWrite {
			continue
		}
		filtered = append(filtered, raw)
	}
	return filtered
}

func removePlanInteractionTools(tools []any) []any {
	if len(tools) == 0 {
		return tools
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		serialized := strings.ToLower(mustJSONString(tool))
		if strings.Contains(serialized, `"name":"update_plan"`) {
			continue
		}
		name := extractFunctionToolName(tool)
		if strings.EqualFold(name, "update_plan") {
			continue
		}
		filtered = append(filtered, raw)
	}
	return filtered
}

func canonicalizeResponsesOutputToolName(name string) string {
	return defaultStreamReconstructionAdapter{}.CanonicalToolName(name)
}

func extractRequestUserInputArguments(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return ""
	}
	if hasNonEmptyQuestionList(parsed["questions"]) {
		return mustJSONString(parsed)
	}
	return ""
}

func renderRequestUserInputQuestionsAsPlainText(arguments string) string {
	normalized := extractRequestUserInputArguments(arguments)
	if normalized == "" {
		return ""
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(normalized), &parsed); err != nil {
		return ""
	}
	rawQuestions, _ := parsed["questions"].([]any)
	if len(rawQuestions) == 0 {
		return ""
	}
	lines := make([]string, 0, len(rawQuestions)*4)
	for idx, rawQuestion := range rawQuestions {
		question, ok := rawQuestion.(map[string]any)
		if !ok {
			continue
		}
		prompt := strings.TrimSpace(cleanFallbackInput(question["question"], ""))
		if prompt == "" {
			continue
		}
		lines = append(lines, fmt.Sprintf("%d. %s", idx+1, prompt))
		if options, ok := question["options"].([]any); ok {
			for _, rawOption := range options {
				option, ok := rawOption.(map[string]any)
				if !ok {
					continue
				}
				label := strings.TrimSpace(cleanFallbackInput(option["label"], ""))
				if label == "" {
					continue
				}
				lines = append(lines, fmt.Sprintf("- %s", label))
			}
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func extractRequestUserInputArgumentsFromToolMarkup(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	match := requestUserInputToolParameterRegexp.FindStringSubmatch(text)
	if len(match) != 2 {
		return ""
	}
	return extractRequestUserInputArguments(`{"questions":` + strings.TrimSpace(match[1]) + `}`)
}

func mergeAssistantQuestionText(base string, questions string) string {
	base = strings.TrimSpace(base)
	questions = strings.TrimSpace(questions)
	switch {
	case base == "":
		return questions
	case questions == "":
		return base
	case strings.Contains(base, questions):
		return base
	default:
		return strings.TrimSpace(base + "\n\n" + questions)
	}
}

func removeNamedToolFromList(tools []any, target string) []any {
	if len(tools) == 0 {
		return tools
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		if strings.EqualFold(extractFunctionToolName(tool), target) {
			continue
		}
		filtered = append(filtered, raw)
	}
	return filtered
}

func extractApplyPatchPlanText(operation map[string]any) string {
	if len(operation) == 0 {
		return ""
	}
	candidates := make([]string, 0, 4)
	if raw, ok := operation["content"]; ok && raw != nil {
		candidates = append(candidates, fmt.Sprintf("%v", raw))
	}
	if raw, ok := operation["input"]; ok && raw != nil {
		candidates = append(candidates, fmt.Sprintf("%v", raw))
	}
	if raw, ok := operation["patch"]; ok && raw != nil {
		candidates = append(candidates, fmt.Sprintf("%v", raw))
	}
	if raw, ok := operation["diff"]; ok && raw != nil {
		if derived, ok := deriveContentFromApplyPatchDiff(fmt.Sprintf("%v", raw)); ok {
			candidates = append(candidates, derived)
		}
	}
	for _, candidate := range candidates {
		if cleaned := sanitizePlanModeCandidateText(candidate); strings.TrimSpace(cleaned) != "" {
			return cleaned
		}
	}
	return ""
}

func sanitizePlanModeCandidateText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = stripLeadingReasoningDirective(strings.TrimSpace(text))
	if content, _ := extractContentAndReasoning(text); strings.TrimSpace(content) != "" {
		text = strings.TrimSpace(content)
	}
	cutMarkers := []string{
		"</parameter>",
		"<parameter=",
		"</function>",
		"<function=",
		"</tool_call>",
		"<tool_call>",
		"<update_plan>",
		"</update_plan>",
		"<explanation>",
		"</explanation>",
		"<status>",
		"</status>",
		"<step>",
		"</step>",
		"*** End Patch",
	}
	lower := strings.ToLower(text)
	cutIdx := -1
	for _, marker := range cutMarkers {
		if idx := strings.Index(lower, strings.ToLower(marker)); idx >= 0 && (cutIdx == -1 || idx < cutIdx) {
			cutIdx = idx
		}
	}
	if cutIdx >= 0 {
		text = text[:cutIdx]
	}
	text = strings.TrimSpace(text)
	for strings.HasSuffix(text, "```") {
		text = strings.TrimSpace(strings.TrimSuffix(text, "```"))
	}
	return strings.TrimSpace(text)
}

func enforcePlanModeResponse(responseBody []byte, upstreamFinishedNormally bool) []byte {
	if !gjson.ValidBytes(responseBody) {
		return responseBody
	}
	var resp map[string]any
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return responseBody
	}

	preferredVisiblePlan := ""
	textParts := make([]string, 0)
	reasoningParts := make([]string, 0)
	outputRaw, _ := resp["output"].([]any)
	for _, rawItem := range outputRaw {
		item, ok := rawItem.(map[string]any)
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", item["type"])))
		if itemType == "reasoning" {
			summaryRaw, _ := item["summary"].([]any)
			for _, rawPart := range summaryRaw {
				part, ok := rawPart.(map[string]any)
				if !ok {
					continue
				}
				text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
				if text != "" {
					reasoningParts = append(reasoningParts, text)
				}
			}
			continue
		}
		if itemType != "message" {
			continue
		}
		contentRaw, _ := item["content"].([]any)
		for _, rawPart := range contentRaw {
			part, ok := rawPart.(map[string]any)
			if !ok {
				continue
			}
			text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
			if text != "" {
				if preferredVisiblePlan == "" {
					preferredVisiblePlan = extractVisibleProposedPlanBlock(text)
				}
				textParts = append(textParts, text)
			}
		}
	}

	planText := strings.TrimSpace(strings.Join(textParts, "\n\n"))
	if preferredVisiblePlan != "" {
		planText = preferredVisiblePlan
	}
	finishReason := strings.TrimSpace(strings.ToLower(gjson.GetBytes(responseBody, "choices.0.finish_reason").String()))
	if preferredVisiblePlan == "" {
		if strings.TrimSpace(planText) == "" && finishReason == "length" {
			planText = planModeLengthDiagnosticText()
		} else {
			return responseBody
		}
	}
	if strings.TrimSpace(planText) == "" && finishReason == "length" {
		planText = planModeLengthDiagnosticText()
	}
	if strings.TrimSpace(planText) == "" && !upstreamFinishedNormally {
		return responseBody
	}
	if strings.TrimSpace(planText) == "" {
		return responseBody
	}
	planText = stripLeadingReasoningDirective(planText)
	if content, _ := extractContentAndReasoning(planText); strings.TrimSpace(content) != "" {
		planText = content
	}
	planText = sanitizeStructuredPlanText(planText)
	planText = ensureProposedPlanWrapper(planText)
	if strings.TrimSpace(planText) == "" {
		return responseBody
	}

	msgID := fmt.Sprintf("msg_plan_mode_%d", time.Now().UnixNano())
	message := map[string]any{
		"id":   msgID,
		"type": "message",
		"role": "assistant",
		"content": []any{
			map[string]any{
				"type": "output_text",
				"text": planText,
			},
		},
	}
	resp["status"] = "completed"
	resp["output"] = []any{message}
	resp["output_text"] = planText
	out, err := json.Marshal(resp)
	if err != nil {
		return responseBody
	}
	return out
}

func shouldEnforcePlanModeSyntheticRewrite(isPlanModeRequested, enforceProxyPlanMode bool, responseBody []byte) bool {
	if enforceProxyPlanMode {
		return true
	}
	if !isPlanModeRequested {
		return false
	}
	if !gjson.ValidBytes(responseBody) {
		return false
	}
	hasToolCall := false
	hasProposedPlan := false
	gjson.GetBytes(responseBody, "output").ForEach(func(_, item gjson.Result) bool {
		itemType := strings.TrimSpace(strings.ToLower(item.Get("type").String()))
		if itemType == "function_call" || strings.HasSuffix(itemType, "_call") {
			hasToolCall = true
			return false
		}
		if itemType != "message" {
			return true
		}
		item.Get("content").ForEach(func(_, part gjson.Result) bool {
			text := strings.ToLower(strings.TrimSpace(part.Get("text").String()))
			if strings.Contains(text, "<proposed_plan>") && strings.Contains(text, "</proposed_plan>") {
				hasProposedPlan = true
				return false
			}
			return true
		})
		return !hasProposedPlan
	})
	if hasToolCall || hasProposedPlan {
		return false
	}
	return true
}

func planModeBlocksMutatingToolCall(responseBody []byte) bool {
	return responseContainsMutatingToolCall(responseBody)
}

func requestHasTools(req map[string]any) bool {
	if req == nil {
		return false
	}
	tools, _ := req["tools"].([]any)
	return len(tools) > 0
}

func appendPlanExecutionInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Execution mode: do not only describe the plan. Immediately execute the next actionable subtask now using available tools. " +
		"After each result, continue to the next subtask automatically until completion or a real blocker."
	prependSystemInstructionOnce(req, instruction)
}

func appendInvalidToolRetryInstruction(req map[string]any, feedback string) {
	if req == nil {
		return
	}
	instruction := "Tool call arguments were invalid/empty in the previous attempt. Retry immediately by either:\n" +
		"- issuing the same tool with concrete non-empty arguments, or\n" +
		"- if no tool is needed, provide the final answer.\n" +
		"Do not return empty tool arguments.\n" +
		"Previous feedback: " + strings.TrimSpace(feedback)
	lowerFeedback := strings.ToLower(feedback)
	if strings.Contains(lowerFeedback, "name=\"apply_patch\"") || strings.Contains(lowerFeedback, "name='apply_patch'") || strings.Contains(lowerFeedback, "apply_patch") {
		instruction += "\nFor apply_patch specifically: issue a tool call with a non-empty `operation` object." +
			"\nExample: {\"operation\":{\"type\":\"update_file\",\"path\":\"path/to/file\",\"diff\":\"@@ ...\"}}" +
			"\nDo not send {} and do not omit `operation`."
	}
	prependSystemInstructionOnce(req, instruction)
}

func maybeForceRetryToolChoice(req map[string]any, feedback string) {
	if req == nil {
		return
	}
	if responsesRequestContractMode(req) == "plan" {
		return
	}
	if body, err := json.Marshal(req); err == nil && rawResponsesBodyLooksLikePlanMode(body) {
		return
	}
	lowerFeedback := strings.ToLower(feedback)
	if !(strings.Contains(lowerFeedback, "name=\"apply_patch\"") || strings.Contains(lowerFeedback, "name='apply_patch'") || strings.Contains(lowerFeedback, "apply_patch")) {
		return
	}
	if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
		filtered := make([]any, 0, 1)
		for _, raw := range tools {
			tool, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			toolType := strings.TrimSpace(fmt.Sprintf("%v", tool["type"]))
			if strings.EqualFold(toolType, "apply_patch") {
				filtered = append(filtered, tool)
				continue
			}
			name := strings.TrimSpace(fmt.Sprintf("%v", tool["name"]))
			if name == "" {
				if fn, ok := tool["function"].(map[string]any); ok {
					name = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
				}
			}
			if strings.EqualFold(name, "apply_patch") || strings.EqualFold(name, llamaSwapApplyPatchFunctionName) {
				filtered = append(filtered, tool)
			}
		}
		if len(filtered) > 0 {
			req["tools"] = filtered
		} else {
			req["tools"] = []any{map[string]any{"type": "apply_patch"}}
		}
	} else {
		req["tools"] = []any{map[string]any{"type": "apply_patch"}}
	}
	req["tool_choice"] = map[string]any{
		"type": "apply_patch",
	}
}

func appendApplyPatchTailConstraintToUserTurn(req map[string]any, constraint string) bool {
	if req == nil {
		return false
	}
	constraint = strings.TrimSpace(constraint)
	if constraint == "" {
		return false
	}
	input, ok := req["input"].([]any)
	if !ok || len(input) == 0 {
		return false
	}
	for idx := len(input) - 1; idx >= 0; idx-- {
		item, ok := input[idx].(map[string]any)
		if !ok {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["type"])), "message") {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["role"])), "user") {
			continue
		}
		item["content"] = appendPolicyToMessageContent(item["content"], constraint)
		input[idx] = item
		req["input"] = input
		return true
	}
	return false
}

func appendApplyPatchFirstAttemptConstraint(req map[string]any) {
	if req == nil {
		return
	}
	if responsesRequestContractMode(req) == "plan" {
		return
	}
	if body, err := json.Marshal(req); err == nil && rawResponsesBodyLooksLikePlanMode(body) {
		return
	}
	workflowState := proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
	userText := extractResponsesUserInputText(req)
	if requestIncludesWebSearchTool(req) &&
		(requestStillNeedsWebSearchBeforeApplyPatch(req, workflowState) ||
			((requestExplicitlyWantsSearchIntent(req, workflowState) ||
				textIndicatesSearchIntent(userText) ||
				textLooksLikeDeferredSearchPromise(userText)) &&
				!requestHistoryContainsResearchArtifacts(req, workflowState))) {
		appendStrictWebSearchOnlyInstruction(req)
		return
	}
	// Shell-first steering is only for explicit mixed-workflow prompts like
	// "first use shell ... then use apply_patch". Native apply_patch remains the
	// default for ordinary file-mutation requests.
	if requestWantsShellInspectionBeforeApplyPatch(req) {
		appendShellInspectionBeforeApplyPatchInstruction(req)
		keepOnlyShellTools(req)
		return
	}
	maybeForceRetryToolChoice(req, `name="apply_patch"`)
	appendApplyPatchTailConstraintToUserTurn(req, applyPatchTailConstraintText)
	if requestInputMentionsApplyPatchDelete(req) {
		prependSystemInstructionOnce(req, "When deleting files in this task, prefer the native apply_patch delete_file operation. Do not choose shell rm/del for requested file deletion steps.")
		appendApplyPatchTailConstraintToUserTurn(req, applyPatchDeleteConstraintText)
	}
}

func requestMentionsAgentOrchestration(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if text == "" {
		return false
	}
	hasAgentVerb := strings.Contains(text, "spawn_agent") ||
		strings.Contains(text, "wait_agent") ||
		strings.Contains(text, "resume_agent") ||
		strings.Contains(text, "close_agent") ||
		strings.Contains(text, "send_input") ||
		strings.Contains(text, "child") ||
		strings.Contains(text, "child agent") ||
		strings.Contains(text, "subagent")
	if !hasAgentVerb {
		return false
	}
	return strings.Contains(text, "spawn") ||
		strings.Contains(text, "wait") ||
		strings.Contains(text, "resume") ||
		strings.Contains(text, "close") ||
		strings.Contains(text, "follow-up")
}

func requestMentionsMultipleAgentSteps(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "spawn_agent twice") ||
		strings.Contains(text, "two child agents") ||
		strings.Contains(text, "two subagents") ||
		strings.Contains(text, "second child") ||
		strings.Contains(text, "second agent") ||
		strings.Contains(text, "both waits") ||
		strings.Contains(text, "close both")
}

func requestMentionsExactlyOneSubagent(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "exactly one subagent") ||
		strings.Contains(text, "exactly one child agent") ||
		strings.Contains(text, "exactly one agent")
}

func requestMentionsPrePatchNonPatchTool(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "view_image") ||
		strings.Contains(text, "file_search") ||
		strings.Contains(text, "playwright") ||
		strings.Contains(text, "browser tool") ||
		strings.Contains(text, "browser tools") ||
		strings.Contains(text, "mcp__playwright__")
}

func requestExplicitlyWantsFileSearch(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "file_search") ||
		strings.Contains(text, "search indexed files") ||
		strings.Contains(text, "search local files")
}

func requestExplicitlyWantsPlaywrightBrowserTools(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "playwright browser tools") ||
		strings.Contains(text, "use playwright browser tools") ||
		strings.Contains(text, "mcp__playwright__browser_") ||
		(strings.Contains(text, "playwright") && strings.Contains(text, "browser"))
}

func requestMentionsPostWaitAgentFollowup(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "resume_agent") ||
		strings.Contains(text, "send_input") ||
		strings.Contains(text, "resume the same agent") ||
		strings.Contains(text, "ask for one more feature") ||
		strings.Contains(text, "one more feature")
}

func appendStrictFileSearchOnlyInstruction(req map[string]any) {
	if req == nil {
		return
	}
	prependSystemInstructionOnce(req,
		"Use the file_search tool first for the requested file search step. Do not substitute shell, grep, find, or prose in place of file_search on this turn.")
	keepOnlyNamedTools(req, "file_search")
	forceFunctionToolChoice(req, "file_search")
	req["parallel_tool_calls"] = false
}

func appendStrictPlaywrightBrowserToolInstruction(req map[string]any) {
	if req == nil {
		return
	}
	prependSystemInstructionOnce(req,
		"Use the real Playwright MCP browser tools for this task. Do not substitute shell scripts, npm setup, Python browser automation, or prose when the Playwright browser tools are exposed.")
	keepOnlyNamedTools(req, "mcp__playwright__")
	removeNamedTools(req, "shell", "exec_command", "write_stdin")
	forceFunctionToolChoice(req, "mcp__playwright__browser_navigate")
	req["parallel_tool_calls"] = false
}

func requestShouldFinalizeExactlyOneSubagentSummary(req map[string]any, workflowState ToolWorkflowState) bool {
	if req == nil || !requestMentionsExactlyOneSubagent(req) {
		return false
	}
	if !requestHistoryContainsCompletedTool(req, "wait_agent") || !requestHistoryContainsCompletedTool(req, "apply_patch") {
		return false
	}
	if requestHistoryContainsCompletedTool(req, "shell") {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if strings.Contains(text, "verify") || strings.Contains(text, "check the file") || strings.Contains(text, "confirm the file") {
		return false
	}
	return true
}

func requestShouldForceApplyPatchAfterSingleSubagentWait(req map[string]any) bool {
	if req == nil || !requestMentionsExactlyOneSubagent(req) {
		return false
	}
	if !requestHistoryContainsCompletedTool(req, "wait_agent") {
		return false
	}
	if requestHistoryContainsCompletedTool(req, "apply_patch") {
		return false
	}
	if !requestInputMentionsApplyPatch(req) {
		return false
	}
	if requestMentionsPostWaitAgentFollowup(req) {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesAllUserInputText(req)))
	if strings.Contains(text, "verify") || strings.Contains(text, "check the file") || strings.Contains(text, "confirm the file") {
		return false
	}
	return true
}

func requestShouldForceSendInputAfterResume(req map[string]any) bool {
	if req == nil || !requestMentionsExactlyOneSubagent(req) {
		return false
	}
	if !requestMentionsPostWaitAgentFollowup(req) {
		return false
	}
	if !requestHistoryContainsCompletedTool(req, "resume_agent") {
		return false
	}
	if requestHistoryContainsCompletedTool(req, "send_input") {
		return false
	}
	if requestHistoryContainsCompletedTool(req, "close_agent") {
		return false
	}
	return true
}

func appendResponsesTool(req map[string]any, tool map[string]any) {
	if req == nil || tool == nil {
		return
	}
	rawTools, _ := req["tools"].([]any)
	name := normalizeToolTierName(extractFunctionToolName(tool))
	if name == "" {
		name = normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", tool["type"])))
	}
	for _, raw := range rawTools {
		existing, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		existingName := normalizeToolTierName(extractFunctionToolName(existing))
		if existingName == "" {
			existingName = normalizeToolTierName(strings.TrimSpace(fmt.Sprintf("%v", existing["type"])))
		}
		if existingName != "" && existingName == name {
			return
		}
	}
	req["tools"] = append(rawTools, tool)
}

func forceFunctionToolChoice(req map[string]any, name string) {
	if req == nil {
		return
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return
	}
	req["tool_choice"] = map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": name,
		},
	}
}

func removeRedundantSubagentNotificationMessages(req map[string]any) {
	if req == nil {
		return
	}
	items, ok := req["input"].([]any)
	if !ok || len(items) == 0 {
		return
	}
	filtered := make([]any, 0, len(items))
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		if strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["type"])), "message") &&
			strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["role"])), "user") {
			parts, _ := item["content"].([]any)
			if len(parts) == 1 {
				if part, ok := parts[0].(map[string]any); ok {
					text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
					if strings.HasPrefix(text, "<subagent_notification>") && strings.HasSuffix(text, "</subagent_notification>") {
						continue
					}
				}
			}
		}
		filtered = append(filtered, raw)
	}
	req["input"] = filtered
}

func appendSerializedAgentOrchestrationInstruction(req map[string]any) {
	if req == nil || !requestMentionsAgentOrchestration(req) {
		return
	}
	instruction := "Agent orchestration mode: execute collaboration steps explicitly and serially." +
		" After each spawn_agent, you must issue wait_agent or another explicitly requested collaboration tool before claiming progress." +
		" Do not say you are waiting unless you actually emit wait_agent." +
		" Do not finish the task until every requested wait_agent, resume_agent, send_input, and close_agent step has been emitted as a real tool call."
	if requestMentionsMultipleAgentSteps(req) {
		instruction += " For multi-agent tasks on this system, keep the sequence strictly serialized: finish the first child's wait/required follow-up before progressing to the second child, then close children explicitly before the final reply."
	}
	if requestMentionsExactlyOneSubagent(req) {
		instruction += " When the user asked for exactly one subagent, do not spawn a second agent after the first successful wait_agent result. Reuse that completed agent result and continue with the requested non-agent follow-up steps."
	}
	if requestShouldForceSendInputAfterResume(req) {
		instruction += " The agent has already been resumed. Emit exactly one send_input tool call next to ask for the one additional requested feature. Do not answer in prose, do not call resume_agent again, and do not close the agent yet."
	}
	if requestShouldForceApplyPatchAfterSingleSubagentWait(req) {
		instruction += " The waited subagent result already contains the requested findings. Do not inspect more files, do not call more agent tools, and do not verify yet. Emit exactly one apply_patch tool call next to write the delegated summary file."
	}
	prependSystemInstructionOnce(req, instruction)
	req["parallel_tool_calls"] = false
}

func keepOnlyNamedTools(req map[string]any, allowedNames ...string) bool {
	if req == nil || len(allowedNames) == 0 {
		return false
	}
	tools, ok := req["tools"].([]any)
	if !ok || len(tools) == 0 {
		return false
	}
	allowed := make(map[string]struct{}, len(allowedNames))
	for _, name := range allowedNames {
		normalized := normalizeToolTierName(name)
		if normalized == "" {
			continue
		}
		allowed[normalized] = struct{}{}
	}
	if len(allowed) == 0 {
		return false
	}
	filtered := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		name := normalizeToolTierName(extractFunctionToolName(tool))
		if name == "" {
			toolType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", tool["type"])))
			name = normalizeToolTierName(toolType)
		}
		if _, ok := allowed[name]; ok {
			filtered = append(filtered, tool)
		}
	}
	if len(filtered) == 0 {
		return false
	}
	req["tools"] = filtered
	delete(req, "tool_choice")
	return true
}

func removeNamedTools(req map[string]any, names ...string) bool {
	if req == nil || len(names) == 0 {
		return false
	}
	tools, ok := req["tools"].([]any)
	if !ok || len(tools) == 0 {
		return false
	}
	blocked := make(map[string]struct{}, len(names))
	for _, name := range names {
		normalized := normalizeToolTierName(name)
		if normalized != "" {
			blocked[normalized] = struct{}{}
		}
	}
	if len(blocked) == 0 {
		return false
	}
	filtered := make([]any, 0, len(tools))
	changed := false
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		name := normalizeToolTierName(extractFunctionToolName(tool))
		if name == "" {
			toolType := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", tool["type"])))
			name = normalizeToolTierName(toolType)
		}
		if _, forbidden := blocked[name]; forbidden {
			changed = true
			continue
		}
		filtered = append(filtered, tool)
	}
	if !changed {
		return false
	}
	req["tools"] = filtered
	return true
}

func appendEmptyPostToolRecoveryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Tool executed. Analyze the result and continue with the next step or final answer. " +
		"Return a non-empty assistant message or exactly one next tool call. Do not stop after the tool phase."
	prependSystemInstructionOnce(req, instruction)
}

func appendMissingWebSearchRecoveryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	if requestExplicitlyForbidsWebSearch(req) {
		return
	}
	instruction := "If current or external information is still needed, emit exactly one real web_search tool call next." +
		" Do not output placeholder progress text, empty <tool_code> wrappers, <websearch> tags, or print(websearch(...))." +
		" Prefer native web_search over Playwright/browser tools for normal research steps when web_search is available."
	prependSystemInstructionOnce(req, instruction)
	req["llamaswap_force_search_intent"] = true
	req["parallel_tool_calls"] = false
}

func appendStrictWebSearchOnlyInstruction(req map[string]any) {
	if req == nil {
		return
	}
	if requestExplicitlyForbidsWebSearch(req) {
		return
	}
	appendMissingWebSearchRecoveryInstruction(req)
	req["tools"] = []any{map[string]any{"type": "web_search", "external_web_access": true}}
	delete(req, "tool_choice")
	req["parallel_tool_calls"] = false
}

func appendWebSearchSynthesisInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "You now have enough gathered web research context to answer." +
		" Do not call any more tools on this turn." +
		" Synthesize the final user-facing answer directly from the gathered web_search results."
	prependSystemInstructionOnce(req, instruction)
	req["llamaswap_force_web_search_synthesis"] = true
	req["tool_choice"] = "none"
	req["parallel_tool_calls"] = false
}

func removeSystemMessagesContaining(req map[string]any, needle string) {
	if req == nil {
		return
	}
	needle = strings.ToLower(strings.TrimSpace(needle))
	if needle == "" {
		return
	}
	input, ok := req["input"].([]any)
	if !ok || len(input) == 0 {
		return
	}
	filtered := make([]any, 0, len(input))
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			filtered = append(filtered, raw)
			continue
		}
		if strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["type"])), "message") &&
			strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["role"])), "system") {
			text := strings.ToLower(strings.TrimSpace(strings.Join(extractMessageTextParts(item["content"]), "\n")))
			if strings.Contains(text, needle) {
				continue
			}
		}
		filtered = append(filtered, raw)
	}
	req["input"] = filtered
}

func clearWebSearchPhaseInstructions(req map[string]any) {
	if req == nil {
		return
	}
	removeSystemMessagesContaining(req, "emit exactly one real web_search tool call next")
	removeSystemMessagesContaining(req, "you now have enough gathered web research context to answer")
	delete(req, "llamaswap_force_web_search_synthesis")
}

func responsesRequestHasForcedWebSearchSynthesis(reqBody []byte) bool {
	if !gjson.ValidBytes(reqBody) {
		return false
	}
	return gjson.GetBytes(reqBody, "llamaswap_force_web_search_synthesis").Bool()
}

func appendMissingShellRecoveryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Emit exactly one real shell tool call next to inspect or verify the file." +
		" Do not output placeholder progress text, and do not call apply_patch on this turn." +
		" Use shell to read or inspect the concrete file state first."
	prependSystemInstructionOnce(req, instruction)
	keepOnlyShellTools(req)
	req["parallel_tool_calls"] = false
}

func appendApplyPatchStateInspectionRecoveryInstruction(req map[string]any, targetPath string) {
	if req == nil {
		return
	}
	targetPath = strings.TrimSpace(targetPath)
	instruction := "Previous apply_patch attempts were malformed or incomplete." +
		" Emit exactly one real shell tool call next to inspect the current filesystem state before any further patching." +
		" Use shell to read the target file if it exists, or list the parent directory if the file may need to be created." +
		" Do not call apply_patch on this turn."
	if targetPath != "" {
		instruction += " Target path: `" + targetPath + "`."
	}
	prependSystemInstructionOnce(req, instruction)
	keepOnlyShellTools(req)
	req["parallel_tool_calls"] = false
}

func appendWeakFinalRecoveryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Your previous reply was only placeholder progress text." +
		" Provide the actual final answer for the user now based on the completed work." +
		" Do not say 'Working on the request.', 'Running tool...', or similar progress text." +
		" Do not call any tool on this turn unless one is strictly necessary to answer."
	prependSystemInstructionOnce(req, instruction)
	req["parallel_tool_calls"] = false
}

func appendWebSearchSynthesisRetryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Your previous reply did not provide the requested final answer." +
		" Use only the gathered web_search results and the original user request." +
		" Do not inspect the workspace, do not mention checking project structure, and do not emit tool markup or pseudo tool calls." +
		" Return the actual final user-facing answer directly on this turn."
	prependSystemInstructionOnce(req, instruction)
	req["tool_choice"] = "none"
	req["parallel_tool_calls"] = false
}

func isWeakProgressPlaceholderText(text string) bool {
	normalized := strings.TrimSpace(strings.ToLower(text))
	switch normalized {
	case "",
		"working on the request.",
		"continuing from the latest tool results.",
		"running tool...",
		"applying the requested file update.",
		"the file has been created. let me close the agent and then provide the final summary.",
		"file created. now verifying the content.",
		"file created. now verifying the saved content.",
		"file created. now verifying the saved file.",
		"let me check the existing codebase to understand what's already here.",
		"file updated. verifying the saved content:",
		"verifying the file content:",
		"good, now let me verify the file content.",
		"good, now let me verify the file content",
		"file updated. verifying the saved content now.",
		"patch applied. verifying the file contents now.",
		"patch applied. verifying the file contents...",
		"the patch was applied successfully. let me verify the file was saved correctly:":
		return true
	default:
		return false
	}
}

func extractResponsesVisibleText(body []byte) string {
	text := strings.TrimSpace(gjson.GetBytes(body, "output_text").String())
	if text != "" {
		return text
	}
	for _, item := range gjson.GetBytes(body, "output").Array() {
		if strings.TrimSpace(strings.ToLower(item.Get("type").String())) != "message" {
			continue
		}
		for _, part := range item.Get("content").Array() {
			if strings.TrimSpace(strings.ToLower(part.Get("type").String())) != "output_text" {
				continue
			}
			text = strings.TrimSpace(part.Get("text").String())
			if text != "" {
				return text
			}
		}
	}
	return ""
}

func responsesOutputContainsAnyToolCall(responseBody []byte) bool {
	if !gjson.ValidBytes(responseBody) {
		return false
	}
	for _, item := range gjson.GetBytes(responseBody, "output").Array() {
		itemType := strings.TrimSpace(strings.ToLower(item.Get("type").String()))
		if strings.HasSuffix(itemType, "_call") || itemType == "function_call" {
			return true
		}
	}
	return false
}

func chatResponseContainsToolCallNamed(responseBody []byte, names ...string) bool {
	if !gjson.ValidBytes(responseBody) {
		return false
	}
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(strings.ToLower(name))
		if trimmed != "" {
			allowed[trimmed] = struct{}{}
		}
	}
	if len(allowed) == 0 {
		return false
	}
	for _, toolCall := range gjson.GetBytes(responseBody, "choices.0.message.tool_calls").Array() {
		name := strings.TrimSpace(strings.ToLower(toolCall.Get("function.name").String()))
		if name == "" {
			name = strings.TrimSpace(strings.ToLower(toolCall.Get("name").String()))
		}
		if _, ok := allowed[name]; ok {
			return true
		}
	}
	functionCall := gjson.GetBytes(responseBody, "choices.0.message.function_call")
	if functionCall.Exists() {
		name := strings.TrimSpace(strings.ToLower(functionCall.Get("name").String()))
		if _, ok := allowed[name]; ok {
			return true
		}
	}
	return false
}

func chatResponseToolCallsOnlyNamed(responseBody []byte, names ...string) bool {
	if !gjson.ValidBytes(responseBody) {
		return false
	}
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(strings.ToLower(name))
		if trimmed != "" {
			allowed[trimmed] = struct{}{}
		}
	}
	if len(allowed) == 0 {
		return false
	}
	found := false
	for _, toolCall := range gjson.GetBytes(responseBody, "choices.0.message.tool_calls").Array() {
		name := strings.TrimSpace(strings.ToLower(toolCall.Get("function.name").String()))
		if name == "" {
			name = strings.TrimSpace(strings.ToLower(toolCall.Get("name").String()))
		}
		if name == "" {
			return false
		}
		if _, ok := allowed[name]; !ok {
			return false
		}
		found = true
	}
	functionCall := gjson.GetBytes(responseBody, "choices.0.message.function_call")
	if functionCall.Exists() {
		name := strings.TrimSpace(strings.ToLower(functionCall.Get("name").String()))
		if name == "" {
			return false
		}
		if _, ok := allowed[name]; !ok {
			return false
		}
		found = true
	}
	return found
}

func extractChatToolCallsNamed(responseBody []byte, names ...string) []map[string]any {
	if !gjson.ValidBytes(responseBody) {
		return nil
	}
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(strings.ToLower(name))
		if trimmed != "" {
			allowed[trimmed] = struct{}{}
		}
	}
	if len(allowed) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, 2)
	for _, toolCall := range gjson.GetBytes(responseBody, "choices.0.message.tool_calls").Array() {
		name := strings.TrimSpace(strings.ToLower(toolCall.Get("function.name").String()))
		if name == "" {
			name = strings.TrimSpace(strings.ToLower(toolCall.Get("name").String()))
		}
		if _, ok := allowed[name]; !ok {
			continue
		}
		action := parseToolArgsMapString(strings.TrimSpace(toolCall.Get("function.arguments").String()))
		if len(action) == 0 {
			action = parseToolArgsMapString(strings.TrimSpace(toolCall.Get("arguments").String()))
		}
		out = append(out, map[string]any{
			"call_id": toolCall.Get("id").String(),
			"name":    name,
			"action":  action,
		})
	}
	return out
}

func shouldRetryMissingWebSearchCall(req map[string]any, translatedChatRequest []byte, chatResponse []byte, translatedResponse []byte, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if !requestIncludesWebSearchTool(req) &&
		!chatRequestIncludesToolName(translatedChatRequest, "web_search", "web_search_preview", llamaSwapWebSearchFunctionName) {
		return false
	}
	if responsesOutputContainsAnyToolCall(translatedResponse) {
		return false
	}
	if chatResponseContainsToolCallNamed(chatResponse, "web_search", "web_search_preview", llamaSwapWebSearchFunctionName) {
		return false
	}
	contentText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.content").String())
	reasoningText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.reasoning_content").String())
	deferredSearchPromise := textLooksLikeDeferredSearchPromise(contentText)
	if !(textLooksLikeWeakProgressPlaceholder(contentText) ||
		textLooksLikeSearchArtifact(contentText) ||
		textLooksLikeEmptyToolArtifact(contentText) ||
		deferredSearchPromise) {
		return false
	}
	combined := strings.TrimSpace(strings.Join([]string{
		extractResponsesUserInputText(req),
		contentText,
		reasoningText,
	}, "\n"))
	if !deferredSearchPromise && !textIndicatesSearchIntent(combined) && !requestExplicitlyWantsSearchIntent(req, workflowState) {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.finish_reason").String()), "stop")
}

func textLooksLikeShellInspectionPromise(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}
	return (strings.Contains(lower, "shell") || strings.Contains(lower, "cat ")) &&
		(strings.Contains(lower, "verify") ||
			strings.Contains(lower, "inspect") ||
			strings.Contains(lower, "read the file") ||
			strings.Contains(lower, "read the current file") ||
			strings.Contains(lower, "check the current file") ||
			strings.Contains(lower, "check the file"))
}

func responseOutputContainsInvalidApplyPatchMessage(responseBody []byte) bool {
	if !gjson.ValidBytes(responseBody) {
		return false
	}
	for _, item := range gjson.GetBytes(responseBody, "output").Array() {
		if strings.TrimSpace(item.Get("type").String()) != "message" {
			continue
		}
		for _, part := range item.Get("content").Array() {
			text := strings.ToLower(strings.TrimSpace(part.Get("text").String()))
			if strings.Contains(text, "apply_patch call was not executed because operation was invalid") {
				return true
			}
		}
	}
	return false
}

func textLooksLikeBrowserOpenPromise(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}
	return (strings.Contains(lower, "open it in your default browser") ||
		strings.Contains(lower, "open it in the browser") ||
		strings.Contains(lower, "open it in your browser") ||
		strings.Contains(lower, "open the browser") ||
		strings.Contains(lower, "open it so the user can see it working")) &&
		(strings.Contains(lower, "let me") ||
			strings.Contains(lower, "let's") ||
			strings.Contains(lower, "i'll") ||
			strings.Contains(lower, "i will"))
}

func shouldRetryInvalidApplyPatchOperation(req map[string]any, chatResponse []byte, translatedResponse []byte) bool {
	if req == nil {
		return false
	}
	if !responsesRequestExposesTool(req, "apply_patch") {
		return false
	}
	if !responseOutputContainsInvalidApplyPatchMessage(translatedResponse) {
		return false
	}
	if !chatResponseContainsToolCallNamed(chatResponse, "apply_patch") {
		return false
	}
	finishReason := strings.TrimSpace(strings.ToLower(gjson.GetBytes(chatResponse, "choices.0.finish_reason").String()))
	return finishReason == "tool_calls" || finishReason == "stop"
}

func shouldRecoverInvalidApplyPatchViaShell(req map[string]any, workflowState ToolWorkflowState, reasonCode string) bool {
	if req == nil {
		return false
	}
	rawTools := strings.ToLower(mustJSONString(req["tools"]))
	hasApplyPatchTool := responsesRequestExposesTool(req, "apply_patch") || strings.Contains(rawTools, "apply_patch")
	if !hasApplyPatchTool {
		return false
	}
	hasShellTool := responsesRequestExposesTool(req, "shell") ||
		responsesRequestExposesTool(req, "exec_command") ||
		responsesRequestExposesTool(req, "shellcommand") ||
		strings.Contains(rawTools, "\"shell\"") ||
		strings.Contains(rawTools, "exec_command") ||
		strings.Contains(rawTools, "shellcommand")
	if !hasShellTool &&
		!responsesRequestExposesTool(req, "exec_command") &&
		!responsesRequestExposesTool(req, "shellcommand") {
		return false
	}
	if workflowState.ApplyPatchSatisfied {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(reasonCode)) {
	case "empty_operation", "invalid_diff", "no_tool_call", "wrong_tool_call", "wrong_tool_call_oversized_shell_write":
		return true
	default:
		return false
	}
}

func shouldRetryMissingShellCall(req map[string]any, translatedChatRequest []byte, chatResponse []byte, translatedResponse []byte, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if requestLooksLikeApplyPatchPolicyInstruction(extractResponsesUserInputText(req)) {
		return false
	}
	contentText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.content").String())
	reasoningText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.reasoning_content").String())
	lowerReasoning := strings.ToLower(reasoningText)
	lowerContent := strings.ToLower(contentText)
	readFileIntent := strings.Contains(lowerReasoning, "read the current file") ||
		strings.Contains(lowerReasoning, "read the file first") ||
		strings.Contains(lowerReasoning, "check the current file") ||
		strings.Contains(lowerReasoning, "inspect the file") ||
		strings.Contains(lowerContent, "check the current file") ||
		strings.Contains(lowerContent, "read the current file") ||
		strings.Contains(lowerContent, "read the file first")
	verifyFileIntent := strings.Contains(lowerReasoning, "verify the file") ||
		strings.Contains(lowerReasoning, "verify the saved file") ||
		strings.Contains(lowerReasoning, "verify the file content") ||
		strings.Contains(lowerReasoning, "check what was saved") ||
		strings.Contains(lowerContent, "verify the file") ||
		strings.Contains(lowerContent, "verify the saved file") ||
		strings.Contains(lowerContent, "verify the file content")
	invalidPatchRecovery := responseOutputContainsInvalidApplyPatchMessage(translatedResponse) &&
		(readFileIntent || verifyFileIntent || workflowState.VerificationExpected)
	wrongApplyPatchInsteadOfShell := chatResponseToolCallsOnlyNamed(chatResponse, "apply_patch") &&
		(readFileIntent || verifyFileIntent)
	if responsesOutputContainsAnyToolCall(translatedResponse) && !invalidPatchRecovery && !wrongApplyPatchInsteadOfShell {
		return false
	}
	if chatResponseContainsToolCallNamed(chatResponse, "exec_command", "write_stdin", "shell", "shellcommand") {
		return false
	}
	shellInspectionStillNeeded := requestWantsShellInspectionBeforeApplyPatch(req) &&
		!completedToolNamesContain(workflowState.CompletedToolNames, "shell")
	verificationShellPending := workflowState.VerificationExpected &&
		completedToolNamesContain(workflowState.CompletedToolNames, "apply_patch") &&
		!workflowState.VerificationCompleted
	finishReason := strings.TrimSpace(strings.ToLower(gjson.GetBytes(chatResponse, "choices.0.finish_reason").String()))
	forcedShell := toolChoiceTargetsNormalizedName(gjson.GetBytes(translatedChatRequest, "tool_choice").Value(), "shell") &&
		(shellInspectionStillNeeded || verificationShellPending)
	shellPromise := textLooksLikeShellInspectionPromise(contentText) || textLooksLikeShellInspectionPromise(reasoningText)
	if !forcedShell && !shellPromise && !invalidPatchRecovery && !wrongApplyPatchInsteadOfShell && !shellInspectionStillNeeded && !verificationShellPending {
		return false
	}
	return finishReason == "stop" || finishReason == "tool_calls"
}

func appendMissingBrowserOpenRecoveryInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Emit exactly one real shell tool call next to open the finished local artifact in the user's default browser." +
		" Do not answer with prose claiming it was opened." +
		" Reuse the concrete file path from the completed work."
	prependSystemInstructionOnce(req, instruction)
	keepOnlyShellTools(req)
	req["parallel_tool_calls"] = false
}

func shouldRetryMissingBrowserOpenCall(req map[string]any, chatResponse []byte, translatedResponse []byte) bool {
	if req == nil {
		return false
	}
	if responsesOutputContainsAnyToolCall(translatedResponse) {
		return false
	}
	if chatResponseContainsToolCallNamed(chatResponse,
		"exec_command",
		"write_stdin",
		"shell",
		"shellcommand",
		"mcp__playwright__browser_navigate",
		"mcp__playwright__browser_click",
		"mcp__playwright__browser_tabs",
		"mcp__playwright__browser_snapshot",
	) {
		return false
	}
	contentText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.content").String())
	reasoningText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.reasoning_content").String())
	visibleText := extractResponsesVisibleText(translatedResponse)
	if !textLooksLikeBrowserOpenPromise(contentText) &&
		!textLooksLikeBrowserOpenPromise(reasoningText) &&
		!textLooksLikeBrowserOpenPromise(visibleText) {
		return false
	}
	finishReason := strings.TrimSpace(strings.ToLower(gjson.GetBytes(chatResponse, "choices.0.finish_reason").String()))
	return finishReason == "stop"
}

func shouldRetryWeakPlaceholderFinal(req map[string]any, translatedChatRequest []byte, chatResponse []byte, translatedResponse []byte, workflowState ToolWorkflowState) bool {
	if req == nil {
		return false
	}
	if requestLooksLikeApplyPatchPolicyInstruction(extractResponsesUserInputText(req)) {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(gjson.GetBytes(translatedResponse, "status").String()), "requires_action") {
		return false
	}
	visibleText := extractResponsesVisibleText(translatedResponse)
	visibleLooksIncomplete := isWeakProgressPlaceholderText(visibleText) ||
		textLooksLikeEmptyToolArtifact(visibleText) ||
		textLooksLikeDeferredToolPromise(visibleText) ||
		textLooksLikeTruncatedAgentContinuation(visibleText, req, workflowState)
	if !visibleLooksIncomplete {
		return false
	}
	if !gjson.ValidBytes(chatResponse) {
		return !responsesOutputContainsAnyToolCall(translatedResponse)
	}
	chatVisibleText := strings.TrimSpace(gjson.GetBytes(chatResponse, "choices.0.message.content").String())
	chatLooksIncomplete := isWeakProgressPlaceholderText(chatVisibleText) ||
		textLooksLikeEmptyToolArtifact(chatVisibleText) ||
		textLooksLikeDeferredToolPromise(chatVisibleText) ||
		textLooksLikeTruncatedAgentContinuation(chatVisibleText, req, workflowState)
	if !chatLooksIncomplete {
		return false
	}
	if chatResponseContainsToolCallNamed(chatResponse, "exec_command", "write_stdin", "shell", "shellcommand", "apply_patch", "web_search") {
		return false
	}
	finishReason := strings.TrimSpace(strings.ToLower(gjson.GetBytes(chatResponse, "choices.0.finish_reason").String()))
	return finishReason == "stop"
}

var exactFinalReplyRegexps = []*regexp.Regexp{
	regexp.MustCompile(`(?is)\bthen\s+reply\s+exactly\s*:\s*["']?([^\r\n"']+)["']?`),
	regexp.MustCompile(`(?is)\breply\s+exactly\s*:\s*["']?([^\r\n"']+)["']?`),
	regexp.MustCompile(`(?is)\bthen\s+reply\s+["']?([A-Za-z0-9_.:-]+(?:\s+[A-Za-z0-9_.:-]+)*)["']?`),
	regexp.MustCompile(`(?is)\bthen\s+respond\s+["']?([A-Za-z0-9_.:-]+(?:\s+[A-Za-z0-9_.:-]+)*)["']?`),
}

func extractExactFinalReplyHint(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	for _, re := range exactFinalReplyRegexps {
		match := re.FindStringSubmatch(text)
		if len(match) < 2 {
			continue
		}
		candidate := strings.TrimSpace(match[1])
		candidate = strings.Trim(candidate, " \t\r\n`\"'.!,;:)")
		if candidate == "" {
			continue
		}
		return candidate
	}
	return ""
}

func extractExactFinalReplyHintFromRequest(req map[string]any) string {
	if req == nil {
		return ""
	}
	return extractExactFinalReplyHint(extractResponsesUserInputText(req))
}

func extractExactFinalReplyHintFromRequestBody(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return extractExactFinalReplyHintFromRequest(req)
}

func requestLooksLikeTextOnlyReply(req map[string]any) bool {
	if req == nil {
		return false
	}
	if proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req).HasToolOutput {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	exactReply := extractExactFinalReplyHintFromRequest(req)
	hasReplyOnlyIntent := exactReply != "" ||
		strings.Contains(text, "reply exactly") ||
		strings.Contains(text, "answer exactly") ||
		strings.Contains(text, "acknowledge receipt and reply") ||
		strings.Contains(text, "reply with")
	if !hasReplyOnlyIntent {
		return false
	}
	toolIntentMarkers := []string{
		"apply_patch",
		"shell",
		"web search",
		"web_search",
		"file search",
		"file_search",
		"code interpreter",
		"code_interpreter",
		"image generation",
		"image_generation",
		"computer",
		"playwright",
		"browser_",
		"spawn_agent",
		"send_input",
		"resume_agent",
		"wait_agent",
		"close_agent",
		"request_user_input",
		"update_plan",
		"navigate to",
		"take screenshot",
		"use shell",
		"use apply_patch",
	}
	for _, marker := range toolIntentMarkers {
		if strings.Contains(text, marker) {
			return false
		}
	}
	return true
}

func requestWantsShellInspectionBeforeApplyPatch(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	wantsPatch := strings.Contains(text, "apply_patch") ||
		strings.Contains(text, "append") ||
		strings.Contains(text, "update file") ||
		strings.Contains(text, "modify file") ||
		strings.Contains(text, "edit file")
	if !wantsPatch {
		return false
	}
	wantsShell := strings.Contains(text, "shell")
	// Require an explicit "inspect/read first" style instruction so mixed
	// shell+patch ordering is preserved only when the user actually asked for it.
	wantsInspectFirst := strings.Contains(text, "inspect first") ||
		strings.Contains(text, "first inspect") ||
		strings.Contains(text, "first use shell") ||
		strings.Contains(text, "read the file first") ||
		strings.Contains(text, "first read the file") ||
		strings.Contains(text, "inspect the current file") ||
		strings.Contains(text, "inspect the file with shell") ||
		strings.Contains(text, "inspect with shell") ||
		strings.Contains(text, "read the file with shell") ||
		strings.Contains(text, "read with shell") ||
		strings.Contains(text, "verify with shell first") ||
		(strings.Contains(text, "with shell") && strings.Contains(text, "then use apply_patch")) ||
		(strings.Contains(text, "with shell") && strings.Contains(text, "then update it with apply_patch"))
	return wantsShell && wantsInspectFirst
}

func requestExplicitlyWantsShellVerification(req map[string]any) bool {
	if req == nil {
		return false
	}
	text := strings.ToLower(strings.TrimSpace(extractResponsesUserInputText(req)))
	if text == "" {
		return false
	}
	return strings.Contains(text, "verify with shell") ||
		strings.Contains(text, "verify the saved content with shell") ||
		strings.Contains(text, "verify the saved file with shell") ||
		strings.Contains(text, "verify the file with shell") ||
		strings.Contains(text, "verify the saved file content with shell") ||
		strings.Contains(text, "verify the final content with shell") ||
		strings.Contains(text, "verify the final file content with shell") ||
		strings.Contains(text, "verify the final file with shell") ||
		strings.Contains(text, "then verify with shell")
}

func summarizeApplyPatchRetryContext(req map[string]any) string {
	if req == nil {
		return "retry_context=nil"
	}
	toolChoice := strings.TrimSpace(mustJSONString(req["tool_choice"]))
	toolsCount := 0
	applyPatchTools := 0
	applyPatchRequired := "unknown"
	toolNames := make([]string, 0, 8)
	if tools, ok := req["tools"].([]any); ok {
		toolsCount = len(tools)
		for _, raw := range tools {
			tool, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			name := strings.TrimSpace(fmt.Sprintf("%v", tool["name"]))
			if name == "" {
				if fn, ok := tool["function"].(map[string]any); ok {
					name = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
					if params, ok := fn["parameters"].(map[string]any); ok &&
						(strings.EqualFold(name, "apply_patch") || strings.EqualFold(name, llamaSwapApplyPatchFunctionName)) {
						if required, ok := params["required"]; ok {
							applyPatchRequired = strings.TrimSpace(mustJSONString(required))
						}
					}
				}
			}
			if cleaned := strings.TrimSpace(cleanFallbackInput(name, "")); cleaned != "" {
				name = cleaned
				toolNames = append(toolNames, name)
			}
			if strings.EqualFold(name, "apply_patch") || strings.EqualFold(name, llamaSwapApplyPatchFunctionName) {
				applyPatchTools++
			}
		}
	}
	return fmt.Sprintf("tool_choice=%s tools=%d apply_patch_tools=%d apply_patch_required=%s tool_names=%s",
		truncateBridgeDebugText(toolChoice, 120),
		toolsCount,
		applyPatchTools,
		truncateBridgeDebugText(applyPatchRequired, 120),
		truncateBridgeDebugText(strings.Join(toolNames, ","), 200),
	)
}

func appendShellInspectionBeforeApplyPatchInstruction(req map[string]any) {
	if req == nil {
		return
	}
	instruction := "Honor the requested tool order. If the user asks to inspect or read a file with shell before mutating it, your first tool call must be a shell read/inspection of that target file. Only after that shell result may you call apply_patch. Do not patch first on this turn."
	if requestInputMentionsApplyPatchDelete(req) {
		instruction += " When the task includes deleting a file and apply_patch is available, perform that deletion with apply_patch using operation.type=delete_file, not shell rm/del."
	}
	prependSystemInstructionOnce(req, instruction)
}

func keepOnlyShellTools(req map[string]any) bool {
	return keepOnlyNamedTools(req, "shell", "exec_command", "shellcommand")
}

func responseMentionsApplyPatch(responseBody []byte) bool {
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		if strings.TrimSpace(item.Get("type").String()) != "message" {
			continue
		}
		content := item.Get("content").Array()
		for _, part := range content {
			text := strings.ToLower(strings.TrimSpace(part.Get("text").String()))
			if strings.Contains(text, "apply_patch") {
				return true
			}
		}
	}
	return false
}

func maybeForceApplyPatchRetryFromOutput(req map[string]any, responseBody []byte) {
	if !responseMentionsApplyPatch(responseBody) {
		return
	}
	maybeForceRetryToolChoice(req, `name="apply_patch"`)
}

func appendStrictApplyPatchToolOnlyInstruction(req map[string]any, reason string) {
	if req == nil {
		return
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "unspecified"
	}
	instruction := "Strict apply_patch recovery mode: return exactly one tool call and no assistant text." +
		"\nTool must be apply_patch." +
		"\nArguments must include non-empty `operation`." +
		"\n`operation` must include `type` and `path`; for `create_file`/`update_file`, include non-empty `diff` or `content`." +
		"\nDo not return `{}` and do not omit `operation`." +
		"\nRecovery reason: " + reason
	prependSystemInstructionOnce(req, instruction)
	keepOnlyNamedTools(req, "apply_patch")
	maybeForceRetryToolChoice(req, `name="apply_patch"`)
	req["parallel_tool_calls"] = false
}

func applyPatchOperationPayloadValid(operation any) bool {
	normalized := normalizeApplyPatchOperation(operation)
	if !hasNonEmptyApplyPatchOperation(normalized) {
		return false
	}
	opMap, ok := normalized.(map[string]any)
	if !ok {
		return true
	}
	opType := normalizeApplyPatchTypeHint(cleanFallbackInput(opMap["type"], ""))
	path := strings.TrimSpace(cleanFallbackInput(opMap["path"], ""))
	if opType == "" {
		return false
	}
	if path == "" {
		return false
	}
	if opType == "update_file" || opType == "create_file" {
		diff := strings.TrimSpace(cleanFallbackInput(opMap["diff"], ""))
		content := strings.TrimSpace(cleanFallbackInput(opMap["content"], ""))
		input := strings.TrimSpace(cleanFallbackInput(opMap["input"], ""))
		patch := strings.TrimSpace(cleanFallbackInput(opMap["patch"], ""))
		if opType == "update_file" && diff != "" && content == "" && input == "" && patch == "" {
			// Prevent repeated invalid-hunk loops: update_file diffs must be real
			// patch hunks (or explicit content/input), not free-form tail text.
			return looksLikePatchHunkOrDocument(diff)
		}
		return diff != "" || content != "" || input != "" || patch != ""
	}
	return true
}

func looksLikePatchHunkOrDocument(diff string) bool {
	normalized := normalizeApplyPatchText(diff)
	trimmed := strings.TrimSpace(normalized)
	if trimmed == "" {
		return false
	}
	if strings.Contains(trimmed, "*** Begin Patch") && strings.Contains(trimmed, "*** End Patch") {
		return true
	}
	// Unified diff or apply_patch hunk markers.
	return strings.Contains(trimmed, "@@")
}

func evaluateApplyPatchOutput(responseBody []byte) (bool, string, string) {
	output := gjson.GetBytes(responseBody, "output").Array()
	sawApplyPatch := false
	sawAnyToolCall := false
	firstToolSample := ""
	for _, item := range output {
		itemType := strings.TrimSpace(item.Get("type").String())
		if strings.HasSuffix(itemType, "_call") || itemType == "function_call" {
			sawAnyToolCall = true
			if firstToolSample == "" {
				if itemType == "function_call" {
					name := strings.TrimSpace(item.Get("name").String())
					args := strings.TrimSpace(item.Get("arguments").String())
					firstToolSample = "function_call:" + name + " args=" + args
				} else {
					firstToolSample = itemType
				}
			}
		}
		switch itemType {
		case "apply_patch_call":
			sawApplyPatch = true
			op := item.Get("operation").Value()
			opRaw := truncateBridgeDebugText(strings.TrimSpace(item.Get("operation").Raw), 220)
			if !hasNonEmptyApplyPatchOperation(op) {
				return false, "empty_operation", opRaw
			}
			if !applyPatchOperationPayloadValid(op) {
				return false, "invalid_diff", opRaw
			}
			return true, "", ""
		case "function_call":
			name := strings.TrimSpace(item.Get("name").String())
			if !strings.EqualFold(name, "apply_patch") {
				continue
			}
			sawApplyPatch = true
			args := parseToolArgsMapString(item.Get("arguments").String())
			op := selectApplyPatchOperation(args)
			sampled := truncateBridgeDebugText(strings.TrimSpace(mustJSONString(args)), 220)
			if !hasNonEmptyApplyPatchOperation(op) {
				return false, "empty_operation", sampled
			}
			if !applyPatchOperationPayloadValid(op) {
				return false, "invalid_diff", sampled
			}
			return true, "", ""
		}
	}
	if !sawAnyToolCall {
		return false, "no_tool_call", ""
	}
	if sawApplyPatch {
		return false, "empty_operation", ""
	}
	return false, "wrong_tool_call", firstToolSample
}

func shouldDeferApplyPatchValidationForCurrentTurn(
	req map[string]any,
	workflowState ToolWorkflowState,
	chatResponse []byte,
	responsesOutput []byte,
) bool {
	if req == nil {
		return false
	}
	researchAlreadyRecorded := requestHistoryContainsSearchArtifacts(req, workflowState) ||
		requestHistoryContainsResearchArtifacts(req, workflowState)
	if !researchAlreadyRecorded &&
		(requestStillNeedsWebSearchBeforeApplyPatch(req, workflowState) ||
			requestShouldStayWebSearchOnlyForResearchPlan(req, workflowState)) {
		return true
	}
	if chatResponseContainsToolCallNamed(chatResponse, "web_search", "web_search_preview", llamaSwapWebSearchFunctionName) {
		return true
	}
	if len(extractPendingBridgeWebSearchCalls(responsesOutput)) > 0 {
		return true
	}
	return false
}

func responseHasAssistantText(responseBody []byte) bool {
	if strings.TrimSpace(gjson.GetBytes(responseBody, "output_text").String()) != "" {
		return true
	}
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		if strings.TrimSpace(item.Get("type").String()) != "message" {
			continue
		}
		content := item.Get("content").Array()
		for _, part := range content {
			if strings.TrimSpace(part.Get("type").String()) != "output_text" {
				continue
			}
			if strings.TrimSpace(part.Get("text").String()) != "" {
				return true
			}
		}
	}
	return false
}

func isOversizedShellFileWrite(args string) bool {
	args = strings.TrimSpace(args)
	if len(args) < 8192 {
		return false
	}
	lower := strings.ToLower(args)
	patterns := []string{
		"set-content",
		"out-file",
		"writealltext",
		"tee -a",
		"@'",
		"@\"",
	}
	for _, pattern := range patterns {
		if strings.Contains(lower, pattern) {
			return true
		}
	}
	return false
}

func shouldForceStrictApplyPatchRetry(reasonCode string) bool {
	return buildApplyPatchRetryLoopDecision(reasonCode, "").ForceStrict
}

func buildApplyPatchRetryFailedResponse(baseResp []byte, reasonCode string, sampledArgs string) []byte {
	id := strings.TrimSpace(gjson.GetBytes(baseResp, "id").String())
	if id == "" {
		id = fmt.Sprintf("resp_%d", time.Now().UnixNano())
	}
	model := strings.TrimSpace(gjson.GetBytes(baseResp, "model").String())
	decision := buildApplyPatchRetryLoopDecision(reasonCode, sampledArgs)
	resp := map[string]any{
		"id":         id,
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      model,
		"output": []any{
			map[string]any{
				"id":   "msg_" + id + "_apply_patch_retry_failed",
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": decision.FailureText,
					},
				},
			},
		},
		"output_text": decision.FailureText,
	}
	out, _ := json.Marshal(resp)
	return out
}

func buildApplyPatchRetryLoopDecision(reasonCode string, sampledArgs string) RetryLoopDecision {
	normalizedReason := strings.ToLower(strings.TrimSpace(reasonCode))
	if normalizedReason == "" {
		normalizedReason = "no_tool_call"
	}
	forceStrict := false
	switch normalizedReason {
	case "no_tool_call", "wrong_tool_call", "wrong_tool_call_oversized_shell_write", "empty_operation", "invalid_diff":
		forceStrict = true
	}
	text := "[proxy] apply_patch retry could not be completed. Retry with apply_patch if the task still requires a file write."
	if strings.HasPrefix(normalizedReason, "wrong_tool_call") {
		text = applyPatchRetryPreferredFailureText
	}
	text += " (`" + normalizedReason + "`)."
	if strings.TrimSpace(sampledArgs) != "" {
		text += " Sampled arguments: " + truncateBridgeDebugText(sampledArgs, 220)
	}
	return RetryLoopDecision{
		ReasonCode:  normalizedReason,
		ForceStrict: forceStrict,
		FailureText: text,
	}
}

func extractFirstApplyPatchInvocationFromResponse(responseBody []byte) (string, map[string]any, bool) {
	if !gjson.ValidBytes(responseBody) {
		return "", nil, false
	}
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
		callID := strings.TrimSpace(item.Get("call_id").String())
		if callID == "" {
			callID = strings.TrimSpace(item.Get("id").String())
		}
		switch itemType {
		case "apply_patch_call":
			var raw any
			if opRaw := strings.TrimSpace(item.Get("operation").Raw); opRaw != "" {
				_ = json.Unmarshal([]byte(opRaw), &raw)
			}
			opAny := normalizeApplyPatchOperation(raw)
			if !applyPatchOperationPayloadValid(opAny) {
				continue
			}
			if op, ok := opAny.(map[string]any); ok {
				return callID, op, true
			}
		case "function_call":
			name := strings.ToLower(strings.TrimSpace(item.Get("name").String()))
			if name != "apply_patch" {
				continue
			}
			argsRaw := strings.TrimSpace(item.Get("arguments").String())
			if argsRaw == "" {
				continue
			}
			args := parseToolArgsMapString(argsRaw)
			opAny := selectApplyPatchOperation(args)
			if !applyPatchOperationPayloadValid(opAny) {
				continue
			}
			if op, ok := opAny.(map[string]any); ok {
				return callID, op, true
			}
		}
	}
	return "", nil, false
}

func findFirstApplyPatchCallItem(responseBody []byte) map[string]any {
	if !gjson.ValidBytes(responseBody) {
		return nil
	}
	output := gjson.GetBytes(responseBody, "output").Array()
	for _, item := range output {
		itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
		if itemType != "apply_patch_call" {
			continue
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(item.Raw), &parsed); err != nil {
			return nil
		}
		return parsed
	}
	return nil
}

func deriveContentFromApplyPatchDiff(diff string) (string, bool) {
	diff = strings.ReplaceAll(diff, "\r\n", "\n")
	diff = strings.TrimSpace(diff)
	if diff == "" {
		return "", false
	}

	// If it's not in patch/diff format, treat it as direct content.
	if !strings.Contains(diff, "@@") && !strings.Contains(diff, "---") && !strings.Contains(diff, "+++") {
		return diff, true
	}

	lines := strings.Split(diff, "\n")
	inHunk := false
	added := make([]string, 0, len(lines))
	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "@@"):
			inHunk = true
		case strings.HasPrefix(line, "---") || strings.HasPrefix(line, "+++"):
			continue
		case !inHunk:
			continue
		case strings.HasPrefix(line, "+"):
			content := strings.TrimPrefix(line, "+")
			// Some upstream variants double-prefix added lines ("++text").
			// Remove exactly one extra prefix when present.
			if strings.HasPrefix(content, "+") {
				content = strings.TrimPrefix(content, "+")
			}
			added = append(added, content)
		}
	}
	if len(added) == 0 {
		return "", false
	}
	joined := strings.Join(added, "\n")
	if !strings.Contains(joined, "\n") && strings.HasPrefix(joined, "+") {
		joined = strings.TrimPrefix(joined, "+")
	}
	return joined, true
}

func canonicalizeLocalApplyPatchPath(pathValue string) (string, error) {
	return canonicalizeLocalApplyPatchPathForHost(pathValue, runtime.GOOS)
}

func canonicalizeLocalApplyPatchPathForHost(pathValue string, hostOS string) (string, error) {
	pathValue = strings.TrimSpace(pathValue)
	if pathValue == "" {
		return "", fmt.Errorf("empty path")
	}
	hostOS = strings.ToLower(strings.TrimSpace(hostOS))
	if hostOS == "" {
		hostOS = "linux"
	}
	workspaceRoot := localApplyPatchWorkspaceRoot(hostOS)

	if looksLikeAbsoluteWindowsOrUNCPath(pathValue) {
		if hostOS == "windows" {
			cleaned := filepath.Clean(filepath.FromSlash(strings.ReplaceAll(pathValue, "\\", "/")))
			return cleaned, nil
		}
		if mapped, ok := tryMapWindowsOrUNCPathToLocal(pathValue); ok {
			pathValue = mapped
		} else {
			return "", fmt.Errorf("unsupported local path for bridge fallback: %s", pathValue)
		}
	}

	if hostOS == "windows" {
		normalized := strings.ReplaceAll(pathValue, "\\", "/")
		normalized = strings.TrimSpace(normalized)
		if mapped, ok := tryMapMntPathToWindows(normalized); ok {
			normalized = mapped
		}
		if filepath.IsAbs(filepath.FromSlash(normalized)) {
			cleaned := filepath.Clean(filepath.FromSlash(normalized))
			return cleaned, nil
		}
		joined := filepath.Clean(filepath.Join(filepath.FromSlash(workspaceRoot), filepath.FromSlash(normalized)))
		return joined, nil
	}

	normalized := strings.ReplaceAll(pathValue, "\\", "/")
	normalized = strings.TrimSpace(normalized)
	normalized = filepath.Clean(normalized)

	// Normalize mirrored duplicate prefix:
	// /home/admmin/llama-swap/home/admmin/llama-swap/...
	dupPrefix := workspaceRoot + workspaceRoot
	if strings.HasPrefix(normalized, dupPrefix) {
		normalized = strings.TrimPrefix(normalized, workspaceRoot)
		if !strings.HasPrefix(normalized, "/") {
			normalized = "/" + normalized
		}
		normalized = filepath.Clean(normalized)
	}

	if strings.HasPrefix(normalized, "/") {
		return normalized, nil
	}

	joined := filepath.Clean(filepath.Join(workspaceRoot, normalized))
	return joined, nil
}

func localApplyPatchWorkspaceRoot(hostOS string) string {
	if wd, err := os.Getwd(); err == nil {
		wd = strings.TrimSpace(wd)
		if wd != "" {
			return filepath.Clean(wd)
		}
	}
	if strings.EqualFold(hostOS, "windows") {
		return filepath.Clean("C:/")
	}
	return filepath.Clean("/")
}

func tryMapWindowsOrUNCPathToLocal(pathValue string) (string, bool) {
	pathValue = strings.TrimSpace(pathValue)
	if pathValue == "" {
		return "", false
	}

	normalized := strings.ReplaceAll(pathValue, "\\", "/")
	normalized = strings.TrimSpace(normalized)

	// Windows drive path: C:\foo\bar -> /mnt/c/foo/bar
	if len(normalized) >= 3 && normalized[1] == ':' && normalized[2] == '/' {
		drive := strings.ToLower(normalized[:1])
		rest := strings.TrimPrefix(normalized[3:], "/")
		if rest == "" {
			return "/mnt/" + drive, true
		}
		return "/mnt/" + drive + "/" + rest, true
	}

	// UNC WSL path: \\wsl$\Ubuntu\home\... -> /home/...
	lower := strings.ToLower(normalized)
	if strings.HasPrefix(lower, "//wsl$/ubuntu/") {
		rest := normalized[len("//wsl$/Ubuntu/"):]
		rest = strings.TrimPrefix(rest, "/")
		if rest == "" {
			return "/", true
		}
		return "/" + rest, true
	}

	return "", false
}

func tryMapMntPathToWindows(pathValue string) (string, bool) {
	pathValue = strings.TrimSpace(strings.ReplaceAll(pathValue, "\\", "/"))
	lower := strings.ToLower(pathValue)
	if !strings.HasPrefix(lower, "/mnt/") || len(pathValue) < len("/mnt/c/") {
		return "", false
	}
	drive := strings.ToUpper(pathValue[len("/mnt/") : len("/mnt/")+1])
	if drive < "A" || drive > "Z" {
		return "", false
	}
	rest := strings.TrimPrefix(pathValue[len("/mnt/x/"):], "/")
	if rest == "" {
		return drive + ":/", true
	}
	return drive + ":/" + rest, true
}

func executeApplyPatchOperationLocally(operation map[string]any) (string, error) {
	if !applyPatchOperationPayloadValid(operation) {
		return "", fmt.Errorf("invalid apply_patch operation payload")
	}
	opType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", operation["type"])))
	pathValue := strings.TrimSpace(fmt.Sprintf("%v", operation["path"]))
	if pathValue == "" {
		return "", fmt.Errorf("missing operation.path")
	}
	displayPath := pathValue
	targetPath, err := canonicalizeLocalApplyPatchPath(pathValue)
	if err != nil {
		return "", err
	}
	switch opType {
	case "delete_file":
		if err := os.Remove(targetPath); err != nil && !os.IsNotExist(err) {
			return "", err
		}
		return fmt.Sprintf("deleted %s", displayPath), nil
	case "create_file", "update_file":
		content := ""
		if raw, ok := operation["content"]; ok && raw != nil {
			switch v := raw.(type) {
			case string:
				content = v
			default:
				encoded, _ := json.Marshal(v)
				content = string(encoded)
			}
		}
		content = strings.TrimSpace(content)
		if content == "" {
			if raw, ok := operation["input"]; ok && raw != nil {
				if s, ok := raw.(string); ok {
					content = strings.TrimSpace(s)
				}
			}
		}
		if content == "" {
			if raw, ok := operation["patch"]; ok && raw != nil {
				if s, ok := raw.(string); ok {
					trimmed := strings.TrimSpace(s)
					if trimmed != "" && !looksLikeFilePathHint(trimmed) {
						content = trimmed
					}
				}
			}
		}
		if content == "" {
			if diff := strings.TrimSpace(fmt.Sprintf("%v", operation["diff"])); diff != "" {
				if derived, ok := deriveContentFromApplyPatchDiff(diff); ok {
					content = strings.TrimSpace(derived)
					// Safety normalization: for single-line fallback content derived from
					// unified diff, strip a leftover leading '+' if present.
					if !strings.Contains(content, "\n") && strings.HasPrefix(content, "+") {
						content = strings.TrimPrefix(content, "+")
					}
				} else {
					return "", fmt.Errorf("diff-only fallback not supported for local apply_patch execution")
				}
			}
		}
		if content == "" {
			return "", fmt.Errorf("missing operation.content for %s", opType)
		}
		if opType == "update_file" {
			if _, err := os.Stat(targetPath); err != nil {
				return "", fmt.Errorf("update target missing: %s", targetPath)
			}
		}
		if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
			return "", err
		}
		return fmt.Sprintf("%s %s", opType, displayPath), nil
	default:
		return "", fmt.Errorf("unsupported operation.type: %s", opType)
	}
}

func executeApplyPatchOperationLocallyWithDisplay(operation map[string]any, preferredDisplayPath string) (string, error) {
	if trimmed := strings.TrimSpace(preferredDisplayPath); trimmed != "" {
		op := cloneMap(operation)
		op["path"] = strings.TrimSpace(fmt.Sprintf("%v", operation["path"]))
		summary, err := executeApplyPatchOperationLocally(op)
		if err != nil {
			return "", err
		}
		parts := strings.SplitN(summary, " ", 2)
		if len(parts) == 2 {
			return parts[0] + " " + trimmed, nil
		}
		return summary, nil
	}
	return executeApplyPatchOperationLocally(operation)
}

func buildSyntheticApplyPatchCompletedResponse(baseResp []byte, originalCall map[string]any, callID string, summary string) []byte {
	output := make([]any, 0, 3)
	if len(originalCall) != 0 {
		preservedCall := cloneMap(originalCall)
		preservedCall["type"] = "apply_patch_call"
		preservedCall["status"] = "completed"
		if strings.TrimSpace(fmt.Sprintf("%v", preservedCall["call_id"])) == "" && strings.TrimSpace(callID) != "" {
			preservedCall["call_id"] = strings.TrimSpace(callID)
		}
		if strings.TrimSpace(fmt.Sprintf("%v", preservedCall["id"])) == "" {
			preservedCall["id"] = fmt.Sprintf("apc_%d", time.Now().UnixNano())
		}
		output = append(output, preservedCall)
	}
	resp := map[string]any{
		"id":         fmt.Sprintf("resp_apply_patch_local_%d", time.Now().UnixNano()),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"output":     output,
	}
	if gjson.ValidBytes(baseResp) {
		if originalID := strings.TrimSpace(gjson.GetBytes(baseResp, "id").String()); originalID != "" {
			resp["id"] = originalID
		}
		if model := strings.TrimSpace(gjson.GetBytes(baseResp, "model").String()); model != "" {
			resp["model"] = model
		}
	}
	if strings.TrimSpace(callID) != "" {
		resp["output"] = append(resp["output"].([]any), map[string]any{
			"id":      fmt.Sprintf("apco_%d", time.Now().UnixNano()),
			"type":    "apply_patch_call_output",
			"call_id": callID,
			"output":  summary,
		})
	}
	resp["output"] = append(resp["output"].([]any), map[string]any{
		"id":   fmt.Sprintf("msg_apply_patch_local_%d", time.Now().UnixNano()),
		"type": "message",
		"role": "assistant",
		"content": []any{
			map[string]any{
				"type": "output_text",
				"text": "apply_patch completed via bridge local fallback: " + strings.TrimSpace(summary),
			},
		},
	})
	out, err := json.Marshal(resp)
	if err != nil {
		return buildApplyPatchRetryFailedResponse(baseResp, "local_fallback_marshal_error", err.Error())
	}
	return out
}

func prependSystemInstructionOnce(req map[string]any, instruction string) {
	if req == nil {
		return
	}
	instruction = strings.TrimSpace(instruction)
	if instruction == "" {
		return
	}
	input, _ := req["input"].([]any)
	for _, raw := range input {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", msg["type"])) != "message" {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", msg["role"])), "system") {
			continue
		}
		content, _ := msg["content"].([]any)
		for _, partRaw := range content {
			part, ok := partRaw.(map[string]any)
			if !ok {
				continue
			}
			text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
			if text == instruction {
				return
			}
		}
	}
	systemMsg := map[string]any{
		"type": "message",
		"role": "system",
		"content": []any{
			map[string]any{
				"type": "input_text",
				"text": instruction,
			},
		},
	}
	req["input"] = append([]any{systemMsg}, input...)
}

type bridgeChatStreamWriter struct {
	mu          sync.Mutex
	header      http.Header
	statusCode  int
	wroteHeader bool
	pipe        *io.PipeWriter
}

func newBridgeChatStreamWriter(pipe *io.PipeWriter) *bridgeChatStreamWriter {
	return &bridgeChatStreamWriter{
		header: make(http.Header),
		pipe:   pipe,
	}
}

func (w *bridgeChatStreamWriter) Header() http.Header {
	return w.header
}

func (w *bridgeChatStreamWriter) WriteHeader(statusCode int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.statusCode = statusCode
}

func (w *bridgeChatStreamWriter) Write(data []byte) (int, error) {
	w.mu.Lock()
	if !w.wroteHeader {
		w.wroteHeader = true
		w.statusCode = http.StatusOK
	}
	w.mu.Unlock()
	if w.pipe == nil {
		return len(data), nil
	}
	return w.pipe.Write(data)
}

func (w *bridgeChatStreamWriter) Flush() {}

func (w *bridgeChatStreamWriter) StatusCode() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.statusCode == 0 {
		return http.StatusOK
	}
	return w.statusCode
}

func writeResponsesStreamFromChatSSE(w http.ResponseWriter, upstream io.Reader, planOutputRequired bool, requestedReasoningSummary string) error {
	return writeResponsesStreamFromChatSSEWithWorkflowAndContext(context.Background(), w, upstream, planOutputRequired, false, requestedReasoningSummary, ToolWorkflowState{}, "", nil)
}

func writeResponsesStreamFromChatSSEWithWorkflow(w http.ResponseWriter, upstream io.Reader, planOutputRequired bool, nativeQuestionRequired bool, requestedReasoningSummary string, workflowState ToolWorkflowState, verificationShellArgs string, originalReq map[string]any) error {
	return writeResponsesStreamFromChatSSEWithWorkflowAndContext(context.Background(), w, upstream, planOutputRequired, nativeQuestionRequired, requestedReasoningSummary, workflowState, verificationShellArgs, originalReq)
}

func writeResponsesStreamFromChatSSEWithWorkflowAndContext(ctx context.Context, w http.ResponseWriter, upstream io.Reader, planOutputRequired bool, nativeQuestionRequired bool, requestedReasoningSummary string, workflowState ToolWorkflowState, verificationShellArgs string, originalReq map[string]any) error {
	// Forward upstream chat.completions SSE as true incremental Responses SSE.
	// In plan mode, mutating tool calls must be suppressed even if the upstream
	// model emits them natively, so the final stream remains plan-only.
	appendProxyStageTrace("stream.chat_sse_to_responses.start", map[string]any{
		"plan_output_required":        planOutputRequired,
		"native_question_required":    nativeQuestionRequired,
		"requested_reasoning_summary": strings.TrimSpace(requestedReasoningSummary),
	})
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		// Shouldn't happen for Codex/VScode clients, but keep the old behavior as a safe fallback.
		raw, err := io.ReadAll(upstream)
		if err != nil {
			return err
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(raw)
		return nil
	}

	respID := ""
	model := ""
	createdAt := int64(0)
	msgID := ""
	commentaryMsgID := ""
	finalTextBuilder := strings.Builder{}
	commentaryText := strings.Builder{}
	reasoningText := strings.Builder{}
	reasoningSummaryText := strings.Builder{}
	reasoningSeen := false
	finishReasonStop := false
	finishReasonLength := false
	toolStates := map[int]*StreamToolCallState{}
	toolOrder := make([]int, 0, 4)
	nextOutputIndex := 0
	streamAdapter := proxyCompatibilityAdapters.Stream
	var latestUsage map[string]any
	var latestTimings map[string]any
	lifecycleStarted := false
	messageStarted := false
	messageOutputIndex := -1
	commentaryStarted := false
	commentaryOutputIndex := -1
	lastProgressCommentary := ""
	reasoningStarted := false
	reasoningOutputIndex := -1
	reasoningItemID := ""
	sequence := 0
	choiceState := newAccumulatedChoiceState(0)
	frameSummaries := make([]QwenStreamFrameTraceSummary, 0, 32)
	var startToolStateIfReady func(state *StreamToolCallState)
	var finalizeToolCallPhase func()

	writeEvent := func(eventType string, payload map[string]any) {
		if _, ok := payload["type"]; !ok {
			payload["type"] = eventType
		}
		payload["sequence_number"] = sequence
		sequence++
		data, _ := json.Marshal(payload)
		_, _ = w.Write([]byte("event: " + eventType + "\n"))
		_, _ = w.Write([]byte("data: " + string(data) + "\n\n"))
		flusher.Flush()
	}
	writeCanonicalStreamEvent := func(event StreamEvent) {
		writeCanonicalResponseStreamEvent(writeEvent, event)
	}
	startToolStateIfReady = func(state *StreamToolCallState) {
		if state == nil || state.Exposed || !streamAdapter.ShouldExposeToolCall(state) {
			return
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			return
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "shell") {
			rawArgs := strings.TrimSpace(normalizePossiblyMixedToolArguments(state.ArgsBuilder.String()))
			if rawArgs == "" || !gjson.Valid(rawArgs) {
				return
			}
			args := parseToolArgsMapString(streamAdapter.CanonicalToolArguments(state, state.ArgsBuilder.String()))
			if len(args) == 0 || shellToolArgumentsLookLikePlainQuestion(args) {
				return
			}
		}
		state.OutputIndex = nextOutputIndex
		state.Exposed = true
		if event, ok := streamAdapter.BuildToolItemAddedEvent(state, respID); ok {
			writeCanonicalStreamEvent(event)
		}
		nextOutputIndex++
	}
	writeReasoningSummaryPartAdded := func() {
		writeEvent("response.reasoning_summary_part.added", map[string]any{
			"response_id":   respID,
			"item_id":       reasoningItemID,
			"output_index":  reasoningOutputIndex,
			"summary_index": 0,
			"part": map[string]any{
				"type": "summary_text",
				"text": "",
			},
		})
	}
	writeReasoningSummaryPartDone := func(text string) {
		writeEvent("response.reasoning_summary_part.done", map[string]any{
			"response_id":   respID,
			"item_id":       reasoningItemID,
			"output_index":  reasoningOutputIndex,
			"summary_index": 0,
			"part": map[string]any{
				"type": "summary_text",
				"text": text,
			},
		})
	}

	startLifecycleIfNeeded := func() {
		if lifecycleStarted {
			return
		}
		if strings.TrimSpace(respID) == "" {
			respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
		}
		if createdAt == 0 {
			createdAt = time.Now().Unix()
		}

		responseSkeleton := map[string]any{
			"id":         respID,
			"object":     "response",
			"created_at": createdAt,
			"model":      model,
			"status":     "in_progress",
			"output":     []any{},
		}
		if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
			responseSkeleton["reasoning"] = map[string]any{"summary": summary}
		}
		writeEvent("response.created", map[string]any{"response": responseSkeleton})
		writeEvent("response.in_progress", map[string]any{"response": responseSkeleton})
		lifecycleStarted = true
	}

	startMessageIfNeeded := func() {
		startLifecycleIfNeeded()
		if messageStarted {
			return
		}
		if messageOutputIndex < 0 {
			messageOutputIndex = nextOutputIndex
			nextOutputIndex++
		}
		msgID = fmt.Sprintf("msg_%s_%d", respID, messageOutputIndex)
		item := map[string]any{
			"id":      msgID,
			"type":    "message",
			"role":    "assistant",
			"channel": "final",
			"content": []any{
				map[string]any{"type": "output_text", "text": ""},
			},
		}
		writeEvent("response.output_item.added", map[string]any{
			"response_id":  respID,
			"output_index": messageOutputIndex,
			"item":         item,
		})
		writeEvent("response.content_part.added", map[string]any{
			"response_id":   respID,
			"item_id":       msgID,
			"output_index":  messageOutputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": ""},
		})
		messageStarted = true
	}

	startCommentaryIfNeeded := func() {
		startLifecycleIfNeeded()
		if commentaryStarted {
			return
		}
		if commentaryOutputIndex < 0 {
			commentaryOutputIndex = nextOutputIndex
			nextOutputIndex++
		}
		commentaryMsgID = fmt.Sprintf("msg_%s_%d", respID, commentaryOutputIndex)
		item := map[string]any{
			"id":      commentaryMsgID,
			"type":    "message",
			"role":    "assistant",
			"channel": "commentary",
			"content": []any{
				map[string]any{"type": "output_text", "text": ""},
			},
		}
		writeEvent("response.output_item.added", map[string]any{
			"response_id":  respID,
			"output_index": commentaryOutputIndex,
			"item":         item,
		})
		writeEvent("response.content_part.added", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryMsgID,
			"output_index":  commentaryOutputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": ""},
		})
		commentaryStarted = true
	}

	startReasoningIfNeeded := func() {
		startLifecycleIfNeeded()
		if reasoningStarted {
			return
		}
		if reasoningOutputIndex < 0 {
			reasoningOutputIndex = nextOutputIndex
			nextOutputIndex++
		}
		reasoningItemID = fmt.Sprintf("rs_%s_%d", respID, reasoningOutputIndex)
		item := map[string]any{
			"id":      reasoningItemID,
			"type":    "reasoning",
			"status":  "in_progress",
			"summary": []any{},
		}
		writeEvent("response.output_item.added", map[string]any{
			"response_id":  respID,
			"output_index": reasoningOutputIndex,
			"item":         item,
		})
		writeReasoningSummaryPartAdded()
		reasoningStarted = true
	}
	emitCommentaryDelta := func(text string) {
		text = strings.TrimSpace(cleanFallbackInput(text, ""))
		if strings.TrimSpace(text) == "" {
			return
		}
		startCommentaryIfNeeded()
		commentaryText.WriteString(text)
		writeEvent("response.output_text.delta", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryMsgID,
			"output_index":  commentaryOutputIndex,
			"content_index": 0,
			"delta":         text,
		})
	}
	emitWorkflowProgressCommentary := func(latestToolName string, finalizing bool) {
		progress := buildWorkflowProgressCommentary(planOutputRequired, nativeQuestionRequired, workflowState, latestToolName, finalizing)
		progress = strings.TrimSpace(progress)
		if progress == "" || progress == lastProgressCommentary {
			return
		}
		lastProgressCommentary = progress
		if commentaryText.Len() > 0 {
			emitCommentaryDelta("\n" + progress)
			return
		}
		emitCommentaryDelta(progress)
	}
	emitFinalDelta := func(text string) {
		text = strings.TrimSpace(cleanFallbackInput(text, ""))
		if strings.TrimSpace(text) == "" {
			return
		}
		startMessageIfNeeded()
		finalTextBuilder.WriteString(text)
		writeEvent("response.output_text.delta", map[string]any{
			"response_id":   respID,
			"item_id":       msgID,
			"output_index":  messageOutputIndex,
			"content_index": 0,
			"delta":         text,
		})
	}
	finalizeToolCallPhase = func() {
		finalizeNormalizedToolCallPhase(
			toolOrder,
			toolStates,
			respID,
			planOutputRequired,
			streamAdapter,
			reasoningText.String(),
			finalTextBuilder.String(),
			originalReq,
			writeCanonicalStreamEvent,
			emitFinalDelta,
		)
	}

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			break
		}
		if !gjson.Valid(payload) {
			continue
		}
		choiceState.addChatCompletionChunk(payload)
		normalizedFrame := buildQwenNormalizedStreamFrame(normalizeQwenStreamChunk(payload))

		startLifecycleIfNeeded()

		for _, normalizedEvent := range normalizedFrame.Events {
			switch normalizedEvent.Kind {
			case QwenNormalizedStreamEventMeta:
				if normalizedFrame.ChunkID != "" && strings.TrimSpace(respID) == "" {
					if strings.HasPrefix(normalizedFrame.ChunkID, "resp_") {
						respID = normalizedFrame.ChunkID
					} else {
						respID = "resp_" + normalizedFrame.ChunkID
					}
				}
				if normalizedFrame.CreatedAt > 0 && createdAt == 0 {
					createdAt = normalizedFrame.CreatedAt
				}
				if normalizedFrame.Model != "" && model == "" {
					model = normalizedFrame.Model
				}
				if normalizedFrame.Usage != nil {
					latestUsage = normalizedFrame.Usage
				}
				if normalizedFrame.Timings != nil {
					latestTimings = normalizedFrame.Timings
					if latestUsage == nil {
						latestUsage = buildResponsesUsageFromTimingMap(normalizedFrame.Timings)
					}
				}
			case QwenNormalizedStreamEventToolCallDelta:
				applyNormalizedToolCallDelta(
					normalizedEvent,
					toolStates,
					&toolOrder,
					respID,
					planOutputRequired,
					workflowState,
					streamAdapter,
					startToolStateIfReady,
					writeCanonicalStreamEvent,
					func(progress string) {
						if commentaryText.Len() > 0 {
							emitCommentaryDelta("\n" + progress)
						} else {
							emitCommentaryDelta(progress)
						}
					},
					&lastProgressCommentary,
				)
			case QwenNormalizedStreamEventFinish:
				if strings.EqualFold(normalizedFrame.FinishReason, "stop") {
					finishReasonStop = true
				}
				if strings.EqualFold(normalizedFrame.FinishReason, "length") {
					finishReasonLength = true
				}
			}
		}

		frameDecision := decideQwenStreamFramePresentation(normalizedFrame, workflowState, planOutputRequired, len(toolOrder))
		frameSummaries = append(frameSummaries, summarizeQwenNormalizedStreamFrame(normalizedFrame, frameDecision))
		if frameDecision.Route == QwenStreamFrameRouteNone {
			if frameDecision.ShouldFinalizeToolCalls {
				finalizeToolCallPhase()
			}
			continue
		}

		if frameDecision.Route == QwenStreamFrameRouteReasoning {
			reasoningSeen = true
			startReasoningIfNeeded()
			reasoningText.WriteString(frameDecision.Text)
			reasoningSummaryText.WriteString(frameDecision.Text)
			writeEvent("response.reasoning_summary_text.delta", map[string]any{
				"response_id":   respID,
				"item_id":       reasoningItemID,
				"output_index":  reasoningOutputIndex,
				"summary_index": 0,
				"delta":         frameDecision.Text,
			})
			if commentaryText.Len() == 0 {
				latestToolName := ""
				if len(toolOrder) > 0 {
					if state := toolStates[toolOrder[len(toolOrder)-1]]; state != nil {
						latestToolName = state.Name
					}
				}
				emitWorkflowProgressCommentary(latestToolName, false)
			}
			if frameDecision.ShouldFinalizeToolCalls {
				finalizeToolCallPhase()
			}
			continue
		}

		delta := frameDecision.Text
		if frameDecision.Route == QwenStreamFrameRouteFinal {
			finalTextBuilder.WriteString(delta)
		} else if frameDecision.Route == QwenStreamFrameRouteCommentary {
			emitCommentaryDelta(delta)
		} else {
			emitFinalDelta(delta)
		}
		if frameDecision.ShouldFinalizeToolCalls {
			finalizeToolCallPhase()
		}
	}
	if err := scanner.Err(); err != nil {
		choiceState.markTruncated()
		return err
	}

	startLifecycleIfNeeded()
	normalizedArtifacts := choiceState.buildNormalizedArtifacts(buildAccumulatedAssistantMessage(choiceState), planOutputRequired, nativeQuestionRequired)
	normalizedView := buildNormalizedArtifactView(normalizedArtifacts)
	addQwenNormalizedStreamCaptureStage(ctx, frameSummaries, normalizedView)

	finalText := strings.TrimSpace(finalTextBuilder.String())
	commentaryTextValue := strings.TrimSpace(commentaryText.String())
	reasoningDoneText := strings.TrimSpace(reasoningText.String())
	reasoningSummaryDoneText := reasoningSummaryText.String()
	finalText = strings.TrimSpace(cleanFallbackInput(finalText, ""))
	commentaryTextValue = strings.TrimSpace(cleanFallbackInput(commentaryTextValue, ""))
	reasoningDoneText = strings.TrimSpace(cleanFallbackInput(reasoningDoneText, ""))
	reasoningSummaryDoneText = strings.TrimSpace(cleanFallbackInput(reasoningSummaryDoneText, ""))
	if normalizedView.PreferredThinking != nil {
		if normalizedText, ok := normalizedView.PreferredThinking.Payload["text"].(string); ok && strings.TrimSpace(normalizedText) != "" {
			reasoningDoneText = normalizedText
			reasoningSummaryDoneText = normalizedText
		}
		if wellFormed, ok := normalizedView.PreferredThinking.Payload["well_formed"].(bool); ok && !wellFormed {
			reasoningDoneText = ""
			reasoningSummaryDoneText = ""
		}
	}
	commentaryReasoningText := proxyCompatibilityAdapters.Reasoning.BuildCommentaryPreview(reasoningDoneText)
	promotedReasoning := false
	if !planOutputRequired && strings.TrimSpace(finalText) == "" && reasoningDoneText != "" && len(toolOrder) == 0 {
		// Codex clients may hide reasoning lanes. If upstream produced reasoning-only output,
		// surface it on the normal output_text lane to avoid empty assistant messages.
		promotedReasoning = true
		finalText = commentaryReasoningText
		if strings.TrimSpace(finalText) == "" {
			finalText = reasoningDoneText
		}
		reasoningDoneText = ""
		reasoningSummaryDoneText = ""
		commentaryReasoningText = ""
		finalTextBuilder.Reset()
		emitFinalDelta(finalText)
	}
	if planOutputRequired {
		nativeVisiblePlan := ""
		if normalizedView.PreferredPlan != nil {
			if planText, ok := normalizedView.PreferredPlan.Payload["text"].(string); ok && strings.TrimSpace(planText) != "" {
				nativeVisiblePlan = extractVisibleProposedPlanBlock(planText)
				if nativeVisiblePlan == "" {
					nativeVisiblePlan = sanitizeStructuredPlanText(planText)
				}
			}
		}
		if nativeVisiblePlan == "" {
			nativeVisiblePlan = extractVisibleProposedPlanBlock(finalText)
		}
		if nativeVisiblePlan != "" {
			finalText = sanitizeStructuredPlanText(nativeVisiblePlan)
		} else if strings.TrimSpace(finalText) == "" && !finishReasonStop && finishReasonLength {
			finalText = planModeLengthDiagnosticText()
		}
		if strings.TrimSpace(finalText) != "" {
			if nativeVisiblePlan != "" || finalText == planModeLengthDiagnosticText() {
				finalText = ensureProposedPlanWrapper(finalText)
			}
			if strings.TrimSpace(finalText) != "" {
				finalTextBuilder.Reset()
				finalTextBuilder.WriteString(finalText)
				emitFinalDelta(finalText)
			}
		}
	}
	visibleToolCount := 0
	for _, toolIdx := range toolOrder {
		state := toolStates[toolIdx]
		if state == nil {
			continue
		}
		if !state.Exposed {
			continue
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			continue
		}
		visibleToolCount++
	}
	if !planOutputRequired && visibleToolCount == 0 && shouldRecoverVerificationShellCall(workflowState, verificationShellArgs, finalText, reasoningDoneText) {
		recoveredState := &StreamToolCallState{
			Index:       len(toolOrder),
			OutputIndex: -1,
			ItemID:      fmt.Sprintf("fc_%s_%d", respID, len(toolOrder)),
			CallID:      fmt.Sprintf("call_%s_recover_verify_shell", respID),
			Name:        "shell",
		}
		recoveredState.ArgsBuilder.WriteString(strings.TrimSpace(verificationShellArgs))
		toolStates[recoveredState.Index] = recoveredState
		toolOrder = append(toolOrder, recoveredState.Index)
		startToolStateIfReady(recoveredState)
		if event, ok := streamAdapter.BuildToolArgsDoneEvent(recoveredState, respID); ok {
			writeCanonicalStreamEvent(event)
		}
		if event, ok := streamAdapter.BuildToolItemDoneEvent(recoveredState, respID); ok {
			writeCanonicalStreamEvent(event)
		}
		finalTextBuilder.Reset()
		commentaryText.Reset()
		reasoningText.Reset()
		reasoningSummaryText.Reset()
		finalText = ""
		commentaryTextValue = ""
		reasoningDoneText = ""
		reasoningSummaryDoneText = ""
		commentaryReasoningText = ""
		visibleToolCount = 1
	}
	if !planOutputRequired && visibleToolCount == 0 && shouldRecoverCodexSkillProofShellCall(originalReq, verificationShellArgs, finalText, reasoningDoneText) {
		recoveredState := &StreamToolCallState{
			Index:       len(toolOrder),
			OutputIndex: -1,
			ItemID:      fmt.Sprintf("fc_%s_%d", respID, len(toolOrder)),
			CallID:      fmt.Sprintf("call_%s_recover_skill_shell", respID),
			Name:        "shell",
		}
		recoveredState.ArgsBuilder.WriteString(strings.TrimSpace(verificationShellArgs))
		toolStates[recoveredState.Index] = recoveredState
		toolOrder = append(toolOrder, recoveredState.Index)
		startToolStateIfReady(recoveredState)
		if event, ok := streamAdapter.BuildToolArgsDoneEvent(recoveredState, respID); ok {
			writeCanonicalStreamEvent(event)
		}
		if event, ok := streamAdapter.BuildToolItemDoneEvent(recoveredState, respID); ok {
			writeCanonicalStreamEvent(event)
		}
		finalTextBuilder.Reset()
		commentaryText.Reset()
		reasoningText.Reset()
		reasoningSummaryText.Reset()
		finalText = ""
		commentaryTextValue = ""
		reasoningDoneText = ""
		reasoningSummaryDoneText = ""
		commentaryReasoningText = ""
		visibleToolCount = 1
	}
	if visibleToolCount == 0 && workflowState.FinalAnswerSafe && strings.TrimSpace(finalText) != "" {
		emitWorkflowProgressCommentary("", true)
		commentaryTextValue = strings.TrimSpace(commentaryText.String())
	}
	finalText = strings.TrimSpace(cleanFallbackInput(finalText, ""))
	emitReasoningCommentary := visibleToolCount == 0
	shouldEmitMessage := visibleToolCount == 0 || (strings.TrimSpace(finalText) != "" && finishReasonStop) || planOutputRequired
	if shouldEmitMessage {
		startMessageIfNeeded()
	}
	if nativeQuestionRequired && normalizedView.PreferredQuestion != nil && commentaryText.Len() == 0 {
		emitWorkflowProgressCommentary("", false)
		commentaryTextValue = strings.TrimSpace(commentaryText.String())
	}
	if reasoningSeen && !promotedReasoning {
		startReasoningIfNeeded()
		writeEvent("response.reasoning_summary_text.done", map[string]any{
			"response_id":   respID,
			"item_id":       reasoningItemID,
			"output_index":  reasoningOutputIndex,
			"summary_index": 0,
			"text":          reasoningSummaryDoneText,
		})
		writeReasoningSummaryPartDone(reasoningSummaryDoneText)
		reasoningDoneItem := map[string]any{
			"id":     reasoningItemID,
			"type":   "reasoning",
			"status": "completed",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": reasoningSummaryDoneText,
				},
			},
		}
		if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
			reasoningDoneItem["summary_mode"] = summary
		}
		writeEvent("response.output_item.done", map[string]any{
			"response_id":  respID,
			"output_index": reasoningOutputIndex,
			"item":         reasoningDoneItem,
		})
	}
	var commentaryItem map[string]any
	if emitReasoningCommentary && !promotedReasoning && strings.TrimSpace(commentaryTextValue) == "" && strings.TrimSpace(commentaryReasoningText) != "" {
		emitCommentaryDelta(commentaryReasoningText)
		commentaryTextValue = strings.TrimSpace(commentaryText.String())
	}
	if commentaryStarted {
		startCommentaryIfNeeded()
		writeEvent("response.output_text.done", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryMsgID,
			"output_index":  commentaryOutputIndex,
			"content_index": 0,
			"text":          commentaryTextValue,
		})
		writeEvent("response.content_part.done", map[string]any{
			"response_id":   respID,
			"item_id":       commentaryMsgID,
			"output_index":  commentaryOutputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": commentaryTextValue},
		})
		commentaryItem = map[string]any{
			"id":      commentaryMsgID,
			"type":    "message",
			"role":    "assistant",
			"channel": "commentary",
			"content": []any{
				map[string]any{"type": "output_text", "text": commentaryTextValue},
			},
		}
		writeEvent("response.output_item.done", map[string]any{
			"response_id":  respID,
			"output_index": commentaryOutputIndex,
			"item":         commentaryItem,
		})
	}
	var finalItem map[string]any
	if shouldEmitMessage {
		writeEvent("response.output_text.done", map[string]any{
			"response_id":   respID,
			"item_id":       msgID,
			"output_index":  messageOutputIndex,
			"content_index": 0,
			"text":          finalText,
		})
		writeEvent("response.content_part.done", map[string]any{
			"response_id":   respID,
			"item_id":       msgID,
			"output_index":  messageOutputIndex,
			"content_index": 0,
			"part":          map[string]any{"type": "output_text", "text": finalText},
		})

		finalItem = map[string]any{
			"id":   msgID,
			"type": "message",
			"role": "assistant",
			"channel": func() string {
				if visibleToolCount > 0 || (commentaryStarted && strings.TrimSpace(finalText) == "") {
					return "commentary"
				}
				return "final"
			}(),
			"content": []any{
				map[string]any{"type": "output_text", "text": finalText},
			},
		}
		writeEvent("response.output_item.done", map[string]any{
			"response_id":  respID,
			"output_index": messageOutputIndex,
			"item":         finalItem,
		})
	}

	fullResponse := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": createdAt,
		"model":      model,
		"status": func() string {
			if visibleToolCount > 0 {
				return "requires_action"
			}
			return "completed"
		}(),
		"output":      []any{},
		"output_text": finalText,
	}
	if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
		fullResponse["reasoning"] = map[string]any{"summary": summary}
	}
	outputByIndex := map[int]any{}
	maxOutputIndex := -1
	if finalItem != nil {
		outputByIndex[messageOutputIndex] = finalItem
		maxOutputIndex = messageOutputIndex
	}
	if commentaryItem != nil {
		outputByIndex[commentaryOutputIndex] = commentaryItem
		if commentaryOutputIndex > maxOutputIndex {
			maxOutputIndex = commentaryOutputIndex
		}
	}
	if reasoningSeen && !promotedReasoning && reasoningOutputIndex >= 0 && reasoningSummaryDoneText != "" {
		reasoningItem := map[string]any{
			"id":   reasoningItemID,
			"type": "reasoning",
			"summary": []any{
				map[string]any{
					"type": "summary_text",
					"text": reasoningSummaryDoneText,
				},
			},
		}
		if summary := normalizeResponsesReasoningSummary(requestedReasoningSummary); summary != "" {
			reasoningItem["summary_mode"] = summary
		}
		outputByIndex[reasoningOutputIndex] = reasoningItem
		if reasoningOutputIndex > maxOutputIndex {
			maxOutputIndex = reasoningOutputIndex
		}
	}
	for _, toolIdx := range toolOrder {
		state := toolStates[toolIdx]
		if state == nil {
			continue
		}
		if !state.Exposed {
			continue
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			continue
		}
		outputByIndex[state.OutputIndex] = streamAdapter.BuildToolOutputItem(state, normalizePossiblyMixedToolArguments(state.ArgsBuilder.String()))
		if state.OutputIndex > maxOutputIndex {
			maxOutputIndex = state.OutputIndex
		}
	}
	orderedOutput := make([]any, 0, maxOutputIndex+1)
	for outputIndex := 0; outputIndex <= maxOutputIndex; outputIndex++ {
		if item, ok := outputByIndex[outputIndex]; ok {
			orderedOutput = append(orderedOutput, item)
		}
	}
	fullResponse["output"] = orderedOutput
	if latestUsage != nil {
		fullResponse["usage"] = latestUsage
	}
	if latestTimings != nil {
		fullResponse["timings"] = latestTimings
	}
	if event, ok := streamAdapter.BuildResponseCompletedEvent(fullResponse); ok {
		writeCanonicalStreamEvent(event)
	} else {
		writeEvent("response.completed", map[string]any{"response": fullResponse})
	}
	appendProxyStageTrace("stream.chat_sse_to_responses.done", map[string]any{
		"response_status": fullResponse["status"],
		"reasoning_seen":  reasoningSeen,
		"tool_count":      len(toolOrder),
		"final_text_len":  len(strings.TrimSpace(finalText)),
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	return nil
}

var architectureRequiresMinBuild = map[string]string{
	"delta_net":   "b5120",
	"gated_delta": "b5120",
}

func bodyLooksLikeArchitectureUnsupported(body []byte) bool {
	lower := strings.ToLower(strings.TrimSpace(string(body)))
	if lower == "" {
		return false
	}
	needles := []string{
		"delta",
		"unknown arch",
		"architecture_unsupported",
		"not implemented",
		"unsupported architecture",
		"gated delta",
	}
	for _, needle := range needles {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}

func buildArchitectureUnsupportedErrorBody(raw []byte) []byte {
	rawText := strings.TrimSpace(string(raw))
	if rawText == "" {
		rawText = "backend returned 500 with no body"
	}
	msg := "Backend does not support this model architecture. Rebuild llama.cpp with DeltaNet support (>= b5120). Raw error: " + rawText
	resp := map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "architecture_unsupported",
			"code":    503,
		},
	}
	encoded, err := json.Marshal(resp)
	if err != nil {
		return []byte(`{"error":{"message":"Backend does not support this model architecture (DeltaNet).","type":"architecture_unsupported","code":503}}`)
	}
	return encoded
}

func modelLikelyRequiresDeltaNet(model string) bool {
	lower := strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(lower, "qwen3.6-27b") || strings.Contains(lower, "qwen3.6")
}

func (pm *ProxyManager) buildResponsesBridgeHandler(
	modelID string,
	bodyBytes []byte,
	nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error,
) func(string, http.ResponseWriter, *http.Request) error {
	return func(_ string, w http.ResponseWriter, r *http.Request) error {
		responsesRequestedStream := gjson.GetBytes(bodyBytes, "stream").Bool() || strings.Contains(strings.ToLower(strings.TrimSpace(r.Header.Get("Accept"))), "text/event-stream")
		var initialReq map[string]any
		_ = json.Unmarshal(bodyBytes, &initialReq)
		initialMode := responsesRequestContractMode(initialReq)
		if initialMode == "default" && rawResponsesBodyLooksLikePlanMode(bodyBytes) {
			initialMode = "plan"
		}
		shellFirstBeforePatch := requestWantsShellInspectionBeforeApplyPatch(initialReq)
		isApplyPatchIntent := initialMode != "plan" && shouldEnableStrictApplyPatchIntent(initialReq, bodyBytes)
		forceLocalApplyPatchFallback := shouldForceLocalApplyPatchFallback(initialReq)
		localWebSearchFallbackEnabled, _, _ := pm.getWebSearchFallbackSettings()
		shouldResolveWebSearchLocally := localWebSearchFallbackEnabled && requestIncludesWebSearchTool(initialReq)
		useNativeStreamForward := responsesRequestedStream && shouldUseNativeResponsesBridgeStream(initialReq)
		addCaptureStage(r.Context(), "bridge.responses_request", bodyBytes)
		emitMonitor(r.Context(), modelID, "bridge", "in", r.URL.Path, "responses.request", summarizeJSONForLog(bodyBytes), false)
		execResponsesRequest := func(responsesReq []byte) (int, http.Header, []byte, []byte, error) {
			isPlanModeRequested := extractResponsesRequestModeFromBody(responsesReq) == "plan"
			codexManagedPlanMode := isCodexManagedPlanModeFromResponsesBody(responsesReq)
			enforceProxyPlanMode := isPlanModeRequested && !codexManagedPlanMode
			nativeQuestionRequired := nativeQuestionOutputRequiredFromRequestBody(responsesReq)
			structuredPlanRequired := structuredPlanOutputRequiredFromRequestBody(responsesReq)
			resolvedWebSearchItems := make([]any, 0, 4)
			webSearchContinuations := 0
			if isApplyPatchIntent && !enforceProxyPlanMode {
				var firstReq map[string]any
				if err := json.Unmarshal(responsesReq, &firstReq); err == nil {
					if !requestMapContainsAnyToolOutput(firstReq) {
						firstWorkflowState := proxyCompatibilityAdapters.Continuation.BuildWorkflowState(firstReq)
						firstUserText := extractResponsesUserInputText(firstReq)
						if (requestIncludesWebSearchTool(firstReq) || rawResponsesBodyIncludesWebSearchTool(responsesReq)) &&
							(requestExplicitlyWantsSearchIntent(firstReq, firstWorkflowState) ||
								textIndicatesSearchIntent(firstUserText) ||
								textLooksLikeDeferredSearchPromise(firstUserText) ||
								rawResponsesBodyMentionsSearchIntent(responsesReq) ||
								strings.Contains(strings.ToLower(firstUserText), "research the web") ||
								strings.Contains(strings.ToLower(firstUserText), "web research") ||
								strings.Contains(strings.ToLower(firstUserText), "search the web")) &&
							!requestHistoryContainsResearchArtifacts(firstReq, firstWorkflowState) {
							appendStrictWebSearchOnlyInstruction(firstReq)
						} else {
							appendApplyPatchFirstAttemptConstraint(firstReq)
						}
					}
					if mutated, marshalErr := json.Marshal(firstReq); marshalErr == nil {
						responsesReq = mutated
					}
				}
			}

			// Bridge requests already come pre-shaped by /v1/responses semantics.
			// Applying chat prompt-size control here has caused unstable behavior
			// for large Codex payloads (e.g. process churn/502). Keep bridge forwarding
			// deterministic by sending translated payload directly.
			logTextTransform(pm.transformLogger, modelID, "bridge_prompt_size_control", "skipped for completions_bridge")

			maxWebSearchContinuations := 3
			if requested := requestedWebSearchContinuationsBeforeMutation(initialReq); requested > 0 && requested < maxWebSearchContinuations {
				maxWebSearchContinuations = requested
			}
			maxBridgeAttempts := bridgeResponsesMaxAttempts(responsesReq, shouldResolveWebSearchLocally)
			perAttemptTimeout := bridgeResponsesAttemptTimeout(responsesReq, shouldResolveWebSearchLocally)
			var lastCode int
			var lastHeader http.Header
			var lastBody []byte
			var lastErr error
			usedApplyPatchShellRecovery := false
			forcedSynthesisWebSearchRecoveries := 0
			const maxForcedSynthesisWebSearchRecoveries = 2

			for attempt := 1; attempt <= maxBridgeAttempts; attempt++ {
				translated, err := translateResponsesToChatCompletionsRequest(responsesReq)
				if err != nil {
					return 0, nil, nil, nil, fmt.Errorf("invalid responses request: %w", err)
				}
				// For streamed /v1/responses requests that need bridge-managed tool
				// continuation, avoid live upstream SSE. A local model/process exit in
				// the middle of upstream streaming tears down the client connection and
				// causes Codex reconnects even when the bridge can recover and finish.
				// Keep upstream SSE only for non-streamed outer requests where we need
				// a collapsed final body for question/plan repair.
				upstreamStreamMode := !responsesRequestedStream && (nativeQuestionRequired || structuredPlanRequired)
				translated, err = sjson.SetBytes(translated, "stream", upstreamStreamMode)
				if err != nil {
					return 0, nil, nil, nil, fmt.Errorf("error forcing bridge request stream mode: %w", err)
				}
				logTextTransform(pm.transformLogger, modelID, "bridge_request_controls", summarizeResponsesBridgeControls(responsesReq))
				logTextTransform(pm.transformLogger, modelID, "bridge_sampling_controls", summarizeBridgeSamplingControls(translated))
				logBodyTransform(pm.transformLogger, modelID, "bridge_translate_responses_to_chat", responsesReq, translated)
				addCaptureStage(r.Context(), "bridge.chat_completions_request", translated)
				if strings.Contains(strings.ToLower(string(responsesReq)), "apply_patch") {
					appendApplyPatchTrace("bridge.chat_request", map[string]any{
						"model_id": modelID,
						"summary":  summarizeApplyPatchResponsesRequest(responsesReq),
						"attempt":  attempt,
					})
					appendApplyPatchTrace("bridge.chat_request_translated", map[string]any{
						"model_id": modelID,
						"summary":  summarizeApplyPatchChatCompletionRequest(translated),
						"attempt":  attempt,
					})
				}
				attemptCtx := r.Context()
				var cancelAttempt context.CancelFunc
				if perAttemptTimeout > 0 {
					attemptCtx, cancelAttempt = context.WithTimeout(attemptCtx, perAttemptTimeout)
				}
				bridgeReq := r.Clone(attemptCtx)
				bridgeReq.URL.Path = "/v1/chat/completions"
				bridgeReq.Body = io.NopCloser(bytes.NewReader(translated))
				bridgeReq.ContentLength = int64(len(translated))
				bridgeReq.Header = r.Header.Clone()
				bridgeReq.Header.Del("transfer-encoding")
				bridgeReq.Header.Set("content-length", strconv.Itoa(len(translated)))
				bridgeReq.Header.Set("Content-Type", "application/json")

				rr := httptest.NewRecorder()
				if err := nextHandler(modelID, rr, bridgeReq); err != nil {
					if cancelAttempt != nil {
						cancelAttempt()
					}
					lastErr = err
					logTextTransform(pm.transformLogger, modelID, "bridge_upstream_retry", fmt.Sprintf("class=handler_error attempt=%d/%d error=%v", attempt, maxBridgeAttempts, err))
				} else {
					if cancelAttempt != nil {
						cancelAttempt()
					}
					lastCode = rr.Code
					lastHeader = rr.Header()
					lastBody = rr.Body.Bytes()
					addCaptureStage(r.Context(), "bridge.chat_completions_response", lastBody)
					if rr.Code >= 200 && rr.Code < 300 {
						translatedUpstreamBody := lastBody
						if upstreamStreamMode {
							if collapsedBody, _, collapseErr := collapseChatCompletionStreamToFinalBody(bytes.NewReader(lastBody)); collapseErr != nil {
								lastErr = collapseErr
								logTextTransform(pm.transformLogger, modelID, "bridge_upstream_retry", fmt.Sprintf("class=stream_collapse_error attempt=%d/%d error=%v", attempt, maxBridgeAttempts, collapseErr))
								continue
							} else {
								translatedUpstreamBody = collapsedBody
							}
						}
						if msg := gjson.GetBytes(translatedUpstreamBody, "choices.0.message"); msg.Exists() {
							toolSummary := summarizeChatCompletionToolCalls(lastBody)
							if strings.Contains(strings.ToLower(toolSummary), "apply_patch") {
								appendApplyPatchTrace("upstream.chat_response", map[string]any{
									"model_id":  modelID,
									"summary":   toolSummary,
									"forensics": summarizeApplyPatchBridgeForensics(lastBody),
								})
								logTextTransform(pm.transformLogger, modelID, "bridge_upstream_apply_patch_summary", toolSummary)
								logTextTransform(pm.transformLogger, modelID, "bridge_upstream_apply_patch_forensics", summarizeApplyPatchBridgeForensics(lastBody))
							}
						}
						hasPriorToolOutput := requestContainsAnyToolOutput(bodyBytes)
						hasPriorApplyPatchOutput := requestContainsApplyPatchToolOutput(bodyBytes)
						applyPatchPathHint := strings.TrimSpace(extractApplyPatchPathHintFromResponsesRequestBody(bodyBytes))
						applyPatchRecoveryPathHint := applyPatchPathHint
						applyPatchContentHint := strings.TrimSpace(extractApplyPatchContentHintFromResponsesRequestBody(bodyBytes))
						applyPatchTypeHint := strings.TrimSpace(extractApplyPatchTypeHintFromResponsesRequestBody(bodyBytes))
						if hasPriorToolOutput && hasPriorApplyPatchOutput {
							applyPatchPathHint = ""
							applyPatchContentHint = ""
							applyPatchTypeHint = ""
						}
						workflowState, verificationShellArgs := extractWorkflowRecoveryContextFromRequestBody(responsesReq)
						var originalReq map[string]any
						_ = json.Unmarshal(responsesReq, &originalReq)
						out, err := translateChatCompletionToResponsesResponseWithWorkflow(translatedUpstreamBody, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint, workflowState, verificationShellArgs, originalReq)
						if err == nil {
							if rewritten, changed, rewriteErr := rewriteResponsesToolCallPayload(out, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint); rewriteErr == nil && changed {
								out = rewritten
							}
							out = recoverNativeQuestionResponseIfRequired(out, translatedUpstreamBody, responsesReq)
							if !hasPriorToolOutput && requestWantsShellInspectionBeforeApplyPatch(initialReq) && responseContainsApplyPatchCall(out) {
								targetPath := applyPatchPathHint
								if strings.TrimSpace(targetPath) == "" {
									targetPath = extractApplyPatchPathFromResponse(out)
								}
								if strings.TrimSpace(targetPath) != "" {
									out = forceShellInspectionResponse(out, targetPath)
								}
							}
							if hasPriorToolOutput {
								if exactReply := extractExactFinalReplyHintFromRequestBody(responsesReq); exactReply != "" {
									forceExactReply := workflowState.FinalAnswerSafe && len(workflowState.PendingToolNames) == 0
									out = enforceExactFinalReplyHint(out, exactReply, forceExactReply)
								}
							}
							if !usedApplyPatchShellRecovery &&
								hasPriorToolOutput &&
								hasPriorApplyPatchOutput &&
								responseOutputContainsInvalidApplyPatchMessage(out) {
								targetPath := strings.TrimSpace(applyPatchRecoveryPathHint)
								if targetPath == "" {
									targetPath = strings.TrimSpace(extractApplyPatchPathFromResponse(out))
								}
								usedApplyPatchShellRecovery = true
								logTextTransform(pm.transformLogger, modelID, "bridge_apply_patch_shell_recovery", fmt.Sprintf("attempt=%d/%d class=invalid_operation target=%s mode=synthetic_shell_inspection", attempt, maxBridgeAttempts, truncateBridgeDebugText(targetPath, 160)))
								out = forceShellInspectionResponse(out, targetPath)
							}
							if shouldRetryInvalidApplyPatchOperation(originalReq, translatedUpstreamBody, out) && attempt < maxBridgeAttempts {
								var retryReq map[string]any
								if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
									appendStrictApplyPatchToolOnlyInstruction(retryReq, "invalid_operation")
									if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
										responsesReq = mutated
										logTextTransform(pm.transformLogger, modelID, "bridge_apply_patch_retry", fmt.Sprintf("attempt=%d/%d class=invalid_operation", attempt, maxBridgeAttempts))
										continue
									}
								}
							}
							if !usedApplyPatchShellRecovery &&
								responseOutputContainsInvalidApplyPatchMessage(out) &&
								chatResponseContainsToolCallNamed(translatedUpstreamBody, "apply_patch") &&
								shouldRecoverInvalidApplyPatchViaShell(originalReq, workflowState, "empty_operation") {
								var retryReq map[string]any
								if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
									targetPath := strings.TrimSpace(applyPatchRecoveryPathHint)
									if targetPath == "" {
										targetPath = strings.TrimSpace(extractApplyPatchPathFromResponse(out))
									}
									appendApplyPatchStateInspectionRecoveryInstruction(retryReq, targetPath)
									if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
										usedApplyPatchShellRecovery = true
										maxBridgeAttempts++
										responsesReq = mutated
										logTextTransform(pm.transformLogger, modelID, "bridge_apply_patch_shell_recovery", fmt.Sprintf("attempt=%d/%d class=invalid_operation target=%s", attempt, maxBridgeAttempts, truncateBridgeDebugText(targetPath, 160)))
										continue
									}
								}
							}
							if shouldRetryMissingShellCall(originalReq, translated, translatedUpstreamBody, out, workflowState) && attempt < maxBridgeAttempts {
								var retryReq map[string]any
								if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
									appendMissingShellRecoveryInstruction(retryReq)
									appendEmptyPostToolRecoveryInstruction(retryReq)
									if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
										responsesReq = mutated
										logTextTransform(pm.transformLogger, modelID, "bridge_shell_retry", fmt.Sprintf("attempt=%d/%d class=missing_shell_call", attempt, maxBridgeAttempts))
										continue
									}
								}
							}
							if shouldRetryMissingBrowserOpenCall(originalReq, translatedUpstreamBody, out) && attempt < maxBridgeAttempts {
								var retryReq map[string]any
								if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
									appendMissingBrowserOpenRecoveryInstruction(retryReq)
									if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
										responsesReq = mutated
										logTextTransform(pm.transformLogger, modelID, "bridge_browser_open_retry", fmt.Sprintf("attempt=%d/%d class=missing_browser_open_call", attempt, maxBridgeAttempts))
										continue
									}
								}
							}
							if shouldResolveWebSearchLocally {
								if shouldRetryMissingWebSearchCall(initialReq, translated, translatedUpstreamBody, out, workflowState) && webSearchContinuations < maxWebSearchContinuations {
									webSearchContinuations++
									var retryReq map[string]any
									if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
										appendMissingWebSearchRecoveryInstruction(retryReq)
										appendEmptyPostToolRecoveryInstruction(retryReq)
										if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
											responsesReq = mutated
											logTextTransform(pm.transformLogger, modelID, "bridge_web_search_retry", fmt.Sprintf("attempt=%d/%d class=missing_tool_call", webSearchContinuations, maxWebSearchContinuations))
											continue
										}
									}
								}
								if pendingSearchCalls := extractPendingBridgeWebSearchCalls(out); len(pendingSearchCalls) > 0 {
									resolvedCalls := make([]map[string]any, 0, len(pendingSearchCalls))
									resolvedOutputs := make([]map[string]any, 0, len(pendingSearchCalls))
									callIDs := make([]string, 0, len(pendingSearchCalls))
									for idx, pendingCall := range pendingSearchCalls {
										callID := strings.TrimSpace(fmt.Sprintf("%v", pendingCall["call_id"]))
										if callID == "" {
											callID = fmt.Sprintf("call_web_search_%d_%d", time.Now().UnixNano(), idx)
											pendingCall["call_id"] = callID
										}
										pendingCall["status"] = "completed"
										resolvedCalls = append(resolvedCalls, pendingCall)
										resolvedOutputs = append(resolvedOutputs, map[string]any{
											"id":      fmt.Sprintf("wsout_%s", callID),
											"type":    "web_search_call_output",
											"call_id": callID,
											"output":  mustJSONString(pm.executeBridgeWebSearch(r.Context(), normalizeMapValue(pendingCall["action"]))),
										})
										callIDs = append(callIDs, callID)
									}
									nextContinuation := webSearchContinuations + 1
									forcedSynthesis := responsesRequestHasForcedWebSearchSynthesis(responsesReq)
									if nextContinuation >= maxWebSearchContinuations && !forcedSynthesis {
										itemsToAppend := make([]map[string]any, 0, len(resolvedCalls)+len(resolvedOutputs))
										for idx := range resolvedCalls {
											itemsToAppend = append(itemsToAppend, resolvedCalls[idx], resolvedOutputs[idx])
										}
										if mutatedReq, appendErr := appendResponsesInputItems(responsesReq, itemsToAppend...); appendErr == nil {
											webSearchContinuations = nextContinuation
											for idx := range resolvedCalls {
												resolvedWebSearchItems = append(resolvedWebSearchItems, cloneMap(resolvedCalls[idx]), cloneMap(resolvedOutputs[idx]))
											}
											var synthesisReq map[string]any
											if err := json.Unmarshal(mutatedReq, &synthesisReq); err == nil {
												if requestInputMentionsApplyPatch(originalReq) && !workflowState.ApplyPatchSatisfied {
													clearWebSearchPhaseInstructions(synthesisReq)
													appendStrictApplyPatchToolOnlyInstruction(synthesisReq, "post_web_search_transition")
												} else {
													appendWebSearchSynthesisInstruction(synthesisReq)
												}
												if synthesized, marshalErr := json.Marshal(synthesisReq); marshalErr == nil {
													mutatedReq = synthesized
												}
											}
											responsesReq = mutatedReq
											logClass := "force_synthesis_turn"
											if requestInputMentionsApplyPatch(originalReq) && !workflowState.ApplyPatchSatisfied {
												logClass = "force_apply_patch_turn"
											}
											logTextTransform(pm.transformLogger, modelID, "bridge_web_search_continue", fmt.Sprintf("attempt=%d/%d call_ids=%s class=%s", webSearchContinuations, maxWebSearchContinuations, strings.Join(callIDs, ","), logClass))
											continue
										}
									}
									if nextContinuation > maxWebSearchContinuations {
										if forcedSynthesis && forcedSynthesisWebSearchRecoveries < maxForcedSynthesisWebSearchRecoveries {
											itemsToAppend := make([]map[string]any, 0, len(resolvedCalls)+len(resolvedOutputs))
											for idx := range resolvedCalls {
												itemsToAppend = append(itemsToAppend, resolvedCalls[idx], resolvedOutputs[idx])
											}
											if mutatedReq, appendErr := appendResponsesInputItems(responsesReq, itemsToAppend...); appendErr == nil {
												webSearchContinuations = nextContinuation
												forcedSynthesisWebSearchRecoveries++
												for idx := range resolvedCalls {
													resolvedWebSearchItems = append(resolvedWebSearchItems, cloneMap(resolvedCalls[idx]), cloneMap(resolvedOutputs[idx]))
												}
												var synthesisReq map[string]any
												if err := json.Unmarshal(mutatedReq, &synthesisReq); err == nil {
													appendWebSearchSynthesisInstruction(synthesisReq)
													appendWebSearchSynthesisRetryInstruction(synthesisReq)
													if synthesized, marshalErr := json.Marshal(synthesisReq); marshalErr == nil {
														mutatedReq = synthesized
													}
												}
												responsesReq = mutatedReq
												if attempt >= maxBridgeAttempts {
													maxBridgeAttempts++
												}
												logTextTransform(pm.transformLogger, modelID, "bridge_web_search_continue", fmt.Sprintf("attempt=%d/%d call_ids=%s class=force_synthesis_search_recovery recoveries=%d/%d", webSearchContinuations, maxWebSearchContinuations, strings.Join(callIDs, ","), forcedSynthesisWebSearchRecoveries, maxForcedSynthesisWebSearchRecoveries))
												continue
											}
										}
										if finalizedOut, finalizeErr := replacePendingWebSearchCallsWithResolvedItems(out, resolvedCalls, resolvedOutputs); finalizeErr == nil {
											out = finalizedOut
										}
										webSearchContinuations = nextContinuation
										logTextTransform(pm.transformLogger, modelID, "bridge_web_search_continue", fmt.Sprintf("attempt=%d/%d call_ids=%s class=finalized_without_retry", webSearchContinuations, maxWebSearchContinuations, strings.Join(callIDs, ",")))
									} else {
										itemsToAppend := make([]map[string]any, 0, len(resolvedCalls)+len(resolvedOutputs))
										for idx := range resolvedCalls {
											itemsToAppend = append(itemsToAppend, resolvedCalls[idx], resolvedOutputs[idx])
										}
										if mutatedReq, appendErr := appendResponsesInputItems(responsesReq, itemsToAppend...); appendErr == nil {
											webSearchContinuations = nextContinuation
											for idx := range resolvedCalls {
												resolvedWebSearchItems = append(resolvedWebSearchItems, cloneMap(resolvedCalls[idx]), cloneMap(resolvedOutputs[idx]))
											}
											responsesReq = mutatedReq
											logTextTransform(pm.transformLogger, modelID, "bridge_web_search_continue", fmt.Sprintf("attempt=%d/%d call_ids=%s", webSearchContinuations, maxWebSearchContinuations, strings.Join(callIDs, ",")))
											continue
										}
									}
								}
							}
							pendingFileSearchCalls := extractPendingBridgeFileSearchCalls(out)
							if len(pendingFileSearchCalls) == 0 {
								for _, toolCall := range extractChatToolCallsNamed(translatedUpstreamBody, "file_search", llamaSwapFileSearchFunctionName) {
									callID := strings.TrimSpace(fmt.Sprintf("%v", toolCall["call_id"]))
									if callID == "" {
										callID = fmt.Sprintf("call_file_search_%d", time.Now().UnixNano())
									}
									pendingFileSearchCalls = append(pendingFileSearchCalls, map[string]any{
										"id":      fmt.Sprintf("fc_%s", callID),
										"type":    "file_search_call",
										"call_id": callID,
										"name":    "file_search",
										"status":  "in_progress",
										"action":  normalizeMapValue(toolCall["action"]),
									})
								}
							}
							if len(pendingFileSearchCalls) > 0 {
								resolvedCalls := make([]map[string]any, 0, len(pendingFileSearchCalls))
								resolvedOutputs := make([]map[string]any, 0, len(pendingFileSearchCalls))
								callIDs := make([]string, 0, len(pendingFileSearchCalls))
								fileSearchRootHint := extractFileSearchRootHintFromRequestBody(responsesReq)
								for idx, pendingCall := range pendingFileSearchCalls {
									callID := strings.TrimSpace(fmt.Sprintf("%v", pendingCall["call_id"]))
									if callID == "" {
										callID = fmt.Sprintf("call_file_search_%d_%d", time.Now().UnixNano(), idx)
										pendingCall["call_id"] = callID
									}
									action := normalizeMapValue(pendingCall["action"])
									if strings.TrimSpace(cleanFallbackInput(action["root"], "")) == "" &&
										strings.TrimSpace(cleanFallbackInput(action["search_root"], "")) == "" &&
										strings.TrimSpace(fileSearchRootHint) != "" {
										action["root"] = fileSearchRootHint
										pendingCall["action"] = action
									}
									pendingCall["status"] = "completed"
									resolvedCalls = append(resolvedCalls, pendingCall)
									resolvedOutputs = append(resolvedOutputs, map[string]any{
										"id":      fmt.Sprintf("fsout_%s", callID),
										"type":    "file_search_call_output",
										"call_id": callID,
										"output":  mustJSONString(map[string]any{"type": "file_search_call_output", "payload": executeBridgeFileSearch(action)}),
									})
									callIDs = append(callIDs, callID)
								}
								itemsToAppend := make([]map[string]any, 0, len(resolvedCalls)+len(resolvedOutputs))
								for idx := range resolvedCalls {
									itemsToAppend = append(itemsToAppend, resolvedCalls[idx], resolvedOutputs[idx])
								}
								if mutatedReq, appendErr := appendResponsesInputItems(responsesReq, itemsToAppend...); appendErr == nil {
									responsesReq = mutatedReq
									logTextTransform(pm.transformLogger, modelID, "bridge_file_search_continue", fmt.Sprintf("call_ids=%s", strings.Join(callIDs, ",")))
									continue
								}
								if finalizedOut, finalizeErr := replacePendingFileSearchCallsWithResolvedItems(out, resolvedCalls, resolvedOutputs); finalizeErr == nil {
									out = finalizedOut
								}
							}
							forcedWebSearchSynthesis := responsesRequestHasForcedWebSearchSynthesis(responsesReq)
							if shouldRetryWeakPlaceholderFinal(originalReq, translated, translatedUpstreamBody, out, workflowState) {
								if forcedWebSearchSynthesis && attempt >= maxBridgeAttempts {
									maxBridgeAttempts++
								}
							}
							if shouldRetryWeakPlaceholderFinal(originalReq, translated, translatedUpstreamBody, out, workflowState) && attempt < maxBridgeAttempts {
								var retryReq map[string]any
								if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
									appendWeakFinalRecoveryInstruction(retryReq)
									if forcedWebSearchSynthesis {
										appendWebSearchSynthesisRetryInstruction(retryReq)
									}
									if requestStillNeedsWebSearchBeforeApplyPatch(originalReq, workflowState) {
										appendStrictWebSearchOnlyInstruction(retryReq)
										appendEmptyPostToolRecoveryInstruction(retryReq)
									} else if requestInputMentionsApplyPatch(originalReq) && !workflowState.ApplyPatchSatisfied {
										appendStrictApplyPatchToolOnlyInstruction(retryReq, "weak_placeholder_final_promised_apply_patch")
									}
									if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
										responsesReq = mutated
										logTextTransform(pm.transformLogger, modelID, "bridge_placeholder_retry", fmt.Sprintf("attempt=%d/%d class=weak_placeholder_final", attempt, maxBridgeAttempts))
										continue
									}
								}
							}
							if shouldEnforcePlanModeSyntheticRewrite(isPlanModeRequested, enforceProxyPlanMode, out) {
								upstreamFinishedNormally := strings.EqualFold(strings.TrimSpace(gjson.GetBytes(lastBody, "choices.0.finish_reason").String()), "stop")
								out = enforcePlanModeResponse(out, upstreamFinishedNormally)
							}
							if len(resolvedWebSearchItems) > 0 {
								if mergedOut, mergeErr := prependResponsesOutputItems(out, resolvedWebSearchItems); mergeErr == nil {
									out = mergedOut
								}
							}
							if requestInputMentionsApplyPatch(originalReq) &&
								responseOutputContainsInvalidApplyPatchMessage(out) &&
								!responseContainsToolCall(out) {
								targetPath := strings.TrimSpace(applyPatchRecoveryPathHint)
								if targetPath == "" {
									targetPath = strings.TrimSpace(extractApplyPatchPathFromResponse(out))
								}
								logTextTransform(pm.transformLogger, modelID, "bridge_apply_patch_shell_recovery", fmt.Sprintf("attempt=%d/%d class=invalid_operation target=%s mode=final_guard", attempt, maxBridgeAttempts, truncateBridgeDebugText(targetPath, 160)))
								out = forceShellInspectionResponse(out, targetPath)
							}
							if strings.Contains(strings.ToLower(string(out)), "apply_patch") {
								appendApplyPatchTrace("bridge.responses_output", map[string]any{
									"model_id": modelID,
									"summary":  summarizeApplyPatchResponsesOutput(out),
								})
							}
							addCaptureStage(r.Context(), "bridge.responses_output", out)
							if isApplyPatchIntent &&
								!enforceProxyPlanMode &&
								!(shellFirstBeforePatch && !hasPriorToolOutput) &&
								!shouldDeferApplyPatchValidationForCurrentTurn(originalReq, workflowState, translatedUpstreamBody, out) {
								valid, reasonCode, sampledArgs := evaluateApplyPatchOutput(out)
								if !valid {
									if reasonCode == "wrong_tool_call" && isOversizedShellFileWrite(sampledArgs) {
										reasonCode = "wrong_tool_call_oversized_shell_write"
									}
									if shouldForceStrictApplyPatchRetry(reasonCode) && attempt < maxBridgeAttempts {
										var retryReq map[string]any
										if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
											appendStrictApplyPatchToolOnlyInstruction(retryReq, reasonCode)
											appendApplyPatchTailConstraintToUserTurn(retryReq, applyPatchTailConstraintText)
											if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
												responsesReq = mutated
												logTextTransform(pm.transformLogger, modelID, "bridge_upstream_retry", fmt.Sprintf("class=apply_patch_%s attempt=%d/%d", reasonCode, attempt, maxBridgeAttempts))
												continue
											}
										}
									}
									if !usedApplyPatchShellRecovery && shouldRecoverInvalidApplyPatchViaShell(originalReq, workflowState, reasonCode) {
										var retryReq map[string]any
										if err := json.Unmarshal(responsesReq, &retryReq); err == nil {
											targetPath := strings.TrimSpace(applyPatchRecoveryPathHint)
											if targetPath == "" {
												targetPath = strings.TrimSpace(extractApplyPatchPathFromResponse(out))
											}
											appendApplyPatchStateInspectionRecoveryInstruction(retryReq, targetPath)
											if mutated, marshalErr := json.Marshal(retryReq); marshalErr == nil {
												usedApplyPatchShellRecovery = true
												maxBridgeAttempts++
												responsesReq = mutated
												logTextTransform(pm.transformLogger, modelID, "bridge_apply_patch_shell_recovery", fmt.Sprintf("attempt=%d/%d class=%s target=%s", attempt, maxBridgeAttempts, reasonCode, truncateBridgeDebugText(targetPath, 160)))
												continue
											}
										}
									}
									return rr.Code, rr.Header(), lastBody, buildApplyPatchRetryFailedResponse(out, reasonCode, sampledArgs), nil
								}
								if forceLocalApplyPatchFallback {
									// Optional safety valve for environments where client-side continuation
									// cannot execute apply_patch. Default behavior preserves native continuation.
									if callID, operation, ok := extractFirstApplyPatchInvocationFromResponse(out); ok {
										if originalCall := findFirstApplyPatchCallItem(out); originalCall != nil {
											opForExec := cloneMap(operation)
											if strings.TrimSpace(applyPatchPathHint) != "" {
												opForExec["path"] = normalizeApplyPatchPathForWorkspace(applyPatchPathHint)
											}
											if summary, execErr := executeApplyPatchOperationLocallyWithDisplay(opForExec, applyPatchPathHint); execErr == nil {
												out = buildSyntheticApplyPatchCompletedResponse(out, originalCall, callID, summary)
											} else {
												appendApplyPatchTrace("bridge.local_apply_patch_exec_error", map[string]any{
													"error": execErr.Error(),
													"path":  fmt.Sprintf("%v", opForExec["path"]),
												})
												return rr.Code, rr.Header(), lastBody, buildApplyPatchRetryFailedResponse(out, "local_exec_error", execErr.Error()), nil
											}
										}
									}
								}
							}
							return rr.Code, rr.Header(), lastBody, out, nil
						}
						lastErr = fmt.Errorf("error translating response: %w", err)
						logTextTransform(pm.transformLogger, modelID, "bridge_upstream_retry", fmt.Sprintf("class=translate_error attempt=%d/%d status=%d", attempt, maxBridgeAttempts, rr.Code))
					} else if rr.Code == http.StatusInternalServerError && bodyLooksLikeArchitectureUnsupported(lastBody) {
						return http.StatusServiceUnavailable, rr.Header(), lastBody, buildArchitectureUnsupportedErrorBody(lastBody), nil
					} else if rr.Code >= 500 || rr.Code == http.StatusTooManyRequests || rr.Code == 0 {
						logTextTransform(pm.transformLogger, modelID, "bridge_upstream_retry", fmt.Sprintf("class=upstream_%d attempt=%d/%d", rr.Code, attempt, maxBridgeAttempts))
					} else {
						return rr.Code, rr.Header(), lastBody, lastBody, nil
					}
				}

				if attempt < maxBridgeAttempts {
					time.Sleep(time.Duration(150*attempt) * time.Millisecond)
				}
			}

			if lastCode > 0 && (lastCode < 200 || lastCode >= 300) {
				if (lastCode == http.StatusBadGateway || lastCode == http.StatusGatewayTimeout || lastCode >= 500) &&
					(requestContainsApplyPatchToolOutput(responsesReq) || requestContainsAnyToolOutput(responsesReq)) {
					logTextTransform(pm.transformLogger, modelID, "bridge_post_tool_failure_passthrough", fmt.Sprintf("class=upstream_%d attempts=%d", lastCode, maxBridgeAttempts))
				}
				logTextTransform(pm.transformLogger, modelID, "bridge_upstream_failure", fmt.Sprintf("class=upstream_%d attempts=%d", lastCode, maxBridgeAttempts))
				return lastCode, lastHeader, lastBody, lastBody, nil
			}
			if lastErr != nil {
				logTextTransform(pm.transformLogger, modelID, "bridge_upstream_failure", fmt.Sprintf("class=terminal_error attempts=%d error=%v", maxBridgeAttempts, lastErr))
				return 0, nil, nil, nil, lastErr
			}
			return 0, nil, nil, nil, fmt.Errorf("responses bridge failed after retries")
		}

		if responsesRequestedStream {
			requestedReasoningSummary := extractResponsesRequestReasoningSummaryFromBody(bodyBytes)
			if isApplyPatchIntent || shouldResolveWebSearchLocally {
				status, headers, _, out, err := execResponsesRequest(bodyBytes)
				if err != nil {
					return err
				}
				if status < 200 || status >= 300 {
					for k, values := range headers {
						for _, value := range values {
							w.Header().Add(k, value)
						}
					}
					w.WriteHeader(status)
					_, _ = w.Write(out)
					return nil
				}
				writeResponsesStream(w, out, requestedReasoningSummary)
				return nil
			}

			codexManagedPlanMode := isCodexManagedPlanModeFromResponsesBody(bodyBytes)
			if responsesRequestedStream && useNativeStreamForward {
				translated, err := translateResponsesToChatCompletionsRequest(bodyBytes)
				if err != nil {
					return fmt.Errorf("invalid responses request: %w", err)
				}
				translated, err = sjson.SetBytes(translated, "stream", true)
				if err != nil {
					return fmt.Errorf("error enabling stream for bridge request: %w", err)
				}
				// Native fast-forward path strips tools by design. Codex-managed plan mode
				// should stay transparent and preserve translated tool settings.
				if useNativeStreamForward && !codexManagedPlanMode {
					translated, _ = sjson.DeleteBytes(translated, "tools")
					translated, _ = sjson.SetBytes(translated, "tool_choice", "none")
					translated, _ = sjson.SetBytes(translated, "parallel_tool_calls", false)
				}
				addCaptureStage(r.Context(), "bridge.chat_completions_request", translated)
				logTextTransform(pm.transformLogger, modelID, "bridge_request_controls", summarizeResponsesBridgeControls(bodyBytes))
				logTextTransform(pm.transformLogger, modelID, "bridge_sampling_controls", summarizeBridgeSamplingControls(translated))
				logBodyTransform(pm.transformLogger, modelID, "bridge_translate_responses_to_chat_stream", bodyBytes, translated)

				bridgeReq := r.Clone(r.Context())
				bridgeReq.URL.Path = "/v1/chat/completions"
				bridgeReq.Body = io.NopCloser(bytes.NewReader(translated))
				bridgeReq.ContentLength = int64(len(translated))
				bridgeReq.Header = r.Header.Clone()
				bridgeReq.Header.Del("transfer-encoding")
				bridgeReq.Header.Set("content-length", strconv.Itoa(len(translated)))
				bridgeReq.Header.Set("Content-Type", "application/json")
				bridgeReq.Header.Set("Accept", "text/event-stream")

				pr, pw := io.Pipe()
				upstreamMon := newSSEMonitor(r.Context(), modelID, "upstream_sse", "in", bridgeReq.URL.Path)
				var upstreamStreamCapture bytes.Buffer
				pwWriter := newPipeResponseWriter(pw)
				teePW := &teeResponseWriter{w: pwWriter, onWrite: func(chunk []byte) {
					upstreamMon.writeChunk(chunk)
					_, _ = upstreamStreamCapture.Write(chunk)
				}}

				errCh := make(chan error, 1)
				go func() {
					err := nextHandler(modelID, teePW, bridgeReq)
					_ = pw.CloseWithError(err)
					errCh <- err
				}()

				var snap headerSnapshot
				// Some local reasoning models take several seconds before the first
				// streamed chunk, even though the SSE stream is healthy. Give the
				// upstream chat-completions bridge enough time to surface headers so
				// live reasoning deltas can flow instead of failing fast after 2s.
				headerWait := 30 * time.Second
				select {
				case snap = <-pwWriter.headerCh:
				case err := <-errCh:
					if err != nil {
						return err
					}
					return fmt.Errorf("upstream stream ended before sending headers")
				case <-time.After(headerWait):
					return fmt.Errorf("timeout waiting for upstream stream headers after %s", headerWait)
				}
				if snap.code < 200 || snap.code >= 300 {
					for k, values := range snap.header {
						for _, value := range values {
							w.Header().Add(k, value)
						}
					}
					w.WriteHeader(snap.code)
					_, _ = io.Copy(w, pr)
					<-errCh
					return nil
				}

				outMon := newSSEMonitor(r.Context(), modelID, "outgoing_sse", "out", r.URL.Path)
				var responsesStreamCapture bytes.Buffer
				commitStreamCaptureStages := func(upstreamBody, responsesBody []byte) {
					if len(upstreamBody) > 0 {
						addCaptureStage(r.Context(), "bridge.chat_completions_response", upstreamBody)
					}
					if len(responsesBody) > 0 {
						addCaptureStage(r.Context(), "bridge.responses_output", responsesBody)
					}
				}
				teeOut := &teeResponseWriter{w: w, onWrite: func(chunk []byte) {
					outMon.writeChunk(chunk)
					_, _ = responsesStreamCapture.Write(chunk)
				}}
				workflowState, verificationShellArgs := extractWorkflowRecoveryContextFromRequestBody(bodyBytes)
				planOutputRequired := structuredPlanOutputRequiredFromRequestBody(bodyBytes)
				nativeQuestionRequired := nativeQuestionOutputRequiredFromRequestBody(bodyBytes)
				var originalReq map[string]any
				_ = json.Unmarshal(bodyBytes, &originalReq)
				err = writeResponsesStreamFromChatSSEWithWorkflowAndContext(r.Context(), teeOut, pr, planOutputRequired, nativeQuestionRequired, requestedReasoningSummary, workflowState, verificationShellArgs, originalReq)
				<-errCh
				commitStreamCaptureStages(upstreamStreamCapture.Bytes(), responsesStreamCapture.Bytes())
				return err
			}

		}

		status, headers, _, out, err := execResponsesRequest(bodyBytes)
		if err != nil {
			return err
		}
		if strings.Contains(strings.ToLower(string(bodyBytes)), "apply_patch") {
			appendApplyPatchTrace("bridge.responses_input", map[string]any{
				"model_id": modelID,
				"summary":  summarizeApplyPatchResponsesRequest(bodyBytes),
			})
		}
		if status < 200 || status >= 300 {
			for k, values := range headers {
				for _, value := range values {
					w.Header().Add(k, value)
				}
			}
			w.WriteHeader(status)
			_, _ = w.Write(out)
			return nil
		}

		if responsesRequestedStream {
			writeResponsesStream(w, out, extractResponsesRequestReasoningSummaryFromBody(bodyBytes))
			return nil
		}

		w.Header().Set("Content-Type", "application/json")
		if status == 0 {
			status = http.StatusOK
		}
		w.WriteHeader(status)
		_, _ = w.Write(out)
		return nil
	}
}

func writeRecorderResponseToWriter(w http.ResponseWriter, rr *httptest.ResponseRecorder) {
	if rr == nil {
		return
	}
	for k, values := range rr.Header() {
		for _, value := range values {
			w.Header().Add(k, value)
		}
	}
	if rr.Code <= 0 {
		rr.Code = http.StatusOK
	}
	w.WriteHeader(rr.Code)
	_, _ = w.Write(rr.Body.Bytes())
}

func (pm *ProxyManager) buildGatewayRetryHandler(modelID string, requestBody []byte, nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error) func(string, http.ResponseWriter, *http.Request) error {
	return func(_ string, w http.ResponseWriter, r *http.Request) error {
		const maxAttempts = 2
		var last *httptest.ResponseRecorder
		for attempt := 1; attempt <= maxAttempts; attempt++ {
			rr := httptest.NewRecorder()
			r2 := r.Clone(r.Context())
			r2.Body = io.NopCloser(bytes.NewReader(requestBody))
			r2.ContentLength = int64(len(requestBody))
			r2.Header = r.Header.Clone()
			r2.Header.Del("transfer-encoding")
			r2.Header.Set("content-length", strconv.Itoa(len(requestBody)))
			err := nextHandler(modelID, rr, r2)
			if err != nil {
				if attempt == maxAttempts {
					return err
				}
				logTextTransform(pm.transformLogger, modelID, "chat_retry", fmt.Sprintf("class=handler_error attempt=%d/%d error=%v", attempt, maxAttempts, err))
				time.Sleep(time.Duration(200*attempt) * time.Millisecond)
				continue
			}
			last = rr
			if rr.Code != http.StatusBadGateway && rr.Code != http.StatusGatewayTimeout {
				writeRecorderResponseToWriter(w, rr)
				return nil
			}
			if attempt < maxAttempts {
				logTextTransform(pm.transformLogger, modelID, "chat_retry", fmt.Sprintf("class=upstream_%d attempt=%d/%d", rr.Code, attempt, maxAttempts))
				time.Sleep(time.Duration(200*attempt) * time.Millisecond)
				continue
			}
		}
		writeRecorderResponseToWriter(w, last)
		return nil
	}
}

func appendIfMissing(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func mustJSONString(v any) string {
	encoded, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

func mustJSONBytes(v any) []byte {
	encoded, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return encoded
}

func logBodyTransform(logger *LogMonitor, modelID string, stage string, before []byte, after []byte) {
	if logger == nil || bytes.Equal(before, after) {
		return
	}
	logger.Infof("<%s> %s\nbefore: %s\nafter: %s", modelID, stage, summarizeJSONForLog(before), summarizeJSONForLog(after))
}

func logTextTransform(logger *LogMonitor, modelID string, stage string, details string) {
	if logger == nil {
		return
	}
	logger.Infof("<%s> %s: %s", modelID, stage, details)
}

func summarizeJSONForLog(body []byte) string {
	const limit = 4000
	if len(body) <= limit {
		return string(body)
	}
	return string(body[:limit]) + "...<truncated>"
}

func normalizeTransformMode(mode RequestTransformMode) RequestTransformMode {
	switch mode {
	case "", TransformModeCompletionsBridge:
		return TransformModeCompletionsBridge
	case TransformModeRaw, TransformModeResponses:
		return mode
	default:
		return ""
	}
}

func (pm *ProxyManager) getTransformMode(modelID string) RequestTransformMode {
	pm.Lock()
	defer pm.Unlock()
	mode := pm.transformModes[modelID]
	switch mode {
	case "", TransformModeRaw, TransformModeCompletionsBridge, TransformModeResponses:
	default:
		logTextTransform(pm.transformLogger, modelID, "invalid_transform_mode", fmt.Sprintf("invalid request transform mode %q", mode))
		return TransformModeCompletionsBridge
	}
	normalized := normalizeTransformMode(mode)
	if normalized == "" {
		logTextTransform(pm.transformLogger, modelID, "invalid_transform_mode", fmt.Sprintf("invalid normalized transform mode %q", mode))
		return TransformModeCompletionsBridge
	}
	return normalized
}

// isResponsesEndpoint returns true if the request path is a responses API endpoint.
// This covers both /v1/responses and the bare /responses route (used by Codex).
func isResponsesEndpoint(path string) bool {
	return path == "/v1/responses" || path == "/responses"
}
func (pm *ProxyManager) isTransformBypassEnabled(modelID string) bool {
	return pm.getTransformMode(modelID) == TransformModeRaw
}

type proxyCtxKey string

type ProxyManager struct {
	sync.Mutex

	config    config.Config
	ginEngine *gin.Engine

	// logging
	proxyLogger     *LogMonitor
	upstreamLogger  *LogMonitor
	muxLogger       *LogMonitor
	transformLogger *LogMonitor

	metricsMonitor *metricsMonitor

	processGroups     map[string]*ProcessGroup
	lastActiveModelID string

	// shutdown signaling
	shutdownCtx    context.Context
	shutdownCancel context.CancelFunc

	// version info
	buildDate string
	commit    string
	version   string

	// peer proxy see: #296, #433
	peerProxy *PeerProxy

	// custom ctx-size per model (stored before loading)
	ctxSizes map[string]int
	// runtime fit-mode per model (stored before loading)
	fitModes map[string]bool
	// fit ctx behavior per model: "max" -> --ctx-size, "min" -> --fit-ctx
	fitCtxModes map[string]string

	// runtime prompt optimization policy per model
	promptPolicies map[string]PromptOptimizationPolicy
	// runtime request transform mode per model
	transformModes map[string]RequestTransformMode

	// latest optimization snapshot for each model (for user visibility and reuse)
	latestPromptOptimizations map[string]PromptOptimizationSnapshot

	// absolute or relative path to active config file
	configPath string

	// lightweight ollama hook
	ollamaEndpoint    string
	ollamaClient      *http.Client
	ollamaModels      map[string]OllamaModel
	ollamaLastRefresh time.Time

	// runtime local web-search fallback settings
	webSearchFallbackEnabled bool
	webSearchFallbackEngine  string
	webSearchFallbackURL     string
	webSearchManagedEnabled  bool
	webSearchManagedCommand  string
	webSearchManagedStopCmd  string
	webSearchManagedService  *managedSidecar
}

type PromptOptimizationPolicy string
type RequestTransformMode string

const (
	PromptOptimizationOff       PromptOptimizationPolicy = "off"
	PromptOptimizationLimitOnly PromptOptimizationPolicy = "limit_only"
	PromptOptimizationAlways    PromptOptimizationPolicy = "always"
	PromptOptimizationLLMAssist PromptOptimizationPolicy = "llm_assisted"

	TransformModeRaw               RequestTransformMode = "raw"
	TransformModeCompletionsBridge RequestTransformMode = "completions_bridge"
	TransformModeResponses         RequestTransformMode = "responses"
)

type PromptOptimizationSnapshot struct {
	Model         string                   `json:"model"`
	Policy        PromptOptimizationPolicy `json:"policy"`
	Applied       bool                     `json:"applied"`
	UpdatedAt     string                   `json:"updatedAt"`
	Note          string                   `json:"note"`
	OriginalBody  string                   `json:"originalBody"`
	OptimizedBody string                   `json:"optimizedBody"`
}

type PromptOptimizationResult struct {
	Policy  PromptOptimizationPolicy
	Applied bool
	Note    string
}

type OllamaModel struct {
	ID           string
	Name         string
	CtxReference int
}

func New(proxyConfig config.Config) *ProxyManager {
	// set up loggers

	var muxLogger, upstreamLogger, proxyLogger, transformLogger *LogMonitor
	switch proxyConfig.LogToStdout {
	case config.LogToStdoutNone:
		muxLogger = NewLogMonitorWriter(io.Discard)
		upstreamLogger = NewLogMonitorWriter(io.Discard)
		proxyLogger = NewLogMonitorWriter(io.Discard)
		transformLogger = NewLogMonitorWriter(io.Discard)
	case config.LogToStdoutBoth:
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(muxLogger)
		proxyLogger = NewLogMonitorWriter(muxLogger)
		transformLogger = NewLogMonitorWriter(muxLogger)
	case config.LogToStdoutUpstream:
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(muxLogger)
		proxyLogger = NewLogMonitorWriter(io.Discard)
		transformLogger = NewLogMonitorWriter(io.Discard)
	default:
		// same as config.LogToStdoutProxy
		// helpful because some old tests create a config.Config directly and it
		// may not have LogToStdout set explicitly
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(io.Discard)
		proxyLogger = NewLogMonitorWriter(muxLogger)
		transformLogger = NewLogMonitorWriter(muxLogger)
	}

	if proxyConfig.LogRequests {
		proxyLogger.Warn("LogRequests configuration is deprecated. Use logLevel instead.")
	}

	switch strings.ToLower(strings.TrimSpace(proxyConfig.LogLevel)) {
	case "debug":
		proxyLogger.SetLogLevel(LevelDebug)
		upstreamLogger.SetLogLevel(LevelDebug)
		transformLogger.SetLogLevel(LevelDebug)
	case "info":
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
		transformLogger.SetLogLevel(LevelInfo)
	case "warn":
		proxyLogger.SetLogLevel(LevelWarn)
		upstreamLogger.SetLogLevel(LevelWarn)
		transformLogger.SetLogLevel(LevelWarn)
	case "error":
		proxyLogger.SetLogLevel(LevelError)
		upstreamLogger.SetLogLevel(LevelError)
		transformLogger.SetLogLevel(LevelError)
	default:
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
		transformLogger.SetLogLevel(LevelInfo)
	}

	// see: https://go.dev/src/time/format.go
	timeFormats := map[string]string{
		"ansic":       time.ANSIC,
		"unixdate":    time.UnixDate,
		"rubydate":    time.RubyDate,
		"rfc822":      time.RFC822,
		"rfc822z":     time.RFC822Z,
		"rfc850":      time.RFC850,
		"rfc1123":     time.RFC1123,
		"rfc1123z":    time.RFC1123Z,
		"rfc3339":     time.RFC3339,
		"rfc3339nano": time.RFC3339Nano,
		"kitchen":     time.Kitchen,
		"stamp":       time.Stamp,
		"stampmilli":  time.StampMilli,
		"stampmicro":  time.StampMicro,
		"stampnano":   time.StampNano,
	}

	if timeFormat, ok := timeFormats[strings.ToLower(strings.TrimSpace(proxyConfig.LogTimeFormat))]; ok {
		proxyLogger.SetLogTimeFormat(timeFormat)
		upstreamLogger.SetLogTimeFormat(timeFormat)
		transformLogger.SetLogTimeFormat(timeFormat)
	}

	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())

	var maxMetrics int
	if proxyConfig.MetricsMaxInMemory <= 0 {
		maxMetrics = 1000 // Default fallback
	} else {
		maxMetrics = proxyConfig.MetricsMaxInMemory
	}

	peerProxy, err := NewPeerProxy(proxyConfig.Peers, proxyLogger)
	if err != nil {
		proxyLogger.Errorf("Disabling Peering. Failed to create proxy peers: %v", err)
		peerProxy = nil
	}

	pm := &ProxyManager{
		config:    proxyConfig,
		ginEngine: gin.New(),

		proxyLogger:     proxyLogger,
		muxLogger:       muxLogger,
		upstreamLogger:  upstreamLogger,
		transformLogger: transformLogger,

		metricsMonitor: newMetricsMonitor(proxyLogger, maxMetrics, proxyConfig.CaptureBuffer),

		processGroups: make(map[string]*ProcessGroup),

		shutdownCtx:    shutdownCtx,
		shutdownCancel: shutdownCancel,

		buildDate: "unknown",
		commit:    "abcd1234",
		version:   "0",

		peerProxy:                 peerProxy,
		ctxSizes:                  make(map[string]int),
		fitModes:                  make(map[string]bool),
		fitCtxModes:               make(map[string]string),
		promptPolicies:            make(map[string]PromptOptimizationPolicy),
		transformModes:            make(map[string]RequestTransformMode),
		latestPromptOptimizations: make(map[string]PromptOptimizationSnapshot),
		configPath:                "config.yaml",
		ollamaEndpoint:            "http://127.0.0.1:11434",
		ollamaClient:              &http.Client{Timeout: 20 * time.Second},
		ollamaModels:              make(map[string]OllamaModel),
		webSearchFallbackEnabled:  true,
		webSearchFallbackEngine:   normalizeWebSearchFallbackEngine("duckduckgo_html"),
		webSearchFallbackURL:      strings.TrimSpace(os.Getenv(llamaSwapWebSearchURLVar)),
		webSearchManagedEnabled:   envBoolTrue(llamaSwapSearxngEnabledVar),
		webSearchManagedCommand:   strings.TrimSpace(os.Getenv(llamaSwapSearxngCommandVar)),
		webSearchManagedStopCmd:   strings.TrimSpace(os.Getenv(llamaSwapSearxngStopCommandVar)),
	}
	pm.webSearchManagedService = newManagedSidecar("searxng", proxyLogger)
	pm.webSearchManagedService.SetStopCommand(pm.webSearchManagedStopCmd)

	// create the process groups
	for groupID := range proxyConfig.Groups {
		processGroup := NewProcessGroup(groupID, proxyConfig, proxyLogger, upstreamLogger, transformLogger)
		pm.processGroups[groupID] = processGroup
		for modelID, process := range processGroup.processes {
			pm.metricsMonitor.registerModelLogMonitor(modelID, process.LogMonitor())
		}
	}

	pm.setupGinEngine()

	// run any startup hooks
	if len(proxyConfig.Hooks.OnStartup.Preload) > 0 {
		// do it in the background, don't block startup -- not sure if good idea yet
		go func() {
			discardWriter := &DiscardWriter{}
			for _, preloadModelName := range proxyConfig.Hooks.OnStartup.Preload {
				modelID, ok := proxyConfig.RealModelName(preloadModelName)

				if !ok {
					proxyLogger.Warnf("Preload model %s not found in config", preloadModelName)
					continue
				}

				proxyLogger.Infof("Preloading model: %s", modelID)
				processGroup, err := pm.swapProcessGroup(modelID)

				if err != nil {
					event.Emit(ModelPreloadedEvent{
						ModelName: modelID,
						Success:   false,
					})
					proxyLogger.Errorf("Failed to preload model %s: %v", modelID, err)
					continue
				} else {
					req, _ := http.NewRequest("GET", "/", nil)
					processGroup.ProxyRequest(modelID, discardWriter, req)
					event.Emit(ModelPreloadedEvent{
						ModelName: modelID,
						Success:   true,
					})
				}
			}
		}()
	}

	pm.webSearchManagedService.StartIfEnabled(pm.shutdownCtx, pm.webSearchManagedEnabled, pm.webSearchManagedCommand)

	return pm
}

func (pm *ProxyManager) setupGinEngine() {

	pm.ginEngine.Use(func(c *gin.Context) {

		// don't log the Wake on Lan proxy health check
		if c.Request.URL.Path == "/wol-health" {
			c.Next()
			return
		}

		// Start timer
		start := time.Now()

		// capture these because /upstream/:model rewrites them in c.Next()
		clientIP := c.ClientIP()
		method := c.Request.Method
		path := c.Request.URL.Path

		// Process request
		c.Next()

		// Stop timer
		duration := time.Since(start)

		statusCode := c.Writer.Status()
		bodySize := c.Writer.Size()

		pm.proxyLogger.Infof("Request %s \"%s %s %s\" %d %d \"%s\" %v",
			clientIP,
			method,
			path,
			c.Request.Proto,
			statusCode,
			bodySize,
			c.Request.UserAgent(),
			duration,
		)
	})

	// see: issue: #81, #77 and #42 for CORS issues
	// respond with permissive OPTIONS for any endpoint
	pm.ginEngine.Use(func(c *gin.Context) {
		if c.Request.Method == "OPTIONS" {
			c.Header("Access-Control-Allow-Origin", "*")
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")

			// allow whatever the client requested by default
			if headers := c.Request.Header.Get("Access-Control-Request-Headers"); headers != "" {
				sanitized := SanitizeAccessControlRequestHeaderValues(headers)
				c.Header("Access-Control-Allow-Headers", sanitized)
			} else {
				c.Header(
					"Access-Control-Allow-Headers",
					"Content-Type, Authorization, Accept, X-Requested-With",
				)
			}
			c.Header("Access-Control-Max-Age", "86400")
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	// Set up routes using the Gin engine
	// Protected routes use pm.apiKeyAuth() middleware
	pm.ginEngine.POST("/v1/chat/completions", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/responses", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support Codex calling /responses without /v1/ prefix
	pm.ginEngine.POST("/responses", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support legacy /v1/completions api, see issue #12
	pm.ginEngine.POST("/v1/completions", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support anthropic /v1/messages (added https://github.com/ggml-org/llama.cpp/pull/17570)
	pm.ginEngine.POST("/v1/messages", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support anthropic count_tokens API (Also added in the above PR)
	pm.ginEngine.POST("/v1/messages/count_tokens", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// Support embeddings and reranking
	pm.ginEngine.POST("/v1/embeddings", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /reranking endpoint + aliases
	pm.ginEngine.POST("/reranking", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/rerank", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/rerank", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/reranking", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /infill endpoint for code infilling
	pm.ginEngine.POST("/infill", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /completion endpoint
	pm.ginEngine.POST("/completion", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// Support audio/speech endpoint
	pm.ginEngine.POST("/v1/audio/speech", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/audio/voices", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.GET("/v1/audio/voices", pm.apiKeyAuth(), pm.proxyGETModelHandler)
	pm.ginEngine.POST("/v1/audio/transcriptions", pm.apiKeyAuth(), pm.proxyOAIPostFormHandler)
	pm.ginEngine.POST("/v1/images/generations", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/images/edits", pm.apiKeyAuth(), pm.proxyOAIPostFormHandler)

	pm.ginEngine.GET("/v1/models", pm.apiKeyAuth(), pm.listModelsHandler)

	// in proxymanager_loghandlers.go
	pm.ginEngine.GET("/logs", pm.apiKeyAuth(), pm.sendLogsHandlers)
	pm.ginEngine.GET("/logs/stream", pm.apiKeyAuth(), pm.streamLogsHandler)
	pm.ginEngine.GET("/logs/stream/*logMonitorID", pm.apiKeyAuth(), pm.streamLogsHandler)

	/**
	 * User Interface Endpoints
	 */
	pm.ginEngine.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui")
	})

	pm.ginEngine.GET("/upstream", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui/models")
	})
	pm.ginEngine.Any("/upstream/*upstreamPath", pm.apiKeyAuth(), pm.proxyToUpstream)
	pm.ginEngine.GET("/unload", pm.apiKeyAuth(), pm.unloadAllModelsHandler)
	pm.ginEngine.GET("/running", pm.apiKeyAuth(), pm.listRunningProcessesHandler)
	pm.ginEngine.GET("/health", func(c *gin.Context) {
		deltaModels := make([]string, 0, 4)
		for modelID := range pm.config.Models {
			if modelLikelyRequiresDeltaNet(modelID) {
				deltaModels = append(deltaModels, modelID)
			}
		}
		sort.Strings(deltaModels)
		c.JSON(http.StatusOK, gin.H{
			"status":              "ok",
			"delta_net_supported": len(deltaModels) == 0,
			"delta_net_models":    deltaModels,
			"architecture_hint":   architectureRequiresMinBuild,
		})
	})

	// see cmd/wol-proxy/wol-proxy.go, not logged
	pm.ginEngine.GET("/wol-health", func(c *gin.Context) {
		c.String(http.StatusOK, "OK")
	})

	pm.ginEngine.GET("/favicon.ico", func(c *gin.Context) {
		if data, err := reactStaticFS.ReadFile("ui_dist/favicon.ico"); err == nil {
			c.Data(http.StatusOK, "image/x-icon", data)
		} else {
			c.String(http.StatusInternalServerError, err.Error())
		}
	})

	reactFS, err := GetReactFS()
	if err != nil {
		pm.proxyLogger.Errorf("Failed to load React filesystem: %v", err)
	} else {
		// Serve files with compression support under /ui/*
		// This handler checks for pre-compressed .br and .gz files
		pm.ginEngine.GET("/ui/*filepath", func(c *gin.Context) {
			filepath := strings.TrimPrefix(c.Param("filepath"), "/")
			// Default to index.html for directory-like paths
			if filepath == "" {
				filepath = "index.html"
			}

			ServeCompressedFile(reactFS, c.Writer, c.Request, filepath)
		})

		// Serve SPA for UI under /ui/* - fallback to index.html for client-side routing
		pm.ginEngine.NoRoute(func(c *gin.Context) {
			if !strings.HasPrefix(c.Request.URL.Path, "/ui") {
				c.AbortWithStatus(http.StatusNotFound)
				return
			}

			// Check if this looks like a file request (has extension)
			path := c.Request.URL.Path
			if strings.Contains(path, ".") && !strings.HasSuffix(path, "/") {
				// This was likely a file request that wasn't found
				c.AbortWithStatus(http.StatusNotFound)
				return
			}

			// Serve index.html for SPA routing
			ServeCompressedFile(reactFS, c.Writer, c.Request, "index.html")
		})
	}

	// see: proxymanager_api.go
	// add API handler functions
	addApiHandlers(pm)

	// Disable console color for testing
	gin.DisableConsoleColor()
}

// ServeHTTP implements http.Handler interface
func (pm *ProxyManager) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	pm.ginEngine.ServeHTTP(w, r)
}

func (pm *ProxyManager) setLastActiveModelID(modelID string) {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return
	}
	pm.Lock()
	pm.lastActiveModelID = modelID
	pm.Unlock()
}

func (pm *ProxyManager) lastActiveModelIDSnapshot() string {
	pm.Lock()
	id := pm.lastActiveModelID
	pm.Unlock()
	return strings.TrimSpace(id)
}

func (pm *ProxyManager) resolveActiveModelAlias(requestedModel string) (string, bool) {
	if !pm.config.IsActiveModelAlias(requestedModel) {
		return "", false
	}
	return pm.lastActiveModelIDSnapshot(), true
}

// StopProcesses acquires a lock and stops all running upstream processes.
// This is the public method safe for concurrent calls.
// Unlike Shutdown, this method only stops the processes but doesn't perform
// a complete shutdown, allowing for process replacement without full termination.
func (pm *ProxyManager) StopProcesses(strategy StopStrategy) {
	pm.Lock()
	service := pm.webSearchManagedService
	processGroups := make([]*ProcessGroup, 0, len(pm.processGroups))
	for _, processGroup := range pm.processGroups {
		processGroups = append(processGroups, processGroup)
	}
	pm.Unlock()

	// stop Processes in parallel
	var wg sync.WaitGroup
	for _, processGroup := range processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.StopProcesses(strategy)
		}(processGroup)
	}

	wg.Wait()
	if service != nil {
		_ = service.Stop()
	}
}

// Shutdown stops all processes managed by this ProxyManager
func (pm *ProxyManager) Shutdown() {
	pm.Lock()
	service := pm.webSearchManagedService
	processGroups := make([]*ProcessGroup, 0, len(pm.processGroups))
	for _, processGroup := range pm.processGroups {
		processGroups = append(processGroups, processGroup)
	}
	pm.Unlock()

	pm.proxyLogger.Debug("Shutdown() called in proxy manager")

	var wg sync.WaitGroup
	// Send shutdown signal to all process in groups
	for _, processGroup := range processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.Shutdown()
		}(processGroup)
	}
	wg.Wait()
	if service != nil {
		_ = service.Stop()
	}
	pm.shutdownCancel()
}

func (pm *ProxyManager) swapProcessGroup(realModelName string) (*ProcessGroup, error) {
	processGroup := pm.findGroupByModelName(realModelName)
	if processGroup == nil {
		return nil, fmt.Errorf("could not find process group for model %s", realModelName)
	}

	if process, ok := processGroup.processes[realModelName]; ok && process != nil {
		pm.Lock()
		ctxSize := pm.ctxSizes[realModelName]
		fitEnabled, fitOverride := pm.fitModes[realModelName]
		fitCtxMode, fitCtxModeOverride := pm.fitCtxModes[realModelName]
		pm.Unlock()

		if !fitOverride {
			if args, err := process.config.SanitizedCommand(); err == nil {
				_, _, parsedFitEnabled, parsedFitCtxMode := parseCtxAndFitFromArgs(args)
				fitEnabled = parsedFitEnabled
				if !fitCtxModeOverride {
					fitCtxMode = parsedFitCtxMode
				}
			}
		}
		if fitCtxMode == "" {
			fitCtxMode = "max"
		}

		process.SetRuntimeCtxSize(ctxSize)
		process.SetRuntimeFitMode(fitEnabled)
		process.SetRuntimeFitCtxMode(fitCtxMode == "min")
	}

	if processGroup.exclusive {
		pm.proxyLogger.Debugf("Exclusive mode for group %s, stopping other process groups", processGroup.id)
		for groupId, otherGroup := range pm.processGroups {
			if groupId != processGroup.id && !otherGroup.persistent {
				otherGroup.StopProcesses(StopWaitForInflightRequest)
			}
		}
	}

	return processGroup, nil
}

func parseCtxAndFitFromArgs(args []string) (ctxSize int, source string, fitEnabled bool, fitCtxMode string) {
	ctxFromCtxSize := 0
	ctxFromFitCtx := 0
	fitCtxMode = "max"

	for i := 0; i < len(args); i++ {
		arg := strings.TrimSpace(args[i])
		if arg == "" {
			continue
		}

		switch {
		case arg == "--fit":
			fitEnabled = true
			if i+1 < len(args) {
				next := strings.ToLower(strings.TrimSpace(args[i+1]))
				switch next {
				case "off", "false", "0", "no":
					fitEnabled = false
				case "on", "true", "1", "yes":
					fitEnabled = true
				}
			}
		case strings.HasPrefix(arg, "--fit="):
			val := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(arg, "--fit=")))
			fitEnabled = val == "on" || val == "true" || val == "1" || val == "yes"
		case arg == "--fit-ctx":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil && n > 0 {
					ctxFromFitCtx = n
				}
			}
		case strings.HasPrefix(arg, "--fit-ctx="):
			if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(arg, "--fit-ctx="))); err == nil && n > 0 {
				ctxFromFitCtx = n
			}
		case arg == "--ctx-size" || arg == "-c":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil && n > 0 {
					ctxFromCtxSize = n
				}
			}
		case strings.HasPrefix(arg, "--ctx-size="):
			if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(arg, "--ctx-size="))); err == nil && n > 0 {
				ctxFromCtxSize = n
			}
		}
	}

	if fitEnabled && ctxFromFitCtx > 0 {
		return ctxFromFitCtx, "fit-ctx", true, "min"
	}
	if ctxFromCtxSize > 0 {
		return ctxFromCtxSize, "ctx-size", fitEnabled, "max"
	}
	if ctxFromFitCtx > 0 {
		return ctxFromFitCtx, "fit-ctx", fitEnabled, "min"
	}
	return 0, "", fitEnabled, fitCtxMode
}

func (pm *ProxyManager) listModelsHandler(c *gin.Context) {
	data := make([]gin.H, 0, len(pm.config.Models))
	createdTime := time.Now().Unix()
	seenIDs := make(map[string]struct{}, len(pm.config.Models))

	newRecord := func(modelId string, modelConfig config.ModelConfig) gin.H {
		record := gin.H{
			"id":       modelId,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "llama-swap",
		}

		if name := strings.TrimSpace(modelConfig.Name); name != "" {
			record["name"] = name
		}
		if desc := strings.TrimSpace(modelConfig.Description); desc != "" {
			record["description"] = desc
		}

		// Add metadata if present
		if len(modelConfig.Metadata) > 0 {
			record["meta"] = gin.H{
				"llamaswap": modelConfig.Metadata,
			}
		}
		return record
	}

	for id, modelConfig := range pm.config.Models {
		if modelConfig.Unlisted {
			continue
		}

		seenIDs[id] = struct{}{}
		data = append(data, newRecord(id, modelConfig))

		// Include aliases
		if pm.config.IncludeAliasesInList {
			for _, alias := range modelConfig.Aliases {
				if alias := strings.TrimSpace(alias); alias != "" {
					seenIDs[alias] = struct{}{}
					data = append(data, newRecord(alias, modelConfig))
				}
			}
		}
	}

	// Include dynamic "active model" aliases as pseudo-models so clients that
	// enumerate /v1/models can still select them.
	for _, alias := range pm.config.UseActiveModelForAliases {
		alias = strings.TrimSpace(alias)
		if alias == "" {
			continue
		}
		if _, exists := seenIDs[alias]; exists {
			continue
		}
		seenIDs[alias] = struct{}{}
		data = append(data, gin.H{
			"id":       alias,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "llama-swap",
			"meta": gin.H{
				"llamaswap": gin.H{
					"active_model_alias": true,
				},
			},
		})
	}

	if pm.peerProxy != nil {
		for peerID, peer := range pm.peerProxy.ListPeers() {
			// add peer models
			for _, modelID := range peer.Models {
				// Skip unlisted models if not showing them
				record := newRecord(modelID, config.ModelConfig{
					Name: fmt.Sprintf("%s: %s", peerID, modelID),
					Metadata: map[string]any{
						"peerID": peerID,
					},
				})

				data = append(data, record)
			}
		}
	}

	for _, ollamaModel := range pm.GetOllamaModels() {
		data = append(data, gin.H{
			"id":       ollamaModel.ID,
			"name":     ollamaModel.Name,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "ollama",
			"meta": gin.H{
				"llamaswap": gin.H{
					"provider":      "ollama",
					"external":      true,
					"ctx_reference": ollamaModel.CtxReference,
				},
			},
		})
	}

	// Sort by the "id" key
	sort.Slice(data, func(i, j int) bool {
		si, _ := data[i]["id"].(string)
		sj, _ := data[j]["id"].(string)
		return si < sj
	})

	// Set CORS headers if origin exists
	if origin := c.GetHeader("Origin"); origin != "" {
		c.Header("Access-Control-Allow-Origin", origin)
	}

	// Use gin's JSON method which handles content-type and encoding
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   data,
	})
}

// findModelInPath searches for a valid model name in a path with slashes.
// It iteratively builds up path segments until it finds a matching model.
// Returns: (searchModelName, realModelName, remainingPath, found)
// Example: "/author/model/endpoint" with model "author/model" -> ("author/model", "author/model", "/endpoint", true)
func (pm *ProxyManager) findModelInPath(path string) (searchName string, realName string, remainingPath string, found bool) {
	parts := strings.Split(strings.TrimSpace(path), "/")
	searchModelName := ""

	for i, part := range parts {
		if part == "" {
			continue
		}

		if searchModelName == "" {
			searchModelName = part
		} else {
			searchModelName = searchModelName + "/" + part
		}

		if modelID, ok := pm.config.RealModelName(searchModelName); ok {
			return searchModelName, modelID, "/" + strings.Join(parts[i+1:], "/"), true
		}
	}

	return "", "", "", false
}

func (pm *ProxyManager) proxyToUpstream(c *gin.Context) {
	upstreamPath := c.Param("upstreamPath")

	searchModelName, modelID, remainingPath, modelFound := pm.findModelInPath(upstreamPath)

	if !modelFound {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model id required in path")
		return
	}

	// Redirect /upstream/modelname to /upstream/modelname/ for URL consistency.
	// This ensures relative URLs in upstream responses resolve correctly and
	// provides canonical URL form. Uses 308 for POST/PUT/etc to preserve the
	// HTTP method (301 would downgrade to GET).
	if remainingPath == "/" && !strings.HasSuffix(upstreamPath, "/") {
		newPath := "/upstream/" + searchModelName + "/"
		if c.Request.URL.RawQuery != "" {
			newPath += "?" + c.Request.URL.RawQuery
		}
		if c.Request.Method == http.MethodGet || c.Request.Method == http.MethodHead {
			c.Redirect(http.StatusMovedPermanently, newPath)
		} else {
			c.Redirect(http.StatusPermanentRedirect, newPath)
		}
		return
	}

	processGroup, err := pm.swapProcessGroup(modelID)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
		return
	}

	// rewrite the path
	originalPath := c.Request.URL.Path
	c.Request.URL.Path = remainingPath

	// attempt to record metrics if it is a POST request
	if pm.metricsMonitor != nil && c.Request.Method == "POST" {
		if err := pm.metricsMonitor.wrapHandler(modelID, c.Writer, c.Request, processGroup.ProxyRequest); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying metrics wrapped request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error proxying wrapped upstream request for model %s, path=%s", modelID, originalPath)
			return
		}
	} else {
		if err := processGroup.ProxyRequest(modelID, c.Writer, c.Request); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error proxying upstream request for model %s, path=%s", modelID, originalPath)
			return
		}
	}
}

func (pm *ProxyManager) proxyInferenceHandler(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not ready request body")
		return
	}

	requestedModel := gjson.GetBytes(bodyBytes, "model").String()
	if requestedModel == "" {
		// Codex sends model_config as a string value directly
		requestedModel = gjson.Get(string(bodyBytes), "model_config").String()
	}
	if requestedModel == "" {
		// Some APIs send it nested: {"model_config": {"model": "..."}}
		requestedModel = gjson.GetBytes(bodyBytes, "model_config.model").String()
	}
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' key")
		return
	}
	if requestedModel == "" {
		// Some APIs send it nested: {"model_config": {"model": "..."}}
		requestedModel = gjson.GetBytes(bodyBytes, "model_config.model").String()
	}
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' key")
		return
	}

	if resolved, matched := pm.resolveActiveModelAlias(requestedModel); matched {
		if resolved == "" {
			pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("model '%s' is configured as an active-model alias, but no active model is set yet", requestedModel))
			return
		}
		pm.proxyLogger.Debugf("Resolved active-model alias '%s' -> '%s'", requestedModel, resolved)
		requestedModel = resolved
	}

	// Look for a matching local model first
	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error
	requestPath := c.Request.URL.Path

	modelID, found := pm.config.RealModelName(requestedModel)
	if found {
		pm.setLastActiveModelID(modelID)
		processGroup, err := pm.swapProcessGroup(modelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}
		if err := processGroup.EnsureStarted(modelID); err != nil {
			lowerErr := strings.ToLower(strings.TrimSpace(err.Error()))
			if strings.Contains(lowerErr, "delta") || strings.Contains(lowerErr, "unknown arch") || strings.Contains(lowerErr, "not implemented") {
				c.Data(http.StatusServiceUnavailable, "application/json", buildArchitectureUnsupportedErrorBody([]byte(err.Error())))
				return
			}
			pm.sendErrorResponse(c, http.StatusBadGateway, fmt.Sprintf("unable to start process: %s", err.Error()))
			return
		}

		// issue #69 allow custom model names to be sent to upstream
		useModelName := pm.config.Models[modelID].UseModelName
		if useModelName != "" {
			beforeBody := append([]byte(nil), bodyBytes...)
			bodyBytes, err = sjson.SetBytes(bodyBytes, "model", useModelName)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting model name in JSON: %s", err.Error()))
				return
			}
			logBodyTransform(pm.transformLogger, modelID, "rewrite_model_name", beforeBody, bodyBytes)
		}

		// Log raw request body for debugging (especially tools)
		pm.proxyLogger.Debugf("<%s> Raw request body (before transforms): %s", modelID, string(bodyBytes))
		if isResponsesEndpoint(requestPath) {
			if missingSkills, extraSkills, ok := detectCodexSkillSnapshotDrift(bodyBytes); ok && (len(missingSkills) > 0 || len(extraSkills) > 0) {
				logTextTransform(pm.transformLogger, modelID, "codex_skill_snapshot_drift",
					fmt.Sprintf("missing_on_request=%q extra_on_request=%q", strings.Join(missingSkills, ","), strings.Join(extraSkills, ",")))
			}
		}

		transformMode := pm.getTransformMode(modelID)
		bypassTransforms := transformMode == TransformModeRaw
		if bypassTransforms {
			logTextTransform(pm.transformLogger, modelID, "bypass_transforms", "skipping request-body filters and prompt control")
		} else if isResponsesEndpoint(requestPath) {
			logTextTransform(pm.transformLogger, modelID, "transform_mode", string(transformMode))
		}

		// Log request body after transforms
		pm.proxyLogger.Debugf("<%s> Request body (after transforms): %s", modelID, string(bodyBytes))

		if !bypassTransforms {
			if isResponsesEndpoint(requestPath) && transformMode == TransformModeResponses {
				var adaptedTools []string
				var unsupportedTools []string
				beforeBody := append([]byte(nil), bodyBytes...)
				bodyBytes, adaptedTools, unsupportedTools, err = normalizeResponsesRequest(bodyBytes)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, "error normalizing responses tools in request")
					return
				}
				logBodyTransform(pm.transformLogger, modelID, "normalize_responses_request", beforeBody, bodyBytes)
				if len(unsupportedTools) > 0 {
					logTextTransform(pm.transformLogger, modelID, "reject_unsupported_responses_tools", strings.Join(unsupportedTools, ", "))
					pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("unsupported /v1/responses tool types for local backend: %s", strings.Join(unsupportedTools, ", ")))
					return
				}
				if len(adaptedTools) > 0 {
					logTextTransform(pm.transformLogger, modelID, "adapted_responses_tools", strings.Join(adaptedTools, ", "))
					c.Request.Header.Set(llamaSwapResponseToolAdapterHeader, strings.Join(adaptedTools, ","))
				}
			}
		}
		if isResponsesEndpoint(requestPath) {
			if hint := extractApplyPatchPathHintFromResponsesRequestBody(bodyBytes); hint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchPathHintHeader, hint)
				logTextTransform(pm.transformLogger, modelID, "apply_patch_path_hint", hint)
			}
			if contentHint := extractApplyPatchContentHintFromResponsesRequestBody(bodyBytes); contentHint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchContentHintHeader, contentHint)
				logTextTransform(pm.transformLogger, modelID, "apply_patch_content_hint", truncateBridgeDebugText(contentHint, 80))
			}
			if typeHint := extractApplyPatchTypeHintFromResponsesRequestBody(bodyBytes); typeHint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchTypeHintHeader, typeHint)
				logTextTransform(pm.transformLogger, modelID, "apply_patch_type_hint", typeHint)
			}
		}

		var optResult PromptOptimizationResult
		if !bypassTransforms && isResponsesEndpoint(requestPath) && transformMode != TransformModeCompletionsBridge {
			beforeOptimization := append([]byte(nil), bodyBytes...)
			if bodyBytes, optResult, err = pm.applyPromptSizeControl(modelID, bodyBytes); err != nil {
				pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("context control rejected request: %s", err.Error()))
				return
			}
			logBodyTransform(pm.transformLogger, modelID, "prompt_size_control", beforeOptimization, bodyBytes)
		}
		c.Header("X-LlamaSwap-Prompt-Optimization-Policy", string(optResult.Policy))
		if optResult.Applied {
			c.Header("X-LlamaSwap-Prompt-Optimized", "true")
		} else {
			c.Header("X-LlamaSwap-Prompt-Optimized", "false")
		}

		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
		if isResponsesEndpoint(requestPath) && transformMode == TransformModeCompletionsBridge {
			nextHandler = pm.buildResponsesBridgeHandler(modelID, append([]byte(nil), bodyBytes...), nextHandler)
		}
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		modelID = requestedModel

		// Log raw request body for debugging
		pm.proxyLogger.Debugf("<%s> Raw request body (before peer transforms): %s", requestedModel, string(bodyBytes))

		transformMode := pm.getTransformMode(modelID)
		bypassTransforms := transformMode == TransformModeRaw
		if bypassTransforms {
			logTextTransform(pm.transformLogger, modelID, "peer_bypass_transforms", "skipping request-body filters and prompt control")
		} else if isResponsesEndpoint(requestPath) {
			logTextTransform(pm.transformLogger, modelID, "peer_transform_mode", string(transformMode))
		}

		// Log request body after peer transforms
		pm.proxyLogger.Debugf("<%s> Request body (after peer transforms): %s", requestedModel, string(bodyBytes))

		if !bypassTransforms {
			if isResponsesEndpoint(requestPath) && transformMode == TransformModeResponses {
				var adaptedTools []string
				var unsupportedTools []string
				beforeBody := append([]byte(nil), bodyBytes...)
				bodyBytes, adaptedTools, unsupportedTools, err = normalizeResponsesRequest(bodyBytes)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, "error normalizing responses tools in request")
					return
				}
				logBodyTransform(pm.transformLogger, modelID, "peer_normalize_responses_request", beforeBody, bodyBytes)
				if len(unsupportedTools) > 0 {
					logTextTransform(pm.transformLogger, modelID, "peer_reject_unsupported_responses_tools", strings.Join(unsupportedTools, ", "))
					pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("unsupported /v1/responses tool types for local backend: %s", strings.Join(unsupportedTools, ", ")))
					return
				}
				if len(adaptedTools) > 0 {
					logTextTransform(pm.transformLogger, modelID, "peer_adapted_responses_tools", strings.Join(adaptedTools, ", "))
					c.Request.Header.Set(llamaSwapResponseToolAdapterHeader, strings.Join(adaptedTools, ","))
				}
			}
		}
		if isResponsesEndpoint(requestPath) {
			if hint := extractApplyPatchPathHintFromResponsesRequestBody(bodyBytes); hint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchPathHintHeader, hint)
				logTextTransform(pm.transformLogger, modelID, "peer_apply_patch_path_hint", hint)
			}
			if contentHint := extractApplyPatchContentHintFromResponsesRequestBody(bodyBytes); contentHint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchContentHintHeader, contentHint)
				logTextTransform(pm.transformLogger, modelID, "peer_apply_patch_content_hint", truncateBridgeDebugText(contentHint, 80))
			}
			if typeHint := extractApplyPatchTypeHintFromResponsesRequestBody(bodyBytes); typeHint != "" {
				c.Request.Header.Set(llamaSwapApplyPatchTypeHintHeader, typeHint)
				logTextTransform(pm.transformLogger, modelID, "peer_apply_patch_type_hint", typeHint)
			}
		}

		nextHandler = pm.peerProxy.ProxyRequest
		if isResponsesEndpoint(requestPath) && transformMode == TransformModeCompletionsBridge {
			nextHandler = pm.buildResponsesBridgeHandler(modelID, append([]byte(nil), bodyBytes...), nextHandler)
		}
	} else if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
		modelID = ollamaModel.ID
		bodyBytes, err = sjson.SetBytes(bodyBytes, "model", ollamaModel.Name)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting ollama model name in JSON: %s", err.Error()))
			return
		}
		logTextTransform(pm.transformLogger, modelID, "rewrite_ollama_model_name", ollamaModel.Name)

		var optResult PromptOptimizationResult
		if !pm.isTransformBypassEnabled(modelID) && isResponsesEndpoint(requestPath) {
			beforeOptimization := append([]byte(nil), bodyBytes...)
			if bodyBytes, optResult, err = pm.applyPromptSizeControl(modelID, bodyBytes); err != nil {
				pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("context control rejected request: %s", err.Error()))
				return
			}
			logBodyTransform(pm.transformLogger, modelID, "prompt_size_control", beforeOptimization, bodyBytes)
		} else {
			logTextTransform(pm.transformLogger, modelID, "bypass_transforms", "skipping prompt control for ollama request")
		}
		c.Header("X-LlamaSwap-Prompt-Optimization-Policy", string(optResult.Policy))
		if optResult.Applied {
			c.Header("X-LlamaSwap-Prompt-Optimized", "true")
		} else {
			c.Header("X-LlamaSwap-Prompt-Optimized", "false")
		}

		pm.proxyLogger.Debugf("ProxyManager using Ollama for model: %s", requestedModel)
		nextHandler = pm.proxyOllamaRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable inference handler for %s", requestedModel))
		return
	}

	if c.Request.Method == "POST" && c.Request.URL.Path == "/v1/chat/completions" {
		beforeBody := append([]byte(nil), bodyBytes...)
		updated, conflictResult, err := stripGrammarToolsConflictJSON(bodyBytes)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("invalid chat request while resolving grammar/tools conflict: %s", err.Error()))
			return
		}
		if conflictResult.removedGrammar || conflictResult.removedJSONSchemaResponse {
			bodyBytes = updated
			logBodyTransform(pm.transformLogger, modelID, "strip_grammar_tools_conflict", beforeBody, bodyBytes)
		}
	}

	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	// dechunk it as we already have all the body bytes see issue #11
	c.Request.Header.Del("transfer-encoding")
	c.Request.Header.Set("content-length", strconv.Itoa(len(bodyBytes)))
	c.Request.ContentLength = int64(len(bodyBytes))

	// issue #366 extract values that downstream handlers may need
	isStreaming := gjson.GetBytes(bodyBytes, "stream").Bool()
	ctx := context.WithValue(c.Request.Context(), proxyCtxKey("streaming"), isStreaming)
	ctx = context.WithValue(ctx, proxyCtxKey("model"), modelID)
	c.Request = c.Request.WithContext(ctx)

	// Mitigate transient upstream 502/504 timeouts for non-stream chat completions.
	if c.Request.Method == "POST" && c.Request.URL.Path == "/v1/chat/completions" && !isStreaming {
		nextHandler = pm.buildGatewayRetryHandler(modelID, append([]byte(nil), bodyBytes...), nextHandler)
	}

	if pm.metricsMonitor != nil && c.Request.Method == "POST" {
		if err := pm.metricsMonitor.wrapHandler(modelID, c.Writer, c.Request, nextHandler); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying metrics wrapped request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error Proxying Metrics Wrapped Request model %s", modelID)
			return
		}
	} else {
		if err := nextHandler(modelID, c.Writer, c.Request); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error Proxying Request for model %s", modelID)
			return
		}
	}
}

func (pm *ProxyManager) applyPromptSizeControl(modelID string, bodyBytes []byte) ([]byte, PromptOptimizationResult, error) {
	pm.Lock()
	ctxSize := pm.ctxSizes[modelID]
	runtimePolicy, hasRuntimePolicy := pm.promptPolicies[modelID]
	pm.Unlock()
	result := PromptOptimizationResult{
		Policy:  PromptOptimizationLimitOnly,
		Applied: false,
		Note:    "no optimization",
	}

	if !gjson.GetBytes(bodyBytes, "messages").IsArray() {
		return bodyBytes, result, nil
	}

	var chatReq ChatRequest
	if err := json.Unmarshal(bodyBytes, &chatReq); err != nil {
		return nil, result, fmt.Errorf("invalid chat request JSON: %w", err)
	}

	modelConfig, exists := pm.config.Models[modelID]
	if !exists {
		if !isOllamaModelID(modelID) {
			return bodyBytes, result, nil
		}
		modelConfig = config.ModelConfig{
			Proxy:          pm.ollamaEndpoint,
			TruncationMode: string(SlidingWindow),
		}
	}

	policy := PromptOptimizationLimitOnly
	if hasRuntimePolicy {
		policy = runtimePolicy
	}
	result.Policy = policy
	if policy == PromptOptimizationOff {
		result.Note = "optimization disabled"
		pm.savePromptOptimizationSnapshot(modelID, policy, false, bodyBytes, bodyBytes, result.Note)
		return bodyBytes, result, nil
	}

	mode := SlidingWindow
	switch policy {
	case PromptOptimizationAlways:
		chatReq.Messages = CompactMessagesForLowVRAM(chatReq.Messages)
		mode = SlidingWindow
		result.Applied = true
		result.Note = "always compacted repeated content"
	case PromptOptimizationLimitOnly:
		switch strings.ToLower(strings.TrimSpace(modelConfig.TruncationMode)) {
		case string(StrictError):
			mode = StrictError
		default:
			mode = SlidingWindow
		}
	case PromptOptimizationLLMAssist:
		assisted, assistedErr := pm.optimizeMessagesWithLLM(modelConfig, chatReq)
		if assistedErr != nil {
			pm.proxyLogger.Warnf("<%s> LLM-assisted optimization failed, falling back to compact mode: %v", modelID, assistedErr)
			assisted.Messages = CompactMessagesForLowVRAM(chatReq.Messages)
		}
		chatReq = assisted
		mode = SlidingWindow
		result.Applied = true
		result.Note = "llm-assisted compression applied"
	default:
		mode = SlidingWindow
	}

	if ctxSize <= 0 {
		if policy != PromptOptimizationAlways {
			updatedBody, err := json.Marshal(chatReq)
			if err != nil {
				return nil, result, fmt.Errorf("failed to serialize optimized chat request: %w", err)
			}
			changed := !bytes.Equal(updatedBody, bodyBytes)
			result.Applied = result.Applied || changed
			if !result.Applied {
				result.Note = "no context limit configured"
			}
			pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
			return updatedBody, result, nil
		}
		updatedBody, err := sjson.SetBytes(bodyBytes, "messages", chatReq.Messages)
		if err != nil {
			return nil, result, fmt.Errorf("failed to update chat messages: %w", err)
		}
		changed := !bytes.Equal(updatedBody, bodyBytes)
		result.Applied = result.Applied || changed
		pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
		return updatedBody, result, nil
	}

	cm := NewContextManager(modelID, ctxSize, mode, pm.proxyLogger, modelConfig.Proxy)
	cropped, err := cm.CropChatRequest(chatReq)
	if err != nil {
		return nil, result, err
	}

	updatedBody := bodyBytes
	updatedBody, err = sjson.SetBytes(updatedBody, "messages", cropped.Messages)
	if err != nil {
		return nil, result, fmt.Errorf("failed to update chat messages: %w", err)
	}

	if len(chatReq.Tools) > 0 || len(cropped.Tools) > 0 {
		updatedBody, err = sjson.SetBytes(updatedBody, "tools", cropped.Tools)
		if err != nil {
			return nil, result, fmt.Errorf("failed to update chat tools: %w", err)
		}
	}

	if cropped.IsCropped() || !bytes.Equal(updatedBody, bodyBytes) {
		result.Applied = true
		if result.Note == "no optimization" {
			result.Note = "cropped to context limit"
		}
		pm.proxyLogger.Infof("<%s> Prompt was compacted to fit ctx-size=%d using mode=%s", modelID, ctxSize, mode)
	}

	pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
	return updatedBody, result, nil
}

func (pm *ProxyManager) savePromptOptimizationSnapshot(
	modelID string,
	policy PromptOptimizationPolicy,
	applied bool,
	originalBody []byte,
	optimizedBody []byte,
	note string,
) {
	const maxSnapshotBytes = 2 * 1024 * 1024
	toSafeString := func(data []byte) string {
		if len(data) <= maxSnapshotBytes {
			return string(data)
		}
		return string(data[:maxSnapshotBytes]) + "\n...<truncated>"
	}

	snapshot := PromptOptimizationSnapshot{
		Model:         modelID,
		Policy:        policy,
		Applied:       applied,
		UpdatedAt:     time.Now().UTC().Format(time.RFC3339),
		Note:          note,
		OriginalBody:  toSafeString(originalBody),
		OptimizedBody: toSafeString(optimizedBody),
	}

	pm.Lock()
	pm.latestPromptOptimizations[modelID] = snapshot
	pm.Unlock()
}

func (pm *ProxyManager) optimizeMessagesWithLLM(modelConfig config.ModelConfig, req ChatRequest) (ChatRequest, error) {
	if len(req.Messages) < 4 {
		return req, nil
	}

	keepTail := 4
	if keepTail > len(req.Messages) {
		keepTail = len(req.Messages)
	}
	middleEnd := len(req.Messages) - keepTail
	if middleEnd <= 1 {
		return req, nil
	}

	keepPrefix := 0
	if req.Messages[0].Role == "system" {
		keepPrefix = 1
	}
	middle := req.Messages[keepPrefix:middleEnd]
	if len(middle) == 0 {
		return req, nil
	}

	var b strings.Builder
	for _, m := range middle {
		if strings.TrimSpace(m.Content) == "" {
			continue
		}
		b.WriteString("[")
		b.WriteString(strings.ToUpper(m.Role))
		b.WriteString("] ")
		b.WriteString(m.Content)
		b.WriteString("\n\n")
		if b.Len() > 12000 {
			break
		}
	}

	summaryInput := b.String()
	if strings.TrimSpace(summaryInput) == "" {
		return req, nil
	}

	upstreamModelName := strings.TrimSpace(modelConfig.UseModelName)
	if upstreamModelName == "" {
		upstreamModelName = strings.TrimSpace(req.Model)
	}
	if upstreamModelName == "" {
		upstreamModelName = "model"
	}

	llmReq := map[string]any{
		"model": upstreamModelName,
		"messages": []map[string]any{
			{
				"role":    "system",
				"content": "Summarize the following chat history for coding continuity. Keep requirements, constraints, file paths, decisions, TODOs, open questions. Be concise. Do not add new facts.",
			},
			{
				"role":    "user",
				"content": summaryInput,
			},
		},
		"max_tokens":  512,
		"temperature": 0,
		"stream":      false,
	}

	reqBytes, err := json.Marshal(llmReq)
	if err != nil {
		return req, err
	}

	url := strings.TrimSuffix(modelConfig.Proxy, "/") + "/v1/chat/completions"
	resp, err := http.Post(url, "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		return req, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return req, fmt.Errorf("llm assistant upstream status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return req, err
	}
	summary := strings.TrimSpace(gjson.GetBytes(body, "choices.0.message.content").String())
	if summary == "" {
		return req, fmt.Errorf("llm assistant returned empty summary")
	}

	newMessages := make([]ChatMessage, 0, keepPrefix+1+keepTail)
	if keepPrefix == 1 {
		newMessages = append(newMessages, req.Messages[0])
	}
	newMessages = append(newMessages, ChatMessage{
		Role:    "system",
		Content: "LLM-assisted context summary:\n" + summary,
	})
	newMessages = append(newMessages, req.Messages[middleEnd:]...)

	req.Messages = newMessages
	return req, nil
}

func (pm *ProxyManager) SetConfigPath(configPath string) {
	pm.Lock()
	defer pm.Unlock()
	pm.configPath = strings.TrimSpace(configPath)
}

func (pm *ProxyManager) proxyOAIPostFormHandler(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil { // 32MB max memory, larger files go to tmp disk
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("error parsing multipart form: %s", err.Error()))
		return
	}

	// Get model parameter from the form
	requestedModel := c.Request.FormValue("model")
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' parameter in form data")
		return
	}

	// Look for a matching local model first, then check peers
	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error
	var useModelName string

	modelID, found := pm.config.RealModelName(requestedModel)
	if found {
		processGroup, err := pm.swapProcessGroup(modelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}

		useModelName = pm.config.Models[modelID].UseModelName
		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		modelID = requestedModel
		nextHandler = pm.peerProxy.ProxyRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable handler for %s", requestedModel))
		return
	}

	// We need to reconstruct the multipart form in any case since the body is consumed
	// Create a new buffer for the reconstructed request
	var requestBuffer bytes.Buffer
	multipartWriter := multipart.NewWriter(&requestBuffer)

	// Copy all form values
	for key, values := range c.Request.MultipartForm.Value {
		for _, value := range values {
			fieldValue := value
			// If this is the model field and we have a profile, use just the model name
			if key == "model" {
				// # issue #69 allow custom model names to be sent to upstream
				if useModelName != "" {
					fieldValue = useModelName
				} else {
					fieldValue = requestedModel
				}
			}
			field, err := multipartWriter.CreateFormField(key)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form field")
				return
			}
			if _, err = field.Write([]byte(fieldValue)); err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error writing form field")
				return
			}
		}
	}

	// Copy all files from the original request
	for key, fileHeaders := range c.Request.MultipartForm.File {
		for _, fileHeader := range fileHeaders {
			formFile, err := multipartWriter.CreateFormFile(key, fileHeader.Filename)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form file")
				return
			}

			file, err := fileHeader.Open()
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error opening uploaded file")
				return
			}

			if _, err = io.Copy(formFile, file); err != nil {
				file.Close()
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error copying file data")
				return
			}
			file.Close()
		}
	}

	// Close the multipart writer to finalize the form
	if err := multipartWriter.Close(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error finalizing multipart form")
		return
	}

	// Create a new request with the reconstructed form data
	modifiedReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		c.Request.URL.String(),
		&requestBuffer,
	)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error creating modified request")
		return
	}

	// Copy the headers from the original request
	modifiedReq.Header = c.Request.Header.Clone()
	modifiedReq.Header.Set("Content-Type", multipartWriter.FormDataContentType())

	// set the content length of the body
	modifiedReq.Header.Set("Content-Length", strconv.Itoa(requestBuffer.Len()))
	modifiedReq.ContentLength = int64(requestBuffer.Len())

	// Use the modified request for proxying
	if err := nextHandler(modelID, c.Writer, modifiedReq); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying Request for model %s", modelID)
		return
	}
}

func (pm *ProxyManager) proxyGETModelHandler(c *gin.Context) {
	requestedModel := c.Query("model")
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing required 'model' query parameter")
		return
	}

	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error
	var modelID string

	if realModelID, found := pm.config.RealModelName(requestedModel); found {
		processGroup, err := pm.swapProcessGroup(realModelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}
		modelID = realModelID
		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		modelID = requestedModel
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		nextHandler = pm.peerProxy.ProxyRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable handler for %s", requestedModel))
		return
	}

	if err := nextHandler(modelID, c.Writer, c.Request); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying GET Request for model %s", modelID)
		return
	}
}

func (pm *ProxyManager) sendErrorResponse(c *gin.Context, statusCode int, message string) {
	acceptHeader := c.GetHeader("Accept")

	if strings.Contains(acceptHeader, "application/json") {
		c.JSON(statusCode, gin.H{"error": message})
	} else {
		c.String(statusCode, message)
	}
}

// apiKeyAuth returns a middleware that validates API keys if configured.
// Returns a pass-through handler if no API keys are configured.
func (pm *ProxyManager) apiKeyAuth() gin.HandlerFunc {
	if len(pm.config.RequiredAPIKeys) == 0 {
		return func(c *gin.Context) { c.Next() }
	}

	return func(c *gin.Context) {
		xApiKey := c.GetHeader("x-api-key")

		var bearerKey string
		var basicKey string
		if auth := c.GetHeader("Authorization"); auth != "" {
			if strings.HasPrefix(auth, "Bearer ") {
				bearerKey = strings.TrimPrefix(auth, "Bearer ")
			} else if strings.HasPrefix(auth, "Basic ") {
				// Basic Auth: base64(username:password), password is the API key
				encoded := strings.TrimPrefix(auth, "Basic ")
				if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
					parts := strings.SplitN(string(decoded), ":", 2)
					if len(parts) == 2 {
						basicKey = parts[1] // password is the API key
					}
				}
			}
		}

		// Use first key found: Basic, then Bearer, then x-api-key
		var providedKey string
		if basicKey != "" {
			providedKey = basicKey
		} else if bearerKey != "" {
			providedKey = bearerKey
		} else {
			providedKey = xApiKey
		}

		// Validate key
		valid := false
		for _, key := range pm.config.RequiredAPIKeys {
			if providedKey == key {
				valid = true
				break
			}
		}

		if !valid {
			c.Header("WWW-Authenticate", `Basic realm="llama-swap"`)
			pm.sendErrorResponse(c, http.StatusUnauthorized, "unauthorized: invalid or missing API key")
			c.Abort()
			return
		}

		// Strip auth headers to prevent leakage to upstream
		c.Request.Header.Del("Authorization")
		c.Request.Header.Del("x-api-key")

		c.Next()
	}
}

func (pm *ProxyManager) unloadAllModelsHandler(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.String(http.StatusOK, "OK")
}

func (pm *ProxyManager) listRunningProcessesHandler(context *gin.Context) {
	context.Header("Content-Type", "application/json")
	runningProcesses := make([]gin.H, 0) // Default to an empty response.

	for _, processGroup := range pm.processGroups {
		for _, process := range processGroup.processes {
			if process.CurrentState() == StateReady {
				runningProcesses = append(runningProcesses, gin.H{
					"model":       process.ID,
					"state":       process.state,
					"cmd":         process.config.Cmd,
					"proxy":       process.config.Proxy,
					"ttl":         process.config.UnloadAfter,
					"name":        process.config.Name,
					"description": process.config.Description,
				})
			}
		}
	}

	// Put the results under the `running` key.
	response := gin.H{
		"running": runningProcesses,
	}

	context.JSON(http.StatusOK, response) // Always return 200 OK
}

func (pm *ProxyManager) findGroupByModelName(modelName string) *ProcessGroup {
	for _, group := range pm.processGroups {
		if group.HasMember(modelName) {
			return group
		}
	}
	return nil
}

func (pm *ProxyManager) SetVersion(buildDate string, commit string, version string) {
	pm.Lock()
	defer pm.Unlock()
	pm.buildDate = buildDate
	pm.commit = commit
	pm.version = version
}
