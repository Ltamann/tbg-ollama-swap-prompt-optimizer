package proxy

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ParsedToolCall struct {
	CallID    string
	Name      string
	Arguments map[string]any
}

type ToolCallParser interface {
	Name() string
	MatchesModel(modelName string) bool
	ParseAssistantOutput(content string) ([]ParsedToolCall, string)
}

type toolCallParserRegistry struct {
	mu      sync.RWMutex
	parsers []ToolCallParser
}

func (r *toolCallParserRegistry) register(parser ToolCallParser) {
	if parser == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.parsers = append(r.parsers, parser)
}

func (r *toolCallParserRegistry) parse(modelName, content string) ([]ParsedToolCall, string) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, parser := range r.parsers {
		if parser.MatchesModel(modelName) {
			return parser.ParseAssistantOutput(content)
		}
	}
	return nil, strings.TrimSpace(content)
}

var defaultToolCallParserRegistry = newDefaultToolCallParserRegistry()

func newDefaultToolCallParserRegistry() *toolCallParserRegistry {
	registry := &toolCallParserRegistry{}
	registry.register(qwenXMLToolCallParser{})
	return registry
}

func parseModelSpecificToolCalls(modelName, content string) ([]ParsedToolCall, string) {
	return defaultToolCallParserRegistry.parse(modelName, content)
}

type qwenXMLToolCallParser struct{}

var (
	qwenFunctionAttrNameRe  = regexp.MustCompile(`(?is)<function\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>`)
	qwenParameterAttrNameRe = regexp.MustCompile(`(?is)<parameter\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>`)
)

func (qwenXMLToolCallParser) Name() string { return "qwen_xml" }

func (qwenXMLToolCallParser) MatchesModel(modelName string) bool {
	modelName = strings.TrimSpace(strings.ToLower(modelName))
	if strings.Contains(modelName, "qwen") {
		return true
	}
	// Local llama-swap configs often expose Qwen backends via Codex-style aliases.
	if strings.Contains(modelName, "codex") || strings.HasPrefix(modelName, "gpt-5") {
		return true
	}
	return false
}

func (qwenXMLToolCallParser) ParseAssistantOutput(content string) ([]ParsedToolCall, string) {
	return parseQwenXMLToolCalls(content)
}

func parseQwenXMLToolCalls(text string) ([]ParsedToolCall, string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, ""
	}

	lower := strings.ToLower(text)
	toolOpen := "<tool_call>"
	toolClose := "</tool_call>"
	searchPos := 0
	parts := make([]string, 0, 2)
	calls := make([]ParsedToolCall, 0, 2)

	for {
		openIdx := strings.Index(lower[searchPos:], toolOpen)
		if openIdx < 0 {
			break
		}
		openIdx += searchPos
		closeIdx := strings.Index(lower[openIdx:], toolClose)
		if closeIdx < 0 {
			break
		}
		closeIdx += openIdx
		blockEnd := closeIdx + len(toolClose)

		if pre := strings.TrimSpace(text[searchPos:openIdx]); pre != "" {
			parts = append(parts, pre)
		}

		block := text[openIdx:blockEnd]
		if call, ok := parseSingleQwenXMLToolCall(block); ok {
			calls = append(calls, call)
		} else if b := strings.TrimSpace(block); b != "" {
			parts = append(parts, b)
		}

		searchPos = blockEnd
	}

	if tail := strings.TrimSpace(text[searchPos:]); tail != "" {
		parts = append(parts, tail)
	}
	if len(calls) == 0 {
		calls, remaining := parseQwenBareFunctionBlocks(text)
		if len(calls) > 0 {
			return calls, remaining
		}
		calls, remaining = parseQwenToolsEnvelopeCalls(text)
		if len(calls) > 0 {
			return calls, remaining
		}
		calls, remaining = parseQwenTaggedEnvelopeCalls(text)
		if len(calls) > 0 {
			return calls, remaining
		}
		calls, remaining = parseDecoratedPlanCalls(text)
		if len(calls) > 0 {
			return calls, remaining
		}
		// Do not synthesize planning tool calls from plain prose blocks.
		if _, stripped := parseGenericTaggedCalls(text); strings.TrimSpace(stripped) != strings.TrimSpace(text) {
			return nil, stripped
		}
		return parseQwenFunctionStyleToolCalls(text)
	}
	return calls, strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func parseQwenToolsEnvelopeCalls(text string) ([]ParsedToolCall, string) {
	normalized := strings.TrimSpace(text)
	if normalized == "" {
		return nil, ""
	}
	normalized = strings.ReplaceAll(normalized, "\\u003c", "<")
	normalized = strings.ReplaceAll(normalized, "\\u003e", ">")
	normalized = strings.ReplaceAll(normalized, "\\u003C", "<")
	normalized = strings.ReplaceAll(normalized, "\\u003E", ">")

	lower := strings.ToLower(normalized)
	openTag := "<tools>"
	closeTag := "</tools>"
	openIdx := strings.Index(lower, openTag)
	if openIdx < 0 {
		return nil, strings.TrimSpace(text)
	}
	closeRel := strings.Index(lower[openIdx:], closeTag)
	if closeRel < 0 {
		return nil, strings.TrimSpace(text)
	}
	closeIdx := openIdx + closeRel
	blockEnd := closeIdx + len(closeTag)

	body := strings.TrimSpace(normalized[openIdx+len(openTag) : closeIdx])
	if body == "" {
		return nil, strings.TrimSpace(text)
	}

	args := map[string]any{}
	if err := json.Unmarshal([]byte(body), &args); err != nil {
		return nil, strings.TrimSpace(text)
	}

	name := inferToolsEnvelopeFunctionName(args)
	if name == "" {
		return nil, strings.TrimSpace(text)
	}
	if name == "shell_command" {
		if cmd := normalizeShellCommandArgument(args["command"]); cmd != "" {
			args["commands"] = []any{cmd}
			delete(args, "command")
		} else if cmds := normalizeShellCommandsFromAny(args["commands"]); len(cmds) > 0 {
			arr := make([]any, 0, len(cmds))
			for _, c := range cmds {
				arr = append(arr, c)
			}
			args["commands"] = arr
			delete(args, "command")
		}
	}

	call := ParsedToolCall{
		CallID:    fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:      name,
		Arguments: args,
	}
	prefix := strings.TrimSpace(normalized[:openIdx])
	suffix := strings.TrimSpace(normalized[blockEnd:])
	remaining := strings.TrimSpace(strings.Join([]string{prefix, suffix}, "\n\n"))
	return []ParsedToolCall{call}, remaining
}

func inferToolsEnvelopeFunctionName(args map[string]any) string {
	if args == nil {
		return ""
	}
	if raw, ok := args["name"]; ok {
		if name := strings.TrimSpace(fmt.Sprintf("%v", raw)); name != "" {
			return name
		}
	}
	if _, ok := args["command"]; ok {
		return "shell_command"
	}
	if _, ok := args["commands"]; ok {
		return "shell_command"
	}
	if _, ok := args["operation"]; ok {
		return "apply_patch"
	}
	return ""
}

func normalizeShellCommandArgument(raw any) string {
	switch val := raw.(type) {
	case string:
		return strings.TrimSpace(val)
	case []any:
		parts := make([]string, 0, len(val))
		for _, item := range val {
			part := strings.TrimSpace(fmt.Sprintf("%v", item))
			if part == "" {
				continue
			}
			if strings.ContainsAny(part, " \t") {
				part = strconv.Quote(part)
			}
			parts = append(parts, part)
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", raw))
	}
}

func normalizeShellCommandsFromAny(raw any) []string {
	switch typed := raw.(type) {
	case nil:
		return nil
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			s := strings.TrimSpace(fmt.Sprintf("%v", item))
			if s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(typed))
		for _, s := range typed {
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
		return out
	case string:
		s := strings.TrimSpace(typed)
		if s == "" {
			return nil
		}
		var arr []string
		if json.Unmarshal([]byte(s), &arr) == nil {
			out := make([]string, 0, len(arr))
			for _, item := range arr {
				item = strings.TrimSpace(item)
				if item != "" {
					out = append(out, item)
				}
			}
			return out
		}
		return []string{s}
	default:
		s := strings.TrimSpace(fmt.Sprintf("%v", typed))
		if s == "" {
			return nil
		}
		return []string{s}
	}
}

func normalizeShellArgsMapForOutput(args map[string]any) map[string]any {
	if args == nil {
		return map[string]any{}
	}
	out := map[string]any{}
	for k, v := range args {
		if k != "command" && k != "commands" {
			out[k] = v
		}
	}
	commands := normalizeShellCommandsFromAny(args["commands"])
	if len(commands) == 0 {
		if cmd := normalizeShellCommandArgument(args["command"]); cmd != "" {
			commands = []string{cmd}
		}
	}
	if len(commands) > 0 {
		out["commands"] = commands
	}
	return out
}

func parseQwenTaggedEnvelopeCalls(text string) ([]ParsedToolCall, string) {
	normalized := strings.TrimSpace(text)
	if normalized == "" {
		return nil, ""
	}
	normalized = strings.ReplaceAll(normalized, "\\u003c", "<")
	normalized = strings.ReplaceAll(normalized, "\\u003e", ">")
	normalized = strings.ReplaceAll(normalized, "\\u003C", "<")
	normalized = strings.ReplaceAll(normalized, "\\u003E", ">")

	working := normalized
	calls := make([]ParsedToolCall, 0, 3)

	applyPatchCalls, next := parseQwenApplyPatchTaggedCalls(working)
	calls = append(calls, applyPatchCalls...)
	working = next

	shellCalls, next := parseQwenShellCommandsTaggedCalls(working)
	calls = append(calls, shellCalls...)
	working = next

	toolUseCalls, next := parseQwenToolUseTaggedCalls(working)
	calls = append(calls, toolUseCalls...)
	working = next

	planCalls, next := parseQwenPlanTaggedCalls(working)
	calls = append(calls, planCalls...)
	working = next

	genericCalls, next := parseGenericTaggedCalls(working)
	calls = append(calls, genericCalls...)
	working = next

	if len(calls) == 0 {
		return nil, strings.TrimSpace(text)
	}
	return calls, strings.TrimSpace(working)
}

func parseGenericTaggedCalls(text string) ([]ParsedToolCall, string) {
	working := text
	out := make([]ParsedToolCall, 0, 2)
	tagStartRe := regexp.MustCompile(`(?is)<([a-zA-Z][a-zA-Z0-9_-]*)[^>]*>`)
	for {
		m := tagStartRe.FindStringSubmatch(working)
		if len(m) < 2 {
			break
		}
		tag := strings.ToLower(strings.TrimSpace(m[1]))
		block, blockStart, blockEnd, ok := extractTagBlockLoose(working, tag)
		if !ok {
			break
		}
		toolName := inferToolNameFromGenericTag(tag, block)
		if toolName != "" {
			if call, ok := buildGenericTaggedToolCall(toolName, block); ok {
				out = append(out, call)
			}
		}
		// Always strip leftover XML-like blocks from assistant text.
		working = strings.TrimSpace(working[:blockStart] + "\n" + working[blockEnd:])
	}
	return out, working
}

func inferToolNameFromGenericTag(tag string, block string) string {
	switch strings.ToLower(strings.TrimSpace(tag)) {
	case "apply_patch", "patch":
		return "apply_patch"
	case "shell", "shell_command", "terminal", "bash", "sh", "command":
		return "shell_command"
	case "web_search", "websearch":
		return "web_search"
	}

	return ""
}

func buildGenericTaggedToolCall(toolName string, block string) (ParsedToolCall, bool) {
	call := ParsedToolCall{
		CallID:    fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:      toolName,
		Arguments: map[string]any{},
	}
	raw := strings.TrimSpace(block)
	if raw == "" {
		return ParsedToolCall{}, false
	}

	switch toolName {
	case "apply_patch":
		patchText := raw
		if strings.Contains(strings.ToLower(patchText), "*** add file:") ||
			strings.Contains(strings.ToLower(patchText), "*** update file:") ||
			strings.Contains(strings.ToLower(patchText), "*** delete file:") {
			patchText = normalizeApplyPatchText(patchText)
		}
		call.Arguments["input"] = patchText
		return call, true
	case "shell_command":
		cmd := ""
		var obj map[string]any
		if json.Unmarshal([]byte(raw), &obj) == nil {
			if c := normalizeShellCommandArgument(obj["command"]); c != "" {
				cmd = c
			} else if list := normalizeShellCommandsFromAny(obj["commands"]); len(list) > 0 {
				return ParsedToolCall{
					CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Name:   toolName,
					Arguments: map[string]any{
						"commands": list,
					},
				}, true
			}
		}
		if cmd == "" {
			cmd = parseTaggedSingleShellCommand(raw)
		}
		if cmd == "" && !strings.Contains(raw, "<") {
			cmd = strings.TrimSpace(raw)
		}
		if cmd == "" {
			return ParsedToolCall{}, false
		}
		call.Arguments["commands"] = []string{cmd}
		return call, true
	default:
		return ParsedToolCall{}, false
	}
}

func parseQwenApplyPatchTaggedCalls(text string) ([]ParsedToolCall, string) {
	working := text
	out := make([]ParsedToolCall, 0, 1)
	for {
		block, blockStart, blockEnd, ok := extractTagBlockLoose(working, "apply_patch")
		if !ok {
			break
		}
		patchText := strings.TrimSpace(block)
		if strings.Contains(strings.ToLower(patchText), "*** add file:") ||
			strings.Contains(strings.ToLower(patchText), "*** update file:") ||
			strings.Contains(strings.ToLower(patchText), "*** delete file:") {
			patchText = normalizeApplyPatchText(patchText)
		}
		if patchText != "" {
			out = append(out, ParsedToolCall{
				CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
				Name:   "apply_patch",
				Arguments: map[string]any{
					"input": patchText,
				},
			})
		}
		working = strings.TrimSpace(working[:blockStart] + "\n" + working[blockEnd:])
	}
	return out, working
}

func parseQwenShellCommandsTaggedCalls(text string) ([]ParsedToolCall, string) {
	working := text
	out := make([]ParsedToolCall, 0, 2)
	for _, shellTag := range []string{"shell_commands", "shell_command", "shell"} {
		for {
			block, blockStart, blockEnd, ok := extractTagBlockLoose(working, shellTag)
			if !ok {
				break
			}
			parsed := parseTaggedShellCommandsFromBlock(block)
			out = append(out, parsed...)
			working = strings.TrimSpace(working[:blockStart] + "\n" + working[blockEnd:])
		}
	}
	return out, working
}

func parseTaggedShellCommandsFromBlock(block string) []ParsedToolCall {
	commandsRaw := strings.TrimSpace(block)
	if nested, _, _, ok := extractTagBlock(block, "commands"); ok {
		commandsRaw = strings.TrimSpace(nested)
	}
	out := make([]ParsedToolCall, 0, 2)
	var commands []any
	if json.Unmarshal([]byte(commandsRaw), &commands) == nil {
		for _, item := range commands {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			command := normalizeShellCommandArgument(obj["command"])
			if command == "" {
				if list := normalizeShellCommandsFromAny(obj["commands"]); len(list) > 0 {
					out = append(out, ParsedToolCall{
						CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
						Name:   "shell_command",
						Arguments: map[string]any{
							"commands": list,
						},
					})
				}
				continue
			}
			out = append(out, ParsedToolCall{
				CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
				Name:   "shell_command",
				Arguments: map[string]any{
					"commands": []string{command},
				},
			})
		}
		return out
	}
	if command := parseTaggedSingleShellCommand(block); command != "" {
		return []ParsedToolCall{{
			CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
			Name:   "shell_command",
			Arguments: map[string]any{
				"commands": []string{command},
			},
		}}
	}
	return nil
}

func parseQwenPlanTaggedCalls(text string) ([]ParsedToolCall, string) {
	working := text
	out := make([]ParsedToolCall, 0, 2)
	for _, tag := range []string{"update_plan", "proposed_plan"} {
		for {
			block, blockStart, blockEnd, ok := extractTagBlockLoose(working, tag)
			if !ok {
				break
			}
			stepsRaw := strings.TrimSpace(block)
			if nested, _, _, ok := extractTagBlock(block, "steps"); ok {
				stepsRaw = strings.TrimSpace(nested)
			}
			plan := normalizeTaggedPlanSteps(stepsRaw)
			if len(plan) > 0 {
				out = append(out, ParsedToolCall{
					CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Name:   "update_plan",
					Arguments: map[string]any{
						"plan": plan,
					},
				})
			}
			working = strings.TrimSpace(working[:blockStart] + "\n" + working[blockEnd:])
		}
	}
	return out, working
}

func parseQwenToolUseTaggedCalls(text string) ([]ParsedToolCall, string) {
	working := text
	out := make([]ParsedToolCall, 0, 2)
	for _, tag := range []string{"tool_use", "tool_call"} {
		for {
			block, blockStart, blockEnd, ok := extractTagBlockLoose(working, tag)
			if !ok {
				break
			}
			out = append(out, parseTaggedToolUseFromBlock(block)...)
			working = strings.TrimSpace(working[:blockStart] + "\n" + working[blockEnd:])
		}
	}
	return out, working
}

func parseTaggedToolUseFromBlock(block string) []ParsedToolCall {
	// Recover nested shell-style tags inside generic wrapper.
	lower := strings.ToLower(block)
	if strings.Contains(lower, "<shell") || strings.Contains(lower, "<shell_command") || strings.Contains(lower, "<shell_commands") {
		if calls := parseTaggedShellCommandsFromBlock(block); len(calls) > 0 {
			return calls
		}
	}
	// Recover update/proposed plan blocks nested inside wrapper.
	if nested, _, _, ok := extractTagBlockLoose(block, "update_plan"); ok {
		if plan := normalizeTaggedPlanSteps(strings.TrimSpace(nested)); len(plan) > 0 {
			return []ParsedToolCall{{
				CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
				Name:   "update_plan",
				Arguments: map[string]any{
					"plan": plan,
				},
			}}
		}
	}
	if nested, _, _, ok := extractTagBlockLoose(block, "proposed_plan"); ok {
		if plan := normalizeTaggedPlanSteps(strings.TrimSpace(nested)); len(plan) > 0 {
			return []ParsedToolCall{{
				CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
				Name:   "update_plan",
				Arguments: map[string]any{
					"plan": plan,
				},
			}}
		}
	}
	return nil
}

func normalizeTaggedPlanSteps(raw string) []map[string]any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	raw = strings.TrimPrefix(raw, "steps=")
	raw = strings.TrimSpace(raw)
	var wrapped map[string]any
	if err := json.Unmarshal([]byte(raw), &wrapped); err == nil {
		if planRaw, ok := wrapped["plan"]; ok {
			return normalizeFunctionStylePlan(planRaw)
		}
		if stepsRaw, ok := wrapped["steps"]; ok {
			return normalizeFunctionStylePlan(stepsRaw)
		}
	}
	if cleaned := cleanLoosePlanText(raw); cleaned != raw {
		if err := json.Unmarshal([]byte(cleaned), &wrapped); err == nil {
			if planRaw, ok := wrapped["plan"]; ok {
				return normalizeFunctionStylePlan(planRaw)
			}
			if stepsRaw, ok := wrapped["steps"]; ok {
				return normalizeFunctionStylePlan(stepsRaw)
			}
		}
	}
	if plan := parsePlanStepsFromLooseString(raw); len(plan) > 0 {
		return plan
	}
	var steps []any
	if err := json.Unmarshal([]byte(raw), &steps); err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(steps))
	for _, item := range steps {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		step := mapValueAsString(obj, "step")
		if step == "" {
			step = mapValueAsString(obj, "content")
		}
		if step == "" {
			continue
		}
		status := mapValueAsString(obj, "status")
		if status == "" {
			status = "pending"
		}
		out = append(out, map[string]any{
			"step":   step,
			"status": status,
		})
	}
	return out
}

func mapValueAsString(obj map[string]any, key string) string {
	if obj == nil {
		return ""
	}
	raw, ok := obj[key]
	if !ok || raw == nil {
		return ""
	}
	return strings.TrimSpace(fmt.Sprintf("%v", raw))
}

func extractTagBlock(text, tag string) (content string, blockStart int, blockEnd int, ok bool) {
	lower := strings.ToLower(text)
	openTag := "<" + strings.ToLower(tag) + ">"
	closeTag := "</" + strings.ToLower(tag) + ">"
	openIdx := strings.Index(lower, openTag)
	if openIdx < 0 {
		return "", 0, 0, false
	}
	closeRel := strings.Index(lower[openIdx+len(openTag):], closeTag)
	if closeRel < 0 {
		return "", 0, 0, false
	}
	closeIdx := openIdx + len(openTag) + closeRel
	return text[openIdx+len(openTag) : closeIdx], openIdx, closeIdx + len(closeTag), true
}

func extractTagBlockLoose(text, tag string) (content string, blockStart int, blockEnd int, ok bool) {
	if content, blockStart, blockEnd, ok = extractTagBlock(text, tag); ok {
		return content, blockStart, blockEnd, true
	}
	lower := strings.ToLower(text)
	openTag := "<" + strings.ToLower(tag) + ">"
	openIdx := strings.Index(lower, openTag)
	if openIdx < 0 {
		return "", 0, 0, false
	}
	start := openIdx + len(openTag)
	return text[start:], openIdx, len(text), true
}

func parseTaggedSingleShellCommand(block string) string {
	lowerBlock := strings.ToLower(block)
	if strings.Contains(lowerBlock, "</command>") {
		if inner, _, _, ok := extractTagBlock(block, "command"); ok {
			inner = strings.TrimSpace(inner)
			if inner != "" {
				return inner
			}
		}
	}
	trimmed := strings.TrimSpace(block)
	if !strings.HasPrefix(trimmed, "<") || !strings.Contains(trimmed, "</") {
		return ""
	}
	start := strings.Index(trimmed, "<")
	end := strings.Index(trimmed[start+1:], ">")
	if start < 0 || end < 0 {
		return ""
	}
	end += start + 1
	tagName := strings.TrimSpace(trimmed[start+1 : end])
	if tagName == "" || strings.ContainsAny(tagName, "<>\"'") {
		return ""
	}
	closeTag := "</" + tagName + ">"
	closeIdx := strings.LastIndex(strings.ToLower(trimmed), strings.ToLower(closeTag))
	if closeIdx <= end {
		return ""
	}
	body := strings.TrimSpace(trimmed[end+1 : closeIdx])
	if body == "" {
		return strings.TrimSpace(tagName)
	}
	return strings.TrimSpace(tagName + " " + body)
}

func parseQwenBareFunctionBlocks(text string) ([]ParsedToolCall, string) {
	lower := strings.ToLower(text)
	funcOpen := "<function="
	funcClose := "</function>"
	searchPos := 0
	parts := make([]string, 0, 2)
	calls := make([]ParsedToolCall, 0, 2)

	for {
		openIdx := strings.Index(lower[searchPos:], funcOpen)
		if openIdx < 0 {
			break
		}
		openIdx += searchPos
		closeRel := strings.Index(lower[openIdx:], funcClose)
		if closeRel < 0 {
			break
		}
		blockEnd := openIdx + closeRel + len(funcClose)

		if pre := strings.TrimSpace(text[searchPos:openIdx]); pre != "" {
			parts = append(parts, pre)
		}

		block := text[openIdx:blockEnd]
		if call, ok := parseSingleQwenXMLToolCall(block); ok {
			calls = append(calls, call)
		} else if b := strings.TrimSpace(block); b != "" {
			parts = append(parts, b)
		}

		searchPos = blockEnd
	}

	if tail := strings.TrimSpace(text[searchPos:]); tail != "" {
		parts = append(parts, tail)
	}

	return calls, strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func parseQwenFunctionStyleToolCalls(text string) ([]ParsedToolCall, string) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, ""
	}
	lower := strings.ToLower(trimmed)
	candidates := []string{"apply_patch(", "shell(", "shell_command(", "web_search(", "file_search(", "code_interpreter(", "image_generation(", "computer(", "update_plan(", "spawn_agent(", "send_input(", "resume_agent(", "wait_agent(", "close_agent("}
	callStart := -1
	for _, c := range candidates {
		if idx := strings.Index(lower, c); idx >= 0 && (callStart < 0 || idx < callStart) {
			callStart = idx
		}
	}
	if callStart < 0 {
		return nil, trimmed
	}

	openParen := strings.Index(trimmed[callStart:], "(")
	if openParen < 0 {
		return nil, trimmed
	}
	openParen += callStart
	name := strings.TrimSpace(trimmed[callStart:openParen])
	if name == "" {
		return nil, trimmed
	}

	closeParen := findMatchingParen(trimmed, openParen)
	if closeParen <= openParen {
		return nil, trimmed
	}

	argExpr := strings.TrimSpace(trimmed[openParen+1 : closeParen])
	args := parseFunctionStyleArguments(argExpr)
	if len(args) == 0 {
		return nil, trimmed
	}

	call := ParsedToolCall{
		CallID:    fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:      name,
		Arguments: args,
	}
	if strings.EqualFold(name, "shell") || strings.EqualFold(name, "shell_command") {
		call.Arguments = normalizeShellArgsMapForOutput(call.Arguments)
	}
	if strings.EqualFold(name, "update_plan") {
		if steps, ok := call.Arguments["steps"]; ok {
			if plan := normalizeFunctionStylePlan(steps); len(plan) > 0 {
				call.Arguments = map[string]any{"plan": plan}
			}
		}
		if planRaw, ok := call.Arguments["plan"]; ok {
			if plan := normalizeFunctionStylePlan(planRaw); len(plan) > 0 {
				call.Arguments = map[string]any{"plan": plan}
			}
		}
	}
	prefix := strings.TrimSpace(trimmed[:callStart])
	suffix := strings.TrimSpace(trimmed[closeParen+1:])
	remaining := strings.TrimSpace(strings.Join([]string{prefix, suffix}, "\n\n"))
	if remaining == "[]" || remaining == "[" || remaining == "]" || remaining == "[\n\n]" {
		remaining = ""
	}
	return []ParsedToolCall{call}, remaining
}

func normalizeFunctionStylePlan(raw any) []map[string]any {
	switch v := raw.(type) {
	case []any:
		out := make([]map[string]any, 0, len(v))
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			step := mapValueAsString(obj, "step")
			if step == "" {
				step = mapValueAsString(obj, "content")
			}
			if step == "" {
				continue
			}
			status := mapValueAsString(obj, "status")
			if status == "" {
				status = "pending"
			}
			out = append(out, map[string]any{
				"step":   step,
				"status": status,
			})
		}
		return out
	case string:
		return parsePlanStepsFromLooseString(v)
	default:
		return nil
	}
}

func parsePlanStepsFromLooseString(raw string) []map[string]any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	if strings.HasPrefix(raw, "steps=") {
		raw = strings.TrimSpace(strings.TrimPrefix(raw, "steps="))
	}
	// First try JSON after normalizing malformed wrappers and single quotes.
	candidates := []string{
		raw,
		cleanLoosePlanText(raw),
	}
	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		normalized := strings.ReplaceAll(candidate, "'", "\"")

		var wrapped map[string]any
		if json.Unmarshal([]byte(normalized), &wrapped) == nil {
			if planRaw, ok := wrapped["plan"]; ok {
				if out := normalizeFunctionStylePlan(planRaw); len(out) > 0 {
					return out
				}
			}
			if stepsRaw, ok := wrapped["steps"]; ok {
				if out := normalizeFunctionStylePlan(stepsRaw); len(out) > 0 {
					return out
				}
			}
		}

		var steps []any
		if json.Unmarshal([]byte(normalized), &steps) == nil {
			out := normalizeFunctionStylePlan(steps)
			if len(out) > 0 {
				return out
			}
		}

		if extracted := extractFirstJSONObjectOrArray(normalized); extracted != "" {
			if json.Unmarshal([]byte(extracted), &wrapped) == nil {
				if planRaw, ok := wrapped["plan"]; ok {
					if out := normalizeFunctionStylePlan(planRaw); len(out) > 0 {
						return out
					}
				}
				if stepsRaw, ok := wrapped["steps"]; ok {
					if out := normalizeFunctionStylePlan(stepsRaw); len(out) > 0 {
						return out
					}
				}
			}
			if json.Unmarshal([]byte(extracted), &steps) == nil {
				out := normalizeFunctionStylePlan(steps)
				if len(out) > 0 {
					return out
				}
			}
		}
	}
	// Fallback: extract step/status pairs from python-like dict snippets.
	re := regexp.MustCompile(`content["']?\s*:\s*["']([^"']+)["'].*?status["']?\s*:\s*["']([^"']+)["']`)
	matches := re.FindAllStringSubmatch(raw, -1)
	if len(matches) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(matches))
	for _, m := range matches {
		if len(m) < 3 {
			continue
		}
		step := strings.TrimSpace(m[1])
		status := strings.TrimSpace(m[2])
		if step == "" {
			continue
		}
		if status == "" {
			status = "pending"
		}
		out = append(out, map[string]any{
			"step":   step,
			"status": status,
		})
	}
	return out
}

func cleanLoosePlanText(raw string) string {
	cleaned := strings.TrimSpace(raw)
	if cleaned == "" {
		return ""
	}
	// Remove stray XML-like tags that models sometimes leak around JSON payloads.
	tagRe := regexp.MustCompile(`</?[A-Za-z_][A-Za-z0-9_:\-]*>`)
	cleaned = tagRe.ReplaceAllString(cleaned, "")
	return strings.TrimSpace(cleaned)
}

func extractFirstJSONObjectOrArray(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	startObj := strings.Index(raw, "{")
	startArr := strings.Index(raw, "[")
	start := -1
	open := byte(0)
	close := byte(0)
	switch {
	case startObj >= 0 && (startArr < 0 || startObj < startArr):
		start = startObj
		open = '{'
		close = '}'
	case startArr >= 0:
		start = startArr
		open = '['
		close = ']'
	default:
		return ""
	}
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(raw); i++ {
		ch := raw[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			continue
		}
		if ch == open {
			depth++
			continue
		}
		if ch == close {
			depth--
			if depth == 0 {
				return strings.TrimSpace(raw[start : i+1])
			}
		}
	}
	return ""
}

func parseDecoratedPlanCalls(text string) ([]ParsedToolCall, string) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, ""
	}
	lower := strings.ToLower(trimmed)
	if !strings.Contains(lower, "begin plan") || !strings.Contains(lower, "end plan") {
		return nil, trimmed
	}
	lines := strings.Split(trimmed, "\n")
	steps := make([]map[string]any, 0, 4)
	for _, line := range lines {
		s := strings.TrimSpace(line)
		if len(s) < 3 {
			continue
		}
		if s[0] >= '1' && s[0] <= '9' && strings.HasPrefix(s[1:], ". ") {
			stepText := strings.TrimSpace(s[3:])
			if stepText == "" {
				continue
			}
			status := "pending"
			if len(steps) == 0 {
				status = "in_progress"
			}
			steps = append(steps, map[string]any{
				"step":   stepText,
				"status": status,
			})
		}
	}
	if len(steps) == 0 {
		return nil, trimmed
	}
	return []ParsedToolCall{{
		CallID: fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:   "update_plan",
		Arguments: map[string]any{
			"plan": steps,
		},
	}}, ""
}

func findMatchingParen(s string, open int) int {
	depth := 0
	inQuote := false
	escaped := false
	for i := open; i < len(s); i++ {
		ch := s[i]
		if inQuote {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inQuote = false
			}
			continue
		}
		switch ch {
		case '"':
			inQuote = true
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

func parseFunctionStyleArguments(expr string) map[string]any {
	args := map[string]any{}
	if strings.TrimSpace(expr) == "" {
		return args
	}
	parts := splitTopLevelComma(expr)
	for _, part := range parts {
		eq := strings.Index(part, "=")
		if eq <= 0 {
			continue
		}
		key := strings.TrimSpace(part[:eq])
		value := strings.TrimSpace(part[eq+1:])
		if key == "" || value == "" {
			continue
		}
		args[key] = parseFunctionStyleValue(value)
	}
	return args
}

func splitTopLevelComma(s string) []string {
	items := []string{}
	start := 0
	braceDepth := 0
	bracketDepth := 0
	parenDepth := 0
	inQuote := false
	escaped := false

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inQuote {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inQuote = false
			}
			continue
		}
		switch ch {
		case '"':
			inQuote = true
		case '{':
			braceDepth++
		case '}':
			if braceDepth > 0 {
				braceDepth--
			}
		case '[':
			bracketDepth++
		case ']':
			if bracketDepth > 0 {
				bracketDepth--
			}
		case '(':
			parenDepth++
		case ')':
			if parenDepth > 0 {
				parenDepth--
			}
		case ',':
			if braceDepth == 0 && bracketDepth == 0 && parenDepth == 0 {
				items = append(items, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	last := strings.TrimSpace(s[start:])
	if last != "" {
		items = append(items, last)
	}
	return items
}

func parseFunctionStyleValue(raw string) any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "\"") && strings.HasSuffix(raw, "\"") {
		if v, err := strconv.Unquote(raw); err == nil {
			return v
		}
		return strings.Trim(raw, "\"")
	}
	if strings.HasPrefix(raw, "{") || strings.HasPrefix(raw, "[") {
		var parsed any
		if json.Unmarshal([]byte(raw), &parsed) == nil {
			return parsed
		}
	}
	if strings.EqualFold(raw, "true") {
		return true
	}
	if strings.EqualFold(raw, "false") {
		return false
	}
	if strings.EqualFold(raw, "null") {
		return nil
	}
	return raw
}

func parseSingleQwenXMLToolCall(block string) (ParsedToolCall, bool) {
	lower := strings.ToLower(block)
	name := ""
	nameEnd := -1
	if match := qwenFunctionAttrNameRe.FindStringSubmatchIndex(block); len(match) >= 4 {
		name = strings.TrimSpace(block[match[2]:match[3]])
		nameEnd = match[1] - 1
	} else {
		funcOpen := "<function="
		funcStart := strings.Index(lower, funcOpen)
		if funcStart < 0 {
			return ParsedToolCall{}, false
		}
		nameStart := funcStart + len(funcOpen)
		nameEndRel := strings.Index(lower[nameStart:], ">")
		if nameEndRel < 0 {
			return ParsedToolCall{}, false
		}
		nameEnd = nameStart + nameEndRel
		name = strings.TrimSpace(block[nameStart:nameEnd])
	}
	if name == "" {
		return ParsedToolCall{}, false
	}

	funcClose := "</function>"
	funcCloseIdx := strings.LastIndex(lower, funcClose)
	if funcCloseIdx < 0 || funcCloseIdx <= nameEnd {
		return ParsedToolCall{}, false
	}

	inner := block[nameEnd+1 : funcCloseIdx]
	call := ParsedToolCall{
		CallID:    fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:      name,
		Arguments: parseQwenXMLParameters(inner),
	}
	if strings.EqualFold(name, "shell") || strings.EqualFold(name, "shell_command") {
		call.Arguments = normalizeShellArgsMapForOutput(call.Arguments)
	}
	return call, true
}

func parseQwenXMLParameters(inner string) map[string]any {
	args := map[string]any{}
	if strings.TrimSpace(inner) == "" {
		return args
	}

	lower := strings.ToLower(inner)
	paramOpen := "<parameter="
	paramOpenAttr := "<parameter"
	paramClose := "</parameter>"
	funcClose := "</function>"
	searchPos := 0

	for {
		openIdxEq := strings.Index(lower[searchPos:], paramOpen)
		openIdxAttr := strings.Index(lower[searchPos:], paramOpenAttr)
		openIdx := -1
		attrMode := false
		switch {
		case openIdxEq >= 0 && openIdxAttr >= 0:
			if openIdxEq <= openIdxAttr {
				openIdx = openIdxEq
			} else {
				openIdx = openIdxAttr
				attrMode = true
			}
		case openIdxEq >= 0:
			openIdx = openIdxEq
		case openIdxAttr >= 0:
			openIdx = openIdxAttr
			attrMode = true
		}
		if openIdx < 0 {
			break
		}
		openIdx += searchPos
		tagEndRel := strings.Index(inner[openIdx:], ">")
		if tagEndRel < 0 {
			break
		}
		tagEnd := openIdx + tagEndRel
		key := ""
		if attrMode {
			tagText := inner[openIdx : tagEnd+1]
			if match := qwenParameterAttrNameRe.FindStringSubmatch(tagText); len(match) == 2 {
				key = strings.TrimSpace(match[1])
			}
		} else {
			keyStart := openIdx + len(paramOpen)
			key = strings.TrimSpace(inner[keyStart:tagEnd])
		}
		if key == "" {
			searchPos = tagEnd + 1
			continue
		}

		valueStart := tagEnd + 1
		closeTagIdx := strings.Index(lower[valueStart:], paramClose)
		nextParamIdx := strings.Index(lower[valueStart:], paramOpen)
		nextParamAttrIdx := strings.Index(lower[valueStart:], paramOpenAttr)
		nextFuncClose := strings.Index(lower[valueStart:], funcClose)

		valueEnd := len(inner)
		if closeTagIdx >= 0 {
			valueEnd = valueStart + closeTagIdx
		} else if nextParamIdx >= 0 && nextParamAttrIdx >= 0 {
			if nextParamIdx <= nextParamAttrIdx {
				valueEnd = valueStart + nextParamIdx
			} else {
				valueEnd = valueStart + nextParamAttrIdx
			}
		} else if nextParamIdx >= 0 {
			valueEnd = valueStart + nextParamIdx
		} else if nextParamAttrIdx >= 0 {
			valueEnd = valueStart + nextParamAttrIdx
		} else if nextFuncClose >= 0 {
			valueEnd = valueStart + nextFuncClose
		}

		value := strings.TrimSpace(inner[valueStart:valueEnd])
		if value != "" {
			var parsed any
			if json.Unmarshal([]byte(value), &parsed) == nil {
				args[key] = parsed
			} else if strings.EqualFold(key, "commands") {
				args[key] = splitNonEmptyLines(value)
			} else {
				args[key] = value
			}
		}

		if closeTagIdx >= 0 {
			searchPos = valueStart + closeTagIdx + len(paramClose)
		} else if nextParamIdx >= 0 && nextParamAttrIdx >= 0 {
			if nextParamIdx <= nextParamAttrIdx {
				searchPos = valueStart + nextParamIdx
			} else {
				searchPos = valueStart + nextParamAttrIdx
			}
		} else if nextParamIdx >= 0 {
			searchPos = valueStart + nextParamIdx
		} else if nextParamAttrIdx >= 0 {
			searchPos = valueStart + nextParamAttrIdx
		} else {
			break
		}
	}
	return args
}
