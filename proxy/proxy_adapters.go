package proxy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var proxyCompatibilityAdapters = newDefaultProxyAdapterSet()

type defaultResponsesChatAdapter struct{}
type defaultReasoningTranslationAdapter struct{}
type defaultToolRepairAdapter struct{}
type defaultStreamReconstructionAdapter struct{}
type defaultContinuationController struct{}

func newDefaultProxyAdapterSet() ProxyAdapterSet {
	return ProxyAdapterSet{
		Responses:    defaultResponsesChatAdapter{},
		Reasoning:    defaultReasoningTranslationAdapter{},
		ToolRepair:   defaultToolRepairAdapter{},
		Stream:       defaultStreamReconstructionAdapter{},
		Continuation: defaultContinuationController{},
	}
}

func (defaultResponsesChatAdapter) TranslateResponsesToChatCompletionsRequest(body []byte) ([]byte, error) {
	return translateResponsesToChatCompletionsRequest(body)
}

func (defaultResponsesChatAdapter) TranslateChatCompletionToResponsesResponse(body []byte, applyPatchPathHint string, applyPatchContentHint string, applyPatchTypeHint string) ([]byte, error) {
	return translateChatCompletionToResponsesResponse(body, applyPatchPathHint, applyPatchContentHint, applyPatchTypeHint)
}

func (defaultReasoningTranslationAdapter) BuildCommentaryPreview(reasoning string) string {
	return buildReasoningCommentaryPreview(reasoning)
}

func (defaultReasoningTranslationAdapter) NormalizeRequestedSummary(summary string) string {
	return normalizeResponsesReasoningSummary(summary)
}

func (defaultToolRepairAdapter) ParseAssistantOutput(modelName, content string) ([]ParsedToolCall, string) {
	return parseModelSpecificToolCalls(modelName, content)
}

func (defaultToolRepairAdapter) RecoverRequestUserInputArguments(reasoningText, text string) (string, bool) {
	return recoverRequestUserInputArgumentsFromTextSources(reasoningText, text)
}

func (defaultToolRepairAdapter) ValidateToolCallItem(item map[string]any) ToolValidationResult {
	normalized := cloneMap(item)
	valid := func(notes ...string) ToolValidationResult {
		return ToolValidationResult{
			Valid:          true,
			LifecycleState: ToolLifecycleValidated,
			NormalizedItem: normalized,
			RepairNotes:    notes,
		}
	}
	rejected := func(warning string) ToolValidationResult {
		return ToolValidationResult{
			Valid:          false,
			Warning:        warning,
			LifecycleState: ToolLifecycleRejected,
			NormalizedItem: normalized,
		}
	}
	itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
	switch itemType {
	case "apply_patch_call":
		operation := normalizeApplyPatchOperation(item["operation"])
		normalized["operation"] = operation
		if input := strings.TrimSpace(buildApplyPatchInputFromOperation(operation)); input != "" {
			normalized["input"] = input
		}
		hasOperation := hasNonEmptyApplyPatchOperation(operation)
		hasInput := hasNonEmptyApplyPatchOperation(item["input"])
		hasPatch := hasNonEmptyApplyPatchOperation(item["patch"])
		hasDiff := hasNonEmptyApplyPatchOperation(item["diff"])
		if !(hasOperation || hasInput || hasPatch || hasDiff) {
			diagnostic := strings.TrimSpace(fmt.Sprintf("%v", item["_bridge_diagnostic"]))
			warning := applyPatchValidationWarningPrefix + " arguments were empty. Provide a non-empty operation with target path and diff/content."
			if diagnostic != "" {
				warning += " Observed arguments: " + diagnostic
			}
			return rejected(warning)
		}
		if hasOperation && !applyPatchOperationPayloadValid(operation) {
			return rejected(applyPatchValidationWarningPrefix + " operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
		}
		if hasInput && !applyPatchOperationPayloadValid(item["input"]) {
			return rejected(applyPatchValidationWarningPrefix + " input operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
		}
		if hasPatch && !applyPatchOperationPayloadValid(item["patch"]) {
			return rejected(applyPatchValidationWarningPrefix + " patch operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
		}
		if hasDiff && !applyPatchOperationPayloadValid(item["diff"]) {
			return rejected(applyPatchValidationWarningPrefix + " diff operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
		}
		return valid("apply_patch_operation_normalized")
	case "custom_tool_call":
		name := strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
		if name == "" {
			return rejected("custom tool call was not executed because tool name was empty.")
		}
		if strings.EqualFold(name, "apply_patch") {
			operation := normalizeApplyPatchOperation(item["operation"])
			normalized["operation"] = operation
			if input := strings.TrimSpace(buildApplyPatchInputFromOperation(operation)); input != "" {
				normalized["input"] = input
			}
			if !hasNonEmptyApplyPatchOperation(operation) && strings.TrimSpace(cleanFallbackInput(item["input"], "")) == "" {
				return rejected(applyPatchValidationWarningPrefix + " arguments were empty. Provide a non-empty operation with target path and diff/content.")
			}
			if hasNonEmptyApplyPatchOperation(operation) && !applyPatchOperationPayloadValid(operation) {
				return rejected(applyPatchValidationWarningPrefix + " operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
			}
			return valid("apply_patch_operation_normalized")
		}
		if strings.TrimSpace(cleanFallbackInput(item["input"], "")) == "" && !strings.EqualFold(name, "apply_patch") {
			return rejected("custom tool call was not executed because input was empty.")
		}
		return valid()
	case "mcp_tool_call":
		normalized["server"] = strings.TrimSpace(fmt.Sprintf("%v", item["server"]))
		normalized["tool"] = strings.TrimSpace(fmt.Sprintf("%v", item["tool"]))
		if fmt.Sprintf("%v", normalized["server"]) == "" || fmt.Sprintf("%v", normalized["tool"]) == "" {
			return rejected("mcp tool call was not executed because server or tool name was empty.")
		}
		return valid()
	case "shell_call", "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "computer_call":
		action, _ := item["action"].(map[string]any)
		if itemType == "shell_call" {
			action = normalizeShellArgumentMap(action)
			normalized["action"] = action
			if !shellToolArgumentsValid(action) {
				return rejected(shellValidationWarningPrefix + " arguments were empty. Provide a non-empty `command` string or `commands` array and retry.")
			}
			return valid("shell_arguments_normalized")
		}
		if len(action) == 0 || !hasAnyNonEmptyValue(action) {
			toolName := strings.TrimSuffix(itemType, "_call")
			return rejected(fmt.Sprintf("%s call was not executed because arguments were empty. Provide concrete action arguments and retry.", toolName))
		}
		return valid()
	case "function_call":
		name := strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
		if name == "" {
			return rejected("function call was not executed because function name was empty.")
		}
		args := parseToolArgsMapString(fmt.Sprintf("%v", item["arguments"]))
		switch {
		case strings.EqualFold(name, "apply_patch"):
			operation := normalizeApplyPatchOperation(selectApplyPatchOperation(args))
			normalizedArgs := buildApplyPatchStreamArgumentPayload(operation, "")
			if len(normalizedArgs) == 0 {
				if input := strings.TrimSpace(cleanFallbackInput(args["input"], "")); input != "" {
					normalizedArgs["input"] = input
				}
				if patch := strings.TrimSpace(cleanFallbackInput(args["patch"], "")); patch != "" {
					normalizedArgs["patch"] = patch
				}
				if diff := strings.TrimSpace(cleanFallbackInput(args["diff"], "")); diff != "" {
					normalizedArgs["diff"] = diff
				}
			}
			normalized["arguments"] = mustJSONString(normalizedArgs)
			if !hasNonEmptyApplyPatchOperation(operation) &&
				strings.TrimSpace(cleanFallbackInput(normalizedArgs["input"], "")) == "" &&
				strings.TrimSpace(cleanFallbackInput(normalizedArgs["patch"], "")) == "" &&
				strings.TrimSpace(cleanFallbackInput(normalizedArgs["diff"], "")) == "" {
				return rejected(applyPatchValidationWarningPrefix + " arguments were empty. Provide a non-empty operation with target path and diff/content.")
			}
			if hasNonEmptyApplyPatchOperation(operation) && !applyPatchOperationPayloadValid(operation) {
				return rejected(applyPatchValidationWarningPrefix + " operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update.")
			}
			return valid("apply_patch_function_arguments_normalized")
		case strings.EqualFold(name, "shell"), strings.EqualFold(name, "shell_command"):
			args = normalizeShellArgumentMapForResponse(args)
			normalized["name"] = "shell"
			normalized["arguments"] = mustJSONString(args)
			if !shellToolArgumentsValid(args) {
				return rejected(shellValidationWarningPrefix + " arguments were empty. Provide a non-empty `command` string or `commands` array and retry.")
			}
			return valid("shell_arguments_normalized")
		case strings.EqualFold(name, "request_user_input"):
			args["questions"] = normalizeQuestionList(args["questions"])
			normalized["arguments"] = mustJSONString(args)
			if !hasNonEmptyQuestionList(args["questions"]) {
				return rejected("request_user_input was not executed because `questions` was empty. Provide at least one question and retry.")
			}
			return valid("request_user_input_questions_normalized")
		case strings.EqualFold(name, "update_plan"):
			args["plan"] = normalizePlanList(args["plan"])
			normalized["arguments"] = mustJSONString(args)
			if !hasNonEmptyPlanList(args["plan"]) {
				return rejected("update_plan was not executed because `plan` was empty. Provide at least one plan step and retry.")
			}
			return valid("update_plan_steps_normalized")
		case strings.EqualFold(name, "multi_tool_use.parallel"):
			args["tool_uses"] = normalizeParallelToolUses(args["tool_uses"])
			normalized["arguments"] = mustJSONString(args)
			if !hasNonEmptyParallelToolUses(args["tool_uses"]) {
				return rejected("multi_tool_use.parallel was not executed because `tool_uses` was empty. Provide at least one tool use and retry.")
			}
			return valid("parallel_tool_uses_normalized")
		}
	}
	return valid()
}

func normalizeQuestionList(raw any) []any {
	items, _ := raw.([]any)
	out := make([]any, 0, len(items))
	for _, item := range items {
		switch typed := item.(type) {
		case string:
			if text := strings.TrimSpace(typed); text != "" {
				out = append(out, text)
			}
		case map[string]any:
			normalized := cloneMap(typed)
			for _, key := range []string{"header", "question", "label", "id"} {
				if text := strings.TrimSpace(fmt.Sprintf("%v", normalized[key])); text != "" {
					normalized[key] = text
				}
			}
			if hasNonEmptyQuestionList([]any{normalized}) {
				out = append(out, normalized)
			}
		}
	}
	return out
}

func hasNonEmptyQuestionList(raw any) bool {
	items, ok := raw.([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, item := range items {
		switch typed := item.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				return true
			}
		case map[string]any:
			for _, key := range []string{"question", "header", "label", "id"} {
				if strings.TrimSpace(fmt.Sprintf("%v", typed[key])) != "" {
					return true
				}
			}
		}
	}
	return false
}

func normalizePlanList(raw any) []any {
	items, _ := raw.([]any)
	out := make([]any, 0, len(items))
	for _, item := range items {
		switch typed := item.(type) {
		case string:
			if text := strings.TrimSpace(typed); text != "" {
				out = append(out, text)
			}
		case map[string]any:
			normalized := cloneMap(typed)
			if step := strings.TrimSpace(fmt.Sprintf("%v", normalized["step"])); step != "" {
				normalized["step"] = step
			}
			status := strings.TrimSpace(strings.ToLower(fmt.Sprintf("%v", normalized["status"])))
			switch status {
			case "pending", "in_progress", "completed":
				normalized["status"] = status
			default:
				if strings.TrimSpace(fmt.Sprintf("%v", normalized["step"])) != "" {
					normalized["status"] = "pending"
				}
			}
			if hasNonEmptyPlanList([]any{normalized}) {
				out = append(out, normalized)
			}
		}
	}
	return out
}

func hasNonEmptyPlanList(raw any) bool {
	items, ok := raw.([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, item := range items {
		switch typed := item.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				return true
			}
		case map[string]any:
			if strings.TrimSpace(fmt.Sprintf("%v", typed["step"])) != "" {
				return true
			}
		}
	}
	return false
}

func normalizeParallelToolUses(raw any) []any {
	items, _ := raw.([]any)
	out := make([]any, 0, len(items))
	for _, item := range items {
		typed, ok := item.(map[string]any)
		if !ok {
			continue
		}
		normalized := cloneMap(typed)
		recipientRaw, hasRecipient := normalized["recipient_name"]
		if !hasRecipient || recipientRaw == nil {
			continue
		}
		recipient := strings.TrimSpace(fmt.Sprintf("%v", recipientRaw))
		if strings.EqualFold(recipient, "<nil>") {
			recipient = ""
		}
		if recipient == "" {
			continue
		}
		normalized["recipient_name"] = recipient
		if params, ok := normalized["parameters"].(map[string]any); ok {
			normalized["parameters"] = cloneMap(params)
		}
		if _, ok := normalized["parameters"].(map[string]any); !ok {
			normalized["parameters"] = map[string]any{}
		}
		out = append(out, normalized)
	}
	return out
}

func hasNonEmptyParallelToolUses(raw any) bool {
	items, ok := raw.([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for _, item := range items {
		typed, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(fmt.Sprintf("%v", typed["recipient_name"])) != "" {
			return true
		}
	}
	return false
}

func (defaultStreamReconstructionAdapter) CanonicalToolName(name string) string {
	name = strings.TrimSpace(name)
	if strings.EqualFold(name, llamaSwapApplyPatchFunctionName) {
		return "apply_patch"
	}
	name = strings.TrimPrefix(name, "__llamaswap_")
	if strings.EqualFold(name, "applypatch") {
		return "apply_patch"
	}
	return name
}

func (a defaultStreamReconstructionAdapter) ShouldExposeToolCall(state *StreamToolCallState) bool {
	if state == nil {
		return false
	}
	toolName := a.CanonicalToolName(state.Name)
	if toolName == "" {
		return false
	}
	if strings.EqualFold(toolName, "shell") {
		args := parseToolArgsMapString(normalizePossiblyMixedToolArguments(state.ArgsBuilder.String()))
		return shellToolArgumentsValid(args)
	}
	return true
}

func (a defaultStreamReconstructionAdapter) CanonicalToolArguments(state *StreamToolCallState, arguments string) string {
	toolName := a.CanonicalToolName(state.Name)
	normalized := normalizePossiblyMixedToolArguments(arguments)
	if strings.EqualFold(toolName, "apply_patch") {
		args := parseToolArgsMapString(normalized)
		op := preferContentDrivenApplyPatchOperation(selectApplyPatchOperation(args))
		if input := strings.TrimSpace(buildApplyPatchInputFromOperation(op)); input != "" {
			appendProxyStageTrace("stream.tool_arguments.canonicalized", map[string]any{
				"tool_name":        toolName,
				"raw_arguments":    truncateBridgeDebugText(arguments, 400),
				"normalized_args":  truncateBridgeDebugText(normalized, 400),
				"parsed_args":      truncateBridgeDebugText(mustJSONString(args), 400),
				"normalized_op":    truncateBridgeDebugText(mustJSONString(op), 400),
				"normalized_input": truncateBridgeDebugText(input, 400),
			})
			return mustJSONString(buildApplyPatchStreamArgumentPayload(op, input))
		}
		if looksLikePatchText(normalized) {
			return mustJSONString(buildApplyPatchStreamArgumentPayload(op, normalizeApplyPatchText(normalized)))
		}
		return mustJSONString(buildApplyPatchStreamArgumentPayload(op, strings.TrimSpace(cleanFallbackInput(args["input"], ""))))
	}
	if strings.EqualFold(toolName, "shell") {
		return mustJSONString(normalizeShellArgumentMapForResponse(parseToolArgsMapString(normalized)))
	}
	return normalized
}

func buildApplyPatchStreamArgumentPayload(operation any, input string) map[string]any {
	payload := map[string]any{}
	if op := normalizeApplyPatchOperation(operation); hasNonEmptyApplyPatchOperation(op) {
		payload["operation"] = op
		if strings.TrimSpace(input) == "" {
			input = strings.TrimSpace(buildApplyPatchInputFromOperation(op))
		}
	}
	if strings.TrimSpace(input) != "" {
		payload["input"] = input
	}
	return payload
}

func canonicalApplyPatchResponsePayload(item map[string]any) map[string]any {
	var args map[string]any
	if item != nil {
		args = parseToolArgsMapString(fmt.Sprintf("%v", item["arguments"]))
	}
	var operation any
	if item != nil {
		operation = item["operation"]
	}
	if !hasNonEmptyApplyPatchOperation(operation) {
		operation = selectApplyPatchOperation(args)
	}
	input := ""
	if item != nil {
		input = strings.TrimSpace(cleanFallbackInput(item["input"], ""))
	}
	if input == "" {
		input = strings.TrimSpace(cleanFallbackInput(args["input"], ""))
	}
	payload := buildApplyPatchStreamArgumentPayload(operation, input)
	if len(payload) == 0 {
		if input != "" {
			payload["input"] = input
		}
		if patch := strings.TrimSpace(cleanFallbackInput(args["patch"], "")); patch != "" {
			payload["patch"] = patch
		}
		if diff := strings.TrimSpace(cleanFallbackInput(args["diff"], "")); diff != "" {
			payload["diff"] = diff
		}
	}
	return payload
}

func (a defaultStreamReconstructionAdapter) BuildToolOutputItem(state *StreamToolCallState, arguments string) map[string]any {
	toolName := a.CanonicalToolName(state.Name)
	if strings.EqualFold(toolName, "apply_patch") {
		args := parseToolArgsMapString(arguments)
		op := preferContentDrivenApplyPatchOperation(selectApplyPatchOperation(args))
		input := strings.TrimSpace(cleanFallbackInput(args["input"], ""))
		payload := buildApplyPatchStreamArgumentPayload(op, input)
		appendProxyStageTrace("stream.tool_output_item.built", map[string]any{
			"tool_name":        toolName,
			"raw_arguments":    truncateBridgeDebugText(arguments, 400),
			"parsed_args":      truncateBridgeDebugText(mustJSONString(args), 400),
			"normalized_op":    truncateBridgeDebugText(mustJSONString(op), 400),
			"normalized_input": truncateBridgeDebugText(fmt.Sprintf("%v", payload["input"]), 400),
		})
		item := map[string]any{
			"id":        state.ItemID,
			"type":      "function_call",
			"call_id":   state.CallID,
			"name":      "apply_patch",
			"status":    "in_progress",
			"arguments": mustJSONString(payload),
		}
		if operation, ok := payload["operation"]; ok {
			item["operation"] = operation
		}
		if patchInput, ok := payload["input"]; ok {
			item["input"] = patchInput
		}
		return item
	}
	if server, subtool, ok := parseMCPToolName(toolName); ok {
		return map[string]any{
			"id":        state.ItemID,
			"type":      "function_call",
			"call_id":   state.CallID,
			"name":      buildMCPToolName(server, subtool),
			"status":    "in_progress",
			"arguments": mustJSONString(parseToolArgsMapString(arguments)),
		}
	}
	return map[string]any{
		"id":        state.ItemID,
		"type":      "function_call",
		"call_id":   state.CallID,
		"name":      toolName,
		"status":    "in_progress",
		"arguments": arguments,
	}
}

func (a defaultStreamReconstructionAdapter) BuildToolItemAddedEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool) {
	if state == nil || !state.Exposed {
		return StreamEvent{}, false
	}
	toolName := a.CanonicalToolName(state.Name)
	arguments := "{}"
	if strings.EqualFold(toolName, "shell") {
		arguments = mustJSONString(normalizeShellArgumentMapForResponse(parseToolArgsMapString(normalizePossiblyMixedToolArguments(state.ArgsBuilder.String()))))
	}
	item := map[string]any{
		"id":        state.ItemID,
		"type":      "function_call",
		"call_id":   state.CallID,
		"name":      toolName,
		"status":    "in_progress",
		"arguments": arguments,
	}
	if server, subtool, ok := parseMCPToolName(toolName); ok {
		item["name"] = buildMCPToolName(server, subtool)
	}
	return StreamEvent{
		Kind:        StreamEventToolItemAdded,
		ResponseID:  responseID,
		ItemID:      state.ItemID,
		OutputIndex: state.OutputIndex,
		ToolName:    toolName,
		CallID:      state.CallID,
		Status:      "in_progress",
		Payload: map[string]any{
			"item": item,
		},
	}, true
}

func (a defaultStreamReconstructionAdapter) BuildToolArgsDeltaEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool) {
	if state == nil || !state.Exposed {
		return StreamEvent{}, false
	}
	toolName := a.CanonicalToolName(state.Name)
	if strings.HasPrefix(toolName, "mcp__") {
		return StreamEvent{}, false
	}
	if strings.EqualFold(toolName, "request_user_input") {
		args := parseToolArgsMapString(a.CanonicalToolArguments(state, state.ArgsBuilder.String()))
		if !hasNonEmptyQuestionList(args["questions"]) {
			return StreamEvent{}, false
		}
	}
	return StreamEvent{
		Kind:       StreamEventToolArgsDelta,
		ResponseID: responseID,
		ItemID:     state.ItemID,
		ToolName:   toolName,
		CallID:     state.CallID,
		Payload: map[string]any{
			"delta": a.CanonicalToolArguments(state, state.ArgsBuilder.String()),
		},
	}, true
}

func (a defaultStreamReconstructionAdapter) BuildToolArgsDoneEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool) {
	if state == nil || !state.Exposed {
		return StreamEvent{}, false
	}
	if strings.HasPrefix(a.CanonicalToolName(state.Name), "mcp__") {
		return StreamEvent{}, false
	}
	return StreamEvent{
		Kind:       StreamEventToolArgsDone,
		ResponseID: responseID,
		ItemID:     state.ItemID,
		ToolName:   state.Name,
		CallID:     state.CallID,
		Payload: map[string]any{
			"arguments": a.CanonicalToolArguments(state, state.ArgsBuilder.String()),
		},
	}, true
}

func (a defaultStreamReconstructionAdapter) BuildToolItemDoneEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool) {
	if state == nil || !state.Exposed {
		return StreamEvent{}, false
	}
	arguments := a.CanonicalToolArguments(state, state.ArgsBuilder.String())
	return StreamEvent{
		Kind:        StreamEventToolOutputReturned,
		ResponseID:  responseID,
		ItemID:      state.ItemID,
		OutputIndex: state.OutputIndex,
		ToolName:    a.CanonicalToolName(state.Name),
		CallID:      state.CallID,
		Status:      "in_progress",
		Payload: map[string]any{
			"item": a.BuildToolOutputItem(state, arguments),
		},
	}, true
}

func (a defaultStreamReconstructionAdapter) NormalizeResponseOutputItem(item map[string]any) map[string]any {
	if item == nil {
		return item
	}
	itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
	if strings.EqualFold(itemType, "shell_call") {
		if action, ok := item["action"].(map[string]any); ok {
			item["action"] = normalizeShellArgumentMapForResponse(action)
		}
	}
	if strings.EqualFold(itemType, "function_call") &&
		strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["name"])), "apply_patch") {
		payload := canonicalApplyPatchResponsePayload(item)
		item["arguments"] = mustJSONString(payload)
		if operation, ok := payload["operation"]; ok {
			item["operation"] = operation
		}
		if input, ok := payload["input"]; ok {
			item["input"] = input
		}
	}
	if strings.EqualFold(itemType, "apply_patch_call") {
		payload := canonicalApplyPatchResponsePayload(item)
		item["type"] = "function_call"
		item["name"] = "apply_patch"
		item["arguments"] = mustJSONString(payload)
		if operation, ok := payload["operation"]; ok {
			item["operation"] = operation
		}
		if input, ok := payload["input"]; ok {
			item["input"] = input
		}
	}
	if strings.EqualFold(itemType, "custom_tool_call") &&
		strings.EqualFold(strings.TrimSpace(fmt.Sprintf("%v", item["name"])), "apply_patch") {
		payload := canonicalApplyPatchResponsePayload(item)
		item["type"] = "function_call"
		item["name"] = "apply_patch"
		item["arguments"] = mustJSONString(payload)
		if operation, ok := payload["operation"]; ok {
			item["operation"] = operation
		}
		if input, ok := payload["input"]; ok {
			item["input"] = input
		}
	}
	return item
}

func (a defaultStreamReconstructionAdapter) BuildResponseToolItemView(item map[string]any) (ResponseToolItemView, bool) {
	if item == nil {
		return ResponseToolItemView{}, false
	}
	itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
	view := ResponseToolItemView{
		Item:                item,
		ItemType:            itemType,
		ItemID:              strings.TrimSpace(fmt.Sprintf("%v", item["id"])),
		CallID:              strings.TrimSpace(fmt.Sprintf("%v", item["call_id"])),
		EmitsArgumentEvents: true,
	}
	switch itemType {
	case "function_call", "shell_call", "apply_patch_call", "mcp_tool_call", "custom_tool_call", "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "computer_call":
	default:
		return ResponseToolItemView{}, false
	}

	view.ToolName = strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
	view.Arguments = fmt.Sprintf("%v", item["arguments"])
	if itemType != "function_call" {
		view.ToolName = strings.TrimSuffix(itemType, "_call")
		switch itemType {
		case "apply_patch_call":
			view.ToolName = "apply_patch"
			view.Arguments = strings.TrimSpace(cleanFallbackInput(item["input"], ""))
			if view.Arguments == "" {
				op := normalizeApplyPatchOperation(item["operation"])
				view.Arguments = strings.TrimSpace(buildApplyPatchInputFromOperation(op))
			}
		case "mcp_tool_call":
			view.ToolName = strings.TrimSpace(fmt.Sprintf("%v", item["tool"]))
			view.Arguments = mustJSONString(normalizeMapValue(item["arguments"]))
		case "custom_tool_call":
			view.ToolName = strings.TrimSpace(fmt.Sprintf("%v", item["name"]))
			view.Arguments = strings.TrimSpace(cleanFallbackInput(item["input"], ""))
			if view.Arguments == "" {
				op := normalizeApplyPatchOperation(item["operation"])
				view.Arguments = strings.TrimSpace(buildApplyPatchInputFromOperation(op))
			}
			view.EmitsArgumentEvents = false
		case "shell_call":
			view.ToolName = "shell"
			if action, ok := item["action"].(map[string]any); ok {
				view.Arguments = mustJSONString(normalizeShellArgumentMapForResponse(action))
			} else {
				view.Arguments = mustJSONString(map[string]any{})
			}
		default:
			view.Arguments = mustJSONString(item["action"])
		}
	}
	if itemType == "mcp_tool_call" {
		view.EmitsArgumentEvents = false
	}
	return view, true
}

func (a defaultStreamReconstructionAdapter) BuildResponseToolArgumentEvents(view ResponseToolItemView, responseID string, outputIndex int) []StreamEvent {
	if !view.EmitsArgumentEvents || strings.TrimSpace(view.ToolName) == "" {
		return nil
	}
	return []StreamEvent{
		{
			Kind:        StreamEventToolArgsDelta,
			ResponseID:  responseID,
			ItemID:      view.ItemID,
			OutputIndex: outputIndex,
			ToolName:    view.ToolName,
			CallID:      view.CallID,
			Payload: map[string]any{
				"delta": view.Arguments,
			},
		},
		{
			Kind:        StreamEventToolArgsDone,
			ResponseID:  responseID,
			ItemID:      view.ItemID,
			OutputIndex: outputIndex,
			ToolName:    view.ToolName,
			CallID:      view.CallID,
			Payload: map[string]any{
				"arguments": view.Arguments,
			},
		},
	}
}

func (defaultStreamReconstructionAdapter) BuildResponseCompletedEvent(response map[string]any) (StreamEvent, bool) {
	if len(response) == 0 {
		return StreamEvent{}, false
	}
	responseID := strings.TrimSpace(fmt.Sprintf("%v", response["id"]))
	return StreamEvent{
		Kind:       StreamEventResponseCompleted,
		ResponseID: responseID,
		Payload: map[string]any{
			"response": response,
		},
	}, true
}

func (defaultContinuationController) DeriveAllowedToolNames(req map[string]any) []string {
	return deriveContinuationAllowedToolNames(req)
}

func (defaultContinuationController) BuildWorkflowState(req map[string]any) ToolWorkflowState {
	return buildToolWorkflowState(req)
}

func buildLoopGuardDecision(workflowState ToolWorkflowState, ctx ContinuationContext, activeToolChoice any, exactReply string) LoopGuardDecision {
	forceFinalAfterSatisfiedApplyPatch := workflowState.ApplyPatchSatisfied
	latestCompletedToolName := workflowState.LatestCompletedToolName
	forceFinalAfterShellVerification := workflowState.VerificationExpected &&
		workflowState.VerificationCompleted &&
		workflowState.FinalAnswerSafe &&
		latestCompletedToolName == "shell"
	repeatedCompletedTool := workflowState.FinalAnswerSafe &&
		workflowState.RepeatedLatestToolFingerprint &&
		latestCompletedToolName != ""
	forceFinalAfterRepeatedCompletedTool := repeatedCompletedTool &&
		latestCompletedToolName != "apply_patch" &&
		latestCompletedToolName != "web_search" &&
		latestCompletedToolName != "request_user_input"

	if repeatedCompletedTool {
		switch latestCompletedToolName {
		case "web_search":
			return LoopGuardDecision{
				Triggered: true,
				State:     ContinuationStateToolCompletedAwaitingFollowup,
				Instructions: []string{
					"Continuation mode: the same web_search call already completed with the same query and results. Do not repeat the same web_search again. Use the existing search results to continue. If more research is still necessary, use a different web_search query; otherwise continue with the next user-facing step or final answer.",
				},
			}
		case "request_user_input":
			return LoopGuardDecision{
				Triggered: true,
				State:     ContinuationStateToolCompletedAwaitingFollowup,
				Instructions: []string{
					"Continuation mode: the same request_user_input call already completed with the same question and answer pattern. Do not ask the same question again. Use the existing answer to continue. Only ask a different clarifying question if it is genuinely necessary for the next step.",
				},
			}
		}
	}

	if forceFinalAfterSatisfiedApplyPatch || forceFinalAfterShellVerification || forceFinalAfterRepeatedCompletedTool {
		guard := LoopGuardDecision{
			Triggered:    true,
			State:        ContinuationStateFinalAnswerRequired,
			DisableTools: true,
		}
		if forceFinalAfterRepeatedCompletedTool {
			guard.Instructions = append(guard.Instructions,
				"Continuation mode: the latest tool call already completed with the same arguments and output pattern again. Do not repeat the same tool call. Do not call any more tools. Provide the final answer immediately.")
		} else if forceFinalAfterShellVerification {
			guard.Instructions = append(guard.Instructions,
				"Continuation mode: the requested shell verification already completed after apply_patch. Do not call any more tools or modify files again. Provide the final answer immediately.")
		} else {
			guard.Instructions = append(guard.Instructions,
				"Continuation mode: the previous apply_patch already produced the requested file change. Do not call any more tools or modify files again. Provide the final answer immediately.")
		}
		if strings.TrimSpace(exactReply) != "" {
			guard.Instructions = append(guard.Instructions,
				"Final answer requirement: reply with exactly "+strconv.Quote(exactReply)+".")
		}
		return guard
	}

	if latestCompletedToolName == "apply_patch" && workflowState.VerificationExpected && !workflowState.VerificationCompleted {
		return LoopGuardDecision{
			Triggered: true,
			State:     ContinuationStateToolCompletedAwaitingFollowup,
			ForceToolChoice: map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": "shell",
				},
			},
			Instructions: []string{
				"Continuation mode: the apply_patch step is complete but the user explicitly required verification with shell. Use shell now to inspect the updated file and only then provide the final answer.",
			},
		}
	}

	return LoopGuardDecision{}
}

func classifyContinuationTurnPhase(ctx ContinuationContext) ContinuationTurnPhase {
	if ctx.TurnPhase != "" {
		return ctx.TurnPhase
	}
	if ctx.ImplementationRetryIntent {
		return ContinuationTurnPhaseImplementationRetry
	}
	if ctx.SearchIntent || ctx.ExplorationFollowupIntent {
		return ContinuationTurnPhaseResearch
	}
	if ctx.NativeQuestionRequested {
		return ContinuationTurnPhaseQuestion
	}
	if ctx.PlanModeRequested {
		if ctx.PlanOutputRequested {
			return ContinuationTurnPhasePlanFinalize
		}
		return ContinuationTurnPhasePlanGather
	}
	return ContinuationTurnPhaseGeneral
}

func (c defaultContinuationController) BuildDecision(req map[string]any, ctx ContinuationContext) ContinuationDecision {
	workflowState := c.BuildWorkflowState(req)
	hasPriorToolOutput := workflowState.HasToolOutput
	hasCompletedRequestUserInputOutput := completedToolNamesContain(workflowState.CompletedToolNames, "request_user_input")
	phase := classifyContinuationTurnPhase(ctx)
	ctx.TurnPhase = phase

	decision := ContinuationDecision{
		State: ContinuationStatePreTool,
	}
	if hasCompletedRequestUserInputOutput {
		if phase == ContinuationTurnPhaseImplementationRetry ||
			phase == ContinuationTurnPhaseResearch ||
			(phase == ContinuationTurnPhasePlanGather && ctx.RequestUserInputAvailable) {
			decision.State = ContinuationStateToolRunning
		} else {
			decision.State = ContinuationStateToolCompletedAwaitingFollowup
		}
	} else if hasPriorToolOutput {
		decision.State = ContinuationStateToolRunning
	}

	if hasPriorToolOutput {
		if exactReply := extractExactFinalReplyHintFromRequest(req); exactReply != "" {
			decision.Instructions = append(decision.Instructions,
				"Continuation mode: if the task is already complete from prior tool results, do not call more tools. Provide the final answer immediately and reply with exactly: "+strconv.Quote(exactReply))
		}
	}
	exactReply := extractExactFinalReplyHintFromRequest(req)
	activeToolChoice := ctx.ActiveToolChoice
	if ctx.RequestedToolChoice != nil {
		activeToolChoice = ctx.RequestedToolChoice
	}
	loopGuard := buildLoopGuardDecision(workflowState, ctx, activeToolChoice, exactReply)
	if loopGuard.Triggered {
		decision.State = loopGuard.State
		decision.DisableTools = loopGuard.DisableTools
		if loopGuard.ForceToolChoice != nil {
			decision.ForceToolChoice = loopGuard.ForceToolChoice
		}
		decision.Instructions = append(decision.Instructions, loopGuard.Instructions...)
	}

	if decision.State == ContinuationStateFinalAnswerRequired {
		decision.DisableTools = true
		return decision
	}

	if !toolChoiceTargetsSpecificTool(decision.ForceToolChoice) {
		decision.AllowedToolNames = c.DeriveAllowedToolNames(req)
	}
	return decision
}

var proxyStageTraceLogPath = filepath.Join(os.TempDir(), "llama-swap-stage-trace.log")

func appendProxyStageTrace(stage string, fields map[string]any) {
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
	f, err := os.OpenFile(proxyStageTraceLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}

func summarizeContinuationState(state ContinuationState, workflowState ToolWorkflowState, forceFinal bool, completedRequestInput bool, hasToolOutput bool) string {
	return fmt.Sprintf("state=%s force_final=%t completed_request_user_input=%t has_tool_output=%t latest_completed=%s latest_fingerprint=%s previous_fingerprint=%s repeated_latest=%t pending=%d apply_patch_satisfied=%t verify_expected=%t verify_completed=%t final_safe=%t",
		state,
		forceFinal,
		completedRequestInput,
		hasToolOutput,
		workflowState.LatestCompletedToolName,
		workflowState.LatestCompletedToolFingerprint,
		workflowState.PreviousCompletedToolFingerprint,
		workflowState.RepeatedLatestToolFingerprint,
		len(workflowState.PendingToolNames),
		workflowState.ApplyPatchSatisfied,
		workflowState.VerificationExpected,
		workflowState.VerificationCompleted,
		workflowState.FinalAnswerSafe,
	)
}
