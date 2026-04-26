package proxy

import (
	"encoding/base64"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

// ChatSSEvent is sent through the SSE stream for the Live chat UI.
type ChatSSEvent struct {
	ID                int                 `json:"id"`
	TraceID           string              `json:"trace_id,omitempty"`
	Timestamp         string              `json:"timestamp"`
	Model             string              `json:"model"`
	Endpoint          string              `json:"endpoint"`
	Status            int                 `json:"status"`
	DurationMs        int                 `json:"duration_ms"`
	InputTokens       int                 `json:"input_tokens"`
	OutputTokens      int                 `json:"output_tokens"`
	CachedTokens      int                 `json:"cached_tokens"`
	Messages          []ChatEventMessage  `json:"messages"`
	Timeline          []ChatTimelineEntry `json:"timeline,omitempty"`
	AssistantResponse *AssistantResponse  `json:"assistant_response,omitempty"`
}

// ChatEventMessage represents a single message in the conversation for the live UI.
type ChatEventMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AssistantResponse holds the assistant's reply from the upstream response.
type AssistantResponse struct {
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
	StopReason       string `json:"stop_reason,omitempty"`
}

type ChatTimelineEntry struct {
	Kind          string `json:"kind"`
	Role          string `json:"role,omitempty"`
	Title         string `json:"title,omitempty"`
	Content       string `json:"content,omitempty"`
	ToolName      string `json:"tool_name,omitempty"`
	CallID        string `json:"call_id,omitempty"`
	Status        string `json:"status,omitempty"`
	OutputPreview string `json:"output_preview,omitempty"`
	Truncated     bool   `json:"truncated,omitempty"`
}

// parseChatEvent extracts structured chat data from captured request/response bodies.
// Returns nil if the bodies don't represent a chat completion request.
func parseChatEvent(reqPath string, reqBody, respBody []byte, tm TokenMetrics) *ChatSSEvent {
	// Check if it's valid JSON chat request
	reqBytes := reqBody
	if len(reqBody) > 0 && reqBody[0] != '{' {
		// Might be base64 encoded
		decoded, err := base64.StdEncoding.DecodeString(string(reqBody))
		if err == nil {
			reqBytes = decoded
		}
	}

	// Check if it's valid JSON chat-ish request
	if !gjson.ValidBytes(reqBytes) {
		return nil
	}

	req := gjson.ParseBytes(reqBytes)

	// Build timestamp
	ts := time.Now().Format(time.RFC3339)

	// Parse model name
	model := req.Get("model").String()
	if model == "" {
		model = tm.Model
	}

	// Parse messages from request across OpenAI/Anthropic/Responses variants.
	messages := parseRequestMessages(req)

	// Build event
	evt := &ChatSSEvent{
		Endpoint:     reqPath,
		ID:           tm.ID,
		TraceID:      tm.TraceID,
		Timestamp:    ts,
		Model:        model,
		Status:       tm.StatusCode,
		DurationMs:   tm.DurationMs,
		InputTokens:  tm.InputTokens,
		OutputTokens: tm.OutputTokens,
		CachedTokens: tm.CachedTokens,
		Messages:     messages,
	}

	// Parse response body for assistant reply and full timeline.
	if len(respBody) > 0 {
		evt.AssistantResponse = parseAssistantResponseFromBody(respBody)
		evt.Timeline = parseTimelineFromBody(respBody)
	}

	return evt
}

// parseMessages extracts ChatEventMessage slice from a gjson array of message objects.
func parseMessages(messagesJSON gjson.Result) []ChatEventMessage {
	var msgs []ChatEventMessage
	messagesJSON.ForEach(func(_, value gjson.Result) bool {
		role := value.Get("role").String()
		contentVal := value.Get("content")

		var content string
		if contentVal.IsArray() {
			// Content is an array (e.g., multi-modal with text + images)
			var parts []string
			contentVal.ForEach(func(_, part gjson.Result) bool {
				if part.Get("type").String() == "text" {
					parts = append(parts, part.Get("text").String())
				}
				return true
			})
			content = joinStrings(parts, "\n")
		} else {
			content = contentVal.String()
		}

		if role != "" && content != "" {
			msgs = append(msgs, ChatEventMessage{Role: role, Content: content})
		}
		return true
	})
	return msgs
}

func parseRequestMessages(req gjson.Result) []ChatEventMessage {
	if req.Get("messages").Exists() {
		return parseMessages(req.Get("messages"))
	}

	if input := req.Get("input"); input.Exists() {
		if input.Type == gjson.String {
			return []ChatEventMessage{{Role: "user", Content: input.String()}}
		}
		if input.IsArray() {
			var out []ChatEventMessage
			input.ForEach(func(_, item gjson.Result) bool {
				role := item.Get("role").String()
				if role == "" {
					role = "user"
				}
				content := extractContent(item.Get("content"))
				if content == "" {
					// Responses-style input items can be {type:"input_text", text:"..."}
					content = item.Get("text").String()
				}
				if content != "" {
					out = append(out, ChatEventMessage{Role: role, Content: content})
				}
				return true
			})
			if len(out) > 0 {
				return out
			}
		}
	}

	if prompt := req.Get("prompt"); prompt.Exists() {
		return []ChatEventMessage{{Role: "user", Content: prompt.String()}}
	}

	return nil
}

func extractContent(contentVal gjson.Result) string {
	if !contentVal.Exists() {
		return ""
	}
	if contentVal.IsArray() {
		var parts []string
		contentVal.ForEach(func(_, part gjson.Result) bool {
			if part.Get("type").String() == "text" {
				parts = append(parts, part.Get("text").String())
				return true
			}
			if txt := part.Get("text").String(); txt != "" {
				parts = append(parts, txt)
			}
			return true
		})
		return joinStrings(parts, "\n")
	}
	return contentVal.String()
}

// parseAssistantResponse extracts the assistant's reply from an OpenAI-style response.
func parseAssistantResponse(resp gjson.Result) *AssistantResponse {
	ar := &AssistantResponse{}

	// Try OpenAI /v1/chat/completions format first
	choices := resp.Get("choices")
	if choices.Exists() && len(choices.Array()) > 0 {
		msg := choices.Get("0.message")
		if msg.Exists() {
			ar.Content = msg.Get("content").String()
			ar.ReasoningContent = msg.Get("reasoning_content").String()
			if ar.ReasoningContent == "" {
				ar.ReasoningContent = msg.Get("reasoning").String()
			}
			ar.StopReason = resp.Get("choices.0.finish_reason").String()
		}
	}

	// Try Anthropic /v1/messages format if no OpenAI choices found
	if ar.Content == "" && ar.ReasoningContent == "" {
		contentBlocks := resp.Get("content")
		if contentBlocks.IsArray() {
			var textParts []string
			var reasoningParts []string
			contentBlocks.ForEach(func(_, block gjson.Result) bool {
				blockType := block.Get("type").String()
				switch blockType {
				case "text":
					textParts = append(textParts, block.Get("text").String())
				case "thinking":
					reasoningParts = append(reasoningParts, block.Get("thinking").String())
				}
				return true
			})
			ar.Content = joinStrings(textParts, "\n")
			ar.ReasoningContent = joinStrings(reasoningParts, "\n")
		}
		ar.StopReason = resp.Get("stop_reason").String()
	}

	// Try OpenAI Responses API shape: output[].content[].text
	if ar.Content == "" && ar.ReasoningContent == "" {
		var textParts []string
		var reasoningParts []string
		resp.Get("output").ForEach(func(_, item gjson.Result) bool {
			item.Get("content").ForEach(func(_, part gjson.Result) bool {
				partType := part.Get("type").String()
				switch partType {
				case "output_text", "text":
					if text := part.Get("text").String(); text != "" {
						textParts = append(textParts, text)
					}
				case "reasoning":
					if text := part.Get("text").String(); text != "" {
						reasoningParts = append(reasoningParts, text)
					}
				}
				return true
			})
			return true
		})
		ar.Content = joinStrings(textParts, "\n")
		ar.ReasoningContent = joinStrings(reasoningParts, "\n")
		if ar.StopReason == "" {
			ar.StopReason = resp.Get("status").String()
		}
	}

	// Only return if we found something
	if ar.Content == "" && ar.ReasoningContent == "" {
		return nil
	}
	return ar
}

func parseAssistantResponseFromBody(respBody []byte) *AssistantResponse {
	if gjson.ValidBytes(respBody) {
		return parseAssistantResponse(gjson.ParseBytes(respBody))
	}

	// Parse SSE payload (responses bridge and chat streaming).
	var contentParts []string
	var reasoningParts []string
	var stopReason string

	lines := strings.Split(string(respBody), "\n")
	currentEvent := ""
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
			continue
		}
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if payload == "" || payload == "[DONE]" || !gjson.Valid(payload) {
			continue
		}
		data := gjson.Parse(payload)

		switch currentEvent {
		case "response.output_text.delta", "response.output_text.done":
			text := strings.TrimSpace(data.Get("delta").String())
			if text == "" {
				text = strings.TrimSpace(data.Get("text").String())
			}
			if text != "" {
				contentParts = append(contentParts, text)
			}
		case "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done":
			text := strings.TrimSpace(data.Get("delta").String())
			if text == "" {
				text = strings.TrimSpace(data.Get("text").String())
			}
			if text != "" {
				reasoningParts = append(reasoningParts, text)
			}
		case "response.completed":
			stopReason = data.Get("response.status").String()
		default:
			// OpenAI chat streaming fallback format
			choices := data.Get("choices")
			if choices.Exists() && len(choices.Array()) > 0 {
				delta := choices.Get("0.delta")
				if c := delta.Get("content").String(); c != "" {
					contentParts = append(contentParts, c)
				}
				if rc := delta.Get("reasoning_content").String(); rc != "" {
					reasoningParts = append(reasoningParts, rc)
				}
				if r := choices.Get("0.finish_reason").String(); r != "" {
					stopReason = r
				}
			}
		}
	}

	content := strings.TrimSpace(strings.Join(contentParts, ""))
	reasoning := strings.TrimSpace(strings.Join(reasoningParts, ""))
	if content == "" && reasoning == "" {
		return nil
	}
	return &AssistantResponse{
		Content:          content,
		ReasoningContent: reasoning,
		StopReason:       stopReason,
	}
}

func parseTimelineFromBody(respBody []byte) []ChatTimelineEntry {
	if gjson.ValidBytes(respBody) {
		return parseTimelineFromJSONResponse(gjson.ParseBytes(respBody))
	}
	return parseTimelineFromSSEResponse(respBody)
}

func parseTimelineFromJSONResponse(resp gjson.Result) []ChatTimelineEntry {
	output := resp.Get("output")
	if !output.IsArray() {
		return nil
	}
	var timeline []ChatTimelineEntry
	hasToolCall := false
	hasToolOutput := false
	hasAssistantMessage := false
	output.ForEach(func(_, item gjson.Result) bool {
		itemType := item.Get("type").String()
		switch itemType {
		case "message":
			text := extractContent(item.Get("content"))
			if text != "" {
				hasAssistantMessage = true
				timeline = append(timeline, ChatTimelineEntry{
					Kind:    "assistant_text",
					Role:    item.Get("role").String(),
					Title:   "Assistant",
					Content: text,
				})
			}
		case "function_call":
			hasToolCall = true
			timeline = append(timeline, ChatTimelineEntry{
				Kind:     "tool_call",
				Title:    "Tool Call",
				ToolName: item.Get("name").String(),
				CallID:   item.Get("call_id").String(),
				Content:  item.Get("arguments").String(),
				Status:   item.Get("status").String(),
			})
		case "shell_call", "apply_patch_call", "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "computer_call":
			hasToolCall = true
			content := item.Get("action").Raw
			if itemType == "apply_patch_call" {
				content = item.Get("operation").Raw
			}
			timeline = append(timeline, ChatTimelineEntry{
				Kind:     "tool_call",
				Title:    "Tool Call",
				ToolName: strings.TrimSuffix(itemType, "_call"),
				CallID:   item.Get("call_id").String(),
				Content:  content,
				Status:   item.Get("status").String(),
			})
		case "function_call_output", "shell_call_output", "apply_patch_call_output", "web_search_call_output", "file_search_call_output", "code_interpreter_call_output", "image_generation_call_output", "computer_call_output":
			hasToolOutput = true
			outputText := item.Get("output").String()
			preview, truncated := truncatePreview(outputText, 3)
			timeline = append(timeline, ChatTimelineEntry{
				Kind:          "tool_output",
				Title:         "Tool Output",
				CallID:        item.Get("call_id").String(),
				Content:       outputText,
				OutputPreview: preview,
				Truncated:     truncated,
			})
		}
		return true
	})
	if strings.EqualFold(resp.Get("status").String(), "completed") && hasToolCall && !hasToolOutput {
		title := "Protocol Warning"
		content := "Stream closed before tool-result continuation."
		if hasAssistantMessage {
			content = "Completed response still contains an unresolved tool call after assistant commentary."
		}
		timeline = append(timeline, ChatTimelineEntry{
			Kind:    "error",
			Title:   title,
			Content: content,
			Status:  "protocol_incomplete_tool_phase",
		})
	}
	return timeline
}

func parseTimelineFromSSEResponse(respBody []byte) []ChatTimelineEntry {
	lines := strings.Split(string(respBody), "\n")
	currentEvent := ""
	var timeline []ChatTimelineEntry
	hasToolCall := false
	hasCompletion := false
	knownToolItemIDs := map[string]struct{}{}
	knownToolCallIDs := map[string]struct{}{}
	orphanDeltaSeen := map[string]struct{}{}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
			continue
		}
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if payload == "" || payload == "[DONE]" || !gjson.Valid(payload) {
			continue
		}
		data := gjson.Parse(payload)
		switch currentEvent {
		case "response.output_item.added":
			item := data.Get("item")
			itemType := item.Get("type").String()
			if itemType == "message" {
				text := extractContent(item.Get("content"))
				if text != "" {
					timeline = append(timeline, ChatTimelineEntry{
						Kind:    "assistant_text",
						Role:    item.Get("role").String(),
						Title:   "Assistant",
						Content: text,
					})
				}
				continue
			}
			if strings.HasSuffix(itemType, "_call") || itemType == "function_call" {
				hasToolCall = true
				itemID := strings.TrimSpace(item.Get("id").String())
				callID := strings.TrimSpace(item.Get("call_id").String())
				if itemID != "" {
					knownToolItemIDs[itemID] = struct{}{}
				}
				if callID != "" {
					knownToolCallIDs[callID] = struct{}{}
				}
				toolName := item.Get("name").String()
				content := item.Get("arguments").String()
				if toolName == "" {
					toolName = strings.TrimSuffix(itemType, "_call")
					if itemType == "apply_patch_call" {
						content = item.Get("operation").Raw
					} else {
						content = item.Get("action").Raw
					}
				}
				timeline = append(timeline, ChatTimelineEntry{
					Kind:     "tool_call",
					Title:    "Tool Call",
					ToolName: toolName,
					CallID:   item.Get("call_id").String(),
					Content:  content,
					Status:   item.Get("status").String(),
				})
			}
		case "response.function_call_arguments.delta":
			itemID := strings.TrimSpace(data.Get("item_id").String())
			callID := strings.TrimSpace(data.Get("call_id").String())
			_, knownItem := knownToolItemIDs[itemID]
			_, knownCall := knownToolCallIDs[callID]
			if knownItem || knownCall {
				continue
			}
			key := itemID + "|" + callID
			if _, seen := orphanDeltaSeen[key]; seen {
				continue
			}
			orphanDeltaSeen[key] = struct{}{}
			timeline = append(timeline, ChatTimelineEntry{
				Kind:   "error",
				Title:  "Protocol Warning",
				Status: "tool_args_orphan_delta",
				Content: "Received function_call_arguments.delta without a prior tool_call start " +
					"(missing matching item_id/call_id).",
				CallID: callID,
			})
		case "response.output_text.delta":
			text := data.Get("delta").String()
			if strings.TrimSpace(text) != "" {
				timeline = append(timeline, ChatTimelineEntry{
					Kind:    "assistant_delta",
					Title:   "Assistant Stream",
					Content: text,
				})
			}
		case "response.reasoning_summary_text.delta":
			text := data.Get("delta").String()
			if strings.TrimSpace(text) != "" {
				timeline = append(timeline, ChatTimelineEntry{
					Kind:    "reasoning",
					Title:   "Reasoning",
					Content: text,
				})
			}
		case "response.function_call_arguments.done":
			name := data.Get("name").String()
			args := data.Get("arguments").String()
			itemID := strings.TrimSpace(data.Get("item_id").String())
			callID := data.Get("call_id").String()
			_, knownItem := knownToolItemIDs[itemID]
			_, knownCall := knownToolCallIDs[strings.TrimSpace(callID)]
			if !knownItem && !knownCall {
				timeline = append(timeline, ChatTimelineEntry{
					Kind:   "error",
					Title:  "Protocol Warning",
					Status: "tool_args_orphan_done",
					Content: "Received function_call_arguments.done without a prior tool_call start " +
						"(missing matching item_id/call_id).",
					ToolName: name,
					CallID:   callID,
				})
			}
			if strings.TrimSpace(args) != "" {
				timeline = append(timeline, ChatTimelineEntry{
					Kind:     "tool_args",
					Title:    "Tool Arguments",
					ToolName: name,
					CallID:   callID,
					Content:  args,
				})
			}
		case "response.completed":
			hasCompletion = true
			status := data.Get("response.status").String()
			if status != "" {
				timeline = append(timeline, ChatTimelineEntry{
					Kind:   "completion",
					Title:  "Completed",
					Status: status,
				})
			}
		default:
			// best effort: chat completions SSE lines
			if choices := data.Get("choices"); choices.Exists() && len(choices.Array()) > 0 {
				delta := choices.Get("0.delta")
				if rc := delta.Get("reasoning_content").String(); rc != "" {
					timeline = append(timeline, ChatTimelineEntry{
						Kind:    "reasoning",
						Title:   "Reasoning",
						Content: rc,
					})
				}
			}
		}
	}
	if hasCompletion && hasToolCall {
		hasToolOutput := false
		for _, entry := range timeline {
			if entry.Kind == "tool_output" {
				hasToolOutput = true
				break
			}
		}
		if !hasToolOutput {
			timeline = append(timeline, ChatTimelineEntry{
				Kind:    "error",
				Title:   "Protocol Warning",
				Content: "Stream closed before tool-result continuation.",
				Status:  "protocol_incomplete_tool_phase",
			})
		}
	}
	return compactTimeline(timeline)
}

func compactTimeline(items []ChatTimelineEntry) []ChatTimelineEntry {
	if len(items) == 0 {
		return items
	}
	out := make([]ChatTimelineEntry, 0, len(items))
	for _, entry := range items {
		if entry.Kind == "assistant_delta" && len(out) > 0 && out[len(out)-1].Kind == "assistant_delta" {
			out[len(out)-1].Content += entry.Content
			continue
		}
		if entry.Kind == "reasoning" && len(out) > 0 && out[len(out)-1].Kind == "reasoning" {
			out[len(out)-1].Content += entry.Content
			continue
		}
		out = append(out, entry)
	}
	return out
}

func truncatePreview(text string, maxLines int) (string, bool) {
	if maxLines <= 0 {
		return text, false
	}
	lines := strings.Split(text, "\n")
	if len(lines) <= maxLines {
		return text, false
	}
	return strings.Join(lines[:maxLines], "\n"), true
}

func joinStrings(parts []string, sep string) string {
	result := ""
	for i, p := range parts {
		if i > 0 {
			result += sep
		}
		result += p
	}
	return result
}
