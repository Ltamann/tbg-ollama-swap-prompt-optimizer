package proxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

type ForensicRequestShape struct {
	Protocol            string   `json:"protocol,omitempty"`
	Model               string   `json:"model,omitempty"`
	ToolChoice          string   `json:"tool_choice,omitempty"`
	ToolNames           []string `json:"tool_names,omitempty"`
	ToolsCount          int      `json:"tools_count,omitempty"`
	ParallelToolCalls   bool     `json:"parallel_tool_calls,omitempty"`
	EnableThinking      bool     `json:"enable_thinking,omitempty"`
	HasCloseThinkBias   bool     `json:"has_close_think_bias,omitempty"`
	HasGrammar          bool     `json:"has_grammar,omitempty"`
	Stream              bool     `json:"stream,omitempty"`
	InputItems          int      `json:"input_items,omitempty"`
	MessagesCount       int      `json:"messages_count,omitempty"`
	HasToolOutputs      bool     `json:"has_tool_outputs,omitempty"`
	InstructionsPreview string   `json:"instructions_preview,omitempty"`
}

type ForensicChatResponseShape struct {
	Model                       string   `json:"model,omitempty"`
	FinishReason                string   `json:"finish_reason,omitempty"`
	ContentPreview              string   `json:"content_preview,omitempty"`
	ReasoningPreview            string   `json:"reasoning_preview,omitempty"`
	ToolCallNames               []string `json:"tool_call_names,omitempty"`
	HasToolCalls                bool     `json:"has_tool_calls,omitempty"`
	HasVisibleAnswerInReasoning bool     `json:"has_visible_answer_in_reasoning,omitempty"`
	IsStream                    bool     `json:"is_stream,omitempty"`
	ChunkCount                  int      `json:"chunk_count,omitempty"`
	HasDoneMarker               bool     `json:"has_done_marker,omitempty"`
	ErrorText                   string   `json:"error_text,omitempty"`
}

type ForensicResponsesOutputShape struct {
	Status            string   `json:"status,omitempty"`
	OutputCount       int      `json:"output_count,omitempty"`
	OutputTextPreview string   `json:"output_text_preview,omitempty"`
	MessageCount      int      `json:"message_count,omitempty"`
	ReasoningCount    int      `json:"reasoning_count,omitempty"`
	ToolCallCount     int      `json:"tool_call_count,omitempty"`
	ToolNames         []string `json:"tool_names,omitempty"`
	HasEmptyCompleted bool     `json:"has_empty_completed,omitempty"`
}

type ForensicQwenStreamShape struct {
	FrameCount         int               `json:"frame_count,omitempty"`
	RouteCounts        map[string]int    `json:"route_counts,omitempty"`
	ToolNames          []string          `json:"tool_names,omitempty"`
	ArtifactCounts     map[string]int    `json:"artifact_counts,omitempty"`
	PreferredOrigins   map[string]string `json:"preferred_origins,omitempty"`
	MessageErrorKinds  []string          `json:"message_error_kinds,omitempty"`
	MalformedReasoning bool              `json:"malformed_reasoning,omitempty"`
}

type ForensicStageSummary struct {
	Name            string                        `json:"name"`
	Request         *ForensicRequestShape         `json:"request,omitempty"`
	ChatResponse    *ForensicChatResponseShape    `json:"chat_response,omitempty"`
	ResponsesOutput *ForensicResponsesOutputShape `json:"responses_output,omitempty"`
	QwenStream      *ForensicQwenStreamShape      `json:"qwen_stream,omitempty"`
	RawTextPreview  string                        `json:"raw_text_preview,omitempty"`
}

type RequestForensicSummary struct {
	ID               int                           `json:"id"`
	TraceID          string                        `json:"trace_id,omitempty"`
	Timestamp        string                        `json:"timestamp,omitempty"`
	Model            string                        `json:"model,omitempty"`
	ReqPath          string                        `json:"req_path,omitempty"`
	StatusCode       int                           `json:"status_code"`
	DurationMs       int                           `json:"duration_ms"`
	InputTokens      int                           `json:"input_tokens"`
	OutputTokens     int                           `json:"output_tokens"`
	HasCapture       bool                          `json:"has_capture"`
	StageNames       []string                      `json:"stage_names,omitempty"`
	Stages           []ForensicStageSummary        `json:"stages,omitempty"`
	ResponsesRequest *ForensicRequestShape         `json:"responses_request,omitempty"`
	ChatRequest      *ForensicRequestShape         `json:"chat_request,omitempty"`
	QwenStream       *ForensicQwenStreamShape      `json:"qwen_stream,omitempty"`
	ChatResponse     *ForensicChatResponseShape    `json:"chat_response,omitempty"`
	ResponsesOutput  *ForensicResponsesOutputShape `json:"responses_output,omitempty"`
	FirstWrongStage  string                        `json:"first_wrong_stage,omitempty"`
	FailureClass     string                        `json:"failure_class,omitempty"`
	Outcome          string                        `json:"outcome,omitempty"`
	Notes            []string                      `json:"notes,omitempty"`
}

func buildRequestForensicSummary(metric TokenMetrics, capture *ReqRespCapture) RequestForensicSummary {
	summary := RequestForensicSummary{
		ID:           metric.ID,
		TraceID:      metric.TraceID,
		Timestamp:    metric.Timestamp.Format(timeFormatRFC3339(metric.Timestamp)),
		Model:        metric.Model,
		StatusCode:   metric.StatusCode,
		DurationMs:   metric.DurationMs,
		InputTokens:  metric.InputTokens,
		OutputTokens: metric.OutputTokens,
		HasCapture:   capture != nil,
		Outcome:      fmt.Sprintf("http_%d", metric.StatusCode),
	}
	if capture == nil {
		if metric.StatusCode >= 500 {
			summary.FirstWrongStage = "capture_missing"
			summary.FailureClass = "no_capture_for_failed_request"
		}
		return summary
	}
	summary.ReqPath = capture.ReqPath
	summary.StageNames = make([]string, 0, len(capture.Stages))
	summary.Stages = make([]ForensicStageSummary, 0, len(capture.Stages))

	for _, stage := range capture.Stages {
		stageSummary := ForensicStageSummary{Name: stage.Name}
		summary.StageNames = append(summary.StageNames, stage.Name)
		switch stage.Name {
		case "bridge.responses_request":
			stageSummary.Request = summarizeForensicRequestShape(capture.ReqBody, "responses")
			if stageSummary.Request == nil {
				stageSummary.Request = summarizeForensicRequestShape(stage.Payload, "responses")
			}
			summary.ResponsesRequest = stageSummary.Request
		case "bridge.chat_completions_request":
			stageSummary.Request = summarizeForensicRequestShape(stage.Payload, "chat_completions")
			summary.ChatRequest = stageSummary.Request
		case "bridge.chat_completions_response":
			stageSummary.ChatResponse = summarizeForensicChatResponse(stage.Payload)
			summary.ChatResponse = stageSummary.ChatResponse
			if stageSummary.ChatResponse == nil {
				stageSummary.RawTextPreview = previewText(stage.Payload, 220)
			}
		case "bridge.responses_output":
			stageSummary.ResponsesOutput = summarizeForensicResponsesOutput(stage.Payload)
			summary.ResponsesOutput = stageSummary.ResponsesOutput
		case "bridge.qwen_stream_normalization":
			stageSummary.QwenStream = summarizeForensicQwenStream(stage.Payload)
			summary.QwenStream = stageSummary.QwenStream
		default:
			stageSummary.RawTextPreview = previewText(stage.Payload, 220)
		}
		summary.Stages = append(summary.Stages, stageSummary)
	}

	classifyRequestForensicSummary(&summary)
	return summary
}

func classifyRequestForensicSummary(summary *RequestForensicSummary) {
	if summary == nil {
		return
	}
	notes := make([]string, 0, 6)
	if summary.ChatRequest != nil && summary.ChatRequest.ToolChoice == "none" && summary.ChatRequest.HasCloseThinkBias {
		notes = append(notes, "tool-less translated request still carried close-think bias")
	}
	if summary.ChatRequest != nil && summary.ChatRequest.ToolChoice == "none" && summary.ChatRequest.HasGrammar {
		notes = append(notes, "tool-less translated request still carried grammar constraints")
	}
	if summary.ChatResponse != nil && summary.ChatResponse.HasVisibleAnswerInReasoning {
		notes = append(notes, "upstream response placed visible answer content inside reasoning")
	}
	if summary.ResponsesOutput != nil && summary.ResponsesOutput.HasEmptyCompleted {
		notes = append(notes, "bridge normalized a stop turn into completed output with no visible items")
	}
	if summary.QwenStream != nil && summary.QwenStream.MalformedReasoning {
		notes = append(notes, "normalized Qwen stream reported malformed reasoning")
	}
	if summary.QwenStream != nil && summary.QwenStream.PreferredOrigins["question"] == "recovered" {
		notes = append(notes, "normalized Qwen stream recovered question semantics from non-native output")
	}
	if summary.QwenStream != nil && summary.QwenStream.PreferredOrigins["plan"] == "recovered" {
		notes = append(notes, "normalized Qwen stream recovered plan semantics from non-native output")
	}
	if summary.QwenStream != nil && len(summary.QwenStream.MessageErrorKinds) > 0 {
		notes = append(notes, "normalized Qwen stream reported semantic message errors")
	}
	if summary.QwenStream != nil && summary.QwenStream.RouteCounts["reasoning"] > 0 && summary.QwenStream.RouteCounts["final"] == 0 && summary.QwenStream.RouteCounts["commentary"] == 0 {
		notes = append(notes, "normalized Qwen stream never reached visible commentary or final output")
	}

	switch {
	case summary.QwenStream != nil && summary.QwenStream.MalformedReasoning:
		summary.FirstWrongStage = "qwen_stream_normalization"
		summary.FailureClass = "malformed_reasoning_detected"
		summary.Outcome = "normalized_malformed_reasoning"
	case summary.StatusCode >= 500 && summary.ChatResponse != nil && strings.Contains(strings.ToLower(summary.ChatResponse.ErrorText), "unable to start process"):
		summary.FirstWrongStage = "upstream_process_lifecycle"
		summary.FailureClass = "upstream_exit_during_request"
		summary.Outcome = "backend_502_process_exit"
	case summary.StatusCode >= 500 && summary.ChatRequest != nil && summary.ChatRequest.ToolChoice == "none" && summary.ChatRequest.HasCloseThinkBias:
		summary.FirstWrongStage = "bridge_chat_translation"
		summary.FailureClass = "close_think_controls_on_toolless_turn"
		summary.Outcome = "backend_502_after_toolless_translation"
	case summary.StatusCode >= 500 && summary.ChatRequest != nil:
		summary.FirstWrongStage = "upstream_process_lifecycle"
		summary.FailureClass = "upstream_502_after_translation"
		summary.Outcome = "backend_502_after_translation"
	case summary.ResponsesOutput != nil && summary.ResponsesOutput.HasEmptyCompleted:
		summary.FirstWrongStage = "bridge_response_normalization"
		summary.FailureClass = "empty_completed_visible_output"
		summary.Outcome = "completed_empty_output"
	case summary.ChatResponse != nil && summary.ChatResponse.HasVisibleAnswerInReasoning:
		summary.FirstWrongStage = "upstream_model_output"
		summary.FailureClass = "visible_answer_leaked_into_reasoning"
		summary.Outcome = "malformed_reasoning_split"
	default:
		summary.FirstWrongStage = "none_detected"
		summary.FailureClass = "no_forensic_anomaly_detected"
	}
	summary.Notes = notes
}

func summarizeForensicRequestShape(payload []byte, protocol string) *ForensicRequestShape {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return nil
	}
	parsed := gjson.ParseBytes(payload)
	shape := &ForensicRequestShape{
		Protocol:            protocol,
		Model:               parsed.Get("model").String(),
		ToolChoice:          normalizeForensicToolChoice(parsed.Get("tool_choice")),
		ParallelToolCalls:   parsed.Get("parallel_tool_calls").Bool(),
		EnableThinking:      parsed.Get("chat_template_kwargs.enable_thinking").Bool(),
		HasCloseThinkBias:   parsed.Get("logit_bias." + qwenCloseThinkTokenID).Exists(),
		HasGrammar:          parsed.Get("grammar").Exists(),
		Stream:              parsed.Get("stream").Bool(),
		InputItems:          len(parsed.Get("input").Array()),
		MessagesCount:       len(parsed.Get("messages").Array()),
		HasToolOutputs:      parsed.Get("input").String() != "" && strings.Contains(parsed.Get("input").Raw, "function_call_output"),
		InstructionsPreview: previewString(parsed.Get("instructions").String(), 220),
	}

	if tools := parsed.Get("tools"); tools.Exists() {
		for _, tool := range tools.Array() {
			name := strings.TrimSpace(tool.Get("function.name").String())
			if name == "" {
				name = strings.TrimSpace(tool.Get("name").String())
			}
			if name == "" {
				toolType := strings.TrimSpace(tool.Get("type").String())
				switch toolType {
				case "web_search", "web_search_preview", "file_search", "computer", "code_interpreter", "image_generation":
					name = toolType
				}
			}
			if name != "" {
				shape.ToolNames = append(shape.ToolNames, name)
			}
		}
		shape.ToolsCount = len(shape.ToolNames)
	}
	return shape
}

func summarizeForensicChatResponse(payload []byte) *ForensicChatResponseShape {
	if len(payload) == 0 {
		return nil
	}
	if !gjson.ValidBytes(payload) {
		text := strings.TrimSpace(string(payload))
		if text == "" {
			return nil
		}
		if strings.Contains(text, "data:") {
			if streamed := summarizeForensicStreamedChatResponse(payload); streamed != nil {
				return streamed
			}
		}
		return &ForensicChatResponseShape{ErrorText: previewString(text, 220)}
	}
	parsed := gjson.ParseBytes(payload)
	resp := &ForensicChatResponseShape{
		Model:            parsed.Get("model").String(),
		FinishReason:     parsed.Get("choices.0.finish_reason").String(),
		ContentPreview:   previewString(parsed.Get("choices.0.message.content").String(), 220),
		ReasoningPreview: previewString(parsed.Get("choices.0.message.reasoning_content").String(), 220),
	}
	toolCalls := parsed.Get("choices.0.message.tool_calls")
	if toolCalls.Exists() {
		for _, call := range toolCalls.Array() {
			name := strings.TrimSpace(call.Get("function.name").String())
			if name != "" {
				resp.ToolCallNames = append(resp.ToolCallNames, name)
			}
		}
		resp.HasToolCalls = len(resp.ToolCallNames) > 0
	}
	reasoningLower := strings.ToLower(strings.TrimSpace(resp.ReasoningPreview))
	if strings.TrimSpace(resp.ContentPreview) == "" && strings.Contains(reasoningLower, "</think>") {
		resp.HasVisibleAnswerInReasoning = true
	}
	return resp
}

func summarizeForensicStreamedChatResponse(payload []byte) *ForensicChatResponseShape {
	lines := strings.Split(string(payload), "\n")
	out := &ForensicChatResponseShape{IsStream: true}
	var contentParts []string
	var reasoningParts []string
	toolNameSet := map[string]struct{}{}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.EqualFold(trimmed, "data: [DONE]") {
			out.HasDoneMarker = true
			continue
		}
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		dataText := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if dataText == "" || !gjson.Valid(dataText) {
			continue
		}
		out.ChunkCount++
		data := gjson.Parse(dataText)
		if out.Model == "" {
			out.Model = strings.TrimSpace(data.Get("model").String())
		}
		choices := data.Get("choices")
		if !choices.Exists() || len(choices.Array()) == 0 {
			continue
		}
		if finish := strings.TrimSpace(choices.Get("0.finish_reason").String()); finish != "" {
			out.FinishReason = finish
		}
		delta := choices.Get("0.delta")
		if text := delta.Get("content").String(); text != "" {
			contentParts = append(contentParts, text)
		}
		if text := delta.Get("reasoning_content").String(); text != "" {
			reasoningParts = append(reasoningParts, text)
		}
		if text := delta.Get("reasoning").String(); text != "" {
			reasoningParts = append(reasoningParts, text)
		}
		for _, call := range delta.Get("tool_calls").Array() {
			name := strings.TrimSpace(call.Get("function.name").String())
			if name != "" {
				toolNameSet[name] = struct{}{}
			}
		}
	}
	for name := range toolNameSet {
		out.ToolCallNames = append(out.ToolCallNames, name)
	}
	out.HasToolCalls = len(out.ToolCallNames) > 0
	out.ContentPreview = previewString(strings.TrimSpace(strings.Join(contentParts, "")), 220)
	out.ReasoningPreview = previewString(strings.TrimSpace(strings.Join(reasoningParts, "")), 220)
	reasoningLower := strings.ToLower(strings.TrimSpace(out.ReasoningPreview))
	if strings.TrimSpace(out.ContentPreview) == "" && strings.Contains(reasoningLower, "</think>") {
		out.HasVisibleAnswerInReasoning = true
	}
	if out.Model == "" && out.FinishReason == "" && out.ContentPreview == "" && out.ReasoningPreview == "" && !out.HasToolCalls && out.ChunkCount == 0 && !out.HasDoneMarker {
		return nil
	}
	return out
}

func summarizeForensicResponsesOutput(payload []byte) *ForensicResponsesOutputShape {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return nil
	}
	parsed := gjson.ParseBytes(payload)
	out := &ForensicResponsesOutputShape{
		Status:            parsed.Get("status").String(),
		OutputCount:       len(parsed.Get("output").Array()),
		OutputTextPreview: previewString(parsed.Get("output_text").String(), 220),
	}
	for _, item := range parsed.Get("output").Array() {
		itemType := strings.TrimSpace(item.Get("type").String())
		switch itemType {
		case "message":
			out.MessageCount++
		case "reasoning":
			out.ReasoningCount++
		case "function_call", "apply_patch_call", "shell_call", "custom_tool_call":
			out.ToolCallCount++
			name := strings.TrimSpace(item.Get("name").String())
			if name == "" && strings.HasSuffix(itemType, "_call") {
				name = strings.TrimSuffix(itemType, "_call")
			}
			if name != "" {
				out.ToolNames = append(out.ToolNames, name)
			}
		}
	}
	out.HasEmptyCompleted = strings.EqualFold(out.Status, "completed") && out.OutputCount == 0 && strings.TrimSpace(out.OutputTextPreview) == ""
	return out
}

func summarizeForensicQwenStream(payload []byte) *ForensicQwenStreamShape {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return nil
	}
	parsed := gjson.ParseBytes(payload)
	out := &ForensicQwenStreamShape{
		RouteCounts:        map[string]int{},
		ArtifactCounts:     map[string]int{},
		PreferredOrigins:   map[string]string{},
		MalformedReasoning: parsed.Get("malformed_reasoning").Bool(),
	}
	for _, frame := range parsed.Get("frames").Array() {
		out.FrameCount++
		route := strings.TrimSpace(frame.Get("route").String())
		if route != "" {
			out.RouteCounts[route]++
		}
		for _, toolName := range frame.Get("tool_names").Array() {
			name := strings.TrimSpace(toolName.String())
			if name != "" {
				out.ToolNames = append(out.ToolNames, name)
			}
		}
	}
	for key, value := range parsed.Get("artifact_counts").Map() {
		out.ArtifactCounts[key] = int(value.Int())
	}
	for key, value := range parsed.Get("preferred_origins").Map() {
		if origin := strings.TrimSpace(value.String()); origin != "" {
			out.PreferredOrigins[key] = origin
		}
	}
	for _, kind := range parsed.Get("message_error_kinds").Array() {
		if text := strings.TrimSpace(kind.String()); text != "" {
			out.MessageErrorKinds = append(out.MessageErrorKinds, text)
		}
	}
	return out
}

func normalizeForensicToolChoice(choice gjson.Result) string {
	if !choice.Exists() {
		return ""
	}
	if choice.Type == gjson.String {
		return strings.TrimSpace(choice.String())
	}
	if fnName := strings.TrimSpace(choice.Get("function.name").String()); fnName != "" {
		return "function:" + fnName
	}
	return strings.TrimSpace(choice.Raw)
}

func previewString(text string, limit int) string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\r\n", "\n"))
	if text == "" || limit <= 0 {
		return text
	}
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}

func previewText(payload []byte, limit int) string {
	return previewString(string(payload), limit)
}

func timeFormatRFC3339(ts interface{}) string {
	return "2006-01-02T15:04:05Z07:00"
}

func summarizeMetricsRow(metric TokenMetrics, capture *ReqRespCapture) RequestForensicSummary {
	return buildRequestForensicSummary(metric, capture)
}

func mustMarshalForensicSummary(summary RequestForensicSummary) []byte {
	data, _ := json.Marshal(summary)
	return data
}
