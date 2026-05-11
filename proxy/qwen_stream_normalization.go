package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

func newAccumulatedChoiceState(choiceIndex int) *AccumulatedChoiceState {
	return &AccumulatedChoiceState{
		ChoiceIndex:       choiceIndex,
		DominantMode:      QwenStreamModeIdle,
		ToolArgumentsByIx: map[int]*ToolArgumentAccumulator{},
	}
}

func (s *AccumulatedChoiceState) nextSequence() int {
	seq := s.Sequence
	s.Sequence++
	return seq
}

func (s *AccumulatedChoiceState) appendRawChunk(payload string) {
	if s == nil {
		return
	}
	s.RawChunks = append(s.RawChunks, RawChunkRecord{
		Sequence: s.nextSequence(),
		Payload:  payload,
	})
}

func (s *AccumulatedChoiceState) toolAccumulator(index int) *ToolArgumentAccumulator {
	if s == nil {
		return nil
	}
	if acc, ok := s.ToolArgumentsByIx[index]; ok {
		return acc
	}
	acc := &ToolArgumentAccumulator{ToolIndex: index}
	s.ToolArgumentsByIx[index] = acc
	s.ToolOrder = append(s.ToolOrder, index)
	sort.Ints(s.ToolOrder)
	return acc
}

func (s *AccumulatedChoiceState) addChatCompletionChunk(payload string) {
	if s == nil || !gjson.Valid(payload) {
		return
	}
	s.appendRawChunk(payload)

	finishReason := strings.TrimSpace(gjson.Get(payload, "choices.0.finish_reason").String())
	if finishReason != "" {
		s.FinishReason = finishReason
		if strings.EqualFold(finishReason, "stop") || strings.EqualFold(finishReason, "tool_calls") {
			s.DominantMode = QwenStreamModeDone
		}
	}

	if content := gjson.Get(payload, "choices.0.delta.content").String(); content != "" {
		s.Content.WriteString(content)
		s.HasAnswerContent = true
		if s.DominantMode != QwenStreamModeToolCalling {
			s.DominantMode = QwenStreamModeAnswering
		}
	}

	reasoning := gjson.Get(payload, "choices.0.delta.reasoning_content").String()
	if reasoning == "" {
		reasoning = gjson.Get(payload, "choices.0.delta.reasoning").String()
	}
	if reasoning != "" {
		s.Reasoning.WriteString(reasoning)
		s.HasReasoning = true
		if s.DominantMode == QwenStreamModeIdle || s.DominantMode == QwenStreamModeThinking {
			s.DominantMode = QwenStreamModeThinking
		}
	}

	toolCalls := gjson.Get(payload, "choices.0.delta.tool_calls")
	if toolCalls.IsArray() {
		s.HasToolDelta = true
		s.DominantMode = QwenStreamModeToolCalling
		for _, tc := range toolCalls.Array() {
			index := int(tc.Get("index").Int())
			acc := s.toolAccumulator(index)
			if id := strings.TrimSpace(tc.Get("id").String()); id != "" {
				acc.ToolID = id
			}
			if name := strings.TrimSpace(tc.Get("function.name").String()); name != "" {
				acc.ToolName = name
			}
			if argDelta := tc.Get("function.arguments").String(); argDelta != "" {
				acc.Arguments.WriteString(argDelta)
			}
		}
	}
}

func (s *AccumulatedChoiceState) markTruncated() {
	if s == nil {
		return
	}
	s.Truncated = true
}

func (s *AccumulatedChoiceState) buildNormalizedArtifacts(message map[string]any, planOutputRequired bool, nativeQuestionRequired bool) []NormalizedArtifact {
	if s == nil {
		return nil
	}
	artifacts := make([]NormalizedArtifact, 0, 8)
	nextSeq := 0
	appendArtifact := func(kind NormalizedArtifactKind, origin NormalizedArtifactOrigin, source string, payload map[string]any) {
		artifacts = append(artifacts, NormalizedArtifact{
			Kind:         kind,
			Origin:       origin,
			Sequence:     nextSeq,
			NativeSource: source,
			Payload:      payload,
		})
		nextSeq++
	}

	rawReasoning := strings.TrimSpace(fmt.Sprintf("%v", message["reasoning_content"]))
	if rawReasoning == "" {
		rawReasoning = strings.TrimSpace(fmt.Sprintf("%v", message["reasoning"]))
	}
	if rawReasoning != "" {
		wellFormed, malformedReason := classifyReasoningBlock(rawReasoning)
		appendArtifact(NormalizedArtifactThinkingBlock, NormalizedArtifactOriginNative, "reasoning_content", map[string]any{
			"text":             rawReasoning,
			"well_formed":      wellFormed,
			"malformed_reason": malformedReason,
		})
		if !wellFormed {
			appendArtifact(NormalizedArtifactMessageError, NormalizedArtifactOriginNative, "reasoning_content", map[string]any{
				"kind":        string(MessageErrorUpstreamMalformedTurn),
				"recoverable": true,
				"details":     malformedReason,
			})
		}
	}

	content := strings.TrimSpace(fmt.Sprintf("%v", message["content"]))
	if content != "" {
		appendArtifact(NormalizedArtifactTextBlock, NormalizedArtifactOriginNative, "content", map[string]any{
			"text": content,
		})
		if strings.Contains(content, "<proposed_plan>") && strings.Contains(content, "</proposed_plan>") {
			appendArtifact(NormalizedArtifactPlanBlock, NormalizedArtifactOriginNative, "content", map[string]any{
				"text": sanitizeStructuredPlanText(content),
			})
		}
	}

	if toolCalls, ok := message["tool_calls"].([]any); ok && len(toolCalls) > 0 {
		for idx, rawTool := range toolCalls {
			tc, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := tc["function"].(map[string]any)
			toolName := strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
			rawArgs := strings.TrimSpace(fmt.Sprintf("%v", fn["arguments"]))
			parsedArgs := parseToolArgsMapString(rawArgs)
			parseStatus := "parsed"
			if rawArgs != "" && len(parsedArgs) == 0 {
				parseStatus = "raw_only"
			}
			appendArtifact(NormalizedArtifactToolBlock, NormalizedArtifactOriginNative, "tool_calls", map[string]any{
				"tool_index":   idx,
				"tool_name":    toolName,
				"tool_id":      strings.TrimSpace(fmt.Sprintf("%v", tc["id"])),
				"raw_args":     rawArgs,
				"parsed_args":  parsedArgs,
				"parse_status": parseStatus,
			})
			if strings.EqualFold(toolName, "request_user_input") {
				appendArtifact(NormalizedArtifactQuestionBlock, NormalizedArtifactOriginNative, "tool_calls", map[string]any{
					"tool_index":  idx,
					"tool_name":   toolName,
					"tool_id":     strings.TrimSpace(fmt.Sprintf("%v", tc["id"])),
					"raw_args":    rawArgs,
					"parsed_args": parsedArgs,
				})
			}
			if rawArgs == "" {
				appendArtifact(NormalizedArtifactMessageError, NormalizedArtifactOriginNative, "tool_calls", map[string]any{
					"kind":        string(MessageErrorPartialToolMissingArgs),
					"recoverable": true,
					"details":     fmt.Sprintf("%s missing tool arguments", toolName),
				})
			}
		}
	}

	if len(artifacts) == 0 && rawReasoning != "" && content == "" {
		appendArtifact(NormalizedArtifactMessageError, NormalizedArtifactOriginNative, "reasoning_content", map[string]any{
			"kind":        string(MessageErrorReasoningOnlyTurn),
			"recoverable": planOutputRequired || nativeQuestionRequired,
			"details":     "upstream turn only contained reasoning without content or tool calls",
		})
	}

	if s.Truncated {
		appendArtifact(NormalizedArtifactMessageError, NormalizedArtifactOriginNative, "stream", map[string]any{
			"kind":        string(MessageErrorTruncatedStream),
			"recoverable": true,
			"details":     "upstream stream ended before a terminal finish_reason was observed",
		})
	}

	return artifacts
}

func normalizeQwenAssistantMessageFromBody(body []byte, planOutputRequired bool, nativeQuestionRequired bool) ([]NormalizedArtifact, *AccumulatedChoiceState) {
	parsed := gjson.ParseBytes(body)
	messageResult := parsed.Get("choices.0.message")
	message := map[string]any{}
	if messageResult.Exists() {
		_ = json.Unmarshal([]byte(messageResult.Raw), &message)
	}
	state := newAccumulatedChoiceState(0)
	state.FinishReason = strings.TrimSpace(parsed.Get("choices.0.finish_reason").String())
	if content := strings.TrimSpace(fmt.Sprintf("%v", message["content"])); content != "" {
		state.Content.WriteString(content)
		state.HasAnswerContent = true
		state.DominantMode = QwenStreamModeAnswering
	}
	if reasoning := strings.TrimSpace(fmt.Sprintf("%v", message["reasoning_content"])); reasoning != "" {
		state.Reasoning.WriteString(reasoning)
		state.HasReasoning = true
		if state.DominantMode == QwenStreamModeIdle {
			state.DominantMode = QwenStreamModeThinking
		}
	}
	if toolCalls, ok := message["tool_calls"].([]any); ok && len(toolCalls) > 0 {
		state.HasToolDelta = true
		state.DominantMode = QwenStreamModeToolCalling
		for idx, rawTool := range toolCalls {
			tc, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			acc := state.toolAccumulator(idx)
			acc.ToolID = strings.TrimSpace(fmt.Sprintf("%v", tc["id"]))
			if fn, ok := tc["function"].(map[string]any); ok {
				acc.ToolName = strings.TrimSpace(fmt.Sprintf("%v", fn["name"]))
				acc.Arguments.WriteString(strings.TrimSpace(fmt.Sprintf("%v", fn["arguments"])))
			}
		}
	}
	if strings.TrimSpace(state.FinishReason) == "" {
		state.markTruncated()
	}
	return state.buildNormalizedArtifacts(message, planOutputRequired, nativeQuestionRequired), state
}

func buildAccumulatedAssistantMessage(state *AccumulatedChoiceState) map[string]any {
	if state == nil {
		return map[string]any{}
	}
	message := map[string]any{}
	content := strings.TrimSpace(state.Content.String())
	reasoning := strings.TrimSpace(state.Reasoning.String())
	content, reasoning = normalizeAssistantReasoningFields(content, reasoning)
	content = strings.TrimSpace(stripLeadingReasoningDirective(content))
	reasoning = strings.TrimSpace(stripLeadingReasoningDirective(reasoning))
	if content != "" {
		message["content"] = content
	}
	if reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	if len(state.ToolOrder) > 0 {
		toolCalls := make([]any, 0, len(state.ToolOrder))
		for _, idx := range state.ToolOrder {
			acc := state.ToolArgumentsByIx[idx]
			if acc == nil {
				continue
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   acc.ToolID,
				"type": "function",
				"function": map[string]any{
					"name":      acc.ToolName,
					"arguments": acc.Arguments.String(),
				},
			})
		}
		if len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
		}
	}
	return message
}

func classifyReasoningBlock(text string) (bool, string) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return true, ""
	}
	if strings.Trim(trimmed, "</think>\n\r\t ") == "" {
		return false, "reasoning only contained think closers"
	}
	closeCount := strings.Count(trimmed, "</think>")
	openCount := strings.Count(trimmed, "<think>")
	if closeCount > 0 && openCount == 0 {
		return false, "reasoning contained orphan think closers"
	}
	return true, ""
}

func normalizedArtifactsPreferNative(artifacts []NormalizedArtifact, kind NormalizedArtifactKind) *NormalizedArtifact {
	var recovered *NormalizedArtifact
	for i := range artifacts {
		if artifacts[i].Kind != kind {
			continue
		}
		if artifacts[i].Origin == NormalizedArtifactOriginNative {
			return &artifacts[i]
		}
		if recovered == nil {
			recovered = &artifacts[i]
		}
	}
	return recovered
}

func buildNormalizedArtifactView(artifacts []NormalizedArtifact) NormalizedArtifactView {
	view := NormalizedArtifactView{
		Artifacts:  artifacts,
		ToolBlocks: make([]NormalizedArtifact, 0, 4),
	}
	for i := range artifacts {
		artifact := artifacts[i]
		switch artifact.Kind {
		case NormalizedArtifactThinkingBlock:
			if view.PreferredThinking == nil || view.PreferredThinking.Origin != NormalizedArtifactOriginNative {
				view.PreferredThinking = &artifacts[i]
			}
		case NormalizedArtifactTextBlock:
			if view.PreferredText == nil || view.PreferredText.Origin != NormalizedArtifactOriginNative {
				view.PreferredText = &artifacts[i]
			}
		case NormalizedArtifactPlanBlock:
			if view.PreferredPlan == nil || view.PreferredPlan.Origin != NormalizedArtifactOriginNative {
				view.PreferredPlan = &artifacts[i]
			}
		case NormalizedArtifactQuestionBlock:
			if view.PreferredQuestion == nil || view.PreferredQuestion.Origin != NormalizedArtifactOriginNative {
				view.PreferredQuestion = &artifacts[i]
			}
		case NormalizedArtifactMessageError:
			if view.PreferredError == nil || view.PreferredError.Origin != NormalizedArtifactOriginNative {
				view.PreferredError = &artifacts[i]
			}
		case NormalizedArtifactToolBlock:
			view.ToolBlocks = append(view.ToolBlocks, artifact)
		}
	}
	if preferred := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactThinkingBlock); preferred != nil {
		view.PreferredThinking = preferred
	}
	if preferred := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactTextBlock); preferred != nil {
		view.PreferredText = preferred
	}
	if preferred := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactPlanBlock); preferred != nil {
		view.PreferredPlan = preferred
	}
	if preferred := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactQuestionBlock); preferred != nil {
		view.PreferredQuestion = preferred
	}
	if preferred := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactMessageError); preferred != nil {
		view.PreferredError = preferred
	}
	return view
}

func collapseChatCompletionStreamToFinalBody(upstream io.Reader) ([]byte, *AccumulatedChoiceState, error) {
	state := newAccumulatedChoiceState(0)
	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	respID := ""
	model := ""
	createdAt := time.Now().Unix()
	latestUsageRaw := ""
	latestTimingsRaw := ""

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
		state.addChatCompletionChunk(payload)
		if chunkID := strings.TrimSpace(gjson.Get(payload, "id").String()); chunkID != "" && respID == "" {
			respID = chunkID
		}
		if chunkModel := strings.TrimSpace(gjson.Get(payload, "model").String()); chunkModel != "" && model == "" {
			model = chunkModel
		}
		if created := gjson.Get(payload, "created").Int(); created > 0 {
			createdAt = created
		}
		if usage := gjson.Get(payload, "usage"); usage.Exists() && usage.Raw != "" && usage.Raw != "null" {
			latestUsageRaw = usage.Raw
		}
		if timings := gjson.Get(payload, "timings"); timings.Exists() && timings.Raw != "" && timings.Raw != "null" {
			latestTimingsRaw = timings.Raw
		}
	}
	if err := scanner.Err(); err != nil {
		state.markTruncated()
		return nil, state, err
	}
	if strings.TrimSpace(state.FinishReason) == "" {
		state.markTruncated()
	}
	if strings.TrimSpace(respID) == "" {
		respID = fmt.Sprintf("chatcmpl_stream_%d", time.Now().UnixNano())
	}

	choice := map[string]any{
		"index":         0,
		"message":       buildAccumulatedAssistantMessage(state),
		"finish_reason": strings.TrimSpace(state.FinishReason),
	}
	if strings.TrimSpace(fmt.Sprintf("%v", choice["finish_reason"])) == "" {
		choice["finish_reason"] = "stop"
	}

	resp := map[string]any{
		"id":      respID,
		"object":  "chat.completion",
		"created": createdAt,
		"model":   model,
		"choices": []any{choice},
	}
	if latestUsageRaw != "" {
		var usage map[string]any
		if json.Unmarshal([]byte(latestUsageRaw), &usage) == nil {
			resp["usage"] = usage
		}
	}
	if latestTimingsRaw != "" {
		var timings map[string]any
		if json.Unmarshal([]byte(latestTimingsRaw), &timings) == nil {
			resp["timings"] = timings
		}
	}
	body, err := json.Marshal(resp)
	if err != nil {
		return nil, state, err
	}
	return body, state, nil
}

func normalizeQwenStreamChunk(payload string) []QwenNormalizedStreamEvent {
	if !gjson.Valid(payload) {
		return nil
	}
	events := make([]QwenNormalizedStreamEvent, 0, 8)
	chunkID := strings.TrimSpace(gjson.Get(payload, "id").String())
	model := strings.TrimSpace(gjson.Get(payload, "model").String())
	createdAt := gjson.Get(payload, "created").Int()
	usageMap := buildResponsesUsageFromChatUsage(gjson.Get(payload, "usage"))
	timingMap := buildResponsesTimingFromChatTiming(gjson.Get(payload, "timings"))

	if chunkID != "" || model != "" || createdAt > 0 || usageMap != nil || timingMap != nil {
		events = append(events, QwenNormalizedStreamEvent{
			Kind:      QwenNormalizedStreamEventMeta,
			ChunkID:   chunkID,
			Model:     model,
			CreatedAt: createdAt,
			Usage:     usageMap,
			Timings:   timingMap,
		})
	}

	if reasoning := gjson.Get(payload, "choices.0.delta.reasoning_content").String(); reasoning != "" {
		events = append(events, QwenNormalizedStreamEvent{
			Kind: QwenNormalizedStreamEventThinkingDelta,
			Text: reasoning,
		})
	} else if reasoning := gjson.Get(payload, "choices.0.delta.reasoning").String(); reasoning != "" {
		events = append(events, QwenNormalizedStreamEvent{
			Kind: QwenNormalizedStreamEventThinkingDelta,
			Text: reasoning,
		})
	}

	if content := gjson.Get(payload, "choices.0.delta.content").String(); content != "" {
		events = append(events, QwenNormalizedStreamEvent{
			Kind: QwenNormalizedStreamEventAssistantDelta,
			Text: content,
		})
	}

	if toolCalls := gjson.Get(payload, "choices.0.delta.tool_calls"); toolCalls.IsArray() {
		for _, tc := range toolCalls.Array() {
			events = append(events, QwenNormalizedStreamEvent{
				Kind:           QwenNormalizedStreamEventToolCallDelta,
				ToolIndex:      int(tc.Get("index").Int()),
				ToolID:         strings.TrimSpace(tc.Get("id").String()),
				ToolName:       strings.TrimSpace(tc.Get("function.name").String()),
				ArgumentsDelta: tc.Get("function.arguments").String(),
			})
		}
	}

	if finishReason := strings.TrimSpace(gjson.Get(payload, "choices.0.finish_reason").String()); finishReason != "" {
		events = append(events, QwenNormalizedStreamEvent{
			Kind:         QwenNormalizedStreamEventFinish,
			FinishReason: finishReason,
		})
	}
	return events
}

func buildQwenNormalizedStreamFrame(events []QwenNormalizedStreamEvent) QwenNormalizedStreamFrame {
	frame := QwenNormalizedStreamFrame{
		Events:         append([]QwenNormalizedStreamEvent(nil), events...),
		ToolCallDeltas: make([]QwenNormalizedStreamEvent, 0, 2),
	}
	for _, event := range events {
		switch event.Kind {
		case QwenNormalizedStreamEventMeta:
			if frame.ChunkID == "" && strings.TrimSpace(event.ChunkID) != "" {
				frame.ChunkID = strings.TrimSpace(event.ChunkID)
			}
			if frame.Model == "" && strings.TrimSpace(event.Model) != "" {
				frame.Model = strings.TrimSpace(event.Model)
			}
			if frame.CreatedAt == 0 && event.CreatedAt > 0 {
				frame.CreatedAt = event.CreatedAt
			}
			if frame.Usage == nil && event.Usage != nil {
				frame.Usage = event.Usage
			}
			if frame.Timings == nil && event.Timings != nil {
				frame.Timings = event.Timings
			}
		case QwenNormalizedStreamEventThinkingDelta:
			frame.ReasoningDeltaText += event.Text
		case QwenNormalizedStreamEventAssistantDelta:
			frame.AssistantDeltaText += event.Text
		case QwenNormalizedStreamEventToolCallDelta:
			frame.ToolCallDeltas = append(frame.ToolCallDeltas, event)
		case QwenNormalizedStreamEventFinish:
			if frame.FinishReason == "" && strings.TrimSpace(event.FinishReason) != "" {
				frame.FinishReason = strings.TrimSpace(event.FinishReason)
			}
		}
	}
	return frame
}

func decideQwenStreamFramePresentation(frame QwenNormalizedStreamFrame, workflowState ToolWorkflowState, planOutputRequired bool, observedToolCount int) QwenStreamFrameDecision {
	decision := QwenStreamFrameDecision{
		HasAssistantDelta:       strings.TrimSpace(frame.AssistantDeltaText) != "",
		HasReasoningDelta:       strings.TrimSpace(frame.ReasoningDeltaText) != "",
		ShouldFinalizeToolCalls: strings.EqualFold(strings.TrimSpace(frame.FinishReason), "tool_calls"),
	}
	switch {
	case !decision.HasAssistantDelta && !decision.HasReasoningDelta:
		decision.Route = QwenStreamFrameRouteNone
	case decision.HasReasoningDelta && !decision.HasAssistantDelta:
		decision.Route = QwenStreamFrameRouteReasoning
		decision.Text = frame.ReasoningDeltaText
	case planOutputRequired || (workflowState.VerificationExpected && !workflowState.VerificationCompleted) || workflowState.FinalAnswerSafe:
		decision.Route = QwenStreamFrameRouteFinal
		decision.Text = frame.AssistantDeltaText
	case observedToolCount > 0 || (workflowState.HasToolOutput && !workflowState.FinalAnswerSafe):
		decision.Route = QwenStreamFrameRouteCommentary
		decision.Text = frame.AssistantDeltaText
	default:
		decision.Route = QwenStreamFrameRouteFinal
		decision.Text = frame.AssistantDeltaText
	}
	return decision
}

func applyNormalizedToolCallDelta(
	event QwenNormalizedStreamEvent,
	toolStates map[int]*StreamToolCallState,
	toolOrder *[]int,
	respID string,
	planOutputRequired bool,
	workflowState ToolWorkflowState,
	streamAdapter StreamReconstructionAdapter,
	startToolStateIfReady func(*StreamToolCallState),
	emitCanonicalStreamEvent func(StreamEvent),
	emitProgressCommentary func(string),
	lastProgressCommentary *string,
) {
	idx := event.ToolIndex
	state, exists := toolStates[idx]
	if !exists {
		callID := strings.TrimSpace(event.ToolID)
		if callID == "" {
			callID = fmt.Sprintf("call_%s_%d", respID, idx)
		}
		state = &StreamToolCallState{
			Index:       idx,
			OutputIndex: -1,
			ItemID:      fmt.Sprintf("fc_%s_%d", respID, idx),
			CallID:      callID,
		}
		toolStates[idx] = state
		*toolOrder = append(*toolOrder, idx)
	}
	if id := strings.TrimSpace(event.ToolID); id != "" {
		state.CallID = id
	}
	if name := strings.TrimSpace(event.ToolName); name != "" {
		state.Name = name
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			if existing := toolStates[idx]; existing != nil {
				existing.Name = name
			}
		}
		if progress := toolStartProgressCommentary(state.Name, workflowState); strings.TrimSpace(progress) != "" && *lastProgressCommentary != progress {
			*lastProgressCommentary = progress
			emitProgressCommentary(progress)
		}
	}
	startToolStateIfReady(state)
	if argDelta := event.ArgumentsDelta; argDelta != "" {
		state.ArgsBuilder.WriteString(argDelta)
		startToolStateIfReady(state)
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			return
		}
		if argsEvent, ok := streamAdapter.BuildToolArgsDeltaEvent(state, respID); ok {
			emitCanonicalStreamEvent(argsEvent)
			state.ArgsEmitted = true
		}
	}
}

func finalizeNormalizedToolCallPhase(
	toolOrder []int,
	toolStates map[int]*StreamToolCallState,
	respID string,
	planOutputRequired bool,
	streamAdapter StreamReconstructionAdapter,
	reasoningText string,
	finalText string,
	originalReq map[string]any,
	emitCanonicalStreamEvent func(StreamEvent),
	emitFinalDelta func(string),
) {
	for _, toolIdx := range toolOrder {
		state := toolStates[toolIdx]
		if state == nil {
			continue
		}
		arguments := streamAdapter.CanonicalToolArguments(state, state.ArgsBuilder.String())
		if strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "request_user_input") {
			if normalized := strings.TrimSpace(arguments); normalized == "" || normalized == "{}" {
				if recovered, ok := streamedRequestUserInputArgumentsFallback(reasoningText, finalText, originalReq); ok {
					state.ArgsBuilder.Reset()
					state.ArgsBuilder.WriteString(recovered)
					arguments = streamAdapter.CanonicalToolArguments(state, state.ArgsBuilder.String())
					if !state.ArgsEmitted {
						if event, ok := streamAdapter.BuildToolArgsDeltaEvent(state, respID); ok {
							emitCanonicalStreamEvent(event)
							state.ArgsEmitted = true
						}
					}
				}
			}
		}
		if strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "shell") && !shellToolArgumentsValid(parseToolArgsMapString(arguments)) {
			if strings.TrimSpace(finalText) == "" {
				emitFinalDelta(shellValidationWarningPrefix + " arguments were empty. Provide a non-empty `command` string or `commands` array and retry.")
			}
			continue
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "shell") &&
			shellToolArgumentsLookLikePlainQuestion(parseToolArgsMapString(arguments)) {
			continue
		}
		if planOutputRequired && strings.EqualFold(streamAdapter.CanonicalToolName(state.Name), "apply_patch") {
			continue
		}
		if event, ok := streamAdapter.BuildToolArgsDoneEvent(state, respID); ok {
			emitCanonicalStreamEvent(event)
		}
		if event, ok := streamAdapter.BuildToolItemDoneEvent(state, respID); ok {
			emitCanonicalStreamEvent(event)
		}
	}
}

func summarizeQwenNormalizedStreamFrame(frame QwenNormalizedStreamFrame, decision QwenStreamFrameDecision) QwenStreamFrameTraceSummary {
	toolNames := make([]string, 0, len(frame.ToolCallDeltas))
	for _, tc := range frame.ToolCallDeltas {
		name := strings.TrimSpace(tc.ToolName)
		if name != "" {
			toolNames = append(toolNames, name)
		}
	}
	return QwenStreamFrameTraceSummary{
		Route:         string(decision.Route),
		FinishReason:  strings.TrimSpace(frame.FinishReason),
		AssistantLen:  len(strings.TrimSpace(frame.AssistantDeltaText)),
		ReasoningLen:  len(strings.TrimSpace(frame.ReasoningDeltaText)),
		ToolCount:     len(frame.ToolCallDeltas),
		ToolNames:     toolNames,
		FinalizeTools: decision.ShouldFinalizeToolCalls,
	}
}

func addQwenNormalizedStreamCaptureStage(ctx context.Context, frameSummaries []QwenStreamFrameTraceSummary, normalizedView NormalizedArtifactView) {
	if ctx == nil {
		return
	}
	artifactCounts := map[string]int{}
	toolNames := make([]string, 0, len(normalizedView.ToolBlocks))
	messageErrorKinds := make([]string, 0, 2)
	malformedReasoning := false
	for _, artifact := range normalizedView.Artifacts {
		artifactCounts[string(artifact.Kind)]++
		if artifact.Kind == NormalizedArtifactToolBlock {
			if name := strings.TrimSpace(cleanFallbackInput(artifact.Payload["tool_name"], "")); name != "" {
				toolNames = append(toolNames, name)
			}
		}
		if artifact.Kind == NormalizedArtifactMessageError {
			if kind := strings.TrimSpace(cleanFallbackInput(artifact.Payload["kind"], "")); kind != "" {
				messageErrorKinds = append(messageErrorKinds, kind)
			}
		}
		if artifact.Kind == NormalizedArtifactThinkingBlock {
			if wellFormed, ok := artifact.Payload["well_formed"].(bool); ok && !wellFormed {
				malformedReasoning = true
			}
		}
	}
	preferredOrigins := map[string]string{}
	if normalizedView.PreferredThinking != nil {
		preferredOrigins["thinking"] = string(normalizedView.PreferredThinking.Origin)
	}
	if normalizedView.PreferredText != nil {
		preferredOrigins["text"] = string(normalizedView.PreferredText.Origin)
	}
	if normalizedView.PreferredPlan != nil {
		preferredOrigins["plan"] = string(normalizedView.PreferredPlan.Origin)
	}
	if normalizedView.PreferredQuestion != nil {
		preferredOrigins["question"] = string(normalizedView.PreferredQuestion.Origin)
	}
	if normalizedView.PreferredError != nil {
		preferredOrigins["error"] = string(normalizedView.PreferredError.Origin)
	}
	payload := map[string]any{
		"frames":              frameSummaries,
		"artifact_counts":     artifactCounts,
		"preferred_origins":   preferredOrigins,
		"tool_names":          toolNames,
		"message_error_kinds": messageErrorKinds,
		"malformed_reasoning": malformedReasoning,
	}
	if encoded, err := json.Marshal(payload); err == nil {
		addCaptureStage(ctx, "bridge.qwen_stream_normalization", encoded)
	}
}
