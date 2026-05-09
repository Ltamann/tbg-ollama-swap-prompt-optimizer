package proxy

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestNormalizeQwenAssistantMessageFromBody_NativeToolPlanAndReasoning(t *testing.T) {
	body := []byte(`{
		"choices":[{
			"finish_reason":"tool_calls",
			"message":{
				"role":"assistant",
				"content":"<proposed_plan>\n1. Inspect\n2. Patch\n</proposed_plan>",
				"reasoning_content":"Need a plan first.",
				"tool_calls":[
					{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"pwd\"}"}},
					{"id":"call_2","type":"function","function":{"name":"request_user_input","arguments":"{\"questions\":[\"What next?\"]}"}}
				]
			}
		}]
	}`)

	artifacts, state := normalizeQwenAssistantMessageFromBody(body, true, true)
	if state == nil {
		t.Fatalf("expected accumulated state")
	}
	if state.DominantMode != QwenStreamModeToolCalling {
		t.Fatalf("expected tool calling mode, got %q", state.DominantMode)
	}
	plan := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactPlanBlock)
	if plan == nil {
		t.Fatalf("expected native plan block")
	}
	question := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactQuestionBlock)
	if question == nil {
		t.Fatalf("expected native question block")
	}
	tool := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactToolBlock)
	if tool == nil {
		t.Fatalf("expected native tool block")
	}
	reasoning := normalizedArtifactsPreferNative(artifacts, NormalizedArtifactThinkingBlock)
	if reasoning == nil {
		t.Fatalf("expected native reasoning block")
	}
}

func TestNormalizeQwenAssistantMessageFromBody_MalformedReasoningProducesError(t *testing.T) {
	body := []byte(`{
		"choices":[{
			"finish_reason":"stop",
			"message":{
				"role":"assistant",
				"content":"",
				"reasoning_content":"</think>\n\n</think>\n"
			}
		}]
	}`)

	artifacts, _ := normalizeQwenAssistantMessageFromBody(body, false, false)
	var foundThinking bool
	var foundError bool
	for _, artifact := range artifacts {
		if artifact.Kind == NormalizedArtifactThinkingBlock {
			foundThinking = true
			if artifact.Payload["well_formed"] != false {
				t.Fatalf("expected malformed thinking block")
			}
		}
		if artifact.Kind == NormalizedArtifactMessageError &&
			artifact.Payload["kind"] == string(MessageErrorUpstreamMalformedTurn) {
			foundError = true
		}
	}
	if !foundThinking {
		t.Fatalf("expected thinking artifact")
	}
	if !foundError {
		t.Fatalf("expected malformed turn error")
	}
}

func TestAccumulatedChoiceState_AddChatCompletionChunk_TracksToolIndexAndOrder(t *testing.T) {
	state := newAccumulatedChoiceState(0)
	state.addChatCompletionChunk(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"name":"apply_patch","arguments":"{\"operation\":"}},{"index":0,"id":"call_0","type":"function","function":{"name":"shell","arguments":"{\"command\":\""}}]}}]}`)
	state.addChatCompletionChunk(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"pwd\"}"}},{"index":1,"id":"call_1","type":"function","function":{"arguments":"{\"type\":\"update_file\"}}"}}],"finish_reason":"tool_calls"}]}`)

	if len(state.ToolOrder) != 2 {
		t.Fatalf("expected 2 tool indexes, got %d", len(state.ToolOrder))
	}
	if state.ToolOrder[0] != 0 || state.ToolOrder[1] != 1 {
		t.Fatalf("expected sorted tool order [0 1], got %#v", state.ToolOrder)
	}
	if got := state.ToolArgumentsByIx[0].Arguments.String(); got != "{\"command\":\"pwd\"}" {
		t.Fatalf("unexpected tool 0 args: %q", got)
	}
	if got := state.ToolArgumentsByIx[1].Arguments.String(); got != "{\"operation\":{\"type\":\"update_file\"}}" {
		t.Fatalf("unexpected tool 1 args: %q", got)
	}
}

func TestCollapseChatCompletionStreamToFinalBody_ReassemblesQuestionToolCall(t *testing.T) {
	stream := strings.NewReader("" +
		"data: {\"id\":\"chatcmpl_123\",\"created\":123,\"model\":\"qwen-test\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Need one question.\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl_123\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_q1\",\"type\":\"function\",\"function\":{\"name\":\"request_user_input\",\"arguments\":\"{\\\"questions\\\":[\\\"Which theme do you prefer?\\\"]}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n" +
		"data: [DONE]\n\n")

	body, state, err := collapseChatCompletionStreamToFinalBody(stream)
	if err != nil {
		t.Fatalf("collapse failed: %v", err)
	}
	if state == nil {
		t.Fatalf("expected accumulated state")
	}
	if got := gjson.GetBytes(body, "choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("unexpected finish reason: %q", got)
	}
	if got := gjson.GetBytes(body, "choices.0.message.tool_calls.0.function.name").String(); got != "request_user_input" {
		t.Fatalf("unexpected tool name: %q", got)
	}
	if got := gjson.GetBytes(body, "choices.0.message.tool_calls.0.function.arguments").String(); got != "{\"questions\":[\"Which theme do you prefer?\"]}" {
		t.Fatalf("unexpected tool args: %q", got)
	}
	if got := gjson.GetBytes(body, "choices.0.message.reasoning_content").String(); got != "Need one question." {
		t.Fatalf("unexpected reasoning: %q", got)
	}
}

func TestBuildNormalizedArtifactView_PrefersNativeQuestionAndPlan(t *testing.T) {
	artifacts := []NormalizedArtifact{
		{
			Kind:   NormalizedArtifactQuestionBlock,
			Origin: NormalizedArtifactOriginRecovered,
			Payload: map[string]any{
				"raw_args": `{"questions":["Recovered?"]}`,
			},
		},
		{
			Kind:   NormalizedArtifactQuestionBlock,
			Origin: NormalizedArtifactOriginNative,
			Payload: map[string]any{
				"raw_args": `{"questions":["Native?"]}`,
			},
		},
		{
			Kind:   NormalizedArtifactPlanBlock,
			Origin: NormalizedArtifactOriginNative,
			Payload: map[string]any{
				"text": "<proposed_plan>Native plan</proposed_plan>",
			},
		},
		{
			Kind:   NormalizedArtifactToolBlock,
			Origin: NormalizedArtifactOriginNative,
			Payload: map[string]any{
				"tool_name": "shell",
			},
		},
	}

	view := buildNormalizedArtifactView(artifacts)
	if view.PreferredQuestion == nil || view.PreferredQuestion.Origin != NormalizedArtifactOriginNative {
		t.Fatalf("expected native preferred question")
	}
	if view.PreferredPlan == nil || view.PreferredPlan.Origin != NormalizedArtifactOriginNative {
		t.Fatalf("expected native preferred plan")
	}
	if len(view.ToolBlocks) != 1 {
		t.Fatalf("expected 1 tool block, got %d", len(view.ToolBlocks))
	}
}

func TestNormalizeQwenStreamChunk_EmitsSemanticEvents(t *testing.T) {
	payload := `{"id":"chatcmpl_evt","created":123,"model":"qwen-test","choices":[{"index":0,"delta":{"reasoning_content":"Think.","content":"Answer","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
	events := normalizeQwenStreamChunk(payload)
	if len(events) < 5 {
		t.Fatalf("expected multiple events, got %d", len(events))
	}
	if events[0].Kind != QwenNormalizedStreamEventMeta {
		t.Fatalf("expected meta event first, got %q", events[0].Kind)
	}
	var sawThinking, sawAssistant, sawTool, sawFinish bool
	for _, event := range events {
		switch event.Kind {
		case QwenNormalizedStreamEventThinkingDelta:
			sawThinking = event.Text == "Think."
		case QwenNormalizedStreamEventAssistantDelta:
			sawAssistant = event.Text == "Answer"
		case QwenNormalizedStreamEventToolCallDelta:
			sawTool = event.ToolIndex == 0 && event.ToolName == "shell"
		case QwenNormalizedStreamEventFinish:
			sawFinish = event.FinishReason == "tool_calls"
		}
	}
	if !sawThinking || !sawAssistant || !sawTool || !sawFinish {
		t.Fatalf("missing expected semantic events: thinking=%v assistant=%v tool=%v finish=%v", sawThinking, sawAssistant, sawTool, sawFinish)
	}
}

func TestBuildQwenNormalizedStreamFrame_FoldsChunkSemantics(t *testing.T) {
	events := []QwenNormalizedStreamEvent{
		{Kind: QwenNormalizedStreamEventMeta, ChunkID: "chatcmpl-1", Model: "qwen-test", CreatedAt: 123, Usage: map[string]any{"input_tokens": 4}},
		{Kind: QwenNormalizedStreamEventThinkingDelta, Text: "Think"},
		{Kind: QwenNormalizedStreamEventThinkingDelta, Text: "."},
		{Kind: QwenNormalizedStreamEventAssistantDelta, Text: "Answer"},
		{Kind: QwenNormalizedStreamEventToolCallDelta, ToolIndex: 0, ToolName: "shell", ArgumentsDelta: `{"commands":["pwd"]}`},
		{Kind: QwenNormalizedStreamEventFinish, FinishReason: "tool_calls"},
	}

	frame := buildQwenNormalizedStreamFrame(events)
	if frame.ChunkID != "chatcmpl-1" {
		t.Fatalf("unexpected chunk id: %q", frame.ChunkID)
	}
	if frame.Model != "qwen-test" {
		t.Fatalf("unexpected model: %q", frame.Model)
	}
	if frame.CreatedAt != 123 {
		t.Fatalf("unexpected created_at: %d", frame.CreatedAt)
	}
	if frame.ReasoningDeltaText != "Think." {
		t.Fatalf("unexpected reasoning delta: %q", frame.ReasoningDeltaText)
	}
	if frame.AssistantDeltaText != "Answer" {
		t.Fatalf("unexpected assistant delta: %q", frame.AssistantDeltaText)
	}
	if len(frame.ToolCallDeltas) != 1 {
		t.Fatalf("expected 1 tool call delta, got %d", len(frame.ToolCallDeltas))
	}
	if frame.ToolCallDeltas[0].ToolName != "shell" {
		t.Fatalf("unexpected tool name: %q", frame.ToolCallDeltas[0].ToolName)
	}
	if frame.FinishReason != "tool_calls" {
		t.Fatalf("unexpected finish reason: %q", frame.FinishReason)
	}
	if frame.Usage == nil || frame.Usage["input_tokens"] != 4 {
		t.Fatalf("unexpected usage payload: %#v", frame.Usage)
	}
}

func TestDecideQwenStreamFramePresentation_RoutesBySemanticState(t *testing.T) {
	tests := []struct {
		name               string
		frame              QwenNormalizedStreamFrame
		workflowState      ToolWorkflowState
		planOutputRequired bool
		observedToolCount  int
		wantRoute          QwenStreamFrameRoute
		wantFinalize       bool
	}{
		{
			name:         "reasoning only routes to reasoning",
			frame:        QwenNormalizedStreamFrame{ReasoningDeltaText: "thinking", FinishReason: "tool_calls"},
			wantRoute:    QwenStreamFrameRouteReasoning,
			wantFinalize: true,
		},
		{
			name:              "assistant during tool work routes to commentary",
			frame:             QwenNormalizedStreamFrame{AssistantDeltaText: "Running tool..."},
			observedToolCount: 1,
			wantRoute:         QwenStreamFrameRouteCommentary,
		},
		{
			name:               "plan mode assistant routes to final buffer",
			frame:              QwenNormalizedStreamFrame{AssistantDeltaText: "1. Scope"},
			planOutputRequired: true,
			wantRoute:          QwenStreamFrameRouteFinal,
		},
		{
			name:          "verification pending assistant routes to final buffer",
			frame:         QwenNormalizedStreamFrame{AssistantDeltaText: "checking"},
			workflowState: ToolWorkflowState{VerificationExpected: true, VerificationCompleted: false},
			wantRoute:     QwenStreamFrameRouteFinal,
		},
		{
			name:      "plain assistant routes to final",
			frame:     QwenNormalizedStreamFrame{AssistantDeltaText: "hello"},
			wantRoute: QwenStreamFrameRouteFinal,
		},
		{
			name:      "empty chunk routes to none",
			frame:     QwenNormalizedStreamFrame{},
			wantRoute: QwenStreamFrameRouteNone,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := decideQwenStreamFramePresentation(tc.frame, tc.workflowState, tc.planOutputRequired, tc.observedToolCount)
			if got.Route != tc.wantRoute {
				t.Fatalf("route mismatch: got %q want %q", got.Route, tc.wantRoute)
			}
			if got.ShouldFinalizeToolCalls != tc.wantFinalize {
				t.Fatalf("finalize mismatch: got %v want %v", got.ShouldFinalizeToolCalls, tc.wantFinalize)
			}
		})
	}
}
