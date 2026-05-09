package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildRequestForensicSummary_ClassifiesUpstreamExitAndRequestShape(t *testing.T) {
	metric := TokenMetrics{
		ID:         7,
		TraceID:    "trace-7",
		Timestamp:  time.Date(2026, 5, 5, 19, 22, 39, 0, time.FixedZone("CEST", 2*3600)),
		Model:      "Qwen3.6-35B-A3B-UD-Q8_K_XL",
		StatusCode: 502,
		DurationMs: 11254,
	}
	capture := &ReqRespCapture{
		ID:      7,
		ReqPath: "/v1/responses",
		Stages: []CaptureStage{
			{
				Name:    "bridge.responses_request",
				Payload: []byte(`{"model":"gpt-5.2","tool_choice":"none","parallel_tool_calls":false,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Plan only"}]}]}`),
			},
			{
				Name:    "bridge.chat_completions_request",
				Payload: []byte(`{"model":"Qwen","tool_choice":"none","parallel_tool_calls":false,"chat_template_kwargs":{"enable_thinking":true},"logit_bias":{"248069":11.8},"messages":[{"role":"system","content":"Plan mode"}]}`),
			},
			{
				Name:    "bridge.chat_completions_response",
				Payload: []byte("unable to start process: upstream command exited prematurely but successfully\n"),
			},
		},
	}

	summary := buildRequestForensicSummary(metric, capture)
	require.Equal(t, 7, summary.ID)
	assert.Equal(t, "/v1/responses", summary.ReqPath)
	assert.Equal(t, "upstream_process_lifecycle", summary.FirstWrongStage)
	assert.Equal(t, "upstream_exit_during_request", summary.FailureClass)
	require.NotNil(t, summary.ChatRequest)
	assert.Equal(t, "none", summary.ChatRequest.ToolChoice)
	assert.True(t, summary.ChatRequest.HasCloseThinkBias)
	require.NotNil(t, summary.ChatResponse)
	assert.Contains(t, summary.ChatResponse.ErrorText, "unable to start process")
	assert.Contains(t, summary.Notes, "tool-less translated request still carried close-think bias")
}

func TestBuildRequestForensicSummary_PromotesQwenStreamAndMalformedReasoning(t *testing.T) {
	metric := TokenMetrics{
		ID:         9,
		TraceID:    "trace-9",
		Timestamp:  time.Date(2026, 5, 8, 10, 15, 0, 0, time.FixedZone("CEST", 2*3600)),
		Model:      "Qwen3.6-35B-A3B-UD-Q8_K_XL",
		StatusCode: 200,
		DurationMs: 4321,
	}
	capture := &ReqRespCapture{
		ID:      9,
		ReqPath: "/v1/responses",
		Stages: []CaptureStage{
			{
				Name:    "bridge.qwen_stream_normalization",
				Payload: []byte(`{"frames":[{"route":"reasoning"}],"artifact_counts":{"thinking_block":1},"preferred_origins":{"thinking":"native","question":"recovered"},"tool_names":[],"message_error_kinds":["reasoning_only_turn"],"malformed_reasoning":true}`),
			},
		},
	}

	summary := buildRequestForensicSummary(metric, capture)
	require.NotNil(t, summary.QwenStream)
	assert.Equal(t, 1, summary.QwenStream.FrameCount)
	assert.Equal(t, "qwen_stream_normalization", summary.FirstWrongStage)
	assert.Equal(t, "malformed_reasoning_detected", summary.FailureClass)
	assert.Contains(t, summary.Notes, "normalized Qwen stream reported malformed reasoning")
	assert.Contains(t, summary.Notes, "normalized Qwen stream recovered question semantics from non-native output")
	assert.Contains(t, summary.Notes, "normalized Qwen stream reported semantic message errors")
	assert.Contains(t, summary.Notes, "normalized Qwen stream never reached visible commentary or final output")
}

func TestApiGetForensics_ReturnsStructuredSummary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pm := &ProxyManager{
		ginEngine:      gin.New(),
		metricsMonitor: newMetricsMonitor(NewLogMonitorWriter(io.Discard), 16, 4),
		processGroups:  map[string]*ProcessGroup{},
		ctxSizes:       map[string]int{},
	}
	addApiHandlers(pm)

	metric := TokenMetrics{
		Timestamp:  time.Date(2026, 5, 5, 19, 35, 25, 0, time.FixedZone("CEST", 2*3600)),
		Model:      "Qwen3.6-35B-A3B-UD-Q8_K_XL",
		StatusCode: 200,
		DurationMs: 12374,
	}
	id := pm.metricsMonitor.addMetrics(metric)
	pm.metricsMonitor.addCapture(ReqRespCapture{
		ID:      id,
		ReqPath: "/v1/responses",
		Stages: []CaptureStage{
			{
				Name:    "bridge.chat_completions_request",
				Payload: []byte(`{"tool_choice":"none","parallel_tool_calls":false,"chat_template_kwargs":{"enable_thinking":true},"messages":[{"role":"user","content":"Plan only"}]}`),
			},
			{
				Name:    "bridge.qwen_stream_normalization",
				Payload: []byte(`{"frames":[{"route":"reasoning","tool_names":["request_user_input"],"finalize_tools":true},{"route":"commentary"},{"route":"final"}],"artifact_counts":{"thinking_block":1,"question_block":1},"preferred_origins":{"thinking":"native","question":"recovered"},"tool_names":["request_user_input"],"message_error_kinds":[],"malformed_reasoning":false}`),
			},
			{
				Name:    "bridge.chat_completions_response",
				Payload: []byte(`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"T22_SENTINEL","reasoning_content":"plan reasoning"}}]}`),
			},
			{
				Name:    "bridge.responses_output",
				Payload: []byte(`{"status":"completed","output":[{"type":"reasoning"},{"type":"message","content":[{"type":"output_text","text":"T22_SENTINEL"}]}],"output_text":"T22_SENTINEL"}`),
			},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/forensics/0", nil)
	rec := httptest.NewRecorder()
	pm.ginEngine.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	var summary RequestForensicSummary
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &summary))
	assert.Equal(t, id, summary.ID)
	assert.Equal(t, "/v1/responses", summary.ReqPath)
	require.NotNil(t, summary.ChatRequest)
	assert.Equal(t, "none", summary.ChatRequest.ToolChoice)
	require.NotNil(t, summary.QwenStream)
	assert.Equal(t, 3, summary.QwenStream.FrameCount)
	assert.Equal(t, "recovered", summary.QwenStream.PreferredOrigins["question"])
	assert.Contains(t, summary.Notes, "normalized Qwen stream recovered question semantics from non-native output")
	foundQwenStage := false
	for _, stage := range summary.Stages {
		if stage.Name == "bridge.qwen_stream_normalization" {
			foundQwenStage = true
			require.NotNil(t, stage.QwenStream)
			assert.Equal(t, 3, stage.QwenStream.FrameCount)
			assert.Equal(t, 1, stage.QwenStream.RouteCounts["reasoning"])
			assert.Equal(t, "recovered", stage.QwenStream.PreferredOrigins["question"])
		}
	}
	assert.True(t, foundQwenStage)
	require.NotNil(t, summary.ResponsesOutput)
	assert.Equal(t, "completed", summary.ResponsesOutput.Status)
}

func TestSummarizeForensicChatResponse_StreamedChatCompletions(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"qwen-test\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think \"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \",\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"tool_calls\"}]}\n\n" +
			"data: [DONE]\n",
	)

	summary := summarizeForensicChatResponse(payload)
	require.NotNil(t, summary)
	assert.True(t, summary.IsStream)
	assert.Equal(t, 3, summary.ChunkCount)
	assert.True(t, summary.HasDoneMarker)
	assert.Equal(t, "qwen-test", summary.Model)
	assert.Equal(t, "tool_calls", summary.FinishReason)
	assert.Equal(t, "hello world", summary.ContentPreview)
	assert.Equal(t, "think", summary.ReasoningPreview)
	assert.True(t, summary.HasToolCalls)
	assert.Contains(t, summary.ToolCallNames, "shell")
	assert.Empty(t, summary.ErrorText)
}
