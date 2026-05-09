package proxy

import "testing"

func TestProxyCompatibilitySuite(t *testing.T) {
	t.Run("translation parity", func(t *testing.T) {
		t.Run("merges developer into leading system", TestTranslateResponsesToChatCompletionsRequest_MergesDeveloperIntoLeadingSystem)
		t.Run("reasoning effort maps controls", TestTranslateResponsesToChatCompletionsRequest_ReasoningEffortMapsReasoningControls)
		t.Run("preserves logit bias and grammar", TestTranslateResponsesToChatCompletionsRequest_PreservesLogitBiasAndGrammar)
		t.Run("slash mode and reasoning commands", TestTranslateResponsesToChatCompletionsRequest_SlashModeAndReasoningCommands)
	})

	t.Run("qwen xml repair parity", func(t *testing.T) {
		t.Run("parses qwen xml tool call envelope", TestParseQwenXMLToolCalls_ParsesFunctionAndParameters)
		t.Run("adapter uses qwen parser compatibility", TestDefaultToolRepairAdapter_UsesQwenParserCompatibility)
		t.Run("recovers request user input from reasoning", TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromReasoning)
		t.Run("recovers request user input from question line", TestTranslateChatCompletionToResponsesResponse_RecoversRequestUserInputArgsFromQuestionLine)
		t.Run("rejects empty parallel tool uses", TestTranslateChatCompletionToResponsesResponse_RejectsEmptyParallelToolUses)
	})

	t.Run("continuation parity", func(t *testing.T) {
		t.Run("forces request user input in plan continuation", TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputInPlanContinuationAfterToolOutput)
		t.Run("forces proposed plan after completed request user input", TestTranslateResponsesToChatCompletionsRequest_ForcesProposedPlanAfterCompletedRequestUserInput)
		t.Run("forces final answer after satisfied apply patch", TestTranslateResponsesToChatCompletionsRequest_ForcesFinalAnswerAfterSatisfiedApplyPatch)
		t.Run("does not re add apply patch for shell first prompt", TestTranslateResponsesToChatCompletionsRequest_DoesNotReAddApplyPatchForShellFirstPrompt)
	})

	t.Run("stream parity", func(t *testing.T) {
		t.Run("reasoning and content stay on separate lanes", TestWriteResponsesStreamFromChatSSE_EmitsReasoningAndContentOnSeparateLanes)
		t.Run("empty shell arguments become validation message", TestWriteResponsesStreamFromChatSSE_EmptyShellArgumentsBecomeValidationMessage)
		t.Run("tool first uses output index zero", TestWriteResponsesStreamFromChatSSE_ToolFirstUsesOutputIndexZeroWithoutEmptyMessage)
		t.Run("plan mode wraps proposed plan", TestWriteResponsesStreamFromChatSSE_PlanModeWrapsOutputAsProposedPlan)
		t.Run("preserves timings and backfills usage", TestWriteResponsesStreamFromChatSSE_PreservesTimingsAndBackfillsUsage)
	})

	t.Run("safety parity", func(t *testing.T) {
		t.Run("empty shell arguments stay non executable", TestWriteResponsesStream_EmptyShellArgumentsBecomeValidationMessage)
		t.Run("empty request user input arguments become validation message", TestWriteResponsesStream_EmptyRequestUserInputArgumentsBecomeValidationMessage)
		t.Run("upstream 502 does not synthesize tool completion", TestBuildResponsesBridgeHandler_DoesNotSynthesizeToolCompletionOnUpstream502)
		t.Run("retries invalid tool args in same cycle", TestBuildResponsesBridgeHandler_RetriesOnInvalidToolArgsInSameCycle)
	})

	t.Run("observability", func(t *testing.T) {
		t.Run("streamed native bridge captures all stages", TestBuildResponsesBridgeHandler_ForwardsNativeStreamWhenSafe_CapturesAllBridgeStages)
		t.Run("capture stages survive metrics wrapper", TestMetricsMonitor_WrapHandler_Capture)
	})
}
