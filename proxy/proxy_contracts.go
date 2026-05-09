package proxy

import "strings"

type ToolLifecycleState string

const (
	ToolLifecycleCandidate      ToolLifecycleState = "candidate"
	ToolLifecycleRepaired       ToolLifecycleState = "repaired"
	ToolLifecycleValidated      ToolLifecycleState = "validated"
	ToolLifecycleRejected       ToolLifecycleState = "rejected"
	ToolLifecycleEmittable      ToolLifecycleState = "emittable"
	ToolLifecycleOutputReturned ToolLifecycleState = "output_returned"
)

type ContinuationState string

const (
	ContinuationStatePreTool                       ContinuationState = "pre_tool"
	ContinuationStateToolRunning                   ContinuationState = "tool_running"
	ContinuationStateToolCompletedAwaitingFollowup ContinuationState = "tool_completed_awaiting_followup"
	ContinuationStateFinalAnswerRequired           ContinuationState = "final_answer_required"
)

type ContinuationTurnPhase string

const (
	ContinuationTurnPhaseGeneral             ContinuationTurnPhase = "general"
	ContinuationTurnPhaseQuestion            ContinuationTurnPhase = "question"
	ContinuationTurnPhasePlanGather          ContinuationTurnPhase = "plan_gather"
	ContinuationTurnPhasePlanFinalize        ContinuationTurnPhase = "plan_finalize"
	ContinuationTurnPhaseResearch            ContinuationTurnPhase = "research"
	ContinuationTurnPhaseImplementationRetry ContinuationTurnPhase = "implementation_retry"
)

type StreamEventKind string

const (
	StreamEventAssistantMessageStarted StreamEventKind = "assistant_message_started"
	StreamEventAssistantMessageDelta   StreamEventKind = "assistant_message_delta"
	StreamEventAssistantMessageDone    StreamEventKind = "assistant_message_done"
	StreamEventReasoningDelta          StreamEventKind = "reasoning_delta"
	StreamEventReasoningDone           StreamEventKind = "reasoning_done"
	StreamEventToolItemAdded           StreamEventKind = "tool_item_added"
	StreamEventToolArgsDelta           StreamEventKind = "tool_args_delta"
	StreamEventToolArgsDone            StreamEventKind = "tool_args_done"
	StreamEventToolRejected            StreamEventKind = "tool_rejected"
	StreamEventToolOutputReturned      StreamEventKind = "tool_output_returned"
	StreamEventResponseCompleted       StreamEventKind = "response_completed"
)

type ToolCandidate struct {
	Name           string
	CallID         string
	Arguments      map[string]any
	LifecycleState ToolLifecycleState
	RepairNotes    []string
}

type ToolValidationResult struct {
	Valid          bool
	Warning        string
	LifecycleState ToolLifecycleState
	NormalizedItem map[string]any
	RepairNotes    []string
}

type ContinuationDecision struct {
	State            ContinuationState
	ForceToolChoice  any
	DisableTools     bool
	AllowedToolNames []string
	Instructions     []string
}

type LoopGuardDecision struct {
	Triggered       bool
	State           ContinuationState
	ForceToolChoice any
	DisableTools    bool
	Instructions    []string
}

type RetryLoopDecision struct {
	ReasonCode  string
	ForceStrict bool
	FailureText string
}

type ToolWorkflowState struct {
	HasToolOutput                    bool
	CompletedToolNames               []string
	PendingToolNames                 []string
	LatestCompletedToolName          string
	LatestCompletedToolFingerprint   string
	PreviousCompletedToolFingerprint string
	RepeatedLatestToolFingerprint    bool
	ApplyPatchSatisfied              bool
	VerificationExpected             bool
	VerificationCompleted            bool
	FinalAnswerSafe                  bool
}

type QwenStreamMode string

const (
	QwenStreamModeIdle        QwenStreamMode = "idle"
	QwenStreamModeThinking    QwenStreamMode = "thinking"
	QwenStreamModeToolCalling QwenStreamMode = "tool_calling"
	QwenStreamModeAnswering   QwenStreamMode = "answering"
	QwenStreamModeDone        QwenStreamMode = "done"
)

type NormalizedArtifactOrigin string

const (
	NormalizedArtifactOriginNative    NormalizedArtifactOrigin = "native"
	NormalizedArtifactOriginRecovered NormalizedArtifactOrigin = "recovered"
	NormalizedArtifactOriginSynthetic NormalizedArtifactOrigin = "synthetic"
)

type NormalizedArtifactKind string

const (
	NormalizedArtifactThinkingBlock NormalizedArtifactKind = "thinking_block"
	NormalizedArtifactTextBlock     NormalizedArtifactKind = "text_block"
	NormalizedArtifactToolBlock     NormalizedArtifactKind = "tool_block"
	NormalizedArtifactPlanBlock     NormalizedArtifactKind = "plan_block"
	NormalizedArtifactQuestionBlock NormalizedArtifactKind = "question_block"
	NormalizedArtifactMessageError  NormalizedArtifactKind = "message_error"
)

type MessageErrorKind string

const (
	MessageErrorUpstreamMalformedTurn  MessageErrorKind = "upstream_malformed_turn"
	MessageErrorPartialToolMissingArgs MessageErrorKind = "partial_tool_missing_args"
	MessageErrorReasoningOnlyTurn      MessageErrorKind = "reasoning_only_turn"
	MessageErrorTruncatedStream        MessageErrorKind = "truncated_stream"
	MessageErrorNativeControlMissing   MessageErrorKind = "native_control_missing"
)

type RawChunkRecord struct {
	Sequence int
	Payload  string
}

type ToolArgumentAccumulator struct {
	ToolIndex int
	ToolID    string
	ToolName  string
	Arguments strings.Builder
}

type AccumulatedChoiceState struct {
	ChoiceIndex       int
	Sequence          int
	RawChunks         []RawChunkRecord
	DominantMode      QwenStreamMode
	HasAnswerContent  bool
	HasToolDelta      bool
	HasReasoning      bool
	Truncated         bool
	FinishReason      string
	Content           strings.Builder
	Reasoning         strings.Builder
	ToolArgumentsByIx map[int]*ToolArgumentAccumulator
	ToolOrder         []int
}

type NormalizedArtifact struct {
	Kind         NormalizedArtifactKind
	Origin       NormalizedArtifactOrigin
	Sequence     int
	NativeSource string
	Payload      map[string]any
}

type NormalizedArtifactView struct {
	Artifacts         []NormalizedArtifact
	PreferredThinking *NormalizedArtifact
	PreferredText     *NormalizedArtifact
	PreferredPlan     *NormalizedArtifact
	PreferredQuestion *NormalizedArtifact
	PreferredError    *NormalizedArtifact
	ToolBlocks        []NormalizedArtifact
}

type MessageError struct {
	Kind        MessageErrorKind `json:"kind"`
	Recoverable bool             `json:"recoverable"`
	Details     string           `json:"details,omitempty"`
}

type ThinkingBlock struct {
	Text            string `json:"text"`
	WellFormed      bool   `json:"well_formed"`
	MalformedReason string `json:"malformed_reason,omitempty"`
}

type ToolBlock struct {
	ToolIndex   int            `json:"tool_index"`
	ToolName    string         `json:"tool_name"`
	ToolID      string         `json:"tool_id,omitempty"`
	RawArgs     string         `json:"raw_args,omitempty"`
	ParsedArgs  map[string]any `json:"parsed_args,omitempty"`
	ParseStatus string         `json:"parse_status,omitempty"`
}

type ContinuationContext struct {
	PlanModeRequested         bool
	PlanOutputRequested       bool
	ProxyPlanEnforcement      bool
	RequestUserInputAvailable bool
	NativeQuestionRequested   bool
	ImplementationRetryIntent bool
	SearchIntent              bool
	ExplorationFollowupIntent bool
	RequestedToolChoice       any
	ActiveToolChoice          any
	TurnPhase                 ContinuationTurnPhase
}

type StreamEvent struct {
	Kind         StreamEventKind
	ResponseID   string
	ItemID       string
	OutputIndex  int
	ContentIndex int
	ToolName     string
	CallID       string
	Status       string
	Payload      map[string]any
}

type QwenNormalizedStreamEventKind string

const (
	QwenNormalizedStreamEventMeta           QwenNormalizedStreamEventKind = "meta"
	QwenNormalizedStreamEventThinkingDelta  QwenNormalizedStreamEventKind = "thinking_delta"
	QwenNormalizedStreamEventAssistantDelta QwenNormalizedStreamEventKind = "assistant_delta"
	QwenNormalizedStreamEventToolCallDelta  QwenNormalizedStreamEventKind = "tool_call_delta"
	QwenNormalizedStreamEventFinish         QwenNormalizedStreamEventKind = "finish"
)

type QwenNormalizedStreamEvent struct {
	Kind           QwenNormalizedStreamEventKind
	ChunkID        string
	Model          string
	CreatedAt      int64
	FinishReason   string
	Text           string
	ToolIndex      int
	ToolID         string
	ToolName       string
	ArgumentsDelta string
	Usage          map[string]any
	Timings        map[string]any
}

type QwenNormalizedStreamFrame struct {
	Events             []QwenNormalizedStreamEvent
	ChunkID            string
	Model              string
	CreatedAt          int64
	FinishReason       string
	AssistantDeltaText string
	ReasoningDeltaText string
	ToolCallDeltas     []QwenNormalizedStreamEvent
	Usage              map[string]any
	Timings            map[string]any
}

type QwenStreamFrameRoute string

const (
	QwenStreamFrameRouteNone       QwenStreamFrameRoute = "none"
	QwenStreamFrameRouteReasoning  QwenStreamFrameRoute = "reasoning"
	QwenStreamFrameRouteCommentary QwenStreamFrameRoute = "commentary"
	QwenStreamFrameRouteFinal      QwenStreamFrameRoute = "final"
)

type QwenStreamFrameDecision struct {
	Route                   QwenStreamFrameRoute
	Text                    string
	HasAssistantDelta       bool
	HasReasoningDelta       bool
	ShouldFinalizeToolCalls bool
}

type QwenStreamFrameTraceSummary struct {
	Route         string   `json:"route,omitempty"`
	FinishReason  string   `json:"finish_reason,omitempty"`
	AssistantLen  int      `json:"assistant_len,omitempty"`
	ReasoningLen  int      `json:"reasoning_len,omitempty"`
	ToolCount     int      `json:"tool_count,omitempty"`
	ToolNames     []string `json:"tool_names,omitempty"`
	FinalizeTools bool     `json:"finalize_tools,omitempty"`
}

type StreamToolCallState struct {
	Index       int
	OutputIndex int
	ItemID      string
	CallID      string
	Name        string
	Exposed     bool
	ArgsEmitted bool
	ArgsBuilder strings.Builder
}

type ResponseToolItemView struct {
	Item                map[string]any
	ItemType            string
	ItemID              string
	CallID              string
	ToolName            string
	Arguments           string
	EmitsArgumentEvents bool
}

type ResponsesChatAdapter interface {
	TranslateResponsesToChatCompletionsRequest(body []byte) ([]byte, error)
	TranslateChatCompletionToResponsesResponse(body []byte, applyPatchPathHint string, applyPatchContentHint string, applyPatchTypeHint string) ([]byte, error)
}

type ReasoningTranslationAdapter interface {
	BuildCommentaryPreview(reasoning string) string
	NormalizeRequestedSummary(summary string) string
}

type ToolRepairAdapter interface {
	ParseAssistantOutput(modelName, content string) ([]ParsedToolCall, string)
	RecoverRequestUserInputArguments(reasoningText, text string) (string, bool)
	ValidateToolCallItem(item map[string]any) ToolValidationResult
}

type StreamReconstructionAdapter interface {
	CanonicalToolName(name string) string
	ShouldExposeToolCall(state *StreamToolCallState) bool
	CanonicalToolArguments(state *StreamToolCallState, arguments string) string
	BuildToolOutputItem(state *StreamToolCallState, arguments string) map[string]any
	BuildToolItemAddedEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool)
	BuildToolArgsDeltaEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool)
	BuildToolArgsDoneEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool)
	BuildToolItemDoneEvent(state *StreamToolCallState, responseID string) (StreamEvent, bool)
	NormalizeResponseOutputItem(item map[string]any) map[string]any
	BuildResponseToolItemView(item map[string]any) (ResponseToolItemView, bool)
	BuildResponseToolArgumentEvents(view ResponseToolItemView, responseID string, outputIndex int) []StreamEvent
	BuildResponseCompletedEvent(response map[string]any) (StreamEvent, bool)
}

type ContinuationController interface {
	DeriveAllowedToolNames(req map[string]any) []string
	BuildWorkflowState(req map[string]any) ToolWorkflowState
	BuildDecision(req map[string]any, ctx ContinuationContext) ContinuationDecision
}

type ProxyAdapterSet struct {
	Responses    ResponsesChatAdapter
	Reasoning    ReasoningTranslationAdapter
	ToolRepair   ToolRepairAdapter
	Stream       StreamReconstructionAdapter
	Continuation ContinuationController
}
