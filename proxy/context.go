package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// TruncationMode defines how context overflow is handled
type TruncationMode string

const (
	// SlidingWindow automatically truncates old messages to fit context
	SlidingWindow TruncationMode = "sliding_window"
	// StrictError returns an error if context would overflow
	StrictError TruncationMode = "strict_error"
	// LowVRAM aggressively removes duplicate/redundant history before sliding window crop
	LowVRAM TruncationMode = "low_vram"
)

// DefaultSafetyMargin is the number of tokens to reserve as safety buffer
const DefaultSafetyMargin = 32
const DefaultReservedOutputTokens = 1024

// ContextManager handles context enforcement and message cropping
type ContextManager struct {
	modelID          string
	ctxSize          int
	safetyMargin     int
	truncationMode   TruncationMode
	proxyLogger      *LogMonitor
	upstreamProxyURL string
}

// NewContextManager creates a new context manager for a model
func NewContextManager(modelID string, ctxSize int, truncationMode TruncationMode, proxyLogger *LogMonitor, upstreamProxyURL string) *ContextManager {
	return &ContextManager{
		modelID:          modelID,
		ctxSize:          ctxSize,
		safetyMargin:     DefaultSafetyMargin,
		truncationMode:   truncationMode,
		proxyLogger:      proxyLogger,
		upstreamProxyURL: upstreamProxyURL,
	}
}

// ContextInfo contains context size and max tokens information
type ContextInfo struct {
	CtxSize            int
	SafePromptTokens   int
	RequestedMaxTokens int
}

// GetContextInfo computes safe context limits
func (cm *ContextManager) GetContextInfo(maxTokens int) ContextInfo {
	if cm.ctxSize <= 0 {
		return ContextInfo{
			CtxSize:            0,
			SafePromptTokens:   0,
			RequestedMaxTokens: maxTokens,
		}
	}

	if maxTokens <= 0 {
		maxTokens = cm.defaultReservedOutputTokens()
	}

	safePromptTokens := cm.ctxSize - maxTokens - cm.safetyMargin
	if safePromptTokens < 0 {
		safePromptTokens = 0
	}

	return ContextInfo{
		CtxSize:            cm.ctxSize,
		SafePromptTokens:   safePromptTokens,
		RequestedMaxTokens: maxTokens,
	}
}

// ChatMessage represents a single chat message
type ChatMessage struct {
	Role         string     `json:"role"`
	Content      string     `json:"content"`
	Name         string     `json:"name,omitempty"`
	FunctionName string     `json:"function_name,omitempty"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall represents a tool call in a message
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function being called
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatRequest represents an OpenAI-style chat completion request
type ChatRequest struct {
	Model            string        `json:"model"`
	Messages         []ChatMessage `json:"messages"`
	MaxTokens        int           `json:"max_tokens,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
	Tools            []ToolSchema  `json:"tools,omitempty"`
	ToolChoice       any           `json:"tool_choice,omitempty"`
	PresencePenalty  float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64       `json:"frequency_penalty,omitempty"`
	Temperature      float64       `json:"temperature,omitempty"`
}

// ToolSchema represents a tool definition
type ToolSchema struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function,omitempty"`
}

// FunctionDef defines a callable function
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// CropChatRequest truncates a chat request to fit within context limits
func (cm *ContextManager) CropChatRequest(originalReq ChatRequest) (*CropResult, error) {
	if cm.ctxSize <= 0 {
		return nil, fmt.Errorf("context size not configured for model %s", cm.modelID)
	}

	workingMessages := cloneMessages(originalReq.Messages)
	workingTools := cloneTools(originalReq.Tools)

	if cm.truncationMode == LowVRAM {
		workingMessages = cm.compactRepeatedMessages(workingMessages)
	}

	info := cm.GetContextInfo(originalReq.MaxTokens)
	if info.SafePromptTokens == 0 {
		return nil, fmt.Errorf("requested max_tokens (%d) exceeds available context", originalReq.MaxTokens)
	}

	totalTokens, err := cm.CountChatTokens(workingMessages, workingTools)
	if err != nil {
		return nil, fmt.Errorf("failed to count tokens: %w", err)
	}

	if totalTokens <= info.SafePromptTokens {
		return &CropResult{
			Messages:         workingMessages,
			Tools:            workingTools,
			OriginalMessages: originalReq.Messages,
			OriginalTools:    originalReq.Tools,
		}, nil
	}

	cm.proxyLogger.Debugf("<%s> Request exceeds context: %d > %d tokens (ctx_size=%d)",
		cm.modelID, totalTokens, info.SafePromptTokens, cm.ctxSize)

	switch cm.truncationMode {
	case StrictError:
		return nil, fmt.Errorf("prompt too long (%d tokens, max %d). Use truncation_mode: sliding_window to enable auto-cropping",
			totalTokens, info.SafePromptTokens)
	}

	croppedMessages, croppedTools := cm.applySlidingWindow(workingMessages, workingTools, info.SafePromptTokens)

	croppedTokens, err := cm.CountChatTokens(croppedMessages, croppedTools)
	if err != nil {
		return nil, fmt.Errorf("failed to count cropped tokens: %w", err)
	}

	cm.proxyLogger.Infof("[%s] Cropped prompt from %d -> %d tokens (ctx_size=%d)",
		cm.modelID, totalTokens, croppedTokens, cm.ctxSize)

	return &CropResult{
		Messages:         croppedMessages,
		Tools:            croppedTools,
		OriginalMessages: originalReq.Messages,
		OriginalTools:    originalReq.Tools,
	}, nil
}

func (cm *ContextManager) defaultReservedOutputTokens() int {
	if cm.ctxSize <= 0 {
		return DefaultReservedOutputTokens
	}

	quarter := cm.ctxSize / 4
	if quarter < 128 {
		quarter = 128
	}
	if quarter > DefaultReservedOutputTokens {
		return DefaultReservedOutputTokens
	}
	return quarter
}

// CountChatTokens counts tokens in chat messages and tools using llama.cpp endpoint
func (cm *ContextManager) CountChatTokens(messages []ChatMessage, tools []ToolSchema) (int, error) {
	if cm.upstreamProxyURL == "" {
		return 0, fmt.Errorf("upstream URL not configured for model %s", cm.modelID)
	}

	payload := map[string]any{
		"content": "",
	}

	textParts := make([]string, 0)

	for _, msg := range messages {
		if msg.Role == "system" || msg.Role == "user" || msg.Role == "assistant" || msg.Role == "tool" {
			if msg.Content != "" {
				textParts = append(textParts, fmt.Sprintf("[%s]: %s", strings.ToUpper(msg.Role), msg.Content))
			}
		}
	}

	if len(textParts) > 0 {
		payload["content"] = strings.Join(textParts, "\n\n")
	}

	if len(tools) > 0 {
		payload["tools"] = tools
	}

	reqBody, err := json.Marshal(payload)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal tokenization payload: %w", err)
	}

	tokenizeURL := strings.TrimSuffix(cm.upstreamProxyURL, "/") + "/tokenize"

	resp, err := http.Post(tokenizeURL, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		cm.proxyLogger.Warnf("<%s> Failed to use llama.cpp /tokenize endpoint: %v (fallback to approximate counting)",
			cm.modelID, err)
		return cm.estimateTokens(textParts), nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		cm.proxyLogger.Warnf("<%s> Failed to read tokenize response: %v", cm.modelID, err)
		return cm.estimateTokens(textParts), nil
	}

	var result struct {
		Tokens []int  `json:"tokens"`
		Count  int    `json:"count"`
		Error  string `json:"error"`
	}

	if json.Unmarshal(body, &result) == nil && result.Error == "" {
		if result.Count > 0 {
			return result.Count, nil
		}
		if len(result.Tokens) > 0 {
			return len(result.Tokens), nil
		}
	}

	cm.proxyLogger.Warnf("<%s> Tokenize endpoint returned unexpected response", cm.modelID)
	return cm.estimateTokens(textParts), nil
}

// estimateTokens provides a rough token count for when llama.cpp endpoint unavailable
func (cm *ContextManager) estimateTokens(textParts []string) int {
	total := 0
	for _, text := range textParts {
		total += len(strings.Fields(text)) * 13 / 10 // Rough approximation: ~1.3 tokens per word
	}
	return total + len(textParts) // Add separators
}

// applySlidingWindow implements the sliding window cropping strategy
func (cm *ContextManager) applySlidingWindow(messages []ChatMessage, tools []ToolSchema, maxTokens int) ([]ChatMessage, []ToolSchema) {
	if maxTokens <= 0 || len(messages) == 0 {
		return messages, tools
	}

	var result []ChatMessage

	for _, msg := range messages {
		result = append(result, msg)
	}

	totalTokens := cm.estimateMessagesTokens(result)

	for totalTokens > maxTokens && len(result) > 1 {
		result = cm.removeOldestNonSystemMessage(result)
		totalTokens = cm.estimateMessagesTokens(result)
	}

	if len(result) == 1 {
		msg := result[0]
		truncatedContent := cm.truncateContent(msg.Content, maxTokens)
		result[0] = ChatMessage{
			Role:         msg.Role,
			Content:      truncatedContent,
			Name:         msg.Name,
			FunctionName: msg.FunctionName,
			ToolCalls:    msg.ToolCalls,
		}
	}

	return result, tools
}

// removeOldestNonSystemMessage removes the oldest message that isn't a system message
func (cm *ContextManager) removeOldestNonSystemMessage(messages []ChatMessage) []ChatMessage {
	if len(messages) <= 1 {
		return messages
	}

	for i := 0; i < len(messages); i++ {
		if messages[i].Role != "system" {
			return append(messages[:i], messages[i+1:]...)
		}
	}

	return messages[1:]
}

// truncateContent truncates content to fit token limit (keeps most recent)
func (cm *ContextManager) truncateContent(content string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}

	lines := strings.Split(content, "\n")
	result := make([]string, 0)

	for i := len(lines) - 1; i >= 0; i-- {
		line := lines[i]
		lineTokens := cm.estimateLineTokens(line)

		if lineTokens <= maxTokens {
			result = append([]string{line}, result...)
			maxTokens -= lineTokens
		} else {
			break
		}
	}

	return strings.Join(result, "\n")
}

// estimateMessagesTokens estimates tokens in messages
func (cm *ContextManager) estimateMessagesTokens(messages []ChatMessage) int {
	total := 0
	for _, msg := range messages {
		total += cm.estimateLineTokens(msg.Content)
		if msg.Name != "" {
			total += len(strings.Fields(msg.Name)) * 13 / 10
		}
		if msg.FunctionName != "" {
			total += len(strings.Fields(msg.FunctionName)) * 13 / 10
		}
	}
	return total + len(messages)*2 // Add overhead for message structure
}

// estimateLineTokens estimates tokens in a single line of text
func (cm *ContextManager) estimateLineTokens(line string) int {
	if line == "" {
		return 0
	}
	return len(strings.Fields(line)) * 13 / 10 // Rough approximation: ~1.3 tokens per word
}

func cloneMessages(messages []ChatMessage) []ChatMessage {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]ChatMessage, 0, len(messages))
	for _, m := range messages {
		next := m
		if len(m.ToolCalls) > 0 {
			next.ToolCalls = append([]ToolCall(nil), m.ToolCalls...)
		}
		cloned = append(cloned, next)
	}
	return cloned
}

func cloneTools(tools []ToolSchema) []ToolSchema {
	if len(tools) == 0 {
		return nil
	}
	return append([]ToolSchema(nil), tools...)
}

func (cm *ContextManager) compactRepeatedMessages(messages []ChatMessage) []ChatMessage {
	if len(messages) <= 2 {
		return messages
	}

	result := make([]ChatMessage, 0, len(messages))
	seen := make(map[string]struct{}, len(messages))

	lastIndex := len(messages) - 1
	for i := lastIndex; i >= 0; i-- {
		msg := messages[i]

		if i == 0 && msg.Role == "system" {
			result = append(result, msg)
			continue
		}
		if i == lastIndex {
			signature := msg.Role + "|" + normalizeMessageContent(msg.Content)
			seen[signature] = struct{}{}
			result = append(result, msg)
			continue
		}

		signature := msg.Role + "|" + normalizeMessageContent(msg.Content)
		if _, found := seen[signature]; found {
			continue
		}

		seen[signature] = struct{}{}
		msg.Content = compactRepeatedLines(msg.Content)
		result = append(result, msg)
	}

	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result
}

// CompactMessagesForLowVRAM performs aggressive dedupe/compaction without ctx enforcement.
func CompactMessagesForLowVRAM(messages []ChatMessage) []ChatMessage {
	cm := &ContextManager{}
	return cm.compactRepeatedMessages(cloneMessages(messages))
}

func normalizeMessageContent(content string) string {
	content = strings.TrimSpace(content)
	if content == "" {
		return ""
	}
	return strings.Join(strings.Fields(content), " ")
}

func compactRepeatedLines(content string) string {
	content = strings.TrimSpace(content)
	if content == "" {
		return ""
	}

	lines := strings.Split(content, "\n")
	compacted := make([]string, 0, len(lines))
	lastLine := ""
	repeatCount := 0

	flush := func() {
		if lastLine == "" {
			return
		}
		compacted = append(compacted, lastLine)
		if repeatCount > 0 {
			compacted = append(compacted, fmt.Sprintf("[repeated %d more line(s) removed]", repeatCount))
		}
	}

	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if line == lastLine {
			repeatCount++
			continue
		}

		flush()
		lastLine = line
		repeatCount = 0
	}
	flush()

	return strings.Join(compacted, "\n")
}

// CropResult contains the cropped request data
type CropResult struct {
	Messages         []ChatMessage `json:"messages,omitempty"`
	Tools            []ToolSchema  `json:"tools,omitempty"`
	OriginalMessages []ChatMessage `json:"-"`
	OriginalTools    []ToolSchema  `json:"-"`
}

// IsCropped returns true if the request was actually cropped
func (cr CropResult) IsCropped() bool {
	if len(cr.Messages) == 0 || len(cr.OriginalMessages) == 0 {
		return false
	}
	if len(cr.Messages) < len(cr.OriginalMessages) {
		return true
	}
	if len(cr.Messages) != len(cr.OriginalMessages) {
		return false
	}
	for i := range cr.Messages {
		if cr.Messages[i].Role != cr.OriginalMessages[i].Role || cr.Messages[i].Content != cr.OriginalMessages[i].Content {
			return true
		}
	}
	return false
}
