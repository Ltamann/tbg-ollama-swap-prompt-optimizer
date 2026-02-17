package proxy

import (
	"net/http"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

type ActivityPromptPreview struct {
	ID             int    `json:"id"`
	Timestamp      string `json:"timestamp"`
	Model          string `json:"model"`
	Kind           string `json:"kind"` // user_request | agent_step
	UserTurn       int    `json:"user_turn"`
	RequestPath    string `json:"request_path"`
	LastRole       string `json:"last_role"`
	LastUserPrompt string `json:"last_user_prompt"`
	PromptPreview  string `json:"prompt_preview"`
	MessageCount   int    `json:"message_count"`
	UserAgent      string `json:"user_agent"`
}

func (pm *ProxyManager) recordActivityPromptPreview(modelID, requestPath string, body []byte, headers http.Header) {
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return
	}

	lastRole := ""
	lastPreview := ""
	lastUserPrompt := ""
	messageCount := 0
	hasAssistantOrTool := false

	for _, msg := range messages.Array() {
		role := strings.TrimSpace(msg.Get("role").String())
		if role != "" {
			lastRole = role
		}
		text := strings.TrimSpace(extractMessageText(msg.Get("content")))
		if text != "" {
			lastPreview = text
		}
		if role == "assistant" || role == "tool" {
			hasAssistantOrTool = true
		}
		if role == "user" && text != "" {
			lastUserPrompt = text
		}
		messageCount++
	}

	if strings.TrimSpace(lastPreview) == "" && strings.TrimSpace(lastUserPrompt) == "" {
		return
	}

	userSignature := strings.TrimSpace(strings.ToLower(lastUserPrompt))

	pm.Lock()
	defer pm.Unlock()

	isNewUserTurn := false
	if userSignature != "" && userSignature != pm.activityCurrentUserSignature {
		pm.activityCurrentUserSignature = userSignature
		pm.activityCurrentTurn++
		pm.activityPromptPreviews = pm.activityPromptPreviews[:0]
		isNewUserTurn = true
	}
	if pm.activityCurrentTurn == 0 {
		pm.activityCurrentTurn = 1
	}

	kind := "agent_step"
	if isNewUserTurn {
		kind = "user_request"
	} else if userSignature != "" && !hasAssistantOrTool {
		kind = "user_request"
	}
	if userSignature != "" && userSignature == pm.activityCurrentUserSignature && !hasAssistantOrTool && lastRole == "user" {
		kind = "user_request"
	}

	pm.activityNextPromptID++
	pm.activityPromptPreviews = append(pm.activityPromptPreviews, ActivityPromptPreview{
		ID:             pm.activityNextPromptID,
		Timestamp:      time.Now().Format(time.RFC3339),
		Model:          strings.TrimSpace(modelID),
		Kind:           kind,
		UserTurn:       pm.activityCurrentTurn,
		RequestPath:    requestPath,
		LastRole:       lastRole,
		LastUserPrompt: lastUserPrompt,
		PromptPreview:  lastPreview,
		MessageCount:   messageCount,
		UserAgent:      trimPreview(strings.TrimSpace(headers.Get("User-Agent")), 180),
	})

	if len(pm.activityPromptPreviews) > 200 {
		pm.activityPromptPreviews = pm.activityPromptPreviews[len(pm.activityPromptPreviews)-200:]
	}
}

func (pm *ProxyManager) getActivityPromptPreviews() []ActivityPromptPreview {
	pm.Lock()
	defer pm.Unlock()
	out := make([]ActivityPromptPreview, len(pm.activityPromptPreviews))
	copy(out, pm.activityPromptPreviews)
	return out
}

func extractMessageText(content gjson.Result) string {
	if !content.Exists() {
		return ""
	}
	if content.Type == gjson.String {
		return content.String()
	}
	if content.IsArray() {
		parts := make([]string, 0, len(content.Array()))
		for _, part := range content.Array() {
			if strings.TrimSpace(part.Get("type").String()) == "text" {
				txt := strings.TrimSpace(part.Get("text").String())
				if txt != "" {
					parts = append(parts, txt)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func trimPreview(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return strings.TrimSpace(s[:max]) + " ..."
}
