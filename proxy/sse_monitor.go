package proxy

import (
	"bytes"
	"context"
	"strings"

	"github.com/tidwall/gjson"
)

type sseMonitor struct {
	ctx       context.Context
	modelID   string
	endpoint  string
	stage     string
	direction string

	buf          []byte
	currentEvent string
}

func newSSEMonitor(ctx context.Context, modelID string, stage string, direction string, endpoint string) *sseMonitor {
	return &sseMonitor{
		ctx:       ctx,
		modelID:   modelID,
		endpoint:  endpoint,
		stage:     stage,
		direction: direction,
		buf:       make([]byte, 0, 4096),
	}
}

func (m *sseMonitor) writeChunk(chunk []byte) {
	if m == nil || len(chunk) == 0 {
		return
	}
	m.buf = append(m.buf, chunk...)

	for {
		idx := bytes.Index(m.buf, []byte("\n"))
		if idx == -1 {
			return
		}
		line := strings.TrimSpace(string(m.buf[:idx]))
		m.buf = m.buf[idx+1:]
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			m.currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		m.emitPayload(payload)
	}
}

func (m *sseMonitor) emitPayload(payload string) {
	if m == nil {
		return
	}
	ev := m.currentEvent
	if ev == "" {
		ev = "data"
	}

	if !gjson.Valid(payload) {
		emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, payload, false)
		return
	}

	parsed := gjson.Parse(payload)

	// Responses SSE: output text + reasoning summary + tool calls are most useful.
	if strings.HasPrefix(ev, "response.") {
		switch ev {
		case "response.output_text.delta":
			delta := strings.TrimSpace(parsed.Get("delta").String())
			if delta != "" {
				emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, delta, false)
			}
			return
		case "response.reasoning_summary_text.delta":
			delta := strings.TrimSpace(parsed.Get("delta").String())
			if delta != "" {
				emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, delta, false)
			}
			return
		case "response.output_item.added":
			item := parsed.Get("item")
			itemType := strings.TrimSpace(item.Get("type").String())
			toolName := strings.TrimSpace(item.Get("name").String())
			callID := strings.TrimSpace(item.Get("call_id").String())
			if strings.HasSuffix(itemType, "_call") || itemType == "function_call" {
				msg := strings.TrimSpace(toolName)
				if msg == "" {
					msg = itemType
				}
				if callID != "" {
					msg += " call_id=" + callID
				}
				emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, msg, false)
			}
			return
		case "response.completed":
			status := strings.TrimSpace(parsed.Get("response.status").String())
			if status == "" {
				status = strings.TrimSpace(parsed.Get("type").String())
			}
			emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, status, false)
			return
		}
	}

	// Chat completions SSE: extract delta content + reasoning_content.
	if choices := parsed.Get("choices"); choices.Exists() && len(choices.Array()) > 0 {
		delta := choices.Get("0.delta")
		if c := strings.TrimSpace(delta.Get("content").String()); c != "" {
			emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, "chat.delta.content", c, false)
		}
		if rc := strings.TrimSpace(delta.Get("reasoning_content").String()); rc != "" {
			emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, "chat.delta.reasoning", rc, false)
		}
		if fr := strings.TrimSpace(choices.Get("0.finish_reason").String()); fr != "" {
			emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, "chat.finish_reason", fr, false)
		}
		return
	}

	emitMonitor(m.ctx, m.modelID, m.stage, m.direction, m.endpoint, ev, payload, false)
}
