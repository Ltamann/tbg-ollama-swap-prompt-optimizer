package proxy

import (
	"context"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
)

type monitorLimiter struct {
	max   uint32
	count atomic.Uint32
}

func newMonitorLimiter(max uint32) *monitorLimiter {
	if max == 0 {
		max = 5000
	}
	return &monitorLimiter{max: max}
}

func (l *monitorLimiter) allow() bool {
	if l == nil {
		return true
	}
	next := l.count.Add(1)
	return next <= l.max
}

func getTraceID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(proxyCtxKey("trace_id")).(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func getMonitorLimiter(ctx context.Context) *monitorLimiter {
	if ctx == nil {
		return nil
	}
	if v, ok := ctx.Value(proxyCtxKey("monitor_limiter")).(*monitorLimiter); ok {
		return v
	}
	return nil
}

func emitMonitor(ctx context.Context, modelID string, stage string, direction string, endpoint string, eventName string, data string, truncated bool) {
	traceID := getTraceID(ctx)
	if traceID == "" {
		return
	}
	limiter := getMonitorLimiter(ctx)
	if limiter != nil && !limiter.allow() {
		return
	}

	const maxData = 8 * 1024
	d := data
	trunc := truncated
	if len(d) > maxData {
		d = d[:maxData] + "...<truncated>"
		trunc = true
	}

	event.Emit(MonitorEvent{
		Event: LiveMonitorEvent{
			TraceID:   traceID,
			Timestamp: time.Now().Format(time.RFC3339Nano),
			Model:     modelID,
			Stage:     stage,
			Direction: direction,
			Endpoint:  endpoint,
			Event:     eventName,
			Data:      d,
			Truncated: trunc,
		},
	})
}
