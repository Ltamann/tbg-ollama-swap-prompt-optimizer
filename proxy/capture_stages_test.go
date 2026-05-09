package proxy

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCaptureStagesStore_DeduplicatesConsecutiveIdenticalStages(t *testing.T) {
	store := newCaptureStagesStore()
	ctx := context.WithValue(context.Background(), proxyCtxKey("capture_stages"), store)

	addCaptureStage(ctx, "bridge.responses_request", []byte(`{"a":1}`))
	addCaptureStage(ctx, "bridge.responses_request", []byte(`{"a":1}`))
	addCaptureStage(ctx, "bridge.chat_completions_request", []byte(`{"b":2}`))
	addCaptureStage(ctx, "bridge.chat_completions_request", []byte(`{"b":2}`))

	stages := getCaptureStages(ctx)
	if assert.Len(t, stages, 2) {
		assert.Equal(t, "bridge.responses_request", stages[0].Name)
		assert.Equal(t, "bridge.chat_completions_request", stages[1].Name)
	}
}
