package proxy

import (
	"context"
	"sync"
)

type captureStagesStore struct {
	mu     sync.Mutex
	stages []CaptureStage

	maxStageBytes int
}

func newCaptureStagesStore() *captureStagesStore {
	return &captureStagesStore{
		stages:        make([]CaptureStage, 0, 8),
		maxStageBytes: 512 * 1024, // per stage
	}
}

func (s *captureStagesStore) add(name string, payload []byte) {
	if s == nil || name == "" || len(payload) == 0 {
		return
	}
	if s.maxStageBytes > 0 && len(payload) > s.maxStageBytes {
		payload = payload[:s.maxStageBytes]
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stages = append(s.stages, CaptureStage{Name: name, Payload: append([]byte(nil), payload...)})
}

func (s *captureStagesStore) snapshot() []CaptureStage {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]CaptureStage, len(s.stages))
	copy(out, s.stages)
	return out
}

func addCaptureStage(ctx context.Context, name string, payload []byte) {
	if ctx == nil {
		return
	}
	store, ok := ctx.Value(proxyCtxKey("capture_stages")).(*captureStagesStore)
	if !ok || store == nil {
		return
	}
	store.add(name, payload)
}

func getCaptureStages(ctx context.Context) []CaptureStage {
	if ctx == nil {
		return nil
	}
	store, ok := ctx.Value(proxyCtxKey("capture_stages")).(*captureStagesStore)
	if !ok || store == nil {
		return nil
	}
	return store.snapshot()
}
