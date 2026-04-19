package proxy

import (
	"encoding/json"
	"testing"
)

func TestChatRequestUnmarshalStringContent(t *testing.T) {
	payload := []byte(`{
		"model": "gemma",
		"messages": [
			{"role": "user", "content": "Reply with OK"}
		]
	}`)

	var req ChatRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("unexpected unmarshal error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}
	if req.Messages[0].Content != "Reply with OK" {
		t.Fatalf("unexpected content: %q", req.Messages[0].Content)
	}
}

func TestChatRequestUnmarshalArrayContent(t *testing.T) {
	payload := []byte(`{
		"model": "gemma",
		"messages": [
			{"role": "user", "content": [{"type":"text","text":"Reply with OK"}]}
		]
	}`)

	var req ChatRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("unexpected unmarshal error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}
	if req.Messages[0].Content != "Reply with OK" {
		t.Fatalf("unexpected content: %q", req.Messages[0].Content)
	}
}

func TestChatRequestUnmarshalMixedContentParts(t *testing.T) {
	payload := []byte(`{
		"model": "gemma",
		"messages": [
			{"role": "user", "content": [
				{"type":"text","text":"First line"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}},
				{"type":"text","text":"Second line"}
			]}
		]
	}`)

	var req ChatRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("unexpected unmarshal error: %v", err)
	}

	if got, want := req.Messages[0].Content, "First line\nSecond line"; got != want {
		t.Fatalf("unexpected content: got %q want %q", got, want)
	}
}
