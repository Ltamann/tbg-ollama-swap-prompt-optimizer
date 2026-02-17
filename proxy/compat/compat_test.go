package compat

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRoute(t *testing.T) {
	assert.Equal(t, EndpointResponses, Route("/v1/responses"))
	assert.Equal(t, EndpointChatCompletions, Route("/v1/chat/completions"))
	assert.Equal(t, EndpointCompletions, Route("/v1/completions"))
	assert.Equal(t, EndpointUnknown, Route("/v1/models"))
}

func TestNormalizeInferenceRequest(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/responses", nil)
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Accept", "")
	body := []byte(`{"model":"x","input":"hello","stream":false}`)

	out, err := NormalizeInferenceRequest(req, body)
	assert.NoError(t, err)
	assert.Equal(t, EndpointResponses, out.Endpoint)
	assert.Equal(t, "application/json", req.Header.Get("Content-Type"))
	assert.Equal(t, "application/json", req.Header.Get("Accept"))
	assert.Equal(t, "x", out.Canonical.Model)
	assert.Equal(t, "hello", out.Canonical.Input)
}

func TestCapabilities(t *testing.T) {
	reg := NewDefaultRegistry()

	err := reg.Validate(CanonicalRequest{
		Endpoint: EndpointCompletions,
		HasTools: true,
	})
	assert.Error(t, err)

	err = reg.Validate(CanonicalRequest{
		Endpoint: EndpointResponses,
		HasTools: true,
		Stream:   true,
	})
	assert.NoError(t, err)
}

func TestGoldenFixturesCanonical(t *testing.T) {
	chatPath := filepath.Join("..", "testdata", "openai_compat", "chat_completions_request.json")
	respPath := filepath.Join("..", "testdata", "openai_compat", "responses_request.json")

	chatBytes, err := os.ReadFile(chatPath)
	assert.NoError(t, err)
	respBytes, err := os.ReadFile(respPath)
	assert.NoError(t, err)

	chatCanonical := ToCanonical(EndpointChatCompletions, chatBytes)
	respCanonical := ToCanonical(EndpointResponses, respBytes)

	assert.Equal(t, chatCanonical.Model, respCanonical.Model)
	assert.Equal(t, chatCanonical.Input, respCanonical.Input)
}

func TestGoldenErrorEnvelope(t *testing.T) {
	path := filepath.Join("..", "testdata", "openai_compat", "error_envelope.json")
	raw, err := os.ReadFile(path)
	assert.NoError(t, err)

	var want ErrorEnvelope
	err = json.Unmarshal(raw, &want)
	assert.NoError(t, err)

	got := NewErrorEnvelope(400, "missing or invalid 'model' key", "")
	assert.Equal(t, want, got)
}

