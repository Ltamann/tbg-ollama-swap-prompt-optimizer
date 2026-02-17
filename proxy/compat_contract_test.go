package proxy

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
)

func TestCompatContract_StrictOpenAI_ErrorEnvelope(t *testing.T) {
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		CompatibilityMode:  "strict_openai",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	pm := New(cfg)
	defer pm.StopProcesses(StopImmediately)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Accept", "text/plain")
	w := CreateTestResponseRecorder()

	pm.ServeHTTP(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Header().Get("Content-Type"), "application/json")
	assert.Equal(t, "missing or invalid 'model' key", gjson.Get(w.Body.String(), "error.message").String())
	assert.Equal(t, "invalid_request_error", gjson.Get(w.Body.String(), "error.type").String())
}

func TestCompatContract_StrictOpenAI_CapabilityValidation(t *testing.T) {
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		CompatibilityMode:  "strict_openai",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	pm := New(cfg)
	defer pm.StopProcesses(StopImmediately)

	reqBody := `{"model":"model1","prompt":"hello","tools":[]}`
	req := httptest.NewRequest("POST", "/v1/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	pm.ServeHTTP(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, gjson.Get(w.Body.String(), "error.message").String(), "does not support tools")
	assert.Equal(t, "invalid_request_error", gjson.Get(w.Body.String(), "error.type").String())
}


