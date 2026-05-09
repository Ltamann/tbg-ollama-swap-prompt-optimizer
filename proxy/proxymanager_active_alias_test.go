package proxy

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
)

func TestProxyManager_ActiveModelAliasRoutesToCurrentModel(t *testing.T) {
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		UseActiveModelForAliases: []string{"gpt-5.4-mini"},
		LogLevel:                 "error",
	})

	proxy := New(cfg)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	// Establish an active model first.
	{
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d (%s)", http.StatusOK, w.Code, w.Body.String())
		}
		if body := w.Body.String(); !bytes.Contains([]byte(body), []byte("model1")) {
			t.Fatalf("expected response to contain %q, got %q", "model1", body)
		}
	}

	// Now request the dynamic alias; it should route to the active model (model1).
	{
		reqBody := `{"model":"gpt-5.4-mini"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d (%s)", http.StatusOK, w.Code, w.Body.String())
		}
		if body := w.Body.String(); !bytes.Contains([]byte(body), []byte("model1")) {
			t.Fatalf("expected response to contain %q, got %q", "model1", body)
		}
	}
}

func TestProxyManager_ActiveModelAliasWithoutActiveModelIsRejected(t *testing.T) {
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		UseActiveModelForAliases: []string{"gpt-5.4-mini"},
		LogLevel:                 "error",
	})

	proxy := New(cfg)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := `{"model":"gpt-5.4-mini"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d (%s)", http.StatusBadRequest, w.Code, w.Body.String())
	}
	want := fmt.Sprintf("model '%s' is configured as an active-model alias", "gpt-5.4-mini")
	if body := w.Body.String(); !bytes.Contains([]byte(body), []byte(want)) {
		t.Fatalf("expected response to contain %q, got %q", want, body)
	}
}
