package proxy

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/rand"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// TestResponseRecorder adds CloseNotify to httptest.ResponseRecorder.
// "If you want to write your own tests around streams you will need a Recorder that can handle CloseNotifier."
// The tests can panic otherwise:
// panic: interface conversion: *httptest.ResponseRecorder is not http.CloseNotifier: missing method CloseNotify
// See: https://github.com/gin-gonic/gin/issues/1815
// TestResponseRecorder is taken from gin's own tests: https://github.com/gin-gonic/gin/blob/ce20f107f5dc498ec7489d7739541a25dcd48463/context_test.go#L1747-L1765
type TestResponseRecorder struct {
	*httptest.ResponseRecorder
	closeChannel chan bool
}

func (r *TestResponseRecorder) CloseNotify() <-chan bool {
	return r.closeChannel
}

func CreateTestResponseRecorder() *TestResponseRecorder {
	return &TestResponseRecorder{
		httptest.NewRecorder(),
		make(chan bool, 1),
	}
}

func TestProxyManager_SwapProcessCorrectly(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	for _, modelName := range []string{"model1", "model2"} {
		reqBody := fmt.Sprintf(`{"model":"%s"}`, modelName)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), modelName)
	}
}
func TestProxyManager_SwapMultiProcess(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		LogLevel: "error",
		Groups: map[string]config.GroupConfig{
			"G1": {
				Swap:      true,
				Exclusive: false,
				Members:   []string{"model1"},
			},
			"G2": {
				Swap:      true,
				Exclusive: false,
				Members:   []string{"model2"},
			},
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	tests := []string{"model1", "model2"}
	for _, requestedModel := range tests {
		t.Run(requestedModel, func(t *testing.T) {
			reqBody := fmt.Sprintf(`{"model":"%s"}`, requestedModel)
			req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
			w := CreateTestResponseRecorder()

			proxy.ServeHTTP(w, req)
			assert.Equal(t, http.StatusOK, w.Code)
			assert.Contains(t, w.Body.String(), requestedModel)
		})
	}

	// make sure there's two loaded models
	assert.Equal(t, proxy.findGroupByModelName("model1").processes["model1"].CurrentState(), StateReady)
	assert.Equal(t, proxy.findGroupByModelName("model2").processes["model2"].CurrentState(), StateReady)
}

// Test that a persistent group is not affected by the swapping behaviour of
// other groups.
func TestProxyManager_PersistentGroupsAreNotSwapped(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"), // goes into the default group
			"model2": getTestSimpleResponderConfig("model2"),
		},
		LogLevel: "error",
		Groups: map[string]config.GroupConfig{
			// the forever group is persistent and should not be affected by model1
			"forever": {
				Swap:       true,
				Exclusive:  false,
				Persistent: true,
				Members:    []string{"model2"},
			},
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	// make requests to load all models, loading model1 should not affect model2
	tests := []string{"model2", "model1"}
	for _, requestedModel := range tests {
		reqBody := fmt.Sprintf(`{"model":"%s"}`, requestedModel)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), requestedModel)
	}

	assert.Equal(t, proxy.findGroupByModelName("model2").processes["model2"].CurrentState(), StateReady)
	assert.Equal(t, proxy.findGroupByModelName("model1").processes["model1"].CurrentState(), StateReady)
}

func TestProxyManager_WebSearchSettingsAPI(t *testing.T) {
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(cfg)
	defer proxy.StopProcesses(StopImmediately)

	req := httptest.NewRequest("POST", "/api/settings/web-search", bytes.NewBufferString(`{
		"enabled": true,
		"engine": "searxng",
		"url": "http://127.0.0.1:18080/search",
		"managedEnabled": false,
		"managedCommand": "sh -lc \"sleep 5\"",
		"managedStopCommand": ""
	}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "searxng", gjson.Get(w.Body.String(), "engine").String())
	assert.Equal(t, "http://127.0.0.1:18080/search", gjson.Get(w.Body.String(), "url").String())
	assert.Equal(t, false, gjson.Get(w.Body.String(), "managedEnabled").Bool())

	getReq := httptest.NewRequest("GET", "/api/settings/web-search", nil)
	getW := httptest.NewRecorder()
	proxy.ServeHTTP(getW, getReq)
	assert.Equal(t, http.StatusOK, getW.Code)
	assert.Equal(t, "searxng", gjson.Get(getW.Body.String(), "engine").String())
	assert.Equal(t, "http://127.0.0.1:18080/search", gjson.Get(getW.Body.String(), "url").String())
}

func TestRequestMapContainsNamedToolOutput(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{"type": "function_call", "call_id": "call_wait_1", "name": "wait_agent"},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": "done"},
		},
	}
	assert.True(t, requestMapContainsNamedToolOutput(req, "wait_agent"))
	assert.False(t, requestMapContainsNamedToolOutput(req, "spawn_agent"))
}

func TestBridgeResponsesMaxAttempts_AgentContinuationIsShorter(t *testing.T) {
	body := []byte(`{
		"input": [
			{"type":"function_call","call_id":"call_wait_1","name":"wait_agent"},
			{"type":"function_call_output","call_id":"call_wait_1","output":"done"}
		]
	}`)
	assert.Equal(t, 2, bridgeResponsesMaxAttempts(body, false))
	assert.Equal(t, 2, bridgeResponsesMaxAttempts(body, true))
}

func TestBridgeResponsesAttemptTimeout_AgentContinuationIsShorter(t *testing.T) {
	body := []byte(`{
		"input": [
			{"type":"function_call","call_id":"call_wait_1","name":"wait_agent"},
			{"type":"function_call_output","call_id":"call_wait_1","output":"done"}
		]
	}`)
	assert.Equal(t, 45*time.Second, bridgeResponsesAttemptTimeout(body, false))
}

func TestTranslateResponsesToChatCompletionsRequest_ExactlyOneSubagentAfterWaitForcesApplyPatchToolChoice(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.4",
		"tools":[
			{"type":"function","function":{"name":"spawn_agent"}},
			{"type":"function","function":{"name":"wait_agent"}},
			{"type":"function","function":{"name":"apply_patch"}}
		],
		"input":[
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"Use exactly one subagent via spawn_agent with model gpt-5.4, wait for that agent, then use apply_patch to create src/agent-report.md."}
			]},
			{"type":"function_call","name":"spawn_agent","call_id":"call_spawn_1","arguments":"{\"model\":\"gpt-5.4\"}"},
			{"type":"function_call_output","call_id":"call_spawn_1","output":"spawned"},
			{"type":"function_call","name":"wait_agent","call_id":"call_wait_1","arguments":"{\"target\":\"agent_1\"}"},
			{"type":"function_call_output","call_id":"call_wait_1","output":"completed"}
		]
	}`)
	out, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "function", gjson.GetBytes(out, "tool_choice.type").String())
	assert.Equal(t, "apply_patch", gjson.GetBytes(out, "tool_choice.function.name").String())
	assert.Equal(t, "apply_patch", gjson.GetBytes(out, "tools.0.function.name").String())
	assert.Len(t, gjson.GetBytes(out, "tools").Array(), 1)
}

// When a request for a different model comes in ProxyManager should wait until
// the first request is complete before swapping. Both requests should complete
func TestProxyManager_SwapMultiProcessParallelRequests(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow test")
	}

	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
			"model3": getTestSimpleResponderConfig("model3"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	results := map[string]string{}

	var wg sync.WaitGroup
	var mu sync.Mutex

	for key := range config.Models {
		wg.Add(1)
		go func(key string) {
			defer wg.Done()

			reqBody := fmt.Sprintf(`{"model":"%s"}`, key)
			req := httptest.NewRequest("POST", "/v1/chat/completions?wait=1000ms", bytes.NewBufferString(reqBody))
			w := CreateTestResponseRecorder()

			proxy.ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				t.Errorf("Expected status OK, got %d for key %s", w.Code, key)
			}

			mu.Lock()
			var response map[string]interface{}
			assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
			result, ok := response["responseMessage"].(string)
			assert.Equal(t, ok, true)
			results[key] = result
			mu.Unlock()
		}(key)

		<-time.After(time.Millisecond)
	}

	wg.Wait()
	assert.Len(t, results, len(config.Models))

	for key, result := range results {
		assert.Equal(t, key, result)
	}
}

func TestProxyManager_ListModelsHandler(t *testing.T) {

	model1Config := getTestSimpleResponderConfig("model1")
	model1Config.Name = "Model 1"
	model1Config.Description = "Model 1 description is used for testing"

	model2Config := getTestSimpleResponderConfig("model2")
	model2Config.Name = "     " // empty whitespace only strings will get ignored
	model2Config.Description = "  "

	cfg := config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": model1Config,
			"model2": model2Config,
			"model3": getTestSimpleResponderConfig("model3"),
		},
		Peers: map[string]config.PeerConfig{
			"peer1": {
				Proxy:  "http://peer1:8080",
				Models: []string{"peer-model-a", "peer-model-b"},
			},
		},
		LogLevel: "error",
	}

	proxy := New(cfg)

	// Create a test request
	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Add("Origin", "i-am-the-origin")
	w := CreateTestResponseRecorder()

	// Call the listModelsHandler
	proxy.ServeHTTP(w, req)

	// Check the response status code
	assert.Equal(t, http.StatusOK, w.Code)

	// Check for Access-Control-Allow-Origin
	assert.Equal(t, req.Header.Get("Origin"), w.Result().Header.Get("Access-Control-Allow-Origin"))

	// Parse the JSON response
	var response struct {
		Data []map[string]interface{} `json:"data"`
	}

	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse JSON response: %v", err)
	}

	// Check the number of models returned (3 local + 2 peer models)
	assert.Len(t, response.Data, 5)

	// Check the details of each model
	expectedModels := map[string]struct{}{
		"model1":       {},
		"model2":       {},
		"model3":       {},
		"peer-model-a": {},
		"peer-model-b": {},
	}

	// make all models
	for _, model := range response.Data {
		modelID, ok := model["id"].(string)
		assert.True(t, ok, "model ID should be a string")
		_, exists := expectedModels[modelID]
		assert.True(t, exists, "unexpected model ID: %s", modelID)
		delete(expectedModels, modelID)

		object, ok := model["object"].(string)
		assert.True(t, ok, "object should be a string")
		assert.Equal(t, "model", object)

		created, ok := model["created"].(float64)
		assert.True(t, ok, "created should be a number")
		assert.Greater(t, created, float64(0)) // Assuming the timestamp is positive

		ownedBy, ok := model["owned_by"].(string)
		assert.True(t, ok, "owned_by should be a string")
		assert.Equal(t, "llama-swap", ownedBy)

		// check for optional name and description
		if modelID == "model1" {
			name, ok := model["name"].(string)
			assert.True(t, ok, "name should be a string")
			assert.Equal(t, "Model 1", name)
			description, ok := model["description"].(string)
			assert.True(t, ok, "description should be a string")
			assert.Equal(t, "Model 1 description is used for testing", description)
		} else if modelID == "peer-model-a" || modelID == "peer-model-b" {
			// Peer models should have meta.llamaswap.peerID
			meta, exists := model["meta"]
			assert.True(t, exists, "peer model should have meta field")
			metaMap, ok := meta.(map[string]interface{})
			assert.True(t, ok, "meta should be a map")
			llamaswap, exists := metaMap["llamaswap"]
			assert.True(t, exists, "meta should have llamaswap field")
			llamaswapMap, ok := llamaswap.(map[string]interface{})
			assert.True(t, ok, "llamaswap should be a map")
			peerID, exists := llamaswapMap["peerID"]
			assert.True(t, exists, "llamaswap should have peerID field")
			assert.Equal(t, "peer1", peerID)
		} else {
			_, exists := model["name"]
			assert.False(t, exists, "unexpected name field for model: %s", modelID)
			_, exists = model["description"]
			assert.False(t, exists, "unexpected description field for model: %s", modelID)
		}
	}

	// Ensure all expected models were returned
	assert.Empty(t, expectedModels, "not all expected models were returned")
}

func TestProxyManager_ListModelsHandler_WithMetadata(t *testing.T) {
	// Process config through LoadConfigFromReader to apply macro substitution
	configYaml := `
healthCheckTimeout: 15
logLevel: error
startPort: 10000
models:
  model1:
    cmd: /path/to/server -p ${PORT}
    macros:
      PORT_NUM: 10001
      TEMP: 0.7
      NAME: "llama"
    metadata:
      port: ${PORT_NUM}
      temperature: ${TEMP}
      enabled: true
      note: "Running on port ${PORT_NUM}"
      nested:
        value: ${TEMP}
  model2:
    cmd: /path/to/server -p ${PORT}
`
	processedConfig, err := config.LoadConfigFromReader(strings.NewReader(configYaml))
	assert.NoError(t, err)

	proxy := New(processedConfig)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response struct {
		Data []map[string]any `json:"data"`
	}

	err = json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Len(t, response.Data, 2)

	// Find model1 and model2 in response
	var model1Data, model2Data map[string]any
	for _, model := range response.Data {
		if model["id"] == "model1" {
			model1Data = model
		} else if model["id"] == "model2" {
			model2Data = model
		}
	}

	// Verify model1 has llamaswap_meta
	assert.NotNil(t, model1Data)
	meta, exists := model1Data["meta"]
	if !assert.True(t, exists, "model1 should have meta key") {
		t.FailNow()
	}

	metaMap := meta.(map[string]any)

	lsmeta, exists := metaMap["llamaswap"]
	if !assert.True(t, exists, "model1 should have meta.llamaswap key") {
		t.FailNow()
	}

	lsmetamap := lsmeta.(map[string]any)

	// Verify type preservation
	assert.Equal(t, float64(10001), lsmetamap["port"]) // JSON numbers are float64
	assert.Equal(t, 0.7, lsmetamap["temperature"])
	assert.Equal(t, true, lsmetamap["enabled"])
	// Verify string interpolation
	assert.Equal(t, "Running on port 10001", lsmetamap["note"])
	// Verify nested structure
	nested := lsmetamap["nested"].(map[string]any)
	assert.Equal(t, 0.7, nested["value"])

	// Verify model2 does NOT have llamaswap_meta
	assert.NotNil(t, model2Data)
	_, exists = model2Data["llamaswap_meta"]
	assert.False(t, exists, "model2 should not have llamaswap_meta")
}

func TestProxyManager_ListModelsHandler_SortedByID(t *testing.T) {
	// Intentionally add models in non-sorted order and with an unlisted model
	config := config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"zeta":  getTestSimpleResponderConfig("zeta"),
			"alpha": getTestSimpleResponderConfig("alpha"),
			"beta":  getTestSimpleResponderConfig("beta"),
			"hidden": func() config.ModelConfig {
				mc := getTestSimpleResponderConfig("hidden")
				mc.Unlisted = true
				return mc
			}(),
		},
		LogLevel: "error",
	}

	proxy := New(config)

	// Request models list
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response struct {
		Data []map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse JSON response: %v", err)
	}

	// We expect only the listed models in sorted order by id
	expectedOrder := []string{"alpha", "beta", "zeta"}
	if assert.Len(t, response.Data, len(expectedOrder), "unexpected number of listed models") {
		got := make([]string, 0, len(response.Data))
		for _, m := range response.Data {
			id, _ := m["id"].(string)
			got = append(got, id)
		}
		assert.Equal(t, expectedOrder, got, "models should be sorted by id ascending")
	}
}

func TestProxyManager_ListModelsHandler_IncludeAliasesInList(t *testing.T) {
	// Configure alias
	config := config.Config{
		HealthCheckTimeout:   15,
		IncludeAliasesInList: true,
		Models: map[string]config.ModelConfig{
			"model1": func() config.ModelConfig {
				mc := getTestSimpleResponderConfig("model1")
				mc.Name = "Model 1"
				mc.Aliases = []string{"alias1"}
				return mc
			}(),
		},
		LogLevel: "error",
	}

	proxy := New(config)

	// Request models list
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response struct {
		Data []map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse JSON response: %v", err)
	}

	// We expect both base id and alias
	var model1Data, alias1Data map[string]any
	for _, model := range response.Data {
		if model["id"] == "model1" {
			model1Data = model
		} else if model["id"] == "alias1" {
			alias1Data = model
		}
	}

	// Verify model1 has name
	assert.NotNil(t, model1Data)
	_, exists := model1Data["name"]
	if !assert.True(t, exists, "model1 should have name key") {
		t.FailNow()
	}
	name1, ok := model1Data["name"].(string)
	assert.True(t, ok, "name1 should be a string")

	// Verify alias1 has name
	assert.NotNil(t, alias1Data)
	_, exists = alias1Data["name"]
	if !assert.True(t, exists, "alias1 should have name key") {
		t.FailNow()
	}
	name2, ok := alias1Data["name"].(string)
	assert.True(t, ok, "name2 should be a string")

	// Name keys should match
	assert.Equal(t, name1, name2)
}

func TestProxyManager_Shutdown(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow test")
	}

	// make broken model configurations
	model1Config := getTestSimpleResponderConfigPort("model1", 9991)
	model1Config.Proxy = "http://localhost:10001/"

	model2Config := getTestSimpleResponderConfigPort("model2", 9992)
	model2Config.Proxy = "http://localhost:10002/"

	model3Config := getTestSimpleResponderConfigPort("model3", 9993)
	model3Config.Proxy = "http://localhost:10003/"

	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": model1Config,
			"model2": model2Config,
			"model3": model3Config,
		},
		LogLevel: "error",
		Groups: map[string]config.GroupConfig{
			"test": {
				Swap:    false,
				Members: []string{"model1", "model2", "model3"},
			},
		},
	})

	proxy := New(config)

	// Start all the processes
	var wg sync.WaitGroup
	for _, modelName := range []string{"model1", "model2", "model3"} {
		wg.Add(1)
		go func(modelName string) {
			defer wg.Done()
			reqBody := fmt.Sprintf(`{"model":"%s"}`, modelName)
			req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
			w := CreateTestResponseRecorder()

			// send a request to trigger the proxy to load ... this should hang waiting for start up
			proxy.ServeHTTP(w, req)
			assert.Equal(t, http.StatusBadGateway, w.Code)
			assert.Contains(t, w.Body.String(), "health check interrupted due to shutdown")
		}(modelName)
	}

	go func() {
		<-time.After(time.Second)
		proxy.Shutdown()
	}()
	wg.Wait()
}

func TestProxyManager_Unload(t *testing.T) {
	conf := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(conf)
	reqBody := fmt.Sprintf(`{"model":"%s"}`, "model1")
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)

	assert.Equal(t, proxy.processGroups[config.DEFAULT_GROUP_ID].processes["model1"].CurrentState(), StateReady)
	req = httptest.NewRequest("GET", "/unload", nil)
	w = CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, w.Body.String(), "OK")

	select {
	case <-proxy.processGroups[config.DEFAULT_GROUP_ID].processes["model1"].cmdWaitChan:
		// good
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for model1 to stop")
	}
	assert.Equal(t, proxy.processGroups[config.DEFAULT_GROUP_ID].processes["model1"].CurrentState(), StateStopped)
}

func TestProxyManager_UnloadSingleModel(t *testing.T) {
	const testGroupId = "testGroup"
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		Groups: map[string]config.GroupConfig{
			testGroupId: {
				Swap:    false,
				Members: []string{"model1", "model2"},
			},
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopImmediately)

	// start both model
	for _, modelName := range []string{"model1", "model2"} {
		reqBody := fmt.Sprintf(`{"model":"%s"}`, modelName)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
	}

	assert.Equal(t, StateReady, proxy.processGroups[testGroupId].processes["model1"].CurrentState())
	assert.Equal(t, StateReady, proxy.processGroups[testGroupId].processes["model2"].CurrentState())

	req := httptest.NewRequest("POST", "/api/models/unload/model1", nil)
	w := CreateTestResponseRecorder()
	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	if !assert.Equal(t, w.Body.String(), "OK") {
		t.FailNow()
	}

	select {
	case <-proxy.processGroups[testGroupId].processes["model1"].cmdWaitChan:
		// good
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for model1 to stop")
	}

	assert.Equal(t, proxy.processGroups[testGroupId].processes["model1"].CurrentState(), StateStopped)
	assert.Equal(t, proxy.processGroups[testGroupId].processes["model2"].CurrentState(), StateReady)
}

// Test issue #61 `Listing the current list of models and the loaded model.`
func TestProxyManager_RunningEndpoint(t *testing.T) {
	// Shared configuration
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		LogLevel: "warn",
	})

	// Define a helper struct to parse the JSON response.
	type RunningResponse struct {
		Running []struct {
			Model       string `json:"model"`
			State       string `json:"state"`
			Cmd         string `json:"cmd"`
			Proxy       string `json:"proxy"`
			TTL         int    `json:"ttl"`
			Name        string `json:"name"`
			Description string `json:"description"`
		} `json:"running"`
	}

	// Create proxy once for all tests
	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	t.Run("no models loaded", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/running", nil)
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var response RunningResponse

		// Check if this is a valid JSON object.
		assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

		// We should have an empty running array here.
		assert.Empty(t, response.Running, "expected no running models")
	})

	t.Run("single model loaded", func(t *testing.T) {
		// Load just a model.
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)

		// Simulate browser call for the `/running` endpoint.
		req = httptest.NewRequest("GET", "/running", nil)
		w = CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)

		var response RunningResponse
		assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

		// Check if we have a single array element.
		assert.Len(t, response.Running, 1)

		// Is this the right model?
		assert.Equal(t, "model1", response.Running[0].Model)

		// Is the model loaded?
		assert.Equal(t, "ready", response.Running[0].State)

		// Verify extended fields are present
		assert.NotEmpty(t, response.Running[0].Cmd, "cmd should be populated")
		assert.NotEmpty(t, response.Running[0].Proxy, "proxy should be populated")
		assert.Equal(t, 0, response.Running[0].TTL, "ttl should default to 0")
	})
}

func TestProxyManager_AudioTranscriptionHandler(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"TheExpectedModel": getTestSimpleResponderConfig("TheExpectedModel"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	// Create a buffer with multipart form data
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add the model field
	fw, err := w.CreateFormField("model")
	assert.NoError(t, err)
	_, err = fw.Write([]byte("TheExpectedModel"))
	assert.NoError(t, err)

	// Add a file field
	fw, err = w.CreateFormFile("file", "test.mp3")
	assert.NoError(t, err)
	// Generate random content length between 10 and 20
	contentLength := rand.Intn(11) + 10 // 10 to 20
	content := make([]byte, contentLength)
	_, err = fw.Write(content)
	assert.NoError(t, err)
	w.Close()

	// Create the request with the multipart form data
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &b)
	req.Header.Set("Content-Type", w.FormDataContentType())
	rec := CreateTestResponseRecorder()
	proxy.ServeHTTP(rec, req)

	// Verify the response
	assert.Equal(t, http.StatusOK, rec.Code)
	var response map[string]string
	err = json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "TheExpectedModel", response["model"])
	assert.Equal(t, response["text"], fmt.Sprintf("The length of the file is %d bytes", contentLength)) // matches simple-responder
	assert.Equal(t, strconv.Itoa(370+contentLength), response["h_content_length"])
}

// Test useModelName in configuration sends overrides what is sent to upstream
func TestProxyManager_UseModelName(t *testing.T) {
	upstreamModelName := "upstreamModel"
	modelConfig := getTestSimpleResponderConfig(upstreamModelName)
	modelConfig.UseModelName = upstreamModelName

	conf := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": modelConfig,
		},
		LogLevel: "error",
	})

	proxy := New(conf)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	requestedModel := "model1"

	t.Run("useModelName over rides requested model: /v1/chat/completions", func(t *testing.T) {
		reqBody := fmt.Sprintf(`{"model":"%s"}`, requestedModel)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), upstreamModelName)

		// make sure the content length was set correctly
		// simple-responder will return the content length it got in the response
		body := w.Body.Bytes()
		contentLength := int(gjson.GetBytes(body, "h_content_length").Int())
		assert.Equal(t, len(fmt.Sprintf(`{"model":"%s"}`, upstreamModelName)), contentLength)
	})

	t.Run("useModelName over rides requested model: /v1/audio/transcriptions", func(t *testing.T) {
		// Create a buffer with multipart form data
		var b bytes.Buffer
		w := multipart.NewWriter(&b)

		// Add the model field
		fw, err := w.CreateFormField("model")
		assert.NoError(t, err)
		_, err = fw.Write([]byte(requestedModel))
		assert.NoError(t, err)

		// Add a file field
		fw, err = w.CreateFormFile("file", "test.mp3")
		assert.NoError(t, err)
		_, err = fw.Write([]byte("test"))
		assert.NoError(t, err)
		w.Close()

		// Create the request with the multipart form data
		req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &b)
		req.Header.Set("Content-Type", w.FormDataContentType())
		rec := CreateTestResponseRecorder()
		proxy.ServeHTTP(rec, req)

		// Verify the response
		assert.Equal(t, http.StatusOK, rec.Code)
		var response map[string]string
		err = json.Unmarshal(rec.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, upstreamModelName, response["model"])
	})
}

func TestProxyManager_AudioVoicesGETHandler(t *testing.T) {
	conf := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(conf)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	t.Run("successful GET with model query param", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/v1/audio/voices?model=model1", nil)
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "voice1")
	})

	t.Run("missing model query param returns 400", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/v1/audio/voices", nil)
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "missing required 'model' query parameter")
	})

	t.Run("unknown model returns 400", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/v1/audio/voices?model=nonexistent", nil)
		w := CreateTestResponseRecorder()
		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "could not find suitable handler")
	})
}

func TestProxyManager_CORSOptionsHandler(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	tests := []struct {
		name            string
		method          string
		requestHeaders  map[string]string
		expectedStatus  int
		expectedHeaders map[string]string
	}{
		{
			name:           "OPTIONS with no headers",
			method:         "OPTIONS",
			expectedStatus: http.StatusNoContent,
			expectedHeaders: map[string]string{
				"Access-Control-Allow-Origin":  "*",
				"Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
				"Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, X-Requested-With",
			},
		},
		{
			name:   "OPTIONS with specific headers",
			method: "OPTIONS",
			requestHeaders: map[string]string{
				"Access-Control-Request-Headers": "X-Custom-Header, Some-Other-Header",
			},
			expectedStatus: http.StatusNoContent,
			expectedHeaders: map[string]string{
				"Access-Control-Allow-Origin":  "*",
				"Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
				"Access-Control-Allow-Headers": "X-Custom-Header, Some-Other-Header",
			},
		},
		{
			name:           "Non-OPTIONS request",
			method:         "GET",
			expectedStatus: http.StatusNotFound, // Since we don't have a GET route defined
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxy := New(config)
			defer proxy.StopProcesses(StopWaitForInflightRequest)

			req := httptest.NewRequest(tt.method, "/v1/chat/completions", nil)
			for k, v := range tt.requestHeaders {
				req.Header.Set(k, v)
			}

			w := CreateTestResponseRecorder()
			proxy.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			for header, expectedValue := range tt.expectedHeaders {
				assert.Equal(t, expectedValue, w.Header().Get(header))
			}
		})
	}
}

func TestProxyManager_Upstream(t *testing.T) {
	configStr := fmt.Sprintf(`
logLevel: error
models:
  model1:
    cmd: %s -port ${PORT} -silent -respond model1
    aliases: [model-alias]
`, getSimpleResponderPath())

	config, err := config.LoadConfigFromReader(strings.NewReader(configStr))
	assert.NoError(t, err)

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)
	t.Run("main model name", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/upstream/model1/test", nil)
		rec := CreateTestResponseRecorder()
		proxy.ServeHTTP(rec, req)
		assert.Equal(t, http.StatusOK, rec.Code)
		assert.Equal(t, "model1", rec.Body.String())
	})

	t.Run("model alias", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/upstream/model-alias/test", nil)
		rec := CreateTestResponseRecorder()
		proxy.ServeHTTP(rec, req)
		assert.Equal(t, http.StatusOK, rec.Code)
		assert.Equal(t, "model1", rec.Body.String())
	})
}

func TestProxyManager_ChatContentLength(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := fmt.Sprintf(`{"model":"%s", "x": "this is just some content to push the length out a bit"}`, "model1")
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	var response map[string]interface{}
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, "81", response["h_content_length"])
	assert.Equal(t, "model1", response["responseMessage"])
}

func TestProxyManager_FiltersStripParams(t *testing.T) {
	modelConfig := getTestSimpleResponderConfig("model1")
	modelConfig.Filters = config.ModelFilters{
		Filters: config.Filters{
			StripParams: "temperature, model, stream",
		},
	}

	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		Models: map[string]config.ModelConfig{
			"model1": modelConfig,
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)
	reqBody := `{"model":"model1", "temperature":0.1, "x_param":"123", "y_param":"abc", "stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	var response map[string]interface{}
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

	// `temperature` and `stream` are gone but model remains
	assert.Equal(t, `{"model":"model1","x_param":"123","y_param":"abc"}`, response["request_body"])

	// assert.Nil(t, response["temperature"])
	// assert.Equal(t, "123", response["x_param"])
	// assert.Equal(t, "abc", response["y_param"])
	// t.Logf("%v", response)
}

func TestNormalizeChatCompletionTools(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[
			{
				"type":"function",
				"name":"say_hello",
				"description":"Say hello",
				"parameters":{
					"type":"object",
					"properties":{"name":{"type":"string"}},
					"required":["name"]
				}
			}
		]
	}`)

	normalized, err := normalizeChatCompletionTools(body)
	assert.NoError(t, err)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	tools, ok := payload["tools"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, tools, 1) {
		return
	}

	tool, ok := tools[0].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	assert.Equal(t, "function", tool["type"])

	function, ok := tool["function"].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	assert.Equal(t, "say_hello", function["name"])
	assert.Equal(t, "Say hello", function["description"])
	parameters, ok := function["parameters"].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	assert.Equal(t, "object", parameters["type"])
}

func TestStripGrammarToolsConflictJSON_StripsGrammarAndJSONSchema(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","function":{"name":"say_hello","parameters":{"type":"object"}}}],
		"grammar":"root ::= custom",
		"response_format":{"type":"json_schema","json_schema":{"name":"demo","schema":{"type":"object"}}}
	}`)

	out, result, err := stripGrammarToolsConflictJSON(body)
	assert.NoError(t, err)
	assert.True(t, result.removedGrammar)
	assert.True(t, result.removedJSONSchemaResponse)
	assert.False(t, gjson.GetBytes(out, "grammar").Exists())
	assert.False(t, gjson.GetBytes(out, "response_format").Exists())
}

func TestStripGrammarToolsConflictJSON_KeepsJSONObjectMode(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","function":{"name":"say_hello","parameters":{"type":"object"}}}],
		"response_format":{"type":"json_object"}
	}`)

	out, result, err := stripGrammarToolsConflictJSON(body)
	assert.NoError(t, err)
	assert.False(t, result.removedGrammar)
	assert.False(t, result.removedJSONSchemaResponse)
	assert.Equal(t, "json_object", gjson.GetBytes(out, "response_format.type").String())
}

func TestProxyManager_ChatCompletionPreservesNativeToolShape(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := `{
		"model":"model1",
		"messages":[{"role":"user","content":"say hi"}],
		"tools":[
			{
				"type":"function",
				"name":"say_hello",
				"description":"Say hello",
				"parameters":{
					"type":"object",
					"properties":{"name":{"type":"string"}},
					"required":["name"]
				}
			}
		]
	}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]any
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

	requestBody, ok := response["request_body"].(string)
	if !assert.True(t, ok) {
		return
	}

	assert.Contains(t, requestBody, `"tools":[`)
	assert.Contains(t, requestBody, `"type":"function"`)
	assert.Contains(t, requestBody, `"name":"say_hello"`)
	assert.Contains(t, requestBody, `"description":"Say hello"`)
	assert.NotContains(t, requestBody, `"function":{"description":"Say hello","name":"say_hello","parameters"`)
	assert.NotContains(t, requestBody, `"parallel_tool_calls":false`)
}

func TestProxyManager_ChatCompletionStripsGrammarWhenToolsPresent(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := `{
		"model":"model1",
		"messages":[{"role":"user","content":"run"}],
		"tools":[{"type":"function","function":{"name":"shell","parameters":{"type":"object"}}}],
		"grammar":"root ::= custom",
		"response_format":{"type":"json_schema","json_schema":{"name":"demo","schema":{"type":"object"}}}
	}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]any
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

	requestBody, ok := response["request_body"].(string)
	if !assert.True(t, ok) {
		return
	}

	assert.NotContains(t, requestBody, `"grammar"`)
	assert.NotContains(t, requestBody, `"response_format":{"type":"json_schema"`)
}

func TestProxyManager_ChatCompletionKeepsJSONResponseObjectWhenToolsPresent(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := `{
		"model":"model1",
		"messages":[{"role":"user","content":"run"}],
		"tools":[{"type":"function","function":{"name":"shell","parameters":{"type":"object"}}}],
		"response_format":{"type":"json_object"}
	}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]any
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))

	requestBody, ok := response["request_body"].(string)
	if !assert.True(t, ok) {
		return
	}

	assert.Contains(t, requestBody, `"response_format":{"type":"json_object"}`)
}

func TestNormalizeResponsesRequest_AdaptsBuiltInTools(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":"hello",
		"tools":[
			{"type":"shell"},
			{"type":"apply_patch"},
			{"type":"web_search_preview"},
			{"type":"web_search"},
			{"type":"file_search"},
			{"type":"code_interpreter"},
			{"type":"image_generation"},
			{"type":"computer"},
			{"type":"function","name":"echo","parameters":{"type":"object"}}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.ElementsMatch(t, []string{"shell", "apply_patch", "web_search", "file_search", "code_interpreter", "image_generation", "computer"}, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	tools, ok := payload["tools"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, tools, 9) {
		return
	}

	first, _ := tools[0].(map[string]any)
	second, _ := tools[1].(map[string]any)
	third, _ := tools[2].(map[string]any)
	fourth, _ := tools[3].(map[string]any)
	fifth, _ := tools[4].(map[string]any)
	sixth, _ := tools[5].(map[string]any)
	seventh, _ := tools[6].(map[string]any)
	eighth, _ := tools[7].(map[string]any)
	ninth, _ := tools[8].(map[string]any)
	assert.Equal(t, "function", first["type"])
	assert.Equal(t, llamaSwapShellFunctionName, first["name"])
	assert.Equal(t, "function", second["type"])
	assert.Equal(t, llamaSwapApplyPatchFunctionName, second["name"])
	secondParams, ok := second["parameters"].(map[string]any)
	if assert.True(t, ok) {
		props, ok := secondParams["properties"].(map[string]any)
		if assert.True(t, ok) {
			_, hasOperation := props["operation"]
			assert.True(t, hasOperation, "apply_patch bridge tool should expose an operation argument for Responses compatibility")
		}
	}
	assert.Equal(t, "function", third["type"])
	assert.Equal(t, "web_search", third["name"])
	assert.Equal(t, "function", fourth["type"])
	assert.Equal(t, "web_search", fourth["name"])
	assert.Equal(t, "function", fifth["type"])
	assert.Equal(t, llamaSwapFileSearchFunctionName, fifth["name"])
	assert.Equal(t, "function", sixth["type"])
	assert.Equal(t, llamaSwapCodeInterpreterFunctionName, sixth["name"])
	assert.Equal(t, "function", seventh["type"])
	assert.Equal(t, llamaSwapImageGenerationFunctionName, seventh["name"])
	assert.Equal(t, "function", eighth["type"])
	assert.Equal(t, llamaSwapComputerFunctionName, eighth["name"])
	assert.Equal(t, "function", ninth["type"])
	assert.Equal(t, "echo", ninth["name"])
}

func TestBuildQwenResponsesToolPolicy_CanonicalizesLegacyWebSearchPreview(t *testing.T) {
	policy := buildQwenResponsesToolPolicy([]string{"web_search_preview"})

	assert.Contains(t, policy, "web_search")
	assert.NotContains(t, policy, "web_search_preview")
	assert.NotContains(t, policy, llamaSwapWebSearchFunctionName)
}

func TestNormalizeResponsesRequest_AdaptsComputerUsePreviewTool(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":"hello",
		"tools":[
			{"type":"computer_use_preview"},
			{"type":"function","name":"echo","parameters":{"type":"object"}}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Equal(t, []string{"computer"}, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	tools, ok := payload["tools"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, tools, 2) {
		return
	}

	first, _ := tools[0].(map[string]any)
	second, _ := tools[1].(map[string]any)
	assert.Equal(t, "function", first["type"])
	assert.Equal(t, llamaSwapComputerFunctionName, first["name"])
	assert.Equal(t, "function", second["type"])
	assert.Equal(t, "echo", second["name"])
}

func TestBuildQwenResponsesToolPolicy_UsesRealWebSearchName(t *testing.T) {
	policy := buildQwenResponsesToolPolicy([]string{"web_search", "file_search", "computer"})

	assert.Contains(t, policy, "web_search")
	assert.Contains(t, policy, "file_search")
	assert.Contains(t, policy, llamaSwapComputerFunctionName)
	assert.NotContains(t, policy, llamaSwapWebSearchFunctionName)
	assert.NotContains(t, policy, llamaSwapFileSearchFunctionName)
	assert.NotContains(t, policy, "- Use computer actions for UI automation requests.")
}

func TestNormalizeResponsesRequest_AdaptsCustomApplyPatchTool(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":"hello",
		"tools":[
			{
				"type":"custom",
				"name":"apply_patch",
				"description":"Use the apply_patch tool.",
				"format":{"type":"grammar","syntax":"lark","definition":"start: begin_patch"}
			}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Equal(t, []string{"apply_patch"}, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	tools, ok := payload["tools"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, tools, 1) {
		return
	}

	first, _ := tools[0].(map[string]any)
	assert.Equal(t, "function", first["type"])
	assert.Equal(t, llamaSwapApplyPatchFunctionName, first["name"])
	params, ok := first["parameters"].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	props, ok := params["properties"].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	_, hasOperation := props["operation"]
	assert.True(t, hasOperation)
}

func TestNormalizeResponsesRequest_RewritesToolOutputs(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":[
			{"type":"shell_call","call_id":"call_0","action":{"commands":["pwd"]}},
			{"type":"shell_call_output","call_id":"call_1","output":"ok","status":"completed"},
			{"type":"apply_patch_call","call_id":"call_1b","operation":{"type":"update_file","path":"README.md","diff":"@@"}},
			{"type":"apply_patch_call_output","call_id":"call_2","success":true},
			{"type":"web_search_call","call_id":"call_2b","action":{"query":"llama-swap github"}},
			{"type":"web_search_call_output","call_id":"call_3","output":[{"title":"A"}]},
			{"type":"file_search_call","call_id":"call_4","action":{"query":"readme","max_num_results":3}},
			{"type":"file_search_call_output","call_id":"call_5","output":[{"filename":"README.md"}]},
			{"type":"code_interpreter_call","call_id":"call_6","action":{"code":"print(1)","language":"python"}},
			{"type":"code_interpreter_call_output","call_id":"call_7","output":"1"},
			{"type":"image_generation_call","call_id":"call_8","action":{"prompt":"cat","size":"1024x1024"}},
			{"type":"image_generation_call_output","call_id":"call_9","output":[{"b64_json":"abc"}]},
			{"type":"computer_call","call_id":"call_10","action":{"action":"click","x":10,"y":20}},
			{"type":"computer_call_output","call_id":"call_11","output":{"ok":true}}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Empty(t, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))
	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 14) {
		return
	}

	first, _ := input[0].(map[string]any)
	second, _ := input[1].(map[string]any)
	third, _ := input[2].(map[string]any)
	fourth, _ := input[3].(map[string]any)
	fifth, _ := input[4].(map[string]any)
	sixth, _ := input[5].(map[string]any)
	seventh, _ := input[6].(map[string]any)
	eighth, _ := input[7].(map[string]any)
	ninth, _ := input[8].(map[string]any)
	tenth, _ := input[9].(map[string]any)
	eleventh, _ := input[10].(map[string]any)
	twelfth, _ := input[11].(map[string]any)
	thirteenth, _ := input[12].(map[string]any)
	fourteenth, _ := input[13].(map[string]any)
	assert.Equal(t, "function_call", first["type"])
	assert.Equal(t, llamaSwapShellFunctionName, first["name"])
	assert.Equal(t, "function_call_output", second["type"])
	assert.Equal(t, "call_1", second["call_id"])
	assert.Equal(t, "function_call", third["type"])
	assert.Equal(t, llamaSwapApplyPatchFunctionName, third["name"])
	assert.Equal(t, "function_call_output", fourth["type"])
	assert.Equal(t, "call_2", fourth["call_id"])
	assert.Equal(t, "function_call", fifth["type"])
	assert.Equal(t, "web_search", fifth["name"])
	assert.Equal(t, "function_call_output", sixth["type"])
	assert.Equal(t, "call_3", sixth["call_id"])
	assert.Equal(t, "function_call", seventh["type"])
	assert.Equal(t, llamaSwapFileSearchFunctionName, seventh["name"])
	assert.Equal(t, "function_call_output", eighth["type"])
	assert.Equal(t, "call_5", eighth["call_id"])
	assert.Equal(t, "function_call", ninth["type"])
	assert.Equal(t, llamaSwapCodeInterpreterFunctionName, ninth["name"])
	assert.Equal(t, "function_call_output", tenth["type"])
	assert.Equal(t, "call_7", tenth["call_id"])
	assert.Equal(t, "function_call", eleventh["type"])
	assert.Equal(t, llamaSwapImageGenerationFunctionName, eleventh["name"])
	assert.Equal(t, "function_call_output", twelfth["type"])
	assert.Equal(t, "call_9", twelfth["call_id"])
	assert.Equal(t, "function_call", thirteenth["type"])
	assert.Equal(t, llamaSwapComputerFunctionName, thirteenth["name"])
	assert.Equal(t, "function_call_output", fourteenth["type"])
	assert.Equal(t, "call_11", fourteenth["call_id"])
}

func TestResponsesRequestToChatMessages_ApplyPatchOperationMappedToOperation(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":    "apply_patch_call",
				"call_id": "call_patch_1",
				"operation": map[string]any{
					"type": "create_file",
					"path": "x.txt",
					"diff": "+ok",
				},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	if !assert.Len(t, messages, 1) {
		return
	}
	assert.Equal(t, "assistant", messages[0]["role"])

	toolCalls, ok := messages[0]["tool_calls"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, toolCalls, 1) {
		return
	}
	toolCall, _ := toolCalls[0].(map[string]any)
	fn, _ := toolCall["function"].(map[string]any)
	assert.Equal(t, "apply_patch", fn["name"])

	argsRaw, _ := fn["arguments"].(string)
	var args map[string]any
	assert.NoError(t, json.Unmarshal([]byte(argsRaw), &args))
	operation, ok := args["operation"].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	assert.Equal(t, "create_file", operation["type"])
	assert.Equal(t, "x.txt", operation["path"])
	assert.Equal(t, "+ok", operation["diff"])
}

func TestResponsesRequestToChatMessages_ApplyPatchEmptyOperationSkipped(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "apply_patch_call",
				"call_id":   "call_patch_empty",
				"operation": map[string]any{},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	assert.Len(t, messages, 0)
}

func TestResponsesRequestToChatMessages_MergesReasoningIntoFollowingFunctionCall(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "reasoning",
				"summary": []any{
					map[string]any{
						"type": "summary_text",
						"text": "run the command first",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_shell_1",
				"arguments": `{"cmd":"pwd"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "/tmp/workspace\n",
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	if !assert.Len(t, messages, 2) {
		return
	}
	assert.Equal(t, "assistant", messages[0]["role"])
	assert.Equal(t, "run the command first", messages[0]["reasoning_content"])
	toolCalls, ok := messages[0]["tool_calls"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, toolCalls, 1) {
		return
	}
	toolCall, _ := toolCalls[0].(map[string]any)
	fn, _ := toolCall["function"].(map[string]any)
	assert.Equal(t, "exec_command", fn["name"])
	assert.Equal(t, "tool", messages[1]["role"])
	assert.Equal(t, "call_shell_1", messages[1]["tool_call_id"])
	assert.Equal(t, "/tmp/workspace", messages[1]["content"])
}

func TestResponsesRequestToChatMessages_DisableReasoningHistoryReplaySkipsStandaloneReasoningItems(t *testing.T) {
	t.Setenv("LLAMASWAP_DISABLE_REASONING_HISTORY_REPLAY", "1")

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "reasoning",
				"summary": []any{
					map[string]any{
						"type": "summary_text",
						"text": "run the command first",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_shell_1",
				"arguments": `{"cmd":"pwd"}`,
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	if !assert.Len(t, messages, 1) {
		return
	}
	assert.Equal(t, "assistant", messages[0]["role"])
	_, hasReasoning := messages[0]["reasoning_content"]
	assert.False(t, hasReasoning)
}

func TestResponsesRequestToChatMessages_DisableReasoningHistoryReplayPreservesInlineAssistantReasoning(t *testing.T) {
	t.Setenv("LLAMASWAP_DISABLE_REASONING_HISTORY_REPLAY", "true")

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "<think>internal reasoning</think>\nVisible answer",
					},
				},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	if !assert.Len(t, messages, 1) {
		return
	}
	assert.Equal(t, "Visible answer", messages[0]["content"])
	assert.Equal(t, "internal reasoning", messages[0]["reasoning_content"])
}

func TestResponsesRequestToChatMessages_PreservesRoleForUntypedRoleContentInput(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"role":    "developer",
				"content": []any{map[string]any{"type": "input_text", "text": "dev rules"}},
			},
			map[string]any{
				"role":    "user",
				"content": []any{map[string]any{"type": "input_text", "text": "hello"}},
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	if !assert.Len(t, messages, 2) {
		return
	}
	assert.Equal(t, "system", messages[0]["role"])
	assert.Equal(t, "dev rules", messages[0]["content"])
	assert.Equal(t, "user", messages[1]["role"])
	assert.Equal(t, "hello", messages[1]["content"])
}

func TestNormalizePossiblyMixedToolArguments_ExtractsXMLInputTag(t *testing.T) {
	raw := `<tool_call><function=apply_patch><input>*** Begin Patch
*** Add File: z.txt
+ok
*** End Patch
</input></function></tool_call>`

	normalized := normalizePossiblyMixedToolArguments(raw)
	var args map[string]any
	assert.NoError(t, json.Unmarshal([]byte(normalized), &args))
	assert.Equal(t, "*** Begin Patch\n*** Add File: z.txt\n+ok\n*** End Patch", args["input"])
}

func TestNormalizeResponsesRequest_PreservesDeveloperRole(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":[
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"system rules"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Empty(t, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 2) {
		return
	}

	first, ok := input[0].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	second, ok := input[1].(map[string]any)
	if !assert.True(t, ok) {
		return
	}

	assert.Equal(t, "message", first["type"])
	assert.Equal(t, "developer", first["role"])
	assert.Equal(t, "message", second["type"])
	assert.Equal(t, "user", second["role"])
}

func TestNormalizeResponsesRequest_MovesSystemMessagesToFront(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":[
			{"type":"message","role":"system","content":[{"type":"input_text","text":"a"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"b"}]},
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"c"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"d"}]}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Empty(t, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 4) {
		return
	}

	roles := make([]string, 0, len(input))
	for _, rawItem := range input {
		item, ok := rawItem.(map[string]any)
		if !assert.True(t, ok) {
			return
		}
		role, _ := item["role"].(string)
		roles = append(roles, role)
	}

	assert.Equal(t, []string{"system", "user", "developer", "user"}, roles)
}

func TestNormalizeResponsesRequest_MergesLeadingSystemMessages(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":[
			{"type":"message","role":"system","content":[{"type":"input_text","text":"a"}]},
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"b"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"c"}]}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.Empty(t, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 3) {
		return
	}

	first, ok := input[0].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	second, ok := input[1].(map[string]any)
	if !assert.True(t, ok) {
		return
	}
	third, ok := input[2].(map[string]any)
	if !assert.True(t, ok) {
		return
	}

	assert.Equal(t, "system", first["role"])
	content, ok := first["content"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, content, 1) {
		return
	}
	assert.Equal(t, "developer", second["role"])
	assert.Equal(t, "user", third["role"])
}

func TestTranslateResponsesToChatCompletionsRequest_MergesDeveloperIntoLeadingSystem(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"instructions":"base system",
		"input":[
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"developer rules"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}
		]
	}`)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	assert.NoError(t, err)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(translated, &payload))

	messages, ok := payload["messages"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, messages, 2) {
		return
	}

	first, _ := messages[0].(map[string]any)
	second, _ := messages[1].(map[string]any)
	assert.Equal(t, "system", first["role"])
	assert.Contains(t, first["content"], "base system")
	assert.Contains(t, first["content"], "developer rules")
	assert.Equal(t, "user", second["role"])
	assert.Equal(t, "hello", second["content"])
}

func TestTranslateResponsesToChatCompletionsRequest_TextOnlyNativeStreamSkipsCloseThinkGuard(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"reasoning":{"effort":"medium"},
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Reply exactly PORT8080_OK"}]}
		]
	}`)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, "none", gjson.GetBytes(translated, "tool_choice").String())
	assert.True(t, gjson.GetBytes(translated, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(translated, "grammar").Exists())
	assert.False(t, gjson.GetBytes(translated, "logit_bias.248069").Exists())
}

func TestTranslateResponsesToChatCompletionsRequest_PlanOnlyNativeStreamSkipsCloseThinkGuard(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":true,
		"reasoning":{"effort":"medium"},
		"instructions":"Planning mode is active. Do NOT execute tasks, claim execution, or start implementing changes.",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Stay in plan mode for a hypothetical patch to repo_mirror/config.yaml, do not edit anything, and explicitly say you are not executing. End with exactly T22_SENTINEL."}]}
		]
	}`)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, "none", gjson.GetBytes(translated, "tool_choice").String())
	assert.True(t, gjson.GetBytes(translated, "chat_template_kwargs.enable_thinking").Bool())
	assert.False(t, gjson.GetBytes(translated, "grammar").Exists())
	assert.False(t, gjson.GetBytes(translated, "logit_bias.248069").Exists())
	assert.False(t, gjson.GetBytes(translated, "parallel_tool_calls").Bool())
}

func TestTranslateResponsesToChatCompletionsRequest_PlanOnlyNoToolDisablesParallelToolCalls(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.2",
		"stream":false,
		"reasoning":{"effort":"medium"},
		"parallel_tool_calls":true,
		"tools":[
			{"type":"function","function":{"name":"shell"}},
			{"type":"function","function":{"name":"apply_patch"}}
		],
		"instructions":"Planning mode is active. Do NOT execute tasks, claim execution, or start implementing changes.",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Stay in plan mode for a hypothetical patch to repo_mirror/config.yaml, do not edit anything, and explicitly say you are not executing. End with exactly T22_SENTINEL."}]}
		]
	}`)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.Equal(t, "none", gjson.GetBytes(translated, "tool_choice").String())
	assert.False(t, gjson.GetBytes(translated, "parallel_tool_calls").Bool())
	assert.False(t, gjson.GetBytes(translated, "tools").Exists())
	assert.False(t, gjson.GetBytes(translated, "grammar").Exists())
	assert.False(t, gjson.GetBytes(translated, "logit_bias.248069").Exists())
}

func TestRequestExplicitlyWantsShellVerification_MatchesFinalFileContentPhrase(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11, then verify the final file content with shell. Finish with exactly T11_SENTINEL.",
					},
				},
			},
		},
	}

	assert.True(t, requestExplicitlyWantsShellVerification(req))
}

func TestNormalizeChatCompletionReasoningBoundary_MovesLeakedFinalAnswerOutOfReasoning(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-bad-reasoning-split",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "content":"",
	      "reasoning_content":"</think>\n\nWhat task would you like me to help you with? T20_SENTINEL"
	    },
	    "finish_reason":"stop"
	  }]
	}`)

	normalized := normalizeChatCompletionReasoningBoundary(body)
	assert.Equal(t, "What task would you like me to help you with? T20_SENTINEL", gjson.GetBytes(normalized, "choices.0.message.content").String())
	assert.False(t, gjson.GetBytes(normalized, "choices.0.message.reasoning_content").Exists())
	assert.NotContains(t, string(normalized), "</think>")
}

func TestNormalizeChatCompletionReasoningBoundary_StripsDanglingCloserFromVisibleContentWhenReasoningExists(t *testing.T) {
	body := []byte(`{
	  "id":"chatcmpl-bad-visible-split",
	  "choices":[{
	    "message":{
	      "role":"assistant",
	      "content":"Visible answer line\n</think>",
	      "reasoning_content":"Hidden reasoning"
	    },
	    "finish_reason":"stop"
	  }]
	}`)

	normalized := normalizeChatCompletionReasoningBoundary(body)
	assert.Equal(t, "Visible answer line", gjson.GetBytes(normalized, "choices.0.message.content").String())
	assert.Equal(t, "Hidden reasoning", gjson.GetBytes(normalized, "choices.0.message.reasoning_content").String())
	assert.NotContains(t, gjson.GetBytes(normalized, "choices.0.message.content").String(), "</think>")
}

func TestExtractContentAndReasoning_ParsesBalancedThinkTagsStructurally(t *testing.T) {
	content, reasoning := extractContentAndReasoning("Intro <think>internal one</think>\nVisible answer")
	assert.Equal(t, "Visible answer", content)
	assert.Equal(t, "internal one", reasoning)
}

func TestExtractContentAndReasoning_ParsesMultipleBalancedThinkTags(t *testing.T) {
	content, reasoning := extractContentAndReasoning("<think>internal one</think>\nVisible\n<thinking>internal two</thinking>")
	assert.Equal(t, "Visible", content)
	assert.Equal(t, "internal one\n\ninternal two", reasoning)
}

func TestNormalizeResponsesRequest_InjectsQwenToolPolicyForBuiltInTools(t *testing.T) {
	body := []byte(`{
		"model":"Qwen3.5-35B-A3B-Q8",
		"input":[
			{"type":"message","role":"system","content":"You are a coding agent."},
			{"type":"message","role":"user","content":"Edit app.js."}
		],
		"tools":[
			{"type":"shell"},
			{"type":"apply_patch"},
			{"type":"web_search"}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.ElementsMatch(t, []string{"shell", "apply_patch", "web_search"}, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 2) {
		return
	}

	first, ok := input[0].(map[string]any)
	if !assert.True(t, ok) {
		return
	}

	content, ok := first["content"].(string)
	if !assert.True(t, ok) {
		return
	}

	assert.Contains(t, content, "Use apply_patch for any file creation, deletion, or modification.")
	assert.Contains(t, content, "Do not use shell to edit files.")
	assert.Contains(t, content, "Use web_search for current or external information.")
}

func TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputInPlanContinuationAfterToolOutput(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "Use the request_user_input tool in native Codex question format and ask exactly one short clarifying question.",
		"tool_choice":  "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "write a plan for a small game"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":"pwd"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "c:/repo\n",
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "ask"},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "request_user_input"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	toolChoice, ok := payload["tool_choice"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", toolChoice["type"])
	fn, ok := toolChoice["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "request_user_input", fn["name"])
}

func TestTranslateResponsesToChatCompletionsRequest_ForcesRequestUserInputAfterExplorationBeforeAnyQuestion(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "Plan mode request.",
		"tool_choice":  "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "write a plan for a small game"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":"pwd"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "c:/repo\n",
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "request_user_input"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	toolChoice, ok := payload["tool_choice"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", toolChoice["type"])
	fn, ok := toolChoice["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "request_user_input", fn["name"])
}

func TestTranslateResponsesToChatCompletionsRequest_ForcesProposedPlanAfterCompletedRequestUserInput(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "Plan mode request.",
		"tool_choice":  "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "write a plan for a small game"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "request_user_input"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	messages, ok := payload["messages"].([]any)
	require.True(t, ok)
	first, ok := messages[0].(map[string]any)
	require.True(t, ok)
	assert.NotContains(t, fmt.Sprintf("%v", first["content"]), "Return exactly one complete <proposed_plan> block now.")
}

func TestTranslateResponsesToChatCompletionsRequest_PlanAndQuestionMatrix(t *testing.T) {
	planDev := func() map[string]any {
		return map[string]any{
			"type": "message",
			"role": "developer",
			"content": []any{
				map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
			},
		}
	}
	defaultDev := func() map[string]any {
		return map[string]any{
			"type": "message",
			"role": "developer",
			"content": []any{
				map[string]any{"type": "input_text", "text": "<collaboration_mode># Collaboration Mode: Default\nYou are in Default mode."},
			},
		}
	}
	noisyDefaultDev := func() map[string]any {
		return map[string]any{
			"type": "message",
			"role": "developer",
			"content": []any{
				map[string]any{"type": "input_text", "text": "<collaboration_mode># Collaboration Mode: Default\nYou are in Default mode.\nIf you truly need clarification, use the request_user_input tool in native Codex question format.\nDo not use it unless the current request explicitly asks for a clarification question."},
			},
		}
	}
	userMsg := func(text string) map[string]any {
		return map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_text", "text": text},
			},
		}
	}
	toolCall := func(name, callID, args string) map[string]any {
		return map[string]any{
			"type":      "function_call",
			"name":      name,
			"call_id":   callID,
			"arguments": args,
		}
	}
	toolOutput := func(callID, output string) map[string]any {
		return map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		}
	}
	baseTools := func(names ...string) []any {
		tools := make([]any, 0, len(names))
		for _, name := range names {
			tools = append(tools, map[string]any{"type": "function", "name": name})
		}
		return tools
	}
	baseReq := func(dev map[string]any, instructions string, user string, inputTail []any, tools []any, toolChoice any) map[string]any {
		input := []any{dev, userMsg(user)}
		input = append(input, inputTail...)
		return map[string]any{
			"model":        "gpt-5.2",
			"instructions": instructions,
			"tool_choice":  toolChoice,
			"input":        input,
			"tools":        tools,
		}
	}
	type testCase struct {
		name                    string
		req                     map[string]any
		wantToolChoiceString    string
		wantToolChoiceFunction  string
		wantInstructionContains string
		wantInstructionExcludes string
		wantToolAbsent          string
	}
	tests := []testCase{
		{
			name: "plan shell exploration does not force request_user_input",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "plan web search exploration does not force request_user_input",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("web_search", "call_web_1", `{"query":"game ideas"}`), toolOutput("call_web_1", "results")},
				baseTools("web_search", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "plan apply_patch history does not force request_user_input",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("apply_patch", "call_patch_1", `{"patch":"*** Begin Patch"}`), toolOutput("call_patch_1", "ok")},
				baseTools("apply_patch", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "explicit native codex question request does not force request_user_input before tool output",
			req: baseReq(planDev(), "Use the request_user_input tool in native Codex question format and ask exactly one short clarifying question.",
				"write a plan for a small game", nil, baseTools("request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "explicit native codex question request after shell output does not force request_user_input",
			req: baseReq(planDev(), "Use the request_user_input tool in native Codex question format and ask exactly one short clarifying question.",
				"write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "ask how they want it built does not force native question tool",
			req: baseReq(planDev(), "Use request_user_input in native question format and ask how they would like this built.",
				"ask the user how they would like this game built and which approach they prefer",
				nil, baseTools("request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "plan continuation with required tool choice stays required",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), "required"),
			wantToolChoiceString: "required",
		},
		{
			name: "specific shell tool choice is preserved in plan continuation",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": "shell",
					},
				}),
			wantToolChoiceFunction: "shell",
		},
		{
			name: "plan continuation without request_user_input tool stays auto",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "default mode exploration never forces request_user_input",
			req: baseReq(defaultDev(), "Default mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "default mode simple hello does not force request_user_input from boilerplate",
			req: baseReq(noisyDefaultDev(), "Default mode request.", "hi",
				nil, baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "default mode web search does not force request_user_input from boilerplate",
			req: baseReq(noisyDefaultDev(), "Default mode request.", "search the web for YLAB",
				nil, baseTools("web_search", "request_user_input", "shell"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "plan followup with web research before plan keeps tools available",
			req: baseReq(planDev(), "Plan mode request.", "write a native plan and do web research before",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("shell", "request_user_input", "web_search"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "later reddit search in plan conversation keeps tools available",
			req: baseReq(planDev(), "Plan mode request.", "find all qwen 3.6 models in reddit webside",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
					toolCall("web_search", "call_ws_1", `{"query":"ylab architects principles"}`),
					toolOutput("call_ws_1", `{"query":"ylab architects principles","results":[{"title":"YLAB"}]}`),
					toolCall("mcp__playwright__browser_navigate", "call_browser_1", `{"url":"https://www.ylab.es/en/"}`),
					toolOutput("call_browser_1", `{"url":"https://www.ylab.es/en/"}`),
				},
				baseTools("request_user_input", "web_search", "mcp__playwright__browser_navigate"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "no questions instruction blocks explicit question forcing",
			req: baseReq(planDev(), "Use the request_user_input tool in native Codex question format and ask exactly one short clarifying question.",
				"no questions please", nil, baseTools("request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "completed request_user_input forces proposed plan return",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`), toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`)},
				baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "completed request_user_input after shell exploration still forces proposed plan return",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{
					toolCall("shell", "call_shell_1", `{"command":"pwd"}`),
					toolOutput("call_shell_1", "c:/repo\n"),
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("shell", "request_user_input"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "multiple completed request_user_input calls still force proposed plan return",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
					toolCall("request_user_input", "call_q_2", `{"questions":["What colors?"]}`),
					toolOutput("call_q_2", `{"answers":[{"id":"colors","value":"bright"}]}`),
				},
				baseTools("request_user_input"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "specific shell tool choice after completed request_user_input is preserved",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("shell", "request_user_input"), map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": "shell",
					},
				}),
			wantToolChoiceFunction:  "shell",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "incomplete request_user_input does not force proposed plan return",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`)},
				baseTools("request_user_input"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "default mode with completed request_user_input still forces proposed plan return",
			req: baseReq(defaultDev(), "Default mode request.", "write a plan for a small game",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("request_user_input"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "plan return still forces without request_user_input tool exposed in tools",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("shell"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "completed request_user_input with required tool choice still forces proposed plan return",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What style?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"style","value":"kid-friendly"}]}`),
				},
				baseTools("request_user_input"), "required"),
			wantToolChoiceString:    "required",
			wantInstructionExcludes: "Return exactly one complete <proposed_plan> block now.",
		},
		{
			name: "default mode write a plan removes update_plan and adds wrapping instruction",
			req: baseReq(defaultDev(), "Default mode request.", "write a plan for a small math game",
				nil, baseTools("update_plan", "shell"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionContains: "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>",
			wantToolAbsent:          "update_plan",
		},
		{
			name: "default mode return a plan also removes update_plan",
			req: baseReq(defaultDev(), "Default mode request.", "return a plan for a portfolio site",
				nil, baseTools("update_plan", "shell"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionContains: "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>",
			wantToolAbsent:          "update_plan",
		},
		{
			name: "default mode implement this plan does not remove update_plan",
			req: baseReq(defaultDev(), "Default mode request.", "please implement this plan using apply_patch",
				nil, baseTools("update_plan", "apply_patch"), "auto"),
			wantInstructionExcludes: "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>",
		},
		{
			name: "retry after invalid apply patch keeps tools available",
			req: baseReq(defaultDev(), "Default mode request.", "try again",
				[]any{
					toolCall("request_user_input", "call_q_1", `{"questions":["What platform?"]}`),
					toolOutput("call_q_1", `{"answers":[{"id":"platform","value":"html"}]}`),
					map[string]any{
						"type": "message",
						"role": "assistant",
						"content": []any{
							map[string]any{"type": "output_text", "text": "apply_patch call was not executed because operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update."},
						},
					},
				},
				baseTools("request_user_input", "apply_patch", "shell"), "auto"),
			wantToolChoiceString: "auto",
		},
		{
			name: "default mode non plan request keeps update_plan untouched",
			req: baseReq(defaultDev(), "Default mode request.", "say hello",
				nil, baseTools("update_plan", "shell"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>",
		},
		{
			name: "plan mode does not inject default mode plan wrapping instruction",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				nil, baseTools("update_plan", "request_user_input"), "auto"),
			wantToolChoiceString:    "auto",
			wantInstructionExcludes: "return the plan directly as assistant text wrapped in <proposed_plan>...</proposed_plan>",
		},
		{
			name: "specific request_user_input tool choice is preserved",
			req: baseReq(planDev(), "Plan mode request.", "write a plan for a small game",
				[]any{toolCall("shell", "call_shell_1", `{"command":"pwd"}`), toolOutput("call_shell_1", "c:/repo\n")},
				baseTools("shell", "request_user_input"), map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": "request_user_input",
					},
				}),
			wantToolChoiceFunction: "request_user_input",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body, err := json.Marshal(tc.req)
			require.NoError(t, err)

			translated, err := translateResponsesToChatCompletionsRequest(body)
			require.NoError(t, err)

			var payload map[string]any
			require.NoError(t, json.Unmarshal(translated, &payload))

			if tc.wantToolChoiceFunction != "" {
				toolChoice, ok := payload["tool_choice"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "function", toolChoice["type"])
				fn, ok := toolChoice["function"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, tc.wantToolChoiceFunction, fn["name"])
			}
			if tc.wantToolChoiceString != "" {
				assert.Equal(t, tc.wantToolChoiceString, payload["tool_choice"])
			}

			messages, ok := payload["messages"].([]any)
			require.True(t, ok)
			var allContent []string
			for _, rawMsg := range messages {
				msg, ok := rawMsg.(map[string]any)
				require.True(t, ok)
				allContent = append(allContent, fmt.Sprintf("%v", msg["content"]))
			}
			joinedContent := strings.Join(allContent, "\n")

			if tc.wantInstructionContains != "" {
				assert.Contains(t, joinedContent, tc.wantInstructionContains)
			}
			if tc.wantInstructionExcludes != "" {
				assert.NotContains(t, joinedContent, tc.wantInstructionExcludes)
			}

			if tc.wantToolAbsent != "" {
				tools, ok := payload["tools"].([]any)
				require.True(t, ok)
				for _, rawTool := range tools {
					tool, ok := rawTool.(map[string]any)
					require.True(t, ok)
					assert.NotEqual(t, tc.wantToolAbsent, tool["name"])
				}
			}
			if tc.name == "retry after invalid apply patch keeps tools available" {
				tools, ok := payload["tools"].([]any)
				require.True(t, ok)
				assert.NotEmpty(t, tools)
			}
		})
	}
}

func TestTranslateResponsesToChatCompletionsRequest_PostPatchContinuationKeepsBroadTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "Default mode request.",
		"tool_choice":  "auto",
		"tools": []any{
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "browser_navigate"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Collaboration Mode: Default\nYou are in Default mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Continue by checking the created file and open it in the browser."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"create_file","path":"biology-quiz.html","content":"<html></html>"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Success. Updated the following files:\nA biology-quiz.html",
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))
	assert.Equal(t, "auto", payload["tool_choice"])

	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 3)

	names := make([]string, 0, len(tools))
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		require.True(t, ok)
		names = append(names, extractFunctionToolName(tool))
	}
	assert.Contains(t, names, "apply_patch")
	assert.Contains(t, names, "shell")
	assert.Contains(t, names, "browser_navigate")
}

func TestTranslateResponsesToChatCompletionsRequest_PostShellContinuationKeepsBroadTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.4",
		"instructions": "Default mode request.",
		"tool_choice":  "auto",
		"tools": []any{
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "shell"},
			map[string]any{"type": "function", "name": "browser_navigate"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Collaboration Mode: Default\nYou are in Default mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Create a skill and write SKILL.md after you inspect the destination folder."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":["powershell.exe","-Command","Get-ChildItem $env:CODEX_HOME/skills"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "Directory listing complete.",
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))
	assert.Equal(t, "auto", payload["tool_choice"])

	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 3)

	names := make([]string, 0, len(tools))
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		require.True(t, ok)
		names = append(names, extractFunctionToolName(tool))
	}
	assert.Contains(t, names, "apply_patch")
	assert.Contains(t, names, "shell")
	assert.Contains(t, names, "browser_navigate")
}

func TestExtractCodexAvailableSkillsFromText(t *testing.T) {
	text := `<skills_instructions>
## Skills
### Available skills
- alpha-skill: First
- beta_skill: Second
### How to use skills
</skills_instructions>`

	assert.Equal(t, []string{"alpha-skill", "beta_skill"}, extractCodexAvailableSkillsFromText(text))
}

func TestExtractCodexSkillRootsFromText(t *testing.T) {
	text := `<skills_instructions>
## Skills
### Available skills
- alpha-skill: First (file: C:/Users/YLAB-Partner/.codex/skills/alpha-skill/SKILL.md)
- beta-skill: Second (file: C:/Users/YLAB-Partner/.codex/skills/beta-skill/SKILL.md)
### How to use skills
</skills_instructions>`

	roots := extractCodexSkillRootsFromText(text)
	require.Len(t, roots, 1)
	assert.Equal(t, "/mnt/c/Users/YLAB-Partner/.codex/skills", filepath.ToSlash(roots[0]))
}

func TestDetectCodexSkillSnapshotDrift(t *testing.T) {
	codexHome := t.TempDir()
	t.Setenv("CODEX_HOME", codexHome)
	require.NoError(t, os.MkdirAll(filepath.Join(codexHome, "skills", "alpha-skill"), 0o755))
	require.NoError(t, os.MkdirAll(filepath.Join(codexHome, "skills", "beta-skill"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(codexHome, "skills", "alpha-skill", "SKILL.md"), []byte("---\nname: alpha-skill\n---\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(codexHome, "skills", "beta-skill", "SKILL.md"), []byte("---\nname: beta-skill\n---\n"), 0o644))

	body := []byte(`{
	  "input": [
	    {
	      "type": "message",
	      "role": "developer",
	      "content": [
	        {
	          "type": "input_text",
	          "text": "<skills_instructions>\n## Skills\n### Available skills\n- alpha-skill: First\n- ghost-skill: Missing on disk\n### How to use skills\n</skills_instructions>"
	        }
	      ]
	    }
	  ]
	}`)

	missing, extra, ok := detectCodexSkillSnapshotDrift(body)
	require.True(t, ok)
	assert.Equal(t, []string{"beta-skill"}, missing)
	assert.Equal(t, []string{"ghost-skill"}, extra)
}

func TestRequestHistoryShowsCodexSkillFilesystemProof(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "update $list-workspace-files and check the skill definition"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":["powershell.exe","-Command","Get-ChildItem $env:CODEX_HOME/skills -Recurse -Filter SKILL.md"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "C:\\Users\\YLAB-Partner\\.codex\\skills\\list-workspace-files\\SKILL.md",
			},
		},
	}

	assert.True(t, requestHistoryShowsCodexSkillFilesystemProof(req))
}

func TestRequestWantsShellInspectionBeforeApplyPatch_DetectsFirstInspectThenPatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First inspect ./tmp/demo/app.js with shell, then use apply_patch to add a line.",
					},
				},
			},
		},
	}

	assert.True(t, requestWantsShellInspectionBeforeApplyPatch(req))
}

func TestRequestExplicitlyWantsShellVerification_DetectsSavedFilePhrase(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Inspect app.js with shell, then use apply_patch, then verify the saved file with shell before finishing.",
					},
				},
			},
		},
	}

	assert.True(t, requestExplicitlyWantsShellVerification(req))
}

func TestDeriveContinuationAllowedToolNames_AfterApplyPatchWithPendingShellVerificationReturnsShell(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "First inspect ./tmp/demo/app.js with shell, then use apply_patch to add a line logging verified-change, then verify the saved file with shell before finishing.",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"update_file","path":"./tmp/demo/app.js","content":"console.log('verified-change')\n"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Patch applied.",
			},
		},
	}

	assert.Equal(t, []string{"shell"}, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_AfterWebSearchDoesNotCollapseToApplyPatch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "exec_command"}},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Investigate the web for a first grade quiz game and then define a plan."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "__llamaswap_web_search_preview",
				"call_id":   "call_web_1",
				"arguments": `{"query":"first grade quiz game ideas"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_web_1",
				"output":  `{"type":"web_search_call_output","output":"results"}`,
			},
		},
	}

	assert.Nil(t, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_AfterSkillCreateReturnsShellForProof(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "exec_command"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "write_stdin"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Create a new Codex skill at ./skills/reward-skill/SKILL.md, then read the skill you created, follow it, and create src/skill-usage.md. Use apply_patch for file writes."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"create_file","path":"./skills/reward-skill/SKILL.md","content":"---\nname: reward-skill\n---\n"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Patch applied.",
			},
		},
	}

	assert.Equal(t, []string{"shell"}, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_AfterWaitAgentInMixedFlowReturnsWriteTools(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "exec_command"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "write_stdin"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "spawn_agent"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "wait_agent"}},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, wait for that agent, integrate its result, then use apply_patch to create src/agent-report.md."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "spawn_agent",
				"call_id":   "call_agent_1",
				"arguments": `{"model":"gpt-5.4","message":"inspect"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_agent_1",
				"output":  "spawned",
			},
			map[string]any{
				"type":      "function_call",
				"name":      "wait_agent",
				"call_id":   "call_wait_1",
				"arguments": `{"target":"agent_123"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_wait_1",
				"output":  "agent completed",
			},
		},
	}

	assert.Equal(t, []string{"apply_patch", "shell"}, deriveContinuationAllowedToolNames(req))
}

func TestRequestMentionsExactlyOneSubagent(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, wait for it, then write the report."},
				},
			},
		},
	}

	assert.True(t, requestMentionsExactlyOneSubagent(req))
	assert.False(t, requestMentionsMultipleAgentSteps(req))
}

func TestRequestMentionsExactlyOneSubagent_UsesExplicitUserInputNotEnvironmentPreamble(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<permissions instructions>\nFilesystem sandboxing defines which files can be read or written.\n</permissions instructions>"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, wait for it, then write the report."},
				},
			},
		},
	}

	assert.True(t, requestMentionsExactlyOneSubagent(req))
	assert.True(t, requestMentionsAgentOrchestration(req))
}

func TestRequestInputMentionsApplyPatch_PreservesOriginalPromptAcrossSubagentNotifications(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<environment_context>\n  <cwd>/home/admmin/llama-swap</cwd>\n</environment_context>"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for it, then use apply_patch to create src/agent-report.md."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"completed\":\"feature summary\"}</subagent_notification>"},
				},
			},
		},
	}

	assert.True(t, requestInputMentionsApplyPatch(req))
	assert.True(t, requestMentionsExactlyOneSubagent(req))
}

func TestNormalizeApplyPatchOperation_StripsTrailingInjectedParameterShellArtifact(t *testing.T) {
	normalized := normalizeApplyPatchOperation(map[string]any{
		"type":    "create_file",
		"path":    "./tmp/demo/agent-report.md",
		"content": "hello world\n</parameter> > /dev/null 2>&1 2>&1; echo $?",
	})

	op, ok := normalized.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "hello world", op["content"])
}

func TestDeriveContinuationAllowedToolNames_ExactlyOneSubagentAfterWaitAgentPrefersWriteTools(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "exec_command"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "write_stdin"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "spawn_agent"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "send_input"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "wait_agent"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "close_agent"}},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, wait for that agent, then use apply_patch to create src/agent-report.md and verify it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "spawn_agent",
				"call_id":   "call_agent_1",
				"arguments": `{"model":"gpt-5.4","message":"inspect"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_agent_1",
				"output":  "spawned",
			},
			map[string]any{
				"type":      "function_call",
				"name":      "wait_agent",
				"call_id":   "call_wait_1",
				"arguments": `{"target":"agent_123"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_wait_1",
				"output":  "agent completed",
			},
		},
	}

	assert.Equal(t, []string{"apply_patch", "shell"}, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_ExactlyOneSubagentAfterWaitAgentForcesApplyPatchWrite(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "exec_command"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "write_stdin"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "apply_patch"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "spawn_agent"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "wait_agent"}},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent via spawn_agent, wait for that agent, integrate its result, then use apply_patch to create src/agent-report.md with a short delegated summary."},
				},
			},
			map[string]any{"type": "function_call", "name": "spawn_agent", "call_id": "call_agent_1", "arguments": `{"model":"gpt-5.4","message":"inspect"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_agent_1", "output": `{"agent_id":"agent_123"}`},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary\"}}</subagent_notification>"},
				},
			},
		},
	}

	assert.Equal(t, []string{"apply_patch"}, deriveContinuationAllowedToolNames(req))
}

func TestRequestShouldForceApplyPatchAfterSingleSubagentWait(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for it, then use apply_patch to create src/agent-report.md with a short delegated summary."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary\"}}</subagent_notification>"},
				},
			},
		},
	}

	assert.True(t, requestShouldForceApplyPatchAfterSingleSubagentWait(req))
}

func TestRequestShouldForceApplyPatchAfterSingleSubagentWait_FalseWhenResumeOrSendInputStillRequested(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for it, then use resume_agent on the same agent and send_input once asking for one more feature. After that use apply_patch to create src/agent-resume-report.md."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary\"}}</subagent_notification>"},
				},
			},
		},
	}

	assert.False(t, requestShouldForceApplyPatchAfterSingleSubagentWait(req))
}

func TestShouldEnableStrictApplyPatchIntent_FalseWhenViewImageOrFileSearchPrecedesPatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use view_image on fixtures/solid-red.png and then use apply_patch to create src/image-report.md."},
				},
			},
		},
	}
	assert.False(t, shouldEnableStrictApplyPatchIntent(req, nil))

	req2 := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use file_search to find the marker, then use apply_patch to create src/file-search-report.md."},
				},
			},
		},
	}
	assert.False(t, shouldEnableStrictApplyPatchIntent(req2, nil))
}

func TestDeriveContinuationAllowedToolNames_PrefersAgentToolsForResumeFollowupAfterWait(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "spawn_agent"},
			map[string]any{"type": "function", "name": "send_input"},
			map[string]any{"type": "function", "name": "resume_agent"},
			map[string]any{"type": "function", "name": "wait_agent"},
			map[string]any{"type": "function", "name": "close_agent"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for that agent, then use resume_agent on the same agent and send_input once asking for one more feature. After that use apply_patch to create src/agent-resume-report.md."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary\"}}</subagent_notification>"},
				},
			},
		},
	}

	assert.Equal(t, []string{"spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent"}, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_SwitchesToApplyPatchAfterCloseAgent(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "spawn_agent"},
			map[string]any{"type": "function", "name": "send_input"},
			map[string]any{"type": "function", "name": "resume_agent"},
			map[string]any{"type": "function", "name": "wait_agent"},
			map[string]any{"type": "function", "name": "close_agent"},
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "exec_command"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for that agent, then use resume_agent on the same agent and send_input once asking for one more feature. Wait for the resumed agent, integrate both suggestions, and use apply_patch to create src/agent-resume-report.md."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{"type": "function_call", "name": "resume_agent", "call_id": "call_resume_1", "arguments": `{"id":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_resume_1", "output": `{"status":"ok"}`},
			map[string]any{"type": "function_call", "name": "send_input", "call_id": "call_send_1", "arguments": `{"target":"agent_123","message":"one more feature"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_send_1", "output": `{"status":"ok"}`},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_2", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_2", "output": `{"status":{"agent_123":{"completed":"second feature"}}}`},
			map[string]any{"type": "function_call", "name": "close_agent", "call_id": "call_close_1", "arguments": `{"target":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_close_1", "output": `{"status":"closed"}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"second feature\"}}</subagent_notification>"},
				},
			},
		},
	}

	assert.Equal(t, []string{"apply_patch", "shell"}, deriveContinuationAllowedToolNames(req))
}

func TestDeriveContinuationAllowedToolNames_ExactlyOneSubagentAfterResumeForcesSendInput(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "spawn_agent"},
			map[string]any{"type": "function", "name": "send_input"},
			map[string]any{"type": "function", "name": "resume_agent"},
			map[string]any{"type": "function", "name": "wait_agent"},
			map[string]any{"type": "function", "name": "close_agent"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for that agent, then use resume_agent on the same agent and send_input once asking for one more feature. Wait for the resumed agent, integrate both suggestions, and use apply_patch to create src/agent-resume-report.md."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature one"}}}`},
			map[string]any{"type": "function_call", "name": "resume_agent", "call_id": "call_resume_1", "arguments": `{"id":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_resume_1", "output": `{"status":{"completed":"feature one"}}`},
		},
	}

	assert.Equal(t, []string{"send_input"}, deriveContinuationAllowedToolNames(req))
}

func TestRequestShouldForceSendInputAfterResume(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, then resume the same agent and send_input once asking for one more feature."},
				},
			},
			map[string]any{"type": "function_call", "name": "resume_agent", "call_id": "call_resume_1", "arguments": `{"id":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_resume_1", "output": `{"status":{"completed":"feature one"}}`},
		},
	}

	assert.True(t, requestShouldForceSendInputAfterResume(req))
}

func TestDeriveContinuationAllowedToolNames_ExplicitFileSearchSwitchesToApplyPatchAfterSearch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "file_search"},
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "shell"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use file_search to find the reward marker, then use apply_patch to create src/file-search-report.md. Do not use shell for the search step."},
				},
			},
			map[string]any{"type": "function_call", "name": "file_search", "call_id": "call_file_1", "arguments": `{"query":"reward marker"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_file_1", "output": `{"type":"file_search_call_output","payload":{"matches":[{"path":"docs/teacher-notes.md"}]}}`},
		},
	}

	assert.Equal(t, []string{"apply_patch"}, deriveContinuationAllowedToolNames(req))
}

func TestRemoveRedundantSubagentNotificationMessages(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for it, then use apply_patch to create src/agent-report.md with a short delegated summary."},
				},
			},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<subagent_notification>{\"agent_path\":\"agent_123\",\"status\":{\"completed\":\"feature summary\"}}</subagent_notification>"},
				},
			},
		},
	}

	removeRedundantSubagentNotificationMessages(req)

	items, ok := req["input"].([]any)
	require.True(t, ok)
	assert.Len(t, items, 3)
}

func TestRequestShouldFinalizeExactlyOneSubagentSummary_AfterApplyPatch(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.4",
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "spawn_agent"},
			map[string]any{"type": "function", "name": "send_input"},
			map[string]any{"type": "function", "name": "resume_agent"},
			map[string]any{"type": "function", "name": "wait_agent"},
			map[string]any{"type": "function", "name": "close_agent"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, wait for it, then use apply_patch to create src/agent-report.md with a short delegated summary."},
				},
			},
			map[string]any{"type": "function_call", "name": "spawn_agent", "call_id": "call_agent_1", "arguments": `{"model":"gpt-5.4","message":"inspect"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_agent_1", "output": "spawned"},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"target":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": "agent completed"},
			map[string]any{"type": "function_call", "name": "apply_patch", "call_id": "call_patch_1", "arguments": `{"operation":{"type":"create_file","path":"src/agent-report.md","content":"summary"}}`},
			map[string]any{"type": "function_call_output", "call_id": "call_patch_1", "output": "Patch applied."},
		},
	}

	assert.True(t, requestShouldFinalizeExactlyOneSubagentSummary(req, buildToolWorkflowState(req)))

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))
	assert.Equal(t, "none", payload["tool_choice"])
	assert.Nil(t, payload["tools"])
}

func TestTranslateResponsesToChatCompletionsRequest_ShellFirstThenPatchStartsWithShellTools(t *testing.T) {
	req := map[string]any{
		"model":       "gpt-5.4",
		"tool_choice": "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Inspect ./tmp/demo/app.js with shell, then use apply_patch to add a line logging verified-change, then verify the saved file with shell before finishing.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)

	names := make([]string, 0, len(tools))
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		require.True(t, ok)
		names = append(names, extractFunctionToolName(tool))
	}

	assert.Contains(t, names, "exec_command")
}

func TestTranslateResponsesToChatCompletionsRequest_ApplyPatchThenShellVerificationKeepsShellTools(t *testing.T) {
	req := map[string]any{
		"model":       "gpt-5.4",
		"tool_choice": "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Inspect ./tmp/demo/app.js with shell, then use apply_patch to add a line logging verified-change, then verify the saved file with shell before finishing.",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"update_file","path":"./tmp/demo/app.js","content":"console.log('verified-change')\n"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Patch applied.",
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)

	names := make([]string, 0, len(tools))
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		require.True(t, ok)
		names = append(names, extractFunctionToolName(tool))
	}

	assert.Contains(t, names, "exec_command")
	assert.NotContains(t, names, "apply_patch")
}

func TestShouldRetryMissingShellCall_WhenForcedShellReturnsOnlyVerificationProse(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}
	translatedChatRequest := []byte(`{"tool_choice":{"type":"function","function":{"name":"shell"}}}`)
	chatResponse := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"Patch applied. Verifying the file contents now.","reasoning_content":"The patch was applied. Now I need to verify the file with shell."}}]}`)
	translatedResponse := []byte(`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Patch applied. Verifying the file contents now."}]}]}`)
	workflowState := ToolWorkflowState{VerificationExpected: true}

	assert.True(t, shouldRetryMissingShellCall(req, translatedChatRequest, chatResponse, translatedResponse, workflowState))
}

func TestExtractLatestCompletedShellArguments_ExecCommandCmdAlias(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_shell_1",
				"arguments": `{"cmd":"cat ./tmp/demo/file.txt"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "demo output",
			},
		},
	}

	args := parseToolArgsMapString(extractLatestCompletedShellArguments(req))
	commands := normalizeShellCommandsValue(args["commands"])
	require.Len(t, commands, 1)
	assert.Equal(t, "cat ./tmp/demo/file.txt", commands[0])
}

func TestShouldRetryMissingShellCall_WhenInvalidApplyPatchNeedsFileRead(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}
	translatedChatRequest := []byte(`{"tool_choice":"auto"}`)
	chatResponse := []byte(`{"choices":[{"finish_reason":"tool_calls","message":{"content":"Let me check the current file contents first.","reasoning_content":"The diff didn't apply. Let me first read the current file contents to get the exact context."}}]}`)
	translatedResponse := []byte(`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Let me check the current file contents first."}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"apply_patch call was not executed because operation was invalid. Provide operation.type, operation.path, and non-empty diff/content for create/update."}]}]}`)

	assert.True(t, shouldRetryMissingShellCall(req, translatedChatRequest, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestShouldRetryMissingShellCall_WhenInvalidApplyPatchNeedsVerification(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}
	translatedChatRequest := []byte(`{"tool_choice":"auto"}`)
	chatResponse := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"Wait — I need to verify the file content, since content replaces the entire file. Let me check what was saved.","reasoning_content":"I need to verify the file to see what actually happened."}}]}`)
	translatedResponse := []byte(`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Wait — I need to verify the file content, since content replaces the entire file. Let me check what was saved."}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"apply_patch call was not executed because operation was invalid. Provide operation.type, operation.path, and non-empty diff/content for create/update."}]}]}`)
	workflowState := ToolWorkflowState{VerificationExpected: true}

	assert.True(t, shouldRetryMissingShellCall(req, translatedChatRequest, chatResponse, translatedResponse, workflowState))
}

func TestShouldRetryInvalidApplyPatchOperation_WhenUpstreamReturnedIncompleteApplyPatch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "exec_command"},
		},
	}
	chatResponse := []byte(`{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"apply_patch","arguments":"{\"operation\":{\"path\":\"quiz/index.html\",\"type\":\"create_file\"}}"}}]}}]}`)
	translatedResponse := []byte(`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"apply_patch call was not executed because operation was invalid. Provide operation.type, operation.path, and non-empty diff/content for create/update."}]}]}`)

	assert.True(t, shouldRetryInvalidApplyPatchOperation(req, chatResponse, translatedResponse))
}

func TestShouldRetryMissingBrowserOpenCall_WhenFinalizedAsProseOnly(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}
	chatResponse := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"Let's open it in your default browser!","reasoning_content":"All three files are created successfully. Let me open it in the browser so the user can see it working."}}]}`)
	translatedResponse := []byte(`{"status":"completed","output_text":"Let's open it in your default browser!","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Let's open it in your default browser!"}]}]}`)

	assert.True(t, shouldRetryMissingBrowserOpenCall(req, chatResponse, translatedResponse))
}

func TestShouldRetryWeakPlaceholderFinal_WhenBridgeWouldCompletePlaceholder(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Read ./skills/example/SKILL.md and summarize it."},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "name: example-skill",
			},
		},
	}
	workflowState := buildToolWorkflowState(req)
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Working on the request.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Working on the request."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Working on the request."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, workflowState))
}

func TestShouldRetryWeakPlaceholderFinal_WhenCompletedPlaceholderStillCarriesHistoricalToolItems(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the workspace briefly, then return a plan."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_shell_1",
				"arguments": `{"cmd":"find . -maxdepth 2 -type f | head"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "README.md\napp.js\nindex.html\n",
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Working on the request.",
		"output":[
			{"type":"function_call","name":"exec_command","call_id":"call_shell_1","arguments":"{\"cmd\":\"find . -maxdepth 2 -type f | head\"}"},
			{"type":"function_call_output","call_id":"call_shell_1","output":"README.md\napp.js\nindex.html\n"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Working on the request."}]}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Working on the request."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenStreamedUpstreamBodyIsRawSSE(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Summarize the workspace and then write a plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Working on the request.",
		"output":[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"Let me inspect the files."}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Working on the request."}]}
		]
	}`)
	chatResponse := []byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me inspect the files.\"}}]}\n\ndata: [DONE]\n")

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestShouldRetryWeakPlaceholderFinal_FalseForRealSummary(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Read ./skills/example/SKILL.md and summarize it."},
				},
			},
		},
	}
	workflowState := buildToolWorkflowState(req)
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"The skill uses shell to inspect files and apply_patch for edits.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The skill uses shell to inspect files and apply_patch for edits."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"The skill uses shell to inspect files and apply_patch for edits."
				}
			}
		]
	}`)

	assert.False(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, workflowState))
}

func TestShouldRetryWeakPlaceholderFinal_WhenVisibleTextIsToolArtifact(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the workspace and return a plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"<tool_call>",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"<tool_call>"}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"<tool_call>"
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenToolArtifactBecomesDeferredPromise(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the workspace and return a plan."},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "README.md",
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"The command was split incorrectly. Let me read the files properly.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The command was split incorrectly. Let me read the files properly."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"<tool_call>=shell>\n<parameter=command>\ncat README.md\n</parameter>\n</function>\n</tool_call>"
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenModelPromisesFirstInspection(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the workspace briefly, then return a plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me first look at what's in that directory.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me first look at what's in that directory."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me first look at what's in that directory."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenModelPromisesToStartExamining(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the workspace briefly, then return a plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me start by examining the directory structure.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me start by examining the directory structure."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"The user wants me to inspect a specific directory and then create a plan for turning it into a quiz game. Let me start by examining the directory structure."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenVerifyingPromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Update the file and verify it before the final answer."},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "updated file",
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"File created. Now verifying the content.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"File created. Now verifying the content."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"File created. Now verifying the content."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenBareVerifyMessageWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Update app.js and verify the saved file before the final answer.",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"update_file","path":"app.js","content":"console.log('ok')\n"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Patch applied.",
			},
		},
	}

	chatResponse := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"Verifying the file content:","reasoning_content":"Good, now let me verify the file content."}}]}`)
	translatedResponse := []byte(`{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Verifying the file content:"}]}]}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenCreatePromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Update the file and verify it before the final answer."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"The file doesn't exist. I'll create it with the correct content.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The file doesn't exist. I'll create it with the correct content."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"The file doesn't exist. I'll create it with the correct content."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenDetailedPlanPromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Ask me how to build it first and then write a detailed plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Got it — a colorful web-based quiz in pure HTML/CSS/JS covering multiple subjects, with score tracking and a results summary. Let me put together a detailed plan.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Got it — a colorful web-based quiz in pure HTML/CSS/JS covering multiple subjects, with score tracking and a results summary. Let me put together a detailed plan."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Got it — a colorful web-based quiz in pure HTML/CSS/JS covering multiple subjects, with score tracking and a results summary. Let me put together a detailed plan."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestShouldRetryWeakPlaceholderFinal_WhenSimplePlanPromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Ask me how to build it first and then write the plan."},
				},
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Great choices - a colorful browser-based quiz will be fun and engaging for a first-grader. Let me put together a plan.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Great choices - a colorful browser-based quiz will be fun and engaging for a first-grader. Let me put together a plan."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Great choices - a colorful browser-based quiz will be fun and engaging for a first-grader. Let me put together a plan."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestShouldRewritePlanModeResponseText_WhenModelPromisesToWritePlanNext(t *testing.T) {
	assert.True(t, shouldRewritePlanModeResponseText("Great choices - a colorful browser-based quiz will be fun and engaging for a first-grader. Let me put together a plan."))
	assert.True(t, shouldRewritePlanModeResponseText("The user wants:\n- HTML + JavaScript\n- Web Browser interface (HTML/CSS)\n\nLet me now design a detailed plan for this."))
}

func TestShouldRetryWeakPlaceholderFinal_WhenStructuredPlanStillRequiredButReplyIsGenericAcknowledgement(t *testing.T) {
	req := map[string]any{
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nOnly output the final plan when it is decision complete.\nWrap the final answer in <proposed_plan>...</proposed_plan> tags exactly.\nYou are in Plan Mode.\n</collaboration_mode>",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Ask me how to build it first and then return the final visible answer as exactly one <proposed_plan>...</proposed_plan> block."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":[{"id":"build_style","question":"How should the game be built?","options":[{"label":"HTML + JavaScript","description":"Single-page app"}]}]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"build_style":{"answers":["HTML + JavaScript"]}}}`,
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"That gives me enough information to proceed.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"That gives me enough information to proceed."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"That gives me enough information to proceed."
				}
			}
		]
	}`)

	assert.True(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenDirectoryCreatePromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Build the quiz app and create the files."},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  `{"output":"Success. Updated the following files:\nD index.html\n"}`,
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Good, old file is gone. Now let me create the first-grade quiz directory and build the full app.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Good, old file is gone. Now let me create the first-grade quiz directory and build the full app."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Good, old file is gone. Now let me create the first-grade quiz directory and build the full app."
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenApplyPatchPromiseWasFinalized(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the file, then update it with apply_patch and verify it."},
				},
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "{\n  \"questions\": []\n}\n",
			},
		},
	}
	translatedResponse := []byte("{\n" +
		"  \"status\":\"completed\",\n" +
		"  \"output_text\":\"Now I'll update the file with `apply_patch`:\",\n" +
		"  \"output\":[\n" +
		"    {\n" +
		"      \"type\":\"message\",\n" +
		"      \"role\":\"assistant\",\n" +
		"      \"content\":[{\"type\":\"output_text\",\"text\":\"Now I'll update the file with `apply_patch`:\"}]\n" +
		"    }\n" +
		"  ]\n" +
		"}")
	chatResponse := []byte("{\n" +
		"  \"choices\":[\n" +
		"    {\n" +
		"      \"finish_reason\":\"stop\",\n" +
		"      \"message\":{\n" +
		"        \"content\":\"Now I'll update the file with `apply_patch`:\"\n" +
		"      }\n" +
		"    }\n" +
		"  ]\n" +
		"}")

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenSynthesisTurnReturnsLetMeAlsoCheck(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "system",
				"content": []any{
					map[string]any{"type": "input_text", "text": "You now have enough gathered web research context to answer. Do not call any more tools on this turn. Synthesize the final user-facing answer directly from the gathered web_search results."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Investigate the web for first-grade quiz game ideas and then write a detailed implementation plan. Do not build the app yet."},
				},
			},
			map[string]any{
				"type":    "web_search_call_output",
				"call_id": "call_web_1",
				"output":  `{"ok":true}`,
			},
		},
	}
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"Let me also check the existing project structure to understand what we're working with.",
		"output":[
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Let me also check the existing project structure to understand what we're working with."}]
			}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{
					"content":"Let me also check the existing project structure to understand what we're working with.\n\n<tool_call>\nshell>\n<command>ls -la /home/admmin/llama-swap/</command>\n</function>"
				}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldRetryWeakPlaceholderFinal_WhenPostAgentCompletionIsSingleLetter(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use exactly one subagent, then use apply_patch to create src/agent-resume-report.md."},
				},
			},
			map[string]any{"type": "function_call", "name": "spawn_agent", "call_id": "call_spawn_1", "arguments": `{"message":"inspect"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_spawn_1", "output": `{"agent_id":"agent_123"}`},
			map[string]any{"type": "function_call", "name": "wait_agent", "call_id": "call_wait_1", "arguments": `{"targets":["agent_123"]}`},
			map[string]any{"type": "function_call_output", "call_id": "call_wait_1", "output": `{"status":{"agent_123":{"completed":"feature summary"}}}`},
			map[string]any{"type": "function_call", "name": "resume_agent", "call_id": "call_resume_1", "arguments": `{"id":"agent_123"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_resume_1", "output": `{"status":"completed"}`},
		},
	}
	workflowState := buildToolWorkflowState(req)
	translatedResponse := []byte(`{
		"status":"completed",
		"output_text":"I",
		"output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I"}]}
		]
	}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"stop",
				"message":{"content":"I"}
			}
		]
	}`)

	assert.True(t, shouldRetryWeakPlaceholderFinal(req, nil, chatResponse, translatedResponse, workflowState))
}

func TestNormalizeResponsesInputItem_ApplyPatchCallIncludesExecutableInput(t *testing.T) {
	item, changed := normalizeResponsesInputItem(map[string]any{
		"type":    "apply_patch_call",
		"call_id": "call_1",
		"operation": map[string]any{
			"type":    "update_file",
			"path":    "README.md",
			"content": "PATCH_OK",
		},
	})
	require.True(t, changed)
	mapped, ok := item.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call", mapped["type"])
	assert.Equal(t, llamaSwapApplyPatchFunctionName, mapped["name"])
	args := fmt.Sprintf("%v", mapped["arguments"])
	assert.Contains(t, args, `"operation"`)
	assert.Contains(t, args, `"input"`)
	assert.Contains(t, args, `*** Begin Patch`)
}

func TestAppendStrictApplyPatchToolOnlyInstruction_PrunesToApplyPatch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "function", "name": "write_stdin"},
		},
	}
	appendStrictApplyPatchToolOnlyInstruction(req, "test")
	tools, ok := req["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]any)
	assert.Equal(t, "apply_patch", tool["name"])
	assert.Equal(t, false, req["parallel_tool_calls"])
}

func TestShouldRetryMissingShellCall_DoesNotForceAnotherShellAfterInspectionBeforePatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the file with shell, then update it with apply_patch, then verify the saved file with shell."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_shell_1",
				"arguments": `{"cmd":"cat src/quiz-data.json"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "{\n  \"questions\": []\n}\n",
			},
		},
	}
	translatedChatRequest := []byte(`{"tool_choice":"auto","tools":[{"type":"function","name":"exec_command"},{"type":"function","name":"apply_patch"}]}`)
	chatResponse := []byte("{\n" +
		"  \"choices\":[\n" +
		"    {\n" +
		"      \"finish_reason\":\"stop\",\n" +
		"      \"message\":{\n" +
		"        \"content\":\"Now applying the update with apply_patch:\",\n" +
		"        \"reasoning_content\":\"Now I need to use apply_patch to update the file with the exact content specified.\"\n" +
		"      }\n" +
		"    }\n" +
		"  ]\n" +
		"}")
	translatedResponse := []byte("{\n" +
		"  \"status\":\"completed\",\n" +
		"  \"output_text\":\"Now applying the update with apply_patch:\",\n" +
		"  \"output\":[\n" +
		"    {\n" +
		"      \"type\":\"message\",\n" +
		"      \"role\":\"assistant\",\n" +
		"      \"content\":[{\"type\":\"output_text\",\"text\":\"Now applying the update with apply_patch:\"}]\n" +
		"    }\n" +
		"  ]\n" +
		"}")

	assert.False(t, shouldRetryMissingShellCall(req, translatedChatRequest, chatResponse, translatedResponse, buildToolWorkflowState(req)))
	assert.True(t, shouldRetryWeakPlaceholderFinal(req, translatedChatRequest, chatResponse, translatedResponse, buildToolWorkflowState(req)))
}

func TestShouldTreatUpdateFragmentAsAppend_DoesNotAppendStandaloneDocumentRewrite(t *testing.T) {
	existing := "<!doctype html><html><body><main>stress-suite</main></body></html>\n"
	fragment := "<!doctype html>\n<html><body><main>quiz home</main></body></html>\n"
	assert.False(t, shouldTreatUpdateFragmentAsAppend(existing, fragment))
}

func TestShouldRetryMissingShellCall_WhenWrongApplyPatchUsedInsteadOfReadingFile(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
	}
	translatedChatRequest := []byte(`{"tool_choice":"auto"}`)
	chatResponse := []byte(`{
		"choices":[
			{
				"finish_reason":"tool_calls",
				"message":{
					"content":"The diff didn't apply because I don't know the exact file contents. Let me read the file first.",
					"reasoning_content":"The diff didn't apply because I don't know the exact file contents. Let me read the file first.",
					"tool_calls":[
						{"function":{"name":"apply_patch","arguments":"{}"}},
						{"function":{"name":"apply_patch","arguments":"{}"}}
					]
				}
			}
		]
	}`)
	translatedResponse := []byte(`{
		"output":[
			{"type":"function_call","name":"apply_patch","arguments":"{}"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"apply_patch call was not executed because operation was invalid. Provide operation.type, operation.path, and non-empty diff/content for create/update."}]}
		]
	}`)

	assert.True(t, shouldRetryMissingShellCall(req, translatedChatRequest, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestTranslateResponsesToChatCompletionsRequest_WebSearchFollowupAfterQuestionKeepsTools(t *testing.T) {
	req := map[string]any{
		"model":       "gpt-5.2",
		"tool_choice": "auto",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "do a web_search for it"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "request_user_input"},
			map[string]any{"type": "web_search"},
			map[string]any{"type": "function", "name": "shell"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
			continue
		}
		names = append(names, strings.TrimSpace(fmt.Sprintf("%v", tool["name"])))
	}
	assert.Contains(t, names, "web_search")
}

func TestResponsesRequestToChatMessages_NormalizesApplyPatchToolOutputTranscript(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"create_file","path":"biology-quiz.html","content":"<html></html>"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Success. Updated the following files:\nA biology-quiz.html",
			},
		},
	}

	messages := responsesRequestToChatMessages(req)
	require.Len(t, messages, 2)
	assert.Equal(t, "tool", messages[1]["role"])
	assert.Equal(t, "Success. Updated the following files:\ncreated: biology-quiz.html", messages[1]["content"])
}

func TestRequestStillWantsStructuredPlan_FalseForImplementationRetry(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What platform?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"platform","value":"html"}]}`,
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "apply_patch call was not executed because operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "try again"},
				},
			},
		},
	}

	assert.False(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestStillWantsStructuredPlan_FalseForExplicitExplorationFollowup(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "do a web_search for it"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	assert.False(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestStillWantsStructuredPlan_FalseForWebResearchBeforePlanFollowup(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "write a native plan and do web research before"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	assert.False(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestExplicitlyWantsSearchIntent_FalseAfterResearchHistoryInProxyPlanMode(t *testing.T) {
	req := map[string]any{
		"mode": "plan",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Investigate the web for a first grade quiz game, define a plan. Use web search and return the final visible answer as exactly one <proposed_plan>...</proposed_plan> block."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_web_1",
				"arguments": `{"query":"first grade quiz game ideas"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_web_1",
				"output":  `{"query":"first grade quiz game ideas","results":[{"title":"A"}]}`,
			},
		},
	}

	workflowState := buildToolWorkflowState(req)
	assert.False(t, requestExplicitlyWantsSearchIntent(req, workflowState))
	assert.True(t, requestStillWantsStructuredPlan(req, workflowState))
}

func TestExtractPendingBridgeWebSearchCalls_ReturnsAllUnresolvedCalls(t *testing.T) {
	body := []byte(`{
		"status":"requires_action",
		"output":[
			{"type":"web_search_call","call_id":"call_1","name":"web_search","status":"completed","action":{"query":"alpha"}},
			{"type":"web_search_call_output","call_id":"call_1","output":"{\"results\":[{\"title\":\"A\"}]}"},
			{"type":"web_search_call","call_id":"call_2","name":"web_search","status":"in_progress","action":{"query":"beta"}},
			{"type":"web_search_call","call_id":"call_3","name":"web_search","status":"in_progress","action":{"query":"gamma"}}
		]
	}`)

	pending := extractPendingBridgeWebSearchCalls(body)
	require.Len(t, pending, 2)
	assert.Equal(t, "call_2", strings.TrimSpace(fmt.Sprintf("%v", pending[0]["call_id"])))
	assert.Equal(t, "call_3", strings.TrimSpace(fmt.Sprintf("%v", pending[1]["call_id"])))
}

func TestExtractPendingBridgeFileSearchCalls_ReturnsAllUnresolvedCalls(t *testing.T) {
	body := []byte(`{
		"status":"requires_action",
		"output":[
			{"type":"file_search_call","call_id":"call_1","name":"file_search","status":"completed","action":{"query":"alpha"}},
			{"type":"file_search_call_output","call_id":"call_1","output":"{\"matches\":[{\"path\":\"docs/a.md\"}]}"},
			{"type":"file_search_call","call_id":"call_2","name":"file_search","status":"in_progress","action":{"query":"beta"}},
			{"type":"file_search_call","call_id":"call_3","name":"file_search","status":"in_progress","action":{"query":"gamma"}}
		]
	}`)

	pending := extractPendingBridgeFileSearchCalls(body)
	require.Len(t, pending, 2)
	assert.Equal(t, "call_2", strings.TrimSpace(fmt.Sprintf("%v", pending[0]["call_id"])))
	assert.Equal(t, "call_3", strings.TrimSpace(fmt.Sprintf("%v", pending[1]["call_id"])))
}

func TestExecuteBridgeFileSearch_FindsWorkspaceMarker(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "docs", "teacher-notes.md"), []byte("# Teacher Notes\nReward marker: GOLDEN_APPLE_REWARD\nMascot marker: HAPPY_FOX_MASCOT\n"), 0o644))

	cwd, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(root))
	defer func() { _ = os.Chdir(cwd) }()

	result := executeBridgeFileSearch(map[string]any{
		"query":           "reward marker ./docs",
		"max_num_results": 3,
	})

	assert.Equal(t, true, result["ok"])
	var first map[string]any
	if typed, ok := result["matches"].([]map[string]any); ok {
		require.NotEmpty(t, typed)
		first = typed[0]
	} else {
		rawMatches, _ := result["matches"].([]any)
		require.NotEmpty(t, rawMatches)
		casted, ok := rawMatches[0].(map[string]any)
		require.True(t, ok)
		first = casted
	}
	assert.Contains(t, fmt.Sprintf("%v", first["path"]), "teacher-notes.md")
	assert.Contains(t, fmt.Sprintf("%v", first["excerpt"]), "GOLDEN_APPLE_REWARD")
}

func TestExecuteBridgeFileSearch_UsesRootOverrideWhenQueryOmitsPath(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "docs", "teacher-notes.md"), []byte("# Teacher Notes\nReward marker: GOLDEN_APPLE_REWARD\nMascot marker: HAPPY_FOX_MASCOT\n"), 0o644))

	result := executeBridgeFileSearch(map[string]any{
		"query":           "reward marker",
		"root":            filepath.Join(root, "docs"),
		"max_num_results": 3,
	})

	assert.Equal(t, true, result["ok"])
	assert.Contains(t, fmt.Sprintf("%v", result["root"]), filepath.ToSlash(filepath.Join(root, "docs")))
	var first map[string]any
	if typed, ok := result["matches"].([]map[string]any); ok {
		require.NotEmpty(t, typed)
		first = typed[0]
	} else {
		rawMatches, _ := result["matches"].([]any)
		require.NotEmpty(t, rawMatches)
		casted, ok := rawMatches[0].(map[string]any)
		require.True(t, ok)
		first = casted
	}
	assert.Contains(t, fmt.Sprintf("%v", first["path"]), "teacher-notes.md")
	assert.Contains(t, fmt.Sprintf("%v", first["excerpt"]), "GOLDEN_APPLE_REWARD")
}

func TestExecuteBridgeFileSearch_IgnoresNilRootFallback(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "docs", "teacher-notes.md"), []byte("Reward marker: GOLDEN_APPLE_REWARD\n"), 0o644))

	cwd, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(root))
	defer func() { _ = os.Chdir(cwd) }()

	result := executeBridgeFileSearch(map[string]any{
		"query": "reward marker ./docs",
	})

	assert.Equal(t, true, result["ok"])
	assert.NotContains(t, fmt.Sprintf("%v", result["root"]), "<nil>")
}

func TestExecuteBridgeFileSearch_IgnoresRootPathTermsInQuery(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(root, "tmp", "wsl_codex_stress_suite", "workspaces", "file_search_then_patch")
	require.NoError(t, os.MkdirAll(filepath.Join(workspace, "docs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(workspace, "docs", "teacher-notes.md"), []byte("# Teacher Notes\nReward marker: GOLDEN_APPLE_REWARD\nMascot marker: HAPPY_FOX_MASCOT\n"), 0o644))

	result := executeBridgeFileSearch(map[string]any{
		"query": "reward marker file_search_then_patch",
		"root":  workspace,
	})

	assert.Equal(t, true, result["ok"])
	var first map[string]any
	if typed, ok := result["matches"].([]map[string]any); ok {
		require.NotEmpty(t, typed)
		first = typed[0]
	} else {
		rawMatches, _ := result["matches"].([]any)
		require.NotEmpty(t, rawMatches)
		casted, ok := rawMatches[0].(map[string]any)
		require.True(t, ok)
		first = casted
	}
	assert.Contains(t, fmt.Sprintf("%v", first["path"]), "teacher-notes.md")
}

func TestExecuteBridgeFileSearch_IgnoresRootPathFragmentTermsInQuery(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(root, "tmp", "wsl_codex_stress_suite", "workspaces", "file_search_then_patch")
	require.NoError(t, os.MkdirAll(filepath.Join(workspace, "docs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(workspace, "docs", "teacher-notes.md"), []byte("# Teacher Notes\nReward marker: GOLDEN_APPLE_REWARD\nMascot marker: HAPPY_FOX_MASCOT\n"), 0o644))

	result := executeBridgeFileSearch(map[string]any{
		"query": "reward marker workspaces/file_search_then_patch",
		"root":  workspace,
	})

	assert.Equal(t, true, result["ok"])
	var first map[string]any
	if typed, ok := result["matches"].([]map[string]any); ok {
		require.NotEmpty(t, typed)
		first = typed[0]
	} else {
		rawMatches, _ := result["matches"].([]any)
		require.NotEmpty(t, rawMatches)
		casted, ok := rawMatches[0].(map[string]any)
		require.True(t, ok)
		first = casted
	}
	assert.Contains(t, fmt.Sprintf("%v", first["path"]), "teacher-notes.md")
}

func TestExtractFileSearchRootHintFromRequest_FindsWorkspacePath(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(root, "tmp", "wsl_codex_stress_suite", "workspaces", "file_search_then_patch")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	cwd, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(root))
	defer func() { _ = os.Chdir(cwd) }()

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use file_search to find the reward marker inside ./tmp/wsl_codex_stress_suite/workspaces/file_search_then_patch. Then use apply_patch to create ./tmp/wsl_codex_stress_suite/workspaces/file_search_then_patch/src/file-search-report.md.",
					},
				},
			},
		},
	}

	assert.Equal(t, "./tmp/wsl_codex_stress_suite/workspaces/file_search_then_patch", filepath.ToSlash(extractFileSearchRootHintFromRequest(req)))
}

func TestExtractChatToolCallsNamed_FileSearch(t *testing.T) {
	body := []byte(`{
	  "choices":[{
	    "message":{
	      "tool_calls":[
	        {"id":"call_file_1","type":"function","function":{"name":"file_search","arguments":"{\"query\":\"reward marker ./docs\"}"}}
	      ]
	    }
	  }]
	}`)

	calls := extractChatToolCallsNamed(body, "file_search")
	require.Len(t, calls, 1)
	assert.Equal(t, "call_file_1", calls[0]["call_id"])
	action := normalizeMapValue(calls[0]["action"])
	assert.Equal(t, "reward marker ./docs", fmt.Sprintf("%v", action["query"]))
}

func TestShouldDeferApplyPatchValidationForCurrentTurn_MixedSearchThenPatch(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Research the web for first grade quiz reward ideas, then use apply_patch to create src/research-brief.md.",
					},
				},
			},
		},
	}

	chatResponse := []byte(`{
	  "choices":[{"message":{"tool_calls":[
	    {"id":"call_web_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"first grade quiz rewards\"}"}}
	  ]}}]
	}`)
	responsesOutput := []byte(`{
	  "output":[
	    {"type":"web_search_call","id":"call_web_1","name":"web_search","arguments":"{\"query\":\"first grade quiz rewards\"}"}
	  ]
	}`)

	assert.True(t, shouldDeferApplyPatchValidationForCurrentTurn(req, ToolWorkflowState{}, chatResponse, responsesOutput))
}

func TestShouldDeferApplyPatchValidationForCurrentTurn_ApplyPatchTurnDoesNotDefer(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
			map[string]any{"type": "function", "name": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Research the web for first grade quiz reward ideas, then use apply_patch to create src/research-brief.md.",
					},
				},
			},
			map[string]any{"type": "web_search_call", "call_id": "call_web_1", "name": "web_search", "query": "first grade quiz reward ideas"},
			map[string]any{"type": "web_search_call_output", "call_id": "call_web_1", "output": "stars and badges"},
		},
	}
	workflowState := buildToolWorkflowState(req)
	chatResponse := []byte(`{
	  "choices":[{"message":{"tool_calls":[
	    {"id":"call_patch_1","type":"function","function":{"name":"apply_patch","arguments":"{\"operation\":{\"type\":\"create_file\",\"path\":\"src/research-brief.md\",\"content\":\"- stars\\n- badges\\n\"}}"}}
	  ]}}]
	}`)
	responsesOutput := []byte(`{
	  "output":[
	    {"type":"function_call","name":"apply_patch","call_id":"call_patch_1","arguments":"{\"operation\":{\"type\":\"create_file\",\"path\":\"src/research-brief.md\",\"content\":\"- stars\\n- badges\\n\"}}"}
	  ]
	}`)

	assert.False(t, shouldDeferApplyPatchValidationForCurrentTurn(req, workflowState, chatResponse, responsesOutput))
}

func TestRequestedWebSearchContinuationsBeforeMutation_TwoSearchesThenWrite(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
			map[string]any{"type": "apply_patch"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Use web_search at least twice with different queries, then create ./tmp/out.md with apply_patch and verify the saved file.",
					},
				},
			},
		},
	}

	assert.Equal(t, 2, requestedWebSearchContinuationsBeforeMutation(req))
}

func TestRequestedWebSearchContinuationsBeforeMutation_ResearchPlanOnlyReturnsZero(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Investigate the web at least twice and then write a detailed plan. Do not build the app yet.",
					},
				},
			},
		},
	}

	assert.Equal(t, 0, requestedWebSearchContinuationsBeforeMutation(req))
}

func TestClearWebSearchPhaseInstructions(t *testing.T) {
	req := map[string]any{
		"llamaswap_force_web_search_synthesis": true,
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "system",
				"content": []any{
					map[string]any{"type": "input_text", "text": "If current or external information is still needed, emit exactly one real web_search tool call next."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "system",
				"content": []any{
					map[string]any{"type": "input_text", "text": "You now have enough gathered web research context to answer."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Research, then create the file with apply_patch."},
				},
			},
		},
	}

	clearWebSearchPhaseInstructions(req)

	assert.NotContains(t, mustJSONString(req), "emit exactly one real web_search tool call next")
	assert.NotContains(t, mustJSONString(req), "enough gathered web research context")
	_, hasFlag := req["llamaswap_force_web_search_synthesis"]
	assert.False(t, hasFlag)
}

func TestTextIndicatesSearchIntent_TrueForInvestigateTheWeb(t *testing.T) {
	assert.True(t, textIndicatesSearchIntent("investigate the web for 10 best chemistry knowledge questions"))
}

func TestRequestStillWantsStructuredPlan_TrueForInvestigateTheWebAfterResearchAndAnsweredQuestionInCodexPlanMode(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_search_1",
				"arguments": `{"query":"10 most difficult chemistry riddles globalquiz.org questions answers"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_search_1",
				"output":  `{"results":[{"title":"10 most difficult chemistry riddles","url":"https://globalquiz.org/en/toughest-chemistry-riddles/"}]}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["How should the game be built?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"quiz_format":{"answers":["HTML/JS single file (Recommended)"]}}}`,
			},
		},
	}

	assert.True(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestShouldRetryMissingWebSearchCall_WhenOriginalRequestHasSearchToolButTranslatedTurnDoesNot(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "dig into the web for 10 chemistry knowledge questions"},
				},
			},
		},
	}

	translatedChatRequest := []byte(`{"tool_choice":"none","messages":[{"role":"user","content":"dig into the web for 10 chemistry knowledge questions"}]}`)
	chatResponse := []byte(`{
		"choices":[{"message":{"role":"assistant","content":"<websearch>\n<query>best chemistry quiz questions</query>\n</websearch>","reasoning_content":"I still need external research before answering."},"finish_reason":"stop"}]
	}`)
	translatedResponse := []byte(`{"output":[]}`)

	assert.True(t, shouldRetryMissingWebSearchCall(req, translatedChatRequest, chatResponse, translatedResponse, ToolWorkflowState{}))
}

func TestTranslateResponsesToChatCompletionsRequest_RemovesWebSearchWhenUserForbidsIt(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.4",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Define a short implementation plan for a first grade quiz game. Do not use web search or inspect the local workspace.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "web_search"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.False(t, chatRequestIncludesToolName(translated, "web_search"))
}

func TestTranslateResponsesToChatCompletionsRequest_DoesNotReAddApplyPatchBeforeRequiredWebSearch(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.4",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Research the web for first grade quiz reward ideas. Then create ./tmp/demo/research-notes.md with three bullet points using apply_patch.",
					},
				},
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "web_search", "external_web_access": true},
			map[string]any{"type": "function", "name": "exec_command"},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.True(t, chatRequestIncludesToolName(translated, "web_search"))
	assert.False(t, chatRequestIncludesToolName(translated, "apply_patch"))
}

func TestTranslateResponsesToChatCompletionsRequest_KeepsResearchThenPlanContinuationOnWebSearch(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.4",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Investigate the web for first grade quiz game ideas and then write a detailed implementation plan. Do not build the app yet.",
					},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_search_1",
				"arguments": `{"query":"first grade quiz game ideas for kids educational"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_search_1",
				"output":  `{"results":[{"title":"Quiz ideas","url":"https://example.com/quiz"}]}`,
			},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "exec_command"},
			map[string]any{"type": "function", "name": "write_stdin"},
			map[string]any{"type": "function", "name": "apply_patch"},
			map[string]any{"type": "web_search", "external_web_access": true},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	assert.True(t, chatRequestIncludesToolName(translated, "web_search"))
	assert.False(t, chatRequestIncludesToolName(translated, "exec_command"))
	assert.False(t, chatRequestIncludesToolName(translated, "apply_patch"))
}

func TestRequestStillWantsStructuredPlan_TrueForCodexPlanModeAfterAnsweredQuestion(t *testing.T) {
	req := map[string]any{
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "help me design a TBG ETUR quiz game"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	assert.True(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestExplicitlyWantsSearchIntent_FalseAfterResearchAndAnsweredQuestionInCodexPlanMode(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_search_1",
				"arguments": `{"query":"10 most difficult chemistry riddles globalquiz.org questions answers"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_search_1",
				"output":  `{"results":[{"title":"10 most difficult chemistry riddles","url":"https://globalquiz.org/en/toughest-chemistry-riddles/"}]}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["How should the game be built?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"quiz_format":{"answers":["HTML/JS single file (Recommended)"]}}}`,
			},
		},
	}

	assert.False(t, requestExplicitlyWantsSearchIntent(req, buildToolWorkflowState(req)))
	assert.True(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestExplicitlyWantsSearchIntent_FalseAfterBrowserResearchAndAnsweredQuestionInCodexPlanMode(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode."},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "mcp__playwright__browser_navigate",
				"call_id":   "call_browser_1",
				"arguments": `{"url":"https://globalquiz.org/en/toughest-chemistry-riddles/"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_browser_1",
				"output":  `{"page":"loaded"}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "mcp__playwright__browser_snapshot",
				"call_id":   "call_browser_2",
				"arguments": `{}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_browser_2",
				"output":  `{"snapshot":"chemistry riddles page"}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["How should the game be built?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"quiz_format":{"answers":["HTML/JS single file (Recommended)"]}}}`,
			},
		},
	}

	assert.False(t, requestExplicitlyWantsSearchIntent(req, buildToolWorkflowState(req)))
	assert.True(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestRequestExplicitlyWantsNativeCodexQuestion_IgnoresInstructionBoilerplate(t *testing.T) {
	req := map[string]any{
		"instructions": "Use the request_user_input tool when a native Codex question is explicitly requested.",
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "find all qwen 3.6 models in reddit webside"},
				},
			},
		},
	}

	assert.False(t, requestExplicitlyWantsNativeCodexQuestion(req))
}

func TestStructuredPlanOutputRequiredFromRequestBody_TrueForCodexPlanModeContract(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemistry questions and ask me how to build the quiz"},
				},
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)
	assert.True(t, structuredPlanOutputRequiredFromRequestBody(body))
	assert.Equal(t, "plan", extractResponsesRequestModeFromBody(body))
}

func TestShellToolArgumentsLookLikePlainQuestion_LiveShape(t *testing.T) {
	args := map[string]any{
		"command": []any{"What", "does", "synthesis", "gas", "(water", "gas)", "consist", "of?"},
	}

	assert.True(t, shellToolArgumentsLookLikePlainQuestion(args))
}

func TestTranslateResponsesToChatCompletionsRequest_TopLevelPlanInstructions_WebResearchBeforePlanKeepsTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{"type": "web_search_preview"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "write a native plan and do web research before"},
				},
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "web_search")
	assert.Contains(t, names, "request_user_input")
}

func TestTranslateResponsesToChatCompletionsRequest_CodexPlanAfterResearchAndAnsweredQuestionRequestsProposedPlan(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{"type": "web_search_preview"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_search_1",
				"arguments": `{"query":"10 most difficult chemistry riddles globalquiz.org questions answers"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_search_1",
				"output":  `{"results":[{"title":"10 most difficult chemistry riddles","url":"https://globalquiz.org/en/toughest-chemistry-riddles/"}]}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["How should the game be built?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"quiz_format":{"answers":["HTML/JS single file (Recommended)"]}}}`,
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "web_search")
	assert.Contains(t, names, "request_user_input")

	messages, ok := payload["messages"].([]any)
	require.True(t, ok)
	systemText := make([]string, 0, len(messages))
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		require.True(t, ok)
		if strings.TrimSpace(fmt.Sprintf("%v", msg["role"])) == "system" {
			systemText = append(systemText, strings.TrimSpace(fmt.Sprintf("%v", msg["content"])))
		}
	}
	joined := strings.Join(systemText, "\n")
	assert.Contains(t, joined, "Codex Plan Mode is still active.")
	assert.Contains(t, joined, "return exactly one <proposed_plan>...</proposed_plan> block now")
}

func TestTranslateResponsesToChatCompletionsRequest_CodexPlanAfterBrowserResearchAndAnsweredQuestionRequestsProposedPlan(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{
				"type":        "function",
				"name":        "mcp__playwright__browser_navigate",
				"description": "Navigate browser",
				"parameters":  map[string]any{"type": "object"},
			},
			map[string]any{
				"type":        "function",
				"name":        "mcp__playwright__browser_snapshot",
				"description": "Snapshot browser",
				"parameters":  map[string]any{"type": "object"},
			},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "investigate the web for 10 best chemie nolege questions, than write a small quiz game with it - ask me question about how to write it."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "mcp__playwright__browser_navigate",
				"call_id":   "call_browser_1",
				"arguments": `{"url":"https://globalquiz.org/en/toughest-chemistry-riddles/"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_browser_1",
				"output":  `{"page":"loaded"}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "mcp__playwright__browser_snapshot",
				"call_id":   "call_browser_2",
				"arguments": `{}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_browser_2",
				"output":  `{"snapshot":"chemistry riddles page"}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_question_1",
				"arguments": `{"questions":["How should the game be built?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_question_1",
				"output":  `{"answers":{"quiz_format":{"answers":["HTML/JS single file (Recommended)"]}}}`,
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	messages, ok := payload["messages"].([]any)
	require.True(t, ok)
	systemText := make([]string, 0, len(messages))
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		require.True(t, ok)
		if strings.TrimSpace(fmt.Sprintf("%v", msg["role"])) == "system" {
			systemText = append(systemText, strings.TrimSpace(fmt.Sprintf("%v", msg["content"])))
		}
	}
	joined := strings.Join(systemText, "\n")
	assert.Contains(t, joined, "Codex Plan Mode is still active.")
	assert.Contains(t, joined, "return exactly one <proposed_plan>...</proposed_plan> block now")
}

func TestTranslateResponsesToChatCompletionsRequest_TopLevelPlanInstructions_LaterRedditSearchKeepsTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{"type": "web_search_preview"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_ws_1",
				"arguments": `{"query":"ylab architects principles"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_ws_1",
				"output":  `{"query":"ylab architects principles","results":[{"title":"YLAB"}]}`,
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "find all qwen 3.6 models in reddit webside"},
				},
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "web_search")
}

func TestRequestStillWantsStructuredPlan_FalseForSearchRetryAfterPseudoSearchOutput(t *testing.T) {
	req := map[string]any{
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "<tool_code>\nprint(websearch(query=\"site:reddit.com qwen 3.6 model\"))\n</tool_code>"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "try again"},
				},
			},
		},
	}

	assert.False(t, requestStillWantsStructuredPlan(req, buildToolWorkflowState(req)))
}

func TestTranslateResponsesToChatCompletionsRequest_SearchRetryAfterPseudoSearchOutputKeepsTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       true,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{"type": "web_search_preview"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
			map[string]any{
				"type":      "function_call",
				"name":      "web_search",
				"call_id":   "call_ws_1",
				"arguments": `{"query":"site:reddit.com qwen 3.6 model"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_ws_1",
				"output":  `{"query":"site:reddit.com qwen 3.6 model","results":[]}`,
			},
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "<tool_code>\nprint(websearch(query=\"site:reddit.com qwen 3.6 model\"))\n</tool_code>"},
				},
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "try again"},
				},
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "web_search")
	assert.Contains(t, names, "request_user_input")
	assert.Contains(t, names, "shell")
}

func TestTranslateResponsesToChatCompletionsRequest_PlanResearchBeforeWritingPlanKeepsSearchTools(t *testing.T) {
	req := map[string]any{
		"model":        "gpt-5.2",
		"instructions": "<collaboration_mode># Plan Mode (Conversational)\nYou are in Plan Mode.",
		"tool_choice":  "auto",
		"stream":       false,
		"tools": []any{
			map[string]any{
				"type":        "function",
				"name":        "request_user_input",
				"description": "Ask structured questions",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"questions": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
					"required": []string{"questions"},
				},
			},
			map[string]any{"type": "web_search"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style should the game use?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "research current OpenAI responses API news before writing the plan"},
				},
			},
		},
	}

	body, err := json.Marshal(req)
	require.NoError(t, err)

	translated, err := translateResponsesToChatCompletionsRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(translated, &payload))

	assert.Equal(t, "auto", payload["tool_choice"])
	tools, ok := payload["tools"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, tools)
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		require.True(t, ok)
		if fn, ok := tool["function"].(map[string]any); ok {
			names = append(names, strings.TrimSpace(fmt.Sprintf("%v", fn["name"])))
		}
	}
	assert.Contains(t, names, "web_search")
	assert.Contains(t, names, "request_user_input")
	assert.Contains(t, names, "shell")
}

func TestShouldRecoverInvalidApplyPatchViaShell(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "apply_patch"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use apply_patch to recreate first-grade-quiz/index.html with the full app."},
				},
			},
		},
	}

	assert.True(t, shouldRecoverInvalidApplyPatchViaShell(req, ToolWorkflowState{}, "empty_operation"))
	assert.True(t, shouldRecoverInvalidApplyPatchViaShell(req, ToolWorkflowState{}, "wrong_tool_call"))
	assert.False(t, shouldRecoverInvalidApplyPatchViaShell(req, ToolWorkflowState{ApplyPatchSatisfied: true}, "empty_operation"))
	assert.False(t, shouldRecoverInvalidApplyPatchViaShell(map[string]any{
		"tools": []any{map[string]any{"type": "apply_patch"}},
		"input": req["input"],
	}, ToolWorkflowState{}, "empty_operation"))
}

func TestShouldEnableStrictApplyPatchIntent_FalseForPolicyInstruction(t *testing.T) {
	req := map[string]any{
		"tools": []any{
			map[string]any{"type": "apply_patch"},
			map[string]any{"type": "shell"},
		},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "If an apply_patch fails due to file drift, re-read the current file, reconcile the patch against the latest contents, reapply safely, and verify the final file state before continuing."},
				},
			},
		},
	}

	assert.False(t, shouldEnableStrictApplyPatchIntent(req, nil))
}

func TestRequestMapContainsApplyPatchToolOutput_NativeApplyPatchCallOutput(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":    "apply_patch_call_output",
				"call_id": "call_patch_1",
				"output":  `{"ok":true}`,
			},
		},
	}

	assert.True(t, requestMapContainsApplyPatchToolOutput(req))
}

func TestNormalizeResponsesRequest_DoesNotInjectQwenToolPolicyForNonQwenModels(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"input":[
			{"type":"message","role":"system","content":"You are a coding agent."},
			{"type":"message","role":"user","content":"Edit app.js."}
		],
		"tools":[
			{"type":"shell"},
			{"type":"apply_patch"}
		]
	}`)

	normalized, adapted, unsupported, err := normalizeResponsesRequest(body)
	assert.NoError(t, err)
	assert.ElementsMatch(t, []string{"shell", "apply_patch"}, adapted)
	assert.Empty(t, unsupported)

	var payload map[string]any
	assert.NoError(t, json.Unmarshal(normalized, &payload))

	input, ok := payload["input"].([]any)
	if !assert.True(t, ok) || !assert.Len(t, input, 2) {
		return
	}

	first, ok := input[0].(map[string]any)
	if !assert.True(t, ok) {
		return
	}

	content, ok := first["content"].(string)
	if !assert.True(t, ok) {
		return
	}

	assert.Equal(t, "You are a coding agent.", content)
}

func TestProxyManager_ResponsesStartsModelWithComputerUsePreviewTool(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		LogLevel:           "error",
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)
	proxy.Lock()
	proxy.transformModes["model1"] = TransformModeResponses
	proxy.Unlock()

	reqBody := `{"model":"model1","input":"hello","tools":[{"type":"computer_use_preview"}]}`
	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewBufferString(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	processGroup := proxy.findGroupByModelName("model1")
	if assert.NotNil(t, processGroup) {
		process, ok := processGroup.GetMember("model1")
		if assert.True(t, ok) {
			assert.Equal(t, StateReady, process.CurrentState())
		}
	}
}

func TestProxyManager_HealthEndpoint(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)
	req := httptest.NewRequest("GET", "/health", nil)
	rec := CreateTestResponseRecorder()
	proxy.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "OK", rec.Body.String())
}

// Ensure the custom llama-server /completion endpoint proxies correctly
func TestProxyManager_CompletionEndpoint(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	reqBody := `{"model":"model1"}`
	req := httptest.NewRequest("POST", "/completion", bytes.NewBufferString(reqBody))
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "model1")
}

func TestProxyManager_StartupHooks(t *testing.T) {

	// using real YAML as the configuration has gotten more complex
	// is the right approach as LoadConfigFromReader() does a lot more
	// than parse YAML now. Eventually migrate all tests to use this approach
	configStr := strings.Replace(`
logLevel: error
hooks:
  on_startup:
    preload:
      - model1
      - model2
groups:
  preloadTestGroup:
    swap: false
    members:
       - model1
       - model2
models:
  model1:
    cmd: ${simpleresponderpath} --port ${PORT} --silent --respond model1
  model2:
      cmd: ${simpleresponderpath} --port ${PORT} --silent --respond model2
`, "${simpleresponderpath}", simpleResponderPath, -1)

	// Create a test model configuration
	config, err := config.LoadConfigFromReader(strings.NewReader(configStr))
	if !assert.NoError(t, err, "Invalid configuration") {
		return
	}

	preloadChan := make(chan ModelPreloadedEvent, 2) // buffer for 2 expected events

	unsub := event.On(func(e ModelPreloadedEvent) {
		preloadChan <- e
	})

	defer unsub()

	// Create the proxy which should trigger preloading
	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	for i := 0; i < 2; i++ {
		select {
		case <-preloadChan:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for models to preload")
		}
	}
	// make sure they are both loaded
	_, foundGroup := proxy.processGroups["preloadTestGroup"]
	if !assert.True(t, foundGroup, "preloadTestGroup should exist") {
		return
	}
	assert.Equal(t, StateReady, proxy.processGroups["preloadTestGroup"].processes["model1"].CurrentState())
	assert.Equal(t, StateReady, proxy.processGroups["preloadTestGroup"].processes["model2"].CurrentState())
}

func TestProxyManager_StreamingEndpointsReturnNoBufferingHeader(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1":       getTestSimpleResponderConfig("model1"),
			"author/model": getTestSimpleResponderConfig("author/model"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	endpoints := []string{
		"/api/events",
		"/logs/stream",
		"/logs/stream/proxy",
		"/logs/stream/upstream",
		"/logs/stream/author/model",
	}

	for _, endpoint := range endpoints {
		t.Run(endpoint, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()

			req := httptest.NewRequest("GET", endpoint, nil)
			req = req.WithContext(ctx)
			rec := CreateTestResponseRecorder()

			// Run handler in goroutine and wait for context timeout
			done := make(chan struct{})
			go func() {
				defer close(done)
				proxy.ServeHTTP(rec, req)
			}()

			// Wait for either the handler to complete or context to timeout
			<-ctx.Done()

			// At this point, the handler has either finished or been cancelled
			// Wait for the goroutine to fully exit before reading
			<-done

			// Now it's safe to read from rec - no more concurrent writes
			assert.Equal(t, http.StatusOK, rec.Code)
			assert.Equal(t, "no", rec.Header().Get("X-Accel-Buffering"))
		})
	}
}

func TestProxyManager_ProxiedStreamingEndpointReturnsNoBufferingHeader(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"streaming-model": getTestSimpleResponderConfig("streaming-model"),
		},
		LogLevel: "error",
	})

	proxy := New(config)
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	// Make a streaming request
	reqBody := `{"model":"streaming-model"}`
	// simple-responder will return text/event-stream when stream=true is in the query
	req := httptest.NewRequest("POST", "/v1/chat/completions?stream=true", bytes.NewBufferString(reqBody))
	rec := CreateTestResponseRecorder()

	proxy.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "no", rec.Header().Get("X-Accel-Buffering"))
	assert.Contains(t, rec.Header().Get("Content-Type"), "text/event-stream")
}

func TestProxyManager_ApiGetVersion(t *testing.T) {
	config := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	// Version test map
	versionTest := map[string]string{
		"build_date": "1970-01-01T00:00:00Z",
		"commit":     "cc915ddb6f04a42d9cd1f524e1d46ec6ed069fdc",
		"version":    "v001",
	}

	proxy := New(config)
	proxy.SetVersion(versionTest["build_date"], versionTest["commit"], versionTest["version"])
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	req := httptest.NewRequest("GET", "/api/version", nil)
	w := CreateTestResponseRecorder()

	proxy.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	// Ensure json response
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))

	// Check for attributes
	response := map[string]string{}
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	for key, value := range versionTest {
		assert.Equal(t, value, response[key], "%s value %s should match response %s", key, value, response[key])
	}
}

func TestProxyManager_APIKeyAuth(t *testing.T) {
	testConfig := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		RequiredAPIKeys: []string{"valid-key-1", "valid-key-2"},
		LogLevel:        "error",
	})

	proxy := New(testConfig)
	defer proxy.StopProcesses(StopImmediately)

	t.Run("valid key in x-api-key header", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		req.Header.Set("x-api-key", "valid-key-1")
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("valid key in Authorization Bearer header", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		req.Header.Set("Authorization", "Bearer valid-key-2")
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("both headers with matching keys", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		req.Header.Set("x-api-key", "valid-key-1")
		req.Header.Set("Authorization", "Bearer valid-key-1")
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("invalid key returns 401", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		req.Header.Set("x-api-key", "invalid-key")
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusUnauthorized, w.Code)
		assert.Contains(t, w.Body.String(), "unauthorized")
	})

	t.Run("missing key returns 401", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusUnauthorized, w.Code)
	})

	t.Run("valid key in Basic Auth header", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		// Basic Auth: base64("anyuser:valid-key-1")
		credentials := base64.StdEncoding.EncodeToString([]byte("anyuser:valid-key-1"))
		req.Header.Set("Authorization", "Basic "+credentials)
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("invalid key in Basic Auth header returns 401", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		credentials := base64.StdEncoding.EncodeToString([]byte("anyuser:wrong-key"))
		req.Header.Set("Authorization", "Basic "+credentials)
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusUnauthorized, w.Code)
		assert.Contains(t, w.Body.String(), "unauthorized")
	})

	t.Run("x-api-key and Basic Auth with matching keys", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		req.Header.Set("x-api-key", "valid-key-1")
		credentials := base64.StdEncoding.EncodeToString([]byte("user:valid-key-1"))
		req.Header.Set("Authorization", "Basic "+credentials)
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("401 response includes WWW-Authenticate header", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusUnauthorized, w.Code)
		assert.Equal(t, `Basic realm="llama-swap"`, w.Header().Get("WWW-Authenticate"))
	})
}

func TestProxyManager_APIKeyAuth_Disabled(t *testing.T) {
	// Config without RequiredAPIKeys - auth should be disabled
	testConfig := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		LogLevel: "error",
	})

	proxy := New(testConfig)
	defer proxy.StopProcesses(StopImmediately)

	t.Run("requests pass without API key when not configured", func(t *testing.T) {
		reqBody := `{"model":"model1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})
}

// TestProxyManager_PeerProxy_InferenceHandler tests the peerProxy integration
// in proxyInferenceHandler for issue #433
func TestProxyManager_PeerProxy_InferenceHandler(t *testing.T) {
	t.Run("requests to peer models are proxied", func(t *testing.T) {
		// Create a test server to act as the peer
		peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"response":"from-peer","model":"peer-model"}`))
		}))
		defer peerServer.Close()

		// Create config with peers but no local model for "peer-model"
		configStr := fmt.Sprintf(`
logLevel: error
peers:
  test-peer:
    proxy: %s
    models:
      - peer-model
models:
  local-model:
    cmd: %s -port ${PORT} -silent -respond local-model
`, peerServer.URL, getSimpleResponderPath())

		testConfig, err := config.LoadConfigFromReader(strings.NewReader(configStr))
		assert.NoError(t, err)

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		reqBody := `{"model":"peer-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "from-peer")
	})

	t.Run("local models take precedence over peer models", func(t *testing.T) {
		// Create a test server to act as the peer - should NOT be called
		peerCalled := false
		peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			peerCalled = true
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"response":"from-peer"}`))
		}))
		defer peerServer.Close()

		// Create config where "shared-model" exists both locally and on peer
		configStr := fmt.Sprintf(`
logLevel: error
peers:
  test-peer:
    proxy: %s
    models:
      - shared-model
models:
  shared-model:
    cmd: %s -port ${PORT} -silent -respond local-response
`, peerServer.URL, getSimpleResponderPath())

		testConfig, err := config.LoadConfigFromReader(strings.NewReader(configStr))
		assert.NoError(t, err)

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		reqBody := `{"model":"shared-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "local-response")
		assert.False(t, peerCalled, "peer should not be called when local model exists")
	})

	t.Run("unknown model returns error", func(t *testing.T) {
		// Create a test server to act as the peer
		peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer peerServer.Close()

		configStr := fmt.Sprintf(`
logLevel: error
peers:
  test-peer:
    proxy: %s
    models:
      - peer-model
models:
  local-model:
    cmd: %s -port ${PORT} -silent -respond local-model
`, peerServer.URL, getSimpleResponderPath())

		testConfig, err := config.LoadConfigFromReader(strings.NewReader(configStr))
		assert.NoError(t, err)

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		reqBody := `{"model":"unknown-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "could not find suitable inference handler")
	})

	t.Run("peer API key is injected into request", func(t *testing.T) {
		var receivedAuthHeader string
		peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedAuthHeader = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"response":"ok"}`))
		}))
		defer peerServer.Close()

		configStr := fmt.Sprintf(`
logLevel: error
peers:
  test-peer:
    proxy: %s
    apiKey: secret-peer-key
    models:
      - peer-model
models:
  local-model:
    cmd: %s -port ${PORT} -silent -respond local-model
`, peerServer.URL, getSimpleResponderPath())

		testConfig, err := config.LoadConfigFromReader(strings.NewReader(configStr))
		assert.NoError(t, err)

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		reqBody := `{"model":"peer-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Equal(t, "Bearer secret-peer-key", receivedAuthHeader)
	})

	t.Run("no peers configured - unknown model returns error", func(t *testing.T) {
		testConfig := config.AddDefaultGroupToConfig(config.Config{
			HealthCheckTimeout: 15,
			Models: map[string]config.ModelConfig{
				"local-model": getTestSimpleResponderConfig("local-model"),
			},
			LogLevel: "error",
		})

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		// peerProxy exists but has no peer models configured
		assert.False(t, proxy.peerProxy.HasPeerModel("unknown-model"))

		reqBody := `{"model":"unknown-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "could not find suitable inference handler")
	})

	t.Run("peer streaming response sets X-Accel-Buffering header", func(t *testing.T) {
		peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("data: test\n\n"))
		}))
		defer peerServer.Close()

		configStr := fmt.Sprintf(`
logLevel: error
peers:
  test-peer:
    proxy: %s
    models:
      - peer-model
models:
  local-model:
    cmd: %s -port ${PORT} -silent -respond local-model
`, peerServer.URL, getSimpleResponderPath())

		testConfig, err := config.LoadConfigFromReader(strings.NewReader(configStr))
		assert.NoError(t, err)

		proxy := New(testConfig)
		defer proxy.StopProcesses(StopImmediately)

		reqBody := `{"model":"peer-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
		w := CreateTestResponseRecorder()

		proxy.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Equal(t, "no", w.Header().Get("X-Accel-Buffering"))
	})
}
