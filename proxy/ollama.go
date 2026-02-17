package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const ollamaPrefix = "ollama/"

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaShowResponse struct {
	Modelfile string         `json:"modelfile"`
	ModelInfo map[string]any `json:"model_info"`
}

func isOllamaModelID(modelID string) bool {
	return strings.HasPrefix(strings.TrimSpace(modelID), ollamaPrefix)
}

func ollamaModelID(name string) string {
	return ollamaPrefix + strings.TrimSpace(name)
}

func (pm *ProxyManager) GetOllamaModels() []OllamaModel {
	_ = pm.refreshOllamaModels(false)

	pm.Lock()
	defer pm.Unlock()

	out := make([]OllamaModel, 0, len(pm.ollamaModels))
	for _, model := range pm.ollamaModels {
		out = append(out, model)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out
}

func (pm *ProxyManager) GetOllamaModelByID(modelID string) (OllamaModel, bool) {
	pm.Lock()
	model, found := pm.ollamaModels[modelID]
	pm.Unlock()
	if found {
		return model, true
	}

	if !isOllamaModelID(modelID) {
		return OllamaModel{}, false
	}
	_ = pm.refreshOllamaModels(true)

	pm.Lock()
	defer pm.Unlock()
	model, found = pm.ollamaModels[modelID]
	return model, found
}

func (pm *ProxyManager) refreshOllamaModels(force bool) error {
	pm.Lock()
	if !force && time.Since(pm.ollamaLastRefresh) < 10*time.Second {
		pm.Unlock()
		return nil
	}
	pm.Unlock()

	var tags ollamaTagsResponse
	endpoint, err := pm.fetchOllamaTags(&tags)
	if err != nil {
		return err
	}

	next := make(map[string]OllamaModel, len(tags.Models))
	for _, item := range tags.Models {
		name := strings.TrimSpace(item.Name)
		if name == "" {
			continue
		}
		ctxRef := pm.fetchOllamaCtxReference(name)
		modelID := ollamaModelID(name)
		next[modelID] = OllamaModel{
			ID:           modelID,
			Name:         name,
			CtxReference: ctxRef,
		}
	}

	pm.Lock()
	pm.ollamaEndpoint = endpoint
	pm.ollamaModels = next
	pm.ollamaLastRefresh = time.Now()
	for id, model := range next {
		if model.CtxReference <= 0 {
			continue
		}
		if _, exists := pm.ctxSizes[id]; !exists {
			pm.ctxSizes[id] = model.CtxReference
		}
	}
	pm.Unlock()
	return nil
}

func (pm *ProxyManager) fetchOllamaTags(tags *ollamaTagsResponse) (string, error) {
	client := &http.Client{Timeout: 2 * time.Second}
	var lastErr error
	for _, endpoint := range pm.ollamaEndpoints() {
		req, err := http.NewRequest(http.MethodGet, strings.TrimSuffix(endpoint, "/")+"/api/tags", nil)
		if err != nil {
			lastErr = err
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			lastErr = fmt.Errorf("ollama tags endpoint status %d", resp.StatusCode)
			resp.Body.Close()
			continue
		}

		decodeErr := json.NewDecoder(resp.Body).Decode(tags)
		resp.Body.Close()
		if decodeErr != nil {
			lastErr = decodeErr
			continue
		}
		return endpoint, nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("ollama not reachable")
	}
	return "", lastErr
}

func (pm *ProxyManager) ollamaEndpoints() []string {
	pm.Lock()
	base := strings.TrimSpace(pm.ollamaEndpoint)
	pm.Unlock()

	if base == "" {
		base = "http://127.0.0.1:11434"
	}

	seen := map[string]bool{}
	var out []string
	add := func(raw string) {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			return
		}
		if !strings.Contains(raw, "://") {
			raw = "http://" + raw
		}
		raw = strings.TrimSuffix(raw, "/")
		if seen[raw] {
			return
		}
		seen[raw] = true
		out = append(out, raw)
	}

	add(base)
	add(os.Getenv("LLAMASWAP_OLLAMA_ENDPOINT"))
	add(os.Getenv("OLLAMA_HOST"))

	if runtime.GOOS == "linux" && (strings.Contains(base, "127.0.0.1") || strings.Contains(base, "localhost")) {
		if content, err := os.ReadFile("/etc/resolv.conf"); err == nil {
			for _, line := range strings.Split(string(content), "\n") {
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "nameserver ") {
					continue
				}
				fields := strings.Fields(line)
				if len(fields) >= 2 && fields[1] != "" {
					add("http://" + fields[1] + ":11434")
				}
				break
			}
		}
	}

	return out
}

func (pm *ProxyManager) fetchOllamaCtxReference(modelName string) int {
	payload, _ := json.Marshal(map[string]any{"model": modelName})
	req, err := http.NewRequest(http.MethodPost, strings.TrimSuffix(pm.ollamaEndpoint, "/")+"/api/show", bytes.NewReader(payload))
	if err != nil {
		return 0
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0
	}

	var show ollamaShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return 0
	}

	for key, value := range show.ModelInfo {
		lower := strings.ToLower(key)
		if !strings.Contains(lower, "context_length") && !strings.Contains(lower, "num_ctx") {
			continue
		}
		if intVal, ok := anyToInt(value); ok && intVal > 0 {
			return intVal
		}
	}

	lines := strings.Split(show.Modelfile, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(strings.ToUpper(trimmed), "PARAMETER ") {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) >= 3 && strings.EqualFold(fields[1], "num_ctx") {
			if n, err := strconv.Atoi(fields[2]); err == nil && n > 0 {
				return n
			}
		}
	}
	return 0
}

func anyToInt(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return 0, false
		}
		return n, true
	default:
		return 0, false
	}
}

func (pm *ProxyManager) proxyOllamaRequest(modelID string, w http.ResponseWriter, r *http.Request) error {
	targetURL := strings.TrimSuffix(pm.ollamaEndpoint, "/") + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	req, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, r.Body)
	if err != nil {
		return err
	}
	req.Header = r.Header.Clone()
	req.Host = ""

	resp, err := pm.ollamaClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(resp.StatusCode)

	_, copyErr := io.Copy(w, resp.Body)
	return copyErr
}
