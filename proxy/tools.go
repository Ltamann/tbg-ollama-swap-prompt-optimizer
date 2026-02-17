package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

type RuntimeToolType string
type RuntimeToolPolicy string

const (
	RuntimeToolHTTP RuntimeToolType = "http"
	RuntimeToolMCP  RuntimeToolType = "mcp"

	ToolPolicyAuto     RuntimeToolPolicy = "auto"
	ToolPolicyAlways   RuntimeToolPolicy = "always"
	ToolPolicyWatchdog RuntimeToolPolicy = "watchdog"
	ToolPolicyNever    RuntimeToolPolicy = "never"
)

type ToolRuntimeSettings struct {
	Enabled                bool   `json:"enabled"`
	WebSearchMode          string `json:"webSearchMode"` // off|auto|force
	WatchdogMode           string `json:"watchdogMode"`  // off|auto
	RequireApprovalHeader  bool   `json:"requireApprovalHeader"`
	ApprovalHeaderName     string `json:"approvalHeaderName"`
	BlockNonLocalEndpoints bool   `json:"blockNonLocalEndpoints"`
	MaxToolRounds          int    `json:"maxToolRounds"`
	KillPreviousOnSwap     bool   `json:"killPreviousOnSwap"`
	MaxRunningModels       int    `json:"maxRunningModels"`
}

type RuntimeTool struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	Type            RuntimeToolType   `json:"type"`
	Endpoint        string            `json:"endpoint"`
	Enabled         bool              `json:"enabled"`
	Description     string            `json:"description,omitempty"`
	RemoteName      string            `json:"remoteName,omitempty"`
	Policy          RuntimeToolPolicy `json:"policy,omitempty"` // auto|always|watchdog|never
	RequireApproval bool              `json:"requireApproval,omitempty"`
	TimeoutSeconds  int               `json:"timeoutSeconds,omitempty"`
}

type ToolApprovalCall struct {
	Name   string         `json:"name"`
	CallID string         `json:"call_id,omitempty"`
	Args   map[string]any `json:"args,omitempty"`
}

type ToolApprovalRequiredError struct {
	HeaderName string             `json:"header_name"`
	ToolCalls  []ToolApprovalCall `json:"tool_calls"`
}

func (e *ToolApprovalRequiredError) Error() string {
	if e == nil {
		return "tool approval required"
	}
	return fmt.Sprintf("tool approval required for %d call(s)", len(e.ToolCalls))
}

type toolsDiskState struct {
	Settings ToolRuntimeSettings `json:"settings"`
	Tools    []RuntimeTool       `json:"tools"`
}

func defaultToolRuntimeSettings() ToolRuntimeSettings {
	return ToolRuntimeSettings{
		Enabled:                true,
		WebSearchMode:          "auto",
		WatchdogMode:           "off",
		RequireApprovalHeader:  false,
		ApprovalHeaderName:     "X-LlamaSwap-Tool-Approval",
		BlockNonLocalEndpoints: true,
		MaxToolRounds:          4,
		KillPreviousOnSwap:     true,
		MaxRunningModels:       1,
	}
}

func normalizeToolRuntimeSettings(in ToolRuntimeSettings) ToolRuntimeSettings {
	out := in
	out.WebSearchMode = strings.ToLower(strings.TrimSpace(out.WebSearchMode))
	if out.WebSearchMode != "off" && out.WebSearchMode != "auto" && out.WebSearchMode != "force" {
		out.WebSearchMode = "auto"
	}
	out.WatchdogMode = strings.ToLower(strings.TrimSpace(out.WatchdogMode))
	if out.WatchdogMode != "off" && out.WatchdogMode != "auto" {
		out.WatchdogMode = "auto"
	}
	if strings.TrimSpace(out.ApprovalHeaderName) == "" {
		out.ApprovalHeaderName = "X-LlamaSwap-Tool-Approval"
	}
	if out.MaxToolRounds <= 0 {
		out.MaxToolRounds = 4
	}
	if out.MaxToolRounds > 16 {
		out.MaxToolRounds = 16
	}
	if out.MaxRunningModels <= 0 {
		out.MaxRunningModels = 1
	}
	if out.MaxRunningModels > 64 {
		out.MaxRunningModels = 64
	}
	return out
}

func normalizeRuntimeTool(t RuntimeTool) RuntimeTool {
	t.ID = strings.TrimSpace(t.ID)
	t.Name = strings.TrimSpace(t.Name)
	t.Endpoint = strings.TrimSpace(t.Endpoint)
	t.Description = strings.TrimSpace(t.Description)
	t.RemoteName = strings.TrimSpace(t.RemoteName)
	switch strings.ToLower(strings.TrimSpace(string(t.Policy))) {
	case string(ToolPolicyAlways):
		t.Policy = ToolPolicyAlways
	case string(ToolPolicyWatchdog):
		t.Policy = ToolPolicyWatchdog
	case string(ToolPolicyNever):
		t.Policy = ToolPolicyNever
	default:
		t.Policy = ToolPolicyAuto
	}
	return t
}

func (pm *ProxyManager) toolsFilePath() string {
	cfg := strings.TrimSpace(pm.configPath)
	if cfg == "" {
		return "tools.json"
	}
	dir := filepath.Dir(cfg)
	return filepath.Join(dir, "tools.json")
}

func (pm *ProxyManager) loadToolsFromDisk() {
	path := pm.toolsFilePath()
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}

	settings := defaultToolRuntimeSettings()
	tools := []RuntimeTool{}

	var state toolsDiskState
	if err := json.Unmarshal(b, &state); err == nil && (len(state.Tools) > 0 || state.Settings != (ToolRuntimeSettings{})) {
		settings = normalizeToolRuntimeSettings(state.Settings)
		if !gjson.GetBytes(b, "settings.killPreviousOnSwap").Exists() {
			settings.KillPreviousOnSwap = true
		}
		if !gjson.GetBytes(b, "settings.maxRunningModels").Exists() {
			settings.MaxRunningModels = 1
		}
		if !gjson.GetBytes(b, "settings.watchdogMode").Exists() {
			settings.WatchdogMode = "off"
		}
		tools = state.Tools
	} else {
		var legacyTools []RuntimeTool
		if err := json.Unmarshal(b, &legacyTools); err != nil {
			pm.proxyLogger.Warnf("failed to parse tools file %s: %v", path, err)
			return
		}
		tools = legacyTools
	}
	for i := range tools {
		tools[i] = normalizeRuntimeTool(tools[i])
	}

	pm.Lock()
	pm.toolSettings = settings
	pm.tools = tools
	pm.Unlock()
}

func (pm *ProxyManager) saveToolsToDisk() error {
	path := pm.toolsFilePath()
	pm.Lock()
	toolsCopy := append([]RuntimeTool(nil), pm.tools...)
	settingsCopy := pm.toolSettings
	pm.Unlock()

	state := toolsDiskState{
		Settings: settingsCopy,
		Tools:    toolsCopy,
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (pm *ProxyManager) getToolRuntimeSettings() ToolRuntimeSettings {
	pm.Lock()
	defer pm.Unlock()
	return pm.toolSettings
}

func (pm *ProxyManager) getEnabledTools() []RuntimeTool {
	pm.Lock()
	defer pm.Unlock()
	out := make([]RuntimeTool, 0, len(pm.tools))
	if !pm.toolSettings.Enabled {
		return out
	}
	for _, t := range pm.tools {
		t = normalizeRuntimeTool(t)
		if t.Enabled && t.Policy != ToolPolicyNever && t.Name != "" && t.Endpoint != "" {
			out = append(out, t)
		}
	}
	return out
}

func (pm *ProxyManager) toolByName(name string) (RuntimeTool, bool) {
	pm.Lock()
	defer pm.Unlock()
	if !pm.toolSettings.Enabled {
		return RuntimeTool{}, false
	}
	for _, t := range pm.tools {
		t = normalizeRuntimeTool(t)
		if t.Enabled && t.Policy != ToolPolicyNever && strings.EqualFold(t.Name, strings.TrimSpace(name)) {
			return t, true
		}
	}
	return RuntimeTool{}, false
}

func (pm *ProxyManager) toolSchemas() []map[string]any {
	tools := pm.getEnabledTools()
	result := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		description := strings.TrimSpace(t.Description)
		if description == "" {
			description = fmt.Sprintf("Tool endpoint: %s", t.Endpoint)
		}
		parameters := toolParametersSchema(t)
		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": description,
				"parameters":  parameters,
			},
		})
	}
	return result
}

func toolParametersSchema(t RuntimeTool) map[string]any {
	// HTTP tools keep query compatibility but also allow named placeholders.
	if t.Type == RuntimeToolHTTP {
		return map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "Primary search/input text. Also used for {query} placeholder.",
				},
			},
			"additionalProperties": true,
		}
	}

	// MCP tools can run in two modes:
	// - fixed remoteName: pass arguments directly
	// - gateway mode: caller provides remote tool name and arguments object
	if strings.TrimSpace(t.RemoteName) != "" {
		return map[string]any{
			"type":                 "object",
			"additionalProperties": true,
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "Optional free-text input for the remote MCP tool.",
				},
			},
		}
	}

	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "Remote MCP tool name to execute (for example: browser_navigate).",
			},
			"arguments": map[string]any{
				"type":                 "object",
				"description":          "Arguments object for the selected MCP tool.",
				"additionalProperties": true,
			},
		},
		"required": []string{"name"},
	}
}

func (pm *ProxyManager) executeToolCall(toolName string, args map[string]any, headers http.Header) (string, error) {
	tool, ok := pm.toolByName(toolName)
	if !ok {
		return "", fmt.Errorf("tool %s not found", toolName)
	}
	settings := pm.getToolRuntimeSettings()
	if !settings.Enabled {
		return "", fmt.Errorf("tool runtime disabled")
	}
	if required, headerName := toolApprovalRequired(tool, settings, headers); required {
		return "", fmt.Errorf("tool %s requires approval header %s=true", toolName, headerName)
	}
	if err := validateToolEndpoint(tool.Endpoint, settings); err != nil {
		return "", err
	}

	timeout := tool.TimeoutSeconds
	if timeout <= 0 {
		if tool.Type == RuntimeToolMCP {
			timeout = 30
		} else {
			timeout = 20
		}
	}
	start := time.Now()
	switch tool.Type {
	case RuntimeToolHTTP:
		out, err := pm.executeHTTPTool(tool, args, timeout)
		errMsg := ""
		if err != nil {
			errMsg = err.Error()
		}
		pm.proxyLogger.Infof("tool call name=%s type=%s duration_ms=%d err=%v err_msg=%q", tool.Name, tool.Type, time.Since(start).Milliseconds(), err != nil, errMsg)
		return out, err
	case RuntimeToolMCP:
		out, err := pm.executeMCPTool(tool, args, timeout)
		errMsg := ""
		if err != nil {
			errMsg = err.Error()
		}
		pm.proxyLogger.Infof("tool call name=%s type=%s duration_ms=%d err=%v err_msg=%q", tool.Name, tool.Type, time.Since(start).Milliseconds(), err != nil, errMsg)
		return out, err
	default:
		return "", fmt.Errorf("unsupported tool type %s", tool.Type)
	}
}

func toolApprovalRequired(tool RuntimeTool, settings ToolRuntimeSettings, headers http.Header) (bool, string) {
	if !(tool.RequireApproval || settings.RequireApprovalHeader) {
		return false, settings.ApprovalHeaderName
	}
	headerName := strings.TrimSpace(settings.ApprovalHeaderName)
	if headerName == "" {
		headerName = "X-LlamaSwap-Tool-Approval"
	}
	val := strings.ToLower(strings.TrimSpace(headers.Get(headerName)))
	if val == "1" || val == "true" || val == "yes" || val == "on" {
		return false, headerName
	}
	return true, headerName
}

func isTruthyHeader(headers http.Header, key string) bool {
	v := strings.ToLower(strings.TrimSpace(headers.Get(key)))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func asMap(v any) (map[string]any, bool) {
	if v == nil {
		return nil, false
	}
	if m, ok := v.(map[string]any); ok {
		return m, true
	}
	if m, ok := v.(map[string]interface{}); ok {
		out := make(map[string]any, len(m))
		for k, val := range m {
			out[k] = val
		}
		return out, true
	}
	return nil, false
}

func decodeJSONStringMap(v any) (map[string]any, bool) {
	raw := strings.TrimSpace(fmt.Sprintf("%v", v))
	if raw == "" || raw == "<nil>" {
		return nil, false
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, false
	}
	return out, true
}

func (pm *ProxyManager) executeHTTPTool(tool RuntimeTool, args map[string]any, timeoutSeconds int) (string, error) {
	raw, err := renderHTTPEndpoint(tool.Endpoint, normalizeHTTPArgs(args))
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	resp, err := client.Get(raw)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("http tool status %d: %s", resp.StatusCode, string(body))
	}

	if strings.Contains(strings.ToLower(tool.Name), "searxng") {
		results := gjson.GetBytes(body, "results")
		if results.IsArray() {
			max := 5
			lines := make([]string, 0, max)
			results.ForEach(func(_, v gjson.Result) bool {
				if len(lines) >= max {
					return false
				}
				title := strings.TrimSpace(v.Get("title").String())
				link := strings.TrimSpace(v.Get("url").String())
				content := strings.TrimSpace(v.Get("content").String())
				lines = append(lines, fmt.Sprintf("- %s\n  %s\n  %s", title, link, content))
				return true
			})
			return strings.Join(lines, "\n"), nil
		}
	}

	return string(body), nil
}

func normalizeHTTPArgs(args map[string]any) map[string]any {
	if len(args) == 0 {
		return args
	}
	out := map[string]any{}
	for k, v := range args {
		out[k] = v
	}

	// Unwrap common nested wrappers from model tool calls.
	for _, key := range []string{"arguments", "args", "input"} {
		if raw, ok := out[key]; ok {
			if m, ok := asMap(raw); ok {
				for mk, mv := range m {
					if _, exists := out[mk]; !exists {
						out[mk] = mv
					}
				}
			} else if m, ok := decodeJSONStringMap(raw); ok {
				for mk, mv := range m {
					if _, exists := out[mk]; !exists {
						out[mk] = mv
					}
				}
			} else {
				// Some models send a plain string in "arguments"/"input".
				// Treat that as the primary query when no explicit query exists.
				rawText := normalizeToolQuery(fmt.Sprintf("%v", raw))
				if rawText != "" {
					if _, exists := out["query"]; !exists {
						out["query"] = rawText
					}
				}
			}
		}
	}

	// Query aliases often produced by smaller models.
	if _, hasQuery := out["query"]; !hasQuery {
		for _, k := range []string{"q", "search", "search_query", "text", "prompt", "term"} {
			if v, ok := out[k]; ok {
				out["query"] = v
				break
			}
		}
	}
	if _, hasQuery := out["query"]; !hasQuery {
		for k, v := range out {
			if strings.EqualFold(k, "name") || strings.EqualFold(k, "tool") || strings.EqualFold(k, "tool_name") {
				continue
			}
			raw := normalizeToolQuery(fmt.Sprintf("%v", v))
			if raw != "" {
				out["query"] = raw
				break
			}
		}
	}
	return out
}

func normalizeToolQuery(raw string) string {
	q := strings.TrimSpace(raw)
	// Some models return {query} or "query" wrappers in function args.
	if len(q) >= 2 && strings.HasPrefix(q, "{") && strings.HasSuffix(q, "}") {
		q = strings.TrimSpace(q[1 : len(q)-1])
	}
	q = strings.Trim(q, `"'`)
	return strings.TrimSpace(q)
}

func renderHTTPEndpoint(endpoint string, args map[string]any) (string, error) {
	out := strings.TrimSpace(endpoint)
	if out == "" {
		return "", fmt.Errorf("tool endpoint is empty")
	}

	query := normalizeToolQuery(fmt.Sprintf("%v", args["query"]))
	if query != "" {
		out = strings.ReplaceAll(out, "{query}", url.QueryEscape(query))
	}

	for k, v := range args {
		key := strings.TrimSpace(k)
		if key == "" || strings.EqualFold(key, "query") {
			continue
		}
		placeholder := "{" + key + "}"
		value := strings.TrimSpace(fmt.Sprintf("%v", v))
		if value == "" || strings.EqualFold(value, "<nil>") {
			continue
		}
		out = strings.ReplaceAll(out, placeholder, url.QueryEscape(value))
	}

	if strings.Contains(out, "{") && strings.Contains(out, "}") {
		return "", fmt.Errorf("missing tool args for endpoint template placeholders")
	}
	return out, nil
}

func (pm *ProxyManager) executeMCPTool(tool RuntimeTool, args map[string]any, timeoutSeconds int) (string, error) {
	remoteName, callArgs, err := resolveMCPCall(tool, args)
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	sessionID, err := mcpInitializeSession(client, tool.Endpoint)
	if err != nil {
		return "", err
	}

	reqBody := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      remoteName,
			"arguments": callArgs,
		},
	}
	body, err := mcpPostJSONRPC(client, tool.Endpoint, sessionID, reqBody)
	if err != nil {
		return "", err
	}

	payload := extractMCPPayload(body)
	if len(payload) == 0 {
		payload = body
	}

	if txt := gjson.GetBytes(payload, "result.content.0.text").String(); strings.TrimSpace(txt) != "" {
		return txt, nil
	}
	if txt := gjson.GetBytes(payload, "result.text").String(); strings.TrimSpace(txt) != "" {
		return txt, nil
	}
	if errMsg := strings.TrimSpace(gjson.GetBytes(payload, "error.message").String()); errMsg != "" {
		return "", fmt.Errorf("mcp error: %s", errMsg)
	}
	return string(payload), nil
}

func resolveMCPCall(tool RuntimeTool, args map[string]any) (string, map[string]any, error) {
	remoteName := strings.TrimSpace(tool.RemoteName)
	callArgs := args

	if remoteName != "" {
		for _, key := range []string{"arguments", "args", "input"} {
			if v, ok := args[key]; ok {
				if m, ok := asMap(v); ok {
					return remoteName, m, nil
				}
				if m, ok := decodeJSONStringMap(v); ok {
					return remoteName, m, nil
				}
			}
		}
		return remoteName, callArgs, nil
	}

	for _, key := range []string{"name", "tool", "tool_name"} {
		if v, ok := args[key]; ok {
			name := strings.TrimSpace(fmt.Sprintf("%v", v))
			if name != "" && !strings.EqualFold(name, "<nil>") {
				remoteName = name
				break
			}
		}
	}
	if remoteName == "" {
		return "", nil, fmt.Errorf("mcp tool requires remote tool name (set remoteName or pass args.name)")
	}

	for _, key := range []string{"arguments", "args", "input"} {
		if v, ok := args[key]; ok {
			if m, ok := asMap(v); ok {
				callArgs = m
				return remoteName, callArgs, nil
			}
			if m, ok := decodeJSONStringMap(v); ok {
				callArgs = m
				return remoteName, callArgs, nil
			}
		}
	}

	// If arguments wrapper isn't present, pass through all fields except selector keys.
	callArgs = make(map[string]any, len(args))
	for k, v := range args {
		lk := strings.ToLower(strings.TrimSpace(k))
		if lk == "name" || lk == "tool" || lk == "tool_name" || lk == "arguments" || lk == "args" || lk == "input" {
			continue
		}
		callArgs[k] = v
	}
	return remoteName, callArgs, nil
}

func mcpInitializeSession(client *http.Client, endpoint string) (string, error) {
	initReq := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2025-06-18",
			"capabilities":    map[string]any{},
			"clientInfo": map[string]any{
				"name":    "tbg-ollama-swap",
				"version": "1.0.0",
			},
		},
	}
	initBody, err := json.Marshal(initReq)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(initBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("mcp initialize status %d: %s", resp.StatusCode, string(respBody))
	}
	sessionID := strings.TrimSpace(resp.Header.Get("mcp-session-id"))
	if sessionID == "" {
		return "", fmt.Errorf("mcp initialize missing session id")
	}

	notifyReq := map[string]any{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
		"params":  map[string]any{},
	}
	if _, err := mcpPostJSONRPC(client, endpoint, sessionID, notifyReq); err != nil {
		return "", err
	}
	return sessionID, nil
}

func mcpPostJSONRPC(client *http.Client, endpoint string, sessionID string, reqBody map[string]any) ([]byte, error) {
	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if strings.TrimSpace(sessionID) != "" {
		req.Header.Set("mcp-session-id", sessionID)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("mcp status %d: %s", resp.StatusCode, string(body))
	}
	return body, nil
}

func extractMCPPayload(raw []byte) []byte {
	body := bytes.TrimSpace(raw)
	if len(body) == 0 {
		return body
	}
	if json.Valid(body) {
		return body
	}
	lines := strings.Split(string(body), "\n")
	lastValid := []byte{}
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		candidate := []byte(data)
		if json.Valid(candidate) {
			lastValid = candidate
		}
	}
	return lastValid
}

func isLocalHost(host string) bool {
	h := strings.TrimSpace(strings.ToLower(host))
	if h == "" {
		return false
	}
	if strings.Contains(h, ":") {
		if parsedHost, _, err := net.SplitHostPort(h); err == nil {
			h = strings.ToLower(parsedHost)
		}
	}
	if h == "localhost" || h == "host.docker.internal" || h == "::1" || h == "[::1]" {
		return true
	}
	if ip := net.ParseIP(strings.Trim(h, "[]")); ip != nil {
		return ip.IsLoopback()
	}
	return strings.HasSuffix(h, ".local")
}

func validateToolEndpoint(endpoint string, settings ToolRuntimeSettings) error {
	u, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil {
		return fmt.Errorf("invalid endpoint URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported endpoint scheme: %s", u.Scheme)
	}
	if settings.BlockNonLocalEndpoints && !isLocalHost(u.Host) {
		return fmt.Errorf("endpoint host %s is blocked by local-only policy", u.Host)
	}
	return nil
}

func extractLastUserMessageText(body []byte) string {
	msgs := gjson.GetBytes(body, "messages")
	if !msgs.IsArray() {
		return ""
	}
	arr := msgs.Array()
	for i := len(arr) - 1; i >= 0; i-- {
		m := arr[i]
		if strings.ToLower(strings.TrimSpace(m.Get("role").String())) != "user" {
			continue
		}
		content := m.Get("content")
		if content.Type == gjson.String {
			return strings.TrimSpace(content.String())
		}
		if content.IsArray() {
			var b strings.Builder
			content.ForEach(func(_, v gjson.Result) bool {
				t := strings.TrimSpace(v.Get("text").String())
				if t != "" {
					if b.Len() > 0 {
						b.WriteString("\n")
					}
					b.WriteString(t)
				}
				return true
			})
			return strings.TrimSpace(b.String())
		}
	}
	return ""
}

func looksLikeWebSearch(text string) bool {
	t := strings.ToLower(strings.TrimSpace(text))
	if t == "" {
		return false
	}
	keywords := []string{
		"search", "seach", "web", "wep", "look up", "find online", "latest", "today", "news", "docs", "documentation", "release notes",
	}
	for _, k := range keywords {
		if strings.Contains(t, k) {
			return true
		}
	}
	return false
}

func (pm *ProxyManager) forcedToolName(body []byte) string {
	settings := pm.getToolRuntimeSettings()
	if !settings.Enabled {
		return ""
	}
	tools := pm.getEnabledTools()
	if len(tools) == 0 {
		return ""
	}
	for _, t := range tools {
		if t.Policy == ToolPolicyAlways {
			return t.Name
		}
	}
	if settings.WebSearchMode != "force" {
		return ""
	}
	if !looksLikeWebSearch(extractLastUserMessageText(body)) {
		return ""
	}
	httpTools := make([]RuntimeTool, 0, len(tools))
	for _, t := range tools {
		if t.Type == RuntimeToolHTTP {
			httpTools = append(httpTools, t)
		}
		n := strings.ToLower(t.Name)
		if t.Type == RuntimeToolHTTP && (strings.Contains(n, "searxng") || strings.Contains(n, "web_search") || strings.Contains(n, "search") || strings.Contains(n, "seach")) {
			return t.Name
		}
	}
	if len(httpTools) == 1 {
		return httpTools[0].Name
	}
	return ""
}
