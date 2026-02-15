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

	ToolPolicyAuto   RuntimeToolPolicy = "auto"
	ToolPolicyAlways RuntimeToolPolicy = "always"
	ToolPolicyNever  RuntimeToolPolicy = "never"
)

type ToolRuntimeSettings struct {
	Enabled                bool   `json:"enabled"`
	WebSearchMode          string `json:"webSearchMode"` // off|auto|force
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
	Policy          RuntimeToolPolicy `json:"policy,omitempty"` // auto|always|never
	RequireApproval bool              `json:"requireApproval,omitempty"`
	TimeoutSeconds  int               `json:"timeoutSeconds,omitempty"`
}

type toolsDiskState struct {
	Settings ToolRuntimeSettings `json:"settings"`
	Tools    []RuntimeTool       `json:"tools"`
}

func defaultToolRuntimeSettings() ToolRuntimeSettings {
	return ToolRuntimeSettings{
		Enabled:                true,
		WebSearchMode:          "auto",
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
		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": description,
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "Search query or tool input",
						},
					},
					"required": []string{"query"},
				},
			},
		})
	}
	return result
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
	if tool.RequireApproval || settings.RequireApprovalHeader {
		headerName := settings.ApprovalHeaderName
		val := strings.ToLower(strings.TrimSpace(headers.Get(headerName)))
		if val != "1" && val != "true" && val != "yes" && val != "on" {
			return "", fmt.Errorf("tool %s requires approval header %s=true", toolName, headerName)
		}
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
		pm.proxyLogger.Infof("tool call name=%s type=%s duration_ms=%d err=%v", tool.Name, tool.Type, time.Since(start).Milliseconds(), err != nil)
		return out, err
	case RuntimeToolMCP:
		out, err := pm.executeMCPTool(tool, args, timeout)
		pm.proxyLogger.Infof("tool call name=%s type=%s duration_ms=%d err=%v", tool.Name, tool.Type, time.Since(start).Milliseconds(), err != nil)
		return out, err
	default:
		return "", fmt.Errorf("unsupported tool type %s", tool.Type)
	}
}

func (pm *ProxyManager) executeHTTPTool(tool RuntimeTool, args map[string]any, timeoutSeconds int) (string, error) {
	query := normalizeToolQuery(fmt.Sprintf("%v", args["query"]))
	if query == "" {
		return "", fmt.Errorf("missing query argument")
	}

	raw := strings.ReplaceAll(tool.Endpoint, "{query}", url.QueryEscape(query))
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

func normalizeToolQuery(raw string) string {
	q := strings.TrimSpace(raw)
	// Some models return {query} or "query" wrappers in function args.
	if len(q) >= 2 && strings.HasPrefix(q, "{") && strings.HasSuffix(q, "}") {
		q = strings.TrimSpace(q[1 : len(q)-1])
	}
	q = strings.Trim(q, `"'`)
	return strings.TrimSpace(q)
}

func (pm *ProxyManager) executeMCPTool(tool RuntimeTool, args map[string]any, timeoutSeconds int) (string, error) {
	remoteName := strings.TrimSpace(tool.RemoteName)
	if remoteName == "" {
		remoteName = tool.Name
	}

	reqBody := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      remoteName,
			"arguments": args,
		},
	}
	b, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	resp, err := client.Post(tool.Endpoint, "application/json", bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("mcp status %d: %s", resp.StatusCode, string(body))
	}

	if txt := gjson.GetBytes(body, "result.content.0.text").String(); strings.TrimSpace(txt) != "" {
		return txt, nil
	}
	if txt := gjson.GetBytes(body, "result.text").String(); strings.TrimSpace(txt) != "" {
		return txt, nil
	}
	return string(body), nil
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
