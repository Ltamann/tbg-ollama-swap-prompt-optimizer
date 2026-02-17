package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/compat"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	PROFILE_SPLIT_CHAR = ":"
)

type proxyCtxKey string

type chatSource struct {
	URL    string `json:"url"`
	Title  string `json:"title,omitempty"`
	Domain string `json:"domain,omitempty"`
}

var toolCallTagRegex = regexp.MustCompile(`(?is)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

type ProxyManager struct {
	sync.Mutex

	config    config.Config
	ginEngine *gin.Engine

	// logging
	proxyLogger    *LogMonitor
	upstreamLogger *LogMonitor
	muxLogger      *LogMonitor

	metricsMonitor *metricsMonitor

	processGroups map[string]*ProcessGroup

	// shutdown signaling
	shutdownCtx    context.Context
	shutdownCancel context.CancelFunc

	// version info
	buildDate string
	commit    string
	version   string

	// peer proxy see: #296, #433
	peerProxy *PeerProxy

	// custom ctx-size per model (stored before loading)
	ctxSizes map[string]int
	// runtime fit-mode per model (stored before loading)
	fitModes map[string]bool
	// fit ctx behavior per model: "max" -> --ctx-size, "min" -> --fit-ctx
	fitCtxModes map[string]string

	// runtime prompt optimization policy per model
	promptPolicies map[string]PromptOptimizationPolicy

	// latest optimization snapshot for each model (for user visibility and reuse)
	latestPromptOptimizations map[string]PromptOptimizationSnapshot

	// absolute or relative path to active config file
	configPath string

	// lightweight ollama hook
	ollamaEndpoint    string
	ollamaClient      *http.Client
	ollamaModels      map[string]OllamaModel
	ollamaLastRefresh time.Time
	tools             []RuntimeTool
	toolSettings      ToolRuntimeSettings

	// in-memory activity prompt timeline for current user turn only
	activityPromptPreviews       []ActivityPromptPreview
	activityCurrentUserSignature string
	activityCurrentTurn          int
	activityNextPromptID         int
	compatCapabilities           compat.Registry
}

type PromptOptimizationPolicy string

const (
	PromptOptimizationOff       PromptOptimizationPolicy = "off"
	PromptOptimizationLimitOnly PromptOptimizationPolicy = "limit_only"
	PromptOptimizationAlways    PromptOptimizationPolicy = "always"
	PromptOptimizationLLMAssist PromptOptimizationPolicy = "llm_assisted"
)

type PromptOptimizationSnapshot struct {
	Model         string                   `json:"model"`
	Policy        PromptOptimizationPolicy `json:"policy"`
	Applied       bool                     `json:"applied"`
	UpdatedAt     string                   `json:"updatedAt"`
	Note          string                   `json:"note"`
	OriginalBody  string                   `json:"originalBody"`
	OptimizedBody string                   `json:"optimizedBody"`
}

type PromptOptimizationResult struct {
	Policy  PromptOptimizationPolicy
	Applied bool
	Note    string
}

type OllamaModel struct {
	ID           string
	Name         string
	CtxReference int
}

func New(proxyConfig config.Config) *ProxyManager {
	// set up loggers

	var muxLogger, upstreamLogger, proxyLogger *LogMonitor
	switch proxyConfig.LogToStdout {
	case config.LogToStdoutNone:
		muxLogger = NewLogMonitorWriter(io.Discard)
		upstreamLogger = NewLogMonitorWriter(io.Discard)
		proxyLogger = NewLogMonitorWriter(io.Discard)
	case config.LogToStdoutBoth:
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(muxLogger)
		proxyLogger = NewLogMonitorWriter(muxLogger)
	case config.LogToStdoutUpstream:
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(muxLogger)
		proxyLogger = NewLogMonitorWriter(io.Discard)
	default:
		// same as config.LogToStdoutProxy
		// helpful because some old tests create a config.Config directly and it
		// may not have LogToStdout set explicitly
		muxLogger = NewLogMonitorWriter(os.Stdout)
		upstreamLogger = NewLogMonitorWriter(io.Discard)
		proxyLogger = NewLogMonitorWriter(muxLogger)
	}

	if proxyConfig.LogRequests {
		proxyLogger.Warn("LogRequests configuration is deprecated. Use logLevel instead.")
	}

	switch strings.ToLower(strings.TrimSpace(proxyConfig.LogLevel)) {
	case "debug":
		proxyLogger.SetLogLevel(LevelDebug)
		upstreamLogger.SetLogLevel(LevelDebug)
	case "info":
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
	case "warn":
		proxyLogger.SetLogLevel(LevelWarn)
		upstreamLogger.SetLogLevel(LevelWarn)
	case "error":
		proxyLogger.SetLogLevel(LevelError)
		upstreamLogger.SetLogLevel(LevelError)
	default:
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
	}

	// see: https://go.dev/src/time/format.go
	timeFormats := map[string]string{
		"ansic":       time.ANSIC,
		"unixdate":    time.UnixDate,
		"rubydate":    time.RubyDate,
		"rfc822":      time.RFC822,
		"rfc822z":     time.RFC822Z,
		"rfc850":      time.RFC850,
		"rfc1123":     time.RFC1123,
		"rfc1123z":    time.RFC1123Z,
		"rfc3339":     time.RFC3339,
		"rfc3339nano": time.RFC3339Nano,
		"kitchen":     time.Kitchen,
		"stamp":       time.Stamp,
		"stampmilli":  time.StampMilli,
		"stampmicro":  time.StampMicro,
		"stampnano":   time.StampNano,
	}

	if timeFormat, ok := timeFormats[strings.ToLower(strings.TrimSpace(proxyConfig.LogTimeFormat))]; ok {
		proxyLogger.SetLogTimeFormat(timeFormat)
		upstreamLogger.SetLogTimeFormat(timeFormat)
	}

	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())

	var maxMetrics int
	if proxyConfig.MetricsMaxInMemory <= 0 {
		maxMetrics = 1000 // Default fallback
	} else {
		maxMetrics = proxyConfig.MetricsMaxInMemory
	}

	peerProxy, err := NewPeerProxy(proxyConfig.Peers, proxyLogger)
	if err != nil {
		proxyLogger.Errorf("Disabling Peering. Failed to create proxy peers: %v", err)
		peerProxy = nil
	}

	pm := &ProxyManager{
		config:    proxyConfig,
		ginEngine: gin.New(),

		proxyLogger:    proxyLogger,
		muxLogger:      muxLogger,
		upstreamLogger: upstreamLogger,

		metricsMonitor: newMetricsMonitor(proxyLogger, maxMetrics, proxyConfig.CaptureBuffer),

		processGroups: make(map[string]*ProcessGroup),

		shutdownCtx:    shutdownCtx,
		shutdownCancel: shutdownCancel,

		buildDate: "unknown",
		commit:    "abcd1234",
		version:   "0",

		peerProxy:                 peerProxy,
		ctxSizes:                  make(map[string]int),
		fitModes:                  make(map[string]bool),
		fitCtxModes:               make(map[string]string),
		promptPolicies:            make(map[string]PromptOptimizationPolicy),
		latestPromptOptimizations: make(map[string]PromptOptimizationSnapshot),
		configPath:                "config.yaml",
		ollamaEndpoint:            "http://127.0.0.1:11434",
		ollamaClient:              &http.Client{Timeout: 20 * time.Second},
		ollamaModels:              make(map[string]OllamaModel),
		tools:                     make([]RuntimeTool, 0),
		toolSettings:              defaultToolRuntimeSettings(),
		activityPromptPreviews:    make([]ActivityPromptPreview, 0),
		compatCapabilities:        compat.NewDefaultRegistry(),
	}
	pm.loadToolsFromDisk()

	// create the process groups
	for groupID := range proxyConfig.Groups {
		processGroup := NewProcessGroup(groupID, proxyConfig, proxyLogger, upstreamLogger)
		pm.processGroups[groupID] = processGroup
	}

	pm.setupGinEngine()

	// run any startup hooks
	if len(proxyConfig.Hooks.OnStartup.Preload) > 0 {
		// do it in the background, don't block startup -- not sure if good idea yet
		go func() {
			discardWriter := &DiscardWriter{}
			for _, preloadModelName := range proxyConfig.Hooks.OnStartup.Preload {
				modelID, ok := proxyConfig.RealModelName(preloadModelName)

				if !ok {
					proxyLogger.Warnf("Preload model %s not found in config", preloadModelName)
					continue
				}

				proxyLogger.Infof("Preloading model: %s", modelID)
				processGroup, err := pm.swapProcessGroup(modelID)

				if err != nil {
					event.Emit(ModelPreloadedEvent{
						ModelName: modelID,
						Success:   false,
					})
					proxyLogger.Errorf("Failed to preload model %s: %v", modelID, err)
					continue
				} else {
					req, _ := http.NewRequest("GET", "/", nil)
					processGroup.ProxyRequest(modelID, discardWriter, req)
					event.Emit(ModelPreloadedEvent{
						ModelName: modelID,
						Success:   true,
					})
				}
			}
		}()
	}

	return pm
}

func (pm *ProxyManager) setupGinEngine() {

	pm.ginEngine.Use(func(c *gin.Context) {

		// don't log the Wake on Lan proxy health check
		if c.Request.URL.Path == "/wol-health" {
			c.Next()
			return
		}

		// Start timer
		start := time.Now()

		// capture these because /upstream/:model rewrites them in c.Next()
		clientIP := c.ClientIP()
		method := c.Request.Method
		path := c.Request.URL.Path

		// Process request
		c.Next()

		// Stop timer
		duration := time.Since(start)

		statusCode := c.Writer.Status()
		bodySize := c.Writer.Size()

		pm.proxyLogger.Infof("Request %s \"%s %s %s\" %d %d \"%s\" %v",
			clientIP,
			method,
			path,
			c.Request.Proto,
			statusCode,
			bodySize,
			c.Request.UserAgent(),
			duration,
		)
	})

	// see: issue: #81, #77 and #42 for CORS issues
	// respond with permissive OPTIONS for any endpoint
	pm.ginEngine.Use(func(c *gin.Context) {
		if c.Request.Method == "OPTIONS" {
			c.Header("Access-Control-Allow-Origin", "*")
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")

			// allow whatever the client requested by default
			if headers := c.Request.Header.Get("Access-Control-Request-Headers"); headers != "" {
				sanitized := SanitizeAccessControlRequestHeaderValues(headers)
				c.Header("Access-Control-Allow-Headers", sanitized)
			} else {
				c.Header(
					"Access-Control-Allow-Headers",
					"Content-Type, Authorization, Accept, X-Requested-With",
				)
			}
			c.Header("Access-Control-Max-Age", "86400")
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	// Set up routes using the Gin engine
	// Protected routes use pm.apiKeyAuth() middleware
	pm.ginEngine.POST("/v1/chat/completions", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/responses", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support legacy /v1/completions api, see issue #12
	pm.ginEngine.POST("/v1/completions", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support anthropic /v1/messages (added https://github.com/ggml-org/llama.cpp/pull/17570)
	pm.ginEngine.POST("/v1/messages", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	// Support anthropic count_tokens API (Also added in the above PR)
	pm.ginEngine.POST("/v1/messages/count_tokens", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// Support embeddings and reranking
	pm.ginEngine.POST("/v1/embeddings", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /reranking endpoint + aliases
	pm.ginEngine.POST("/reranking", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/rerank", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/rerank", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/reranking", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /infill endpoint for code infilling
	pm.ginEngine.POST("/infill", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// llama-server's /completion endpoint
	pm.ginEngine.POST("/completion", pm.apiKeyAuth(), pm.proxyInferenceHandler)

	// Support audio/speech endpoint
	pm.ginEngine.POST("/v1/audio/speech", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/audio/voices", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.GET("/v1/audio/voices", pm.apiKeyAuth(), pm.proxyGETModelHandler)
	pm.ginEngine.POST("/v1/audio/transcriptions", pm.apiKeyAuth(), pm.proxyOAIPostFormHandler)
	pm.ginEngine.POST("/v1/images/generations", pm.apiKeyAuth(), pm.proxyInferenceHandler)
	pm.ginEngine.POST("/v1/images/edits", pm.apiKeyAuth(), pm.proxyOAIPostFormHandler)

	pm.ginEngine.GET("/v1/models", pm.apiKeyAuth(), pm.listModelsHandler)

	// in proxymanager_loghandlers.go
	pm.ginEngine.GET("/logs", pm.apiKeyAuth(), pm.sendLogsHandlers)
	pm.ginEngine.GET("/logs/stream", pm.apiKeyAuth(), pm.streamLogsHandler)
	pm.ginEngine.GET("/logs/stream/*logMonitorID", pm.apiKeyAuth(), pm.streamLogsHandler)

	/**
	 * User Interface Endpoints
	 */
	pm.ginEngine.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui")
	})

	pm.ginEngine.GET("/upstream", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui/models")
	})
	pm.ginEngine.Any("/upstream/*upstreamPath", pm.apiKeyAuth(), pm.proxyToUpstream)
	pm.ginEngine.GET("/unload", pm.apiKeyAuth(), pm.unloadAllModelsHandler)
	pm.ginEngine.GET("/running", pm.apiKeyAuth(), pm.listRunningProcessesHandler)
	pm.ginEngine.GET("/health", func(c *gin.Context) {
		c.String(http.StatusOK, "OK")
	})

	// see cmd/wol-proxy/wol-proxy.go, not logged
	pm.ginEngine.GET("/wol-health", func(c *gin.Context) {
		c.String(http.StatusOK, "OK")
	})

	pm.ginEngine.GET("/favicon.ico", func(c *gin.Context) {
		if data, err := reactStaticFS.ReadFile("ui_dist/favicon.ico"); err == nil {
			c.Data(http.StatusOK, "image/x-icon", data)
		} else {
			c.String(http.StatusInternalServerError, err.Error())
		}
	})

	reactFS, err := GetReactFS()
	if err != nil {
		pm.proxyLogger.Errorf("Failed to load React filesystem: %v", err)
	} else {
		// Serve files with compression support under /ui/*
		// This handler checks for pre-compressed .br and .gz files
		pm.ginEngine.GET("/ui/*filepath", func(c *gin.Context) {
			filepath := strings.TrimPrefix(c.Param("filepath"), "/")
			// Default to index.html for directory-like paths
			if filepath == "" {
				filepath = "index.html"
			}

			ServeCompressedFile(reactFS, c.Writer, c.Request, filepath)
		})

		// Serve SPA for UI under /ui/* - fallback to index.html for client-side routing
		pm.ginEngine.NoRoute(func(c *gin.Context) {
			if !strings.HasPrefix(c.Request.URL.Path, "/ui") {
				c.AbortWithStatus(http.StatusNotFound)
				return
			}

			// Check if this looks like a file request (has extension)
			path := c.Request.URL.Path
			if strings.Contains(path, ".") && !strings.HasSuffix(path, "/") {
				// This was likely a file request that wasn't found
				c.AbortWithStatus(http.StatusNotFound)
				return
			}

			// Serve index.html for SPA routing
			ServeCompressedFile(reactFS, c.Writer, c.Request, "index.html")
		})
	}

	// see: proxymanager_api.go
	// add API handler functions
	addApiHandlers(pm)

	// Disable console color for testing
	gin.DisableConsoleColor()
}

// ServeHTTP implements http.Handler interface
func (pm *ProxyManager) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	pm.ginEngine.ServeHTTP(w, r)
}

// StopProcesses acquires a lock and stops all running upstream processes.
// This is the public method safe for concurrent calls.
// Unlike Shutdown, this method only stops the processes but doesn't perform
// a complete shutdown, allowing for process replacement without full termination.
func (pm *ProxyManager) StopProcesses(strategy StopStrategy) {
	pm.Lock()
	defer pm.Unlock()

	// stop Processes in parallel
	var wg sync.WaitGroup
	for _, processGroup := range pm.processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.StopProcesses(strategy)
		}(processGroup)
	}

	wg.Wait()
}

// Shutdown stops all processes managed by this ProxyManager
func (pm *ProxyManager) Shutdown() {
	pm.Lock()
	defer pm.Unlock()

	pm.proxyLogger.Debug("Shutdown() called in proxy manager")

	var wg sync.WaitGroup
	// Send shutdown signal to all process in groups
	for _, processGroup := range pm.processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.Shutdown()
		}(processGroup)
	}
	wg.Wait()
	pm.shutdownCancel()
}

func (pm *ProxyManager) swapProcessGroup(realModelName string) (*ProcessGroup, error) {
	processGroup := pm.findGroupByModelName(realModelName)
	if processGroup == nil {
		return nil, fmt.Errorf("could not find process group for model %s", realModelName)
	}

	if process, ok := processGroup.processes[realModelName]; ok && process != nil {
		pm.Lock()
		ctxSize := pm.ctxSizes[realModelName]
		fitEnabled, fitOverride := pm.fitModes[realModelName]
		fitCtxMode, fitCtxModeOverride := pm.fitCtxModes[realModelName]
		pm.Unlock()

		if !fitOverride {
			if args, err := process.config.SanitizedCommand(); err == nil {
				_, _, parsedFitEnabled, parsedFitCtxMode := parseCtxAndFitFromArgs(args)
				fitEnabled = parsedFitEnabled
				if !fitCtxModeOverride {
					fitCtxMode = parsedFitCtxMode
				}
			}
		}
		if fitCtxMode == "" {
			fitCtxMode = "max"
		}

		process.SetRuntimeCtxSize(ctxSize)
		process.SetRuntimeFitMode(fitEnabled)
		process.SetRuntimeFitCtxMode(fitCtxMode == "min")
	}

	if processGroup.exclusive {
		pm.proxyLogger.Debugf("Exclusive mode for group %s, stopping other process groups", processGroup.id)
		for groupId, otherGroup := range pm.processGroups {
			if groupId != processGroup.id && !otherGroup.persistent {
				otherGroup.StopProcesses(StopWaitForInflightRequest)
			}
		}
	}
	pm.enforceRuntimeProcessPolicy(realModelName)

	return processGroup, nil
}

func (pm *ProxyManager) enforceRuntimeProcessPolicy(targetModel string) {
	settings := pm.getToolRuntimeSettings()
	type runningProc struct {
		groupID string
		modelID string
	}
	running := make([]runningProc, 0)
	for groupID, group := range pm.processGroups {
		for modelID, process := range group.processes {
			if process == nil {
				continue
			}
			if process.CurrentState() == StateReady {
				running = append(running, runningProc{groupID: groupID, modelID: modelID})
			}
		}
	}

	killSet := map[string]struct{}{}
	if settings.KillPreviousOnSwap {
		for _, rp := range running {
			if rp.modelID != targetModel {
				killSet[rp.groupID+":"+rp.modelID] = struct{}{}
			}
		}
	}

	maxKeep := settings.MaxRunningModels
	if maxKeep < 1 {
		maxKeep = 1
	}
	keep := 1 // always keep target
	for _, rp := range running {
		if rp.modelID == targetModel {
			continue
		}
		if _, forcedKill := killSet[rp.groupID+":"+rp.modelID]; forcedKill {
			continue
		}
		if keep < maxKeep {
			keep++
			continue
		}
		killSet[rp.groupID+":"+rp.modelID] = struct{}{}
	}

	for key := range killSet {
		parts := strings.SplitN(key, ":", 2)
		if len(parts) != 2 {
			continue
		}
		groupID, modelID := parts[0], parts[1]
		group := pm.processGroups[groupID]
		if group == nil {
			continue
		}
		if err := group.StopProcess(modelID, StopImmediately); err != nil {
			pm.proxyLogger.Warnf("runtime policy stop failed for %s: %v", modelID, err)
			continue
		}
		pm.proxyLogger.Infof("runtime policy stopped previous model %s", modelID)
	}
}

func parseCtxAndFitFromArgs(args []string) (ctxSize int, source string, fitEnabled bool, fitCtxMode string) {
	ctxFromCtxSize := 0
	ctxFromFitCtx := 0
	fitCtxMode = "max"

	for i := 0; i < len(args); i++ {
		arg := strings.TrimSpace(args[i])
		if arg == "" {
			continue
		}

		switch {
		case arg == "--fit":
			fitEnabled = true
			if i+1 < len(args) {
				next := strings.ToLower(strings.TrimSpace(args[i+1]))
				switch next {
				case "off", "false", "0", "no":
					fitEnabled = false
				case "on", "true", "1", "yes":
					fitEnabled = true
				}
			}
		case strings.HasPrefix(arg, "--fit="):
			val := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(arg, "--fit=")))
			fitEnabled = val == "on" || val == "true" || val == "1" || val == "yes"
		case arg == "--fit-ctx":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil && n > 0 {
					ctxFromFitCtx = n
				}
			}
		case strings.HasPrefix(arg, "--fit-ctx="):
			if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(arg, "--fit-ctx="))); err == nil && n > 0 {
				ctxFromFitCtx = n
			}
		case arg == "--ctx-size" || arg == "-c":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil && n > 0 {
					ctxFromCtxSize = n
				}
			}
		case strings.HasPrefix(arg, "--ctx-size="):
			if n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(arg, "--ctx-size="))); err == nil && n > 0 {
				ctxFromCtxSize = n
			}
		}
	}

	if fitEnabled && ctxFromFitCtx > 0 {
		return ctxFromFitCtx, "fit-ctx", true, "min"
	}
	if ctxFromCtxSize > 0 {
		return ctxFromCtxSize, "ctx-size", fitEnabled, "max"
	}
	if ctxFromFitCtx > 0 {
		return ctxFromFitCtx, "fit-ctx", fitEnabled, "min"
	}
	return 0, "", fitEnabled, fitCtxMode
}

func (pm *ProxyManager) listModelsHandler(c *gin.Context) {
	data := make([]gin.H, 0, len(pm.config.Models))
	createdTime := time.Now().Unix()

	newRecord := func(modelId string, modelConfig config.ModelConfig) gin.H {
		record := gin.H{
			"id":       modelId,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "llama-swap",
		}

		if name := strings.TrimSpace(modelConfig.Name); name != "" {
			record["name"] = name
		}
		if desc := strings.TrimSpace(modelConfig.Description); desc != "" {
			record["description"] = desc
		}

		// Add metadata if present
		if len(modelConfig.Metadata) > 0 {
			record["meta"] = gin.H{
				"llamaswap": modelConfig.Metadata,
			}
		}
		return record
	}

	for id, modelConfig := range pm.config.Models {
		if modelConfig.Unlisted {
			continue
		}

		data = append(data, newRecord(id, modelConfig))

		// Include aliases
		if pm.config.IncludeAliasesInList {
			for _, alias := range modelConfig.Aliases {
				if alias := strings.TrimSpace(alias); alias != "" {
					data = append(data, newRecord(alias, modelConfig))
				}
			}
		}
	}

	if pm.peerProxy != nil {
		for peerID, peer := range pm.peerProxy.ListPeers() {
			// add peer models
			for _, modelID := range peer.Models {
				// Skip unlisted models if not showing them
				record := newRecord(modelID, config.ModelConfig{
					Name: fmt.Sprintf("%s: %s", peerID, modelID),
					Metadata: map[string]any{
						"peerID": peerID,
					},
				})

				data = append(data, record)
			}
		}
	}

	for _, ollamaModel := range pm.GetOllamaModels() {
		data = append(data, gin.H{
			"id":       ollamaModel.ID,
			"name":     ollamaModel.Name,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "ollama",
			"meta": gin.H{
				"llamaswap": gin.H{
					"provider":      "ollama",
					"external":      true,
					"ctx_reference": ollamaModel.CtxReference,
				},
			},
		})
	}

	// Sort by the "id" key
	sort.Slice(data, func(i, j int) bool {
		si, _ := data[i]["id"].(string)
		sj, _ := data[j]["id"].(string)
		return si < sj
	})

	// Set CORS headers if origin exists
	if origin := c.GetHeader("Origin"); origin != "" {
		c.Header("Access-Control-Allow-Origin", origin)
	}

	// Use gin's JSON method which handles content-type and encoding
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   data,
	})
}

// findModelInPath searches for a valid model name in a path with slashes.
// It iteratively builds up path segments until it finds a matching model.
// Returns: (searchModelName, realModelName, remainingPath, found)
// Example: "/author/model/endpoint" with model "author/model" -> ("author/model", "author/model", "/endpoint", true)
func (pm *ProxyManager) findModelInPath(path string) (searchName string, realName string, remainingPath string, found bool) {
	parts := strings.Split(strings.TrimSpace(path), "/")
	searchModelName := ""

	for i, part := range parts {
		if part == "" {
			continue
		}

		if searchModelName == "" {
			searchModelName = part
		} else {
			searchModelName = searchModelName + "/" + part
		}

		if modelID, ok := pm.config.RealModelName(searchModelName); ok {
			return searchModelName, modelID, "/" + strings.Join(parts[i+1:], "/"), true
		}
	}

	return "", "", "", false
}

func (pm *ProxyManager) proxyToUpstream(c *gin.Context) {
	upstreamPath := c.Param("upstreamPath")

	searchModelName, modelID, remainingPath, modelFound := pm.findModelInPath(upstreamPath)

	if !modelFound {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model id required in path")
		return
	}

	// Redirect /upstream/modelname to /upstream/modelname/ for URL consistency.
	// This ensures relative URLs in upstream responses resolve correctly and
	// provides canonical URL form. Uses 308 for POST/PUT/etc to preserve the
	// HTTP method (301 would downgrade to GET).
	if remainingPath == "/" && !strings.HasSuffix(upstreamPath, "/") {
		newPath := "/upstream/" + searchModelName + "/"
		if c.Request.URL.RawQuery != "" {
			newPath += "?" + c.Request.URL.RawQuery
		}
		if c.Request.Method == http.MethodGet || c.Request.Method == http.MethodHead {
			c.Redirect(http.StatusMovedPermanently, newPath)
		} else {
			c.Redirect(http.StatusPermanentRedirect, newPath)
		}
		return
	}

	processGroup, err := pm.swapProcessGroup(modelID)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
		return
	}

	// rewrite the path
	originalPath := c.Request.URL.Path
	c.Request.URL.Path = remainingPath

	// attempt to record metrics if it is a POST request
	if pm.metricsMonitor != nil && c.Request.Method == "POST" {
		if err := pm.metricsMonitor.wrapHandler(modelID, c.Writer, c.Request, processGroup.ProxyRequest); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying metrics wrapped request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error proxying wrapped upstream request for model %s, path=%s", modelID, originalPath)
			return
		}
	} else {
		if err := processGroup.ProxyRequest(modelID, c.Writer, c.Request); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error proxying upstream request for model %s, path=%s", modelID, originalPath)
			return
		}
	}
}

func (pm *ProxyManager) proxyInferenceHandler(c *gin.Context) {
	rawBodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not ready request body")
		return
	}
	bodyBytes := rawBodyBytes
	if decoded, err := decodeRequestByContentEncoding(rawBodyBytes, c.Request.Header.Get("Content-Encoding")); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("invalid compressed request body: %s", err.Error()))
		return
	} else {
		bodyBytes = decoded
	}
	pm.proxyLogger.Warnf(
		"Incoming API request: method=%s path=%s headers=%s body=%s",
		c.Request.Method,
		c.Request.URL.Path,
		safeHeadersJSON(c.Request.Header),
		truncateForLog(string(bodyBytes), 120000),
	)

	norm, err := compat.NormalizeInferenceRequest(c.Request, bodyBytes)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, err.Error())
		return
	}
	bodyBytes = norm.Body
	pm.proxyLogger.Warnf("compat endpoint=%s path=%s", norm.Endpoint, c.Request.URL.Path)
	isResponsesEndpoint := norm.Endpoint == compat.EndpointResponses
	if pm.compatibilityMode() == "strict_openai" {
		if err := pm.compatCapabilities.Validate(norm.Canonical); err != nil {
			pm.sendErrorResponse(c, http.StatusBadRequest, err.Error())
			return
		}
	}
	c.Set("compat_endpoint", string(norm.Endpoint))
	c.Set("compat_canonical_model", norm.Canonical.Model)

	requestedModel := gjson.GetBytes(bodyBytes, "model").String()
	if requestedModel == "" {
		if isResponsesEndpoint {
			if fallbackModel, ok := pm.resolveResponsesFallbackModel(); ok {
				requestedModel = fallbackModel
				bodyBytes, err = sjson.SetBytes(bodyBytes, "model", fallbackModel)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error setting fallback model in JSON: %s", err.Error()))
					return
				}
				pm.proxyLogger.Warnf("Responses request missing model; falling back to '%s'", fallbackModel)
			}
		}
	}

	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' key")
		return
	}

	// Look for a matching local model first
	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error

	modelID, found := pm.config.RealModelName(requestedModel)
	if !found && pm.compatibilityMode() != "strict_openai" {
		trimmedModel := strings.TrimSpace(requestedModel)
		if strings.EqualFold(trimmedModel, "localhost") {
			if fallbackModel, ok := pm.resolveResponsesFallbackModel(); ok && fallbackModel != "" && fallbackModel != requestedModel {
				bodyBytes, err = sjson.SetBytes(bodyBytes, "model", fallbackModel)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error setting fallback model in JSON: %s", err.Error()))
					return
				}
				pm.proxyLogger.Warnf("Model placeholder '%s' mapped to '%s'", requestedModel, fallbackModel)
				requestedModel = fallbackModel
				modelID, found = pm.config.RealModelName(requestedModel)
			}
		}
	}
	if !found && isResponsesEndpoint {
		peerHasModel := false
		if pm.peerProxy != nil {
			peerHasModel = pm.peerProxy.HasPeerModel(requestedModel)
		}
		_, ollamaHasModel := pm.GetOllamaModelByID(requestedModel)
		if !peerHasModel && !ollamaHasModel {
			if fallbackModel, ok := pm.resolveResponsesFallbackModel(); ok && fallbackModel != requestedModel {
				bodyBytes, err = sjson.SetBytes(bodyBytes, "model", fallbackModel)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error setting fallback model in JSON: %s", err.Error()))
					return
				}
				pm.proxyLogger.Warnf("Responses request model '%s' not found; falling back to '%s'", requestedModel, fallbackModel)
				requestedModel = fallbackModel
				modelID, found = pm.config.RealModelName(requestedModel)
			}
		}
	}

	if found {
		processGroup, err := pm.swapProcessGroup(modelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}

		// issue #69 allow custom model names to be sent to upstream
		useModelName := pm.config.Models[modelID].UseModelName
		if useModelName != "" {
			bodyBytes, err = sjson.SetBytes(bodyBytes, "model", useModelName)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting model name in JSON: %s", err.Error()))
				return
			}
		}

		// issue #174 strip parameters from the JSON body
		stripParams, err := pm.config.Models[modelID].Filters.SanitizedStripParams()
		if err != nil { // just log it and continue
			pm.proxyLogger.Errorf("Error sanitizing strip params string: %s, %s", pm.config.Models[modelID].Filters.StripParams, err.Error())
		} else {
			for _, param := range stripParams {
				pm.proxyLogger.Debugf("<%s> stripping param: %s", modelID, param)
				bodyBytes, err = sjson.DeleteBytes(bodyBytes, param)
				if err != nil {
					pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error deleting parameter %s from request", param))
					return
				}
			}
		}

		// issue #453 set/override parameters in the JSON body
		setParams, setParamKeys := pm.config.Models[modelID].Filters.SanitizedSetParams()
		for _, key := range setParamKeys {
			pm.proxyLogger.Debugf("<%s> setting param: %s", modelID, key)
			bodyBytes, err = sjson.SetBytes(bodyBytes, key, setParams[key])
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error setting parameter %s in request", key))
				return
			}
		}

		var optResult PromptOptimizationResult
		if bodyBytes, optResult, err = pm.applyPromptSizeControl(modelID, bodyBytes); err != nil {
			pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("context control rejected request: %s", err.Error()))
			return
		}
		c.Header("X-LlamaSwap-Prompt-Optimization-Policy", string(optResult.Policy))
		if optResult.Applied {
			c.Header("X-LlamaSwap-Prompt-Optimized", "true")
		} else {
			c.Header("X-LlamaSwap-Prompt-Optimized", "false")
		}

		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		modelID = requestedModel

		// issue #453 apply filters for peer requests
		peerFilters := pm.peerProxy.GetPeerFilters(requestedModel)

		// Apply stripParams - remove specified parameters from request
		stripParams := peerFilters.SanitizedStripParams()
		for _, param := range stripParams {
			pm.proxyLogger.Debugf("<%s> stripping param: %s", requestedModel, param)
			bodyBytes, err = sjson.DeleteBytes(bodyBytes, param)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error stripping parameter %s from request", param))
				return
			}
		}

		// Apply setParams - set/override specified parameters in request
		setParams, setParamKeys := peerFilters.SanitizedSetParams()
		for _, key := range setParamKeys {
			pm.proxyLogger.Debugf("<%s> setting param: %s", requestedModel, key)
			bodyBytes, err = sjson.SetBytes(bodyBytes, key, setParams[key])
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error setting parameter %s in request", key))
				return
			}
		}

		nextHandler = pm.peerProxy.ProxyRequest
	} else if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
		modelID = ollamaModel.ID
		bodyBytes, err = sjson.SetBytes(bodyBytes, "model", ollamaModel.Name)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting ollama model name in JSON: %s", err.Error()))
			return
		}

		var optResult PromptOptimizationResult
		if bodyBytes, optResult, err = pm.applyPromptSizeControl(modelID, bodyBytes); err != nil {
			pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("context control rejected request: %s", err.Error()))
			return
		}
		c.Header("X-LlamaSwap-Prompt-Optimization-Policy", string(optResult.Policy))
		if optResult.Applied {
			c.Header("X-LlamaSwap-Prompt-Optimized", "true")
		} else {
			c.Header("X-LlamaSwap-Prompt-Optimized", "false")
		}

		pm.proxyLogger.Debugf("ProxyManager using Ollama for model: %s", requestedModel)
		nextHandler = pm.proxyOllamaRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable inference handler for %s", requestedModel))
		return
	}

	bridgeResponses := isResponsesEndpoint
	responsesRequestedStream := false
	if bridgeResponses {
		acceptHeader := strings.ToLower(strings.TrimSpace(c.Request.Header.Get("Accept")))
		acceptsEventStream := strings.Contains(acceptHeader, "text/event-stream")
		responsesRequestedStream = gjson.GetBytes(bodyBytes, "stream").Bool() || acceptsEventStream
		translated, err := translateResponsesToChatCompletionsRequest(bodyBytes)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("invalid responses request: %s", err.Error()))
			return
		}
		bodyBytes = translated
		// Most local backends (including llama.cpp OpenAI server) are chat-completions-first.
		c.Request.URL.Path = "/v1/chat/completions"
	}

	if !bridgeResponses && strings.HasPrefix(c.Request.URL.Path, "/v1/chat/completions") {
		handled, err := pm.proxyWithToolsIfNeeded(c, modelID, nextHandler, bodyBytes)
		if err != nil {
			var approvalErr *ToolApprovalRequiredError
			if errors.As(err, &approvalErr) {
				c.JSON(http.StatusConflict, gin.H{
					"error": gin.H{
						"type":        "tool_approval_required",
						"code":        "tool_approval_required",
						"message":     "Tool execution requires user approval",
						"header_name": approvalErr.HeaderName,
						"tool_calls":  approvalErr.ToolCalls,
					},
				})
				return
			}
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("tool execution failed: %s", err.Error()))
			return
		}
		if handled {
			return
		}
	}

	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	// dechunk it as we already have all the body bytes see issue #11
	c.Request.Header.Del("transfer-encoding")
	c.Request.Header.Del("Transfer-Encoding")
	// Body is rewritten as plain JSON in proxy, so remove any inbound encoding marker.
	c.Request.Header.Del("content-encoding")
	c.Request.Header.Del("Content-Encoding")
	// Some clients send non-JSON content types to OpenAI-compatible endpoints.
	// We always forward a JSON body, so normalize this header to avoid upstream 415s.
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("content-length", strconv.Itoa(len(bodyBytes)))
	c.Request.ContentLength = int64(len(bodyBytes))

	// issue #366 extract values that downstream handlers may need
	isStreaming := gjson.GetBytes(bodyBytes, "stream").Bool()
	ctx := context.WithValue(c.Request.Context(), proxyCtxKey("streaming"), isStreaming)
	ctx = context.WithValue(ctx, proxyCtxKey("model"), modelID)
	c.Request = c.Request.WithContext(ctx)
	pm.recordActivityPromptPreview(modelID, c.Request.URL.Path, bodyBytes, c.Request.Header)

	if bridgeResponses {
		pm.proxyLogger.Warnf("Responses bridge active for model=%s stream=%v", modelID, responsesRequestedStream)
		pm.proxyLogger.Warnf("Responses bridge request payload: %s", truncateForLog(string(bodyBytes), 8000))
		pm.proxyLogger.Warnf(
			"Responses bridge request headers: %s path=%s",
			safeHeadersJSON(c.Request.Header),
			c.Request.URL.Path,
		)
		var (
			statusCode int
			respBody   []byte
		)
		// Reuse the existing tool loop for bridged responses so tool_calls are executed
		// instead of being dropped during chat->responses translation.
		if len(pm.getEnabledTools()) > 0 && gjson.GetBytes(bodyBytes, "messages").IsArray() {
			working, err := sjson.SetBytes(bodyBytes, "stream", false)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error preparing bridged request: %s", err.Error()))
				return
			}
			working, err = pm.injectToolSchemas(working)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error injecting tool schemas: %s", err.Error()))
				return
			}
			maxIterations := pm.getToolRuntimeSettings().MaxToolRounds
			respBody, statusCode, err = pm.runToolLoop(modelID, nextHandler, c.Request, working, maxIterations)
			if err != nil {
				var approvalErr *ToolApprovalRequiredError
				if errors.As(err, &approvalErr) {
					c.JSON(http.StatusConflict, gin.H{
						"error": gin.H{
							"type":        "tool_approval_required",
							"code":        "tool_approval_required",
							"message":     "Tool execution requires user approval",
							"header_name": approvalErr.HeaderName,
							"tool_calls":  approvalErr.ToolCalls,
						},
					})
					return
				}
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
				pm.proxyLogger.Errorf("Error Proxying Bridged Responses Tool Request for model %s", modelID)
				return
			}
		} else {
			rr := &bridgeResponseRecorder{
				ResponseRecorder: httptest.NewRecorder(),
				closeChannel:     make(chan bool, 1),
			}
			if err := nextHandler(modelID, rr, c.Request); err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
				pm.proxyLogger.Errorf("Error Proxying Bridged Responses Request for model %s", modelID)
				return
			}
			statusCode = rr.Code
			if statusCode == 0 {
				statusCode = http.StatusOK
			}
			respBody = rr.Body.Bytes()
		}
		if statusCode < 200 || statusCode >= 300 {
			pm.proxyLogger.Warnf(
				"Responses bridge upstream error: status=%d content-type=%q body=%s",
				statusCode,
				"",
				truncateForLog(string(respBody), 8000),
			)
			c.Data(statusCode, "application/json", respBody)
			return
		}
		respBody = bytes.TrimSpace(respBody)
		if len(respBody) == 0 {
			pm.proxyLogger.Warn("Responses bridge upstream returned empty body with success status")
			pm.sendErrorResponse(c, http.StatusBadGateway, "responses bridge upstream returned empty body")
			return
		}
		if !json.Valid(respBody) {
			pm.proxyLogger.Warnf(
				"Responses bridge upstream returned invalid JSON body: %s",
				truncateForLog(string(respBody), 8000),
			)
			pm.sendErrorResponse(c, http.StatusBadGateway, "responses bridge upstream returned invalid JSON")
			return
		}
		pm.proxyLogger.Warnf(
			"Responses bridge upstream success: status=%d content-type=%q body=%s",
			statusCode,
			"",
			truncateForLog(string(respBody), 120000),
		)
		out, err := translateChatCompletionToResponsesResponse(respBody)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error translating response: %s", err.Error()))
			return
		}
		pm.proxyLogger.Warnf("Responses bridge translated output: %s", truncateForLog(string(out), 120000))
		if responsesRequestedStream {
			writeResponsesStream(c, out)
			return
		}
		c.Data(statusCode, "application/json", out)
		return
	}

	if pm.metricsMonitor != nil && c.Request.Method == "POST" {
		if err := pm.metricsMonitor.wrapHandler(modelID, c.Writer, c.Request, nextHandler); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying metrics wrapped request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error Proxying Metrics Wrapped Request model %s", modelID)
			return
		}
	} else {
		if err := nextHandler(modelID, c.Writer, c.Request); err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
			pm.proxyLogger.Errorf("Error Proxying Request for model %s", modelID)
			return
		}
	}
}

type bridgeResponseRecorder struct {
	*httptest.ResponseRecorder
	closeChannel chan bool
}

func (r *bridgeResponseRecorder) CloseNotify() <-chan bool {
	return r.closeChannel
}

func translateResponsesToChatCompletionsRequest(body []byte) ([]byte, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	out := map[string]any{}
	copyField := func(key string) {
		if v, ok := req[key]; ok {
			out[key] = v
		}
	}
	for _, key := range []string{
		"model",
		"temperature",
		"top_p",
		"presence_penalty",
		"frequency_penalty",
		"stop",
		"n",
		"tool_choice",
		"parallel_tool_calls",
		"metadata",
	} {
		copyField(key)
	}
	// Responses API tool entries can differ from chat/completions.
	// Normalize to OpenAI chat format: {type:"function", function:{...}}.
	if toolsRaw, ok := req["tools"].([]any); ok {
		if normalized := normalizeChatTools(toolsRaw); len(normalized) > 0 {
			out["tools"] = normalized
		}
	}

	if v, ok := req["max_output_tokens"]; ok {
		out["max_tokens"] = v
	} else if v, ok := req["max_tokens"]; ok {
		out["max_tokens"] = v
	}

	messages := responsesRequestToChatMessages(req)
	if len(messages) == 0 {
		userText := extractResponsesInputText(req["input"])
		userText = strings.TrimSpace(cleanFallbackInput(req["input"], userText))
		messages = []map[string]any{{"role": "user", "content": userText}}
	}
	for _, m := range messages {
		content, _ := m["content"].(string)
		if strings.TrimSpace(content) == "" {
			m["content"] = " "
		}
	}
	out["messages"] = messages

	// Keep this non-streaming for now; we translate to a stable JSON response object.
	out["stream"] = false
	return json.Marshal(out)
}

func normalizeChatTools(toolsRaw []any) []any {
	out := make([]any, 0, len(toolsRaw))
	for _, t := range toolsRaw {
		m, ok := t.(map[string]any)
		if !ok {
			continue
		}

		// Already in chat format.
		if fn, ok := m["function"].(map[string]any); ok {
			if name, _ := fn["name"].(string); strings.TrimSpace(name) != "" {
				out = append(out, map[string]any{
					"type":     "function",
					"function": fn,
				})
			}
			continue
		}

		typ, _ := m["type"].(string)
		if strings.TrimSpace(typ) != "" && typ != "function" {
			// Drop tool kinds unsupported by chat/completions backends.
			continue
		}

		name, _ := m["name"].(string)
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}

		fn := map[string]any{"name": name}
		if desc, ok := m["description"].(string); ok && strings.TrimSpace(desc) != "" {
			fn["description"] = desc
		}
		if params, ok := m["parameters"]; ok {
			fn["parameters"] = params
		}
		if strict, ok := m["strict"]; ok {
			fn["strict"] = strict
		}

		out = append(out, map[string]any{
			"type":     "function",
			"function": fn,
		})
	}
	return out
}

func extractResponsesInputText(input any) string {
	parts := make([]string, 0)
	collectResponseText(input, &parts)
	return strings.Join(parts, "\n")
}

func collectResponseText(v any, out *[]string) {
	switch x := v.(type) {
	case string:
		s := strings.TrimSpace(x)
		if s != "" {
			*out = append(*out, s)
		}
	case []any:
		for _, item := range x {
			collectResponseText(item, out)
		}
	case map[string]any:
		for _, key := range []string{"input_text", "text", "content", "value"} {
			if child, ok := x[key]; ok {
				collectResponseText(child, out)
			}
		}
	}
}

func responsesRequestToChatMessages(req map[string]any) []map[string]any {
	out := make([]map[string]any, 0)

	if instructions, ok := req["instructions"].(string); ok && strings.TrimSpace(instructions) != "" {
		out = append(out, map[string]any{
			"role":    "system",
			"content": strings.TrimSpace(instructions),
		})
	}

	convertOne := func(role string, content any) {
		if content == nil {
			return
		}
		r := normalizeChatCompletionRole(role)
		txt := extractResponsesInputText(content)
		txt = strings.TrimSpace(cleanFallbackInput(content, txt))
		if txt == "" {
			return
		}
		out = append(out, map[string]any{
			"role":    r,
			"content": txt,
		})
	}

	appendAssistantToolCall := func(name string, arguments any, callID string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		args := encodeAnyAsJSONString(arguments)
		callID = strings.TrimSpace(callID)
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(out))
		}
		out = append(out, map[string]any{
			"role":    "assistant",
			"content": "",
			"tool_calls": []any{
				map[string]any{
					"id":   callID,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": args,
					},
				},
			},
		})
	}

	appendToolResult := func(callID string, output any) {
		callID = strings.TrimSpace(callID)
		if callID == "" {
			return
		}
		txt := extractResponsesInputText(output)
		txt = strings.TrimSpace(txt)
		if txt == "" {
			txt = encodeAnyAsJSONString(output)
		}
		txt = strings.TrimSpace(cleanFallbackInput(output, txt))
		if txt == "" {
			txt = " "
		}
		out = append(out, map[string]any{
			"role":         "tool",
			"tool_call_id": callID,
			"content":      txt,
		})
	}

	if messages, ok := req["messages"].([]any); ok {
		for _, m := range messages {
			obj, ok := m.(map[string]any)
			if !ok {
				continue
			}
			role, _ := obj["role"].(string)
			if strings.EqualFold(strings.TrimSpace(role), "assistant") {
				if tc, ok := obj["tool_calls"]; ok {
					assistantText := extractResponsesInputText(obj["content"])
					assistantText = strings.TrimSpace(cleanFallbackInput(obj["content"], assistantText))
					out = append(out, map[string]any{
						"role":       "assistant",
						"content":    assistantText,
						"tool_calls": tc,
					})
					continue
				}
			}
			if strings.EqualFold(strings.TrimSpace(role), "tool") {
				appendToolResult(
					strings.TrimSpace(fmt.Sprintf("%v", obj["tool_call_id"])),
					obj["content"],
				)
				continue
			}
			convertOne(role, obj["content"])
		}
		return out
	}

	if inputArr, ok := req["input"].([]any); ok {
		for _, it := range inputArr {
			obj, ok := it.(map[string]any)
			if !ok {
				continue
			}
			inputType := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", obj["type"])))
			switch inputType {
			case "function_call":
				appendAssistantToolCall(
					strings.TrimSpace(fmt.Sprintf("%v", obj["name"])),
					obj["arguments"],
					strings.TrimSpace(fmt.Sprintf("%v", obj["call_id"])),
				)
				continue
			case "function_call_output":
				appendToolResult(
					strings.TrimSpace(fmt.Sprintf("%v", obj["call_id"])),
					obj["output"],
				)
				continue
			case "message":
				role, _ := obj["role"].(string)
				convertOne(role, obj["content"])
				continue
			}
			role, _ := obj["role"].(string)
			if c, ok := obj["content"]; ok {
				convertOne(role, c)
			} else {
				convertOne(role, obj)
			}
		}
		return out
	}

	if input, ok := req["input"]; ok {
		convertOne("user", input)
	}
	return out
}

func normalizeChatCompletionRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "user", "assistant", "tool":
		return strings.ToLower(strings.TrimSpace(role))
	case "developer":
		// Local chat-completions backends often don't support "developer".
		// Preserve intent by mapping it to system instructions.
		return "system"
	default:
		return "user"
	}
}

func cleanFallbackInput(raw any, preferred string) string {
	if strings.TrimSpace(preferred) != "" {
		return preferred
	}
	if raw == nil {
		return ""
	}
	s := strings.TrimSpace(fmt.Sprintf("%v", raw))
	lower := strings.ToLower(s)
	if s == "" || lower == "<nil>" || lower == "null" || lower == "[]" || lower == "{}" {
		return ""
	}
	return s
}

func encodeAnyAsJSONString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(x)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return strings.TrimSpace(fmt.Sprintf("%v", v))
		}
		return strings.TrimSpace(string(b))
	}
}

func translateChatCompletionToResponsesResponse(body []byte) ([]byte, error) {
	message := gjson.GetBytes(body, "choices.0.message")
	if !message.Exists() || strings.TrimSpace(message.Raw) == "" || message.Type == gjson.Null {
		return nil, errors.New("chat completion missing choices[0].message")
	}

	id := strings.TrimSpace(gjson.GetBytes(body, "id").String())
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	text := message.Get("content").String()
	output := make([]any, 0, 2)

	toolCalls := message.Get("tool_calls")
	functionCall := message.Get("function_call")
	hasToolCalls := toolCalls.IsArray() && len(toolCalls.Array()) > 0
	hasFunctionCall := functionCall.Exists() && strings.TrimSpace(functionCall.Get("name").String()) != ""

	if strings.TrimSpace(text) != "" || (!hasToolCalls && !hasFunctionCall) {
		output = append(output, map[string]any{
			"id":   "msg_" + id,
			"type": "message",
			"role": "assistant",
			"content": []any{
				map[string]any{
					"type": "output_text",
					"text": text,
				},
			},
		})
	}

	appendFunctionCall := func(callID, name, arguments string, index int) {
		callID = strings.TrimSpace(callID)
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), index)
		}
		output = append(output, map[string]any{
			"id":        fmt.Sprintf("fc_%s", callID),
			"type":      "function_call",
			"call_id":   callID,
			"name":      strings.TrimSpace(name),
			"arguments": arguments,
			"status":    "completed",
		})
	}

	if hasToolCalls {
		idx := 0
		toolCalls.ForEach(func(_, tc gjson.Result) bool {
			appendFunctionCall(
				tc.Get("id").String(),
				tc.Get("function.name").String(),
				tc.Get("function.arguments").String(),
				idx,
			)
			idx++
			return true
		})
	} else if hasFunctionCall {
		appendFunctionCall(
			"",
			functionCall.Get("name").String(),
			functionCall.Get("arguments").String(),
			0,
		)
	}

	resp := map[string]any{
		"id":          "resp_" + id,
		"object":      "response",
		"created_at":  time.Now().Unix(),
		"status":      "completed",
		"model":       model,
		"output":      output,
		"output_text": text,
	}
	if created := gjson.GetBytes(body, "created").Int(); created > 0 {
		resp["created_at"] = created
	}

	usage := gjson.GetBytes(body, "usage")
	if usage.Exists() {
		resp["usage"] = map[string]any{
			"input_tokens":  usage.Get("prompt_tokens").Int(),
			"output_tokens": usage.Get("completion_tokens").Int(),
			"total_tokens":  usage.Get("total_tokens").Int(),
		}
	}
	return json.Marshal(resp)
}

func truncateForLog(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "...<truncated>"
}

func safeHeadersJSON(h http.Header) string {
	clone := make(map[string][]string, len(h))
	for k, v := range h {
		keyLower := strings.ToLower(strings.TrimSpace(k))
		if keyLower == "authorization" || keyLower == "x-api-key" {
			clone[k] = []string{"<redacted>"}
			continue
		}
		clone[k] = append([]string(nil), v...)
	}
	b, err := json.Marshal(clone)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func decodeRequestByContentEncoding(body []byte, encodingHeader string) ([]byte, error) {
	encoding := strings.ToLower(strings.TrimSpace(encodingHeader))
	if encoding == "" || encoding == "identity" {
		return body, nil
	}

	// Handle headers such as "zstd, br" by taking the first encoding token.
	if idx := strings.Index(encoding, ","); idx > 0 {
		encoding = strings.TrimSpace(encoding[:idx])
	}

	switch encoding {
	case "gzip":
		r, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer r.Close()
		return io.ReadAll(r)
	case "deflate":
		r := flate.NewReader(bytes.NewReader(body))
		defer r.Close()
		return io.ReadAll(r)
	case "zstd":
		r, err := zstd.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer r.Close()
		return io.ReadAll(r)
	default:
		return nil, fmt.Errorf("unsupported content-encoding: %s", encoding)
	}
}

func writeResponsesStream(c *gin.Context, responseJSON []byte) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	sequence := 0
	writeEvent := func(eventType string, payload map[string]any) {
		if _, ok := payload["type"]; !ok {
			payload["type"] = eventType
		}
		payload["sequence_number"] = sequence
		sequence++
		data, _ := json.Marshal(payload)
		_, _ = c.Writer.Write([]byte("event: " + eventType + "\n"))
		_, _ = c.Writer.Write([]byte("data: " + string(data) + "\n\n"))
		c.Writer.Flush()
	}

	var full map[string]any
	if err := json.Unmarshal(responseJSON, &full); err != nil {
		full = map[string]any{}
	}

	respID := strings.TrimSpace(gjson.GetBytes(responseJSON, "id").String())
	if respID == "" {
		respID = fmt.Sprintf("resp_%d", time.Now().UnixNano())
	}
	createdAt := gjson.GetBytes(responseJSON, "created_at").Int()
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	model := strings.TrimSpace(gjson.GetBytes(responseJSON, "model").String())
	responseSkeleton := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": createdAt,
		"model":      model,
		"status":     "in_progress",
		"output":     []any{},
	}

	writeEvent("response.created", map[string]any{
		"type":     "response.created",
		"response": responseSkeleton,
	})
	writeEvent("response.in_progress", map[string]any{
		"type":     "response.in_progress",
		"response": responseSkeleton,
	})

	emitMessagePart := func(itemID string, outputIndex int, contentIndex int, part map[string]any) {
		partType := strings.TrimSpace(fmt.Sprintf("%v", part["type"]))
		if partType == "" {
			partType = "output_text"
		}
		text := strings.TrimSpace(fmt.Sprintf("%v", part["text"]))
		partAdded := map[string]any{"type": partType}
		partDone := map[string]any{"type": partType}
		if partType == "output_text" {
			partAdded["text"] = ""
			partDone["text"] = text
		} else if text != "" {
			partAdded["text"] = text
			partDone["text"] = text
		}
		writeEvent("response.content_part.added", map[string]any{
			"type":          "response.content_part.added",
			"response_id":   respID,
			"item_id":       itemID,
			"output_index":  outputIndex,
			"content_index": contentIndex,
			"part":          partAdded,
		})
		if partType == "output_text" && text != "" {
			writeEvent("response.output_text.delta", map[string]any{
				"type":          "response.output_text.delta",
				"response_id":   respID,
				"item_id":       itemID,
				"output_index":  outputIndex,
				"content_index": contentIndex,
				"delta":         text,
			})
		}
		if partType == "output_text" {
			writeEvent("response.output_text.done", map[string]any{
				"type":          "response.output_text.done",
				"response_id":   respID,
				"item_id":       itemID,
				"output_index":  outputIndex,
				"content_index": contentIndex,
				"text":          text,
			})
		}
		writeEvent("response.content_part.done", map[string]any{
			"type":          "response.content_part.done",
			"response_id":   respID,
			"item_id":       itemID,
			"output_index":  outputIndex,
			"content_index": contentIndex,
			"part":          partDone,
		})
	}

	output := gjson.GetBytes(responseJSON, "output").Array()
	if len(output) == 0 {
		// Fallback for text-only responses missing output array.
		text := gjson.GetBytes(responseJSON, "output_text").String()
		if strings.TrimSpace(text) != "" {
			fallbackItemID := "msg_" + respID
			item := map[string]any{
				"id":      fallbackItemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": []any{map[string]any{"type": "output_text", "text": text}},
			}
			writeEvent("response.output_item.added", map[string]any{
				"type":         "response.output_item.added",
				"response_id":  respID,
				"output_index": 0,
				"item":         item,
			})
			emitMessagePart(fallbackItemID, 0, 0, map[string]any{
				"type": "output_text",
				"text": text,
			})
			writeEvent("response.output_item.done", map[string]any{
				"type":         "response.output_item.done",
				"response_id":  respID,
				"output_index": 0,
				"item":         item,
			})
		}
	}

	for i, itemResult := range output {
		itemRaw := strings.TrimSpace(itemResult.Raw)
		if itemRaw == "" {
			continue
		}
		item := map[string]any{}
		if err := json.Unmarshal([]byte(itemRaw), &item); err != nil {
			continue
		}
		itemType := strings.TrimSpace(fmt.Sprintf("%v", item["type"]))
		itemID := strings.TrimSpace(fmt.Sprintf("%v", item["id"]))
		if itemID == "" {
			if itemType == "function_call" {
				itemID = fmt.Sprintf("fc_%s_%d", respID, i)
			} else {
				itemID = fmt.Sprintf("msg_%s_%d", respID, i)
			}
			item["id"] = itemID
		}

		writeEvent("response.output_item.added", map[string]any{
			"type":         "response.output_item.added",
			"response_id":  respID,
			"output_index": i,
			"item":         item,
		})

		if itemType == "message" {
			if content, ok := item["content"].([]any); ok {
				for contentIndex, contentPart := range content {
					part, ok := contentPart.(map[string]any)
					if !ok {
						continue
					}
					emitMessagePart(itemID, i, contentIndex, part)
				}
			}
		}

		if itemType == "function_call" {
			args := encodeAnyAsJSONString(item["arguments"])
			callID := strings.TrimSpace(fmt.Sprintf("%v", item["call_id"]))
			if callID == "" {
				callID = itemID
			}
			if args != "" {
				writeEvent("response.function_call_arguments.delta", map[string]any{
					"type":         "response.function_call_arguments.delta",
					"response_id":  respID,
					"output_index": i,
					"item_id":      itemID,
					"delta":        args,
				})
			}
			writeEvent("response.function_call_arguments.done", map[string]any{
				"type":         "response.function_call_arguments.done",
				"response_id":  respID,
				"output_index": i,
				"item_id":      itemID,
				"call_id":      callID,
				"arguments":    args,
			})
		}

		writeEvent("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"response_id":  respID,
			"output_index": i,
			"item":         item,
		})
	}

	if len(full) > 0 {
		writeEvent("response.completed", map[string]any{
			"type":     "response.completed",
			"response": full,
		})
	} else {
		writeEvent("response.completed", map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":     respID,
				"object": "response",
				"status": "completed",
			},
		})
	}

	_, _ = c.Writer.Write([]byte("data: [DONE]\n\n"))
	c.Writer.Flush()
}

func (pm *ProxyManager) proxyWithToolsIfNeeded(
	c *gin.Context,
	modelID string,
	nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error,
	bodyBytes []byte,
) (bool, error) {
	if len(pm.getEnabledTools()) == 0 {
		return false, nil
	}
	if !gjson.GetBytes(bodyBytes, "messages").IsArray() {
		return false, nil
	}

	originalStream := gjson.GetBytes(bodyBytes, "stream").Bool()
	working, err := sjson.SetBytes(bodyBytes, "stream", false)
	if err != nil {
		return false, err
	}
	working, err = pm.injectToolSchemas(working)
	if err != nil {
		return false, err
	}

	maxIterations := pm.getToolRuntimeSettings().MaxToolRounds
	finalBody, statusCode, err := pm.runToolLoop(modelID, nextHandler, c.Request, working, maxIterations)
	if err != nil {
		return false, err
	}

	if !originalStream {
		c.Data(statusCode, "application/json", finalBody)
		return true, nil
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	content := gjson.GetBytes(finalBody, "choices.0.message.content").String()
	reasoning := gjson.GetBytes(finalBody, "choices.0.message.reasoning_content").String()
	sourcesRaw := gjson.GetBytes(finalBody, "choices.0.message.sources").Raw
	var sources any = []any{}
	if strings.TrimSpace(sourcesRaw) != "" {
		_ = json.Unmarshal([]byte(sourcesRaw), &sources)
	}
	chunk := map[string]any{
		"id":      fmt.Sprintf("chatcmpl-tools-%d", time.Now().UnixNano()),
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   modelID,
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"role":              "assistant",
					"content":           content,
					"reasoning_content": reasoning,
					"sources":           sources,
				},
				"finish_reason": "stop",
			},
		},
	}
	data, _ := json.Marshal(chunk)
	_, _ = c.Writer.Write([]byte("data: " + string(data) + "\n\n"))
	_, _ = c.Writer.Write([]byte("data: [DONE]\n\n"))
	c.Writer.Flush()
	return true, nil
}

func (pm *ProxyManager) injectToolSchemas(body []byte) ([]byte, error) {
	schemas := pm.toolSchemas()
	if len(schemas) == 0 {
		return body, nil
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	existing, _ := req["tools"].([]any)
	existingNames := map[string]struct{}{}
	for _, t := range existing {
		m, ok := t.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := m["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if strings.TrimSpace(name) != "" {
			existingNames[name] = struct{}{}
		}
	}

	merged := append([]any{}, existing...)
	for _, s := range schemas {
		fn, _ := s["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if _, found := existingNames[name]; found {
			continue
		}
		merged = append(merged, s)
	}
	req["tools"] = merged
	if _, hasChoice := req["tool_choice"]; !hasChoice {
		if forced := pm.forcedToolName(body); strings.TrimSpace(forced) != "" {
			req["tool_choice"] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": forced,
				},
			}
		}
	}
	req["stream"] = false
	return json.Marshal(req)
}

func (pm *ProxyManager) invokeInferenceOnce(
	modelID string,
	nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error,
	orig *http.Request,
	body []byte,
) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(orig.Context(), orig.Method, orig.URL.String(), bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header = orig.Header.Clone()
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Length", strconv.Itoa(len(body)))
	req.Header.Del("Transfer-Encoding")
	req.ContentLength = int64(len(body))
	pm.recordActivityPromptPreview(modelID, req.URL.Path, body, req.Header)

	rr := httptest.NewRecorder()
	testCtx, _ := gin.CreateTestContext(rr)
	testCtx.Request = req
	if pm.metricsMonitor != nil && req.Method == http.MethodPost {
		if err := pm.metricsMonitor.wrapHandler(modelID, testCtx.Writer, req, nextHandler); err != nil {
			return nil, 0, err
		}
	} else {
		if err := nextHandler(modelID, testCtx.Writer, req); err != nil {
			return nil, 0, err
		}
	}
	status := rr.Code
	if status == 0 {
		status = http.StatusOK
	}
	return rr.Body.Bytes(), status, nil
}

func (pm *ProxyManager) runToolLoop(
	modelID string,
	nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error,
	orig *http.Request,
	initialBody []byte,
	maxIterations int,
) ([]byte, int, error) {
	working := initialBody
	finalBody := initialBody
	finalStatus := http.StatusOK
	sourceMap := map[string]chatSource{}
	attachSources := func(body []byte, status int) []byte {
		if len(sourceMap) == 0 || status < 200 || status >= 300 {
			return body
		}
		sources := make([]chatSource, 0, len(sourceMap))
		for _, src := range sourceMap {
			sources = append(sources, src)
		}
		sort.Slice(sources, func(i, j int) bool {
			return sources[i].URL < sources[j].URL
		})
		if b, err := json.Marshal(sources); err == nil {
			if withSources, err := sjson.SetRawBytes(body, "choices.0.message.sources", b); err == nil {
				return withSources
			}
		}
		return body
	}
	interactiveApproval := isTruthyHeader(orig.Header, "X-LlamaSwap-Tool-Approval-Interactive")
	approvalHeaderName := pm.getToolRuntimeSettings().ApprovalHeaderName
	if strings.TrimSpace(approvalHeaderName) == "" {
		approvalHeaderName = "X-LlamaSwap-Tool-Approval"
	}
	approvedNow := isTruthyHeader(orig.Header, approvalHeaderName)

	for i := 0; i < maxIterations; i++ {
		respBody, statusCode, err := pm.invokeInferenceOnce(modelID, nextHandler, orig, working)
		if err != nil {
			return nil, 0, err
		}
		finalBody = respBody
		finalStatus = statusCode
		if statusCode < 200 || statusCode >= 300 {
			return attachSources(finalBody, finalStatus), finalStatus, nil
		}

		toolCalls := gjson.GetBytes(respBody, "choices.0.message.tool_calls")
		hasToolCalls := toolCalls.IsArray() && len(toolCalls.Array()) > 0
		functionCall := gjson.GetBytes(respBody, "choices.0.message.function_call")
		hasFunctionCall := functionCall.Exists() && strings.TrimSpace(functionCall.Get("name").String()) != ""
		settings := pm.getToolRuntimeSettings()
		embeddedCalls := make([]ToolApprovalCall, 0)
		if !hasToolCalls && !hasFunctionCall && settings.WatchdogMode != "off" {
			assistantText := strings.TrimSpace(gjson.GetBytes(respBody, "choices.0.message.content").String())
			embeddedCalls = parseEmbeddedToolCalls(assistantText)
		} else if !hasToolCalls && !hasFunctionCall {
			assistantText := strings.TrimSpace(gjson.GetBytes(respBody, "choices.0.message.content").String())
			parsed := parseEmbeddedToolCalls(assistantText)
			for _, call := range parsed {
				name := strings.TrimSpace(call.Name)
				if name == "" {
					continue
				}
				tool, ok := pm.toolByName(name)
				if !ok {
					continue
				}
				if tool.Policy == ToolPolicyWatchdog {
					embeddedCalls = append(embeddedCalls, call)
				}
			}
		}
		if !hasToolCalls && !hasFunctionCall && len(embeddedCalls) == 0 {
			return attachSources(finalBody, finalStatus), finalStatus, nil
		}

		var reqMap map[string]any
		if err := json.Unmarshal(working, &reqMap); err != nil {
			return nil, 0, err
		}
		rawMessages, _ := reqMap["messages"].([]any)

		var assistantMsg map[string]any
		if err := json.Unmarshal([]byte(gjson.GetBytes(respBody, "choices.0.message").Raw), &assistantMsg); err == nil {
			rawMessages = append(rawMessages, assistantMsg)
		}

		pendingCalls := make([]ToolApprovalCall, 0)
		if hasToolCalls {
			toolCalls.ForEach(func(_, tc gjson.Result) bool {
				callID := strings.TrimSpace(tc.Get("id").String())
				toolName := strings.TrimSpace(tc.Get("function.name").String())
				argText := tc.Get("function.arguments").String()
				args := map[string]any{}
				if strings.TrimSpace(argText) != "" {
					_ = json.Unmarshal([]byte(argText), &args)
				}
				pendingCalls = append(pendingCalls, ToolApprovalCall{Name: toolName, CallID: callID, Args: args})
				return true
			})
		} else if hasFunctionCall {
			toolName := strings.TrimSpace(functionCall.Get("name").String())
			argText := functionCall.Get("arguments").String()
			args := map[string]any{}
			if strings.TrimSpace(argText) != "" {
				_ = json.Unmarshal([]byte(argText), &args)
			}
			pendingCalls = append(pendingCalls, ToolApprovalCall{Name: toolName, Args: args})
		} else if len(embeddedCalls) > 0 {
			pendingCalls = append(pendingCalls, embeddedCalls...)
		}

		if interactiveApproval && !approvedNow && len(pendingCalls) > 0 {
			return nil, 0, &ToolApprovalRequiredError{
				HeaderName: approvalHeaderName,
				ToolCalls:  pendingCalls,
			}
		}

		for _, call := range pendingCalls {
			toolName := strings.TrimSpace(call.Name)
			args := call.Args
			if args == nil {
				args = map[string]any{}
			}
			out, execErr := pm.executeToolCall(toolName, args, orig.Header)
			if execErr != nil {
				out = fmt.Sprintf("tool error: %v", execErr)
			}
			for _, src := range extractSourcesFromToolOutput(out) {
				if strings.TrimSpace(src.URL) == "" {
					continue
				}
				sourceMap[src.URL] = src
			}
			msg := map[string]any{
				"role":    "tool",
				"name":    toolName,
				"content": out,
			}
			if strings.TrimSpace(call.CallID) != "" {
				msg["tool_call_id"] = call.CallID
			}
			rawMessages = append(rawMessages, msg)
		}

		reqMap["messages"] = rawMessages
		reqMap["stream"] = false
		// After executing at least one tool call, force the next pass to produce
		// a final assistant answer instead of repeatedly calling tools.
		reqMap["tool_choice"] = "none"
		nextBody, err := json.Marshal(reqMap)
		if err != nil {
			return nil, 0, err
		}
		working = nextBody
	}
	return attachSources(finalBody, finalStatus), finalStatus, nil
}

func parseEmbeddedToolCalls(content string) []ToolApprovalCall {
	text := strings.TrimSpace(content)
	if text == "" {
		return nil
	}
	matches := toolCallTagRegex.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	out := make([]ToolApprovalCall, 0, len(matches))
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		raw := strings.TrimSpace(match[1])
		if raw == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		name := strings.TrimSpace(fmt.Sprintf("%v", obj["name"]))
		if name == "" || strings.EqualFold(name, "<nil>") {
			continue
		}
		args := map[string]any{}
		if v, ok := obj["arguments"]; ok {
			if m, ok := asMap(v); ok {
				args = m
			} else if m, ok := decodeJSONStringMap(v); ok {
				args = m
			}
		}
		out = append(out, ToolApprovalCall{
			Name: name,
			Args: args,
		})
	}
	return out
}

func extractSourcesFromToolOutput(out string) []chatSource {
	s := strings.TrimSpace(out)
	if s == "" {
		return nil
	}

	sources := map[string]chatSource{}

	// Try JSON first (searxng direct payloads).
	if gjson.Valid(s) {
		results := gjson.Get(s, "results")
		if results.IsArray() {
			results.ForEach(func(_, v gjson.Result) bool {
				u := strings.TrimSpace(v.Get("url").String())
				if u == "" {
					return true
				}
				title := strings.TrimSpace(v.Get("title").String())
				domain := sourceDomainFromURL(u)
				sources[u] = chatSource{URL: u, Title: title, Domain: domain}
				return true
			})
		}
	}

	// Parse plain-text URLs from compact tool output.
	for _, token := range strings.Fields(s) {
		t := strings.TrimSpace(token)
		t = strings.Trim(t, "[](){}<>,.;'\"")
		if !strings.HasPrefix(t, "http://") && !strings.HasPrefix(t, "https://") {
			continue
		}
		if _, err := url.ParseRequestURI(t); err != nil {
			continue
		}
		if _, exists := sources[t]; !exists {
			sources[t] = chatSource{
				URL:    t,
				Domain: sourceDomainFromURL(t),
			}
		}
	}

	outSources := make([]chatSource, 0, len(sources))
	for _, src := range sources {
		outSources = append(outSources, src)
	}
	return outSources
}

func sourceDomainFromURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(u.Hostname())
}

func (pm *ProxyManager) applyPromptSizeControl(modelID string, bodyBytes []byte) ([]byte, PromptOptimizationResult, error) {
	pm.Lock()
	ctxSize := pm.ctxSizes[modelID]
	runtimePolicy, hasRuntimePolicy := pm.promptPolicies[modelID]
	pm.Unlock()
	result := PromptOptimizationResult{
		Policy:  PromptOptimizationLimitOnly,
		Applied: false,
		Note:    "no optimization",
	}

	if !gjson.GetBytes(bodyBytes, "messages").IsArray() {
		return bodyBytes, result, nil
	}

	var chatReq ChatRequest
	if err := json.Unmarshal(bodyBytes, &chatReq); err != nil {
		return nil, result, fmt.Errorf("invalid chat request JSON: %w", err)
	}

	modelConfig, exists := pm.config.Models[modelID]
	if !exists {
		if !isOllamaModelID(modelID) {
			return bodyBytes, result, nil
		}
		modelConfig = config.ModelConfig{
			Proxy:          pm.ollamaEndpoint,
			TruncationMode: string(SlidingWindow),
		}
	}

	policy := PromptOptimizationLimitOnly
	if hasRuntimePolicy {
		policy = runtimePolicy
	}
	result.Policy = policy
	if policy == PromptOptimizationOff {
		result.Note = "optimization disabled"
		pm.savePromptOptimizationSnapshot(modelID, policy, false, bodyBytes, bodyBytes, result.Note)
		return bodyBytes, result, nil
	}

	mode := SlidingWindow
	switch policy {
	case PromptOptimizationAlways:
		chatReq.Messages = CompactMessagesForLowVRAM(chatReq.Messages)
		mode = SlidingWindow
		result.Applied = true
		result.Note = "always compacted repeated content"
	case PromptOptimizationLimitOnly:
		switch strings.ToLower(strings.TrimSpace(modelConfig.TruncationMode)) {
		case string(StrictError):
			mode = StrictError
		default:
			mode = SlidingWindow
		}
	case PromptOptimizationLLMAssist:
		assisted, assistedErr := pm.optimizeMessagesWithLLM(modelConfig, chatReq)
		if assistedErr != nil {
			pm.proxyLogger.Warnf("<%s> LLM-assisted optimization failed, falling back to compact mode: %v", modelID, assistedErr)
			assisted.Messages = CompactMessagesForLowVRAM(chatReq.Messages)
		}
		chatReq = assisted
		mode = SlidingWindow
		result.Applied = true
		result.Note = "llm-assisted compression applied"
	default:
		mode = SlidingWindow
	}

	if ctxSize <= 0 {
		if policy != PromptOptimizationAlways {
			updatedBody, err := json.Marshal(chatReq)
			if err != nil {
				return nil, result, fmt.Errorf("failed to serialize optimized chat request: %w", err)
			}
			changed := !bytes.Equal(updatedBody, bodyBytes)
			result.Applied = result.Applied || changed
			if !result.Applied {
				result.Note = "no context limit configured"
			}
			pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
			return updatedBody, result, nil
		}
		updatedBody, err := sjson.SetBytes(bodyBytes, "messages", chatReq.Messages)
		if err != nil {
			return nil, result, fmt.Errorf("failed to update chat messages: %w", err)
		}
		changed := !bytes.Equal(updatedBody, bodyBytes)
		result.Applied = result.Applied || changed
		pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
		return updatedBody, result, nil
	}

	cm := NewContextManager(modelID, ctxSize, mode, pm.proxyLogger, modelConfig.Proxy)
	cropped, err := cm.CropChatRequest(chatReq)
	if err != nil {
		return nil, result, err
	}

	updatedBody := bodyBytes
	updatedBody, err = sjson.SetBytes(updatedBody, "messages", cropped.Messages)
	if err != nil {
		return nil, result, fmt.Errorf("failed to update chat messages: %w", err)
	}

	if len(chatReq.Tools) > 0 || len(cropped.Tools) > 0 {
		updatedBody, err = sjson.SetBytes(updatedBody, "tools", cropped.Tools)
		if err != nil {
			return nil, result, fmt.Errorf("failed to update chat tools: %w", err)
		}
	}

	if cropped.IsCropped() || !bytes.Equal(updatedBody, bodyBytes) {
		result.Applied = true
		if result.Note == "no optimization" {
			result.Note = "cropped to context limit"
		}
		pm.proxyLogger.Infof("<%s> Prompt was compacted to fit ctx-size=%d using mode=%s", modelID, ctxSize, mode)
	}

	pm.savePromptOptimizationSnapshot(modelID, policy, result.Applied, bodyBytes, updatedBody, result.Note)
	return updatedBody, result, nil
}

func (pm *ProxyManager) savePromptOptimizationSnapshot(
	modelID string,
	policy PromptOptimizationPolicy,
	applied bool,
	originalBody []byte,
	optimizedBody []byte,
	note string,
) {
	const maxSnapshotBytes = 2 * 1024 * 1024
	toSafeString := func(data []byte) string {
		if len(data) <= maxSnapshotBytes {
			return string(data)
		}
		return string(data[:maxSnapshotBytes]) + "\n...<truncated>"
	}

	snapshot := PromptOptimizationSnapshot{
		Model:         modelID,
		Policy:        policy,
		Applied:       applied,
		UpdatedAt:     time.Now().UTC().Format(time.RFC3339),
		Note:          note,
		OriginalBody:  toSafeString(originalBody),
		OptimizedBody: toSafeString(optimizedBody),
	}

	pm.Lock()
	pm.latestPromptOptimizations[modelID] = snapshot
	pm.Unlock()
}

func (pm *ProxyManager) optimizeMessagesWithLLM(modelConfig config.ModelConfig, req ChatRequest) (ChatRequest, error) {
	if len(req.Messages) < 4 {
		return req, nil
	}

	keepTail := 4
	if keepTail > len(req.Messages) {
		keepTail = len(req.Messages)
	}
	middleEnd := len(req.Messages) - keepTail
	if middleEnd <= 1 {
		return req, nil
	}

	keepPrefix := 0
	if req.Messages[0].Role == "system" {
		keepPrefix = 1
	}
	middle := req.Messages[keepPrefix:middleEnd]
	if len(middle) == 0 {
		return req, nil
	}

	var b strings.Builder
	for _, m := range middle {
		contentText := chatContentToText(m.Content)
		if strings.TrimSpace(contentText) == "" {
			continue
		}
		b.WriteString("[")
		b.WriteString(strings.ToUpper(m.Role))
		b.WriteString("] ")
		b.WriteString(contentText)
		b.WriteString("\n\n")
		if b.Len() > 12000 {
			break
		}
	}

	summaryInput := b.String()
	if strings.TrimSpace(summaryInput) == "" {
		return req, nil
	}

	upstreamModelName := strings.TrimSpace(modelConfig.UseModelName)
	if upstreamModelName == "" {
		upstreamModelName = strings.TrimSpace(req.Model)
	}
	if upstreamModelName == "" {
		upstreamModelName = "model"
	}

	llmReq := map[string]any{
		"model": upstreamModelName,
		"messages": []map[string]any{
			{
				"role":    "system",
				"content": "Summarize the following chat history for coding continuity. Keep requirements, constraints, file paths, decisions, TODOs, open questions. Be concise. Do not add new facts.",
			},
			{
				"role":    "user",
				"content": summaryInput,
			},
		},
		"max_tokens":  512,
		"temperature": 0,
		"stream":      false,
	}

	reqBytes, err := json.Marshal(llmReq)
	if err != nil {
		return req, err
	}

	url := strings.TrimSuffix(modelConfig.Proxy, "/") + "/v1/chat/completions"
	resp, err := http.Post(url, "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		return req, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return req, fmt.Errorf("llm assistant upstream status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return req, err
	}
	summary := strings.TrimSpace(gjson.GetBytes(body, "choices.0.message.content").String())
	if summary == "" {
		return req, fmt.Errorf("llm assistant returned empty summary")
	}

	newMessages := make([]ChatMessage, 0, keepPrefix+1+keepTail)
	if keepPrefix == 1 {
		newMessages = append(newMessages, req.Messages[0])
	}
	newMessages = append(newMessages, ChatMessage{
		Role:    "system",
		Content: "LLM-assisted context summary:\n" + summary,
	})
	newMessages = append(newMessages, req.Messages[middleEnd:]...)

	req.Messages = newMessages
	return req, nil
}

func (pm *ProxyManager) SetConfigPath(configPath string) {
	pm.Lock()
	pm.configPath = strings.TrimSpace(configPath)
	pm.Unlock()
	pm.loadToolsFromDisk()
}

func (pm *ProxyManager) proxyOAIPostFormHandler(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil { // 32MB max memory, larger files go to tmp disk
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("error parsing multipart form: %s", err.Error()))
		return
	}

	// Get model parameter from the form
	requestedModel := c.Request.FormValue("model")
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' parameter in form data")
		return
	}

	// Look for a matching local model first, then check peers
	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error
	var useModelName string

	modelID, found := pm.config.RealModelName(requestedModel)
	if found {
		processGroup, err := pm.swapProcessGroup(modelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}

		useModelName = pm.config.Models[modelID].UseModelName
		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		modelID = requestedModel
		nextHandler = pm.peerProxy.ProxyRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable handler for %s", requestedModel))
		return
	}

	// We need to reconstruct the multipart form in any case since the body is consumed
	// Create a new buffer for the reconstructed request
	var requestBuffer bytes.Buffer
	multipartWriter := multipart.NewWriter(&requestBuffer)

	// Copy all form values
	for key, values := range c.Request.MultipartForm.Value {
		for _, value := range values {
			fieldValue := value
			// If this is the model field and we have a profile, use just the model name
			if key == "model" {
				// # issue #69 allow custom model names to be sent to upstream
				if useModelName != "" {
					fieldValue = useModelName
				} else {
					fieldValue = requestedModel
				}
			}
			field, err := multipartWriter.CreateFormField(key)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form field")
				return
			}
			if _, err = field.Write([]byte(fieldValue)); err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error writing form field")
				return
			}
		}
	}

	// Copy all files from the original request
	for key, fileHeaders := range c.Request.MultipartForm.File {
		for _, fileHeader := range fileHeaders {
			formFile, err := multipartWriter.CreateFormFile(key, fileHeader.Filename)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form file")
				return
			}

			file, err := fileHeader.Open()
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error opening uploaded file")
				return
			}

			if _, err = io.Copy(formFile, file); err != nil {
				file.Close()
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error copying file data")
				return
			}
			file.Close()
		}
	}

	// Close the multipart writer to finalize the form
	if err := multipartWriter.Close(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error finalizing multipart form")
		return
	}

	// Create a new request with the reconstructed form data
	modifiedReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		c.Request.URL.String(),
		&requestBuffer,
	)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error creating modified request")
		return
	}

	// Copy the headers from the original request
	modifiedReq.Header = c.Request.Header.Clone()
	modifiedReq.Header.Set("Content-Type", multipartWriter.FormDataContentType())

	// set the content length of the body
	modifiedReq.Header.Set("Content-Length", strconv.Itoa(requestBuffer.Len()))
	modifiedReq.ContentLength = int64(requestBuffer.Len())

	// Use the modified request for proxying
	if err := nextHandler(modelID, c.Writer, modifiedReq); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying Request for model %s", modelID)
		return
	}
}

func (pm *ProxyManager) proxyGETModelHandler(c *gin.Context) {
	requestedModel := c.Query("model")
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing required 'model' query parameter")
		return
	}

	var nextHandler func(modelID string, w http.ResponseWriter, r *http.Request) error
	var modelID string

	if realModelID, found := pm.config.RealModelName(requestedModel); found {
		processGroup, err := pm.swapProcessGroup(realModelID)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
			return
		}
		modelID = realModelID
		pm.proxyLogger.Debugf("ProxyManager using local Process for model: %s", requestedModel)
		nextHandler = processGroup.ProxyRequest
	} else if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		modelID = requestedModel
		pm.proxyLogger.Debugf("ProxyManager using ProxyPeer for model: %s", requestedModel)
		nextHandler = pm.peerProxy.ProxyRequest
	}

	if nextHandler == nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find suitable handler for %s", requestedModel))
		return
	}

	if err := nextHandler(modelID, c.Writer, c.Request); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying GET Request for model %s", modelID)
		return
	}
}

func (pm *ProxyManager) sendErrorResponse(c *gin.Context, statusCode int, message string) {
	acceptHeader := c.GetHeader("Accept")
	isInference := compat.IsInferencePath(c.Request.URL.Path)

	if isInference && pm.compatibilityMode() == "strict_openai" {
		c.JSON(statusCode, compat.NewErrorEnvelope(statusCode, message, ""))
		return
	}

	if strings.Contains(acceptHeader, "application/json") || isInference {
		c.JSON(statusCode, compat.NewErrorEnvelope(statusCode, message, ""))
	} else {
		c.String(statusCode, message)
	}
}

func (pm *ProxyManager) compatibilityMode() string {
	mode := strings.ToLower(strings.TrimSpace(pm.config.CompatibilityMode))
	if mode == "" {
		return "legacy"
	}
	return mode
}

// apiKeyAuth returns a middleware that validates API keys if configured.
// Returns a pass-through handler if no API keys are configured.
func (pm *ProxyManager) apiKeyAuth() gin.HandlerFunc {
	if len(pm.config.RequiredAPIKeys) == 0 {
		return func(c *gin.Context) { c.Next() }
	}

	return func(c *gin.Context) {
		xApiKey := c.GetHeader("x-api-key")

		var bearerKey string
		var basicKey string
		if auth := c.GetHeader("Authorization"); auth != "" {
			if strings.HasPrefix(auth, "Bearer ") {
				bearerKey = strings.TrimPrefix(auth, "Bearer ")
			} else if strings.HasPrefix(auth, "Basic ") {
				// Basic Auth: base64(username:password), password is the API key
				encoded := strings.TrimPrefix(auth, "Basic ")
				if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
					parts := strings.SplitN(string(decoded), ":", 2)
					if len(parts) == 2 {
						basicKey = parts[1] // password is the API key
					}
				}
			}
		}

		// Use first key found: Basic, then Bearer, then x-api-key
		var providedKey string
		if basicKey != "" {
			providedKey = basicKey
		} else if bearerKey != "" {
			providedKey = bearerKey
		} else {
			providedKey = xApiKey
		}

		// Validate key
		valid := false
		for _, key := range pm.config.RequiredAPIKeys {
			if providedKey == key {
				valid = true
				break
			}
		}

		if !valid {
			c.Header("WWW-Authenticate", `Basic realm="llama-swap"`)
			pm.sendErrorResponse(c, http.StatusUnauthorized, "unauthorized: invalid or missing API key")
			c.Abort()
			return
		}

		// Strip auth headers to prevent leakage to upstream
		c.Request.Header.Del("Authorization")
		c.Request.Header.Del("x-api-key")

		c.Next()
	}
}

func (pm *ProxyManager) unloadAllModelsHandler(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.String(http.StatusOK, "OK")
}

func (pm *ProxyManager) listRunningProcessesHandler(context *gin.Context) {
	context.Header("Content-Type", "application/json")
	runningProcesses := make([]gin.H, 0) // Default to an empty response.

	for _, processGroup := range pm.processGroups {
		for _, process := range processGroup.processes {
			if process.CurrentState() == StateReady {
				runningProcesses = append(runningProcesses, gin.H{
					"model":       process.ID,
					"state":       process.state,
					"cmd":         process.config.Cmd,
					"proxy":       process.config.Proxy,
					"ttl":         process.config.UnloadAfter,
					"name":        process.config.Name,
					"description": process.config.Description,
				})
			}
		}
	}

	// Put the results under the `running` key.
	response := gin.H{
		"running": runningProcesses,
	}

	context.JSON(http.StatusOK, response) // Always return 200 OK
}

func (pm *ProxyManager) findGroupByModelName(modelName string) *ProcessGroup {
	for _, group := range pm.processGroups {
		if group.HasMember(modelName) {
			return group
		}
	}
	return nil
}

func (pm *ProxyManager) resolveResponsesFallbackModel() (string, bool) {
	pm.Lock()
	defer pm.Unlock()

	readyModels := make([]string, 0)
	for _, processGroup := range pm.processGroups {
		for _, process := range processGroup.processes {
			if process.CurrentState() == StateReady {
				readyModels = append(readyModels, process.ID)
			}
		}
	}
	if len(readyModels) > 0 {
		sort.Strings(readyModels)
		return readyModels[0], true
	}

	if len(pm.config.Models) == 0 {
		return "", false
	}

	modelIDs := make([]string, 0, len(pm.config.Models))
	for modelID := range pm.config.Models {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)
	return modelIDs[0], true
}

func (pm *ProxyManager) SetVersion(buildDate string, commit string, version string) {
	pm.Lock()
	defer pm.Unlock()
	pm.buildDate = buildDate
	pm.commit = commit
	pm.version = version
}
