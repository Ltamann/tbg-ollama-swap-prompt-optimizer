package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
)

type Model struct {
	Id                         string  `json:"id"`
	Name                       string  `json:"name"`
	Description                string  `json:"description"`
	State                      string  `json:"state"`
	Unlisted                   bool    `json:"unlisted"`
	PeerID                     string  `json:"peerID"`
	Provider                   string  `json:"provider,omitempty"`
	External                   bool    `json:"external,omitempty"`
	CtxReference               int     `json:"ctxReference,omitempty"`
	CtxConfigured              int     `json:"ctxConfigured,omitempty"`
	CtxSource                  string  `json:"ctxSource,omitempty"`
	FitEnabled                 bool    `json:"fitEnabled,omitempty"`
	FitCtxMode                 string  `json:"fitCtxMode,omitempty"`
	TempConfigured             float64 `json:"tempConfigured"`
	TopPConfigured             float64 `json:"topPConfigured"`
	TopKConfigured             int     `json:"topKConfigured"`
	MinPConfigured             float64 `json:"minPConfigured"`
	PresencePenaltyConfigured  float64 `json:"presencePenaltyConfigured"`
	FrequencyPenaltyConfigured float64 `json:"frequencyPenaltyConfigured"`
}

func addApiHandlers(pm *ProxyManager) {
	// Add API endpoints for React to consume
	// Protected with API key authentication
	apiGroup := pm.ginEngine.Group("/api", pm.apiKeyAuth())
	{
		apiGroup.POST("/models/unload", pm.apiUnloadAllModels)
		apiGroup.POST("/models/kill-llama-cpp", pm.apiKillAllLlamaCpp)
		apiGroup.POST("/models/unload/*model", pm.apiUnloadSingleModelHandler)
		apiGroup.GET("/tools", pm.apiListTools)
		apiGroup.POST("/tools", pm.apiCreateTool)
		apiGroup.PUT("/tools/:id", pm.apiUpdateTool)
		apiGroup.DELETE("/tools/:id", pm.apiDeleteTool)
		apiGroup.GET("/tools/settings", pm.apiGetToolSettings)
		apiGroup.PUT("/tools/settings", pm.apiSetToolSettings)
		apiGroup.GET("/events", pm.apiSendEvents)
		apiGroup.GET("/metrics", pm.apiGetMetrics)
		apiGroup.GET("/activity/prompts", pm.apiGetActivityPrompts)
		apiGroup.GET("/version", pm.apiGetVersion)
		apiGroup.GET("/captures/:id", pm.apiGetCapture)
		apiGroup.GET("/config/path", pm.apiGetConfigPath)
		apiGroup.POST("/config/reload", pm.apiReloadConfig)
		apiGroup.POST("/restart", pm.apiRestartTBG)
	}

	// Add ctx-size endpoint handlers
	ctxSizeGroup := pm.ginEngine.Group("/api/model", pm.apiKeyAuth())
	ctxSizeGroup.POST("/:model/ctxsize", pm.apiSetCtxSize)
	ctxSizeGroup.GET("/:model/ctxsize", pm.apiGetCtxSize)
	ctxSizeGroup.POST("/:model/fit", pm.apiSetFitMode)
	ctxSizeGroup.GET("/:model/fit", pm.apiGetFitMode)
	ctxSizeGroup.POST("/:model/prompt-optimization", pm.apiSetPromptOptimization)
	ctxSizeGroup.GET("/:model/prompt-optimization", pm.apiGetPromptOptimization)
	ctxSizeGroup.GET("/:model/prompt-optimization/latest", pm.apiGetLatestPromptOptimization)
}

func (pm *ProxyManager) apiUnloadAllModels(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.JSON(http.StatusOK, gin.H{"msg": "ok"})
}

func (pm *ProxyManager) apiKillAllLlamaCpp(c *gin.Context) {
	// First stop all processes managed by llama-swap.
	pm.StopProcesses(StopImmediately)

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		// Best effort for typical llama.cpp binary names on Windows.
		cmd = exec.Command("taskkill", "/F", "/IM", "llama-server.exe", "/T")
	default:
		// Linux/macOS (including WSL): kill all llama-server processes.
		cmd = exec.Command("pkill", "-9", "-f", "llama-server")
	}

	out, err := cmd.CombinedOutput()
	outStr := strings.TrimSpace(string(out))

	// If no process matched, treat as success.
	if err != nil {
		lower := strings.ToLower(outStr)
		if strings.Contains(lower, "no process found") ||
			strings.Contains(lower, "not found running instance") ||
			strings.Contains(lower, "no running instance") {
			c.JSON(http.StatusOK, gin.H{"msg": "ok", "detail": "no matching llama.cpp processes"})
			return
		}
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("failed to kill llama.cpp processes: %v (%s)", err, outStr))
		return
	}

	if outStr == "" {
		outStr = "llama.cpp processes killed"
	}
	c.JSON(http.StatusOK, gin.H{"msg": "ok", "detail": outStr})
}

func (pm *ProxyManager) getModelStatus() []Model {
	// Extract keys and sort them
	models := []Model{}

	modelIDs := make([]string, 0, len(pm.config.Models))
	for modelID := range pm.config.Models {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)

	// Iterate over sorted keys
	for _, modelID := range modelIDs {
		// Get process state
		processGroup := pm.findGroupByModelName(modelID)
		state := "unknown"
		if processGroup != nil {
			process := processGroup.processes[modelID]
			if process != nil {
				var stateStr string
				switch process.CurrentState() {
				case StateReady:
					stateStr = "ready"
				case StateStarting:
					stateStr = "starting"
				case StateStopping:
					stateStr = "stopping"
				case StateShutdown:
					stateStr = "shutdown"
				case StateStopped:
					stateStr = "stopped"
				default:
					stateStr = "unknown"
				}
				state = stateStr
			}
		}
		modelCfg := pm.config.Models[modelID]
		args, _ := (&modelCfg).SanitizedCommand()
		configCtx, ctxSource, fitFromConfig, fitCtxMode := parseCtxAndFitFromArgs(args)
		samplingConfigured := parseSamplingFromArgs(args)
		pm.Lock()
		runtimeFit, hasFitOverride := pm.fitModes[modelID]
		runtimeFitCtxMode, hasFitCtxModeOverride := pm.fitCtxModes[modelID]
		pm.Unlock()
		fitEnabled := fitFromConfig
		if hasFitOverride {
			fitEnabled = runtimeFit
		}
		if hasFitCtxModeOverride {
			fitCtxMode = runtimeFitCtxMode
		}
		if fitCtxMode == "" {
			fitCtxMode = "max"
		}

		model := Model{
			Id:            modelID,
			Name:          pm.config.Models[modelID].Name,
			Description:   pm.config.Models[modelID].Description,
			State:         state,
			Unlisted:      pm.config.Models[modelID].Unlisted,
			Provider:      "llama",
			CtxConfigured: configCtx,
			CtxSource:     ctxSource,
			FitEnabled:    fitEnabled,
			FitCtxMode:    fitCtxMode,
		}
		model.TempConfigured = samplingConfigured.temp
		model.TopPConfigured = samplingConfigured.topP
		model.TopKConfigured = samplingConfigured.topK
		model.MinPConfigured = samplingConfigured.minP
		model.PresencePenaltyConfigured = samplingConfigured.presencePenalty
		model.FrequencyPenaltyConfigured = samplingConfigured.frequencyPenalty
		models = append(models, model)
	}

	// Iterate over the peer models
	if pm.peerProxy != nil {
		for peerID, peer := range pm.peerProxy.ListPeers() {
			for _, modelID := range peer.Models {
				models = append(models, Model{
					Id:     modelID,
					PeerID: peerID,
				})
			}
		}
	}

	for _, ollamaModel := range pm.GetOllamaModels() {
		models = append(models, Model{
			Id:           ollamaModel.ID,
			Name:         ollamaModel.Name,
			State:        "ready",
			Provider:     "ollama",
			External:     true,
			CtxReference: ollamaModel.CtxReference,
		})
	}

	return models
}

type samplingConfigured struct {
	temp             float64
	topP             float64
	topK             int
	minP             float64
	presencePenalty  float64
	frequencyPenalty float64
}

func parseSamplingFromArgs(args []string) samplingConfigured {
	cfg := samplingConfigured{
		// llama.cpp defaults from -h
		temp:             0.8,
		topP:             0.95,
		topK:             40,
		minP:             0.05,
		presencePenalty:  0.0,
		frequencyPenalty: 0.0,
	}

	for i := 0; i < len(args); i++ {
		arg := strings.TrimSpace(args[i])
		if arg == "" {
			continue
		}
		if arg == "--temp" {
			if i+1 < len(args) {
				if f, err := strconv.ParseFloat(strings.TrimSpace(args[i+1]), 64); err == nil {
					cfg.temp = f
				}
			}
			continue
		}
		if strings.HasPrefix(arg, "--temp=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--temp=")), 64); err == nil {
				cfg.temp = f
			}
		}
		if arg == "--top-p" && i+1 < len(args) {
			if f, err := strconv.ParseFloat(strings.TrimSpace(args[i+1]), 64); err == nil {
				cfg.topP = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--top-p=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--top-p=")), 64); err == nil {
				cfg.topP = f
			}
			continue
		}
		if arg == "--top-k" && i+1 < len(args) {
			if v, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil {
				cfg.topK = v
			}
			continue
		}
		if strings.HasPrefix(arg, "--top-k=") {
			if v, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(arg, "--top-k="))); err == nil {
				cfg.topK = v
			}
			continue
		}
		if arg == "--min-p" && i+1 < len(args) {
			if f, err := strconv.ParseFloat(strings.TrimSpace(args[i+1]), 64); err == nil {
				cfg.minP = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--min-p=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--min-p=")), 64); err == nil {
				cfg.minP = f
			}
			continue
		}
		if (arg == "--presence-penalty" || arg == "--presence_penalty") && i+1 < len(args) {
			if f, err := strconv.ParseFloat(strings.TrimSpace(args[i+1]), 64); err == nil {
				cfg.presencePenalty = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--presence-penalty=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--presence-penalty=")), 64); err == nil {
				cfg.presencePenalty = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--presence_penalty=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--presence_penalty=")), 64); err == nil {
				cfg.presencePenalty = f
			}
			continue
		}
		if (arg == "--frequency-penalty" || arg == "--frequency_penalty") && i+1 < len(args) {
			if f, err := strconv.ParseFloat(strings.TrimSpace(args[i+1]), 64); err == nil {
				cfg.frequencyPenalty = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--frequency-penalty=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--frequency-penalty=")), 64); err == nil {
				cfg.frequencyPenalty = f
			}
			continue
		}
		if strings.HasPrefix(arg, "--frequency_penalty=") {
			if f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(arg, "--frequency_penalty=")), 64); err == nil {
				cfg.frequencyPenalty = f
			}
			continue
		}
	}
	return cfg
}

type messageType string

const (
	msgTypeModelStatus messageType = "modelStatus"
	msgTypeLogData     messageType = "logData"
	msgTypeMetrics     messageType = "metrics"
)

type messageEnvelope struct {
	Type messageType `json:"type"`
	Data string      `json:"data"`
}

// sends a stream of different message types that happen on the server
func (pm *ProxyManager) apiSendEvents(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Content-Type-Options", "nosniff")
	// prevent nginx from buffering SSE
	c.Header("X-Accel-Buffering", "no")

	sendBuffer := make(chan messageEnvelope, 25)
	ctx, cancel := context.WithCancel(c.Request.Context())
	sendModels := func() {
		data, err := json.Marshal(pm.getModelStatus())
		if err == nil {
			msg := messageEnvelope{Type: msgTypeModelStatus, Data: string(data)}
			select {
			case sendBuffer <- msg:
			case <-ctx.Done():
				return
			default:
			}

		}
	}

	sendLogData := func(source string, data []byte) {
		data, err := json.Marshal(gin.H{
			"source": source,
			"data":   string(data),
		})
		if err == nil {
			select {
			case sendBuffer <- messageEnvelope{Type: msgTypeLogData, Data: string(data)}:
			case <-ctx.Done():
				return
			default:
			}
		}
	}

	sendMetrics := func(metrics []TokenMetrics) {
		jsonData, err := json.Marshal(metrics)
		if err == nil {
			select {
			case sendBuffer <- messageEnvelope{Type: msgTypeMetrics, Data: string(jsonData)}:
			case <-ctx.Done():
				return
			default:
			}
		}
	}

	/**
	 * Send updated models list
	 */
	defer event.On(func(e ProcessStateChangeEvent) {
		sendModels()
	})()
	defer event.On(func(e ConfigFileChangedEvent) {
		sendModels()
	})()

	/**
	 * Send Log data
	 */
	defer pm.proxyLogger.OnLogData(func(data []byte) {
		sendLogData("proxy", data)
	})()
	defer pm.upstreamLogger.OnLogData(func(data []byte) {
		sendLogData("upstream", data)
	})()

	/**
	 * Send Metrics data
	 */
	defer event.On(func(e TokenMetricsEvent) {
		sendMetrics([]TokenMetrics{e.Metrics})
	})()

	// send initial batch of data
	sendLogData("proxy", pm.proxyLogger.GetHistory())
	sendLogData("upstream", pm.upstreamLogger.GetHistory())
	sendModels()
	sendMetrics(pm.metricsMonitor.getMetrics())

	for {
		select {
		case <-c.Request.Context().Done():
			cancel()
			return
		case <-pm.shutdownCtx.Done():
			cancel()
			return
		case msg := <-sendBuffer:
			c.SSEvent("message", msg)
			c.Writer.Flush()
		}
	}
}

func (pm *ProxyManager) apiGetMetrics(c *gin.Context) {
	jsonData, err := pm.metricsMonitor.getMetricsJSON()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get metrics"})
		return
	}
	c.Data(http.StatusOK, "application/json", jsonData)
}

func (pm *ProxyManager) apiGetActivityPrompts(c *gin.Context) {
	c.JSON(http.StatusOK, pm.getActivityPromptPreviews())
}

func (pm *ProxyManager) apiUnloadSingleModelHandler(c *gin.Context) {
	requestedModel := strings.TrimPrefix(c.Param("model"), "/")
	realModelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		pm.sendErrorResponse(c, http.StatusNotFound, "Model not found")
		return
	}

	processGroup := pm.findGroupByModelName(realModelName)
	if processGroup == nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("process group not found for model %s", requestedModel))
		return
	}

	if err := processGroup.StopProcess(realModelName, StopImmediately); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error stopping process: %s", err.Error()))
		return
	} else {
		c.String(http.StatusOK, "OK")
	}
}

func (pm *ProxyManager) apiGetVersion(c *gin.Context) {
	c.JSON(http.StatusOK, map[string]string{
		"version":    pm.version,
		"commit":     pm.commit,
		"build_date": pm.buildDate,
	})
}

func (pm *ProxyManager) apiGetConfigPath(c *gin.Context) {
	pm.Lock()
	defer pm.Unlock()
	c.JSON(http.StatusOK, gin.H{
		"configPath": pm.configPath,
	})
}

func (pm *ProxyManager) reloadConfigFromDisk(stopModels bool) error {
	pm.Lock()
	cfgPath := strings.TrimSpace(pm.configPath)
	pm.Unlock()
	if cfgPath == "" {
		cfgPath = "config.yaml"
	}

	newCfg, err := config.LoadConfig(cfgPath)
	if err != nil {
		return err
	}

	event.Emit(ConfigFileChangedEvent{ReloadingState: ReloadingStateStart})
	defer event.Emit(ConfigFileChangedEvent{ReloadingState: ReloadingStateEnd})

	if stopModels {
		pm.StopProcesses(StopImmediately)
	}

	pm.Lock()
	defer pm.Unlock()

	pm.config = newCfg

	// Keep processGroups in sync with config groups.
	for groupID := range pm.config.Groups {
		if _, ok := pm.processGroups[groupID]; !ok {
			pm.processGroups[groupID] = NewProcessGroup(groupID, pm.config, pm.proxyLogger, pm.upstreamLogger)
		}
	}
	for groupID := range pm.processGroups {
		if _, ok := pm.config.Groups[groupID]; !ok {
			delete(pm.processGroups, groupID)
		}
	}

	// Soft restart clears runtime overrides when requested.
	if stopModels {
		pm.ctxSizes = make(map[string]int)
		pm.fitModes = make(map[string]bool)
		pm.fitCtxModes = make(map[string]string)
		pm.promptPolicies = make(map[string]PromptOptimizationPolicy)
		pm.latestPromptOptimizations = make(map[string]PromptOptimizationSnapshot)
		pm.activityPromptPreviews = pm.activityPromptPreviews[:0]
		pm.activityCurrentUserSignature = ""
		pm.activityCurrentTurn = 0
	}

	return nil
}

func (pm *ProxyManager) apiReloadConfig(c *gin.Context) {
	if err := pm.reloadConfigFromDisk(false); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to reload config: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, gin.H{"msg": "ok", "detail": "config reloaded"})
}

func (pm *ProxyManager) apiRestartTBG(c *gin.Context) {
	if err := pm.reloadConfigFromDisk(true); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to restart TBG: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, gin.H{"msg": "ok", "detail": "TBG soft restart complete"})
}

func (pm *ProxyManager) apiGetCapture(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid capture ID"})
		return
	}

	capture := pm.metricsMonitor.getCaptureByID(id)
	if capture == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "capture not found"})
		return
	}

	c.JSON(http.StatusOK, capture)
}

func (pm *ProxyManager) apiListTools(c *gin.Context) {
	pm.Lock()
	defer pm.Unlock()
	tools := append([]RuntimeTool(nil), pm.tools...)
	for i := range tools {
		tools[i] = normalizeRuntimeTool(tools[i])
	}
	c.JSON(http.StatusOK, tools)
}

func (pm *ProxyManager) apiGetToolSettings(c *gin.Context) {
	settings := pm.getToolRuntimeSettings()
	c.JSON(http.StatusOK, settings)
}

func (pm *ProxyManager) apiSetToolSettings(c *gin.Context) {
	var req ToolRuntimeSettings
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	req = normalizeToolRuntimeSettings(req)
	pm.Lock()
	pm.toolSettings = req
	pm.Unlock()
	if err := pm.saveToolsToDisk(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to save tools: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, req)
}

func (pm *ProxyManager) apiCreateTool(c *gin.Context) {
	var req RuntimeTool
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	req.ID = strings.TrimSpace(req.ID)
	if req.ID == "" {
		req.ID = fmt.Sprintf("tool_%d", time.Now().UnixNano())
	}
	req = normalizeRuntimeTool(req)
	if req.Name == "" || req.Endpoint == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "name and endpoint are required")
		return
	}
	if req.Type != RuntimeToolHTTP && req.Type != RuntimeToolMCP {
		pm.sendErrorResponse(c, http.StatusBadRequest, "type must be http or mcp")
		return
	}
	settings := pm.getToolRuntimeSettings()
	if err := validateToolEndpoint(req.Endpoint, settings); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, err.Error())
		return
	}

	pm.Lock()
	for _, t := range pm.tools {
		if t.ID == req.ID {
			pm.Unlock()
			pm.sendErrorResponse(c, http.StatusBadRequest, "tool id already exists")
			return
		}
	}
	pm.tools = append(pm.tools, req)
	pm.Unlock()

	if err := pm.saveToolsToDisk(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to save tools: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, req)
}

func (pm *ProxyManager) apiUpdateTool(c *gin.Context) {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "id required")
		return
	}
	var req RuntimeTool
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	req.ID = id
	req = normalizeRuntimeTool(req)
	if req.Name == "" || req.Endpoint == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "name and endpoint are required")
		return
	}
	if req.Type != RuntimeToolHTTP && req.Type != RuntimeToolMCP {
		pm.sendErrorResponse(c, http.StatusBadRequest, "type must be http or mcp")
		return
	}
	settings := pm.getToolRuntimeSettings()
	if err := validateToolEndpoint(req.Endpoint, settings); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, err.Error())
		return
	}

	pm.Lock()
	updated := false
	for i, t := range pm.tools {
		if t.ID == id {
			pm.tools[i] = req
			updated = true
			break
		}
	}
	pm.Unlock()
	if !updated {
		pm.sendErrorResponse(c, http.StatusNotFound, "tool not found")
		return
	}
	if err := pm.saveToolsToDisk(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to save tools: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, req)
}

func (pm *ProxyManager) apiDeleteTool(c *gin.Context) {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "id required")
		return
	}
	pm.Lock()
	next := make([]RuntimeTool, 0, len(pm.tools))
	found := false
	for _, t := range pm.tools {
		if t.ID == id {
			found = true
			continue
		}
		next = append(next, t)
	}
	pm.tools = next
	pm.Unlock()
	if !found {
		pm.sendErrorResponse(c, http.StatusNotFound, "tool not found")
		return
	}
	if err := pm.saveToolsToDisk(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to save tools: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, gin.H{"msg": "ok"})
}

type SetCtxSizeRequest struct {
	CtxSize int `json:"ctxSize"`
}

type SetFitModeRequest struct {
	Fit  bool   `json:"fit"`
	Mode string `json:"mode,omitempty"`
}

func (pm *ProxyManager) apiSetCtxSize(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if _, exists := pm.GetOllamaModelByID(requestedModel); exists {
			pm.sendErrorResponse(c, http.StatusBadRequest, "ctxSize for ollama models is read-only")
			return
		}
		pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
		return
	}

	var req SetCtxSizeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}

	if req.CtxSize <= 0 {
		pm.sendErrorResponse(c, http.StatusBadRequest, "ctxSize must be a positive integer")
		return
	}

	pm.Lock()
	defer pm.Unlock()

	pm.ctxSizes[modelName] = req.CtxSize
	c.JSON(http.StatusOK, gin.H{"msg": "ctxSize set successfully", "model": modelName, "ctxSize": req.CtxSize})
}

func (pm *ProxyManager) apiGetCtxSize(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
			c.JSON(http.StatusOK, gin.H{"model": ollamaModel.ID, "ctxSize": ollamaModel.CtxReference})
			return
		}
		pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
		return
	}

	pm.Lock()
	defer pm.Unlock()

	ctxSize, found := pm.ctxSizes[modelName]
	if !found {
		c.JSON(http.StatusOK, gin.H{"model": modelName, "ctxSize": 0})
		return
	}

	c.JSON(http.StatusOK, gin.H{"model": modelName, "ctxSize": ctxSize})
}

func (pm *ProxyManager) apiSetFitMode(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if _, exists := pm.GetOllamaModelByID(requestedModel); exists {
			pm.sendErrorResponse(c, http.StatusBadRequest, "fit mode for ollama models is read-only")
			return
		}
		pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
		return
	}

	var req SetFitModeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	mode := strings.ToLower(strings.TrimSpace(req.Mode))
	if mode == "" {
		mode = "max"
	}
	if mode != "max" && mode != "min" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "fit mode must be one of: max, min")
		return
	}

	pm.Lock()
	pm.fitModes[modelName] = req.Fit
	pm.fitCtxModes[modelName] = mode
	pm.Unlock()

	c.JSON(http.StatusOK, gin.H{"msg": "fit mode set successfully", "model": modelName, "fit": req.Fit, "mode": mode})
}

func (pm *ProxyManager) apiGetFitMode(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if _, exists := pm.GetOllamaModelByID(requestedModel); exists {
			c.JSON(http.StatusOK, gin.H{"model": requestedModel, "fit": false})
			return
		}
		pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
		return
	}

	pm.Lock()
	fit, hasOverride := pm.fitModes[modelName]
	mode, hasModeOverride := pm.fitCtxModes[modelName]
	pm.Unlock()
	if !hasOverride {
		modelCfg := pm.config.Models[modelName]
		args, _ := (&modelCfg).SanitizedCommand()
		_, _, fit, mode = parseCtxAndFitFromArgs(args)
	}
	if !hasModeOverride && mode == "" {
		mode = "max"
	}

	c.JSON(http.StatusOK, gin.H{"model": modelName, "fit": fit, "mode": mode})
}

type SetPromptOptimizationRequest struct {
	Policy PromptOptimizationPolicy `json:"policy"`
}

func (pm *ProxyManager) apiSetPromptOptimization(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
			modelName = ollamaModel.ID
			found = true
		}
		if !found {
			pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
			return
		}
	}

	var req SetPromptOptimizationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}

	switch req.Policy {
	case PromptOptimizationOff, PromptOptimizationLimitOnly, PromptOptimizationAlways, PromptOptimizationLLMAssist:
	default:
		pm.sendErrorResponse(c, http.StatusBadRequest, "policy must be one of: off, limit_only, always, llm_assisted")
		return
	}

	pm.Lock()
	pm.promptPolicies[modelName] = req.Policy
	pm.Unlock()

	c.JSON(http.StatusOK, gin.H{
		"msg":    "prompt optimization policy set successfully",
		"model":  modelName,
		"policy": req.Policy,
	})
}

func (pm *ProxyManager) apiGetPromptOptimization(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
			modelName = ollamaModel.ID
			found = true
		}
		if !found {
			pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
			return
		}
	}

	pm.Lock()
	policy, hasRuntimePolicy := pm.promptPolicies[modelName]
	pm.Unlock()

	if !hasRuntimePolicy {
		policy = PromptOptimizationLimitOnly
	}

	c.JSON(http.StatusOK, gin.H{
		"model":  modelName,
		"policy": policy,
	})
}

func (pm *ProxyManager) apiGetLatestPromptOptimization(c *gin.Context) {
	requestedModel := strings.TrimSpace(c.Param("model"))
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	modelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		if ollamaModel, exists := pm.GetOllamaModelByID(requestedModel); exists {
			modelName = ollamaModel.ID
			found = true
		}
		if !found {
			pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
			return
		}
	}

	pm.Lock()
	snapshot, exists := pm.latestPromptOptimizations[modelName]
	pm.Unlock()
	if !exists {
		pm.sendErrorResponse(c, http.StatusNotFound, "no optimization snapshot found")
		return
	}

	c.JSON(http.StatusOK, snapshot)
}

