package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/mostlygeek/llama-swap/event"
)

type Model struct {
	Id           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	State        string `json:"state"`
	Unlisted     bool   `json:"unlisted"`
	PeerID       string `json:"peerID"`
	Provider     string `json:"provider,omitempty"`
	External     bool   `json:"external,omitempty"`
	CtxReference int    `json:"ctxReference,omitempty"`
}

func addApiHandlers(pm *ProxyManager) {
	// Add API endpoints for React to consume
	// Protected with API key authentication
	apiGroup := pm.ginEngine.Group("/api", pm.apiKeyAuth())
	{
		apiGroup.POST("/models/unload", pm.apiUnloadAllModels)
		apiGroup.POST("/models/unload/*model", pm.apiUnloadSingleModelHandler)
		apiGroup.GET("/events", pm.apiSendEvents)
		apiGroup.GET("/metrics", pm.apiGetMetrics)
		apiGroup.GET("/version", pm.apiGetVersion)
		apiGroup.GET("/captures/:id", pm.apiGetCapture)
		apiGroup.GET("/config/path", pm.apiGetConfigPath)
	}

	// Add ctx-size endpoint handlers
	ctxSizeGroup := pm.ginEngine.Group("/api/model", pm.apiKeyAuth())
	ctxSizeGroup.POST("/:model/ctxsize", pm.apiSetCtxSize)
	ctxSizeGroup.GET("/:model/ctxsize", pm.apiGetCtxSize)
	ctxSizeGroup.POST("/:model/prompt-optimization", pm.apiSetPromptOptimization)
	ctxSizeGroup.GET("/:model/prompt-optimization", pm.apiGetPromptOptimization)
	ctxSizeGroup.GET("/:model/prompt-optimization/latest", pm.apiGetLatestPromptOptimization)
}

func (pm *ProxyManager) apiUnloadAllModels(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.JSON(http.StatusOK, gin.H{"msg": "ok"})
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
		models = append(models, Model{
			Id:          modelID,
			Name:        pm.config.Models[modelID].Name,
			Description: pm.config.Models[modelID].Description,
			State:       state,
			Unlisted:    pm.config.Models[modelID].Unlisted,
			Provider:    "llama",
		})
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

type SetCtxSizeRequest struct {
	CtxSize int `json:"ctxSize"`
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
