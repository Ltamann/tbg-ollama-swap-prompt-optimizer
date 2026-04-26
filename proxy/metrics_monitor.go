package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/event"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

var errNoStreamingMetrics = errors.New("stream contained no usage/timings payload")

// TokenMetrics represents parsed token statistics from llama-server logs
type TokenMetrics struct {
	ID              int       `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	Model           string    `json:"model"`
	StatusCode      int       `json:"status_code"`
	CachedTokens    int       `json:"cache_tokens"`
	InputTokens     int       `json:"input_tokens"`
	OutputTokens    int       `json:"output_tokens"`
	PromptPerSecond float64   `json:"prompt_per_second"`
	TokensPerSecond float64   `json:"tokens_per_second"`
	DurationMs      int       `json:"duration_ms"`
	HasCapture      bool      `json:"has_capture"`
}

type ReqRespCapture struct {
	ID          int               `json:"id"`
	ReqPath     string            `json:"req_path"`
	ReqHeaders  map[string]string `json:"req_headers"`
	ReqBody     []byte            `json:"req_body"`
	RespHeaders map[string]string `json:"resp_headers"`
	RespBody    []byte            `json:"resp_body"`
}

// Size returns the approximate memory usage of this capture in bytes
func (c *ReqRespCapture) Size() int {
	size := len(c.ReqPath) + len(c.ReqBody) + len(c.RespBody)
	for k, v := range c.ReqHeaders {
		size += len(k) + len(v)
	}
	for k, v := range c.RespHeaders {
		size += len(k) + len(v)
	}
	return size
}

// TokenMetricsEvent represents a token metrics event
type TokenMetricsEvent struct {
	Metrics TokenMetrics
}

func (e TokenMetricsEvent) Type() uint32 {
	return TokenMetricsEventID // defined in events.go
}

// metricsMonitor parses llama-server output for token statistics
type metricsMonitor struct {
	mu         sync.RWMutex
	metrics    []TokenMetrics
	maxMetrics int
	nextID     int
	logger     *LogMonitor

	// capture fields
	enableCaptures bool
	captures       map[int]ReqRespCapture // map for O(1) lookup by ID
	captureOrder   []int                  // track insertion order for FIFO eviction
	captureSize    int                    // current total size in bytes
	maxCaptureSize int                    // max bytes for captures

	// chatEvent callbacks — invoked when a chat capture is added
	chatMu        sync.RWMutex
	chatCallbacks []func(reqPath string, reqBody, respBody []byte, tm TokenMetrics)

	upstreamMu       sync.Mutex
	upstreamStates   map[string]*upstreamLogState
	maxUpstreamQueue int
}

// newMetricsMonitor creates a new metricsMonitor. captureBufferMB is the
// capture buffer size in megabytes; 0 disables captures.
func newMetricsMonitor(logger *LogMonitor, maxMetrics int, captureBufferMB int) *metricsMonitor {
	return &metricsMonitor{
		logger:           logger,
		metrics:          make([]TokenMetrics, 0),
		maxMetrics:       maxMetrics,
		enableCaptures:   captureBufferMB > 0,
		captures:         make(map[int]ReqRespCapture),
		captureOrder:     make([]int, 0),
		captureSize:      0,
		maxCaptureSize:   captureBufferMB * 1024 * 1024,
		upstreamStates:   make(map[string]*upstreamLogState),
		maxUpstreamQueue: 32,
	}
}

type upstreamTimingCandidate struct {
	InputTokens     int
	OutputTokens    int
	PromptPerSecond float64
	TokensPerSecond float64
	DurationMs      int
}

type upstreamRequestMetric struct {
	Path       string
	StatusCode int
	Timestamp  time.Time
	Metrics    upstreamTimingCandidate
}

type upstreamLogState struct {
	buffer    string
	pending   *upstreamTimingCandidate
	completed []upstreamRequestMetric
}

var (
	promptEvalTimingRegex = regexp.MustCompile(`prompt eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s+tokens.*?([0-9.]+)\s+tokens per second`)
	evalTimingRegex       = regexp.MustCompile(`eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s+tokens.*?([0-9.]+)\s+tokens per second`)
	totalTimingRegex      = regexp.MustCompile(`total time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s+tokens`)
	doneRequestRegex      = regexp.MustCompile(`done request:\s+[A-Z]+\s+(\S+)\s+\S+\s+([0-9]{3})`)
)

func (mp *metricsMonitor) registerModelLogMonitor(modelID string, monitor *LogMonitor) {
	if mp == nil || monitor == nil || modelID == "" {
		return
	}
	monitor.OnLogData(func(data []byte) {
		mp.ingestUpstreamLog(modelID, data)
	})
}

func (mp *metricsMonitor) ingestUpstreamLog(modelID string, data []byte) {
	mp.upstreamMu.Lock()
	defer mp.upstreamMu.Unlock()

	state := mp.ensureUpstreamStateLocked(modelID)
	state.buffer += string(data)

	for {
		idx := strings.IndexByte(state.buffer, '\n')
		if idx == -1 {
			break
		}
		line := strings.TrimRight(state.buffer[:idx], "\r")
		state.buffer = state.buffer[idx+1:]
		mp.parseUpstreamLogLineLocked(state, strings.TrimSpace(line))
	}
}

func (mp *metricsMonitor) ensureUpstreamStateLocked(modelID string) *upstreamLogState {
	state, ok := mp.upstreamStates[modelID]
	if !ok {
		state = &upstreamLogState{}
		mp.upstreamStates[modelID] = state
	}
	return state
}

func (mp *metricsMonitor) parseUpstreamLogLineLocked(state *upstreamLogState, line string) {
	if line == "" {
		return
	}

	if matches := promptEvalTimingRegex.FindStringSubmatch(line); len(matches) == 4 {
		promptMs, _ := strconv.ParseFloat(matches[1], 64)
		inputTokens, _ := strconv.Atoi(matches[2])
		promptTPS, _ := strconv.ParseFloat(matches[3], 64)
		state.pending = &upstreamTimingCandidate{
			InputTokens:     inputTokens,
			PromptPerSecond: promptTPS,
			DurationMs:      int(promptMs),
		}
		return
	}

	if state.pending != nil {
		if matches := evalTimingRegex.FindStringSubmatch(line); len(matches) == 4 {
			evalMs, _ := strconv.ParseFloat(matches[1], 64)
			outputTokens, _ := strconv.Atoi(matches[2])
			outputTPS, _ := strconv.ParseFloat(matches[3], 64)
			state.pending.OutputTokens = outputTokens
			state.pending.TokensPerSecond = outputTPS
			state.pending.DurationMs += int(evalMs)
			return
		}

		if matches := totalTimingRegex.FindStringSubmatch(line); len(matches) == 3 {
			totalMs, _ := strconv.ParseFloat(matches[1], 64)
			if int(totalMs) > state.pending.DurationMs {
				state.pending.DurationMs = int(totalMs)
			}
			return
		}
	}

	if matches := doneRequestRegex.FindStringSubmatch(line); len(matches) == 3 {
		if state.pending == nil {
			return
		}
		statusCode, _ := strconv.Atoi(matches[2])
		state.completed = append(state.completed, upstreamRequestMetric{
			Path:       matches[1],
			StatusCode: statusCode,
			Timestamp:  time.Now(),
			Metrics:    *state.pending,
		})
		if len(state.completed) > mp.maxUpstreamQueue {
			state.completed = state.completed[len(state.completed)-mp.maxUpstreamQueue:]
		}
		state.pending = nil
	}
}

func (mp *metricsMonitor) consumeUpstreamMetric(modelID, path string, requestStart time.Time, wait time.Duration) *upstreamTimingCandidate {
	deadline := time.Now().Add(wait)
	for {
		if metric := mp.tryConsumeUpstreamMetric(modelID, path, requestStart); metric != nil {
			return metric
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (mp *metricsMonitor) consumeUpstreamMetricAny(modelID string, paths []string, requestStart time.Time, wait time.Duration) *upstreamTimingCandidate {
	deadline := time.Now().Add(wait)
	for {
		for _, path := range paths {
			if path == "" {
				continue
			}
			if metric := mp.tryConsumeUpstreamMetric(modelID, path, requestStart); metric != nil {
				return metric
			}
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (mp *metricsMonitor) tryConsumeUpstreamMetric(modelID, path string, requestStart time.Time) *upstreamTimingCandidate {
	mp.upstreamMu.Lock()
	defer mp.upstreamMu.Unlock()

	state := mp.upstreamStates[modelID]
	if state == nil || len(state.completed) == 0 {
		return nil
	}

	cutoff := requestStart.Add(-1 * time.Second)
	for idx, item := range state.completed {
		if item.Path != path {
			continue
		}
		if item.Timestamp.Before(cutoff) {
			continue
		}
		metric := item.Metrics
		state.completed = append(state.completed[:idx], state.completed[idx+1:]...)
		return &metric
	}
	return nil
}

// addMetrics adds a new metric to the collection and publishes an event.
// Returns the assigned metric ID.
func (mp *metricsMonitor) addMetrics(metric TokenMetrics) int {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	metric.ID = mp.nextID
	mp.nextID++
	mp.metrics = append(mp.metrics, metric)
	if len(mp.metrics) > mp.maxMetrics {
		mp.metrics = mp.metrics[len(mp.metrics)-mp.maxMetrics:]
	}
	event.Emit(TokenMetricsEvent{Metrics: metric})
	return metric.ID
}

// addCapture adds a new capture to the buffer with size-based eviction.
// Captures are skipped if enableCaptures is false or if capture exceeds maxCaptureSize.
func (mp *metricsMonitor) addCapture(capture ReqRespCapture) {
	if !mp.enableCaptures {
		return
	}

	mp.mu.Lock()
	defer mp.mu.Unlock()

	captureSize := capture.Size()
	if captureSize > mp.maxCaptureSize {
		mp.logger.Warnf("capture size %d exceeds max %d, skipping", captureSize, mp.maxCaptureSize)
		return
	}

	// Evict oldest (FIFO) until room available
	for mp.captureSize+captureSize > mp.maxCaptureSize && len(mp.captureOrder) > 0 {
		oldestID := mp.captureOrder[0]
		mp.captureOrder = mp.captureOrder[1:]
		if evicted, exists := mp.captures[oldestID]; exists {
			mp.captureSize -= evicted.Size()
			delete(mp.captures, oldestID)
		}
	}

	mp.captures[capture.ID] = capture
	mp.captureOrder = append(mp.captureOrder, capture.ID)
	mp.captureSize += captureSize
}

// getCaptureByID returns a capture by its ID, or nil if not found.
func (mp *metricsMonitor) getCaptureByID(id int) *ReqRespCapture {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	capture, ok := mp.captures[id]
	if !ok {
		return nil
	}
	return &capture
}

// OnChatCapture registers a callback that is invoked when a chat request/response
// capture is added. Returns a function to unregister the callback.
func (mp *metricsMonitor) OnChatCapture(fn func(reqPath string, reqBody, respBody []byte, tm TokenMetrics)) func() {
	mp.chatMu.Lock()
	defer mp.chatMu.Unlock()
	mp.chatCallbacks = append(mp.chatCallbacks, fn)
	return func() {}
}

// invokeChatCaptureCallbacks calls all registered chat capture callbacks.
func (mp *metricsMonitor) invokeChatCaptureCallbacks(reqPath string, reqBody, respBody []byte, tm TokenMetrics) {
	mp.chatMu.RLock()
	fns := make([]func(reqPath string, reqBody, respBody []byte, tm TokenMetrics), len(mp.chatCallbacks))
	copy(fns, mp.chatCallbacks)
	mp.chatMu.RUnlock()

	for _, fn := range fns {
		fn(reqPath, reqBody, respBody, tm)
	}
}

// getMetrics returns a copy of the current metrics
func (mp *metricsMonitor) getMetrics() []TokenMetrics {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	result := make([]TokenMetrics, len(mp.metrics))
	copy(result, mp.metrics)
	return result
}

// getMetricsJSON returns metrics as JSON
func (mp *metricsMonitor) getMetricsJSON() ([]byte, error) {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	if mp.metrics == nil {
		return json.Marshal([]TokenMetrics{})
	}
	return json.Marshal(mp.metrics)
}

// wrapHandler wraps the proxy handler to extract token metrics
// if wrapHandler returns an error it is safe to assume that no
// data was sent to the client
func (mp *metricsMonitor) wrapHandler(
	modelID string,
	writer gin.ResponseWriter,
	request *http.Request,
	next func(modelID string, w http.ResponseWriter, r *http.Request) error,
) error {
	requestStart := time.Now()

	// Capture request body and headers if captures enabled
	var reqBody []byte
	var reqHeaders map[string]string
	if mp.enableCaptures {
		if request.Body != nil {
			var err error
			reqBody, err = io.ReadAll(request.Body)
			if err != nil {
				return fmt.Errorf("failed to read request body for capture: %w", err)
			}
			request.Body.Close()
			request.Body = io.NopCloser(bytes.NewBuffer(reqBody))
		}
		reqHeaders = make(map[string]string)
		for key, values := range request.Header {
			if len(values) > 0 {
				reqHeaders[key] = values[0]
			}
		}
		redactHeaders(reqHeaders)
	}

	recorder := newBodyCopier(writer)

	// Filter Accept-Encoding to only include encodings we can decompress for metrics
	if ae := request.Header.Get("Accept-Encoding"); ae != "" {
		request.Header.Set("Accept-Encoding", filterAcceptEncoding(ae))
	}

	if err := next(modelID, recorder, request); err != nil {
		return err
	}

	// after this point we have to assume that data was sent to the client
	// and we can only log errors but not send them to clients

	// Initialize default metrics - these will always be recorded
	tm := TokenMetrics{
		Timestamp:  time.Now(),
		Model:      modelID,
		StatusCode: recorder.Status(),
		DurationMs: int(time.Since(requestStart).Milliseconds()),
	}

	body := recorder.body.Bytes()
	if len(body) == 0 {
		mp.logger.Warn("metrics: empty body, recording minimal metrics")
		return mp.finalizeMetricCapture(request, recorder, reqBody, reqHeaders, tm, nil)
	}

	// Decompress if needed
	if encoding := recorder.Header().Get("Content-Encoding"); encoding != "" {
		var err error
		body, err = decompressBody(body, encoding)
		if err != nil {
			mp.logger.Warnf("metrics: decompression failed: %v, path=%s, recording minimal metrics", err, request.URL.Path)
			return mp.finalizeMetricCapture(request, recorder, reqBody, reqHeaders, tm, nil)
		}
	}
	if recorder.Status() != http.StatusOK {
		mp.logger.Warnf("metrics recorded failed request, HTTP status=%d, path=%s", recorder.Status(), request.URL.Path)
		return mp.finalizeMetricCapture(request, recorder, reqBody, reqHeaders, tm, body)
	}
	if strings.Contains(recorder.Header().Get("Content-Type"), "text/event-stream") {
		if parsed, err := processStreamingResponse(modelID, requestStart, body); err != nil {
			if errors.Is(err, errNoStreamingMetrics) {
				mp.logger.Debugf("streaming response had no usage/timings payload, path=%s", request.URL.Path)
			} else {
				mp.logger.Warnf("error processing streaming response: %v, path=%s, recording minimal metrics", err, request.URL.Path)
			}
		} else {
			tm = parsed
			tm.StatusCode = recorder.Status()
			mergeRequestDuration(&tm, requestStart)
		}
	} else {
		if gjson.ValidBytes(body) {
			parsed := gjson.ParseBytes(body)
			usage := parsed.Get("usage")
			timings := parsed.Get("timings")

			// extract timings for infill - response is an array, timings are in the last element
			// see #463
			if strings.HasPrefix(request.URL.Path, "/infill") {
				if arr := parsed.Array(); len(arr) > 0 {
					timings = arr[len(arr)-1].Get("timings")
				}
			}

			if usage.Exists() || timings.Exists() {
				if parsedMetrics, err := parseMetrics(modelID, requestStart, usage, timings); err != nil {
					mp.logger.Warnf("error parsing metrics: %v, path=%s, recording minimal metrics", err, request.URL.Path)
				} else {
					tm = parsedMetrics
					tm.StatusCode = recorder.Status()
					mergeRequestDuration(&tm, requestStart)
				}
			}
		} else {
			mp.logger.Warnf("metrics: invalid JSON in response body path=%s, recording minimal metrics", request.URL.Path)
		}
	}

	if isResponsesEndpoint(request.URL.Path) && (tm.PromptPerSecond < 0 || tm.TokensPerSecond < 0) {
		paths := []string{
			"/v1/chat/completions",
			request.URL.Path,
			"/v1/responses",
			"/responses",
			"/completion",
		}
		if upstreamMetric := mp.consumeUpstreamMetricAny(modelID, paths, requestStart, 250*time.Millisecond); upstreamMetric != nil {
			applyUpstreamTimingMetric(&tm, upstreamMetric)
		}
	}

	return mp.finalizeMetricCapture(request, recorder, reqBody, reqHeaders, tm, body)
}

func (mp *metricsMonitor) finalizeMetricCapture(
	request *http.Request,
	recorder *responseBodyCopier,
	reqBody []byte,
	reqHeaders map[string]string,
	tm TokenMetrics,
	body []byte,
) error {
	// Build capture if enabled and determine if it will be stored
	var capture *ReqRespCapture
	if mp.enableCaptures {
		respHeaders := make(map[string]string)
		for key, values := range recorder.Header() {
			if len(values) > 0 {
				respHeaders[key] = values[0]
			}
		}
		redactHeaders(respHeaders)
		delete(respHeaders, "Content-Encoding")
		capture = &ReqRespCapture{
			ReqPath:     request.URL.Path,
			ReqHeaders:  reqHeaders,
			ReqBody:     reqBody,
			RespHeaders: respHeaders,
			RespBody:    body,
		}
		// Only set HasCapture if the capture will actually be stored (not too large)
		if capture.Size() <= mp.maxCaptureSize {
			tm.HasCapture = true
		}
	}

	metricID := mp.addMetrics(tm)

	// Store capture if enabled
	if capture != nil {
		capture.ID = metricID
		mp.addCapture(*capture)
	}

	// Invoke chat event callbacks for streaming live UI
	mp.invokeChatCaptureCallbacks(request.URL.Path, reqBody, body, tm)

	return nil
}

func processStreamingResponse(modelID string, start time.Time, body []byte) (TokenMetrics, error) {
	if gjson.ValidBytes(body) {
		parsed := gjson.ParseBytes(body)
		usage, timings := extractUsageAndTimings(parsed)
		if usage.Exists() || timings.Exists() {
			return parseMetrics(modelID, start, usage, timings)
		}
		return TokenMetrics{}, errNoStreamingMetrics
	}

	// Iterate **backwards** through the body looking for the data payload with
	// usage data. This avoids allocating a slice of all lines via bytes.Split.

	// Start from the end of the body and scan backwards for newlines
	pos := len(body)
	for pos > 0 {
		// Find the previous newline (or start of body)
		lineStart := bytes.LastIndexByte(body[:pos], '\n')
		if lineStart == -1 {
			lineStart = 0
		} else {
			lineStart++ // Move past the newline
		}

		line := bytes.TrimSpace(body[lineStart:pos])
		pos = lineStart - 1 // Move position before the newline for next iteration

		if len(line) == 0 {
			continue
		}

		// SSE payload always follows "data:"
		prefix := []byte("data:")
		if !bytes.HasPrefix(line, prefix) {
			continue
		}
		data := bytes.TrimSpace(line[len(prefix):])

		if len(data) == 0 {
			continue
		}

		if bytes.Equal(data, []byte("[DONE]")) {
			// [DONE] line itself contains nothing of interest.
			continue
		}

		if gjson.ValidBytes(data) {
			parsed := gjson.ParseBytes(data)
			usage, timings := extractUsageAndTimings(parsed)

			if usage.Exists() || timings.Exists() {
				return parseMetrics(modelID, start, usage, timings)
			}
		}
	}

	return TokenMetrics{}, errNoStreamingMetrics
}

func extractUsageAndTimings(parsed gjson.Result) (gjson.Result, gjson.Result) {
	usage := parsed.Get("usage")
	timings := parsed.Get("timings")

	// Responses streaming payloads often nest metrics under response.{usage,timings}.
	if !usage.Exists() {
		usage = parsed.Get("response.usage")
	}
	if !timings.Exists() {
		timings = parsed.Get("response.timings")
	}
	return usage, timings
}

func parseMetrics(modelID string, start time.Time, usage, timings gjson.Result) (TokenMetrics, error) {
	// default values
	cachedTokens := -1 // unknown or missing data
	outputTokens := 0
	inputTokens := 0

	// timings data
	tokensPerSecond := -1.0
	promptPerSecond := -1.0
	durationMs := int(time.Since(start).Milliseconds())

	if usage.Exists() {
		if pt := usage.Get("prompt_tokens"); pt.Exists() {
			// v1/chat/completions
			inputTokens = int(pt.Int())
		} else if it := usage.Get("input_tokens"); it.Exists() {
			// v1/messages
			inputTokens = int(it.Int())
		}

		if ct := usage.Get("completion_tokens"); ct.Exists() {
			// v1/chat/completions
			outputTokens = int(ct.Int())
		} else if ot := usage.Get("output_tokens"); ot.Exists() {
			outputTokens = int(ot.Int())
		}

		if ct := usage.Get("cache_read_input_tokens"); ct.Exists() {
			cachedTokens = int(ct.Int())
		}
	}

	// use llama-server's timing data for tok/sec and duration as it is more accurate
	if timings.Exists() {
		inputTokens = int(timings.Get("prompt_n").Int())
		outputTokens = int(timings.Get("predicted_n").Int())
		promptPerSecond = timings.Get("prompt_per_second").Float()
		tokensPerSecond = timings.Get("predicted_per_second").Float()
		durationMs = int(timings.Get("prompt_ms").Float() + timings.Get("predicted_ms").Float())

		if cachedValue := timings.Get("cache_n"); cachedValue.Exists() {
			cachedTokens = int(cachedValue.Int())
		}
	}

	return TokenMetrics{
		Timestamp:       time.Now(),
		Model:           modelID,
		CachedTokens:    cachedTokens,
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		PromptPerSecond: promptPerSecond,
		TokensPerSecond: tokensPerSecond,
		DurationMs:      durationMs,
	}, nil
}

func mergeRequestDuration(tm *TokenMetrics, requestStart time.Time) {
	if tm == nil {
		return
	}

	requestDurationMs := int(time.Since(requestStart).Milliseconds())
	if requestDurationMs > tm.DurationMs {
		tm.DurationMs = requestDurationMs
	}
}

func applyUpstreamTimingMetric(tm *TokenMetrics, upstreamMetric *upstreamTimingCandidate) {
	if tm == nil || upstreamMetric == nil {
		return
	}
	if upstreamMetric.InputTokens > 0 {
		tm.InputTokens = upstreamMetric.InputTokens
	}
	if upstreamMetric.OutputTokens >= 0 {
		tm.OutputTokens = upstreamMetric.OutputTokens
	}
	if upstreamMetric.PromptPerSecond > 0 {
		tm.PromptPerSecond = upstreamMetric.PromptPerSecond
	}
	if upstreamMetric.TokensPerSecond > 0 {
		tm.TokensPerSecond = upstreamMetric.TokensPerSecond
	}
	if upstreamMetric.DurationMs > tm.DurationMs {
		tm.DurationMs = upstreamMetric.DurationMs
	}
}

// decompressBody decompresses the body based on Content-Encoding header
func decompressBody(body []byte, encoding string) ([]byte, error) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		return io.ReadAll(reader)
	case "deflate":
		reader := flate.NewReader(bytes.NewReader(body))
		defer reader.Close()
		return io.ReadAll(reader)
	default:
		return body, nil // Return as-is for unknown/no encoding
	}
}

// responseBodyCopier records the response body and writes to the original response writer
// while also capturing it in a buffer for later processing
type responseBodyCopier struct {
	gin.ResponseWriter
	body  *bytes.Buffer
	tee   io.Writer
	start time.Time
}

func newBodyCopier(w gin.ResponseWriter) *responseBodyCopier {
	bodyBuffer := &bytes.Buffer{}
	return &responseBodyCopier{
		ResponseWriter: w,
		body:           bodyBuffer,
		tee:            io.MultiWriter(w, bodyBuffer),
	}
}

func (w *responseBodyCopier) Write(b []byte) (int, error) {
	if w.start.IsZero() {
		w.start = time.Now()
	}

	// Single write operation that writes to both the response and buffer
	return w.tee.Write(b)
}

func (w *responseBodyCopier) WriteHeader(statusCode int) {
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *responseBodyCopier) Header() http.Header {
	return w.ResponseWriter.Header()
}

func (w *responseBodyCopier) StartTime() time.Time {
	return w.start
}

// sensitiveHeaders lists headers that should be redacted in captures
var sensitiveHeaders = map[string]bool{
	"authorization":       true,
	"proxy-authorization": true,
	"cookie":              true,
	"set-cookie":          true,
	"x-api-key":           true,
}

// redactHeaders replaces sensitive header values in-place with "[REDACTED]"
func redactHeaders(headers map[string]string) {
	for key := range headers {
		if sensitiveHeaders[strings.ToLower(key)] {
			headers[key] = "[REDACTED]"
		}
	}
}

// filterAcceptEncoding filters the Accept-Encoding header to only include
// encodings we can decompress (gzip, deflate). This respects the client's
// preferences while ensuring we can parse response bodies for metrics.
func filterAcceptEncoding(acceptEncoding string) string {
	if acceptEncoding == "" {
		return ""
	}

	supported := map[string]bool{"gzip": true, "deflate": true}
	var filtered []string

	for _, part := range strings.Split(acceptEncoding, ",") {
		// Parse encoding and optional quality value (e.g., "gzip;q=1.0")
		encoding := strings.TrimSpace(strings.Split(part, ";")[0])
		if supported[strings.ToLower(encoding)] {
			filtered = append(filtered, strings.TrimSpace(part))
		}
	}

	return strings.Join(filtered, ", ")
}
