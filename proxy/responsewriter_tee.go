package proxy

import (
	"io"
	"net/http"
	"sync"
)

type teeResponseWriter struct {
	w       http.ResponseWriter
	onWrite func([]byte)
}

func (t *teeResponseWriter) Header() http.Header { return t.w.Header() }

func (t *teeResponseWriter) WriteHeader(statusCode int) { t.w.WriteHeader(statusCode) }

func (t *teeResponseWriter) Write(b []byte) (int, error) {
	if t.onWrite != nil && len(b) > 0 {
		t.onWrite(b)
	}
	return t.w.Write(b)
}

func (t *teeResponseWriter) Flush() {
	if f, ok := t.w.(http.Flusher); ok {
		f.Flush()
	}
}

type statusCaptureResponseWriter struct {
	w           http.ResponseWriter
	statusCode  int
	wroteHeader bool
}

func newStatusCaptureResponseWriter(w http.ResponseWriter) *statusCaptureResponseWriter {
	return &statusCaptureResponseWriter{
		w:          w,
		statusCode: http.StatusOK,
	}
}

func (w *statusCaptureResponseWriter) Header() http.Header { return w.w.Header() }

func (w *statusCaptureResponseWriter) WriteHeader(statusCode int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	if statusCode > 0 {
		w.statusCode = statusCode
	}
	w.w.WriteHeader(statusCode)
}

func (w *statusCaptureResponseWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	return w.w.Write(b)
}

func (w *statusCaptureResponseWriter) Flush() {
	if f, ok := w.w.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *statusCaptureResponseWriter) StatusCode() int {
	if w == nil {
		return 0
	}
	return w.statusCode
}

type headerSnapshot struct {
	code   int
	header http.Header
}

type pipeResponseWriter struct {
	pw *io.PipeWriter

	mu          sync.Mutex
	header      http.Header
	wroteHeader bool
	code        int

	headerCh chan headerSnapshot
}

func newPipeResponseWriter(pw *io.PipeWriter) *pipeResponseWriter {
	return &pipeResponseWriter{
		pw:       pw,
		header:   make(http.Header),
		code:     http.StatusOK,
		headerCh: make(chan headerSnapshot, 1),
	}
}

func (w *pipeResponseWriter) Header() http.Header { return w.header }

func (w *pipeResponseWriter) WriteHeader(statusCode int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	if statusCode > 0 {
		w.code = statusCode
	}
	h := w.header.Clone()
	select {
	case w.headerCh <- headerSnapshot{code: w.code, header: h}:
	default:
	}
}

func (w *pipeResponseWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	needsHeader := !w.wroteHeader
	w.mu.Unlock()
	if needsHeader {
		w.WriteHeader(http.StatusOK)
	}
	return w.pw.Write(b)
}

func (w *pipeResponseWriter) Flush() {}
