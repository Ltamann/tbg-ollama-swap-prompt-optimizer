package compat

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

type NormalizeResult struct {
	Body      []byte
	Endpoint  EndpointKind
	Canonical CanonicalRequest
}

func NormalizeInferenceRequest(r *http.Request, body []byte) (NormalizeResult, error) {
	kind := Route(r.URL.Path)
	if kind == EndpointUnknown {
		return NormalizeResult{}, fmt.Errorf("unsupported inference endpoint: %s", r.URL.Path)
	}

	result := NormalizeResult{
		Body:     body,
		Endpoint: kind,
	}

	if IsJSONBodyEndpoint(kind) {
		// Upstreams generally require JSON for OpenAI-compatible endpoints.
		r.Header.Set("Content-Type", "application/json")
	}

	if strings.TrimSpace(r.Header.Get("Accept")) == "" {
		r.Header.Set("Accept", "application/json")
	}
	r.Header.Del("transfer-encoding")
	r.Header.Del("Transfer-Encoding")
	r.Header.Set("content-length", strconv.Itoa(len(body)))
	r.ContentLength = int64(len(body))

	result.Canonical = ToCanonical(kind, body)
	return result, nil
}
