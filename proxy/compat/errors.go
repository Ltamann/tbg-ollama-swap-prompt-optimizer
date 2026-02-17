package compat

import (
	"net/http"
	"strings"
)

type ErrorEnvelope struct {
	Error ErrorBody `json:"error"`
}

type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}

func NewErrorEnvelope(statusCode int, message string, code string) ErrorEnvelope {
	errType := ErrorTypeFromStatus(statusCode)
	if strings.TrimSpace(code) == "" {
		code = strings.ToLower(strings.ReplaceAll(http.StatusText(statusCode), " ", "_"))
	}
	return ErrorEnvelope{
		Error: ErrorBody{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	}
}

func ErrorTypeFromStatus(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest, http.StatusUnsupportedMediaType:
		return "invalid_request_error"
	case http.StatusUnauthorized, http.StatusForbidden:
		return "authentication_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusConflict:
		return "conflict_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	default:
		if statusCode >= 500 {
			return "server_error"
		}
		return "invalid_request_error"
	}
}

