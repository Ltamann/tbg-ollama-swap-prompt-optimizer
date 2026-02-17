package compat

import (
	"strings"

	"github.com/tidwall/gjson"
)

// CanonicalRequest is a lightweight, endpoint-agnostic summary used for
// compatibility checks and logging.
type CanonicalRequest struct {
	Endpoint EndpointKind `json:"endpoint"`
	Model    string       `json:"model,omitempty"`
	Input    string       `json:"input,omitempty"`
	Stream   bool         `json:"stream,omitempty"`
	HasTools bool         `json:"has_tools,omitempty"`
}

func ToCanonical(kind EndpointKind, body []byte) CanonicalRequest {
	c := CanonicalRequest{Endpoint: kind}
	if len(body) == 0 {
		return c
	}

	c.Model = strings.TrimSpace(gjson.GetBytes(body, "model").String())
	c.Stream = gjson.GetBytes(body, "stream").Bool()
	c.HasTools = gjson.GetBytes(body, "tools").IsArray()

	switch kind {
	case EndpointResponses:
		c.Input = strings.TrimSpace(gjson.GetBytes(body, "input").String())
		if c.Input == "" {
			c.Input = strings.TrimSpace(gjson.GetBytes(body, "messages.0.content").String())
		}
	case EndpointChatCompletions, EndpointMessages:
		c.Input = strings.TrimSpace(gjson.GetBytes(body, "messages.-1.content").String())
		if c.Input == "" {
			c.Input = strings.TrimSpace(gjson.GetBytes(body, "messages.0.content").String())
		}
	case EndpointCompletions:
		c.Input = strings.TrimSpace(gjson.GetBytes(body, "prompt").String())
	default:
		c.Input = strings.TrimSpace(gjson.GetBytes(body, "input").String())
	}
	return c
}
