package proxy

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

type ParsedToolCall struct {
	CallID    string
	Name      string
	Arguments map[string]any
}

type ToolCallParser interface {
	Name() string
	MatchesModel(modelName string) bool
	ParseAssistantOutput(content string) ([]ParsedToolCall, string)
}

type toolCallParserRegistry struct {
	mu      sync.RWMutex
	parsers []ToolCallParser
}

func (r *toolCallParserRegistry) register(parser ToolCallParser) {
	if parser == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.parsers = append(r.parsers, parser)
}

func (r *toolCallParserRegistry) parse(modelName, content string) ([]ParsedToolCall, string) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, parser := range r.parsers {
		if parser.MatchesModel(modelName) {
			return parser.ParseAssistantOutput(content)
		}
	}
	return nil, strings.TrimSpace(content)
}

var defaultToolCallParserRegistry = newDefaultToolCallParserRegistry()

func newDefaultToolCallParserRegistry() *toolCallParserRegistry {
	registry := &toolCallParserRegistry{}
	registry.register(qwenXMLToolCallParser{})
	return registry
}

func parseModelSpecificToolCalls(modelName, content string) ([]ParsedToolCall, string) {
	return defaultToolCallParserRegistry.parse(modelName, content)
}

type qwenXMLToolCallParser struct{}

func (qwenXMLToolCallParser) Name() string { return "qwen_xml" }

func (qwenXMLToolCallParser) MatchesModel(modelName string) bool {
	modelName = strings.TrimSpace(strings.ToLower(modelName))
	return strings.Contains(modelName, "qwen")
}

func (qwenXMLToolCallParser) ParseAssistantOutput(content string) ([]ParsedToolCall, string) {
	return parseQwenXMLToolCalls(content)
}

func parseQwenXMLToolCalls(text string) ([]ParsedToolCall, string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, ""
	}

	lower := strings.ToLower(text)
	toolOpen := "<tool_call>"
	toolClose := "</tool_call>"
	searchPos := 0
	parts := make([]string, 0, 2)
	calls := make([]ParsedToolCall, 0, 2)

	for {
		openIdx := strings.Index(lower[searchPos:], toolOpen)
		if openIdx < 0 {
			break
		}
		openIdx += searchPos
		closeIdx := strings.Index(lower[openIdx:], toolClose)
		if closeIdx < 0 {
			break
		}
		closeIdx += openIdx
		blockEnd := closeIdx + len(toolClose)

		if pre := strings.TrimSpace(text[searchPos:openIdx]); pre != "" {
			parts = append(parts, pre)
		}

		block := text[openIdx:blockEnd]
		if call, ok := parseSingleQwenXMLToolCall(block); ok {
			calls = append(calls, call)
		} else if b := strings.TrimSpace(block); b != "" {
			parts = append(parts, b)
		}

		searchPos = blockEnd
	}

	if tail := strings.TrimSpace(text[searchPos:]); tail != "" {
		parts = append(parts, tail)
	}
	return calls, strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func parseSingleQwenXMLToolCall(block string) (ParsedToolCall, bool) {
	lower := strings.ToLower(block)
	funcOpen := "<function="
	funcStart := strings.Index(lower, funcOpen)
	if funcStart < 0 {
		return ParsedToolCall{}, false
	}
	nameStart := funcStart + len(funcOpen)
	nameEndRel := strings.Index(lower[nameStart:], ">")
	if nameEndRel < 0 {
		return ParsedToolCall{}, false
	}
	nameEnd := nameStart + nameEndRel
	name := strings.TrimSpace(block[nameStart:nameEnd])
	if name == "" {
		return ParsedToolCall{}, false
	}

	funcClose := "</function>"
	funcCloseIdx := strings.LastIndex(lower, funcClose)
	if funcCloseIdx < 0 || funcCloseIdx <= nameEnd {
		return ParsedToolCall{}, false
	}

	inner := block[nameEnd+1 : funcCloseIdx]
	return ParsedToolCall{
		CallID:    fmt.Sprintf("call_%d", time.Now().UnixNano()),
		Name:      name,
		Arguments: parseQwenXMLParameters(inner),
	}, true
}

func parseQwenXMLParameters(inner string) map[string]any {
	args := map[string]any{}
	if strings.TrimSpace(inner) == "" {
		return args
	}

	lower := strings.ToLower(inner)
	paramOpen := "<parameter="
	paramClose := "</parameter>"
	funcClose := "</function>"
	searchPos := 0

	for {
		openIdx := strings.Index(lower[searchPos:], paramOpen)
		if openIdx < 0 {
			break
		}
		openIdx += searchPos
		keyStart := openIdx + len(paramOpen)
		keyEndRel := strings.Index(lower[keyStart:], ">")
		if keyEndRel < 0 {
			break
		}
		keyEnd := keyStart + keyEndRel
		key := strings.TrimSpace(inner[keyStart:keyEnd])
		if key == "" {
			searchPos = keyEnd + 1
			continue
		}

		valueStart := keyEnd + 1
		closeTagIdx := strings.Index(lower[valueStart:], paramClose)
		nextParamIdx := strings.Index(lower[valueStart:], paramOpen)
		nextFuncClose := strings.Index(lower[valueStart:], funcClose)

		valueEnd := len(inner)
		if closeTagIdx >= 0 {
			valueEnd = valueStart + closeTagIdx
		} else if nextParamIdx >= 0 {
			valueEnd = valueStart + nextParamIdx
		} else if nextFuncClose >= 0 {
			valueEnd = valueStart + nextFuncClose
		}

		value := strings.TrimSpace(inner[valueStart:valueEnd])
		if value != "" {
			var parsed any
			if json.Unmarshal([]byte(value), &parsed) == nil {
				args[key] = parsed
			} else if strings.EqualFold(key, "commands") {
				args[key] = splitNonEmptyLines(value)
			} else {
				args[key] = value
			}
		}

		if closeTagIdx >= 0 {
			searchPos = valueStart + closeTagIdx + len(paramClose)
		} else if nextParamIdx >= 0 {
			searchPos = valueStart + nextParamIdx
		} else {
			break
		}
	}
	return args
}
