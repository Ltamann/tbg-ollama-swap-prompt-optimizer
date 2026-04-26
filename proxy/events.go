package proxy

// package level registry of the different event types

const ProcessStateChangeEventID = 0x01
const ChatCompletionStatsEventID = 0x02
const ConfigFileChangedEventID = 0x03
const LogDataEventID = 0x04
const TokenMetricsEventID = 0x05
const ModelPreloadedEventID = 0x06
const MonitorEventID = 0x07

type ProcessStateChangeEvent struct {
	ProcessName string
	NewState    ProcessState
	OldState    ProcessState
}

func (e ProcessStateChangeEvent) Type() uint32 {
	return ProcessStateChangeEventID
}

type ChatCompletionStats struct {
	TokensGenerated int
}

func (e ChatCompletionStats) Type() uint32 {
	return ChatCompletionStatsEventID
}

type ReloadingState int

const (
	ReloadingStateStart ReloadingState = iota
	ReloadingStateEnd
)

type ConfigFileChangedEvent struct {
	ReloadingState ReloadingState
}

func (e ConfigFileChangedEvent) Type() uint32 {
	return ConfigFileChangedEventID
}

type LogDataEvent struct {
	Data []byte
}

func (e LogDataEvent) Type() uint32 {
	return LogDataEventID
}

type ModelPreloadedEvent struct {
	ModelName string
	Success   bool
}

func (e ModelPreloadedEvent) Type() uint32 {
	return ModelPreloadedEventID
}

type LiveMonitorEvent struct {
	TraceID   string `json:"trace_id"`
	Timestamp string `json:"timestamp"`
	Model     string `json:"model"`
	Stage     string `json:"stage"`
	Direction string `json:"direction"`
	Endpoint  string `json:"endpoint,omitempty"`
	Event     string `json:"event,omitempty"`
	Data      string `json:"data,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

type MonitorEvent struct {
	Event LiveMonitorEvent
}

func (e MonitorEvent) Type() uint32 {
	return MonitorEventID
}
