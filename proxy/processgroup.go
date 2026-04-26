package proxy

import (
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
)

type ProcessGroup struct {
	sync.Mutex

	config     config.Config
	id         string
	swap       bool
	exclusive  bool
	persistent bool

	proxyLogger     *LogMonitor
	upstreamLogger  *LogMonitor
	transformLogger *LogMonitor

	// map of current processes
	processes       map[string]*Process
	lastUsedProcess string
}

func NewProcessGroup(id string, config config.Config, proxyLogger *LogMonitor, upstreamLogger *LogMonitor, transformLogger *LogMonitor) *ProcessGroup {
	groupConfig, ok := config.Groups[id]
	if !ok {
		panic("Unable to find configuration for group id: " + id)
	}

	pg := &ProcessGroup{
		id:              id,
		config:          config,
		swap:            groupConfig.Swap,
		exclusive:       groupConfig.Exclusive,
		persistent:      groupConfig.Persistent,
		proxyLogger:     proxyLogger,
		upstreamLogger:  upstreamLogger,
		transformLogger: transformLogger,
		processes:       make(map[string]*Process),
	}

	// Create a Process for each member in the group
	for _, modelID := range groupConfig.Members {
		modelConfig, modelID, _ := pg.config.FindConfig(modelID)
		processLogger := NewLogMonitorWriter(upstreamLogger)
		process := NewProcess(modelID, pg.config.HealthCheckTimeout, modelConfig, processLogger, pg.proxyLogger, pg.transformLogger)
		pg.processes[modelID] = process
	}

	return pg
}

// ProxyRequest proxies a request to the specified model
func (pg *ProcessGroup) ProxyRequest(modelID string, writer http.ResponseWriter, request *http.Request) error {
	if !pg.HasMember(modelID) {
		return fmt.Errorf("model %s not part of group %s", modelID, pg.id)
	}

	if pg.swap {
		pg.Lock()
		if pg.lastUsedProcess != modelID {

			// is there something already running?
			if pg.lastUsedProcess != "" {
				pg.processes[pg.lastUsedProcess].Stop()
			}

			// wait for the request to the new model to be fully handled
			// and prevent race conditions see issue #277
			pg.processes[modelID].ProxyRequest(writer, request)
			pg.lastUsedProcess = modelID

			// short circuit and exit
			pg.Unlock()
			return nil
		}
		pg.Unlock()
	}

	pg.processes[modelID].ProxyRequest(writer, request)
	return nil
}

func (pg *ProcessGroup) HasMember(modelName string) bool {
	return slices.Contains(pg.config.Groups[pg.id].Members, modelName)
}

func (pg *ProcessGroup) GetMember(modelName string) (*Process, bool) {
	if pg.HasMember(modelName) {
		return pg.processes[modelName], true
	}
	return nil, false
}

func (pg *ProcessGroup) EnsureStarted(modelID string) error {
	if !pg.HasMember(modelID) {
		return fmt.Errorf("model %s not part of group %s", modelID, pg.id)
	}

	process := pg.processes[modelID]
	if process == nil {
		return fmt.Errorf("process not found for model %s", modelID)
	}

	if process.CurrentState() == StateReady {
		return nil
	}
	const maxStartAttempts = 8
	var lastErr error
	for attempt := 1; attempt <= maxStartAttempts; attempt++ {
		if process.CurrentState() == StateReady {
			return nil
		}
		if err := process.start(); err != nil {
			lastErr = err
			if !isTransientStartError(err) || attempt == maxStartAttempts {
				return err
			}
			pg.proxyLogger.Infof("<%s> transient startup failure in EnsureStarted, retrying (%d/%d): %v", modelID, attempt, maxStartAttempts, err)
			time.Sleep(time.Duration(attempt) * 250 * time.Millisecond)
			continue
		}
		return nil
	}
	return lastErr
}

func isTransientStartError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(strings.TrimSpace(err.Error()))
	if msg == "" {
		return false
	}
	transientMarkers := []string{
		"upstream command exited prematurely but successfully",
		"process was already starting but wound up in state stopped",
		"processes was in state starting when start() was called",
	}
	for _, marker := range transientMarkers {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

func (pg *ProcessGroup) StopProcess(modelID string, strategy StopStrategy) error {
	pg.Lock()

	process, exists := pg.processes[modelID]
	if !exists {
		pg.Unlock()
		return fmt.Errorf("process not found for %s", modelID)
	}

	if pg.lastUsedProcess == modelID {
		pg.lastUsedProcess = ""
	}
	pg.Unlock()

	switch strategy {
	case StopImmediately:
		process.StopImmediately()
	default:
		process.Stop()
	}
	return nil
}

func (pg *ProcessGroup) StopProcesses(strategy StopStrategy) {
	pg.Lock()
	defer pg.Unlock()

	if len(pg.processes) == 0 {
		return
	}

	// stop Processes in parallel
	var wg sync.WaitGroup
	for _, process := range pg.processes {
		wg.Add(1)
		go func(process *Process) {
			defer wg.Done()
			switch strategy {
			case StopImmediately:
				process.StopImmediately()
			default:
				process.Stop()
			}
		}(process)
	}
	wg.Wait()
}

func (pg *ProcessGroup) Shutdown() {
	var wg sync.WaitGroup
	for _, process := range pg.processes {
		wg.Add(1)
		go func(process *Process) {
			defer wg.Done()
			process.Shutdown()
		}(process)
	}
	wg.Wait()
}
