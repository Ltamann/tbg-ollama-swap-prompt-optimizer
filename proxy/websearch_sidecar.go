package proxy

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Ltamann/tbg-ollama-swap-prompt-optimizer/proxy/config"
)

type managedSidecar struct {
	name       string
	logger     *LogMonitor
	cmd        *exec.Cmd
	cmdString  string
	stopString string
	running    bool
	mu         sync.Mutex
}

func newManagedSidecar(name string, logger *LogMonitor) *managedSidecar {
	return &managedSidecar{name: name, logger: logger}
}

func shellCommand(command string) (*exec.Cmd, error) {
	args, err := config.SanitizeCommand(command)
	if err != nil {
		return nil, err
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("empty command")
	}
	cmd := exec.Command(args[0], args[1:]...)
	setProcAttributes(cmd)
	return cmd, nil
}

func (m *managedSidecar) Start(command string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	command = strings.TrimSpace(command)
	if command == "" {
		return fmt.Errorf("%s start command is empty", m.name)
	}
	if m.running {
		if strings.TrimSpace(m.cmdString) == command {
			return nil
		}
		if err := m.stopLocked(); err != nil {
			return err
		}
	}

	cmd, err := shellCommand(command)
	if err != nil {
		return err
	}
	cmd.Stdout = m.logger
	cmd.Stderr = m.logger
	if err := cmd.Start(); err != nil {
		return err
	}
	m.cmd = cmd
	m.cmdString = command
	m.running = true
	m.logger.Infof("[%s] started managed sidecar: %s", m.name, command)

	go func(localCmd *exec.Cmd, expected string) {
		err := localCmd.Wait()
		m.mu.Lock()
		defer m.mu.Unlock()
		if m.cmd == localCmd {
			m.running = false
			m.cmd = nil
		}
		if err != nil {
			m.logger.Warnf("[%s] sidecar exited: %v", m.name, err)
		} else {
			m.logger.Infof("[%s] sidecar exited cleanly", m.name)
		}
		if strings.TrimSpace(m.cmdString) == expected && m.cmd == nil {
			m.cmdString = expected
		}
	}(cmd, command)

	return nil
}

func (m *managedSidecar) SetStopCommand(command string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopString = strings.TrimSpace(command)
}

func (m *managedSidecar) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.stopLocked()
}

func (m *managedSidecar) stopLocked() error {
	if !m.running || m.cmd == nil || m.cmd.Process == nil {
		m.running = false
		m.cmd = nil
		return nil
	}

	if strings.TrimSpace(m.stopString) != "" {
		stopCmd, err := shellCommand(m.stopString)
		if err != nil {
			return err
		}
		stopCmd.Stdout = m.logger
		stopCmd.Stderr = m.logger
		if err := stopCmd.Run(); err != nil {
			return err
		}
	} else if runtime.GOOS == "windows" {
		if err := m.cmd.Process.Kill(); err != nil {
			return err
		}
	} else {
		if err := m.cmd.Process.Signal(syscall.SIGTERM); err != nil {
			return err
		}
	}

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if !m.running {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	if m.cmd != nil && m.cmd.Process != nil {
		_ = m.cmd.Process.Kill()
	}
	m.running = false
	m.cmd = nil
	return nil
}

func (m *managedSidecar) Restart(command string) error {
	if err := m.Stop(); err != nil {
		return err
	}
	return m.Start(command)
}

func (m *managedSidecar) Status() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.running {
		return "running"
	}
	return "stopped"
}

func (m *managedSidecar) StartIfEnabled(ctx context.Context, enabled bool, command string) {
	if !enabled || strings.TrimSpace(command) == "" {
		return
	}
	go func() {
		select {
		case <-ctx.Done():
			return
		default:
			if err := m.Start(command); err != nil {
				m.logger.Warnf("[%s] failed to start managed sidecar: %v", m.name, err)
			}
		}
	}()
}
