package proxy

import (
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManagedSidecar_StartAndStop(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell command test is written for posix environments")
	}

	sidecar := newManagedSidecar("test-searxng", testLogger)
	err := sidecar.Start(`sh -lc "sleep 30"`)
	require.NoError(t, err)
	assert.Equal(t, "running", sidecar.Status())

	require.NoError(t, sidecar.Stop())
	time.Sleep(200 * time.Millisecond)
	assert.Equal(t, "stopped", sidecar.Status())
}
