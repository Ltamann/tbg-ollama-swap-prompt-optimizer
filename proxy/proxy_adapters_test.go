package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultToolRepairAdapter_UsesQwenParserCompatibility(t *testing.T) {
	content := `<tool_call><function=shell><parameter=command>pwd</parameter></function></tool_call>`

	calls, remaining := proxyCompatibilityAdapters.ToolRepair.ParseAssistantOutput("gpt-5.3-codex", content)
	require.Len(t, calls, 1)
	assert.Equal(t, "shell", calls[0].Name)
	assert.Equal(t, "pwd", firstCommand(t, calls[0].Arguments["commands"]))
	assert.Equal(t, "", remaining)
}

func TestDefaultReasoningTranslationAdapter_UsesPreviewCompatibility(t *testing.T) {
	reasoning := "The user wants a plan. " + `<think>` + "detail " + `</think>`

	preview := proxyCompatibilityAdapters.Reasoning.BuildCommentaryPreview(reasoning)
	assert.NotEmpty(t, preview)
	assert.NotContains(t, preview, "<think>")
	assert.Contains(t, preview, "The user wants a plan.")
}

func TestDefaultStreamReconstructionAdapter_CanonicalizesKnownTools(t *testing.T) {
	state := &StreamToolCallState{Name: "__llamaswap_apply_patch"}
	assert.Equal(t, "apply_patch", proxyCompatibilityAdapters.Stream.CanonicalToolName(state.Name))

	state = &StreamToolCallState{Name: "applypatch"}
	assert.Equal(t, "apply_patch", proxyCompatibilityAdapters.Stream.CanonicalToolName(state.Name))
}

func TestDefaultStreamReconstructionAdapter_RejectsEmptyShellExposure(t *testing.T) {
	state := &StreamToolCallState{Name: "shell"}
	state.ArgsBuilder.WriteString(`{}`)

	assert.False(t, proxyCompatibilityAdapters.Stream.ShouldExposeToolCall(state))

	state.ArgsBuilder.Reset()
	state.ArgsBuilder.WriteString(`{"command":"pwd"}`)
	assert.True(t, proxyCompatibilityAdapters.Stream.ShouldExposeToolCall(state))
}

func TestDefaultContinuationController_ReflectsCurrentCompatibilityHeuristics(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":   "apply_patch_call_output",
				"output": `{"ok":true}`,
			},
		},
	}

	state := proxyCompatibilityAdapters.Continuation.BuildWorkflowState(req)
	assert.True(t, state.HasToolOutput)
	assert.False(t, completedToolNamesContain(state.CompletedToolNames, "request_user_input"))
}

func TestDefaultToolRepairAdapter_ValidatesStructuredToolContracts(t *testing.T) {
	t.Run("request_user_input requires questions", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "request_user_input",
			"arguments": `{}`,
		})
		assert.False(t, result.Valid)
		assert.Contains(t, result.Warning, "`questions` was empty")
	})

	t.Run("update_plan requires plan steps", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "update_plan",
			"arguments": `{"plan":[]}`,
		})
		assert.False(t, result.Valid)
		assert.Contains(t, result.Warning, "`plan` was empty")
	})

	t.Run("parallel requires tool uses", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "multi_tool_use.parallel",
			"arguments": `{"tool_uses":[]}`,
		})
		assert.False(t, result.Valid)
		assert.Contains(t, result.Warning, "`tool_uses` was empty")
	})

	t.Run("valid request_user_input survives", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "request_user_input",
			"arguments": `{"questions":[{"header":"Need","question":"Which file?"}]}`,
		})
		assert.True(t, result.Valid)
		assert.Equal(t, ToolLifecycleValidated, result.LifecycleState)
	})

	t.Run("shell command is canonicalized to shell with response safe command shape", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "shell_command",
			"arguments": `{"command":["powershell.exe","-Command","Get-Content mutations/base_a.txt"]}`,
		})
		require.True(t, result.Valid)
		require.NotNil(t, result.NormalizedItem)
		assert.Equal(t, "shell", result.NormalizedItem["name"])
		args := parseToolArgsMapString(fmt.Sprintf("%v", result.NormalizedItem["arguments"]))
		assert.Equal(t, []any{"powershell.exe", "-Command", "Get-Content mutations/base_a.txt"}, args["command"])
		_, hasCommands := args["commands"]
		assert.False(t, hasCommands)
	})

	t.Run("request_user_input canonicalizes question list", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "request_user_input",
			"arguments": `{"questions":["",{"header":" Need ","question":" Which file? "}]}`,
		})
		require.True(t, result.Valid)
		args := parseToolArgsMapString(fmt.Sprintf("%v", result.NormalizedItem["arguments"]))
		questions, _ := args["questions"].([]any)
		require.Len(t, questions, 1)
		q := questions[0].(map[string]any)
		assert.Equal(t, "Need", q["header"])
		assert.Equal(t, "Which file?", q["question"])
	})

	t.Run("update_plan canonicalizes plan steps", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "update_plan",
			"arguments": `{"plan":[{"step":"  Add tests  ","status":"INVALID"},""]}`,
		})
		require.True(t, result.Valid)
		args := parseToolArgsMapString(fmt.Sprintf("%v", result.NormalizedItem["arguments"]))
		plan, _ := args["plan"].([]any)
		require.Len(t, plan, 1)
		step := plan[0].(map[string]any)
		assert.Equal(t, "Add tests", step["step"])
		assert.Equal(t, "pending", step["status"])
	})

	t.Run("parallel canonicalizes tool uses", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type":      "function_call",
			"name":      "multi_tool_use.parallel",
			"arguments": `{"tool_uses":[{"recipient_name":" functions.shell_command ","parameters":{"command":"pwd"}},{"parameters":{"command":"skip"}}]}`,
		})
		require.True(t, result.Valid)
		args := parseToolArgsMapString(fmt.Sprintf("%v", result.NormalizedItem["arguments"]))
		toolUses, _ := args["tool_uses"].([]any)
		require.Len(t, toolUses, 1)
		use := toolUses[0].(map[string]any)
		assert.Equal(t, "functions.shell_command", use["recipient_name"])
	})

	t.Run("apply_patch canonicalizes operation input", func(t *testing.T) {
		result := proxyCompatibilityAdapters.ToolRepair.ValidateToolCallItem(map[string]any{
			"type": "apply_patch_call",
			"operation": map[string]any{
				"type":    "update_file",
				"path":    "README.md",
				"content": "PATCH_OK\n",
			},
		})
		require.True(t, result.Valid)
		require.NotNil(t, result.NormalizedItem)
		assert.NotEmpty(t, strings.TrimSpace(fmt.Sprintf("%v", result.NormalizedItem["input"])))
	})
}

func TestDefaultContinuationController_BuildDecision(t *testing.T) {
	t.Run("forces final answer after satisfied apply patch", func(t *testing.T) {
		dir := t.TempDir()
		target := filepath.Join(dir, "demo.txt")
		require.NoError(t, os.WriteFile(target, []byte("before\nPATCH_OK\nafter\n"), 0o644))
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{
							"type": "input_text",
							"text": fmt.Sprintf("Use apply_patch to update %s by appending one line: PATCH_OK. Then reply exactly DONE.", target),
						},
					},
				},
				map[string]any{"type": "apply_patch_call_output", "call_id": "call_1", "output": `{"ok":true}`},
			},
		}
		decision := proxyCompatibilityAdapters.Continuation.BuildDecision(req, ContinuationContext{})
		assert.Equal(t, ContinuationStateFinalAnswerRequired, decision.State)
		assert.True(t, decision.DisableTools)
		require.NotEmpty(t, decision.Instructions)
		assert.Contains(t, strings.Join(decision.Instructions, "\n"), "previous apply_patch already produced the requested file change")
	})

	t.Run("forces request user input during codex managed plan continuation", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{"type": "function_call_output", "call_id": "call_1", "output": `{"cwd":"/tmp"}`},
			},
		}
		decision := proxyCompatibilityAdapters.Continuation.BuildDecision(req, ContinuationContext{
			PlanModeRequested:         true,
			RequestUserInputAvailable: true,
			ActiveToolChoice:          "auto",
		})
		assert.Equal(t, ContinuationStateToolRunning, decision.State)
		forced, ok := decision.ForceToolChoice.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "function", forced["type"])
		assert.Equal(t, "request_user_input", forced["function"].(map[string]any)["name"])
	})

	t.Run("implementation retry after native question keeps tools enabled", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type":      "function_call",
					"name":      "request_user_input",
					"call_id":   "call_q_1",
					"arguments": `{"questions":["What platform?"]}`,
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call_q_1",
					"output":  `{"answers":[{"id":"platform","value":"html"}]}`,
				},
				map[string]any{
					"type": "message",
					"role": "assistant",
					"content": []any{
						map[string]any{"type": "output_text", "text": "apply_patch call was not executed because operation was invalid. Provide `operation.type`, `operation.path`, and non-empty `diff/content` for create/update."},
					},
				},
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "try again"},
					},
				},
			},
		}

		decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{
			PlanOutputRequested:       false,
			PlanModeRequested:         false,
			RequestUserInputAvailable: true,
			ImplementationRetryIntent: true,
			ActiveToolChoice:          "auto",
		})

		assert.Equal(t, ContinuationStateToolRunning, decision.State)
		assert.False(t, decision.DisableTools)
		assert.Nil(t, decision.ForceToolChoice)
	})
}

func TestDefaultContinuationController_BuildDecision_ExplicitNativeQuestionAddsInstruction(t *testing.T) {
	req := map[string]any{
		"tool_choice": "auto",
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{
		PlanModeRequested:         true,
		RequestUserInputAvailable: true,
		NativeQuestionRequested:   true,
	})

	require.NotNil(t, decision.ForceToolChoice)
	assert.Equal(t, ContinuationStatePreTool, decision.State)
	assert.Contains(t, strings.Join(decision.Instructions, "\n"), "Return a native function call named request_user_input")
}

func TestDefaultContinuationController_BuildDecision_ForcesFinalAfterShellVerification(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11, then verify the final file content with shell. Finish with exactly T11_SENTINEL."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_0",
				"arguments": `{"command":"Get-Content mutations/base_a.txt"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_0",
				"output":  "BASE_A\n",
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Success. Updated the following files:\nM mutations/base_a.txt",
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_1",
				"arguments": `{"command":"Get-Content mutations/base_a.txt"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_1",
				"output":  "BASE_A\nORDERED_T11\n",
			},
		},
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{})

	assert.Equal(t, ContinuationStateFinalAnswerRequired, decision.State)
	assert.True(t, decision.DisableTools)
	assert.Contains(t, strings.Join(decision.Instructions, "\n"), "requested shell verification already completed")
}

func TestBuildLoopGuardDecision_CentralizesLoopProtections(t *testing.T) {
	t.Run("forces final after satisfied apply patch", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			ApplyPatchSatisfied: true,
		}, ContinuationContext{}, nil, "")
		require.True(t, guard.Triggered)
		assert.Equal(t, ContinuationStateFinalAnswerRequired, guard.State)
		assert.True(t, guard.DisableTools)
		assert.Contains(t, strings.Join(guard.Instructions, "\n"), "previous apply_patch already produced the requested file change")
	})

	t.Run("forces shell verification followup before final answer", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			LatestCompletedToolName: "apply_patch",
			VerificationExpected:    true,
			VerificationCompleted:   false,
		}, ContinuationContext{}, nil, "")
		require.True(t, guard.Triggered)
		assert.Equal(t, ContinuationStateToolCompletedAwaitingFollowup, guard.State)
		forced, ok := guard.ForceToolChoice.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "shell", forced["function"].(map[string]any)["name"])
	})

	t.Run("does not close tools after answered question without repeat evidence", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			CompletedToolNames: []string{"request_user_input"},
		}, ContinuationContext{
			PlanModeRequested:   true,
			PlanOutputRequested: true,
		}, "auto", "")
		assert.False(t, guard.Triggered)
	})

	t.Run("allows explicit exploration followup after answers", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			CompletedToolNames: []string{"request_user_input"},
			HasToolOutput:      true,
		}, ContinuationContext{
			PlanModeRequested:         true,
			ExplorationFollowupIntent: true,
		}, "auto", "")
		assert.False(t, guard.Triggered)
	})

	t.Run("forces final after repeated completed tool fingerprint once workflow is safe", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			LatestCompletedToolName:          "shell",
			LatestCompletedToolFingerprint:   `shell|{"command":"Get-Content demo.txt"}`,
			PreviousCompletedToolFingerprint: `shell|{"command":"Get-Content demo.txt"}`,
			RepeatedLatestToolFingerprint:    true,
			FinalAnswerSafe:                  true,
		}, ContinuationContext{}, nil, "DONE_SENTINEL")
		require.True(t, guard.Triggered)
		assert.Equal(t, ContinuationStateFinalAnswerRequired, guard.State)
		assert.True(t, guard.DisableTools)
		assert.Contains(t, strings.Join(guard.Instructions, "\n"), "already completed with the same arguments and output pattern again")
		assert.Contains(t, strings.Join(guard.Instructions, "\n"), `reply with exactly "DONE_SENTINEL"`)
	})

	t.Run("keeps tools available after repeated request user input fingerprint", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			LatestCompletedToolName:          "request_user_input",
			LatestCompletedToolFingerprint:   `request_user_input|{"questions":["What style?"]}`,
			PreviousCompletedToolFingerprint: `request_user_input|{"questions":["What style?"]}`,
			RepeatedLatestToolFingerprint:    true,
			FinalAnswerSafe:                  true,
		}, ContinuationContext{}, nil, "")
		require.True(t, guard.Triggered)
		assert.Equal(t, ContinuationStateToolCompletedAwaitingFollowup, guard.State)
		assert.False(t, guard.DisableTools)
		assert.Contains(t, strings.Join(guard.Instructions, "\n"), "Do not ask the same question again")
	})

	t.Run("keeps tools available after repeated web search fingerprint", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			LatestCompletedToolName:          "web_search",
			LatestCompletedToolFingerprint:   `web_search|{"query":"10 best chemistry trivia questions and answers"}`,
			PreviousCompletedToolFingerprint: `web_search|{"query":"10 best chemistry trivia questions and answers"}`,
			RepeatedLatestToolFingerprint:    true,
			FinalAnswerSafe:                  true,
		}, ContinuationContext{}, nil, "")
		require.True(t, guard.Triggered)
		assert.Equal(t, ContinuationStateToolCompletedAwaitingFollowup, guard.State)
		assert.False(t, guard.DisableTools)
		assert.Contains(t, strings.Join(guard.Instructions, "\n"), "Do not repeat the same web_search again")
	})

	t.Run("does not force final after repeated apply patch fingerprint alone", func(t *testing.T) {
		guard := buildLoopGuardDecision(ToolWorkflowState{
			LatestCompletedToolName:          "apply_patch",
			LatestCompletedToolFingerprint:   `apply_patch|Success. Updated the following files:\ndeleted: biology-quiz.html`,
			PreviousCompletedToolFingerprint: `apply_patch|Success. Updated the following files:\ndeleted: biology-quiz.html`,
			RepeatedLatestToolFingerprint:    true,
			FinalAnswerSafe:                  true,
		}, ContinuationContext{}, nil, "")
		assert.False(t, guard.Triggered)
	})
}

func TestDefaultContinuationController_BuildDecision_RequiresVerificationAfterApplyPatch(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11, then verify the final file content with shell. Finish with exactly T11_SENTINEL."},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "shell",
				"call_id":   "call_shell_0",
				"arguments": `{"command":"Get-Content mutations/base_a.txt"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_shell_0",
				"output":  "BASE_A\n",
			},
			map[string]any{
				"type":      "function_call",
				"name":      "apply_patch",
				"call_id":   "call_patch_1",
				"arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_patch_1",
				"output":  "Success. Updated the following files:\nM mutations/base_a.txt",
			},
		},
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{})

	assert.Equal(t, ContinuationStateToolCompletedAwaitingFollowup, decision.State)
	assert.False(t, decision.DisableTools)
	forced, ok := decision.ForceToolChoice.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", forced["type"])
	assert.Equal(t, "shell", forced["function"].(map[string]any)["name"])
	assert.Contains(t, strings.Join(decision.Instructions, "\n"), "apply_patch step is complete but the user explicitly required verification with shell")
}

func TestDefaultContinuationController_BuildDecision_ExplorationFollowupAfterRequestUserInputKeepsTools(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "do a web_search for it"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{
		PlanModeRequested:         true,
		ExplorationFollowupIntent: true,
		ActiveToolChoice:          "auto",
	})

	assert.Equal(t, ContinuationStateToolRunning, decision.State)
	assert.False(t, decision.DisableTools)
	assert.Nil(t, decision.ForceToolChoice)
}

func TestDefaultContinuationController_BuildDecision_PlanModeAfterAnsweredQuestionKeepsQuestionLoopAlive(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{
		PlanModeRequested:         true,
		RequestUserInputAvailable: true,
		ActiveToolChoice:          "auto",
	})

	assert.Equal(t, ContinuationStateToolRunning, decision.State)
	assert.False(t, decision.DisableTools)
	forced, ok := decision.ForceToolChoice.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", forced["type"])
	assert.Equal(t, "request_user_input", forced["function"].(map[string]any)["name"])
}

func TestDefaultContinuationController_BuildDecision_PlanOutputRequestedKeepsToolsAvailable(t *testing.T) {
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":      "function_call",
				"name":      "request_user_input",
				"call_id":   "call_q_1",
				"arguments": `{"questions":["What style?"]}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_q_1",
				"output":  `{"answers":[{"id":"style","value":"kid-friendly"}]}`,
			},
		},
	}

	decision := defaultContinuationController{}.BuildDecision(req, ContinuationContext{
		PlanModeRequested:         true,
		PlanOutputRequested:       true,
		RequestUserInputAvailable: true,
		ActiveToolChoice:          "auto",
	})

	assert.Equal(t, ContinuationStateToolCompletedAwaitingFollowup, decision.State)
	assert.False(t, decision.DisableTools)
	assert.Nil(t, decision.ForceToolChoice)
}

func TestClassifyContinuationTurnPhase(t *testing.T) {
	t.Run("plan gather", func(t *testing.T) {
		phase := classifyContinuationTurnPhase(ContinuationContext{
			PlanModeRequested:         true,
			RequestUserInputAvailable: true,
		})
		assert.Equal(t, ContinuationTurnPhasePlanGather, phase)
	})

	t.Run("plan finalize", func(t *testing.T) {
		phase := classifyContinuationTurnPhase(ContinuationContext{
			PlanModeRequested:   true,
			PlanOutputRequested: true,
		})
		assert.Equal(t, ContinuationTurnPhasePlanFinalize, phase)
	})

	t.Run("research overrides answered question followup", func(t *testing.T) {
		phase := classifyContinuationTurnPhase(ContinuationContext{
			PlanModeRequested:         true,
			RequestUserInputAvailable: true,
			SearchIntent:              true,
		})
		assert.Equal(t, ContinuationTurnPhaseResearch, phase)
	})

	t.Run("implementation retry has highest priority", func(t *testing.T) {
		phase := classifyContinuationTurnPhase(ContinuationContext{
			PlanModeRequested:         true,
			PlanOutputRequested:       true,
			ImplementationRetryIntent: true,
			SearchIntent:              true,
		})
		assert.Equal(t, ContinuationTurnPhaseImplementationRetry, phase)
	})
}

func TestDefaultContinuationController_DeriveAllowedToolNames_UsesWorkflowState(t *testing.T) {
	t.Run("returns shell only while apply patch verification is still pending", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11, then verify the final file content with shell."},
					},
				},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_0", "arguments": `{"command":"Get-Content mutations/base_a.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_0", "output": "BASE_A\n"},
				map[string]any{"type": "function_call", "name": "apply_patch", "call_id": "call_patch_1", "arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`},
				map[string]any{"type": "function_call_output", "call_id": "call_patch_1", "output": "Success. Updated the following files:\nM mutations/base_a.txt"},
			},
		}

		allowed := defaultContinuationController{}.DeriveAllowedToolNames(req)
		assert.Equal(t, []string{"shell"}, allowed)
	})

	t.Run("does not collapse to apply patch only after a completed patch", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "Continue by checking the created file and open it in the browser."},
					},
				},
				map[string]any{"type": "function_call", "name": "apply_patch", "call_id": "call_patch_1", "arguments": `{"operation":{"type":"create_file","path":"biology-quiz.html","content":"<html></html>"}}`},
				map[string]any{"type": "function_call_output", "call_id": "call_patch_1", "output": "Success. Updated the following files:\nA biology-quiz.html"},
			},
		}

		allowed := defaultContinuationController{}.DeriveAllowedToolNames(req)
		assert.Nil(t, allowed)
	})

	t.Run("returns shell after only shell output", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "Use shell to inspect demo.txt."},
					},
				},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_0", "arguments": `{"command":"Get-Content demo.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_0", "output": "BASE\n"},
			},
		}

		allowed := defaultContinuationController{}.DeriveAllowedToolNames(req)
		assert.Equal(t, []string{"shell"}, allowed)
	})
}

func TestBuildToolWorkflowState(t *testing.T) {
	t.Run("tracks apply patch verification workflow by sequence", func(t *testing.T) {
		dir := t.TempDir()
		target := filepath.Join(dir, "base_a.txt")
		require.NoError(t, os.WriteFile(target, []byte("BASE_A\nORDERED_T11\n"), 0o644))
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": fmt.Sprintf("First use shell to inspect %s, then use apply_patch to append ORDERED_T11, then verify the final file content with shell. Finish with exactly T11_SENTINEL.", target)},
					},
				},
				map[string]any{
					"type":      "function_call",
					"name":      "shell",
					"call_id":   "call_shell_0",
					"arguments": `{"command":"Get-Content mutations/base_a.txt"}`,
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call_shell_0",
					"output":  "BASE_A\n",
				},
				map[string]any{
					"type":      "function_call",
					"name":      "apply_patch",
					"call_id":   "call_patch_1",
					"arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`,
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call_patch_1",
					"output":  "Success. Updated the following files:\nM mutations/base_a.txt",
				},
			},
		}

		state := buildToolWorkflowState(req)

		assert.True(t, state.HasToolOutput)
		assert.Equal(t, []string{"shell", "apply_patch"}, state.CompletedToolNames)
		assert.Empty(t, state.PendingToolNames)
		assert.Equal(t, "apply_patch", state.LatestCompletedToolName)
		assert.Contains(t, state.LatestCompletedToolFingerprint, "apply_patch|")
		assert.Contains(t, state.LatestCompletedToolFingerprint, "mutations/base_a.txt")
		assert.False(t, state.ApplyPatchSatisfied)
		assert.True(t, state.VerificationExpected)
		assert.False(t, state.VerificationCompleted)
		assert.False(t, state.FinalAnswerSafe)
	})

	t.Run("marks final answer safe only after verification shell completes", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "First use shell to inspect mutations/base_a.txt, then use apply_patch to append ORDERED_T11, then verify the final file content with shell. Finish with exactly T11_SENTINEL."},
					},
				},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_0", "arguments": `{"command":"Get-Content mutations/base_a.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_0", "output": "BASE_A\n"},
				map[string]any{"type": "function_call", "name": "apply_patch", "call_id": "call_patch_1", "arguments": `{"operation":{"type":"update_file","path":"mutations/base_a.txt","content":"BASE_A\nORDERED_T11"}}`},
				map[string]any{"type": "function_call_output", "call_id": "call_patch_1", "output": "Success. Updated the following files:\nM mutations/base_a.txt"},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_1", "arguments": `{"command":"Get-Content mutations/base_a.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_1", "output": "BASE_A\nORDERED_T11\n"},
			},
		}

		state := buildToolWorkflowState(req)

		assert.True(t, state.HasToolOutput)
		assert.Equal(t, []string{"shell", "apply_patch", "shell"}, state.CompletedToolNames)
		assert.Empty(t, state.PendingToolNames)
		assert.Equal(t, "shell", state.LatestCompletedToolName)
		assert.Contains(t, state.LatestCompletedToolFingerprint, "shell|")
		assert.Contains(t, state.LatestCompletedToolFingerprint, "Get-Content mutations/base_a.txt")
		assert.Contains(t, state.PreviousCompletedToolFingerprint, "apply_patch|")
		assert.False(t, state.RepeatedLatestToolFingerprint)
		assert.True(t, state.VerificationExpected)
		assert.True(t, state.VerificationCompleted)
		assert.True(t, state.FinalAnswerSafe)
	})

	t.Run("keeps pending tools visible until output returns", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{
					"type": "message",
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "Use shell to inspect demo.txt, then use apply_patch."},
					},
				},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_0", "arguments": `{"command":"Get-Content demo.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_0", "output": "BASE\n"},
				map[string]any{"type": "function_call", "name": "apply_patch", "call_id": "call_patch_1", "arguments": `{"operation":{"type":"update_file","path":"demo.txt","content":"BASE\nPATCH_OK"}}`},
			},
		}

		state := buildToolWorkflowState(req)

		assert.True(t, state.HasToolOutput)
		assert.Equal(t, []string{"shell"}, state.CompletedToolNames)
		assert.Equal(t, []string{"apply_patch"}, state.PendingToolNames)
		assert.Equal(t, "shell", state.LatestCompletedToolName)
		assert.Contains(t, state.LatestCompletedToolFingerprint, "shell|")
		assert.False(t, state.FinalAnswerSafe)
	})

	t.Run("tracks repeated completed tool fingerprints", func(t *testing.T) {
		req := map[string]any{
			"input": []any{
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_0", "arguments": `{"command":"Get-Content demo.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_0", "output": "BASE\n"},
				map[string]any{"type": "function_call", "name": "shell", "call_id": "call_shell_1", "arguments": `{"command":"Get-Content demo.txt"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_shell_1", "output": "BASE\n"},
			},
		}

		state := buildToolWorkflowState(req)

		assert.Equal(t, []string{"shell", "shell"}, state.CompletedToolNames)
		assert.True(t, state.RepeatedLatestToolFingerprint)
		assert.Equal(t, state.PreviousCompletedToolFingerprint, state.LatestCompletedToolFingerprint)
	})
}

func TestBuildToolWorkflowState_TracksSatisfiedApplyPatch(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "demo.txt")
	require.NoError(t, os.WriteFile(target, []byte("before\nPATCH_OK\nafter\n"), 0o644))

	req := map[string]any{
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": fmt.Sprintf("Use apply_patch to update %s by appending one line: PATCH_OK. Then reply exactly DONE.", target),
					},
				},
			},
			map[string]any{
				"type":    "apply_patch_call_output",
				"call_id": "call_1",
				"output":  `{"ok":true}`,
			},
		},
	}

	state := buildToolWorkflowState(req)

	assert.True(t, state.HasToolOutput)
	assert.True(t, state.ApplyPatchSatisfied)
	assert.True(t, state.FinalAnswerSafe)
	assert.False(t, state.VerificationExpected)
}

func TestSummarizeContinuationState_UsesWorkflowState(t *testing.T) {
	summary := summarizeContinuationState(
		ContinuationStateToolCompletedAwaitingFollowup,
		ToolWorkflowState{
			LatestCompletedToolName: "apply_patch",
			PendingToolNames:        []string{"shell"},
			VerificationExpected:    true,
			VerificationCompleted:   false,
			FinalAnswerSafe:         false,
		},
		false,
		false,
		true,
	)

	assert.Contains(t, summary, "state=tool_completed_awaiting_followup")
	assert.Contains(t, summary, "has_tool_output=true")
	assert.Contains(t, summary, "latest_completed=apply_patch")
	assert.Contains(t, summary, "pending=1")
	assert.Contains(t, summary, "apply_patch_satisfied=false")
	assert.Contains(t, summary, "verify_expected=true")
	assert.Contains(t, summary, "verify_completed=false")
	assert.Contains(t, summary, "final_safe=false")
}

func TestDefaultStreamReconstructionAdapter_ResponseToolViews(t *testing.T) {
	t.Run("normalizes apply patch response item into apply patch call view", func(t *testing.T) {
		item := map[string]any{
			"type":      "function_call",
			"name":      "apply_patch",
			"call_id":   "call_1",
			"arguments": `{"operation":{"type":"update_file","path":"README.md","content":"PATCH_OK"}}`,
		}
		normalized := proxyCompatibilityAdapters.Stream.NormalizeResponseOutputItem(item)
		assert.Equal(t, "function_call", normalized["type"])
		assert.Equal(t, "apply_patch", normalized["name"])
		assert.Contains(t, fmt.Sprintf("%v", normalized["arguments"]), `"operation"`)

		view, ok := proxyCompatibilityAdapters.Stream.BuildResponseToolItemView(normalized)
		require.True(t, ok)
		assert.Equal(t, "function_call", view.ItemType)
		assert.Equal(t, "apply_patch", view.ToolName)
		assert.True(t, view.EmitsArgumentEvents)
		assert.Contains(t, view.Arguments, `"operation"`)
	})

	t.Run("content-driven apply patch normalization does not synthesize operation diff", func(t *testing.T) {
		item := map[string]any{
			"type":      "function_call",
			"name":      "apply_patch",
			"call_id":   "call_2",
			"arguments": `{"operation":{"type":"update_file","path":"README.md","content":"BASE_A\nORDERED_T11"}}`,
		}
		normalized := proxyCompatibilityAdapters.Stream.NormalizeResponseOutputItem(item)
		args := fmt.Sprintf("%v", normalized["arguments"])
		assert.Contains(t, args, `"content":"BASE_A\nORDERED_T11"`)
		assert.NotContains(t, args, `"diff"`)
	})

	t.Run("builds shell tool view with emitted argument events", func(t *testing.T) {
		item := map[string]any{
			"id":      "fc_1",
			"type":    "shell_call",
			"call_id": "call_shell",
			"action":  map[string]any{"commands": []any{"pwd"}},
		}
		view, ok := proxyCompatibilityAdapters.Stream.BuildResponseToolItemView(item)
		require.True(t, ok)
		assert.Equal(t, "shell", view.ToolName)
		assert.True(t, view.EmitsArgumentEvents)
		assert.Contains(t, view.Arguments, "pwd")
	})
}

func TestDefaultStreamReconstructionAdapter_ResponseToolArgumentEvents(t *testing.T) {
	t.Run("builds non-stream tool argument replay events", func(t *testing.T) {
		view := ResponseToolItemView{
			ItemID:              "fc_1",
			CallID:              "call_1",
			ToolName:            "shell",
			Arguments:           `{"command":"pwd"}`,
			EmitsArgumentEvents: true,
		}
		events := proxyCompatibilityAdapters.Stream.BuildResponseToolArgumentEvents(view, "resp_1", 2)
		require.Len(t, events, 2)
		assert.Equal(t, StreamEventToolArgsDelta, events[0].Kind)
		assert.Equal(t, 2, events[0].OutputIndex)
		assert.Equal(t, "shell", events[0].ToolName)
		assert.Contains(t, fmt.Sprintf("%v", events[0].Payload["delta"]), "pwd")
		assert.Equal(t, StreamEventToolArgsDone, events[1].Kind)
		assert.Contains(t, fmt.Sprintf("%v", events[1].Payload["arguments"]), "pwd")
	})

	t.Run("suppresses replay events for tool items that do not emit arguments", func(t *testing.T) {
		view := ResponseToolItemView{
			ItemID:              "fc_2",
			CallID:              "call_2",
			ToolName:            "mcp__playwright__browser_navigate",
			Arguments:           `{"url":"https://example.com"}`,
			EmitsArgumentEvents: false,
		}
		assert.Nil(t, proxyCompatibilityAdapters.Stream.BuildResponseToolArgumentEvents(view, "resp_1", 3))
	})
}

func TestDefaultStreamReconstructionAdapter_ResponseCompletedEvent(t *testing.T) {
	event, ok := proxyCompatibilityAdapters.Stream.BuildResponseCompletedEvent(map[string]any{
		"id":     "resp_1",
		"object": "response",
		"status": "completed",
	})
	require.True(t, ok)
	assert.Equal(t, StreamEventResponseCompleted, event.Kind)
	assert.Equal(t, "resp_1", event.ResponseID)
	response, ok := event.Payload["response"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "completed", response["status"])
}

func TestDefaultStreamReconstructionAdapter_StreamToolEvents(t *testing.T) {
	t.Run("builds shell tool lifecycle events", func(t *testing.T) {
		state := &StreamToolCallState{
			Index:       0,
			OutputIndex: 2,
			ItemID:      "fc_1",
			CallID:      "call_1",
			Name:        "shell",
			Exposed:     true,
		}
		state.ArgsBuilder.WriteString(`{"command":"pwd"}`)
		assert.True(t, proxyCompatibilityAdapters.Stream.ShouldExposeToolCall(state))

		added, ok := proxyCompatibilityAdapters.Stream.BuildToolItemAddedEvent(state, "resp_1")
		require.True(t, ok)
		assert.Equal(t, StreamEventToolItemAdded, added.Kind)
		assert.Equal(t, 2, added.OutputIndex)
		assert.Equal(t, "shell", added.ToolName)
		item := added.Payload["item"].(map[string]any)
		assert.Equal(t, "function_call", item["type"])
		assert.Equal(t, "shell", item["name"])

		delta, ok := proxyCompatibilityAdapters.Stream.BuildToolArgsDeltaEvent(state, "resp_1")
		require.True(t, ok)
		assert.Equal(t, StreamEventToolArgsDelta, delta.Kind)
		assert.Contains(t, fmt.Sprintf("%v", delta.Payload["delta"]), "pwd")

		done, ok := proxyCompatibilityAdapters.Stream.BuildToolArgsDoneEvent(state, "resp_1")
		require.True(t, ok)
		assert.Equal(t, StreamEventToolArgsDone, done.Kind)
		assert.Contains(t, fmt.Sprintf("%v", done.Payload["arguments"]), "pwd")

		outputDone, ok := proxyCompatibilityAdapters.Stream.BuildToolItemDoneEvent(state, "resp_1")
		require.True(t, ok)
		assert.Equal(t, StreamEventToolOutputReturned, outputDone.Kind)
		outItem := outputDone.Payload["item"].(map[string]any)
		assert.Equal(t, "function_call", outItem["type"])
		assert.Equal(t, "shell", outItem["name"])
	})

	t.Run("suppresses mcp argument events but keeps lifecycle events", func(t *testing.T) {
		state := &StreamToolCallState{
			Index:       1,
			OutputIndex: 3,
			ItemID:      "fc_2",
			CallID:      "call_2",
			Name:        "mcp__playwright__browser_navigate",
			Exposed:     true,
		}
		state.ArgsBuilder.WriteString(`{"url":"https://example.com"}`)

		assert.True(t, proxyCompatibilityAdapters.Stream.ShouldExposeToolCall(state))

		added, ok := proxyCompatibilityAdapters.Stream.BuildToolItemAddedEvent(state, "resp_1")
		require.True(t, ok)
		item := added.Payload["item"].(map[string]any)
		assert.Equal(t, "mcp__playwright__browser_navigate", item["name"])

		_, ok = proxyCompatibilityAdapters.Stream.BuildToolArgsDeltaEvent(state, "resp_1")
		assert.False(t, ok)
		_, ok = proxyCompatibilityAdapters.Stream.BuildToolArgsDoneEvent(state, "resp_1")
		assert.False(t, ok)

		outputDone, ok := proxyCompatibilityAdapters.Stream.BuildToolItemDoneEvent(state, "resp_1")
		require.True(t, ok)
		outItem := outputDone.Payload["item"].(map[string]any)
		assert.Equal(t, "mcp__playwright__browser_navigate", outItem["name"])
	})
}
