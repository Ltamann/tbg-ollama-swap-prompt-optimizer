package proxy

import "testing"

func TestResponseLooksLikePlanningOnly_TodoListOnly(t *testing.T) {
	body := []byte(`{
		"output": [
			{
				"type": "todo_list",
				"items": [
					{"text":"Step 1","completed":false},
					{"text":"Step 2","completed":false}
				]
			}
		]
	}`)
	if !responseLooksLikePlanningOnly(body) {
		t.Fatalf("expected todo_list-only response to be treated as planning-only")
	}
}

func TestResponseLooksLikePlanningOnly_TodoListWithToolCall(t *testing.T) {
	body := []byte(`{
		"output": [
			{
				"type": "todo_list",
				"items": [{"text":"Step 1","completed":false}]
			},
			{
				"type": "function_call",
				"name": "shell_command",
				"call_id": "call_1",
				"arguments": "{\"command\":\"pwd\"}"
			}
		]
	}`)
	if responseLooksLikePlanningOnly(body) {
		t.Fatalf("expected tool-call response to not be treated as planning-only")
	}
}

