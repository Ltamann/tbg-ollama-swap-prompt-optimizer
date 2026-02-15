import type { ChatMessage, ChatCompletionRequest, ChatSource } from "./types";

export interface StreamChunk {
  content: string;
  reasoning_content?: string;
  sources?: ChatSource[];
  done: boolean;
}

export interface ChatOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  max_tokens?: number;
}

function parseSSELine(line: string): StreamChunk | null {
  const trimmed = line.trim();
  if (!trimmed || !trimmed.startsWith("data: ")) {
    return null;
  }

  const data = trimmed.slice(6);
  if (data === "[DONE]") {
    return { content: "", done: true };
  }

  try {
    const parsed = JSON.parse(data);
    const delta = parsed.choices?.[0]?.delta;
    const content = delta?.content || "";
    const reasoning_content = delta?.reasoning_content || "";
    const sources = (delta?.sources || parsed.choices?.[0]?.message?.sources || []) as ChatSource[];

    if (content || reasoning_content || (Array.isArray(sources) && sources.length > 0)) {
      return { content, reasoning_content, sources, done: false };
    }
    return null;
  } catch {
    return null;
  }
}

export async function* streamChatCompletion(
  model: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
  options?: ChatOptions
): AsyncGenerator<StreamChunk> {
  const request: ChatCompletionRequest = {
    model,
    messages,
    stream: true,
    temperature: options?.temperature,
    top_p: options?.top_p,
    top_k: options?.top_k,
    min_p: options?.min_p,
    presence_penalty: options?.presence_penalty,
    frequency_penalty: options?.frequency_penalty,
    max_tokens: options?.max_tokens,
  };

  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat API error: ${response.status} - ${errorText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Response body is not readable");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const result = parseSSELine(line);
        if (result?.done) {
          yield result;
          return;
        }
        if (result) {
          yield result;
        }
      }
    }

    // Process any remaining buffer
    const result = parseSSELine(buffer);
    if (result && !result.done) {
      yield result;
    }

    yield { content: "", done: true };
  } finally {
    reader.releaseLock();
  }
}
