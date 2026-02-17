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

export class ChatAPIError extends Error {
  status: number;
  body: string;
  jsonBody: any;

  constructor(status: number, body: string, jsonBody: any = null) {
    super(`Chat API error: ${status} - ${body}`);
    this.name = "ChatAPIError";
    this.status = status;
    this.body = body;
    this.jsonBody = jsonBody;
  }
}

export interface ChatRequestHeaders {
  approval?: boolean;
  interactiveApproval?: boolean;
  approvalHeaderName?: string;
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
  options?: ChatOptions,
  requestHeaders?: ChatRequestHeaders
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

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (requestHeaders?.interactiveApproval) {
    headers["X-LlamaSwap-Tool-Approval-Interactive"] = "true";
  }
  if (requestHeaders?.approval) {
    const headerName = (requestHeaders.approvalHeaderName || "X-LlamaSwap-Tool-Approval").trim() || "X-LlamaSwap-Tool-Approval";
    headers[headerName] = "true";
  }

  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers,
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let parsed: any = null;
    try {
      parsed = JSON.parse(errorText);
    } catch {
      parsed = null;
    }
    throw new ChatAPIError(response.status, errorText, parsed);
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
