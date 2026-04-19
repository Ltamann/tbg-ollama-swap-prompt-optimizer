import { get, writable } from "svelte/store";
import type { ChatMessage, ContentPart } from "../lib/types";
import { streamChatCompletion } from "../lib/chatApi";
import { persistentStore } from "./persistent";
import { contextSize } from "./theme";
import { models, getLatestPromptOptimization } from "./api";

export const chatMessagesStore = persistentStore<ChatMessage[]>("playground-chat-messages", []);
export const chatIsStreamingStore = writable(false);
export const chatIsReasoningStore = writable(false);

export interface SamplingSettings {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  presence_penalty: number;
  frequency_penalty: number;
  max_tokens?: number;
}

let reasoningStartTime = 0;
let abortController: AbortController | null = null;

function estimateContextTokensFromMessages(messages: ChatMessage[]): number {
  let text = "";
  for (const message of messages) {
    const content = message.content;
    if (typeof content === "string") {
      text += content;
      continue;
    }
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part.type === "text" && typeof part.text === "string") {
          text += part.text;
        }
      }
    }
  }
  return Math.max(0, Math.round(text.length / 4));
}

function estimateContextTokensFromRawBody(rawBody: string): number {
  const fallback = Math.max(0, Math.round((rawBody || "").length / 4));
  if (!rawBody || !rawBody.trim()) {
    return 0;
  }

  try {
    const parsed = JSON.parse(rawBody) as { messages?: Array<{ content?: string | Array<{ type?: string; text?: string }> }> };
    let text = "";
    const messages = Array.isArray(parsed?.messages) ? parsed.messages : [];
    for (const message of messages) {
      const content = message?.content;
      if (typeof content === "string") {
        text += content;
        continue;
      }
      if (Array.isArray(content)) {
        for (const part of content) {
          if (part?.type === "text" && typeof part?.text === "string") {
            text += part.text;
          }
        }
      }
    }
    if (text.length > 0) {
      return Math.max(0, Math.round(text.length / 4));
    }
  } catch {
    return fallback;
  }

  return fallback;
}

function getModelCtxSize(modelId: string): number {
  const match = get(models).find((m) => m.id === modelId);
  return match?.ctxConfigured || match?.ctxReference || 0;
}

async function syncPromptOptimizationSnapshot(
  modelId: string,
  requestStartedAt: number,
  fallbackInputCtx: number
): Promise<void> {
  const snapshot = await getLatestPromptOptimization(modelId);
  if (!snapshot) {
    return;
  }
  const updatedAt = Date.parse(snapshot.updatedAt || "");
  if (Number.isFinite(updatedAt) && updatedAt + 500 < requestStartedAt) {
    return;
  }

  const inputCtx = estimateContextTokensFromRawBody(snapshot.originalBody);
  const optimizedCtx = estimateContextTokensFromRawBody(snapshot.optimizedBody);
  contextSize.update((current) => {
    if (current.modelId !== modelId) {
      return current;
    }
    return {
      ...current,
      inputCtx: inputCtx > 0 ? inputCtx : fallbackInputCtx,
      optimizedCtx: optimizedCtx > 0 ? optimizedCtx : current.optimizedCtx,
    };
  });
}

function updateLastAssistant(mutator: (msg: ChatMessage) => ChatMessage): void {
  const messages = get(chatMessagesStore);
  if (messages.length === 0) return;
  const lastIndex = messages.length - 1;
  chatMessagesStore.set(messages.map((msg, i) => (i === lastIndex ? mutator(msg) : msg)));
}

export function cancelChatStreaming(): void {
  abortController?.abort();
}

export function newChatSession(): void {
  if (get(chatIsStreamingStore)) {
    cancelChatStreaming();
  }
  chatMessagesStore.set([]);
  contextSize.update((current) => ({
    ...current,
    inputCtx: 0,
    optimizedCtx: 0,
  }));
  chatIsReasoningStore.set(false);
  reasoningStartTime = 0;
}

export async function regenerateFromIndex(
	idx: number,
	model: string,
	systemPrompt: string,
	settings: SamplingSettings
): Promise<void> {
  if (!model || get(chatIsStreamingStore)) return;

  const current = get(chatMessagesStore);
  let messages = current.slice(0, idx + 1);
  messages = [...messages, { role: "assistant", content: "" }];
  chatMessagesStore.set(messages);

  chatIsStreamingStore.set(true);
  chatIsReasoningStore.set(false);
  reasoningStartTime = 0;
  abortController = new AbortController();

  try {
    const requestStartedAt = Date.now();
    const apiMessages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    apiMessages.push(...messages.slice(0, -1));

    const estimatedInputCtx = estimateContextTokensFromMessages(apiMessages);
    contextSize.set({
      modelId: model,
      modelCtx: getModelCtxSize(model),
      inputCtx: estimatedInputCtx,
      optimizedCtx: 0,
    });
    void syncPromptOptimizationSnapshot(model, requestStartedAt, estimatedInputCtx);

	const stream = streamChatCompletion(model, apiMessages, abortController.signal, settings);

    for await (const chunk of stream) {
      if (chunk.done) break;

      if (chunk.reasoning_content) {
        if (!get(chatIsReasoningStore)) {
          chatIsReasoningStore.set(true);
          reasoningStartTime = Date.now();
        }

        updateLastAssistant((msg) => ({
          ...msg,
          reasoning_content: (msg.reasoning_content || "") + chunk.reasoning_content,
        }));
      }

      if (chunk.content) {
        if (get(chatIsReasoningStore)) {
          const reasoningTimeMs = Date.now() - reasoningStartTime;
          chatIsReasoningStore.set(false);
          updateLastAssistant((msg) => ({ ...msg, reasoningTimeMs }));
        }

        updateLastAssistant((msg) => ({
          ...msg,
          content: (typeof msg.content === "string" ? msg.content : "") + chunk.content,
        }));
      }
    }
    void syncPromptOptimizationSnapshot(model, requestStartedAt, estimatedInputCtx);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      if (get(chatIsReasoningStore) && reasoningStartTime > 0) {
        const reasoningTimeMs = Date.now() - reasoningStartTime;
        updateLastAssistant((msg) => ({ ...msg, reasoningTimeMs }));
      }
    } else {
      const errorMessage = error instanceof Error ? error.message : "An error occurred";
      updateLastAssistant((msg) => ({
        ...msg,
        content: (typeof msg.content === "string" ? msg.content : "") + `\n\n**Error:** ${errorMessage}`,
      }));
    }
  } finally {
    chatIsStreamingStore.set(false);
    chatIsReasoningStore.set(false);
    abortController = null;
  }
}

export async function sendUserMessage(
	userInput: string,
	attachedImages: string[],
	model: string,
	systemPrompt: string,
	settings: SamplingSettings
): Promise<boolean> {
  const trimmedInput = userInput.trim();
  if ((!trimmedInput && attachedImages.length === 0) || !model || get(chatIsStreamingStore)) {
    return false;
  }

  let content: string | ContentPart[];
  if (attachedImages.length > 0) {
    const parts: ContentPart[] = [];
    if (trimmedInput) {
      parts.push({ type: "text", text: trimmedInput });
    }
    for (const url of attachedImages) {
      parts.push({ type: "image_url", image_url: { url } });
    }
    content = parts;
  } else {
    content = trimmedInput;
  }

  const messages = get(chatMessagesStore);
  const userIndex = messages.length;
  chatMessagesStore.set([...messages, { role: "user", content }]);

	await regenerateFromIndex(userIndex, model, systemPrompt, settings);
  return true;
}

export async function editUserMessage(
	idx: number,
	newContent: string,
	model: string,
	systemPrompt: string,
	settings: SamplingSettings
): Promise<void> {
  if (get(chatIsStreamingStore) || !model) return;

  const messages = get(chatMessagesStore).map((msg, i) => (i === idx ? { ...msg, content: newContent } : msg));
  chatMessagesStore.set(messages);
	await regenerateFromIndex(idx, model, systemPrompt, settings);
}
