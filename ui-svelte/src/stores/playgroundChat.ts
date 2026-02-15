import { get, writable } from "svelte/store";
import type { ChatMessage, ContentPart, ChatSource } from "../lib/types";
import { streamChatCompletion } from "../lib/chatApi";
import { persistentStore } from "./persistent";

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

export interface UploadedAttachment {
  id: string;
  name: string;
  mimeType: string;
  size: number;
  kind: "image" | "file";
  dataUrl?: string;
  textContent?: string;
}

let reasoningStartTime = 0;
let abortController: AbortController | null = null;

function mergeSources(existing: ChatSource[] | undefined, incoming: ChatSource[] | undefined): ChatSource[] | undefined {
  if (!incoming || incoming.length === 0) {
    return existing;
  }
  const map = new Map<string, ChatSource>();
  for (const src of existing || []) {
    if (src.url) map.set(src.url, src);
  }
  for (const src of incoming) {
    if (!src.url) continue;
    map.set(src.url, { ...map.get(src.url), ...src });
  }
  return Array.from(map.values());
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
    const apiMessages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    apiMessages.push(...messages.slice(0, -1));

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
      if (chunk.sources && chunk.sources.length > 0) {
        updateLastAssistant((msg) => ({
          ...msg,
          sources: mergeSources(msg.sources, chunk.sources),
        }));
      }
    }
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
	attachments: UploadedAttachment[],
	model: string,
	systemPrompt: string,
	settings: SamplingSettings
): Promise<boolean> {
  const trimmedInput = userInput.trim();
  if ((!trimmedInput && attachments.length === 0) || !model || get(chatIsStreamingStore)) {
    return false;
  }

  let content: string | ContentPart[];
  if (attachments.length > 0) {
    const parts: ContentPart[] = [];
    if (trimmedInput) {
      parts.push({ type: "text", text: trimmedInput });
    }
    for (const attachment of attachments) {
      if (attachment.kind === "image" && attachment.dataUrl) {
        parts.push({ type: "image_url", image_url: { url: attachment.dataUrl } });
        continue;
      }
      const sizeKb = Math.max(1, Math.round(attachment.size / 1024));
      const header = `[Attached file: ${attachment.name} (${sizeKb} KB)]`;
      const textBody = (attachment.textContent || "").trim();
      const block = textBody ? `${header}\n${textBody}` : `${header}\n(binary or unsupported text extraction)`;
      parts.push({ type: "text", text: block });
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

export async function deleteUserMessageWithReply(
  idx: number,
  model: string,
  systemPrompt: string,
  settings: SamplingSettings
): Promise<void> {
  if (get(chatIsStreamingStore)) return;

  const current = get(chatMessagesStore);
  if (idx < 0 || idx >= current.length) return;
  if (current[idx].role !== "user") return;

  let end = idx;
  if (idx+1 < current.length && current[idx + 1].role === "assistant") {
    end = idx + 1;
  }

  const next = [...current.slice(0, idx), ...current.slice(end + 1)];
  chatMessagesStore.set(next);

  // If there is still a user message before the removed segment, regenerate
  // the next assistant turn to keep conversation continuity.
  let previousUserIdx = -1;
  for (let i = idx - 1; i >= 0; i--) {
    if (next[i]?.role === "user") {
      previousUserIdx = i;
      break;
    }
  }

  if (previousUserIdx >= 0 && model) {
    await regenerateFromIndex(previousUserIdx, model, systemPrompt, settings);
  }
}
