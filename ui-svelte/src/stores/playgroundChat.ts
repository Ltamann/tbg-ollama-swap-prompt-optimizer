import { get, writable } from "svelte/store";
import type { ChatMessage, ContentPart } from "../lib/types";
import { streamChatCompletion } from "../lib/chatApi";
import { persistentStore } from "./persistent";

export const chatMessagesStore = persistentStore<ChatMessage[]>("playground-chat-messages", []);
export const chatIsStreamingStore = writable(false);
export const chatIsReasoningStore = writable(false);

let reasoningStartTime = 0;
let abortController: AbortController | null = null;

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
  temperature: number
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

    const stream = streamChatCompletion(model, apiMessages, abortController.signal, { temperature });

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
  temperature: number
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

  await regenerateFromIndex(userIndex, model, systemPrompt, temperature);
  return true;
}

export async function editUserMessage(
  idx: number,
  newContent: string,
  model: string,
  systemPrompt: string,
  temperature: number
): Promise<void> {
  if (get(chatIsStreamingStore) || !model) return;

  const messages = get(chatMessagesStore).map((msg, i) => (i === idx ? { ...msg, content: newContent } : msg));
  chatMessagesStore.set(messages);
  await regenerateFromIndex(idx, model, systemPrompt, temperature);
}
