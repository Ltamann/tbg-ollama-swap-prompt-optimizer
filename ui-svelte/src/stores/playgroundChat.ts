import { get, writable } from "svelte/store";
import type { ChatMessage, ContentPart, ChatSource } from "../lib/types";
import { ChatAPIError, streamChatCompletion, type ChatRequestHeaders } from "../lib/chatApi";
import { persistentStore } from "./persistent";
import { contextSize } from "./theme";
import { models, getModelCtxSize as fetchModelCtxSize, getLatestPromptOptimization } from "./api";

export const chatMessagesStore = persistentStore<ChatMessage[]>("playground-chat-messages", []);
export const chatIsStreamingStore = writable(false);
export const chatIsReasoningStore = writable(false);

export interface ToolApprovalCall {
  name: string;
  call_id?: string;
  args?: Record<string, any>;
}

export interface PendingToolApproval {
  headerName: string;
  toolCalls: ToolApprovalCall[];
  userIndex: number;
  model: string;
  systemPrompt: string;
  settings: SamplingSettings;
}

export const pendingToolApprovalStore = writable<PendingToolApproval | null>(null);

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
let generationStartTime = 0;
let liveGenerationTokens = 0;

function estimateStreamTokens(chunkText: string): number {
  if (!chunkText || !chunkText.trim()) return 0;
  // Streaming chunks are typically token-like fragments, so count one per chunk.
  return 1;
}

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

function applyReasoningFallbackIfNeeded(): void {
  const messages = get(chatMessagesStore);
  if (messages.length === 0) return;
  const lastIndex = messages.length - 1;
  const last = messages[lastIndex];
  if (!last || last.role !== "assistant") return;

  const content = typeof last.content === "string" ? last.content.trim() : "";
  const reasoning = (last.reasoning_content || "").trim();
  if (content === "" && reasoning !== "") {
    chatMessagesStore.set(
      messages.map((msg, i) =>
        i === lastIndex
          ? {
              ...msg,
              content: reasoning,
            }
          : msg
      )
    );
  }
}

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

function getModelCtxSize(modelId: string): number {
  const match = get(models).find((m) => m.id === modelId);
  return match?.ctxConfigured || match?.ctxReference || 0;
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
    // Keep fallback for non-JSON payloads.
  }

  return fallback;
}

async function ensureModelCtx(modelId: string): Promise<number> {
  const fromStore = getModelCtxSize(modelId);
  if (fromStore > 0) {
    return fromStore;
  }
  const fromApi = await fetchModelCtxSize(modelId);
  if (fromApi > 0) {
    contextSize.update((current) =>
      current.modelId === modelId
        ? { ...current, modelCtx: fromApi }
        : current
    );
  }
  return fromApi;
}

async function syncPromptOptimizationSnapshot(
  modelId: string,
  requestStartedAt: number,
  fallbackInputCtx: number
): Promise<boolean> {
  const snapshot = await getLatestPromptOptimization(modelId);
  if (!snapshot) {
    return false;
  }
  const updatedAt = Date.parse(snapshot.updatedAt || "");
  if (Number.isFinite(updatedAt) && updatedAt + 500 < requestStartedAt) {
    return false;
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
  return optimizedCtx > 0;
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
  generationStartTime = 0;
  liveGenerationTokens = 0;
}

export async function regenerateFromIndex(
	idx: number,
	model: string,
	systemPrompt: string,
	settings: SamplingSettings,
  requestHeaders?: ChatRequestHeaders
): Promise<void> {
  if (!model || get(chatIsStreamingStore)) return;

  const current = get(chatMessagesStore);
  let messages = current.slice(0, idx + 1);
  messages = [...messages, { role: "assistant", content: "", generationTokens: 0, generationTokensPerSecond: 0 }];
  chatMessagesStore.set(messages);

  chatIsStreamingStore.set(true);
  chatIsReasoningStore.set(false);
  reasoningStartTime = 0;
  generationStartTime = 0;
  liveGenerationTokens = 0;
  abortController = new AbortController();

  try {
    const requestStartedAt = Date.now();
    let lastSnapshotSyncAt = 0;
    const apiMessages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    apiMessages.push(...messages.slice(0, -1));

	const stream = streamChatCompletion(model, apiMessages, abortController.signal, settings, {
      interactiveApproval: requestHeaders?.interactiveApproval ?? false,
      approval: requestHeaders?.approval ?? false,
      approvalHeaderName: requestHeaders?.approvalHeaderName,
    });
    const estimatedInputCtx = estimateContextTokensFromMessages(apiMessages);
    const modelCtx = getModelCtxSize(model);
    contextSize.set({
      modelId: model,
      modelCtx,
      inputCtx: estimatedInputCtx,
      optimizedCtx: 0,
    });
    void ensureModelCtx(model);
    void syncPromptOptimizationSnapshot(model, requestStartedAt, estimatedInputCtx).then((hasOptimized) => {
      if (hasOptimized) {
        lastSnapshotSyncAt = Date.now();
      }
    });

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

        if (generationStartTime === 0) {
          generationStartTime = Date.now();
        }
        liveGenerationTokens += estimateStreamTokens(chunk.content);
        const liveElapsedMs = Math.max(1, Date.now() - generationStartTime);
        const liveGenerationTokensPerSecond = liveGenerationTokens / (liveElapsedMs / 1000);

        updateLastAssistant((msg) => ({
          ...msg,
          content: (typeof msg.content === "string" ? msg.content : "") + chunk.content,
          generationTokens: liveGenerationTokens,
          generationTokensPerSecond: liveGenerationTokensPerSecond,
          totalDurationMs: liveElapsedMs,
        }));
      }
      if (chunk.sources && chunk.sources.length > 0) {
        updateLastAssistant((msg) => ({
          ...msg,
          sources: mergeSources(msg.sources, chunk.sources),
        }));
      }
      if (chunk.usage || chunk.timings) {
        const promptTokens =
          chunk.timings?.prompt_n ??
          chunk.usage?.prompt_tokens ??
          chunk.usage?.input_tokens ??
          0;
        const outputTokens =
          chunk.timings?.predicted_n ??
          chunk.usage?.completion_tokens ??
          chunk.usage?.output_tokens ??
          liveGenerationTokens;
        const promptTokensPerSecond = chunk.timings?.prompt_per_second ?? 0;
        const generationTokensPerSecond = chunk.timings?.predicted_per_second ?? 0;
        const totalDurationMs = Math.round((chunk.timings?.prompt_ms ?? 0) + (chunk.timings?.predicted_ms ?? 0));

        updateLastAssistant((msg) => ({
          ...msg,
          promptTokens: promptTokens > 0 ? promptTokens : msg.promptTokens,
          promptTokensPerSecond: promptTokensPerSecond > 0 ? promptTokensPerSecond : msg.promptTokensPerSecond,
          generationTokens: outputTokens > 0 ? outputTokens : (msg.generationTokens ?? 0),
          generationTokensPerSecond: generationTokensPerSecond > 0 ? generationTokensPerSecond : (msg.generationTokensPerSecond ?? 0),
          totalDurationMs: totalDurationMs > 0 ? totalDurationMs : msg.totalDurationMs,
        }));

        if (promptTokens > 0) {
          contextSize.update((current) => ({
            modelId: model,
            modelCtx: current.modelId === model && current.modelCtx > 0 ? current.modelCtx : modelCtx,
            inputCtx: current.modelId === model ? current.inputCtx : estimatedInputCtx,
            optimizedCtx: promptTokens,
          }));
        } else {
          const now = Date.now();
          if (now - lastSnapshotSyncAt > 300) {
            lastSnapshotSyncAt = now;
            void syncPromptOptimizationSnapshot(model, requestStartedAt, estimatedInputCtx);
          }
        }
      }
    }
    void syncPromptOptimizationSnapshot(model, requestStartedAt, estimatedInputCtx);
    applyReasoningFallbackIfNeeded();
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      if (get(chatIsReasoningStore) && reasoningStartTime > 0) {
        const reasoningTimeMs = Date.now() - reasoningStartTime;
        updateLastAssistant((msg) => ({ ...msg, reasoningTimeMs }));
      }
    } else if (error instanceof ChatAPIError && error.status === 409 && error.jsonBody?.error?.code === "tool_approval_required") {
      const payload = error.jsonBody?.error || {};
      const headerName = String(payload.header_name || "X-LlamaSwap-Tool-Approval");
      const toolCalls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
      pendingToolApprovalStore.set({
        headerName,
        toolCalls,
        userIndex: idx,
        model,
        systemPrompt,
        settings: { ...settings },
      });
      updateLastAssistant((msg) => ({
        ...msg,
        content: "Tool execution requested. Please approve or deny in the dialog.",
      }));
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
    generationStartTime = 0;
    liveGenerationTokens = 0;
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

	// Fire-and-forget so UI can clear input immediately after submit.
	void regenerateFromIndex(userIndex, model, systemPrompt, settings);
  return true;
}

export async function approvePendingToolExecution(): Promise<void> {
  const pending = get(pendingToolApprovalStore);
  if (!pending || get(chatIsStreamingStore)) return;
  pendingToolApprovalStore.set(null);
  await regenerateFromIndex(
    pending.userIndex,
    pending.model,
    pending.systemPrompt,
    pending.settings,
    { approval: true, interactiveApproval: true, approvalHeaderName: pending.headerName }
  );
}

export function denyPendingToolExecution(reason?: string): void {
  const pending = get(pendingToolApprovalStore);
  if (!pending) return;
  pendingToolApprovalStore.set(null);
  const messages = get(chatMessagesStore);
  if (messages.length === 0) return;
  const denial = reason?.trim() ? `Tool execution denied by user. ${reason.trim()}` : "Tool execution denied by user.";
  const lastIndex = messages.length - 1;
  chatMessagesStore.set(messages.map((m, i) => (
    i === lastIndex && m.role === "assistant"
      ? { ...m, content: denial }
      : m
  )));
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
