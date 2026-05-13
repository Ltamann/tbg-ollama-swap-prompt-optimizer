export type ParsedResponseItem = {
  type: string;
  name?: string;
  callId?: string;
  role?: string;
  contentText?: string;
  argumentsText?: string;
  outputText?: string;
  webQuery?: string;
  webUrl?: string;
};

export type ParsedFileChange = {
  action: "created" | "edited" | "deleted";
  path: string;
};

export type CaptureSummary = {
  totalToolCalls: number;
  toolNames: string[];
  webSearchCount: number;
  webQueries: string[];
  fileChanges: ParsedFileChange[];
  finalAssistantText: string;
  responseItems: ParsedResponseItem[];
};

function tryParseJson(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const record = part as Record<string, unknown>;
    if ((record.type === "output_text" || record.type === "text" || record.type === "input_text") && typeof record.text === "string") {
      parts.push(record.text);
    }
  }
  return parts.join("\n").trim();
}

function parseOutputEnvelope(rawBody: string): ParsedResponseItem[] {
  const parsed = tryParseJson(rawBody);
  if (!parsed || typeof parsed !== "object") return [];
  const record = parsed as Record<string, unknown>;
  const output = record.output;
  if (!Array.isArray(output)) return [];
  const items: ParsedResponseItem[] = [];
  for (const entry of output) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as Record<string, unknown>;
    const type = typeof item.type === "string" ? item.type : "";
    if (!type) continue;
    const parsedItem: ParsedResponseItem = {
      type,
      name: typeof item.name === "string" ? item.name : undefined,
      callId: typeof item.call_id === "string" ? item.call_id : undefined,
      role: typeof item.role === "string" ? item.role : undefined,
    };
    if (type === "message") {
      parsedItem.contentText = textFromContent(item.content);
    }
    if (type === "function_call") {
      parsedItem.argumentsText = typeof item.arguments === "string" ? item.arguments : "";
    }
    if (type === "function_call_output" || type.endsWith("_call_output")) {
      if (typeof item.output === "string") {
        parsedItem.outputText = item.output;
      } else if (item.output !== undefined) {
        parsedItem.outputText = JSON.stringify(item.output);
      }
    }
    if (type === "web_search_call") {
      const action = item.action;
      if (action && typeof action === "object") {
        const actionRecord = action as Record<string, unknown>;
        if (typeof actionRecord.query === "string") parsedItem.webQuery = actionRecord.query;
        if (typeof actionRecord.url === "string") parsedItem.webUrl = actionRecord.url;
      }
    }
    items.push(parsedItem);
  }
  return items;
}

function parseSSEItems(rawBody: string): ParsedResponseItem[] {
  const items: ParsedResponseItem[] = [];
  const contentParts: string[] = [];
  const reasoningParts: string[] = [];
  for (const line of rawBody.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data: ")) continue;
    const data = trimmed.slice(6);
    if (!data || data === "[DONE]") continue;
    const parsed = tryParseJson(data);
    if (!parsed || typeof parsed !== "object") continue;
    const record = parsed as Record<string, unknown>;
    const eventType = typeof record.type === "string" ? record.type : "";
    if (eventType === "response.output_text.delta" && typeof record.delta === "string") {
      contentParts.push(record.delta);
      continue;
    }
    if (eventType === "response.reasoning_summary_text.delta" && typeof record.delta === "string") {
      reasoningParts.push(record.delta);
      continue;
    }
    if (eventType === "response.function_call_arguments.done") {
      items.push({
        type: "function_call",
        name: typeof record.name === "string" ? record.name : "",
        callId: typeof record.call_id === "string" ? record.call_id : "",
        argumentsText: typeof record.arguments === "string" ? record.arguments : "",
      });
      continue;
    }
    const delta = (record.choices && Array.isArray(record.choices) ? record.choices[0] : null) as Record<string, unknown> | null;
    const deltaContent = delta?.delta && typeof delta.delta === "object" ? delta.delta as Record<string, unknown> : null;
    if (deltaContent?.content && typeof deltaContent.content === "string") {
      contentParts.push(deltaContent.content);
    }
    if (deltaContent?.reasoning_content && typeof deltaContent.reasoning_content === "string") {
      reasoningParts.push(deltaContent.reasoning_content);
    }
  }
  if (reasoningParts.length > 0) {
    items.unshift({ type: "reasoning", contentText: reasoningParts.join("") });
  }
  if (contentParts.length > 0) {
    items.push({ type: "message", role: "assistant", contentText: contentParts.join("") });
  }
  return items;
}

function parseFileChangesFromOutput(text: string): ParsedFileChange[] {
  const changes: ParsedFileChange[] = [];
  const lines = text.split(/\r?\n/);
  let captureChanges = false;
  for (const line of lines) {
    if (/updated the following files:/i.test(line)) {
      captureChanges = true;
      continue;
    }
    if (!captureChanges) continue;
    const trimmed = line.trim();
    if (!trimmed) continue;
    const match = /^([AMD])\s+(.+)$/.exec(trimmed);
    if (!match) continue;
    const action = match[1] === "A" ? "created" : match[1] === "D" ? "deleted" : "edited";
    changes.push({ action, path: match[2] });
  }
  return changes;
}

export function summarizeCaptureResponse(rawBody: string): CaptureSummary {
  const responseItems = parseOutputEnvelope(rawBody);
  const items = responseItems.length > 0 ? responseItems : parseSSEItems(rawBody);
  const toolNames: string[] = [];
  const webQueries: string[] = [];
  const fileChanges: ParsedFileChange[] = [];
  let finalAssistantText = "";

  for (const item of items) {
    if (item.type === "function_call" && item.name) {
      toolNames.push(item.name);
    }
    if (item.type === "web_search_call") {
      toolNames.push("web_search");
      if (item.webQuery) webQueries.push(item.webQuery);
    }
    if (item.outputText) {
      fileChanges.push(...parseFileChangesFromOutput(item.outputText));
    }
    if (item.type === "message" && item.role === "assistant" && item.contentText) {
      finalAssistantText = item.contentText;
    }
  }

  return {
    totalToolCalls: toolNames.length,
    toolNames,
    webSearchCount: toolNames.filter((name) => name === "web_search").length,
    webQueries,
    fileChanges,
    finalAssistantText,
    responseItems: items,
  };
}
