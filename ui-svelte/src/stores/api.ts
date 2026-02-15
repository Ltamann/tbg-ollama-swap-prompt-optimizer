import { writable } from "svelte/store";
import type { Model, Metrics, VersionInfo, LogData, APIEventEnvelope, ReqRespCapture } from "../lib/types";
import { connectionState } from "./theme";

const LOG_LENGTH_LIMIT = 1024 * 100; /* 100KB of log data */

// Stores
export const models = writable<Model[]>([]);
export const proxyLogs = writable<string>("");
export const upstreamLogs = writable<string>("");
export const metrics = writable<Metrics[]>([]);
export const versionInfo = writable<VersionInfo>({
  build_date: "unknown",
  commit: "unknown",
  version: "unknown",
});

let apiEventSource: EventSource | null = null;

function appendLog(newData: string, store: typeof proxyLogs | typeof upstreamLogs): void {
  store.update((prev) => {
    const updatedLog = prev + newData;
    return updatedLog.length > LOG_LENGTH_LIMIT ? updatedLog.slice(-LOG_LENGTH_LIMIT) : updatedLog;
  });
}

export function enableAPIEvents(enabled: boolean): void {
  if (!enabled) {
    apiEventSource?.close();
    apiEventSource = null;
    metrics.set([]);
    return;
  }

  let retryCount = 0;
  const initialDelay = 1000; // 1 second

  const connect = () => {
    apiEventSource?.close();
    apiEventSource = new EventSource("/api/events");

    connectionState.set("connecting");

    apiEventSource.onopen = () => {
      // Clear everything on connect to keep things in sync
      proxyLogs.set("");
      upstreamLogs.set("");
      metrics.set([]);
      models.set([]);
      retryCount = 0;
      connectionState.set("connected");
    };

    apiEventSource.onmessage = (e: MessageEvent) => {
      try {
        const message = JSON.parse(e.data) as APIEventEnvelope;
        switch (message.type) {
          case "modelStatus": {
            const newModels = JSON.parse(message.data) as Model[];
            // Sort models by name and id
            newModels.sort((a, b) => {
              return (a.name + a.id).localeCompare(b.name + b.id);
            });
            models.set(newModels);
            break;
          }

          case "logData": {
            const logData = JSON.parse(message.data) as LogData;
            switch (logData.source) {
              case "proxy":
                appendLog(logData.data, proxyLogs);
                break;
              case "upstream":
                appendLog(logData.data, upstreamLogs);
                break;
            }
            break;
          }

          case "metrics": {
            const newMetrics = JSON.parse(message.data) as Metrics[];
            metrics.update((prevMetrics) => [...newMetrics, ...prevMetrics]);
            break;
          }
        }
      } catch (err) {
        console.error(e.data, err);
      }
    };

    apiEventSource.onerror = () => {
      apiEventSource?.close();
      retryCount++;
      const delay = Math.min(initialDelay * Math.pow(2, retryCount - 1), 5000);
      connectionState.set("disconnected");
      setTimeout(connect, delay);
    };
  };

  connect();
}

// Fetch version info when connected
connectionState.subscribe(async (status) => {
  if (status === "connected") {
    try {
      const response = await fetch("/api/version");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: VersionInfo = await response.json();
      versionInfo.set(data);
    } catch (error) {
      console.error(error);
    }
  }
});

export async function listModels(): Promise<Model[]> {
  try {
    const response = await fetch("/api/models/");
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data || [];
  } catch (error) {
    console.error("Failed to fetch models:", error);
    return [];
  }
}

export async function unloadAllModels(): Promise<void> {
  try {
    const response = await fetch(`/api/models/unload`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to unload models: ${response.status}`);
    }
  } catch (error) {
    console.error("Failed to unload models:", error);
    throw error;
  }
}

export async function unloadSingleModel(model: string): Promise<void> {
  try {
    const response = await fetch(`/api/models/unload/${model}`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to unload model: ${response.status}`);
    }
  } catch (error) {
    console.error("Failed to unload model", model, error);
    throw error;
  }
}

export async function loadModel(model: string): Promise<void> {
  try {
    const response = await fetch(`/upstream/${model}/`, {
      method: "GET",
    });
    if (!response.ok) {
      throw new Error(`Failed to load model: ${response.status}`);
    }
  } catch (error) {
    console.error("Failed to load model:", error);
    throw error;
  }
}

export async function getModelCtxSize(model: string): Promise<number> {
  try {
    const response = await fetch(`/api/model/${encodeURIComponent(model)}/ctxsize`);
    if (!response.ok) {
      throw new Error(`Failed to fetch ctx size for ${model}: ${response.status}`);
    }
    const data = (await response.json()) as { ctxSize?: number };
    return typeof data.ctxSize === "number" && data.ctxSize > 0 ? data.ctxSize : 0;
  } catch (error) {
    console.error("Failed to fetch model ctx size:", model, error);
    return 0;
  }
}

export async function setModelCtxSize(model: string, ctxSize: number): Promise<void> {
  const value = Math.floor(ctxSize);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("ctxSize must be a positive integer");
  }

  const response = await fetch(`/api/model/${encodeURIComponent(model)}/ctxsize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ctxSize: value }),
  });

  if (!response.ok) {
    throw new Error(`Failed to set ctx size for ${model}: ${response.status}`);
  }
}

export async function getModelFitMode(model: string): Promise<boolean> {
  try {
    const response = await fetch(`/api/model/${encodeURIComponent(model)}/fit`);
    if (!response.ok) {
      throw new Error(`Failed to fetch fit mode for ${model}: ${response.status}`);
    }
    const data = (await response.json()) as { fit?: boolean };
    return data.fit === true;
  } catch (error) {
    console.error("Failed to fetch fit mode:", model, error);
    return false;
  }
}

export async function setModelFitMode(model: string, fit: boolean): Promise<void> {
  const response = await fetch(`/api/model/${encodeURIComponent(model)}/fit`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ fit }),
  });

  if (!response.ok) {
    throw new Error(`Failed to set fit mode for ${model}: ${response.status}`);
  }
}

export type PromptOptimizationPolicy = "off" | "limit_only" | "always" | "llm_assisted";
export interface PromptOptimizationSnapshot {
  model: string;
  policy: PromptOptimizationPolicy | "llm_assisted";
  applied: boolean;
  updatedAt: string;
  note: string;
  originalBody: string;
  optimizedBody: string;
}

export async function getPromptOptimizationPolicy(model: string): Promise<PromptOptimizationPolicy> {
  try {
    const response = await fetch(`/api/model/${encodeURIComponent(model)}/prompt-optimization`);
    if (!response.ok) {
      throw new Error(`Failed to fetch prompt optimization policy for ${model}: ${response.status}`);
    }
    const data = (await response.json()) as { policy?: PromptOptimizationPolicy };
    if (data.policy === "off" || data.policy === "always" || data.policy === "limit_only" || data.policy === "llm_assisted") {
      return data.policy;
    }
    return "limit_only";
  } catch (error) {
    console.error("Failed to fetch prompt optimization policy:", model, error);
    return "limit_only";
  }
}

export async function setPromptOptimizationPolicy(model: string, policy: PromptOptimizationPolicy): Promise<void> {
  const response = await fetch(`/api/model/${encodeURIComponent(model)}/prompt-optimization`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ policy }),
  });

  if (!response.ok) {
    throw new Error(`Failed to set prompt optimization policy for ${model}: ${response.status}`);
  }
}

export async function getConfigPath(): Promise<string> {
  try {
    const response = await fetch("/api/config/path");
    if (!response.ok) {
      throw new Error(`Failed to fetch config path: ${response.status}`);
    }
    const data = (await response.json()) as { configPath?: string };
    return data.configPath || "config.yaml";
  } catch (error) {
    console.error("Failed to fetch config path:", error);
    return "config.yaml";
  }
}

export async function getLatestPromptOptimization(model: string): Promise<PromptOptimizationSnapshot | null> {
  try {
    const response = await fetch(`/api/model/${encodeURIComponent(model)}/prompt-optimization/latest`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch latest prompt optimization for ${model}: ${response.status}`);
    }
    return (await response.json()) as PromptOptimizationSnapshot;
  } catch (error) {
    console.error("Failed to fetch latest prompt optimization:", model, error);
    return null;
  }
}

export async function getCapture(id: number): Promise<ReqRespCapture | null> {
  try {
    const response = await fetch(`/api/captures/${id}`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch capture: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch capture:", error);
    return null;
  }
}
