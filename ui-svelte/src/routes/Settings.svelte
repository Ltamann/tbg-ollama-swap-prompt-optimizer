<script lang="ts">
  import { onMount } from "svelte";
  import {
    models,
    type PromptOptimizationPolicy,
    type PromptOptimizationSnapshot,
    type WebSearchSettings,
    getPromptOptimizationPolicy,
    setPromptOptimizationPolicy,
    getConfigPath,
    getLatestPromptOptimization,
    getWebSearchSettings,
    setWebSearchSettings,
  } from "../stores/api";
  import type { Model } from "../lib/types";

  type PolicyOption = {
    value: PromptOptimizationPolicy;
    label: string;
    help: string;
  };

  const options: PolicyOption[] = [
    {
      value: "off",
      label: "Prompt Optimizer: Off",
      help: "Send prompts as-is. No automatic cleanup or compression.",
    },
    {
      value: "limit_only",
      label: "Prompt Optimizer: Smart (Only Near Limit)",
      help: "Only optimize when prompt size is near or above your context limit.",
    },
    {
      value: "always",
      label: "Prompt Optimizer: Aggressive (Always)",
      help: "Always remove repeated content before sending prompts.",
    },
    {
      value: "llm_assisted",
      label: "Prompt Optimizer: LLM-Assisted",
      help: "Use the running model to summarize older context before request forwarding.",
    },
  ];

  let loading = $state<Record<string, boolean>>({});
  let loaded = $state<Record<string, boolean>>({});
  let policyByModel = $state<Record<string, PromptOptimizationPolicy>>({});
  let configPath = $state("config.yaml");
  let latestInfo = $state<Record<string, string>>({});
  let latestSnapshotByModel = $state<Record<string, PromptOptimizationSnapshot | null>>({});
  let copyStatusByModel = $state<Record<string, string>>({});
  let webSearchSettings = $state<WebSearchSettings>({
    enabled: true,
    engine: "duckduckgo_html",
    url: "",
  });
  let webSearchStatus = $state("");

  const localModels = $derived.by(() => $models.filter((m) => !m.peerID).sort((a, b) => a.id.localeCompare(b.id)));

  $effect(() => {
    for (const model of localModels) {
      void ensureLoaded(model.id);
    }
  });
  onMount(() => {
    void loadConfigPath();
    void loadWebSearchSettings();
  });

  async function ensureLoaded(modelId: string): Promise<void> {
    if (loaded[modelId] || loading[modelId]) {
      return;
    }
    loading = { ...loading, [modelId]: true };
    try {
      const policy = await getPromptOptimizationPolicy(modelId);
      policyByModel = { ...policyByModel, [modelId]: policy };
      loaded = { ...loaded, [modelId]: true };
    } finally {
      loading = { ...loading, [modelId]: false };
    }
  }

  async function onPolicyChange(model: Model, value: string): Promise<void> {
    const policy = value as PromptOptimizationPolicy;
    if (policy !== "off" && policy !== "limit_only" && policy !== "always" && policy !== "llm_assisted") {
      return;
    }
    policyByModel = { ...policyByModel, [model.id]: policy };
    try {
      await setPromptOptimizationPolicy(model.id, policy);
    } catch (error) {
      console.error(error);
    }
  }

  function getPolicyLabel(value: PromptOptimizationPolicy): string {
    return options.find((o) => o.value === value)?.label || value;
  }

  async function loadConfigPath(): Promise<void> {
    configPath = await getConfigPath();
  }

  async function loadWebSearchSettings(): Promise<void> {
    webSearchSettings = await getWebSearchSettings();
  }

  async function saveWebSearchSettings(): Promise<void> {
    try {
      webSearchSettings = await setWebSearchSettings(webSearchSettings);
      webSearchStatus = "Saved.";
    } catch (error) {
      console.error(error);
      webSearchStatus = "Save failed.";
    }
  }

  async function loadLatestInfo(modelId: string): Promise<void> {
    const latest = await getLatestPromptOptimization(modelId);
    if (!latest) {
      latestSnapshotByModel = { ...latestSnapshotByModel, [modelId]: null };
      latestInfo = { ...latestInfo, [modelId]: "No optimization snapshot yet." };
      return;
    }
    latestSnapshotByModel = { ...latestSnapshotByModel, [modelId]: latest };
    const label = `${latest.updatedAt} | ${latest.policy} | applied=${latest.applied}`;
    latestInfo = { ...latestInfo, [modelId]: label };
  }

  async function copyText(text: string): Promise<boolean> {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }

      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(textArea);
      return ok;
    } catch (error) {
      console.error("Failed to copy optimized prompt:", error);
      return false;
    }
  }

  async function copyLatestOptimizedPrompt(modelId: string): Promise<void> {
    const existing = latestSnapshotByModel[modelId];
    let snapshot = existing;
    if (!snapshot) {
      await loadLatestInfo(modelId);
      snapshot = latestSnapshotByModel[modelId];
    }

    const optimizedBody = snapshot?.optimizedBody?.trim();
    if (!optimizedBody) {
      copyStatusByModel = { ...copyStatusByModel, [modelId]: "No optimized prompt available to copy." };
      return;
    }

    const copied = await copyText(optimizedBody);
    copyStatusByModel = {
      ...copyStatusByModel,
      [modelId]: copied ? "Optimized prompt copied." : "Copy failed.",
    };
  }
</script>

<div class="card">
  <h2>Settings</h2>
  <p class="text-sm text-txtsecondary mb-2">
    Config file:
    <a href={`file://${configPath}`} title={configPath} class="underline">{configPath}</a>
  </p>
  <p class="text-sm text-txtsecondary mb-4">Control how prompts are optimized before requests are sent upstream.</p>

  <div class="mb-6 rounded border border-gray-200 dark:border-white/10 p-4">
    <h3 class="font-semibold mb-2">Local Web Search Fallback</h3>
    <p class="text-sm text-txtsecondary mb-3">
      Codex browser/MCP stays preferred. These settings control llama-swap’s local `web_search` fallback when the client cannot execute the first-party web search call.
    </p>
    <div class="flex items-center gap-2 mb-3">
      <input id="ws-enabled" type="checkbox" bind:checked={webSearchSettings.enabled} />
      <label for="ws-enabled">Enable llama-swap local web-search fallback</label>
    </div>
    <div class="grid gap-3 md:grid-cols-2">
      <div>
        <label class="block text-sm mb-1" for="ws-engine">Search Engine</label>
        <select
          id="ws-engine"
          class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
          bind:value={webSearchSettings.engine}
        >
          <option value="duckduckgo_html">DuckDuckGo HTML</option>
          <option value="searxng">SearXNG</option>
        </select>
      </div>
      <div>
        <label class="block text-sm mb-1" for="ws-url">SearXNG Endpoint</label>
        <input
          id="ws-url"
          class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
          bind:value={webSearchSettings.url}
          placeholder="http://127.0.0.1:8081/search"
        />
      </div>
    </div>
    <div class="mt-4 rounded border border-gray-200 dark:border-white/10 p-3">
      <div class="font-medium mb-2">Managed SearXNG Sidecar</div>
      <div class="flex items-center gap-2 mb-3">
        <input id="ws-managed-enabled" type="checkbox" bind:checked={webSearchSettings.managedEnabled} />
        <label for="ws-managed-enabled">Start SearXNG with llama-swap</label>
      </div>
      <div class="grid gap-3">
        <div>
          <label class="block text-sm mb-1" for="ws-managed-command">Start Command</label>
          <input
            id="ws-managed-command"
            class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
            bind:value={webSearchSettings.managedCommand}
            placeholder="docker run ... searxng or custom launcher"
          />
        </div>
        <div>
          <label class="block text-sm mb-1" for="ws-managed-stop-command">Stop Command</label>
          <input
            id="ws-managed-stop-command"
            class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
            bind:value={webSearchSettings.managedStopCommand}
            placeholder="optional custom stop command"
          />
        </div>
      </div>
      <div class="text-xs text-txtsecondary mt-2">
        Status: {webSearchSettings.managedStatus || "stopped"}. When enabled, llama-swap starts this command during server startup and stops it on restart/shutdown.
      </div>
    </div>
    <div class="text-xs text-txtsecondary mt-2">
      Use `SearXNG` with a JSON endpoint like `/search?format=json`. Leave the endpoint blank to use DuckDuckGo HTML fallback.
    </div>
    <div class="mt-3 flex items-center gap-2">
      <button class="btn btn--sm" onclick={saveWebSearchSettings}>Save Web Search Settings</button>
      <button class="btn btn--sm" onclick={loadWebSearchSettings}>Reload</button>
    </div>
    {#if webSearchStatus}
      <div class="text-xs text-txtsecondary mt-2">{webSearchStatus}</div>
    {/if}
  </div>

  <table class="w-full">
    <thead>
      <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
        <th class="py-2">Model</th>
        <th class="py-2">Prompt Optimization</th>
      </tr>
    </thead>
    <tbody>
      {#each localModels as model (model.id)}
        {@const selected = policyByModel[model.id] || "limit_only"}
        <tr class="border-b hover:bg-secondary-hover border-gray-200 dark:border-white/10">
          <td class="py-2">
            <div class="font-semibold">{model.name || model.id}</div>
            {#if model.name}
              <div class="text-xs text-txtsecondary">{model.id}</div>
            {/if}
            {#if model.external}
              <div class="text-xs text-txtsecondary">Provider: Ollama (external)</div>
            {/if}
          </td>
          <td class="py-2">
            <select
              class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
              value={selected}
              disabled={loading[model.id]}
              onchange={(e) => onPolicyChange(model, (e.currentTarget as HTMLSelectElement).value)}
              title={options.find((o) => o.value === selected)?.help || ""}
            >
              {#each options as option}
                <option value={option.value} title={option.help}>
                  {option.label}
                </option>
              {/each}
            </select>
            <div class="text-xs text-txtsecondary mt-1" title={options.find((o) => o.value === selected)?.help || ""}>
              {getPolicyLabel(selected)}
            </div>
            <div class="mt-2 flex items-center gap-2">
              <button class="btn btn--sm" onclick={() => loadLatestInfo(model.id)}>Latest Optimized Prompt Info</button>
              <button class="btn btn--sm" onclick={() => copyLatestOptimizedPrompt(model.id)}>Copy Optimized Prompt</button>
            </div>
            {#if latestInfo[model.id]}
              <div class="text-xs text-txtsecondary mt-1 break-all">{latestInfo[model.id]}</div>
            {/if}
            {#if copyStatusByModel[model.id]}
              <div class="text-xs text-txtsecondary mt-1">{copyStatusByModel[model.id]}</div>
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
