<script lang="ts">
  import { onMount } from "svelte";
  import {
    models,
    type PromptOptimizationPolicy,
    type PromptOptimizationSnapshot,
    getPromptOptimizationPolicy,
    setPromptOptimizationPolicy,
    getConfigPath,
    getLatestPromptOptimization,
    listTools,
    createTool,
    updateTool,
    deleteTool,
    type RuntimeTool,
    type RuntimeToolType,
    type RuntimeToolPolicy,
    type ToolRuntimeSettings,
    getToolRuntimeSettings,
    setToolRuntimeSettings,
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
  let draftToolName = $state("");
  let draftToolType = $state<RuntimeToolType>("http");
  let draftToolEndpoint = $state("");
  let draftToolDescription = $state("");
  let tools = $state<RuntimeTool[]>([]);
  let toolEdits = $state<Record<string, RuntimeTool>>({});
  let toolSettings = $state<ToolRuntimeSettings>({
    enabled: true,
    webSearchMode: "auto",
    watchdogMode: "off",
    requireApprovalHeader: false,
    approvalHeaderName: "X-LlamaSwap-Tool-Approval",
    blockNonLocalEndpoints: true,
    maxToolRounds: 4,
    killPreviousOnSwap: true,
    maxRunningModels: 1,
  });

  const localModels = $derived.by(() => $models.filter((m) => !m.peerID).sort((a, b) => a.id.localeCompare(b.id)));

  $effect(() => {
    for (const model of localModels) {
      void ensureLoaded(model.id);
    }
  });
  onMount(() => {
    void loadConfigPath();
    void loadTools();
    void loadToolSettings();
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

  async function loadTools(): Promise<void> {
    try {
      tools = await listTools();
      const nextEdits: Record<string, RuntimeTool> = {};
      for (const t of tools) {
        nextEdits[t.id] = {
          ...t,
          policy: t.policy || "auto",
          timeoutSeconds: t.timeoutSeconds || (t.type === "mcp" ? 30 : 20),
        };
      }
      toolEdits = nextEdits;
    } catch (error) {
      console.error("Failed to load tools", error);
    }
  }

  async function loadToolSettings(): Promise<void> {
    try {
      const loaded = await getToolRuntimeSettings();
      toolSettings = {
        ...loaded,
        watchdogMode: loaded.watchdogMode || "off",
      };
    } catch (error) {
      console.error("Failed to load tool runtime settings", error);
    }
  }

  async function persistToolSettings(): Promise<void> {
    try {
      toolSettings = await setToolRuntimeSettings(toolSettings);
    } catch (error) {
      console.error("Failed to save tool runtime settings", error);
      await loadToolSettings();
    }
  }

  async function addTool(): Promise<void> {
    const name = draftToolName.trim();
    const endpoint = draftToolEndpoint.trim();
    if (!name || !endpoint) {
      return;
    }

    const next: Omit<RuntimeTool, "id"> = {
      name,
      type: draftToolType,
      endpoint,
      enabled: true,
      description: draftToolDescription.trim() || undefined,
      policy: "auto",
    };
    try {
      await createTool(next);
      await loadTools();
    } catch (error) {
      console.error("Failed to create tool", error);
    }
    draftToolName = "";
    draftToolType = "http";
    draftToolEndpoint = "";
    draftToolDescription = "";
  }

  async function removeTool(id: string): Promise<void> {
    const prevTools = tools;
    const prevEdits = toolEdits;
    tools = tools.filter((t) => t.id !== id);
    const nextEdits = { ...toolEdits };
    delete nextEdits[id];
    toolEdits = nextEdits;
    try {
      await deleteTool(id);
    } catch (error) {
      console.error("Failed to delete tool", error);
      tools = prevTools;
      toolEdits = prevEdits;
    }
  }

  function setToolEdit(id: string, patch: Partial<RuntimeTool>): void {
    const existing = toolEdits[id] || tools.find((t) => t.id === id);
    if (!existing) return;
    toolEdits = {
      ...toolEdits,
      [id]: {
        ...existing,
        ...patch,
      },
    };
  }

  function resetToolEdit(id: string): void {
    const tool = tools.find((t) => t.id === id);
    if (!tool) {
      return;
    }
    setToolEdit(id, {
      ...tool,
      policy: tool.policy || "auto",
      timeoutSeconds: tool.timeoutSeconds || (tool.type === "mcp" ? 30 : 20),
    });
  }

  async function saveToolEdit(id: string): Promise<void> {
    const edit = toolEdits[id];
    if (!edit) {
      return;
    }

    const normalized: RuntimeTool = {
      ...edit,
      name: (edit.name || "").trim(),
      endpoint: (edit.endpoint || "").trim(),
      description: (edit.description || "").trim() || undefined,
      remoteName: (edit.remoteName || "").trim() || undefined,
      policy: (edit.policy || "auto") as RuntimeToolPolicy,
      timeoutSeconds: Math.max(1, Math.round(edit.timeoutSeconds || (edit.type === "mcp" ? 30 : 20))),
    };
    if (!normalized.name || !normalized.endpoint) {
      return;
    }

    try {
      const saved = await updateTool(normalized);
      tools = tools.map((t) => (t.id === id ? saved : t));
      setToolEdit(id, {
        ...saved,
        policy: saved.policy || "auto",
        timeoutSeconds: saved.timeoutSeconds || (saved.type === "mcp" ? 30 : 20),
      });
    } catch (error) {
      console.error("Failed to save tool", error);
    }
  }

  async function addPresetSearXNG(): Promise<void> {
    const exists = tools.some((t) => t.name.toLowerCase() === "searxng_web_search");
    if (exists) return;
    try {
      await createTool({
        name: "searxng_web_search",
        type: "http",
        endpoint: "http://host.docker.internal:8081/search?format=json&q={query}",
        enabled: true,
        description: "Web search via SearXNG (query placeholder: {query})",
      });
      await loadTools();
    } catch (error) {
      console.error("Failed to add searxng preset", error);
    }
  }

  async function addPresetPlaywright(): Promise<void> {
    const exists = tools.some((t) => t.name.toLowerCase() === "playwright_mcp");
    if (exists) return;
    try {
      await createTool({
        name: "playwright_mcp",
        type: "mcp",
        endpoint: "http://host.docker.internal:8931/mcp",
        enabled: true,
        description: "Playwright MCP server endpoint",
      });
      await loadTools();
    } catch (error) {
      console.error("Failed to add playwright preset", error);
    }
  }
</script>

<div class="card">
  <h2>Settings</h2>
  <p class="text-sm text-txtsecondary mb-2">
    Config file:
    <a href={`file://${configPath}`} title={configPath} class="underline">{configPath}</a>
  </p>
  <p class="text-sm text-txtsecondary mb-4">Control how prompts are optimized before requests are sent upstream.</p>

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
              class="settings-select w-full rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-sm"
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

<div class="card mt-4">
  <h2>Tools</h2>
  <p class="text-sm text-txtsecondary mb-3">
    Define tool endpoints for automatic function-calling wiring (HTTP/MCP).
  </p>

  <div class="mb-4 p-3 rounded border border-gray-200 dark:border-white/10 bg-surface">
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={toolSettings.enabled} onchange={(e) => { toolSettings = { ...toolSettings, enabled: (e.currentTarget as HTMLInputElement).checked }; void persistToolSettings(); }} />
        Tools Runtime Enabled
      </label>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={toolSettings.blockNonLocalEndpoints} onchange={(e) => { toolSettings = { ...toolSettings, blockNonLocalEndpoints: (e.currentTarget as HTMLInputElement).checked }; void persistToolSettings(); }} />
        Local Endpoints Only
      </label>
      <label class="text-sm">
        Web Search Routing
        <select class="settings-select w-full mt-1 rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-sm" bind:value={toolSettings.webSearchMode} onchange={() => void persistToolSettings()}>
          <option value="off">Off</option>
          <option value="auto">Auto</option>
          <option value="force">Force First (search-like prompts)</option>
        </select>
      </label>
      <label class="text-sm">
        Watchdog
        <select class="settings-select w-full mt-1 rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-sm" bind:value={toolSettings.watchdogMode} onchange={() => void persistToolSettings()}>
          <option value="off">off</option>
          <option value="auto">auto (global)</option>
        </select>
      </label>
      <label class="text-sm">
        Max Tool Rounds
        <input class="w-full mt-1 rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-sm" type="number" min="1" max="16" bind:value={toolSettings.maxToolRounds} onchange={() => void persistToolSettings()} />
      </label>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={toolSettings.killPreviousOnSwap} onchange={(e) => { toolSettings = { ...toolSettings, killPreviousOnSwap: (e.currentTarget as HTMLInputElement).checked }; void persistToolSettings(); }} />
        Kill Previous Model On Swap
      </label>
      <label class="text-sm">
        Max Running Models
        <input class="w-full mt-1 rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-sm" type="number" min="1" max="64" bind:value={toolSettings.maxRunningModels} onchange={() => void persistToolSettings()} />
      </label>
    </div>
  </div>

  <div class="mb-3 flex flex-wrap gap-2">
    <button class="btn btn--sm" onclick={addPresetSearXNG}>Add SearXNG Preset</button>
    <button class="btn btn--sm" onclick={addPresetPlaywright}>Add Playwright MCP Preset</button>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-4 gap-2 mb-2">
    <input
      class="rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
      placeholder="tool name (e.g. searxng_web_search)"
      bind:value={draftToolName}
    />
    <select
      class="settings-select rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-sm"
      bind:value={draftToolType}
    >
      <option value="http">http</option>
      <option value="mcp">mcp</option>
    </select>
    <input
      class="rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm md:col-span-2"
      placeholder="endpoint URL"
      bind:value={draftToolEndpoint}
    />
  </div>
  <div class="flex gap-2 mb-4">
    <input
      class="flex-1 rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
      placeholder="description (optional)"
      bind:value={draftToolDescription}
    />
    <button class="btn btn--sm" onclick={addTool}>Add Tool</button>
  </div>

  <table class="w-full">
    <thead>
      <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
        <th class="py-2">Enabled</th>
        <th class="py-2">Name</th>
        <th class="py-2">Type</th>
        <th class="py-2">Policy</th>
        <th class="py-2">Endpoint</th>
        <th class="py-2">Security</th>
        <th class="py-2">Action</th>
      </tr>
    </thead>
    <tbody>
      {#if tools.length === 0}
        <tr>
          <td class="py-3 text-sm text-txtsecondary" colspan="7">No tools configured yet.</td>
        </tr>
      {:else}
        {#each tools as tool (tool.id)}
          {@const edit = toolEdits[tool.id] || tool}
          <tr class="border-b hover:bg-secondary-hover border-gray-200 dark:border-white/10">
            <td class="py-2">
              <input
                type="checkbox"
                checked={edit.enabled}
                onchange={(e) => setToolEdit(tool.id, { enabled: (e.currentTarget as HTMLInputElement).checked })}
              />
            </td>
            <td class="py-2">
              <input
                class="w-full rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-sm"
                value={edit.name}
                onchange={(e) => setToolEdit(tool.id, { name: (e.currentTarget as HTMLInputElement).value })}
              />
              <input
                class="w-full mt-1 rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-xs"
                placeholder="description"
                value={edit.description || ""}
                onchange={(e) => setToolEdit(tool.id, { description: (e.currentTarget as HTMLInputElement).value })}
              />
            </td>
            <td class="py-2">
              <select
                class="settings-select rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-xs"
                value={edit.type}
                onchange={(e) => setToolEdit(tool.id, { type: (e.currentTarget as HTMLSelectElement).value as RuntimeToolType })}
              >
                <option value="http">http</option>
                <option value="mcp">mcp</option>
              </select>
            </td>
            <td class="py-2">
              <select
                class="settings-select rounded border border-gray-300 dark:border-white/20 px-2 py-1 text-xs"
                value={edit.policy || "auto"}
                onchange={(e) => setToolEdit(tool.id, { policy: (e.currentTarget as HTMLSelectElement).value as RuntimeToolPolicy })}
              >
                <option value="auto">auto</option>
                <option value="always">always</option>
                <option value="watchdog">watchdog</option>
                <option value="never">never</option>
              </select>
            </td>
            <td class="py-2">
              <input
                class="w-full rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-xs"
                value={edit.endpoint}
                onchange={(e) => setToolEdit(tool.id, { endpoint: (e.currentTarget as HTMLInputElement).value })}
              />
              <input
                class="w-full mt-1 rounded border border-gray-300 dark:border-white/20 bg-card px-2 py-1 text-xs"
                placeholder="remoteName (optional, for MCP)"
                value={edit.remoteName || ""}
                onchange={(e) => setToolEdit(tool.id, { remoteName: (e.currentTarget as HTMLInputElement).value })}
              />
            </td>
            <td class="py-2 text-xs">
              <label class="flex items-center gap-1">
                timeout
                <input class="w-16 rounded border border-gray-300 dark:border-white/20 bg-card px-1 py-0.5 text-xs" type="number" min="1" max="300" value={edit.timeoutSeconds || (edit.type === "mcp" ? 30 : 20)} onchange={(e) => setToolEdit(tool.id, { timeoutSeconds: parseInt((e.currentTarget as HTMLInputElement).value, 10) })} />
                s
              </label>
            </td>
            <td class="py-2 flex gap-2">
              <button class="btn btn--sm" onclick={() => saveToolEdit(tool.id)}>Save</button>
              <button class="btn btn--sm" onclick={() => resetToolEdit(tool.id)}>Reset</button>
              <button class="btn btn--sm" onclick={() => removeTool(tool.id)}>Delete</button>
            </td>
          </tr>
        {/each}
      {/if}
    </tbody>
  </table>
</div>

<style>
  .settings-select {
    background-color: var(--color-surface);
    color: var(--color-txtmain);
  }

  .settings-select option {
    background-color: var(--color-surface);
    color: var(--color-txtmain);
  }
</style>
