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

  const localModels = $derived.by(() => $models.filter((m) => !m.peerID).sort((a, b) => a.id.localeCompare(b.id)));

  $effect(() => {
    for (const model of localModels) {
      void ensureLoaded(model.id);
    }
  });
  onMount(() => {
    void loadConfigPath();
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
