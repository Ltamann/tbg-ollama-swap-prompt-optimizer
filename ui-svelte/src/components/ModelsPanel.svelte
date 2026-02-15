<script lang="ts">
  import {
    models,
    loadModel,
    unloadAllModels,
    unloadSingleModel,
    getModelCtxSize,
    setModelCtxSize,
    getModelFitMode,
    setModelFitMode,
  } from "../stores/api";
  import { isNarrow } from "../stores/theme";
  import { persistentStore } from "../stores/persistent";
  import type { Model } from "../lib/types";

  let isUnloading = $state(false);
  let menuOpen = $state(false);

  const showUnlistedStore = persistentStore<boolean>("showUnlisted", true);
  const showIdorNameStore = persistentStore<"id" | "name">("showIdorName", "id");

  const defaultCtxSize = 32768;
  const requestedCtxSizes = [8096, 32768, 65536, 98304, 131072, 180224, 196608, 202752, 262144];
  const ctxSizeSet = new Set<number>(requestedCtxSizes);
  for (let v = 4096; v <= 262144; v += 4096) {
    ctxSizeSet.add(v);
  }
  const ctxOptions = Array.from(ctxSizeSet).sort((a, b) => a - b);

  let ctxSelections = $state<Record<string, number>>({});
  let ctxLoading = $state<Record<string, boolean>>({});
  let ctxKnown = $state<Record<string, boolean>>({});
  let fitSelections = $state<Record<string, boolean>>({});

  let filteredModels = $derived.by(() => {
    const filtered = $models.filter((model) => $showUnlistedStore || !model.unlisted);
    const peerModels = filtered.filter((m) => m.peerID);

    // Group peer models by peerID
    const grouped = peerModels.reduce(
      (acc, model) => {
        const peerId = model.peerID || "unknown";
        if (!acc[peerId]) acc[peerId] = [];
        acc[peerId].push(model);
        return acc;
      },
      {} as Record<string, Model[]>
    );

    return {
      regularModels: filtered.filter((m) => !m.peerID),
      peerModelsByPeerId: grouped,
    };
  });

  $effect(() => {
    for (const model of filteredModels.regularModels) {
      if (model.external) {
        continue;
      }
      if (fitSelections[model.id] === undefined) {
        fitSelections = { ...fitSelections, [model.id]: model.fitEnabled === true };
      }
      void ensureCtxSizeLoaded(model.id);
    }
  });

  async function handleUnloadAllModels(): Promise<void> {
    isUnloading = true;
    try {
      await unloadAllModels();
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(() => (isUnloading = false), 1000);
    }
  }

  function toggleIdorName(): void {
    showIdorNameStore.update((prev) => (prev === "name" ? "id" : "name"));
  }

  function toggleShowUnlisted(): void {
    showUnlistedStore.update((prev) => !prev);
  }

  function getModelDisplay(model: Model): string {
    return $showIdorNameStore === "id" ? model.id : (model.name || model.id);
  }

  function formatCtxLabel(value: number): string {
    const k = Math.round(value / 1024);
    return `${value.toLocaleString()} (${k}k)`;
  }

  async function ensureCtxSizeLoaded(modelId: string): Promise<void> {
    if (ctxKnown[modelId] || ctxLoading[modelId]) {
      return;
    }

    ctxLoading = { ...ctxLoading, [modelId]: true };
    try {
      const remote = await getModelCtxSize(modelId);
      const model = filteredModels.regularModels.find((m) => m.id === modelId);
      const configuredCtx = model?.ctxConfigured && model.ctxConfigured > 0 ? model.ctxConfigured : defaultCtxSize;
      ctxSelections = { ...ctxSelections, [modelId]: remote > 0 ? remote : configuredCtx };
      ctxKnown = { ...ctxKnown, [modelId]: true };
    } finally {
      ctxLoading = { ...ctxLoading, [modelId]: false };
    }
  }

  async function persistCtxSize(modelId: string): Promise<void> {
    const selected = ctxSelections[modelId] || defaultCtxSize;
    await setModelCtxSize(modelId, selected);
  }

  async function handleCtxChange(modelId: string, value: string): Promise<void> {
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return;
    }

    ctxSelections = { ...ctxSelections, [modelId]: parsed };
    try {
      await persistCtxSize(modelId);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleLoadModel(modelId: string): Promise<void> {
    try {
      await persistCtxSize(modelId);
      await setModelFitMode(modelId, fitSelections[modelId] === true);
      await loadModel(modelId);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleFitToggle(modelId: string, checked: boolean): Promise<void> {
    fitSelections = { ...fitSelections, [modelId]: checked };
    try {
      await setModelFitMode(modelId, checked);
    } catch (e) {
      console.error(e);
      const fallback = await getModelFitMode(modelId);
      fitSelections = { ...fitSelections, [modelId]: fallback };
    }
  }

  function isExternalModel(model: Model): boolean {
    return model.external === true || model.provider === "ollama";
  }
</script>

<div class="card h-full flex flex-col">
  <div class="shrink-0">
    <div class="flex justify-between items-baseline">
      <h2 class={$isNarrow ? "text-xl" : ""}>Models</h2>
      {#if $isNarrow}
        <div class="relative">
          <button class="btn text-base flex items-center gap-2 py-1" onclick={() => (menuOpen = !menuOpen)} aria-label="Toggle menu">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
              <path fill-rule="evenodd" d="M3 6.75A.75.75 0 0 1 3.75 6h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 6.75ZM3 12a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 12Zm0 5.25a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75a.75.75 0 0 1-.75-.75Z" clip-rule="evenodd" />
            </svg>
          </button>
          {#if menuOpen}
            <div class="absolute right-0 mt-2 w-48 bg-surface border border-gray-200 dark:border-white/10 rounded shadow-lg z-20">
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { toggleIdorName(); menuOpen = false; }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                  <path fill-rule="evenodd" d="M15.97 2.47a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1 0 1.06l-4.5 4.5a.75.75 0 1 1-1.06-1.06l3.22-3.22H7.5a.75.75 0 0 1 0-1.5h11.69l-3.22-3.22a.75.75 0 0 1 0-1.06Zm-7.94 9a.75.75 0 0 1 0 1.06l-3.22 3.22H16.5a.75.75 0 0 1 0 1.5H4.81l3.22 3.22a.75.75 0 1 1-1.06 1.06l-4.5-4.5a.75.75 0 0 1 0-1.06l4.5-4.5a.75.75 0 0 1 1.06 0Z" clip-rule="evenodd" />
                </svg>
                {$showIdorNameStore === "id" ? "Show Name" : "Show ID"}
              </button>
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { toggleShowUnlisted(); menuOpen = false; }}
              >
                {#if $showUnlistedStore}
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M3.53 2.47a.75.75 0 0 0-1.06 1.06l18 18a.75.75 0 1 0 1.06-1.06l-18-18ZM22.676 12.553a11.249 11.249 0 0 1-2.631 4.31l-3.099-3.099a5.25 5.25 0 0 0-6.71-6.71L7.759 4.577a11.217 11.217 0 0 1 4.242-.827c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113Z" />
                    <path d="M15.75 12c0 .18-.013.357-.037.53l-4.244-4.243A3.75 3.75 0 0 1 15.75 12ZM12.53 15.713l-4.243-4.244a3.75 3.75 0 0 0 4.244 4.243Z" />
                    <path d="M6.75 12c0-.619.107-1.213.304-1.764l-3.1-3.1a11.25 11.25 0 0 0-2.63 4.31c-.12.362-.12.752 0 1.114 1.489 4.467 5.704 7.69 10.675 7.69 1.5 0 2.933-.294 4.242-.827l-2.477-2.477A5.25 5.25 0 0 1 6.75 12Z" />
                  </svg>
                {:else}
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                    <path fill-rule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z" clip-rule="evenodd" />
                  </svg>
                {/if}
                {$showUnlistedStore ? "Hide Unlisted" : "Show Unlisted"}
              </button>
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { handleUnloadAllModels(); menuOpen = false; }}
                disabled={isUnloading}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
                  <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm.53 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v5.69a.75.75 0 0 0 1.5 0v-5.69l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
                </svg>
                {isUnloading ? "Unloading..." : "Unload All"}
              </button>
            </div>
          {/if}
        </div>
      {/if}
    </div>
    {#if !$isNarrow}
      <div class="flex justify-between">
        <div class="flex gap-2">
          <button class="btn text-base flex items-center gap-2" onclick={toggleIdorName} style="line-height: 1.2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
              <path fill-rule="evenodd" d="M15.97 2.47a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1 0 1.06l-4.5 4.5a.75.75 0 1 1-1.06-1.06l3.22-3.22H7.5a.75.75 0 0 1 0-1.5h11.69l-3.22-3.22a.75.75 0 0 1 0-1.06Zm-7.94 9a.75.75 0 0 1 0 1.06l-3.22 3.22H16.5a.75.75 0 0 1 0 1.5H4.81l3.22 3.22a.75.75 0 1 1-1.06 1.06l-4.5-4.5a.75.75 0 0 1 0-1.06l4.5-4.5a.75.75 0 0 1 1.06 0Z" clip-rule="evenodd" />
            </svg>
            {$showIdorNameStore === "id" ? "ID" : "Name"}
          </button>

          <button class="btn text-base flex items-center gap-2" onclick={toggleShowUnlisted} style="line-height: 1.2">
            {#if $showUnlistedStore}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                <path fill-rule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z" clip-rule="evenodd" />
              </svg>
            {:else}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                <path d="M3.53 2.47a.75.75 0 0 0-1.06 1.06l18 18a.75.75 0 1 0 1.06-1.06l-18-18ZM22.676 12.553a11.249 11.249 0 0 1-2.631 4.31l-3.099-3.099a5.25 5.25 0 0 0-6.71-6.71L7.759 4.577a11.217 11.217 0 0 1 4.242-.827c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113Z" />
                <path d="M15.75 12c0 .18-.013.357-.037.53l-4.244-4.243A3.75 3.75 0 0 1 15.75 12ZM12.53 15.713l-4.243-4.244a3.75 3.75 0 0 0 4.244 4.243Z" />
                <path d="M6.75 12c0-.619.107-1.213.304-1.764l-3.1-3.1a11.25 11.25 0 0 0-2.63 4.31c-.12.362-.12.752 0 1.114 1.489 4.467 5.704 7.69 10.675 7.69 1.5 0 2.933-.294 4.242-.827l-2.477-2.477A5.25 5.25 0 0 1 6.75 12Z" />
              </svg>
            {/if}
            unlisted
          </button>
        </div>
        <button class="btn text-base flex items-center gap-2" onclick={handleUnloadAllModels} disabled={isUnloading}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
            <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm.53 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v5.69a.75.75 0 0 0 1.5 0v-5.69l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
          </svg>
          {isUnloading ? "Unloading..." : "Unload All"}
        </button>
      </div>
    {/if}
  </div>

  <div class="flex-1 overflow-y-auto">
    <table class="w-full">
      <thead class="sticky top-0 bg-card z-10">
        <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
          <th>{$showIdorNameStore === "id" ? "Model ID" : "Name"}</th>
          <th>Ctx</th>
          <th></th>
          <th>State</th>
        </tr>
      </thead>
      <tbody>
        {#each filteredModels.regularModels as model (model.id)}
          {@const isExternal = isExternalModel(model)}
          {@const currentCtx = isExternal ? (model.ctxReference || 0) : (ctxSelections[model.id] || model.ctxConfigured || defaultCtxSize)}
          {@const fitEnabled = fitSelections[model.id] === true}
          {@const ctxSource = fitEnabled ? "fit-ctx" : (model.ctxSource || "ctx-size")}
          {@const sourceClass = ctxSource === "fit-ctx" ? "text-emerald-600 dark:text-emerald-400" : "text-sky-600 dark:text-sky-400"}
          <tr class="border-b hover:bg-secondary-hover border-gray-200">
            <td class={model.unlisted ? "text-txtsecondary" : ""}>
              {#if isExternal}
                <span class="font-semibold">{getModelDisplay(model)}</span>
              {:else}
                <a href="/upstream/{model.id}/" class="font-semibold" target="_blank">
                  {getModelDisplay(model)}
                </a>
              {/if}
              {#if model.description}
                <p class={model.unlisted ? "text-opacity-70" : ""}><em>{model.description}</em></p>
              {/if}
              {#if isExternal}
                <p class="text-xs text-txtsecondary">Ollama model (external)</p>
              {/if}
            </td>
            <td class="w-44">
              <div class="flex items-center gap-2">
                {#if !isExternal}
                  <label class="inline-flex items-center gap-1 text-xs">
                    <input
                      type="checkbox"
                      checked={fitEnabled}
                      onchange={(e) => handleFitToggle(model.id, (e.currentTarget as HTMLInputElement).checked)}
                    />
                    fit
                  </label>
                {/if}
                <select
                  class="w-full rounded border border-gray-300 dark:border-white/20 bg-surface px-2 py-1 text-sm"
                  value={currentCtx}
                  disabled={isExternal || ctxLoading[model.id]}
                  onchange={(e) => handleCtxChange(model.id, (e.currentTarget as HTMLSelectElement).value)}
                >
                  {#if isExternal}
                    <option value={currentCtx}>
                      {currentCtx > 0 ? formatCtxLabel(currentCtx) : "From Ollama model file"}
                    </option>
                  {:else}
                    {#each ctxOptions as ctx}
                      <option value={ctx}>{formatCtxLabel(ctx)}</option>
                    {/each}
                  {/if}
                </select>
              </div>
              {#if !isExternal}
                <div class={`mt-1 text-[10px] leading-tight ${sourceClass}`}>
                  {ctxSource}
                </div>
              {/if}
            </td>
            <td class="w-12">
              {#if isExternal}
                <span class="text-xs text-txtsecondary">auto</span>
              {:else if model.state === "stopped"}
                <button class="btn btn--sm" onclick={() => handleLoadModel(model.id)}>Load</button>
              {:else}
                <button class="btn btn--sm" onclick={() => unloadSingleModel(model.id)} disabled={model.state !== "ready"}>Unload</button>
              {/if}
            </td>
            <td class="w-20">
              <span class="w-16 text-center status status--{model.state}">{model.state}</span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if Object.keys(filteredModels.peerModelsByPeerId).length > 0}
      <h3 class="mt-8 mb-2">Peer Models</h3>
      {#each Object.entries(filteredModels.peerModelsByPeerId).sort(([a], [b]) => a.localeCompare(b)) as [peerId, peerModels] (peerId)}
        <div class="mb-4">
          <table class="w-full">
            <thead class="sticky top-0 bg-card z-10">
              <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
                <th class="font-semibold">{peerId}</th>
              </tr>
            </thead>
            <tbody>
              {#each peerModels as model (model.id)}
                <tr class="border-b hover:bg-secondary-hover border-gray-200">
                  <td class="pl-8 {model.unlisted ? 'text-txtsecondary' : ''}">
                    <span>{model.id}</span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/each}
    {/if}
  </div>
</div>
