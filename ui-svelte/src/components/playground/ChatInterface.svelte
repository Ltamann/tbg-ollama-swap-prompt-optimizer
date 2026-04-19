<script lang="ts">
  import { get } from "svelte/store";
  import { models, upstreamLogs } from "../../stores/api";
  import { persistentStore } from "../../stores/persistent";
  import {
    chatMessagesStore,
    chatIsStreamingStore,
    chatIsReasoningStore,
    type SamplingSettings,
    cancelChatStreaming,
    newChatSession,
    regenerateFromIndex,
    sendUserMessage,
    editUserMessage,
  } from "../../stores/playgroundChat";
  import ChatMessageComponent from "./ChatMessage.svelte";
  import ModelSelector from "./ModelSelector.svelte";
  import ExpandableTextarea from "./ExpandableTextarea.svelte";

  const selectedModelStore = persistentStore<string>("playground-selected-model", "");
  const systemPromptStore = persistentStore<string>("playground-system-prompt", "");
  const temperatureByModelStore = persistentStore<Record<string, number>>("playground-temperature-by-model", {});
  const topPByModelStore = persistentStore<Record<string, number>>("playground-top-p-by-model", {});
  const topKByModelStore = persistentStore<Record<string, number>>("playground-top-k-by-model", {});
  const minPByModelStore = persistentStore<Record<string, number>>("playground-min-p-by-model", {});
  const presencePenaltyByModelStore = persistentStore<Record<string, number>>("playground-presence-penalty-by-model", {});
  const frequencyPenaltyByModelStore = persistentStore<Record<string, number>>("playground-frequency-penalty-by-model", {});
  const maxTokensByModelStore = persistentStore<Record<string, number>>("playground-max-tokens-by-model", {});
  let currentTemperature = $state(0.8);
  let currentTopP = $state(0.95);
  let currentTopK = $state(40);
  let currentMinP = $state(0.05);
  let currentPresencePenalty = $state(0);
  let currentFrequencyPenalty = $state(0);
  let currentMaxTokens = $state<number>(0);
  let initializedSamplingModelID = $state("");
  let userInput = $state("");
  let messagesContainer: HTMLDivElement | undefined = $state();
  let showSettings = $state(false);
  let attachedImages = $state<string[]>([]);
  let fileInput = $state<HTMLInputElement | null>(null);
  let imageError = $state<string | null>(null);

  let hasModels = $derived($models.some((m) => !m.unlisted));
  let selectedModelState = $derived(
    $models.find((m) => m.id === $selectedModelStore)?.state || "unknown"
  );
  let isModelLoading = $derived(selectedModelState === "starting" || selectedModelState === "stopping");

  type LoadStage = { pct: number; label: string; patterns: string[] };

  const LOAD_STAGES: LoadStage[] = [
    { pct: 8, label: "Starting model process", patterns: ["srv    load_model: loading model", "main: loading model"] },
    { pct: 18, label: "Reading model metadata", patterns: ["llama_model_loader: loaded meta data"] },
    { pct: 36, label: "Loading tensors", patterns: ["load_tensors: loading model tensors"] },
    { pct: 56, label: "Offloading layers to GPU", patterns: ["load_tensors: offloaded"] },
    { pct: 72, label: "Building context", patterns: ["llama_context: constructing llama_context"] },
    { pct: 84, label: "Allocating compute buffers", patterns: ["sched_reserve: reserve took", "sched_reserve:"] },
    { pct: 92, label: "Warming up model", patterns: ["warming up the model"] },
    { pct: 100, label: "Model ready", patterns: ["main: model loaded", "srv    load_model: initialized slots"] },
  ];

  function parseModelLoadProgress(logs: string): { pct: number; label: string } {
    const raw = (logs || "").toLowerCase();
    if (!raw) {
      return { pct: 4, label: "Waiting for load logs..." };
    }

    const markerA = raw.lastIndexOf("srv    load_model: loading model");
    const markerB = raw.lastIndexOf("main: loading model");
    const start = Math.max(markerA, markerB, 0);
    const section = raw.slice(start);

    let pct = 4;
    let label = "Starting model process";
    for (const stage of LOAD_STAGES) {
      if (stage.patterns.some((pattern) => section.includes(pattern.toLowerCase()))) {
        if (stage.pct >= pct) {
          pct = stage.pct;
          label = stage.label;
        }
      }
    }
    return { pct, label };
  }

  let loadProgress = $derived(parseModelLoadProgress($upstreamLogs));

  function configuredTemperatureForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.tempConfigured === "number" && Number.isFinite(model.tempConfigured)) {
      return model.tempConfigured;
    }
    return 0.8;
  }

  function configuredTopPForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.topPConfigured === "number" && Number.isFinite(model.topPConfigured)) {
      return model.topPConfigured;
    }
    return 0.95;
  }

  function configuredTopKForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.topKConfigured === "number" && Number.isFinite(model.topKConfigured)) {
      return model.topKConfigured;
    }
    return 40;
  }

  function configuredMinPForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.minPConfigured === "number" && Number.isFinite(model.minPConfigured)) {
      return model.minPConfigured;
    }
    return 0.05;
  }

  function configuredPresencePenaltyForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.presencePenaltyConfigured === "number" && Number.isFinite(model.presencePenaltyConfigured)) {
      return model.presencePenaltyConfigured;
    }
    return 0;
  }

  function configuredFrequencyPenaltyForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.frequencyPenaltyConfigured === "number" && Number.isFinite(model.frequencyPenaltyConfigured)) {
      return model.frequencyPenaltyConfigured;
    }
    return 0;
  }

  // Keep sampling settings model-specific and initialize from model config when available.
  // If missing in config, fall back to llama.cpp defaults.
  $effect(() => {
    const modelID = $selectedModelStore;
    if (!modelID) {
      initializedSamplingModelID = "";
      currentTemperature = 0.8;
      currentTopP = 0.95;
      currentTopK = 40;
      currentMinP = 0.05;
      currentPresencePenalty = 0;
      currentFrequencyPenalty = 0;
      currentMaxTokens = 0;
      return;
    }
    if (initializedSamplingModelID === modelID) {
      return;
    }
    initializedSamplingModelID = modelID;

    const configuredTemp = configuredTemperatureForModel(modelID);
    const configuredTopP = configuredTopPForModel(modelID);
    const configuredTopK = configuredTopKForModel(modelID);
    const configuredMinP = configuredMinPForModel(modelID);
    const configuredPresence = configuredPresencePenaltyForModel(modelID);
    const configuredFrequency = configuredFrequencyPenaltyForModel(modelID);

    currentTemperature = configuredTemp;
    currentTopP = configuredTopP;
    currentTopK = configuredTopK;
    currentMinP = configuredMinP;
    currentPresencePenalty = configuredPresence;
    currentFrequencyPenalty = configuredFrequency;
    currentMaxTokens = 0;

    temperatureByModelStore.set({ ...get(temperatureByModelStore), [modelID]: configuredTemp });
    topPByModelStore.set({ ...get(topPByModelStore), [modelID]: configuredTopP });
    topKByModelStore.set({ ...get(topKByModelStore), [modelID]: configuredTopK });
    minPByModelStore.set({ ...get(minPByModelStore), [modelID]: configuredMinP });
    presencePenaltyByModelStore.set({ ...get(presencePenaltyByModelStore), [modelID]: configuredPresence });
    frequencyPenaltyByModelStore.set({ ...get(frequencyPenaltyByModelStore), [modelID]: configuredFrequency });
    maxTokensByModelStore.set({ ...get(maxTokensByModelStore), [modelID]: 0 });
  });

  function currentSamplingSettings(): SamplingSettings {
    return {
      temperature: currentTemperature,
      top_p: currentTopP,
      top_k: Math.round(currentTopK),
      min_p: currentMinP,
      presence_penalty: currentPresencePenalty,
      frequency_penalty: currentFrequencyPenalty,
      max_tokens: currentMaxTokens > 0 ? Math.round(currentMaxTokens) : undefined,
    };
  }

  // Auto-scroll when messages change
  $effect(() => {
    if ($chatMessagesStore.length > 0 && messagesContainer) {
      messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: "smooth",
      });
    }
  });

  async function sendMessage() {
    const sent = await sendUserMessage(userInput, attachedImages, $selectedModelStore, $systemPromptStore, currentSamplingSettings());
    if (sent) {
      userInput = "";
      attachedImages = [];
      imageError = null;
    }
  }

  function cancelStreaming() {
    cancelChatStreaming();
  }

  function newChat() {
    newChatSession();
  }

  async function editMessage(idx: number, newContent: string) {
    await editUserMessage(idx, newContent, $selectedModelStore, $systemPromptStore, currentSamplingSettings());
  }

  function handleTemperatureInput(value: number): void {
    currentTemperature = value;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    temperatureByModelStore.set({
      ...$temperatureByModelStore,
      [modelID]: value,
    });
  }

  function handleTopPInput(value: number): void {
    currentTopP = value;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    topPByModelStore.set({ ...$topPByModelStore, [modelID]: value });
  }

  function handleTopKInput(value: number): void {
    const rounded = Math.max(0, Math.round(value));
    currentTopK = rounded;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    topKByModelStore.set({ ...$topKByModelStore, [modelID]: rounded });
  }

  function handleMinPInput(value: number): void {
    currentMinP = value;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    minPByModelStore.set({ ...$minPByModelStore, [modelID]: value });
  }

  function handlePresencePenaltyInput(value: number): void {
    currentPresencePenalty = value;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    presencePenaltyByModelStore.set({ ...$presencePenaltyByModelStore, [modelID]: value });
  }

  function handleFrequencyPenaltyInput(value: number): void {
    currentFrequencyPenalty = value;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    frequencyPenaltyByModelStore.set({ ...$frequencyPenaltyByModelStore, [modelID]: value });
  }

  function handleMaxTokensInput(value: number): void {
    const rounded = Math.max(0, Math.round(value));
    currentMaxTokens = rounded;
    const modelID = $selectedModelStore;
    if (!modelID) return;
    maxTokensByModelStore.set({ ...$maxTokensByModelStore, [modelID]: rounded });
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  const ACCEPTED_IMAGE_FORMATS = ["image/jpeg", "image/png", "image/gif", "image/webp"];
  const MAX_IMAGE_SIZE = 20 * 1024 * 1024; // 20MB
  const MAX_IMAGES_PER_MESSAGE = 5;

  function validateImageFile(file: File): string | null {
    if (!ACCEPTED_IMAGE_FORMATS.includes(file.type)) {
      return `Invalid file type: ${file.type}. Accepted formats: JPG, PNG, GIF, WEBP`;
    }
    if (file.size > MAX_IMAGE_SIZE) {
      return `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum size: 20MB`;
    }
    return null;
  }

  function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsDataURL(file);
    });
  }

  async function processImageFiles(files: File[]): Promise<void> {
    imageError = null;

    if (attachedImages.length + files.length > MAX_IMAGES_PER_MESSAGE) {
      imageError = `Maximum ${MAX_IMAGES_PER_MESSAGE} images per message`;
      return;
    }

    for (const file of files) {
      const error = validateImageFile(file);
      if (error) {
        imageError = error;
        return;
      }
    }

    try {
      const dataUrls = await Promise.all(files.map(fileToDataUrl));
      attachedImages = [...attachedImages, ...dataUrls];
    } catch (error) {
      imageError = error instanceof Error ? error.message : "Failed to process images";
    }
  }

  function handleImageSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      processImageFiles(Array.from(input.files));
    }
    // Reset the input so the same file can be selected again
    input.value = "";
  }

  function removeImage(idx: number) {
    attachedImages = attachedImages.filter((_, i) => i !== idx);
    imageError = null;
  }
</script>

<div class="h-full flex flex-col">
  <div class="shrink-0 sticky top-0 z-20 border-b border-gray-200/70 dark:border-white/10 bg-background/92 backdrop-blur">
    <div class="max-w-4xl mx-auto w-full px-4 py-2">
      <div class="flex items-center gap-2">
        <div class="flex-1"></div>
        <button
          class="btn px-2.5 py-2 rounded-lg"
          onclick={() => (showSettings = !showSettings)}
          title="Settings"
          aria-label="Settings"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
            <path fill-rule="evenodd" d="M8.34 1.804A1 1 0 0 1 9.32 1h1.36a1 1 0 0 1 .98.804l.295 1.473c.497.144.971.342 1.416.587l1.25-.834a1 1 0 0 1 1.262.125l.962.962a1 1 0 0 1 .125 1.262l-.834 1.25c.245.445.443.919.587 1.416l1.473.295a1 1 0 0 1 .804.98v1.36a1 1 0 0 1-.804.98l-1.473.295a6.95 6.95 0 0 1-.587 1.416l.834 1.25a1 1 0 0 1-.125 1.262l-.962.962a1 1 0 0 1-1.262.125l-1.25-.834a6.953 6.953 0 0 1-1.416.587l-.295 1.473a1 1 0 0 1-.98.804H9.32a1 1 0 0 1-.98-.804l-.295-1.473a6.957 6.957 0 0 1-1.416-.587l-1.25.834a1 1 0 0 1-1.262-.125l-.962-.962a1 1 0 0 1-.125-1.262l.834-1.25a6.957 6.957 0 0 1-.587-1.416l-1.473-.295A1 1 0 0 1 1 10.68V9.32a1 1 0 0 1 .804-.98l1.473-.295c.144-.497.342-.971.587-1.416l-.834-1.25a1 1 0 0 1 .125-1.262l.962-.962A1 1 0 0 1 5.38 3.03l1.25.834a6.957 6.957 0 0 1 1.416-.587l.294-1.473ZM13 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" clip-rule="evenodd" />
          </svg>
        </button>
        <button class="btn px-2.5 py-2 rounded-lg" onclick={newChat} disabled={$chatMessagesStore.length === 0 && !$chatIsStreamingStore} title="New chat" aria-label="New chat">
          New Chat
        </button>
      </div>
      {#if isModelLoading}
        <div class="mt-2 max-w-md ml-auto px-3 py-2 rounded-xl border border-gray-200 dark:border-white/10 bg-surface shadow-sm">
          <div class="flex items-center justify-between text-xs text-txtsecondary mb-1">
            <span>Model loading</span>
            <span>{Math.min(100, Math.max(0, Math.round(loadProgress.pct)))}%</span>
          </div>
          <div class="h-2 rounded-full bg-gray-200 dark:bg-white/10 overflow-hidden">
            <div class="h-full bg-primary transition-all duration-300 ease-out" style={`width: ${Math.min(100, Math.max(0, loadProgress.pct))}%`}></div>
          </div>
          <div class="text-[11px] text-txtsecondary mt-1">{loadProgress.label}</div>
        </div>
      {/if}
      {#if showSettings}
        <div class="mt-3 p-4 bg-surface border border-gray-200 dark:border-white/10 rounded-xl shadow-sm">
          <div class="mb-4">
            <label class="block text-sm font-medium mb-1" for="system-prompt">System Prompt</label>
            <textarea
              id="system-prompt"
              class="w-full px-3 py-2 rounded border border-gray-200 dark:border-white/10 bg-card focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              placeholder="You are a helpful assistant..."
              rows="3"
              bind:value={$systemPromptStore}
              disabled={$chatIsStreamingStore}
            ></textarea>
          </div>
          <div>
            <label class="block text-sm font-medium mb-1" for="temperature">
              Temperature: {currentTemperature.toFixed(2)}
            </label>
            <input
              id="temperature"
              type="range"
              min="0"
              max="2"
              step="0.05"
              class="w-full"
              bind:value={currentTemperature}
              oninput={(e) => handleTemperatureInput(parseFloat((e.currentTarget as HTMLInputElement).value))}
              disabled={$chatIsStreamingStore}
            />
            <div class="flex justify-between text-xs text-txtsecondary mt-1">
              <span>Precise (0)</span>
              <span>Creative (2)</span>
            </div>
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="top-p">Top P: {currentTopP.toFixed(2)}</label>
            <input id="top-p" type="range" min="0" max="1" step="0.01" class="w-full" bind:value={currentTopP} oninput={(e) => handleTopPInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="top-k">Top K: {Math.round(currentTopK)}</label>
            <input id="top-k" type="range" min="0" max="200" step="1" class="w-full" bind:value={currentTopK} oninput={(e) => handleTopKInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="min-p">Min P: {currentMinP.toFixed(2)}</label>
            <input id="min-p" type="range" min="0" max="1" step="0.01" class="w-full" bind:value={currentMinP} oninput={(e) => handleMinPInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="presence-penalty">Presence Penalty: {currentPresencePenalty.toFixed(2)}</label>
            <input id="presence-penalty" type="range" min="-2" max="2" step="0.01" class="w-full" bind:value={currentPresencePenalty} oninput={(e) => handlePresencePenaltyInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="frequency-penalty">Frequency Penalty: {currentFrequencyPenalty.toFixed(2)}</label>
            <input id="frequency-penalty" type="range" min="-2" max="2" step="0.01" class="w-full" bind:value={currentFrequencyPenalty} oninput={(e) => handleFrequencyPenaltyInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
          <div class="mt-3">
            <label class="block text-sm font-medium mb-1" for="max-tokens">Max Tokens (0 = model default): {Math.round(currentMaxTokens)}</label>
            <input id="max-tokens" type="range" min="0" max="8192" step="64" class="w-full" bind:value={currentMaxTokens} oninput={(e) => handleMaxTokensInput(parseFloat((e.currentTarget as HTMLInputElement).value))} disabled={$chatIsStreamingStore} />
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Empty state for no models configured -->
  {#if !hasModels}
    <div class="flex-1 flex items-center justify-center text-txtsecondary">
      <p>No models configured. Add models to your configuration to start chatting.</p>
    </div>
  {:else}
    <!-- Messages area -->
    <div
      class="flex-1 overflow-y-auto px-2"
      bind:this={messagesContainer}
    >
      <div class="max-w-4xl mx-auto w-full py-6 px-2">
      {#if $chatMessagesStore.length === 0}
        <div class="h-full flex items-center justify-center text-txtsecondary">
          <p>Start a conversation by typing a message below.</p>
        </div>
      {:else}
        {#each $chatMessagesStore as message, idx (idx)}
          <ChatMessageComponent
            role={message.role}
            content={message.content}
            reasoning_content={message.reasoning_content}
            reasoningTimeMs={message.reasoningTimeMs}
            isStreaming={$chatIsStreamingStore && idx === $chatMessagesStore.length - 1 && message.role === "assistant"}
            isReasoning={$chatIsReasoningStore && idx === $chatMessagesStore.length - 1 && message.role === "assistant"}
            onEdit={message.role === "user" ? (newContent) => editMessage(idx, newContent) : undefined}
            onRegenerate={message.role === "assistant" && idx > 0 && $chatMessagesStore[idx - 1].role === "user"
              ? () => regenerateFromIndex(idx - 1, $selectedModelStore, $systemPromptStore, currentSamplingSettings())
              : undefined}
          />
        {/each}
      {/if}
      </div>
    </div>

    <!-- Input area -->
    <div class="shrink-0 sticky bottom-0 z-20 bg-background/95 backdrop-blur border-t border-gray-200/70 dark:border-white/10">
      <div class="max-w-4xl mx-auto w-full px-4 py-3">
      <!-- Image preview strip -->
      {#if attachedImages.length > 0}
        <div class="mb-2 flex flex-wrap gap-2">
          {#each attachedImages as imageUrl, idx (idx)}
            <div class="relative group">
              <img
                src={imageUrl}
                alt="Attached image {idx + 1}"
                class="w-20 h-20 object-cover rounded border border-gray-200 dark:border-white/10"
              />
              <button
                class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                onclick={() => removeImage(idx)}
                title="Remove image"
              >
                ×
              </button>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Error message -->
      {#if imageError}
        <div class="mb-2 p-2 bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded text-sm">
          {imageError}
        </div>
      {/if}

      <div role="group" aria-label="Message input and attachments" class="flex items-end gap-2 rounded-[1.6rem] border border-gray-300/80 dark:border-white/10 bg-surface px-2.5 py-2 shadow-sm">
        <!-- Hidden file input -->
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.gif,.webp"
          multiple
          class="hidden"
          bind:this={fileInput}
          onchange={handleImageSelect}
        />

        <div class="relative shrink-0">
        {#if !$chatIsStreamingStore}
          <button
            class="btn w-9 h-9 rounded-full p-0 flex items-center justify-center"
            onclick={() => fileInput?.click()}
            disabled={$chatIsStreamingStore || !$selectedModelStore || isModelLoading}
            title="Attach image"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
              <path fill-rule="evenodd" d="M1 5.25A2.25 2.25 0 0 1 3.25 3h13.5A2.25 2.25 0 0 1 19 5.25v9.5A2.25 2.25 0 0 1 16.75 17H3.25A2.25 2.25 0 0 1 1 14.75v-9.5Zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 0 0 .75-.75v-2.69l-2.22-2.219a.75.75 0 0 0-1.06 0l-1.91 1.909.47.47a.75.75 0 1 1-1.06 1.06L6.53 8.091a.75.75 0 0 0-1.06 0l-2.97 2.97ZM12 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z" clip-rule="evenodd" />
            </svg>
          </button>
        {/if}
        </div>

        <ExpandableTextarea
          bind:value={userInput}
          placeholder="Ask anything..."
          rows={3}
          onkeydown={handleKeyDown}
          disabled={$chatIsStreamingStore || !$selectedModelStore || isModelLoading}
        />
        <div class="relative shrink-0">
          <ModelSelector bind:value={$selectedModelStore} placeholder="Select a model..." disabled={$chatIsStreamingStore} />
        </div>
        <div class="shrink-0">
          {#if $chatIsStreamingStore}
            <button class="btn bg-red-500 hover:bg-red-600 text-white w-9 h-9 rounded-full p-0 flex items-center justify-center" onclick={cancelStreaming} title="Cancel">
              Cancel
            </button>
          {:else}
            <button
              class="btn bg-primary text-btn-primary-text hover:opacity-90 w-9 h-9 rounded-full p-0 flex items-center justify-center"
              onclick={sendMessage}
              disabled={(!userInput.trim() && attachedImages.length === 0) || !$selectedModelStore || isModelLoading}
              title="Send message"
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4">
                <path d="M12 4l6.5 8h-4v8h-5v-8h-4L12 4Z"></path>
              </svg>
            </button>
          {/if}
        </div>
      </div>
      </div>
    </div>
  {/if}
</div>
