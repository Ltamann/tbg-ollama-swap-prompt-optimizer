<script lang="ts">
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { models, listTools, getToolRuntimeSettings } from "../../stores/api";
  import { persistentStore } from "../../stores/persistent";
  import {
    chatMessagesStore,
    chatIsStreamingStore,
    chatIsReasoningStore,
    type SamplingSettings,
    type UploadedAttachment,
    cancelChatStreaming,
    newChatSession,
    regenerateFromIndex,
    sendUserMessage,
    editUserMessage,
    deleteUserMessageWithReply,
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
  let attachments = $state<UploadedAttachment[]>([]);
  let fileInput = $state<HTMLInputElement | null>(null);
  let attachmentError = $state<string | null>(null);
  let isDraggingFiles = $state(false);
  let toolStatusText = $state("");

  let hasModels = $derived($models.some((m) => !m.unlisted));
  let selectableModels = $derived(
    $models.filter((m) => !m.unlisted && !m.peerID)
  );

  function configuredTemperatureForModel(modelID: string): number {
    const model = $models.find((m) => m.id === modelID);
    if (model && typeof model.tempConfigured === "number" && Number.isFinite(model.tempConfigured)) {
      return model.tempConfigured;
    }
    return 0.8;
  }

  onMount(() => {
    void refreshToolStatus();
  });

  async function refreshToolStatus(): Promise<void> {
    try {
      const [settings, tools] = await Promise.all([getToolRuntimeSettings(), listTools()]);
      if (!settings.enabled) {
        toolStatusText = "Tools: Off";
        return;
      }
      const enabledTools = tools.filter((t) => t.enabled && (t.policy || "auto") !== "never");
      const names = enabledTools.slice(0, 3).map((t) => t.name);
      const suffix = enabledTools.length > 3 ? ` +${enabledTools.length - 3}` : "";
      toolStatusText = `Tools: ${enabledTools.length === 0 ? "none" : names.join(", ")}${suffix} | mode=${settings.webSearchMode}`;
    } catch {
      toolStatusText = "Tools: status unavailable";
    }
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

  // Keep chat model selection aligned with actual running model on load/reconnect.
  // If a ready model exists, prefer it; otherwise keep current if valid, else first available.
  $effect(() => {
    const available = selectableModels;
    if (available.length === 0) {
      return;
    }

    const selected = $selectedModelStore;
    const selectedIsValid = available.some((m) => m.id === selected);
    const readyModel = available.find((m) => m.state === "ready");
    const target = readyModel?.id ?? (selectedIsValid ? selected : available[0].id);

    if (target && target !== selected) {
      selectedModelStore.set(target);
      initializedSamplingModelID = "";
    }
  });

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
    const sent = await sendUserMessage(userInput, attachments, $selectedModelStore, $systemPromptStore, currentSamplingSettings());
    if (sent) {
      userInput = "";
      attachments = [];
      attachmentError = null;
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

  async function deleteMessage(idx: number) {
    await deleteUserMessageWithReply(idx, $selectedModelStore, $systemPromptStore, currentSamplingSettings());
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
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
  const MAX_ATTACHMENTS_PER_MESSAGE = 8;
  const MAX_TEXT_EXTRACT_BYTES = 128 * 1024;
  const TEXT_EXTENSIONS = [".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".csv", ".log", ".js", ".ts", ".tsx", ".jsx", ".py", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".rb", ".php", ".sh", ".sql", ".xml", ".html", ".css"];

  function validateAttachment(file: File): string | null {
    if (file.size > MAX_FILE_SIZE) {
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

  function fileToText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("Failed to read text file"));
      reader.readAsText(file);
    });
  }

  function hasTextExtension(name: string): boolean {
    const lower = name.toLowerCase();
    return TEXT_EXTENSIONS.some((ext) => lower.endsWith(ext));
  }

  function shouldExtractText(file: File): boolean {
    if (file.type.startsWith("text/") || file.type === "application/json" || file.type === "application/xml") {
      return true;
    }
    return hasTextExtension(file.name);
  }

  async function processFiles(files: File[]): Promise<void> {
    attachmentError = null;

    if (attachments.length + files.length > MAX_ATTACHMENTS_PER_MESSAGE) {
      attachmentError = `Maximum ${MAX_ATTACHMENTS_PER_MESSAGE} attachments per message`;
      return;
    }

    for (const file of files) {
      const error = validateAttachment(file);
      if (error) {
        attachmentError = error;
        return;
      }
    }

    try {
      const nextAttachments: UploadedAttachment[] = [];
      for (const file of files) {
        const id = `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`;
        if (ACCEPTED_IMAGE_FORMATS.includes(file.type)) {
          const dataUrl = await fileToDataUrl(file);
          nextAttachments.push({
            id,
            name: file.name,
            mimeType: file.type,
            size: file.size,
            kind: "image",
            dataUrl,
          });
          continue;
        }

        let textContent = "";
        if (shouldExtractText(file)) {
          const text = await fileToText(file);
          textContent = text.length > MAX_TEXT_EXTRACT_BYTES ? `${text.slice(0, MAX_TEXT_EXTRACT_BYTES)}\n...<truncated>` : text;
        }
        nextAttachments.push({
          id,
          name: file.name,
          mimeType: file.type || "application/octet-stream",
          size: file.size,
          kind: "file",
          textContent,
        });
      }
      attachments = [...attachments, ...nextAttachments];
    } catch (error) {
      attachmentError = error instanceof Error ? error.message : "Failed to process attachments";
    }
  }

  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      processFiles(Array.from(input.files));
    }
    // Reset the input so the same file can be selected again
    input.value = "";
  }

  function removeAttachment(id: string) {
    attachments = attachments.filter((a) => a.id !== id);
    attachmentError = null;
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    if ($chatIsStreamingStore || !$selectedModelStore) {
      return;
    }
    isDraggingFiles = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    isDraggingFiles = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDraggingFiles = false;
    if ($chatIsStreamingStore || !$selectedModelStore) {
      return;
    }
    const dropped = event.dataTransfer?.files;
    if (!dropped || dropped.length === 0) {
      return;
    }
    void processFiles(Array.from(dropped));
  }
</script>

<div class="flex flex-col h-full">
  <div class="shrink-0 sticky top-0 z-20 bg-card pb-3">
    <!-- Model selector and controls -->
    <div class="flex flex-wrap gap-2 mb-3">
      <ModelSelector bind:value={$selectedModelStore} placeholder="Select a model..." disabled={$chatIsStreamingStore} />
      <div class="flex gap-2">
        <button
          class="btn"
          onclick={() => (showSettings = !showSettings)}
          title="Settings"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
            <path fill-rule="evenodd" d="M8.34 1.804A1 1 0 0 1 9.32 1h1.36a1 1 0 0 1 .98.804l.295 1.473c.497.144.971.342 1.416.587l1.25-.834a1 1 0 0 1 1.262.125l.962.962a1 1 0 0 1 .125 1.262l-.834 1.25c.245.445.443.919.587 1.416l1.473.295a1 1 0 0 1 .804.98v1.36a1 1 0 0 1-.804.98l-1.473.295a6.95 6.95 0 0 1-.587 1.416l.834 1.25a1 1 0 0 1-.125 1.262l-.962.962a1 1 0 0 1-1.262.125l-1.25-.834a6.953 6.953 0 0 1-1.416.587l-.295 1.473a1 1 0 0 1-.98.804H9.32a1 1 0 0 1-.98-.804l-.295-1.473a6.957 6.957 0 0 1-1.416-.587l-1.25.834a1 1 0 0 1-1.262-.125l-.962-.962a1 1 0 0 1-.125-1.262l.834-1.25a6.957 6.957 0 0 1-.587-1.416l-1.473-.295A1 1 0 0 1 1 10.68V9.32a1 1 0 0 1 .804-.98l1.473-.295c.144-.497.342-.971.587-1.416l-.834-1.25a1 1 0 0 1 .125-1.262l.962-.962A1 1 0 0 1 5.38 3.03l1.25.834a6.957 6.957 0 0 1 1.416-.587l.294-1.473ZM13 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" clip-rule="evenodd" />
          </svg>
        </button>
        <button class="btn" onclick={newChat} disabled={$chatMessagesStore.length === 0 && !$chatIsStreamingStore}>
          New Chat
        </button>
      </div>
    </div>
    <div class="text-xs text-txtsecondary mb-2">{toolStatusText}</div>

    <!-- Settings panel -->
    {#if showSettings}
      <div class="p-4 bg-surface border border-gray-200 dark:border-white/10 rounded">
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

  <!-- Messages area -->
  <div
    class="flex-1 overflow-y-auto mb-4 px-2"
    bind:this={messagesContainer}
  >
    {#if $chatMessagesStore.length === 0}
      <div class="h-full flex items-center justify-center text-txtsecondary">
        {#if !hasModels}
          <p>No models configured. Add models to your configuration to start chatting.</p>
        {:else}
          <p>Start a conversation by typing a message below.</p>
        {/if}
      </div>
    {:else}
      {#each $chatMessagesStore as message, idx (idx)}
        <ChatMessageComponent
          role={message.role}
          content={message.content}
          reasoning_content={message.reasoning_content}
          reasoningTimeMs={message.reasoningTimeMs}
          sources={message.sources}
          isStreaming={$chatIsStreamingStore && idx === $chatMessagesStore.length - 1 && message.role === "assistant"}
          isReasoning={$chatIsReasoningStore && idx === $chatMessagesStore.length - 1 && message.role === "assistant"}
          onEdit={message.role === "user" ? (newContent) => editMessage(idx, newContent) : undefined}
          onDelete={message.role === "user" ? () => deleteMessage(idx) : undefined}
          onRegenerate={message.role === "assistant" && idx > 0 && $chatMessagesStore[idx - 1].role === "user"
            ? () => regenerateFromIndex(idx - 1, $selectedModelStore, $systemPromptStore, currentSamplingSettings())
            : undefined}
        />
      {/each}
    {/if}
  </div>

  <!-- Input area -->
  <div class="shrink-0">
      <!-- Attachment strip -->
      {#if attachments.length > 0}
        <div class="mb-2 flex flex-wrap gap-2">
          {#each attachments as attachment (attachment.id)}
            <div class="group flex items-center gap-2 rounded border border-gray-200 dark:border-white/10 bg-surface px-2 py-1 text-xs max-w-[340px]">
              <span class="shrink-0 text-[10px] text-txtsecondary">{attachment.kind === "image" ? "IMG" : "FILE"}</span>
              <div class="truncate">
                <div class="truncate font-medium">{attachment.name}</div>
                <div class="text-txtsecondary">{Math.max(1, Math.round(attachment.size / 1024))} KB</div>
              </div>
              <button
                class="ml-auto rounded px-1 text-txtsecondary hover:text-red-500"
                onclick={() => removeAttachment(attachment.id)}
                title="Remove attachment"
              >×</button>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Error message -->
      {#if attachmentError}
        <div class="mb-2 p-2 bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded text-sm">
          {attachmentError}
        </div>
      {/if}

      <div
        role="group"
        aria-label="Message input and attachments"
        class={`flex gap-2 rounded ${isDraggingFiles ? "ring-2 ring-primary/60 bg-secondary-hover" : ""}`}
        ondragover={handleDragOver}
        ondragleave={handleDragLeave}
        ondrop={handleDrop}
      >
        <!-- Hidden file input -->
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.gif,.webp,.txt,.md,.json,.yaml,.yml,.toml,.csv,.log,.js,.ts,.tsx,.jsx,.py,.go,.java,.c,.cpp,.h,.hpp,.rs,.rb,.php,.sh,.sql,.xml,.html,.css,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.zip"
          multiple
          class="hidden"
          bind:this={fileInput}
          onchange={handleFileSelect}
        />

        <ExpandableTextarea
          bind:value={userInput}
          placeholder="Type a message..."
          rows={3}
          onkeydown={handleKeyDown}
          disabled={$chatIsStreamingStore || !$selectedModelStore}
        />
        <div class="flex flex-col gap-2">
          {#if $chatIsStreamingStore}
            <button
              class="btn bg-red-500 hover:bg-red-600 text-white px-3"
              onclick={cancelStreaming}
              title="Stop generation"
              aria-label="Stop generation"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4">
                <path d="M7 7.75A.75.75 0 0 1 7.75 7h8.5a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75h-8.5a.75.75 0 0 1-.75-.75v-8.5Z" />
              </svg>
            </button>
          {:else}
            <button
              class="btn px-3"
              onclick={() => fileInput?.click()}
              disabled={$chatIsStreamingStore || !$selectedModelStore}
              title="Add files or images"
              aria-label="Add files or images"
            >
              +
            </button>
            <button
              class="btn bg-primary text-btn-primary-text hover:opacity-90 px-3"
              onclick={sendMessage}
              disabled={(!userInput.trim() && attachments.length === 0) || !$selectedModelStore}
              title="Send message"
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4">
                <path d="M3.32 2.43a.75.75 0 0 1 .78-.1l16.5 8.25a.75.75 0 0 1 0 1.34L4.1 20.17a.75.75 0 0 1-1.06-.83l1.44-6.1a.75.75 0 0 1 .55-.55l8.88-2.22-8.88-2.22a.75.75 0 0 1-.55-.55l-1.44-6.1a.75.75 0 0 1 .28-.77Z" />
              </svg>
            </button>
          {/if}
        </div>
      </div>
  </div>
</div>
