<script lang="ts">
  import { models } from "../../stores/api";
  import { persistentStore } from "../../stores/persistent";
  import {
    chatMessagesStore,
    chatIsStreamingStore,
    chatIsReasoningStore,
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
  const temperatureStore = persistentStore<number>("playground-temperature", 0.7);
  let userInput = $state("");
  let messagesContainer: HTMLDivElement | undefined = $state();
  let showSettings = $state(false);
  let attachedImages = $state<string[]>([]);
  let fileInput = $state<HTMLInputElement | null>(null);
  let imageError = $state<string | null>(null);

  let hasModels = $derived($models.some((m) => !m.unlisted));

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
    const sent = await sendUserMessage(userInput, attachedImages, $selectedModelStore, $systemPromptStore, $temperatureStore);
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
    await editUserMessage(idx, newContent, $selectedModelStore, $systemPromptStore, $temperatureStore);
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

<div class="flex flex-col h-full">
  <!-- Model selector and controls -->
  <div class="shrink-0 flex flex-wrap gap-2 mb-4">
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

  <!-- Settings panel -->
  {#if showSettings}
    <div class="shrink-0 mb-4 p-4 bg-surface border border-gray-200 dark:border-white/10 rounded">
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
          Temperature: {$temperatureStore.toFixed(2)}
        </label>
        <input
          id="temperature"
          type="range"
          min="0"
          max="2"
          step="0.05"
          class="w-full"
          bind:value={$temperatureStore}
          disabled={$chatIsStreamingStore}
        />
        <div class="flex justify-between text-xs text-txtsecondary mt-1">
          <span>Precise (0)</span>
          <span>Creative (2)</span>
        </div>
      </div>
    </div>
  {/if}

  <!-- Empty state for no models configured -->
  {#if !hasModels}
    <div class="flex-1 flex items-center justify-center text-txtsecondary">
      <p>No models configured. Add models to your configuration to start chatting.</p>
    </div>
  {:else}
    <!-- Messages area -->
    <div
      class="flex-1 overflow-y-auto mb-4 px-2"
      bind:this={messagesContainer}
    >
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
              ? () => regenerateFromIndex(idx - 1, $selectedModelStore, $systemPromptStore, $temperatureStore)
              : undefined}
          />
        {/each}
      {/if}
    </div>

    <!-- Input area -->
    <div class="shrink-0">
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

      <div class="flex gap-2">
        <!-- Hidden file input -->
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.gif,.webp"
          multiple
          class="hidden"
          bind:this={fileInput}
          onchange={handleImageSelect}
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
            <button class="btn bg-red-500 hover:bg-red-600 text-white" onclick={cancelStreaming}>
              Cancel
            </button>
          {:else}
            <button
              class="btn"
              onclick={() => fileInput?.click()}
              disabled={$chatIsStreamingStore || !$selectedModelStore}
              title="Attach image"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
                <path fill-rule="evenodd" d="M1 5.25A2.25 2.25 0 0 1 3.25 3h13.5A2.25 2.25 0 0 1 19 5.25v9.5A2.25 2.25 0 0 1 16.75 17H3.25A2.25 2.25 0 0 1 1 14.75v-9.5Zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 0 0 .75-.75v-2.69l-2.22-2.219a.75.75 0 0 0-1.06 0l-1.91 1.909.47.47a.75.75 0 1 1-1.06 1.06L6.53 8.091a.75.75 0 0 0-1.06 0l-2.97 2.97ZM12 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z" clip-rule="evenodd" />
              </svg>
            </button>
            <button
              class="btn bg-primary text-btn-primary-text hover:opacity-90"
              onclick={sendMessage}
              disabled={(!userInput.trim() && attachedImages.length === 0) || !$selectedModelStore}
            >
              Send
            </button>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>
