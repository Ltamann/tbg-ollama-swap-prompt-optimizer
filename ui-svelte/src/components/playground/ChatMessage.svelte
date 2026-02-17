<script lang="ts">
  import { onDestroy } from "svelte";
  import { renderMarkdown, escapeHtml } from "../../lib/markdown";
  import { Copy, Check, Pencil, X, Save, RefreshCw, ChevronDown, ChevronRight, Brain, Code, Trash2, Globe } from "lucide-svelte";
  import { getTextContent, getImageUrls } from "../../lib/types";
  import type { ContentPart, ChatSource } from "../../lib/types";

  interface Props {
    role: "user" | "assistant" | "system";
    content: string | ContentPart[];
    sources?: ChatSource[];
    reasoning_content?: string;
    reasoningTimeMs?: number;
    promptTokens?: number;
    promptTokensPerSecond?: number;
    generationTokens?: number;
    generationTokensPerSecond?: number;
    totalDurationMs?: number;
    isStreaming?: boolean;
    isReasoning?: boolean;
    onEdit?: (newContent: string) => void;
    onDelete?: () => void;
    onRegenerate?: () => void;
  }

  let {
    role,
    content,
    sources = [],
    reasoning_content = "",
    reasoningTimeMs = 0,
    promptTokens = 0,
    promptTokensPerSecond = 0,
    generationTokens = 0,
    generationTokensPerSecond = 0,
    totalDurationMs = 0,
    isStreaming = false,
    isReasoning = false,
    onEdit,
    onDelete,
    onRegenerate,
  }: Props = $props();

  let textContent = $derived(getTextContent(content));
  let imageUrls = $derived(getImageUrls(content));
  let hasImages = $derived(imageUrls.length > 0);
  let hasSources = $derived(sources.length > 0);
  let canEdit = $derived(onEdit !== undefined && !hasImages);
  let canDelete = $derived(onDelete !== undefined);

  let renderedContent = $derived(
    role === "assistant" && !isStreaming
      ? renderMarkdown(textContent)
      : escapeHtml(textContent).replace(/\n/g, '<br>')
  );
  let copied = $state(false);
  let showRaw = $state(false);
  let isEditing = $state(false);
  let editContent = $state("");
  let showReasoning = $state(false);
  let modalImageUrl = $state<string | null>(null);
  let showSourcesPopup = $state(false);
  let sourcesPopupEl = $state<HTMLDivElement | null>(null);

  function formatDuration(ms: number): string {
    if (ms < 1000) {
      return `${ms.toFixed(0)}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function formatSpeed(tokensPerSecond: number): string {
    if (!Number.isFinite(tokensPerSecond) || tokensPerSecond <= 0) {
      return "0.0";
    }
    return tokensPerSecond >= 100 ? tokensPerSecond.toFixed(0) : tokensPerSecond.toFixed(1);
  }

  async function copyToClipboard() {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(textContent);
      } else {
        // Fallback for non-secure contexts (HTTP)
        const textarea = document.createElement("textarea");
        textarea.value = textContent;
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }

  function startEdit() {
    editContent = textContent;
    isEditing = true;
  }

  function cancelEdit() {
    isEditing = false;
    editContent = "";
  }

  function saveEdit() {
    if (onEdit && editContent.trim() !== textContent) {
      onEdit(editContent.trim());
    }
    isEditing = false;
    editContent = "";
  }

  function openModal(imageUrl: string) {
    modalImageUrl = imageUrl;
    document.body.style.overflow = "hidden";
  }

  function closeModal(event?: MouseEvent) {
    // Only close if clicking the background, not the image
    if (event && event.target !== event.currentTarget) {
      return;
    }
    modalImageUrl = null;
    document.body.style.overflow = "";
  }

  function handleModalKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      closeModal();
    }
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      saveEdit();
    } else if (event.key === "Escape") {
      cancelEdit();
    }
  }

  function toggleSourcesPopup() {
    showSourcesPopup = !showSourcesPopup;
  }

  function onDocumentClick(event: MouseEvent) {
    if (!showSourcesPopup) return;
    const target = event.target as Node | null;
    if (!target) return;
    if (sourcesPopupEl && !sourcesPopupEl.contains(target)) {
      showSourcesPopup = false;
    }
  }

  if (typeof document !== "undefined") {
    document.addEventListener("click", onDocumentClick);
  }

  onDestroy(() => {
    if (typeof document !== "undefined") {
      document.removeEventListener("click", onDocumentClick);
    }
  });

  function sourceLabel(source: ChatSource): string {
    return source.title || source.domain || source.url;
  }

  function sourceDomain(source: ChatSource): string {
    if (source.domain) return source.domain;
    try {
      return new URL(source.url).hostname;
    } catch {
      return source.url;
    }
  }

  function sourceFavicon(source: ChatSource): string {
    const d = sourceDomain(source);
    return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(d)}&sz=32`;
  }
</script>

<div class="flex {role === 'user' ? 'justify-end' : 'justify-start'} mb-6">
  <div
    class="relative group w-full {role === 'user'
      ? 'max-w-[78%] rounded-[1.35rem] px-4 py-3 bg-surface border border-gray-200 dark:border-white/10 shadow-sm'
      : role === 'assistant'
        ? 'max-w-full rounded-xl px-0 py-0 bg-transparent border-0'
        : 'max-w-full rounded-xl px-4 py-3 bg-surface border border-gray-200 dark:border-white/10'}"
  >
    {#if role === "assistant"}
      {#if reasoning_content || isReasoning}
        <div class="mb-3 border border-gray-200 dark:border-white/10 rounded overflow-hidden">
          <button
            class="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-white/5 hover:bg-gray-100 dark:hover:bg-white/10 transition-colors text-sm"
            onclick={() => showReasoning = !showReasoning}
          >
            {#if showReasoning}
              <ChevronDown class="w-4 h-4" />
            {:else}
              <ChevronRight class="w-4 h-4" />
            {/if}
            <Brain class="w-4 h-4" />
            <span class="font-medium">Reasoning</span>
            <span class="text-txtsecondary ml-2">
              ({reasoning_content.length} chars{#if !isReasoning && reasoningTimeMs > 0}, {formatDuration(reasoningTimeMs)}{/if})
            </span>
            {#if isReasoning}
              <span class="ml-auto flex items-center gap-1 text-txtsecondary">
                <span class="w-1.5 h-1.5 bg-primary rounded-full animate-pulse"></span>
                reasoning...
              </span>
            {/if}
          </button>
          {#if showReasoning}
            <div class="px-3 py-2 bg-gray-50/50 dark:bg-white/[0.02] text-sm text-txtsecondary whitespace-pre-wrap font-mono">
              {reasoning_content}{#if isReasoning}<span class="inline-block w-1.5 h-4 bg-current animate-pulse ml-0.5"></span>{/if}
            </div>
          {/if}
        </div>
      {/if}
      {#if hasImages}
        <div class="mb-3 flex flex-wrap gap-2">
          {#each imageUrls as imageUrl, idx (idx)}
            <button
              onclick={() => openModal(imageUrl)}
              class="cursor-pointer rounded border border-gray-200 dark:border-white/10 hover:opacity-80 transition-opacity"
            >
              <img
                src={imageUrl}
                alt="Image {idx + 1}"
                class="max-h-64 rounded"
              />
            </button>
          {/each}
        </div>
      {/if}
      {#if showRaw}
        <div class="whitespace-pre-wrap font-mono text-sm">{textContent}</div>
      {:else}
        <div class="prose prose-sm dark:prose-invert max-w-none">
          {@html renderedContent}
          {#if isStreaming && !isReasoning}
            <span class="inline-block w-2 h-4 bg-current animate-pulse ml-0.5"></span>
          {/if}
        </div>
      {/if}
      {#if generationTokens > 0 || promptTokens > 0 || isStreaming}
        <div class="mt-2 text-[11px] font-mono text-txtsecondary">
          {#if promptTokens > 0}
            <span>prompt {promptTokens} tok @ {formatSpeed(promptTokensPerSecond)} tok/s</span>
            <span class="mx-1">|</span>
          {/if}
          <span>gen {Math.max(0, generationTokens)} tok @ {formatSpeed(generationTokensPerSecond)} tok/s</span>
          {#if totalDurationMs > 0}
            <span class="mx-1">|</span>
            <span>{formatDuration(totalDurationMs)}</span>
          {/if}
          {#if isStreaming}
            <span class="ml-2">live</span>
          {/if}
        </div>
      {/if}
      {#if !isStreaming}
        {#if hasSources}
          <div class="sources-wrap mt-2 mb-1" bind:this={sourcesPopupEl}>
            <button class="source-www-btn" title="Sources" onclick={toggleSourcesPopup}>
              <Globe class="w-3.5 h-3.5" />
            </button>
            {#if showSourcesPopup}
              <div class="sources-popup">
                <div class="sources-popup-title">Sources</div>
                {#each sources as source, idx (source.url + idx)}
                  <a
                    class="sources-popup-link"
                    href={source.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    title={sourceLabel(source)}
                  >
                    <img src={sourceFavicon(source)} alt={sourceDomain(source)} />
                    <span>{sourceLabel(source)}</span>
                  </a>
                {/each}
              </div>
            {/if}
          </div>
        {/if}
        <div class="flex gap-1 mt-2 pt-1 border-t border-gray-200 dark:border-white/10">
          {#if onRegenerate}
            <button
              class="p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 text-txtsecondary"
              onclick={onRegenerate}
              title="Regenerate response"
            >
              <RefreshCw class="w-4 h-4" />
            </button>
          {/if}
          <button
            class="p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 text-txtsecondary"
            onclick={copyToClipboard}
            title={copied ? "Copied!" : "Copy to clipboard"}
          >
            {#if copied}
              <Check class="w-4 h-4 text-green-500" />
            {:else}
              <Copy class="w-4 h-4" />
            {/if}
          </button>
          <button
            class="p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 {showRaw ? 'text-primary' : 'text-txtsecondary'}"
            onclick={() => showRaw = !showRaw}
            title={showRaw ? "Show rendered" : "Show raw"}
          >
            <Code class="w-4 h-4" />
          </button>
        </div>
      {/if}
    {:else}
      {#if isEditing}
        <div class="flex flex-col gap-2 min-w-[300px]">
          <textarea
            class="w-full px-3 py-2 rounded border border-gray-200 dark:border-white/10 bg-surface text-txtmain focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            rows="3"
            bind:value={editContent}
            onkeydown={handleKeyDown}
          ></textarea>
          <div class="flex justify-end gap-2">
            <button
              class="p-1.5 rounded hover:bg-white/20"
              onclick={cancelEdit}
              title="Cancel"
            >
              <X class="w-4 h-4" />
            </button>
            <button
              class="p-1.5 rounded hover:bg-white/20"
              onclick={saveEdit}
              title="Save"
            >
              <Save class="w-4 h-4" />
            </button>
          </div>
        </div>
      {:else}
        {#if hasImages}
          <div class="mb-2 flex flex-wrap gap-2">
            {#each imageUrls as imageUrl, idx (idx)}
              <button
                onclick={() => openModal(imageUrl)}
                class="cursor-pointer rounded border border-white/20 hover:opacity-80 transition-opacity"
              >
                <img
                  src={imageUrl}
                  alt="Image {idx + 1}"
                  class="max-w-[200px] rounded"
                />
              </button>
            {/each}
          </div>
        {/if}
        <div class="whitespace-pre-wrap pr-8">{textContent}</div>
        {#if canEdit || canDelete}
          <div class="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {#if canEdit}
              <button
                class="p-1.5 rounded-lg bg-white/20 hover:bg-white/30 shadow-sm"
                onclick={startEdit}
                title="Edit message"
              >
                <Pencil class="w-4 h-4" />
              </button>
            {/if}
            {#if canDelete}
              <button
                class="p-1.5 rounded-lg bg-white/20 hover:bg-red-500/80 shadow-sm"
                onclick={onDelete}
                title="Delete user message and related answer"
              >
                <Trash2 class="w-4 h-4" />
              </button>
            {/if}
          </div>
        {/if}
      {/if}
    {/if}
  </div>
</div>

<!-- Full-size image modal -->
{#if modalImageUrl}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    onclick={(e) => closeModal(e)}
    onkeydown={handleModalKeyDown}
    role="button"
    tabindex="-1"
  >
    <button
      class="absolute top-4 right-4 p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
      onclick={() => closeModal()}
      title="Close"
    >
      <X class="w-6 h-6" />
    </button>
    <img
      src={modalImageUrl}
      alt=""
      class="max-w-full max-h-full rounded pointer-events-none"
    />
  </div>
{/if}

<style>
  .prose :global(pre) {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    border-radius: 0.375rem;
    padding: 0.75rem;
    overflow-x: auto;
    margin: 0.5rem 0;
  }

  .prose :global(code) {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.875em;
  }

  .prose :global(pre code) {
    background: none;
    padding: 0;
  }

  .prose :global(code:not(pre code)) {
    background-color: var(--color-surface);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
  }

  .prose :global(p) {
    margin: 0.5rem 0;
  }

  .prose :global(p:first-child) {
    margin-top: 0;
  }

  .prose :global(p:last-child) {
    margin-bottom: 0;
  }

  .prose :global(ul),
  .prose :global(ol) {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
  }

  .prose :global(li) {
    margin: 0.25rem 0;
  }

  .prose :global(h1),
  .prose :global(h2),
  .prose :global(h3),
  .prose :global(h4) {
    margin: 1rem 0 0.5rem 0;
    font-weight: 600;
  }

  .prose :global(h1:first-child),
  .prose :global(h2:first-child),
  .prose :global(h3:first-child),
  .prose :global(h4:first-child) {
    margin-top: 0;
  }

  .prose :global(blockquote) {
    border-left: 3px solid var(--color-primary);
    padding-left: 1rem;
    margin: 0.5rem 0;
    font-style: italic;
  }

  .prose :global(a) {
    color: var(--color-primary);
    text-decoration: underline;
  }

  .prose :global(table) {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
  }

  .prose :global(th),
  .prose :global(td) {
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    padding: 0.5rem;
    text-align: left;
  }

  .prose :global(th) {
    background-color: var(--color-surface);
    font-weight: 600;
  }

  /* Highlight.js theme overrides for dark mode */
  :global(.dark) .prose :global(.hljs) {
    background: transparent;
  }

  .sources-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
  }

  .source-www-btn {
    width: 24px;
    height: 24px;
    min-width: 24px;
    border-radius: 9999px;
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    background: var(--color-surface);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-txtsecondary);
    cursor: pointer;
  }

  .source-www-btn:hover {
    color: var(--color-txtmain);
  }

  .sources-popup {
    position: absolute;
    left: 0;
    bottom: 30px;
    z-index: 15;
    min-width: 260px;
    max-width: 420px;
    max-height: 240px;
    overflow: auto;
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    background: var(--color-surface);
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  }

  .sources-popup-title {
    font-size: 11px;
    color: var(--color-txtsecondary);
    margin-bottom: 6px;
  }

  .sources-popup-link {
    display: flex;
    align-items: center;
    gap: 8px;
    text-decoration: none;
    color: var(--color-txtmain);
    padding: 6px;
    border-radius: 8px;
    font-size: 12px;
  }

  .sources-popup-link:hover {
    background: color-mix(in srgb, var(--color-surface), var(--color-txtmain) 7%);
  }

  .sources-popup-link img {
    width: 14px;
    height: 14px;
    flex: 0 0 auto;
  }

  .sources-popup-link span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
