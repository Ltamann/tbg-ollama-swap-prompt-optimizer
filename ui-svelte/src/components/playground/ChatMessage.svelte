<script lang="ts">
  import { renderMarkdown, escapeHtml } from "../../lib/markdown";
  import { Copy, Check, Pencil, X, Save, RefreshCw, ChevronDown, ChevronRight, Brain, Code, Trash2 } from "lucide-svelte";
  import { getTextContent, getImageUrls } from "../../lib/types";
  import type { ContentPart, ChatSource } from "../../lib/types";

  interface Props {
    role: "user" | "assistant" | "system";
    content: string | ContentPart[];
    sources?: ChatSource[];
    reasoning_content?: string;
    reasoningTimeMs?: number;
    isStreaming?: boolean;
    isReasoning?: boolean;
    onEdit?: (newContent: string) => void;
    onDelete?: () => void;
    onRegenerate?: () => void;
  }

  let { role, content, sources = [], reasoning_content = "", reasoningTimeMs = 0, isStreaming = false, isReasoning = false, onEdit, onDelete, onRegenerate }: Props = $props();

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

  function formatDuration(ms: number): string {
    if (ms < 1000) {
      return `${ms.toFixed(0)}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
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

<div class="flex {role === 'user' ? 'justify-end' : 'justify-start'} mb-4">
  <div
    class="relative group max-w-[85%] rounded-lg px-4 py-2 {role === 'user'
      ? 'bg-primary text-btn-primary-text'
      : 'bg-surface border border-gray-200 dark:border-white/10'}"
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
      {#if !isStreaming}
        {#if hasSources}
          <div class="sources-tray mt-2 mb-1" title="Sources">
            {#each sources as source, idx (source.url + idx)}
              <a
                class="source-badge"
                href={source.url}
                target="_blank"
                rel="noreferrer noopener"
                title={sourceLabel(source)}
              >
                <img src={sourceFavicon(source)} alt={sourceDomain(source)} />
              </a>
            {/each}
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

  .sources-tray {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    overflow: hidden;
    max-width: 30px;
    transition: max-width 220ms ease;
  }

  .sources-tray:hover {
    max-width: 420px;
  }

  .source-badge {
    width: 24px;
    height: 24px;
    min-width: 24px;
    border-radius: 9999px;
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    background: var(--color-surface);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    margin-left: -6px;
    transition: margin-left 220ms ease, transform 120ms ease;
  }

  .source-badge:first-child {
    margin-left: 0;
  }

  .sources-tray:hover .source-badge {
    margin-left: 0;
  }

  .source-badge:hover {
    transform: translateY(-1px);
  }

  .source-badge img {
    width: 14px;
    height: 14px;
  }
</style>
