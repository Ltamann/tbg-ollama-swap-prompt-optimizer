<script lang="ts">
  import { liveChatEvents, liveMonitorEvents } from "../stores/api";
  import type { LiveChatEvent, LiveChatTimelineEntry, LiveMonitorEvent } from "../lib/types";
  import { renderMarkdown } from "../lib/markdown";
  import { Copy, Check, ChevronRight, Brain, Wrench, TerminalSquare, FileText } from "lucide-svelte";

  let events = $state<LiveChatEvent[]>([]);
  let monitor = $state<LiveMonitorEvent[]>([]);
  let scrollContainer: HTMLDivElement | undefined = undefined;
  let isAtBottom = $state(true);
  let copiedId = $state<number | null>(null);
  let reasoningStates = $state<Record<number, boolean>>({});
  let timelineExpandStates = $state<Record<string, boolean>>({});
  let monitorExpandStates = $state<Record<number, boolean>>({});
  let monitorByTrace = $state<Record<string, LiveMonitorEvent[]>>({});
  let userMessagesByEvent = $state<Record<number, Array<{ role: string; content: string }>>>({});
  let systemMessagesByEvent = $state<Record<number, Array<{ role: string; content: string }>>>({});
  let recentMonitorItems = $state<LiveMonitorEvent[]>([]);

  $effect(() => {
    events = [...$liveChatEvents];
  });

  $effect(() => {
    monitor = [...$liveMonitorEvents];
  });

  $effect(() => {
    const grouped: Record<string, LiveMonitorEvent[]> = {};
    for (const item of monitor) {
      const trace = item.trace_id || "";
      if (!trace) continue;
      if (!grouped[trace]) grouped[trace] = [];
      grouped[trace].push(item);
    }
    for (const trace of Object.keys(grouped)) {
      const items = grouped[trace];
      grouped[trace] = items.length > 160 ? items.slice(-160) : items;
    }
    monitorByTrace = grouped;
    recentMonitorItems = monitor.length > 160 ? monitor.slice(-160) : monitor;
  });

  $effect(() => {
    const users: Record<number, Array<{ role: string; content: string }>> = {};
    const systems: Record<number, Array<{ role: string; content: string }>> = {};
    for (const event of events) {
      users[event.id] = event.messages.filter((m) => m.role === "user");
      systems[event.id] = event.messages.filter((m) => m.role === "system");
    }
    userMessagesByEvent = users;
    systemMessagesByEvent = systems;
  });

  function checkBottom(): void {
    if (!scrollContainer) return;
    const threshold = 80;
    isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop <= scrollContainer.clientHeight + threshold;
  }

  function scrollToBottom(smooth: boolean): void {
    if (!scrollContainer) return;
    scrollContainer.scrollTo({
      top: scrollContainer.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }

  $effect(() => {
    if (isAtBottom && events.length > 0 && scrollContainer) {
      scrollToBottom(true);
    }
  });

  $effect(() => {
    if (scrollContainer) {
      const handler = () => checkBottom();
      scrollContainer.addEventListener("scroll", handler);
      return function cleanup() {
        scrollContainer?.removeEventListener("scroll", handler);
      };
    }
  });

  function formatDuration(ms: number): string {
    if (ms < 1000) return ms + "ms";
    return (ms / 1000).toFixed(1) + "s";
  }

  function formatTime(timestamp: string): string {
    const d = new Date(timestamp);
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  async function copyText(text: string, eventId: number): Promise<void> {
    try { await navigator.clipboard.writeText(text); } catch { /* ignore */ }
    copiedId = eventId;
    setTimeout(function() { copiedId = null; }, 2000);
  }

  function toggleReasoning(eventId: number): void {
    reasoningStates = { ...reasoningStates, [eventId]: !reasoningStates[eventId] };
  }

  function getReasoningState(eventId: number): boolean {
    return !!reasoningStates[eventId];
  }

  function toggleTimelineExpand(eventId: number, idx: number): void {
    const key = `${eventId}:${idx}`;
    timelineExpandStates = { ...timelineExpandStates, [key]: !timelineExpandStates[key] };
  }

  function isTimelineExpanded(eventId: number, idx: number): boolean {
    return !!timelineExpandStates[`${eventId}:${idx}`];
  }

  // Pre-compute all values we need from an event so the template stays simple
  function evtContent(evt: LiveChatEvent): string {
    var r = evt.assistant_response;
    return r ? (r.content || "") : "";
  }

  function evtReasoning(evt: LiveChatEvent): string {
    var r = evt.assistant_response;
    return r ? (r.reasoning_content || "") : "";
  }

  function evtStopReason(evt: LiveChatEvent): string | undefined {
    var r = evt.assistant_response;
    return r ? r.stop_reason : undefined;
  }

  function evtHasContent(evt: LiveChatEvent): boolean {
    var c = evtContent(evt);
    var rc = evtReasoning(evt);
    return !!(c || rc || (evt.timeline && evt.timeline.length > 0));
  }

  function evtTimeline(evt: LiveChatEvent): LiveChatTimelineEntry[] {
    return evt.timeline || [];
  }

  function evtTraceID(evt: LiveChatEvent): string {
    return evt.trace_id || "";
  }

  function toggleMonitor(eventId: number): void {
    monitorExpandStates = { ...monitorExpandStates, [eventId]: !monitorExpandStates[eventId] };
  }

  function isMonitorExpanded(eventId: number): boolean {
    return !!monitorExpandStates[eventId];
  }

  function monitorForEvent(evt: LiveChatEvent): LiveMonitorEvent[] {
    const trace = evtTraceID(evt);
    if (!trace) return [];
    return monitorByTrace[trace] || [];
  }

  function recentMonitor(): LiveMonitorEvent[] {
    return recentMonitorItems;
  }

  function userMessagesForEvent(eventId: number): Array<{ role: string; content: string }> {
    return userMessagesByEvent[eventId] || [];
  }

  function systemMessagesForEvent(eventId: number): Array<{ role: string; content: string }> {
    return systemMessagesByEvent[eventId] || [];
  }
</script>

<div class="h-full flex flex-col">
  <div class="shrink-0 px-4 py-2.5 border-b border-border/50 bg-surface/90 backdrop-blur">
    <div class="max-w-4xl mx-auto w-full flex items-center gap-3">
      <div class="flex items-center gap-2">
        <div class="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
        <h2 class="text-sm font-semibold text-txtmain">Live</h2>
      </div>
      <span class="text-xs text-txtsecondary hidden sm:inline">Real-time chat stream</span>
      <span class="ml-auto text-xs text-txtsecondary tabular-nums">
        {events.length} turn{events.length !== 1 ? "s" : ""}
      </span>
    </div>
  </div>

  <div class="flex-1 overflow-y-auto" bind:this={scrollContainer}>
    <div class="max-w-4xl mx-auto w-full py-4 px-4 space-y-6">
      {#if monitor.length > 0}
        <details class="border border-gray-200 dark:border-white/10 rounded bg-surface p-3">
          <summary class="cursor-pointer text-xs font-semibold text-txtsecondary">
            Monitor feed ({monitor.length})
          </summary>
          <div class="mt-2 border border-gray-200 dark:border-white/10 rounded bg-surface max-h-[240px] overflow-auto">
            {#each recentMonitor() as m, idx (m.timestamp + ':' + idx)}
              <div class="px-3 py-2 border-b border-gray-200 dark:border-white/10 text-[11px] font-mono">
                <div class="flex flex-wrap gap-x-2 gap-y-1 text-txtsecondary">
                  <span class="opacity-70">{formatTime(m.timestamp)}</span>
                  <span class="opacity-50">trace {m.trace_id.slice(-8)}</span>
                  <span class="text-primary">{m.stage}</span>
                  <span>{m.direction}</span>
                  {#if m.endpoint}<span class="opacity-70">{m.endpoint}</span>{/if}
                  {#if m.event}<span class="opacity-70">{m.event}</span>{/if}
                  {#if m.truncated}<span class="opacity-70">(truncated)</span>{/if}
                </div>
                {#if m.data}
                  <pre class="mt-1 whitespace-pre-wrap break-all text-txtsecondary">{m.data}</pre>
                {/if}
              </div>
            {/each}
          </div>
        </details>
      {/if}

      {#each events as event (event.id)}
        {#if evtHasContent(event) || event.messages.length > 0}
          <div class="space-y-4">
            <div class="flex items-center gap-2 text-[11px] text-txtsecondary font-mono select-none">
              <span class="opacity-60">{formatTime(event.timestamp)}</span>
              <span class="text-primary font-semibold">⚡ {event.model}</span>
              {#if event.input_tokens > 0 || event.output_tokens > 0}
                <span class="opacity-70">{event.input_tokens}↑ {event.output_tokens}↓</span>
              {/if}
              {#if event.duration_ms > 0}
                <span class="opacity-70">· {formatDuration(event.duration_ms)}</span>
              {/if}
              {#if event.cached_tokens > 0}
                <span class="text-green-500 opacity-70">· {event.cached_tokens} cached</span>
              {/if}
              {#if evtTraceID(event)}
                <span class="opacity-50">· trace {evtTraceID(event).slice(-8)}</span>
              {/if}
            </div>

            <!-- User messages -->
            {#each userMessagesForEvent(event.id) as msg (msg.content + event.id)}
              <div class="flex justify-end">
                <div class="max-w-[78%] rounded-[1.35rem] px-4 py-3 bg-surface border border-gray-200 dark:border-white/10 shadow-sm">
                  <div class="whitespace-pre-wrap text-txtmain text-sm">{msg.content}</div>
                </div>
              </div>
            {/each}

            <!-- System messages -->
            {#each systemMessagesForEvent(event.id) as msg (msg.content + event.id)}
              <div class="flex justify-start">
                <div class="max-w-full rounded-xl px-4 py-3 bg-surface border border-gray-200 dark:border-white/10 shadow-sm">
                  <div class="whitespace-pre-wrap text-txtsecondary text-sm">{msg.content}</div>
                </div>
              </div>
            {/each}

            <!-- Assistant response -->
            {#if event.assistant_response}
              <div class="flex justify-start">
                <div class="max-w-full rounded-xl px-0 py-0 bg-transparent border-0">

                  <!-- Reasoning section -->
                  {#if evtReasoning(event) && evtReasoning(event).length > 5}
                    <div class="mb-3 border border-gray-200 dark:border-white/10 rounded overflow-hidden">
                      <button
                        class="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-white/5 hover:bg-gray-100 dark:hover:bg-white/10 transition-colors text-sm"
                        onclick={function() { toggleReasoning(event.id); }}
                      >
                        {#if getReasoningState(event.id)}
                          <ChevronRight class="w-4 h-4 rotate-90" />
                        {:else}
                          <ChevronRight class="w-4 h-4" />
                        {/if}
                        <Brain class="w-4 h-4" />
                        <span class="font-medium">Reasoning</span>
                        <span class="text-txtsecondary ml-2">({evtReasoning(event).length} chars)</span>
                      </button>
                      {#if getReasoningState(event.id)}
                        <div class="px-3 py-2 bg-gray-50/50 dark:bg-white/[0.02] text-sm text-txtsecondary whitespace-pre-wrap font-mono max-h-[300px] overflow-y-auto">
                          {@html renderMarkdown(evtReasoning(event))}
                        </div>
                      {/if}
                    </div>
                  {/if}

                  <!-- Main content -->
                  {#if evtContent(event)}
                    <div class="prose prose-sm dark:prose-invert max-w-none">
                      {@html renderMarkdown(evtContent(event))}
                    </div>
                  {/if}

                  <!-- Action bar -->
                  <div class="flex gap-1 mt-2 pt-1 border-t border-gray-200 dark:border-white/10">
                    <button
                      class="p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 text-txtsecondary"
                      onclick={function() { copyText(evtContent(event) || evtReasoning(event) || "", event.id); }}
                      title={copiedId === event.id ? "Copied!" : "Copy to clipboard"}
                    >
                      {#if copiedId === event.id}
                        <Check class="w-4 h-4 text-green-500" />
                      {:else}
                        <Copy class="w-4 h-4" />
                      {/if}
                    </button>
                  </div>
                </div>
              </div>
            {:else}
              <div class="flex justify-start">
                <div class="max-w-full rounded-xl px-4 py-2 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 text-sm text-txtsecondary">
                  Request completed with status {event.status}
                </div>
              </div>
            {/if}

            <!-- Timeline entries -->
            {#if evtTimeline(event).length > 0}
              <div class="space-y-2">
                {#each evtTimeline(event) as t, idx (`${event.id}-${idx}-${t.kind}`)}
                  {#if t.kind === "tool_call" || t.kind === "tool_args" || t.kind === "tool_output"}
                    <div class="border border-gray-200 dark:border-white/10 rounded bg-surface">
                      <button
                        class="w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-white/5 flex items-center gap-2"
                        onclick={function() { toggleTimelineExpand(event.id, idx); }}
                      >
                        <ChevronRight class={`w-4 h-4 ${isTimelineExpanded(event.id, idx) ? "rotate-90" : ""}`} />
                        {#if t.kind === "tool_output"}
                          <TerminalSquare class="w-4 h-4 text-primary" />
                        {:else}
                          <Wrench class="w-4 h-4 text-primary" />
                        {/if}
                        <span class="text-sm font-medium">
                          {t.title || (t.kind === "tool_output" ? "Tool Output" : "Tool Call")}
                          {#if t.tool_name} · {t.tool_name}{/if}
                        </span>
                        {#if t.call_id}
                          <span class="text-xs text-txtsecondary font-mono">{t.call_id}</span>
                        {/if}
                      </button>
                      <div class="px-3 pb-3 text-sm text-txtsecondary">
                        {#if t.kind === "tool_output" && t.output_preview}
                          {#if isTimelineExpanded(event.id, idx) && t.content}
                            <pre class="font-mono whitespace-pre-wrap break-all">{t.content}</pre>
                          {:else}
                            <pre class="font-mono whitespace-pre-wrap break-all">{t.output_preview}</pre>
                            {#if t.truncated}
                              <div class="text-xs mt-1">Preview truncated to 3 lines</div>
                            {/if}
                          {/if}
                        {:else if t.content}
                          {#if isTimelineExpanded(event.id, idx)}
                            <pre class="font-mono whitespace-pre-wrap break-all">{t.content}</pre>
                          {:else}
                            <pre class="font-mono whitespace-pre-wrap break-all">{t.content.length > 240 ? t.content.slice(0, 240) + "..." : t.content}</pre>
                          {/if}
                        {/if}
                      </div>
                    </div>
                  {:else if t.kind === "completion"}
                    <div class="text-xs text-txtsecondary flex items-center gap-2">
                      <FileText class="w-3.5 h-3.5" />
                      <span>Completed {t.status ? `(${t.status})` : ""}</span>
                    </div>
                  {/if}
                {/each}
              </div>
            {/if}

            {#if monitorForEvent(event).length > 0}
              <div class="space-y-2">
                <button class="btn btn--sm" onclick={function() { toggleMonitor(event.id); }}>
                  {isMonitorExpanded(event.id) ? "Hide" : "Show"} monitor ({monitorForEvent(event).length})
                </button>

                {#if isMonitorExpanded(event.id)}
                  <div class="border border-gray-200 dark:border-white/10 rounded bg-surface max-h-[240px] overflow-auto">
                    {#each monitorForEvent(event) as m, idx (m.timestamp + ':' + idx)}
                      <div class="px-3 py-2 border-b border-gray-200 dark:border-white/10 text-[11px] font-mono">
                        <div class="flex flex-wrap gap-x-2 gap-y-1 text-txtsecondary">
                          <span class="opacity-70">{formatTime(m.timestamp)}</span>
                          <span class="text-primary">{m.stage}</span>
                          <span>{m.direction}</span>
                          {#if m.endpoint}<span class="opacity-70">{m.endpoint}</span>{/if}
                          {#if m.event}<span class="opacity-70">{m.event}</span>{/if}
                          {#if m.truncated}<span class="opacity-70">(truncated)</span>{/if}
                        </div>
                        {#if m.data}
                          <pre class="mt-1 whitespace-pre-wrap break-all text-txtsecondary">{m.data}</pre>
                        {/if}
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/if}
      {/each}

      <!-- Scroll indicator -->
      {#if !isAtBottom && events.length > 0}
        <div class="flex justify-center py-2">
          <button
            class="btn text-sm px-4 py-1.5 flex items-center gap-2"
            onclick={function() { isAtBottom = true; scrollToBottom(true); }}
          >
            New activity
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4">
              <path fill-rule="evenodd" d="M12 2.25a.75.75 0 0 1 .75.75v16.19l6.22-6.22a.75.75 0 1 1 1.06 1.06l-7.5 7.5a.75.75 0 0 1-1.06 0l-7.5-7.5a.75.75 0 1 1 1.06-1.06l6.22 6.22V3a.75.75 0 0 1 .75-.75Z" clip-rule="evenodd" />
            </svg>
          </button>
        </div>
      {/if}

      {#if events.length === 0}
        <div class="flex-1 flex items-center justify-center text-txtsecondary py-20">
          <p class="text-sm">Waiting for chat activity...</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  :global(.prose pre) {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
    border-radius: 0.375rem;
    padding: 0.75rem;
    overflow-x: auto;
    margin: 0.5rem 0;
  }

  :global(.prose code) {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.875em;
  }

  :global(.prose pre code) { background: none; padding: 0; }

  :global(.prose code:not(pre code)) {
    background-color: var(--color-surface);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    border: 1px solid var(--color-border, rgba(128, 128, 128, 0.2));
  }

  :global(.prose p) { margin: 0.5rem 0; }
  :global(.prose p:first-child) { margin-top: 0; }
  :global(.prose p:last-child) { margin-bottom: 0; }
  :global(.prose ul), :global(.prose ol) { margin: 0.5rem 0; padding-left: 1.5rem; }
  :global(.prose li) { margin: 0.25rem 0; }
  :global(.prose h1), :global(.prose h2), :global(.prose h3), :global(.prose h4) { margin: 1rem 0 0.5rem 0; font-weight: 600; }
  :global(.prose a) { color: var(--color-primary); text-decoration: underline; }
  :global(.dark .prose .hljs) { background: transparent; }
</style>
