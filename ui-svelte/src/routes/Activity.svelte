<script lang="ts">
  import { onMount } from "svelte";
  import { metrics, getCapture, liveMonitorEvents } from "../stores/api";
  import Tooltip from "../components/Tooltip.svelte";
  import CaptureDialog from "../components/CaptureDialog.svelte";
  import type { Metrics, ReqRespCapture, LiveMonitorEvent } from "../lib/types";
  import { summarizeCaptureResponse, type CaptureSummary } from "../lib/captureSummary";

  type PromptFlowItem = {
    metric: Metrics;
    capture: ReqRespCapture | null;
    lastUserPrompt: string;
    promptPreview: string;
    requestPath: string;
    userAgent: string;
    stageNames: string[];
    responseSummary: CaptureSummary | null;
  };

  function formatSpeed(speed: number): string {
    return speed < 0 ? "unknown" : speed.toFixed(2) + " t/s";
  }

  function formatDuration(ms: number): string {
    return (ms / 1000).toFixed(2) + "s";
  }

  function formatRelativeTime(timestamp: string): string {
    const now = new Date();
    const date = new Date(timestamp);
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 5) {
      return "now";
    }
    if (diffInSeconds < 60) {
      return `${diffInSeconds}s ago`;
    }

    const diffInMinutes = Math.floor(diffInSeconds / 60);
    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    }

    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    }

    return "a while ago";
  }

  function decodeBody(body: string | null | undefined): string {
    if (!body) return "";
    try {
      const binary = atob(body);
      const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
      return new TextDecoder().decode(bytes);
    } catch {
      return body;
    }
  }

  function formatJson(str: string): string {
    try {
      return JSON.stringify(JSON.parse(str), null, 2);
    } catch {
      return str;
    }
  }

  function extractTextContent(content: unknown): string {
    if (typeof content === "string") {
      return content;
    }
    if (!Array.isArray(content)) {
      return "";
    }
    const parts: string[] = [];
    for (const part of content) {
      if (!part || typeof part !== "object") continue;
      const record = part as Record<string, unknown>;
      if (record.type === "text" && typeof record.text === "string") {
        parts.push(record.text);
      } else if (record.type === "input_text" && typeof record.text === "string") {
        parts.push(record.text);
      }
    }
    return parts.join("\n");
  }

  function extractLastUserPrompt(rawRequestBody: string): string {
    try {
      const parsed = JSON.parse(rawRequestBody) as Record<string, unknown>;

      const input = parsed.input;
      if (Array.isArray(input)) {
        let last = "";
        for (const item of input) {
          if (!item || typeof item !== "object") continue;
          const record = item as Record<string, unknown>;
          if (record.role === "user") {
            const text = extractTextContent(record.content);
            if (text) last = text;
          }
        }
        if (last) return last;
      }

      const messages = parsed.messages;
      if (Array.isArray(messages)) {
        let last = "";
        for (const item of messages) {
          if (!item || typeof item !== "object") continue;
          const record = item as Record<string, unknown>;
          if (record.role === "user") {
            const text = extractTextContent(record.content);
            if (text) {
              last = text;
            } else if (typeof record.content === "string") {
              last = record.content;
            }
          }
        }
        return last;
      }
    } catch {
      return "";
    }
    return "";
  }

  function buildPromptPreview(rawRequestBody: string): string {
    try {
      return formatJson(rawRequestBody);
    } catch {
      return rawRequestBody;
    }
  }

  let sortedMetrics = $derived([...$metrics].sort((a, b) => b.id - a.id));
  let promptFlow = $state<PromptFlowItem[]>([]);
  let loadingPromptFlow = $state(false);
  let groupedMonitor = $state<Record<string, LiveMonitorEvent[]>>({});

  let selectedCapture = $state<ReqRespCapture | null>(null);
  let dialogOpen = $state(false);
  let loadingCaptureId = $state<number | null>(null);

  async function viewCapture(id: number) {
    loadingCaptureId = id;
    const capture = await getCapture(id);
    loadingCaptureId = null;
    if (capture) {
      selectedCapture = capture;
      dialogOpen = true;
    }
  }

  async function refreshPromptFlow() {
    loadingPromptFlow = true;
    try {
      const candidates = sortedMetrics.filter((metric) => metric.has_capture).slice(0, 8);
      const captures = await Promise.all(candidates.map(async (metric) => ({ metric, capture: await getCapture(metric.id) })));
      promptFlow = captures.map(({ metric, capture }) => {
        const rawRequestBody = capture ? decodeBody(capture.req_body) : "";
        return {
          metric,
          capture,
          lastUserPrompt: extractLastUserPrompt(rawRequestBody),
          promptPreview: buildPromptPreview(rawRequestBody),
          requestPath: capture?.req_path || "",
          userAgent: capture?.req_headers?.["User-Agent"] || capture?.req_headers?.["user-agent"] || "",
          stageNames: capture?.stages?.map((stage) => stage.name) || [],
          responseSummary: capture ? summarizeCaptureResponse(decodeBody(capture.resp_body)) : null,
        };
      });
    } finally {
      loadingPromptFlow = false;
    }
  }

  function closeDialog() {
    dialogOpen = false;
    selectedCapture = null;
  }

  onMount(() => {
    void refreshPromptFlow();
  });

  $effect(() => {
    if ($metrics.length > 0) {
      void refreshPromptFlow();
    }
  });

  $effect(() => {
    const grouped: Record<string, LiveMonitorEvent[]> = {};
    for (const item of $liveMonitorEvents) {
      const trace = item.trace_id || "";
      if (!trace) continue;
      if (!grouped[trace]) grouped[trace] = [];
      grouped[trace].push(item);
    }
    for (const trace of Object.keys(grouped)) {
      const items = grouped[trace];
      grouped[trace] = items.length > 12 ? items.slice(-12) : items;
    }
    groupedMonitor = grouped;
  });
</script>

<div class="p-2">
  <h1 class="text-2xl font-bold">Activity</h1>

  {#if $metrics.length === 0}
    <div class="text-center py-8">
      <p class="text-gray-600">No metrics data available</p>
    </div>
  {:else}
    <div class="card overflow-auto mt-3 mb-4">
      <table class="min-w-full divide-y">
        <thead class="border-gray-200 dark:border-white/10">
          <tr class="text-left text-xs uppercase tracking-wider">
            <th class="px-6 py-3">ID</th>
            <th class="px-6 py-3">Time</th>
            <th class="px-6 py-3">Model</th>
            <th class="px-6 py-3">Status</th>
            <th class="px-6 py-3">
              Cached <Tooltip content="prompt tokens from cache" />
            </th>
            <th class="px-6 py-3">
              Prompt <Tooltip content="new prompt tokens processed" />
            </th>
            <th class="px-6 py-3">Generated</th>
            <th class="px-6 py-3">Prompt Processing</th>
            <th class="px-6 py-3">Generation Speed</th>
            <th class="px-6 py-3">Duration</th>
            <th class="px-6 py-3">Capture</th>
          </tr>
        </thead>
        <tbody class="divide-y">
          {#each sortedMetrics as metric (metric.id)}
            <tr class="whitespace-nowrap text-sm border-gray-200 dark:border-white/10">
              <td class="px-4 py-4">{metric.id + 1}</td>
              <td class="px-6 py-4">{formatRelativeTime(metric.timestamp)}</td>
              <td class="px-6 py-4">{metric.model}</td>
              <td class="px-6 py-4">{metric.status_code || 200}</td>
              <td class="px-6 py-4">{metric.cache_tokens > 0 ? metric.cache_tokens.toLocaleString() : "-"}</td>
              <td class="px-6 py-4">{metric.input_tokens > 0 ? metric.input_tokens.toLocaleString() : "-"}</td>
              <td class="px-6 py-4">{metric.output_tokens > 0 ? metric.output_tokens.toLocaleString() : "-"}</td>
              <td class="px-6 py-4">{formatSpeed(metric.prompt_per_second)}</td>
              <td class="px-6 py-4">{formatSpeed(metric.tokens_per_second)}</td>
              <td class="px-6 py-4">{formatDuration(metric.duration_ms)}</td>
              <td class="px-6 py-4">
                {#if metric.has_capture}
                  <button
                    onclick={() => viewCapture(metric.id)}
                    disabled={loadingCaptureId === metric.id}
                    class="btn btn--sm"
                  >
                    {loadingCaptureId === metric.id ? "..." : "View"}
                  </button>
                {:else}
                  <span class="text-txtsecondary">-</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <div class="card p-4 mt-3 mb-4">
    <div class="flex items-center justify-between mb-2">
      <h2 class="text-lg font-semibold">Latest Prompt Flow</h2>
      <button class="btn btn--sm" onclick={() => refreshPromptFlow()} disabled={loadingPromptFlow}>
        {loadingPromptFlow ? "..." : "Refresh"}
      </button>
    </div>
    <p class="text-xs text-txtsecondary mb-3">
      Latest captured requests sent through the proxy. Use this to inspect the exact prompt body Codex sent.
    </p>

    {#if promptFlow.length === 0}
      <p class="text-sm text-txtsecondary">No prompt activity yet.</p>
    {:else}
      <div class="space-y-3 max-h-[420px] overflow-auto pr-1">
        {#each promptFlow as item (item.metric.id)}
          <div class="border border-gray-200 dark:border-white/10 rounded p-3 bg-surface">
            <div class="flex flex-wrap items-center gap-2 mb-2">
              <span class="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                HTTP {item.metric.status_code || 200}
              </span>
              <span class="text-xs text-txtsecondary">#{item.metric.id + 1}</span>
              <span class="text-xs text-txtsecondary">{item.metric.model}</span>
              <span class="text-xs text-txtsecondary">{formatRelativeTime(item.metric.timestamp)}</span>
            </div>
            <div class="text-xs text-txtsecondary mb-1">
              path: <span class="font-mono">{item.requestPath || "(unknown)"}</span>
              {#if item.userAgent}
                | ua: <span class="font-mono">{item.userAgent}</span>
              {/if}
              {#if item.metric.trace_id}
                | trace: <span class="font-mono">{item.metric.trace_id}</span>
              {/if}
            </div>
            {#if item.stageNames.length > 0}
              <div class="mb-2">
                <div class="text-xs font-semibold mb-1">Internal proxy calls</div>
                <div class="flex flex-wrap gap-1">
                  {#each item.stageNames as stageName}
                    <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300">
                      {stageName}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}
            {#if item.responseSummary}
              <div class="mb-2">
                <div class="text-xs font-semibold mb-1">Stream summary</div>
                <div class="flex flex-wrap gap-1">
                  <span class="text-[11px] px-2 py-0.5 rounded bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
                    Worked for {formatDuration(item.metric.duration_ms)}
                  </span>
                  {#if item.responseSummary.webSearchCount > 0}
                    <span class="text-[11px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                      Searched web {item.responseSummary.webSearchCount} {item.responseSummary.webSearchCount === 1 ? "time" : "times"}
                    </span>
                  {/if}
                  {#if item.responseSummary.totalToolCalls > 0}
                    <span class="text-[11px] px-2 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
                      Called {item.responseSummary.totalToolCalls} {item.responseSummary.totalToolCalls === 1 ? "tool" : "tools"}
                    </span>
                  {/if}
                  {#if item.responseSummary.fileChanges.length > 0}
                    <span class="text-[11px] px-2 py-0.5 rounded bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300">
                      {item.responseSummary.fileChanges.filter((change) => change.action === "created").length > 0
                        ? `Created ${item.responseSummary.fileChanges.filter((change) => change.action === "created").length} files`
                        : `Edited ${item.responseSummary.fileChanges.length} files`}
                    </span>
                  {/if}
                </div>
                {#if item.responseSummary.webQueries.length > 0}
                  <div class="mt-2 space-y-1">
                    {#each item.responseSummary.webQueries as query}
                      <div class="text-[11px] font-mono text-txtsecondary">Searched web for {query}</div>
                    {/each}
                  </div>
                {/if}
                {#if item.responseSummary.toolNames.length > 0}
                  <div class="mt-2 flex flex-wrap gap-1">
                    {#each item.responseSummary.toolNames as toolName}
                      <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-card border border-gray-200 dark:border-white/10">
                        {toolName}
                      </span>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
            {#if item.metric.trace_id && groupedMonitor[item.metric.trace_id]?.length}
              <div class="mb-2">
                <div class="text-xs font-semibold mb-1">Latest monitor events</div>
                <div class="bg-card border border-gray-200 dark:border-white/10 rounded p-2 space-y-1">
                  {#each groupedMonitor[item.metric.trace_id] as monitorEvent}
                    <div class="text-[11px] font-mono text-txtsecondary">
                      <span class="text-sky-600 dark:text-sky-300">{monitorEvent.stage}</span>
                      <span> {monitorEvent.direction}</span>
                      {#if monitorEvent.event}
                        <span> · {monitorEvent.event}</span>
                      {/if}
                      {#if monitorEvent.data}
                        <div class="whitespace-pre-wrap break-all opacity-80">{monitorEvent.data}</div>
                      {/if}
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
            {#if item.lastUserPrompt}
              <div class="mb-2">
                <div class="text-xs font-semibold mb-1">Last user prompt</div>
                <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{item.lastUserPrompt}</pre>
              </div>
            {/if}
            <div>
              <div class="text-xs font-semibold mb-1">Request sent to proxy</div>
              <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{item.promptPreview || "(empty)"}</pre>
            </div>
            <div class="mt-3">
              <button class="btn btn--sm" onclick={() => viewCapture(item.metric.id)}>
                Open full capture
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<CaptureDialog capture={selectedCapture} open={dialogOpen} onclose={closeDialog} />
