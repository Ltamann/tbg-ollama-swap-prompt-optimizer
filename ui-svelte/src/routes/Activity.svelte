<script lang="ts">
  import { onMount } from "svelte";
  import { metrics, getCapture, listActivityPromptPreviews } from "../stores/api";
  import Tooltip from "../components/Tooltip.svelte";
  import CaptureDialog from "../components/CaptureDialog.svelte";
  import type { ActivityPromptPreview, Metrics, ReqRespCapture } from "../lib/types";

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

    // Handle future dates by returning "just now"
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

  let sortedMetrics = $derived([...$metrics].sort((a, b) => b.id - a.id));
  let promptPreviews = $state<ActivityPromptPreview[]>([]);

  let selectedCapture = $state<ReqRespCapture | null>(null);
  let dialogOpen = $state(false);
  let loadingCaptureId = $state<number | null>(null);
  let loadingPrompts = $state(false);
  let inlineCaptureByPrompt = $state<Record<number, ReqRespCapture | null>>({});
  let loadingInlinePromptId = $state<number | null>(null);
  let inlineOpenByPrompt = $state<Record<number, boolean>>({});

  function metricForPreview(item: ActivityPromptPreview): Metrics | null {
    const previewTs = new Date(item.timestamp).getTime();
    let best: Metrics | null = null;
    let bestDiff = Number.MAX_SAFE_INTEGER;
    for (const m of sortedMetrics) {
      if ((m.model || "").trim() !== (item.model || "").trim()) continue;
      const diff = Math.abs(new Date(m.timestamp).getTime() - previewTs);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = m;
      }
    }
    return best;
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
      const parsed = JSON.parse(str);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return str;
    }
  }

  function parseSSEOutput(text: string): { reasoning: string; content: string } {
    const out = { reasoning: "", content: "" };
    for (const line of (text || "").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const payload = trimmed.slice(6);
      if (!payload || payload === "[DONE]") continue;
      try {
        const parsed = JSON.parse(payload);
        const delta = parsed?.choices?.[0]?.delta;
        if (delta?.reasoning_content) out.reasoning += String(delta.reasoning_content);
        if (delta?.content) out.content += String(delta.content);
      } catch {
        // ignore non-JSON SSE fragments
      }
    }
    return out;
  }

  function contentType(headers: Record<string, string> | null | undefined): string {
    if (!headers) return "";
    return String(headers["Content-Type"] || headers["content-type"] || "").toLowerCase();
  }

  function extractLastUserFromRequest(rawRequestBody: string): string {
    try {
      const parsed = JSON.parse(rawRequestBody) as any;
      const msgs = Array.isArray(parsed?.messages) ? parsed.messages : [];
      let last = "";
      for (const msg of msgs) {
        if (msg?.role !== "user") continue;
        const c = msg?.content;
        if (typeof c === "string") {
          last = c;
          continue;
        }
        if (Array.isArray(c)) {
          const textParts: string[] = [];
          for (const part of c) {
            if (part?.type === "text" && typeof part?.text === "string") {
              textParts.push(part.text);
            }
          }
          if (textParts.length > 0) {
            last = textParts.join("\n");
          }
        }
      }
      return last;
    } catch {
      return "";
    }
  }

  async function toggleInlineFlow(item: ActivityPromptPreview, metric: Metrics | null): Promise<void> {
    const isOpen = inlineOpenByPrompt[item.id] === true;
    if (isOpen) {
      inlineOpenByPrompt = { ...inlineOpenByPrompt, [item.id]: false };
      return;
    }

    inlineOpenByPrompt = { ...inlineOpenByPrompt, [item.id]: true };
    if (inlineCaptureByPrompt[item.id] !== undefined) {
      return;
    }
    if (!metric?.has_capture) {
      inlineCaptureByPrompt = { ...inlineCaptureByPrompt, [item.id]: null };
      return;
    }

    loadingInlinePromptId = item.id;
    const capture = await getCapture(metric.id);
    loadingInlinePromptId = null;
    inlineCaptureByPrompt = { ...inlineCaptureByPrompt, [item.id]: capture };
  }

  async function refreshPromptPreviews() {
    loadingPrompts = true;
    promptPreviews = await listActivityPromptPreviews();
    loadingPrompts = false;
  }

  onMount(() => {
    void refreshPromptPreviews();
    const timer = setInterval(() => {
      void refreshPromptPreviews();
    }, 1500);
    return () => clearInterval(timer);
  });

  async function viewCapture(id: number) {
    loadingCaptureId = id;
    const capture = await getCapture(id);
    loadingCaptureId = null;
    if (capture) {
      selectedCapture = capture;
      dialogOpen = true;
    }
  }

  function closeDialog() {
    dialogOpen = false;
    selectedCapture = null;
  }
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
              <td class="px-6 py-4">{metric.cache_tokens > 0 ? metric.cache_tokens.toLocaleString() : "-"}</td>
              <td class="px-6 py-4">{metric.input_tokens.toLocaleString()}</td>
              <td class="px-6 py-4">{metric.output_tokens.toLocaleString()}</td>
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
      <button class="btn btn--sm" onclick={() => refreshPromptPreviews()} disabled={loadingPrompts}>
        {loadingPrompts ? "..." : "Refresh"}
      </button>
    </div>
    <p class="text-xs text-txtsecondary mb-3">
      Current user turn only. Resets automatically when a new user request is detected.
    </p>

    {#if promptPreviews.length === 0}
      <p class="text-sm text-txtsecondary">No prompt activity yet.</p>
    {:else}
      <div class="space-y-3 max-h-[420px] overflow-auto pr-1">
        {#each [...promptPreviews].reverse() as item (item.id)}
          {@const previewMetric = metricForPreview(item)}
          <div class="border border-gray-200 dark:border-white/10 rounded p-3 bg-surface">
            <div class="flex flex-wrap items-center gap-2 mb-2">
              <span class="text-xs font-semibold px-2 py-0.5 rounded {item.kind === 'user_request' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'}">
                {item.kind === "user_request" ? "New User Request" : "Agent Step"}
              </span>
              <span class="text-xs text-txtsecondary">#{item.id}</span>
              <span class="text-xs text-txtsecondary">{item.model}</span>
              <span class="text-xs text-txtsecondary">{formatRelativeTime(item.timestamp)}</span>
              <span class="text-xs text-txtsecondary">messages: {item.message_count}</span>
            </div>
            {#if previewMetric}
              <div class="text-xs mb-2 flex flex-wrap items-center gap-2 text-txtsecondary">
                <span class="font-semibold text-txt">Metrics</span>
                <span>cached: {previewMetric.cache_tokens > 0 ? previewMetric.cache_tokens.toLocaleString() : "-"}</span>
                <span>prompt: {previewMetric.input_tokens.toLocaleString()}</span>
                <span>generated: {previewMetric.output_tokens.toLocaleString()}</span>
                <span>prompt/s: {formatSpeed(previewMetric.prompt_per_second)}</span>
                <span>gen/s: {formatSpeed(previewMetric.tokens_per_second)}</span>
                <span>dur: {formatDuration(previewMetric.duration_ms)}</span>
              </div>
            {/if}
            <div class="text-xs text-txtsecondary mb-1">
              path: <span class="font-mono">{item.request_path}</span>
              {#if item.user_agent}
                | ua: <span class="font-mono">{item.user_agent}</span>
              {/if}
            </div>
            {#if item.last_user_prompt}
              <div class="mb-2">
                <div class="text-xs font-semibold mb-1">Last user prompt</div>
                <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{item.last_user_prompt}</pre>
              </div>
            {/if}
            <div>
              <div class="text-xs font-semibold mb-1">Request sent to model (preview)</div>
              <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{item.prompt_preview}</pre>
            </div>
            <div class="mt-3">
              <button class="btn btn--sm" onclick={() => toggleInlineFlow(item, previewMetric)}>
                {inlineOpenByPrompt[item.id] ? "Hide full communication flow" : "Show full communication flow"}
              </button>
            </div>
            {#if inlineOpenByPrompt[item.id]}
              <div class="mt-3 space-y-2">
                {#if loadingInlinePromptId === item.id}
                  <div class="text-xs text-txtsecondary">Loading full communication...</div>
                {:else}
                  {@const capture = inlineCaptureByPrompt[item.id]}
                  {#if capture}
                    {@const reqRaw = decodeBody(capture.req_body)}
                    {@const reqPretty = formatJson(reqRaw)}
                    {@const respRaw = decodeBody(capture.resp_body)}
                    {@const reqCT = contentType(capture.req_headers)}
                    {@const respCT = contentType(capture.resp_headers)}
                    {@const parsedSSE = respCT.includes("text/event-stream") ? parseSSEOutput(respRaw) : { reasoning: "", content: "" }}
                    {@const cliInput = extractLastUserFromRequest(reqRaw) || item.last_user_prompt}
                    <div class="text-xs font-semibold">CLI input (decoded)</div>
                    <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{cliInput || "(not found)"}</pre>

                    <div class="text-xs font-semibold">Proxy forwarded request (decoded)</div>
                    <div class="text-[11px] text-txtsecondary mb-1">path: <span class="font-mono">{capture.req_path}</span>{#if reqCT} | content-type: <span class="font-mono">{reqCT}</span>{/if}</div>
                    <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{reqPretty || "(empty)"}</pre>

                    <div class="text-xs font-semibold">Server response (decoded)</div>
                    {#if respCT.includes("text/event-stream")}
                      <div class="text-[11px] text-txtsecondary mb-1">content-type: <span class="font-mono">{respCT}</span></div>
                      {#if parsedSSE.reasoning}
                        <div class="text-[11px] font-semibold text-txtsecondary">reasoning</div>
                        <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{parsedSSE.reasoning}</pre>
                      {/if}
                      <div class="text-[11px] font-semibold text-txtsecondary">assistant output</div>
                      <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{parsedSSE.content || "(empty)"}</pre>
                      <details>
                        <summary class="text-xs cursor-pointer text-txtsecondary">Raw SSE</summary>
                        <pre class="mt-1 text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{respRaw || "(empty)"}</pre>
                      </details>
                    {:else}
                      <div class="text-[11px] text-txtsecondary mb-1">content-type: <span class="font-mono">{respCT || "unknown"}</span></div>
                      <pre class="text-xs whitespace-pre-wrap break-words bg-card border border-gray-200 dark:border-white/10 rounded p-2">{formatJson(respRaw) || "(empty)"}</pre>
                    {/if}
                  {:else}
                    <div class="text-xs text-txtsecondary">No capture available for this request (capture may be disabled or request exceeded capture buffer).</div>
                  {/if}
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

</div>

<CaptureDialog capture={selectedCapture} open={dialogOpen} onclose={closeDialog} />
