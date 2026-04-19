<script lang="ts">
  interface Props {
    modelCtx?: number;
    inputCtx?: number;
    optimizedCtx?: number;
    modelId?: string;
  }

  let { modelCtx = 0, inputCtx = 0, optimizedCtx = 0, modelId = "" }: Props = $props();

  function toK(value: number): string {
    if (!Number.isFinite(value) || value <= 0) return "0k";
    return `${Math.round(value / 1000)}k`;
  }

  let safeModelCtx = $derived(Math.max(1, Math.floor(modelCtx)));
  let inputPercent = $derived((Math.max(0, inputCtx) / safeModelCtx) * 100);
  let optimizedPercent = $derived((Math.max(0, optimizedCtx) / safeModelCtx) * 100);
  let inputWidth = $derived(Math.min(100, inputPercent));
  let optimizedWidth = $derived(Math.min(100, optimizedPercent));
  let inputOverflow = $derived(inputPercent > 100);
  let optimizedOverflow = $derived(optimizedPercent > 100);
  let hasData = $derived(modelCtx > 0);
</script>

<div class="w-full max-w-[560px] px-2">
  <div class="flex items-center justify-between text-[10px] leading-none text-txtsecondary mb-1">
    <span>input total</span>
    <span>{modelId ? `${modelId} ` : ""}{toK(modelCtx)}</span>
  </div>

  <div class="relative border rounded-full border-gray-400/60 dark:border-gray-500/70 bg-transparent px-[3px] py-[3px]">
    <div class="relative h-[4px] w-full rounded-full overflow-hidden bg-gray-300/80 dark:bg-gray-700/90">
      {#if hasData}
        <div class="absolute inset-y-0 left-0 rounded-full bg-gray-500/80 dark:bg-gray-500/85" style={`width:${inputWidth}%`}></div>
      {/if}
      {#if inputOverflow}
        <div class="absolute inset-y-0 right-0 w-[10px] rounded-full border border-red-500/90 bg-red-500/10"></div>
      {/if}
    </div>

    <div class="mt-[3px] relative h-[4px] w-full rounded-full overflow-hidden bg-gray-300/80 dark:bg-gray-700/90">
      {#if hasData}
        <div class="absolute inset-y-0 left-0 rounded-full bg-primary" style={`width:${optimizedWidth}%`}></div>
      {/if}
      {#if optimizedOverflow}
        <div class="absolute inset-y-0 right-0 w-[10px] rounded-full border border-red-500/90 bg-red-500/10"></div>
      {/if}
    </div>
  </div>

  <div class="mt-1 text-[10px] leading-none text-txtsecondary">output optimized: {toK(optimizedCtx)}</div>
</div>
