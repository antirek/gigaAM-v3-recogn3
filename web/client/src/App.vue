<script setup>
import { ref } from "vue";
import CallsTable from "./components/CallsTable.vue";
import BatchSummaryView from "./components/BatchSummaryView.vue";

const tab = ref("calls");
const callsRef = ref(null);
const jumpCallId = ref("");

function onJumpToCall(callId) {
  jumpCallId.value = callId;
  tab.value = "calls";
  setTimeout(() => {
    callsRef.value?.openCall?.(callId);
  }, 50);
}
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <h1>GigaAM — аналитика звонков</h1>
      <span class="muted">MongoDB + API</span>
    </header>

    <nav class="tabs">
      <button
        class="tab-btn"
        :class="{ active: tab === 'calls' }"
        type="button"
        @click="tab = 'calls'"
      >
        Список звонков
      </button>
      <button
        class="tab-btn"
        :class="{ active: tab === 'batch' }"
        type="button"
        @click="tab = 'batch'"
      >
        Саммари за день
      </button>
    </nav>

    <CallsTable
      v-if="tab === 'calls'"
      ref="callsRef"
      :initial-call-id="jumpCallId"
    />
    <BatchSummaryView
      v-if="tab === 'batch'"
      @jump-to-call="onJumpToCall"
    />
  </div>
</template>
