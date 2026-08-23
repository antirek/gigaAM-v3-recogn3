<script setup>
import { onMounted, ref, watch } from "vue";
import { fetchBatch, fetchBatches } from "../api.js";

const emit = defineEmits(["jump-to-call"]);

const batches = ref([]);
const selectedTag = ref("");
const batch = ref(null);
const loading = ref(false);
const error = ref("");

async function loadBatches() {
  const res = await fetchBatches();
  batches.value = res.items || [];
  if (!selectedTag.value && batches.value.length) {
    selectedTag.value = batches.value[0].batchTag;
  }
}

async function loadBatch() {
  if (!selectedTag.value) return;
  loading.value = true;
  error.value = "";
  try {
    batch.value = await fetchBatch(selectedTag.value);
  } catch (e) {
    error.value = e.message;
    batch.value = null;
  } finally {
    loading.value = false;
  }
}

watch(selectedTag, loadBatch);

onMounted(async () => {
  await loadBatches();
  await loadBatch();
});
</script>

<template>
  <div class="card">
    <div class="filters" style="grid-template-columns: 1fr auto">
      <label>
        Батч / день
        <select v-model="selectedTag">
          <option v-for="b in batches" :key="b.batchTag" :value="b.batchTag">
            {{ b.date }} — {{ b.batchTag }} ({{ b.nCalls }} зв., эск. {{ b.nEscalations ?? "?" }})
          </option>
        </select>
      </label>
      <label class="checkbox-row">
        <button class="btn primary" type="button" @click="loadBatch">Обновить</button>
      </label>
    </div>

    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="loading" class="muted">Загрузка…</div>
    <div v-else-if="batch" class="batch-grid">
      <section class="batch-section">
        <h3>Общая картина</h3>
        <p>{{ batch.data?.executive_summary || "—" }}</p>
        <p class="muted">
          Звонков: {{ batch.data?.n_calls ?? batch.data?.n_calls_total ?? "?" }},
          эскалаций: {{ batch.data?.n_escalations ?? "?" }}
        </p>
      </section>

      <section v-if="batch.data?.key_moments?.length" class="batch-section">
        <h3>Ключевые моменты</h3>
        <ul>
          <li v-for="(m, i) in batch.data.key_moments" :key="i">{{ m }}</li>
        </ul>
      </section>

      <section v-if="batch.data?.top_topics?.length" class="batch-section">
        <h3>Топ темы</h3>
        <div class="reason-tags">
          <span v-for="t in batch.data.top_topics" :key="t" class="reason-tag">{{ t }}</span>
        </div>
      </section>

      <section v-if="batch.data?.recommendations?.length" class="batch-section">
        <h3>Рекомендации</h3>
        <ul>
          <li v-for="(r, i) in batch.data.recommendations" :key="i">{{ r }}</li>
        </ul>
      </section>

      <section v-if="batch.data?.supervisor_escalations?.length" class="batch-section">
        <h3>Эскалации руководителю ({{ batch.data.supervisor_escalations.length }})</h3>
        <div class="escalation-list">
          <article
            v-for="esc in batch.data.supervisor_escalations"
            :key="esc.call_id"
            class="escalation-item"
          >
            <div>
              <span
                class="call-link"
                @click="emit('jump-to-call', esc.call_id)"
              >{{ esc.call_id }}</span>
              <span class="badge" :class="esc.severity">{{ esc.severity }}</span>
            </div>
            <div class="reason-tags" style="margin: 6px 0">
              <span v-for="r in esc.reasons || []" :key="r" class="reason-tag">{{ r }}</span>
            </div>
            <p>{{ esc.summary_for_manager }}</p>
          </article>
        </div>
      </section>
    </div>
    <div v-else class="muted">Нет данных batch summary. Импортируйте батч через API/скрипт.</div>
  </div>
</template>
