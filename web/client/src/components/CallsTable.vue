<script setup>
import { onMounted, reactive, ref, watch } from "vue";
import { fetchCall, fetchCalls, fetchFilterMeta, fetchBatches } from "../api.js";
import AppModal from "./AppModal.vue";

const props = defineProps({
  initialCallId: { type: String, default: "" },
});

const emit = defineEmits(["open-call"]);

const batches = ref([]);
const reasonOptions = ref([]);
const loading = ref(false);
const error = ref("");
const result = ref({ items: [], total: 0, page: 1, pages: 1 });
const ready = ref(false);
let loadSeq = 0;

const filters = reactive({
  batchTag: "",
  date: "",
  escalation: "",
  severity: "",
  reason: "",
  q: "",
  page: 1,
  limit: 50,
});

const modal = reactive({
  show: false,
  title: "",
  mode: "",
  content: "",
  call: null,
});

const severityLabel = (severity) => {
  if (!severity) return "—";
  return severity;
};

async function loadMeta() {
  const [batchRes, filterRes] = await Promise.all([fetchBatches(), fetchFilterMeta()]);
  batches.value = batchRes.items || [];
  reasonOptions.value = filterRes.escalationReasons || [];
  if (!filters.batchTag && batches.value.length) {
    filters.batchTag = batches.value[0].batchTag;
  }
}

async function loadCalls() {
  const seq = ++loadSeq;
  loading.value = true;
  error.value = "";
  try {
    const data = await fetchCalls({ ...filters });
    if (seq !== loadSeq) return;
    result.value = data;
  } catch (e) {
    if (seq !== loadSeq) return;
    error.value = e.message;
  } finally {
    if (seq === loadSeq) loading.value = false;
  }
}

function resetPageAndLoad() {
  if (filters.page !== 1) {
    filters.page = 1;
  } else {
    loadCalls();
  }
}

function formatTime(call) {
  if (call.startedAt) {
    return new Date(call.startedAt).toLocaleTimeString("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
  return call.callId?.slice(11, 19)?.replace(/-/g, ":") || "—";
}

async function openModal(mode, row) {
  modal.mode = mode;
  modal.title =
    mode === "transcript"
      ? `Диалог: ${row.callId}`
      : mode === "json"
        ? `JSON summary: ${row.callId}`
        : mode === "extract"
          ? `Факты (extract): ${row.callId}`
          : `Саммари: ${row.callId}`;
  modal.show = true;
  modal.content = "";
  modal.call = null;

  try {
    const full = await fetchCall(row.callId);
    modal.call = full;
    if (mode === "transcript") {
      modal.content = full.transcript || "(пустой транскрипт)";
    } else if (mode === "json") {
      modal.content = JSON.stringify(full.summary || {}, null, 2);
    } else if (mode === "extract") {
      modal.content = formatExtract(full);
    } else {
      modal.content = buildSummaryText(full);
    }
  } catch (e) {
    modal.content = `Ошибка: ${e.message}`;
  }
}

function formatExtract(call) {
  const ex = call.extract || {};
  if (!ex || (!ex.phones?.length && !ex.addresses?.length && !ex.amounts?.length && !ex.commitments?.length)) {
    return JSON.stringify(ex || {}, null, 2) || "(extract пуст — прогоните summarize-call / extract)";
  }
  const lines = [];
  if (ex.phones?.length) {
    lines.push("Phones:");
    for (const p of ex.phones) {
      const digits = typeof p === "string" ? p : p.digits;
      const ev = typeof p === "object" ? p.evidence : "";
      lines.push(`- ${digits}${ev ? `  (${ev})` : ""}`);
    }
  }
  if (ex.addresses?.length) {
    lines.push("\nAddresses:");
    for (const a of ex.addresses) {
      const text = typeof a === "string" ? a : a.text;
      lines.push(`- ${text}`);
    }
  }
  if (ex.amounts?.length) {
    lines.push("\nAmounts:");
    for (const a of ex.amounts) {
      if (typeof a === "string") lines.push(`- ${a}`);
      else lines.push(`- ${a.value || ""} ${a.currency || ""} ${a.what || ""}`.trim());
    }
  }
  if (ex.commitments?.length) {
    lines.push("\nCommitments:");
    for (const c of ex.commitments) {
      if (typeof c === "string") lines.push(`- ${c}`);
      else lines.push(`- ${c.promise || ""}${c.when ? ` (когда: ${c.when})` : ""}`);
    }
  }
  if (ex.notes) lines.push(`\nNotes: ${ex.notes}`);
  lines.push("\n---\n" + JSON.stringify(ex, null, 2));
  return lines.join("\n");
}

function factsPreview(row) {
  const bits = [];
  if (row.phones?.length) bits.push(`☎ ${row.phones.slice(0, 2).join(", ")}`);
  if (row.addresses?.length) bits.push(`⌂ ${row.addresses[0]}`);
  if (row.amounts?.length) bits.push(`₽ ${row.amounts[0]}`);
  if (row.commitments?.length) bits.push(`✓ ${row.commitments[0].slice(0, 40)}`);
  return bits.length ? bits.join(" · ") : "—";
}

function buildSummaryText(call) {
  const s = call.summary || {};
  const lines = [];
  if (s.intent) lines.push(`Intent:\n${s.intent}`);
  if (s.topics?.length) lines.push(`\nTopics:\n- ${s.topics.join("\n- ")}`);
  if (s.issues_detected?.length) {
    lines.push("\nIssues:");
    for (const i of s.issues_detected) {
      lines.push(`- (${i.severity}) ${i.issue}`);
    }
  }
  if (s.actions?.length) {
    lines.push("\nActions:");
    for (const a of s.actions) {
      lines.push(`- ${a.who}: ${a.action}`);
    }
  }
  const esc = s.escalation || {};
  if (esc.needed) {
    lines.push(`\nEscalation (${esc.severity}): ${esc.summary_for_manager || ""}`);
  }
  if (call.summaryMd) {
    lines.push(`\n---\n${call.summaryMd}`);
  }
  return lines.join("\n") || "(нет саммари)";
}

function escalationBadge(row) {
  if (!row.escalationNeeded) return { cls: "none", text: "нет" };
  return { cls: row.escalationSeverity || "low", text: row.escalationSeverity || "да" };
}

watch(
  () => [filters.batchTag, filters.date, filters.escalation, filters.severity, filters.reason],
  () => {
    if (ready.value) resetPageAndLoad();
  },
);

watch(
  () => filters.page,
  () => {
    if (ready.value) loadCalls();
  },
);

onMounted(async () => {
  await loadMeta();
  ready.value = true;
  await loadCalls();
  if (props.initialCallId) {
    await openModal("summary", { callId: props.initialCallId });
  }
});

defineExpose({ openCall: (callId) => openModal("summary", { callId }) });
</script>

<template>
  <div class="card">
    <div class="filters">
      <label>
        Батч
        <select v-model="filters.batchTag">
          <option value="">Все</option>
          <option v-for="b in batches" :key="b.batchTag" :value="b.batchTag">
            {{ b.batchTag }}
          </option>
        </select>
      </label>
      <label>
        Дата
        <input v-model="filters.date" type="date" />
      </label>
      <label>
        Эскалация
        <select v-model="filters.escalation">
          <option value="">Все</option>
          <option value="true">Только с эскалацией</option>
          <option value="false">Без эскалации</option>
        </select>
      </label>
      <label>
        Severity
        <select v-model="filters.severity">
          <option value="">Все</option>
          <option value="high">high</option>
          <option value="medium">medium</option>
          <option value="low">low</option>
        </select>
      </label>
      <label>
        Причина
        <select v-model="filters.reason">
          <option value="">Все</option>
          <option v-for="r in reasonOptions" :key="r" :value="r">{{ r }}</option>
        </select>
      </label>
      <label>
        Поиск
        <input v-model="filters.q" placeholder="ID, intent, topic" @keyup.enter="resetPageAndLoad" />
      </label>
      <label class="checkbox-row">
        <button class="btn primary" type="button" @click="resetPageAndLoad">Применить</button>
      </label>
    </div>

    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="loading" class="muted">Загрузка…</div>

    <div v-else class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Время</th>
            <th>ID звонка</th>
            <th>Intent</th>
            <th>Темы</th>
            <th>Факты</th>
            <th>Эскалация</th>
            <th>Действия</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in result.items" :key="row.callId">
            <td>{{ formatTime(row) }}</td>
            <td><code>{{ row.callId }}</code></td>
            <td class="intent-cell">{{ row.intent || "—" }}</td>
            <td>
              <div class="reason-tags">
                <span v-for="t in (row.topics || []).slice(0, 3)" :key="t" class="reason-tag">{{ t }}</span>
              </div>
            </td>
            <td class="intent-cell muted">{{ factsPreview(row) }}</td>
            <td>
              <span class="badge" :class="escalationBadge(row).cls">
                {{ escalationBadge(row).text }}
              </span>
              <div v-if="row.escalationReasons?.length" class="reason-tags" style="margin-top: 4px">
                <span v-for="r in row.escalationReasons" :key="r" class="reason-tag">{{ r }}</span>
              </div>
            </td>
            <td>
              <div class="actions">
                <button class="btn" type="button" @click="openModal('transcript', row)">Диалог</button>
                <button class="btn" type="button" @click="openModal('extract', row)">Факты</button>
                <button class="btn" type="button" @click="openModal('json', row)">JSON</button>
                <button class="btn" type="button" @click="openModal('summary', row)">Саммари</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="pagination">
      <span>Всего: {{ result.total }}</span>
      <div>
        <button class="btn" :disabled="filters.page <= 1" @click="filters.page -= 1">←</button>
        <span style="margin: 0 8px">{{ filters.page }} / {{ result.pages }}</span>
        <button class="btn" :disabled="filters.page >= result.pages" @click="filters.page += 1">→</button>
      </div>
    </div>
  </div>

  <AppModal :show="modal.show" :title="modal.title" @close="modal.show = false">
    <pre :class="modal.mode === 'transcript' ? 'transcript' : ''">{{ modal.content }}</pre>
  </AppModal>
</template>
