<script setup>
defineProps({
  title: { type: String, required: true },
  show: { type: Boolean, default: false },
});

const emit = defineEmits(["close"]);
</script>

<template>
  <div v-if="show" class="modal-backdrop" @click.self="emit('close')">
    <div class="modal card">
      <button class="btn modal-close" type="button" @click="emit('close')">✕</button>
      <h2>{{ title }}</h2>
      <slot />
    </div>
  </div>
</template>
