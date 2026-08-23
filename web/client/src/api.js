const BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export function fetchCalls(params) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== "" && v !== null && v !== undefined) qs.set(k, v);
  });
  return request(`/api/calls?${qs}`);
}

export function fetchCall(callId) {
  return request(`/api/calls/${encodeURIComponent(callId)}`);
}

export function fetchBatches() {
  return request("/api/batches");
}

export function fetchBatch(batchTag) {
  return request(`/api/batches/${encodeURIComponent(batchTag)}`);
}

export function fetchFilterMeta() {
  return request("/api/calls/meta/filters");
}
