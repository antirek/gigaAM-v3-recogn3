import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import mongoose from "mongoose";
import { createApp } from "../src/app.js";
import { Call } from "../src/models/Call.js";
import { BatchSummary } from "../src/models/BatchSummary.js";
import { normalizeCallRecord } from "../src/services/importBatch.js";

const TEST_URI =
  process.env.MONGODB_URI_TEST || "mongodb://127.0.0.1:27017/gigaam_calls_test";

const sampleCall = normalizeCallRecord({
  callId: "2026-08-19_09-08-00_b494c500",
  batchTag: "test_batch",
  transcript: "[00:00] Спикер 1: тест",
  summary: {
    call_id: "2026-08-19_09-08-00_b494c500",
    intent: "Тестовый звонок",
    topics: ["тест"],
    escalation: {
      needed: true,
      severity: "medium",
      reasons: ["process_failure"],
      evidence: [],
      summary_for_manager: "Нужен разбор",
    },
  },
  summaryMd: "# Test",
});

test("normalizeCallRecord parses date and escalation", () => {
  assert.equal(sampleCall.date, "2026-08-19");
  assert.equal(sampleCall.escalationNeeded, true);
  assert.deepEqual(sampleCall.escalationReasons, ["process_failure"]);
});

test("normalizeCallRecord flattens extract facts", () => {
  const doc = normalizeCallRecord({
    callId: "2026-08-19_09-08-00_b494c500",
    batchTag: "test_batch",
    transcript: "x",
    summary: { intent: "i", escalation: { needed: false } },
    extract: {
      phones: [{ digits: "79001234567" }],
      addresses: [{ text: "Тверь" }],
      amounts: [{ value: "1000", currency: "RUB" }],
      commitments: [{ promise: "перезвоним" }],
    },
  });
  assert.deepEqual(doc.phones, ["79001234567"]);
  assert.deepEqual(doc.addresses, ["Тверь"]);
  assert.equal(doc.amounts[0], "1000 RUB");
  assert.deepEqual(doc.commitments, ["перезвоним"]);
});

test("API: import, list, filter, get call, batch", async (t) => {
  await mongoose.connect(TEST_URI);
  await Call.deleteMany({});
  await BatchSummary.deleteMany({});

  const app = createApp();

  t.after(async () => {
    await Call.deleteMany({});
    await BatchSummary.deleteMany({});
    await mongoose.disconnect();
  });

  const importRes = await request(app)
    .post("/api/import/batch")
    .send({
      batchTag: "test_batch",
      calls: [
        {
          callId: sampleCall.callId,
          transcript: sampleCall.transcript,
          summary: sampleCall.summary,
          summaryMd: sampleCall.summaryMd,
        },
      ],
      batchSummary: {
        date: "2026-08-19",
        n_calls: 1,
        executive_summary: "День тестовый",
        supervisor_escalations: [],
        n_escalations: 1,
      },
    });

  assert.equal(importRes.status, 200);
  assert.equal(importRes.body.nCalls, 1);

  const listRes = await request(app).get("/api/calls?escalation=true");
  assert.equal(listRes.status, 200);
  assert.equal(listRes.body.total, 1);
  assert.equal(listRes.body.items[0].callId, sampleCall.callId);

  const getRes = await request(app).get(`/api/calls/${sampleCall.callId}`);
  assert.equal(getRes.status, 200);
  assert.match(getRes.body.transcript, /тест/);

  const batchRes = await request(app).get("/api/batches/test_batch");
  assert.equal(batchRes.status, 200);
  assert.equal(batchRes.body.data.executive_summary, "День тестовый");
});
