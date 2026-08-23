import fs from "fs";
import path from "path";
import { Call } from "../models/Call.js";
import { BatchSummary } from "../models/BatchSummary.js";

const CALL_DIR_RE = /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-f0-9]+$/i;

export function parseCallMeta(callId) {
  const m = callId.match(/^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_/);
  if (!m) {
    return { date: callId.slice(0, 10) || "unknown", startedAt: null };
  }
  const [, date, hh, mm, ss] = m;
  const startedAt = new Date(`${date}T${hh}:${mm}:${ss}`);
  return { date, startedAt: Number.isNaN(startedAt.getTime()) ? null : startedAt };
}

export function normalizeCallRecord({ callId, batchTag, transcript, summary, summaryMd }) {
  const { date, startedAt } = parseCallMeta(callId);
  const esc = summary?.escalation || {};
  return {
    callId,
    batchTag,
    date,
    startedAt,
    transcript: transcript || "",
    summary: summary || {},
    summaryMd: summaryMd || "",
    intent: String(summary?.intent || ""),
    topics: Array.isArray(summary?.topics) ? summary.topics : [],
    escalationNeeded: Boolean(esc.needed),
    escalationSeverity: esc.needed ? String(esc.severity || "low") : "",
    escalationReasons: Array.isArray(esc.reasons) ? esc.reasons : [],
  };
}

export async function upsertCalls(calls) {
  let upserted = 0;
  for (const raw of calls) {
    const doc = normalizeCallRecord(raw);
    await Call.findOneAndUpdate({ callId: doc.callId }, doc, {
      upsert: true,
      new: true,
      setDefaultsOnInsert: true,
    });
    upserted += 1;
  }
  return upserted;
}

export async function upsertBatchSummary({ batchTag, batchSummary }) {
  const date =
    batchSummary?.date ||
    batchTag.match(/(\d{4}-\d{2}-\d{2})/)?.[1] ||
    "unknown";
  await BatchSummary.findOneAndUpdate(
    { batchTag },
    { batchTag, date, data: batchSummary || {} },
    { upsert: true, new: true, setDefaultsOnInsert: true },
  );
  return { batchTag, date };
}

function readTextIfExists(filePath) {
  if (!fs.existsSync(filePath)) return "";
  return fs.readFileSync(filePath, "utf-8");
}

function readJsonIfExists(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

export function loadBatchFromDir(batchDir) {
  const abs = path.resolve(batchDir);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
    throw new Error(`batch directory not found: ${abs}`);
  }

  const batchTag = path.basename(abs);
  const calls = [];

  for (const name of fs.readdirSync(abs)) {
    if (!CALL_DIR_RE.test(name)) continue;
    const callDir = path.join(abs, name);
    if (!fs.statSync(callDir).isDirectory()) continue;

    const summary = readJsonIfExists(path.join(callDir, "call_summary.json"));
    if (!summary) continue;

    calls.push({
      callId: name,
      batchTag,
      transcript: readTextIfExists(path.join(callDir, "transcript.txt")),
      summary,
      summaryMd: readTextIfExists(path.join(callDir, "call_summary.md")),
    });
  }

  const batchSummary = readJsonIfExists(path.join(abs, "batch_summary.json"));
  return { batchTag, calls, batchSummary };
}

export async function importBatchFromDir(batchDir) {
  const { batchTag, calls, batchSummary } = loadBatchFromDir(batchDir);
  const nCalls = await upsertCalls(calls);
  let batch = null;
  if (batchSummary) {
    batch = await upsertBatchSummary({ batchTag, batchSummary });
  }
  return { batchTag, nCalls, batch };
}
