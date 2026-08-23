import { Router } from "express";
import {
  importBatchFromDir,
  upsertBatchSummary,
  upsertCalls,
} from "../services/importBatch.js";

const router = Router();

router.post("/batch", async (req, res, next) => {
  try {
    const { batchTag, calls, batchSummary } = req.body || {};
    if (!batchTag || !Array.isArray(calls)) {
      res.status(400).json({
        error: "batchTag and calls[] are required",
      });
      return;
    }

    const normalized = calls.map((c) => ({
      callId: c.callId,
      batchTag,
      transcript: c.transcript || "",
      summary: c.summary || {},
      summaryMd: c.summaryMd || "",
      extract: c.extract || {},
    }));

    const nCalls = await upsertCalls(normalized);
    let batch = null;
    if (batchSummary) {
      batch = await upsertBatchSummary({ batchTag, batchSummary });
    }

    res.json({ ok: true, batchTag, nCalls, batch });
  } catch (err) {
    next(err);
  }
});

router.post("/batch-from-path", async (req, res, next) => {
  try {
    const { path: batchPath } = req.body || {};
    if (!batchPath) {
      res.status(400).json({ error: "path is required" });
      return;
    }
    const result = await importBatchFromDir(batchPath);
    res.json({ ok: true, ...result });
  } catch (err) {
    next(err);
  }
});

export default router;
