import { Router } from "express";
import { BatchSummary } from "../models/BatchSummary.js";

const router = Router();

router.get("/", async (_req, res, next) => {
  try {
    const batches = await BatchSummary.find({})
      .sort({ date: -1, batchTag: -1 })
      .select("batchTag date data.n_calls data.n_escalations updatedAt")
      .lean();

    res.json({
      items: batches.map((b) => ({
        batchTag: b.batchTag,
        date: b.date,
        nCalls: b.data?.n_calls ?? b.data?.n_calls_total ?? null,
        nEscalations: b.data?.n_escalations ?? null,
        updatedAt: b.updatedAt,
      })),
    });
  } catch (err) {
    next(err);
  }
});

router.get("/:batchTag", async (req, res, next) => {
  try {
    const batch = await BatchSummary.findOne({ batchTag: req.params.batchTag }).lean();
    if (!batch) {
      res.status(404).json({ error: "batch not found" });
      return;
    }
    res.json(batch);
  } catch (err) {
    next(err);
  }
});

export default router;
