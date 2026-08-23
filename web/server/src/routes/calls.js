import { Router } from "express";
import { Call } from "../models/Call.js";

const router = Router();

const ESCALATION_REASONS = [
  "complaint_threat",
  "billing_dispute",
  "agent_quality",
  "unresolved_repeat",
  "process_failure",
];

function buildCallFilter(query) {
  const filter = {};

  if (query.batchTag) filter.batchTag = query.batchTag;
  if (query.date) filter.date = query.date;

  if (query.escalation === "true" || query.escalation === "1") {
    filter.escalationNeeded = true;
  } else if (query.escalation === "false" || query.escalation === "0") {
    filter.escalationNeeded = false;
  }

  if (query.severity) {
    filter.escalationSeverity = query.severity;
  }

  if (query.reason) {
    const reasons = String(query.reason)
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (reasons.length) filter.escalationReasons = { $in: reasons };
  }

  const q = String(query.q || "").trim();
  if (q) {
    filter.$or = [
      { callId: { $regex: q, $options: "i" } },
      { intent: { $regex: q, $options: "i" } },
      { topics: { $elemMatch: { $regex: q, $options: "i" } } },
    ];
  }

  return filter;
}

router.get("/meta/filters", (_req, res) => {
  res.json({ escalationReasons: ESCALATION_REASONS });
});

router.get("/", async (req, res, next) => {
  try {
    const page = Math.max(1, Number(req.query.page) || 1);
    const limit = Math.min(200, Math.max(1, Number(req.query.limit) || 50));
    const sortBy = req.query.sortBy === "callId" ? "callId" : "startedAt";
    const sortDir = req.query.sortDir === "asc" ? 1 : -1;

    const filter = buildCallFilter(req.query);
    const [items, total] = await Promise.all([
      Call.find(filter)
        .sort({ [sortBy]: sortDir })
        .skip((page - 1) * limit)
        .limit(limit)
        .select(
          "callId batchTag date startedAt intent topics phones addresses amounts commitments escalationNeeded escalationSeverity escalationReasons",
        )
        .lean(),
      Call.countDocuments(filter),
    ]);

    res.json({
      items,
      total,
      page,
      limit,
      pages: Math.ceil(total / limit) || 1,
    });
  } catch (err) {
    next(err);
  }
});

router.get("/:callId", async (req, res, next) => {
  try {
    const call = await Call.findOne({ callId: req.params.callId }).lean();
    if (!call) {
      res.status(404).json({ error: "call not found" });
      return;
    }
    res.json(call);
  } catch (err) {
    next(err);
  }
});

export default router;
