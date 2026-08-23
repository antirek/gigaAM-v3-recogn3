import cors from "cors";
import express from "express";
import { config } from "./config.js";
import callsRouter from "./routes/calls.js";
import batchesRouter from "./routes/batches.js";
import importRouter from "./routes/import.js";

export function createApp() {
  const app = express();

  app.use(cors({ origin: config.corsOrigin }));
  app.use(express.json({ limit: "50mb" }));

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true });
  });

  app.use("/api/calls", callsRouter);
  app.use("/api/batches", batchesRouter);
  app.use("/api/import", importRouter);

  app.use((err, _req, res, _next) => {
    console.error(err);
    res.status(500).json({ error: err.message || "internal error" });
  });

  return app;
}
