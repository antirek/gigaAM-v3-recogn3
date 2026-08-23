#!/usr/bin/env node
import path from "path";
import { fileURLToPath } from "url";
import { connectDb, disconnectDb } from "../src/db.js";
import { importBatchFromDir } from "../src/services/importBatch.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..", "..");

async function main() {
  const batchArg = process.argv[2];
  if (!batchArg) {
    console.error("Usage: npm run import -- <batch-dir>");
    console.error("Example: npm run import -- out/outgoing_answered_gt30_2026-08-19");
    process.exit(1);
  }

  const batchDir = path.isAbsolute(batchArg)
    ? batchArg
    : path.join(projectRoot, batchArg);

  await connectDb();
  try {
    const result = await importBatchFromDir(batchDir);
    console.log(
      `[import] batch=${result.batchTag} calls=${result.nCalls} batchSummary=${result.batch ? "yes" : "no"}`,
    );
  } finally {
    await disconnectDb();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
