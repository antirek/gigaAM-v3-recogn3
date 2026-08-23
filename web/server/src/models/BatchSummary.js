import mongoose from "mongoose";

const batchSummarySchema = new mongoose.Schema(
  {
    batchTag: { type: String, required: true, unique: true, index: true },
    date: { type: String, required: true, index: true },
    data: { type: mongoose.Schema.Types.Mixed, default: {} },
  },
  { timestamps: true },
);

export const BatchSummary = mongoose.model("BatchSummary", batchSummarySchema);
