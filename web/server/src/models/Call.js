import mongoose from "mongoose";

const callSchema = new mongoose.Schema(
  {
    callId: { type: String, required: true, unique: true, index: true },
    batchTag: { type: String, required: true, index: true },
    date: { type: String, required: true, index: true },
    startedAt: { type: Date, index: true },
    transcript: { type: String, default: "" },
    summary: { type: mongoose.Schema.Types.Mixed, default: {} },
    summaryMd: { type: String, default: "" },
    intent: { type: String, default: "", index: "text" },
    topics: { type: [String], default: [] },
    escalationNeeded: { type: Boolean, default: false, index: true },
    escalationSeverity: {
      type: String,
      enum: ["low", "medium", "high", ""],
      default: "",
      index: true,
    },
    escalationReasons: { type: [String], default: [], index: true },
  },
  { timestamps: true },
);

callSchema.index({ intent: "text", callId: "text" });

export const Call = mongoose.model("Call", callSchema);
