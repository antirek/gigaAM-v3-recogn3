import mongoose from "mongoose";
import { config } from "./config.js";

export async function connectDb() {
  mongoose.set("strictQuery", true);
  await mongoose.connect(config.mongoUri);
  console.log(`[db] connected: ${config.mongoUri}`);
}

export async function disconnectDb() {
  await mongoose.disconnect();
}
