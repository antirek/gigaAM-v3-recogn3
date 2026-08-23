export const config = {
  port: Number(process.env.WEB_PORT || 3000),
  mongoUri: process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/gigaam_calls",
  corsOrigin: process.env.CORS_ORIGIN || "http://localhost:5173",
};
