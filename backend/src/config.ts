import "dotenv/config";

// Centralized configuration with sensible defaults for local dev.
export const config = {
  port: Number(process.env.PORT ?? 4000),
  wsPath: "/ws",
  uiOrigin: process.env.UI_ORIGIN ?? "*",
  deviceToken: process.env.DEVICE_TOKEN ?? "", // optional shared token for devices
  recentLimit: 300,
};
