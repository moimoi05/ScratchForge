import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  testMatch: "real.spec.ts",
  timeout: 180000,
  use: {
    baseURL: "http://127.0.0.1:8000",
    channel: "chromium",
    headless: true,
    viewport: { width: 1440, height: 1100 },
  },
});
