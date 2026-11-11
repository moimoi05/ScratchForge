import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  testMatch: ["studio.spec.ts", "import.spec.ts", "library.spec.ts"],
  use: {
    baseURL: "http://127.0.0.1:5173",
    headless: true,
    channel: "chromium",
  },
  webServer: {
    command: "npm run dev -- --strictPort",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
  },
});
