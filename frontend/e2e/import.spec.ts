import { expect, test, type Page } from "@playwright/test";

const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";
const background = (id: string) => ({
  id, width: 600, height: 400, source_image: `${id}.tif`,
  url: `/files/${id}.png`, thumbnail: "/files/thumb.png", split: "train",
});

function gate() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => { release = resolve; });
  return { promise, release };
}

async function fixture(page: Page, options: {
  openGate?: ReturnType<typeof gate>;
  previewGate?: ReturnType<typeof gate>;
  failure?: "open" | "preview";
  rejectFirstUpload?: boolean;
} = {}) {
  let uploads = 0;
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/backgrounds")
      return route.fulfill({ json: [background("first"), background("second")] });
    if (path === "/api/upload") {
      uploads++;
      if (options.rejectFirstUpload && uploads === 1)
        return route.fulfill({ status: 400, json: { detail: "Invalid TIFF" } });
      return route.fulfill({ json: background("uploaded") });
    }
    if (path.endsWith("/open")) {
      const id = path.split("/")[3];
      if (id === "second") {
        await options.openGate?.promise;
        if (options.failure === "open")
          return route.fulfill({ status: 422, json: { detail: "TIFF source changed" } });
      }
      return route.fulfill({ json: background(id) });
    }
    if (path.startsWith("/files/")) {
      if (path === "/files/second.png" || path === "/files/uploaded.png") {
        await options.previewGate?.promise;
        if (options.failure === "preview") return route.fulfill({ status: 500, body: "Preview failed" });
      }
      return route.fulfill({ contentType: "image/png", body: Buffer.from(png, "base64") });
    }
    return route.fulfill({ json: path === "/api/library" ? [] : {} });
  });
  await page.goto("/");
}

test("background import stays visible until the preview is ready", async ({ page }) => {
  const opening = gate(), preview = gate();
  await fixture(page, { openGate: opening, previewGate: preview });
  await page.getByRole("button", { name: "Chọn second.tif", exact: true }).click();
  await expect(page.getByRole("progressbar", { name: /import/i })).toBeVisible();
  await expect(page.getByRole("button", { name: "Chọn first.tif", exact: true })).toBeDisabled();
  opening.release();
  await expect(page.getByRole("status")).not.toContainText("Import thành công");
  await expect(page.getByRole("progressbar", { name: /import/i })).toBeVisible();
  preview.release();
  await expect(page.getByRole("status")).toContainText("Import thành công");
  await expect(page.locator(".editor-footer")).toContainText("second.tif");
});

for (const failure of ["open", "preview"] as const) {
  test(`failed ${failure} import keeps the previous background`, async ({ page }) => {
    await fixture(page, { failure });
    await page.getByRole("button", { name: "Chọn first.tif", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Import thành công");
    await page.getByRole("button", { name: /^Nền OK/ }).click();
    await page.getByRole("button", { name: "Chọn second.tif", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Import thất bại");
    await expect(page.locator(".editor-footer")).toContainText("first.tif");
    await expect(page.getByRole("button", { name: "Chọn first.tif", exact: true })).toBeEnabled();
  });
}

test("TIFF upload reports server processing and waits for preview", async ({ page }) => {
  await page.addInitScript(() => {
    type Controls = { progress: (loaded: number) => void; finish: () => void; respond: () => void };
    const testWindow = window as typeof window & { testUpload: Controls };
    class UploadRequest {
      upload: { onprogress?: (event: ProgressEvent) => void; onload?: () => void } = {};
      onload?: () => void;
      status = 0;
      response: unknown;
      responseType = "";
      open() {}
      send() {
        testWindow.testUpload = {
          progress: (loaded) => this.upload.onprogress?.(new ProgressEvent("progress", {
            lengthComputable: true, loaded, total: 100,
          })),
          finish: () => this.upload.onload?.(),
          respond: () => {
            this.status = 200;
            this.response = { id: "uploaded", width: 600, height: 400, source_image: "uploaded.tif", url: "/files/uploaded.png" };
            this.onload?.();
          },
        };
      }
    }
    window.XMLHttpRequest = UploadRequest as unknown as typeof XMLHttpRequest;
  });
  const preview = gate();
  await fixture(page, { previewGate: preview });
  await page.locator('input[type="file"][accept*=".tif"]').setInputFiles({
    name: "uploaded.tif", mimeType: "image/tiff", buffer: Buffer.alloc(128 * 1024, 1),
  });
  await expect(page.getByRole("progressbar", { name: /import/i })).toBeVisible();
  await page.evaluate(() => (window as typeof window & { testUpload: { progress: (loaded: number) => void } }).testUpload.progress(45));
  await expect(page.getByRole("progressbar", { name: /import/i })).toHaveAttribute("value", "45");
  await expect(page.getByRole("status")).toContainText("45%");
  await page.evaluate(() => (window as typeof window & { testUpload: { finish: () => void } }).testUpload.finish());
  await expect(page.getByRole("status")).toContainText(/xử lý/i);
  await expect(page.getByRole("progressbar", { name: /import/i })).not.toHaveAttribute("value");
  await page.evaluate(() => (window as typeof window & { testUpload: { respond: () => void } }).testUpload.respond());
  await expect(page.getByRole("status")).toContainText(/xem trước/i);
  await expect(page.getByRole("status")).not.toContainText("Import thành công");
  preview.release();
  await expect(page.getByRole("status")).toContainText("Import thành công");
  await expect(page.locator(".editor-footer")).toContainText("uploaded.tif");
});

test("failed TIFF upload can be retried by dropping a file onto the studio", async ({ page }) => {
  await fixture(page, { rejectFirstUpload: true });
  const input = page.locator('input[type="file"][accept*=".tif"]');
  await input.setInputFiles({ name: "uploaded.tif", mimeType: "image/tiff", buffer: Buffer.alloc(128, 1) });
  await expect(page.getByRole("status")).toContainText("Import thất bại");
  const transfer = await page.evaluateHandle(() => {
    const data = new DataTransfer();
    data.items.add(new File([new Uint8Array(128)], "uploaded.tif", { type: "image/tiff" }));
    return data;
  });
  await page.getByTestId("studio-canvas").dispatchEvent("drop", { dataTransfer: transfer });
  await expect(page.getByRole("status")).toContainText("Import thành công");
  await expect(page.locator(".editor-footer")).toContainText("uploaded.tif");
});
