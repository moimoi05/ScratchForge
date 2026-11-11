import { test, expect, type Page } from "@playwright/test";

const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";
const original = { id: "stamp-real", rgb: "stamp-real/stamp_crop.tif", mask: "stamp-real/mask.png",
  preview_alpha: "stamp-real/preview_alpha.png", width: 600, height: 400,
  review_status: "APPROVED", origin: "import", segmentation: [[20, 20, 100, 20, 100, 40, 20, 40]] };

async function fixture(page: Page) {
  let stamps = [original];
  let cancelled = false;
  let operation = "";
  const posts: { path: string; body: unknown }[] = [];
  const downloads: string[] = [];
  await page.route("http://127.0.0.1:8000/**", async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.startsWith("/files/") || path.startsWith("/library-files/")) {
      expect(path.endsWith(".tif")).toBeFalsy();
      return route.fulfill({ contentType: "image/png", body: Buffer.from(png, "base64") });
    }
    if (path === "/api/library") return route.fulfill({ json: stamps });
    if (path === "/api/training") return route.fulfill({ json: { checkpoints: [
      { id: "lora-sd15", name: "F971 SD15", base_model: "sd15" },
      { id: "lora-sdxl", name: "F971 SDXL", base_model: "sdxl" },
    ] } });
    if (path === "/api/models/status") return route.fulfill({ json: { models: [
      { model_id: "sd15", status: "ready" }, { model_id: "sdxl", status: "ready" },
    ] } });
    if (path === "/api/models/download") { downloads.push(route.request().postDataJSON().model_id); return route.fulfill({ json: {} }); }
    if (path === "/api/health") return route.fulfill({ json: { brush_models: { sd15: { model_available: true }, sdxl: { model_available: true } } } });
    if (path === "/api/library/import") {
      posts.push({ path, body: route.request().postData() });
      stamps = [...stamps, { ...original, id: "stamp-imported", review_status: "pending" }];
      return route.fulfill({ json: { stamps: [stamps[1]] } });
    }
    if (path.startsWith("/api/library/") && path.endsWith("/approve")) {
      const id = path.split("/")[3];
      expect(route.request().postDataJSON()).toEqual({ confirmed: true });
      stamps = stamps.map(stamp => stamp.id === id ? { ...stamp, review_status: "APPROVED" } : stamp);
      return route.fulfill({ json: stamps.find(stamp => stamp.id === id) });
    }
    if (["/api/library/augment", "/api/training/lora", "/api/library/generate"].includes(path)) {
      operation = path;
      posts.push({ path, body: route.request().postDataJSON() });
      cancelled = false;
      return route.fulfill({ json: { id: "job-library", status: "queued", progress: 0 } });
    }
    if (path === "/api/jobs/job-library/cancel") { cancelled = true; return route.fulfill({ json: { id: "job-library", status: "cancelled", progress: 25 } }); }
    if (path === "/api/jobs/job-library") {
      if (operation === "/api/library/augment") {
        stamps = [...stamps.filter(s => s.id !== "stamp-augmented"), { ...original, id: "stamp-augmented", review_status: "pending", origin: "augment" }];
        return route.fulfill({ json: { id: "job-library", status: "succeeded", progress: 100, operation_result: { stamps: [stamps[stamps.length - 1]] } } });
      }
      return route.fulfill({ json: { id: "job-library", status: cancelled ? "cancelled" : "running", progress: 25 } });
    }
    return route.fulfill({ json: ["/api/backgrounds", "/api/exports"].includes(path) ? [] : {} });
  });
  return { posts, downloads };
}

test("hover explains disabled toolbar and training actions without enabling them", async ({ page }) => {
  const { posts } = await fixture(page);
  await page.goto("/");
  const draw = page.getByRole("button", { name: /Vẽ xước/ });
  await expect(draw).toBeDisabled();
  await draw.hover();
  await expect(page.getByRole("tooltip")).toContainText(/mở ảnh nền/i);
  await page.getByRole("button", { name: /Undo/ }).hover();
  await expect(page.getByRole("tooltip")).toContainText(/chưa có thao tác/i);
  await page.getByRole("button", { name: "Lật", exact: true }).hover();
  await expect(page.getByRole("tooltip")).toContainText(/chọn một stamp/i);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("tooltip")).toHaveCount(0);

  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const train = page.getByRole("button", { name: "B · Train LoRA F971", exact: true });
  await expect(train).toBeDisabled();
  await train.hover();
  await expect(page.getByRole("tooltip")).toContainText(/tích ít nhất một stamp/i);
  await page.getByLabel("Chọn stamp-real", { exact: true }).check();
  await expect(train).toBeEnabled();
  await train.focus();
  await expect(page.getByRole("tooltip")).toContainText(/huấn luyện/i);
  expect(posts).toHaveLength(0);
});

test("stamp cards zoom into the scratch and can open preview without a background", async ({ page }) => {
  await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: /^Vết xước/ }).click();
  const card = page.getByRole("button", { name: "Thêm stamp-real" });
  await expect(card).toBeDisabled();
  await expect(card.locator("svg")).not.toHaveAttribute("viewBox", "0 0 600 400");
  await card.hover();
  await page.getByRole("button", { name: "Phóng to stamp-real" }).click();
  await expect(page.getByRole("dialog", { name: "Xem vết xước" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Xem vết xước" })).toHaveCount(0);
});

test("source sidebar owns A/B/C and publishes augmentation into studio library", async ({ page }) => {
  const { posts } = await fixture(page);
  await page.goto("/");
  await expect(page.getByTestId("source-library-panel")).toHaveCount(0);
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  await expect(panel).toBeVisible();
  await panel.getByLabel("Chọn stamp-real", { exact: true }).check();
  await panel.getByRole("button", { name: "A · Augment Stamp", exact: true }).click();
  await expect.poll(() => posts.length).toBe(1);
  expect(posts[0].body).toMatchObject({ stamp_ids: ["stamp-real"], copies: 1 });
  await expect(panel.getByText("stamp-augmented", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Bàn dựng", exact: true }).click();
  await page.getByRole("button", { name: /^Vết xước/ }).click();
  await expect(page.getByRole("button", { name: "Thêm stamp-augmented", exact: true })).toBeVisible();
});

test("source pair import validates declaration and refreshes stamps", async ({ page }) => {
  const { posts } = await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  await panel.getByLabel("Kiểu import stamp").selectOption("crop");
  await panel.getByLabel("Ảnh TIFF stamp").setInputFiles({ name: "crop.tif", mimeType: "image/tiff", buffer: Buffer.from("fixture") });
  await panel.getByLabel("Mask stamp PNG").setInputFiles({ name: "mask.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") });
  await panel.getByLabel("ID nguồn train").fill("new-f971");
  await expect(panel.getByRole("button", { name: "Import vào kho", exact: true })).toBeDisabled();
  await expect(panel.getByRole("button", { name: "Import vào kho", exact: true })).toHaveAccessibleDescription(/tích.*Tôi xác nhận nguồn thuộc tập train/i);
  await panel.getByLabel("Tôi xác nhận nguồn thuộc tập train").check();
  await panel.getByRole("button", { name: "Import vào kho", exact: true }).click();
  await expect(panel.getByText("stamp-imported", { exact: true })).toBeVisible();
  expect(String(posts[0].body)).toContain('name="confirmed_train"');
});

test("LoRA training reports real progress and supports cancel", async ({ page }) => {
  const { posts } = await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  const train = panel.getByRole("button", { name: "B · Train LoRA F971", exact: true });
  await expect(train).toBeDisabled();
  await expect(train).toHaveAccessibleDescription(/tích ít nhất một stamp/i);
  await expect(panel.getByRole("button", { name: "C · Sinh stamp mới" })).toHaveAccessibleDescription(/chọn.*LoRA tương thích/i);
  await panel.getByLabel("Chọn stamp-real", { exact: true }).check();
  await expect(train).toBeEnabled();
  await train.click();
  await expect(panel.getByRole("progressbar", { name: "Tiến trình kho stamp" })).toHaveAttribute("value", "25");
  expect(posts[0].body).toMatchObject({ base_model: "sd15", steps: 1000, stamp_ids: ["stamp-real"] });
  await panel.getByRole("button", { name: "Hủy job kho stamp", exact: true }).click();
  await expect(panel.getByText("Đã hủy", { exact: true })).toBeVisible();
});

test("choosing an unavailable base starts download and locks B/C until verified", async ({ page }) => {
  await fixture(page);
  let status = "not_downloaded";
  const downloads: string[] = [];
  await page.route("**/api/models/status", route => route.fulfill({ json: { models: [
    { model_id: "sd15", status: "ready" }, { model_id: "sdxl", status, progress: 50 },
  ] } }));
  await page.route("**/api/models/download", route => {
    downloads.push(route.request().postDataJSON().model_id);
    status = "downloading";
    return route.fulfill({ json: { model_id: "sdxl", status } });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  const base = panel.getByLabel("Base model kho stamp");
  const train = panel.getByRole("button", { name: "B · Train LoRA F971", exact: true });
  await panel.getByLabel("Chọn stamp-real", { exact: true }).check();
  await expect(base.locator('option[value="sdxl"]')).toContainText("Chưa tải");
  await base.selectOption("sdxl");
  await expect.poll(() => downloads).toEqual(["sdxl"]);
  await expect(panel.getByRole("progressbar", { name: "Tiến trình SDXL" })).toBeVisible();
  await expect(train).toBeDisabled();
  await expect(train).toHaveAccessibleDescription(/đợi.*model/i);
  await expect(panel.getByRole("button", { name: "Tải SDXL", exact: true })).toHaveCount(0);
  status = "ready";
  await expect(base).toHaveValue("sdxl");
  await expect(train).toBeEnabled();
});

test("Stamp Brush never downloads a model and AI LoRA is filtered by base", async ({ page }) => {
  const { downloads } = await fixture(page);
  await page.goto("/");
  await page.getByLabel("Phương pháp Vẽ nét", { exact: true }).selectOption("stamp_brush");
  await expect(page.getByLabel("Nguồn Stamp Brush")).toBeVisible();
  await expect(page.getByLabel("Nguồn Stamp Brush")).toHaveValue("random");
  await expect(page.getByLabel("Nguồn Stamp Brush").locator("option").first()).toHaveAttribute("value", "random");
  await expect(page.getByLabel("Mức biến đổi", { exact: true })).toBeDisabled();
  expect(downloads).toEqual([]);
  await page.getByLabel("Phương pháp Vẽ nét", { exact: true }).selectOption("sd15");
  await expect(page.getByLabel("LoRA F971 cho nét AI").locator("option")).toHaveCount(2);
  await page.getByLabel("LoRA F971 cho nét AI").selectOption("lora-sd15");
  await expect(page.getByText("Đã chọn LoRA · sẽ nạp khi chạy", { exact: true })).toBeVisible();
  await expect(page.getByText("Đã nạp LoRA F971", { exact: true })).toHaveCount(0);
});

test("full-frame pair import reports failure and can retry without losing the library", async ({ page }) => {
  await fixture(page);
  let attempts = 0;
  await page.route("**/api/library/import", route => {
    attempts++;
    expect(route.request().postData()).toContain('filename="polygons.json"');
    return route.fulfill(attempts === 1 ? { status: 422, json: { detail: "COCO dimensions do not match TIFF" } } : { json: [original] });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  await panel.getByLabel("Ảnh TIFF stamp").setInputFiles({ name: "frame.tif", mimeType: "image/tiff", buffer: Buffer.from("fixture") });
  await panel.getByLabel("Polygon stamp JSON").setInputFiles({ name: "polygons.json", mimeType: "application/json", buffer: Buffer.from('{"images":[],"annotations":[]}') });
  await panel.getByLabel("ID nguồn train").fill("train-frame");
  await panel.getByLabel("Tôi xác nhận nguồn thuộc tập train").check();
  await panel.getByRole("button", { name: "Import vào kho", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("COCO dimensions do not match TIFF");
  await expect(panel.getByText("stamp-real", { exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Import vào kho", exact: true }).click();
  await expect(panel.getByText(/Import thành công/)).toBeVisible();
  expect(attempts).toBe(2);
});

test("stamp polygons can be confirmed without redrawing and synthetic failures remain visible", async ({ page }) => {
  await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Dữ liệu nguồn", exact: true }).click();
  const panel = page.getByTestId("source-library-panel");
  await panel.getByRole("button", { name: "Duyệt stamp-real", exact: true }).click();
  const review = page.getByRole("dialog", { name: "Duyệt stamp", exact: true });
  await expect(review.getByTestId("stamp-review-canvas")).toHaveAttribute("data-image-loaded", "true");
  await expect(review.getByRole("button", { name: "Duyệt stamp", exact: true })).toBeDisabled();
  await review.getByLabel("Tôi đã kiểm tra stamp và polygon").check();
  await review.getByRole("button", { name: "Duyệt stamp", exact: true }).click();
  await expect(review.getByText(/APPROVED/)).toBeVisible();
  await review.getByRole("button", { name: "Đóng duyệt stamp", exact: true }).click();
  await panel.getByLabel("Chọn stamp-real", { exact: true }).check();
  await panel.getByLabel("LoRA sinh stamp").selectOption("lora-sd15");
  await page.route("**/api/jobs/job-library", route => route.fulfill({ json: { id: "job-library", status: "failed", progress: 0, error: "CUDA ran out of VRAM" } }));
  await panel.getByRole("button", { name: "C · Sinh stamp mới", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("CUDA ran out of VRAM");
  await expect(panel.getByRole("button", { name: "C · Sinh stamp mới", exact: true })).toBeEnabled();
});

test("restoring an automatically labelled result preserves Stamp Brush source and LoRA settings", async ({ page }) => {
  await fixture(page);
  const result = { id: "result-library", preview: "/files/result.png", edit_mask: "/files/edit.png", width: 600, height: 400,
    polygons: original.segmentation, review_status: "pending", scene: {
      image_id: "input-library", placements: [], brushes: [{ points: [20, 20, 100, 40], width: 8, stamp_id: "stamp-real" }],
      brush_method: "stamp_brush", brush_stamp_mode: "random", brush_stamp_id: "stamp-real", seed: 76, strength: .55,
      stamp_method: "classical_laplacian", lora_id: "lora-sd15",
    } };
  await page.route("**/api/exports", route => route.fulfill({ json: [result] }));
  await page.route("**/api/results/result-library", route => route.fulfill({ json: result }));
  await page.goto("/");
  await page.getByRole("button", { name: "Ảnh đã tạo", exact: true }).click();
  await page.getByRole("button", { name: /result-libra/ }).click();
  await expect(page.getByRole("button", { name: "Xuất bộ training", exact: true })).toBeDisabled();
  await expect(page.getByLabel("Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc")).toBeEnabled();
  await page.getByRole("button", { name: "Mở lại scene để chỉnh cách đặt xước", exact: true }).click();
  await expect(page.getByLabel("Phương pháp Vẽ nét", { exact: true })).toHaveValue("stamp_brush");
  await expect(page.getByLabel("Nguồn Stamp Brush")).toHaveValue("random");
  await expect(page.getByLabel("Seed", { exact: true })).toHaveValue("76");
  await expect(page.getByLabel("Phương pháp ghép Stamp", { exact: true })).toHaveValue("classical_laplacian");
});
