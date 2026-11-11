import { test, expect } from "@playwright/test";

const png =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";
test("full-frame studio requires final label review before export", async ({
  page,
}) => {
  let approved = false;
  let polygons: number[][] = [];
  let submitted: Record<string, unknown> = {};
  let legacyScene = false;
  const result = () => ({
    id: "result-1",
    preview: "/files/result.png",
    edit_mask: "/files/mask.png",
    width: 600,
    height: 400,
    polygons,
    review_status: approved ? "approved" : "pending",
    scene: legacyScene
      ? { image_id: "image-1", placements: [], brushes: [], engine: "sdxl" }
      : submitted,
    warnings: ["PCT-Net checkpoint chưa có; đã dùng Classical."],
    provenance: { layers: [{ kind: "stamp", requested_method: "pctnet", effective_method: "classical_mixed" }] },
  });
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith("/api/"))
      return route.fulfill({
        contentType: "image/png",
        body: Buffer.from(png, "base64"),
      });
    let body: unknown = {};
    if (path === "/api/health") body = { engine: "ready", trained: false, ai: { model_available: false } };
    else if (path === "/api/models/status") body = { models: ["pctnet", "harmonizer", "rainnet", "sdxl", "sd15"].map(model_id => ({ model_id, status: "ready" })) };
    else if (path === "/api/dataset") body = { status: "PASS", images: 186 };
    else if (path === "/api/library")
      body = [
        {
          id: "stamp-1",
          rgb: "rgb.png",
          mask: "mask.png",
          thumbnail: "thumb.png",
          width: 40,
          height: 20,
          review_status: "PENDING_VISUAL_REVIEW",
        },
      ];
    else if (path === "/api/backgrounds")
      body = [
        {
          id: "ok-1",
          width: 600,
          height: 400,
          url: "/files/bg.png",
          source_image: "F971-OK.jpg",
          split: "train",
        },
      ];
    else if (path.endsWith("/open"))
      body = { id: "image-1", width: 600, height: 400, url: "/files/bg.png" };
    else if (path === "/api/jobs" && route.request().method() === "POST") {
      submitted = route.request().postDataJSON();
      body = { id: "job-1", status: "queued" };
    }
    else if (path === "/api/jobs/job-1")
      body = { id: "job-1", status: "succeeded", result: result() };
    else if (path.endsWith("/annotations")) {
      polygons = route.request().postDataJSON().polygons;
      approved = false;
      body = result();
    } else if (path.endsWith("/approve")) {
      approved = true;
      body = result();
    } else if (path.endsWith("/export")) body = { url: "/files/training.zip" };
    else if (path === "/api/exports") body = [result()];
    else if (path === "/api/results/result-1") body = result();
    return route.fulfill({ json: body });
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Tạo dữ liệu xước F971" }),
  ).toBeVisible();
  await expect(page.getByLabel("Phương pháp ghép Stamp", { exact: true })).toHaveValue("classical_mixed");
  await expect(page.getByLabel("Phương pháp Vẽ nét", { exact: true })).toHaveValue("procedural");
  await page.getByRole("button", { name: "Chọn F971-OK.jpg" }).click();
  await page.getByRole("button", { name: "Thêm stamp-1" }).click();
  await page.getByLabel("Phương pháp ghép Stamp", { exact: true }).selectOption("pctnet");
  await page.getByLabel("Phương pháp Vẽ nét", { exact: true }).selectOption("sd15");
  await page.getByRole("button", { name: "╱ Vẽ xước" }).click();
  const drawingBox = (await page.getByTestId("studio-canvas").boundingBox())!;
  await page.mouse.move(drawingBox.x + 100, drawingBox.y + 100);
  await page.mouse.down();
  await page.mouse.move(drawingBox.x + 160, drawingBox.y + 130);
  await page.mouse.up();
  await page.getByRole("button", { name: "Tạo ảnh" }).click();
  await expect(
    page.getByRole("heading", { name: "Duyệt ảnh và nhãn" }),
  ).toBeVisible();
  expect(submitted.stamp_method).toBe("pctnet");
  expect(submitted.brush_method).toBe("sd15");
  expect(submitted).not.toHaveProperty("engine");
  expect((submitted.placements as unknown[]).length).toBe(1);
  expect((submitted.brushes as unknown[]).length).toBe(1);
  await expect(page.getByRole("status")).toContainText("đã dùng Classical");
  await expect(page.getByTestId("effective-methods")).toContainText("Mixed SeamlessClone");
  await expect(
    page.getByRole("button", { name: "Xuất bộ training" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Vẽ polygon mới" }).click();
  const canvas = page.getByTestId("review-canvas");
  const box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + 60, box.y + 60);
  await page.mouse.click(box.x + 160, box.y + 60);
  await page.mouse.click(box.x + 120, box.y + 140);
  await page.getByRole("button", { name: "Đóng polygon" }).click();
  // Vertex handles must move independently of the panning stage.
  const start = await canvas.boundingBox();
  await page.mouse.move(start!.x + 60, start!.y + 60);
  await page.mouse.down();
  await page.mouse.move(start!.x + 75, start!.y + 70);
  await page.mouse.up();
  await page.getByRole("button", { name: "Lưu nhãn" }).click();
  await page
    .getByLabel("Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc")
    .check();
  await page.getByRole("button", { name: "Duyệt ảnh và nhãn" }).click();
  await page.getByRole("button", { name: "Xuất bộ training" }).click();
  await expect(
    page.getByRole("link", { name: "Tải bộ training ZIP" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Xóa polygon 1" }).click();
  await expect(
    page.getByRole("button", { name: "Xuất bộ training" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("link", { name: "Tải bộ training ZIP" }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Mở lại scene để chỉnh cách đặt xước" })
    .click();
  await expect(page.getByLabel("Phương pháp ghép Stamp", { exact: true })).toHaveValue("pctnet");
  await expect(page.getByLabel("Phương pháp Vẽ nét", { exact: true })).toHaveValue("sd15");
  legacyScene = true;
  await page.getByRole("button", { name: "Ảnh đã tạo", exact: true }).click();
  await expect(page.locator(".result-card").first()).toContainText("Mixed SeamlessClone");
  await page.locator(".result-card").first().click();
  await page.getByRole("button", { name: "Mở lại scene để chỉnh cách đặt xước" }).click();
  await expect(page.getByLabel("Phương pháp ghép Stamp", { exact: true })).toHaveValue("classical_mixed");
  await expect(page.getByLabel("Phương pháp Vẽ nét", { exact: true })).toHaveValue("sdxl");
  await expect(
    page.getByRole("heading", { name: "Tạo dữ liệu xước F971" }),
  ).toBeVisible();
});

test("API failure is visible and never reported as generated AI", async ({
  page,
}) => {
  await page.route("http://127.0.0.1:8000/**", (route) =>
    route.fulfill({ status: 503, json: { detail: "Dataset chưa được nhập" } }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert").first()).toContainText("Dataset chưa được nhập");
  await expect(page.getByRole("button", { name: "Tạo ảnh" })).toBeDisabled();
});


test("model downloads preserve selection until verified and cancellation preserves old method", async ({ page }) => {
  let state = "not_downloaded";
  let cancelled = false;
  await page.route("http://127.0.0.1:8000/**", async route => {
    const path = new URL(route.request().url()).pathname;
    let body: unknown = [];
    if (path === "/api/models/status") body = {models: [{model_id: "pctnet", status: state, total_bytes: 100000000, downloaded_bytes: 25000000, progress: 25, size_label: "100 MB"}]};
    if (path === "/api/models/download") { state = "downloading"; body = {model_id: "pctnet", status: state}; }
    if (path === "/api/models/cancel") { cancelled = true; state = "cancelled"; body = {model_id: "pctnet", status: state}; }
    if (path === "/api/health") body = {};
    if (path === "/api/dataset") body = {};
    await route.fulfill({json: body});
  });
  await page.goto("/");
  const select = page.getByLabel("Phương pháp ghép Stamp", {exact: true});
  await expect(page.getByTestId("model-pctnet")).toContainText("Chưa tải");
  await select.selectOption("pctnet");
  await expect(select).toHaveValue("classical_mixed");
  await expect(page.getByTestId("model-pctnet")).toContainText("25%");
  await page.getByRole("button", {name: "Hủy PCT-Net"}).click();
  await expect.poll(() => cancelled).toBe(true);
  await expect(select).toHaveValue("classical_mixed");
  await page.getByRole("button", {name: "Tải PCT-Net"}).click();
  state = "verifying";
  await expect(page.getByTestId("model-pctnet")).toContainText("Đang kiểm tra");
  await expect(select).toHaveValue("classical_mixed");
  state = "ready";
  await expect(select).toHaveValue("pctnet");
  await expect(page.getByTestId("model-pctnet")).toContainText("Đã tải");
  state = "not_downloaded";
  await expect(select).toHaveValue("classical_mixed");
  await expect(page.getByText("Model đang chọn chưa sẵn sàng; đã chuyển về phương pháp CPU.")).toBeVisible();
});
