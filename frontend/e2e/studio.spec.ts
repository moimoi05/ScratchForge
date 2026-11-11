import { test, expect } from "@playwright/test";

const png =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";

test("Segment edits TIFF annotations and closes a polygon at its first vertex", async ({ page }) => {
  type Annotation = { id: number | null; category_id: number; segmentation: number[][] };
  let annotations: Annotation[] = [{ id: 23, category_id: 1, segmentation: [[60, 60, 160, 60, 120, 140, 60, 60]] }];
  let revision = 0;
  let approved = false;
  let allowPreview: () => void;
  const previewGate = new Promise<void>((resolve) => { allowPreview = resolve; });
  const frame = (large: boolean) => ({
    id: large ? "s942" : "f971",
    group: large ? "s942-ok" : "f971-scratch",
    file_name: large ? "large.tif" : "scratch.tif",
    width: large ? 11004 : 600,
    height: large ? 5599 : 400,
    split: large ? null : "train",
    preview: "/files/segment-fixture.png",
    thumbnail: "/files/segment-fixture.png",
    annotation_count: large ? 0 : annotations.length,
    review_status: !large && approved ? "approved" : "pending",
    revision: large ? 0 : revision,
  });
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/files/segment-fixture.png") {
      await previewGate;
      return route.fulfill({ contentType: "image/png", body: Buffer.from(png, "base64") });
    }
    if (path === "/api/segment/images") return route.fulfill({ json: [frame(false), frame(true)] });
    if (path.endsWith("/annotations")) {
      const body = route.request().postDataJSON();
      expect(body.revision).toBe(revision);
      annotations = body.annotations.map((item: Annotation, index: number) => ({ ...item, id: item.id ?? 100 + index }));
      revision++;
      approved = false;
    }
    if (path.endsWith("/approve")) {
      expect(route.request().postDataJSON().revision).toBe(revision);
      approved = true;
      revision++;
    }
    if (path.startsWith("/api/segment/images/")) {
      const large = path.endsWith("/s942");
      return route.fulfill({ json: { ...frame(large), annotations: large ? [] : annotations } });
    }
    return route.fulfill({ json: ["/api/backgrounds", "/api/library"].includes(path) ? [] : {} });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Segment", exact: true }).click();
  await page.getByRole("button", { name: "Mở scratch.tif", exact: true }).click();
  const canvas = page.getByTestId("segment-canvas");
  await expect(page.getByRole("button", { name: "Duyệt nhãn", exact: true })).toBeDisabled();
  allowPreview!();
  await expect(canvas).toHaveAttribute("data-image-loaded", "true");
  await expect(page.getByText("3 đỉnh", { exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "100%", exact: true }).click();
  const box = (await canvas.locator(".konvajs-content").boundingBox())!;
  await page.getByRole("button", { name: "Thêm đỉnh", exact: true }).click();
  await page.mouse.click(box.x + 110, box.y + 62);
  await expect(page.getByText("4 đỉnh", { exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "Xóa đỉnh", exact: true }).click();
  await page.mouse.click(box.x + 110, box.y + 60);
  await expect(page.getByText("3 đỉnh", { exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "Xóa đỉnh", exact: true }).click();
  await page.mouse.move(box.x + 60, box.y + 60);
  await page.mouse.down();
  await page.mouse.move(box.x + 76, box.y + 67);
  await page.mouse.up();
  await page.getByRole("button", { name: "Vẽ polygon mới" }).click();
  for (const [x, y] of [[230, 100], [280, 100], [265, 150], [234, 101]])
    await page.mouse.click(box.x + x, box.y + y);
  await expect(page.getByRole("button", { name: "Đóng polygon" })).toHaveCount(0);
  await page.getByRole("button", { name: "Lưu nhãn" }).click();
  await expect.poll(() => annotations.length).toBe(2);
  expect(annotations[0].segmentation[0].map(Math.round)).toEqual([76, 67, 160, 60, 120, 140]);
  const drawn = annotations[1].segmentation[0];
  expect(drawn).toHaveLength(6);
  drawn.forEach((coordinate, i) => expect(Math.abs(coordinate - [230, 100, 280, 100, 265, 150][i])).toBeLessThan(1));
  await page.getByRole("button", { name: "Duyệt nhãn", exact: true }).click();
  await expect(page.getByTestId("segment-status")).toHaveText("ĐÃ DUYỆT");
  await page.reload();
  await page.getByRole("button", { name: "Segment", exact: true }).click();
  await page.getByRole("button", { name: "Mở scratch.tif", exact: true }).click();
  await expect(page.getByTestId("segment-status")).toHaveText("ĐÃ DUYỆT");
  await expect(page.getByText("3 đỉnh", { exact: true })).toHaveCount(2);
  await page.getByRole("button", { name: "Xóa polygon 2" }).click();
  await page.getByRole("button", { name: "Lưu nhãn" }).click();
  await expect(page.getByTestId("segment-status")).toHaveText("CHỜ DUYỆT");
  await page.getByRole("button", { name: "S942 OK" }).click();
  await page.getByRole("button", { name: "Mở large.tif" }).click();
  await expect(canvas).toHaveAttribute("data-image-loaded", "true");
  await page.getByRole("button", { name: "100%", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-zoom", "1");
  await expect(page.getByText("11004 × 5599", { exact: true })).toBeVisible();
});

test("Segment advances to the next pending TIFF after successful approval", async ({ page }) => {
  const queue = [
    { id: "first", group: "f971-scratch" },
    { id: "reviewed", group: "f971-scratch" },
    { id: "last", group: "f971-scratch" },
    { id: "background", group: "f971-ok" },
  ];
  const approved = new Set(["reviewed"]);
  let rejectApproval = true;
  const frame = ({ id, group }: (typeof queue)[number]) => ({
    id, group, file_name: `${id}.tif`, width: 600, height: 400, split: "train",
    preview: `/files/${id}.png`, annotation_count: 0, annotations: [],
    review_status: approved.has(id) ? "approved" : "pending",
    revision: approved.has(id) ? 1 : 0,
  });
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.startsWith("/files/"))
      return route.fulfill({ contentType: "image/png", body: Buffer.from(png, "base64") });
    if (path === "/api/segment/images") return route.fulfill({ json: queue.map(frame) });
    const item = queue.find(({ id }) => path === `/api/segment/images/${id}` || path === `/api/segment/images/${id}/approve`);
    if (item) {
      if (path.endsWith("/approve")) {
        if (rejectApproval) {
          rejectApproval = false;
          return route.fulfill({ status: 409, json: { detail: "Review changed" } });
        }
        approved.add(item.id);
      }
      return route.fulfill({ json: frame(item) });
    }
    return route.fulfill({ json: ["/api/backgrounds", "/api/library"].includes(path) ? [] : {} });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Segment", exact: true }).click();
  const first = page.getByRole("button", { name: "Mở first.tif", exact: true });
  const last = page.getByRole("button", { name: "Mở last.tif", exact: true });
  const approve = page.getByRole("button", { name: "Duyệt nhãn", exact: true });
  await first.click();
  await approve.click();
  await expect(page.getByText(/Review changed/)).toBeVisible();
  await expect(first).toHaveAttribute("aria-current", "true");
  await approve.click();
  await expect(last).toHaveAttribute("aria-current", "true");
  await expect(first).toContainText("Đã duyệt");
  await expect(page.getByTestId("segment-status")).toHaveText("CHỜ DUYỆT");
  await approve.click();
  await expect(last).toHaveAttribute("aria-current", "true");
  await expect(page.getByTestId("segment-status")).toHaveText("ĐÃ DUYỆT");
});

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
  await expect(page.getByRole("status").filter({ hasText: "đã dùng Classical" })).toBeVisible();
  await expect(page.getByTestId("effective-methods")).toContainText("Mixed SeamlessClone");
  await expect(
    page.getByRole("button", { name: "Xuất bộ training" }),
  ).toBeDisabled();
  await expect(page.getByRole("button", { name: "Xuất bộ training" }))
    .toHaveAccessibleDescription("Bấm Duyệt ảnh và nhãn trước khi xuất.");
  await page.getByRole("button", { name: "Vẽ polygon mới" }).click();
  const canvas = page.getByTestId("review-canvas");
  await expect(canvas).toHaveAttribute("data-image-loaded", "true");
  const box = (await canvas.locator(".konvajs-content").boundingBox())!;
  await page.mouse.click(box.x + 60, box.y + 60);
  await page.mouse.click(box.x + 160, box.y + 60);
  await page.mouse.click(box.x + 120, box.y + 140);
  await page.mouse.click(box.x + 60, box.y + 60);
  // Vertex handles must move independently of the panning stage.
  await page.mouse.move(box.x + 60, box.y + 60);
  await page.mouse.down();
  await page.mouse.move(box.x + 75, box.y + 70);
  await page.mouse.up();
  await page.getByRole("button", { name: "Lưu nhãn" }).click();
  await expect(page.getByRole("button", { name: "Duyệt ảnh và nhãn" }))
    .toHaveAccessibleDescription("Tích ô “Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc”.");
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
  await expect(select.locator('option[value="pctnet"]')).toContainText("Chưa tải");
  await expect(page.getByTestId("model-pctnet")).toHaveCount(0);
  await select.selectOption("pctnet");
  await expect(select).toHaveValue("classical_mixed");
  await expect(page.getByTestId("model-pctnet")).toContainText("25%");
  await page.getByRole("button", {name: "Hủy PCT-Net"}).click();
  await expect.poll(() => cancelled).toBe(true);
  await expect(select).toHaveValue("classical_mixed");
  await expect(page.getByRole("button", {name: "Tải PCT-Net", exact: true})).toHaveCount(0);
  await Promise.all([
    page.waitForResponse(response => response.url().endsWith("/api/models/download") && response.request().method() === "POST"),
    select.selectOption("pctnet"),
  ]);
  state = "verifying";
  await expect(page.getByTestId("model-pctnet")).toContainText("Đang kiểm tra");
  await expect(select).toHaveValue("classical_mixed");
  state = "ready";
  await expect(select).toHaveValue("pctnet");
  await expect(select.locator('option[value="pctnet"]')).toContainText("Đã tải");
  await expect(page.getByTestId("model-pctnet")).toHaveCount(0);
  state = "not_downloaded";
  await expect(select).toHaveValue("classical_mixed");
  await expect(page.getByText("Model đang chọn chưa sẵn sàng; đã chuyển về phương pháp CPU.")).toBeVisible();
});
