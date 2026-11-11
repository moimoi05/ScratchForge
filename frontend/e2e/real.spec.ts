import { test, expect } from "@playwright/test";

test("Segment renders a native S942 TIFF at 100% zoom", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Segment", exact: true }).click();
  await page.getByRole("button", { name: "S942 OK", exact: true }).click();
  await page.getByRole("button", { name: /^Mở .*\.tif$/i }).first().click();
  const segment = page.getByTestId("segment-canvas");
  await expect(segment).toHaveAttribute("data-image-loaded", "true", { timeout: 90000 });
  await page.getByRole("button", { name: "100%", exact: true }).click();
  await expect(segment).toHaveAttribute("data-zoom", "1");
  await expect(page.getByText("11004 × 5599", { exact: true })).toBeVisible();
  await expect.poll(() => segment.locator("canvas").evaluate((element: HTMLCanvasElement) =>
    element.getContext("2d")?.getImageData(20, 20, 1, 1).data[3],
  )).toBe(255);
});

test("real F971 classical stamp keeps full frame and awaits manual review", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Tạo dữ liệu xước F971" }),
  ).toBeVisible();
  const background = page.getByRole("button", { name: /^Chọn / }).first();
  await expect(background).toBeVisible();
  await background.click();
  const stamp = page.getByRole("button", { name: /^Thêm (original|review)-/ }).first();
  await expect(stamp).toBeEnabled({ timeout: 60000 });
  await stamp.click();
  await page.getByLabel("Phương pháp ghép Stamp", { exact: true }).selectOption("classical_mixed");
  await page.getByRole("button", { name: "Tạo ảnh" }).click();
  await expect(
    page.getByRole("heading", { name: "Duyệt ảnh và nhãn" }),
  ).toBeVisible({ timeout: 120000 });
  await expect(
    page.getByRole("button", { name: "Xuất bộ training" }),
  ).toBeDisabled();
  await expect(page.locator(".review-panel .badge")).toContainText("CHỜ DUYỆT");
  await expect(page.getByTestId("review-canvas")).toHaveAttribute(
    "data-image-loaded",
    "true",
    { timeout: 60000 },
  );
  await page.screenshot({ path: "../outputs/ui-smoke.png", fullPage: true });
  expect(errors).toEqual([]);
});
