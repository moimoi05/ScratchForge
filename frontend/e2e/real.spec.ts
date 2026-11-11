import { test, expect } from "@playwright/test";

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
  const stamp = page.getByRole("button", { name: /^Thêm original-/ }).first();
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
