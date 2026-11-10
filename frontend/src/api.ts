export const API =
  window.location.port === "5173"
    ? "http://127.0.0.1:8000"
    : window.location.origin;
export const url = (path: string) =>
  path.startsWith("http") ? path : API + path;
export type Stamp = {
  id: string;
  rgb: string;
  mask: string;
  thumbnail?: string;
  width: number;
  height: number;
  review_status: string;
  source_image?: string;
};
export type Placement = {
  stamp_id: string;
  x: number;
  y: number;
  scale: number;
  rotation: number;
  flip: boolean;
};
export type Brush = { points: number[]; width: number };
export type Scene = {
  image_id?: string;
  placements: Placement[];
  brushes: Brush[];
  engine?: "baseline" | "sdxl";
  seed?: number;
  strength?: number;
  edit_margin?: number;
};
export type Background = {
  id: string;
  width: number;
  height: number;
  url: string;
  thumbnail?: string;
  source_image?: string;
  split?: string;
};
export type Result = {
  id: string;
  preview: string;
  edit_mask: string;
  width: number;
  height: number;
  polygons: number[][];
  review_status: string;
  scene: Scene;
};
export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch(API + path, {
    method,
    ...(body instanceof FormData
      ? { body }
      : body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
  });
  const payload = await response
    .json()
    .catch(() => ({ detail: `HTTP ${response.status}` }));
  if (!response.ok)
    throw Error(
      typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail ?? payload),
    );
  return payload;
}
export const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
