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
  preview_alpha?: string;
  polygon?: string;
  segmentation?: number[][];
  origin?: string;
  split?: string;
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
export type Brush = { points: number[]; width: number; stamp_id?: string };
export type StampMethod = "classical_mixed" | "classical_laplacian" | "pctnet" | "harmonizer" | "rainnet";
export type BrushMethod = "sdxl" | "sd15" | "procedural" | "stamp_brush" | "stamp_brush_v2";
export type LoraCheckpoint = { id: string; base_model: "sd15" | "sdxl"; name?: string; revision?: string; weights?: string };
export type ModelHealth = { model_available?: boolean; loaded_lora_id?: string | null; lora_loaded?: boolean };
export type Health = { ai?: ModelHealth; brush_models?: { sdxl?: ModelHealth; sd15?: ModelHealth }; engine?: string };
export type LibraryJob = { id: string; status: string; progress?: number; error?: string;
  operation_result?: { stamps?: Stamp[]; rejected?: unknown[]; lora_id?: string; checkpoint?: LoraCheckpoint } };
export type Scene = {
  image_id?: string;
  placements: Placement[];
  brushes: Brush[];
  engine?: "baseline" | "sdxl";
  stamp_method?: StampMethod;
  brush_method?: BrushMethod;
  seed?: number;
  strength?: number;
  edit_margin?: number;
  brush_stamp_mode?: "selected" | "random";
  brush_stamp_id?: string;
  lora_id?: string | null;
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
  input_preview?: string;
  edit_mask: string;
  native_image?: string;
  label_mask?: string;
  polygon_file?: string;
  width: number;
  height: number;
  polygons: number[][];
  review_status: string;
  scene: Scene;
  warnings?: string[];
  provenance?: {
    stamps?: Stamp[];
    layers?: { kind: "stamp" | "brush"; requested_method: string; effective_method: string }[];
  };
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
