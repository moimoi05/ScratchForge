import type { BrushMethod, Result, Scene, StampMethod } from "./api";

export const stampMethods: { value: StampMethod; label: string }[] = [
  { value: "classical_mixed", label: "Classical · Reinhard + Mixed SeamlessClone" },
  { value: "classical_laplacian", label: "Classical · Reinhard + Laplacian Pyramid" },
  { value: "pctnet", label: "AI · PCT-Net (Full-res Harmonization)" },
  { value: "harmonizer", label: "AI · Harmonizer (Filter-based)" },
  { value: "rainnet", label: "AI · RainNet (Region-aware AdaIN)" },
];
export const brushMethods: { value: BrushMethod; label: string }[] = [
  { value: "sdxl", label: "AI · SDXL Inpainting" },
  { value: "sd15", label: "AI · SD 1.5 Inpainting (nhẹ / CPU)" },
  { value: "procedural", label: "Procedural · Sinh xước giải thuật (CPU)" },
];
export type GenerationSettings = {
  stamp_method: StampMethod;
  brush_method: BrushMethod;
  seed: number;
  strength: number;
  edit_margin: number;
};
export const settingsFromScene = (scene?: Scene): GenerationSettings => ({
  stamp_method: scene?.stamp_method ?? "classical_mixed",
  brush_method: scene?.brush_method ?? (scene?.engine === "sdxl" ? "sdxl" : "procedural"),
  seed: scene?.seed ?? 42,
  strength: scene?.strength ?? 0.65,
  edit_margin: scene?.edit_margin ?? 24,
});
export function resultMethodSummary(result: Result): string {
  const layers = result.provenance?.layers;
  if (layers?.length) {
    return [...new Set(layers.map(layer => {
      const methods = layer.kind === "stamp" ? stampMethods : brushMethods;
      const label = methods.find(item => item.value === layer.effective_method)?.label ?? layer.effective_method;
      return `${layer.kind === "stamp" ? "Stamp" : "Nét"}: ${label}`;
    }))].join(" · ");
  }
  const settings = settingsFromScene(result.scene);
  const chosen = [
    ...(result.scene.placements?.length ? [stampMethods.find(item => item.value === settings.stamp_method)?.label] : []),
    ...(result.scene.brushes?.length ? [brushMethods.find(item => item.value === settings.brush_method)?.label] : []),
  ].filter(Boolean);
  if (result.warnings?.length) return "Có phương pháp thay thế — xem thông báo";
  return chosen.join(" · ") || (result.scene.engine === "sdxl" ? "SDXL (scene cũ)" : "Classical / Procedural");
}
