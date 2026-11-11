import { brushMethods, stampMethods, type GenerationSettings } from "./generationMethods";
import type { BrushMethod, StampMethod } from "./api";
import "./GenerationPanel.css";
import { ModelSelect } from "./ModelSelect";

type Health = {
  ai?: { model_available?: boolean };
  brush_models?: { sdxl?: { model_available?: boolean }; sd15?: { model_available?: boolean } };
};
export function GenerationPanel({ settings, onChange, busy, canGenerate, jobStatus, hasBrushes, health, onGenerate }: {
  settings: GenerationSettings;
  onChange: (next: GenerationSettings) => void;
  busy: boolean;
  canGenerate: boolean;
  jobStatus?: string;
  hasBrushes: boolean;
  health?: Health;
  onGenerate: () => void;
}) {
  const brushAI = settings.brush_method !== "procedural";
  const available = settings.brush_method === "sdxl"
    ? health?.brush_models?.sdxl?.model_available ?? health?.ai?.model_available
    : health?.brush_models?.sd15?.model_available;
  return <section className="generation-panel">
    <div><span className="eyebrow">02 / TẠO ẢNH</span><h2>Hòa hợp với bề mặt</h2>
      <p>Stamp được ghép ở độ phân giải gốc, không đi qua VAE. Nét vẽ được xử lý riêng.</p></div>
    <div className="generation-methods">
      <ModelSelect label="Phương pháp ghép Stamp" value={settings.stamp_method} methods={stampMethods} disabled={busy}
        onChange={value => onChange({ ...settings, stamp_method: value as StampMethod })}/>
      <ModelSelect label="Phương pháp Vẽ nét" value={settings.brush_method} methods={brushMethods} disabled={busy}
        onChange={value => onChange({ ...settings, brush_method: value as BrushMethod })}/>
    </div>
    <div className="generation-options">
      <label>Seed nét vẽ<input aria-label="Seed" type="number" min="0" max="2147483647" value={settings.seed} disabled={busy}
        onChange={e => onChange({ ...settings, seed: Math.max(0, Math.min(2147483647, +e.target.value)) })}/></label>
      <label>Mức biến đổi nét AI <b>{settings.strength.toFixed(2)}</b><input aria-label="Mức biến đổi" type="range" min=".1" max=".99" step=".01" value={settings.strength} disabled={busy || !brushAI || !hasBrushes}
        onChange={e => onChange({ ...settings, strength: +e.target.value })}/></label>
      <label>Biên chỉnh sửa (px)<input aria-label="Biên chỉnh sửa" type="number" min="0" max="128" value={settings.edit_margin} disabled={busy}
        onChange={e => onChange({ ...settings, edit_margin: Math.max(0, Math.min(128, +e.target.value)) })}/></label>
    </div>
    <button className="primary generate" disabled={!canGenerate || busy} onClick={onGenerate}>{busy ? "Đang xử lý…" : "Tạo ảnh"}</button>
    <p className="model-note">{busy ? `Job ${jobStatus ?? "đang gửi"} · giữ nguyên scene trong lúc xử lý`
      : "Chỉ xử lý phương pháp của layer hiện có. Nếu thiếu checkpoint harmonization, hệ thống dùng Classical và báo rõ trong kết quả."}
      {hasBrushes && brushAI && available === false ? " Chưa thấy model vẽ nét đã chọn; cần tải model trước khi tạo nét bằng AI." : ""}</p>
  </section>;
}
