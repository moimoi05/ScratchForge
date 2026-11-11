import { brushMethods, stampMethods, type GenerationSettings } from "./generationMethods";
import type { BrushMethod, StampMethod, Stamp, LoraCheckpoint, Health } from "./api";
import "./GenerationPanel.css";
import { ModelSelect } from "./ModelSelect";

export function GenerationPanel({ settings, onChange, busy, hasImage, canGenerate, jobStatus, hasBrushes, health, stamps = [], checkpoints = [], currentStampId, onGenerate }: {
  settings: GenerationSettings;
  onChange: (next: GenerationSettings) => void;
  busy: boolean;
  canGenerate: boolean;
  hasImage: boolean;
  jobStatus?: string;
  hasBrushes: boolean;
  health?: Health;
  stamps?: Stamp[];
  checkpoints?: LoraCheckpoint[];
  currentStampId?: string;
  onGenerate: () => void;
}) {
  const brushAI = settings.brush_method === "sd15" || settings.brush_method === "sdxl";
  const compatible = checkpoints.filter(item => item.base_model === settings.brush_method);
  const missingLora = !!settings.lora_id && !compatible.some(item => item.id === settings.lora_id);
  const model = settings.brush_method === "sd15" ? health?.brush_models?.sd15 : health?.brush_models?.sdxl ?? health?.ai;
  const loaded = !!settings.lora_id && model?.loaded_lora_id === settings.lora_id && model?.lora_loaded === true;
  const available = settings.brush_method === "sdxl"
    ? health?.brush_models?.sdxl?.model_available ?? health?.ai?.model_available
    : health?.brush_models?.sd15?.model_available;
  const generateHint = busy ? "Đợi thao tác hiện tại hoàn tất."
    : !hasImage ? "Mở ảnh nền TIFF trước."
    : !canGenerate ? "Thêm một stamp hoặc vẽ ít nhất một nét xước trên ảnh."
    : hasBrushes && brushAI && missingLora ? "Chọn LoRA tương thích hoặc chọn ‘Không dùng LoRA’." : "";
  const strengthHint = busy ? "Đợi thao tác hiện tại hoàn tất."
    : !brushAI ? "Chọn SDXL hoặc SD 1.5 ở Phương pháp Vẽ nét."
    : !hasBrushes ? "Vẽ ít nhất một nét xước để chỉnh mức biến đổi AI." : "";
  const waitHint = busy ? "Đợi thao tác hiện tại hoàn tất." : "";
  return <section className="generation-panel">
    <div><span className="eyebrow">02 / TẠO ẢNH</span><h2>Hòa hợp với bề mặt</h2>
      <p>Stamp được ghép ở độ phân giải gốc, không đi qua VAE. Nét vẽ được xử lý riêng.</p></div>
    <div className="generation-methods">
      <ModelSelect label="Phương pháp ghép Stamp" value={settings.stamp_method} methods={stampMethods} disabled={busy}
        onChange={value => onChange({ ...settings, stamp_method: value as StampMethod })}/>
      <ModelSelect label="Phương pháp Vẽ nét" value={settings.brush_method} methods={brushMethods} disabled={busy}
        onChange={value => onChange({ ...settings, brush_method: value as BrushMethod,
          lora_id: value === settings.brush_method ? settings.lora_id : null })}/>
    </div>
    {settings.brush_method.startsWith("stamp_brush") && <div className="brush-source-controls">
      <label>Nguồn Stamp Brush<select aria-label="Nguồn Stamp Brush" disabled={busy} value={settings.brush_stamp_mode}
        data-tooltip={waitHint || "Dùng một mẫu cụ thể hoặc chọn ngẫu nhiên theo seed cho từng nét."}
        onChange={e => onChange({ ...settings, brush_stamp_mode: e.target.value as "selected" | "random" })}>
        <option value="random">Ngẫu nhiên theo seed · mỗi nét một mẫu</option><option value="selected">Theo mẫu đang chọn</option>
      </select></label>
      {settings.brush_stamp_mode === "selected" && <label>Mẫu xước để uốn<select aria-label="Mẫu xước để uốn" disabled={busy || !stamps.length}
        data-tooltip={waitHint || (!stamps.length ? "Thêm stamp TIFF ở tab Dữ liệu nguồn trước." : "Chọn chất liệu xước sẽ được uốn theo nét vẽ.")}
        value={settings.brush_stamp_id ?? currentStampId ?? stamps[0]?.id ?? ""}
        onChange={e => onChange({ ...settings, brush_stamp_id: e.target.value })}>
        {!stamps.length && <option value="">Chưa có stamp</option>}
        {stamps.map(stamp => <option key={stamp.id} value={stamp.id}>{stamp.id}</option>)}
      </select></label>}
      <small>{settings.brush_method === "stamp_brush_v2"
        ? "Đầu và cuối mỗi nét thuôn khác nhau, phối nhẹ sắc độ từ TIFF nguồn. Ô 1 áp dụng cho stamp đặt trực tiếp."
        : "Uốn trực tiếp TIFF rồi hòa hợp theo Ô 1. Phép uốn nội suy pixel, không qua VAE."}</small>
    </div>}
    {brushAI && <div className="brush-source-controls">
      <label>LoRA F971<select aria-label="LoRA F971 cho nét AI" disabled={busy} value={settings.lora_id ?? ""}
        data-tooltip={waitHint || "Chọn LoRA được huấn luyện với cùng base model để áp dụng cho nét AI."}
        onChange={e => onChange({ ...settings, lora_id: e.target.value || null })}>
        <option value="">Không dùng LoRA</option>
        {missingLora && <option value={settings.lora_id!} disabled>LoRA đã lưu không còn khả dụng · {settings.lora_id}</option>}
        {compatible.map(item => <option key={item.id} value={item.id}>{item.name ?? item.id}</option>)}
      </select></label>
      <span role="status">{missingLora ? "Chọn LoRA tương thích hoặc bỏ dùng LoRA để tiếp tục" : loaded ? "Đã nạp LoRA F971" : settings.lora_id ? "Đã chọn LoRA · sẽ nạp khi chạy" : "Chưa dùng LoRA F971"}</span>
    </div>}
    <div className="generation-options">
      <label>Seed nét vẽ<input aria-label="Seed" type="number" min="0" max="2147483647" value={settings.seed} disabled={busy}
        data-tooltip={waitHint || "Giữ cùng seed và thông số để lặp lại kết quả sinh nét."}
        onChange={e => onChange({ ...settings, seed: Math.max(0, Math.min(2147483647, +e.target.value)) })}/></label>
      <label>Mức biến đổi nét AI <b>{settings.strength.toFixed(2)}</b><input aria-label="Mức biến đổi" type="range" min=".1" max=".99" step=".01" value={settings.strength} disabled={!!strengthHint} aria-describedby={strengthHint ? "ai-strength-hint" : undefined}
        data-tooltip="Giá trị càng cao thì AI càng thay đổi vùng nét vẽ."
        onChange={e => onChange({ ...settings, strength: +e.target.value })}/>{strengthHint && <small className="action-hint" id="ai-strength-hint">{strengthHint}</small>}</label>
      <label>Biên chỉnh sửa (px)<input aria-label="Biên chỉnh sửa" type="number" min="0" max="128" value={settings.edit_margin} disabled={busy}
        data-tooltip={waitHint || "Số pixel nền xung quanh để thuật toán hòa hợp lấy ngữ cảnh."}
        onChange={e => onChange({ ...settings, edit_margin: Math.max(0, Math.min(128, +e.target.value)) })}/></label>
    </div>
    <button className="primary generate" disabled={!!generateHint} aria-describedby={generateHint ? "generate-image-hint" : undefined} onClick={onGenerate}>{busy ? "Đang xử lý…" : "Tạo ảnh"}</button>
    {generateHint && <p className="action-hint" id="generate-image-hint">{generateHint}</p>}
    <p className="model-note">{busy ? `Job ${jobStatus ?? "đang gửi"} · giữ nguyên scene trong lúc xử lý`
      : "Chỉ xử lý phương pháp của layer hiện có. Nếu thiếu checkpoint harmonization, hệ thống dùng Classical và báo rõ trong kết quả."}
      {hasBrushes && brushAI && available === false ? " Chưa thấy model vẽ nét đã chọn; cần tải model trước khi tạo nét bằng AI." : ""}</p>
  </section>;
}
