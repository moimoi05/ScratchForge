import { useEffect, useId, useRef, useState } from "react";
import { api, message } from "./api";

export type ModelState = {
  model_id: string;
  status: "not_downloaded" | "downloading" | "verifying" | "ready" | "cancelled" | "error";
  downloaded_bytes?: number;
  total_bytes?: number;
  progress?: number;
  speed_bps?: number;
  error?: string;
  size_label?: string;
};
const names: Record<string, string> = { pctnet: "PCT-Net", harmonizer: "Harmonizer", rainnet: "RainNet", sdxl: "SDXL", sd15: "SD 1.5" };
const immediate = (id: string) => id.startsWith("classical_") || id === "procedural" || id === "stamp_brush" || id === "stamp_brush_v2";
const optionStatus: Record<ModelState["status"], string> = {
  ready: "Đã tải", not_downloaded: "Chưa tải", downloading: "Đang tải",
  verifying: "Đang kiểm tra", cancelled: "Chưa tải", error: "Lỗi tải",
};

export function ModelSelect({ label, value, methods, disabled, onChange, onModels, onPending, placeholder }: {
  label: string; value: string; methods: {value: string; label: string}[];
  disabled: boolean; onChange: (value: string) => void;
  onModels?: (models: ModelState[]) => void;
  onPending?: (id: string | null) => void;
  placeholder?: string;
}) {
  const hintId = useId();
  const [models, setModels] = useState<ModelState[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pollError, setPollError] = useState("");
  const current = useRef(value);
  current.current = value;
  const [requesting, setRequesting] = useState<string | null>(null);
  const pending = useRef<string | null>(null);
  const change = useRef(onChange);
  change.current = onChange;
  const reportModels = useRef(onModels);
  reportModels.current = onModels;
  const reportPending = useRef(onPending);
  reportPending.current = onPending;
  const updatePending = (id: string | null) => { pending.current = id; reportPending.current?.(id); };
  const locked = useRef(disabled);
  locked.current = disabled;
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const result = await api<{models: ModelState[]}>("/api/models/status");
        if (!alive) return;
        const rows = Array.isArray(result.models) ? result.models : [];
        setModels(rows);
        reportModels.current?.(rows);
        setPollError("");
        const active = rows.find(row => row.model_id === current.current);
        if (!placeholder && !immediate(current.current) && active && active.status !== "ready" && !locked.current) {
          setNotice("Model đang chọn chưa sẵn sàng; đã chuyển về phương pháp CPU.");
          change.current(methods.some(method => method.value === "procedural") ? "procedural" : "classical_mixed");
        }
        const selected = rows.find(row => row.model_id === pending.current);
        if (selected?.status === "ready" && !locked.current) {
          updatePending(null);
          change.current(selected.model_id);
        }
        if (selected && ["cancelled", "error"].includes(selected.status)) updatePending(null);
      } catch (reason) {
        if (alive) setPollError(message(reason));
      } finally {
        if (alive) timer = setTimeout(poll, 750);
      }
    }
    void poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  async function choose(id: string) {
    if (disabled || requesting) return;
    updatePending(null);
    setNotice("");
    setError("");
    const row = models.find(model => model.model_id === id);
    if (immediate(id) || row?.status === "ready") { onChange(id); return; }
    updatePending(id);
    if (row?.status === "downloading" || row?.status === "verifying") return;
    setRequesting(id);
    try {
      await api("/api/models/download", "POST", {model_id: id});
    } catch (reason) {
      updatePending(null);
      setError(message(reason));
    } finally { setRequesting(null); }
  }
  async function cancel(id: string) {
    setRequesting(id);
    try {
      await api("/api/models/cancel", "POST", {model_id: id});
      if (pending.current === id) updatePending(null);
      setModels(rows => rows.map(row => row.model_id === id ? {...row, status: "cancelled"} : row));
    } catch (reason) { setError(message(reason)); }
    finally { setRequesting(null); }
  }
  const hint = requesting ? "Đợi phản hồi từ trình tải model." : disabled ? "Đợi thao tác hiện tại hoàn tất." : "";
  const downloadRows = models.filter(row => methods.some(method => method.value === row.model_id)
    && (row.status === "downloading" || row.status === "verifying" || row.status === "error"));
  return <div className="model-select">
    <label>{label}<select aria-label={label} value={value} disabled={!!hint} aria-describedby={hint ? hintId : undefined}
      data-tooltip="Chọn model chưa tải để bắt đầu tải tự động."
      onChange={event => { const id = event.target.value; event.target.value = value; void choose(id); }}>
      {placeholder && <option value="" disabled>{placeholder}</option>}
      {methods.map(method => {
        const status = models.find(row => row.model_id === method.value)?.status;
        return <option key={method.value} value={method.value}>
          {method.label}{immediate(method.value) ? "" : ` · ${status ? optionStatus[status] : "Chưa tải"}`}
        </option>;
      })}
    </select></label>
    {!!downloadRows.length && <div className="model-status-list">
      {downloadRows.map(row => {
        const loading = row.status === "downloading" || row.status === "verifying";
        const name = names[row.model_id];
        const progress = Math.min(100, Math.max(0, row.progress ?? 0));
        return <div className="model-status-row" key={row.model_id} data-testid={`model-${row.model_id}`}>
          <div><b>{name}</b><small>{row.status === "verifying" ? "Đang kiểm tra file…" : loading ? `${progress.toFixed(0)}% · ${((row.downloaded_bytes ?? 0)/1e6).toFixed(1)} MB${row.speed_bps ? ` · ${(row.speed_bps/1e6).toFixed(1)} MB/s` : ""}` : "Tải thất bại. Chọn lại model trong danh sách để thử lại."}</small>
            {loading && <progress aria-label={`Tiến trình ${name}`} value={progress} max={100}/>}
            {row?.error && <small className="model-download-error">{row.error}</small>}
          </div>
          {loading && <button type="button" aria-label={`Hủy ${name}`} disabled={!!requesting} aria-describedby={requesting ? hintId : undefined}
            data-tooltip={`Dừng tải ${name}.`} onClick={() => void cancel(row.model_id)}>Hủy</button>}
        </div>;
      })}
    </div>}
    {notice && <p className="model-download-error">{notice}</p>}
    {hint && <p className="action-hint" id={hintId}>{hint}</p>}
    {(error || pollError) && <p role="alert" className="model-download-error">{error || pollError}</p>}
  </div>;
}
