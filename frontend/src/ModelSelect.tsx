import { useEffect, useRef, useState } from "react";
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
const immediate = (id: string) => id.startsWith("classical_") || id === "procedural";

export function ModelSelect({ label, value, methods, disabled, onChange }: {
  label: string; value: string; methods: {value: string; label: string}[];
  disabled: boolean; onChange: (value: string) => void;
}) {
  const [models, setModels] = useState<ModelState[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const current = useRef(value);
  current.current = value;
  const [requesting, setRequesting] = useState<string | null>(null);
  const pending = useRef<string | null>(null);
  const change = useRef(onChange);
  change.current = onChange;
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
        setError("");
        const active = rows.find(row => row.model_id === current.current);
        if (!immediate(current.current) && active && active.status !== "ready" && !locked.current) {
          setNotice("Model đang chọn chưa sẵn sàng; đã chuyển về phương pháp CPU.");
          change.current(methods.some(method => method.value === "procedural") ? "procedural" : "classical_mixed");
        }
        const selected = rows.find(row => row.model_id === pending.current);
        if (selected?.status === "ready" && !locked.current) {
          pending.current = null;
          change.current(selected.model_id);
        }
        if (selected && ["cancelled", "error"].includes(selected.status)) pending.current = null;
      } catch (reason) {
        if (alive) setError(message(reason));
      } finally {
        if (alive) timer = setTimeout(poll, 750);
      }
    }
    void poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  async function choose(id: string) {
    if (disabled || requesting) return;
    pending.current = null;
    setNotice("");
    const row = models.find(model => model.model_id === id);
    if (immediate(id) || row?.status === "ready") { onChange(id); return; }
    pending.current = id;
    if (row?.status === "downloading" || row?.status === "verifying") return;
    setRequesting(id);
    try {
      await api("/api/models/download", "POST", {model_id: id});
    } catch (reason) {
      pending.current = null;
      setError(message(reason));
    } finally { setRequesting(null); }
  }
  async function cancel(id: string) {
    if (pending.current === id) pending.current = null;
    setRequesting(id);
    try {
      await api("/api/models/cancel", "POST", {model_id: id});
      setModels(rows => rows.map(row => row.model_id === id ? {...row, status: "cancelled"} : row));
    } catch (reason) { setError(message(reason)); }
    finally { setRequesting(null); }
  }
  return <div className="model-select">
    <label>{label}<select aria-label={label} value={value} disabled={disabled || !!requesting}
      onChange={event => { const id = event.target.value; event.target.value = value; void choose(id); }}>
      {methods.map(method => <option key={method.value} value={method.value}>{method.label}{immediate(method.value) ? "" : models.find(row => row.model_id === method.value)?.status === "ready" ? " · Đã tải" : " · Chưa tải"}</option>)}
    </select></label>
    <div className="model-status-list">
      {methods.filter(method => !immediate(method.value)).map(method => {
        const row = models.find(model => model.model_id === method.value);
        const loading = row?.status === "downloading" || row?.status === "verifying";
        const ready = row?.status === "ready";
        const name = names[method.value];
        const progress = Math.min(100, Math.max(0, row?.progress ?? 0));
        return <div className="model-status-row" key={method.value} data-testid={`model-${method.value}`}>
          <div><b>{name}</b><small>{ready ? "Đã tải" : row?.status === "verifying" ? "Đang kiểm tra file…" : loading ? `${progress.toFixed(0)}% · ${((row?.downloaded_bytes ?? 0)/1e6).toFixed(1)} MB${row?.speed_bps ? ` · ${(row.speed_bps/1e6).toFixed(1)} MB/s` : ""}` : `Chưa tải — ${row?.size_label ?? "Đang xác định dung lượng"}`}</small>
            {loading && <progress aria-label={`Tiến trình ${name}`} value={progress} max={100}/>}
            {row?.error && <small className="model-download-error">{row.error}</small>}
          </div>
          {!ready && <button type="button" aria-label={`${loading ? "Hủy" : "Tải"} ${name}`} disabled={!!requesting || (!loading && disabled)}
            onClick={() => void (loading ? cancel(method.value) : choose(method.value))}>{requesting === method.value ? "Đang xử lý…" : loading ? "Hủy" : "Tải"}</button>}
        </div>;
      })}
    </div>
    {notice && <p className="model-download-error">{notice}</p>}
    {error && <p role="alert" className="model-download-error">{error}</p>}
  </div>;
}
