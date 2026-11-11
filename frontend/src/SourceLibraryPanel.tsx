import { useEffect, useState } from "react";
import { api, message, url, type LibraryJob, type LoraCheckpoint, type Stamp } from "./api";
import { StampReview } from "./StampReview";
import { ModelSelect, type ModelState } from "./ModelSelect";
import "./SourceLibraryPanel.css";

const terminal = (status: string) => ["succeeded", "completed", "failed", "cancelled"].includes(status);
const synthetic = (stamp: Stamp) => ["ai", "generated", "synthetic"].includes(stamp.origin ?? "");
const native = (stamp: Stamp) => /\.tiff?$/i.test(stamp.rgb);
const baseMethods = [{ value: "sd15", label: "SD 1.5 Inpainting · 512" }, { value: "sdxl", label: "SDXL Inpainting · 1024" }];
const statusNames: Record<string, string> = { queued: "Đang chờ", running: "Đang xử lý", succeeded: "Hoàn tất", completed: "Hoàn tất", failed: "Thất bại", cancelled: "Đã hủy" };

export function SourceLibraryPanel({ stamps, checkpoints, onLibraryChange, onTrainingChange, onError }: {
  stamps: Stamp[]; checkpoints: LoraCheckpoint[]; onLibraryChange: () => Promise<void>;
  onTrainingChange: () => Promise<void>; onError: (error: string) => void;
}) {
  const [mode, setMode] = useState<"fullframe" | "crop">("fullframe");
  const [image, setImage] = useState<File>();
  const [labels, setLabels] = useState<File>();
  const [sourceId, setSourceId] = useState("");
  const [declared, setDeclared] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [copies, setCopies] = useState(1);
  const [steps, setSteps] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [base, setBase] = useState<"sd15" | "sdxl">("sd15");
  const [models, setModels] = useState<ModelState[]>([]);
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [lora, setLora] = useState("");
  const [count, setCount] = useState(4);
  const [threshold, setThreshold] = useState(8);
  const [minArea, setMinArea] = useState(3);
  const [job, setJob] = useState<LibraryJob>();
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState("");
  const [review, setReview] = useState<Stamp>();
  const busy = submitting || !!job && !terminal(job.status);
  const compatible = checkpoints.filter(item => item.base_model === base);
  const selectedStamps = stamps.filter(stamp => selected.includes(stamp.id));
  const waiting = busy ? "Đợi yêu cầu hiện tại hoàn tất." : "";
  const baseReady = models.some(model => model.model_id === base && model.status === "ready");
  const baseHint = pendingModel ? "Đợi model tải xong và được kiểm tra trước khi chạy B/C."
    : !baseReady && "Chọn base model ở trên để hệ thống tải và kiểm tra model.";
  const selectHint = !selectedStamps.length && "Tích ít nhất một stamp trong mục Chọn stamp ở trên.";
  const tiffHint = selectedStamps.some(stamp => !native(stamp)) && "Bỏ chọn các mẫu PNG cũ. Thao tác này cần stamp TIFF.";
  const reviewHint = selectedStamps.some(stamp => stamp.review_status.toLowerCase() !== "approved") &&
    "Bấm Xem ở mẫu chờ duyệt, tích ‘Tôi đã kiểm tra stamp và polygon’, rồi bấm Duyệt stamp.";
  const syntheticHint = selectedStamps.some(synthetic) && "Bỏ chọn stamp do AI sinh và chọn stamp nguồn hoặc augment đã duyệt.";
  const importHint = waiting || [!image && "Chọn ảnh TIFF.",
    !labels && (mode === "fullframe" ? "Chọn file Polygon COCO JSON." : "Chọn file mask PNG."),
    !sourceId.trim() && "Nhập ID nguồn train.", !declared && "Tích ‘Tôi xác nhận nguồn thuộc tập train’."].filter(Boolean).join(" ");
  const augmentHint = waiting || [selectHint, tiffHint,
    selectedStamps.length > 100 && "Chọn tối đa 100 stamp mỗi lượt.",
    selectedStamps.length * copies > 500 && "Giảm số mẫu hoặc số biến thể để tổng không vượt 500."].filter(Boolean).join(" ");
  const trainHint = waiting || [baseHint, selectHint, tiffHint, reviewHint, syntheticHint].filter(Boolean).join(" ");
  const generateHint = waiting || [baseHint, selectHint, tiffHint, reviewHint, syntheticHint,
    selectedStamps.length > 100 && "Chọn tối đa 100 stamp mỗi lượt.",
    !compatible.some(item => item.id === lora) && (compatible.length
      ? "Chọn một LoRA tương thích ở ô LoRA đã train."
      : "Train LoRA cho base model này ở mục B, hoặc đổi sang base model đã có LoRA.")].filter(Boolean).join(" ");
  useEffect(() => { setSelected(ids => ids.filter(id => stamps.some(stamp => stamp.id === id))); }, [stamps]);
  useEffect(() => {
    if (!job || terminal(job.status)) return;
    let live = true;
    const timer = setTimeout(() => {
      api<LibraryJob>(`/api/jobs/${job.id}`).then(async next => {
        if (!live) return;
        setJob(next);
        if (next.status === "failed") onError(next.error ?? "Không thể hoàn thành job kho stamp.");
        if (["succeeded", "completed"].includes(next.status)) {
          await Promise.all([onLibraryChange(), onTrainingChange()]);
          if (next.operation_result?.rejected?.length) setNotice(`${next.operation_result.rejected.length} mẫu không đạt; xem lại tham số tách mask.`);
        }
      }).catch(error => { if (live) { onError(message(error)); setJob({ ...job, status: "failed", error: message(error) }); } });
    }, 600);
    return () => { live = false; clearTimeout(timer); };
  }, [job, onLibraryChange, onTrainingChange, onError]);

  async function importPair() {
    if (!image || !labels || !sourceId.trim() || !declared || busy) return;
    setSubmitting(true); setNotice("");
    try {
      if (!/\.tiff?$/i.test(image.name)) throw Error("Chọn TIFF RGB 24-bit (.tif/.tiff).");
      if (mode === "fullframe" && !/\.json$/i.test(labels.name)) throw Error("Chọn Polygon JSON chuẩn COCO.");
      if (mode === "crop" && !/\.png$/i.test(labels.name)) throw Error("Chọn mask PNG nhị phân trắng/đen.");
      const data = new FormData();
      data.append("image", image); data.append("mode", mode); data.append("source_id", sourceId.trim());
      data.append("confirmed_train", "true"); data.append(mode === "fullframe" ? "annotations" : "mask", labels);
      await api("/api/library/import", "POST", data);
      await onLibraryChange();
      setNotice("Import thành công · kiểm tra và duyệt nhãn stamp trước khi train.");
    } catch (error) { onError(message(error)); }
    finally { setSubmitting(false); }
  }
  async function start(path: string, body: unknown) {
    if (busy) return;
    setSubmitting(true); setNotice("");
    try { setJob(await api<LibraryJob>(path, "POST", body)); }
    catch (error) { onError(message(error)); }
    finally { setSubmitting(false); }
  }
  async function cancel() {
    if (!job || terminal(job.status)) return;
    setSubmitting(true);
    try { setJob(await api<LibraryJob>(`/api/jobs/${job.id}/cancel`, "POST")); }
    catch (error) { onError(message(error)); }
    finally { setSubmitting(false); }
  }
  return <aside className="source-library-panel" data-testid="source-library-panel" aria-label="Mở rộng và huấn luyện Kho Xước">
    <div><span className="eyebrow">KHO XƯỚC TIFF</span><h2>Mở rộng & huấn luyện</h2><p>RGB gốc, mask và COCO luôn đi cùng nhau.</p></div>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>Import dữ liệu mới</legend>
      <label>Kiểu import<select aria-label="Kiểu import stamp" value={mode} onChange={e => { setMode(e.target.value as typeof mode); setLabels(undefined); }}>
        <option value="fullframe">TIFF nguyên khung + COCO Polygon JSON</option><option value="crop">Crop TIFF + Mask PNG</option>
      </select></label>
      <label>Ảnh TIFF<input aria-label="Ảnh TIFF stamp" type="file" accept=".tif,.tiff,image/tiff" onChange={e => setImage(e.target.files?.[0])}/></label>
      <label>{mode === "fullframe" ? "Polygon COCO JSON" : "Mask nhị phân PNG"}<input key={mode} aria-label={mode === "fullframe" ? "Polygon stamp JSON" : "Mask stamp PNG"}
        type="file" accept={mode === "fullframe" ? ".json,application/json" : ".png,image/png"} onChange={e => setLabels(e.target.files?.[0])}/></label>
      <label>ID nguồn train<input aria-label="ID nguồn train" maxLength={150} value={sourceId} onChange={e => setSourceId(e.target.value)}/></label>
      <label className="confirmation"><input type="checkbox" checked={declared} onChange={e => setDeclared(e.target.checked)}/>Tôi xác nhận nguồn thuộc tập train</label>
      <button disabled={!!importHint} aria-describedby={importHint ? "import-stamp-hint" : undefined} data-tooltip="Thêm TIFF và nhãn vào kho để kiểm tra và duyệt." onClick={() => void importPair()}>Import vào kho</button>
      {importHint && <p className="action-hint" id="import-stamp-hint">{importHint}</p>}
    </fieldset>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>Chọn stamp · {selected.length}/{stamps.length}</legend>
      <div className="button-row"><button onClick={() => setSelected(stamps.map(s => s.id))}>Chọn tất cả</button><button onClick={() => setSelected([])}>Bỏ chọn</button></div>
      <div className="source-stamp-list">{stamps.map(stamp => <div className="source-stamp-row" key={stamp.id}>
        <input type="checkbox" aria-label={`Chọn ${stamp.id}`} checked={selected.includes(stamp.id)}
          onChange={e => setSelected(ids => e.target.checked ? [...ids, stamp.id] : ids.filter(id => id !== stamp.id))}/>
        <img loading="lazy" src={url(`/library-files/${stamp.thumbnail ?? stamp.preview_alpha ?? stamp.rgb}`)} alt=""/>
        <span><strong>{stamp.id}</strong><small>{stamp.width} × {stamp.height} · {stamp.review_status.toLowerCase() === "approved" ? "Đã duyệt" : "Chờ duyệt"}</small></span>
        <button aria-label={`Duyệt ${stamp.id}`} data-tooltip={waiting || "Mở stamp để xem, sửa hoặc duyệt polygon."} onClick={() => setReview(stamp)}>Xem</button>
      </div>)}{!stamps.length && <p>Chưa có stamp. Import TIFF và nhãn để bắt đầu.</p>}</div>
      <label>Seed kho stamp<input aria-label="Seed kho stamp" data-tooltip={waiting || "Dùng cùng seed và thông số để lặp lại lựa chọn ngẫu nhiên."} type="number" min="0" max="4294967295" value={seed}
        onChange={e => setSeed(Math.max(0, Math.min(4294967295, +e.target.value)))}/></label>
    </fieldset>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>A · Biến dạng ngẫu nhiên</legend>
      <p>Elastic, spline, xoay và co giãn đồng bộ TIFF/mask/polygon.</p>
      <label>Biến thể mỗi mẫu<input aria-label="Biến thể mỗi mẫu" type="number" min="1" max="20" value={copies} onChange={e => setCopies(Math.max(1, Math.min(20, +e.target.value)))}/></label>
      <button disabled={!!augmentHint} aria-describedby={augmentHint ? "augment-stamp-hint" : undefined} data-tooltip="Tạo biến thể từ các stamp đã tích, giữ TIFF, mask và polygon đồng bộ." onClick={() => void start("/api/library/augment", { stamp_ids: selected, copies, seed })}>A · Augment Stamp</button>
      {augmentHint && <p className="action-hint" id="augment-stamp-hint">{augmentHint}</p>}
    </fieldset>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>Base model cho B / C</legend>
      <ModelSelect label="Base model kho stamp" value={baseReady ? base : ""} methods={baseMethods}
        placeholder="Chọn base model" disabled={busy} onModels={setModels} onPending={setPendingModel}
        onChange={value => { setBase(value as typeof base); if (value !== base) setLora(""); }}/>
    </fieldset>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>B · Train LoRA F971</legend>
      <p>Chỉ stamp thực/import và augment đã duyệt. Stamp AI không dùng train lại.</p>
      <label>Số steps<input aria-label="Số steps LoRA" type="number" min="1" max="100000" value={steps} onChange={e => setSteps(Math.max(1, Math.min(100000, +e.target.value)))}/></label>
      <button disabled={!!trainHint} aria-describedby={trainHint ? "train-lora-hint" : undefined} data-tooltip="Huấn luyện LoRA F971 từ các stamp đã tích và lưu weights trên máy." onClick={() => void start("/api/training/lora", { base_model: base, stamp_ids: selected, steps, seed })}>B · Train LoRA F971</button>
      {trainHint && <p className="action-hint" id="train-lora-hint">{trainHint}</p>}
    </fieldset>
    <fieldset disabled={busy} data-tooltip={waiting || undefined}>
      <legend>C · Inference sinh stamp</legend>
      <label>LoRA đã train<select aria-label="LoRA sinh stamp" value={lora} onChange={e => setLora(e.target.value)}>
        <option value="">Chọn LoRA tương thích</option>{compatible.map(item => <option key={item.id} value={item.id}>{item.name ?? item.id}</option>)}
      </select></label>
      <label>Số stamp mới<input aria-label="Số stamp mới" type="number" min="1" max="100" value={count} onChange={e => setCount(Math.max(1, Math.min(100, +e.target.value)))}/></label>
      <div className="source-compact-fields"><label>Ngưỡng RGB<input aria-label="Ngưỡng tách mask" data-tooltip={waiting || "Chênh lệch màu tối thiểu với nền để tính một pixel vào mask."} type="number" min="1" max="255" value={threshold} onChange={e => setThreshold(Math.max(1, Math.min(255, +e.target.value)))}/></label>
        <label>Diện tích tối thiểu<input aria-label="Diện tích mask tối thiểu" data-tooltip={waiting || "Loại các vùng mask nhỏ hơn số pixel này."} type="number" min="1" max="10000" value={minArea} onChange={e => setMinArea(Math.max(1, Math.min(10000, +e.target.value)))}/></label></div>
      <button disabled={!!generateHint} aria-describedby={generateHint ? "generate-stamp-hint" : undefined} data-tooltip="Dùng LoRA đã chọn để sinh stamp mới và đưa vào kho chờ duyệt." onClick={() => void start("/api/library/generate", { base_model: base, lora_id: lora, stamp_ids: selected, count, seed, threshold, min_area: minArea })}>C · Sinh stamp mới</button>
      {generateHint && <p className="action-hint" id="generate-stamp-hint">{generateHint}</p>}
      <small>Mask lấy từ chênh lệch với nền giả lập; kiểm tra polygon trước khi dùng.</small>
    </fieldset>
    {job && <div className="source-job" role="status" aria-live="polite">
      <strong>{statusNames[job.status] ?? job.status}</strong><span>{Math.round(job.progress ?? 0)}%</span>
      <progress aria-label="Tiến trình kho stamp" max={100} value={job.progress ?? 0}/>
      {job.error && <p>{job.error}</p>}
      {!terminal(job.status) && <button disabled={submitting} data-tooltip={submitting ? "Đợi máy chủ phản hồi." : "Yêu cầu dừng job kho stamp đang chạy."} onClick={() => void cancel()} aria-label="Hủy job kho stamp">Hủy</button>}
    </div>}
    {submitting && <p role="status">Đang xử lý yêu cầu…</p>}
    {notice && <p role="status">{notice}</p>}
    <div className="lora-registry"><h3>LoRA local · {checkpoints.length}</h3>{checkpoints.map(item => <p key={item.id}>{item.name ?? item.id}<small>{item.base_model} · {item.revision?.slice(0, 12) ?? "Local"}</small></p>)}</div>
    {review && <StampReview key={review.id} stamp={review} onClose={() => setReview(undefined)} onSaved={onLibraryChange} onError={onError}/>}
  </aside>;
}
