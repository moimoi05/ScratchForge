import { useId, useState } from "react";
import { api, message, type Stamp } from "./api";
import { PolygonCanvas, type PolygonTool } from "./PolygonCanvas";

export function StampReview({ stamp, onClose, onSaved, onError }: {
  stamp: Stamp; onClose: () => void; onSaved: () => Promise<void>; onError: (error: string) => void;
}) {
  const [current, setCurrent] = useState(stamp);
  const [polygons, setPolygons] = useState(stamp.segmentation ?? []);
  const [draft, setDraft] = useState<number[]>([]);
  const [tool, setTool] = useState<PolygonTool>("move");
  const [selected, setSelected] = useState("0");
  const [dirty, setDirty] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const hintId = useId();
  const waitHint = busy ? "Đợi thao tác hiện tại hoàn tất." : "";
  const drawHint = tool === "draw" ? "Khép polygon hoặc bấm Hủy polygon stamp." : "";
  const saveHint = waitHint || drawHint || (!dirty ? "Chỉnh polygon trước khi lưu." : "");
  const confirmHint = waitHint || (!loaded ? "Đợi ảnh stamp tải xong." : !polygons.length
    ? "Vẽ ít nhất một polygon stamp trước khi xác nhận." : dirty ? "Bấm Lưu nhãn stamp trước khi xác nhận." : "");
  const approveHint = waitHint || drawHint || confirmHint || (!confirmed ? "Tích ô “Tôi đã kiểm tra stamp và polygon”." : "");
  function edit(next: number[][]) { setPolygons(next); setDirty(true); setConfirmed(false); }
  async function save() {
    setBusy(true); setError("");
    try {
      const next = await api<Stamp>(`/api/library/${current.id}/annotations`, "PUT", { polygons });
      setCurrent(next);
      setPolygons(next.segmentation ?? polygons);
      setDirty(false);
      setConfirmed(false);
      await onSaved();
    } catch (error) { setError(message(error)); onError(message(error)); }
    finally { setBusy(false); }
  }
  async function approve() {
    setBusy(true); setError("");
    try {
      const next = await api<Stamp>(`/api/library/${current.id}/approve`, "POST", { confirmed: true });
      setCurrent(next);
      await onSaved();
    } catch (error) { setError(message(error)); onError(message(error)); }
    finally { setBusy(false); }
  }
  return <div className="stamp-review-overlay"><section role="dialog" aria-modal="true" aria-label="Duyệt stamp" className="stamp-review-dialog"
    aria-describedby={busy ? `${hintId}-save` : undefined}>
    <div className="button-row"><h2>Duyệt stamp · {current.id}</h2><button disabled={busy} data-tooltip={waitHint || undefined} onClick={onClose}>Đóng duyệt stamp</button></div>
    {error && <p role="alert" className="model-download-error">{error}</p>}
    <p>{current.width} × {current.height} · {dirty ? "NHÃN CHƯA LƯU" : current.review_status.toUpperCase()}</p>
    <PolygonCanvas key={current.id} imageUrl={`/library-files/${current.preview_alpha ?? current.thumbnail ?? current.rgb}`}
      imageWidth={current.width} imageHeight={current.height}
      rings={polygons.map((points, i) => ({ id: String(i), points }))} selectedRing={selected}
      tool={tool} draft={draft} disabled={busy} testId="stamp-review-canvas" zoomLabel="Zoom duyệt stamp"
      onImageLoaded={setLoaded} onSelect={setSelected} onDraft={setDraft}
      onClose={points => { edit([...polygons, points]); setDraft([]); setTool("move"); }}
      onEdit={(id, points) => edit(polygons.map((p, i) => String(i) === id ? points : p))}
      toolbar={<><button disabled={busy} data-tooltip={waitHint || "Chọn polygon rồi kéo đỉnh cần chỉnh."} onClick={() => setTool("move")}>Di chuyển đỉnh</button>
        <button disabled={busy} data-tooltip={waitHint || "Bấm các đỉnh, rồi bấm lại đỉnh đầu để khép polygon."} onClick={() => { setTool("draw"); setDraft([]); }}>Vẽ polygon stamp</button>
        <button disabled={busy} data-tooltip={waitHint || "Chọn polygon rồi bấm gần cạnh để thêm đỉnh."} onClick={() => setTool("insert")}>Thêm đỉnh</button>
        <button disabled={busy} data-tooltip={waitHint || "Bấm đỉnh cần xóa. Giữ ít nhất ba đỉnh trong polygon."} onClick={() => setTool("delete")}>Xóa đỉnh</button></>}/>
    {tool === "draw" && <button onClick={() => { setDraft([]); setTool("move"); }}>Hủy polygon stamp</button>}
    <div className="stamp-polygon-list">{polygons.map((p, i) => <div key={i}>
      <button aria-pressed={selected === String(i)} onClick={() => setSelected(String(i))}>Polygon {i + 1} · {p.length / 2} đỉnh</button>
      <button disabled={busy} aria-label={`Xóa polygon stamp ${i + 1}`} data-tooltip={waitHint || `Xóa polygon ${i + 1}. Bấm Lưu nhãn stamp để áp dụng.`} onClick={() => edit(polygons.filter((_, j) => i !== j))}>×</button>
    </div>)}</div>
    <div className="button-row"><button disabled={!!saveHint} aria-describedby={saveHint ? `${hintId}-save` : undefined}
        onClick={() => void save()}>Lưu nhãn stamp</button>
      <label className="confirmation"><input type="checkbox" checked={confirmed} disabled={dirty || busy || !loaded || !polygons.length}
        data-tooltip={confirmHint || undefined}
        aria-describedby={confirmHint ? `${hintId}-${confirmHint === saveHint ? "save" : "approve"}` : undefined}
        onChange={e => setConfirmed(e.target.checked)}/>Tôi đã kiểm tra stamp và polygon</label>
      <button className="primary" disabled={!!approveHint}
        aria-describedby={approveHint ? `${hintId}-${approveHint === saveHint ? "save" : "approve"}` : undefined}
        onClick={() => void approve()}>Duyệt stamp</button></div>
    {saveHint && <p className="action-hint" id={`${hintId}-save`}>{saveHint}</p>}
    {approveHint && approveHint !== saveHint && <p className="action-hint" id={`${hintId}-approve`}>{approveHint}</p>}
  </section></div>;
}
