import { useEffect, useId, useState } from "react";
import { api, message, url, type Result } from "./api";
import { PolygonCanvas } from "./PolygonCanvas";
import { resultMethodSummary } from "./generationMethods";

export function Review({
  result,
  onUpdate,
  onError,
  onBack,
}: {
  result: Result;
  onUpdate: (r: Result) => void;
  onError: (s: string) => void;
  onBack: () => void;
}) {
  const [polygons, setPolygons] = useState(result.polygons),
    [draft, setDraft] = useState<number[]>([]),
    [drawing, setDrawing] = useState(false);
  const [dirty, setDirty] = useState(false),
    [confirmed, setConfirmed] = useState(false),
    [busy, setBusy] = useState(false),
    [download, setDownload] = useState("");
  const [showBefore, setShowBefore] = useState(false),
    [showLabels, setShowLabels] = useState(true);
  useEffect(() => {
    setPolygons(result.polygons);
    setDirty(false);
    setConfirmed(false);
    setDownload("");
    setDraft([]);
    setDrawing(false);
  }, [result.id]);
  const preview = showBefore && result.scene.image_id
    ? result.input_preview ?? `/files/${result.scene.image_id}-input.png`
    : result.preview;
  const approved = result.review_status.toLowerCase() === "approved" && !dirty;
  const hintId = useId();
  const waitHint = busy ? "Đợi thao tác hiện tại hoàn tất." : drawing ? "Khép polygon hoặc bấm Hủy nét." : "";
  const saveHint = waitHint || (!dirty ? "Chỉnh polygon trước khi lưu." : "");
  const confirmHint = (busy ? waitHint : "") || (!polygons.length ? "Vẽ ít nhất một polygon trước khi xác nhận." : dirty ? "Bấm Lưu nhãn trước khi xác nhận." : "");
  const approveHint = waitHint || confirmHint || (!confirmed ? "Tích ô “Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc”." : "");
  const exportHint = waitHint || (!approved ? "Bấm Duyệt ảnh và nhãn trước khi xuất." : "");
  function edit(next: number[][]) {
    setPolygons(next);
    setDirty(true);
    setConfirmed(false);
    setDownload("");
  }
  async function action(task: () => Promise<void>) {
    setBusy(true);
    try {
      await task();
    } catch (e) {
      onError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    await action(async () => {
      const saved = await api<Result>(
        `/api/results/${result.id}/annotations`,
        "PUT",
        { polygons },
      );
      onUpdate({ ...result, ...saved, polygons, review_status: "pending" });
      setDirty(false);
      setConfirmed(false);
    });
  }
  function finish(points: number[]) {
    edit([...polygons, points]);
    setDraft([]);
    setDrawing(false);
  }
  return (
    <section className="review">
      <div className="section-heading">
        <div>
          <span className="eyebrow">03 / KIỂM TRA VÀ GÁN NHÃN</span>
          <h1>Duyệt ảnh và nhãn</h1>
          <p>
            Nhãn đã được tạo tự động. Kiểm tra theo vết xước thực tế, chỉ sửa
            polygon khi cần rồi xác nhận trước khi xuất.
          </p>
        </div>
        <button onClick={onBack}>← Về bàn dựng</button>
      </div>
      {!!result.warnings?.length && <div className="method-warnings" role="status">
        {result.warnings.map((warning, index) => <p key={index}>{warning}</p>)}
      </div>}
      <p className="effective-methods" data-testid="effective-methods">Đã xử lý: {resultMethodSummary(result)}</p>
      <div className="review-layout">
        <div>
          <PolygonCanvas
            key={result.id}
            imageUrl={preview}
            imageWidth={result.width}
            imageHeight={result.height}
            rings={showLabels && !showBefore ? polygons.map((points, i) => ({ id: String(i), points })) : []}
            tool={drawing ? "draw" : "move"}
            draft={draft}
            disabled={busy || showBefore}
            testId="review-canvas"
            zoomLabel="Zoom duyệt ảnh"
            onDraft={setDraft}
            onClose={finish}
            onEdit={(id, points) => edit(polygons.map((polygon, i) => String(i) === id ? points : polygon))}
            toolbar={<>
              <button className={showBefore ? "active" : ""} disabled={drawing || !result.scene.image_id}
                data-tooltip={drawing ? waitHint : !result.scene.image_id ? undefined : showBefore ? "Trở về ảnh kết quả để chỉnh nhãn." : "So sánh với ảnh nền trước khi tạo xước."}
                aria-describedby={drawing ? `${hintId}-canvas` : !result.scene.image_id ? `${hintId}-original` : undefined}
                onClick={() => setShowBefore(!showBefore)}>
                {showBefore ? "Đang xem ảnh gốc" : "Đang xem kết quả"}
              </button>
              <button disabled={drawing} data-tooltip={drawing ? waitHint : undefined} aria-describedby={drawing ? `${hintId}-canvas` : undefined} onClick={() => setShowLabels(!showLabels)}>
                {showLabels ? "Ẩn nhãn" : "Hiện nhãn"}
              </button>
              {!result.scene.image_id && <small id={`${hintId}-original`} className="action-hint">Kết quả này chưa có ảnh gốc để so sánh.</small>}
            </>}
          />
          <p className="hint" id={`${hintId}-canvas`}>
            {drawing
              ? "Bấm các đỉnh rồi bấm lại gần đỉnh đầu để khép vòng."
              : "Kéo các đỉnh để sửa nhãn. Kéo nền để di chuyển khi xem 100%."}
          </p>
        </div>
        <aside className="review-panel">
          <span className={`badge ${approved ? "good" : "warn"}`}>
            {dirty ? "NHÃN CHƯA LƯU" : approved ? "ĐÃ DUYỆT" : "CHỜ DUYỆT"}
          </span>
          <h2>Polygon cuối cùng</h2>
          <p>Kiểm tra polygon đề xuất và xác nhận nhãn cuối cùng.</p>
          <button
            className="wide"
            disabled={busy}
            data-tooltip={busy ? waitHint : "Bấm lần lượt các đỉnh, rồi bấm lại đỉnh đầu để khép polygon."}
            aria-describedby={busy ? `${hintId}-save` : undefined}
            onClick={() => {
              setDrawing(true);
              setShowBefore(false);
              setShowLabels(true);
              setDraft([]);
            }}
          >
            Vẽ polygon mới
          </button>
          {drawing && (
            <div className="button-row">
              <button
                onClick={() => {
                  setDraft([]);
                  setDrawing(false);
                }}
              >
                Hủy nét
              </button>
            </div>
          )}
          <div className="polygon-list">
            {polygons.map((p, i) => (
              <div key={i}>
                <span>
                  Polygon {i + 1} · {p.length / 2} đỉnh
                </span>
                <button
                  aria-label={`Xóa polygon ${i + 1}`}
                  data-tooltip={`Xóa polygon ${i + 1} khỏi nhãn. Bấm Lưu nhãn để áp dụng.`}
                  onClick={() => edit(polygons.filter((_, j) => i !== j))}
                >
                  ×
                </button>
              </div>
            ))}
            {!polygons.length && (
              <p>Chưa có nhãn. Vẽ polygon bao quanh các vết xước nhìn thấy.</p>
            )}
          </div>
          <button
            className="wide"
            disabled={!dirty || busy || drawing}
            aria-describedby={saveHint ? `${hintId}-save` : undefined}
            onClick={save}
          >
            Lưu nhãn
          </button>
          {saveHint && <p className="action-hint" id={`${hintId}-save`}>{saveHint}</p>}
          <label className="confirmation">
            <input
              type="checkbox"
              checked={confirmed}
              disabled={dirty || !polygons.length || busy}
              aria-describedby={confirmHint ? `${hintId}-${confirmHint === saveHint ? "save" : "confirm"}` : undefined}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
            Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc
          </label>
          {confirmHint && confirmHint !== saveHint && <p className="action-hint" id={`${hintId}-confirm`}>{confirmHint}</p>}
          <button
            className="wide primary"
            disabled={
              dirty || !confirmed || !polygons.length || busy || drawing
            }
            aria-describedby={approveHint ? `${hintId}-approve` : undefined}
            onClick={() =>
              action(async () => {
                const next = await api<Result>(
                  `/api/results/${result.id}/approve`,
                  "POST",
                  { confirmed: true },
                );
                onUpdate({ ...result, ...next, review_status: "approved" });
              })
            }
          >
            Duyệt ảnh và nhãn
          </button>
          {approveHint && <p className="action-hint" id={`${hintId}-approve`}>{approveHint}</p>}
          <button
            className="wide export"
            disabled={!approved || busy || drawing}
            data-tooltip={exportHint || "Tạo bộ ZIP gồm TIFF, COCO, mask và scene."}
            aria-describedby={exportHint ? `${hintId}-export` : undefined}
            onClick={() =>
              action(async () => {
                const out = await api<{ url: string }>(
                  `/api/results/${result.id}/export`,
                  "POST",
                );
                setDownload(out.url);
              })
            }
          >
            Xuất bộ training
          </button>
          {exportHint && <p className="action-hint" id={`${hintId}-export`}>{exportHint}</p>}
          {download && (
            <a className="download" href={url(download)}>
              Tải bộ training ZIP
            </a>
          )}
          {result.label_mask && <a href={url(result.label_mask)} target="_blank" rel="noreferrer">Xem mask_full.png</a>}
          <small>
            TIFF nguyên khung + COCO + mask + scene + nguồn gốc. Chỉ nguồn train
            đã xác thực được phép xuất.
          </small>
        </aside>
      </div>
    </section>
  );
}
