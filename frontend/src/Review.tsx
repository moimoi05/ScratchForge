import { useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KImage, Line, Circle } from "react-konva";
import useImage from "use-image";
import { api, message, url, type Result } from "./api";

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
    [showLabels, setShowLabels] = useState(true),
    [zoom, setZoom] = useState(1),
    [width, setWidth] = useState(900),
    [pan, setPan] = useState({ x: 0, y: 0 });
  const holder = useRef<HTMLDivElement>(null);
  const [generated, loadStatus] = useImage(url(result.preview), "anonymous");
  const [original, setOriginal] = useState("");
  const [before] = useImage(original, "anonymous");
  useEffect(() => {
    setPolygons(result.polygons);
    setDirty(false);
    setConfirmed(false);
    setDownload("");
    setDraft([]);
  }, [result.id]);
  useEffect(() => {
    const obs = new ResizeObserver((entries) =>
      setWidth(Math.max(280, entries[0].contentRect.width)),
    );
    if (holder.current) obs.observe(holder.current);
    return () => obs.disconnect();
  }, []);
  useEffect(() => {
    setZoom(Math.min(width / result.width, 560 / result.height));
    setPan({ x: 0, y: 0 });
  }, [result.id, width]);
  useEffect(() => {
    setOriginal(
      result.scene.image_id
        ? url(`/files/${result.scene.image_id}-input.png`)
        : "",
    );
  }, [result.scene.image_id]);
  const approved = result.review_status.toLowerCase() === "approved" && !dirty;
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
  function finish() {
    if (draft.length >= 6) {
      edit([...polygons, draft]);
      setDraft([]);
      setDrawing(false);
    }
  }
  return (
    <section className="review">
      <div className="section-heading">
        <div>
          <span className="eyebrow">03 / KIỂM TRA VÀ GÁN NHÃN</span>
          <h1>Duyệt ảnh và nhãn</h1>
          <p>
            AI có thể thay đổi hình dạng. Hãy sửa polygon theo vết xước thực tế
            trước khi xuất.
          </p>
        </div>
        <button onClick={onBack}>← Về bàn dựng</button>
      </div>
      <div className="review-layout">
        <div>
          <div className="canvas-toolbar">
            <button
              className={showBefore ? "active" : ""}
              onClick={() => setShowBefore(!showBefore)}
            >
              {showBefore ? "Đang xem ảnh gốc" : "Đang xem kết quả"}
            </button>
            <button onClick={() => setShowLabels(!showLabels)}>
              {showLabels ? "Ẩn nhãn" : "Hiện nhãn"}
            </button>
            <button
              onClick={() => {
                setZoom(Math.min(width / result.width, 560 / result.height));
                setPan({ x: 0, y: 0 });
              }}
            >
              Vừa khung
            </button>
            <button
              onClick={() => {
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
            >
              100%
            </button>
            <input
              aria-label="Zoom duyệt ảnh"
              type="range"
              min=".02"
              max="3"
              step=".01"
              value={zoom}
              onChange={(e) => setZoom(+e.target.value)}
            />
            <span>{Math.round(zoom * 100)}%</span>
          </div>
          <div
            className="canvas"
            ref={holder}
            data-testid="review-canvas"
            data-image-loaded={!!generated}
          >
            <Stage
              width={width}
              height={560}
              scaleX={zoom}
              scaleY={zoom}
              x={pan.x}
              y={pan.y}
              draggable={!drawing}
              onDragEnd={(e) => {
                if (e.target === e.target.getStage())
                  setPan({ x: e.target.x(), y: e.target.y() });
              }}
              onClick={(e) => {
                if (!drawing || showBefore || !generated) return;
                const p = e.target.getStage()!.getRelativePointerPosition();
                if (p)
                  setDraft([
                    ...draft,
                    Math.max(0, Math.min(result.width, p.x)),
                    Math.max(0, Math.min(result.height, p.y)),
                  ]);
              }}
            >
              <Layer>
                <KImage image={showBefore ? before : generated} />
                {showLabels &&
                  !showBefore &&
                  polygons.map((poly, i) => (
                    <Line
                      key={i}
                      points={poly}
                      closed
                      stroke="#62e6c1"
                      strokeWidth={1.5 / zoom}
                      fill="rgba(98,230,193,.12)"
                      listening={false}
                    />
                  ))}
                {showLabels &&
                  !showBefore &&
                  !drawing &&
                  polygons.flatMap((poly, i) =>
                    Array.from({ length: poly.length / 2 }, (_, v) => (
                      <Circle
                        key={`${i}-${v}`}
                        x={poly[v * 2]}
                        y={poly[v * 2 + 1]}
                        radius={4 / zoom}
                        fill="#62e6c1"
                        stroke="#0e191c"
                        strokeWidth={1 / zoom}
                        draggable
                        onDragEnd={(e) => {
                          const next = poly.map((value, k) =>
                            k === v * 2
                              ? Math.max(
                                  0,
                                  Math.min(result.width, e.target.x()),
                                )
                              : k === v * 2 + 1
                                ? Math.max(
                                    0,
                                    Math.min(result.height, e.target.y()),
                                  )
                                : value,
                          );
                          edit(polygons.map((p, k) => (k === i ? next : p)));
                        }}
                      />
                    )),
                  )}
                <Line
                  points={draft}
                  stroke="#ffca82"
                  strokeWidth={2 / zoom}
                  listening={false}
                />
                {draft.map((_, i) =>
                  i % 2 === 0 ? (
                    <Circle
                      key={i}
                      x={draft[i]}
                      y={draft[i + 1]}
                      radius={3 / zoom}
                      fill="#ffca82"
                      listening={false}
                    />
                  ) : null,
                )}
              </Layer>
            </Stage>
            {!generated && (
              <div className="canvas-empty">
                <p role="status">
                  {loadStatus === "failed"
                    ? "Không tải được ảnh kết quả. Hãy mở lại từ Ảnh đã tạo."
                    : "Đang tải ảnh nguyên khung…"}
                </p>
              </div>
            )}
          </div>
          <p className="hint">
            {drawing
              ? "Bấm từng đỉnh trên ảnh, sau đó đóng polygon."
              : "Kéo các đỉnh để sửa nhãn. Kéo nền để di chuyển khi xem 100%."}
          </p>
        </div>
        <aside className="review-panel">
          <span className={`badge ${approved ? "good" : "warn"}`}>
            {dirty ? "NHÃN CHƯA LƯU" : approved ? "ĐÃ DUYỆT" : "CHỜ DUYỆT"}
          </span>
          <h2>Polygon cuối cùng</h2>
          <p>Nét hướng dẫn không tự động trở thành nhãn đúng.</p>
          <button
            className="wide"
            disabled={busy}
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
              <button disabled={draft.length < 6} onClick={finish}>
                Đóng polygon
              </button>
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
            onClick={save}
          >
            Lưu nhãn
          </button>
          <label className="confirmation">
            <input
              type="checkbox"
              checked={confirmed}
              disabled={dirty || !polygons.length || busy}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
            Tôi đã kiểm tra ảnh và polygon ở độ phân giải gốc
          </label>
          <button
            className="wide primary"
            disabled={
              dirty || !confirmed || !polygons.length || busy || drawing
            }
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
          <button
            className="wide export"
            disabled={!approved || busy || drawing}
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
          {download && (
            <a className="download" href={url(download)}>
              Tải bộ training ZIP
            </a>
          )}
          <small>
            PNG nguyên khung + COCO + mask + scene + nguồn gốc. Chỉ nguồn train
            đã xác thực được phép xuất.
          </small>
        </aside>
      </div>
    </section>
  );
}
