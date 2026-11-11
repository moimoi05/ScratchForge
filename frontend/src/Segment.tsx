import { useEffect, useId, useRef, useState } from "react";
import { api, message, type Background } from "./api";
import { PolygonCanvas, type PolygonTool } from "./PolygonCanvas";

const groups = [
  ["f971-scratch", "F971 Scratch"],
  ["f971-ok", "F971 OK"],
  ["s942-ok", "S942 OK"],
] as const;
type Group = (typeof groups)[number][0];
type Annotation = { id: number | null; category_id: number; segmentation: number[][] };
type DraftAnnotation = Annotation & { key: string };
type Frame = {
  id: string;
  group: Group;
  file_name: string;
  width: number;
  height: number;
  split: string | null;
  preview: string;
  review_status: "pending" | "approved";
  annotation_count: number;
  revision: number;
};
type Detail = Frame & { annotations: Annotation[] };
const base = "/api/segment";

function openRing(points: number[]) {
  let end = points.length;
  while (end > 6 && points[end - 2] === points[0] && points[end - 1] === points[1]) end -= 2;
  return end === points.length ? points : points.slice(0, end);
}

const draftOf = (annotations: Annotation[]): DraftAnnotation[] => annotations.map((item) => ({
  ...item,
  key: String(item.id),
  segmentation: item.segmentation.map(openRing),
}));
const ringKey = (annotation: DraftAnnotation, index: number) => `${annotation.key}-${index}`;

export function Segment({
  onError,
  onLibraryChange,
  canOpenBackground,
  onOpenBackground,
}: {
  onError: (error: string) => void;
  onLibraryChange: () => void;
  canOpenBackground: boolean;
  onOpenBackground: (image: Background) => void;
}) {
  const [frames, setFrames] = useState<Frame[]>();
  const [frame, setFrame] = useState<Detail>();
  const [group, setGroup] = useState<Group>("f971-scratch");
  const [query, setQuery] = useState("");
  const [annotations, setAnnotations] = useState<DraftAnnotation[]>([]);
  const [selectedRing, setSelectedRing] = useState<string>();
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [tool, setTool] = useState<PolygonTool>("move");
  const [draft, setDraft] = useState<number[]>([]);
  const [showLabels, setShowLabels] = useState(true);
  const [previewReady, setPreviewReady] = useState(false);
  const [notice, setNotice] = useState("");
  const nextKey = useRef(0);
  const framesRef = useRef<HTMLDivElement>(null);
  const basePath = frame ? `${base}/images/${frame.id}` : "";
  const drawing = tool === "draw";
  const locked = dirty || drawing || busy;
  const rings = annotations.flatMap((item) =>
    item.segmentation.map((points, index) => ({
      id: ringKey(item, index), points, annotation: item, index,
    })),
  );
  const visible = frames?.filter(
    (item) => item.group === group && item.file_name.toLowerCase().includes(query.trim().toLowerCase()),
  ) ?? [];

  useEffect(() => {
    let current = true;
    api<Frame[]>(`${base}/images`).then(
      (items) => { if (current) setFrames(items); },
      (error) => { if (current) { setFrames([]); onError(message(error)); } },
    );
    return () => { current = false; };
  }, [onError]);

  function resetTool() {
    setTool("move");
    setDraft([]);
  }

  function applyDetail(detail: Detail, keepSelection = false) {
    const updated = draftOf(detail.annotations);
    if (frame?.id !== detail.id || frame.preview !== detail.preview) setPreviewReady(false);
    setFrame(detail);
    setAnnotations(updated);
    setDirty(false);
    resetTool();
    setSelectedRing((current) => {
      if (keepSelection && updated.some((item) => item.segmentation.some((_, index) => ringKey(item, index) === current)))
        return current;
      return updated.length ? ringKey(updated[0], 0) : undefined;
    });
    setFrames((items) => items?.map((item) => item.id === detail.id ? detail : item));
  }

  async function action(run: () => Promise<void>) {
    setBusy(true);
    setNotice("");
    onError("");
    try {
      await run();
    } catch (error) {
      onError(message(error));
    } finally {
      setBusy(false);
    }
  }

  async function loadFrame(item: Frame) {
    applyDetail(await api<Detail>(`${base}/images/${item.id}`));
    setShowLabels(true);
  }

  function open(item: Frame) {
    return action(() => loadFrame(item));
  }

  function edit(next: DraftAnnotation[]) {
    setAnnotations(next);
    setDirty(true);
    setNotice("");
  }

  function editRing(id: string, points: number[]) {
    edit(annotations.map((item) => ({
      ...item,
      segmentation: item.segmentation.map((ring, index) => ringKey(item, index) === id ? points : ring),
    })));
  }

  function removeRing(annotation: DraftAnnotation, index: number) {
    const remaining = annotation.segmentation.filter((_, i) => i !== index);
    const next = annotations.flatMap((item) => {
      if (item !== annotation) return [item];
      return remaining.length ? [{ ...item, segmentation: remaining }] : [];
    });
    edit(next);
    setSelectedRing(next.length ? ringKey(next[0], 0) : undefined);
    resetTool();
  }

  function close(points: number[]) {
    const item = { id: null, key: `new-${nextKey.current++}`, category_id: 1, segmentation: [points] };
    edit([...annotations, item]);
    setSelectedRing(ringKey(item, 0));
    resetTool();
  }

  function importCoco(file: File) {
    return action(async () => {
      const form = new FormData();
      form.append("file", file);
      const imported = await api<{ images: number; annotations: number }>(`${base}/import`, "POST", form);
      setFrames(await api<Frame[]>(`${base}/images`));
      if (frame) applyDetail(await api<Detail>(basePath), true);
      setNotice(`Đã nạp COCO · ${imported.images} ảnh · ${imported.annotations} nhãn`);
      onLibraryChange();
    });
  }

  function save() {
    if (!frame) return;
    return action(async () => {
      const saved = await api<Detail>(`${basePath}/annotations`, "PUT", {
        revision: frame.revision,
        annotations: annotations.map(({ id, category_id, segmentation }) => ({ id, category_id, segmentation })),
      });
      applyDetail(saved, true);
      setNotice("Đã lưu nhãn");
      onLibraryChange();
    });
  }

  function approve() {
    if (!frame) return;
    return action(async () => {
      applyDetail(await api<Detail>(`${basePath}/approve`, "POST", { revision: frame.revision }), true);
      onLibraryChange();
      const position = visible.findIndex((item) => item.id === frame.id);
      const next = position >= 0
        ? visible.slice(position + 1).find((item) => item.review_status === "pending")
        : undefined;
      setNotice(`Đã duyệt nhãn ${frame.file_name}`);
      if (next) await loadFrame(next);
    });
  }

  const approved = frame?.review_status === "approved" && !dirty;
  const hintId = useId();
  const waitHint = busy ? "Đợi thao tác hiện tại hoàn tất." : "";
  const drawHint = drawing ? "Khép polygon hoặc bấm Hủy nét." : "";
  const switchHint = waitHint || drawHint || (dirty ? "Bấm Lưu nhãn hoặc Bỏ sửa chưa lưu trước khi đổi ảnh hay nạp COCO." : "");
  const previewHint = waitHint || drawHint || (!previewReady ? "Đợi ảnh TIFF tải xong." : "");
  const verticesHint = !drawing ? waitHint || (!showLabels ? "Bấm Hiện nhãn để chọn polygon."
    : !selectedRing ? (rings.length ? "Chọn một polygon để sửa đỉnh." : "Vẽ một polygon trước khi sửa đỉnh.") : "") : "";
  const saveHint = waitHint || drawHint || (!dirty ? "Chỉnh polygon trước khi lưu." : "");
  const approveHint = previewHint || (dirty ? "Bấm Lưu nhãn trước khi duyệt." : approved ? "Nhãn này đã được duyệt." : "");
  const hint = {
    draw: "Bấm các đỉnh rồi bấm lại gần đỉnh đầu để khép vòng.",
    insert: "Bấm gần cạnh của polygon đã chọn để chèn đỉnh.",
    delete: "Bấm một đỉnh để xóa. Polygon cần ít nhất ba đỉnh.",
    move: "Kéo đỉnh để sửa. Kéo nền để di chuyển và cuộn chuột để zoom.",
  }[tool];

  return (
    <section className="segment-page" onKeyDown={(event) => {
      if (event.key === "Escape") resetTool();
    }}>
      <div className="section-heading">
        <div>
          <h1>Segment</h1>
          <p>Duyệt nhãn COCO trên ảnh TIFF và sửa từng đỉnh.</p>
        </div>
        <label className="segment-import">
          Nạp COCO JSON
          <input type="file" accept=".json,application/json" disabled={locked}
            aria-describedby={locked ? `${hintId}-switch` : undefined} onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (file) void importCoco(file);
          }} />
        </label>
      </div>
      {notice && <p className="segment-notice" role="status">{notice}</p>}
      <div className="segment-layout">
        <aside className="segment-sources">
          <div className="segment-groups" aria-label="Nhóm ảnh TIFF">
            {groups.map(([id, name]) => (
              <button type="button" key={id} aria-pressed={group === id}
                className={group === id ? "active" : ""} disabled={locked}
                aria-describedby={locked ? `${hintId}-switch` : undefined}
                onClick={() => { setGroup(id); setQuery(""); framesRef.current?.scrollTo(0, 0); }}>
                {name}
              </button>
            ))}
          </div>
          <input className="search" aria-label="Tìm ảnh TIFF" placeholder="Tìm ảnh TIFF…"
            value={query} disabled={busy} data-tooltip={waitHint || undefined} onChange={(event) => setQuery(event.target.value)} />
          <p className="asset-note">{visible.length} ảnh</p>
          {switchHint && <p className="action-hint" id={`${hintId}-switch`}>{switchHint}</p>}
          <div className="segment-frame-list" ref={framesRef}>
            {visible.map((item) => (
              <button type="button" key={item.id} title={item.file_name} aria-label={`Mở ${item.file_name}`}
                aria-current={frame?.id === item.id ? "true" : undefined}
                className={frame?.id === item.id ? "active" : ""}
                disabled={locked} aria-describedby={locked ? `${hintId}-switch` : undefined} onClick={() => void open(item)}>
                <span>{item.file_name}</span>
                <small>{item.annotation_count} nhãn · {item.review_status === "approved" ? "Đã duyệt" : "Chờ duyệt"}</small>
              </button>
            ))}
            {!visible.length && <p>{frames ? "Không tìm thấy ảnh trong nhóm này." : "Đang tải danh mục TIFF…"}</p>}
          </div>
        </aside>
        {frame ? (
          <>
            <div className="segment-workspace">
              <div className="segment-frame-title">
                <strong>{frame.file_name}</strong>
                <span>{frame.width} × {frame.height}</span>
                {frame.split && <span>{frame.split.toUpperCase()}</span>}
                {frame.group === "f971-ok" && frame.split === "train" &&
                  <button disabled={locked || !canOpenBackground}
                    aria-describedby={locked ? `${hintId}-switch` : !canOpenBackground ? `${hintId}-background` : undefined} onClick={() =>
                    onOpenBackground({ id: frame.id, width: frame.width, height: frame.height,
                      url: frame.preview, source_image: frame.file_name, split: "train" })
                  }>Dùng nền ở bàn dựng</button>}
                {frame.group === "f971-ok" && frame.split === "train" && !locked && !canOpenBackground &&
                  <p className="action-hint" id={`${hintId}-background`}>Đợi bàn dựng xử lý xong trước khi đổi nền.</p>}
              </div>
              <PolygonCanvas key={frame.id} imageUrl={frame.preview}
                imageWidth={frame.width} imageHeight={frame.height}
                rings={showLabels ? rings : []} selectedRing={selectedRing} tool={tool}
                draft={draft} disabled={!previewReady || busy} testId="segment-canvas" zoomLabel="Zoom Segment"
                onDraft={setDraft} onClose={close} onEdit={editRing} onSelect={setSelectedRing}
                onImageLoaded={setPreviewReady}
                toolbar={<button disabled={drawing} data-tooltip={drawHint || undefined} onClick={() => {
                  resetTool(); setShowLabels(!showLabels);
                }}>{showLabels ? "Ẩn nhãn" : "Hiện nhãn"}</button>}
              />
              <p className="hint">{hint}</p>
            </div>
            <aside className="review-panel segment-labels">
              <span data-testid="segment-status" className={`badge ${approved ? "good" : "warn"}`}>
                {dirty ? "NHÃN CHƯA LƯU" : approved ? "ĐÃ DUYỆT" : "CHỜ DUYỆT"}
              </span>
              <h2>Polygon</h2>
              <button className="wide" disabled={!!previewHint} data-tooltip={previewHint || "Bấm các đỉnh, rồi bấm lại đỉnh đầu để khép polygon."} aria-describedby={previewHint ? `${hintId}-preview` : undefined} onClick={() => {
                setShowLabels(true); setTool("draw"); setDraft([]);
              }}>Vẽ polygon mới</button>
              {previewHint && <p className="action-hint" id={`${hintId}-preview`}>{previewHint}</p>}
              {drawing ? <button className="wide" onClick={resetTool}>Hủy nét</button> : (
                <div className="button-row">
                  <button aria-pressed={tool === "insert"} className={tool === "insert" ? "active" : ""}
                    disabled={!selectedRing || busy || !showLabels}
                    data-tooltip={verticesHint || "Bấm gần cạnh của polygon đã chọn để thêm đỉnh."}
                    aria-describedby={verticesHint ? `${hintId}-vertices` : undefined}
                    onClick={() => setTool(tool === "insert" ? "move" : "insert")}>Thêm đỉnh</button>
                  <button aria-pressed={tool === "delete"} className={tool === "delete" ? "active" : ""}
                    disabled={!selectedRing || busy || !showLabels}
                    data-tooltip={verticesHint || "Bấm đỉnh cần xóa. Giữ ít nhất ba đỉnh trong polygon."}
                    aria-describedby={verticesHint ? `${hintId}-vertices` : undefined}
                    onClick={() => setTool(tool === "delete" ? "move" : "delete")}>Xóa đỉnh</button>
                </div>
              )}
              {verticesHint && <p className="action-hint" id={`${hintId}-vertices`}>{verticesHint}</p>}
              <div className="polygon-list">
                {rings.map((ring, i) => (
                    <div key={ring.id}>
                      <button className={selectedRing === ring.id ? "active" : ""} aria-pressed={selectedRing === ring.id}
                        disabled={busy || drawing} data-tooltip={waitHint || drawHint || undefined} onClick={() => {
                          setShowLabels(true); setSelectedRing(ring.id);
                        }}>Polygon {i + 1}</button>
                      <span>{ring.points.length / 2} đỉnh</span>
                      <button aria-label={`Xóa polygon ${i + 1}`} disabled={busy || drawing}
                        data-tooltip={waitHint || drawHint || `Xóa polygon ${i + 1}. Bấm Lưu nhãn để áp dụng.`}
                        onClick={() => removeRing(ring.annotation, ring.index)}>×</button>
                    </div>
                ))}
                {!rings.length && <p>Ảnh này chưa có polygon.</p>}
              </div>
              <button className="wide" disabled={!!saveHint} aria-describedby={saveHint ? `${hintId}-${previewHint === saveHint ? "preview" : "save"}` : undefined}
                onClick={save}>Lưu nhãn</button>
              {saveHint && saveHint !== previewHint && <p className="action-hint" id={`${hintId}-save`}>{saveHint}</p>}
              {dirty && <button className="wide" disabled={busy || drawing}
                data-tooltip={waitHint || drawHint || "Khôi phục nhãn đã lưu gần nhất."}
                onClick={() => applyDetail(frame, true)}>Bỏ sửa chưa lưu</button>}
              <button className="wide primary" disabled={!!approveHint}
                aria-describedby={approveHint ? `${hintId}-${previewHint === approveHint ? "preview" : "approve"}` : undefined}
                onClick={approve}>Duyệt nhãn</button>
              {approveHint && approveHint !== previewHint && <p className="action-hint" id={`${hintId}-approve`}>{approveHint}</p>}
              {frame.group === "f971-scratch" && frame.split === "train" &&
                <small>Nhãn đã duyệt cập nhật stamp của ảnh này.</small>}
            </aside>
          </>
        ) : (
          <div className="empty-state segment-empty"><p>Chọn ảnh TIFF để xem và chỉnh nhãn.</p></div>
        )}
      </div>
    </section>
  );
}
