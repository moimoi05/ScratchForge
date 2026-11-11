import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  api,
  message,
  url,
  type Background,
  type Result,
  type Scene,
  type Stamp,
  type Health,
  type LoraCheckpoint,
} from "./api";
import { StudioCanvas } from "./StudioCanvas";
import { Review } from "./Review";
import { Segment } from "./Segment";
import { GenerationPanel } from "./GenerationPanel";
import { SourceLibraryPanel } from "./SourceLibraryPanel";
import { HoverHelp } from "./HoverHelp";
import { StampCard, StampPreview } from "./StampCard";
import { loadBackgroundPreview, uploadTiff, type ImportState } from "./imageImport";
import { resultMethodSummary, settingsFromScene, type GenerationSettings } from "./generationMethods";
import "./style.css";

const emptyScene: Scene = { placements: [], brushes: [] };
type Job = { id: string; status: string; error?: string; result?: Result };
function App() {
  const [previewStamp, setPreviewStamp] = useState<Stamp>();
  const [page, setPage] = useState<"studio" | "dataset" | "history" | "review" | "segment">(
    "studio",
  );
  const [segmentOpened, setSegmentOpened] = useState(false);
  const [datasetOpened, setDatasetOpened] = useState(false);
  const [checkpoints, setCheckpoints] = useState<LoraCheckpoint[]>([]);
  const [stamps, setStamps] = useState<Stamp[]>([]),
    [sceneStamps, setSceneStamps] = useState<Stamp[]>([]),
    [backgrounds, setBackgrounds] = useState<Background[]>([]),
    [image, setImage] = useState<Background>();
  const [scene, setScene] = useState<Scene>(emptyScene),
    [past, setPast] = useState<Scene[]>([]),
    [future, setFuture] = useState<Scene[]>([]);
  const [selected, setSelected] = useState(-1),
    [tool, setTool] = useState<"select" | "brush">("select"),
    [brushWidth, setBrushWidth] = useState(12);
  const [settings, setSettings] = useState<GenerationSettings>(settingsFromScene());
  const [result, setResult] = useState<Result>(),
    [history, setHistory] = useState<Result[]>([]),
    [dataset, setDataset] = useState<Record<string, unknown>>();
  const [health, setHealth] = useState<Health>(),
    [job, setJob] = useState<Job>(),
    [error, setError] = useState(""),
    [working, setBusy] = useState(false);
  const [importState, setImportState] = useState<ImportState>();
  const [importedPreview, setImportedPreview] = useState<{ id: string; image: HTMLImageElement }>();
  const importLock = useRef(false);
  const importing = !!importState && !["success", "error"].includes(importState.phase);
  const busy = working || importing;
  const [search, setSearch] = useState(""),
    [assetTab, setAssetTab] = useState<"backgrounds" | "stamps">("backgrounds");
  const refreshLibrary = useCallback(async () => { setStamps(await api<Stamp[]>("/api/library")); }, []);
  const refreshTraining = useCallback(async () => {
    const data = await api<{ checkpoints?: LoraCheckpoint[] }>("/api/training");
    setCheckpoints(Array.isArray(data.checkpoints) ? data.checkpoints : []);
  }, []);
  useEffect(() => {
    let live = true;
    Promise.allSettled([
      api<Stamp[]>("/api/library").then((data) => live && setStamps(data)),
      api<Background[]>("/api/backgrounds").then(
        (data) => live && setBackgrounds(data),
      ),
      api<Record<string, unknown>>("/api/dataset").then(
        (data) => live && setDataset(data),
      ),
      api<typeof health>("/api/health").then((data) => live && setHealth(data)),
      api<{ checkpoints?: LoraCheckpoint[] }>("/api/training").then(data => live && setCheckpoints(Array.isArray(data.checkpoints) ? data.checkpoints : [])),
    ]).then((results) => {
      const failure = results.find((item) => item.status === "rejected");
      if (live && failure?.status === "rejected")
        setError(message(failure.reason));
    });
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (page === "history")
      api<Result[]>("/api/exports")
        .then(setHistory)
        .catch((e) => setError(message(e)));
  }, [page]);
  useEffect(() => {
    if (page !== "studio" || !["sd15", "sdxl"].includes(settings.brush_method)) return;
    let live = true;
    const poll = () => { void api<Health>("/api/health").then(data => { if (live) setHealth(data); }).catch(e => { if (live) setError(message(e)); }); };
    poll();
    const timer = setInterval(poll, 2000);
    return () => { live = false; clearInterval(timer); };
  }, [page, settings.brush_method]);
  useEffect(() => {
    if (!job || ["succeeded", "completed", "failed", "cancelled"].includes(job.status))
      return;
    let live = true;
    const timer = setTimeout(() => {
      api<Job>(`/api/jobs/${job.id}`)
        .then((next) => {
          if (!live) return;
          setJob(next);
          if (next.status === "failed") {
            setBusy(false);
            setError(
              next.error || "Không thể tạo ảnh. Kiểm tra model và bộ nhớ GPU.",
            );
          } else if (next.result) {
            setBusy(false);
            setResult(next.result);
            setPage("review");
            void api<Health>("/api/health").then(setHealth).catch(e => setError(message(e)));
          }
        })
        .catch((e) => {
          if (live) {
            setBusy(false);
            setError(message(e));
            setJob({ ...job, status: "failed" });
          }
        });
    }, 650);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [job]);
  const change = (next: Scene) => {
    setPast((items) => [...items, scene]);
    setFuture([]);
    setScene(next);
    setResult(undefined);
  };
  function opened(next: Background) {
    setImportedPreview(undefined);
    setImage(next);
    setScene(emptyScene);
    setSceneStamps([]);
    setPast([]);
    setFuture([]);
    setSelected(-1);
    setResult(undefined);
    setAssetTab("stamps");
    setJob(undefined);
  }
  async function importBackground(source: Background | File) {
    if (busy || importLock.current) return;
    importLock.current = true;
    const file = source instanceof File ? source : undefined;
    const name = file?.name ?? (source as Background).source_image ?? (source as Background).id;
    setError("");
    setImportState({ name, phase: file ? "upload" : "processing", percent: file ? 0 : undefined });
    try {
      if (file && !/\.tiff?$/i.test(file.name)) throw Error("Hãy chọn ảnh .tif hoặc .tiff.");
      if (file && file.size > 200 * 1024 * 1024) throw Error("Ảnh TIFF vượt giới hạn upload 200 MiB.");
      const next = file
        ? { ...(await uploadTiff(file, (percent) => setImportState({
            name, phase: percent === 100 ? "processing" : "upload", percent: percent === 100 ? undefined : percent,
          }))), source_image: file.name }
        : { ...(await api<Background>(`/api/backgrounds/${(source as Background).id}/open`, "POST")),
            source_image: (source as Background).source_image, split: (source as Background).split };
      setImportState({ name, phase: "preview" });
      const preview = await loadBackgroundPreview(next);
      opened(next);
      setImportedPreview({ id: next.id, image: preview });
      setImportState({ name, phase: "success" });
    } catch (error) {
      setImportState({ name, phase: "error" });
      setError(message(error));
    } finally {
      importLock.current = false;
    }
  }
  async function act(work: () => Promise<void>) {
    setError("");
    try {
      await work();
    } catch (e) {
      setError(message(e));
    }
  }
  function add(stamp: Stamp, x = image!.width / 2, y = image!.height / 2) {
    if (!image || busy) return;
    setSceneStamps((items) => items.some((item) => item.id === stamp.id) ? items : [...items, stamp]);
    change({
      ...scene,
      placements: [
        ...scene.placements,
        { stamp_id: stamp.id, x, y, scale: 1, rotation: 0, flip: false },
      ],
    });
    setSelected(scene.placements.length);
    setTool("select");
  }
  async function generate() {
    if (scene.placements.some((placement) => !stamps.some((stamp) => stamp.id === placement.stamp_id))) {
      setError("Scene có stamp đã rút khỏi thư viện. Hãy xóa các stamp cũ và chọn stamp đang có trước khi tạo ảnh.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const next = await api<Job>("/api/jobs", "POST", {
        image_id: image!.id,
        placements: scene.placements,
        brushes: scene.brushes,
        ...settings,
        ...(settings.brush_method.startsWith("stamp_brush") && settings.brush_stamp_mode === "selected"
          ? { brush_stamp_id: settings.brush_stamp_id ?? scene.placements[selected]?.stamp_id ?? stamps[0]?.id }
          : {}),
      });
      setJob(next);
    } catch (e) {
      setBusy(false);
      setError(message(e));
    }
  }
  async function reopen(id: string) {
    await act(async () => {
      const next = await api<Result>(`/api/results/${id}`);
      setResult(next);
      setPage("review");
    });
  }
  function restoreScene() {
    if (importLock.current || !result?.scene.image_id) return;
    const s = result.scene;
    opened({
      id: s.image_id!,
      width: result.width,
      height: result.height,
      url: result.input_preview ?? `/files/${s.image_id}-input.png`,
    });
    setSceneStamps(result.provenance?.stamps ?? []);
    setScene({ placements: s.placements ?? [], brushes: s.brushes ?? [] });
    setSettings(settingsFromScene(s));
    setPage("studio");
  }
  const count = scene.placements.length + scene.brushes.length;
  const waitHint = busy ? "Đợi thao tác hiện tại hoàn tất." : "";
  const importHint = importing ? "Đợi ảnh TIFF tải và mở xong." : undefined;
  const chooseStampHint = !scene.placements[selected] ? "Chọn một stamp trên ảnh trước." : "";
  const assets = backgrounds.filter((item) =>
    (item.source_image ?? item.id).toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
      <HoverHelp />
      {previewStamp && <StampPreview stamp={previewStamp} onClose={() => setPreviewStamp(undefined)} />}
      <header>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("studio");
          }}
        >
          <span className="brand-mark">╱</span>
          <b>
            ScratchForge<span>F971 SYNTHETIC DATA STUDIO</span>
          </b>
        </a>
        <nav aria-label="Điều hướng">
          <button
            className={page === "studio" ? "active" : ""}
            onClick={() => setPage("studio")}
          >
            Bàn dựng
          </button>
          <button
            disabled={importing}
            data-tooltip={importHint}
            className={page === "segment" ? "active" : ""}
            onClick={() => {
              setSegmentOpened(true);
              setPage("segment");
            }}
          >
            Segment
          </button>
          <button
            className={page === "dataset" ? "active" : ""}
            disabled={importing}
            data-tooltip={importHint}
            onClick={() => { setDatasetOpened(true); setPage("dataset"); }}
          >
            Dữ liệu nguồn
          </button>
          <button
            className={page === "history" ? "active" : ""}
            disabled={importing}
            data-tooltip={importHint}
            onClick={() => setPage("history")}
          >
            Ảnh đã tạo
          </button>
        </nav>
        <span className="local-status">
          <i />
          LOCAL WORKSPACE
        </span>
      </header>
      {error && (
        <div role="alert" className="error">
          <strong>Chưa thể hoàn thành thao tác.</strong> {error}
          <button aria-label="Đóng thông báo" onClick={() => setError("")}>
            ×
          </button>
        </div>
      )}
      {segmentOpened && (
        <div hidden={page !== "segment"}>
          <Segment
            onError={setError}
            onLibraryChange={() => { void act(refreshLibrary); }}
            canOpenBackground={!busy}
            onOpenBackground={(next) => {
              setPage("studio");
              void importBackground(next);
            }}
          />
        </div>
      )}
      {page === "studio" && (
        <>
          <div className="section-heading studio-heading">
            <div>
              <span className="eyebrow">01 / DỰNG VẾT XƯỚC</span>
              <h1>Tạo dữ liệu xước F971</h1>
              <p>
                Đặt vết xước thật hoặc vẽ nét mới. Giữ nguyên toàn bộ bức ảnh.
              </p>
            </div>
            <span className="badge">NG → OK → SYNTHETIC</span>
          </div>
          <main className="workspace">
            <aside className="assets">
              <div className="asset-tabs">
                <button
                  className={assetTab === "backgrounds" ? "active" : ""}
                  onClick={() => setAssetTab("backgrounds")}
                >
                  Nền OK <span>{backgrounds.length}</span>
                </button>
                <button
                  className={assetTab === "stamps" ? "active" : ""}
                  onClick={() => setAssetTab("stamps")}
                >
                  Vết xước <span>{stamps.length}</span>
                </button>
              </div>
              {assetTab === "backgrounds" ? (
                <>
                  <input
                    className="search"
                    aria-label="Tìm ảnh nền"
                    placeholder="Tìm ảnh F971 OK…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  <p className="asset-note">Ảnh nguyên khung · chỉ tập train</p>
                  <div className="asset-scroll">
                    {assets.map((bg) => (
                      <button
                        className="background-card"
                        key={bg.id}
                        aria-label={`Chọn ${bg.source_image ?? bg.id}`}
                        disabled={busy}
                        data-tooltip={waitHint || "Mở ảnh này làm nền cho cảnh."}
                        draggable={!busy}
                        onDragStart={(event) => event.dataTransfer.setData("background", bg.id)}
                        onClick={() => void importBackground(bg)}
                      >
                        <img
                          loading="lazy"
                          src={url(bg.thumbnail ?? bg.url)}
                          alt=""
                        />
                        <span title={bg.source_image}>
                          {bg.source_image ?? bg.id}
                        </span>
                        <small>
                          {bg.width} × {bg.height} · TRAIN
                        </small>
                      </button>
                    ))}
                    {!assets.length && (
                      <p className="empty-message">
                        Chưa có nền OK. Chạy setup để nhập dữ liệu F971.
                      </p>
                    )}
                  </div>
                  <label className="upload">
                    Mở TIFF ngoài để thử
                    <input
                      type="file"
                      accept=".tif,.tiff,image/tiff"
                      disabled={busy}
                      data-tooltip={waitHint || "Mở TIFF để thử dựng cảnh. Ảnh ngoài không dùng để xuất bộ training."}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        e.target.value = "";
                        if (file) void importBackground(file);
                      }}
                    />
                  </label>
                  <small>Ảnh ngoài không được xuất thành bộ training.</small>
                </>
              ) : (
                <>
                  <p className={image && !busy ? "asset-note" : "action-hint"} id="place-stamp-hint">
                    {busy ? "Đợi thao tác hiện tại hoàn tất." : !image ? "Chọn ảnh ở mục Nền OK hoặc mở TIFF trước khi thêm stamp." : "Kéo vào ảnh hoặc bấm để thêm · nguồn train"}
                  </p>
                  <div className="asset-scroll stamp-grid">
                    {stamps.map((stamp) => (
                      <StampCard key={stamp.id} stamp={stamp} disabled={!image || busy}
                        onAdd={() => add(stamp)} onPreview={() => setPreviewStamp(stamp)} />
                    ))}
                    {!stamps.length && (
                      <p>
                        Chưa có thư viện xước. Kiểm tra bước build library trong
                        setup.
                      </p>
                    )}
                  </div>
                  <p className="asset-note">
                    Nhãn nguồn cần kiểm tra trực quan. Nhãn ảnh sinh phải được
                    duyệt lại.
                  </p>
                </>
              )}
            </aside>
            <article className="editor">
              <div className="tools">
                <div className="tool-group">
                  <button
                    className={tool === "select" ? "active" : ""}
                    data-tooltip="Chọn stamp trên ảnh để di chuyển hoặc chỉnh kích thước."
                    onClick={() => setTool("select")}
                  >
                    ↖ Chọn / Di chuyển
                  </button>
                  <button
                    className={tool === "brush" ? "active" : ""}
                    disabled={!image || busy}
                    data-tooltip={waitHint || (!image ? "Mở ảnh nền TIFF trước khi vẽ xước." : "Kéo chuột trên ảnh để vẽ nét xước.")}
                    onClick={() => setTool("brush")}
                  >
                    ╱ Vẽ xước
                  </button>
                </div>
                <label className="brush-size">
                  Nét{" "}
                  <input
                    aria-label="Độ rộng nét"
                    data-tooltip="Độ rộng nét vẽ, tính bằng pixel của ảnh gốc."
                    type="number"
                    min="1"
                    max="512"
                    value={brushWidth}
                    onChange={(e) =>
                      setBrushWidth(Math.min(512, Math.max(1, +e.target.value)))
                    }
                  />{" "}
                  px
                </label>
                <div className="tool-group">
                  <button
                    disabled={!past.length || busy}
                    data-tooltip={waitHint || (!past.length ? "Chưa có thao tác nào để hoàn tác." : "Hoàn tác thay đổi gần nhất trên ảnh.")}
                    onClick={() => {
                      setFuture([scene, ...future]);
                      setScene(past[past.length - 1]);
                      setPast(past.slice(0, -1));
                      setSelected(-1);
                    }}
                  >
                    ↶ Undo
                  </button>
                  <button
                    disabled={!future.length || busy}
                    data-tooltip={waitHint || (!future.length ? "Hoàn tác một thao tác trước khi làm lại." : "Khôi phục thao tác vừa hoàn tác.")}
                    onClick={() => {
                      setPast([...past, scene]);
                      setScene(future[0]);
                      setFuture(future.slice(1));
                      setSelected(-1);
                    }}
                  >
                    ↷ Redo
                  </button>
                </div>
                <button
                  disabled={!scene.placements[selected] || busy}
                  data-tooltip={waitHint || chooseStampHint || "Lật ngang stamp đang chọn."}
                  onClick={() =>
                    change({
                      ...scene,
                      placements: scene.placements.map((p, i) =>
                        i === selected ? { ...p, flip: !p.flip } : p,
                      ),
                    })
                  }
                >
                  Lật
                </button>
                <button
                  disabled={!scene.placements[selected] || busy}
                  data-tooltip={waitHint || chooseStampHint || "Xóa stamp đang chọn khỏi ảnh."}
                  onClick={() => {
                    change({
                      ...scene,
                      placements: scene.placements.filter(
                        (_, i) => i !== selected,
                      ),
                    });
                    setSelected(-1);
                  }}
                >
                  Xóa nét chọn
                </button>
                <button
                  disabled={!count || busy}
                  data-tooltip={waitHint || (!count ? "Chưa có stamp hoặc nét vẽ để xóa." : "Xóa tất cả stamp và nét vẽ khỏi ảnh.")}
                  onClick={() => {
                    change(emptyScene);
                    setSelected(-1);
                  }}
                >
                  Xóa tất cả
                </button>
              </div>
              {importState && (
                <div className={`import-status import-${importState.phase}`} role="status" aria-live="polite">
                  <span>{({
                    upload: "Đang import · tải TIFF lên",
                    processing: "Đang import · máy chủ đang xử lý TIFF",
                    preview: "Đang import · tải và giải mã ảnh xem trước",
                    success: "Import thành công",
                    error: "Import thất bại",
                  })[importState.phase]} · {importState.name}
                  {importState.phase === "upload" && importState.percent !== undefined && ` · ${importState.percent}%`}</span>
                  {importing && <progress aria-label="Tiến trình import ảnh TIFF" max={100}
                    value={importState.phase === "upload" ? importState.percent : undefined} />}
                </div>
              )}
              <StudioCanvas
                image={image}
                preview={importedPreview?.id === image?.id ? importedPreview?.image : undefined}
                disabled={busy}
                onImportFile={(file) => void importBackground(file)}
                onImportBackground={(id) => {
                  const source = backgrounds.find((item) => item.id === id);
                  if (source) void importBackground(source);
                }}
                scene={scene}
                stamps={[...stamps, ...sceneStamps.filter((item) => !stamps.some((stamp) => stamp.id === item.id))]}
                tool={tool}
                brushWidth={brushWidth}
                selected={selected}
                onSelect={setSelected}
                onChange={(next) => {
                  if (!busy) change(next);
                }}
                onAdd={add}
              />
              <div className="editor-footer">
                <span>
                  {image?.source_image ??
                    (image ? "Ảnh thử ngoài dataset" : "Chưa chọn ảnh nền")}
                </span>
                <span>
                  {scene.placements.length} stamp · {scene.brushes.length} nét
                  vẽ
                </span>
              </div>
              <GenerationPanel
                settings={settings} onChange={setSettings} busy={busy}
                canGenerate={!!image && !!count} hasImage={!!image} jobStatus={job?.status}
                hasBrushes={scene.brushes.length > 0} health={health} onGenerate={generate}
                stamps={stamps} checkpoints={checkpoints} currentStampId={scene.placements[selected]?.stamp_id}
              />
            </article>
          </main>
        </>
      )}
      {page === "review" && result && (
        <Review
          key={result.id}
          result={result}
          onUpdate={setResult}
          onError={setError}
          onBack={() => setPage("studio")}
        />
      )}
      {page === "review" && result && (
        <div className="restore-scene">
          <button disabled={importing} data-tooltip={importHint || "Mở lại cảnh đã lưu để chỉnh sửa stamp và nét vẽ."} onClick={restoreScene}>
            Mở lại scene để chỉnh cách đặt xước
          </button>
          <span>Tạo lại ảnh sẽ cần duyệt nhãn mới.</span>
        </div>
      )}
      {datasetOpened && (
        <section className="data-page" hidden={page !== "dataset"}>
          <div className="section-heading">
            <div>
              <span className="eyebrow">DỮ LIỆU CÓ NGUỒN GỐC</span>
              <h1>F971 · ảnh gốc và polygon</h1>
              <p>
                Nguồn GAN_Project được giữ nguyên. Không dùng patch legacy để
                thay thế ảnh gốc.
              </p>
            </div>
          </div>
          <div className="source-layout">
            <SourceLibraryPanel stamps={stamps} checkpoints={checkpoints} onLibraryChange={refreshLibrary}
              onTrainingChange={refreshTraining} onError={setError}/>
            <div className="source-reports">
          <div className="stats">
            <div>
              <strong>49</strong>
              <span>Ảnh NG nguyên khung</span>
            </div>
            <div>
              <strong>245</strong>
              <span>Polygon xước nguồn</span>
            </div>
            <div>
              <strong>137</strong>
              <span>Ảnh nền OK</span>
            </div>
            <div>
              <strong>{stamps.length}</strong>
              <span>Stamp train khả dụng</span>
            </div>
          </div>
          <div className="data-columns">
            <article>
              <h2>Phân chia theo ảnh nguồn</h2>
              <table>
                <thead>
                  <tr>
                    <th>Nguồn</th>
                    <th>Train</th>
                    <th>Validation</th>
                    <th>Test</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>NG</th>
                    <td>33</td>
                    <td>8</td>
                    <td>8</td>
                  </tr>
                  <tr>
                    <th>OK</th>
                    <td>96</td>
                    <td>21</td>
                    <td>20</td>
                  </tr>
                </tbody>
              </table>
              <p>
                Chỉ nguồn train đi vào bộ sinh training. Validation và test thật
                được giữ riêng. Instance chạm biên được giữ HOLD.
              </p>
            </article>
            <article>
              <h2>Kết quả kiểm tra dataset</h2>
              <pre>
                {dataset
                  ? JSON.stringify(dataset, null, 2)
                  : "Chưa có báo cáo. Chạy setup và verify_dataset."}
              </pre>
            </article>
          </div>
            </div>
          </div>
        </section>
      )}
      {page === "history" && (
        <section className="history-page">
          <div className="section-heading">
            <div>
              <span className="eyebrow">KẾT QUẢ ĐÃ LƯU</span>
              <h1>Ảnh đã tạo</h1>
              <p>Mở lại ảnh để sửa nhãn, duyệt hoặc xuất dữ liệu.</p>
            </div>
          </div>
          <div className="history-grid">
            {history.map((item) => (
              <button
                className="result-card"
                key={item.id}
                onClick={() => reopen(item.id)}
              >
                <img src={url(item.preview)} alt="Kết quả F971" />
                <span
                  className={`badge ${item.review_status.toLowerCase() === "approved" ? "good" : "warn"}`}
                >
                  {item.review_status.toLowerCase() === "approved"
                    ? "ĐÃ DUYỆT"
                    : "CHỜ DUYỆT"}
                </span>
                <strong>{item.id.slice(0, 12)}</strong>
                <small>
                  {item.width} × {item.height} ·{" "}
                  {resultMethodSummary(item)}
                </small>
                {!!item.warnings?.length && <small className="fallback-label">Có fallback · xem chi tiết kết quả</small>}
              </button>
            ))}
          </div>
          {!history.length && (
            <div className="empty-state">
              <h2>Chưa có ảnh đã tạo</h2>
              <p>Bắt đầu từ bàn dựng để thêm vết xước lên ảnh OK.</p>
              <button onClick={() => setPage("studio")}>Đến bàn dựng</button>
            </div>
          )}
        </section>
      )}
      <footer>
        <span>ScratchForge / F971</span>
        <span>Ảnh nguyên khung · Nhãn được duyệt · Nguồn có thể truy xuất</span>
        <span>LOCAL ONLY</span>
      </footer>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
