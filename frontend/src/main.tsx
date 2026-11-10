import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  api,
  message,
  url,
  type Background,
  type Result,
  type Scene,
  type Stamp,
} from "./api";
import { StudioCanvas } from "./StudioCanvas";
import { Review } from "./Review";
import "./style.css";

const emptyScene: Scene = { placements: [], brushes: [] };
type Job = { id: string; status: string; error?: string; result?: Result };
function App() {
  const [page, setPage] = useState<"studio" | "dataset" | "history" | "review">(
    "studio",
  );
  const [stamps, setStamps] = useState<Stamp[]>([]),
    [backgrounds, setBackgrounds] = useState<Background[]>([]),
    [image, setImage] = useState<Background>();
  const [scene, setScene] = useState<Scene>(emptyScene),
    [past, setPast] = useState<Scene[]>([]),
    [future, setFuture] = useState<Scene[]>([]);
  const [selected, setSelected] = useState(-1),
    [tool, setTool] = useState<"select" | "brush">("select"),
    [brushWidth, setBrushWidth] = useState(12);
  const [engine, setEngine] = useState<"baseline" | "sdxl">("sdxl"),
    [seed, setSeed] = useState(42),
    [strength, setStrength] = useState(0.65),
    [margin, setMargin] = useState(24);
  const [result, setResult] = useState<Result>(),
    [history, setHistory] = useState<Result[]>([]),
    [dataset, setDataset] = useState<Record<string, unknown>>();
  const [health, setHealth] = useState<{
      ai?: { model_available?: boolean };
      engine?: string;
    }>(),
    [job, setJob] = useState<Job>(),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [search, setSearch] = useState(""),
    [assetTab, setAssetTab] = useState<"backgrounds" | "stamps">("backgrounds");
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
    if (!job || ["succeeded", "completed", "failed"].includes(job.status))
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
    setImage(next);
    setScene(emptyScene);
    setPast([]);
    setFuture([]);
    setSelected(-1);
    setResult(undefined);
    setAssetTab("stamps");
    setJob(undefined);
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
    setBusy(true);
    setError("");
    try {
      const next = await api<Job>("/api/jobs", "POST", {
        image_id: image!.id,
        ...scene,
        engine,
        seed,
        strength,
        edit_margin: margin,
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
    if (!result?.scene.image_id) return;
    const s = result.scene;
    opened({
      id: s.image_id!,
      width: result.width,
      height: result.height,
      url: `/files/${s.image_id}-input.png`,
    });
    setScene({ placements: s.placements ?? [], brushes: s.brushes ?? [] });
    if (s.engine) setEngine(s.engine);
    if (s.seed !== undefined) setSeed(s.seed);
    if (s.strength !== undefined) setStrength(s.strength);
    if (s.edit_margin !== undefined) setMargin(s.edit_margin);
    setPage("studio");
  }
  const count = scene.placements.length + scene.brushes.length;
  const assets = backgrounds.filter((item) =>
    (item.source_image ?? item.id).toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
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
            className={page === "dataset" ? "active" : ""}
            onClick={() => setPage("dataset")}
          >
            Dữ liệu nguồn
          </button>
          <button
            className={page === "history" ? "active" : ""}
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
                        onClick={() =>
                          act(async () =>
                            opened({
                              ...(await api<Background>(
                                `/api/backgrounds/${bg.id}/open`,
                                "POST",
                              )),
                              source_image: bg.source_image,
                              split: bg.split,
                            }),
                          )
                        }
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
                    Mở ảnh ngoài để thử
                    <input
                      type="file"
                      accept="image/png,image/jpeg"
                      disabled={busy}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file)
                          act(async () => {
                            const form = new FormData();
                            form.append("file", file);
                            opened(
                              await api<Background>(
                                "/api/upload",
                                "POST",
                                form,
                              ),
                            );
                          });
                      }}
                    />
                  </label>
                  <small>Ảnh ngoài không được xuất thành bộ training.</small>
                </>
              ) : (
                <>
                  <p className="asset-note">
                    Kéo vào ảnh hoặc bấm để thêm · nguồn train
                  </p>
                  <div className="asset-scroll stamp-grid">
                    {stamps.map((stamp) => (
                      <button
                        key={stamp.id}
                        className="stamp-card"
                        aria-label={`Thêm ${stamp.id}`}
                        draggable={!!image && !busy}
                        disabled={!image || busy}
                        onDragStart={(e) =>
                          e.dataTransfer.setData("stamp", stamp.id)
                        }
                        onClick={() => add(stamp)}
                      >
                        <div>
                          <img
                            loading="lazy"
                            src={url(
                              "/library-files/" +
                                (stamp.thumbnail ?? stamp.rgb),
                            )}
                            alt=""
                          />
                        </div>
                        <span>{stamp.id}</span>
                        <small>
                          {stamp.width} × {stamp.height}
                        </small>
                      </button>
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
                    onClick={() => setTool("select")}
                  >
                    ↖ Chọn / Di chuyển
                  </button>
                  <button
                    className={tool === "brush" ? "active" : ""}
                    disabled={!image || busy}
                    onClick={() => setTool("brush")}
                  >
                    ╱ Vẽ xước
                  </button>
                </div>
                <label className="brush-size">
                  Nét{" "}
                  <input
                    aria-label="Độ rộng nét"
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
                  onClick={() => {
                    change(emptyScene);
                    setSelected(-1);
                  }}
                >
                  Xóa tất cả
                </button>
              </div>
              <StudioCanvas
                image={image}
                scene={scene}
                stamps={stamps}
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
              <section className="generation-panel">
                <div>
                  <span className="eyebrow">02 / TẠO ẢNH</span>
                  <h2>Hòa hợp với bề mặt</h2>
                  <p>
                    AI xử lý ánh sáng và texture trong vùng chỉnh sửa. Ảnh xuất
                    luôn nguyên khung.
                  </p>
                </div>
                <div className="generation-options">
                  <label>
                    Phương pháp
                    <select
                      aria-label="Phương pháp"
                      value={engine}
                      disabled={busy}
                      onChange={(e) =>
                        setEngine(e.target.value as "baseline" | "sdxl")
                      }
                    >
                      <option value="sdxl">AI · SDXL Inpainting</option>
                      <option value="baseline">Baseline · Ghép ảnh</option>
                    </select>
                  </label>
                  <label>
                    Seed
                    <input
                      aria-label="Seed"
                      type="number"
                      min="0"
                      max="2147483647"
                      value={seed}
                      disabled={busy}
                      onChange={(e) =>
                        setSeed(
                          Math.max(0, Math.min(2147483647, +e.target.value)),
                        )
                      }
                    />
                  </label>
                  <label>
                    Mức biến đổi <b>{strength.toFixed(2)}</b>
                    <input
                      aria-label="Mức biến đổi"
                      type="range"
                      min=".1"
                      max=".99"
                      step=".01"
                      value={strength}
                      disabled={busy}
                      onChange={(e) => setStrength(+e.target.value)}
                    />
                  </label>
                  <label>
                    Biên chỉnh sửa (px)
                    <input
                      aria-label="Biên chỉnh sửa"
                      type="number"
                      min="0"
                      max="128"
                      value={margin}
                      disabled={busy}
                      onChange={(e) =>
                        setMargin(Math.max(0, Math.min(128, +e.target.value)))
                      }
                    />
                  </label>
                </div>
                <button
                  className="primary generate"
                  disabled={!image || !count || busy}
                  onClick={generate}
                >
                  {busy ? "Đang xử lý…" : "Tạo ảnh"}
                </button>
                <p className="model-note">
                  {busy
                    ? `Job ${job?.status ?? "đang gửi"} · giữ nguyên scene trong lúc xử lý`
                    : engine === "sdxl"
                      ? "Model có sẵn, chưa được huấn luyện riêng cho F971. Kết quả cần được duyệt."
                      : "Baseline là ghép pixel và hòa màu; không phải ảnh sinh bởi AI."}
                  {engine === "sdxl" && health?.ai?.model_available === false
                    ? " Chưa thấy model cục bộ — chạy bước tải model trên máy GPU."
                    : ""}
                </p>
              </section>
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
          <button onClick={restoreScene}>
            Mở lại scene để chỉnh cách đặt xước
          </button>
          <span>Tạo lại ảnh sẽ cần duyệt nhãn mới.</span>
        </div>
      )}
      {page === "dataset" && (
        <section className="data-page">
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
                  {item.scene?.engine ?? "baseline"}
                </small>
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
