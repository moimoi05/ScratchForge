# ScratchForge Studio

Independent R&D workspace, planned repository name `scratchforge-studio`. No Defect-GAN source, trainer or loss was copied. No remote, commit, push, or training is performed.

## Windows native / Python 3.11

PowerShell, from `D:\RTC\RD\ScratchForge` (activation is optional):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python scripts/migrate_legacy_data.py
.\.venv\Scripts\python scripts/verify_dataset.py
.\.venv\Scripts\python scripts/build_library.py
.\.venv\Scripts\python scripts/build_roi_audit.py
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Second PowerShell:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run dev
```

Open http://127.0.0.1:5173; API docs http://127.0.0.1:8000/docs. Use RGB PNG/JPEG, select stamps or drag from library, transform via handles, brush, undo/redo, generate baseline, download ZIP. Zoom slider and background drag provide zoom/pan. All scene positions are original image pixels; stamp position is its center. Uniform scale, clockwise rotation in screen coordinates and optional horizontal flip are applied by backend inverse affine transform. Canvas is an interactive approximation; generated backend PNG is authoritative.

## Data and provenance

`data/raw/F971` and `data/raw/S942` are checksum-verified copies. `data/provenance.json` tracks raw old/new paths, sizes and hashes; `data/manifest.json` tracks all 114 reference patches, source names, source paths, splits, RGB dimensions, crop offsets and hashes. Original source is read-only. Copy collisions with different hashes abort; identical files are reused. Migration validates every patch against an exact decoded RGB source crop before converting polygon coordinates. It never fabricates missing mappings. Run migration again to check idempotence.

**114 legacy patches are reference only, never a default new training dataset.** Their 39 clean replacements are retained. Counts: train 72, val 17, test 25. The curated COCO is a reference mapping only.

**New ROI source is original NG F971 RGB plus original-image COCO coordinates.** `scripts/build_library.py` uses verified source split assignments. It creates train-only native-resolution 256/512/adaptive crops with 16 px context, clipped context at image boundaries, never resizing source scratches. Each instance has RGB, binary mask, polygon, source hash, source offset, split and pending review status. Frame-touching polygons and unknown splits are HOLD. Val/test are excluded from the stamp library. COCO JSON originals remain byte-for-byte copies. No pretrained model exists here.

`data/library/contact-*.jpg` shows every train ROI overlay and mask. Contact sheets are scaled for inspection only. Every ROI remains PENDING_VISUAL_REVIEW until detailed human review at original resolution; sheet review alone does not authorize training. See reports for suspect cases.

## Baseline / export

Current preview copies transformed scratch RGB inside a binary instance mask. Brush uses a fixed scratch-colored baseline. No GAN or inpainting result is claimed. Export ZIP includes RGB PNG, union mask PNG, exact per-instance uncompressed COCO RLE, and editable scene JSON. RLE preserves holes and pixel coverage; export polygons are not approximated. Overlap belongs to both instance masks. Outside the union mask pixels remain exactly unchanged. Preview realism, local color matching and blend-edge quality require further work.

## AI extension

`scratchforge/engine.py`: local mask-conditioned generator, ROI inference, loading, training monitoring, checkpoint and quality-evaluation contracts, and RGB compositing. Implement a LocalGenerator adapter for future GAN/inpainting; keep request coordinates and export interface stable. Training page intentionally reports no model/checkpoints. There is no training command or automatic epoch loop.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts/verify_dataset.py
npm.cmd --prefix frontend run build
.\.venv\Scripts\python scripts/check_gpu.py
.\.venv\Scripts\python scripts/make_example.py
```

See `reports/TEST_RESULTS.md` for actual results and limitations. `data/`, outputs, dependencies and caches are git-ignored. Do not commit proprietary images. App binds loopback only and is intended for a single local user; deployment/auth, persistent scene editing, detailed dataset review decisions and training orchestration remain future work.

GPU wheels pinned from the official PyTorch previous versions page: https://pytorch.org/get-started/previous-versions/ . CUDA kernel execution must pass on the local RTX 5060 Ti; no driver/system changes are performed.

`data/roi_audit/` contains all 245 original instances for visual audit only, partitioned as train 133, val 16, test 84, HOLD 12 (8 unknown splits and 4 boundary instances). The training Stamp Library excludes evaluation data; its 9 HOLD items refer only to the pre-library train/unknown-split selection.
