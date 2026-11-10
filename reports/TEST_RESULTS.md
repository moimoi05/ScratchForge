# ScratchForge verification — 2026-11-10

This report supersedes the earlier clone's setup/GPU report. Evidence below was
collected in the current checkout on Windows with Python 3.11.9 and Node 24.11.

## Data

- Imported **186 full-frame RGB images**: 49 NG, 137 OK, 245 original scratch polygons.
- Preserved source-image splits: NG 33/8/8 and OK 96/21/20 (train/val/test).
- Original sources remained read-only. Checksum-verified copies, original COCO and
  authoritative source split CSV are recorded in fullframe provenance.
- Re-import was idempotent; full-frame verification PASS. No legacy patch/S942
  dependency in the default setup.
- Library: **138 train stamps**, 106 evaluation annotations excluded. All four
  frame-touching annotations are HOLD: 53, 194, 203, 233. Assets contain the complete
  polygon and native-resolution context; all 138 RGB/mask checksum pairs verified.
- No native-resolution human approval of source stamps or generated images is
  claimed. Final labels and image quality still require user review.

## Software

- **42 Python tests passed**, combined coverage **92.37%** over backend,
  scratchforge and the three new data workflow modules. Every measured module >=80%.
- Valid RED/GREEN evidence captured for missing importer/API/generation interfaces,
  source/asset tampering, annotation deletion, split reassignment, missing model,
  off-frame placement, and failed ZIP publication.
- Tests cover fixtures independently of proprietary data: reviewed polygon export,
  mask/COCO roundtrip, approval invalidation, blocked old download links, exploratory
  upload restrictions, finite bounds, unsafe paths, failed jobs, interrupted-job
  recovery, immutable background pixels, native stamp transforms and merged ROIs.
- Security review findings fixed: unbounded retained full-frame masks, offscreen
  tint-sampling crop, ZIP partial publication, background/stamp source integrity.
- TypeScript/Vite production build PASS. Two mocked Playwright E2E tests PASS:
  scene placement, job polling, final polygon creation/vertex correction, saving,
  explicit approval/export, invalidation after changes, scene restore and API errors.
- Real browser/API smoke PASS on **5239 x 3359** F971 OK: choose source, add stamp,
  baseline generation, decoded full-frame review image, pending labels and blocked
  training export, no browser JavaScript errors. Screenshot: `outputs/ui-smoke.png`.
- Real-image baseline smoke separately verified full dimensions and pixel-exact
  preservation outside the edit mask. Results remain pending human review.
- Windows `setup.ps1 -Dev` completed end-to-end, including re-import, library,
  verification, npm clean install and build. `start.ps1` launched the studio;
  root HTTP 200, dataset PASS, 96 train backgrounds and 138 stamps returned.

## Dependencies and hardware

- Python dependency audit after updates: **no known vulnerabilities found**;
  `pip check` passed. npm audit: **0 vulnerabilities**.
- Pinned AI dependencies: Diffusers 0.42.1, Transformers 5.22.2, Accelerate 1.15.0,
  Safetensors 0.8.0, Hugging Face Hub 2.4.0. Actual SDXL pipeline import checked on CPU.
- GPU-host setup pins PyTorch 2.10.0+cu128 and Torchvision 0.25.0+cu128. That wheel
  combination is not claimed hardware-tested in this environment.
- Current host: Intel graphics; installed development PyTorch is CPU-only.
  `check_gpu.py --require-cuda` correctly fails with NOT_AVAILABLE.
- **Actual CUDA model inference: NOT RUN.** No model weights were downloaded on
  this CPU-only host. Run `setup.ps1 -Gpu -DownloadModel` and GPU/model smoke on the
  target NVIDIA machine. Adapter tests with CPU fakes establish software contracts
  only, not CUDA support, VRAM consumption, throughput or F971 realism.

## Acceptance remaining on the GPU machine

Run kernel/model smoke, then inspect representative real F971 outputs for stamp
and brush modes at full view and 100% zoom. Reject erased/implausible scratches,
visible seams or incorrect labels. Pretrained SDXL has not been fine-tuned to F971;
no guarantee of training-ready realism is made. Native/context ROIs can be resized
inside the model; verify very small and long scratches rather than assuming fidelity.

The Vite bundle is approximately 589 KB and emits its size advisory. The app is
local-only, single-user, single-GPU-worker; deployment/auth and fine-tuning remain
outside this release. No training runs or driver changes performed.

## GPU handoff archive

`outputs/transfer/F971-GPU-data-20261110.zip` contains 188 required source files
(186 full-frame JPGs, original COCO, source split CSV), a per-file checksum manifest
and data README. Size: 447,072,604 bytes (426.36 MiB). All archived member SHA256,
byte sizes and ZIP CRCs passed. The archive and SHA256 sidecar stay outside Git;
`guide.md` documents transfer, extraction, model setup and startup on Windows.
