# ScratchForge verification results

Date: 2026-11-09, Windows native, Python 3.11.

- Phase 1: independent empty-history Git repository initialized. No remote, commits or push. No legacy architecture/loss/trainer copied. Separate Python venv and npm lockfile created.
- Phase 2: 222 raw files copied (~1.28 GB), disk preflight passed (~138 GB free). 114 reference patches, 39 clean replacements, 47 defect / 67 clean. Splits 72 train / 17 val / 25 test. 98 patch-local reference polygons. Every crop verified pixel-exact against original RGB source before offset conversion. All raw and curated SHA256, dimensions and source split checks PASS. No HOLD/DROP patches imported into reference set.
- Phase 3: original-image COCO has 49 images / 245 annotations. 133 native-pixel train instances in library, 103 evaluation annotations excluded, 9 automatic HOLD. All 133 ROI RGB/mask pairs passed source-pixel and original-polygon coordinate tests. Nine contact sheets visually inspected; detailed findings in VISUAL_REVIEW.md. All instances remain pending native-resolution human approval. Legacy 114 patches are explicitly reference-only.
- Phase 4: React/TypeScript/Vite/Konva production build PASS. Backend loopback health HTTP 200 and Vite HTTP 200. API upload/generate/export tests PASS. Two stamps exported on real F971 OK image; exact unchanged pixels outside union mask PASS. See example.json for source, scene and output links. Web includes Dataset, Library, Training status, Generation Center and Review/Export listing.
- Phase 5: generator/load/ROI/compositor/monitor/checkpoint/evaluation contracts present. No training started or trained fidelity claim.
- pytest: 10 passed in 41.58s. Includes collision refusal/idempotent copy, invalid COCO cases, adaptive crop/translation/rasterization, every original-image train ROI, transformed multi-stamp invariance, API validation, per-instance RLE and export ZIP.

## Remaining limits

- Browser UI automation unavailable: browser inventory empty; Edge unavailable. Actual browser drag/drop, transformer, brush and undo/redo interactions have not been verified. TypeScript and API tests do not replace this acceptance test.
- All 133 train ROI/masks require detailed original-resolution review; contact-sheet screening is preliminary. Evaluation polygons have not been made into training stamps. Unknown split/boundary HOLD cases are unresolved; no reassignment inferred.
- Baseline copies texture inside binary mask and can show edges/color discontinuity. No learned synthesis. Brush preview is fixed-color baseline.
- Training UI is status-only. Engine contracts need actual model adapters, training jobs, checkpoint persistence, monitoring and quality evaluation implementation later.
- Review/Export currently lists downloadable exports; accept/reject workflow and scene reload UI need further implementation. Random uses random stamp placement; no learned shape synthesis.
- Vite bundle warning: ~571 KB minified. Starlette TestClient emits httpx deprecation warning; tests pass. No dependency vulnerabilities reported by npm audit at install time.
- Git ignored all datasets, generated outputs and environments. No source file was intentionally written; migration only opens legacy data for reading.

## Final checks

- Full original audit: 245 ROI/masks produced, 0 errors, all pairs pixel-checked against original RGB and original polygon translations by the tenth test. Audit partitions: train 133 / val 16 / test 84 / HOLD 12. All contact sheets screened; native-resolution approval is still pending.
- Migration rerun PASS: required_bytes=0, no conflicting copies. Provenance includes raw image dimensions, mode, source name and verified split when available; unknown splits remain null. Verifier also compares hashes against legacy originals.
- GPU PASS: PyTorch 2.7.1+cu128, CUDA runtime 12.8, RTX 5060 Ti, compute capability 12.0, sm_120 included. Real CUDA matrix multiplication and synchronize passed. No driver changes or training. Initial 2.11 download was replaced by the officially published cached 2.7.1 CUDA 12.8 wheel to complete installation; requirements match installed version.
- Real-image exported example visually inspected: two visible stamp edits, baseline color/edge limitations expected. All outside-mask pixels remain exact.
