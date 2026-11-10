# ScratchForge project instructions

## Product goal
Build a local Windows web studio for generating realistic F971 scratch training data.
Users place polygon-segmented scratch stamps on clean F971 images or draw guidance
by hand, run local AI, correct the resulting polygons, and approve training exports.

## Data contract
- Default source: `../GAN_Project/F971`; source files are read-only.
- Import full-frame NG/OK images and `NG/annotations/coco.json` with checksums.
- Use `../GAN_Project/reports/source_image_splits.csv` for source-level splits.
- Expected source inventory: 49 NG images, 245 scratch annotations, 137 OK images.
- Original RGB dimensions and full-frame coordinates are authoritative.
- Internal regions with context are allowed, but must contain complete scratch
  instances. Never replace the source dataset or final outputs with cropped patches.
- Only train backgrounds and train stamps may produce training exports. Preserve
  real validation/test data. Keep the four frame-touching source instances on HOLD.
- Legacy 114 patches and S942 are optional references, not setup dependencies.
- Dataset, models, outputs, caches, and proprietary images stay out of Git.

## Generation and annotation
- First release uses pretrained SDXL inpainting; no F971 fine-tuning by default.
- Distinguish baseline compositing from actual AI. Never silently substitute one.
- Keep guidance masks, editable AI regions, and final labels separate.
- AI may change scratch shape within the editable region. Preserve every pixel
  outside that region when compositing onto the full-resolution source.
- Generated results start pending review. Users must correct/confirm final polygons
  before training export. Any image/label change invalidates prior approval.
- Export full-frame PNG, reviewed COCO polygons, corresponding mask, editable scene,
  and provenance including source IDs/splits, model revision, seed, and parameters.
- Serialize GPU jobs; persist status and report missing models/CUDA/OOM explicitly.

## Engineering workflow
- Delegate independent domain work with explicit file ownership; do not revert
  another contributor's work.
- Write and execute failing tests before production changes. Use isolated fixtures
  for unit/API tests and real-image verification separately. Target >=80% coverage
  for changed Python modules and cover the critical browser journey with E2E tests.
- Review correctness and security after implementation. Validate input sizes,
  coordinates, IDs, paths, and provenance at boundaries.
- Prefer small focused modules and immutable request/state updates.
- Do not commit/push or modify drivers/system CUDA without explicit authorization.
- Default to localhost, Python 3.11, and the existing React/FastAPI stack.
- GPU smoke tests only pass after real CUDA inference on the target machine;
  cached reports and mocked tests are not hardware evidence.
- Use CodeGraph before code navigation only when `.codegraph/` already exists;
  do not initialize an index without a user request.
