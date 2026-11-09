# ROI visual inspection ? preliminary, not training approval

All 9 contact sheets (133 original-image train instances, orange polygon overlay plus white binary mask) were inspected. No mask displacement is apparent at sheet scale. Exact pixel equality and original polygon-to-local offset are separately tested for every ROI.

Cases requiring full-resolution review:
- 8,9,83,84: long scratches overlap printed markings; thin parallel/repeated instances (9/83) require annotation duplicate review.
- 106?110,113,114,117,122,125,133,142: broad/irregular polygon shapes; verify taxonomy and tightness.
- 160,161,166?168,174,206,208,223?225,231: small, blurred or low-contrast marks, pending HOLD decision.
- 151,152,156,157,163?165,169?189,205,207,209,210,212?221,229,232,234?243: inspect printed text, edges, surrounding metal and neighboring defects in context; nearby defects may exist outside target mask.
- Other instances: no obvious sheet-scale alignment failure, still pending native-resolution review.

Nine automatic HOLD cases and reasons are in library.json. Eight are unverified source splits; one frame-boundary case. Evaluation annotations remain excluded. These are not fabricated or reassigned.

Every instance retains PENDING_VISUAL_REVIEW. A reviewer must inspect full-resolution RGB against mask before any training set is approved. Training is disabled. Legacy 114 patches remain reference only.

## Full original COCO audit extension

All 245 original instances were cropped separately in data/roi_audit, with zero mapping/rasterization errors. Counts: train 133, val 16, test 84, HOLD 12. Train sheets match the nine inspected library sheets; eight additional val/test/HOLD sheets were inspected. All 245 are thus screened at sheet scale, but remain pending detailed native-resolution approval.

HOLD audit: 53,194,203,233 touch original-image boundary; 158,159,191,202,204,230,241,245 lack verified source split. No full-extent claim is made for boundary cases. Val 6,238,239 and test 226 are tiny/blurred and need detailed review. Test 56,58 contain multiple close thin strokes represented by irregular connected polygons, requiring annotation topology review. Repeated similar polygons on the same source should be checked for overlap and duplication before training.

Audit data is partitioned and never consumed as training stamps. No val/test data was added to library.
