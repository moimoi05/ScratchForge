# ScratchForge · F971 Studio

Web chạy cục bộ để đặt vết xước đã segment hoặc vẽ nét lên ảnh F971 OK,
tạo ảnh nguyên khung, sửa nhãn và xuất dữ liệu training đã duyệt.

**Có hai engine riêng:** baseline ghép ảnh/hòa màu và SDXL inpainting có sẵn.
SDXL chưa được fine-tune F971; chất lượng xước công nghiệp cần được duyệt thực tế.
Không có quy trình training model trong bản này.

## Chạy trên Windows

Cần Python **3.11**, Node.js/npm, TIFF gốc và dữ liệu đối chiếu. Bố cục mặc định:

```text
RD/
  GAN_Project/
    F971/NG/annotations/coco.json
    reports/source_image_splits.csv
  GAN Industry Scratch.v1i.coco-segmentation/train/_annotations.coco.json
  image/
    F971_Scratch/*.tif
    S942_OK/*.tif
    2026-09-14/ORG/MATTRUOC/CAM2/DOME-1/*.tif
  ScratchForge/
```

Mở PowerShell tại thư mục ScratchForge. Setup thường để dùng web/baseline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Mở **http://127.0.0.1:8000**. Dừng bằng Ctrl+C. Có thể chỉ định vị trí dữ liệu khác:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Source "D:\duong-dan\GAN_Project" -ImageRoot "D:\duong-dan\image" -Coco "D:\duong-dan\_annotations.coco.json"
```

Trên máy NVIDIA dự kiến RTX 5060 Ti **16 GB VRAM**, setup AI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu -DownloadModel
.\.venv\Scripts\python scripts/check_gpu.py --require-cuda --inference-smoke
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Lần đầu tải model cần Internet và vài GB đĩa. Script chỉ tải model về, không gửi
ảnh sản phẩm ra ngoài. Sau khi tải, inference dùng file cục bộ. CUDA Toolkit có
sẵn chưa đủ: kiểm tra phải chạy được kernel PyTorch và inference thật. Script không
cài/thay driver hoặc system CUDA. Chưa có GPU thì vẫn setup dữ liệu và chạy baseline.

## Segment trên TIFF gốc

1. Mở **Segment** và chọn **F971 Scratch**, **F971 OK** hoặc **S942 OK**. F971 Scratch có sẵn polygon từ bản xuất COCO. Các ảnh OK có thể được gán nhãn thủ công.
2. Chọn polygon để kéo, thêm hoặc xóa từng đỉnh. Khi vẽ, bấm gần đỉnh đầu sau khi đặt ít nhất ba đỉnh hợp lệ để khép vòng.
3. **Lưu nhãn** rồi **Duyệt** theo từng ảnh. Khi duyệt ảnh F971 train, thư viện stamp dùng polygon đã duyệt. Sửa lại một ảnh đã duyệt sẽ tạm rút các stamp tương ứng cho đến khi duyệt lại.

Tab Segment xử lý tọa độ theo TIFF nguyên khung. Khi nhập COCO, tên và kích thước phải khớp với ảnh nguồn. Nhãn từ JPG giữ nguyên tọa độ nếu JPG và TIFF có cùng khung hình và hướng ảnh.

## Tạo ảnh xước

1. Chọn ảnh trong **Nền OK**. Danh sách chỉ chứa 96 nền thuộc train.
2. Chọn **Vết xước** để bấm thêm/kéo thả stamp. Dùng các tay nắm để xoay/phóng,
   nút lật/xóa và Undo/Redo; hoặc chọn **Vẽ xước** để vẽ hướng dẫn bằng tay.
3. Chọn **AI · SDXL Inpainting** hoặc **Baseline · Ghép ảnh**, seed, mức biến đổi,
   biên chỉnh sửa; bấm **Tạo ảnh**. Baseline không phải kết quả AI.
4. So sánh trước/sau ở 100% và kiểm tra polygon đã tự tạo. Kéo đỉnh để sửa khi cần;
   không phải vẽ nhãn lại từ đầu. Nhãn AI được đề xuất từ xước phát hiện trên ảnh,
   không lấy nguyên mask hướng dẫn làm ground truth.
5. **Lưu nhãn**, xác nhận đã kiểm tra ảnh/nhãn, rồi **Duyệt ảnh và nhãn**.
   Sau đó mới **Xuất bộ training**. Sửa nhãn sẽ hủy duyệt và vô hiệu link xuất cũ.
6. Tab **Ảnh đã tạo** cho mở lại kết quả. Từ màn duyệt có thể mở lại scene để dựng tiếp.

Web nhận ảnh upload TIFF RGB 8-bit một trang. Ảnh upload ngoài dataset chỉ để thử nghiệm, không được xuất thành bộ training.
Mọi ảnh thật còn cần người dùng duyệt chất lượng và độ khớp nhãn trước sử dụng.

## Dữ liệu và đầu ra

- Nguồn đã xác minh: **49 NG / 245 polygon**, **137 OK**, tất cả nguyên khung.
- Split NG: **33 train / 8 validation / 8 test**; OK: **96 / 21 / 20**.
  Dùng `source_image_splits.csv`; không suy split từ bộ patch cũ.
- `data/raw/F971/`: bản sao TIFF nguyên khung được kiểm tra checksum.
- `data/fullframe/index.json`, `coco.json`, `provenance.json`: toàn bộ ảnh, nhãn
  tọa độ gốc, split và checksum. Ảnh OK có annotation rỗng.
- `data/library/`: **138 stamp train** từ polygon đầy đủ có context; 4 instance
  chạm biên HOLD (`53, 194, 203, 233`), 106 annotation evaluation không dùng làm stamp.
  Asset stamp là vùng trích nội bộ; dữ liệu chính và ảnh kết quả vẫn nguyên khung.
- `outputs/`: ảnh input/kết quả, mask hướng dẫn, vùng AI được sửa, scene, trạng thái
  job/duyệt và ZIP. Không dùng 114 patch legacy hoặc S942 làm setup mặc định.

Mỗi ZIP mới đã duyệt gồm `image.tif` RGB 24-bit nguyên khung, LZW lossless, 300 DPI;
`mask_full.png`, `polygon.json` COCO từ nhãn cuối, `scene.json`, `provenance.json`.
Tên tương thích `mask.png` và `annotations.json` cũng được giữ. Provenance lưu
nguồn nền/stamp, split, seed, engine, tham số và model revision. Chỉ nguồn train
được xuất; validation/test thật được giữ riêng để đánh giá.

AI xử lý vùng bao trọn nét cùng ngữ cảnh rồi ghép lại ảnh gốc. Những vùng giao nhau
được gộp, không chia tile làm cắt nét. Adapter giữ tỉ lệ vùng xử lý khi đưa vào model;
nét rất nhỏ hoặc quá dài vẫn cần kiểm tra ở 100%. Pixel ngoài vùng sửa giữ nguyên.

## Kiểm thử và phát triển

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q --cov=backend --cov=scratchforge --cov=scripts.import_f971 --cov=scripts.import_tiff_dataset --cov=scripts.build_library --cov=scripts.verify_dataset --cov-fail-under=80
.\.venv\Scripts\python scripts/verify_dataset.py
.\.venv\Scripts\python scripts/make_example.py
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend exec -- playwright install chromium
npm.cmd --prefix frontend run test:e2e
# Khi web đang chạy và dữ liệu thật đã nhập:
npm.cmd --prefix frontend run test:real
```

Unit/API tests dùng fixture độc lập, không cần proprietary dataset hoặc CUDA.
`make_example.py` chạy baseline trên ảnh thật và giữ kết quả chờ người duyệt.
Browser tests có fixture API riêng; kiểm thử GPU thực và duyệt chất lượng F971
là các bước độc lập, không được suy từ mock hoặc báo cáo cũ.

Nếu phát triển UI riêng: chạy API như trên, rồi `npm.cmd --prefix frontend run dev`
và mở http://127.0.0.1:5173. API docs: http://127.0.0.1:8000/docs.

Model: [`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).
Model download lưu commit cụ thể vào `model-provenance.json`; có thể cấu hình
`SCRATCHFORGE_MODEL_DIR` trước khi chạy web nếu model nằm ngoài thư mục mặc định.
PyTorch GPU được khóa bản CUDA 12.8 theo [hướng dẫn chính thức](https://pytorch.org/get-started/previous-versions/).

Ứng dụng dành cho một người dùng cục bộ, chạy **một Uvicorn worker**, bind localhost.
Dataset/model/output/dependency được git-ignore. Không commit ảnh sản phẩm.
Xem `AGENTS.md` cho quy tắc dự án và `reports/TEST_RESULTS.md` cho bằng chứng kiểm thử.

## Kho TIFF, augment và LoRA F971

Cột trái tab **Dữ liệu nguồn** chứa import và ba chế độ A/B/C. Mỗi stamp mới có
`stamp_crop.tif` gốc, `mask.png` nhị phân, `preview_alpha.png`, `polygon.json` COCO
tọa độ crop và metadata/checksum. Preview chỉ để hiển thị; ghép và Stamp Brush
đọc TIFF trực tiếp, không đi qua VAE. Uốn TPS nội suy pixel và hòa hợp có điều chỉnh
màu/sáng; đây không phải bảo đảm mọi giá trị RGB còn nguyên sau biến hình.

- **Import:** chọn TIFF nguyên khung + COCO hoặc Crop TIFF + Mask PNG cùng kích thước.
  Khai báo ID nguồn và xác nhận quyền dùng train. Nguồn val/test/HOLD đã biết bị chặn.
- **A / Augment:** chọn stamp, số biến thể và seed; xem/sửa rồi duyệt mẫu mới.
- **B / Train LoRA:** chọn stamp thực/import hoặc augment đã duyệt, base SD 1.5/SDXL
  Inpainting, steps và seed. Stamp AI và các mẫu hậu duệ AI không dùng train lại.
  Tiến trình và Hủy phản ánh job thật; weights hoàn chỉnh lưu local trong `data/models/lora`.
- **C / Sinh stamp:** chọn LoRA khớp base/revision, template train và số lượng; chỉnh
  ngưỡng tách mask nếu cần. Kết quả chờ kiểm tra/duyệt và tự đồng bộ sang Bàn dựng.

Ô 2 trong **Bàn dựng** có Stamp Brush (mẫu đã chọn/ngẫu nhiên), AI kèm LoRA và
Procedural CPU. LoRA đã chọn chỉ được báo **đã nạp** sau xác nhận từ backend.
Train/inference dùng chung hàng đợi; lỗi CUDA/model/VRAM không bị đổi thành kết quả
Procedural. Máy phát triển không GPU vẫn dùng import, augment, Stamp Brush và kiểm thử CPU.

Trên máy Windows RTX 5060 Ti 16 GB, cài PyTorch CUDA rồi các dependency AI/PEFT:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
# Tải cả hai base qua UI hoặc scripts/download_model.py; duyệt stamp train trước.
# Dừng web trước smoke để không có tiến trình khác chiếm GPU.
.\.venv\Scripts\python scripts/smoke_lora.py --base-model both --steps 2
```

Smoke dùng TIFF OK train đã xác minh ở kích thước gốc, ưu tiên 5861×3421 nếu có, chạy train và hủy sau một step,
nạp lại LoRA, inference và kiểm tra TIFF/pixel ngoài mask. Có thể truyền nhiều
`--stamp-id <ID>` để chọn mẫu; mặc định lấy tối đa bốn stamp TIFF thực đã duyệt.
Kết quả ở `reports/lora-smoke-runtime.json`; CPU hoặc thiếu một kiểm tra sẽ trả mã lỗi.
Test fixture và báo cáo cũ không phải bằng chứng CUDA. Mở TIFF kết quả bằng HALCON
trên máy đích để nghiệm thu riêng; script không tự đánh dấu HALCON đạt.
