# ScratchForge · F971 Studio

Web chạy cục bộ để đặt vết xước đã segment hoặc vẽ nét lên ảnh F971 OK,
tạo ảnh nguyên khung, sửa nhãn và xuất dữ liệu training đã duyệt.

**Có hai engine riêng:** baseline ghép ảnh/hòa màu và SDXL inpainting có sẵn.
SDXL chưa được fine-tune F971; chất lượng xước công nghiệp cần được duyệt thực tế.
Không có quy trình training model trong bản này.

## Chạy trên Windows

Cần Python **3.11**, Node.js/npm và dữ liệu nguồn. Mặc định hai repo đặt cạnh nhau:

```text
RD/
  GAN_Project/
    F971/NG/images/*.jpg
    F971/NG/annotations/coco.json
    F971/OK/images/*.jpg
    reports/source_image_splits.csv
  ScratchForge/
```

Mở PowerShell tại thư mục ScratchForge. Setup thường để dùng web/baseline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Mở **http://127.0.0.1:8000**. Dừng bằng Ctrl+C. Source khác vị trí mặc định:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Source "D:\duong-dan\GAN_Project"
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

## Cách sử dụng

1. Chọn ảnh trong **Nền OK**. Danh sách chỉ chứa 96 nền thuộc train.
2. Chọn **Vết xước** để bấm thêm/kéo thả stamp. Dùng các tay nắm để xoay/phóng,
   nút lật/xóa và Undo/Redo; hoặc chọn **Vẽ xước** để vẽ hướng dẫn bằng tay.
3. Chọn **AI · SDXL Inpainting** hoặc **Baseline · Ghép ảnh**, seed, mức biến đổi,
   biên chỉnh sửa; bấm **Tạo ảnh**. Baseline không phải kết quả AI.
4. So sánh trước/sau, xem 100%, vẽ polygon quanh vết xước thực tế và kéo đỉnh để sửa.
   AI được đổi hình nên mask hướng dẫn không tự trở thành nhãn ground truth.
5. **Lưu nhãn**, xác nhận đã kiểm tra ảnh/nhãn, rồi **Duyệt ảnh và nhãn**.
   Sau đó mới **Xuất bộ training**. Sửa nhãn sẽ hủy duyệt và vô hiệu link xuất cũ.
6. Tab **Ảnh đã tạo** cho mở lại kết quả. Từ màn duyệt có thể mở lại scene để dựng tiếp.

Ảnh upload ngoài dataset chỉ để thử nghiệm, không được xuất thành bộ training.
Mọi ảnh thật còn cần người dùng duyệt chất lượng và độ khớp nhãn trước sử dụng.

## Dữ liệu và đầu ra

- Nguồn đã xác minh: **49 NG / 245 polygon**, **137 OK**, tất cả nguyên khung.
- Split NG: **33 train / 8 validation / 8 test**; OK: **96 / 21 / 20**.
  Dùng `source_image_splits.csv`; không suy split từ bộ patch cũ.
- `data/raw/F971/`: bản sao checksum-verified; source không bị sửa.
- `data/fullframe/index.json`, `coco.json`, `provenance.json`: toàn bộ ảnh, nhãn
  tọa độ gốc, split và checksum. Ảnh OK có annotation rỗng.
- `data/library/`: **138 stamp train** từ polygon đầy đủ có context; 4 instance
  chạm biên HOLD (`53, 194, 203, 233`), 106 annotation evaluation không dùng làm stamp.
  Asset stamp là vùng trích nội bộ; dữ liệu chính và ảnh kết quả vẫn nguyên khung.
- `outputs/`: ảnh input/kết quả, mask hướng dẫn, vùng AI được sửa, scene, trạng thái
  job/duyệt và ZIP. Không dùng 114 patch legacy hoặc S942 làm setup mặc định.

Mỗi ZIP đã duyệt gồm `image.png` nguyên khung, `mask.png` từ nhãn cuối,
`annotations.json` COCO polygon, `scene.json`, `provenance.json`. Provenance lưu
nguồn nền/stamp, split, seed, engine, tham số và model revision. Chỉ nguồn train
được xuất; validation/test thật được giữ riêng để đánh giá.

AI xử lý vùng bao trọn nét cùng ngữ cảnh rồi ghép lại ảnh gốc. Những vùng giao nhau
được gộp, không chia tile làm cắt nét. Adapter giữ tỉ lệ vùng xử lý khi đưa vào model;
nét rất nhỏ hoặc quá dài vẫn cần kiểm tra ở 100%. Pixel ngoài vùng sửa giữ nguyên.

## Kiểm thử và phát triển

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q --cov=backend --cov=scratchforge --cov=scripts.import_f971 --cov=scripts.build_library --cov=scripts.verify_dataset --cov-fail-under=80
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
