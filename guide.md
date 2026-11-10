# Chuyển ScratchForge sang máy GPU Windows

Mục tiêu: chạy web, đặt/vẽ xước trên ảnh F971 OK và sinh ảnh nguyên khung bằng
SDXL Inpainting cục bộ. Không cần Docker cho quy trình này.

## 1. Chuẩn bị máy GPU

- Windows, GPU NVIDIA dự kiến RTX 5060 Ti 16 GB VRAM.
- Driver NVIDIA hoạt động: mở PowerShell và chạy `nvidia-smi`.
- Cài **Python 3.11** từ https://www.python.org/downloads/windows/;
  kiểm tra `py -3.11 --version`.
- Cài Node.js LTS kèm npm từ https://nodejs.org/ và Git từ
  https://git-scm.com/download/win. Kiểm tra `node --version`, `npm.cmd --version`
  và `git --version`.
- Cần Internet trong lần setup đầu để tải thư viện và model, cùng dung lượng
  trống cho môi trường PyTorch, model và các ảnh PNG kết quả.

Không chuyển `.venv` hoặc `node_modules` từ máy cũ. Script sẽ tạo lại môi trường
cho đúng máy. Script không thay driver hoặc system CUDA.

## 2. Lấy code và giải nén dữ liệu

Copy hai file sau từ máy cũ sang máy GPU:

- `F971-GPU-data-20261110.zip`
- `F971-GPU-data-20261110.zip.sha256`

Hai file được đặt tại `outputs/transfer/` trên máy đóng gói. ZIP chứa 186 ảnh JPG
nguyên khung (49 NG, 137 OK), 245 polygon trong COCO và bảng chia tập theo ảnh gốc.
Không chứa model, crop legacy, S942 hoặc môi trường Python.

Ví dụ đặt ZIP trong `D:\Transfer`, và code/data trong `D:\Projects`:

```powershell
New-Item -ItemType Directory -Force D:\Projects | Out-Null
Set-Location D:\Projects
git clone https://github.com/moimoi05/ScratchForge.git
```

Kiểm tra ZIP đã chuyển nguyên vẹn, rồi giải nén:

```powershell
$zipPath = 'D:\Transfer\F971-GPU-data-20261110.zip'
$checksumPath = "$zipPath.sha256"
$expectedHash = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]
$actualHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) { throw 'ZIP bi thay doi hoac copy chua du. Hay chuyen lai file.' }
Expand-Archive -LiteralPath $zipPath -DestinationPath D:\Projects
```

Giải nén vào thư mục cha `D:\Projects`, vì ZIP đã có thư mục `GAN_Project` bên trong.
Không thêm một lớp `GAN_Project` nữa. Kết quả cần là:

```text
D:\Projects\
  ScratchForge\
    guide.md
    scripts\setup.ps1
  GAN_Project\
    F971\
      NG\images\*.jpg
      NG\annotations\coco.json
      OK\images\*.jpg
    reports\source_image_splits.csv
  F971-DATA-CHECKSUMS.json
  F971-DATA-README.txt
```

Data không được push lên GitHub. Nếu chỉ clone code mà chưa chuyển ZIP,
setup sẽ thiếu dữ liệu nguồn.

## 3. Cài môi trường GPU và tải model

Mở PowerShell trong thư mục code:

```powershell
Set-Location D:\Projects\ScratchForge
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu -DownloadModel
```

Script sẽ thực hiện:

1. Tạo `.venv` với Python 3.11 và cài backend.
2. Cài PyTorch `2.10.0+cu128` và Torchvision `0.25.0+cu128`, sau đó cài thư viện AI.
3. Kiểm tra PyTorch nhận CUDA và chạy được kernel trên GPU thật.
4. Tải trọng số FP16 Safetensors của model
   **`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`** về
   `data/models/sdxl-inpaint/`. Commit model được ghi trong `model-provenance.json`.
5. Nhập ảnh nguyên khung, dựng 138 stamp train, verify dữ liệu và build frontend.

Nếu data không nằm cạnh repo, truyền đúng thư mục chứa `F971` và `reports`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Source 'E:\Data\GAN_Project' -Gpu -DownloadModel
```

Model có sẵn chưa được fine-tune F971. Tải model không đồng nghĩa đã huấn luyện
hoặc đã đạt chất lượng dữ liệu công nghiệp. Inference sau tải dùng file cục bộ;
ảnh sản phẩm không được gửi tới API sinh ảnh bên ngoài.

## 4. Kiểm tra inference và mở web

```powershell
.\.venv\Scripts\python scripts/check_gpu.py --require-cuda --inference-smoke
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1
```

Lệnh kiểm tra phải PASS cả CUDA kernel và model inference. Kết quả được ghi vào
`reports/gpu.json`, ảnh smoke vào `outputs/gpu-smoke.png`. Ảnh smoke chỉ chứng minh
model chạy được; vẫn cần kiểm tra chất lượng F971 thật.

Mở **http://127.0.0.1:8000** trên máy GPU. Giữ cửa sổ PowerShell đang chạy;
Ctrl+C để dừng. Những lần sau chỉ cần chạy `scripts/start.ps1`.

## 5. Dùng web và xuất dữ liệu

1. Chọn **Nền OK** rồi chọn ảnh F971 nguyên khung.
2. Chọn stamp ở **Vết xước** để kéo/thả, xoay, đổi kích thước; hoặc **Vẽ xước**.
3. Chọn **AI · SDXL Inpainting** và bấm **Tạo ảnh**. Baseline chỉ dùng thử ghép ảnh.
4. So sánh trước/sau và xem 100%. AI có thể đổi hình dạng, nên cần vẽ/sửa polygon
   trên ảnh kết quả theo vết xước thực tế.
5. **Lưu nhãn** → xác nhận đã kiểm tra → **Duyệt ảnh và nhãn** → **Xuất bộ training**.

ZIP đầu ra gồm PNG nguyên khung, mask, COCO polygon đã duyệt, scene và provenance.
Đổi nhãn sẽ hủy duyệt cũ. Chỉ nền/stamp từ train được xuất; validation/test thật
được giữ riêng. Không duyệt ảnh nếu vết xước bị xóa, texture không thật hoặc nhãn lệch.

## 6. Xử lý lỗi thường gặp

**`SDXL model missing`** — kiểm tra đúng thư mục repo và chạy lại:

```powershell
.\.venv\Scripts\python scripts/download_model.py
```

**CUDA unavailable / kernel lỗi** — kiểm tra `nvidia-smi`, sau đó:

```powershell
.\.venv\Scripts\python scripts/check_gpu.py --require-cuda
```

Có CUDA Toolkit chưa chứng minh PyTorch wheel tương thích. Giữ log lỗi để kiểm tra
driver, wheel và GPU thực tế; không cần tự cài lại toàn bộ hệ thống ngay.

**Hết VRAM** — đóng các ứng dụng đang chiếm GPU, chạy một tác vụ mỗi lần và thử lại.
Ứng dụng không tự đổi sang baseline rồi ghi là AI.

**Thiếu file nguồn** — kiểm tra cấu trúc ở bước 2 hoặc truyền `-Source` đúng vị trí.

**Đã tải model, muốn chạy lại các bước data/build**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu
```

**Không có GPU nhưng muốn xem giao diện** — chạy setup không có `-Gpu`, sau đó
start web và chọn Baseline. Xem `README.md` và `reports/TEST_RESULTS.md` cho kiểm thử.
