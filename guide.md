# Chuyển ScratchForge sang máy GPU Windows

Mục tiêu: chạy web, ghép stamp hoặc vẽ xước trên ảnh F971 OK và xuất ảnh nguyên
khung. Stamp dùng harmonization riêng; nét vẽ dùng diffusion hoặc thuật toán CPU.
Không cần Docker cho quy trình này.

## 1. Chuẩn bị máy GPU

- Windows, GPU NVIDIA RTX 5060 Ti 16 GB VRAM, RAM 32 GB trên máy test đích.
- Driver NVIDIA hoạt động: mở PowerShell và chạy `nvidia-smi`.
- Cài **Python 3.11** từ https://www.python.org/downloads/windows/;
  kiểm tra `py -3.11 --version`.
- Cài Node.js LTS kèm npm từ https://nodejs.org/ và Git từ
  https://git-scm.com/download/win. Kiểm tra `node --version`, `npm.cmd --version`
  và `git --version`.
- Cần Internet trong lần setup đầu để tải thư viện và model, cùng dung lượng
  trống cho môi trường PyTorch, model và các ảnh TIFF kết quả.

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
Bản ZIP cũ cung cấp COCO và bảng split. Chép thêm ba thư mục TIFF gốc và bản xuất
COCO Roboflow từ máy nguồn để setup phiên bản hiện tại:

```text
D:\Projects\
  ScratchForge\
    guide.md
    scripts\setup.ps1
  GAN_Project\
    F971\NG\annotations\coco.json
    reports\source_image_splits.csv
  GAN Industry Scratch.v1i.coco-segmentation\train\_annotations.coco.json
  image\
    F971_Scratch\*.tif
    S942_OK\*.tif
    2026-09-14\ORG\MATTRUOC\CAM2\DOME-1\*.tif
```

Giữ dữ liệu ảnh và COCO trên máy cục bộ.

## 3. Cài môi trường GPU và tải model

Mở PowerShell trong thư mục code:

```powershell
Set-Location D:\Projects\ScratchForge
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu -DownloadModel
```

Script sẽ thực hiện:

1. Tạo `.venv` với Python 3.11 và cài backend.
2. Cài PyTorch `2.10.0+cu128` và Torchvision `0.25.0+cu128`, sau đó cài thư viện
   diffusion và harmonization.
3. Kiểm tra PyTorch nhận CUDA và chạy được kernel trên GPU thật.
4. Tải trọng số FP16 Safetensors của model
   **`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`** về
   `data/models/sdxl-inpaint/`. Commit model được ghi trong `model-provenance.json`.
5. Nhập ảnh nguyên khung, dựng 138 stamp train, verify dữ liệu và build frontend.

Nếu dữ liệu ở vị trí khác, chỉ định thư mục chứa `F971` và `reports`, thư mục TIFF và file COCO:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Source 'E:\Data\GAN_Project' -ImageRoot 'E:\Data\image' -Coco 'E:\Data\_annotations.coco.json' -Gpu -DownloadModel
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
3. Trong **02 / TẠO ẢNH**, chọn hai phương pháp độc lập:
   - **Phương pháp ghép Stamp**: Classical Mixed Clone/Laplacian, PCT-Net,
     Harmonizer hoặc RainNet. Chỉ áp dụng cho stamp.
   - **Phương pháp Vẽ nét**: SDXL, SD 1.5 hoặc Procedural (CPU).
     Chỉ áp dụng cho nét vẽ. Bấm **Tạo ảnh**.
4. So sánh trước/sau và xem 100%. AI có thể đổi hình dạng, nên cần vẽ/sửa polygon
   trên ảnh kết quả theo vết xước thực tế.
5. **Lưu nhãn** → xác nhận đã kiểm tra → **Duyệt ảnh và nhãn** → **Xuất bộ training**.

ZIP đầu ra mới gồm TIFF nguyên khung, mask PNG, COCO polygon đã duyệt, scene và provenance.
Đổi nhãn sẽ hủy duyệt cũ. Chỉ nền/stamp từ train được xuất; validation/test thật
được giữ riêng. Không duyệt ảnh nếu vết xước bị xóa, texture không thật hoặc nhãn lệch.

Stamp được xử lý trên ROI có biên ngữ cảnh ở độ phân giải gốc, không đi qua VAE.
Hai nhánh nhận nền gốc; stamp được ưu tiên khi chồng lên nét vẽ. Ảnh xuất giữ kích
thước nguồn, mọi pixel ngoài vùng chỉnh sửa giữ nguyên. Điều chỉnh màu vẫn có thể
đổi giá trị pixel của texture: cần kiểm tra ở mức zoom 100%.

### Tải và quản lý model ngay trên web

Hai ô phương pháp hiển thị trạng thái từng model: **Đã tải** hoặc **Chưa tải**
kèm dung lượng. Classical và Procedural dùng được ngay. Bấm model chưa tải hoặc
nút **Tải** để bắt đầu tải nền; phương pháp đang chọn vẫn được giữ nguyên.

Trong lúc tải, dòng model hiển thị tiến trình và nút **Hủy**. Hủy sẽ dừng tải,
dọn file tạm chưa hoàn chỉnh và giữ phương pháp cũ. Chỉ sau khi tải xong và kiểm
tra đủ file, model mới chuyển thành **Đã tải** và tự được chọn. Lỗi tải/verify
hiển thị ngay tại model; có thể tải lại. Tải xong chưa chứng minh inference hay
chất lượng F971 đạt yêu cầu.

Môi trường CUDA vẫn cần cài theo bước 3. Máy phát triển hiện tại chưa kiểm thử
inference thực trên RTX 5060 Ti; chạy lệnh kiểm tra GPU ở bước 4 trên máy đích.
Có thể bỏ `-DownloadModel` ở bước setup để tải model trực tiếp trên web.
SDXL/SD 1.5 ưu tiên CUDA FP16 khi có GPU; SD 1.5 dùng FP32 khi chạy CPU.
Nếu cần giảm VRAM, đặt `$env:SCRATCHFORGE_CPU_OFFLOAD = '1'` trong PowerShell
trước khi chạy `scripts/start.ps1`; mặc định diffusion được đưa lên CUDA.

### Đăng ký checkpoint stamp thủ công (tùy chọn)

Mặc định **Classical** chạy CPU, không cần checkpoint. Có thể đăng ký checkpoint
từ hướng dẫn chính thức của [PCT-Net](https://github.com/rakutentech/PCT-Net-Image-Harmonization),
[Harmonizer](https://github.com/ZHKKKe/Harmonizer) hoặc
[RainNet](https://github.com/junleen/RainNet), rồi đăng ký file cục bộ:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-harmonization.txt
.\.venv\Scripts\python scripts/setup_harmonization.py --model harmonizer --checkpoint 'D:\Models\harmonizer.pth'
```

Môi trường cần cặp PyTorch/Torchvision tương thích (đã cài ở bước GPU). Các adapter
không tự cài thư viện. PCT-Net mặc định kiến trúc `CNN_pct`; nếu checkpoint khác,
tạo `data/models/harmonization/pctnet/config.json` với `{"variant": "tên kiến trúc"}`
khớp cấu hình chính thức. RainNet có thể thay đổi texture dù không đổi kích thước
đầu ra; không có bảo đảm giữ 100% micro-texture với bất kỳ phương pháp hòa màu nào.

Thay `harmonizer` bằng `pctnet` hoặc `rainnet` tương ứng. Script clone mã nguồn
chính thức ở commit cố định vào `data/models/harmonization/<model>/source`,
lưu `checkpoint.pth` và checksum/provenance. Không ghi đè checkout đã sửa hoặc
checkpoint khác. Trọng số chỉ được nạp khi chọn phương pháp và chạy job đầu tiên.
Thiếu checkpoint hoặc nạp model lỗi sẽ dùng Classical và hiển thị cảnh báo;
provenance lưu cả phương pháp yêu cầu lẫn phương pháp thực tế.

### SD 1.5 cho nét vẽ trên CPU

```powershell
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
.\.venv\Scripts\python scripts/download_model.py --model sd15
```

Model nằm tại `data/models/sd15-inpaint`. Downloader dùng bản public kế nhiệm
`stable-diffusion-v1-5/stable-diffusion-inpainting` của model Runway cũ. CPU dùng
float32; chạy chậm hơn GPU. Nếu không muốn tải model, chọn **Procedural**.
Nhánh diffusion không tự chuyển sang Procedural khi có lỗi.

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
Nhánh nét vẽ không tự đổi sang Procedural rồi ghi là AI. Nhánh stamp có fallback
Classical với cảnh báo và provenance rõ ràng.

**Thiếu file nguồn** — kiểm tra cấu trúc ở bước 2 hoặc truyền `-Source` đúng vị trí.

**Đã tải model, muốn chạy lại các bước data/build**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu
```

**Không có GPU** — chạy setup không có `-Gpu`, sau đó start web và chọn stamp
Classical, nét vẽ Procedural. Xem `README.md` và `reports/TEST_RESULTS.md` cho kiểm thử.

## 7. Mở rộng kho TIFF và train LoRA

Mở **Dữ liệu nguồn**, dùng cột trái để import TIFF nguyên khung + COCO Polygon JSON
hoặc Crop TIFF + Mask PNG nhị phân cùng kích thước. Điền ID nguồn và xác nhận thuộc
tập train. Không dùng nguồn validation/test/HOLD; khai báo không thể biến nguồn
evaluation đã biết thành dữ liệu train. Polygon của stamp dùng tọa độ crop; polygon
ảnh kết quả dùng tọa độ nguyên khung.

Chọn stamp rồi dùng **A / Augment** để tạo biến thể. Kiểm tra preview/mask, chỉnh
polygon nếu cần và duyệt mẫu trước khi train. **B / Train LoRA** nhận stamp TIFF
thực/import hoặc augment đã duyệt, chọn SD 1.5/SDXL Inpainting, steps, seed; mặc định
SD 1.5 và 1.000 steps. Có tiến trình thực và nút Hủy. Chỉ weights `.safetensors`
hoàn chỉnh được đưa vào danh sách LoRA; mẫu AI không được train lại mặc định.

**C / Sinh stamp** dùng LoRA tương thích và mask mẫu train trên nền giả lập. Mask
tự tách từ chênh lệch ảnh trước/sau trong vùng cho phép; ngưỡng quá cao có thể làm
mất xước mảnh. Xem/sửa/duyệt nhãn trước sử dụng; mẫu hợp lệ tự hiện ở **Vết xước**.

Trong **Bàn dựng**, Stamp Brush uốn TIFF theo polyline bằng TPS rồi hòa hợp theo Ô 1.
Chọn một mẫu hoặc ngẫu nhiên; ID mẫu thực dùng được lưu theo từng nét. Chế độ AI
hiển thị LoRA đã chọn và trạng thái nạp thực; Procedural tiếp tục chạy CPU.
Ảnh mới có polygon tự đề xuất: kiểm tra ở 100%, sửa nếu cần rồi xác nhận duyệt.
ZIP gồm TIFF RGB24, LZW, 300 DPI, nguyên kích thước; `mask_full.png`, `polygon.json`
COCO, scene/provenance và các tên tương thích cũ. Sửa nhãn hủy duyệt và link export cũ.

Máy này chỉ phát triển mã; kiểm chứng GPU thực trên Windows RTX 5060 Ti 16 GB,
RAM 32 GB. Sau khi cài `requirements-gpu.txt` và `requirements-ai.txt` (gồm PEFT),
tải cả hai base, duyệt stamp train và dừng web:

```powershell
.\.venv\Scripts\python scripts/smoke_lora.py --base-model both --steps 2
# Hoặc chọn cụ thể stamp TIFF đã duyệt:
.\.venv\Scripts\python scripts/smoke_lora.py --base-model sd15 --stamp-id '<ID>' --steps 2
```

Lệnh dùng TIFF OK train đã xác minh ở kích thước gốc, ưu tiên 5861×3421 nếu có. Nó kiểm tra
CUDA kernel, train/hủy thật, nạp lại LoRA, inference và TIFF/pixel ngoài mask; ghi
`reports/lora-smoke-runtime.json`. Thiếu CUDA hoặc kiểm tra thất bại trả mã khác 0.
Chạy test CPU không chứng minh GPU đạt. HALCON vẫn cần mở TIFF thật trên máy đích
và xác nhận thủ công; báo cáo smoke để trạng thái HALCON là `NOT_RUN`.
