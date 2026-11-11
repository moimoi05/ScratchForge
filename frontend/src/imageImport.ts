import { API, url, type Background } from "./api";

export type ImportState = {
  name: string;
  phase: "upload" | "processing" | "preview" | "success" | "error";
  percent?: number;
};

export function uploadTiff(file: File, onProgress: (percent?: number) => void): Promise<Background> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", API + "/api/upload");
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      onProgress(event.lengthComputable ? Math.round(event.loaded / event.total * 100) : undefined);
    };
    request.upload.onload = () => onProgress(100);
    request.onerror = () => reject(Error("Mất kết nối khi import TIFF. Hãy thử lại."));
    request.onabort = () => reject(Error("Import TIFF đã bị ngắt."));
    request.onload = () => {
      const payload = request.response;
      if (request.status < 200 || request.status >= 300) {
        reject(Error(typeof payload?.detail === "string" ? payload.detail : `HTTP ${request.status}`));
      } else if (!payload?.id || !payload?.url) {
        reject(Error("Máy chủ trả về thông tin ảnh không hợp lệ."));
      } else {
        resolve(payload);
      }
    };
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}

export function loadBackgroundPreview(background: Background): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      image.decode().then(() => resolve(image), () => reject(Error("Không thể giải mã ảnh xem trước. Hãy thử import lại.")));
    };
    image.onerror = () => reject(Error("Không thể tải ảnh xem trước. Hãy thử import lại."));
    image.src = url(background.url);
  });
}
