import { useEffect, useRef, useState } from "react";
import { url, type Stamp } from "./api";

function scratchBounds(stamp: Stamp) {
  let left = stamp.width, top = stamp.height, right = 0, bottom = 0;
  for (const ring of stamp.segmentation ?? []) {
    for (let i = 0; i + 1 < ring.length; i += 2) {
      left = Math.min(left, ring[i]); right = Math.max(right, ring[i]);
      top = Math.min(top, ring[i + 1]); bottom = Math.max(bottom, ring[i + 1]);
    }
  }
  if (!(left <= right && top <= bottom)) return `0 0 ${stamp.width} ${stamp.height}`;
  const padding = Math.max(2, Math.max(right - left, bottom - top) * .07);
  const x = Math.max(0, left - padding), y = Math.max(0, top - padding);
  return `${x} ${y} ${Math.min(stamp.width, right + padding + 1) - x} ${Math.min(stamp.height, bottom + padding + 1) - y}`;
}

function StampImage({ stamp, detail = false }: { stamp: Stamp; detail?: boolean }) {
  const preview = detail ? stamp.preview_alpha ?? stamp.thumbnail : stamp.thumbnail ?? stamp.preview_alpha;
  return <svg viewBox={scratchBounds(stamp)} aria-hidden="true">
    <foreignObject width={stamp.width} height={stamp.height}>
      <img loading={detail ? "eager" : "lazy"} src={url(`/library-files/${preview ?? stamp.rgb}`)} alt="" />
    </foreignObject>
  </svg>;
}

export function StampCard({ stamp, disabled, onAdd, onPreview }: {
  stamp: Stamp; disabled: boolean; onAdd: () => void; onPreview: () => void;
}) {
  return <div className="stamp-card-wrap">
    <button className="stamp-card" aria-label={`Thêm ${stamp.id}`} disabled={disabled} draggable={!disabled}
      aria-describedby={disabled ? "place-stamp-hint" : undefined}
      data-tooltip="Bấm để đặt stamp hoặc kéo tới vị trí mong muốn."
      onDragStart={event => event.dataTransfer.setData("stamp", stamp.id)} onClick={onAdd}>
      <div><StampImage stamp={stamp} /></div>
      <span>{stamp.id}</span>
      <small>{stamp.width} × {stamp.height}</small>
    </button>
    <button type="button" className="stamp-magnify" aria-label={`Phóng to ${stamp.id}`} data-tooltip="Mở ảnh vết xước để xem chi tiết." onClick={onPreview}>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <circle cx="10" cy="10" r="6" /><path d="m14.5 14.5 5 5M7 10h6m-3-3v6" />
      </svg>
    </button>
  </div>;
}

export function StampPreview({ stamp, onClose }: { stamp: Stamp; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [zoom, setZoom] = useState(1);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog ref={dialog} className="stamp-preview" aria-label="Xem vết xước" onClose={onClose}>
    <div className="stamp-preview-toolbar">
      <label>Phóng to <input type="range" aria-label="Mức phóng to vết xước" min="1" max="8" step=".5" value={zoom} onChange={e => setZoom(+e.target.value)} /> {zoom}×</label>
      <button autoFocus onClick={() => dialog.current?.close()}>Đóng</button>
    </div>
    <div className="stamp-preview-image"><div style={{ width: `${zoom * 100}%`, height: `${zoom * 100}%` }}><StampImage stamp={stamp} detail /></div></div>
    <small>{stamp.id} · {stamp.width} × {stamp.height} px</small>
  </dialog>;
}
