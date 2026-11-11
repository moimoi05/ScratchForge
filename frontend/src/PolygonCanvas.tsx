import { useEffect, useRef, useState, type ReactNode } from "react";
import type { KonvaEventObject } from "konva/lib/Node";
import { Circle, Image as KImage, Layer, Line, Stage } from "react-konva";
import { url } from "./api";

type Point = { x: number; y: number };
export type Ring = { id: string; points: number[] };
export type PolygonTool = "move" | "draw" | "insert" | "delete";

const height = 560;
const closeRadius = 10;
const minZoom = 0.02;
const maxZoom = 3;
const origin = { x: 0, y: 0 };
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

function usePreview(imageUrl: string) {
  const [preview, setPreview] = useState<{ source: string; image?: HTMLImageElement }>();
  useEffect(() => {
    const image = new window.Image();
    let live = true;
    image.crossOrigin = "anonymous";
    image.onload = () => {
      void image.decode().catch(() => undefined).then(() => {
        if (live) setPreview({ source: imageUrl, image });
      });
    };
    image.onerror = () => setPreview({ source: imageUrl });
    image.src = url(imageUrl);
    return () => {
      live = false;
      image.onload = null;
      image.onerror = null;
    };
  }, [imageUrl]);
  return preview?.source === imageUrl ? preview : undefined;
}

function hasArea(points: number[]) {
  let twiceArea = 0;
  for (let i = 0; i < points.length; i += 2) {
    const j = (i + 2) % points.length;
    twiceArea += points[i] * points[j + 1] - points[j] * points[i + 1];
  }
  return points.length >= 6 && Math.abs(twiceArea) >= 1;
}

function closestEdge(points: number[], point: Point) {
  let best: { index: number; point: Point; distance: number } | undefined;
  for (let i = 0; i < points.length; i += 2) {
    const j = (i + 2) % points.length;
    const dx = points[j] - points[i];
    const dy = points[j + 1] - points[i + 1];
    const lengthSquared = dx * dx + dy * dy;
    if (!lengthSquared) continue;
    const t = clamp(((point.x - points[i]) * dx + (point.y - points[i + 1]) * dy) / lengthSquared, 1);
    if (t === 0 || t === 1) continue;
    const projected = { x: points[i] + t * dx, y: points[i + 1] + t * dy };
    const gap = distance(projected, point);
    if (!best || gap < best.distance) best = { index: i + 2, point: projected, distance: gap };
  }
  return best;
}

type Props = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  rings: Ring[];
  selectedRing?: string;
  tool: PolygonTool;
  draft: number[];
  disabled?: boolean;
  testId: string;
  zoomLabel: string;
  toolbar?: ReactNode;
  onDraft: (points: number[]) => void;
  onClose: (points: number[]) => void;
  onEdit: (id: string, points: number[]) => void;
  onSelect?: (id: string) => void;
  onImageLoaded?: (loaded: boolean) => void;
};

export function PolygonCanvas({
  imageUrl, imageWidth, imageHeight, rings, selectedRing, tool, draft, disabled,
  testId, zoomLabel, toolbar, onDraft, onClose, onEdit, onSelect, onImageLoaded,
}: Props) {
  const holder = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [view, setView] = useState({ zoom: 0, pan: origin });
  const [dragging, setDragging] = useState<{ id: string; points: number[] }>();
  const preview = usePreview(imageUrl);
  const image = preview?.image;
  const fit = Math.min(1, width / imageWidth, height / imageHeight);
  const zoom = view.zoom || fit;
  const pan = view.pan;

  useEffect(() => {
    onImageLoaded?.(!!image);
  }, [image, onImageLoaded]);

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(Math.max(1, Math.floor(entry.contentRect.width)));
    });
    if (holder.current) observer.observe(holder.current);
    return () => observer.disconnect();
  }, []);

  function changeZoom(next: number, anchor = { x: width / 2, y: height / 2 }) {
    const ratio = next / zoom;
    setView({
      zoom: next,
      pan: { x: anchor.x - (anchor.x - pan.x) * ratio, y: anchor.y - (anchor.y - pan.y) * ratio },
    });
  }

  function click(point: Point) {
    if (tool === "draw") {
      if (draft.length && distance(point, { x: draft[0], y: draft[1] }) * zoom <= closeRadius && hasArea(draft)) {
        onClose(draft);
        return;
      }
      for (let i = 0; i < draft.length; i += 2) {
        if (point.x === draft[i] && point.y === draft[i + 1]) return;
      }
      onDraft([...draft, point.x, point.y]);
    } else if (tool === "insert") {
      const ring = rings.find((item) => item.id === selectedRing);
      if (!ring) return;
      const edge = closestEdge(ring.points, point);
      if (edge && edge.distance * zoom <= closeRadius)
        onEdit(ring.id, [...ring.points.slice(0, edge.index), edge.point.x, edge.point.y, ...ring.points.slice(edge.index)]);
    }
  }

  function canvasClick(e: KonvaEventObject<MouseEvent | TouchEvent>) {
    if (!image || disabled) return;
    const point = e.target.getStage()?.getRelativePointerPosition();
    if (point && point.x >= 0 && point.x <= imageWidth && point.y >= 0 && point.y <= imageHeight)
      click(point);
  }

  return (
    <>
      <div className="canvas-toolbar">
        {toolbar}
        <button data-tooltip="Thu phóng để thấy toàn bộ ảnh." onClick={() => setView({ zoom: 0, pan: origin })}>Vừa khung</button>
        <button data-tooltip="Xem một pixel ảnh trên một pixel canvas." onClick={() => setView({ zoom: 1, pan: origin })}>100%</button>
        <input
          aria-label={zoomLabel}
          type="range"
          min={Math.min(minZoom, fit)}
          max={maxZoom}
          step="any"
          value={zoom}
          onChange={(e) => changeZoom(+e.target.value)}
        />
        <output aria-label="Tỷ lệ zoom">{Math.round(zoom * 100)}%</output>
      </div>
      <div
        className="canvas polygon-canvas"
        ref={holder}
        data-testid={testId}
        data-image-loaded={!!image}
        data-zoom={zoom}
      >
        <Stage
          width={width}
          height={height}
          scaleX={zoom}
          scaleY={zoom}
          x={pan.x}
          y={pan.y}
          draggable={tool === "move"}
          onDragEnd={(e) => {
            if (e.target === e.target.getStage())
              setView({ zoom, pan: { x: e.target.x(), y: e.target.y() } });
          }}
          onWheel={(e) => {
            e.evt.preventDefault();
            const anchor = e.target.getStage()?.getPointerPosition();
            const next = Math.max(Math.min(minZoom, fit), Math.min(maxZoom, zoom * (e.evt.deltaY < 0 ? 1.2 : 1 / 1.2)));
            changeZoom(next, anchor ?? undefined);
          }}
          onClick={canvasClick}
          onTap={canvasClick}
        >
          <Layer>
            <KImage image={image} width={imageWidth} height={imageHeight} listening={false} />
            {rings.map((ring) => {
              const selected = !onSelect || ring.id === selectedRing;
              const points = dragging?.id === ring.id ? dragging.points : ring.points;
              return (
                <Line
                  key={ring.id}
                  points={points}
                  closed
                  stroke={selected ? "#62e6c1" : "#6eb6d6"}
                  strokeWidth={(selected ? 2 : 1.5) / zoom}
                  fill={selected ? "rgba(98,230,193,.14)" : "rgba(110,182,214,.10)"}
                  hitStrokeWidth={10 / zoom}
                  listening={!!onSelect && tool === "move" && !disabled}
                  onClick={() => onSelect?.(ring.id)}
                  onTap={() => onSelect?.(ring.id)}
                />
              );
            })}
            {!disabled && (tool === "move" || tool === "delete") &&
              rings.filter((ring) => !onSelect || ring.id === selectedRing).flatMap((ring) =>
                Array.from({ length: ring.points.length / 2 }, (_, vertex) => {
                  const offset = vertex * 2;
                  const moved = (point: Point) => [
                    ...ring.points.slice(0, offset),
                    clamp(point.x, imageWidth), clamp(point.y, imageHeight),
                    ...ring.points.slice(offset + 2),
                  ];
                  const remove = () => {
                    if (tool !== "delete") return;
                    const points = [...ring.points.slice(0, offset), ...ring.points.slice(offset + 2)];
                    if (hasArea(points)) onEdit(ring.id, points);
                  };
                  return (
                    <Circle
                      key={`${ring.id}-${vertex}`}
                      x={ring.points[offset]}
                      y={ring.points[offset + 1]}
                      radius={4 / zoom}
                      fill={tool === "delete" ? "#ffca82" : "#62e6c1"}
                      stroke="#0e191c"
                      strokeWidth={1 / zoom}
                      draggable={tool === "move"}
                      dragBoundFunc={(position) => ({
                        x: pan.x + clamp((position.x - pan.x) / zoom, imageWidth) * zoom,
                        y: pan.y + clamp((position.y - pan.y) / zoom, imageHeight) * zoom,
                      })}
                      onDragMove={(e) => setDragging({ id: ring.id, points: moved(e.target.position()) })}
                      onDragEnd={(e) => {
                        const points = moved(e.target.position());
                        setDragging(undefined);
                        if (hasArea(points)) onEdit(ring.id, points);
                        else e.target.position({ x: ring.points[offset], y: ring.points[offset + 1] });
                      }}
                      onClick={remove}
                      onTap={remove}
                    />
                  );
                }),
              )}
            <Line points={draft} stroke="#ffca82" strokeWidth={2 / zoom} listening={false} />
            {Array.from({ length: draft.length / 2 }, (_, i) => (
              <Circle
                key={i}
                x={draft[i * 2]}
                y={draft[i * 2 + 1]}
                radius={(i === 0 ? 6 : 3) / zoom}
                fill="#ffca82"
                listening={false}
              />
            ))}
          </Layer>
        </Stage>
        {!image && (
          <div className="canvas-empty">
            <p role="status">{preview ? "Không tải được ảnh xem trước." : "Đang tải ảnh nguyên khung…"}</p>
          </div>
        )}
      </div>
    </>
  );
}
