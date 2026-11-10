import { useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KImage, Line, Transformer } from "react-konva";
import type Konva from "konva";
import useImage from "use-image";
import {
  url,
  type Background,
  type Placement,
  type Scene,
  type Stamp,
} from "./api";

export function ScratchSprite({
  stamp,
  placement,
  selected,
  onSelect,
  onChange,
}: {
  stamp: Stamp;
  placement: Placement;
  selected: boolean;
  onSelect: () => void;
  onChange: (p: Placement) => void;
}) {
  const [rgb] = useImage(url("/library-files/" + stamp.rgb), "anonymous");
  const [mask] = useImage(url("/library-files/" + stamp.mask), "anonymous");
  const [cut, setCut] = useState<HTMLCanvasElement>();
  const node = useRef<Konva.Image>(null),
    transformer = useRef<Konva.Transformer>(null);
  useEffect(() => {
    if (!rgb || !mask) return;
    const canvas = document.createElement("canvas");
    canvas.width = stamp.width;
    canvas.height = stamp.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(rgb, 0, 0);
    const alpha = document.createElement("canvas");
    alpha.width = stamp.width;
    alpha.height = stamp.height;
    const ax = alpha.getContext("2d")!;
    ax.drawImage(mask, 0, 0);
    const pixels = ax.getImageData(0, 0, stamp.width, stamp.height);
    for (let i = 0; i < pixels.data.length; i += 4)
      pixels.data[i + 3] = pixels.data[i];
    ax.putImageData(pixels, 0, 0);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(alpha, 0, 0);
    setCut(canvas);
  }, [rgb, mask, stamp.width, stamp.height]);
  useEffect(() => {
    if (selected && node.current) transformer.current?.nodes([node.current]);
  }, [selected]);
  return (
    <>
      <KImage
        ref={node}
        image={cut}
        x={placement.x}
        y={placement.y}
        offsetX={stamp.width / 2}
        offsetY={stamp.height / 2}
        scaleX={placement.scale * (placement.flip ? -1 : 1)}
        scaleY={placement.scale}
        rotation={placement.rotation}
        draggable
        onClick={(e) => {
          e.cancelBubble = true;
          onSelect();
        }}
        onDragEnd={(e) =>
          onChange({ ...placement, x: e.target.x(), y: e.target.y() })
        }
        onTransformEnd={() => {
          const n = node.current!;
          onChange({
            ...placement,
            x: n.x(),
            y: n.y(),
            rotation: n.rotation(),
            scale: Math.max(0.05, Math.min(8, Math.abs(n.scaleX()))),
          });
        }}
      />
      {selected && (
        <Transformer
          ref={transformer}
          keepRatio
          enabledAnchors={[
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
          ]}
          flipEnabled={false}
        />
      )}
    </>
  );
}

export function StudioCanvas({
  image,
  scene,
  stamps,
  tool,
  brushWidth,
  selected,
  onSelect,
  onChange,
  onAdd,
}: {
  image?: Background;
  scene: Scene;
  stamps: Stamp[];
  tool: "select" | "brush";
  brushWidth: number;
  selected: number;
  onSelect: (i: number) => void;
  onChange: (s: Scene) => void;
  onAdd: (stamp: Stamp, x: number, y: number) => void;
}) {
  const [bg] = useImage(image ? url(image.url) : "", "anonymous");
  const [zoom, setZoom] = useState(1),
    [pan, setPan] = useState({ x: 0, y: 0 }),
    [width, setWidth] = useState(900);
  const holder = useRef<HTMLDivElement>(null),
    drawing = useRef(false),
    draft = useRef<number[]>([]);
  const [stroke, setStroke] = useState<number[]>([]);
  const height = 560;
  useEffect(() => {
    const observer = new ResizeObserver((entries) =>
      setWidth(Math.max(280, entries[0].contentRect.width)),
    );
    if (holder.current) observer.observe(holder.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (image) {
      setZoom(
        Math.min((width - 40) / image.width, (height - 40) / image.height),
      );
      setPan({ x: 20, y: 20 });
    }
  }, [image?.id, width]);
  function point(stage: Konva.Stage) {
    const p = stage.getRelativePointerPosition();
    if (!p || !image) return;
    return {
      x: Math.max(0, Math.min(image.width, p.x)),
      y: Math.max(0, Math.min(image.height, p.y)),
    };
  }
  function finish() {
    if (!drawing.current) return;
    drawing.current = false;
    if (draft.current.length >= 4)
      onChange({
        ...scene,
        brushes: [
          ...scene.brushes,
          { points: draft.current, width: brushWidth },
        ],
      });
    draft.current = [];
    setStroke([]);
  }
  return (
    <>
      <div className="canvas-toolbar">
        <span>
          {image
            ? `${image.width} × ${image.height} px · nguyên khung`
            : "Chọn một ảnh OK để bắt đầu"}
        </span>
        <button
          onClick={() => {
            if (image) {
              setZoom(
                Math.min(
                  (width - 40) / image.width,
                  (height - 40) / image.height,
                ),
              );
              setPan({ x: 20, y: 20 });
            }
          }}
        >
          Vừa khung
        </button>
        <button
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          100%
        </button>
        <label>
          Zoom{" "}
          <input
            aria-label="Zoom canvas"
            type="range"
            min="0.02"
            max="3"
            step="0.01"
            value={zoom}
            onChange={(e) => setZoom(+e.target.value)}
          />
        </label>
        <b>{Math.round(zoom * 100)}%</b>
      </div>
      <div
        ref={holder}
        className="canvas"
        data-testid="studio-canvas"
        onMouseLeave={finish}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!image) return;
          const stamp = stamps.find(
            (s) => s.id === e.dataTransfer.getData("stamp"),
          );
          const rect = e.currentTarget.getBoundingClientRect();
          if (stamp)
            onAdd(
              stamp,
              (e.clientX - rect.left - pan.x) / zoom,
              (e.clientY - rect.top - pan.y) / zoom,
            );
        }}
      >
        <Stage
          width={width}
          height={height}
          scaleX={zoom}
          scaleY={zoom}
          x={pan.x}
          y={pan.y}
          draggable={tool === "select"}
          onDragEnd={(e) => {
            if (e.target === e.target.getStage())
              setPan({ x: e.target.x(), y: e.target.y() });
          }}
          onMouseDown={(e) => {
            if (tool !== "brush" || !image) return;
            const p = point(e.target.getStage()!);
            if (!p) return;
            drawing.current = true;
            draft.current = [p.x, p.y, p.x, p.y];
            setStroke(draft.current);
          }}
          onMouseMove={(e) => {
            if (!drawing.current) return;
            const p = point(e.target.getStage()!);
            if (p) {
              draft.current = [...draft.current, p.x, p.y];
              setStroke(draft.current);
            }
          }}
          onMouseUp={finish}
        >
          <Layer>
            <KImage image={bg} />
            {scene.placements.map((p, i) => {
              const stamp = stamps.find((s) => s.id === p.stamp_id);
              return stamp ? (
                <ScratchSprite
                  key={i}
                  stamp={stamp}
                  placement={p}
                  selected={selected === i && tool === "select"}
                  onSelect={() => onSelect(i)}
                  onChange={(next) =>
                    onChange({
                      ...scene,
                      placements: scene.placements.map((item, k) =>
                        k === i ? next : item,
                      ),
                    })
                  }
                />
              ) : null;
            })}
            {scene.brushes.map((b, i) => (
              <Line
                key={i}
                points={b.points}
                stroke="#efba71"
                opacity={0.8}
                strokeWidth={b.width}
                lineCap="round"
                lineJoin="round"
                listening={false}
              />
            ))}
            <Line
              points={stroke}
              stroke="#efba71"
              strokeWidth={brushWidth}
              lineCap="round"
              listening={false}
            />
          </Layer>
        </Stage>
        {!image && (
          <div className="canvas-empty">
            <span>＋</span>
            <h2>Nền sạch. Vết xước do bạn quyết định.</h2>
            <p>Chọn ảnh F971 OK ở cột bên trái, rồi đặt hoặc vẽ nét xước.</p>
          </div>
        )}
      </div>
    </>
  );
}
