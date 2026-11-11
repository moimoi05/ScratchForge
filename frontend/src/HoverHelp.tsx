import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const controls = "button, input, select, textarea, a[href], [data-tooltip]";

function hintFor(element: Element) {
  // The existing descriptions also explain the prerequisite to screen readers.
  if (element.matches(":disabled, [aria-disabled='true']")) {
    const ids = element.getAttribute("aria-describedby")?.split(/\s+/) ?? [];
    const description = ids.map(id => document.getElementById(id)?.textContent?.trim()).filter(Boolean).join(" ");
    if (description) return description;
  }
  return element.closest("[data-tooltip]")?.getAttribute("data-tooltip")?.trim();
}

export function HoverHelp() {
  const [hint, setHint] = useState<{ anchor: Element; text: string }>();
  const bubble = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let anchor: Element | null = null;
    let dismissed: Element | null = null;
    let text: string | undefined;
    let pointer: { x: number; y: number } | null = null;
    function show(target: Element | null, reposition = false) {
      const next = target?.closest(controls) ?? null;
      if (dismissed && next !== dismissed) dismissed = null;
      const message = next && next !== dismissed ? hintFor(next) : undefined;
      if (!reposition && anchor === next && text === message) return;
      anchor = next;
      text = message;
      setHint(next && message ? { anchor: next, text: message } : undefined);
    }
    // Hit-test at document level so a native disabled control can still show help.
    const move = (event: PointerEvent) => {
      if (event.pointerType === "mouse" || event.pointerType === "pen") {
        pointer = { x: event.clientX, y: event.clientY };
        show(document.elementFromPoint(event.clientX, event.clientY));
      }
    };
    const focus = (event: FocusEvent) => {
      pointer = null;
      show(event.target instanceof Element ? event.target : null);
    };
    const clear = () => show(null);
    const leave = (event: PointerEvent) => { if (!event.relatedTarget) { pointer = null; clear(); } };
    const reposition = () => {
      const target = pointer ? document.elementFromPoint(pointer.x, pointer.y) : anchor;
      show(target?.isConnected ? target : null, true);
    };
    const dismiss = () => { dismissed = anchor; text = undefined; setHint(undefined); };
    const key = (event: KeyboardEvent) => { if (event.key === "Escape") dismiss(); };
    document.addEventListener("pointermove", move, true);
    document.addEventListener("pointerout", leave, true);
    document.addEventListener("pointerdown", dismiss, true);
    document.addEventListener("focusin", focus);
    document.addEventListener("focusout", clear);
    document.addEventListener("keydown", key);
    document.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    return () => {
      document.removeEventListener("pointermove", move, true);
      document.removeEventListener("pointerout", leave, true);
      document.removeEventListener("pointerdown", dismiss, true);
      document.removeEventListener("focusin", focus);
      document.removeEventListener("focusout", clear);
      document.removeEventListener("keydown", key);
      document.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
    };
  }, []);

  useLayoutEffect(() => {
    if (!hint || !bubble.current) return;
    const anchor = hint.anchor.getBoundingClientRect();
    const { offsetWidth: width, offsetHeight: height } = bubble.current;
    const gap = 8;
    const left = Math.max(gap, Math.min(anchor.left + (anchor.width - width) / 2, window.innerWidth - width - gap));
    const top = anchor.bottom + height + 2 * gap <= window.innerHeight
      ? anchor.bottom + gap : Math.max(gap, anchor.top - height - gap);
    bubble.current.style.left = `${left}px`;
    bubble.current.style.top = `${top}px`;
  }, [hint]);

  return hint ? createPortal(<div className="hover-help" role="tooltip" ref={bubble}>{hint.text}</div>, document.body) : null;
}
