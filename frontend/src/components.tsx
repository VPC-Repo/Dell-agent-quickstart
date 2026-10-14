import { createPortal } from "react-dom";
import { useEffect, useRef } from "react";
import { X } from "lucide-react";

export function Drawer({
  title,
  onClose,
  children,
  busy = false,
  wide = false,
  footer = null,
}: any) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    ref.current?.focus();
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Tab") {
        const els = Array.from(
          ref.current?.querySelectorAll<HTMLElement>(
            'button:not(:disabled),input:not(:disabled),textarea:not(:disabled),select:not(:disabled),[tabindex="0"]',
          ) || [],
        );
        if (!els.length) return;
        if (
          e.shiftKey &&
          (document.activeElement === els[0] ||
            document.activeElement === ref.current)
        ) {
          e.preventDefault();
          els.at(-1)?.focus();
        } else if (!e.shiftKey && document.activeElement === els.at(-1)) {
          e.preventDefault();
          els[0].focus();
        }
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, []);
  // Portal to the body: a fixed overlay inside the blurred top bar (or any transformed panel)
  // would be confined to that element and show only its header.
  return createPortal(
    <div
      className="cw-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        tabIndex={-1}
        className={"cw-drawer" + (wide ? " cw-drawer-wide" : "")}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <header>
          <h2>{title}</h2>
          <button aria-label="Close details" onClick={onClose}>
            <X size={20} />
          </button>
        </header>
        <div inert={busy} className="cw-drawer-body">
          {children}
        </div>
        {footer && (
          <div inert={busy} className="cw-drawer-foot">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
