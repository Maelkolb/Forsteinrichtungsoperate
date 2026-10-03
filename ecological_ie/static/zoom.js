(function () {
  const SVG = "http://www.w3.org/2000/svg";
  const STYLE = `
.zoom { position: relative; overflow: hidden; touch-action: none; cursor: grab; background: var(--surface-2, #f2f2ef); outline: none; }
.zoom.dragging { cursor: grabbing; }
.zoom:focus-visible { box-shadow: inset 0 0 0 2px var(--accent, #1f5a4e); }
.zoom-stage { position: absolute; left: 0; top: 0; transform-origin: 0 0; will-change: transform; }
.zoom-stage.animate { transition: transform .28s ease; }
.zoom-stage img { display: block; width: 100%; height: 100%; user-select: none; -webkit-user-drag: none; pointer-events: none; }
.zoom-overlay { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
.zoom-overlay * { vector-effect: non-scaling-stroke; }
.zoom-overlay .hit { pointer-events: all; fill: transparent; stroke: none; cursor: pointer; }
.zoom-controls { position: absolute; top: 8px; right: 8px; display: inline-flex; border: 1px solid var(--rule-2, #cbccc6); border-radius: 4px; overflow: hidden; background: var(--surface, #fff); z-index: 2; }
.zoom-controls button { border: 0; background: none; font: inherit; font-size: .82rem; padding: 2px 10px; cursor: pointer; color: var(--ink-2, #4d5358); }
.zoom-controls button + button { border-left: 1px solid var(--rule-2, #cbccc6); }
.zoom-controls button:hover { color: var(--ink, #1a1c1e); background: var(--surface-2, #f2f2ef); }
.zoom-empty { padding: 30px 24px; color: var(--muted, #868b8f); font-style: italic; }
@media (prefers-reduced-motion: reduce) { .zoom-stage.animate { transition: none; } }`;

  function injectStyle() {
    if (document.getElementById("zoom-style")) return;
    const style = document.createElement("style");
    style.id = "zoom-style";
    style.textContent = STYLE;
    document.head.append(style);
  }

  class Zoom {
    constructor(host, options = {}) {
      injectStyle();
      this.host = host;
      this.options = { wheel: "always", max: 5, controls: false, ...options };
      this.width = options.width || 1000;
      this.height = options.height || 1000;
      this.scale = 1; this.x = 0; this.y = 0; this.touched = false; this.pointers = new Map();
      host.classList.add("zoom");
      host.tabIndex = 0;
      host.setAttribute("role", "img");
      if (options.alt) host.setAttribute("aria-label", options.alt);
      this.stage = document.createElement("div");
      this.stage.className = "zoom-stage";
      this.svg = document.createElementNS(SVG, "svg");
      this.svg.setAttribute("class", "zoom-overlay");
      this.svg.setAttribute("viewBox", "0 0 1000 1000");
      this.svg.setAttribute("preserveAspectRatio", "none");
      if (options.src) {
        this.img = new Image();
        this.img.alt = options.alt || "";
        this.img.draggable = false;
        this.img.decoding = "async";
        this.img.addEventListener("load", () => {
          if (!options.width) { this.width = this.img.naturalWidth; this.height = this.img.naturalHeight; this.size(); this.fit(); }
        });
        this.img.src = options.src;
        this.stage.append(this.img);
      }
      this.stage.append(this.svg);
      host.append(this.stage);
      if (this.options.controls) this.buildControls();
      this.size();
      this.listen();
      new ResizeObserver(() => { if (!this.touched) this.fit(); else { this.clamp(); this.apply(); } }).observe(host);
      requestAnimationFrame(() => this.fit());
    }

    size() { this.stage.style.width = this.width + "px"; this.stage.style.height = this.height + "px"; }
    get fitScale() {
      const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
      return Math.min(w / this.width, h / this.height);
    }
    get relative() { return this.scale / this.fitScale; }

    buildControls() {
      const bar = document.createElement("div");
      bar.className = "zoom-controls";
      const button = (label, title, action) => {
        const b = document.createElement("button");
        b.type = "button"; b.textContent = label; b.title = title;
        b.addEventListener("click", e => { e.stopPropagation(); action(); });
        b.addEventListener("pointerdown", e => e.stopPropagation());
        bar.append(b);
      };
      button("−", "Zoom out", () => this.zoomBy(1 / 1.5));
      button("Fit", "Whole page", () => this.fit(true));
      button("+", "Zoom in", () => this.zoomBy(1.5));
      this.host.append(bar);
    }

    apply() {
      this.stage.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.scale})`;
      this.host.dispatchEvent(new CustomEvent("zoomchange", { detail: { relative: this.relative } }));
    }

    clamp() {
      const w = this.host.clientWidth, h = this.host.clientHeight;
      const sw = this.width * this.scale, sh = this.height * this.scale;
      this.x = sw <= w ? (w - sw) / 2 : Math.min(0, Math.max(w - sw, this.x));
      this.y = sh <= h ? (h - sh) / 2 : Math.min(0, Math.max(h - sh, this.y));
    }

    fit(animate) {
      this.scale = this.fitScale; this.touched = false;
      this.clamp(); this.animate(animate); this.apply();
    }

    animate(on) {
      if (!on) return;
      this.stage.classList.add("animate");
      clearTimeout(this.animTimer);
      this.animTimer = setTimeout(() => this.stage.classList.remove("animate"), 320);
    }

    zoomAt(factor, clientX, clientY, animate) {
      const r = this.host.getBoundingClientRect();
      const px = clientX - r.left, py = clientY - r.top;
      const next = Math.max(this.fitScale * 0.9, Math.min(this.options.max, this.scale * factor));
      const ix = (px - this.x) / this.scale, iy = (py - this.y) / this.scale;
      this.scale = next; this.x = px - ix * next; this.y = py - iy * next;
      this.touched = true; this.clamp(); this.animate(animate); this.apply();
    }

    zoomBy(factor) {
      const r = this.host.getBoundingClientRect();
      this.zoomAt(factor, r.left + r.width / 2, r.top + r.height / 2, true);
    }

    show(box, how = {}) {
      if (!box || box.length !== 4) return;
      const [y0, x0, y1, x1] = box.map(v => v / 1000);
      const w = this.host.clientWidth, h = this.host.clientHeight;
      const bw = (x1 - x0) * this.width, bh = (y1 - y0) * this.height;
      if (how.zoom === "width") this.scale = Math.min(this.options.max, Math.max(this.fitScale, 0.94 * w / Math.max(bw, 1)));
      else if (how.zoom === "box") this.scale = Math.min(this.options.max, Math.max(this.fitScale, Math.min(0.8 * w / Math.max(bw, 1), 0.6 * h / Math.max(bh, 1))));
      const cx = (x0 + x1) / 2 * this.width * this.scale, cy = (y0 + y1) / 2 * this.height * this.scale;
      if (how.zoom || how.center !== false) { this.x = w / 2 - cx; this.y = h / 2 - cy; }
      this.touched = this.scale > this.fitScale * 1.01 || this.touched;
      this.clamp(); this.animate(true); this.apply();
    }

    visible(box) {
      const [y0, x0, y1, x1] = box.map(v => v / 1000);
      const w = this.host.clientWidth, h = this.host.clientHeight;
      const top = y0 * this.height * this.scale + this.y, bottom = y1 * this.height * this.scale + this.y;
      const left = x0 * this.width * this.scale + this.x, right = x1 * this.width * this.scale + this.x;
      return top >= 0 && bottom <= h && left >= -0.25 * (right - left) && right <= w + 0.25 * (right - left);
    }

    layer(name) {
      let g = this.svg.querySelector(`g[data-layer="${name}"]`);
      if (!g) { g = document.createElementNS(SVG, "g"); g.dataset.layer = name; this.svg.append(g); }
      return g;
    }
    clear(name) { this.layer(name).replaceChildren(); }
    shape(name, tag, attrs, title) {
      const node = document.createElementNS(SVG, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      if (title) { const t = document.createElementNS(SVG, "title"); t.textContent = title; node.append(t); }
      this.layer(name).append(node);
      return node;
    }
    rect(name, box, cls, title) {
      const [y0, x0, y1, x1] = box;
      return this.shape(name, "rect", { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0), class: cls || "" }, title);
    }
    polygon(name, points, cls, title) {
      return this.shape(name, "polygon", { points: points.map(p => p.join(",")).join(" "), class: cls || "" }, title);
    }

    listen() {
      const host = this.host;
      host.addEventListener("wheel", e => {
        if (this.options.wheel !== "always" && !e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        this.zoomAt(Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)), e.clientX, e.clientY);
      }, { passive: false });
      host.addEventListener("dblclick", e => { e.preventDefault(); this.zoomAt(e.shiftKey ? 1 / 2 : 2, e.clientX, e.clientY, true); });
      host.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        this.moved = 0;
        if (this.pointers.size === 1) this.drag = { x: e.clientX, y: e.clientY, ox: this.x, oy: this.y };
        if (this.pointers.size === 2) {
          const [a, b] = [...this.pointers.values()];
          this.pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), scale: this.scale };
        }
      });
      host.addEventListener("pointermove", e => {
        if (!this.pointers.has(e.pointerId)) return;
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (this.pointers.size === 2 && this.pinch) {
          const [a, b] = [...this.pointers.values()];
          const factor = Math.hypot(a.x - b.x, a.y - b.y) / this.pinch.d * this.pinch.scale / this.scale;
          this.zoomAt(factor, (a.x + b.x) / 2, (a.y + b.y) / 2);
          this.moved = 99;
          return;
        }
        if (!this.drag) return;
        const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
        this.moved = Math.max(this.moved, Math.abs(dx) + Math.abs(dy));
        if (this.moved > 4) {
          if (!host.hasPointerCapture(e.pointerId)) host.setPointerCapture(e.pointerId);
          host.classList.add("dragging");
          this.x = this.drag.ox + dx; this.y = this.drag.oy + dy; this.touched = true;
          this.clamp(); this.apply();
        }
      });
      const end = e => {
        this.pointers.delete(e.pointerId);
        if (this.pointers.size < 2) this.pinch = null;
        if (!this.pointers.size) { this.drag = null; host.classList.remove("dragging"); }
      };
      host.addEventListener("pointerup", end);
      host.addEventListener("pointercancel", end);
      host.addEventListener("click", e => { if (this.moved > 4) { e.stopPropagation(); e.preventDefault(); } }, true);
      host.addEventListener("keydown", e => {
        const step = 60;
        const actions = { "+": () => this.zoomBy(1.4), "=": () => this.zoomBy(1.4), "-": () => this.zoomBy(1 / 1.4), "0": () => this.fit(true),
          ArrowUp: () => { this.y += step; }, ArrowDown: () => { this.y -= step; }, ArrowLeft: () => { this.x += step; }, ArrowRight: () => { this.x -= step; } };
        const action = actions[e.key];
        if (!action) return;
        e.preventDefault(); e.stopPropagation();
        action(); this.touched = true; this.clamp(); this.apply();
      });
    }
  }
  window.Zoom = Zoom;
})();
