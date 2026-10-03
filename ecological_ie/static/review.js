(function () {
  function splitLines(container) {
    for (const p of container.querySelectorAll("p, li")) {
      const breaks = [...p.querySelectorAll(".lb")].filter(b => !b.closest(".marg"));
      if (!breaks.length && !p.querySelector(".la")) continue;
      const pieces = [];
      for (let i = 0; i <= breaks.length; i++) {
        const range = document.createRange();
        if (i === 0) range.setStart(p, 0); else range.setStartAfter(breaks[i - 1]);
        if (i === breaks.length) range.setEnd(p, p.childNodes.length); else range.setEndBefore(breaks[i]);
        pieces.push(range.cloneContents());
      }
      p.replaceChildren(...pieces.map(fragment => { const line = document.createElement("span"); line.className = "ln"; line.append(fragment); return line; }));
    }
    for (const node of container.querySelectorAll(".ln, h1, h2, h3, h4")) {
      const anchor = node.querySelector(":scope > .la, :scope > * > .la");
      if (anchor && !anchor.closest(".marg")) node.dataset.l = anchor.dataset.l;
    }
  }

  function boundsOf(polygons) {
    const xs = polygons.flatMap(p => p.map(q => q[0])), ys = polygons.flatMap(p => p.map(q => q[1]));
    return [Math.min(...ys), Math.min(...xs), Math.max(...ys), Math.max(...xs)];
  }

  function setUp(section) {
    const data = JSON.parse(section.querySelector("script.page-data").textContent);
    const host = section.querySelector(".viewer");
    const content = section.querySelector(".content");
    const zoom = new Zoom(host, { src: data.src, width: data.w, height: data.h, alt: data.alt, wheel: "modifier", controls: true });
    const reveal = box => { if (zoom.touched && !zoom.visible(box)) zoom.show(box); };

    const grid = content.querySelectorAll("table.grid tbody tr[data-b]");
    for (const tr of grid) {
      const box = tr.dataset.b.split(",").map(Number);
      const failed = tr.querySelector("td.bad");
      zoom.rect("rows", box, failed ? "row-outline failed" : "row-outline");
      const hit = zoom.rect("hits", box, "hit");
      hit.addEventListener("pointerenter", () => { tr.classList.add("hl"); zoom.clear("hover"); zoom.rect("hover", box, "band"); });
      hit.addEventListener("pointerleave", () => { tr.classList.remove("hl"); zoom.clear("hover"); });
      hit.addEventListener("click", () => tr.scrollIntoView({ block: "center", behavior: "smooth" }));
      tr.addEventListener("pointerover", e => {
        zoom.clear("hover");
        zoom.rect("hover", box, "band");
        const cell = e.target.closest("td");
        const range = cell && cell.colSpan === 1 ? (data.cols || {})[tr.closest("table.grid").dataset.t]?.[cell.cellIndex] : null;
        if (range && range[1] > range[0]) zoom.rect("hover", [box[0], range[0], box[2], range[1]], "cell");
        reveal(box);
      });
      tr.addEventListener("click", () => zoom.show(box, { zoom: "width" }));
    }
    if (grid.length) content.addEventListener("pointerleave", () => zoom.clear("hover"));

    const lines = data.lines || {};
    if (Object.keys(lines).length) {
      const text = content.querySelector(".tx");
      splitLines(text);
      for (const [kind, box] of data.regions || []) zoom.rect("regions", box, `region ${kind}`, `${kind} region`);
      const target = l => text.querySelector(`[data-l="${l}"]`);
      for (const [l, polygons] of Object.entries(lines)) for (const polygon of polygons) {
        const hit = zoom.polygon("hits", polygon, "hit");
        hit.addEventListener("pointerenter", () => { zoom.clear("hover"); for (const p of polygons) zoom.polygon("hover", p, "line-hl"); target(l)?.classList.add("hl"); });
        hit.addEventListener("pointerleave", () => { zoom.clear("hover"); target(l)?.classList.remove("hl"); });
        hit.addEventListener("click", () => target(l)?.scrollIntoView({ block: "center", behavior: "smooth" }));
      }
      text.addEventListener("pointerover", e => {
        const node = e.target.closest("[data-l], .marg");
        if (!node) return;
        const ids = node.dataset.l != null ? [node.dataset.l] : [...node.querySelectorAll(".la")].map(a => a.dataset.l);
        const polygons = ids.flatMap(l => lines[l] || []);
        zoom.clear("hover");
        for (const p of polygons) zoom.polygon("hover", p, "line-hl");
        if (polygons.length) reveal(boundsOf(polygons));
      });
      text.addEventListener("click", e => {
        const node = e.target.closest("[data-l]");
        const polygons = node ? lines[node.dataset.l] || [] : [];
        if (polygons.length) zoom.show(boundsOf(polygons), { zoom: "width" });
      });
      text.addEventListener("pointerleave", () => zoom.clear("hover"));
    }

    for (const [label, box] of data.labels || []) zoom.rect("labels", box, "label", label);
  }

  const pending = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      pending.unobserve(entry.target);
      setUp(entry.target);
    }
  }, { rootMargin: "600px 0px" });
  for (const section of document.querySelectorAll("section.pg")) {
    if (section.querySelector("script.page-data")) pending.observe(section);
  }
})();
