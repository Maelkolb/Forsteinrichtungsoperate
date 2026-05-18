"""
viewer_editor_addon.py — Shared CSS/JS/HTML assets that turn a
Forsteinrichtungsoperate viewer HTML file into an interactive
transcription editor.

This module is the SINGLE SOURCE OF TRUTH for the editor features:

    * Click-to-edit transcriptions (contenteditable)
    * Per-page "Mark done" with automatic CER / WER computation
    * Per-page "Redo" flag
    * Progress counter + per-page metrics chip
    * Auto-save to browser localStorage every 1.5 s
    * Restore-banner on reload, with last-edited-page resume
    * Portable JSON export / import

Both scripts/upgrade_viewer.py (retrofit existing HTMLs) and
scripts/build_viewer.py (new HTMLs going forward) pull from here,
so the editor behaviour stays in lockstep across both flows.

Reference for the CER / WER algorithm and JSON shape: ported from
the HistOrniGraph project (https://github.com/Maelkolb/HistOrniGraph,
``Create_GUIs.py``) and adapted to the scrolling, no-pagination
layout of the Forsteinrichtungsoperate viewer.
"""

from __future__ import annotations

# Sentinel comment we drop into every upgraded HTML so the same file
# isn't double-upgraded.  Bump the version suffix when the addon
# changes in a backward-incompatible way.
ADDON_MARKER = "<!-- forsteinrichtungsoperate-editor-addon-v1 -->"


# ───────────────────────────────────────────────────────────
# CSS additions.  All selectors are scoped tightly so we don't
# accidentally restyle pre-existing parts of the viewer.
# ───────────────────────────────────────────────────────────
CSS_ADDON = r"""
/* === EDITOR ADDON CSS === */

/* Toolbar buttons in the top header */
header.top .editor-btn {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 4px 12px;
  font: inherit;
  font-size: 13px;
  color: var(--fg);
  cursor: pointer;
  white-space: nowrap;
}
header.top .editor-btn:hover { background: var(--bg-sidebar); }
header.top .editor-btn.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
header.top .editor-btn.primary {
  border-color: var(--accent);
  color: var(--accent);
}
header.top .editor-btn.primary:hover {
  background: rgba(139, 58, 47, .08);
}

/* Progress + save indicator */
header.top .editor-progress {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--fg-muted);
  white-space: nowrap;
}
header.top .editor-progress .bar {
  width: 60px; height: 5px;
  background: var(--bg-sidebar);
  border-radius: 3px;
  overflow: hidden;
}
header.top .editor-progress .bar-fill {
  height: 100%;
  background: #3a7d44;
  border-radius: 3px;
  transition: width .3s ease;
}
header.top .editor-save-status {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--fg-muted);
}

/* Editable transcription — only when edit mode is on. */
body.editor-edit-mode .transcription {
  cursor: text;
}
body.editor-edit-mode .transcription:focus-within,
.transcription[contenteditable="true"]:focus {
  background: #fffbe6;
  outline: 2px solid var(--accent);
  outline-offset: -1px;
  border-radius: var(--radius);
}
.transcription[contenteditable="true"] {
  /* Subtle hint that the box is editable, even before it's focussed. */
  background: rgba(255, 251, 230, .55);
  box-shadow: inset 0 0 0 1px rgba(139, 58, 47, .25);
  border-radius: var(--radius);
  min-height: 60px;
}

/* Article header: lay it out so the action buttons sit on the right. */
article.page > header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}
article.page > header h2 { flex-shrink: 0; }
article.page > header .page-actions {
  margin-left: auto;
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}

/* Done / redo buttons */
article.page .done-btn,
article.page .redo-btn {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  padding: 4px 10px;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--fg);
  border-radius: var(--radius);
  cursor: pointer;
  white-space: nowrap;
}
article.page .done-btn:hover {
  border-color: #3a7d44; color: #3a7d44; background: rgba(58, 125, 68, .05);
}
article.page .done-btn.is-done {
  background: #3a7d44; color: #fff; border-color: #3a7d44; font-weight: 600;
}
article.page .redo-btn:hover {
  border-color: #c07d16; color: #c07d16; background: rgba(192, 125, 22, .05);
}
article.page .redo-btn.is-redo {
  background: #c07d16; color: #fff; border-color: #c07d16; font-weight: 600;
}

/* Metrics chip */
article.page .editor-metrics {
  display: none;
  align-items: center;
  gap: 4px 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--fg-muted);
  flex-wrap: wrap;
}
article.page .editor-metrics.visible { display: inline-flex; }
article.page .editor-metrics .m-pair {
  display: inline-flex; gap: 3px; align-items: baseline;
}
article.page .editor-metrics .m-label {
  color: var(--fg-muted);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: .5px;
}
article.page .editor-metrics .m-value { font-weight: 600; }
article.page .editor-metrics .m-value.excellent { color: #3a7d44; }
article.page .editor-metrics .m-value.good      { color: #5a9e3a; }
article.page .editor-metrics .m-value.moderate  { color: #c07d16; }
article.page .editor-metrics .m-value.poor      { color: #c04030; }

/* Dirty indicator (the page has been edited but not yet marked done) */
article.page.is-dirty > header h2::after {
  content: " • modified";
  color: var(--accent);
  font-size: 12px;
  font-weight: 400;
}

/* Restore banner — sits just below the top header */
#editor-restore-banner {
  position: sticky;
  top: 49px;
  z-index: 4;
  background: rgba(192, 125, 22, .14);
  border-bottom: 1px solid rgba(192, 125, 22, .35);
  padding: 8px 24px;
  font-size: 13px;
  color: var(--fg);
  display: flex;
  align-items: center;
  gap: 12px;
}
#editor-restore-banner.hidden { display: none; }
#editor-restore-banner .banner-text { flex: 1; }
#editor-restore-banner .restore-btn,
#editor-restore-banner .dismiss-btn {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  padding: 4px 12px;
  border-radius: var(--radius);
  cursor: pointer;
}
#editor-restore-banner .restore-btn {
  border: 1px solid #c07d16;
  background: rgba(192, 125, 22, .2);
  color: #c07d16;
  font-weight: 500;
}
#editor-restore-banner .restore-btn:hover { background: rgba(192, 125, 22, .3); }
#editor-restore-banner .dismiss-btn {
  border: 1px solid var(--border);
  background: transparent;
  color: var(--fg-muted);
}
#editor-restore-banner .dismiss-btn:hover { color: var(--fg); border-color: var(--fg-muted); }

/* Toast */
#editor-toast {
  position: fixed;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%) translateY(0);
  background: var(--fg);
  color: #fff;
  padding: 9px 18px;
  border-radius: var(--radius);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  z-index: 9999;
  opacity: 0;
  pointer-events: none;
  transition: opacity .2s, transform .2s;
  white-space: nowrap;
  box-shadow: 0 4px 12px rgba(0,0,0,.2);
}
#editor-toast.show {
  opacity: 1;
  transform: translateX(-50%) translateY(-4px);
}

/* Aggregate metrics summary in the sidebar */
aside.sidebar .editor-summary {
  margin: 14px;
  padding: 10px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: 12px;
  color: var(--fg-muted);
  display: none;
}
aside.sidebar .editor-summary.visible { display: block; }
aside.sidebar .editor-summary h3 {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--fg-muted);
  margin: 0 0 6px;
}
aside.sidebar .editor-summary .stat {
  display: flex; justify-content: space-between; gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  padding: 2px 0;
}
aside.sidebar .editor-summary .stat .v { color: var(--fg); font-weight: 600; }

/* Sidebar items: prefix when done / redo */
ol.page-list li.is-done a .num { color: #3a7d44; }
ol.page-list li.is-done a .num::before { content: "✓ "; }
ol.page-list li.is-redo a .num { color: #c07d16; }
ol.page-list li.is-redo a .num::before { content: "↻ "; }

/* Print: hide all editor chrome */
@media print {
  header.top .editor-btn,
  header.top .editor-progress,
  header.top .editor-save-status,
  article.page .page-actions,
  article.page .editor-metrics,
  aside.sidebar .editor-summary,
  #editor-restore-banner,
  #editor-toast { display: none !important; }
  .transcription[contenteditable="true"] {
    background: transparent !important;
    box-shadow: none !important;
  }
}
/* === END EDITOR ADDON CSS === */
"""


# ───────────────────────────────────────────────────────────
# HTML widgets
# ───────────────────────────────────────────────────────────

# Inserted just before </header> of the page's top header.
TOOLBAR_HTML = r"""
<button id="editor-edit-toggle" class="editor-btn" type="button" title="Click on any transcription to edit (E)">✎ Edit mode</button>
<button id="editor-export-btn" class="editor-btn primary" type="button" title="Export edits, metrics and progress as a portable JSON file">↓ Export JSON</button>
<button id="editor-import-btn" class="editor-btn" type="button" title="Import a previously exported JSON to restore edits and progress">↑ Import JSON</button>
<input type="file" id="editor-import-file" accept=".json" style="display:none">
<span class="editor-progress" title="Pages marked as done">
  <span class="bar"><span class="bar-fill" id="editor-progress-fill" style="width:0%"></span></span>
  <span id="editor-progress-text">0 / 0 done</span>
</span>
<span class="editor-save-status" id="editor-save-status"></span>
"""

# Per-article widgets (injected into each <article><header>).
# {page_id} is substituted at injection time.
ARTICLE_WIDGETS_TEMPLATE = r"""<span class="editor-metrics" data-pageid="{page_id_attr}">
  <span class="m-pair"><span class="m-label">CER</span><span class="m-value" data-metric="cer">—</span></span>
  <span class="m-pair"><span class="m-label">WER</span><span class="m-value" data-metric="wer">—</span></span>
  <span class="m-pair"><span class="m-label">Edits</span><span class="m-value" data-metric="edits" style="color:var(--fg)">—</span></span>
</span>
<span class="page-actions">
  <button class="redo-btn" type="button" data-pageid="{page_id_attr}" title="Flag this page for reprocessing (Shift+D)">↻ Redo</button>
  <button class="done-btn" type="button" data-pageid="{page_id_attr}" title="Mark this page as validated; computes CER & WER vs the original transcription (D)">☐ Mark done</button>
</span>"""

# Restore banner + toast, inserted right after </header> of the top bar.
RESTORE_AND_TOAST_HTML = r"""
<div id="editor-restore-banner" class="hidden">
  <span class="banner-text" id="editor-restore-text">Saved session found.</span>
  <button class="restore-btn" id="editor-restore-btn" type="button">Restore session</button>
  <button class="dismiss-btn" id="editor-restore-dismiss" type="button">Dismiss</button>
</div>
<div id="editor-toast" role="status" aria-live="polite"></div>
"""

# Sidebar summary block — slotted in just above the page list.
SIDEBAR_SUMMARY_HTML = r"""
<div class="editor-summary" id="editor-summary">
  <h3>Validation summary</h3>
  <div class="stat"><span>Pages done</span><span class="v" id="sum-done">0 / 0</span></div>
  <div class="stat"><span>Pages flagged redo</span><span class="v" id="sum-redo">0</span></div>
  <div class="stat" title="Micro-averaged Character Error Rate across all marked-done pages">
    <span>Micro CER</span><span class="v" id="sum-cer">—</span>
  </div>
  <div class="stat" title="Micro-averaged Word Error Rate across all marked-done pages">
    <span>Micro WER</span><span class="v" id="sum-wer">—</span>
  </div>
</div>
"""


# ───────────────────────────────────────────────────────────
# JavaScript
# ───────────────────────────────────────────────────────────
# Wrapped in its own IIFE so it doesn't collide with the existing
# viewer JS (sidebar filter, regions toggle, lightbox).
JS_ADDON = r"""
// === FORSTEINRICHTUNGSOPERATE EDITOR ADDON ===
(function() {
  'use strict';

  // ─── Session key (stable across reloads of the same viewer) ───
  function _sessionKey() {
    var t = (document.title || 'forst_viewer').replace(/\s+/g, '_');
    var arts = document.querySelectorAll('article.page');
    var n = arts.length;
    var firstId = arts[0] ? arts[0].id : 'none';
    return 'forst_editor::' + t + '::' + n + '::' + firstId;
  }
  var STORAGE_KEY = _sessionKey();

  // ─── State ───
  var pages = [];   // array of page objects, see init()
  var pageById = {};
  var editMode = false;
  var lastViewedPageId = null;
  var lastEditedPageId = null;
  var autoSaveTimer = null;
  var toastTimer = null;
  var initialized = false;

  // ─── Levenshtein / CER / WER ───
  // Ported verbatim from HistOrniGraph/Create_GUIs.py (the same
  // algorithm Anthropic-recommended ASR/HTR pipelines use):
  //   CER = edit-distance(chars) / max(1, len(reference_chars))
  //   WER = edit-distance(words) / max(1, len(reference_words))
  // Reference = the AI-generated transcription as it was the moment
  // the page was first loaded in this browser; hypothesis = the
  // human-corrected version.
  function levenshtein(a, b) {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    if (a.length > b.length) { var tmp = a; a = b; b = tmp; }
    var prev = new Array(a.length + 1);
    var curr = new Array(a.length + 1);
    for (var i = 0; i <= a.length; i++) prev[i] = i;
    for (var j = 1; j <= b.length; j++) {
      curr[0] = j;
      for (var k = 1; k <= a.length; k++) {
        if (a[k - 1] === b[j - 1]) curr[k] = prev[k - 1];
        else curr[k] = 1 + Math.min(prev[k - 1], prev[k], curr[k - 1]);
      }
      var s = prev; prev = curr; curr = s;
    }
    return prev[a.length];
  }
  function computeCER(ref, hyp) {
    var rc = Array.from(ref);
    var hc = Array.from(hyp);
    var d = levenshtein(rc, hc);
    return { cer: d / Math.max(rc.length, 1), distance: d, refLen: rc.length };
  }
  function computeWER(ref, hyp) {
    var rw = ref.split(/\s+/).filter(function(w) { return w.length > 0; });
    var hw = hyp.split(/\s+/).filter(function(w) { return w.length > 0; });
    var d = levenshtein(rw, hw);
    return { wer: d / Math.max(rw.length, 1), distance: d, refLen: rw.length };
  }
  function rateTier(r) {
    if (r <= 0.02) return 'excellent';
    if (r <= 0.08) return 'good';
    if (r <= 0.20) return 'moderate';
    return 'poor';
  }
  function formatRate(r) { return (r * 100).toFixed(2) + '%'; }

  // ─── Text extraction (normalised whitespace) ───
  // We use innerText (not textContent) because innerText collapses
  // hidden elements and respects rendering — i.e. it gives us what
  // the user actually SEES, which is what they actually corrected.
  function pageText(p) {
    return (p.transEl.innerText || '').replace(/\s+/g, ' ').trim();
  }

  // ─── Init ───
  function init() {
    if (initialized) return;
    initialized = true;

    document.querySelectorAll('article.page').forEach(function(a) {
      var trans = a.querySelector('.transcription');
      if (!trans || !a.id) return;
      var p = {
        id: a.id,
        articleEl: a,
        transEl: trans,
        // Cache the AI-original at first load so CER/WER is always
        // measured against the same baseline, even if the user edits.
        originalText: '',
        originalHTML: trans.innerHTML,
        isDone: false,
        isRedo: false,
        metrics: null,
        dirty: false
      };
      pages.push(p);
      pageById[p.id] = p;
    });
    // originalText must be captured AFTER push because we use the
    // shared pageText() helper which normalises whitespace.
    pages.forEach(function(p) { p.originalText = pageText(p); });

    bindControls();
    setupScrollTracking();
    setupEditListeners();
    updateProgress();
    updateSidebarSummary();
    checkRestoreSession();
  }

  // ─── Edit-mode toggle ───
  function setEditMode(on) {
    editMode = !!on;
    document.body.classList.toggle('editor-edit-mode', editMode);
    var btn = document.getElementById('editor-edit-toggle');
    btn.classList.toggle('active', editMode);
    btn.textContent = editMode ? '✎ Exit edit mode' : '✎ Edit mode';
    pages.forEach(function(p) {
      p.transEl.setAttribute('contenteditable', editMode ? 'true' : 'false');
      p.transEl.setAttribute('spellcheck', 'false');
    });
    if (editMode) showToast('Edit mode ON — click any transcription to edit');
  }

  // ─── Mark done / undone ───
  function toggleDone(pageId) {
    var p = pageById[pageId];
    if (!p) return;
    if (p.isDone) {
      p.isDone = false;
      p.metrics = null;
      hideMetrics(p.id);
      showToast('Page “' + p.id + '” unmarked');
    } else {
      var current = pageText(p);
      var cer = computeCER(p.originalText, current);
      var wer = computeWER(p.originalText, current);
      p.metrics = {
        cer: cer.cer, wer: wer.wer,
        charEdits: cer.distance, wordEdits: wer.distance,
        refChars: cer.refLen, refWords: wer.refLen,
        doneAt: new Date().toISOString()
      };
      p.isDone = true;
      showMetrics(p.id, p.metrics);
      showToast('Done — CER ' + formatRate(cer.cer) + ', WER ' + formatRate(wer.wer));
    }
    updateDoneButton(p);
    updateProgress();
    updateSidebarItem(p);
    updateSidebarSummary();
    scheduleAutoSave();
  }

  // ─── Mark redo ───
  function toggleRedo(pageId) {
    var p = pageById[pageId];
    if (!p) return;
    p.isRedo = !p.isRedo;
    updateRedoButton(p);
    updateSidebarItem(p);
    updateProgress();
    showToast(p.isRedo ? 'Page “' + p.id + '” flagged for redo' : 'Redo flag removed');
    scheduleAutoSave();
  }

  // ─── UI updates ───
  function showMetrics(pageId, m) {
    var el = document.querySelector('.editor-metrics[data-pageid="' + cssAttrEscape(pageId) + '"]');
    if (!el) return;
    var cer = el.querySelector('[data-metric="cer"]');
    var wer = el.querySelector('[data-metric="wer"]');
    var ed  = el.querySelector('[data-metric="edits"]');
    cer.textContent = formatRate(m.cer);
    cer.className = 'm-value ' + rateTier(m.cer);
    wer.textContent = formatRate(m.wer);
    wer.className = 'm-value ' + rateTier(m.wer);
    ed.textContent = m.charEdits + ' chars / ' + m.wordEdits + ' words';
    el.classList.add('visible');
  }
  function hideMetrics(pageId) {
    var el = document.querySelector('.editor-metrics[data-pageid="' + cssAttrEscape(pageId) + '"]');
    if (el) el.classList.remove('visible');
  }
  function updateDoneButton(p) {
    var btn = document.querySelector('.done-btn[data-pageid="' + cssAttrEscape(p.id) + '"]');
    if (!btn) return;
    btn.classList.toggle('is-done', p.isDone);
    btn.textContent = p.isDone ? '✓ Done' : '☐ Mark done';
  }
  function updateRedoButton(p) {
    var btn = document.querySelector('.redo-btn[data-pageid="' + cssAttrEscape(p.id) + '"]');
    if (!btn) return;
    btn.classList.toggle('is-redo', p.isRedo);
    btn.textContent = p.isRedo ? '↻ Redo flagged' : '↻ Redo';
  }
  function updateSidebarItem(p) {
    // Find the sidebar <li> by matching its <a href="#pageId">
    var a = document.querySelector('ol.page-list li a[href="#' + cssAttrEscape(p.id) + '"]');
    if (!a) return;
    var li = a.parentElement;
    li.classList.toggle('is-done', p.isDone);
    li.classList.toggle('is-redo', p.isRedo);
  }
  function updateProgress() {
    var done = pages.filter(function(p) { return p.isDone; }).length;
    var redo = pages.filter(function(p) { return p.isRedo; }).length;
    var pct = pages.length ? (done / pages.length * 100) : 0;
    var fill = document.getElementById('editor-progress-fill');
    var txt = document.getElementById('editor-progress-text');
    if (fill) fill.style.width = pct.toFixed(1) + '%';
    if (txt) {
      var s = done + ' / ' + pages.length + ' done';
      if (redo > 0) s += ', ' + redo + ' redo';
      txt.textContent = s;
    }
  }
  function updateSidebarSummary() {
    var sum = document.getElementById('editor-summary');
    if (!sum) return;
    var done = pages.filter(function(p) { return p.isDone; });
    var redo = pages.filter(function(p) { return p.isRedo; }).length;
    if (done.length === 0 && redo === 0) {
      sum.classList.remove('visible');
      return;
    }
    sum.classList.add('visible');
    document.getElementById('sum-done').textContent = done.length + ' / ' + pages.length;
    document.getElementById('sum-redo').textContent = String(redo);
    if (done.length > 0) {
      var tcd = 0, tcr = 0, twd = 0, twr = 0;
      done.forEach(function(p) {
        if (!p.metrics) return;
        tcd += p.metrics.charEdits || 0; tcr += p.metrics.refChars || 0;
        twd += p.metrics.wordEdits || 0; twr += p.metrics.refWords || 0;
      });
      var microCER = tcr > 0 ? tcd / tcr : 0;
      var microWER = twr > 0 ? twd / twr : 0;
      var cerEl = document.getElementById('sum-cer');
      var werEl = document.getElementById('sum-wer');
      cerEl.textContent = formatRate(microCER); cerEl.className = 'v ' + rateTier(microCER);
      werEl.textContent = formatRate(microWER); werEl.className = 'v ' + rateTier(microWER);
    } else {
      document.getElementById('sum-cer').textContent = '—';
      document.getElementById('sum-wer').textContent = '—';
    }
  }

  // ─── Track scroll position (which page is in view) ───
  function setupScrollTracking() {
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(e) {
        if (e.isIntersecting && e.target.id) {
          lastViewedPageId = e.target.id;
        }
      });
    }, { rootMargin: '-25% 0px -65% 0px' });
    pages.forEach(function(p) { observer.observe(p.articleEl); });
  }

  // ─── Track edits to transcriptions (debounced autosave) ───
  function setupEditListeners() {
    pages.forEach(function(p) {
      p.transEl.addEventListener('input', function() {
        var current = pageText(p);
        var wasDirty = p.dirty;
        p.dirty = current !== p.originalText;
        p.articleEl.classList.toggle('is-dirty', p.dirty);
        if (p.dirty) lastEditedPageId = p.id;
        // If a previously-done page is edited, recompute the metrics
        // live (without un-marking it).
        if (p.isDone) {
          var cer = computeCER(p.originalText, current);
          var wer = computeWER(p.originalText, current);
          p.metrics = {
            cer: cer.cer, wer: wer.wer,
            charEdits: cer.distance, wordEdits: wer.distance,
            refChars: cer.refLen, refWords: wer.refLen,
            doneAt: p.metrics ? p.metrics.doneAt : new Date().toISOString(),
            updatedAt: new Date().toISOString()
          };
          showMetrics(p.id, p.metrics);
          updateSidebarSummary();
        }
        scheduleAutoSave();
      });
    });
  }

  // ─── Autosave to localStorage ───
  function scheduleAutoSave() {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(doAutoSave, 1500);
  }
  function doAutoSave() {
    try {
      var session = buildSessionObject();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
      var now = new Date();
      var hh = String(now.getHours()).padStart(2, '0');
      var mm = String(now.getMinutes()).padStart(2, '0');
      var ss = String(now.getSeconds()).padStart(2, '0');
      var el = document.getElementById('editor-save-status');
      if (el) el.textContent = 'Auto-saved ' + hh + ':' + mm + ':' + ss;
    } catch (e) {
      var el = document.getElementById('editor-save-status');
      if (el) el.textContent = 'Save failed: ' + e.message;
    }
  }

  // ─── Session serialisation ───
  function buildSessionObject() {
    var session = {
      _meta: {
        format: 'forsteinrichtungsoperate_editor_v1',
        documentTitle: document.title,
        pageCount: pages.length,
        firstPageId: pages[0] ? pages[0].id : null,
        lastViewedPageId: lastViewedPageId,
        lastEditedPageId: lastEditedPageId,
        savedAt: new Date().toISOString()
      },
      pages: {}
    };
    pages.forEach(function(p) {
      var current = pageText(p);
      var currentHTML = p.transEl.innerHTML;
      var modified = currentHTML !== p.originalHTML;
      if (modified || p.isDone || p.isRedo) {
        session.pages[p.id] = {
          originalText: p.originalText,
          originalHTML: p.originalHTML,
          editedHTML: modified ? currentHTML : null,
          editedText: modified ? current : null,
          modified: modified,
          done: p.isDone,
          metrics: p.metrics,
          redo: p.isRedo
        };
      }
    });
    // Aggregate (handy for at-a-glance review of the exported JSON)
    var doneList = Object.values(session.pages).filter(function(x) { return x.done; });
    if (doneList.length > 0) {
      var tcd = 0, tcr = 0, twd = 0, twr = 0;
      doneList.forEach(function(x) {
        if (!x.metrics) return;
        tcd += x.metrics.charEdits || 0; tcr += x.metrics.refChars || 0;
        twd += x.metrics.wordEdits || 0; twr += x.metrics.refWords || 0;
      });
      session._meta.aggregate = {
        donePages: doneList.length,
        redoPages: Object.values(session.pages).filter(function(x) { return x.redo; }).length,
        totalPages: pages.length,
        microCER: tcr > 0 ? tcd / tcr : 0,
        microWER: twr > 0 ? twd / twr : 0,
        totalCharEdits: tcd,
        totalWordEdits: twd
      };
    }
    return session;
  }

  // ─── Restore on reload ───
  function checkRestoreSession() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var session = JSON.parse(raw);
      if (!session || !session._meta || !session.pages) return;
      if (Object.keys(session.pages).length === 0) return;

      var meta = session._meta;
      var pgs = Object.values(session.pages);
      var editedCount = pgs.filter(function(x) { return x.modified; }).length;
      var doneCount   = pgs.filter(function(x) { return x.done; }).length;
      var redoCount   = pgs.filter(function(x) { return x.redo; }).length;

      var dt = meta.savedAt ? new Date(meta.savedAt) : null;
      var dateStr = '?';
      try {
        dateStr = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                + ' at ' + dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      } catch (e) {}

      var parts = [editedCount + ' edited', doneCount + ' done'];
      if (redoCount > 0) parts.push(redoCount + ' redo');
      var resumeTo = meta.lastEditedPageId || meta.lastViewedPageId;
      if (resumeTo) parts.push('last on “' + resumeTo + '”');

      var text = 'Saved session from <strong>' + dateStr + '</strong> — ' + parts.join(', ') + '.';
      document.getElementById('editor-restore-text').innerHTML = text;

      var banner = document.getElementById('editor-restore-banner');
      banner.classList.remove('hidden');
      // Stash session on the button so the click handler can find it.
      document.getElementById('editor-restore-btn')._session = session;
    } catch (e) {
      // Corrupt session — drop it silently.
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
    }
  }

  function applySession(session) {
    var applied = { edits: 0, done: 0, redo: 0 };
    var pgs = session.pages || {};
    Object.keys(pgs).forEach(function(pid) {
      var data = pgs[pid];
      var p = pageById[pid];
      if (!p) return;  // page no longer exists in this viewer
      if (data.modified && data.editedHTML != null) {
        p.transEl.innerHTML = data.editedHTML;
        p.dirty = pageText(p) !== p.originalText;
        p.articleEl.classList.toggle('is-dirty', p.dirty);
        applied.edits++;
      }
      if (data.done) {
        p.isDone = true;
        p.metrics = data.metrics || null;
        if (p.metrics) showMetrics(p.id, p.metrics);
        updateDoneButton(p);
        applied.done++;
      }
      if (data.redo) {
        p.isRedo = true;
        updateRedoButton(p);
        applied.redo++;
      }
      updateSidebarItem(p);
    });
    updateProgress();
    updateSidebarSummary();

    // Jump to where they left off
    var meta = session._meta || {};
    var resumeTo = meta.lastEditedPageId || meta.lastViewedPageId;
    if (resumeTo && pageById[resumeTo]) {
      setTimeout(function() {
        pageById[resumeTo].articleEl.scrollIntoView({ block: 'start', behavior: 'smooth' });
      }, 100);
    }

    var msg = '✓ Restored: ' + applied.edits + ' edit(s), ' + applied.done + ' done';
    if (applied.redo > 0) msg += ', ' + applied.redo + ' redo';
    if (resumeTo) msg += ' — jumped to “' + resumeTo + '”';
    showToast(msg, 3500);
    scheduleAutoSave();
  }

  // ─── JSON export / import ───
  function exportJSON() {
    var session = buildSessionObject();
    var blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    var safeTitle = (document.title || 'viewer').replace(/[^\w\-]+/g, '_');
    a.download = safeTitle + '_edits.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    var n = Object.keys(session.pages).length;
    showToast(n > 0 ? 'Exported ' + n + ' page(s)' : 'Exported (no edits yet)');
  }

  function importJSON(file) {
    var reader = new FileReader();
    reader.onload = function(e) {
      try {
        var session = JSON.parse(e.target.result);
        if (!session || !session._meta) {
          showToast('⚠ Not a valid editor session JSON', 4000);
          return;
        }
        if (session._meta.documentTitle &&
            session._meta.documentTitle !== document.title) {
          if (!confirm('This session is from “' + session._meta.documentTitle +
                       '”, but you are viewing “' + document.title +
                       '”. Apply it anyway?')) return;
        }
        applySession(session);
      } catch (err) {
        showToast('⚠ Import failed: ' + err.message, 4000);
      }
    };
    reader.readAsText(file);
  }

  // ─── Toast ───
  function showToast(msg, ms) {
    var t = document.getElementById('editor-toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function() { t.classList.remove('show'); }, ms || 2500);
  }

  // ─── Wire up events ───
  function bindControls() {
    document.getElementById('editor-edit-toggle').addEventListener('click', function() {
      setEditMode(!editMode);
    });
    document.getElementById('editor-export-btn').addEventListener('click', exportJSON);
    document.getElementById('editor-import-btn').addEventListener('click', function() {
      document.getElementById('editor-import-file').click();
    });
    document.getElementById('editor-import-file').addEventListener('change', function(e) {
      var f = e.target.files && e.target.files[0];
      if (f) importJSON(f);
      e.target.value = '';
    });
    // Delegated handler for per-article done/redo buttons
    document.addEventListener('click', function(e) {
      var t = e.target;
      if (!t || !t.classList) return;
      var pid = t.getAttribute && t.getAttribute('data-pageid');
      if (!pid) return;
      if (t.classList.contains('done-btn')) toggleDone(pid);
      else if (t.classList.contains('redo-btn')) toggleRedo(pid);
    });
    // Restore banner
    document.getElementById('editor-restore-btn').addEventListener('click', function() {
      var s = this._session;
      document.getElementById('editor-restore-banner').classList.add('hidden');
      if (s) applySession(s);
    });
    document.getElementById('editor-restore-dismiss').addEventListener('click', function() {
      document.getElementById('editor-restore-banner').classList.add('hidden');
      try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    });
    // Keyboard shortcuts: D / Shift+D / E
    document.addEventListener('keydown', function(e) {
      // Ignore when typing inside an editable transcription
      if (e.target && e.target.isContentEditable) return;
      if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey) {
        // Find the page currently in view and toggle it
        var pid = lastViewedPageId;
        if (!pid) return;
        if (e.shiftKey) toggleRedo(pid);
        else toggleDone(pid);
        e.preventDefault();
      } else if ((e.key === 'e' || e.key === 'E') && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        setEditMode(!editMode);
        e.preventDefault();
      }
    });
    // Save on unload as a last resort
    window.addEventListener('beforeunload', doAutoSave);
  }

  // ─── Tiny CSS attribute-selector escape helper ───
  function cssAttrEscape(s) {
    // Page IDs are alphanumeric + dashes in this viewer, but be defensive.
    return String(s).replace(/(["\\])/g, '\\$1');
  }

  // ─── Boot ───
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
// === END EDITOR ADDON ===
"""


# ───────────────────────────────────────────────────────────
# Convenience exports
# ───────────────────────────────────────────────────────────

__all__ = [
    "ADDON_MARKER",
    "CSS_ADDON",
    "TOOLBAR_HTML",
    "ARTICLE_WIDGETS_TEMPLATE",
    "RESTORE_AND_TOAST_HTML",
    "SIDEBAR_SUMMARY_HTML",
    "JS_ADDON",
]
