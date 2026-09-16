'use strict';

const $ = (id) => document.getElementById(id);
const dom = {
    crumb: $('crumb'), crumbName: $('crumbName'), crumbExtra: $('crumbExtra'),
    loading: $('loading'), csvBtn: $('csvBtn'), clearBtn: $('clearBtn'), themeBtn: $('themeBtn'),
    progress: $('progress'), progressFill: $('progressFill'), progressText: $('progressText'),
    landing: $('landing'), drop: $('drop'), dropTitle: $('dropTitle'), picker: $('picker'),
    strip: $('strip'), scan: $('scan'), scroller: $('scroller'), zoombox: $('zoombox'),
    stage: $('stage'), frame: $('frame'), splitter: $('splitter'), pane: $('pane'),
    boxesBtn: $('boxesBtn'), zoomIn: $('zoomIn'), zoomOut: $('zoomOut'), zoomFit: $('zoomFit'),
    scanState: $('scanState'), zoomValue: $('zoomValue'), veil: $('veil'), notes: $('notes'),
};

const state = {
    frames: [],
    index: 0,
    pinned: false,
    active: null,
    zoom: 1,
    manual: false,
    boxes: true,
    split: 52,
    ratio: 0.6,
};

const IMAGES = /\.(png|jpe?g|webp|bmp|tiff?|avif)$/i;
const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const pct = (v) => `${Math.round(v * 100)}%`;
const current = () => state.frames[state.index] || null;

function note(text, bad) {
    const node = document.createElement('div');
    node.className = bad ? 'note bad' : 'note';
    node.textContent = text;
    dom.notes.append(node);
    setTimeout(() => {
        node.classList.add('out');
        setTimeout(() => node.remove(), 220);
    }, 3400);
}

function plural(n, one, few, many) {
    const m10 = n % 10;
    const m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
}

/* ── theme ───────────────────────────────────────────────── */

function applyTheme(dark) {
    document.documentElement.classList.toggle('dark', dark);
    try {
        localStorage.setItem('plates-theme', dark ? 'dark' : 'light');
    } catch (error) { /* ignore */
    }
}

(function initTheme() {
    let saved = null;
    try {
        saved = localStorage.getItem('plates-theme');
    } catch (error) { /* ignore */
    }
    const dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(dark);
})();

dom.themeBtn.addEventListener('click', () => applyTheme(!document.documentElement.classList.contains('dark')));

/* ── upload ──────────────────────────────────────────────── */

async function upload(files) {
    const picked = Array.from(files).filter((f) => f.type.startsWith('image/') || IMAGES.test(f.name));
    if (!picked.length) return note('Это не изображения', true);

    dom.loading.hidden = false;
    const form = new FormData();
    picked.forEach((f) => form.append('files', f, f.name));
    try {
        const response = await fetch('/api/jobs', {method: 'POST', body: form});
        if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
        const payload = await response.json();
        const byName = new Map();
        picked.forEach((file) => {
            if (!byName.has(file.name)) byName.set(file.name, file);
        });

        const fresh = state.frames.length === 0;
        payload.accepted.forEach((job) => {
            const file = byName.get(job.filename);
            state.frames.push({
                id: job.id,
                filename: job.filename,
                status: job.status,
                position: job.position,
                error: null,
                result: null,
                preview: file ? URL.createObjectURL(file) : null,
            });
        });
        payload.rejected.forEach((item) => note(`${item.filename}: ${item.reason}`, true));
        if (fresh) {
            state.index = 0;
            state.pinned = false;
        }
        renderChrome();
        renderBody();
    } catch (error) {
        note(error.message, true);
    } finally {
        dom.loading.hidden = true;
    }
}

dom.drop.addEventListener('click', () => dom.picker.click());
dom.picker.addEventListener('change', () => {
    if (dom.picker.files.length) upload(dom.picker.files);
    dom.picker.value = '';
});

let depth = 0;
document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    depth += 1;
    if (state.frames.length) dom.veil.classList.add('on');
    else dom.drop.classList.add('over');
    dom.dropTitle.textContent = 'Отпустите файлы';
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    depth = Math.max(0, depth - 1);
    if (!depth) resetDrag();
});
document.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    resetDrag();
    if (e.dataTransfer && e.dataTransfer.files.length) upload(e.dataTransfer.files);
});
document.addEventListener('paste', (e) => {
    const files = e.clipboardData ? Array.from(e.clipboardData.files) : [];
    if (files.length) upload(files);
});

function resetDrag() {
    dom.veil.classList.remove('on');
    dom.drop.classList.remove('over');
    dom.dropTitle.textContent = 'Перетащите фото';
}

/* ── chrome ──────────────────────────────────────────────── */

function renderChrome() {
    const frames = state.frames;
    const has = frames.length > 0;
    const frame = current();

    dom.landing.hidden = has;
    dom.scan.hidden = !has;
    dom.pane.hidden = !has;
    dom.splitter.hidden = !has;
    dom.strip.hidden = frames.length < 2;
    dom.crumb.hidden = !frame;
    dom.csvBtn.hidden = !has;
    dom.clearBtn.hidden = !has;

    if (frame) {
        dom.crumbName.textContent = frame.filename;
        dom.crumbExtra.textContent = frames.length > 1 ? `${frames.length} ${plural(frames.length, 'кадр', 'кадра', 'кадров')}` : '';
    }

    const settled = frames.filter((f) => f.status === 'done' || f.status === 'failed' || f.status === 'cancelled').length;
    const busy = has && settled < frames.length;
    dom.progress.hidden = !busy;
    if (busy) {
        dom.progressFill.style.width = `${Math.max((settled / frames.length) * 100, 4)}%`;
        dom.progressText.textContent = `${settled} / ${frames.length}`;
    }

    if (frames.length > 1) renderStrip();
    dom.scan.style.width = `${state.split}%`;
}

const MARKS = {
    done: '<svg class="ok" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    running: '<svg class="run" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>',
    failed: '<svg class="err" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.7 18-8-14a2 2 0 0 0-3.5 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3M12 9v4M12 17h.01"/></svg>',
    queued: '<svg class="wait" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="3 3"><circle cx="12" cy="12" r="9"/></svg>',
};

function renderStrip() {
    dom.strip.innerHTML = state.frames.map((frame, index) => `
    <button type="button" class="tile${index === state.index ? ' on' : ''}" data-index="${index}" title="${esc(frame.filename)}">
      <span>${index + 1}</span>${MARKS[frame.status] || MARKS.queued}
    </button>`).join('');
    dom.strip.querySelectorAll('.tile').forEach((tile) => {
        tile.addEventListener('click', () => {
            state.pinned = true;
            state.index = Number(tile.dataset.index);
            state.active = null;
            state.manual = false;
            renderChrome();
            renderBody();
        });
    });
    const on = dom.strip.querySelector('.tile.on');
    if (on) on.scrollIntoView({block: 'nearest', inline: 'nearest'});
}

/* ── body ────────────────────────────────────────────────── */

let shownId = null;

function renderBody() {
    const frame = current();
    if (!frame) {
        shownId = null;
        dom.pane.innerHTML = '';
        return;
    }
    if (shownId !== frame.id) {
        shownId = frame.id;
        state.manual = false;
        dom.scroller.scrollTop = 0;
        dom.frame.removeAttribute('src');
        if (frame.preview) dom.frame.src = frame.preview;
    }
    renderBoxes();
    renderPane();
    renderScanFoot();
}

function renderBoxes() {
    dom.stage.querySelectorAll('.hit').forEach((node) => node.remove());
    const frame = current();
    const result = frame && frame.result;
    if (!result || !state.boxes) return;

    const {width, height} = result.source;
    result.plates.forEach((plate, index) => {
        const [x0, y0, x1, y1] = plate.box;
        const hit = document.createElement('button');
        hit.type = 'button';
        hit.className = `hit${plate.confidence < 0.6 ? ' weak' : ''}${state.active === index ? ' on' : ''}`;
        hit.style.left = `${(x0 / width) * 100}%`;
        hit.style.top = `${(y0 / height) * 100}%`;
        hit.style.width = `${((x1 - x0) / width) * 100}%`;
        hit.style.height = `${((y1 - y0) / height) * 100}%`;
        hit.style.animationDelay = `${Math.min(index * 18, 360)}ms`;
        hit.innerHTML = `<span class="hit-tag">${esc(plate.text || plate.subtype)}</span>`;
        hit.addEventListener('click', () => pick(index === state.active ? null : index, true));
        dom.stage.append(hit);
    });
}

function renderPane() {
    const frame = current();
    if (!frame) return;

    if (frame.status === 'failed') {
        dom.pane.innerHTML = `<div class="state bad">${esc(frame.error || 'ошибка обработки')}</div>`;
        return;
    }
    if (frame.status === 'cancelled') {
        dom.pane.innerHTML = '<div class="state">Снято</div>';
        return;
    }
    if (frame.status !== 'done' || !frame.result) {
        const queued = frame.status === 'queued';
        dom.pane.innerHTML = `<div class="state">
      <span>${queued ? 'В очереди' : 'Читаем номера'}</span>
      ${queued && frame.position ? `<span class="sub">Перед этим кадром: ${frame.position}</span>` : ''}
    </div>`;
        return;
    }
    if (!frame.result.plates.length) {
        dom.pane.innerHTML = '<div class="state">Номеров не найдено</div>';
        return;
    }

    dom.pane.innerHTML = frame.result.plates.map((plate, index) => rowMarkup(plate, index)).join('');
    dom.pane.querySelectorAll('.row').forEach((row) => {
        row.addEventListener('click', () => pick(Number(row.dataset.index) === state.active ? null : Number(row.dataset.index), false));
    });
}

function rowMarkup(plate, index) {
    const weak = plate.confidence < 0.6;
    const meta = [`<span>${esc(plate.subtype)}</span>`, `<span class="${weak ? 'flag' : ''}">${pct(plate.confidence)}</span>`];
    if (plate.readable) meta.push(plate.valid ? '<span class="good">ГОСТ</span>' : '<span class="flag">вне ГОСТ</span>');

    const glyphs = plate.characters.map((item) => {
        const alts = item.alternatives.length
            ? `<span class="alts">${item.alternatives.map((a) => `<div><span>${esc(a.char)}</span><span>${pct(a.probability)}</span></div>`).join('')}</span>`
            : '';
        return `<div class="glyph${item.probability < 0.9 ? ' weak' : ''}">
      <span class="g">${esc(item.char)}</span><span class="p">${Math.round(item.probability * 100)}</span>${alts}
    </div>`;
    }).join('');

    return `<article class="row${plate.readable ? '' : ' dim'}${state.active === index ? ' on' : ''}" data-index="${index}">
    <div class="row-patch"><img src="${plate.patch_png}" alt="" loading="lazy"></div>
    <div class="row-body">
      <div class="row-plate${plate.text ? '' : ' none'}">${plate.text ? esc(plate.text) : 'не прочитан'}</div>
      <div class="row-meta">${meta.join('')}</div>
      ${glyphs ? `<div class="glyphs">${glyphs}</div>` : ''}
    </div>
  </article>`;
}

function pick(index, fromBox) {
    state.active = index;
    dom.stage.querySelectorAll('.hit').forEach((hit, position) => hit.classList.toggle('on', position === index));
    dom.pane.querySelectorAll('.row').forEach((row) => {
        const on = Number(row.dataset.index) === index;
        row.classList.toggle('on', on);
        if (on && fromBox) row.scrollIntoView({block: 'nearest', behavior: 'smooth'});
    });
    if (index !== null && !fromBox) {
        const hit = dom.stage.querySelectorAll('.hit')[index];
        if (hit) hit.scrollIntoView({block: 'nearest', behavior: 'smooth'});
    }
}

function renderScanFoot() {
    const frame = current();
    const result = frame && frame.result;
    if (result) {
        const n = result.plates.length;
        dom.scanState.textContent = n
            ? `${n} ${plural(n, 'номер', 'номера', 'номеров')} · ${Math.round(result.timing_ms.detect)} + ${Math.round(result.timing_ms.recognize)} ms`
            : 'ничего не найдено';
    } else if (frame && frame.status === 'running') {
        dom.scanState.innerHTML = '<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>Обрабатывается';
    } else {
        dom.scanState.textContent = frame && frame.status === 'queued' ? 'В очереди' : '';
    }
    dom.zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
}

/* ── zoom ────────────────────────────────────────────────── */

function fitZoom() {
    const box = dom.scroller.getBoundingClientRect();
    const width = box.width - 40;
    const height = box.height - 40;
    if (width <= 0 || height <= 0) return 1;
    return Math.max(0.05, Math.min(1, Math.round((height / (width * state.ratio)) * 200) / 200));
}

function applyZoom(value) {
    state.zoom = Math.min(5, Math.max(0.1, value));
    dom.zoombox.style.width = `${state.zoom * 100}%`;
    dom.zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
}

dom.frame.addEventListener('load', () => {
    if (dom.frame.naturalWidth) state.ratio = dom.frame.naturalHeight / dom.frame.naturalWidth;
    if (!state.manual) applyZoom(fitZoom());
    renderBoxes();
});

dom.zoomIn.addEventListener('click', () => {
    state.manual = true;
    applyZoom(state.zoom + 0.2);
});
dom.zoomOut.addEventListener('click', () => {
    state.manual = true;
    applyZoom(state.zoom - 0.2);
});
dom.zoomFit.addEventListener('click', () => {
    state.manual = false;
    applyZoom(fitZoom());
});
dom.boxesBtn.addEventListener('click', () => {
    state.boxes = !state.boxes;
    dom.boxesBtn.classList.toggle('on', !state.boxes);
    renderBoxes();
});

let resizeTimer = 0;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (!state.manual) applyZoom(fitZoom());
    }, 140);
});

/* ── splitter ────────────────────────────────────────────── */

let dragging = false;
dom.splitter.addEventListener('mousedown', () => {
    dragging = true;
    dom.splitter.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
});
window.addEventListener('mousemove', (event) => {
    if (!dragging) return;
    const offset = dom.strip.hidden ? 0 : dom.strip.offsetWidth;
    const usable = window.innerWidth - offset;
    state.split = Math.min(76, Math.max(24, ((event.clientX - offset) / usable) * 100));
    dom.scan.style.width = `${state.split}%`;
});
window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    dom.splitter.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    if (!state.manual) applyZoom(fitZoom());
});

/* ── actions ─────────────────────────────────────────────── */

dom.csvBtn.addEventListener('click', () => window.open('/api/export.csv', '_blank'));
dom.clearBtn.addEventListener('click', reset);

async function reset() {
    state.frames.forEach((frame) => {
        if (frame.preview) URL.revokeObjectURL(frame.preview);
    });
    state.frames = [];
    state.index = 0;
    state.active = null;
    state.pinned = false;
    shownId = null;
    dom.frame.removeAttribute('src');
    dom.stage.querySelectorAll('.hit').forEach((node) => node.remove());
    dom.pane.innerHTML = '';
    renderChrome();
    try {
        await fetch('/api/jobs/clear', {method: 'POST'});
    } catch (error) { /* ignore */
    }
}

/* ── stream ──────────────────────────────────────────────── */

async function loadResult(frame) {
    try {
        const response = await fetch(`/api/jobs/${frame.id}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        frame.result = (await response.json()).result;
        if (current() === frame) {
            renderBoxes();
            renderPane();
            renderScanFoot();
        }
    } catch (error) {
        note(error.message, true);
    }
}

function onJob(job) {
    const frame = state.frames.find((item) => item.id === job.id);
    if (!frame) return;
    const wasDone = frame.status === 'done';
    frame.status = job.status;
    frame.position = job.position;
    frame.error = job.error;

    if (job.status === 'done' && !wasDone && !frame.result) loadResult(frame);
    if (job.status === 'running' && !state.pinned) {
        const index = state.frames.indexOf(frame);
        if (index !== state.index) {
            state.index = index;
            state.active = null;
            renderBody();
        }
    }
    renderChrome();
    if (current() === frame) {
        renderPane();
        renderScanFoot();
    }
}

new EventSource('/api/stream').onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'job') onJob(payload.job);
};

renderChrome();
