'use strict';
// The Simple view's front-to-back line, under the pictures: one dot per placed section at
// its position along the cutting axis, its two ends named (`jobs/<n>/strip`'s axis). The
// sections the agent is working on are in the accent, the ones shown above it ringed. A
// dot glides to its new place when the agent moves its section, so a reshuffle is seen
// as it happens; a section placed for the first time drops in. Hovering names a section,
// clicking shows it. The sections not placed yet are counted at the right end.
const Axis = (() => {
  const $ = id => document.getElementById(id);
  const { el } = Ui;
  // A move shorter than this (mm) is a nudge, not shown as a move.
  const MOVED_MM = 0.02;
  // More sections than this moving at once is a reshuffle.
  const FEW = 3;

  let onPick = () => {};
  let rows = [];
  let range = null;              // {lo, hi, low, high} from the strip's axis
  let dots = new Map();          // section id -> {node, mm}
  let all = [], shown = [];

  function init({ pick }) {
    onPick = pick;
    const line = $('axis-line');
    line.addEventListener('pointerover', event => {
      const dot = event.target.closest('.axis-dot');
      if (dot) tip(dot);
    });
    line.addEventListener('pointerout', event => {
      if (event.target.closest('.axis-dot')) $('axis-tip').hidden = true;
    });
  }
  function setJob() {
    rows = []; range = null; all = []; shown = [];
    for (const dot of dots.values()) dot.slot.remove();
    dots = new Map();
    ticked = '';
    $('axis-tip').hidden = true;
    draw();
  }
  function setInfo(info) {
    const a = info?.axis;
    range = a && Number.isFinite(a.min_mm) && Number.isFinite(a.max_mm) && a.max_mm > a.min_mm
      ? { lo: a.min_mm, hi: a.max_mm, low: a.low || '', high: a.high || '' } : null;
    draw();
  }
  function setRows(value) { rows = Array.isArray(value) ? value : []; draw(); }
  function setTargets(everyId, pageIds) { all = everyId || []; shown = pageIds || []; mark(); }

  // The span the dots are placed on: the atlas's, else the sections' own with a margin.
  function span() {
    // The line spans the sections placed so far, half a millimetre beyond them, so the dots
    // use its width; with fewer than two placed, the atlas's whole axis.
    const mm = rows.map(r => r.position_mm).filter(Number.isFinite);
    const low = range?.low || 'anterior', high = range?.high || 'posterior';
    if (mm.length < 2) return range;
    const lo = Math.min(...mm), hi = Math.max(...mm);
    return { lo: lo - 0.5, hi: Math.max(hi, lo + 1) + 0.5, low, high };
  }
  function draw() {
    const s = span();
    $('axis').hidden = !rows.length;
    $('axis-low').textContent = s?.low || 'anterior';
    $('axis-high').textContent = s?.high || 'posterior';
    ticks(s);
    const line = $('axis-line');
    const n = rows.length;
    let waiting = 0;
    // How many placed sections moved since the last look: one or a few are each marked;
    // a reshuffle of many moves front to back as a wave, unmarked.
    const moving = rows.filter(r => {
      const dot = dots.get(r.id);
      return dot && dot.mm !== null && Number.isFinite(r.position_mm) && Math.abs(dot.mm - r.position_mm) > MOVED_MM;
    }).length;
    $('axis-line').classList.toggle('wave', moving > FEW);
    rows.forEach((r, order) => {
      const placed = Number.isFinite(r.position_mm) && s;
      // A section not placed yet waits, dim, in stack order along the whole line; placed,
      // it glides to its position.
      const x = placed ? Math.min(100, Math.max(0, (100 * (r.position_mm - s.lo)) / (s.hi - s.lo))) : (100 * (order + 0.5)) / n;
      if (!placed) waiting += 1;
      let dot = dots.get(r.id);
      if (!dot) {
        // Each dot rides in a slot as wide as the line, moved by a transform.
        const slot = el('span', 'axis-slot new');
        const node = el('button', 'axis-dot');
        node.type = 'button';
        node.dataset.id = r.id;
        node.setAttribute('aria-label', Ui.short(r.id));
        node.addEventListener('click', () => onPick(r.id));
        slot.append(node);
        slot.style.transform = `translateX(${x}%)`;
        line.append(slot);
        dot = { slot, node, mm: placed ? r.position_mm : null, x };
        dots.set(r.id, dot);
        requestAnimationFrame(() => requestAnimationFrame(() => slot.classList.remove('new')));
      } else if (placed && dot.mm !== null && Math.abs(dot.mm - r.position_mm) > MOVED_MM && moving <= FEW) {
        const ghost = el('span', 'axis-ghost');
        ghost.style.left = `${dot.x}%`;
        line.append(ghost);
        setTimeout(() => ghost.remove(), 1400);
        dot.node.classList.remove('moved');
        void dot.node.offsetWidth;
        dot.node.classList.add('moved');
      }
      dot.mm = placed ? r.position_mm : null;
      dot.x = x;
      dot.slot.style.transform = `translateX(${x}%)`;
      dot.slot.style.setProperty('--x', String(Math.round(x)));
      dot.node.classList.toggle('unplaced', !placed);
      dot.node.dataset.mm = placed ? Ui.mm(r.position_mm) : '';
    });
    const ids = new Set(rows.map(r => r.id));
    for (const [id, dot] of dots) if (!ids.has(id)) { dot.slot.remove(); dots.delete(id); }
    $('axis-waiting').textContent = !n || !waiting ? ''
      : waiting === n ? `${n} sections, not placed yet` : `${waiting} not placed yet`;
    mark();
  }
  // Faint ticks a millimetre apart, so the line reads as a distance.
  let ticked = '';
  function ticks(s) {
    const key = s ? `${s.lo}|${s.hi}` : '';
    if (key === ticked) return;
    ticked = key;
    const line = $('axis-line');
    for (const t of line.querySelectorAll('.axis-tick')) t.remove();
    if (!s) return;
    for (let mm = Math.ceil(s.lo); mm <= s.hi; mm += 1) {
      const tick = el('span', 'axis-tick');
      tick.style.left = `${(100 * (mm - s.lo)) / (s.hi - s.lo)}%`;
      line.prepend(tick);
    }
  }
  function mark() {
    for (const [id, dot] of dots) {
      dot.node.classList.toggle('target', all.includes(id) && all.length < rows.length);
      dot.node.classList.toggle('shown', shown.includes(id));
    }
  }
  function tip(dot) {
    const box = $('axis-tip');
    box.replaceChildren(el('span', 'tip-name', Ui.short(dot.dataset.id)), el('span', 'tip-line', dot.dataset.mm ? ` ${dot.dataset.mm} mm` : ' not placed yet'));
    const line = $('axis-line').getBoundingClientRect(), r = dot.getBoundingClientRect();
    box.hidden = false;
    const left = r.left + r.width / 2 - line.left - box.offsetWidth / 2;
    box.style.left = `${Math.max(0, Math.min(line.width - box.offsetWidth, left))}px`;
  }

  return { init, setJob, setInfo, setRows, setTargets };
})();
