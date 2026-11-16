'use strict';
// The journal and the narration. The journal is the run so far as a timeline: the agent's
// messages and reasoning summaries are its entries, in the serif face, each at its minute
// of the run; between them each tool call is one quiet line in plain words (the tool's
// label; its name on hover and inside; opened, its inputs, step and status), with a
// `N pictures` link when it saved some. Calls of one tool in a row fold into one line
// (`xn`) that opens to show each. The journal follows its end until the person scrolls
// away. The narration band over the window says the newest entry's first paragraph, large,
// and under it what the agent is doing now. Under the journal, the tray: one call's
// pictures at a time, four per page, with `<- n/N ->` across the calls and Newest. The
// lightbox shows one picture as large as the window allows, with its caption, and the
// arrow keys go through every picture of the job.
const AgentLog = (() => {
  const PER_PAGE = 4;
  // Pictures no tool call owns: the job's opening ones, and the pages of the whole series
  // the agent reads first.
  const GROUP_TITLES = { opening: 'Opening pictures', show_stack: 'The whole series at a glance' };
  const $ = id => document.getElementById(id);
  const { el } = Ui;

  let job = null;
  let calls = new Map();      // tool calls by key
  let claims = new Map();     // a view folder a call's end names -> that call
  let groups = new Map();     // pictures no call claims, by tool and call number
  let owners = [];            // the calls and groups with pictures, in picture order
  let pictures = [];          // every picture, by number
  let tray = { follow: true, owner: null, page: 0, shown: '' };
  let tail = null;            // the last row of the transcript (a call, a run, or other)
  let words = false;          // the agent's own words are in the folder (a trace)
  let activity = false;
  let box = { list: null, index: 0 };
  let status = { word: '', sentence: '' };
  let lastCall = null;        // the newest tool call
  let started = null;         // when the agent session's first record was written
  let spoken = null;          // the timer that ends the figure's talking
  let following = true;       // the journal keeps its end in view
  let latest = null;          // the newest of the agent's own entries

  function init() {
    const scroll = $('log-scroll');
    // The journal follows its end until the person scrolls away from it (rows also grow
    // between additions, when a call's pictures arrive).
    scroll.addEventListener('scroll', () => { following = atEnd(); $('log-jump').hidden = following; });
    $('log-jump').addEventListener('click', () => { following = true; toEnd(); $('log-jump').hidden = true; });
    $('log-expand').addEventListener('click', () => {
      const on = !$('log').classList.contains('expanded');
      $('log').classList.toggle('expanded', on);
      $('log-expand').setAttribute('aria-expanded', String(on));
      $('log-expand').textContent = on ? 'Last lines' : 'Whole journal';
      toEnd();
    });
    $('tray-previous').addEventListener('click', () => trayStep(-1));
    $('tray-next').addEventListener('click', () => trayStep(1));
    $('tray-follow').addEventListener('click', () => { tray.follow = true; renderTray(); });
    $('lightbox-previous').addEventListener('click', () => lightboxStep(-1));
    $('lightbox-next').addEventListener('click', () => lightboxStep(1));
    $('lightbox-close').addEventListener('click', () => $('lightbox').close());
    $('lightbox').addEventListener('keydown', event => {
      if (event.key === 'ArrowLeft') { lightboxStep(-1); event.preventDefault(); }
      else if (event.key === 'ArrowRight') { lightboxStep(1); event.preventDefault(); }
    });
    $('lightbox').addEventListener('close', () => { $('lightbox-image').removeAttribute('src'); });
    traySize();
    new ResizeObserver(() => {
      $('log').style.setProperty('--tray-size', $('tray').offsetHeight + 'px');
    }).observe($('tray'));
  }
  function setJob(index) {
    job = index;
    calls = new Map(); claims = new Map(); groups = new Map(); owners = []; pictures = [];
    tray = { follow: true, owner: null, page: 0, shown: '' };
    words = false; activity = false; tail = null;
    lastCall = null; started = null; latest = null; following = true;
    $('now-words').textContent = '';
    $('log-items').replaceChildren();
    $('log-jump').hidden = true;
    $('log-empty').hidden = true;
    $('log-trace-note').hidden = true;
    renderTray();
  }
  // The narration band's two lines: the agent's latest words (`#now-words`, written by
  // `message`) and, under them, what it is doing now in plain words: the running call,
  // else the last one, else the server's sentence.
  function setStatus(word, sentence) {
    status = { word, sentence: sentence || '' };
    renderDoing();
  }
  function renderDoing() {
    const running = [...calls.values()].filter(c => c.root.classList.contains('running')).pop();
    const last = running || lastCall;
    const what = last ? last.title + sectionWords(last.sections) : '';
    let text;
    if (running) text = 'Now: ' + what;
    else if (status.word === 'Running' && last) text = 'Last step: ' + what;
    else text = status.sentence.replace(/\s*Last: .*$/, '') || (last ? 'Last step: ' + what : '');
    $('sentence').textContent = text;
    $('sentence').title = status.sentence;
    $('statusline').classList.toggle('wordless', !$('now-words').textContent);
  }
  const sectionWords = ids => (!ids?.length ? '' : ids.length === 1 ? ', on ' + Ui.short(ids[0]) : `, on ${ids.length} sections`);
  // Minutes into the run, from the agent session's first record.
  function elapsed(at) {
    const t = Date.parse(at || '');
    if (!Number.isFinite(t)) return '';
    if (started === null) started = t;
    const minutes = Math.round((t - started) / 60000);
    return minutes < 1 ? 'Start' : `${minutes} min`;
  }
  const toEnd = () => { const scroll = $('log-scroll'); scroll.scrollTop = scroll.scrollHeight; };
  const atEnd = () => {
    const scroll = $('log-scroll');
    return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 60;
  };

  // --- the transcript ----------------------------------------------------------------------
  function add(items) {
    let pictured = false;
    for (const item of items) {
      if (item.kind === 'tool') toolItem(item);
      else if (item.kind === 'picture') { pictureItem(item); pictured = true; }
      else if (item.kind === 'agent' || item.kind === 'reasoning') message(item);
      else if (item.kind === 'note') note(item);
    }
    $('log-empty').hidden = $('log-items').children.length > 0;
    $('log-empty').textContent = 'No activity yet.';
    $('log-trace-note').hidden = words || !activity;
    if (pictured || items.some(i => i.kind === 'tool')) renderTray();
    if (items.length) renderDoing();
    if (following && items.length) toEnd();
  }
  // An agent message: plain text with paragraphs, `inline code` in violet, **bold** names
  // and list lines; nothing from the text is read as markup.
  function message(item) {
    words = true;
    const root = el('article', 'msg ' + (item.kind === 'agent' ? 'agent' : 'reasoning'));
    const when = el('div', 'msg-when', elapsed(item.at));
    when.title = item.kind === 'agent' ? 'A message from the agent' : "A summary of the agent's reasoning";
    root.append(when);
    const body = el('div', 'msg-body');
    for (const block of String(item.text || '').split(/\n\s*\n/)) {
      if (!block.trim()) continue;
      const lines = block.split('\n');
      if (lines.every(line => /^\s*([-*]|\d+\.)\s+/.test(line))) {
        for (const line of lines) {
          const [, marker, rest] = line.match(/^\s*([-*]|\d+\.)\s+(.*)$/);
          const p = el('p', 'item');
          p.append(el('span', 'marker', (marker === '*' ? '-' : marker) + ' '));
          inline(p, rest);
          body.append(p);
        }
      } else {
        const p = el('p');
        lines.forEach((line, i) => { if (i) p.append(document.createElement('br')); inline(p, line.replace(/^#{1,6}\s+/, '')); });
        body.append(p);
      }
    }
    root.append(body);
    $('log-items').append(root);
    tail = null;
    latest?.classList.remove('latest');
    latest = root;
    root.classList.add('latest');
    narrate(body);
  }
  // The narration band says the agent's latest words: the entry's first paragraph (the
  // whole entry on hover, and in the journal).
  // The words come in one after another, as said, and the agent's figure talks meanwhile.
  function narrate(body) {
    const now = $('now-words');
    const text = body.firstElementChild?.textContent || '';
    now.replaceChildren();
    text.split(/(\s+)/).forEach((part, i) => {
      if (!part.trim()) { now.append(document.createTextNode(part)); return; }
      const word = el('span', 'word-in', part);
      word.style.animationDelay = `${Math.min(i * 22, 2600)}ms`;
      now.append(word);
    });
    now.title = body.textContent || '';
    const band = $('statusline');
    band.classList.add('speaking');
    clearTimeout(spoken);
    spoken = setTimeout(() => band.classList.remove('speaking'), Math.min(4200, 1200 + text.length * 14));
  }
  function inline(parent, text) {
    for (const part of text.split(/(`[^`]+`|\*\*[^*]+\*\*)/)) {
      if (!part) continue;
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) parent.append(el('code', '', part.slice(1, -1)));
      else if (part.startsWith('**') && part.endsWith('**') && part.length > 4) parent.append(el('strong', '', part.slice(2, -2)));
      else parent.append(document.createTextNode(part));
    }
  }
  function note(item) {
    if (/^Agent session/.test(item.text || '')) { words = true; elapsed(item.at); }
    $('log-items').append(el('p', 'note' + (item.error ? ' error' : ''), item.text));
    tail = null;
  }
  // A call's sections: one is named (and puts it in focus), more are counted, their names
  // on hover.
  function sectionsLine(ids) {
    const line = el('span', 'call-targets');
    ids = ids || [];
    if (ids.length === 1) {
      const id = ids[0];
      const link = el('span', 'section-link', Ui.short(id));
      link.setAttribute('role', 'button');
      link.tabIndex = 0;
      link.title = 'Show ' + id;
      const go = event => { event.preventDefault(); event.stopPropagation(); Viewer.pick(id); };
      link.addEventListener('click', go);
      link.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') go(event); });
      line.append(link);
    } else if (ids.length) {
      line.textContent = `${ids.length} sections`;
      line.title = ids.join(', ');
    }
    return line;
  }
  function toolItem(item) {
    activity = true;
    let call = calls.get(item.key);
    if (!call) {
      call = makeCall(item);
      calls.set(item.key, call);
      lastCall = call;
      place(call);
    }
    if (item.phase === 'start') {
      call.start = item;
      if (!call.end) setState(call, 'running');
    } else {
      call.end = item;
      setState(call, item.failed ? 'failed' : 'done');
      for (const path of item.views || []) {
        claims.set(path, call);
        // A picture that came before the call's end moves from its group to the call.
        for (const group of groups.values()) {
          const index = group.pictures.findIndex(p => p.path === path);
          if (index < 0) continue;
          const [picture] = group.pictures.splice(index, 1);
          own(call, picture);
          if (!group.pictures.length) dropGroup(group);
          else link(group);
        }
      }
    }
    details(call);
    if (call.run) updateRun(call.run);
  }
  function makeCall(item) {
    const root = el('details', 'call');
    const summary = el('summary');
    const body = el('span', 'call-body');
    const title = el('span', 'call-title', item.title || item.name);
    title.title = item.name;
    body.append(title, sectionsLine(item.sections));
    // The tool's own name, beside its plain words.
    if (item.title && item.title !== item.name) body.append(el('span', 'call-name', item.name));
    const state = el('span', 'call-state');
    summary.append(body, state);
    const detail = el('div', 'call-detail');
    root.append(summary, detail);
    return { kind: 'call', key: item.key, name: item.name, title: item.title || item.name, sections: item.sections || [], root, state, detail, pictures: [], link: null, start: null, end: null, run: null };
  }
  // A new call goes at the end; the second call of one tool in a row starts a run (one
  // row, `×n`, that opens to show the calls), and later ones join it.
  function place(call) {
    if (tail?.kind === 'run' && tail.name === call.name) {
      tail.list.append(call.root);
      call.run = tail;
      tail.calls.push(call);
      return;
    }
    if (tail?.kind === 'call' && tail.name === call.name) {
      const first = tail;
      const run = makeRun(first);
      first.root.before(run.root);
      run.root.after(run.link);
      run.list.append(first.root);
      if (first.link) run.list.append(first.link);
      run.list.append(call.root);
      first.run = call.run = run;
      run.calls.push(first, call);
      tail = run;
      updateRun(run);
      return;
    }
    $('log-items').append(call.root);
    tail = call;
  }
  function makeRun(first) {
    const root = el('details', 'call run');
    const summary = el('summary');
    const body = el('span', 'call-body');
    const title = el('span', 'call-title');
    title.title = first.name;
    const targets = el('span', 'call-targets');
    body.append(title, targets);
    if (first.title !== first.name) body.append(el('span', 'call-name', first.name));
    const state = el('span', 'call-state');
    summary.append(body, state);
    const list = el('div', 'run-calls');
    root.append(summary, list);
    const link = el('button', 'view-images');
    link.hidden = true;
    const run = { kind: 'run', name: first.name, label: first.title, root, title, targets, state, list, link, calls: [] };
    link.addEventListener('click', () => {
      const owner = run.calls.find(c => c.pictures.length);
      if (!owner) return;
      tray.follow = false;
      tray.owner = owner;
      tray.page = 0;
      renderTray();
    });
    root.addEventListener('toggle', () => updateRun(run));
    return run;
  }
  function updateRun(run) {
    run.title.textContent = `${run.label} ×${run.calls.length}`;
    const ids = [];
    for (const call of run.calls) for (const id of call.sections) if (!ids.includes(id)) ids.push(id);
    run.targets.replaceChildren(...sectionsLine(ids).childNodes);
    const running = run.calls.some(c => c.root.classList.contains('running'));
    const failed = run.calls.filter(c => c.root.classList.contains('failed')).length;
    run.root.classList.toggle('running', running);
    run.root.classList.toggle('failed', !running && failed > 0);
    run.state.textContent = running ? 'now' : failed ? `${failed} failed` : '';
    // The run's own link while it is closed; open, each call shows its own.
    const n = run.calls.reduce((sum, c) => sum + c.pictures.length, 0);
    run.link.textContent = `${n} picture${n === 1 ? '' : 's'}`;
    run.link.hidden = run.root.open || !n;
  }
  function setState(call, word) {
    call.root.classList.toggle('running', word === 'running');
    call.root.classList.toggle('failed', word === 'failed');
    // Done is the usual end and says nothing more than the row's grey rule: no word.
    call.state.textContent = word === 'done' ? '' : word === 'running' ? 'now' : word;
    call.state.title = word;
  }
  function details(call) {
    const args = call.end?.args ?? call.start?.args;
    const parts = [];
    if (args !== undefined && args !== null && !(typeof args === 'object' && !Object.keys(args).length)) {
      parts.push(el('h4', '', 'Tool inputs'), el('pre', '', typeof args === 'string' ? args : JSON.stringify(args, null, 2)));
    }
    const end = call.end;
    const meta = [];
    if (end && typeof end.step === 'number') meta.push(`step ${end.step}`);
    if (end?.status) meta.push(String(end.status));
    else if (end) meta.push(end.failed ? 'failed' : 'done');
    else meta.push('running');
    if (end?.door || call.start?.door) meta.push('via ' + (end?.door || call.start?.door));
    meta.unshift(call.name);
    parts.push(el('p', 'call-meta', meta.join(' · ')));
    call.detail.replaceChildren(...parts);
  }

  // --- pictures ------------------------------------------------------------------------------
  function pictureItem(item) {
    activity = true;
    const picture = {
      number: item.number, path: item.path, tool: item.tool, call: item.call,
      sections: item.sections || [], mode: item.mode, caption: item.caption || '', label: item.label,
    };
    pictures.push(picture);
    if (pictures.length > 1 && pictures[pictures.length - 2].number > picture.number) pictures.sort((a, b) => a.number - b.number);
    const call = claims.get(item.path);
    if (call) { own(call, picture); return; }
    // A picture no tool event names (the opening, or a folder from before tool events):
    // one row per call that saved it.
    // The pictures read at the start are one row, whatever number of calls saved them.
    const key = GROUP_TITLES[item.tool] ? item.tool : `${item.tool}:${item.call}`;
    let group = groups.get(key);
    if (!group) {
      const root = el('div', 'call static');
      const line = el('div', 'call-line');
      const body = el('span', 'call-body');
      const opening = Boolean(GROUP_TITLES[item.tool]);
      const title = GROUP_TITLES[item.tool] || item.title || item.tool || 'Pictures';
      body.append(el('span', 'call-title', title));
      const targets = el('span', 'call-targets');
      body.append(targets);
      line.append(body);
      root.append(line);
      group = { kind: 'group', key, name: item.tool, title, root, targets, ids: [], pictures: [], link: null };
      groups.set(key, group);
      // The pictures read at the start come first, after the line saying the job opened.
      const items = $('log-items');
      if (opening) {
        const before = [...groups.values()].filter(g => g !== group && GROUP_TITLES[g.name]).pop();
        const anchor = before ? (before.link || before.root) : items.firstElementChild?.classList.contains('note') ? items.firstElementChild : null;
        if (anchor) anchor.after(root); else items.prepend(root);
      } else { items.append(root); tail = null; }
    }
    for (const id of picture.sections) if (!group.ids.includes(id)) group.ids.push(id);
    group.targets.replaceChildren(...sectionsLine(group.ids).childNodes);
    own(group, picture);
  }
  function own(owner, picture) {
    owner.pictures.push(picture);
    owner.pictures.sort((a, b) => a.number - b.number);
    if (!owners.includes(owner)) owners.push(owner);
    owners.sort((a, b) => a.pictures[0].number - b.pictures[0].number);
    link(owner);
    if (owner.run) updateRun(owner.run);
  }
  function dropGroup(group) {
    group.root.remove();
    group.link?.remove();
    groups.delete(group.key);
    owners = owners.filter(o => o !== group);
  }
  function link(owner) {
    if (!owner.link) {
      owner.link = el('button', 'view-images');
      owner.link.addEventListener('click', () => {
        tray.follow = false;
        tray.owner = owner;
        tray.page = 0;
        renderTray();
      });
      owner.root.after(owner.link);
    }
    const n = owner.pictures.length;
    owner.link.textContent = `${n} picture${n === 1 ? '' : 's'}`;
  }
  const url = picture => `jobs/${job}/picture/${picture.number}`;

  // --- the tray --------------------------------------------------------------------------------
  function batches() {
    const out = [];
    for (const owner of owners) {
      const pages = Math.ceil(owner.pictures.length / PER_PAGE);
      for (let page = 0; page < pages; page++) {
        out.push({ owner, page, pages, pictures: owner.pictures.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE) });
      }
    }
    return out;
  }
  function renderTray() {
    const list = batches();
    const empty = !list.length;
    $('tray').classList.toggle('empty', empty);
    $('tray-divider').hidden = empty;
    for (const owner of owners) { owner.link?.classList.remove('shown'); owner.run?.link.classList.remove('shown'); }
    if (empty) {
      $('tray-tool').textContent = 'nothing yet';
      $('tray-grid').replaceChildren();
      tray.shown = '';
      return;
    }
    let index = tray.follow ? list.length - 1 : list.findIndex(b => b.owner === tray.owner && b.page === tray.page);
    if (index < 0) index = list.length - 1;
    const batch = list[index];
    if (!tray.follow) { tray.owner = batch.owner; tray.page = batch.page; }
    const owner = batch.owner;
    owner.link?.classList.add('shown');
    owner.run?.link.classList.add('shown');
    $('tray-tool').textContent = owner.title || owner.name || 'pictures';
    const total = owner.pictures.length;
    const from = batch.page * PER_PAGE + 1, to = from + batch.pictures.length - 1;
    const ids = owner.kind === 'call' ? owner.sections : owner.ids;
    const sections = ids.length > 1 ? `${ids.length} sections` : ids.map(Ui.short).join('');
    $('tray-caption').textContent = [sections, total > PER_PAGE ? `pictures ${from} to ${to} of ${total}` : `${total} picture${total === 1 ? '' : 's'}`].filter(Boolean).join(', ');
    $('tray-caption').title = ids.join(', ');
    $('tray-count').textContent = `${index + 1}/${list.length}`;
    $('tray-previous').disabled = index === 0;
    $('tray-next').disabled = index === list.length - 1;
    $('tray-follow').setAttribute('aria-pressed', String(tray.follow));
    // The tiles are made again only when the batch changes (each picture is fetched anew).
    const shown = batch.pictures.map(p => p.number).join(',');
    if (shown === tray.shown) return;
    tray.shown = shown;
    const grid = $('tray-grid');
    grid.classList.toggle('one', batch.pictures.length === 1);
    grid.replaceChildren(...batch.pictures.map(tile));
  }
  function tile(picture) {
    const button = el('button', 'tile');
    const img = el('img');
    img.src = url(picture);
    img.alt = picture.caption || picture.label || 'A picture the agent saw';
    img.decoding = 'async';
    const label = el('span', 'tile-label');
    const what = [picture.sections.length === 1 ? Ui.short(picture.sections[0]) : picture.sections.length ? `${picture.sections.length} sections` : '', picture.mode].filter(Boolean).join(', ');
    label.textContent = what || picture.label || '';
    button.title = picture.caption || '';
    button.append(img, label);
    button.addEventListener('click', () => openPicture(picture.number));
    return button;
  }
  function trayStep(delta) {
    const list = batches();
    if (!list.length) return;
    let index = tray.follow ? list.length - 1 : list.findIndex(b => b.owner === tray.owner && b.page === tray.page);
    index = Math.max(0, Math.min(list.length - 1, (index < 0 ? list.length - 1 : index) + delta));
    tray.follow = false;
    tray.owner = list[index].owner;
    tray.page = list[index].page;
    renderTray();
  }
  function traySize() {
    const trayEl = $('tray'), log = $('log');
    const saved = Number(Ui.store.get('trayHeight'));
    if (saved > 0) trayEl.style.setProperty('--tray-height', saved + 'px');
    let start = 0;
    Ui.drag($('tray-divider'), {
      start: () => { start = trayEl.getBoundingClientRect().height; },
      move: (dx, dy) => {
        const height = Math.round(Math.min(log.clientHeight - 160, Math.max(180, start - dy)));
        trayEl.style.setProperty('--tray-height', height + 'px');
      },
      end: () => Ui.store.set('trayHeight', parseInt(trayEl.style.getPropertyValue('--tray-height'), 10) || ''),
    });
  }

  // --- the lightbox ----------------------------------------------------------------------------
  function openPicture(number) {
    box = { list: pictures, index: Math.max(0, pictures.findIndex(p => p.number === number)) };
    showBox();
    if (!$('lightbox').open) $('lightbox').showModal();
  }
  // A focus panel's picture: shown at once as the panel has it, then drawn again at the
  // lightbox's size (`large(edge)` gives that picture's address); no others to step through.
  function openRender(shown, large, label, caption) {
    box = { list: null, index: 0, large: '' };
    const image = $('lightbox-image');
    image.src = shown;
    image.alt = label;
    $('lightbox-label').textContent = label;
    $('lightbox-caption').textContent = caption;
    $('lightbox-count').textContent = '';
    $('lightbox-previous').hidden = $('lightbox-next').hidden = true;
    if (!$('lightbox').open) $('lightbox').showModal();
    const frame = image.parentElement.getBoundingClientRect();
    const edge = Math.min(2048, Math.ceil((Math.max(frame.width, frame.height) * (window.devicePixelRatio || 1)) / 128) * 128);
    const url = large(edge);
    box.large = url;
    const sharper = new Image();
    sharper.onload = () => { sharper.release(); if (box.large === url && $('lightbox').open) image.src = url; };
    sharper.onerror = () => sharper.release();
    sharper.release = Ui.loads.add(() => 0, () => { sharper.src = url; });
  }
  function showBox() {
    const picture = box.list?.[box.index];
    if (!picture) return;
    $('lightbox-previous').hidden = $('lightbox-next').hidden = false;
    $('lightbox-previous').disabled = box.index === 0;
    $('lightbox-next').disabled = box.index === box.list.length - 1;
    $('lightbox-image').src = url(picture);
    $('lightbox-image').alt = picture.caption || picture.label || '';
    $('lightbox-label').textContent = `Picture ${picture.number}` + (picture.sections.length === 1 ? `, ${Ui.short(picture.sections[0])}` : '');
    $('lightbox-count').textContent = `${box.index + 1} / ${box.list.length}`;
    $('lightbox-caption').textContent = picture.caption || picture.sections.join(', ');
  }
  function lightboxStep(delta) {
    if (!box.list) return;
    box.index = Math.max(0, Math.min(box.list.length - 1, box.index + delta));
    showBox();
  }

  return { init, setJob, setStatus, add, openRender };
})();
