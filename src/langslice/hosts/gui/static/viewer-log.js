'use strict';
// The agent log, the ABBA viewer's browser log as a column: the agent's messages and
// reasoning summaries, each tool call as a row (its name, its sections, its state;
// opened, its inputs, step and status), notes, and under a call that saved pictures a
// `[View N images]` link to them. The transcript follows its end while the person is
// there. Under it, the tool images tray: one call's pictures at a time, four per page,
// with `← n/N →` across the calls and Follow (the newest). The lightbox shows one picture
// as large as the window allows, `←` / `→` through every picture of the job.
const AgentLog = (() => {
  const PER_PAGE = 4;
  const SECTIONS_SHOWN = 5;
  const $ = id => document.getElementById(id);
  const { el } = Ui;

  let job = null;
  let calls = new Map();      // tool calls by key
  let claims = new Map();     // a view folder a call's end names -> that call
  let groups = new Map();     // pictures no call claims, by tool and call number
  let owners = [];            // the calls and groups with pictures, in picture order
  let pictures = [];          // every picture, by number
  let tray = { follow: true, owner: null, page: 0, shown: '' };
  let words = false;          // the agent's own words are in the folder (a trace)
  let activity = false;
  let box = { list: null, index: 0 };

  function init() {
    const scroll = $('log-scroll');
    scroll.addEventListener('scroll', () => { $('log-jump').hidden = atEnd(); });
    $('log-jump').addEventListener('click', () => { scroll.scrollTop = scroll.scrollHeight; $('log-jump').hidden = true; });
    $('log-expand').addEventListener('click', () => {
      const on = !$('log').classList.contains('expanded');
      $('log').classList.toggle('expanded', on);
      $('log-expand').setAttribute('aria-expanded', String(on));
      $('log-expand').textContent = on ? 'Last lines' : 'Whole log';
      scroll.scrollTop = scroll.scrollHeight;
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
    words = false; activity = false;
    $('log-items').replaceChildren();
    $('log-jump').hidden = true;
    $('log-empty').hidden = true;
    $('log-trace-note').hidden = true;
    renderTray();
  }
  function setStatus(word) {
    $('log-status').textContent = word;
    $('log-status').className = 'log-status word-' + word;
  }
  const atEnd = () => {
    const scroll = $('log-scroll');
    return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 60;
  };

  // --- the transcript ----------------------------------------------------------------------
  function add(items) {
    const follow = atEnd();
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
    if (follow && items.length) {
      const scroll = $('log-scroll');
      scroll.scrollTop = scroll.scrollHeight;
    }
  }
  // An agent message: plain text with paragraphs, `inline code` in violet, **bold** names
  // and list lines; nothing from the text is read as markup.
  function message(item) {
    words = true;
    const root = el('article', 'msg ' + (item.kind === 'agent' ? 'agent' : 'reasoning'));
    root.append(el('div', 'msg-label', item.kind === 'agent' ? 'Agent' : 'Reasoning summary'));
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
    if (/^Agent session/.test(item.text || '')) words = true;
    $('log-items').append(el('p', 'note' + (item.error ? ' error' : ''), item.text));
  }
  // A call's sections, the first five (then …), each one a way to put it in focus.
  function sectionsLine(ids) {
    const line = el('span', 'call-targets');
    (ids || []).slice(0, SECTIONS_SHOWN).forEach((id, i) => {
      if (i) line.append(document.createTextNode(', '));
      const link = el('span', 'section-link', id);
      link.setAttribute('role', 'button');
      link.tabIndex = 0;
      link.title = 'Show ' + id + ' in focus';
      const go = event => { event.preventDefault(); event.stopPropagation(); Viewer.pick(id); };
      link.addEventListener('click', go);
      link.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') go(event); });
      line.append(link);
    });
    if ((ids || []).length > SECTIONS_SHOWN) line.append(document.createTextNode(' …'));
    return line;
  }
  function toolItem(item) {
    activity = true;
    let call = calls.get(item.key);
    if (!call) {
      call = makeCall(item);
      calls.set(item.key, call);
      $('log-items').append(call.root);
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
  }
  function makeCall(item) {
    const root = el('details', 'call');
    const summary = el('summary');
    const body = el('span', 'call-body');
    body.append(el('span', 'call-name', item.name), sectionsLine(item.sections));
    const state = el('span', 'call-state');
    summary.append(body, state);
    const detail = el('div', 'call-detail');
    root.append(summary, detail);
    return { kind: 'call', key: item.key, name: item.name, sections: item.sections || [], root, state, detail, pictures: [], link: null, start: null, end: null };
  }
  function setState(call, word) {
    call.root.classList.toggle('running', word === 'running');
    call.root.classList.toggle('failed', word === 'failed');
    call.state.textContent = `[${word}]`;
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
    parts.push(el('p', 'call-meta', meta.join(' · ')));
    call.detail.replaceChildren(...parts);
  }

  // --- pictures ------------------------------------------------------------------------------
  function pictureItem(item) {
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
    const key = `${item.tool}:${item.call}`;
    let group = groups.get(key);
    if (!group) {
      const root = el('div', 'call static');
      const line = el('div', 'call-line');
      const body = el('span', 'call-body');
      const opening = item.tool === 'opening';
      body.append(el('span', 'call-name' + (opening ? ' plain' : ''), opening ? 'Opening pictures' : item.tool || 'pictures'));
      const targets = el('span', 'call-targets');
      body.append(targets);
      line.append(body, el('span', 'call-state', 'pictures'));
      root.append(line);
      group = { kind: 'group', key, name: item.tool, root, targets, ids: [], pictures: [], link: null };
      groups.set(key, group);
      // The opening pictures come first, after the line saying the job opened.
      const items = $('log-items');
      if (opening) {
        const before = [...groups.values()].filter(g => g !== group && g.name === 'opening').pop();
        const anchor = before ? (before.link || before.root) : items.firstElementChild?.classList.contains('note') ? items.firstElementChild : null;
        if (anchor) anchor.after(root); else items.prepend(root);
      } else items.append(root);
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
    owner.link.textContent = `[View ${n} image${n === 1 ? '' : 's'}]`;
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
    for (const owner of owners) owner.link?.classList.remove('shown');
    if (empty) {
      $('tray-tool').textContent = '· no pictures yet';
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
    $('tray-tool').textContent = '· ' + (owner.name === 'opening' ? 'opening' : owner.name || 'pictures');
    const total = owner.pictures.length;
    const from = batch.page * PER_PAGE + 1, to = from + batch.pictures.length - 1;
    const ids = owner.kind === 'call' ? owner.sections : owner.ids;
    const sections = ids.length > 3 ? `${ids.slice(0, 3).join(', ')} and ${ids.length - 3} more` : ids.join(', ');
    $('tray-caption').textContent = [sections, total > PER_PAGE ? `pictures ${from}-${to} of ${total}` : `${total} picture${total === 1 ? '' : 's'}`].filter(Boolean).join(' · ');
    $('tray-caption').title = $('tray-caption').textContent;
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
    const what = [picture.sections.length === 1 ? picture.sections[0] : picture.sections.length ? `${picture.sections.length} sections` : '', picture.mode].filter(Boolean).join(' · ');
    label.append(el('span', 'tile-number', String(picture.number)), document.createTextNode(what || picture.label || ''));
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
    sharper.onload = () => { if (box.large === url && $('lightbox').open) image.src = url; };
    sharper.src = url;
  }
  function showBox() {
    const picture = box.list?.[box.index];
    if (!picture) return;
    $('lightbox-previous').hidden = $('lightbox-next').hidden = false;
    $('lightbox-previous').disabled = box.index === 0;
    $('lightbox-next').disabled = box.index === box.list.length - 1;
    $('lightbox-image').src = url(picture);
    $('lightbox-image').alt = picture.caption || picture.label || '';
    $('lightbox-label').textContent = `Picture ${picture.number} · ${picture.tool || 'picture'}` + (picture.sections.length === 1 ? ` · ${picture.sections[0]}` : '');
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
