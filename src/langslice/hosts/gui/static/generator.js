'use strict';
// The prompt generator tab: a form that runs in the page and makes one message
// to paste into an agent app (a short request and the job's
// `langslice-job FOLDER init` command, which the register-brain skill runs).
// What the form offers comes from the code (GET generator: hosts/gui/generator.py).
// The message is built by the pure functions in `Generator` (tested under
// node by tests/test_gui_generator.py); the rest draws the form.

const Generator = (() => {
  const SEP = ':';
  const SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;
  const ORDERED = 'The file names sort in the order the sections were cut.';

  // The form as the page starts it: the default atlas and tasks, nothing else.
  function startForm(data) {
    return { folder: '', atlas: data.default_atlas, plane: '', thickness: '', interval: '', pixelSize: '',
             tasks: data.tasks.filter(t => t.default).map(t => t.name), off: [], ordered: false,
             damaged: '', keep: {}, notes: '', imageProvider: '', imageModel: '',
             abba: null, keepRegistrations: true, frozen: {} };
  }

  // The form started from a job's own settings (`prefill`, GET generator?job=n):
  // a job prepared in ABBA carries `abba`, which fixes the folder, atlas, plane
  // and pixel size, and keeps its registered sections unless unticked.
  function prefilled(data, prefill) {
    const form = startForm(data);
    if (!prefill) return form;
    const tasks = data.tasks.map(t => t.name).filter(name => (prefill.tasks || []).includes(name));
    return { ...form, folder: prefill.folder || '', atlas: prefill.atlas || '', plane: prefill.plane || '',
             thickness: prefill.thickness || '', interval: prefill.interval || '', pixelSize: prefill.pixel_size || '',
             tasks, off: [...(prefill.off || [])], ordered: (prefill.facts || []).includes(ORDERED),
             damaged: prefill.damaged || '', keep: { ...(prefill.keep || {}) }, notes: prefill.notes || '',
             frozen: { ...(prefill.frozen || {}) }, abba: prefill.abba || null };
  }

  // A word for a POSIX shell: as it is when it needs no quoting, else in
  // single quotes, or in double quotes when it holds a single quote and
  // nothing a double-quoted word expands.
  function shellQuote(value) {
    const s = String(value);
    if (SAFE.test(s)) return s;
    if (!s.includes("'")) return "'" + s + "'";
    if (!/["$`\\!]/.test(s)) return '"' + s + '"';
    return "'" + s.replace(/'/g, `'"'"'`) + "'";
  }

  // A pasted folder path without the quotes Windows' "Copy as path" adds, or
  // a file:// address's prefix.
  function cleanPath(raw) {
    let p = String(raw || '').trim();
    if (p.length >= 2 && ((p[0] === '"' && p.endsWith('"')) || (p[0] === "'" && p.endsWith("'")))) p = p.slice(1, -1).trim();
    if (p.startsWith('file://')) {
      try { p = decodeURIComponent(p.slice('file://'.length)); } catch { /* kept as typed */ }
    }
    return p;
  }

  const isOff = (off, name) => off.has(name) || off.has(name.split(SEP)[0]);

  function chosenTasks(data, form) {
    return data.tasks.map(t => t.name).filter(name => (form.tasks || []).includes(name));
  }
  // The tasks a run of this form has: the chosen ones, or init's default.
  function runTasks(data, form) {
    const chosen = chosenTasks(data, form);
    return chosen.length ? chosen : data.tasks.filter(t => t.default).map(t => t.name);
  }
  // The tools a run of this form has: its tasks', without the image model's
  // when the form says there is none.
  function present(data, form) {
    const key = runTasks(data, form).join(','), none = form.imageProvider === 'none';
    return new Set(data.tools.filter(t => t.runs_in.includes(key) && !(none && t.image_model)).map(t => t.name));
  }

  // Whether the image-model choice applies: the run warps, and a tool that
  // calls the image model is not switched off (whatever provider is chosen).
  function imageModelApplies(data, form) {
    const tasks = runTasks(data, form);
    if (!tasks.includes('nonlinear')) return false;
    const key = tasks.join(','), off = new Set(form.off || []), held = keptOff(data, form);
    return data.tools.some(t => t.image_model && t.runs_in.includes(key)
      && !(t.disableable && off.has(t.name)) && !held.has(t.name));
  }
  // The image-model flags: [setting, value] pairs, none when nothing is chosen.
  function imageModelFlags(data, form) {
    const provider = String(form.imageProvider || '').trim(), model = String(form.imageModel || '').trim();
    if (!provider || !imageModelApplies(data, form)) return [];
    const out = [['nonlinear.provider', provider]];
    if (model && provider !== 'none') out.push(['nonlinear.image_model', model]);
    return out;
  }

  // The names the job's `--disabled` takes: the tools of this run switched off
  // whole, then the options switched off of the tools still on (a name a kept
  // property already switches off left out).
  function offNames(data, form) {
    const off = new Set(form.off || []), here = present(data, form), held = keptOff(data, form), names = [];
    for (const tool of data.tools) {
      if (!here.has(tool.name) || held.has(tool.name)) continue;
      if (tool.disableable && off.has(tool.name)) { names.push(tool.name); continue; }
      for (const option of tool.options) if (off.has(option.name) && !held.has(option.name)) names.push(option.name);
    }
    return names;
  }

  function labelOf(data, name) {
    for (const tool of data.tools) {
      if (tool.name === name) return tool.label;
      for (const option of tool.options) if (option.name === name) return `${tool.label}: ${option.label}`;
    }
    return name;
  }

  function damagedMap(text) {
    const out = {};
    for (const raw of String(text || '').split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      const at = line.indexOf(':');
      const file = (at < 0 ? line : line.slice(0, at)).trim();
      const note = at < 0 ? '' : line.slice(at + 1).trim();
      if (file) out[file] = note || 'damaged';
    }
    return out;
  }

  const blank = value => String(value ?? '').trim() === '';
  const wholeNumber = value => /^[1-9][0-9]*$/.test(String(value).trim());
  const positiveNumber = value => /^([0-9]+\.?[0-9]*|\.[0-9]+)$/.test(String(value).trim()) && Number(value) > 0;

  // What stops the message: one sentence each (the spec would refuse the
  // same tool choices when the job is made).
  function problems(data, form) {
    const out = [];
    if (blank(cleanPath(form.folder))) out.push('Enter the folder that holds the section images.');
    if (!blank(form.thickness) && !wholeNumber(form.thickness)) out.push('Section thickness is a whole number of micrometres (µm).');
    if (!blank(form.interval) && !wholeNumber(form.interval)) out.push('Section interval is a whole number of micrometres (µm).');
    if (!form.abba && !blank(form.pixelSize) && !positiveNumber(form.pixelSize)) out.push('Pixel size is a number of micrometres per pixel, above 0.');
    const off = new Set(offNames(data, form)), here = present(data, form);
    const look = data.tools.find(t => t.name === 'look');
    if (look && look.options.length && look.options.every(o => isOff(off, o.name))) {
      out.push(`"${look.label}" needs at least one kind of picture: turn one back on.`);
    }
    for (const task of runTasks(data, form)) {
      const writers = data.task_writers[task] || [];
      if (writers.length && !writers.some(name => here.has(name) && !isOff(off, name))) {
        const label = data.tasks.find(t => t.name === task).label;
        out.push(`"${label}" needs one of these tools on: ${writers.map(name => `"${labelOf(data, name)}"`).join(', ')}.`);
      }
    }
    const needing = new Map();  // a needed group, joined -> the tools on that need it
    for (const tool of data.tools) {
      if (!here.has(tool.name) || isOff(off, tool.name)) continue;
      for (const group of tool.needs) {
        if (!group.every(name => isOff(off, name))) continue;
        const key = group.join('|');
        needing.set(key, [...(needing.get(key) || []), tool.label]);
      }
    }
    for (const [key, users] of needing) {
      const needed = key.split('|').map(name => `"${labelOf(data, name)}"`).join(' or ');
      const quoted = users.map(label => `"${label}"`);
      out.push(`${joinWords(quoted)} ${users.length > 1 ? 'need' : 'needs'} ${needed}: turn it back on, or turn ${users.length > 1 ? 'those' : 'that'} off too.`);
    }
    return out;
  }

  // What the person should know but may send anyway.
  function warnings(data, form) {
    const chosen = chosenTasks(data, form);
    if (chosen.length === 1 && chosen[0] === 'nonlinear') {
      return ['Warping alone starts from sections already placed and aligned: say in the notes where that registration is (a QuickNII, VisuAlign, DeepSlice or LangSlice file).'];
    }
    return [];
  }

  function leftOut(data, form) {
    const out = [];
    const fixed = Boolean(form.abba);  // ABBA's job already holds these
    if (!chosenTasks(data, form).length) out.push('what to do (place, align, warp)');
    if (!fixed && blank(form.atlas)) out.push('the atlas');
    if (!fixed && blank(form.plane)) out.push('the cutting plane');
    if (blank(form.thickness)) out.push('the section thickness');
    if (blank(form.interval)) out.push('the section interval');
    if (!fixed && blank(form.pixelSize)) out.push('the pixel size (when the image files do not record it)');
    return out;
  }

  // The job's `--frozen`, or null for none: the kept properties on every
  // section, and a job prepared in ABBA's own per-section freezes (its
  // registered sections) while they are kept; unticked, `{}` frees them.
  function frozenFlag(data, form) {
    const keep = kept(data, form), saved = form.frozen || {};
    const freed = Boolean(form.abba) && !form.keepRegistrations && Object.keys(saved).length > 0;
    const sections = freed ? {} : saved;
    if (keep.length) return JSON.stringify({ '*': keep, ...sections });
    return freed ? '{}' : null;
  }

  // The properties kept on every section (the job's `--frozen "*"`): each
  // box ticked whose task is chosen (the property's task off keeps it anyway).
  function kept(data, form) {
    const tasks = chosenTasks(data, form);
    return Object.keys(data.keep).filter(prop => form.keep && form.keep[prop] && tasks.includes(data.keep[prop]));
  }
  // The tools and options a kept property leaves nothing to do.
  function keptOff(data, form) {
    return new Set(kept(data, form).flatMap(prop => data.frozen_off[prop] || []));
  }

  // The command, each flag spelled as the settings table spells it (`data.flags`,
  // setting name -> flag).
  function command(data, form) {
    const args = [[shellQuote(cleanPath(form.folder)), 'init']];
    const flag = (setting, value) => args.push([data.flags[setting], shellQuote(value)]);
    const tasks = chosenTasks(data, form);
    if (tasks.length) flag('tasks', tasks.join(','));
    const fixed = Boolean(form.abba);  // ABBA's job already holds these
    if (!fixed && !blank(form.atlas)) flag('atlas', form.atlas.trim());
    if (!fixed && !blank(form.plane)) flag('plane', form.plane.trim());
    if (!blank(form.thickness)) flag('position.thickness_um', String(form.thickness).trim());
    if (!blank(form.interval)) flag('position.interval_um', String(form.interval).trim());
    if (!fixed && !blank(form.pixelSize)) flag('inputs.pixel_size_um', String(form.pixelSize).trim());
    const off = offNames(data, form);
    if (off.length) flag('disabled', off.join(','));
    for (const [setting, value] of imageModelFlags(data, form)) flag(setting, value);
    const frozen = frozenFlag(data, form);
    if (frozen !== null) flag('frozen', frozen);
    const damaged = damagedMap(form.damaged);
    if (Object.keys(damaged).length) flag('inputs.damaged', JSON.stringify(damaged));
    if (form.ordered) flag('facts', ORDERED);
    if (!blank(form.notes)) flag('notes', String(form.notes).trim());
    return { text: 'langslice-job ' + args.map(words => words.join(' ')).join(' \\\n  '),
             quoted: args.some(words => words.some(word => /^['"]/.test(word))) };
  }

  function joinWords(items) {
    return items.length < 2 ? items.join('') : items.slice(0, -1).join(', ') + ' and ' + items[items.length - 1];
  }

  // The message: `{text, problems, warnings}`; `text` is empty while a
  // problem stands.
  function message(data, form) {
    const stops = problems(data, form);
    const notes = warnings(data, form);
    if (stops.length) return { text: '', problems: stops, warnings: notes };
    const cmd = command(data, form);
    const lines = ['Please register my brain sections with LangSlice, following its register-brain skill: create the job with the init command below, then launch the registration and check the result.'];
    const missing = leftOut(data, form);
    if (missing.length) lines.push(`The command leaves out ${joinWords(missing)}: choose ${missing.length > 1 ? 'these' : 'it'} yourself, or ask me when the images do not tell you.`);
    if (cmd.quoted) lines.push('The command is quoted for bash; if your shell is another, quote it for yours.');
    const text = lines.join(' ') + '\n\n```bash\n' + cmd.text + '\n```\n';
    return { text, problems: stops, warnings: notes };
  }

  // The From ABBA block's keep box, for n registered sections.
  function keepLabel(n) {
    return n === 1
      ? 'Keep the 1 section’s existing ABBA registrations (its position may still move)'
      : `Keep the ${n} sections’ existing ABBA registrations (their positions may still move)`;
  }

  return { shellQuote, cleanPath, startForm, prefilled, offNames, present, keptOff, imageModelApplies,
           problems, warnings, leftOut, command, message, keepLabel };
})();

if (typeof module !== 'undefined') module.exports = Generator;

// --- the form -------------------------------------------------------------------
// Numbered black cards on the left (the form), the message on the right: sticky, as tall
// as the window allows, with the Copy button. Built once from GET generator; update()
// redraws what depends on the form after every change. Every text goes in by textContent.
if (typeof document !== 'undefined') (() => {
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const root = document.getElementById('tab-prompt');
  if (!root) return;
  let data = null;
  const form = {};  // Generator.startForm / prefilled, filled in by build()
  const toolRows = [];  // [{tool, input, row, note, options: [{option, input, row, note}]}]
  const groupRows = [];  // [{group, list, note}]
  const fixedControls = [];  // the inputs a job prepared in ABBA fixes
  const ui = {};  // the controls update() redraws
  const KEPT = 'Off: you said what this would change stays as it is.';
  let ids = 0, message = '', copiedTimer = 0;

  const capital = word => (word ? word[0].toUpperCase() + word.slice(1) : '');
  const fixed = control => { fixedControls.push(control); return control; };
  const nextId = () => `gen-${++ids}`;

  // A numbered card, `1 Images`; the From ABBA card has no number.
  function card(step, title, cls) {
    const box = make('section', 'gen-card' + (cls ? ' ' + cls : ''));
    const head = make('h2', 'gen-card-title');
    if (step) head.append(make('span', 'gen-step', String(step)));
    head.append(make('span', '', title));
    box.append(head);
    return box;
  }

  // A label above its control (`target`, when the control is a wrapper), a help line under it.
  function field(text, control, help, target = control) {
    const wrap = make('div', 'gen-field');
    let label;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) {
      label = make('label', 'gen-label', text);
      target.id = target.id || nextId();
      label.htmlFor = target.id;
    } else {
      label = make('p', 'gen-label', text);
      label.id = nextId();
      target.setAttribute('aria-labelledby', label.id);
    }
    wrap.append(label, control);
    if (help) wrap.append(make('p', 'gen-help', help));
    return wrap;
  }
  function textInput(key, { placeholder, rows, mode, path } = {}) {
    const box = make(rows ? 'textarea' : 'input', 'gen-input' + (path ? ' gen-path' : ''));
    if (rows) box.rows = rows; else box.type = 'text';
    if (placeholder) box.placeholder = placeholder;
    if (mode) box.inputMode = mode;
    box.spellcheck = false;
    box.autocomplete = 'off';
    box.value = form[key];
    box.addEventListener('input', () => { form[key] = box.value; update(); });
    return box;
  }
  // A short number field with its unit beside it.
  function unitInput(key, unit, placeholder, mode) {
    const box = textInput(key, { placeholder, mode });
    const wrap = make('span', 'gen-unit');
    wrap.append(box, make('span', 'gen-unit-name', unit));
    wrap.addEventListener('mousedown', e => { if (e.target !== box && !box.disabled) { e.preventDefault(); box.focus(); } });
    return { wrap, box };
  }

  // A checkbox drawn as a box or as a switch; the native input stays, for the keyboard
  // and screen readers.
  function toggle(kind, checked) {
    const input = make('input', 'gen-native');
    input.type = 'checkbox';
    input.checked = checked;
    if (kind === 'switch') input.setAttribute('role', 'switch');
    const mark = make('span', kind === 'switch' ? 'gen-switch' : 'gen-box');
    mark.setAttribute('aria-hidden', 'true');
    return { input, mark };
  }
  // A ticked fact: its box, its words, and a note under them.
  function check(text, help, checked, onChange) {
    const { input, mark } = toggle('box', checked);
    const row = make('label', 'gen-check');
    const words = make('span', 'gen-check-words');
    words.append(make('span', 'gen-check-text', text));
    if (help) words.append(make('span', 'gen-help', help));
    row.append(input, mark, words);
    input.addEventListener('change', () => { onChange(input.checked); update(); });
    return { row, input };
  }

  function atlasSelect() {
    const box = make('select', 'gen-input gen-select');
    box.append(new Option('Not set: the agent chooses or asks', ''));
    for (const group of data.species) {
      const og = make('optgroup'); og.label = group;
      for (const atlas of data.atlases.filter(a => a.species === group)) {
        og.append(new Option(atlas.name + (atlas.name === data.default_atlas ? ' (default)' : ''), atlas.name));
      }
      if (og.children.length) box.append(og);
    }
    box.value = form.atlas;
    box.addEventListener('change', () => { form.atlas = box.value; update(); });
    return box;
  }

  // BrainGlobe's paragraph on the chosen atlas: three lines, until `more` opens it.
  function atlasAbout() {
    const wrap = make('div', 'gen-about');
    const text = make('p', 'gen-about-text gen-clamped');
    const more = make('button', 'gen-link', 'more');
    more.type = 'button';
    more.hidden = true;
    more.setAttribute('aria-expanded', 'false');
    const measure = () => {
      if (text.classList.contains('gen-clamped')) more.hidden = text.scrollHeight <= text.clientHeight + 1;
    };
    more.addEventListener('click', () => {
      const clamped = text.classList.toggle('gen-clamped');
      more.textContent = clamped ? 'more' : 'less';
      more.setAttribute('aria-expanded', String(!clamped));
      measure();
    });
    if ('ResizeObserver' in window) new ResizeObserver(measure).observe(text);  // measured once shown
    wrap.append(text, more);
    ui.about = { text, more, words: null, measure };
    return wrap;
  }
  function showAbout(words) {
    const about = ui.about;
    if (about.words === words) return;
    about.words = words;
    about.text.textContent = words;
    about.text.classList.add('gen-clamped');
    about.more.textContent = 'more';
    about.more.setAttribute('aria-expanded', 'false');
    about.measure();
  }

  // The cutting plane: one of the three, or none (the agent decides).
  function planeControl() {
    const wrap = make('div', 'gen-plane');
    const group = make('div', 'gen-segments');
    group.setAttribute('role', 'group');
    const buttons = data.planes.map(plane => {
      const button = fixed(make('button', 'gen-segment', capital(plane)));
      button.type = 'button';
      button.addEventListener('click', () => { form.plane = form.plane === plane ? '' : plane; update(); });
      group.append(button);
      return [plane, button];
    });
    const unset = make('span', 'gen-plane-note', 'Not set: the agent decides.');
    const clear = fixed(make('button', 'gen-link', 'Let the agent decide'));
    clear.type = 'button';
    clear.addEventListener('click', () => { form.plane = ''; update(); });
    wrap.append(group, unset, clear);
    ui.plane = { group, buttons, unset, clear };
    return wrap;
  }

  function folderHelp() {
    const d = make('details', 'gen-more');
    d.append(make('summary', '', 'How to copy a folder’s path'));
    const list = make('ul', 'gen-howto');
    for (const [os, how] of [
      ['Windows', 'In File Explorer, right-click the folder (on Windows 10, hold Shift first) and choose “Copy as path”.'],
      ['macOS', 'In Finder, select the folder, hold Option, right-click it and choose “Copy … as Pathname”.'],
      ['Linux', 'In the Files app, select the folder and press Ctrl+C; pasting it here gives its path.'],
    ]) { const li = make('li'); li.append(make('strong', '', os + ': '), how); list.append(li); }
    d.append(list, make('p', 'gen-help', 'Paste the path into the box above. The quotes Windows adds are taken off.'));
    return d;
  }

  // A job prepared in ABBA: what ABBA fixed, the snapshots' slice names, and whether the
  // sections ABBA already registered stay as they are.
  function abbaCard() {
    const abba = form.abba;
    const box = card(null, 'From ABBA', 'gen-abba');
    const facts = make('dl', 'gen-facts');
    const atlas = make('dd');
    atlas.append(make('span', 'gen-path', form.atlas));
    if (abba.atlas) atlas.append(make('span', 'gen-muted', ` (in ABBA: ${abba.atlas})`));
    facts.append(make('dt', '', 'Images'), make('dd', 'gen-path', form.folder),
      make('dt', '', 'Atlas'), atlas,
      make('dt', '', 'Cutting plane'), make('dd', '', capital(form.plane)),
      make('dt', '', 'Snapshot pixel size'), make('dd', '', form.pixelSize ? `${form.pixelSize} µm per pixel` : 'not recorded'));
    box.append(facts, make('p', 'gen-help', 'ABBA set these when it exported the snapshots; the job already holds them, so the message leaves them out.'));
    const registered = abba.registered || [];
    if (registered.length && Object.keys(form.frozen).length) {
      const n = registered.length;
      box.append(check(Generator.keepLabel(n),
        n === 1 ? 'Untick to let the agent redo its alignment and warp too.' : 'Untick to let the agent redo their alignment and warps too.',
        !!form.keepRegistrations, on => { form.keepRegistrations = on; }).row);
    }
    const names = Object.entries(abba.sections || {});
    if (names.length) {
      const d = make('details', 'gen-more');
      d.open = names.length <= 12;
      d.append(make('summary', '', `Snapshot files and their ABBA slices (${names.length})`));
      const wrap = make('div', 'gen-slice-wrap'), table = make('table', 'gen-slice-table');
      const head = make('tr');
      for (const title of ['File', 'ABBA slice', 'In ABBA']) head.append(make('th', '', title));
      const thead = make('thead'); thead.append(head);
      const body = make('tbody');
      const warped = new Set(abba.existing_warp || []), done = new Set(registered);
      for (const [file, slice] of names) {
        const tr = make('tr');
        const status = warped.has(file) ? 'registered, warped' : done.has(file) ? 'registered' : '';
        tr.append(make('td', 'gen-path', file), make('td', '', slice), make('td', 'gen-muted', status));
        body.append(tr);
      }
      table.append(thead, body);
      wrap.append(table);
      d.append(wrap, make('p', 'gen-help', 'Name a damaged section below by its file name.'));
      box.append(d);
    }
    return box;
  }

  // The three tasks, as cards to switch on and off.
  function taskCards() {
    const grid = make('div', 'gen-tasks');
    ui.tasks = data.tasks.map(task => {
      const { input, mark } = toggle('box', form.tasks.includes(task.name));
      const tile = make('label', 'gen-task');
      const head = make('span', 'gen-task-head');
      head.append(mark, make('span', 'gen-task-label', task.label));
      tile.append(input, head, make('span', 'gen-task-about', task.about));
      input.addEventListener('change', () => {
        form.tasks = data.tasks.map(t => t.name).filter(n => (n === task.name ? input.checked : form.tasks.includes(n)));
        update();
      });
      grid.append(tile);
      return { tile, input };
    });
    return grid;
  }

  // One row of the tools table: its switch, its label (and a note), its own name.
  function toolRow(label, name, on, cls) {
    const { input, mark } = toggle('switch', on);
    input.dataset.name = name;
    const row = make('label', 'gen-tool' + (cls ? ' ' + cls : ''));
    const words = make('span', 'gen-tool-words');
    const note = make('span', 'gen-tool-note');
    words.append(make('span', 'gen-tool-label', label), note);
    row.append(input, mark, words, make('code', 'gen-tool-name', name));
    return { row, input, note };
  }
  function setOff(name, off) {
    const names = new Set(form.off);
    if (off) names.add(name); else names.delete(name);
    form.off = [...names];
  }
  function toolsPanel() {
    const d = make('details', 'gen-card gen-advanced');
    const summary = make('summary', 'gen-card-title');
    summary.append(make('span', 'gen-step gen-disclosure'), make('span', '', 'Advanced: the tools the registration agent may use'));
    d.append(summary);
    const body = make('div', 'gen-advanced-body');
    body.append(make('p', 'gen-help', 'Every tool is on unless you turn it off. A task’s tools apply only when that task is chosen above. The violet name is the tool’s own name, as the Agent viewer shows it.'));
    for (const group of data.groups) {
      const list = make('div', 'gen-tools');
      list.setAttribute('role', 'group');
      const head = make('div', 'gen-tools-head');
      const title = make('span', 'gen-tools-title', group.label);
      title.id = nextId();
      list.setAttribute('aria-labelledby', title.id);
      const note = make('span', 'gen-tools-note');
      head.append(title, note);
      list.append(head);
      for (const tool of data.tools.filter(t => t.group === group.name)) {
        const row = toolRow(tool.label, tool.name, !form.off.includes(tool.name), tool.options.length ? 'gen-parent' : '');
        if (!tool.disableable) { row.input.disabled = true; row.input.title = 'Every run needs this tool.'; row.row.classList.add('gen-fixed'); }
        row.input.addEventListener('change', () => { setOff(tool.name, !row.input.checked); update(); });
        list.append(row.row);
        const options = tool.options.map((option, i) => {
          const o = toolRow(option.label, option.name, !form.off.includes(option.name),
            'gen-option' + (i === tool.options.length - 1 ? ' gen-last' : ''));
          o.input.addEventListener('change', () => { setOff(option.name, !o.input.checked); update(); });
          list.append(o.row);
          return { option, ...o };
        });
        toolRows.push({ tool, ...row, options });
      }
      groupRows.push({ group, list, note });
      body.append(list);
      if (group.task === 'nonlinear' && !ui.image) body.append(imageModelPanel());
    }
    if (!ui.image) body.append(imageModelPanel());
    d.append(body);
    return d;
  }

  // The image model the warp's border tracing asks (`--nonlinear-provider`,
  // `--nonlinear-image-model`): the providers and models LangSlice lists.
  function imageModelPanel() {
    const box = make('div', 'gen-tools gen-image');
    box.setAttribute('role', 'group');
    const head = make('div', 'gen-tools-head');
    const title = make('span', 'gen-tools-title', 'Image model');
    title.id = nextId();
    box.setAttribute('aria-labelledby', title.id);
    const note = make('span', 'gen-tools-note');
    head.append(title, note);
    const tracer = data.tools.find(t => t.image_model);
    const about = make('p', 'gen-help', `The model that draws a section’s borders for ${tracer ? `“${tracer.label}”` : 'the warp'}. Applies when warping to fit.`);
    const provider = make('select', 'gen-input gen-select');
    const fallback = data.image_models.find(p => p.default);
    provider.append(new Option(`Not set: LangSlice’s default${fallback ? ` (${fallback.label})` : ''}`, ''));
    for (const p of data.image_models) provider.append(new Option(p.label, p.provider));
    provider.value = form.imageProvider;
    const model = make('select', 'gen-input gen-select');
    const hint = make('p', 'gen-help');
    function models() {
      const chosen = data.image_models.find(p => p.provider === form.imageProvider);
      model.replaceChildren(new Option(chosen && chosen.default_model ? `The provider’s default (${chosen.default_model})` : 'The provider’s default', ''));
      for (const name of chosen ? chosen.models : []) model.append(new Option(name, name));
      model.value = form.imageModel;
      if (!chosen || chosen.provider === 'none') hint.textContent = chosen ? 'The warp fits each section to its stain alone.' : '';
      else hint.textContent = chosen.connected ? 'Set up on this computer.' : 'Not set up on this computer yet: Settings connects it.';
    }
    provider.addEventListener('change', () => { form.imageProvider = provider.value; form.imageModel = ''; models(); update(); });
    model.addEventListener('change', () => { form.imageModel = model.value; update(); });
    models();
    const pair = make('div', 'gen-pair');
    pair.append(field('Provider', provider), field('Model', model));
    box.append(head, about, pair, hint);
    ui.image = { box, note, provider, model };
    return box;
  }

  function outputPanel() {
    const panel = make('aside', 'gen-output');
    panel.setAttribute('aria-label', 'Your message');
    const head = make('div', 'gen-output-head');
    ui.copy = make('button', 'gen-copy', 'Copy');
    ui.copy.type = 'button';
    ui.copy.addEventListener('click', copy);
    head.append(make('h2', 'gen-output-title', 'Your message'), ui.copy);
    ui.problems = make('ul', 'gen-problems');
    ui.warnings = make('ul', 'gen-warnings');
    ui.output = make('pre', 'gen-message');
    ui.output.tabIndex = 0;
    ui.empty = make('p', 'gen-message-empty', 'Your message appears here.');
    panel.append(head, ui.problems, ui.warnings, ui.output, ui.empty,
      make('p', 'gen-help', 'Paste it into a new chat in your agent app (Claude Code, Codex or Antigravity) with LangSlice installed. The agent sets up the job, asks about anything left open, and starts the registration; follow it in the Agent viewer.'));
    return panel;
  }

  // The message, word for word as it is copied: the request, then the command block (its
  // fences dim, the command in violet).
  function showMessage(text) {
    if (text !== message) resetCopied();
    message = text;
    ui.output.hidden = !text;
    ui.empty.hidden = Boolean(text);
    ui.copy.disabled = !text;
    const open = '\n\n```bash\n', at = text.indexOf(open), close = text.lastIndexOf('\n```');
    if (!text || at < 0 || close <= at) { ui.output.textContent = text; return; }
    ui.output.replaceChildren(make('span', 'gen-request', text.slice(0, at)), '\n\n',
      make('span', 'gen-fence', '```bash'), '\n',
      make('span', 'gen-command', text.slice(at + open.length, close)), '\n',
      make('span', 'gen-fence', '```'), text.slice(close + '\n```'.length));
    if (ui.output.textContent !== text) ui.output.textContent = text;  // never other words than the copy's
  }

  async function copy() {
    if (!message) return;
    try { await navigator.clipboard.writeText(message); }
    catch {
      const range = document.createRange(); range.selectNodeContents(ui.output);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      document.execCommand('copy'); sel.removeAllRanges();
    }
    ui.copy.textContent = 'Copied';
    ui.copy.classList.add('gen-copied');
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(resetCopied, 2000);
  }
  function resetCopied() {
    clearTimeout(copiedTimer);
    if (!ui.copy) return;
    ui.copy.textContent = 'Copy';
    ui.copy.classList.remove('gen-copied');
  }

  function update() {
    if (!data) return;
    const here = Generator.present(data, form), off = new Set(form.off), keptOff = Generator.keptOff(data, form);
    const noImageModel = form.imageProvider === 'none';
    for (const { group, list, note } of groupRows) {
      const inRun = data.tools.some(t => t.group === group.name && here.has(t.name));
      const task = data.tasks.find(t => t.name === group.task);
      list.classList.toggle('gen-absent', !inRun);
      note.textContent = inRun || !task ? '' : `Not in this run: “${task.label}” is not chosen above.`;
    }
    for (const { tool, input, row, note, options } of toolRows) {
      const inRun = here.has(tool.name), held = keptOff.has(tool.name);
      const noModel = !inRun && noImageModel && tool.image_model;
      row.classList.toggle('gen-absent', !inRun);
      row.title = inRun ? '' : (noModel ? 'Not in this run: you chose no image model.' : 'Not in this run: its task is not chosen.');
      if (tool.disableable) {
        input.disabled = !inRun || held;
        input.checked = !held && !off.has(tool.name);
      }
      note.textContent = !tool.disableable ? 'always on' : noModel ? 'not in this run: you chose no image model' : held ? 'off: you said this stays as it is' : '';
      const toolOff = tool.disableable && off.has(tool.name);
      for (const o of options) {
        const kept = keptOff.has(o.option.name);
        o.input.disabled = !inRun || kept || toolOff;
        o.input.checked = !kept && !toolOff && !off.has(o.option.name);  // shown off while the tool is off
        o.row.classList.toggle('gen-absent', !inRun);
        o.row.title = kept ? KEPT : '';
        o.note.textContent = kept ? 'off: you said this stays as it is' : '';
      }
    }
    const imageOn = Generator.imageModelApplies(data, form);
    const imageOff = 'Not in this run: warping to fit is not chosen, or its border tracing is off.';
    ui.image.box.classList.toggle('gen-absent', !imageOn);
    ui.image.box.title = imageOn ? '' : imageOff;
    ui.image.note.textContent = imageOn ? '' : imageOff;
    ui.image.provider.disabled = !imageOn;
    ui.image.model.disabled = !imageOn || !form.imageProvider || noImageModel;
    for (const { tile, input } of ui.tasks) tile.classList.toggle('gen-on', input.checked);
    const tasks = new Set(form.tasks);
    for (const [prop, item] of Object.entries(ui.keep)) item.input.disabled = !tasks.has(data.keep[prop]);
    for (const control of fixedControls) control.disabled = Boolean(form.abba);
    for (const wrap of ui.units) wrap.classList.toggle('gen-fixed', wrap.firstChild.disabled);
    for (const [plane, button] of ui.plane.buttons) {
      button.classList.toggle('gen-on', form.plane === plane);
      button.setAttribute('aria-pressed', String(form.plane === plane));
    }
    ui.plane.unset.hidden = Boolean(form.plane);
    ui.plane.clear.hidden = !form.plane || Boolean(form.abba);
    const atlas = data.atlases.find(a => a.name === form.atlas);
    showAbout(atlas ? atlas.about : 'The agent picks the atlas, or asks you.');
    const made = Generator.message(data, form);
    ui.problems.replaceChildren(...made.problems.map(p => make('li', '', p)));
    ui.warnings.replaceChildren(...made.warnings.map(w => make('li', '', w)));
    showMessage(made.text);
  }

  function build(prefill) {
    Object.assign(form, Generator.prefilled(data, prefill));
    const page = make('div', 'gen-page'), main = make('div', 'gen-form');
    main.append(make('p', 'gen-intro', form.abba
      ? 'ABBA prepared this job: its snapshots, positions and cutting angles are in it. Add what you know, copy the message, and paste it into your agent app.'
      : 'Fill in what you know, copy the message, and paste it into your agent app. Only the image folder is needed: anything left blank, the agent decides or asks you about.'));
    if (form.abba) main.append(abbaCard());

    const fixedHelp = form.abba ? 'Set by ABBA.' : undefined;
    const images = card(1, 'Images');
    images.append(field('Folder of section images (required)',
      fixed(textInput('folder', { placeholder: 'e.g. C:\\Users\\me\\brain1 or /home/me/brain1', path: true })), fixedHelp));
    if (!form.abba) images.append(folderHelp());
    main.append(images);

    const atlas = card(2, 'Atlas');
    atlas.append(field('Atlas', fixed(atlasSelect())), atlasAbout(), field('Cutting plane', planeControl(), fixedHelp, ui.plane.group));
    main.append(atlas);

    const cut = card(3, 'Sections');
    const thickness = unitInput('thickness', 'µm', 'e.g. 40', 'numeric');
    const interval = unitInput('interval', 'µm', 'e.g. 120', 'numeric');
    const pixel = unitInput('pixelSize', 'µm per pixel', 'e.g. 0.65', 'decimal');
    fixed(pixel.box);
    ui.units = [thickness.wrap, interval.wrap, pixel.wrap];
    const units = make('div', 'gen-units');
    units.append(
      field('Thickness', thickness.wrap, form.abba && form.thickness ? 'ABBA’s, from its slice spacing: check it.' : undefined, thickness.box),
      field('Interval', interval.wrap, 'From one kept section to the next: every third 40 µm section is 120 µm.', interval.box),
      field('Pixel size', pixel.wrap, form.abba ? 'ABBA’s snapshots.' : 'Only if the image files do not record it.', pixel.box));
    cut.append(units);
    main.append(cut);

    const todo = card(4, 'What to do');
    todo.append(taskCards());
    main.append(todo);

    const known = card(5, 'What you already know');
    const checks = make('div', 'gen-checks');
    ui.keep = {
      angles: check('The sections were cut straight: do not tilt the cutting plane', 'Applies when placing along the atlas.',
        !!form.keep.angles, on => { form.keep = { ...form.keep, angles: on }; }),
      orientation: check('Every section is the right way up and not mirrored: do not flip or turn them', 'Applies when aligning in the plane.',
        !!form.keep.orientation, on => { form.keep = { ...form.keep, orientation: on }; }),
    };
    checks.append(check('The file names sort in the order the sections were cut', '', !!form.ordered, on => { form.ordered = on; }).row,
      ui.keep.angles.row, ui.keep.orientation.row);
    known.append(checks, field('Damaged sections',
      textInput('damaged', { rows: 3, placeholder: 'One per line: file name: what is wrong\ne.g. s012.tif: left cortex torn' })));
    main.append(known);

    const notes = card(6, 'Notes for the agent');
    const notesBox = textInput('notes', { rows: 3, placeholder: 'e.g. Injection in the right hemisphere; the olfactory bulbs were removed.' });
    notesBox.setAttribute('aria-label', 'Notes for the agent');
    notes.append(notesBox, make('p', 'gen-help', 'Saved with the job; the registration agent reads them.'));
    main.append(notes, toolsPanel());

    page.append(main, outputPanel());
    root.replaceChildren(page);
    update();
  }

  // The job the window was opened on (`#tab=prompt&job=<n>`, langslice gui
  // FOLDER --tab prompt): the form starts from its job.json.
  function startJob() {
    const job = new URLSearchParams(location.hash.slice(1)).get('job');
    return job !== null && /^[0-9]+$/.test(job) ? job : null;
  }

  async function load() {
    try {
      const job = startJob();
      const r = await fetch(job === null ? 'generator' : `generator?job=${job}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(String(r.status));
      data = await r.json();
      build(data.prefill || null);
    } catch (error) {
      console.error(error);
      root.replaceChildren(make('p', 'gen-intro gen-failed', 'The prompt generator could not load its choices from LangSlice. Close the window and open it again.'));
    }
  }
  load();
})();
