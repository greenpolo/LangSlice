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
  function present(data, form) {
    const key = runTasks(data, form).join(',');
    return new Set(data.tools.filter(t => t.runs_in.includes(key)).map(t => t.name));
  }

  // The names the job's `--disable` takes: the tools of this run switched off
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
    if (!blank(form.pixelSize) && !positiveNumber(form.pixelSize)) out.push('Pixel size is a number of micrometres per pixel, above 0.');
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
    if (!chosenTasks(data, form).length) out.push('what to do (place, align, warp)');
    if (blank(form.atlas)) out.push('the atlas');
    if (blank(form.plane)) out.push('the cutting plane');
    if (blank(form.thickness)) out.push('the section thickness');
    if (blank(form.interval)) out.push('the section interval');
    if (blank(form.pixelSize)) out.push('the pixel size (when the image files do not record it)');
    return out;
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

  function command(data, form) {
    const args = [[shellQuote(cleanPath(form.folder)), 'init']];
    const flag = (name, value) => args.push([name, shellQuote(value)]);
    const tasks = chosenTasks(data, form);
    if (tasks.length) flag('--tasks', tasks.join(','));
    if (!blank(form.atlas)) flag('--atlas', form.atlas.trim());
    if (!blank(form.plane)) flag('--plane', form.plane.trim());
    if (!blank(form.thickness)) flag('--thickness', String(form.thickness).trim());
    if (!blank(form.interval)) flag('--interval', String(form.interval).trim());
    if (!blank(form.pixelSize)) flag('--pixel-size-um', String(form.pixelSize).trim());
    const off = offNames(data, form);
    if (off.length) flag('--disable', off.join(','));
    const keep = kept(data, form);
    if (keep.length) flag('--frozen', JSON.stringify({ '*': keep }));
    const damaged = damagedMap(form.damaged);
    if (Object.keys(damaged).length) flag('--damaged', JSON.stringify(damaged));
    if (form.ordered) flag('--fact', 'The file names sort in the order the sections were cut.');
    if (!blank(form.notes)) flag('--notes', String(form.notes).trim());
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

  return { shellQuote, cleanPath, offNames, present, keptOff, problems, warnings, leftOut, command, message };
})();

if (typeof module !== 'undefined') module.exports = Generator;

// --- the form -------------------------------------------------------------------
if (typeof document !== 'undefined') (() => {
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const root = document.getElementById('tab-prompt');
  if (!root) return;
  let data = null;
  const form = { folder: '', atlas: '', plane: '', thickness: '', interval: '', pixelSize: '',
                 tasks: [], off: [], ordered: false, damaged: '', keep: {}, notes: '' };
  const toolBoxes = [];  // [{tool, box, row, optionBoxes}]
  let output, problemList, warningList, copyButton, copied, aboutAtlas, keepBoxes;

  function field(labelText, control, help) {
    const wrap = make('label', 'gen-field');
    wrap.append(make('span', 'gen-label', labelText), control);
    if (help) wrap.append(make('span', 'gen-help', help));
    return wrap;
  }
  function input(key, attrs = {}) {
    const box = make(attrs.multiline ? 'textarea' : 'input');
    if (!attrs.multiline) box.type = 'text';
    if (attrs.placeholder) box.placeholder = attrs.placeholder;
    if (attrs.rows) box.rows = attrs.rows;
    if (attrs.mode) box.inputMode = attrs.mode;
    box.spellcheck = false;
    box.value = form[key];
    box.oninput = () => { form[key] = box.value; update(); };
    return box;
  }
  function select(key, choices, notSet) {
    const box = make('select');
    box.append(new Option(notSet, ''));
    for (const [value, text] of choices) box.append(new Option(text, value));
    box.value = form[key];
    box.onchange = () => { form[key] = box.value; update(); };
    return box;
  }
  function check(text, key, help, prop) {
    const wrap = make('label', 'gen-check'), box = make('input');
    box.type = 'checkbox';
    if (prop) {
      box.checked = !!form.keep[prop];
      box.onchange = () => { form.keep = { ...form.keep, [prop]: box.checked }; update(); };
    } else {
      box.checked = !!form[key];
      box.onchange = () => { form[key] = box.checked; update(); };
    }
    const words = make('span');
    words.append(make('span', '', text));
    if (help) words.append(make('span', 'gen-help', help));
    wrap.append(box, words);
    return { wrap, box };
  }
  function section(title) {
    const s = make('section', 'gen-section');
    s.append(make('h3', '', title));
    return s;
  }

  function atlasSelect() {
    const box = make('select');
    box.append(new Option('Not set: the agent chooses or asks', ''));
    for (const group of data.species) {
      const og = make('optgroup'); og.label = group;
      for (const atlas of data.atlases.filter(a => a.species === group)) {
        og.append(new Option(atlas.name + (atlas.name === data.default_atlas ? ' (default)' : ''), atlas.name));
      }
      box.append(og);
    }
    box.value = form.atlas;
    box.onchange = () => { form.atlas = box.value; update(); };
    return box;
  }

  function folderHelp() {
    const d = make('details', 'gen-howto');
    d.append(make('summary', '', 'How to copy a folder’s path'));
    const list = make('ul');
    for (const [os, how] of [
      ['Windows', 'In File Explorer, right-click the folder (on Windows 10, hold Shift first) and choose “Copy as path”.'],
      ['macOS', 'In Finder, select the folder, hold Option, right-click it and choose “Copy … as Pathname”.'],
      ['Linux', 'In the Files app, select the folder and press Ctrl+C; pasting it here gives its path.'],
    ]) { const li = make('li'); li.append(make('strong', '', os + ': '), how); list.append(li); }
    d.append(list, make('p', 'gen-help', 'Paste the path into the box above. The quotes Windows adds are taken off.'));
    return d;
  }

  function toolsPanel() {
    const d = make('details', 'gen-advanced');
    d.append(make('summary', '', 'Advanced: the tools the registration agent may use'));
    d.append(make('p', 'gen-help', 'Every tool is on unless you turn it off. A task’s tools apply only when that task is chosen above. The grey name is the tool’s own name, as the Agent viewer shows it.'));
    for (const group of data.groups) {
      const box = make('fieldset', 'gen-group');
      box.append(make('legend', '', group.label));
      for (const tool of data.tools.filter(t => t.group === group.name)) {
        const row = make('div', 'gen-tool');
        const label = make('label', 'gen-check'), cb = make('input');
        cb.type = 'checkbox'; cb.checked = true; cb.dataset.name = tool.name;
        if (!tool.disableable) { cb.disabled = true; cb.title = 'Every run needs this tool.'; }
        cb.onchange = () => { setOff(tool.name, !cb.checked); update(); };
        const words = make('span');
        words.append(make('span', '', tool.label), make('code', 'gen-name', tool.name));
        if (!tool.disableable) words.append(make('span', 'gen-help gen-inline', 'always on'));
        label.append(cb, words);
        row.append(label);
        const optionBoxes = [];
        if (tool.options.length) {
          const opts = make('div', 'gen-options');
          for (const option of tool.options) {
            const ol = make('label', 'gen-check'), ob = make('input');
            ob.type = 'checkbox'; ob.checked = true; ob.dataset.name = option.name;
            ob.onchange = () => { setOff(option.name, !ob.checked); update(); };
            ol.append(ob, make('span', '', option.label));
            opts.append(ol); optionBoxes.push(ob);
          }
          row.append(opts);
        }
        box.append(row);
        toolBoxes.push({ tool, box: cb, row, optionBoxes });
      }
      d.append(box);
    }
    return d;
  }
  function setOff(name, off) {
    const names = new Set(form.off);
    if (off) names.add(name); else names.delete(name);
    form.off = [...names];
  }

  function outputPanel() {
    const panel = make('aside', 'gen-output');
    const head = make('div', 'gen-output-head');
    head.append(make('h3', '', 'Your message'));
    copyButton = make('button', 'gen-copy', 'Copy');
    copyButton.onclick = copy;
    copied = make('span', 'gen-help', '');
    head.append(copied, copyButton);
    problemList = make('ul', 'gen-problems');
    warningList = make('ul', 'gen-warnings');
    output = make('pre', 'gen-message');
    panel.append(head, problemList, warningList, output,
      make('p', 'gen-help', 'Paste it into a new chat in your agent app (Claude Code, Codex or Antigravity) with LangSlice installed. The agent sets up the job, asks about anything left open, and starts the registration; follow it in the Agent viewer.'));
    return panel;
  }

  async function copy() {
    const text = output.textContent;
    if (!text) return;
    try { await navigator.clipboard.writeText(text); }
    catch {
      const range = document.createRange(); range.selectNodeContents(output);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      document.execCommand('copy'); sel.removeAllRanges();
    }
    copied.textContent = 'Copied';
    setTimeout(() => { copied.textContent = ''; }, 2000);
  }

  function update() {
    if (!data) return;
    const here = Generator.present(data, form), off = new Set(form.off), keptOff = Generator.keptOff(data, form);
    const KEPT = 'Off: you said what this would change stays as it is.';
    for (const { tool, box, row, optionBoxes } of toolBoxes) {
      const inRun = here.has(tool.name);
      row.classList.toggle('gen-absent', !inRun);
      row.title = inRun ? '' : 'Not in this run: its task is not chosen.';
      for (const ob of optionBoxes) {
        const held = keptOff.has(ob.dataset.name);
        ob.disabled = !inRun || held || (tool.disableable && off.has(tool.name));
        ob.parentElement.classList.toggle('gen-kept', held);
        ob.parentElement.title = held ? KEPT : '';
      }
      if (tool.disableable) box.disabled = !inRun || keptOff.has(tool.name);
    }
    const tasks = new Set(form.tasks);
    for (const [prop, item] of Object.entries(keepBoxes)) item.box.disabled = !tasks.has(data.keep[prop]);
    const atlas = data.atlases.find(a => a.name === form.atlas);
    aboutAtlas.textContent = atlas ? atlas.about : 'The agent picks the atlas, or asks you.';
    const made = Generator.message(data, form);
    problemList.replaceChildren(...made.problems.map(p => make('li', '', p)));
    warningList.replaceChildren(...made.warnings.map(w => make('li', '', w)));
    output.textContent = made.text;
    output.hidden = !made.text;
    copyButton.disabled = !made.text;
  }

  function build() {
    form.atlas = data.default_atlas;
    form.tasks = data.tasks.filter(t => t.default).map(t => t.name);
    const page = make('div', 'gen-page'), main = make('div', 'gen-form');
    main.append(make('p', 'gen-intro', 'Fill in what you know, copy the message, and paste it into your agent app. Only the image folder is needed: anything left blank, the agent decides or asks you about.'));

    const where = section('Images');
    where.append(field('Folder of section images (required)', input('folder', { placeholder: 'e.g. C:\\Users\\me\\brain1 or /home/me/brain1' })), folderHelp());
    main.append(where);

    const atlas = section('Atlas');
    aboutAtlas = make('p', 'gen-help gen-about');
    atlas.append(field('Atlas', atlasSelect()), aboutAtlas,
      field('Cutting plane', select('plane', data.planes.map(p => [p, p[0].toUpperCase() + p.slice(1)]), 'Not set: the agent decides')));
    main.append(atlas);

    const cut = section('Sections');
    const row = make('div', 'gen-row');
    row.append(field('Thickness (µm)', input('thickness', { placeholder: 'e.g. 40', mode: 'numeric' })),
      field('Interval (µm)', input('interval', { placeholder: 'e.g. 120', mode: 'numeric' }), 'From one kept section to the next: every third 40 µm section is 120 µm.'),
      field('Pixel size (µm per pixel)', input('pixelSize', { placeholder: 'e.g. 0.65', mode: 'decimal' }), 'Only if the image files do not record it.'));
    cut.append(row);
    main.append(cut);

    const todo = section('What to do');
    for (const task of data.tasks) {
      const wrap = make('label', 'gen-check'), box = make('input');
      box.type = 'checkbox'; box.checked = form.tasks.includes(task.name);
      box.onchange = () => {
        form.tasks = data.tasks.map(t => t.name).filter(n => (n === task.name ? box.checked : form.tasks.includes(n)));
        update();
      };
      const words = make('span');
      words.append(make('span', '', task.label), make('span', 'gen-help', task.about));
      wrap.append(box, words);
      todo.append(wrap);
    }
    main.append(todo);

    const known = section('What you already know');
    known.append(check('The file names sort in the order the sections were cut', 'ordered').wrap);
    keepBoxes = {
      angles: check('The sections were cut straight: do not tilt the cutting plane', '', 'Applies when placing along the atlas.', 'angles'),
      orientation: check('Every section is the right way up and not mirrored: do not flip or turn them', '', 'Applies when aligning in the plane.', 'orientation'),
    };
    known.append(keepBoxes.angles.wrap, keepBoxes.orientation.wrap,
      field('Damaged sections', input('damaged', { multiline: true, rows: 3, placeholder: 'One per line: file name: what is wrong\ne.g. s012.tif: left cortex torn' })));
    main.append(known);

    const notes = section('Notes for the agent');
    notes.append(field('Notes', input('notes', { multiline: true, rows: 3, placeholder: 'e.g. Injection in the right hemisphere; the olfactory bulbs were removed.' })));
    main.append(notes, toolsPanel());

    page.append(main, outputPanel());
    root.replaceChildren(page);
    update();
  }

  async function load() {
    try {
      const r = await fetch('generator', { cache: 'no-store' });
      if (!r.ok) throw new Error(String(r.status));
      data = await r.json();
      build();
    } catch {
      root.replaceChildren(make('p', 'gen-intro', 'The prompt generator could not load its choices from LangSlice. Close the window and open it again.'));
    }
  }
  load();
})();
