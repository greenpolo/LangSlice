'use strict';
// The prompt generator's message, as pure functions (`Generator`): the form's state in,
// the message to paste into an agent app out (a short request and the job's
// `langslice-job FOLDER init` command, which the register-brain skill runs). One source
// for the older page (generator.js), the React window (web/src/prompt, through
// generator-logic.d.ts) and tests/test_gui_generator.py, which runs them under node.
// What the form offers comes from the code (GET generator: hosts/gui/generator.py).

const Generator = (() => {
  const SEP = ':';
  const SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;
  const ORDERED = 'The file names sort in the order the sections were cut.';

  // The form as the page starts it: the default atlas and tasks, nothing else.
  function startForm(data) {
    return { folder: '', atlas: data.default_atlas, plane: '', thickness: '', interval: '', pixelSize: '',
             tasks: data.tasks.filter(t => t.default).map(t => t.name), off: [], on: [], ordered: false,
             damaged: '', keep: {}, notes: '', imageProvider: '', imageModel: '',
             abba: null, keepRegistrations: true, frozen: {}, integrations: [] };
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
             tasks, off: [...(prefill.off || [])], on: [...(prefill.on || [])], ordered: (prefill.facts || []).includes(ORDERED),
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
    const key = tasks.join(','), held = keptOff(data, form);
    return data.tools.some(t => t.image_model && t.runs_in.includes(key)
      && !(t.disableable && toolOff(form, t)) && !held.has(t.name));
  }
  // The image-model flags: [setting, value] pairs, none when nothing is chosen.
  function imageModelFlags(data, form) {
    const provider = String(form.imageProvider || '').trim(), model = String(form.imageModel || '').trim();
    if (!provider || !imageModelApplies(data, form)) return [];
    const out = [['nonlinear.provider', provider]];
    if (model && provider !== 'none') out.push(['nonlinear.image_model', model]);
    return out;
  }

  // Whether the form has a tool off: a tool that starts off (`default_off`)
  // is off unless `form.on` names it, any other when `form.off` does.
  function toolOff(form, tool) {
    return tool.default_off ? !(form.on || []).includes(tool.name) : (form.off || []).includes(tool.name);
  }
  // The tools that start off and the form leaves off.
  function startsOff(data, form) {
    return data.tools.filter(t => t.default_off && toolOff(form, t)).map(t => t.name);
  }
  // The names the job's `--enabled` takes: the tools of this run that start
  // off and the form switches on.
  function onNames(data, form) {
    const here = present(data, form), held = keptOff(data, form);
    return data.tools.filter(t => t.default_off && here.has(t.name) && !held.has(t.name) && !toolOff(form, t))
      .map(t => t.name);
  }

  // The names the job's `--disabled` takes: the tools of this run switched off
  // whole, then the options switched off of the tools still on (a name a kept
  // property already switches off left out).
  function offNames(data, form) {
    const off = new Set(form.off || []), here = present(data, form), held = keptOff(data, form), names = [];
    for (const tool of data.tools) {
      if (!here.has(tool.name) || held.has(tool.name)) continue;
      if (tool.disableable && toolOff(form, tool)) { if (!tool.default_off) names.push(tool.name); continue; }
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
    const off = new Set([...offNames(data, form), ...startsOff(data, form)]), here = present(data, form);
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
    const on = onNames(data, form);
    if (on.length) flag('enabled', on.join(','));
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

  // The programs the person will use the results in (`data.integrations`, in
  // its order), named for the main agent; never in the command, so the
  // registration agent never reads them. A job prepared in ABBA has none: its
  // results already go to ABBA.
  function integrationWords(data, form) {
    if (form.abba) return [];
    const chosen = new Set(form.integrations || []);
    return data.integrations.filter(i => chosen.has(i.name)).map(i => i.words);
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
    const programs = integrationWords(data, form);
    if (programs.length) lines.push(`When the registration is done, prepare the results for ${joinWords(programs)}.`);
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

  return { shellQuote, cleanPath, startForm, prefilled, offNames, onNames, toolOff, present, keptOff, imageModelApplies,
           problems, warnings, leftOut, integrationWords, command, message, keepLabel };
})();

if (typeof module !== 'undefined') module.exports = Generator;
