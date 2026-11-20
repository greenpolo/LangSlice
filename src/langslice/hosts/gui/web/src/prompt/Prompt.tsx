// The Prompt generator tab: a form that makes one message to paste into an agent app (a
// short request and the job's `langslice job FOLDER init` command, which the
// register-brain skill runs). What the form offers comes from GET generator
// (hosts/gui/generator.py); the message is built by static/generator-logic.js, the same
// pure functions tests/test_gui_generator.py runs under node. The form on the left in
// numbered steps, the message on the right, sticky, with Copy.
import { motion } from 'motion/react';
import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import Generator, { type Data, type Form, type Tool } from '../../../static/generator-logic.js';
import { Check, Disclosure, Field, Select, Switch, Tick } from '../controls';
import { DUR, EASE_IN_OUT } from '../motion';
import { Message } from './Message';
import './prompt.css';

const capital = (word: string) => (word ? word[0].toUpperCase() + word.slice(1) : '');

// The job the window was opened on (`#tab=prompt&job=<n>`): the form starts from its job.json.
function startJob() {
  const job = new URLSearchParams(location.hash.slice(1)).get('job');
  return job !== null && /^[0-9]+$/.test(job) ? job : null;
}

export function Prompt() {
  const [data, setData] = useState<Data | null>(null);
  const [form, setForm] = useState<Form | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const job = startJob();
    fetch(job === null ? 'generator' : `generator?job=${job}`, { cache: 'no-store' })
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d: Data) => { setData(d); setForm(Generator.prefilled(d, d.prefill || null)); })
      .catch(error => { console.error(error); setFailed(true); });
  }, []);
  if (failed) return <div className="page prompt"><p className="lead">The prompt generator could not load its choices from LangSlice. Close the window and open it again.</p></div>;
  if (!data || !form) return <div className="page prompt" aria-busy="true" />;
  return <PromptForm data={data} form={form} setForm={setForm} />;
}

function PromptForm({ data, form, setForm }: { data: Data; form: Form; setForm: (f: Form) => void }) {
  const up = (patch: Partial<Form>) => setForm({ ...form, ...patch });
  const abba = form.abba;
  const fixedHelp = abba ? 'Set by ABBA.' : undefined;
  const made = Generator.message(data, form);
  const ids = useId();
  const id = (name: string) => `${ids}-${name}`;
  return (
    <div className="page prompt">
      <div className="prompt-form">
        <p className="lead">
          {abba
            ? 'ABBA prepared this job: its snapshots, positions and cutting angles are in it. Add what you know, copy the message, and paste it into your agent app.'
            : 'Fill in what you know, copy the message, and paste it into your agent app. Only the image folder is needed: anything left blank, the agent decides or asks you about.'}
        </p>
        {abba && <AbbaBlock data={data} form={form} up={up} />}

        <Step n={1} title="Images">
          <Field label="Folder of section images (required)" id={id('folder')} help={fixedHelp}>
            <input id={id('folder')} className="input path" type="text" spellCheck={false} autoComplete="off"
              placeholder="e.g. C:\Users\me\brain1 or /home/me/brain1" disabled={!!abba}
              value={form.folder} onChange={e => up({ folder: e.target.value })} />
          </Field>
          {!abba && <FolderHelp />}
        </Step>

        <Step n={2} title="Atlas">
          <Field label="Atlas" id={id('atlas')}>
            <Select id={id('atlas')} value={form.atlas} disabled={!!abba} onChange={e => up({ atlas: e.target.value })}>
              <option value="">Not set: the agent chooses or asks</option>
              {data.species.map(group => {
                const atlases = data.atlases.filter(a => a.species === group);
                return atlases.length ? (
                  <optgroup key={group} label={group}>
                    {atlases.map(a => <option key={a.name} value={a.name}>{a.name + (a.name === data.default_atlas ? ' (default)' : '')}</option>)}
                  </optgroup>
                ) : null;
              })}
            </Select>
          </Field>
          <About words={data.atlases.find(a => a.name === form.atlas)?.about ?? 'The agent picks the atlas, or asks you.'} />
          <Field label="Cutting plane" labelId={id('plane')} help={fixedHelp}>
            <Plane data={data} form={form} up={up} labelledBy={id('plane')} />
          </Field>
        </Step>

        <Step n={3} title="Sections">
          <div className="units">
            <Field label="Thickness" id={id('thick')} help={abba && form.thickness ? 'ABBA’s, from its slice spacing: check it.' : undefined}>
              <Unit id={id('thick')} unit="µm" placeholder="e.g. 40" mode="numeric" value={form.thickness} onChange={v => up({ thickness: v })} />
            </Field>
            <Field label="Interval" id={id('interval')} help="From one kept section to the next: every third 40 µm section is 120 µm.">
              <Unit id={id('interval')} unit="µm" placeholder="e.g. 120" mode="numeric" value={form.interval} onChange={v => up({ interval: v })} />
            </Field>
            <Field label="Pixel size" id={id('pixel')} help={abba ? 'ABBA’s snapshots.' : 'Only if the image files do not record it.'}>
              <Unit id={id('pixel')} unit="µm per pixel" placeholder="e.g. 0.65" mode="decimal" disabled={!!abba} value={form.pixelSize} onChange={v => up({ pixelSize: v })} />
            </Field>
          </div>
        </Step>

        <Step n={4} title="What to do">
          <div className="tasks">
            {data.tasks.map(task => {
              const on = form.tasks.includes(task.name);
              return (
                <label key={task.name} className={'task' + (on ? ' on' : '')}>
                  <input type="checkbox" className="native" checked={on} onChange={e => up({
                    tasks: data.tasks.map(t => t.name).filter(n => (n === task.name ? e.target.checked : form.tasks.includes(n))),
                  })} />
                  <span className="task-head"><Tick /><span className="task-label">{task.label}</span></span>
                  <span className="task-about">{task.about}</span>
                </label>
              );
            })}
          </div>
        </Step>

        <Step n={5} title="What you already know">
          <div className="checks">
            <Check checked={form.ordered} onChange={on => up({ ordered: on })}>The file names sort in the order the sections were cut</Check>
            <Check checked={!!form.keep.angles} disabled={!form.tasks.includes(data.keep.angles)} help="Applies when placing along the atlas."
              onChange={on => up({ keep: { ...form.keep, angles: on } })}>The sections were cut straight: do not tilt the cutting plane</Check>
            <Check checked={!!form.keep.orientation} disabled={!form.tasks.includes(data.keep.orientation)} help="Applies when aligning in the plane."
              onChange={on => up({ keep: { ...form.keep, orientation: on } })}>Every section is the right way up and not mirrored: do not flip or turn them</Check>
          </div>
          <Field label="Damaged sections" id={id('damaged')}>
            <textarea id={id('damaged')} className="input" rows={3} spellCheck={false} autoComplete="off"
              placeholder={'One per line: file name: what is wrong\ne.g. s012.tif: left cortex torn'}
              value={form.damaged} onChange={e => up({ damaged: e.target.value })} />
          </Field>
        </Step>

        <Step n={6} title="Notes for the agent">
          <div className="field">
            <textarea className="input" rows={3} spellCheck={false} autoComplete="off" aria-label="Notes for the agent"
              placeholder="e.g. Injection in the right hemisphere; the olfactory bulbs were removed."
              value={form.notes} onChange={e => up({ notes: e.target.value })} />
            <p className="help">Saved with the job; the registration agent reads them.</p>
          </div>
        </Step>

        {!abba && (
          <Step n={7} title="Where you’ll use the results">
            <div className="checks">
              {data.integrations.map(program => (
                <Check key={program.name} checked={form.integrations.includes(program.name)} onChange={on => up({
                  integrations: data.integrations.map(i => i.name).filter(name => (name === program.name ? on : form.integrations.includes(name))),
                })}>{program.label}</Check>
              ))}
            </div>
            <p className="help">Your agent prepares the files each program opens once the registration is checked, and tells you where they are. The registration agent is not told. For ABBA, start the job from ABBA instead (Register › LangSlice…).</p>
          </Step>
        )}

        <Disclosure className="step advanced" summary={<span className="step-title">Advanced: the tools the registration agent may use</span>}>
          <Tools data={data} form={form} up={up} />
        </Disclosure>
      </div>
      <Message made={made} />
    </div>
  );
}

// A numbered step: its number in the gutter, its title, its body under the title.
function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className="step" aria-labelledby={id}>
      <span className="step-n" aria-hidden="true">{n}</span>
      <h2 className="step-title" id={id}>{title}</h2>
      <div className="step-body">{children}</div>
    </section>
  );
}

function FolderHelp() {
  return (
    <Disclosure className="more" summary="How to copy a folder’s path">
      <ul className="howto">
        <li><strong>Windows: </strong>In File Explorer, right-click the folder (on Windows 10, hold Shift first) and choose “Copy as path”.</li>
        <li><strong>macOS: </strong>In Finder, select the folder, hold Option, right-click it and choose “Copy … as Pathname”.</li>
        <li><strong>Linux: </strong>In the Files app, select the folder and press Ctrl+C; pasting it here gives its path.</li>
      </ul>
      <p className="help">Paste the path into the box above. The quotes Windows adds are taken off.</p>
    </Disclosure>
  );
}

// BrainGlobe's paragraph on the chosen atlas: three lines, until `more` opens it.
function About({ words }: { words: string }) {
  const text = useRef<HTMLParagraphElement>(null);
  const [open, setOpen] = useState(false);
  const [long, setLong] = useState(false);
  useEffect(() => { setOpen(false); }, [words]);
  useLayoutEffect(() => {
    const el = text.current;
    if (!el) return;
    const measure = () => { if (!el.classList.contains('open')) setLong(el.scrollHeight > el.clientHeight + 1); };
    measure();
    const watch = new ResizeObserver(measure);
    watch.observe(el);
    return () => watch.disconnect();
  }, [words, open]);
  return (
    <div className="about">
      <p ref={text} className={'about-text' + (open ? ' open' : '')}>{words}</p>
      {(long || open) && (
        <button type="button" className="link" aria-expanded={open} onClick={() => setOpen(o => !o)}>{open ? 'less' : 'more'}</button>
      )}
    </div>
  );
}

// The cutting plane: one of the planes, or none (the agent decides). The chosen segment's
// ground slides to it.
function Plane({ data, form, up, labelledBy }: { data: Data; form: Form; up: (p: Partial<Form>) => void; labelledBy: string }) {
  const fixed = !!form.abba;
  const layout = useId();
  return (
    <div className="plane">
      <div className="segments" role="group" aria-labelledby={labelledBy}>
        {data.planes.map(plane => {
          const on = form.plane === plane;
          return (
            <button key={plane} type="button" className={'segment' + (on ? ' on' : '')} aria-pressed={on} disabled={fixed}
              onClick={() => up({ plane: on ? '' : plane })}>
              {on && <motion.span layoutId={layout} className="segment-ground" transition={{ duration: DUR.indicator, ease: EASE_IN_OUT }} />}
              <span className="segment-text">{capital(plane)}</span>
            </button>
          );
        })}
      </div>
      {!form.plane && <span className="help">Not set: the agent decides.</span>}
      {form.plane && !fixed && <button type="button" className="link" onClick={() => up({ plane: '' })}>Let the agent decide</button>}
    </div>
  );
}

// A short number field with its unit inside the same box.
function Unit({ id, unit, placeholder, mode, value, onChange, disabled }: {
  id: string; unit: string; placeholder: string; mode: 'numeric' | 'decimal'; value: string;
  onChange: (v: string) => void; disabled?: boolean;
}) {
  const box = useRef<HTMLInputElement>(null);
  return (
    <span className="unit" data-disabled={disabled || undefined}
      onMouseDown={e => { if (e.target !== box.current && !disabled) { e.preventDefault(); box.current?.focus(); } }}>
      <input ref={box} id={id} className="unit-input" type="text" inputMode={mode} spellCheck={false} autoComplete="off"
        placeholder={placeholder} disabled={disabled} value={value} onChange={e => onChange(e.target.value)} />
      <span className="unit-name">{unit}</span>
    </span>
  );
}

// A job prepared in ABBA: what ABBA fixed, the snapshots' slice names, and whether the
// sections ABBA already registered stay as they are.
function AbbaBlock({ form, up }: { data: Data; form: Form; up: (p: Partial<Form>) => void }) {
  const abba = form.abba!;
  const registered = abba.registered || [];
  const names = Object.entries(abba.sections || {});
  const warped = new Set(abba.existing_warp || []), done = new Set(registered);
  const n = registered.length;
  return (
    <section className="abba" aria-label="From ABBA">
      <h2 className="step-title">From ABBA</h2>
      <dl className="facts">
        <dt>Images</dt><dd className="path">{form.folder}</dd>
        <dt>Atlas</dt><dd><span className="path">{form.atlas}</span>{abba.atlas && <span className="muted"> (in ABBA: {abba.atlas})</span>}</dd>
        <dt>Cutting plane</dt><dd>{capital(form.plane)}</dd>
        <dt>Snapshot pixel size</dt><dd>{form.pixelSize ? `${form.pixelSize} µm per pixel` : 'not recorded'}</dd>
      </dl>
      <p className="help">ABBA set these when it exported the snapshots; the job already holds them, so the message leaves them out.</p>
      {n > 0 && Object.keys(form.frozen).length > 0 && (
        <Check checked={!!form.keepRegistrations} onChange={on => up({ keepRegistrations: on })}
          help={n === 1 ? 'Untick to let the agent redo its alignment and warp too.' : 'Untick to let the agent redo their alignment and warps too.'}>
          {Generator.keepLabel(n)}
        </Check>
      )}
      {names.length > 0 && (
        <Disclosure className="more" startOpen={names.length <= 12} summary={`Snapshot files and their ABBA slices (${names.length})`}>
          <div className="slices">
            <table>
              <thead><tr><th>File</th><th>ABBA slice</th><th>In ABBA</th></tr></thead>
              <tbody>
                {names.map(([file, slice]) => (
                  <tr key={file}>
                    <td className="path">{file}</td><td>{slice}</td>
                    <td className="muted">{warped.has(file) ? 'registered, warped' : done.has(file) ? 'registered' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="help">Name a damaged section below by its file name.</p>
        </Disclosure>
      )}
    </section>
  );
}

// The tools table: a switch per tool, each greyed row saying why; the image
// model after the warp's tools.
function Tools({ data, form, up }: { data: Data; form: Form; up: (p: Partial<Form>) => void }) {
  const here = Generator.present(data, form), keptOff = Generator.keptOff(data, form);
  const noImageModel = form.imageProvider === 'none';
  const setOff = (name: string, isOff: boolean) => {
    const tool = data.tools.find(t => t.name === name);
    if (tool && tool.default_off) {  // a tool that starts off is named when it is on
      const on = new Set(form.on || []);
      if (isOff) on.delete(name); else on.add(name);
      up({ on: [...on] });
      return;
    }
    const names = new Set(form.off);
    if (isOff) names.add(name); else names.delete(name);
    up({ off: [...names] });
  };
  let imageShown = false;
  const groups = data.groups.map(group => {
    const inRun = data.tools.some(t => t.group === group.name && here.has(t.name));
    const task = data.tasks.find(t => t.name === group.task);
    const showImage = group.task === 'nonlinear' && !imageShown;
    if (showImage) imageShown = true;
    return (
      <div key={group.name}>
        <ToolGroup title={group.label} note={inRun || !task ? '' : `Not in this run: “${task.label}” is not chosen above.`} absent={!inRun}>
          {data.tools.filter(t => t.group === group.name).map(tool => (
            <ToolRows key={tool.name} data={data} form={form} tool={tool} here={here} keptOff={keptOff} noImageModel={noImageModel} setOff={setOff} />
          ))}
        </ToolGroup>
        {showImage && <ImageModel data={data} form={form} up={up} />}
      </div>
    );
  });
  return (
    <>
      <p className="help measure">Every tool is on unless you turn it off, except DeepSlice, Brainways and the image model’s border trace, which start off. A task’s tools apply only when that task is chosen above. The violet name is the tool’s own name, as the Agent viewer shows it.</p>
      {groups}
      {!imageShown && <ImageModel data={data} form={form} up={up} />}
    </>
  );
}

function ToolGroup({ title, note, absent, children }: { title: string; note: string; absent: boolean; children: ReactNode }) {
  const id = useId();
  return (
    <div className={'tools' + (absent ? ' absent' : '')} role="group" aria-labelledby={id}>
      <div className="tools-head"><span className="tools-title" id={id}>{title}</span>{note && <span className="tools-note">{note}</span>}</div>
      {children}
    </div>
  );
}

function ToolRows({ data, form, tool, here, keptOff, noImageModel, setOff }: {
  data: Data; form: Form; tool: Tool; here: Set<string>; keptOff: Set<string>;
  noImageModel: boolean; setOff: (name: string, off: boolean) => void;
}) {
  void data;
  const inRun = here.has(tool.name), held = keptOff.has(tool.name);
  const noModel = !inRun && noImageModel && tool.image_model;
  const note = !tool.disableable ? 'always on' : noModel ? 'not in this run: you chose no image model' : held ? 'off: you said this stays as it is' : '';
  const checked = tool.disableable ? !held && !Generator.toolOff(form, tool) : true;
  return (
    <>
      <label className={'tool' + (inRun ? '' : ' absent') + (tool.disableable ? '' : ' fixed')}
        title={inRun ? '' : (noModel ? 'Not in this run: you chose no image model.' : 'Not in this run: its task is not chosen.')}>
        <Switch checked={checked} disabled={!tool.disableable || !inRun || held}
          title={tool.disableable ? undefined : 'Every run needs this tool.'} onChange={on => setOff(tool.name, !on)} />
        <span className="tool-words"><span className="tool-label">{tool.label}</span>{note && <span className="tool-note">{note}</span>}</span>
        <code className="tool-name">{tool.name}</code>
      </label>
    </>
  );
}

// The image model the warp's border tracing asks (`--nonlinear-provider`,
// `--nonlinear-image-model`): the providers and models LangSlice lists.
function ImageModel({ data, form, up }: { data: Data; form: Form; up: (p: Partial<Form>) => void }) {
  const ids = useId();
  const on = Generator.imageModelApplies(data, form);
  const offWords = 'Not in this run: warping to fit is not chosen, or its border tracing is off.';
  const tracer = data.tools.find(t => t.image_model);
  const fallback = data.image_models.find(p => p.default);
  const chosen = data.image_models.find(p => p.provider === form.imageProvider);
  const hint = !chosen ? '' : chosen.provider === 'none' ? 'The warp fits each section to its stain alone.'
    : chosen.connected ? 'Set up on this computer.' : 'Not set up on this computer yet: Settings connects it.';
  return (
    <div className={'tools image' + (on ? '' : ' absent')} role="group" aria-labelledby={ids + '-t'} title={on ? '' : offWords}>
      <div className="tools-head"><span className="tools-title" id={ids + '-t'}>Image model</span>{!on && <span className="tools-note">{offWords}</span>}</div>
      <p className="help">The model that draws a section’s borders for {tracer ? `“${tracer.label}”` : 'the warp'}. Applies when warping to fit.</p>
      <div className="pair">
        <Field label="Provider" id={ids + '-p'}>
          <Select id={ids + '-p'} value={form.imageProvider} disabled={!on} onChange={e => up({ imageProvider: e.target.value, imageModel: '' })}>
            <option value="">{`Not set: the connection saved in Settings${fallback ? ` (${fallback.label})` : ''}`}</option>
            {data.image_models.map(p => <option key={p.provider} value={p.provider}>{p.label}</option>)}
          </Select>
        </Field>
        <Field label="Model" id={ids + '-m'}>
          <Select id={ids + '-m'} value={form.imageModel} disabled={!on || !form.imageProvider || form.imageProvider === 'none'}
            onChange={e => up({ imageModel: e.target.value })}>
            <option value="">{chosen && chosen.default_model ? `The provider’s default (${chosen.default_model})` : 'The provider’s default'}</option>
            {(chosen ? chosen.models : []).map(name => <option key={name} value={name}>{name}</option>)}
          </Select>
        </Field>
      </div>
      {hint && <p className="help">{hint}</p>}
    </div>
  );
}
