// The window's form controls, shared by the Prompt generator and Settings: each keeps its
// native input (the keyboard, screen readers, forms) and draws its own mark beside it.
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useId, useState, type ReactNode } from 'react';
import { DUR, EASE_OUT } from './motion';
import './controls.css';

// A tick box: the native checkbox, its drawn square with a tick, its words and a note.
export function Check({ checked, onChange, disabled, children, help, className }: {
  checked: boolean; onChange: (on: boolean) => void; disabled?: boolean;
  children: ReactNode; help?: ReactNode; className?: string;
}) {
  return (
    <label className={'check' + (className ? ' ' + className : '')} data-disabled={disabled || undefined}>
      <input type="checkbox" className="native" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)} />
      <Tick />
      <span className="check-words">
        <span className="check-text">{children}</span>
        {help && <span className="help">{help}</span>}
      </span>
    </label>
  );
}

// The drawn square of a checkbox; its tick is drawn by the stroke, so it can be traced in.
export function Tick() {
  return (
    <span className="tick" aria-hidden="true">
      <svg viewBox="0 0 12 12"><path d="M2.6 6.3 4.9 8.6 9.4 3.6" /></svg>
    </span>
  );
}

// A switch: a native checkbox with role="switch", its track and thumb drawn after it.
export function Switch({ checked, onChange, disabled, label, title }: {
  checked: boolean; onChange: (on: boolean) => void; disabled?: boolean; label?: string; title?: string;
}) {
  return (
    <>
      <input type="checkbox" role="switch" className="native" checked={checked} disabled={disabled}
        aria-label={label} title={title} onChange={e => onChange(e.target.checked)} />
      <span className="switch" aria-hidden="true" />
    </>
  );
}

// A radio drawn as a ring with a dot.
export function Radio({ name, value, checked, disabled, onPick, label }: {
  name: string; value: string; checked: boolean; disabled?: boolean; onPick: (value: string) => void; label?: string;
}) {
  return (
    <>
      <input type="radio" className="native" name={name} value={value} checked={checked} disabled={disabled}
        aria-label={label} onChange={e => { if (e.target.checked) onPick(value); }} />
      <span className="radio" aria-hidden="true" />
    </>
  );
}

// A native select in the window's dress, with a drawn chevron.
export function Select({ className, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <span className={'select' + (className ? ' ' + className : '')} data-disabled={props.disabled || undefined}>
      <select {...props} />
    </span>
  );
}

// A label over its control, a note under it. `id` names the control the label is for.
export function Field({ label, id, help, children, labelId }: {
  label: ReactNode; id?: string; help?: ReactNode; children: ReactNode; labelId?: string;
}) {
  return (
    <div className="field">
      {id ? <label className="label" htmlFor={id}>{label}</label> : <p className="label" id={labelId}>{label}</p>}
      {children}
      {help && <p className="help">{help}</p>}
    </div>
  );
}

// A section of a page that opens and closes: its summary a button, its body grown to its
// height (an occasional action, so it moves; under reduced motion it only fades).
export function Disclosure({ summary, children, startOpen = false, className }: {
  summary: ReactNode; children: ReactNode; startOpen?: boolean; className?: string;
}) {
  const [open, setOpen] = useState(startOpen);
  const id = useId();
  const reduce = useReducedMotion();
  const closed = reduce ? { opacity: 0 } : { height: 0, opacity: 0 };
  return (
    <div className={'disclosure' + (open ? ' open' : '') + (className ? ' ' + className : '')}>
      <button type="button" className="disclosure-summary" aria-expanded={open} aria-controls={id} onClick={() => setOpen(o => !o)}>
        <span className="chevron" aria-hidden="true" />
        {summary}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            className="disclosure-body"
            initial={closed}
            animate={{ height: 'auto', opacity: 1 }}
            exit={closed}
            transition={{ duration: DUR.disclose, ease: EASE_OUT }}
          >
            <div className="disclosure-inner">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// A sentence whose literal parts are code: `prose('Run ', ['x'], '.')`.
export function Prose({ className, parts }: { className?: string; parts: (string | [string])[] }) {
  return <p className={className}>{parts.map((p, i) => (Array.isArray(p) ? <code key={i} className="literal">{p[0]}</code> : p))}</p>;
}
