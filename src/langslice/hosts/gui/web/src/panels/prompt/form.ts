// The prompt generator's data and form. GET generator gives what the form offers and the
// form to start from (hosts/gui/forms/generator.py, hosts/gui/forms/message.py); every change of the
// form is posted to generator/message, which answers with the message, its problems and
// how each tool stands. The page holds the form and draws the answer, nothing more.
import { useEffect, useRef, useState } from 'react';

export type Task = { name: string; label: string; about: string; default: boolean };
export type Group = { name: string; label: string; task: string | null };
export type Tool = {
  name: string; group: string; label: string; disableable: boolean; needs: string[][];
  image_model: boolean; default_off: boolean; runs_in: string[];
};
export type Atlas = { name: string; about: string; species: string };
export type ImageProvider = {
  provider: string; label: string; default: boolean; models: string[];
  default_model?: string | null; connected?: boolean;
};
export type Integration = { name: string; label: string; words: string };
export type Abba = {
  atlas?: string; registered?: string[]; existing_warp?: string[];
  sections?: Record<string, string>;
};
export type Form = {
  folder: string; atlas: string; plane: string; thickness: string; interval: string;
  pixelSize: string; tasks: string[]; off: string[]; on: string[]; ordered: boolean;
  damaged: string; keep: Record<string, boolean>; notes: string; imageProvider: string;
  imageModel: string; abba: Abba | null; keepRegistrations: boolean;
  frozen: Record<string, unknown>; integrations: string[];
};
export type Data = {
  tasks: Task[]; groups: Group[]; tools: Tool[]; keep: Record<string, string>; atlases: Atlas[];
  species: string[]; default_atlas: string; planes: string[];
  image_models: ImageProvider[]; integrations: Integration[]; form: Form;
};
export type ToolStand = { present: boolean; held: boolean; on: boolean };
export type Made = {
  text: string; problems: string[]; warnings: string[];
  tools: Record<string, ToolStand>; image_model_applies: boolean;
};

const UNREACHABLE: Made = {
  text: '', problems: ['LangSlice is not reachable: the message cannot be made.'], warnings: [],
  tools: {}, image_model_applies: false,
};

// The server's answer for *form*, the newest one kept while the next is on its way.
export function useMade(form: Form): Made | null {
  const [made, setMade] = useState<Made | null>(null);
  const asked = useRef(0);
  useEffect(() => {
    const n = ++asked.current;
    const stop = new AbortController();
    fetch('generator/message', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form),
      cache: 'no-store', signal: stop.signal,
    })
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((answer: Made) => { if (n === asked.current) setMade(answer); })
      .catch(error => { if (!stop.signal.aborted && n === asked.current) { console.error(error); setMade(UNREACHABLE); } });
    return () => stop.abort();
  }, [form]);
  return made;
}
