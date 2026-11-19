// Types for generator-logic.js, the prompt generator's pure functions, as the React
// window imports them (web/src/prompt). The payload is GET generator's answer
// (hosts/gui/generator.py); the form is the plain object the functions take.

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
export type Prefill = {
  folder?: string; atlas?: string; plane?: string; thickness?: string; interval?: string;
  pixel_size?: string; tasks?: string[]; off?: string[]; on?: string[]; facts?: string[];
  damaged?: string; keep?: Record<string, boolean>; notes?: string;
  frozen?: Record<string, unknown>; abba?: Abba | null;
};
export type Data = {
  tasks: Task[]; groups: Group[]; tools: Tool[]; task_writers: Record<string, string[]>;
  keep: Record<string, string>; frozen_off: Record<string, string[]>; atlases: Atlas[];
  species: string[]; default_atlas: string; planes: string[]; flags: Record<string, string>;
  image_models: ImageProvider[]; integrations: Integration[]; prefill?: Prefill | null;
};
export type Form = {
  folder: string; atlas: string; plane: string; thickness: string; interval: string;
  pixelSize: string; tasks: string[]; off: string[]; on: string[]; ordered: boolean;
  damaged: string; keep: Record<string, boolean>; notes: string; imageProvider: string;
  imageModel: string; abba: Abba | null; keepRegistrations: boolean;
  frozen: Record<string, unknown>; integrations: string[];
};
export type Message = { text: string; problems: string[]; warnings: string[] };

declare const Generator: {
  shellQuote(value: string): string;
  cleanPath(raw: string): string;
  startForm(data: Data): Form;
  prefilled(data: Data, prefill: Prefill | null): Form;
  offNames(data: Data, form: Form): string[];
  onNames(data: Data, form: Form): string[];
  toolOff(form: Form, tool: Tool): boolean;
  present(data: Data, form: Form): Set<string>;
  keptOff(data: Data, form: Form): Set<string>;
  imageModelApplies(data: Data, form: Form): boolean;
  problems(data: Data, form: Form): string[];
  warnings(data: Data, form: Form): string[];
  leftOut(data: Data, form: Form): string[];
  integrationWords(data: Data, form: Form): string[];
  command(data: Data, form: Form): { text: string; quoted: boolean };
  message(data: Data, form: Form): Message;
  keepLabel(n: number): string;
};
export default Generator;
