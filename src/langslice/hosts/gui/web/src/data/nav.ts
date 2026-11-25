// What the window shows, kept in its address (`#tab=viewer&view=simple&job=0`), so a
// reload or `langslice gui --tab prompt` opens the same place: the tab (the agent viewer,
// the prompt generator or Settings) and the viewer's view (Simple or Extended, also kept
// per browser).
import { useSyncExternalStore } from 'react';

export type Tab = 'viewer' | 'prompt' | 'settings';
export type View = 'simple' | 'extended';
const TABS: Tab[] = ['viewer', 'prompt', 'settings'];

const read = () => new URLSearchParams(location.hash.slice(1));
const stored = (): View => {
  try { return localStorage.getItem('langslice.detail') === 'extended' ? 'extended' : 'simple'; } catch { return 'simple'; }
};

let place = (() => {
  const p = read();
  const tab = TABS.includes(p.get('tab') as Tab) ? (p.get('tab') as Tab) : 'viewer';
  const view: View = p.get('view') === 'extended' ? 'extended' : p.get('view') === 'simple' ? 'simple' : stored();
  return { tab, view };
})();
const listeners = new Set<() => void>();

export function useNav() {
  return useSyncExternalStore(cb => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => place);
}
export function go(next: Partial<typeof place>) {
  place = { ...place, ...next };
  if (next.view) { try { localStorage.setItem('langslice.detail', next.view); } catch { /* not kept */ } }
  const p = read();
  p.set('tab', place.tab);
  p.set('view', place.view);
  history.replaceState(null, '', '#' + p.toString());
  for (const l of listeners) l();
}
