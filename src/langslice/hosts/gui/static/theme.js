'use strict';
// The window's look, dark (the start) or light, kept per browser. Loaded before the page
// is drawn, so a window kept light does not flash dark; app.js wires the button.
const Theme = (() => {
  const KEY = 'langslice.theme';
  const LOCKUPS = { dark: 'lockup.svg', light: 'lockup-light.svg' };
  let name = 'dark';
  try { if (localStorage.getItem(KEY) === 'light') name = 'light'; } catch { /* dark */ }
  document.documentElement.dataset.theme = name;
  // The button names the look it switches to; the lockup is the one drawn for the ground.
  function show() {
    document.documentElement.dataset.theme = name;
    const button = document.getElementById('theme');
    if (button) button.textContent = name === 'dark' ? 'Light' : 'Dark';
    const lockup = document.getElementById('lockup');
    if (lockup) lockup.src = LOCKUPS[name];
  }
  function toggle() {
    name = name === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem(KEY, name); } catch { /* not kept */ }
    show();
  }
  return { show, toggle };
})();
