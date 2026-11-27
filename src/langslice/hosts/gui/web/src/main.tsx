import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../../static/tokens.css';
import './ui/window.css';
import { App } from './App';

try {
  if (localStorage.getItem('langslice.theme') === 'light') document.documentElement.dataset.theme = 'light';
} catch { /* dark */ }

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
