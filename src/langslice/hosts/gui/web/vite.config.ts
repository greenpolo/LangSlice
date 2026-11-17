// Builds the window's page into ../static as index.html, ui.js and ui.css: fixed names
// directly in static/ (the server serves only those), no inline script or style (the
// page's content policy allows 'self' only), the fonts and logos the page already ships
// left where they are.
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  publicDir: false,
  build: {
    outDir: '../static',
    emptyOutDir: false,
    assetsInlineLimit: 0,
    cssCodeSplit: false,
    modulePreload: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      input: 'index.html',
      output: {
        entryFileNames: 'ui.js',
        chunkFileNames: 'ui-[name].js',
        assetFileNames: info => (info.names?.[0]?.endsWith('.css') ? 'ui.css' : '[name][extname]'),
      },
    },
  },
});
