import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Default "./" makes the Tauri shell build produce working relative
  // asset URLs out of the box. GitHub Pages workflow overrides explicitly
  // via VITE_BASE_PATH=/LangSlice/.
  base: process.env.VITE_BASE_PATH ?? "./",
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
