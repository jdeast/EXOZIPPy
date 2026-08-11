import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production bundle is committed to the wheel at
// src/exozippy/gui/static/ and served by FastAPI. `npm run dev` runs a Vite
// dev server that proxies /api (HTTP + WebSocket) to the FastAPI dev server;
// set EXOZIPPY_API_PORT to match the port `exozippy-gui` printed.
const apiPort = process.env.EXOZIPPY_API_PORT || "8000";
const apiTarget = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  plugins: [react()],
  build: {
    // Emit the built bundle straight into the Python package's static dir.
    outDir: "../../src/exozippy/gui/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
