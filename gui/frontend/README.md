# EXOZIPPy GUI frontend

React + TypeScript + Vite sources for the EXOZIPPy GUI. The **built** bundle is
committed to `src/exozippy/gui/static/` and shipped in the wheel, so end users
install with `pip install exozippy[gui]` and never need Node. Node/npm is a
**dev-only** dependency for rebuilding this bundle.

## Develop

```bash
cd gui/frontend
npm install

# In one terminal: start the Python API (note the port it prints).
poetry run exozippy-gui --browser --no-window --port 8000

# In another: start Vite with HMR. It proxies /api (HTTP + WebSocket) to the
# FastAPI server. Set EXOZIPPY_API_PORT if you used a non-8000 port above.
EXOZIPPY_API_PORT=8000 npm run dev
```

Vite serves the app at http://localhost:5173 and forwards `/api/*` to FastAPI.

## Build (refresh the committed bundle)

```bash
cd gui/frontend
npm run build      # type-checks, then writes src/exozippy/gui/static/
```

`vite.config.ts` sets `build.outDir` to `../../src/exozippy/gui/static`
(`emptyOutDir: true`). **Commit the refreshed `static/` bundle** whenever you
change frontend sources -- CI/end users run the committed bundle, not a fresh
build.

## Layout

- `src/api.ts` -- typed client for the FastAPI endpoints.
- `src/plotspec.ts` -- TypeScript mirror of `plotspec.py`'s PlotSpec contract.
- `src/plotly-adapter.ts` -- the one place PlotSpec roles map to plotly traces.
- `src/components/` -- shell (TopBar, Sidebar, LogTerminal), PlotView, tabs.
- `src/fixtures/` -- canned PlotSpec used by the Welcome tab demo (no fit
  needed).
- `src/assets/` -- brand logo + wordmark.
