# EXOZIPPy GUI

> **Status: experimental.** The GUI is still buggy and has never been
> verified driving a real fit end to end, on any platform. It is not part
> of what the README means by a supported platform, and nothing in CI
> exercises it beyond unit tests of the modules below. Treat everything
> here as a description of intent as much as of proven behavior.

A local, **optional** graphical wrapper around the EXOZIPPy backend. Nothing
here is required for the scripting/CLI workflow, and no component-specific
knowledge is hardcoded in the GUI: it consumes only the contracts that
components declare (introspection schema, utility registry, PlotSpec, solve
provenance). A future component author gets the GUI for free.

This document describes how the GUI is built so a developer (or Claude Code)
can extend it without re-reading every file.

## Running it

- Entry point: `exozippy-gui` (console script -> `exozippy.gui.app:main`).
  Optional positional arg `[project]`: a project directory, or a specific
  config file relative/absolute (e.g. `exozippy-gui kelt4.yaml`) -- resolved
  by `resolve_project_arg()` into `(project_dir, initial_config)`, where a
  file's parent dir becomes the project and the file is pre-selected in the
  Config tab. Defaults to the current directory when omitted. Other flags:
  `--browser` (open a browser tab instead of a native window), `--no-window`
  (serve only, for tests), `--host`, `--port` (default: an OS-assigned free
  port).
- Dependencies are the optional `gui` extra: `pip install exozippy[gui]`
  (developers: `poetry install -E gui`). The plain CLI and `import exozippy`
  must keep working WITHOUT the extra -- every GUI import is guarded, and there
  is a test asserting `import exozippy` does not import fastapi.
- Two-process-at-runtime model: a **FastAPI** server (uvicorn on 127.0.0.1
  only, in a background thread) plus a **pywebview** native window (falls back
  to `webbrowser.open`). The React frontend is a prebuilt static bundle served
  by FastAPI -- end users never need Node.

## The backend contracts it wraps (Phase 1, G1-G6)

The GUI is deliberately thin. All the hard work lives in core modules that are
useful to scripting too; the server just exposes them over HTTP/JSON:

| Module | What it provides | GUI use |
|--------|------------------|---------|
| `exozippy/introspect.py` | `full_schema()` -- every component's parameters + config keys, JSON-safe, no System needed | drives all auto-generated forms and the component-agnostic menus |
| `exozippy/utilities/registry.py` | `all_utilities()`, `UtilitySpec.to_schema()`, `run_utility()` | the Tools menu + utility runner |
| `exozippy/solve_api.py` | `solve()` (values/bounds/priors + provenance), `validate()` (structured diagnostics) | the Tune-tab Solve button + Config validation |
| `exozippy/plotspec.py` | `PlotSpec` contract; `Component.plot_data(system, point=None)` | all charts (data-only previews + model traces) |
| `exozippy/evaluator.py` | `compile_evaluator()`, `Evaluator.set_value/eval_plots/structural_hash` | millisecond live-slider plot updates |
| `exozippy/gui/runner.py` + `status.py` | subprocess fit launch, status/snapshot files, graceful stop | the pinned RunControl |

If you are tempted to write component logic in the server or frontend, stop and
push it into one of these contracts instead.

## Server (`src/exozippy/gui/`)

- `app.py` -- the FastAPI app factory (`create_app`) and `main()`. Owns the
  HTTP/WebSocket surface, static-bundle serving (with an SPA/placeholder
  fallback), free-port selection, the uvicorn thread, and the pywebview window.
  Per-project mutable state (open document, run handle, tune session, preview
  cache) lives on closures inside `create_app()` so each app instance -- and
  each per-test app -- is isolated. Blocking work runs off the event loop:
  endpoints that call into the backend are plain `def` (FastAPI runs them in a
  threadpool) and the seconds-long jobs use dedicated `ThreadPoolExecutor`s or
  a worker subprocess/process.
- `document.py` (G8) -- `ProjectDocument`: both user files (system `*.yaml` +
  `*.params.yaml`) as **ruamel round-trip** trees so comments and key order
  survive edits. Edits are reversible `Command` objects (`SetConfigKey`,
  `SetParamField`, `AddComponentInstance`, `DeleteInstance`, `RenameInstance`,
  `DuplicateInstance`, `AssociateDatafile`) with server-side undo/redo stacks.
  `RenameInstance` rewrites every cross-reference (orbit body groups, `band:`,
  `star_ndx`/`orbit:` keys, and `linking.py` expressions) purely from the
  schema -- no hardcoded component names. Undo uses TEXT snapshots, not
  `deepcopy` (ruamel drops comments on deepcopy). `command_from_json` dispatches
  the API command payloads. Both files are screened on the read from disk by
  the shared `exozippy.yamlio.check_yaml_booleans`, which refuses YAML-1.1-only
  boolean spellings (`yes/no/on/off`): ruamel is YAML **1.2** and reads those as
  strings while the fit's PyYAML is YAML **1.1** and reads them as booleans, so
  `finite_source: no` was `False` to the fit and the truthy string `"no"` here
  -- the editor showed, and could save, the opposite of what the fit does. The
  fit's own loaders call the same guard, so neither side can accept a spelling
  the other reads differently; `true/True/TRUE/false/False/FALSE` are the
  agreement set, and quoting is the escape hatch when a string was meant.
- `datafiles.py` (G9) -- pure, component-agnostic helpers: `list_directory`
  (project-rooted browser that cannot escape the root), `eligible_associations`
  (which instance/key a filename may attach to, by matching the schema's
  `kind: "datafile"` globs), `current_associations` (chip data). The latter two
  are the unwired association seam -- see "Known unwired seams".
  (There is no `preview.py`/`preview_worker.py` any more: a per-component
  data-file preview subprocess served the removed Welcome/Data tabs, and the
  Tune landing's `/api/tune/plots/data` -- the same `plot_data(point=None)`
  specs, built in the tune worker process -- superseded it. Deleted 2026-08-12
  with its `/api/preview` endpoint and mtime cache; recover from git history if
  a per-component preview is ever wanted without a full solve.)
- `tune.py` (G10) -- `TuneSession` (server-side phase tracking:
  solving -> compiling -> live -> error) driving an `EvaluatorWorker`, a
  dedicated **worker process** (spawn context, request/response over
  multiprocessing queues) that holds the System/model/`Evaluator` so pytensor
  compile + eval stay off the API event loop. One session per open project.
- `runner.py` + `status.py` (G6) -- `start_run(config, cwd) -> RunHandle`
  (launches `python -m exozippy.cli <config>` as a fresh subprocess with
  `EXOZIPPY_GUI_SNAPSHOT=1`), `RunHandle.status()/stop(force=)`,
  `list_runs(dir)`; `GuiReporter` writes the atomic `_gui_status.json` +
  `_gui_snapshot/` artifacts the samplers emit at each convergence check.
  **A crashed run must never render as the previous run's success.** Every run
  at a prefix writes the same status file, so `start_run` mints a run id per
  launch, passes it in `EXOZIPPY_GUI_RUN_ID`, `GuiReporter` stamps it into the
  document, and `RunHandle.status()` refuses a document carrying any other id
  -- a fit that dies before writing anything then reports "unknown" (alive:
  "starting", dead: "error"), never someone else's "done". `start_run` also
  captures the child's stdout+stderr to `<prefix>_gui_console.log`: a crash
  before `run_fit` installs its reporter (an unreadable config, an import
  error) -- and any crash the interpreter cannot catch (SIGKILL, OOM) -- has
  no other trace, and with the streams inherited its traceback landed in
  whatever terminal started the GUI. `status()` reports the exit status plus
  that tail as the run's `error`; the capture is best-effort and a failure to
  open it only logs. In-process crashes still record their traceback through
  `GuiReporter.terminal(error=...)` (the PR #46 mechanism, called from
  `run.run_fit`) -- this is the path for everything that never reaches it.
- `__init__.py` -- intentionally light (no eager fastapi/numpy imports) so
  `import exozippy.gui` stays cheap; exports `TERMINAL_PHASES`.

## HTTP / WebSocket API

All under `/api`, JSON in/out, served on 127.0.0.1 only.

Core (G7):
- `GET /api/health` -- liveness.
- `GET /api/config` -- client bootstrap: `{initial_project, initial_config}`,
  which project to auto-open and (optionally) which config file within it to
  pre-select in the Config tab.
- `GET /api/schema` -- `introspect.full_schema()`.
- `GET /api/utilities` -- utility argument schemas (G2 registry).
- `POST /api/project/open` `{path}` -- classify a dir's yaml/data files, and
  RESET the per-project server state: the Tune session (closed on a detached
  thread, since its worker may be mid-solve), the preview cache, and the open
  document when it lives outside the newly opened project (autosaved first if
  dirty). Each of those describes the project that was open; leaving them made
  project B show A's solved values and plots, and let an edit typed against them
  land in B's params file. The frontend mirrors this: `TuneTab` is keyed by
  `configPath` so a switch remounts it, and its `ensureDoc` re-opens the
  document whenever the server's open path is not the config it is tuning.
- `WS  /api/logs?file=...` -- tail a log file (follows rotation/truncation).

Config document (G8): `POST /api/doc/open`, `GET /api/doc`,
`POST /api/doc/{command,undo,redo,save,autosave}`, `POST /api/doc/validate`
(async: returns a job id) + `GET /api/doc/validate/{job_id}`.
`doc/open` is edit-preserving: re-opening the path that is already open
returns the dirty in-memory document unchanged (tabs call open on mount, and
a naive reload-from-disk silently reverted unsaved edits on tab switches); a
clean same-path doc IS reloaded so external file edits are picked up.

Data manager (G9): `GET /api/files` (project-rooted browser),
`GET /api/browse` (unconfined, for the sidebar project picker),
`POST /api/files/eligible`, `GET /api/files/associations` (both unwired --
see "Known unwired seams").

Run controls (G11): `POST /api/run` (one active run per project; copies the
exact config/params into the output dir as `.used.*` for reproducibility -- the
params file is the config's own `parameter_file`, the one the fit subprocess
reads, unless the request names another; this is content, complementary to the
structural fingerprint `trace_meta.py` stamps into the trace, which is a hash
and is deliberately blind to initval/mu values),
`GET /api/run/status` (adds `terminal` -- the run is over, however it ended --
plus `run_id`, `stale_status`, `returncode`, `error` and `console_path`, so
`RunControl` can offer Run again and show WHY a run stopped instead of leaving
a Stop button on a dead process), `POST /api/run/stop`, `GET /api/run/plots`,
`GET /api/run/image?path=` (path-restricted to the run tree via
realpath+commonpath -- the last two are unwired, see "Known unwired seams"),
`POST /api/utilities/run`.

Tune (G10): `POST /api/tune/solve` (the tune worker process is spawned by the
solve itself; there is no separate prewarm call -- the Tune tab is the landing
page and auto-Solves on mount, so a prewarm had nothing left to overlap with),
`GET /api/tune/status`,
`GET /api/tune/result`, `POST /api/tune/eval`, `GET /api/tune/hash`,
`GET /api/tune/plots/data` (data-only PlotSpecs, available from the
"compiling" phase on -- the worker builds them right after `prepare()` via
`plot_data(point=None)` and ships them with its progress message, so the
Tune tab draws the observations while the model compiles).

## Frontend (`gui/frontend/`)

React + TypeScript + Vite. The **built** bundle is committed to
`src/exozippy/gui/static/` and shipped in the wheel; Node is a dev-only
dependency. See `gui/frontend/README.md` for the dev/build loop.

- `src/main.tsx` -- entry; mounts `App`.
- `src/App.tsx` -- the shell: top bar + left sidebar + center tabbed workspace
  + bottom log terminal. Tabs are registered in the `TABS` array; each tab's
  `render` receives a shared `TabContext`
  `{listing, setLogFile, configPath, setActiveTab, active}`.
  Current tabs: Config, Tune, Tools (no Welcome tab -- the backend-health
  line lives at the bottom of the sidebar and the Config empty state is the
  "open a project" landing; no Data tab -- its association flow is the
  Config form's working Browse button, wired to the undoable
  `associate_datafile` command, and data previews are the Tune landing; no
  Run tab -- a pinned bottom-right `RunControl` saves the open document and
  then launches the fit, so the run is always THE CONFIGURATION ON SCREEN,
  with progress/Stop inline and the log attached to the bottom terminal).
  **Landing rule** (startup and
  sidebar project-open alike): when the app knows which config to run -- an
  explicit file on the command line, or a project with exactly one config --
  it lands on Tune, which AUTO-RUNS the first Solve, so the first screen is
  the data with the solved model over it; several configs (or none) land on
  Config. Tabs stay MOUNTED once visited
  (hidden with `display: none`, and a window resize is dispatched on reveal so
  plotly recomputes sizes) -- unmount-on-switch would discard tab state; the
  `active` flag tells a tab it is the visible one (ConfigTab resyncs from
  `GET /api/doc` on reveal so edits from other tabs show up).
- `src/api.ts` -- the single typed client for every endpoint, plus
  `openLogSocket(file)` and `runImageUrl(path)`. Client methods with no caller
  are the unwired seams listed at the bottom of this file and say so in a
  comment; anything else without a caller is dead and should be deleted with
  its endpoint.
- `src/components/PathPicker.tsx` -- the ONE server-side path browser, used
  twice: the sidebar picks a project **directory** (`select="dir"`, unconfined
  `api.browse`, confirmed with a button) and the Config form picks a data
  **file** (`select="file"`, project-rooted `api.files`, `relativeToRoot` so
  the stored path is relative to the config dir, picked by clicking the row).
  It browses through the server rather than a native OS dialog so it behaves
  identically in the pywebview window and in a plain browser tab. Both call
  sites had their own near-identical copy until 2026-08-12 (review 4.13); add
  a prop rather than a third copy.
- `src/plotspec.ts` -- TypeScript mirror of `plotspec.py`'s PlotSpec.
- `src/plotly-adapter.ts` -- the ONE place PlotSpec trace roles map to plotly
  encodings (data = markers+error bars, model = line unless kind "scatter").
  It is the GUI half of a two-renderer pair: `src/exozippy/plotrender.py`
  renders the SAME specs with matplotlib for the saved PDFs, and the two
  share a meta/style vocabulary (see plotrender.py's module docstring) --
  extend both together. Plots draw on WHITE figure cards with matplotlib's
  tab10 colors by fixed `series_index`, red model curves, and matched
  legend/reference-line conventions, so a GUI chart looks like its PDF.
  The dark app chrome around the cards is styled separately in styles.css.
- `src/components/PlotView.tsx` -- thin wrapper over `plotly.js-dist-min`
  (`Plotly.react`, no react-plotly.js) so repeated renders patch in place.
- `src/components/` -- shell parts (`TopBar`, `Sidebar`, `LogTerminal`, the
  pinned `RunControl`) and the tab bodies (`ConfigTab`, `TuneTab`,
  `ToolsTab`).

## The signature interaction: Solve, then live sliders (G10)

Hybrid model. The first Solve runs AUTOMATICALLY when the Tune tab mounts
with a config open (it is the landing page; re-Solves stay manual) -> the
server runs `solve()` (relaxation engine, seconds) then `compile_evaluator()`
(pytensor compile, seconds) in the tune worker process; the data-only plots
render as soon as the relaxation finishes (see `/api/tune/plots/data`), then
the panel fills with values + provenance, the model traces land at the solved
start point, and the app enters LIVE mode. Slider drags then call
`POST /api/tune/eval` (debounced ~50 ms) -> `Evaluator.set_value` (inverts the
slider's user-unit value into a new raw point) + `eval_plots` -> updated
traces patched into the charts in milliseconds. Model traces always; DATA
traces too (with errors) on specs whose meta declares `dynamic_data` --
phase-folded panels re-fold the observations with tc/P, RV panels subtract
gamma, mulens panels re-align onto the reference flux system -- otherwise a
tc slider moves nothing visible (the phased model grid is tc-anchored) and
the panel looks frozen. Components whose data prep uses point values in
numpy must also list those params explicitly in `param_deps` (rv gamma,
transit baseline): the graph walk cannot see them, and a missing dep means
the `changed_label` filter skips the component entirely.
The Tune layout is a two-column split: the parameter tree (the selected row
expands into the slider/bounds/prior editor inline, right below itself) and
a responsive plot GRID (as many ~420px panels as fit). `eval_plots` re-renders by
calling each affected component's own `plot_data(system, point)` again at the
new point -- the SAME code that built the base specs and that the CLI's
matplotlib `plot()` reuses -- rather than a second, parallel plotting
implementation; the only optimization is a single cached raw->internal-point
pytensor function (`Evaluator.internal_point`, built once per Solve) plus an
optional `changed_label` filter that skips components the moved parameter's
`param_deps` don't cover. This is what makes phase-folded curves (sorted/
column-selected from a multi-orbit node) and SED spectra (NumPy spectral-
library interpolation) update live along with everything else -- an earlier
affine-calibrated-pytensor fast path could not recover either.
Any structural change (bound/prior/fixed edit, add/remove component) flips the
`structural_hash`; the UI shows a "Config changed -- re-Solve" banner and
freezes the live plots until the next Solve. Slider/bound/prior edits are still
real G8 `set_param_field` commands (undoable, RANK_USER, saved to params.yaml);
the Tune toolbar has its own Save button (the shared document's dirty flag)
so tuned values can be written to disk without leaving the tab.

## Invariants (do not break these)

1. **Component-agnostic.** No `if comp_type == "transit"` in server or frontend
   code. If you need per-component behavior, it belongs in a component-declared
   schema/contract, not here.
2. **Optional + guarded.** `import exozippy` and the CLI must work without the
   `gui` extra. Keep GUI imports lazy/guarded.
3. **Round-trip YAML.** All config writes go through `ProjectDocument`
   (ruamel) so the user's comments and ordering survive.
4. **Process isolation for heavy/blocking work.** Fits run as subprocesses
   (never threads -- GIL + pytensor compile locks). The tune evaluator runs in
   its own worker process. Never block the event loop.
5. **Local only.** The server binds 127.0.0.1. File-serving endpoints must stay
   path-restricted to their intended tree.

## Testing

A **kept** seam still has to be tested end to end on the server side, or it
rots into something that only looks alive: `/api/files/eligible`,
`/api/files/associations`, `/api/run/plots` and `/api/run/image` are all
exercised by the files below even though no frontend calls them today.

Fast GUI tests (fastapi TestClient, no real compile): `tests/test_gui_app.py`,
`tests/test_gui_document.py`, `tests/test_gui_data.py`, `tests/test_gui_tune.py`,
`tests/test_run_endpoints.py`. Real-compile / real-fit paths are marked `slow`.
Run the set with `poetry run pytest tests/test_gui_*.py tests/test_run_endpoints.py -m "not slow"`.

Note: the repo's pre-commit hook runs the FULL suite, which hangs on a cold
pytensor cache in a fresh worktree; GUI work is typically committed after
running the targeted GUI tests.

## Extending

- **New tab:** add a `*Tab.tsx` under `components/`, register it in `App.tsx`'s
  `TABS` (use the shared `TabContext`), and add any client methods to `api.ts`.
- **New endpoint:** add it inside `create_app()` in `app.py`, additively; keep
  it a plain `def` if it blocks, and hang any state off the per-app closures.
- **New utility:** declare a `UtilitySpec` on the owning component
  (`get_utilities()`); it surfaces in the Tools menu automatically via the G2
  registry -- no GUI change needed.
- After frontend changes, run `npm run build` in `gui/frontend/` and commit the
  refreshed `src/exozippy/gui/static/` bundle.

## Status / roadmap

Implemented: Phase 1 backend contracts (G1-G6) and Phase 2 core GUI (G7 shell,
G8 config editor, G9 data manager, G10 tune panel, G11 run controls).

Known unwired seams (polish backlog). Each is a **live, tested backend
contract with a typed client method and no caller** -- kept deliberately, not
overlooked. Anything not on this list and not called is dead; delete it with
its endpoint rather than growing the list.

1. **Associate a produced file** (G9 stub). The Tools tab renders a disabled
   "Associate" button next to each file a utility produced
   (`ToolsTab.tsx`, `TODO(G9)`). Wiring it up is `api.filesEligible(filename)`
   for the instance/key menu, then the existing undoable `associate_datafile`
   command; `api.filesAssociations()` is the same contract's other half (which
   instances already reference a file -- the chip data the removed Data tab
   drew). Server: `POST /api/files/eligible`, `GET /api/files/associations`,
   over the pure helpers in `datafiles.py`.
2. **The run-plot gallery** (`api.runPlots` + `runImageUrl`, served by
   `GET /api/run/plots` and `GET /api/run/image`). This one is a **dropped
   feature, not a backlog item**: the removed Run tab polled `/api/run/plots`
   on the same tick as the status poll and rendered the run's
   `<prefix>_start*` and `<prefix>_mcmc*` images as a thumbnail grid, each
   thumbnail linking to the full-size file (the CSS, `.plot-gallery` /
   `.gallery-grid`, is still in `styles.css`). It went with RunTab in commit
   `75bf1ba` (2026-08-03) and nothing replaced it, so a running fit's plots
   are reachable only through the filesystem.
   **It never actually displayed an EXOZIPPy plot**, though, and that is the
   real work in restoring it: `plotrender` writes `{prefix}_{tag}.pdf`, while
   `app.py`'s `_IMAGE_EXTS` has listed only raster formats + svg since G11, so
   `/api/run/plots` returns empty lists for every real run (`_corner.png` is
   the one raster the fit writes, and it matches neither glob). The gallery
   therefore rendered for its fixtures and stayed blank in practice.
   Restoring it means deciding how PDFs reach a browser -- serve them and
   render an `<embed>`/link grid rather than `<img>`, or have the fit emit
   raster thumbnails alongside -- and then the UI is small: the two
   `<section className="plot-gallery">` blocks from
   `git show 75bf1ba^:gui/frontend/src/components/RunTab.tsx`, plus an
   `api.runPlots()` call on RunControl's existing poll. The endpoints, the
   `RunPlots` type, the client methods and the CSS are all kept so that
   remains the whole job; the format decision is why this is recorded rather
   than patched over. Where it should live -- expanded out of RunControl, or
   as the G13 results browser -- is the other open question.
   `tests/test_run_endpoints.py::test_run_plots_lists_raster_images_but_not_pdfs`
   pins the current behavior and is where the format fix would show up.

(The old Run-button doc-dirty gating seam is gone: RunControl saves the
document before every launch.)

Not yet built (Phase 3): G12 node canvas (React Flow view over the G8
document), G13 results browser, G14 run queue + settings + packaging.
