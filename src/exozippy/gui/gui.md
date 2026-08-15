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
  Per-project mutable state (open document, open project directory, run
  handle, tune session) lives on closures inside `create_app()` so each app
  instance -- and each per-test app -- is isolated. Blocking work runs off the event loop:
  endpoints that call into the backend are plain `def` (FastAPI runs them in a
  threadpool) and the seconds-long jobs use dedicated `ThreadPoolExecutor`s or
  a worker subprocess/process.
- `document.py` (G8) -- `ProjectDocument`: both user files (system `*.yaml` +
  `*.params.yaml`) as **ruamel round-trip** trees so comments and key order
  survive edits. Edits are reversible `Command` objects (`SetConfigKey`,
  `SetParamField`, `AddComponentInstance`, `DeleteInstance`, `RenameInstance`,
  `DuplicateInstance`, `AssociateDatafile`) with server-side undo/redo stacks.
  `SetParamField` writes through `ProjectDocument.param_key_for`, which finds
  the spelling the params file ALREADY uses for that element and edits it in
  place. The GUI addresses parameters by the NAME form (`star.A.teff`, what
  `introspect` and `export_solution` display) while a file may equally spell
  the same element by INDEX (`star.0.teff` -- `examples/kelt4/kelt4.params.yaml`
  does), and a literal keyed write appended a TWIN rather than updating the
  entry. Both spellings are equally specific, so nothing downstream could
  adjudicate: `ConfigManager` refuses such a file outright (see CLAUDE.md's
  "Parameter naming convention"), and before it did, `standardize_param_names`
  kept whichever key came last -- the GUI's -- silently discarding the user's
  whole original entry, `sigma` prior included. Only the two specific
  spellings are matched; a 2-part broadcast entry is a coarser statement that
  a specific entry legitimately refines.

  **`DeleteInstance` and `DuplicateInstance` are spelling-aware too, and for a
  sharper reason: the INDEX form does not survive a change to the instance
  list.** `star.A.teff` follows its star wherever the list moves it;
  `star.0.teff` follows the index, so deleting an earlier instance repoints it
  at a different body. Both commands used to scan for the `comp.<name>.`
  prefix, so deleting star "A" left every `star.0.*` entry in the file, now
  silently applying to whichever star became index 0 -- a WRONG-ELEMENT bug,
  not litter (review 11.10). The decided semantics, implemented in
  `_retarget_params_for_delete` / `_copy_param_keys`:
  - **The deleted instance's own entries go under BOTH specific spellings.**
    `star.A.teff` and `star.0.teff` name one element; honouring only the first
    leaves a live entry behind.
  - **A survivor's index-form entry whose index shifts is rewritten to the
    NAME form** (`star.1.teff` -> `star.B.teff`). The name form means the same
    element before and after any list mutation, so this converts a fragile
    spelling into a stable one exactly where the fragility would bite.
    Re-indexing was rejected because it fixes today's delete and leaves the
    same trap set for the next one; refusing the delete was rejected because
    it gates an edit the GUI can repair exactly. An UNNAMED instance has no
    name form, so its keys are re-indexed instead -- the same guarantee by the
    only means available.
  - **Everything else is byte-identical.** A survivor at an index *below* the
    deleted one still spells its own element correctly, and the 2-part
    BROADCAST form (`star.teff`) was never specific to the deleted instance --
    it covers whatever elements remain, which is exactly what it said before.
  - **Link expressions get the same treatment**, since a reference is a
    parameter path too (`initval: star.2.teff` retargets just as silently),
    with one difference: a reference to the DELETED instance is respelled to
    its *name* rather than removed, turning a silent mis-address into the loud
    "no instance named 'A'" the name form has always produced.
  - **`DuplicateInstance` is the mirror case.** It reads the source's entries
    under both specific spellings (the prefix scan matched none of an
    index-spelled instance's, so the clone silently arrived with no
    parameters) and writes them under the clone's NAME form -- never its index
    form, and exactly one spelling per element.
  - Either command **raises, atomically**, if the params file already names
    one element under both specific spellings (which `ConfigManager` refuses
    anyway): merging them silently would bury a fault carrying a value, a
    bound or a prior. `Command.apply` restores its own before-snapshot when
    `_do` raises, since a failed command is not on the undo stack and must
    leave nothing behind.

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
  Its wire protocol is **addressed by request id**, and waiting is **bounded**
  -- see "The tune worker protocol" below.
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
  `import exozippy.gui` stays cheap; it is the ONE definition of
  `TERMINAL_PHASES` (`status.py` re-exports it rather than restating it).

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
  thread, since its worker may be mid-solve) and the open document when it
  lives outside the newly opened project (autosaved first if dirty). It also
  records the newly opened directory as the server's current project, which is
  what `/api/files` and the log socket confine to -- that root used to be a
  closure over the LAUNCH project that nothing ever updated. Each of those
  describes the project that was open; leaving them made project B show A's
  solved values and plots, and let an edit typed against them land in B's
  params file. The frontend mirrors this: `TuneTab` is keyed by `configPath`
  so a switch remounts it, and its `ensureDoc` re-opens the document whenever
  the server's open path is not the config it is tuning.
- `WS  /api/logs?file=...` -- tail a log file (follows rotation/truncation).
  Two guards, both invariant 5 (see "Invariants"): the handshake's `Origin`,
  if present, must be loopback (WebSocket handshakes bypass CORS entirely, so
  any page the user visits could otherwise open this socket against the
  loopback server), and `file` must lie inside the open project, the open
  document's directory, or the active run's cwd. The handler races its poll
  loop against `websocket.receive()`, so a tail on a QUIET file ends when the
  client goes away instead of looping forever on an open handle. Content the
  seed tail already sent is skipped once; every later reopen (rotation or
  truncation) streams the new file from its start. Covered by
  `tests/test_gui_logs.py`.

Config document (G8): `POST /api/doc/open`, `GET /api/doc`,
`POST /api/doc/{command,undo,redo,save}`, `POST /api/doc/validate`
(async: returns a job id) + `GET /api/doc/validate/{job_id}`.
`doc/open` is edit-preserving: re-opening the path that is already open
returns the dirty in-memory document unchanged (tabs call open on mount, and
a naive reload-from-disk silently reverted unsaved edits on tab switches); a
clean same-path doc IS reloaded so external file edits are picked up. It also
returns `recovery`: any autosave sidecar newer than its real file, which
`ConfigTab` renders as a banner offering the undoable `restore_autosave`
command. A validation job is retired the first time it is polled in a terminal
state, and abandoned ones are evicted oldest-first past a cap -- the tab starts
a fresh validation after every edit burst, and each entry holds a full
diagnostics list.

Every command goes through `POST /api/doc/command`; bad input answers 400 (that
includes an `IndexError`/`TypeError` from a path that indexes past a list or
traverses a scalar -- the command's snapshot restore has already rolled the
edit back, so the document is intact).

Data manager (G9): `GET /api/files` (browser confined to the CURRENTLY open
project -- a directory outside it is refused, not merely denied a parent link),
`GET /api/browse` (unconfined, for the sidebar project picker),
`POST /api/files/eligible`, `GET /api/files/associations` (both unwired --
see "Known unwired seams"). An unreadable directory answers 400 on both.

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
`GET /api/run/image?path=` (confined to the run's working directory -- i.e.
the whole project dir, not just the results dir -- and serving whatever file
type it finds there; the last two are unwired, see "Known unwired seams"),
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
  `active` flag tells a tab it is the visible one, and passing it is not
  optional bookkeeping: ConfigTab resyncs from `GET /api/doc` on reveal so
  edits from other tabs show up, and TuneTab re-checks `/api/tune/hash` on
  reveal so an edit made in ConfigTab actually raises its stale banner.
- `src/api.ts` -- the single typed client for every endpoint, plus
  `openLogSocket(file)` and `runImageUrl(path)`. Failures throw `ApiError`,
  which carries the HTTP status: at least one caller has to tell a 409 ("the
  evaluator is gone -- Solve again") from a transient blip. Client methods
  with no caller are the unwired seams listed at the bottom of this file and
  say so in a comment; anything else without a caller is dead and should be
  deleted with its endpoint.
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
  **Every field in `ConfigTab` commits `onBlur`, so every one needs a change
  guard.** `onBlur` fires on a plain click-through, and without the guard
  (`unchanged()`) tabbing across a row pushed one undo entry per cell, marked
  the document dirty -- which changes what `RunControl` writes -- and rewrote
  each untouched value through JSON round-trip + `coerce`, so a typed `1e-3`
  became `0.001` on disk and a flow-style `CommentedSeq` came back a plain
  list with its comments gone. `TuneTab.setFieldIfChanged` is the same guard.
  `ConfigTab` also renders the **autosave-recovery banner** from
  `doc/open`'s `recovery` list.

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

**The banner covers edits made ANYWHERE, not only in the Tune tab.** It used
not to: `TuneTab` checked `/api/tune/hash` at mount and after its own commands
only, and tabs stay MOUNTED when hidden, so a bound or prior edited in
`ConfigTab` left the sliders live and still committing RANK_USER initvals
against a model the document no longer described. It now takes an `active`
prop and re-checks the hash on reveal and on a slow timer while visible and
live. Two independent staleness sources are tracked separately and OR'd --
the document/evaluator hash mismatch (re-derivable from the server at any
time) and a verdict the evaluator itself returned (`needs_resolve`, or a 409
from a worker that died and was respawned). One boolean carried both, so any
hash re-check silently cleared a `needs_resolve` banner. A 409 from
`/api/tune/eval` is surfaced through that second channel rather than swallowed
by a bare catch: the status poll stops once the solve goes live, so a
timed-out eval was otherwise invisible until the user next pressed Solve.
Solve and the banner's Re-Solve are both disabled while a solve is in flight.

## The tune worker protocol (`tune.py`)

Two multiprocessing queues carry request dicts one way and response dicts the
other. Two rules make that safe; both exist because the naive version was
racy in ways that were invisible until they were not (review 1.4 and 1.5).

**1. Every request carries a uuid, and the worker echoes it on every message it
sends back** -- the final answer, the error, and the mid-solve
`{"progress": "compiling", "data_plots": [...]}` alike. The parent runs ONE
reader thread that drains the response queue and files each message under the
id it echoes; each caller waits only on its own id, and a message addressed to
an id nobody is waiting on is dropped at the reader (the only place that
decision is made).

This is what lets a slider eval and a Solve be in flight at once, which the
server really does produce: a Solve runs on the single-slot `tune_pool` while
`POST /api/tune/eval` runs on FastAPI's threadpool. Before ids, whichever
thread called `get()` first took whatever message arrived -- an eval could
return the Solve's payload, and the eval's wait silently ATE the Solve's
progress message, losing the data-only plots. `TuneSession.eval`'s phase gate
narrowed that window but could not close it: the gate is read **unlocked** and
`solve()`'s put+await is deliberately **outside** the session lock (taking the
lock across either would serialize the slider behind a seconds-long Solve).
The request id, not the lock, is what makes those unlocked reads safe -- do not
"fix" the lock discipline by widening the lock.

**2. Silence has a deadline, and expiry means terminate + respawn.** A worker
that dies raises immediately (the response queue is polled, not blocked on). A
worker that hangs while still ALIVE -- a pytensor compile deadlock, a wedge in
native code, an OOM-frozen child -- used to wedge every later Solve until the
server was restarted. Now each request has a silence budget
(`SOLVE_TIMEOUT_S = 900`, `EVAL_TIMEOUT_S = 120`; every message restarts the
clock, so a job is only ever killed for going quiet, never for being long).
On expiry the parent terminates the process, replaces BOTH queues (a process
killed mid-write can leave a partial pickle in the pipe), spawns a clean
worker and raises `WorkerTimeout` -- a `RuntimeError`, so `TuneSession.solve`
surfaces it as the error phase and `/api/tune/eval` as a 409. A restart also
bumps a generation counter that releases any OTHER in-flight request at once,
instead of leaving it to burn its own deadline and terminate the fresh worker
in turn. A timed-out eval sets the session to the error phase, because the
respawned worker holds no compiled evaluator: the UI has to Solve again.

The deadlines are deliberately generous and are read at call time (never
captured), so a slow machine can monkeypatch them. A false positive is worse
than a late detection -- terminating a healthy worker throws away a compile
that was about to finish and looks, from the UI, exactly like the bug the
deadline is there to fix.

**3. A healthy worker is reused, but not forever.** Reuse is the main speed
lever (a spawn re-imports pytensor/pymc/exozippy and cold-starts the compile
cache), so `_do_solve` rebuilds the System/model/evaluator in place and drops
the previous one first. What that cannot reclaim is pytensor's compiled C
modules, which can never be `dlclose`d: a long-lived worker's RSS ratchets up
with every solve. `TuneSession._should_recycle` closes and respawns it past a
solve count (`_RECYCLE_AFTER_SOLVES`) or an RSS ceiling (`_RECYCLE_RSS_MB`, via
psutil when readable). This is a different failure from rule 2's: the silence
deadline fires only on a WEDGED worker, and one slowly eating the machine
answers every message promptly right up until it swaps. Both thresholds are
deliberately generous -- recycling too eagerly would simply undo the reuse.

There is **no cancel endpoint**: nothing lets a user abandon an in-flight Solve
early. The deadline covers the wedged case; a deliberate cancel is a separate
feature (it needs a UI affordance and a request-scoped abort, not just
`worker.close()`). The UI does disable Solve and Re-Solve while one is running,
so a second cannot be queued behind the first.

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
   path-restricted to their intended tree -- really restricted, not merely
   documented as such: `/api/files` and `WS /api/logs` both REFUSE a path
   outside their root, and `datafiles.is_within` is the one predicate that
   answers the question (there were two, with different symlink behavior). A
   WebSocket endpoint additionally needs its own `Origin` check, since
   handshakes are not subject to CORS and binding to loopback does not stop a
   page the user visits from connecting.

## Testing

A **kept** seam still has to be tested end to end on the server side, or it
rots into something that only looks alive: `/api/files/eligible`,
`/api/files/associations`, `/api/run/plots` and `/api/run/image` are all
exercised by the files below even though no frontend calls them today.

Fast GUI tests (fastapi TestClient, no real compile): `tests/test_gui_app.py`,
`tests/test_gui_document.py`, `tests/test_gui_data.py`, `tests/test_gui_tune.py`,
`tests/test_gui_logs.py`, `tests/test_run_endpoints.py`. Real-compile /
real-fit paths are marked `slow`.
Run the set with `poetry run pytest tests/test_gui_*.py tests/test_run_endpoints.py -m "not slow"`.

The frontend has no test harness, so anything it and the server must agree on
is pinned SERVER-side by parsing the source: `test_gui_document.py` reads
`ConfigTab.tsx`'s `PARAM_FIELDS` literal and asserts `set_param_field` accepts
every column the table renders. That pair had already drifted -- `bound_scale`
was rendered and 400'd on every blur, because `_PARAM_FIELDS` was defined as
`set(LINKABLE_FIELDS)` and `bound_scale` is deliberately not linkable.

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
