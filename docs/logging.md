# What EXOZIPPy logs, and at which level

Read this before adding or changing a log message.

The console is for the user watching a fit. It must tell them where the run
is and, above all, what they have to act on. A wall of confirmations buries
the one line that matters: on a microlensing fit in 2026-11 the most
consequential message of the setup -- "no zeropoint prior stated ... so
f_source is NOT tied to the SED", after the user's own zeropoint key had been
silently ignored -- was one INFO line among 221, sandwiched between
"zeropoint magnitude system Vega" lines. This page is the rule that keeps
that from happening again.

`<prefix>.log` keeps everything at DEBUG (`logger.setup_logging`: the file
handler is always DEBUG, the console is `logger_level:` in the system config,
default INFO), so demoting a message to DEBUG loses nothing: it is still in
the file, with a timestamp and the module that wrote it.

## The four levels

**ERROR** -- the run cannot continue, or a result is invalid. Usually the
code raises instead; log at ERROR only where the run deliberately carries on
past the failure (a wrap-up stage that records its failure before
re-raising, a worker that died).

**WARNING** -- the user should know, and usually act. The test: *would a
careful user change their config, or how they read the result, after
reading this?* That covers:

- a user input ignored, overridden, or not reproduced by the model;
- an approximation or modeling choice made on the user's behalf that changes
  the result (a tie switched off, a grid edge clipped where the fit is, a
  fallback model);
- a bad start, a parameter near a bound, a convergence problem, a data
  problem.

It does NOT cover a default taking effect, a choice that follows from what the
user wrote, or internal bookkeeping a user cannot act on (a grid trimmed far
from where the fit is) -- those are DEBUG.

**INFO** -- a short narrative of the run, so a user watching it knows where
it is:

- ONE line summarizing the model (`Model: star (3), lens (2), ...`), not a
  line per component type;
- stage milestones, with timing where a stage takes long (`Done solving
  (0.16 s)`, the seed-polish summary, the wrap-up stage announcements);
- files written, aggregated per stage (`Start plots written: <prefix>_start_*`),
  not one line per file;
- progress at a coarse cadence (see below), plus the start and finish of a
  long stage;
- the startup parameter table (`run.inspect_start`) and the final summary.

**DEBUG** -- everything else: confirmations that a default or a config choice
took effect, internal bookkeeping (grid trimming and re-axing internals,
whitening details, worker counts, per-rung start statistics), per-instance
decisions that follow from the config, per-step diagnostics.

When in doubt between INFO and DEBUG, it is DEBUG. When in doubt between
WARNING and INFO, apply the test above; if the honest answer is "no, but it
is good to know", it is DEBUG, not INFO.

## Format

**Component prefix.** A message from a component starts with its tag,
`Component.log_tag(i)`: `[<component>]` for one about the component as a whole
(`[star] ...`), `[<component> <instance name>]` for one about one element
(`[mulensinstrument OGLE4] ...`), where the instance name is what the user
wrote in the config. Not `mulensinstrument OGLE4: ...`, not `Band I: ...`, not
`SED.plot: ...`. Non-component modules use a short stage name the same way
when one helps (`Seed polish: ...`, `Whitening: ...`, `PTDE: ...`). A message about
one parameter element names it by its params-file key,
`Parameter 'orbit.b.period': ...`, because that is the line to edit.

**A WARNING** is:

1. one headline sentence, on the first line: what happened, and its
   consequence;
2. then, after a line break, the remedy: what to write, and where (`set
   mulensinstrument.<name>.zeropoint: {mu, sigma}` in the params file);
3. a list of several things (keys, files, instances) as bullets, one per
   line, built with `logger.bullets(items)` rather than joined with `; `.

Length is not a limit in itself -- say what the user needs -- but keep it
short enough to read in one glance; the detail behind it belongs at DEBUG.

It names the parameter, component instance or file it is about. A WARNING
that reports a caught exception gives the exception in the headline
(`... failed (ValueError: ...); <what is missing>`) and logs the traceback at
DEBUG, so the file has it and the console does not. It contains
no numpy reprs (`np.float64(2300.0)` -- format numbers with `:g` or
`float()`) and no dict or list dumps (`{'teff': (...)}` -- say it in words).
The headline is what the recap repeats (below), so it must stand on its own.

**Wrapped at 80 columns.** Every WARNING and ERROR the console and the file
log print is wrapped at `logger.WRAP_COLUMNS` (80) by `logger.wrap_message`,
which `logger.WrappingFormatter` applies to both handlers -- so a call site
writes the message as one logical line per paragraph and never wraps by
hand. The layout is fixed in that one place (`examples/kelt4`, RV only, as
the console prints it):

```
The seed polish moved 2 of your start values by more than 3 sigma:
  - orbit.0.cosi: 0.11996 -> 0.999944 (1.6e+05 sigma)
  - planet.0.radius: 1.706 -> 1.0041 (16 sigma)
  Set "sampler: {seed_polish: false}" to start at the seeds or "sigma: 0" in the
  params file to fix it throughout the fit.
```

The headline starts flush left and its continuation is indented two spaces;
each later line the message breaks explicitly (`\n`) is indented two spaces;
a later line starting with `- ` is a bullet (`  - `, continuation under its
text). Words are never split, so a long path stays on a line of its own.
INFO and DEBUG are not wrapped (the startup table and progress lines keep
their own layout), and the record itself is untouched, so the recap and
pytest's caplog see the message as written. An exception whose text is a
remedy for the user (the unknown-key raises) calls `wrap_message` itself;
the function is idempotent, so a message wrapped twice is unchanged.
`tests/test_logging_policy.py` checks the layout, both handlers, the
unknown-key raises, and every WARNING of a shipped example's setup.

**Plain ASCII**, as everywhere in the codebase (CLAUDE.md): no box-drawing
banners (a file log written in the locale's encoding turned one into a line of
`????`), no arrows, multiplication signs or Greek letters (write `->`, `x`,
`gamma`). A warning that needs to stand out is a WARNING; it
does not need a frame.

**The same situation gets the same phrasing** wherever it is logged, and is
logged ONCE: a decision that applies to several instances is one aggregated
line naming them (`[star] nothing reads feh (Lens, Companion)`), not a line per
instance. `System.prepare` may run stages 3 and 4 twice (a topology revision
or a start re-read); `system._RepeatedRecords` drops exact repeats of the
first pass's records on the second, so a message whose text does not change
is not printed twice. A message that interpolates something that changes
between the passes defeats that filter -- keep such numbers out of a
per-pass message or log it at DEBUG.

## Progress cadence

A long stage logs at most one INFO progress line per
`logger.INFO_PROGRESS_S` (300 s) of wall clock, plus its start and finish
lines. Finer heartbeats (the seed polish's 30 s `POLISH_PROGRESS_S`, the
whitening probe's 60 s `PROBE_PROGRESS_S`) keep their own intervals but go to
DEBUG in between; `logger.ProgressCadence` decides which line is promoted.
The finer cadence exists to answer "computing or hung?" (run.md), and the file
log, which keeps DEBUG, is where that question is answered.

A WARNING whose cause is reached several times per run with different
numbers (once per light curve, once per solve pass, once per seed) logs the
detail at DEBUG each time and the headline once, guarded by
`logger.first_time(key)` (reset by `setup_logging` and `System.__init__`).

## Warnings must not be buried: the recap

`logger.WarningRecap`, installed by `setup_logging`, collects every WARNING
(and ERROR) record. `run._run_fit` repeats their headlines -- one line each,
identical headlines counted (`(x3)`) -- right before sampling starts
("Recap: N warning(s) during setup") and again when the run ends ("... during
this run"). A recap is itself one WARNING record marked so the collector does
not collect it. Nothing threads warning lists through call sites: log the
warning where it happens and the recap finds it.

## `warnings.warn` and `print`

Library code logs. `warnings.warn` is reserved for a genuine Python-API
`DeprecationWarning` (a caller's code should change), where Python's own
filters are the right channel; a message about a FIT goes through the logger,
which writes it to the fit's log file and into the recap -- a
`warnings.warn` does neither, and Python shows each one once per process.

`print` is for a command-line tool's deliberate output (a script whose
product is text on stdout: `exozippy-summary`, `scripts/`, the data-fetching
utilities' progress, the table-generation scripts under `models/`). Inside the
fitting library a `print` bypasses the log file and the console level, so it
is a logger call.

## User input is never silently ignored

A key the code does not recognize is a typo or a misplaced setting, and
ignoring it delivers something other than what the user wrote with no
message (CLAUDE.md, "No fallbacks for internal invariants"). So an unknown
key RAISES, at both levels of the system config:

- **Top level** (JDE 2026-11-08). Every top-level key must be a registered
  component or one of `system.RESERVED_CONFIG_KEYS`; `System.__init__`
  checks them all before building any component and raises naming every
  unknown key, with a did-you-mean among the components and reserved blocks
  (`'samplr': did you mean 'sampler'?`). A key naming a component module
  that failed to import raises that ImportError instead -- a different fix.
  This used to be a "will be ignored" WARNING.
- **Inside a component block.** Every entry is checked against its
  component's `config_schema()` at construction
  (`Component.validate_config_keys`, called by `System.__init__`), and an
  unknown key raises, naming the key and the component, and suggesting the
  right spelling where there is an obvious one (a near-miss of a known key,
  or a parameter that belongs in the params file -- `zero_point:` in a
  `mulensinstrument` block is `mulensinstrument.<name>.zeropoint: {mu,
  sigma}` in the params file).

Spellings are never case-folded: `Star:` raises, suggesting `star`. The
keys INSIDE a reserved block (`sampler:`, `modes:`, ...) still warn
(`run.warn_unknown_sampler_keys`, `run.warn_unknown_block_keys`).

## The seed polish's moves: judged in sigma, not percent

`run.report_polish_moves` lists every user start value the seed polish moved
at DEBUG, and warns about a move only when it is larger than
`run.LARGE_POLISH_MOVE_SIGMA` (3) of that element's own 1-sigma width at the
start (JDE 2026-11-08). The width is `Parameter.start_width`: for a bounded
element the startup probe's measured 0.5-nat step (`init_scale` after
whitening, src/exozippy/whitening.md), for a Gaussian-prior element its prior
sigma times the probe's measured factor, and the preliminary scale when
nothing was measured (`measure_scales: false`). An element the sampler does
not move (derived, pinned) has no width and never warns: its move follows
from the sampled elements it is computed from, which are judged themselves.

A percentage is right only by coincidence of units: a noise variance that
moves 10% is irrelevant, and a 1% move in t_0 can be hundreds of sigma. Why
3: the probe's width is the CONDITIONAL one (curvature along one axis, the
others held), which is at most the marginal width, so a good seed can sit a
sigma or two off in this unit through ordinary correlations; past 3 sigma the
move costs at least ~4.5 nats along that axis alone, which a seed fitted to
the same model and data does not. Measured on the shipped examples (setup to
the sampling boundary, `polish_timeout: 300`; the number of parameters
named in the one warning):

| example | 10% rule (before) | 3 sigma (after) |
|---|---|---|
| kelt4 `kelt4_rvonly.yaml` | 2: cosi, planet radius | 2: the same (cosi 1.6e5 sigma, radius 16 sigma) |
| ob140939 | 1: Spitzer err_scale | 1: the same (5.9 sigma) |
| hat3 `hat3_mist_claret.yaml` | 7: av, sesinw, secosw, sed errscale, three jitter variances | 2: planet radius (a 1.2% move, 6.6 sigma), teffsed (3.1 sigma) |

On hat3 the two rules disagree in exactly the way that motivated the change:
the 22% move of the HIRES jitter variance is 0.4 sigma, and the 1.2% move of
the planet radius is 6.6 sigma.

## Tests

`tests/test_logging_policy.py` pins the shape: setup of a small shipped
example emits at most a handful of INFO records and every WARNING it emits
names a remedy and is wrapped at 80 columns; the recap collects, counts and
is not re-collected; an unknown key, top-level or in a component block,
raises; every shipped YAML uses only known keys.
`tests/test_user_start_check.py` pins the sigma rule for polish moves. When a test asserts on a log message,
assert at the level this page assigns -- and never weaken an assertion that
guards a real warning to make a demotion pass.
