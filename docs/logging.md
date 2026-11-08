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

1. one headline sentence: what happened, and its consequence;
2. then the remedy: what to write, and where (`set
   mulensinstrument.<name>.zeropoint: {mu, sigma}` in the params file);
3. at most about three lines in all.

It names the parameter, component instance or file it is about. A WARNING
that reports a caught exception gives the exception in the headline
(`... failed (ValueError: ...); <what is missing>`) and logs the traceback at
DEBUG, so the file has it and the console does not. It contains
no numpy reprs (`np.float64(2300.0)` -- format numbers with `:g` or
`float()`) and no dict or list dumps (`{'teff': (...)}` -- say it in words).
The headline is what the recap repeats (below), so it must stand on its own.

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
message (CLAUDE.md, "No fallbacks for internal invariants"). Every component's
config entries are checked against its `config_schema()` at construction
(`Component.validate_config_keys`, called by `System.__init__`), and an unknown key RAISES, naming the
key and the component, and suggesting the right spelling where there is an
obvious one (a near-miss of a known key, or a parameter that belongs in the
params file -- `zero_point:` in a `mulensinstrument` block is
`mulensinstrument.<name>.zeropoint: {mu, sigma}` in the params file). Spellings
are never case-folded.

## Tests

`tests/test_logging_policy.py` pins the shape: setup of a small shipped
example emits at most a handful of INFO records and every WARNING it emits
names a remedy; the recap collects, counts and is not re-collected; an
unknown component config key raises. When a test asserts on a log message,
assert at the level this page assigns -- and never weaken an assertion that
guards a real warning to make a demotion pass.
