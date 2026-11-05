# EXOZIPPy as an RGES-PIT component pipeline

This directory packages EXOZIPPy as a component of the Roman Galactic
Exoplanet Survey Project Infrastructure Team (RGES-PIT) global pipeline. It
is the event-modeling stage (working group 3): it receives an event message
about a microlensing event, fits the light curves, and emits an event message
pointing at the posterior.

It follows the NENA containerization framework's Python template
([golmschenk/nena_minimal_python_component_template](https://github.com/golmschenk/nena_minimal_python_component_template),
[nena_component_tools](https://github.com/golmschenk/nena_component_tools)).
The science is all in `pipeline_function` (`component_pipeline.py`), which
takes the input event as a dictionary and returns the output event. The
framework's task loop (`container_loop.py`) handles everything else: JSON
files on disk locally, SQS and EventBridge in the cloud.

The sections below are the component's input/output specification, in the
form WG14 asks every component pipeline to provide.

## Summary

| | |
|---|---|
| Component pipeline name | `exozippy_event_modeling` (placeholder) |
| Working group | WG3, event modeling |
| Triggered by | A new candidate event or anomaly (WG5) |
| Reads now | The event's light curves (Roman and ground-based), with each observer's position from an ephemeris file or a site |
| Reads when available | WG5 detection parameters, WG4 lens/source flux and position posteriors, WG13/MSOS astrometry, WG8 precursor HST/Euclid imaging, the field's red clump and extinction, per-band zeropoints, per-exposure spacecraft positions |
| Writes | One directory per event: the posterior (NetCDF), a parameter table (CSV), the generated EXOZIPPy config, plots |
| Emits | One event per fit, with the parameter table inline and paths to the files |
| Consumers | WG12 (efficiency and occurrence rates), WG13 (joint photometric-astrometric fits), event catalogs |

What this component adds to the pipeline is physical parameters: a lens mass
and distance, a planet mass and projected separation in physical units, and
the relative proper motion. These come as posteriors from one joint model of
the light curves, the galactic structure and kinematics along the line of
sight, and, as they become available, the lens flux, source color, extinction
and astrometry. The planned inputs below are what turn the first
version's prior-dominated physical posterior into a measurement.

## Versioning

Every input and output event may carry `schema_version` (currently `"0.1"`).
Adding an optional key does not change the version. Renaming or removing a
key, or changing its meaning, does. Readers should ignore keys they do not
know. Producers should leave out a key they cannot fill rather than send
`null`.

## Input event

One JSON object per event to fit. Keys are **required**, **optional** (read
when present), or **planned** (accepted now, read in a later version). An
upstream pipeline can start sending a planned key as soon as it has it. Every
planned key that arrives is listed in the output event's `inputs_not_used`, so
nothing supplied is dropped silently. Unknown keys are rejected, so a
misspelled key fails loudly.

Minimal example (this directory's `input_events/sim_pspl_0001.json`):

```json
{
  "event_id": "sim_pspl_0001",
  "ra": 268.0,
  "dec": -29.0,
  "architecture": "1S1L",
  "light_curves": [
    {
      "path": "sim_pspl_F146.dat",
      "band": "F146",
      "data_format": "magnitude",
      "observer_location": "roman_simulated_2018dc",
      "time_frame": "bjd",
      "time_scale": "tdb"
    }
  ]
}
```

### Top-level keys

| Key | Status | Type | Meaning |
|---|---|---|---|
| `event_id` | required | string | Unique event name. Also names the output directory, so it may not contain `/`. |
| `ra`, `dec` | required | number, degrees | Event position (ICRS). |
| `light_curves` | required | list, at least one | One entry per light curve: per filter, and per observatory (Roman, or contemporaneous ground data such as KMTNet). |
| `schema_version` | optional | string | See "Versioning". |
| `architecture` | optional | `"1S1L"` or `"1S2L"` | The model: a single lens, or a binary or planetary lens. When absent, the component chooses: for now a single lens. An architecture search that fits both and compares their evidence is planned for 2027-01. |
| `detection` | planned | object | WG5's detection: `t_0`, `u_0`, `t_E`, an anomaly flag, and the anomaly's time range. Used as start values and to choose the architecture. |
| `lens_flux` | planned | object | WG4's lens and source position and brightness posteriors: `{"path": ..., "format": ...}`. Constrains the lens mass and distance through its brightness. |
| `astrometry` | planned | list | WG13 and MSOS astrometry: per-epoch centroid shifts of the varying light, or resolved lens and source positions, each `{"path": ..., "kind": ...}`. Gives the relative proper motion's size and direction, and astrometric microlensing. |
| `precursor_imaging` | planned | list | WG8 HST and Euclid catalogs or posteriors of the stars at the event position. Gives the source's identity and colors, and resolves the lens and source later. |
| `field` | planned | object | The field's color-magnitude diagram, or at least the red clump's centroid in each Roman band. Calibrates the source color and magnitude, and so the angular source size. |
| `extinction` | planned | object | Extinction and reddening toward the event in Roman bands, with uncertainties, for when no red-clump measurement is available. |

### Light-curve keys

| Key | Status | Type | Meaning |
|---|---|---|---|
| `path` | required | string | File path, relative to the input data root. A `.parquet` file needs `columns`. Any other file is read as whitespace-separated columns: time, value, error. |
| `band` | required | string | Filter, e.g. `F146`, `F087`, `F213` for Roman WFI. |
| `data_format` | required | `"magnitude"` or `"flux"` | What the value column holds. |
| `observer_location` | required | string | Where the photometry was taken. Either a file under the input data root (a spacecraft ephemeris; format in `src/exozippy/ephemerides/README.md`), or a built-in name: `earth`, an astropy site name, a `"lon_deg,lat_deg[,height_m]"` string, or a shipped ephemeris such as `roman_simulated_2018dc`. |
| `time_frame` | required | string | `bjd`, `hjd` or `jd`: the reference point of the time stamps. |
| `time_scale` | required | string | An astropy time scale: `tdb`, `tt`, `tai`, `utc`, `ut1`, `tcb` or `tcg`. Both time keys are required because a wrong time system silently shifts the parallax and the event time. |
| `time_offset` | optional | number | Added to every time stamp, e.g. `2450000` for HJD' / BJD'. |
| `columns` | optional (Parquet only) | object | `{"time": "<name>", "value": "<name>", "error": "<name>"}`, naming the file's columns. Required for a Parquet file. |
| `observatory` | planned | string | Telescope and camera, for reporting and for per-observatory systematics. |
| `zeropoint` | planned | object | `{"magnitude_system": "AB" or "Vega", "zp": ..., "zp_err": ...}`: the calibration that turns the fitted source flux into a magnitude. With it, the source flux in two or more bands gives the source color, and from that its angular size. |
| `observer_positions` | planned | object | Column names for per-epoch spacecraft positions in the light-curve file (`{"x": ..., "y": ..., "z": ...}`, barycentric ICRS, AU). Roman's data products carry each exposure's position, which would make `observer_location` unnecessary for Roman data. |
| `ephemeris_revision` | planned | string | Which revision of the spacecraft ephemeris the photometry used, so a fit can be redone when the ephemeris is revised. |

The input event is small; light curves and other products are files. In the
cloud, the input data root is the mounted S3 bucket(s). Locally, it is
`./input_data`.

## Output event

Emitted once per completed fit. This is the example event's real output from
a local run with the quick settings (`parameters` cut to 3 of its 75 rows):

```json
{
  "schema_version": "0.1",
  "status": "complete",
  "event_id": "sim_pspl_0001",
  "architecture": "1S1L",
  "exozippy_version": "0.1.0",
  "observer_locations": {"F146_sim_pspl_F146": "roman_simulated_2018dc"},
  "inputs_not_used": [],
  "output_directory": "sim_pspl_0001",
  "files": {
    "results_csv": "sim_pspl_0001/fitresults/sim_pspl_0001_results.csv",
    "trace": "sim_pspl_0001/fitresults/sim_pspl_0001_trace.nc",
    "config": "sim_pspl_0001/sim_pspl_0001.yaml",
    "params": "sim_pspl_0001/sim_pspl_0001.params.yaml"
  },
  "parameters": [
    {"parname": "star.Lens.mass", "mode": "all", "weight": 1.0, "weight_err": null, "value": 0.38, "up_err": 0.11, "low_err": 0.076},
    {"parname": "star.Lens.distance", "mode": "all", "weight": 1.0, "weight_err": null, "value": 6450.0, "up_err": 1100.0, "low_err": 540.0},
    {"parname": "mulensevent.t_E", "mode": "all", "weight": 1.0, "weight_err": null, "value": 24.94, "up_err": 0.21, "low_err": 0.26}
  ]
}
```

| Key | Status | Meaning |
|---|---|---|
| `schema_version` | now | See "Versioning". |
| `status` | now | `"complete"`. `"accepted"` is planned, for the long-fit pattern below. |
| `event_id`, `architecture` | now | As fit. |
| `exozippy_version` | now | The EXOZIPPy release or commit that ran the fit. |
| `observer_locations` | now | The ephemeris or site used for each light curve, so fits can be redone when an ephemeris is revised. |
| `inputs_not_used` | now | Planned input keys that were supplied but not read by this version. |
| `output_directory`, `files` | now | Paths relative to the output data root (the WG3 S3 bucket in the cloud). |
| `parameters` | now | The `<prefix>_results.csv` table, one object per row: the median and the upper and lower credible-interval errors, in EXOZIPPy's parameter-table units (solar masses, pc, days, mas/yr). A field with no value (e.g. the error of a pinned parameter) is `null`. It already includes the physical parameters: lens mass and distance, source distance, proper motions, and for a binary lens the planet mass. `mode` is `"all"` for a unimodal posterior, or the mode's index when the posterior has several (degenerate solutions), with `weight` its posterior probability from bridge-sampling evidence. Rows with `mode` `"rejected-seed<k>"` describe a candidate solution the posterior rejected, with a Laplace approximation and its relative weight. Those rows are what WG12 needs to weight degenerate solutions. |
| `summary` | planned | A short, fixed set of headline values with no EXOZIPPy parameter names to learn: lens mass, lens distance, planet mass, projected separation in AU, relative proper motion, `t_E`, `pi_E`, `theta_E`. Each as median and 68% interval, per mode. |
| `modes` | planned | One entry per degenerate solution: its probability, its evidence and that evidence's error, and a short label (e.g. close/wide, `u_0` +/-). |
| `architecture_comparison` | planned | Log-evidence of each architecture fit, and the preferred one. |
| `quality` | planned | Convergence diagnostics (maximum R-hat, minimum effective sample size, stuck-chain flag), and whether the physical posterior is prior-dominated. |
| `files.light_curve_model` | planned | Parquet: the best-fit and posterior-band model light curve of each band on a time grid. This is the "Microlensing event models and light curves" product WG12 consumes. |
| `files.posterior_samples` | planned | Parquet: thinned posterior draws of the headline parameters, for consumers who do not want to read NetCDF. |

- The message is capped at the framework's 100 KB. The task fails, rather
  than truncating, if a table ever exceeds that.
- A fit that fails emits nothing. The task logs the failure (error and
  traceback) through `task.log_failure`, and the queue retries it.
- **Planned long-fit pattern:** emit `status: "accepted"` when a task
  starts, then `"complete"` when it finishes, possibly after the 12-hour
  queue limit. This depends on the long-job path asked about below.

## Output files

Everything for one event goes under `<output root>/<event_id>/`:

| File | Format | Contents |
|---|---|---|
| `fitresults/<event_id>_trace.nc` | NetCDF4 (ArviZ InferenceData) | The full posterior: every draw of every parameter, sampler statistics, and the per-mode labels. This is the product for anything downstream that needs more than medians. |
| `fitresults/<event_id>_results.csv` | CSV | The parameter table also sent in the output event. |
| `fitresults/<event_id>_*.pdf` / `.png` | PDF, PNG | Light-curve, residual, corner and trace plots. |
| `fitresults/<event_id>_paper.tex` | LaTeX | Parameter table and methods prose. |
| `<event_id>.yaml`, `<event_id>.params.yaml` | YAML | The exact EXOZIPPy inputs used. `exozippy <event_id>.yaml` reruns the fit by hand. |
| `input_event.json` | JSON | The input event as received. |
| `data/` | text | Present only when an input light curve was Parquet: the converted copy that was fit. |

## The fit

Each event gets the same model as the Roman 2018 Data Challenge fits in
`examples/DC2018`:

- The lens and source are physical stars, with a galactic-model prior on
  their masses, distances and proper motions along the event's line of
  sight.
- The microlensing start values come from EXOZIPPy's built-in peak finder.
  No upstream seed is needed.
- Every light curve uses the inlier/outlier mixture likelihood instead of a
  hard outlier mask, with a fitted error scaling.
- Sampling uses parallel-tempered differential evolution.
- A binary lens is fit with finite-source effects.
- The physical posterior in this first version comes from the light curve
  (t_E, parallax, finite source) plus the galactic model. The planned inputs
  (lens flux, source color and extinction, astrometry) are what sharpen it.

These settings live in `fit_settings.yaml`, which is not part of the event
message. `fit_settings_quick.yaml` is the same model with a tiny sample, for
smoke tests only.

## Run time

The framework limits a task to 6 hours of wall-clock time: the queue hides a
task for twice its declared run time, and SQS caps that at 12 hours.

- A single-lens fit, including the physical posterior, fits inside that
  window.
- A full binary-lens posterior does not. On the 2026 Roman Microlensing Data
  Challenge, some binary fits on 32 cores took longer than a 24-hour cluster
  limit.

Measured so far: the example event (1728 points, one band) with the quick
settings took about 30 minutes wall-clock on a 36-core workstation, of which
5.6 minutes were wrap-up (plots, tables, the paper draft). Production
settings take 50,000 draws instead of 300. Production timings on the target AWS
instance type are TBD.

## Running it locally

With Docker (as in the framework's guide):

```shell
cd containers/rges_pit
docker compose up --build --force-recreate
```

This fits `input_events/sim_pspl_0001.json`, a simulated single-lens Roman
F146 event made by `make_example_event.py`. The output event goes to
`output_events/` and the products to `output_data/`.

Without Docker, from an EXOZIPPy environment with `nena_component_tools`
installed:

```shell
cd containers/rges_pit
EXOZIPPY_FIT_SETTINGS=fit_settings_quick.yaml python container_loop.py
```

Environment variables: `EXOZIPPY_INPUT_ROOT` (default `input_data`),
`EXOZIPPY_OUTPUT_ROOT` (default `output_data`) and `EXOZIPPY_FIT_SETTINGS`
(default `fit_settings.yaml`).

## Questions for WG14

1. **Long fits.** A binary-lens posterior takes one to several days, past the
   12-hour SQS limit. Is there, or could there be, a supported path for
   long tasks? Slide 5 already wraps components in a Step Function, which
   can run an ECS task far longer. Other options are AWS Batch, or
   submitting to the SMDC Slurm cluster. Failing that, may a component
   delete its message at the start (by emitting an "accepted" event with
   `mark_task_complete=True`), run for days, and emit its result with
   `mark_task_complete=False`? If so, does the service's autoscaling stop
   containers whose queue looks empty, and who retries a task whose container
   dies?
2. **Cores and memory per task.** What instance size does a component get?
   Fargate's cap is 16 vCPU. Our sampler uses tens of cores, and a binary fit
   peaks near 8 GB.
3. **Network access.** Can a container reach the internet? We would use it
   for JPL Horizons (Roman's ephemeris) and catalog services (VizieR, MAST,
   the Gaia archive). If not, can those products be mirrored into a bucket?
4. **Roman's ephemeris.** Who provides it in the pipeline, how often, and is
   each revision kept? JPL Horizons has Roman as -211, a long-range
   prediction to 2028-10-25 that is revised as tracking comes in. Do the
   photometry products carry each exposure's spacecraft position (predicted
   or definitive)? If so, per-epoch positions in the light curves would be
   simplest.
5. **Upstream schemas.** When will the WG5 detection event and the
   photometry product formats be defined? This component adapts to their
   names. It needs coordinates, light-curve locations and their time system;
   the planned keys above are what it would like.
6. **A `nena_component_tools` bug (0.3.2).** Locally, `LocalTask.log_failure`
   calls `json.dumps` on a dictionary holding `input_event_path`, a `Path`.
   It raises `TypeError`, so every local failure crashes the loop instead of
   being logged. Converting the path with `str()` fixes it.
7. **Names.** The component name, image registry, S3 bucket names and
   upstream source names in `nena_configuration.toml` are placeholders.
