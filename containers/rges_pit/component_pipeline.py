"""EXOZIPPy as an RGES-PIT component pipeline (WG3, event modeling).

The contract is the NENA containerization framework's
(golmschenk/nena_minimal_python_component_template): one pure function,
``pipeline_function(input_dictionary) -> output_dictionary``, that
container_loop.py calls once per input event.  The dictionaries are the JSON
event messages; bulky products go to files under the output root and the
returned dictionary only points at them (the framework caps a message at
100 KB).

What one task does:

1. validate the input event (README.md, "Input event") and raise naming the
   offending key -- a malformed upstream message is a bug to report, not
   something to guess around;
2. stage each light curve where EXOZIPPy can read it (a Parquet light curve
   is rewritten as whitespace columns; a text file is used in place);
3. build the system config and the params overrides for the requested
   architecture, and dump both to YAML so the fit can be rerun by hand with
   ``exozippy <event_id>.yaml``;
4. run the fit (``exozippy.run.run_fit``);
5. return the output event: where the products are, plus the
   ``<prefix>_results.csv`` table inline.

The fit itself follows examples/DC2018/run_event.py: no microlensing start
values are passed, so EXOZIPPy's built-in PSPL peak finder seeds t_0, u_0
and t_E from the light curves, and a binary lens keeps the defaults.yaml
starts for s, q and alpha.
"""

import contextlib
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

#: Where input light curves (and any ephemeris file) are read from.  Every
#: path in an input event is relative to this root.  In the cloud it is the
#: mounted input S3 bucket; locally a directory (compose.yaml mounts one).
INPUT_ROOT = Path(os.environ.get("EXOZIPPY_INPUT_ROOT", "input_data"))
#: Where each event's products are written, one directory per event_id.
OUTPUT_ROOT = Path(os.environ.get("EXOZIPPY_OUTPUT_ROOT", "output_data"))
#: Sampler and model settings that do not come from the event message.
FIT_SETTINGS = Path(
    os.environ.get("EXOZIPPY_FIT_SETTINGS", HERE / "fit_settings.yaml")
)

#: The framework's own message size cap (nena_component_tools docs,
#: "Output JSON EventBridge message").
MAX_MESSAGE_BYTES = 100_000

#: The input/output specification version (README.md).  Bumped when a key
#: is renamed or removed, or changes meaning; adding an optional key is not
#: a version change.
SCHEMA_VERSION = "0.1"

ARCHITECTURES = ("1S1L", "1S2L")
DATA_FORMATS = ("magnitude", "flux")
LC_REQUIRED = (
    "path",
    "band",
    "data_format",
    "observer_location",
    "time_frame",
    "time_scale",
)
LC_OPTIONAL = ("time_offset", "columns")
PARQUET_ROLES = ("time", "value", "error")

EVENT_REQUIRED = ("event_id", "ra", "dec", "light_curves")
EVENT_OPTIONAL = ("schema_version", "architecture")
#: Inputs the specification ASKS for but this version does not read yet
#: (README.md, "Planned inputs").  They are accepted, so an upstream can
#: start sending them now, and every one that arrives is listed in the
#: output event's ``inputs_not_used`` -- nothing supplied is dropped
#: silently.
EVENT_PLANNED = (
    "detection",
    "lens_flux",
    "astrometry",
    "precursor_imaging",
    "field",
    "extinction",
)
LC_PLANNED = (
    "observatory",
    "zeropoint",
    "observer_positions",
    "ephemeris_revision",
)


class InputEventError(ValueError):
    """The input event does not follow README.md's input specification."""


def _require(mapping, key, where):
    if key not in mapping:
        raise InputEventError(f"{where}: required key {key!r} is missing.")
    return mapping[key]


def validate_input(input_dictionary):
    """Check the input event against README.md and return it unchanged.

    Raises InputEventError naming the key at fault.  Unknown keys raise too:
    a misspelled optional key would otherwise be silently ignored.  Planned
    keys (EVENT_PLANNED, LC_PLANNED) are accepted and not checked further;
    ``inputs_not_used`` reports them.
    """
    if not isinstance(input_dictionary, dict):
        raise InputEventError(
            f"input event must be a JSON object; got "
            f"{type(input_dictionary).__name__}."
        )
    known = set(EVENT_REQUIRED + EVENT_OPTIONAL + EVENT_PLANNED)
    extra = sorted(set(input_dictionary) - known)
    if extra:
        raise InputEventError(
            f"input event: unknown key(s) {extra}; expected {sorted(known)}."
        )
    event_id = _require(input_dictionary, "event_id", "input event")
    if not isinstance(event_id, str) or not event_id or "/" in event_id:
        raise InputEventError(
            f"input event: event_id must be a non-empty string with no '/' "
            f"(it names the output directory); got {event_id!r}."
        )
    where = f"event {event_id}"
    for key in ("ra", "dec"):
        val = _require(input_dictionary, key, where)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InputEventError(
                f"{where}: {key} must be a number in degrees; got {val!r}."
            )
    arch = input_dictionary.get("architecture")
    if "architecture" in input_dictionary and arch not in ARCHITECTURES:
        raise InputEventError(
            f"{where}: architecture must be one of {list(ARCHITECTURES)}; "
            f"got {arch!r}."
        )
    lcs = _require(input_dictionary, "light_curves", where)
    if not isinstance(lcs, list) or not lcs:
        raise InputEventError(
            f"{where}: light_curves must be a non-empty list."
        )
    for k, lc in enumerate(lcs):
        lc_where = f"{where}, light_curves[{k}]"
        if not isinstance(lc, dict):
            raise InputEventError(f"{lc_where}: must be a JSON object.")
        lc_known = LC_REQUIRED + LC_OPTIONAL + LC_PLANNED
        extra = sorted(set(lc) - set(lc_known))
        if extra:
            raise InputEventError(
                f"{lc_where}: unknown key(s) {extra}; expected "
                f"{list(lc_known)}."
            )
        for key in LC_REQUIRED:
            _require(lc, key, lc_where)
        if lc["data_format"] not in DATA_FORMATS:
            raise InputEventError(
                f"{lc_where}: data_format must be one of "
                f"{list(DATA_FORMATS)}; got {lc['data_format']!r}."
            )
        is_parquet = str(lc["path"]).endswith(".parquet")
        if is_parquet:
            cols = _require(lc, "columns", lc_where)
            missing = [r for r in PARQUET_ROLES if r not in cols]
            if (
                not isinstance(cols, dict)
                or missing
                or set(cols) - set(PARQUET_ROLES)
            ):
                raise InputEventError(
                    f"{lc_where}: a Parquet light curve needs columns: "
                    f"{{time: <name>, value: <name>, error: <name>}} naming "
                    f"its columns; got {cols!r}."
                )
        elif "columns" in lc:
            raise InputEventError(
                f"{lc_where}: 'columns' applies only to a .parquet file; a "
                f"text light curve is read as time, value, error columns."
            )
    names = [_instrument_name(lc) for lc in lcs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise InputEventError(
            f"{where}: two light curves share the instrument name(s) "
            f"{dupes} (built from band + file stem); rename one file."
        )
    return input_dictionary


def inputs_not_used(event):
    """The planned keys present in ``event``, as dotted paths."""
    out = [k for k in EVENT_PLANNED if k in event]
    for k, lc in enumerate(event["light_curves"]):
        out += [f"light_curves[{k}].{key}" for key in LC_PLANNED if key in lc]
    return out


def architecture(event, settings):
    """The architecture to fit: the event's, else fit_settings.yaml's
    ``default_architecture`` (the place an architecture search will take
    over this choice)."""
    return event.get("architecture", settings["default_architecture"])


def _instrument_name(lc):
    return f"{lc['band']}_{Path(lc['path']).stem}"


def _observer_location(lc):
    """An exozippy built-in name ('earth', 'roman_simulated_2018dc', ...)
    or a file under the input root.  A path that exists under the input
    root wins, so an uploaded Roman ephemeris is used as given."""
    loc = lc["observer_location"]
    candidate = INPUT_ROOT / loc
    return str(candidate.resolve()) if candidate.exists() else loc


def stage_light_curve(lc, staging_dir):
    """Path of a file EXOZIPPy can read for light curve ``lc``.

    A text file is read in place.  A Parquet file is rewritten as three
    whitespace columns (time, value, error) in ``staging_dir``.
    """
    src = INPUT_ROOT / lc["path"]
    if not src.exists():
        raise FileNotFoundError(
            f"light curve {lc['path']!r} not found under the input root "
            f"{str(INPUT_ROOT)!r}."
        )
    if not str(src).endswith(".parquet"):
        return str(src.resolve())
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(src)
    cols = [lc["columns"][r] for r in PARQUET_ROLES]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise InputEventError(
            f"light curve {lc['path']!r} has no column(s) {absent}; it has "
            f"{list(df.columns)}."
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    out = staging_dir / f"{_instrument_name(lc)}.dat"
    np.savetxt(out, df[cols].to_numpy(dtype=float), fmt="%.10f")
    return str(out.resolve())


def build_config(event, files, prefix, settings):
    """The EXOZIPPy system config for ``event``.

    ``files`` maps each light curve's instrument name to its staged path.
    """
    arch = architecture(event, settings)
    binary = arch == "1S2L"
    bands = list(dict.fromkeys(lc["band"] for lc in event["light_curves"]))
    config = {
        "run": {"name": event["event_id"]},
        "prefix": str(prefix),
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [
            {"finite_source": bool(settings["finite_source"][arch])}
        ],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        # The line of sight is the source's (index 1 in star:).
        "galacticmodel": [{"name": event["event_id"], "anchor_idx": 1}],
        "band": [
            # No Roman WFI filter has a Claret table, so the limb-darkening
            # tie is stated explicitly (it raises otherwise).  false is what
            # this pipeline has always done; ld_source: phoenix would tie
            # each band to tables built through its own WFI profile.
            {
                "name": b,
                "filter": f"Roman/WFI.{b}",
                "ld_law": "linear",
                "star_constrains_ld": False,
            }
            for b in bands
        ],
        "mulensinstrument": [],
        "sampler": dict(settings["sampler"]),
        "modes": dict(settings["modes"]),
    }
    if binary:
        config["planet"] = [{"name": "Companion"}]
        config["lens"].append({"body": "planet.Companion"})
    for lc in event["light_curves"]:
        name = _instrument_name(lc)
        entry = {
            "name": name,
            "file": files[name],
            "data_format": lc["data_format"],
            "observer_location": _observer_location(lc),
            "band": lc["band"],
            "time_frame": lc["time_frame"],
            "time_scale": lc["time_scale"],
            # Hogg inlier/outlier mixture in place of a hard bad-data mask
            # (examples/DC2018/README.md): no point is dropped, and each
            # point's outlier probability is reported afterwards.
            "likelihood": "hogg",
        }
        if "time_offset" in lc:
            entry["time_offset"] = lc["time_offset"]
        config["mulensinstrument"].append(entry)
    return config


def build_user_params(event, arch):
    """Params overrides: only the sky position, pinned, for both stars.

    No microlensing start values, so the built-in peak finder seeds them
    from the data (a user entry would outrank it).  The source's position
    is the one the galactic model's line of sight is read from.
    """
    params = {}
    for star in ("Lens", "Source"):
        params[f"star.{star}.ra"] = {"initval": float(event["ra"]), "sigma": 0}
        params[f"star.{star}.dec"] = {
            "initval": float(event["dec"]),
            "sigma": 0,
        }
    if arch == "1S2L":
        params["planet.Companion.radius"] = {"sigma": 0}
    return params


def read_results_csv(path):
    """``<prefix>_results.csv`` as a list of row dicts.

    The first line is a ``# col, col, ...`` header whose columns depend on
    whether the run reported modes; later ``#`` lines are notes and are
    skipped.  Numeric fields become floats and empty fields None.
    """
    with open(path) as f:
        header = f.readline()
        if not header.startswith("#"):
            raise ValueError(f"{path}: expected a '# col, ...' header line.")
        names = [c.strip() for c in header.lstrip("#").split(",")]
        rows = []
        for rec in csv.reader(line for line in f if not line.startswith("#")):
            if not rec:
                continue
            if len(rec) != len(names):
                raise ValueError(
                    f"{path}: row {rec!r} has {len(rec)} fields; the header "
                    f"names {len(names)}."
                )
            row = {}
            for name, val in zip(names, rec, strict=True):
                if val == "":
                    row[name] = None
                    continue
                try:
                    row[name] = float(val)
                except ValueError:
                    row[name] = val
            rows.append(row)
    return rows


def load_settings(path=None):
    with open(path or FIT_SETTINGS) as f:
        settings = yaml.safe_load(f)
    for key in ("sampler", "modes", "finite_source", "default_architecture"):
        if key not in settings:
            raise KeyError(f"{path or FIT_SETTINGS}: missing {key!r}.")
    return settings


def pipeline_function(input_dictionary):
    """Fit one microlensing event; return the output event dictionary."""
    from exozippy.run import run_fit

    event = validate_input(input_dictionary)
    settings = load_settings()
    event_id = event["event_id"]
    event_dir = (OUTPUT_ROOT / event_id).resolve()
    event_dir.mkdir(parents=True, exist_ok=True)

    files = {
        _instrument_name(lc): stage_light_curve(lc, event_dir / "data")
        for lc in event["light_curves"]
    }
    prefix = event_dir / "fitresults" / event_id
    config = build_config(event, files, prefix, settings)
    arch = architecture(event, settings)
    user_params = build_user_params(event, arch)

    params_yaml = event_dir / f"{event_id}.params.yaml"
    config_yaml = event_dir / f"{event_id}.yaml"
    with open(params_yaml, "w") as f:
        yaml.safe_dump(user_params, f, sort_keys=False)
    with open(config_yaml, "w") as f:
        yaml.safe_dump(
            {**config, "parameter_file": str(params_yaml)}, f, sort_keys=False
        )
    with open(event_dir / "input_event.json", "w") as f:
        json.dump(event, f, indent=2)

    # From the event directory: run_fit writes its restart params file
    # (mkparam) to the working directory.
    with contextlib.chdir(event_dir):
        run_fit(config, user_params=user_params)

    results_csv = Path(f"{prefix}_results.csv")
    if not results_csv.exists():
        raise FileNotFoundError(
            f"event {event_id}: the fit finished without writing "
            f"{results_csv}."
        )

    def rel(p):
        return str(Path(p).resolve().relative_to(OUTPUT_ROOT.resolve()))

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "event_id": event_id,
        "architecture": arch,
        "exozippy_version": version("exozippy"),
        "observer_locations": {
            _instrument_name(lc): lc["observer_location"]
            for lc in event["light_curves"]
        },
        "inputs_not_used": inputs_not_used(event),
        "output_directory": rel(event_dir),
        "files": {
            "results_csv": rel(results_csv),
            "trace": rel(f"{prefix}_trace.nc"),
            "config": rel(config_yaml),
            "params": rel(params_yaml),
        },
        "parameters": read_results_csv(results_csv),
    }
    size = len(json.dumps(output).encode())
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"event {event_id}: output message is {size} bytes, over the "
            f"framework's {MAX_MESSAGE_BYTES}-byte cap; drop 'parameters' "
            f"and point at files.results_csv instead."
        )
    return output
