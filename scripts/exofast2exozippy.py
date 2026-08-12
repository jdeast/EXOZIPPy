#!/usr/bin/env python3
"""Convert an EXOFASTv2 driver .pro file into EXOZIPPy YAML inputs.

Reads the IDL procedure that calls ``exofastv2, ...``, translates the call's
keywords, the ``priorfile=`` and the ``sedfile=`` into the EXOZIPPy config
trio, and copies the referenced data files, so that

    cd examples/gj1214
    poetry run python ../../scripts/exofast2exozippy.py ~/modeling/gj1214/fitclass.pro
    poetry run exozippy gj1214.yaml

runs a comparable fit. Generated files (base name defaults to the .pro
file's parent directory name, override with --name):

    <name>.yaml            system config
    <name>.params.yaml     parameter overrides (from priorfile=)
    <name>.sed.yaml        SED photometry (from sedfile=)
    <data files>           copied next to the YAMLs

Sampling controls (maxsteps/nthin/ntemps/nthreads/...) are deliberately NOT
carried over: the emitted sampler block follows EXOZIPPy's HMC best
practices instead of EXOFASTv2's DE-MCMC settings. Every EXOFASTv2 feature
that has no EXOZIPPy equivalent yet produces a WARNING (collected at the
end and embedded as comments in the generated config).
"""

import argparse
import glob
import math
import re
import shutil
import sys
from pathlib import Path

MSUN_PER_MJUP = 1047.348644  # IAU nominal GMsun/GMjup
RAD_TO_DEG = 180.0 / math.pi

# ----------------------------------------------------------------------
# Warning / info collection
# ----------------------------------------------------------------------

WARNINGS = []
INFOS = []


def warn(msg):
    WARNINGS.append(msg)


def info(msg):
    INFOS.append(msg)


# ----------------------------------------------------------------------
# IDL .pro parsing
# ----------------------------------------------------------------------


def _strip_idl_comment(line):
    """Remove an IDL ``;`` comment, ignoring semicolons inside quotes."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == ";":
            break
        else:
            out.append(ch)
    return "".join(out)


def _logical_lines(text):
    """Yield comment-stripped IDL lines with ``$`` continuations joined."""
    pending = ""
    for raw in text.splitlines():
        line = _strip_idl_comment(raw).rstrip()
        if not line.strip():
            continue
        if line.rstrip().endswith("$"):
            pending += line.rstrip()[:-1]
            continue
        yield (pending + line).strip()
        pending = ""
    if pending.strip():
        yield pending.strip()


def _split_top_level(s, sep=","):
    """Split on ``sep`` outside quotes and brackets/parens."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


_IDL_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([deDE][+-]?\d+)?$")


def _parse_idl_value(tok, variables):
    """Parse an IDL expression token into a Python value."""
    tok = tok.strip()
    if not tok:
        return None
    if tok[0] in ("'", '"') and tok[-1] == tok[0] and len(tok) >= 2:
        return tok[1:-1]
    if tok.startswith("[") and tok.endswith("]"):
        return [
            _parse_idl_value(t, variables) for t in _split_top_level(tok[1:-1])
        ]
    if _IDL_NUM.match(tok):
        num = tok.lower().replace("d", "e")
        return float(num) if ("." in num or "e" in num) else int(num)
    low = tok.lower()
    if low in ("!values.d_infinity", "!values.f_infinity"):
        return math.inf
    if low in variables:
        return variables[low]
    warn(f"could not evaluate IDL expression '{tok}'; keeping it as a string")
    return tok


def parse_pro_file(path):
    """Return the keyword dict of the ``exofastv2, ...`` call in ``path``."""
    text = Path(path).read_text()
    variables = {}
    call_args = None
    for line in _logical_lines(text):
        low = line.lower()
        if low.startswith("exofastv2"):
            rest = line[len("exofastv2") :].lstrip()
            if rest.startswith(","):
                rest = rest[1:]
            call_args = rest
            continue
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if m and call_args is None:
            variables[m.group(1).lower()] = _parse_idl_value(
                m.group(2), variables
            )
    if call_args is None:
        sys.exit(f"ERROR: no 'exofastv2, ...' call found in {path}")

    keywords = {}
    for tok in _split_top_level(call_args):
        if tok.startswith("/"):
            keywords[tok[1:].lower()] = True
        elif "=" in tok:
            key, val = tok.split("=", 1)
            keywords[key.strip().lower()] = _parse_idl_value(val, variables)
        else:
            warn(f"ignoring positional argument '{tok}' in exofastv2 call")
    return keywords


# ----------------------------------------------------------------------
# EXOFASTv2 prior file parsing
# ----------------------------------------------------------------------


def _prior_float(tok):
    t = tok.lower().replace("d", "e")
    if t in ("inf", "+inf", "infinity"):
        return math.inf
    if t in ("-inf", "-infinity"):
        return -math.inf
    return float(t)


def parse_priorfile(path):
    """Parse an EXOFASTv2 prior file.

    Each line: ``name value [width [lower [upper [start]]]]``.
      width  > 0 -> Gaussian prior (center=value, sigma=width)
      width == 0 -> parameter fixed at value
      width  < 0 or absent -> starting value only
      start (5th number) -> starting value distinct from the prior center

    Returns a list of dicts with keys name, index, value, width, lower,
    upper, start.
    """
    priors = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        name = toks[0].lower()
        try:
            vals = [_prior_float(t) for t in toks[1:6]]
        except ValueError:
            warn(f"priorfile line not understood, skipped: '{raw.strip()}'")
            continue
        if not vals:
            warn(f"priorfile line has no value, skipped: '{raw.strip()}'")
            continue
        index = None
        m = re.match(r"^(.*)_(\d+)$", name)
        if m:
            name, index = m.group(1), int(m.group(2))
        vals += [None] * (5 - len(vals))
        value, width, lower, upper, start = vals
        if lower is not None and math.isinf(lower) and lower < 0:
            lower = None
        if upper is not None and math.isinf(upper) and upper > 0:
            upper = None
        priors.append(
            dict(
                name=name,
                index=index,
                value=value,
                width=width,
                lower=lower,
                upper=upper,
                start=start,
            )
        )
    return priors


# ----------------------------------------------------------------------
# EXOFASTv2 -> EXOZIPPy prior name mapping
# ----------------------------------------------------------------------

# axis: which instance list the (optional) _N index runs over.
#   star | planet | band | transit | telescope | sed
# scale: multiply value/width/lower/upper/start by this on the way over.
PRIOR_MAP = {
    # --- star ---
    "mstar": ("star", "star.{n}.mass", 1.0),
    "logmstar": ("star", "star.{n}.logmass", 1.0),
    "rstar": ("star", "star.{n}.radius", 1.0),
    "rstarsed": ("star", "star.{n}.radiussed", 1.0),
    "teff": ("star", "star.{n}.teff", 1.0),
    "teffsed": ("star", "star.{n}.teffsed", 1.0),
    "feh": ("star", "star.{n}.feh", 1.0),
    "logg": ("star", "star.{n}.logg", 1.0),
    "rhostar": ("star", "star.{n}.density", 1.0),  # both g/cm3
    "age": ("star", "star.{n}.age", 1.0),
    "av": ("star", "star.{n}.av", 1.0),
    "distance": ("star", "star.{n}.distance", 1.0),
    "parallax": ("star", "star.{n}.parallax", 1.0),
    "fbol": ("star", "star.{n}.fbol", 1.0),
    "errscale": ("sed", "sed.errscale", 1.0),
    # --- orbit (planet-indexed) ---
    "period": ("planet", "orbit.{n}.period", 1.0),
    "logp": ("planet", "orbit.{n}.logP", 1.0),
    "tc": ("planet", "orbit.{n}.tc", 1.0),
    "tp": ("planet", "orbit.{n}.tp", 1.0),
    "cosi": ("planet", "orbit.{n}.cosi", 1.0),
    "ideg": ("planet", "orbit.{n}.inc", 1.0),
    "b": ("planet", "orbit.{n}.b", 1.0),
    "e": ("planet", "orbit.{n}.ecc", 1.0),
    "omega": ("planet", "orbit.{n}.omega", RAD_TO_DEG),
    "omegadeg": ("planet", "orbit.{n}.omega", 1.0),
    "secosw": ("planet", "orbit.{n}.secosw", 1.0),
    "sesinw": ("planet", "orbit.{n}.sesinw", 1.0),
    "ecosw": ("planet", "orbit.{n}.ecosw", 1.0),
    "esinw": ("planet", "orbit.{n}.esinw", 1.0),
    "vcve": ("planet", "orbit.{n}.vcve", 1.0),
    "k": ("planet", "orbit.{n}.K", 1.0),
    # --- planet ---
    "p": ("planet", "planet.{n}.p", 1.0),
    "rp": ("planet", "planet.{n}.radius", 1.0),
    "mp": ("planet", "planet.{n}.mass", 1.0),
    "mpsun": ("planet", "planet.{n}.mass", MSUN_PER_MJUP),
    "ar": ("planet", "planet.{n}.ar", 1.0),
    # --- band ---
    "u1": ("band", "band.{n}.u1", 1.0),
    "u2": ("band", "band.{n}.u2", 1.0),
    "thermal": ("band", "band.{n}.thermal", 1.0),
    # --- per transit file ---
    "variance": ("transit", "transit.{n}.jitter_variance", 1.0),
    "f0": ("transit", "transit.{n}.baseline", 1.0),
    # --- per RV telescope ---
    "gamma": ("telescope", "rvinstrument.{n}.gamma", 1.0),
    "jittervar": ("telescope", "rvinstrument.{n}.jitter_variance", 1.0),
    "jitter": ("telescope", "rvinstrument.{n}.jitter", 1.0),
}

# Priors EXOFASTv2 accepts but EXOZIPPy has no home for (yet).
PRIOR_UNSUPPORTED = {
    "initfeh": "MIST evolutionary tracks are not implemented",
    "eep": "MIST evolutionary tracks are not implemented",
    "alpha": "MIST alpha enhancement is not implemented",
    "vsini": "map it to orbit.<planet>.vsini by hand if fitting RM",
    "reflect": "reflected-light phase curves are not implemented",
    "dilute": "explicit dilution priors are not implemented (EXOZIPPy "
    "dilutes transits automatically from the SED flux fractions)",
    "ttv": "per-transit TTVs are not implemented",
    "tiv": "per-transit inclination variations are not implemented",
    "tdeltav": "per-transit depth variations are not implemented",
    "slope": "RV slope (fitslope) is not implemented",
    "quad": "RV quadratic trend (fitquad) is not implemented",
    "chord": "the chord parameterization is not used; set orbit cosi/b",
    "sign": "the vcve sign parameter is not used",
    "logk": "put a prior on orbit.<planet>.K instead",
    "msini": "put a prior on planet.<planet>.mass instead",
    "absks": "put the constraint on mann ks/ks_err instead",
    "phottobary": "photocenter-to-barycenter priors are not implemented",
}


# ----------------------------------------------------------------------
# Filter-name translation (EXOFASTv2/Keivan -> SVO) via the alias table
# ----------------------------------------------------------------------


def _load_filter_aliases():
    try:
        from exozippy.components.sed.bc_grid import _load_alias_table

        return _load_alias_table()
    except Exception as exc:  # pragma: no cover - env without exozippy
        warn(
            f"could not load the EXOZIPPy filter alias table ({exc}); "
            "SED filter names are copied verbatim"
        )
        return None


def _to_svo(name, alias_df):
    """Translate an EXOFASTv2 band name to its canonical SVO name."""
    if alias_df is None:
        return name, True
    hit = alias_df[alias_df.eq(name).any(axis=1)]
    if len(hit) == 0:
        return name, False
    svo = str(hit.iloc[0]["SVO"]).strip()
    if svo.lower() == "unsupported":
        return name, False
    return svo, True


# ----------------------------------------------------------------------
# Data-file discovery
# ----------------------------------------------------------------------


def _resolve_glob(pattern, pro_dir):
    """Expand an EXOFASTv2 path glob relative to the .pro file's directory."""
    pattern = str(Path(pattern).expanduser())
    if not Path(pattern).is_absolute():
        pattern = str(pro_dir / pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        warn(f"no files match '{pattern}'")
    return [Path(f) for f in files]


def _transit_meta(path):
    """EXOFASTv2 transit filename convention: nYYYYMMDD.FILTER.TELESCOPE.*"""
    parts = path.name.split(".")
    if len(parts) >= 4 and parts[0].startswith("n"):
        date, band, telescope = parts[0][1:], parts[1], parts[2]
        name = f"{telescope}_UT{date}"
    else:
        warn(
            f"transit file '{path.name}' does not follow the "
            "nYYYYMMDD.FILTER.TELESCOPE.* convention; band unknown -- "
            "edit the generated transit/band blocks by hand"
        )
        name, band = path.stem.replace(".", "_"), "UNKNOWN"
    return name, band


def _rv_meta(path):
    """EXOFASTv2 RV filename convention: <target>.<INSTRUMENT>.rv"""
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else path.stem


# ----------------------------------------------------------------------
# YAML emission helpers (hand-rolled so we can write comments)
# ----------------------------------------------------------------------


def _fmt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return repr(v)
        return repr(v)
    if isinstance(v, (int,)):
        return str(v)
    return f'"{v}"'


def _emit_params_yaml(entries, header_lines):
    """entries: list of (path, {field: value}, comment_or_None)."""
    out = [f"# {h}" for h in header_lines]
    out.append("")
    for path, fields, comment in entries:
        if comment:
            for c in comment.splitlines():
                out.append(f"# {c}")
        out.append(f"{path}:")
        for key in ("initval", "mu", "sigma", "lower", "upper"):
            if key in fields:
                out.append(f"  {key}: {_fmt(fields[key])}")
        out.append("")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------
# The converter
# ----------------------------------------------------------------------

SAMPLER_BLOCK = """\
# Sampling controls are deliberately NOT translated from EXOFASTv2
# (maxsteps/nthin/ntemps are DE-MCMC knobs); this block follows EXOZIPPy's
# HMC best practices instead.
sampler:
  method: numpyro      # nuts, numpyro, blackjax, nutpie, ptde
  tune: 2000
  draws: 4000
  init: adapt_diag
  target_accept: 0.95
  recompute_trace: True   # if False and a prefix.nc file exists, load it instead of re-sampling
"""

# exofastv2 keywords that only steer its own MCMC/outputs: silently ignored.
IGNORED_KEYWORDS = {
    "maxsteps",
    "nthin",
    "ntemps",
    "nthreads",
    "nthread",
    "maxgr",
    "mintz",
    "dontstop",
    "ntry",
    "stretch",
    "randomfunc",
    "seed",
    "debug",
    "verbose",
    "plotonly",
    "bestonly",
    "skiptt",
    "nprint",
    "stopnow",
    "maxtime",
}

# keywords that pick a sampling parameterization exofastv2-side; EXOZIPPy
# always samples cosi/secosw/sesinw, so these change nothing.
NOOP_KEYWORDS = {"nochord", "novcve", "noyy", "notorres", "nomistsed"}

UNSUPPORTED_KEYWORDS = {
    "fitspline": "EXOFASTv2's Kepler-spline detrending is not implemented; "
    "the closest EXOZIPPy analog is 'gp: sho' on the transit "
    "file entry (correlated-noise model) or detrend columns",
    "splinespace": "see fitspline",
    "fitreflect": "reflected-light phase curves are not implemented",
    "fitdilute": "explicit dilution fitting is not implemented (EXOZIPPy "
    "dilutes transits automatically from the SED when several "
    "stars are modeled)",
    "fitbeam": "Doppler beaming is not implemented",
    "fitellip": "ellipsoidal variations are not implemented",
    "fitphase": "phase-curve fitting is not implemented",
    "ttvs": "per-transit TTVs are not implemented",
    "tivs": "per-transit inclination variations are not implemented",
    "tdvs": "per-transit depth variations are not implemented",
    "fitslope": "RV slopes are not implemented",
    "fitquad": "RV quadratic trends are not implemented",
    "fitlogmp": "log-mass sampling is not a user knob in EXOZIPPy",
    "fluxfile": "EXOFASTv2 flux files are not supported; convert the "
    "photometry to a .sed.yaml by hand",
    "mistsedfile": "MIST SED files are not supported; use sedfile",
    "dtpath": "Doppler tomography is not implemented",
    "yy": "Yonsei-Yale tracks are not implemented",
    "parsec": "PARSEC tracks are not implemented",
    "rejectflatmodel": "no EXOZIPPy equivalent (start-value sanity check)",
    "earth": "observer-frame keywords are not applicable",
    "bjd": "observer-frame keywords are not applicable",
}


def _bool_array(val, n, keyword):
    """Normalize an exofastv2 per-star flag (scalar or array) to n bools."""
    if val is None:
        return [False] * n
    if isinstance(val, list):
        flags = [bool(v) for v in val]
    else:
        flags = [bool(val)]
    if len(flags) < n:
        flags += [False] * (n - len(flags))
    elif len(flags) > n:
        warn(
            f"{keyword} has {len(flags)} entries but only {n} star(s); "
            "extra entries ignored"
        )
        flags = flags[:n]
    return flags


def convert(pro_path, outdir, base):
    pro_path = Path(pro_path).expanduser().resolve()
    pro_dir = pro_path.parent
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    kw = parse_pro_file(pro_path)
    consumed = set()

    def take(key, default=None):
        consumed.add(key)
        return kw.get(key, default)

    # ---- model dimensions and instance names -------------------------
    nplanets = int(take("nplanets", 1))
    nstars = int(take("nstars", 1))
    star_names = [chr(ord("A") + i) for i in range(nstars)]
    planet_names = [chr(ord("b") + i) for i in range(nplanets)]

    # ---- data files ---------------------------------------------------
    transits = []  # dicts: name, file(Path), band
    bands = []  # ordered unique band names (exofastv2 band index)
    tranpath = take("tranpath")
    if tranpath:
        for f in _resolve_glob(tranpath, pro_dir):
            name, band = _transit_meta(f)
            if any(t["name"] == name for t in transits):
                name = f"{name}_{len(transits)}"
            transits.append(dict(name=name, file=f, band=band))
            if band not in bands:
                bands.append(band)

    rvs = []  # dicts: name, file(Path)
    rvpath = take("rvpath")
    if rvpath:
        for f in _resolve_glob(rvpath, pro_dir):
            name = _rv_meta(f)
            if any(r["name"] == name for r in rvs):
                warn(
                    f"duplicate RV instrument label '{name}' from "
                    f"'{f.name}'; suffixing"
                )
                name = f"{name}_{len(rvs)}"
            rvs.append(dict(name=name, file=f))

    # per-file cadence smearing
    exptime = take("exptime")
    ninterp = take("ninterp")
    if take("longcadence"):
        exptime = exptime or 30.0
        ninterp = ninterp or 10
        info("longcadence -> exptime: 30 min, ninterp: 10 on every transit")
    for key, val in (("exptime", exptime), ("ninterp", ninterp)):
        if val is None:
            continue
        vals = val if isinstance(val, list) else [val] * len(transits)
        if len(vals) != len(transits):
            warn(
                f"{key} has {len(vals)} entries for {len(transits)} "
                "transit files; ignored"
            )
            continue
        for t, v in zip(transits, vals):
            t[key] = v

    # ---- SED ----------------------------------------------------------
    sed_yaml_name = None
    sedfile = take("sedfile")
    sed_lines = []
    alias_df = _load_filter_aliases()
    if sedfile:
        sed_src = (
            (pro_dir / Path(str(sedfile)).expanduser())
            if not Path(str(sedfile)).expanduser().is_absolute()
            else Path(str(sedfile)).expanduser()
        )
        if not sed_src.exists():
            warn(f"sedfile '{sed_src}' not found; skipping SED translation")
        else:
            sed_yaml_name = f"{base}.sed.yaml"
            sed_lines = _translate_sedfile(
                sed_src, alias_df, nstars, star_names
            )

    # ---- empirical relations -------------------------------------------
    mann_mass = _bool_array(take("mannmass"), nstars, "mannmass")
    mann_rad = _bool_array(take("mannrad"), nstars, "mannrad")
    mann_syn_mass = _bool_array(take("mannsynmass"), nstars, "mannsynmass")
    mann_syn_rad = _bool_array(take("mannsynrad"), nstars, "mannsynrad")
    torres_flags = _bool_array(take("torres"), nstars, "torres")

    # ---- MIST / Claret defaults ----------------------------------------
    if not take("nomist"):
        warn(
            "the EXOFASTv2 fit used MIST evolutionary tracks; EXOZIPPy has "
            "no evolutionary model yet. Setting mist: False -- constrain "
            "the star with mann/torres or explicit priors instead"
        )
    if not take("noclaret"):
        warn(
            "the EXOFASTv2 fit used Claret limb-darkening priors; the LD "
            "table prior is not implemented yet, so the limb darkening is "
            "constrained by the transit data alone"
        )

    # ---- thermal emission ----------------------------------------------
    fitthermal = take("fitthermal") or []
    if not isinstance(fitthermal, list):
        fitthermal = [fitthermal]
    for b in fitthermal:
        if b not in bands:
            warn(
                f"fitthermal band '{b}' does not match any transit band "
                f"({bands}); ignored"
            )

    # ---- prefix ---------------------------------------------------------
    prefix = take("prefix", f"fitresults/{base}")
    prefix = str(prefix).rstrip(".")

    # ---- priors ----------------------------------------------------------
    priorfile = take("priorfile")
    priors = []
    if priorfile:
        prior_src = (
            (pro_dir / Path(str(priorfile)).expanduser())
            if not Path(str(priorfile)).expanduser().is_absolute()
            else Path(str(priorfile)).expanduser()
        )
        if not prior_src.exists():
            warn(f"priorfile '{prior_src}' not found")
        else:
            priors = parse_priorfile(prior_src)

    axis_names = {
        "star": star_names,
        "planet": planet_names,
        "band": bands,
        "transit": [t["name"] for t in transits],
        "telescope": [r["name"] for r in rvs],
        "sed": [None],
    }

    param_entries = []  # (path, fields, comment)
    mann_ks = {}  # star index -> (ks, ks_err, ks_offset_initval)
    ld_priors = {}  # band name -> {"u1": prior, "u2": prior}

    for p in priors:
        name = p["name"]
        # u1/u2 need special handling: EXOZIPPy samples Kipping q1/q2 and
        # derives u1/u2, so start values must be inverted onto q1/q2
        # (initvals on derived parameters are ignored). Collect per band
        # and translate after the loop.
        if name in ("u1", "u2"):
            if not bands:
                warn(
                    f"prior '{name}' needs a band instance but none "
                    "exists; dropped"
                )
                continue
            if p["index"] is not None and p["index"] >= len(bands):
                warn(
                    f"prior '{name}_{p['index']}' indexes past the last "
                    "band; dropped"
                )
                continue
            targets = [bands[p["index"]]] if p["index"] is not None else bands
            for b in targets:
                ld_priors.setdefault(b, {})[name] = p
            continue
        # appks feeds the mann relation's Ks pathway, not a params entry
        if name == "appks":
            idx = p["index"] or 0
            if p["width"] and p["width"] > 0:
                off = None
                if p["start"] is not None:
                    off = (p["start"] - p["value"]) / p["width"]
                mann_ks[idx] = (p["value"], p["width"], off)
            else:
                warn(
                    "appks prior has no width; mann needs ks + ks_err -- "
                    "falling back to ks: synthetic"
                )
            continue
        if name in PRIOR_UNSUPPORTED:
            warn(f"prior '{name}' dropped: {PRIOR_UNSUPPORTED[name]}")
            continue
        if re.match(r"^[cm]\d+$", name):
            warn(
                f"detrending-coefficient prior '{name}' dropped; wire "
                "detrend columns on the instrument entry instead"
            )
            continue
        if name not in PRIOR_MAP:
            warn(
                f"prior '{name}' not recognized; add it to the generated "
                "params file by hand"
            )
            continue

        axis, template, scale = PRIOR_MAP[name]
        names = axis_names[axis]
        if not names:
            warn(
                f"prior '{name}' needs a {axis} instance but none exists; "
                "dropped"
            )
            continue
        if p["index"] is not None:
            if p["index"] >= len(names):
                warn(
                    f"prior '{name}_{p['index']}' indexes past the last "
                    f"{axis} instance; dropped"
                )
                continue
            targets = [names[p["index"]]]
        else:
            targets = names  # exofastv2: unindexed applies to all instances

        for n in targets:
            path = template.format(n=n)
            fields = {}
            value = p["value"] * scale
            width = None if p["width"] is None else p["width"] * abs(scale)
            start = None if p["start"] is None else p["start"] * scale
            if width is not None and width > 0:
                fields["mu"] = value
                fields["sigma"] = width
                fields["initval"] = start if start is not None else value
            elif width is not None and width == 0:
                fields["initval"] = value
                fields["sigma"] = 0.0
            else:
                fields["initval"] = start if start is not None else value
            if p["lower"] is not None:
                fields["lower"] = p["lower"] * abs(scale)
            if p["upper"] is not None:
                fields["upper"] = p["upper"] * abs(scale)
            comment = None
            if name == "mpsun":
                comment = f"mpsun {p['value']:.10g} Msun -> {value:.10g} Mjup"
            elif name == "omega":
                comment = "omega converted rad -> deg"
            param_entries.append((path, fields, comment))

    _translate_ld_priors(ld_priors, param_entries)

    # thermal priors imply fitthermal on that band
    for path, fields, _ in param_entries:
        m = re.match(r"^band\.(.+)\.thermal$", path)
        if m and m.group(1) not in fitthermal:
            fitthermal.append(m.group(1))
            info(f"thermal prior on band '{m.group(1)}' -> fitthermal: true")

    # ---- mann config entries ---------------------------------------------
    mann_entries = []
    for i, sname in enumerate(star_names):
        constrain = []
        syn = False
        if mann_mass[i] or mann_syn_mass[i]:
            constrain.append("mass")
        if mann_rad[i] or mann_syn_rad[i]:
            constrain.append("radius")
        if not constrain:
            continue
        syn = (mann_syn_mass[i] or mann_syn_rad[i]) or i not in mann_ks
        entry = dict(star=sname, constrain=constrain)
        if syn:
            if i not in mann_ks and not (mann_syn_mass[i] or mann_syn_rad[i]):
                if sedfile:
                    warn(
                        f"mann relation for star {sname} has no appks "
                        "prior; using ks: synthetic (needs the SED)"
                    )
                else:
                    warn(
                        f"mann relation for star {sname} has no appks "
                        "prior and no SED; supply ks/ks_err by hand"
                    )
            entry["ks"] = "synthetic"
        else:
            ks, ks_err, off = mann_ks[i]
            entry["ks"] = ks
            entry["ks_err"] = ks_err
            if off is not None and abs(off) > 1e-12:
                param_entries.append(
                    (
                        f"mann.{sname}.ks_offset",
                        {"initval": off},
                        "start value of the appks prior, as a non-centered "
                        "offset from its center in units of ks_err",
                    )
                )
        mann_entries.append(entry)

    torres_entries = [
        dict(star=star_names[i], constrain=["mass", "radius"])
        for i in range(nstars)
        if torres_flags[i]
    ]

    # ---- leftover keywords ------------------------------------------------
    for key, val in kw.items():
        if key in consumed:
            continue
        if key in IGNORED_KEYWORDS:
            info(f"'{key}={val}' ignored (EXOFASTv2 MCMC/output control)")
        elif key in NOOP_KEYWORDS:
            info(
                f"'/{key}' is a no-op: EXOZIPPy always samples "
                "cosi/secosw/sesinw"
            )
        elif key in UNSUPPORTED_KEYWORDS:
            warn(f"'{key}' not translated: {UNSUPPORTED_KEYWORDS[key]}")
        else:
            warn(
                f"exofastv2 keyword '{key}={val}' is not handled by this "
                "converter; check whether EXOZIPPy has an equivalent"
            )

    # ---- copy data files ---------------------------------------------------
    copied = []
    for entry in transits + rvs:
        src = entry["file"]
        dst = outdir / src.name
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(src.name)
        else:
            warn(f"data file '{src}' not found; not copied")

    # ---- write the files -----------------------------------------------------
    config_text = _emit_config(
        base=base,
        prefix=prefix,
        star_names=star_names,
        planet_names=planet_names,
        transits=transits,
        rvs=rvs,
        bands=bands,
        fitthermal=fitthermal,
        sed_yaml_name=sed_yaml_name,
        mann_entries=mann_entries,
        torres_entries=torres_entries,
        pro_path=pro_path,
    )
    (outdir / f"{base}.yaml").write_text(config_text)

    if param_entries:
        header = [
            f"Generated by scripts/exofast2exozippy.py from "
            f"{priorfile} (via {pro_path.name})",
            "EXOFASTv2 semantics: width>0 -> Gaussian prior (mu/sigma), "
            "width=0 -> fixed (sigma: 0),",
            "no width -> starting value only; a 5th column overrides the "
            "starting value.",
        ]
        (outdir / f"{base}.params.yaml").write_text(
            _emit_params_yaml(param_entries, header)
        )

    if sed_lines:
        (outdir / sed_yaml_name).write_text("\n".join(sed_lines) + "\n")

    # ---- report ---------------------------------------------------------------
    print(f"wrote {outdir / (base + '.yaml')}")
    if param_entries:
        print(f"wrote {outdir / (base + '.params.yaml')}")
    if sed_lines:
        print(f"wrote {outdir / sed_yaml_name}")
    for f in copied:
        print(f"copied {f}")
    if INFOS:
        print("\nnotes:")
        for m in INFOS:
            print(f"  - {m}")
    if WARNINGS:
        print(
            "\nWARNINGS (also embedded at the top of the config):",
            file=sys.stderr,
        )
        for m in WARNINGS:
            print(f"  - {m}", file=sys.stderr)
    print(f"\nnext: cd {outdir} && exozippy {base}.yaml")


# ----------------------------------------------------------------------
# Limb-darkening prior translation
# ----------------------------------------------------------------------


def _translate_ld_priors(ld_priors, param_entries):
    """Translate EXOFASTv2 u1/u2 priors onto EXOZIPPy's sampled q1/q2.

    EXOZIPPy samples the Kipping (2013) q1/q2 and derives u1/u2, so:
      - start values must be inverted onto q1/q2 (an initval on a derived
        parameter is ignored): q1 = (u1+u2)^2, q2 = u1 / (2*(u1+u2));
      - Gaussian priors (width > 0) stay on u1/u2 -- a sigma on a derived
        parameter is applied as a Gaussian potential on its value;
      - fixed values (width == 0) can only be honored exactly when BOTH
        u1 and u2 are fixed (then q1/q2 are fixed at the inversion).
    """
    for band, pri in ld_priors.items():
        u1p, u2p = pri.get("u1"), pri.get("u2")

        def _val(p):
            if p is None:
                return None
            return p["start"] if p["start"] is not None else p["value"]

        u1v, u2v = _val(u1p), _val(u2p)

        # Gaussian priors ride on the derived u1/u2 directly.
        for uname, p in (("u1", u1p), ("u2", u2p)):
            if p is None:
                continue
            if p["lower"] is not None or p["upper"] is not None:
                warn(
                    f"bounds on '{uname}' (band {band}) dropped: bounds "
                    "on the derived u1/u2 are not enforced (the Kipping "
                    "q1/q2 sampling already guarantees physical LD)"
                )
            if p["width"] is not None and p["width"] > 0:
                param_entries.append(
                    (
                        f"band.{band}.{uname}",
                        {"mu": p["value"], "sigma": p["width"]},
                        "Gaussian prior applied as a potential on the derived "
                        f"{uname}",
                    )
                )

        # Start values / fixed values invert onto q1/q2.
        if u1v is None or u2v is None:
            if u1v is not None or u2v is not None:
                warn(
                    f"band {band}: only one of u1/u2 has a value; cannot "
                    "invert to the sampled q1/q2 -- LD starts at the "
                    "component defaults"
                )
            continue
        both_fixed = u1p["width"] == 0 and u2p["width"] == 0
        if (u1p["width"] == 0) != (u2p["width"] == 0):
            warn(
                f"band {band}: only one of u1/u2 is fixed (width 0); "
                "EXOZIPPy can only fix the q1/q2 pair together -- both "
                "left free, seeded at the requested values"
            )
        usum = u1v + u2v
        q1 = usum**2
        q2 = u1v / (2.0 * usum) if usum != 0 else None
        if q2 is None or not (0 <= q1 <= 1 and 0 <= q2 <= 1):
            warn(
                f"band {band}: u1={u1v}, u2={u2v} falls outside the "
                "Kipping q1/q2 domain; LD start values dropped"
            )
            continue
        comment = (
            f"u1={u1v:.6g}, u2={u2v:.6g} inverted onto the sampled "
            "Kipping q1/q2"
        )
        for qname, qval in (("q1", q1), ("q2", q2)):
            fields = {"initval": qval}
            if both_fixed:
                fields["sigma"] = 0.0
            param_entries.append((f"band.{band}.{qname}", fields, comment))
            comment = None


# ----------------------------------------------------------------------
# SED file translation
# ----------------------------------------------------------------------


def _translate_sedfile(path, alias_df, nstars, star_names):
    """EXOFASTv2 sed file (band mag used_err [catalog_err [starndx]]) ->
    EXOZIPPy .sed.yaml lines."""
    lines = [
        f"# Generated by scripts/exofast2exozippy.py from {path.name}.",
        "# err is EXOFASTv2's used_errors column (3rd column); the catalog",
        "# error (4th column, if present) is kept as a comment.",
        "",
        "model: NextGen",
        f"nstars: {nstars}",
        "filters:",
    ]
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 3:
            warn(f"sed line not understood, skipped: '{raw.strip()}'")
            continue
        band = toks[0]
        try:
            mag = float(toks[1])
            err = float(toks[2])
            cat = float(toks[3]) if len(toks) > 3 else None
        except ValueError:
            warn(f"sed line not understood, skipped: '{raw.strip()}'")
            continue
        svo, ok = _to_svo(band, alias_df)
        if not ok:
            warn(
                f"SED filter '{band}' not found in the alias table; "
                "copied verbatim -- fix the name if the fit rejects it"
            )
        lines.append(f'    - name: "{svo}"')
        lines.append(f"      mag: {mag}")
        suffix = f"   # catalog err {cat}" if cat is not None else ""
        lines.append(f"      err: {err}{suffix}")
        # optional per-row star list (multi-star EXOFASTv2 sed files)
        if len(toks) > 4:
            try:
                idxs = [int(t) for t in toks[4].split(",")]
                names = [star_names[i] for i in idxs]
                lines.append("      photType:")
                lines.append(f"        pos: [{', '.join(names)}]")
            except (ValueError, IndexError):
                warn(
                    f"sed star-index column not understood on: '{raw.strip()}'"
                )
        lines.append("")
    return lines


# ----------------------------------------------------------------------
# Config emission
# ----------------------------------------------------------------------


def _emit_config(
    base,
    prefix,
    star_names,
    planet_names,
    transits,
    rvs,
    bands,
    fitthermal,
    sed_yaml_name,
    mann_entries,
    torres_entries,
    pro_path,
):
    L = []
    L.append(f"# Generated by scripts/exofast2exozippy.py from {pro_path}")
    if WARNINGS:
        L.append("#")
        L.append(
            "# Conversion warnings (features of the EXOFASTv2 fit that "
            "did not translate):"
        )
        for m in WARNINGS:
            for i, chunk in enumerate(_wrap(m, 72)):
                L.append(f"#   {'- ' if i == 0 else '  '}{chunk}")
    L.append("")
    L.append("run:")
    L.append(f'  name: "{base}"')
    L.append("")
    L.append(f'prefix: "{prefix}"')
    L.append("logger_level: INFO   # DEBUG, INFO, or WARNING")
    L.append("")

    L.append("star:")
    for s in star_names:
        L.append(f'  - name: "{s}"')
        L.append("    mist: False")
    L.append("")

    if sed_yaml_name:
        L.append("sed:")
        L.append(f'  file: "{sed_yaml_name}"')
        L.append("")

    for comp, entries in (("mann", mann_entries), ("torres", torres_entries)):
        if not entries:
            continue
        L.append(f"{comp}:")
        for e in entries:
            L.append(f'  - star: "{e["star"]}"')
            L.append(f"    constrain: [{', '.join(e['constrain'])}]")
            if "ks" in e:
                if e["ks"] == "synthetic":
                    L.append("    ks: synthetic")
                else:
                    L.append(f"    ks: {e['ks']}")
                    L.append(f"    ks_err: {e['ks_err']}")
        L.append("")

    L.append("planet:")
    for p in planet_names:
        L.append(f'  - name: "{p}"')
    L.append("")

    L.append("orbit:")
    for p in planet_names:
        L.append(f'  - name: "{p}"')
        L.append(f'    primary: ["{star_names[0]}"]')
        L.append(f'    companion: ["{p}"]')
    L.append("")

    if transits:
        L.append("transit:")
        for t in transits:
            L.append(f'  - name: "{t["name"]}"')
            L.append(f'    file: "{t["file"].name}"')
            L.append(f'    band: "{t["band"]}"')
            if "exptime" in t:
                L.append(f"    exptime: {t['exptime']} # minutes")
            if "ninterp" in t:
                L.append(f"    ninterp: {int(t['ninterp'])}")
        L.append("")

    if bands:
        L.append("band:")
        for b in bands:
            L.append(f'  - name: "{b}"')
            L.append(f'    filter: "{b}"')
            if b in fitthermal:
                L.append("    fitthermal: true")
        L.append("")

    if rvs:
        L.append("rvinstrument:")
        for r in rvs:
            L.append(f'  - name: "{r["name"]}"')
            L.append(f'    file: "{r["file"].name}"')
        L.append("")

    L.append(f'parameter_file: "{base}.params.yaml"')
    L.append("")
    L.append(SAMPLER_BLOCK)
    return "\n".join(L)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Convert an EXOFASTv2 driver .pro file into EXOZIPPy "
        "YAML inputs (config, params, sed) plus copied data "
        "files.",
        epilog="Run from the directory where the EXOZIPPy fit should live, "
        "e.g. cd examples/gj1214 && python "
        "../../scripts/exofast2exozippy.py "
        "~/modeling/gj1214/fitclass.pro",
    )
    parser.add_argument("profile", help="EXOFASTv2 driver .pro file")
    parser.add_argument(
        "-o", "--outdir", default=".", help="output directory (default: cwd)"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="base name for the generated YAML files "
        "(default: the .pro file's parent directory "
        "name)",
    )
    args = parser.parse_args()

    pro_path = Path(args.profile).expanduser()
    if not pro_path.exists():
        sys.exit(f"ERROR: {pro_path} not found")
    base = args.name or pro_path.resolve().parent.name
    convert(pro_path, args.outdir, base)


if __name__ == "__main__":
    main()
