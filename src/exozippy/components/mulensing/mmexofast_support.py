"""Consume (and optionally produce) MMEXOFAST output for the mulensing stack.

MMEXOFAST (https://github.com/jenniferyee/MMEXOFAST) produces a JSON dict of
lightly-optimized microlensing solutions spanning the standard degeneracies:

    {"fits": [{"parameters": {...}, "sigmas": {...}}, ...],
     "errfacs": {"<file basename>": float, ...},
     "mag_methods": [...],
     "coords": "17:54:19.20 -30:22:38.00" | null,     # newer versions only
     "jd_offset": 2450000.0 | 0.0,                     # newer versions only
     "excluded_points": {"<file basename>": {"n_data": int,
                                             "indices": [int, ...],
                                             "times": [float, ...]}}}

Three consumers share this module:

- ``Lens._load_mmexofast_seeds`` (stage 2) pushes each fit as a per-seed hint
  set when the lens block names an explicit ``mmexofast: <file>``.
- ``MulensInstrument._resolve_mmexofast`` (stage 1a) applies the bad-data
  mask (``excluded_points``) and error-rescaling factors (``errfacs``) to its
  own files, and -- when the user supplied no sufficient start values and no
  explicit file -- runs MMEXOFAST on the raw light curves to generate all of
  the above automatically ("data-driven hints").
- ``utilities/mmexofast_to_params.py`` translates the same JSON to a
  params.yaml for humans; it deliberately does not import this module so the
  CLI works without the package installed.

Only ``run_or_load`` touches the mmexofast package, and imports it lazily:
exozippy must stay importable (and publishable) without it, since mmexofast
is installable only from git (see the microlensing group in pyproject.toml).

Time systems: newer MMEXOFAST adds ``jd_offset`` to every epoch parameter so
the JSON is always full JD. EXOZIPPy's t_0 must live in the DATA's own time
system (the initval is compared against raw file times), so seed extraction
subtracts ``jd_offset`` back out. ``excluded_points`` are consumed as
INDICES (positions in the file's own row order), which are offset-free.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from ...config import RANK_DERIVED_DATA

logger = logging.getLogger(__name__)


def load_json(path):
    """Read an MMEXOFAST JSON; return the dict, or None with a warning."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read mmexofast file '{path}': {e}.")
        return None


def user_hints_sufficient(user_params, is_binary, want_rho):
    """True when the user supplied a start value for every microlensing
    observable this topology needs (t_0, u_0, t_E; rho when finite-source;
    s-or-log_s, alpha, q when a lens companion exists).

    ``user_params`` is the ConfigManager's standardized dict, so lens paths
    are already in ``lens.0.<param>`` form. Bounds-only entries do not count:
    the point of the check is whether the relaxation engine has a START, and
    only initval (or a fixed mu) provides one.
    """

    def has_start(*paths):
        for p in paths:
            entry = user_params.get(p)
            if isinstance(entry, dict) and (
                entry.get("initval") is not None or entry.get("mu") is not None
            ):
                return True
        return False

    ok = (
        has_start("lens.0.t_0")
        and has_start("lens.0.u_0")
        and has_start("lens.0.t_E")
    )
    if want_rho:
        ok = ok and has_start("lens.0.rho")
    if is_binary:
        ok = (
            ok
            and has_start("lens.0.log_s", "lens.0.s")
            and has_start("lens.0.alpha")
            and has_start("lens.0.q")
        )
    return ok


def push_seed_hints(data, config_manager, want_rho, is_binary, source="?"):
    """Push every MMEXOFAST fit as a per-seed hint set (P4 multi-seed
    sampling), plus scale hints from fit 0's sigmas.

    Alpha convention: MMEXOFAST alpha maps to EXOZIPPy alpha by the IDENTITY
    transform (verified against examples/DC2018_128; the ob161003
    ``alpha_MM = 180 - alpha_paper`` note is a paper-vs-MM relation on a
    different event and does NOT apply).

    s is seeded as log_s = log10(s) because the mulensing manifest samples
    log_s (s = 10**log_s is derived).

    Epochs are shifted back into the data's own time system by subtracting
    the JSON's ``jd_offset`` (0.0 when absent, and for pre-jd_offset files).

    Rank sits between RANK_DERIVED_DATA and RANK_USER (config.add_seed_hints
    default) so an explicit user initval list still wins.

    Returns the number of seed solutions pushed (0 when the file has none).
    """
    fits = data.get("fits", [])
    if not fits:
        logger.warning(
            f"mmexofast output '{source}' has no 'fits'; no seeds loaded."
        )
        return 0

    jd_offset = float(data.get("jd_offset", 0.0) or 0.0)

    seed_sets = []
    for fit in fits:
        p = fit.get("parameters", {})
        d = {}
        if "t_0" in p:
            d["lens.0.t_0"] = float(p["t_0"]) - jd_offset
        for key, path in (
            ("u_0", "lens.0.u_0"),
            ("t_E", "lens.0.t_E"),
        ):
            if key in p:
                d[path] = float(p[key])
        if want_rho and "rho" in p:
            d["lens.0.rho"] = float(p["rho"])
        # s/q/alpha are companion (binary-lens) geometry only.
        if is_binary:
            if "s" in p and float(p["s"]) > 0:
                d["lens.0.log_s"] = float(np.log10(float(p["s"])))
            if "alpha" in p:
                d["lens.0.alpha"] = float(p["alpha"])  # identity convention
            if "q" in p:
                d["lens.0.q"] = float(p["q"])
        seed_sets.append(d)

    config_manager.add_seed_hints(seed_sets)
    logger.info(
        f"MMEXOFAST: loaded {len(seed_sets)} seed solution(s) from '{source}'."
    )

    # Scale hints from fit 0's sigmas (bounds/scales resolve from seed 0
    # only; other seeds move only the start). log_rho/log_s/log_q sigmas are
    # in dex -> convert to a linear scale as sigma_x = x * ln(10) * sigma_logx
    # (matches examples/DC2018_128/compare_results.py).
    s0 = fits[0].get("sigmas", {})
    p0 = fits[0].get("parameters", {})
    ln10 = np.log(10.0)

    def _sh(path, val):
        if val is not None and np.isfinite(val) and val > 0:
            config_manager.add_scale_hint(path, float(val))

    _sh("lens.0.t_0", s0.get("t_0"))
    _sh("lens.0.u_0", s0.get("u_0"))
    _sh("lens.0.t_E", s0.get("t_E"))
    if want_rho and "log_rho" in s0 and "rho" in p0:
        _sh("lens.0.rho", float(p0["rho"]) * float(s0["log_rho"]) * ln10)
    if is_binary:
        if "log_s" in s0:
            _sh("lens.0.log_s", float(s0["log_s"]))
        if "alpha" in s0:
            _sh("lens.0.alpha", float(s0["alpha"]))
        if "log_q" in s0 and "q" in p0:
            _sh("lens.0.q", float(p0["q"]) * float(s0["log_q"]) * ln10)

    return len(seed_sets)


def apply_excluded_points(data, files, mask_specs, context, robust_kinds=None):
    """Merge the JSON's ``excluded_points`` into an instrument's mask specs.

    MMEXOFAST keys each dataset by file basename and reports 0-based indices
    into the file's own row order -- exactly the index-list form
    ``Instrument._apply_mask`` accepts. An explicit user ``mask:`` on a file
    wins: only entries whose spec is still None are filled. Returns the
    updated list (mutated in place too).

    ``robust_kinds`` is the instrument's per-file ``likelihood_kinds`` list;
    a file that opted into a robust likelihood (hogg mixture or Student-t)
    is skipped: the robust likelihood supersedes the hard mask -- it refits
    the outlier verdict at every posterior draw instead of freezing
    MMEXOFAST's, and keeps the points auditable via
    ``Instrument.outlier_prob_at_data``. MMEXOFAST still used the exclusions
    internally (protecting its own chi2-based anomaly search), and the
    errfacs remain consumed either way -- only the propagation of the hard
    mask into EXOZIPPy is dropped. A user's own ``mask:`` still applies.
    """
    excluded = data.get("excluded_points") or {}
    if not excluded:
        return mask_specs
    by_base = {os.path.basename(str(f)): i for i, f in enumerate(files)}
    for label, info in excluded.items():
        i = by_base.get(os.path.basename(str(label)))
        if i is None:
            logger.warning(
                f"[{context}] mmexofast excluded_points entry '{label}' "
                f"matches none of this component's files; ignored."
            )
            continue
        indices = list(info.get("indices") or [])
        if not indices:
            continue
        if robust_kinds is not None and robust_kinds[i]:
            logger.info(
                f"[{context}] file '{label}' has likelihood: "
                f"{robust_kinds[i]}; leaving its {len(indices)} "
                f"mmexofast-excluded point(s) unmasked -- the robust "
                f"likelihood handles them (and reports their posterior "
                f"outlier probabilities)."
            )
            continue
        if mask_specs[i] is not None:
            logger.info(
                f"[{context}] file '{label}' already has a user mask; "
                f"ignoring the {len(indices)} mmexofast-excluded points."
            )
            continue
        mask_specs[i] = [int(k) for k in indices]
        logger.info(
            f"[{context}] masking {len(indices)} mmexofast-excluded "
            f"point(s) in '{label}'."
        )
    return mask_specs


def push_errfac_hints(data, files, prefix, config_manager):
    """Seed each instrument's err_scale initval from the JSON's ``errfacs``.

    MMEXOFAST reports the factor its own error renormalization multiplied
    the file's errors by; EXOZIPPy reads the ORIGINAL file and samples a
    multiplicative err_scale, so the factor is the natural start value.
    RANK_DERIVED_DATA: beats defaults.yaml, yields to the user.
    """
    errfacs = data.get("errfacs") or {}
    if not errfacs:
        return
    by_base = {os.path.basename(str(f)): i for i, f in enumerate(files)}
    for label, fac in errfacs.items():
        i = by_base.get(os.path.basename(str(label)))
        if i is None:
            logger.warning(
                f"[{prefix}] mmexofast errfacs entry '{label}' matches none "
                f"of this component's files; ignored."
            )
            continue
        fac = float(fac)
        if not (np.isfinite(fac) and fac > 0):
            continue
        config_manager.add_hint(
            f"{prefix}.{i}.err_scale", fac, rank=RANK_DERIVED_DATA
        )
        logger.info(
            f"[{prefix}] err_scale[{i}] seeded at the mmexofast error "
            f"renormalization factor {fac:.4f} ('{label}')."
        )


def run_or_load(
    json_path,
    files,
    coords=None,
    fit_type="point_lens",
    renormalize_errors=True,
    no_parallax=True,
    options=None,
    log_file=None,
):
    """Run MMEXOFAST on ``files`` and cache its exozippy-init JSON, or load
    the cached JSON if it already exists.

    The cache makes reruns (recompute_trace, restarts on a cluster) free:
    delete the JSON to force a fresh MMEXOFAST fit. ``options`` is an
    optional dict of extra ``MMEXOFASTFitter`` keyword arguments (e.g.
    ``limb_darkening_coeffs_gamma``) forwarded verbatim, so anything the
    fitter accepts is reachable from YAML without a new exozippy knob.

    Raises ImportError with install instructions when the package is
    missing: silently continuing would start the fit from defaults.yaml
    values, which for a raw light curve is a hopeless (and on a cluster,
    expensive) start. Opt out with ``mmexofast: false`` on the lens block.
    """
    json_path = Path(json_path)
    if json_path.exists():
        logger.info(f"MMEXOFAST: using cached output '{json_path}'.")
        return load_json(json_path)

    try:
        import mmexofast as mmexo
    except ImportError as e:
        raise ImportError(
            "MMEXOFAST auto-initialization needs the 'mmexofast' package "
            "(poetry install --with microlensing, or pip install "
            "git+https://github.com/jenniferyee/MMEXOFAST.git). Either "
            "install it, supply start values for the microlensing "
            "parameters in the params file, or set 'mmexofast: false' on "
            "the lens block to opt out."
        ) from e

    kwargs = dict(
        files=[str(f) for f in files],
        fit_type=fit_type,
        renormalize_errors=renormalize_errors,
        no_parallax=no_parallax,
        verbose=False,
        # Default the fitter's own log next to the JSON so a slow or stuck
        # stage is diagnosable (MMEXOFAST logs each workflow step).
        log_file=(
            str(log_file)
            if log_file is not None
            else str(json_path.with_suffix(".log"))
        ),
    )
    if coords is not None:
        kwargs["coords"] = coords
    kwargs.update(options or {})

    logger.info(
        f"MMEXOFAST: no sufficient user start values -- fitting "
        f"{len(files)} light curve(s) (fit_type={fit_type}, "
        f"renormalize_errors={kwargs['renormalize_errors']}, "
        f"no_parallax={kwargs['no_parallax']}). This can take a while; the "
        f"result is cached at '{json_path}'."
    )
    with mmexo.MMEXOFASTFitter(**kwargs) as fitter:
        fitter.fit()
        data = fitter.initialize_exozippy()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"MMEXOFAST: wrote '{json_path}'.")
    return data
