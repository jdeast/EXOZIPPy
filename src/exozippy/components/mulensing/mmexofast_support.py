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

- ``Lens._load_mmexofast_seeds`` (stage 3) pushes each fit as a per-seed hint
  set when the lens block names an explicit ``mmexofast: <file>``.
- ``MulensInstrument._resolve_mmexofast`` (stage 1) applies the bad-data
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

Failure policy: seeding that cannot happen must SAY so. Starting a raw
light-curve fit from defaults.yaml is hopeless, and on a cluster expensively
so, which is why the missing-package case raises rather than continuing (see
``run_or_load``). A JSON that exists but cannot be parsed is the same failure
wearing a quieter coat, so ``load_json`` distinguishes three cases and never
collapses them:

- ABSENT: ``None`` plus a warning; the caller decides (an explicit
  ``mmexofast: <file>`` warns and skips, the auto path generates the file).
- PRESENT and well-formed: the dict.
- PRESENT and unreadable / unparseable / structurally wrong: raises
  ``CorruptMMEXOFASTFileError``. ``run_or_load`` catches it for its OWN
  cache -- a derived artifact with an unambiguous recovery (regenerate),
  the same call ``utilities/zenodo.py`` makes for a corrupt download -- and
  moves the bad file aside first so nothing is overwritten silently. A
  user-named file is not ours to rebuild, so there the exception propagates.

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

from ...config import PRECEDENCE_DERIVED_DATA

logger = logging.getLogger(__name__)


class CorruptMMEXOFASTFileError(ValueError):
    """An MMEXOFAST JSON exists but cannot be used.

    Distinct from "absent" on purpose. Absent is a normal state with a
    normal recovery (generate it, or run without seeds because the user said
    so); present-but-broken is a state nobody chose, and the observed cause
    -- a job killed while ``run_or_load`` was writing the cache -- leaves a
    file whose only visible symptom, before this exception existed, was a
    fit that quietly started from defaults.yaml.
    """


# The one key every MMEXOFAST exozippy-init JSON carries. Its absence means
# the file is not one (wrong path, half-written, hand-edited), which is
# worth catching here: `push_seed_hints` would only log "no 'fits'" and let
# the run continue seedless.
_REQUIRED_KEY = "fits"


def load_json(path):
    """Read an MMEXOFAST JSON.

    Returns
    -------
    dict or None
        The parsed JSON object, or None when ``path`` does not exist (logged
        as a warning; absence is the caller's decision to make).

    Raises
    ------
    CorruptMMEXOFASTFileError
        The file exists but could not be opened, is not valid JSON, is not a
        JSON object, or lacks the ``fits`` key. Every one of those means the
        seeds this file was supposed to supply will not be applied, and the
        module's contract (see the module docstring) is that seeding which
        cannot happen is reported, not absorbed. ``fits: []`` is legal -- an
        MMEXOFAST run that found no solutions is a real answer, and
        ``push_seed_hints`` already warns about it.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"mmexofast file '{path}' does not exist.")
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise CorruptMMEXOFASTFileError(
            f"mmexofast file '{path}' exists but could not be read: {e}. "
            f"A partially written file is usually a job killed mid-write. "
            f"Delete it to regenerate, point 'mmexofast:' at a good file, "
            f"supply start values for the microlensing parameters in the "
            f"params file, or set 'mmexofast: false' on the lens block to "
            f"opt out -- but do not let the fit start from defaults."
        ) from e

    if not isinstance(data, dict) or _REQUIRED_KEY not in data:
        raise CorruptMMEXOFASTFileError(
            f"mmexofast file '{path}' parsed but has no '{_REQUIRED_KEY}' "
            f"key, so it is not MMEXOFAST exozippy-init output (got "
            f"{type(data).__name__} with keys "
            f"{sorted(data)[:6] if isinstance(data, dict) else 'n/a'}). "
            f"Check the path, or delete the file to regenerate it."
        )
    return data


def user_hints_sufficient(config_manager, is_binary, want_rho):
    """True when every microlensing observable this topology needs (t_0, u_0,
    t_E; rho when finite-source; s-or-log_s, alpha, q when a lens companion
    exists) was either given outright or can be **derived** from what was.

    Derivability, not literal presence, is the right question, because
    several of these are derived parameters a params file legitimately never
    names.  The case that matters is a restart file written by mkparam: it
    carries only sampled coordinates, so `lens.q` (derived from the body
    masses) and `lens.t_E` (from theta_E / mu_rel) are both absent -- yet
    both are fully determined by the `planet.log_q`/`planet.mass`,
    `star.logmass`, distance and proper-motion entries it does carry.
    Scanning for literal keys therefore declared a complete restart file
    "insufficient" and re-ran MMEXOFAST on every second-iteration fit.

    So ask the relaxation engine (`ConfigManager.probe_derivable`), which
    follows exactly the relations that will set these values for real at
    stage 4.  Bounds-only entries still do not count: a bound is not a start.

    A literal entry for every observable short-circuits the probe: naming a
    value outright makes it PRECEDENCE_USER, which is derivable by definition, so
    the engine cannot change the answer.  That keeps the common hand-written
    params file on the old zero-cost path and spends the extra solve only
    where the literal scan would have been wrong.
    """
    required = ["lens.0.t_0", "lens.0.u_0", "lens.0.t_E"]
    if want_rho:
        required.append("lens.0.rho")
    if is_binary:
        required += ["lens.0.alpha", "lens.0.q"]

    def named(path):
        entry = config_manager.user_params.get(path)
        return isinstance(entry, dict) and (
            entry.get("initval") is not None or entry.get("mu") is not None
        )

    # s and log_s are one fact in two coordinates; either satisfies it.
    if all(named(p) for p in required) and (
        not is_binary or named("lens.0.s") or named("lens.0.log_s")
    ):
        return True

    derivable = config_manager.probe_derivable(
        required + ["lens.0.s", "lens.0.log_s"]
    )
    ok = all(p in derivable for p in required)
    if is_binary:
        ok = ok and ("lens.0.s" in derivable or "lens.0.log_s" in derivable)
    if not ok:
        missing = [p for p in required if p not in derivable]
        logger.debug(
            f"MMEXOFAST trigger: {missing} cannot be derived from the "
            f"supplied parameters; a fit is needed."
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

    Rank is PRECEDENCE_DERIVED_DATA (config.add_seed_hints): MMEXOFAST is a very
    fancy derivation FROM THE DATA, not a user statement, so it sits in the
    same tier as any other data-driven hint and EVERY user entry -- an initval
    list, and equally a plain scalar initval -- outranks it.

    A fit missing some of the observables this topology needs seeds only the
    ones it has -- PARTIAL seeding, which is worse than none because the
    unseeded parameters fall back to defaults.yaml while the seeded ones do
    not, putting the start in a place no MMEXOFAST solution ever occupied.
    That cannot be silent, so every skipped observable is named in a warning.
    It is not fatal: which observables matter is the topology's call (a PSPL
    JSON reused for a binary lens is a real, if degraded, start), and
    ``user_hints_sufficient`` re-asks the relaxation engine afterwards.

    Returns the number of seed solutions pushed (0 when the file has none).
    """
    fits = data.get("fits") or []
    if not fits:
        logger.warning(
            f"mmexofast output '{source}' has no 'fits'; no seeds loaded."
        )
        return 0

    jd_offset = float(data.get("jd_offset", 0.0) or 0.0)

    wanted = ["t_0", "u_0", "t_E"]
    if want_rho:
        wanted.append("rho")
    if is_binary:
        wanted += ["s", "alpha", "q"]

    seed_sets = []
    for i, fit in enumerate(fits):
        p = fit.get("parameters", {})
        absent = [k for k in wanted if k not in p]
        if absent:
            logger.warning(
                f"mmexofast output '{source}' fit {i} has no {absent}; "
                f"those parameters keep their defaults.yaml start while the "
                f"rest are seeded from this fit. Check that the file matches "
                f"this lens topology (binary={is_binary}, "
                f"finite_source={want_rho})."
            )
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
    PRECEDENCE_DERIVED_DATA: beats defaults.yaml, yields to the user.
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
            f"{prefix}.{i}.err_scale", fac, rank=PRECEDENCE_DERIVED_DATA
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

    A cache that exists but cannot be parsed is REGENERATED, not tolerated
    and not raised on: this file is exozippy's own derived artifact, so the
    recovery is unambiguous (delete it and refit) -- the same call
    ``utilities/zenodo.py`` makes for a corrupt download. It is never
    overwritten silently, though: the unreadable file is moved aside to
    ``<name>.corrupt`` and the warning names it, so a half-written cache is
    still there to look at. Contrast ``load_json`` on a user-named
    ``mmexofast: <file>``, which raises -- exozippy did not write that file
    and cannot rebuild it.

    Raises ImportError with install instructions when the package is
    missing: silently continuing would start the fit from defaults.yaml
    values, which for a raw light curve is a hopeless (and on a cluster,
    expensive) start. Opt out with ``mmexofast: false`` on the lens block.
    """
    json_path = Path(json_path)
    corrupt = None
    if json_path.exists():
        try:
            data = load_json(json_path)
        except CorruptMMEXOFASTFileError as e:
            corrupt = e
        else:
            # Logged only once the cache is known to be usable -- the old
            # order announced "using cached output" and then used nothing.
            logger.info(f"MMEXOFAST: using cached output '{json_path}'.")
            return data

    try:
        import mmexofast as mmexo
    except ImportError as e:
        preamble = (
            f"The cached MMEXOFAST output is unusable ({corrupt}) and "
            f"cannot be regenerated: "
            if corrupt is not None
            else ""
        )
        raise ImportError(
            preamble
            + "MMEXOFAST auto-initialization needs the 'mmexofast' package "
            "(poetry install --with microlensing, or pip install "
            "git+https://github.com/jenniferyee/MMEXOFAST.git). Either "
            "install it, supply start values for the microlensing "
            "parameters in the params file, or set 'mmexofast: false' on "
            "the lens block to opt out."
        ) from e

    if corrupt is not None:
        # Quarantine rather than clobber: the bad file is the only evidence
        # of how the previous run died, and an overwrite that says nothing
        # is the failure mode this whole path exists to stop.
        quarantine = json_path.with_name(json_path.name + ".corrupt")
        json_path.replace(quarantine)
        logger.warning(
            f"MMEXOFAST: {corrupt} Regenerating it -- the unreadable file "
            f"was moved to '{quarantine}' and MMEXOFAST is being re-run."
        )

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

    # Write through a .part file and rename: json.dump straight onto the
    # cache path is what produced the truncated caches this function now has
    # to detect (a cluster job killed mid-write). os.replace is atomic
    # within a filesystem, so the cache path only ever holds a complete file.
    json_path.parent.mkdir(parents=True, exist_ok=True)
    part = json_path.with_name(json_path.name + ".part")
    with open(part, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(part, json_path)
    logger.info(f"MMEXOFAST: wrote '{json_path}'.")
    return data
