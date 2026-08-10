"""mkparam - Seed a params.yaml from the MAP of a previous trace."""

import logging
import re
from pathlib import Path

import arviz as az
import numpy as np
import yaml

from exozippy.config import validate_sigma_has_center
from exozippy.samplers import convergence
from exozippy.trace_meta import check_trace_freshness

logger = logging.getLogger(__name__)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get_instance_names(config, comp_key):
    entries = config.get(comp_key, [])
    if not isinstance(entries, list):
        return []
    return [str(e.get("name", i)) for i, e in enumerate(entries)]


def _find_existing(existing_params, comp_key, idx, name, param):
    for key in (
        f"{comp_key}.{name}.{param}",
        f"{comp_key}.{idx}.{param}",
        f"{comp_key}.{param}",
    ):
        if key in existing_params:
            return key, existing_params[key]
    return None, None


def _apply_existing_constraints(entry, existing_entry):
    """Layer an existing params entry's constraint fields onto a fresh entry.

    ``entry`` already carries the new ``initval`` (the trace MAP, or the
    length-K seed list). The existing entry's mu/sigma/bounds are copied over
    unchanged -- mu is the prior center, not the starting point, so it must
    never drift toward the MAP.

    If the original carried a Gaussian prior (sigma > 0) but no explicit mu,
    its initval WAS the prior center (``Parameter.build_pymc`` centers the
    potential on initval whenever mu is absent, for sampled and derived
    parameters alike), so promote it to mu. Without the promotion the prior
    would silently follow the MAP on every successive mkprior run.

    Returns ``entry`` (mutated in place) for convenience.
    """
    if not isinstance(existing_entry, dict):
        return entry
    for prior_key in ("mu", "sigma", "lower", "upper"):
        if prior_key in existing_entry:
            entry[prior_key] = existing_entry[prior_key]
    existing_sigma = existing_entry.get("sigma")
    if (
        existing_sigma is not None
        and float(existing_sigma) != 0
        and "mu" not in existing_entry
        and "initval" in existing_entry
    ):
        entry["mu"] = existing_entry["initval"]
    return entry


def _normalize_key(key, config):
    """Rewrite comp.0.param index notation to comp.Name.param for readability."""
    parts = key.split(".", 2)
    if len(parts) == 3:
        comp_key, idx_or_name, param = parts
        try:
            idx = int(idx_or_name)
            instance_names = _get_instance_names(config, comp_key)
            if idx < len(instance_names):
                return f"{comp_key}.{instance_names[idx]}.{param}"
        except ValueError:
            pass
    return key


def _next_versioned_path(param_path):
    """Return the path with its version suffix incremented by one.

    foo.params.yaml      → foo.params.2.yaml
    foo.params.2.yaml    → foo.params.3.yaml
    foo.params.12.yaml   → foo.params.13.yaml
    """
    p = Path(param_path)
    suffix = p.suffix  # ".yaml"
    # Strip the last extension to expose possible version number
    stem = p.name[
        : p.name.rfind(suffix)
    ]  # e.g. "foo.params" or "foo.params.2"
    m = re.search(r"^(.*?)\.(\d+)$", stem)
    if m:
        base, n = m.group(1), int(m.group(2))
    else:
        base, n = stem, 1
    return p.parent / f"{base}.{n + 1}{suffix}"


def _mode_seed_quotas(weights, n):
    """Allocate ``n`` seeds across modes: every mode gets at least one seed
    (a restart must never erase a discovered mode), the rest go by weight
    (largest remainder). With more modes than seeds, the top-n by weight
    get one each."""
    order = np.argsort(weights)[::-1]
    if n <= len(weights):
        quotas = np.zeros(len(weights), dtype=int)
        quotas[order[:n]] = 1
        return quotas
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    remaining = n - len(w)
    ideal = w * remaining
    quotas = np.ones(len(w), dtype=int) + np.floor(ideal).astype(int)
    shortfall = n - int(quotas.sum())
    if shortfall > 0:
        frac_order = np.argsort(ideal - np.floor(ideal))[::-1]
        quotas[frac_order[:shortfall]] += 1
    return quotas


def _sample_seed_draws(idata, n, exclude, rng_seed=0):
    """Pick ``n`` random JOINT (chain, draw) index pairs for multi-seed starts.

    When the trace is multimodal (outputs.modes.identify_modes finds more
    than one mode), the draws are STRATIFIED BY MODE -- every mode gets at
    least one seed and the rest are allocated by mode weight -- so a restart
    can never launder a multimodal posterior into a single-basin start set.
    (The old good-chain pooling did exactly that: good_chain_mask is a
    stuck-chain detector, and against a multimodal trace it classifies mode
    membership as sickness -- on DC2018 event 128 it flagged the 52
    majority-branch chains as bad because the 2 true-branch chains' lp was
    582 nats higher.)

    Within a mode (and in the unimodal fall-back path) draws are taken from
    the POST-BURN-IN region of the good chains, so every seed is a real
    point in the equilibrated posterior. Whole draws are returned (a chain
    and draw index), never per-parameter marginals, so a downstream consumer
    that reads all parameters at (chain, draw) gets one self-consistent
    point.

    Returns (pairs, good_mask, burnin) where ``good_mask`` and ``burnin``
    also describe the pool used, so a caller can compute statistics over
    exactly the same post-burn-in good draws.
    """
    post = idata["posterior"]
    var_names = convergence.default_var_names(post)
    arrays = {v: post[v].values for v in var_names}
    lp = None
    ss = idata.get("sample_stats") if hasattr(idata, "get") else None
    if ss is not None and "lp" in ss.data_vars:
        lp = ss["lp"].values

    diag = convergence.find_burnin(arrays, lp=lp, var_names=var_names)
    burnin, good_mask = diag["burnin"], diag["good_mask"]
    good_chains = np.nonzero(good_mask)[0]
    n_draws = int(post.sizes["draw"])
    rng = np.random.default_rng(rng_seed)
    draw_lo = min(burnin, max(0, n_draws - 1))

    # ---- mode-stratified path -------------------------------------------
    labels = None
    try:
        from exozippy.outputs.modes import identify_modes

        report = identify_modes(idata, attach=False)
        if report.n_modes > 1:
            labels = report.labels  # (chain, draw); -1 = invalid/unassigned
            weights = [m.weight for m in report.modes]
    except Exception as e:  # never let mode analysis break seed emission
        logger.warning(
            f"mkprior: mode identification failed ({e}); falling back to "
            f"unstratified seed draws."
        )

    if labels is not None:
        quotas = _mode_seed_quotas(weights, n)
        logger.info(
            "mkprior: multimodal trace -- stratifying %d seeds across %d "
            "modes (quotas %s, weights %s)",
            n,
            len(quotas),
            quotas.tolist(),
            [f"{w:.3f}" for w in weights],
        )
        pairs, seen = [], {tuple(exclude)}
        for m, quota in enumerate(quotas):
            cs, ds = np.nonzero(labels == m)
            post_burn = ds >= draw_lo
            if post_burn.sum() >= quota:  # prefer equilibrated draws
                cs, ds = cs[post_burn], ds[post_burn]
            idx = rng.permutation(len(cs))
            taken = 0
            for j in idx:
                if taken >= quota:
                    break
                pair = (int(cs[j]), int(ds[j]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
                    taken += 1
        return pairs, good_mask, burnin

    # ---- unimodal / fallback path ---------------------------------------
    pairs, seen = [], {tuple(exclude)}
    attempts = 0
    while len(pairs) < n and attempts < 50 * max(n, 1):
        attempts += 1
        c = int(rng.choice(good_chains))
        d = int(rng.integers(draw_lo, n_draws))
        if (c, d) not in seen:
            seen.add((c, d))
            pairs.append((c, d))
    return pairs, good_mask, burnin


def mkprior(
    config, base_dir=None, trace_path=None, output_path=None, n_seeds=None
):
    """
    Write a params.yaml seeded from a previous trace.

    With ``n_seeds == 1`` (default) every sampled parameter gets a scalar
    ``initval`` at the trace MAP. With ``n_seeds > 1`` the ``initval`` becomes
    a length-K list of mutually-consistent JOINT posterior draws (seed 0 = the
    MAP; seeds 1..K-1 = random post-burn-in draws from the good chains), which
    the next run consumes as P4 multi-seed starts so its walkers begin already
    spread across the posterior covariance (notes/todo.txt #3). Bounds stay
    scalar (from seed 0), matching config._build_seed_overrides.  No
    ``init_scale`` is written: whitening scales are measured from the data at
    startup and the key would be warn-ignored.

    Parameters
    ----------
    config : dict or str or Path
        Loaded config dict, or path to the config YAML file.
    base_dir : Path, optional
        Directory relative to which parameter_file and prefix are resolved.
        Defaults to config's parent when config is a path, else CWD.
    trace_path : str or Path, optional
        Trace file; defaults to ``<prefix>_trace.nc``.
    output_path : str or Path, optional
        Output file; defaults to ``<prefix>_mkprior.params.yaml``.
    n_seeds : int, optional
        Number of multi-seed start points to emit. When None, read from
        ``config['mkprior']['n_seeds']`` (default 1 = legacy scalar behavior).

    Returns
    -------
    Path
        The path of the written file.
    """
    if isinstance(config, (str, Path)):
        base_dir = Path(config).parent
        config = _load_yaml(str(config))
    else:
        base_dir = Path(base_dir or ".")

    if n_seeds is None:
        n_seeds = (config.get("mkprior") or {}).get("n_seeds", 1)
    n_seeds = max(1, int(n_seeds))

    prefix = config.get("prefix", "fitresults/model")
    run_name = Path(prefix).stem  # e.g. "KELT-4A" from "fitresults/KELT-4A"
    if trace_path is None:
        trace_path = base_dir / f"{prefix}_trace.nc"
    if output_path is None:
        param_file = config.get("parameter_file")
        if param_file:
            output_path = _next_versioned_path(base_dir / param_file)
        else:
            output_path = base_dir / f"{run_name}.params.2.yaml"

    param_file = config.get("parameter_file")
    existing_params = {}
    if param_file:
        param_path = base_dir / param_file
        if param_path.exists():
            existing_params = _load_yaml(str(param_path))
            # mkprior reads the params file directly, bypassing ConfigManager,
            # so the same check has to run here.  Every entry mkprior
            # synthesizes carries an initval, but the pass-through loop below
            # copies constraint-bearing entries verbatim -- a legacy
            # '{sigma: 0.5}' would be re-emitted into the restart file.  Fail
            # on the input: it is the actual source and the file to edit.
            validate_sigma_has_center(existing_params, source=str(param_path))

    idata = az.from_netcdf(str(trace_path))

    # The restart file this writes IS the next fit's start, so seeding it
    # from a trace sampled under a different model corrupts that fit
    # silently.  This load most often fires automatically at the end of a
    # run, where the trace was just written by the same System -- so the
    # mismatch case is genuinely exceptional and worth failing on rather
    # than papering over.  A trace with no fingerprint (written before this
    # metadata existed) only warns.
    #
    # The fingerprint is recomputed here from the same two inputs the output
    # file is built from -- this `config` and the parameter_file on disk --
    # rather than being handed down from the live System.  That is
    # deliberate and measured: across kelt4 (RV-only and rv+transit+sed),
    # ob08092 and ob140939, the config dict is mutated ZERO times by stages
    # 1-6, so the recomputation reproduces System's snapshot exactly.  (The
    # one mutation that does exist -- Mann/Torres deriving `name:` from
    # their `star:` key -- happens inside System.__init__, which is why the
    # snapshot is taken at the END of __init__.)  Should some future
    # component start writing into config during load_data, the raise here
    # would be CORRECT, not spurious: existing_params and config are what
    # this function merges the MAP into, so if they are not what was
    # fitted, the restart file is wrong no matter what the trace says.
    from .evaluator import structural_hash, structural_payload

    check_trace_freshness(
        idata,
        (
            structural_hash(config, existing_params),
            structural_payload(config, existing_params),
        ),
        trace_path,
    )

    # Find the MAP draw. lp is present for NUTS and for Metropolis traces saved
    # after the fix that persists it right after pm.sample(). Fall back to the
    # posterior median for old Metropolis trace files without lp.
    ss = idata.get("sample_stats")
    has_lp = ss is not None and "lp" in ss.data_vars
    if has_lp:
        lp = ss["lp"]
        flat_lp = lp.values.flatten()
        map_idx = int(np.argmax(flat_lp))
        n_draws = lp.sizes["draw"]
        map_chain = map_idx // n_draws
        map_draw = map_idx % n_draws
        map_lp = float(flat_lp[map_idx])
        logger.info(
            f"mkprior: MAP chain={map_chain} draw={map_draw} lp={map_lp:.4f}"
        )
    else:
        # No lp → use last draw of chain 0 as a self-consistent fallback.
        # Per-parameter medians would be inconsistent (the joint point may not
        # exist in the posterior); any real draw is always self-consistent.
        logger.warning(
            "mkprior: lp not in trace — using last draw of chain 0 as fallback"
        )
        map_chain, map_draw, map_lp = (
            0,
            idata.posterior.sizes["draw"] - 1,
            float("nan"),
        )

    posterior = idata["posterior"]
    # Only include physically sampled variables (those with a _raw counterpart).
    # Derived Deterministics (e.g. orbit.period from orbit.logP) must be excluded:
    # writing them to params.yaml creates redundant constraints that confuse the
    # relaxation engine.
    raw_var_names = {v[:-4] for v in posterior.data_vars if v.endswith("_raw")}
    sampled_vars = sorted(v for v in posterior.data_vars if v in raw_var_names)

    # Multi-seed (P4): seed 0 is the MAP; seeds 1..K-1 are random post-burn-in
    # draws from the good chains. All seeds are JOINT draws (a (chain, draw)
    # pair each), so reading every parameter at those indices yields K
    # mutually-consistent start points that span the posterior covariance.
    seed_pairs = [(map_chain, map_draw)]
    if n_seeds > 1:
        extra, _pool_mask, _pool_burnin = _sample_seed_draws(
            idata, n_seeds - 1, exclude=(map_chain, map_draw)
        )
        seed_pairs += extra
        if len(seed_pairs) < n_seeds:
            logger.warning(
                "mkprior: requested %d seeds but only %d distinct draws were "
                "available; emitting %d.",
                n_seeds,
                len(seed_pairs),
                len(seed_pairs),
            )
    K = len(seed_pairs)
    logger.info("mkprior: emitting %d seed(s) per parameter", K)

    output = {}
    consumed_existing = set()
    # out_key -> (comp_key, element index, instance name), so the direction-pair
    # -> angle pass below can look the angle's existing entry up the same way.
    key_context = {}

    for var_name in sampled_vars:
        comp_key, param = var_name.rsplit(".", 1)
        da = posterior[var_name]
        # (K, n_elements) joint values across the seed draws.
        seed_vals = np.stack(
            [np.atleast_1d(da.values[c, d]) for (c, d) in seed_pairs]
        )
        n_elements = seed_vals.shape[1]

        instance_names = _get_instance_names(config, comp_key)

        for i in range(n_elements):
            mv_list = [float(np.round(seed_vals[k, i], 8)) for k in range(K)]
            mv = mv_list[0]
            if instance_names:
                name = instance_names[i] if i < len(instance_names) else str(i)
                out_key = f"{comp_key}.{name}.{param}"
            elif n_elements == 1:
                # Component uses a flat-dict config (no named instances).
                # Write the 2-part key to match the trace variable name so the
                # next run can find the entry without hitting a name-lookup crash.
                name = None
                out_key = f"{comp_key}.{param}"
            else:
                # Multiple unnamed instances — fall back to numeric index.
                name = str(i)
                out_key = f"{comp_key}.{i}.{param}"

            existing_key, existing_entry = _find_existing(
                existing_params, comp_key, i, name, param
            )
            if existing_key:
                consumed_existing.add(existing_key)

            # Set initval to the seed value(s) so the next run starts there: a
            # scalar for single-seed, a length-K list for multi-seed (P4).
            # mu/sigma/bounds carry over from the existing entry unchanged.
            # No init_scale: whitening scales are measured from the data at
            # startup, and the key would be warn-ignored anyway.
            entry = _apply_existing_constraints(
                {"initval": mv if K == 1 else mv_list}, existing_entry
            )

            output[out_key] = entry
            key_context[out_key] = (comp_key, i, name)

    # Convert direction-vector pairs (x, y) → their angle (degrees).
    # These pairs (lens xalpha/yalpha, orbit xbigomega/ybigomega) are sampled
    # on wide bounds (±100) so that only the direction arctan2(y, x) matters;
    # their individual values are not meaningful cosine/sine values and must
    # not be written to params.yaml as-is.  The relaxation engine derives the
    # pair from the angle via cos/sin, so writing the angle is correct.
    for x_name, y_name, angle_name in [
        ("xalpha", "yalpha", "alpha"),
        ("xbigomega", "ybigomega", "bigomega"),
    ]:
        _x_keys = {
            k[: -len(f".{x_name}")]: k
            for k in list(output)
            if k.endswith(f".{x_name}")
        }
        _y_keys = {
            k[: -len(f".{y_name}")]: k
            for k in list(output)
            if k.endswith(f".{y_name}")
        }
        for prefix in set(_x_keys) & set(_y_keys):
            x_key, y_key = _x_keys[prefix], _y_keys[prefix]
            xv, yv = output[x_key]["initval"], output[y_key]["initval"]
            # initval may be a scalar (single-seed) or a length-K list
            # (multi-seed): convert every seed's (x, y) to its own angle.
            xv_list = xv if isinstance(xv, list) else [xv]
            yv_list = yv if isinstance(yv, list) else [yv]
            angles = [
                float(np.round(np.degrees(np.arctan2(y, x)), 8))
                for x, y in zip(xv_list, yv_list)
            ]
            del output[x_key]
            del output[y_key]
            angle_entry = {
                "initval": angles[0] if len(angles) == 1 else angles,
            }
            # The angle itself is never in ``sampled_vars`` (only the x/y pair
            # is), so its existing entry has NOT been consumed above.  Consume
            # it here and merge it: a user prior/bound on alpha (or bigomega)
            # survives, while the initval comes from the trace MAP.  Without
            # this the pass-through loop below would overwrite the fresh MAP
            # angle with the stale entry, breaking the restart contract.
            comp_key, idx, name = key_context.get(
                x_key, (prefix.split(".", 1)[0], 0, None)
            )
            existing_key, existing_entry = _find_existing(
                existing_params, comp_key, idx, name, angle_name
            )
            if existing_key:
                consumed_existing.add(existing_key)
            output[f"{prefix}.{angle_name}"] = _apply_existing_constraints(
                angle_entry, existing_entry
            )

    _CONSTRAINT_FIELDS = {"sigma", "upper", "lower"}

    # Pass through existing entries not touched by the trace only if they carry
    # a constraint (prior, bound, or fixed value).  Pure initval-only entries
    # on non-sampled parameters are stale guesses — discard them.
    # Normalize all keys to name notation (star.A.param) regardless of how the
    # existing file expressed them (star.0.param or star.param).
    for key, val in existing_params.items():
        if key not in consumed_existing:
            if isinstance(val, dict) and not (_CONSTRAINT_FIELDS & val.keys()):
                continue
            # For non-sampled constraint parameters (e.g. a Gaia parallax prior
            # applied as a potential on distance), promote initval→mu so the
            # prior center is explicit and cannot accidentally drift if initval
            # is ever edited.  Same logic as for sampled parameters above.
            if isinstance(val, dict):
                sigma = val.get("sigma")
                if (
                    sigma is not None
                    and float(sigma) != 0
                    and "mu" not in val
                    and "initval" in val
                ):
                    val = dict(val)
                    val["mu"] = val["initval"]
            output[_normalize_key(key, config)] = val

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(
            f"# Generated by mkprior from {Path(str(trace_path)).name}"
            f"  (MAP lp={map_lp:.4f})\n"
        )
        if K > 1:
            f.write(
                f"# Multi-seed: initval is a length-{K} list of joint posterior\n"
                f"# draws (seed 0 = MAP; 1..{K - 1} = random post-burn-in draws\n"
                f"# from the good chains). Bounds are scalar (seed 0).\n"
            )
        yaml.dump(output, f, default_flow_style=False, sort_keys=True)

    logger.info(f"mkprior: written {output_path}")
    return output_path
