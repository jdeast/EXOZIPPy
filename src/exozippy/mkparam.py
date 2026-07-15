"""mkparam - Seed a params.yaml from the MAP of a previous trace."""

import logging
import re
from pathlib import Path

import arviz as az
import numpy as np
import yaml

from exozippy.samplers import convergence

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
    stem = p.name[: p.name.rfind(suffix)]  # e.g. "foo.params" or "foo.params.2"
    m = re.search(r'^(.*?)\.(\d+)$', stem)
    if m:
        base, n = m.group(1), int(m.group(2))
    else:
        base, n = stem, 1
    return p.parent / f"{base}.{n + 1}{suffix}"


def _sample_seed_draws(idata, n, exclude, rng_seed=0):
    """Pick ``n`` random JOINT (chain, draw) index pairs for multi-seed starts.

    Draws are taken from the GOOD chains and the POST-BURN-IN region only
    (samplers.convergence.find_burnin drops the initial transient and any
    stuck chain), so every seed is a real point in the equilibrated posterior
    and the set spans the true covariance. Whole draws are returned (a chain
    and draw index), never per-parameter marginals, so a downstream consumer
    that reads all parameters at (chain, draw) gets one self-consistent point.

    Returns (pairs, good_mask, burnin) where ``good_mask`` and ``burnin`` also
    describe the pool used, so the caller can compute init_scale over exactly
    the same post-burn-in good draws.
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


def mkprior(config, base_dir=None, trace_path=None, output_path=None,
            n_seeds=None):
    """
    Write a params.yaml seeded from a previous trace.

    With ``n_seeds == 1`` (default) every sampled parameter gets a scalar
    ``initval`` at the trace MAP. With ``n_seeds > 1`` the ``initval`` becomes
    a length-K list of mutually-consistent JOINT posterior draws (seed 0 = the
    MAP; seeds 1..K-1 = random post-burn-in draws from the good chains), which
    the next run consumes as P4 multi-seed starts so its walkers begin already
    spread across the posterior covariance (notes/todo.txt #3). ``init_scale``
    and bounds stay scalar (from seed 0), matching config._build_seed_overrides.

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

    idata = az.from_netcdf(str(trace_path))

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
        logger.info(f"mkprior: MAP chain={map_chain} draw={map_draw} lp={map_lp:.4f}")
    else:
        # No lp → use last draw of chain 0 as a self-consistent fallback.
        # Per-parameter medians would be inconsistent (the joint point may not
        # exist in the posterior); any real draw is always self-consistent.
        logger.warning("mkprior: lp not in trace — using last draw of chain 0 as fallback")
        map_chain, map_draw, map_lp = 0, idata.posterior.sizes["draw"] - 1, float("nan")

    posterior = idata["posterior"]
    # Only include physically sampled variables (those with a _raw counterpart).
    # Derived Deterministics (e.g. orbit.period from orbit.logP) must be excluded:
    # writing them to params.yaml creates redundant constraints that confuse the
    # relaxation engine and lead to conflicting init_scale values.
    raw_var_names = {v[:-4] for v in posterior.data_vars if v.endswith("_raw")}
    sampled_vars = sorted(v for v in posterior.data_vars if v in raw_var_names)

    # Multi-seed (P4): seed 0 is the MAP; seeds 1..K-1 are random post-burn-in
    # draws from the good chains. All seeds are JOINT draws (a (chain, draw)
    # pair each), so reading every parameter at those indices yields K
    # mutually-consistent start points that span the posterior covariance.
    seed_pairs = [(map_chain, map_draw)]
    pool_mask, pool_burnin = None, 0
    if n_seeds > 1:
        extra, pool_mask, pool_burnin = _sample_seed_draws(
            idata, n_seeds - 1, exclude=(map_chain, map_draw))
        seed_pairs += extra
        if len(seed_pairs) < n_seeds:
            logger.warning(
                "mkprior: requested %d seeds but only %d distinct draws were "
                "available; emitting %d.", n_seeds, len(seed_pairs),
                len(seed_pairs))
    K = len(seed_pairs)
    logger.info("mkprior: emitting %d seed(s) per parameter", K)

    output = {}
    consumed_existing = set()

    for var_name in sampled_vars:
        comp_key, param = var_name.rsplit(".", 1)
        da = posterior[var_name]
        # init_scale is a scalar step hint from the equilibrated posterior: use
        # the post-burn-in good draws when multi-seed (the transient inflates
        # std), else all draws (legacy single-seed behavior, unchanged).
        if pool_mask is not None:
            pool = da.values[pool_mask][:, pool_burnin:].reshape(-1, *da.shape[2:])
        else:
            pool = da.values.reshape(-1, *da.shape[2:])
        std_vals = np.atleast_1d(np.std(pool, axis=0))
        # (K, n_elements) joint values across the seed draws.
        seed_vals = np.stack(
            [np.atleast_1d(da.values[c, d]) for (c, d) in seed_pairs])

        instance_names = _get_instance_names(config, comp_key)

        for i, sv in enumerate(std_vals):
            mv_list = [float(np.round(seed_vals[k, i], 8)) for k in range(K)]
            mv = mv_list[0]
            if instance_names:
                name = instance_names[i] if i < len(instance_names) else str(i)
                out_key = f"{comp_key}.{name}.{param}"
            elif len(std_vals) == 1:
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
            # Preserve mu/sigma/bounds from the existing entry unchanged — mu is
            # the prior center, not the starting point, so it must not move.
            entry = {
                "initval": mv if K == 1 else mv_list,
                "init_scale": float(np.round(sv, 8)),
            }
            if isinstance(existing_entry, dict):
                for prior_key in ("mu", "sigma", "lower", "upper"):
                    if prior_key in existing_entry:
                        entry[prior_key] = existing_entry[prior_key]
                # If the original had a Gaussian prior (sigma > 0) but no
                # explicit mu, the original initval was the prior center.
                # Promote it to mu so the prior doesn't shift as initval
                # moves to the MAP on successive mkparam runs.
                existing_sigma = existing_entry.get("sigma")
                if (existing_sigma is not None and float(existing_sigma) != 0
                        and "mu" not in existing_entry
                        and "initval" in existing_entry):
                    entry["mu"] = existing_entry["initval"]

            output[out_key] = entry

    # Convert direction-vector pairs (x, y) → their angle (degrees).
    # These pairs (lens xalpha/yalpha, orbit xbigomega/ybigomega) are sampled
    # on wide bounds (±100) so that only the direction arctan2(y, x) matters;
    # their individual values are not meaningful cosine/sine values and must
    # not be written to params.yaml as-is.  The relaxation engine derives the
    # pair from the angle via cos/sin, so writing the angle is correct.
    for x_name, y_name, angle_name in [("xalpha", "yalpha", "alpha"),
                                       ("xbigomega", "ybigomega", "bigomega")]:
        _x_keys = {k[:-len(f".{x_name}")]: k for k in list(output) if k.endswith(f".{x_name}")}
        _y_keys = {k[:-len(f".{y_name}")]: k for k in list(output) if k.endswith(f".{y_name}")}
        for prefix in set(_x_keys) & set(_y_keys):
            x_key, y_key = _x_keys[prefix], _y_keys[prefix]
            xv, yv = output[x_key]["initval"], output[y_key]["initval"]
            xs, ys = output[x_key].get("init_scale", 0.0), output[y_key].get("init_scale", 0.0)
            # initval may be a scalar (single-seed) or a length-K list
            # (multi-seed): convert every seed's (x, y) to its own angle.
            xv_list = xv if isinstance(xv, list) else [xv]
            yv_list = yv if isinstance(yv, list) else [yv]
            angles = [float(np.round(np.degrees(np.arctan2(y, x)), 8))
                      for x, y in zip(xv_list, yv_list)]
            # init_scale is scalar (seed 0), so propagate from the seed-0 (x, y).
            x0, y0, r_sq = xv_list[0], yv_list[0], xv_list[0]**2 + yv_list[0]**2
            sigma_deg = float(np.round(np.degrees(
                np.sqrt((y0 / r_sq)**2 * xs**2 + (x0 / r_sq)**2 * ys**2)
            ), 8)) if r_sq > 0 else 0.0
            del output[x_key]
            del output[y_key]
            output[f"{prefix}.{angle_name}"] = {
                "initval": angles[0] if len(angles) == 1 else angles,
                "init_scale": sigma_deg,
            }

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
                if (sigma is not None and float(sigma) != 0
                        and "mu" not in val
                        and "initval" in val):
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
                f"# from the good chains). init_scale/bounds are scalar (seed 0).\n")
        yaml.dump(output, f, default_flow_style=False, sort_keys=True)

    logger.info(f"mkprior: written {output_path}")
    return output_path
