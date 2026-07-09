"""mkparam - Seed a params.yaml from the MAP of a previous trace."""

import logging
import re
from pathlib import Path

import arviz as az
import numpy as np
import yaml

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


def mkprior(config, base_dir=None, trace_path=None, output_path=None):
    """
    Write a params.yaml seeded at the MAP of a previous trace.

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

    output = {}
    consumed_existing = set()

    for var_name in sampled_vars:
        comp_key, param = var_name.rsplit(".", 1)
        da = posterior[var_name]
        all_samples = da.values.reshape(-1, *da.shape[2:])
        map_vals = np.atleast_1d(da.values[map_chain, map_draw])
        std_vals = np.atleast_1d(np.std(all_samples, axis=0))

        instance_names = _get_instance_names(config, comp_key)

        for i, (mv, sv) in enumerate(zip(map_vals, std_vals)):
            if instance_names:
                name = instance_names[i] if i < len(instance_names) else str(i)
                out_key = f"{comp_key}.{name}.{param}"
            elif len(map_vals) == 1:
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

            # Always set initval to the MAP value so the next run starts there.
            # Preserve mu/sigma/bounds from the existing entry unchanged — mu is
            # the prior center, not the starting point, so it must not move.
            entry = {
                "initval": float(np.round(mv, 8)),
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
            angle_deg = float(np.degrees(np.arctan2(yv, xv)))
            r_sq = xv**2 + yv**2
            sigma_deg = float(np.degrees(
                np.sqrt((yv / r_sq)**2 * xs**2 + (xv / r_sq)**2 * ys**2)
            )) if r_sq > 0 else 0.0
            del output[x_key]
            del output[y_key]
            output[f"{prefix}.{angle_name}"] = {
                "initval": float(np.round(angle_deg, 8)),
                "init_scale": float(np.round(sigma_deg, 8)),
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
        yaml.dump(output, f, default_flow_style=False, sort_keys=True)

    logger.info(f"mkprior: written {output_path}")
    return output_path
