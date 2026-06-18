"""mkparam - Seed a params.yaml from the MAP of a previous trace."""

import logging
import shutil
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


def backup_params(config, base_dir=None):
    """
    Copy the parameter file to a versioned backup (params.2.yaml, params.3.yaml, …).

    Returns the backup Path, or None if no parameter file is configured.
    """
    if isinstance(config, (str, Path)):
        base_dir = Path(config).parent
        config = _load_yaml(str(config))
    else:
        base_dir = Path(base_dir or ".")

    param_file = config.get("parameter_file")
    if not param_file:
        return None
    param_path = base_dir / param_file
    if not param_path.exists():
        return None

    stem = param_path.stem    # e.g. "ob140939.params"
    suffix = param_path.suffix  # ".yaml"
    n = 2
    while True:
        backup = param_path.parent / f"{stem}.{n}{suffix}"
        if not backup.exists():
            break
        n += 1

    shutil.copy2(param_path, backup)
    logger.info(f"mkprior: params backup → {backup}")
    return backup


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
            p = base_dir / param_file
            param_dir = p.parent
            # Strip the trailing ".yaml"/".yml" to get the base stem
            # e.g. "kelt4.params.yaml" → "kelt4.params"
            base_stem = p.name[: p.name.rfind(".")]
        else:
            param_dir = base_dir
            base_stem = f"{run_name}.params"
        n = 2
        while True:
            candidate = param_dir / f"{base_stem}.{n}.yaml"
            if not candidate.exists():
                break
            n += 1
        output_path = candidate

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
            name = instance_names[i] if i < len(instance_names) else str(i)
            out_key = f"{comp_key}.{name}.{param}"

            existing_key, existing_entry = _find_existing(
                existing_params, comp_key, i, name, param
            )
            if existing_key:
                consumed_existing.add(existing_key)

            has_prior = isinstance(existing_entry, dict) and (
                "mu" in existing_entry or "sigma" in existing_entry
            )
            if has_prior:
                entry = {
                    "mu": float(np.round(mv, 8)),
                    "init_scale": float(np.round(sv, 8)),
                }
                if "sigma" in existing_entry:
                    entry["sigma"] = existing_entry["sigma"]
            else:
                entry = {
                    "initval": float(np.round(mv, 8)),
                    "init_scale": float(np.round(sv, 8)),
                }

            output[out_key] = entry

    _CONSTRAINT_FIELDS = {"sigma", "upper", "lower"}

    # Pass through existing entries not touched by the trace only if they carry
    # a constraint (prior, bound, or fixed value).  Pure initval-only entries
    # on non-sampled parameters are stale guesses — discard them.
    for key, val in existing_params.items():
        if key not in consumed_existing:
            if isinstance(val, dict) and not (_CONSTRAINT_FIELDS & val.keys()):
                continue
            output[key] = val

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(
            f"# Generated by mkprior from {Path(str(trace_path)).name}"
            f"  (MAP lp={map_lp:.4f})\n"
        )
        yaml.dump(output, f, default_flow_style=False, sort_keys=True)

    logger.info(f"mkprior: written {output_path}")
    return output_path
