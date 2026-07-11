
import gc
import importlib
import time
import yaml
import numpy as np
import math
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#import pytensor
#pytensor.config.optimizer_excluding = "local_elemwise_fusion"
#pytensor.config.allow_gc = True
#pytensor.config.linker = "py"

import multiprocessing as mp
import pymc as pm
import arviz as az
from pymc.initial_point import make_initial_point_fn


# PyMC 5.25.1 bug fix: stats_dtypes_shapes declares scaling/lambda as scalar []
# but np.atleast_1d always produces a 1-D array, crashing the NDArray backend.
# Not currently used (PTDE replaced DEMetropolis), kept for future experiments.
def _fix_de_stats(astep_fn):
    def wrapper(self, q0):
        result, stats = astep_fn(self, q0)
        for s in stats:
            for key in ("scaling", "lambda"):
                if key in s and np.ndim(s[key]) > 0:
                    s[key] = float(np.ravel(s[key])[0])
        return result, stats
    return wrapper

class DEMetropolisZ(pm.DEMetropolisZ):
    astep = _fix_de_stats(pm.DEMetropolisZ.astep)

class DEMetropolis(pm.DEMetropolis):
    astep = _fix_de_stats(pm.DEMetropolis.astep)

# local imports
from .logger import setup_logging
from .mkparam import mkprior
from .outputs.latex import build_latex_output, build_csv_output
from .diagnostics import ModelAuditor
from exozippy.system import System
from exozippy.samplers.ptde import ptde_sample


import pytensor

logger = logging.getLogger(__name__)

# debugging imports
# import ipdb

KNOWN_SAMPLER_KEYS = {
    "init", "tune", "draws", "chains", "cores", "target_accept",
    "method", "n_temps", "T_max", "n_chains", "recompute_trace",
    "nthin", "check_curvatures", "profile", "min_ess", "max_rhat",
    "maxtime", "chain_method", "eval_timeout",
    "rung_thin_factor", "rung_thin_start", "collect_rung_timing",
}


def run_fit(config):
    """
    The main library entry point to run an orbital fit.
    """

    # 1. Prepare output directory
    prefix = Path(config.get("prefix", "fitresults/planet"))
    parent_dir = prefix.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(prefix, config.get("logger_level", "INFO"))

    # 2. Load the sampler settings (flat under sampler:)
    sampler_cfg = config.get("sampler", {})
    init          = sampler_cfg.get("init", "adapt_diag")
    tune          = int(sampler_cfg.get("tune", 2000))
    draws         = int(sampler_cfg.get("draws", 2000))
    chains        = int(sampler_cfg.get("chains", 4))
    _cores_raw    = sampler_cfg.get("cores", None)
    if _cores_raw is not None:
        cores = int(_cores_raw)
    else:
        _phys = mp.cpu_count()
        cores = max(1, min(int(_phys * 0.75), _phys - 1))
    target_accept   = sampler_cfg.get("target_accept", 0.9)
    method          = sampler_cfg.get("method", None)   # None → auto-select after system is built
    n_temps         = int(sampler_cfg.get("n_temps", 8))
    T_max           = float(sampler_cfg.get("T_max", 200.0))
    _n_chains_raw   = sampler_cfg.get("n_chains", None)
    n_chains        = int(_n_chains_raw) if _n_chains_raw is not None else None
    recompute_trace = sampler_cfg.get("recompute_trace", False)
    nthin           = int(sampler_cfg.get("nthin", 1))
    check_curvatures = sampler_cfg.get("check_curvatures", True)
    profile         = sampler_cfg.get("profile", False)
    _min_ess_raw    = sampler_cfg.get("min_ess", 1000)
    min_ess         = int(_min_ess_raw) if _min_ess_raw is not None else None
    _max_rhat_raw   = sampler_cfg.get("max_rhat", 1.01)
    max_rhat        = float(_max_rhat_raw) if _max_rhat_raw is not None else None
    _maxtime_raw    = sampler_cfg.get("maxtime", None)
    maxtime         = float(_maxtime_raw) if _maxtime_raw is not None else None
    _eval_timeout_raw = sampler_cfg.get("eval_timeout", None)
    eval_timeout    = float(_eval_timeout_raw) if _eval_timeout_raw is not None else None
    rung_thin_factor = int(sampler_cfg.get("rung_thin_factor", 1))
    _rung_thin_start_raw = sampler_cfg.get("rung_thin_start", None)
    rung_thin_start = int(_rung_thin_start_raw) if _rung_thin_start_raw is not None else None
    collect_rung_timing = bool(sampler_cfg.get("collect_rung_timing", False))
    if profile: pytensor.config.profile = True

    # Warn about unrecognized keys in the sampler block so they are never silently ignored.
    _unknown_sampler_keys = sorted(set(sampler_cfg) - KNOWN_SAMPLER_KEYS)
    if _unknown_sampler_keys:
        logger.warning(
            f"Unrecognized key(s) in the sampler block will be ignored: "
            f"{_unknown_sampler_keys}. "
            f"Did you mean 'method'? Valid sampler keys: {sorted(KNOWN_SAMPLER_KEYS)}"
        )

    # 3. Build the stellar system into a PyMC Graph
    system = System(config)
    system.prepare() # this triggers I/O
    model = system.build_model()

    # Aggregate sampler requirements from all active components.
    # Components advertise incompatible/recommended samplers via sampler_requirements();
    # run.py stays agnostic about which component imposes the constraint.
    _incompatible, _recommended, _reasons = set(), set(), []
    for comp in system.active_components.values():
        reqs = comp.sampler_requirements()
        _incompatible.update(reqs.get('incompatible', set()))
        if 'recommended' in reqs:
            _recommended.add(reqs['recommended'])
        if 'reason' in reqs:
            _reasons.append(reqs['reason'])

    if method is None:
        method = next(iter(_recommended)) if _recommended else "nuts"
    elif method.lower() in _incompatible:
        rec_str = next(iter(_recommended)) if _recommended else "ptde"
        reason_str = "; ".join(_reasons) if _reasons else "incompatible with this model"
        logger.warning(
            f"Sampler '{method}' cannot be used with this model ({reason_str}). "
            f"Set 'method: {rec_str}' in the sampler block."
        )
    method = method.lower()

    # 4. Sample
    # We use adapt_diag to start exactly at our estimated means
    with model:

        # 1. Get your starting dictionaries
        nuts_scales, phys_scales, phys_inits, transformed_inits = system.get_mcmc_init(model)
        inspect_start(model, system, transformed_inits, phys_inits, phys_scales, check_curvatures)

        # Build the raw starting point explicitly: 0 for logit params,
        # (initval - mu)/sigma for Gaussian-path params, so the physical
        # start is always our initval.
        raw_start = system.get_raw_start(model)

        # convert raw starting point to the internal starting point
        internal_start = system.get_internal_point(model, raw_start)

        # make all the component plots
        for comp in system.active_components.values():
            comp.plot(system, [internal_start], filename_prefix=str(prefix) + "_start")

        #### profiling ####
        if profile:
            func = model.logp_dlogp_function(profile=True)
            func.profile.summary()
            #ipdb.set_trace()
        ###################
        #ipdb.set_trace()

        trace_path = str(prefix) + "_trace.nc"
        if os.path.exists(trace_path) and not recompute_trace:
            # if we've already done the sampling and don't want to redo it, load it
            idata = az.from_netcdf(trace_path)
        else:
            # do the sampling and save the results
            nuts_scales = np.array(nuts_scales).flatten()
            if method in ("numpyro", "blackjax", "nutpie"):
                try:
                    importlib.import_module(method)
                except ImportError:
                    logger.warning(
                        f"{method} is not installed — falling back to PyMC NUTS. "
                        f"Install with: poetry install --extras jax"
                    )
                    method = "nuts"

            if method == "ptde":
                idata = ptde_sample(
                    model, system, draws, tune,
                    n_temps=n_temps,
                    T_max=T_max,
                    n_chains=n_chains,
                    cores=cores,
                    plot_prefix=str(prefix),
                    min_ess=min_ess,
                    max_rhat=max_rhat,
                    maxtime=maxtime,
                    eval_timeout=eval_timeout,
                    rung_thin_factor=rung_thin_factor,
                    rung_thin_start=rung_thin_start,
                    collect_rung_timing=collect_rung_timing,
                )
            elif method in ("numpyro", "blackjax"):
                import jax
                jax.config.update("jax_enable_x64", True)
                from pymc.sampling.jax import sample_jax_nuts
                chain_method = sampler_cfg.get("chain_method", "parallel")
                # jitter=False: the JAX samplers default to jittering each
                # chain by U(-1, 1) in raw (whitened) space, i.e. +/- one
                # init_scale per parameter.  We deliberately construct the
                # start from the relaxation-engine solution; when an
                # init_scale is much wider than the posterior (common for
                # conservative user scales), the jitter launches chains at
                # logp ~ -1e6 and the step size collapses to zero (100%
                # divergences).  Opt back in with 'jitter: true'.
                idata = sample_jax_nuts(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    initvals=internal_start,
                    jitter=sampler_cfg.get("jitter", False),
                    chain_method=chain_method,
                    nuts_sampler=method,
                )
            elif method == "nutpie":
                # nutpie ignores initvals; it uses init_mean: a flat float64
                # array in model.free_RVs order (raw/unconstrained space).
                nutpie_init_mean = np.concatenate([
                    np.asarray(raw_start[v.name], dtype=float).ravel()
                    for v in model.free_RVs
                ])
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    nuts_sampler="nutpie",
                    target_accept=target_accept,
                    nuts_sampler_kwargs={"init_mean": nutpie_init_mean},
                    cores=cores,
                    return_inferencedata=True,
                )
            else:
                nuts_callback = None
                if maxtime is not None:
                    _nuts_start = time.time()
                    def nuts_callback(trace, draw):
                        if time.time() - _nuts_start > maxtime:
                            logger.info(f"NUTS: wall-clock limit {maxtime:.0f}s reached")
                            raise KeyboardInterrupt
                step = pm.NUTS(target_accept=target_accept)
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    init=init,
                    step=step,
                    cores=cores,
                    return_inferencedata=True,
                    callback=nuts_callback,
                )
            if nthin > 1:
                idata = idata.sel(draw=slice(None, None, nthin))
            # Ensure lp is in sample_stats; compute and persist if missing.
            ss_vars = (list(idata.sample_stats.data_vars)
                       if hasattr(idata, "sample_stats") else [])
            if "lp" not in ss_vars:
                import xarray as xr
                lp_vals = _compute_lp_from_model(model, idata)
                if lp_vals is not None:
                    idata.sample_stats["lp"] = xr.DataArray(
                        lp_vals, dims=["chain", "draw"],
                        coords={"chain": idata.posterior.chain,
                                "draw": idata.posterior.draw})
            # Convert sampled variables to user-facing units before archiving.
            # This makes the trace file, trace plots, ArviZ summary, and
            # mkparam output all use the same units the user specified.
            _convert_posterior_to_user_units(idata, system.get_parameter_lookup())
            _sanitize_netcdf_attrs(idata)
            idata.to_netcdf(trace_path)

        # compute the loglikelihoods (super slow? I can't believe this can't be stored/recalled...
        #loglike = pm.compute_log_likelihood(idata)

    # populate the parameters with the posteriors
    system.distribute_posterior(idata)

    summary_path = Path(str(prefix) + "_summary.txt")
    summary_path.write_text(str(az.summary(idata)), encoding="utf-8")

    # make a corner plot of fitted parameters (similar to EXOFASTv2 covar plot)
    make_corner(model, idata, str(prefix) + "_corner.png")

    # Save a 1D trace plot (similar to EXOFASTv2 chain file)
    all_params = system.get_all_parameters()
    plot_vars = [p.label for p in all_params if p.label in idata["posterior"]]
    save_multipage_trace(idata, plot_vars, str(prefix) + "_trace_detailed.pdf",
                         model=model)

    # Pick the suspected troublemakers
    # List every tracked parameter in the posterior
    #available_vars = list(idata.posterior.data_vars)
    #print("All available variables:\n", available_vars)

    # Automatically filter for the ones we care about
    #vars_to_check = [v for v in available_vars if any(sub in v for sub in ['secosw', 'sesinw', 'ecc', 'omega', 'mass'])]
    #print("\nFiltered variables to plot:\n", vars_to_check)
    #az.plot_pair(
    #    idata,
    #    var_names=vars_to_check,
    #    kind='scatter',
    #    divergences=True,
    #    divergences_kwargs={'color': 'C3', 'alpha': 0.5, 'markersize': 5}  # C3 is usually red
    #)
    #plt.show()

    # Generate latex table and machine-readable CSV
    build_latex_output(system,
                       var_filename=str(prefix) + '_definitions.tex',
                       template_filename=str(prefix) + '_template.tex',
                       caption=r"Median and 68\% Confidence intervals for " + prefix.stem)
    build_csv_output(system, csv_filename=str(prefix) + '_results.csv')

    # Generate final plots
    draws = get_draws(idata, param_lookup=system.get_parameter_lookup())
    for comp in system.active_components.values():
        comp.plot(system, draws, filename_prefix=str(prefix) + "_mcmc")

    try:
        mkprior(config, trace_path=trace_path)
    except Exception:
        logger.exception("mkprior failed (non-fatal)")

def inspect_start(model, system, transformed_inits, phys_inits, phys_scales, calc_curvature=True):
    auditor = ModelAuditor(model, system, transformed_inits)
    param_logps, other_nodes = auditor.get_aggregated_logps()
    curvature_map = auditor.get_curvatures() if calc_curvature else {}
    unused_yaml = auditor.check_unused_yaml()

    # Dynamic Width Logic
    display_labels = [p.get_display_label(i) for p in auditor.all_params
                      for i in range(np.prod(p.shape).astype(int) if p.shape != () else 1)]
    max_label_len = max([len(l) for l in display_labels] + [len(k) for k in other_nodes.keys()] + [24])

    table_width = 127
    logger.info("-" * table_width)
    logger.info("--------           Starting points and penalties (Physical Space) with Sampler Curvature (Unity Space)                 --------")
    logger.info("--------           Ideal curvature=-1.0. Tune by changing init_scale = Scale/sqrt(abs(Curv)) in param.yaml             --------")
    logger.info("--------           The deviation from ideal primarily impacts tuning efficiency.                                       --------")
    logger.info("--------           It can easily tolerate factors of 10,000+ from ideal with a longer tuning phase.                    --------")
    logger.info("--------           However, the initial scale does impact the steepness of bounds on derived parameters.               --------")
    logger.info("--------           Scales that are too large will create softer bounds that might introduce real biases.               --------")
    logger.info("--------           Scales that are too small will create harder bounds that lead to divergences.                       --------")
    logger.info("--------           Log-Prob for parameters includes summed penalties from bounds and priors.                           --------")
    logger.info("-" * table_width)
    header = f"{'Parameter':>{max_label_len}} | {'Value':>15} | {'Scale':>10} | {'Units':>12} | {'Log-Prob':>10} | {'Unity Curv':>10} | Priors & Bounds (*=user) |"
    logger.info(header)
    logger.info("-" * table_width)

    flat_warnings = []

    # --- PART 1: CORE PARAMETERS ---
    for p in auditor.all_params:
        should_print = getattr(p, 'debug_print', None)
        if should_print is None:
            should_print = np.any(getattr(p, 'is_sampled', False))
            # Handle vectorized boolean flags
            if isinstance(should_print, np.ndarray):
                should_print = np.any(should_print)
        if not should_print:
            continue

        raw_v = p.initval
        raw_s = p.init_scale

        if raw_v is None:
            # 1. Try to get it from the expression's dependency-solved graph
            try:
                if p.label in auditor.system.config_manager.user_params:
                    user_val = auditor.system.config_manager.user_params[p.label].get("initval")
                    if user_val is not None:
                        # Convert to internal units so it matches expectations
                        raw_v = p.to_internal(user_val)
            except:
                pass

            # 2. Last resort: Eval the expression if it exists
            if raw_v is None and p.expression is not None:
                try:
                    # 'deps' often need to be resolved. This is a hacky but effective way
                    # to visualize the starting point of a deterministic.
                    raw_v = p.expression().eval() if hasattr(p.expression(), 'eval') else p.expression()
                except:
                    pass

        if raw_v is None:
            continue

        v_phys = np.atleast_1d(raw_v)
        s_phys = np.atleast_1d(raw_s)
        c_phys = np.atleast_1d(curvature_map.get(p.label, [np.nan] * len(v_phys)))

        user_flag = "*" if getattr(p, 'user_prior_modified', False) else ""

        for i in range(len(v_phys)):
            row_label = p.get_display_label(i)

            # Grab the solver's reconciled value if it exists ---
            if row_label in auditor.system.config_manager.user_params:
                resolved_data = auditor.system.config_manager.user_params[row_label]
                if "initval" in resolved_data:
                    # Fetch the conversion factor and apply it backwards
                    f = p._get_conversion_factors()
                    f_val = f[i] if np.size(f) > 1 else (f[0] if np.size(f) == 1 else f)
                    v_phys[i] = float(resolved_data["initval"]) / float(f_val)

            def safe_float(x):
                if x is None or (hasattr(x, 'size') and x.size == 0):
                    return np.nan
                if hasattr(x, 'eval'):
                    x = x.eval()
                    # Extract scalar from numpy arrays/scalars
                val = x.item() if hasattr(x, 'item') else x
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return np.nan

            raw_val = safe_float(v_phys[i])
            # Pass through component conversion
            internal_res = p.from_internal(raw_val)
            # FORCE extraction to a standard Python float
            val_out = float(internal_res.item()) if hasattr(internal_res, 'item') else float(internal_res)

            # Do the same for scale
            raw_scale = safe_float(s_phys[i])
            internal_scale = p.from_internal(raw_scale)
            scale_out = float(internal_scale.item()) if hasattr(internal_scale, 'item') else float(internal_scale)

            #val_out = float(p.from_internal(safe_float(v_phys[i])))
            #scale_out = float(p.from_internal(safe_float(s_phys[i])))

            # Float/Scientific formatting logic ---
            def smart_format(val, width):
                # 2. Print a clean N/A instead of 'nan'
                if np.isnan(val):
                    return f"{'N/A':>{width}}"
                if val == 0:
                    return f"{0.0:>{width}.6f}"

                abs_v = abs(val)
                # Use scientific notation if it's outside the "clean" range
                if abs_v < 1e-4 or abs_v > 1e6:
                    precision = max(0, width - 7)
                    return f"{val:>{width}.{precision}e}"

                # Otherwise, use standard fixed-point
                precision = max(0, width - 7)
                return f"{val:>{width}.{precision}f}"

            val_str = smart_format(val_out, width=15)
            scale_str = smart_format(scale_out, width=10)

            is_fixed = (p.sigma is not None and np.atleast_1d(p.sigma)[i] == 0) or p.expression is not None
            raw_c = c_phys[i]

            # Curvature Warning and Display Logic
            if is_fixed or not calc_curvature:
                c_str = "N/A"
            elif np.isnan(raw_c) or np.isinf(raw_c) or raw_c == 0.0:
                c_str = "NaN (WARN)"
                flat_warnings.append(f"{row_label} (NaN/Inf)")  # Tag it in the warning list below
            else:
                c_str = f"{raw_c:.5f}"
                if abs(raw_c) < 1e-4:
                    flat_warnings.append(row_label)

            prior_str = p.get_prior_str(i, latex=False)

            logger.info(
                f"{row_label:>{max_label_len}} | {val_str} | {scale_str} | {p.get_unit_str(i):>12} | {param_logps.get(p.label, 0.0):10.2f} | {c_str:>10} | {prior_str}{user_flag}")

    # --- 2. Potentials & Likelihoods ---
    for node, lp in other_nodes.items():
        # logit_uniform_prior nodes are constant log-volume factors (−log range) that
        # never change during sampling and add nothing informative to the table.
        if node.startswith("logit_uniform_prior"):
            continue

        clean_node = node.replace("up_bound.", "").replace("low_bound.", "").replace("prior.", "").replace(
            "user_prior.", "")
        parent = auditor.param_lookup.get(clean_node)
        is_bound = "low_bound" in node or "up_bound" in node

        # Bug fix: skip inactive bounds — lp≈0 means we're well within the bound.
        # They clutter the table without conveying useful information.
        if is_bound and abs(lp) < 1e-6:
            continue

        # Bug fix: for bound nodes mark * only when the user explicitly set the
        # prior/bounds (sigma, lower, upper), NOT merely because they set initval.
        # user_modified is True for any user touch; user_prior_modified requires
        # an explicit physics override (sigma, lower, upper, mu).
        if is_bound:
            is_user = parent and getattr(parent, 'user_prior_modified', False)
        else:
            is_user = (parent and parent.user_modified) or (clean_node in auditor.user_params)

        if abs(lp) > 1e-6 or is_user:
            p_info = "Likelihood/Det."

            if is_user:
                if "up_bound" in node and parent:
                    val = parent.upper[0] if parent.upper is not None else 'N/A'
                    p_info = f"< {val}"
                elif "low_bound" in node and parent:
                    val = parent.lower[0] if parent.lower is not None else 'N/A'
                    p_info = f"> {val}"
                elif parent:
                    p_info = parent.get_prior_str(latex=False)

            logger.info(
                f"{node:>{max_label_len}} | {'N/A':>15} | {'N/A':>10} | {'---':>12} | {lp:10.2f} | {'N/A':>10} | {p_info}{' *' if is_user else ''}")
    logger.info("-" * table_width)

    # --- 3. THE FATAL CHECK ---
    bad_params = {k: v for k, v in param_logps.items() if not np.isfinite(v)}
    bad_nodes = {k: v for k, v in other_nodes.items() if not np.isfinite(v)}

    # if we start at a bad spot, PyMC will draw randomly from the prior, which will never work
    # raise an error here
    if bad_params or bad_nodes:
        bad_list = "\n".join(f"  -> {k}: {v}" for k, v in {**bad_params, **bad_nodes}.items())
        logger.error(
            "!" * 40 + "\n"
            "Fatal error: the starting model returned an infinite/NaN penalty!\n"
            "The following nodes have Infinite or NaN Log-Probability:\n"
            f"{bad_list}\n"
            "Check your initial values against your bounds/priors!\n"
            + "!" * 40)
        raise ValueError("Initialization failed due to non-finite Log-Probability.")

    if flat_warnings:
        logger.warning(
            "?" * 60 + "\n"
            f"WARNING: No curvature detected for: {flat_warnings}. Check your bounds/initialization.\n"
            "Even a single unconstrained parameter will destroy HMC efficiency.\n"
            + "?" * 60)

    if unused_yaml:
        logger.warning(
            f"The following parameters in the parameter.yaml file did not match any model parameter "
            f"and were not applied: {unused_yaml}\n"
            "This can be safely ignored if intentional, but check for typos.")

def make_corner(model, idata, filename, max_samples=1000):
    import corner
    all_vars = list(idata["posterior"].data_vars)
    physical_vars = [v for v in all_vars if "_raw" not in v and "_interval" not in v]

    samples_list = []
    labels = []

    for v in physical_vars:
        arr = idata["posterior"][v].stack(sample=("chain", "draw")).values
        n_samples = arr.shape[-1]

        if arr.ndim == 1:
            samples_list.append(arr)
            labels.append(v)
        else:
            arr_flat = arr.reshape(-1, n_samples)
            n_elem = arr_flat.shape[0]
            if n_elem > 100:
                logger.warning(f"make_corner: {v} has {n_elem} elements — possible shape mismatch")
            for i in range(n_elem):
                samples_list.append(arr_flat[i])
                labels.append(f"{v}[{i}]")

    samples = np.array(samples_list).T

    # Thin to cap corner.corner() memory — 1000 samples is sufficient for visual quality.
    if samples.shape[0] > max_samples:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
        idx.sort()
        samples = samples[idx]

    minrank = 0.5 - math.erf(1.0 / math.sqrt(2)) / 2.0
    maxrank = 0.5 + math.erf(1.0 / math.sqrt(2)) / 2.0
    n_samples_total = samples.shape[0]

    try:
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[minrank, 0.5, maxrank],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        # PDF vector backend holds every scatter marker path in memory during
        # serialization → balloons to tens of GB for a 26×26 subplot figure.
        # PNG (Agg) rasterizes to a fixed pixel buffer bounded by DPI × size.
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Corner plot failed (sample size {n_samples_total}): {e}")

# Module-level globals for fork-based parallel lp evaluation.
# PyTensor compiled functions can't be pickled, so they're set here before
# forking; child processes inherit them via copy-on-write without IPC.
_LP_FN = None
_LP_POINT_MAP = None


def _lp_eval_chain(args):
    """Evaluate logp for every draw in one chain (runs in a forked child)."""
    chain_data, chain_idx, n_draws = args
    lp_chain = np.full(n_draws, np.nan)
    for d in range(n_draws):
        point = {_LP_POINT_MAP[tname]: np.atleast_1d(chain_data[tname][d])
                 for tname in chain_data}
        lp_chain[d] = float(_LP_FN(point))
    return chain_idx, lp_chain


def _compute_lp_from_model(model, idata):
    """Compute log posterior at each draw by evaluating the compiled model logp.

    Used when the sampler (Metropolis) doesn't write lp to sample_stats.
    Chains are processed in parallel via fork so the PyTensor compiled function
    is inherited without pickling (numpy chain data is all that's sent over IPC).
    Returns an (n_chains, n_draws) float64 array, or None on failure.
    """
    try:
        n_chains = idata.posterior.sizes["chain"]
        n_draws = idata.posterior.sizes["draw"]

        with model:
            logp_fn = model.compile_logp(jacobian=False)

        # In EXOZIPPy's non-centered parameterization, the free RVs ARE the raw
        # unconstrained variables (e.g. "star.logmass_raw"). ArviZ stores them in
        # the posterior under the same name. Do NOT append another "_raw" here.
        point_map = {}   # trace var name → logp_fn input name
        for rv in model.free_RVs:
            vv = model.rvs_to_values.get(rv)
            if vv is None:
                continue
            if rv.name in idata.posterior.data_vars:
                point_map[rv.name] = vv.name

        if not point_map:
            logger.warning("_compute_lp_from_model: no unconstrained vars found in trace")
            return None

        logger.info(f"Computing lp for {n_chains}×{n_draws} draws "
                    f"({len(point_map)} unconstrained vars)")

        # Extract per-chain numpy arrays (picklable; logp_fn is NOT pickled —
        # it's inherited by child processes via fork).
        chain_arrays = []
        for c in range(n_chains):
            chain_arrays.append({tname: idata.posterior[tname].values[c]
                                  for tname in point_map})

        # Set module-level globals so forked workers inherit them without pickling.
        global _LP_FN, _LP_POINT_MAP
        _LP_FN = logp_fn
        _LP_POINT_MAP = point_map

        n_workers = min(n_chains, mp.cpu_count())
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_lp_eval_chain, [(arr, c, n_draws)
                                                for c, arr in enumerate(chain_arrays)])

        lp_vals = np.full((n_chains, n_draws), np.nan)
        for chain_idx, chain_lp in results:
            lp_vals[chain_idx] = chain_lp
            logger.info(f"  chain {chain_idx}: lp range "
                        f"[{chain_lp.min():.1f}, {chain_lp.max():.1f}]")

        return lp_vals

    except Exception as e:
        logger.warning(f"Could not compute lp from model: {e}")
        return None


def _n_trace_rows(idata, var_names, group="posterior"):
    """Total rows az.plot_trace needs: 1 row per element (shape product) per var."""
    rows = 0
    dataset = idata[group]
    for v in var_names:
        shape = dataset[v].shape[2:]  # drop (chain, draw) dims
        rows += int(np.prod(shape)) if shape else 1
    return rows


def _chunk_by_rows(idata, var_names, rows_per_page):
    """Yield (chunk, n_rows) pairs sized so each page needs <= rows_per_page rows."""
    chunk, chunk_rows = [], 0
    for v in var_names:
        r = _n_trace_rows(idata, [v])
        if chunk_rows + r > rows_per_page and chunk:
            yield chunk, chunk_rows
            chunk, chunk_rows = [v], r
        else:
            chunk.append(v)
            chunk_rows += r
    if chunk:
        yield chunk, chunk_rows


def save_multipage_trace(idata, var_names, filename, rows_per_page=4,
                         max_samples=2000, model=None):
    n_chains, n_draws = idata.posterior.chain.size, idata.posterior.draw.size

    # Thin to cap matplotlib memory and render time
    total_samples = n_chains * n_draws
    if total_samples > max_samples:
        thin_factor = max(1, total_samples // max_samples)
        sl = slice(None, None, thin_factor)
        thin_kwargs = {"posterior": idata.posterior.isel(draw=sl)}
        if hasattr(idata, "sample_stats"):
            thin_kwargs["sample_stats"] = idata.sample_stats.isel(draw=sl)
        idata = az.from_dict(thin_kwargs)

    # lp is in sample_stats for NUTS traces and for Metropolis traces saved after
    # the fix that computes and persists it right after pm.sample().
    # Fall back to computing it for old trace files.
    ss_vars = list(idata.sample_stats.data_vars) if hasattr(idata, "sample_stats") else []
    if "lp" in ss_vars:
        lp_idata, lp_var = idata, "lp"
    elif model is not None:
        logger.info("lp not in trace — computing from model (old trace file)")
        import xarray as xr
        lp_vals = _compute_lp_from_model(model, idata)
        if lp_vals is not None:
            if not hasattr(idata, "sample_stats") or idata.sample_stats is None:
                idata.add_groups({"sample_stats": xr.Dataset()})
            idata.sample_stats["lp"] = xr.DataArray(
                lp_vals, dims=["chain", "draw"],
                coords={"chain": idata.posterior.chain,
                        "draw": idata.posterior.draw})
            lp_idata, lp_var = idata, "lp"
        else:
            lp_idata, lp_var = None, None
    else:
        lp_idata, lp_var = None, None

    with PdfPages(filename) as pdf:
        # lp gets its own first page when available — mixing two different
        # datasets (sample_stats + posterior) in a pre-allocated axes grid
        # caused ArviZ 0.19 to silently ignore the passed axes and render
        # into its own floating figure, leaving our fig blank.  Let ArviZ
        # own the figure and retrieve it from the returned axes instead.
        if lp_var and lp_idata is not None:
            fig_lp = _render_trace_page(lp_idata, [lp_var], n_rows=1,
                                        title="Trace Plots: log-posterior (lp)",
                                        group="sample_stats")
            pdf.savefig(fig_lp)
            plt.close(fig_lp)
            gc.collect()

        for page_num, (chunk, n_rows) in enumerate(
            _chunk_by_rows(idata, var_names, rows_per_page), start=1
        ):
            fig = _render_trace_page(idata, chunk, n_rows,
                                     title=f"Trace Plots: Page {page_num}")
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()


def _render_trace_page(idata, var_names, n_rows, title, group="posterior"):
    """One trace-plot page: dist column + trace column, one row per element.

    plot_trace_dist (not plot_trace) is the ArviZ 1.0 equivalent of the old
    dist + trace two-column layout; plain plot_trace now renders only the
    trace lines.  compact=False keeps one row per vector element, matching
    the rows_per_page pagination math.
    """
    pc = az.plot_trace_dist(idata, var_names=var_names, group=group,
                            compact=False,
                            figure_kwargs={"figsize": (12, 3 * n_rows)})
    fig = pc.viz["figure"].item()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def _sanitize_netcdf_attrs(idata):
    """Flatten dict-valued attrs to JSON strings so xarray can serialize to netCDF.

    nutpie stores rich metadata (dicts) in sample_stats attrs; netCDF only allows
    scalars/strings/arrays.
    """
    import json
    for group in idata.children:
        ds = getattr(idata, group, None)
        if ds is None or not hasattr(ds, "attrs"):
            continue
        for k, v in list(ds.attrs.items()):
            if isinstance(v, dict):
                ds.attrs[k] = json.dumps(v)


def _convert_posterior_to_user_units(idata, param_lookup):
    """Convert idata.posterior in-place from internal math units to user units.

    Each non-raw variable in the posterior whose Parameter has a non-trivial
    unit conversion is multiplied by the internal→user factor.  This is called
    once after sampling so that the saved trace, trace plots, ArviZ summary,
    and mkparam output are all in user-facing units (e.g. jupiterMass, m/s).
    """
    for var_name in list(idata.posterior.data_vars):
        if var_name.endswith('_raw') or var_name not in param_lookup:
            continue
        factor = np.squeeze(np.asarray(
            param_lookup[var_name]._get_conversion_factors(), dtype=float))
        if np.all(factor == 1.0):
            continue
        idata.posterior[var_name] = idata.posterior[var_name] * factor


def get_draws(idata, n_draws=50, param_lookup=None):
    """
    Extracts a random subset of draws from the posterior for plotting.

    The trace is stored in user units.  Component physics functions expect
    internal units, so each variable is divided by its conversion factor
    before being returned when ``param_lookup`` is provided.
    """
    # 1. Flatten chains/draws into a single 'sample' dimension
    post = az.extract(idata, combined=True, keep_dataset=True)

    total_available = post.sample.size
    n_to_extract = min(n_draws, total_available)

    # 2. Pick random indices
    indices = np.random.choice(total_available, size=n_to_extract, replace=False)

    draw_list = []
    for idx in indices:
        point = {}
        for var in post.data_vars:
            val = post[var].isel(sample=idx).values
            if param_lookup is not None and var in param_lookup and not var.endswith('_raw'):
                factor = np.squeeze(
                    np.asarray(param_lookup[var]._get_conversion_factors(), dtype=float))
                val = val / factor
            point[var] = val
        draw_list.append(point)

    return draw_list

def _initialize_internal_maps(self):
    """
    The Bridge Phase: Converts YAML indices into PyTensor variables
    so Step 3 can use them for vectorized math.
    """
    # 1. Planet Mappings (Planet -> Star and Planet -> Orbit)
    if "planets" in self.active_components:
        planet_comp = self.active_components["planets"]

        # Read 'star_ndx' from each planet entry in the YAML
        star_map_indices = np.array([
            p_cfg.get("star_ndx", 0) for p_cfg in planet_comp.config
        ])
        self.star_map = pt.as_tensor_variable(star_map_indices).astype("int32")
        # Also attach it to the component so it can use 'self.star_map'
        planet_comp.star_map = self.star_map

        # Read 'orbit_ndx' from each planet entry in the YAML
        orbit_map_indices = np.array([
            p_cfg.get("orbit_ndx", 0) for p_cfg in planet_comp.config
        ])
        self.orbit_map = pt.as_tensor_variable(orbit_map_indices).astype("int32")
        planet_comp.orbit_map = self.orbit_map

    # 2. RV Instrument Mapping (Observation -> Instrument Offset)
    # We find any component that looks like an RVInstrument
    for comp in self.active_components.values():
        if hasattr(comp, 'inst_map') and not isinstance(comp, (Star, Planet, Orbit)):
            # This handles the gamma/jitter slicing for RVs and Transits
            comp.inst_map_tensor = pt.as_tensor_variable(comp.inst_map).astype("int32")

def get_diagonal_curvature(model, point):
    import pytensor.gradient as ptg
    logp_node = model.logp()
    vars_to_check = model.value_vars
    curvatures = []

    free_vars = [var.name for var in model.value_vars]
    filtered_point = {k: point[k] for k in free_vars if k in point}

    n_vars = len(vars_to_check)
    logger.info(f"Computing sampler curvature for {n_vars} parameter group(s) "
                f"(this compiles one gradient graph per group and can take a while)...")
    t_start = time.time()

    for i, var in enumerate(vars_to_check):
        grad = ptg.grad(logp_node, var)

        try:
            curv = ptg.grad(grad.sum(), var)
            fn = model.compile_fn(curv, on_unused_input='ignore')
            val = np.atleast_1d(fn(filtered_point))
        except ValueError:
            # Some physics ops (e.g. exoplanet_core's limb-darkening solution-vector
            # op used by the transit component) only implement a first-order
            # pullback, so a second nested ptg.grad() through them raises
            # "Backpropagation is only supported for the solution vector".
            # Fall back to a central-difference estimate of the Hessian diagonal
            # built from the (working) first-order gradient function.
            logger.info(f"  [{i + 1}/{n_vars}] {var.name}: exact 2nd derivative "
                        f"unsupported by an op in the graph, falling back to "
                        f"finite differences")
            grad_fn = model.compile_fn(grad, on_unused_input='ignore')
            x0 = np.atleast_1d(np.asarray(filtered_point[var.name], dtype=float))
            orig_shape = np.asarray(filtered_point[var.name]).shape
            val = np.empty(x0.size)
            eps = 1e-5 * np.maximum(np.abs(x0), 1.0)
            for j in range(x0.size):
                xp, xm = x0.copy(), x0.copy()
                xp[j] += eps[j]
                xm[j] -= eps[j]
                pt_plus = dict(filtered_point, **{var.name: xp.reshape(orig_shape)})
                pt_minus = dict(filtered_point, **{var.name: xm.reshape(orig_shape)})
                gp = np.atleast_1d(grad_fn(pt_plus)).sum()
                gm = np.atleast_1d(grad_fn(pt_minus)).sum()
                val[j] = (gp - gm) / (2 * eps[j])

        curvatures.append(val)
        elapsed = time.time() - t_start
        logger.info(f"  [{i + 1}/{n_vars}] {var.name} done ({elapsed:.1f}s elapsed)")

    return np.concatenate(curvatures)
