
import yaml
import numpy as np
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pytensor
#pytensor.config.optimizer_excluding = "local_elemwise_fusion"
#pytensor.config.allow_gc = True
#pytensor.config.linker = "py"

import pymc as pm
import arviz as az
from pymc.initial_point import make_initial_point_fn

# local imports
from .outputs.latex import build_latex_output
from .diagnostics import ModelAuditor
from exozippy.system import System

# debugging imports
import ipdb

def run_fit(config):
    """
    The main library entry point to run an orbital fit.
    """

    # 1. Prepare output directory
    prefix = Path(config.get("prefix", "fitresults/planet"))
    parent_dir = prefix.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the sampler settings
    sampler_cfg = config.get("sampler", {})
    pymc_cfg = sampler_cfg.get("pymc", {})
    init = pymc_cfg.get("adapt_diag")
    tune = int(pymc_cfg.get("tune", 2000))
    draws = int(pymc_cfg.get("draws", 2000))
    chains = int(pymc_cfg.get("chains", 4))
    cores = int(pymc_cfg.get("cores", None))
    target_accept = pymc_cfg.get("target_accept", 0.9)
    recompute_trace = pymc_cfg.get("recompute_trace", False)
    check_curvatures = pymc_cfg.get("check_curvatures", True)
    profile = pymc_cfg.get("profile", False)
    if profile: pytensor.config.profile = True

    # 3. Build the stellar system into a PyMC Graph
    system = System(config)
    model = system.build_model()

    # 4. Sample
    # We use adapt_diag to start exactly at our heuristic means
    with model:

        # 1. Get your starting dictionaries
        nuts_scales, phys_scales, phys_inits, transformed_inits = system.get_mcmc_init(model)
        inspect_start(model, system, transformed_inits, phys_inits, phys_scales, check_curvatures)

        # this makes random draws within our 1000 sigma range
        raw_start = model.initial_point()

        # nuke them and start at our actual starting points
        for key in raw_start:
            raw_start[key] = np.zeros_like(raw_start[key])

        # convert raw starting point to the internal starting point
        internal_start = system.get_internal_point(model, raw_start)

        # make all the component plots
        for comp in system.active_components.values():
            comp.plot(system, [internal_start], filename_prefix=str(prefix) + "_start")

        #### profiling ####
        if profile:
            func = model.logp_dlogp_function(profile=True)
            func.profile.summary()
            ipdb.set_trace()
        ###################
        #ipdb.set_trace()

        trace_path = str(prefix) + "_trace.nc"
        if os.path.exists(trace_path) and not recompute_trace:
            # if we've already done the sampling and don't want to redo it, load it
            idata = az.from_netcdf(trace_path)
        else:
            # do the sampling and save the results
            nuts_scales = np.array(nuts_scales).flatten()
            step = pm.NUTS(target_accept=target_accept)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                init=init,
                step=step,
                cores=cores,
                #initvals=ordered_inits,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": False}
            )
            az.to_netcdf(idata, trace_path)

        # compute the loglikelihoods (super slow? I can't believe this can't be stored/recalled...
        #loglike = pm.compute_log_likelihood(idata)

    # populate the parameters with the posteriors
    system.distribute_posterior(idata)
    summary_path = Path(str(prefix) + "_summary.txt")
    summary_path.write_text(str(az.summary(idata)), encoding="utf-8")

    # make a corner plot of fitted parameters (similar to EXOFASTv2 covar plot)
    make_corner(model, idata, str(prefix) + "_corner.pdf")

    # Save a 1D trace plot (similar to EXOFASTv2 chain file)
    # Create a list of physical parameter labels that actually have traces
    all_params = system.get_all_parameters()
    plot_vars = [p.label for p in all_params if p.label in idata.posterior]
    save_multipage_trace(idata, plot_vars, str(prefix) + "_trace_detailed.pdf")

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

    # Generate latex table
    build_latex_output(system,
                       var_filename=str(prefix) + '_definitions.tex',
                       template_filename=str(prefix) + '_template.tex',
                       caption=r"Median and 68\% Confidence intervals for " + prefix.stem)

    # Generate final plots
    draws = get_draws(idata)
    for comp in system.active_components.values():
        comp.plot(system, draws, filename_prefix=str(prefix) + "_mcmc")

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
    print("-" * table_width )
    print("--------           Starting points and penalties (Physical Space) with Sampler Curvature (Unity Space)                 --------")
    print("--------           Ideal curvature=-1.0. Tune by changing init_scale = Scale/sqrt(abs(Curv)) in param.yaml             --------")
    print("--------           abs(Curv) >~ 1e4 will impact efficiency and require more tuning steps                               --------")
    print("--------           abs(Curv) ~< 1e-4 may artificially truncate your posteriors or severely impact efficiency           --------")
    print("--------           Log-Prob for parameters includes summed penalties from bounds and priors.                           --------")
    print("-" * table_width )
    header = f"{'Parameter':>{max_label_len}} | {'Value':>15} | {'Scale':>10} | {'Units':>12} | {'Log-Prob':>10} | {'Unity Curv':>10} | Priors & Bounds (*=user) |"
    print(header)
    print("-" * table_width )

    flat_warnings = []

    # --- PART 1: CORE PARAMETERS ---
    for p in auditor.all_params:
        raw_v = phys_inits.get(p.label)
        raw_s = phys_scales.get(p.label)

        if raw_v is None or raw_s is None:
            continue

        v_phys = np.atleast_1d(raw_v)
        s_phys = np.atleast_1d(raw_s)
        c_phys = np.atleast_1d(curvature_map.get(p.label, [np.nan] * len(v_phys)))

        user_flag = "*" if getattr(p, 'user_prior_modified', False) else ""

        for i in range(len(v_phys)):
            row_label = p.get_display_label(i)

            def safe_float(x):
                if hasattr(x, 'eval'): return float(x.eval())
                return float(x)

            val_out = float(p.from_internal(safe_float(v_phys[i])))
            scale_out = float(p.from_internal(safe_float(s_phys[i])))

            is_fixed = (p.sigma is not None and np.atleast_1d(p.sigma)[i] == 0) or p.expression is not None
            raw_c = c_phys[i]

            # Curvature Warning and Display Logic
            if is_fixed:
                c_str = "N/A"
            elif np.isnan(raw_c) or np.isinf(raw_c) or raw_c == 0.0:
                c_str = "NaN (WARN)"
                flat_warnings.append(f"{row_label} (NaN/Inf)")  # Tag it in the warning list below
            else:
                c_str = f"{raw_c:.5f}"
                if abs(raw_c) < 1e-4:
                    flat_warnings.append(row_label)

            prior_str = p.get_prior_str(i, latex=False)

            print(
                f"{row_label:>{max_label_len}} | {val_out:15.6f} | {scale_out:10.5f} | {p.get_unit_str(i):>12} | {param_logps.get(p.label, 0.0):10.2f} | {c_str:>10} | {prior_str}{user_flag}")

    # --- 2. Potentials & Likelihoods ---
    for node, lp in other_nodes.items():
        clean_node = node.replace("up_bound.", "").replace("low_bound.", "").replace("prior.", "").replace(
            "user_prior.", "")
        parent = auditor.param_lookup.get(clean_node)

        # Flag if the user touched this parameter in the YAML
        is_user = (parent and parent.user_modified) or (clean_node in auditor.user_params)

        # SHOW CONDITION: User requested it OR it's actively penalizing the model
        if abs(lp) > 1e-6 or is_user:
            p_info = "Likelihood/Det."

            if is_user:
                if "up_bound" in node and parent:
                    # Safely grab the parsed upper bound array
                    val = parent.upper[0] if parent.upper is not None else 'N/A'
                    p_info = f"< {val}"
                elif "low_bound" in node and parent:
                    val = parent.lower[0] if parent.lower is not None else 'N/A'
                    p_info = f"> {val}"
                elif parent:
                    p_info = parent.get_prior_str(latex=False)

            print(
                f"{node:>{max_label_len}} | {'N/A':>15} | {'N/A':>10} | {'---':>12} | {lp:10.2f} | {'N/A':>10} | {p_info}{' *' if is_user else ''}")
    print("-" * table_width )

    # --- 3. THE FATAL CHECK ---
    bad_params = {k: v for k, v in param_logps.items() if not np.isfinite(v)}
    bad_nodes = {k: v for k, v in other_nodes.items() if not np.isfinite(v)}

    # if we start at a bad spot, PyMC will draw randomly from the prior, which will never work
    # raise an error here
    if bad_params or bad_nodes:
        print("\n\033[91m" + "!" * 40)
        print("Fatal error: the starting model returned an infinite/NaN penalty!")
        print("The following nodes have Infinite or NaN Log-Probability:")
        for k, v in {**bad_params, **bad_nodes}.items():
            print(f"  -> {k}: {v}")
        print("Check your initial values against your bounds/priors!")
        print("!" * 40 + "\033[0m\n")
        raise ValueError("Initialization failed due to non-finite Log-Probability.")

    if flat_warnings:
        print("\n\033[93m" + "?" * 60)
        print(f"WARNING: No curvature detected for: {flat_warnings}. Check your bounds/initialization.")
        print("Even a single unconstrained parameter will destroy HMC efficiency.")
        print("?" * 60 + "\033[0m\n")

    if unused_yaml:
        print(f"\n!!!! Warning: the following parameters in the parameter.yaml file did not match a parameter in the model and its entries were not applied: {unused_yaml}")
        print(f"\n!!!!          This warning can be safely ignored if that was intentional, but check for typos.")

def make_corner(model, idata, filename):
    import corner
    physical_vars = [v for v in idata.posterior.data_vars if "_raw" not in v and "_interval" not in v]

    samples_list = []
    labels = []

    for v in physical_vars:
        # Stack chains and draws into a single 'sample' dimension at the end
        arr = idata.posterior[v].stack(sample=("chain", "draw")).values
        n_samples = arr.shape[-1]

        # If it's a scalar parameter (1D array of samples)
        if arr.ndim == 1:
            samples_list.append(arr)
            labels.append(v)
        # If it's a vector parameter (e.g., 2 instruments -> 2D array)
        else:
            arr_flat = arr.reshape(-1, n_samples)
            for i in range(arr_flat.shape[0]):
                samples_list.append(arr_flat[i])
                labels.append(f"{v}[{i}]")

    samples = np.array(samples_list).T

    minrank = 0.5 - math.erf(1.0 / math.sqrt(2)) / 2.0
    maxrank = 0.5 + math.erf(1.0 / math.sqrt(2)) / 2.0
    n_samples_total = samples.shape[0]
    plot_contours = n_samples_total > 100

    try:
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[minrank, 0.5, maxrank],
            show_titles=True,
            plot_contours=plot_contours,
            plot_density=plot_contours,
            title_kwargs={"fontsize": 12}
        )
        fig.savefig(filename)
    except Exception as e:
        print(f"Warning: Corner plot failed (Sample size {n_samples_total}). Error: {e}")

def save_multipage_trace(idata, var_names, filename, params_per_page=4):
    with PdfPages(filename) as pdf:
        # --- PAGE 1 (Special: LP + first 3 params) ---
        num_on_first = params_per_page - 1
        first_page_vars = var_names[:num_on_first]

        # Create a figure with enough rows for 4 parameters (8 subplots)
        fig, axes = plt.subplots(params_per_page, 2, figsize=(12, 3 * params_per_page))

        # 1. Plot LP into the first row
        az.plot_trace(idata.sample_stats, var_names=["lp"], axes=axes[0:1, :])

        # 2. Plot the parameters into the remaining rows
        if first_page_vars:
            az.plot_trace(idata, var_names=first_page_vars, axes=axes[1:params_per_page, :])

        fig.suptitle("Trace Plots: Page 1 (Likelihood & Parameters)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # --- PAGES 2+ (Standard Loop) ---
        for i in range(num_on_first, len(var_names), params_per_page):
            chunk = var_names[i: i + params_per_page]
            n_rows = len(chunk)

            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
            # Ensure axes is 2D even if there's only 1 row
            if n_rows == 1: axes = axes[np.newaxis, :]

            az.plot_trace(idata, var_names=chunk, axes=axes)

            page_num = (i - num_on_first) // params_per_page + 2
            fig.suptitle(f"Trace Plots: Page {page_num}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)
            plt.close(fig)

def get_draws(idata, n_draws=50):
    """
    Extracts a random subset of draws from the posterior for plotting.
    """
    # 1. Flatten chains/draws into a single 'sample' dimension
    # This gives us a dataset where every variable has a 'sample' axis
    post = az.extract(idata, combined=True)

    total_available = post.sample.size
    n_to_extract = min(n_draws, total_available)

    # 2. Pick random indices
    indices = np.random.choice(total_available, size=n_to_extract, replace=False)

    draw_list = []
    for idx in indices:
        # Create a point dictionary for this specific draw
        # .isel(sample=idx) selects the values at that index across all variables
        point = {var: post[var].isel(sample=idx).values for var in post.data_vars}
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

    for var in vars_to_check:
        grad = ptg.grad(logp_node, var)
        curv = ptg.grad(grad.sum(), var)

        # Compile the function
        fn = model.compile_fn(curv, on_unused_input='ignore')

        # fn.f.maker.inputs contains the expected variable objects
        #expected_names = [v.name for v in fn.f.maker.inputs]
        #filtered_point = {k: v for k, v in point.items() if k in expected_names}

        free_vars = [var.name for var in model.value_vars]
        filtered_point = {k: point[k] for k in free_vars if k in point}

        # Pass the filtered dictionary as a single positional argument
        val = np.atleast_1d(fn(filtered_point))
        curvatures.append(val)

    return np.concatenate(curvatures)
