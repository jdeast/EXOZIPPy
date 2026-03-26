
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
from .outputs.plot_step import plot_step
from .components.stellarsystem import StellarSystem
from .components.parameter import Parameter
from .components.component import Component

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
    init = pymc_cfg.get("adapt_diag"),
    tune = int(pymc_cfg.get("tune", 2000))
    draws = int(pymc_cfg.get("draws", 2000))
    chains = int(pymc_cfg.get("chains", 4))
    cores = int(pymc_cfg.get("cores", None))
    target_accept = pymc_cfg.get("target_accept", 0.9)
    recompute_trace = pymc_cfg.get("recompute_trace", False)
    profile = pymc_cfg.get("profile", False)
    if profile: pytensor.config.profile = True

    # 3. Build the stellar system into a PyMC Graph
    stellar_system = StellarSystem(config)
    model = stellar_system.build_model()

    # debugging?
    """
    logp_fn = model.compile_logp()

    # 2. Create the Chi-Square Shim
    def chi2func(point):
        #Translates PyMC Log-Probability to Chi-Square.
        #PyMC handles the transformations (log, etc.) automatically.
        try:
            # Chi^2 = -2 * ln(Likelihood * Prior)
            # We use the Log-Posterior so priors constrain the line-search
            lp = logp_fn(point)
            if np.isnan(lp) or np.isinf(lp):
                return 1e20  # "Soft" infinity for the line-search
            return -2.0 * lp
        except Exception:
            return 1e20

    # 3. Initialize from the 'Heuristic' point
    # model.initial_point() returns a dict of the 'Transformed' values
    bestpars = model.initial_point()
    """

    user_init = {p.label: p.initval for p in stellar_system.get_all_parameters()
                 if p.expression is None and p.initval is not None}

    # 4. Sample
    # We use adapt_diag to start exactly at our heuristic means
    with model:

        # 1. Get your starting dictionaries
        ordered_scales, ordered_inits, transformed_inits = stellar_system.get_mcmc_init(model)

        # 2. Map the scales and parameters back to their names
        param_lookup = stellar_system.get_parameter_lookup()

        # 3. map scales to random variable names
        scale_dict = {rv.name: scale for rv, scale in zip(model.free_RVs, ordered_scales)}

        print("\n--- These are the starting points, scales, and corresponding penalties of the model ---")
        print("--- If the log-Prob is big, you may need to revisit its initialization in the parameter.yaml file ---\n")

        logps = model.point_logps(transformed_inits)

        header = f"{'Parameter':>25} | {'Value':>15} | {'Scale':>12} | {'Units':>12} | {'Log-Prob':>10} "
        print(header)
        print("-" * len(header))

        bad_nodes = {}

        for name, logp_val in logps.items():
            if np.isinf(logp_val) or np.isnan(logp_val):
                bad_nodes[name] = logp_val

            phys_vals = ordered_inits.get(name)
            scale_vals = scale_dict.get(name)
            param_obj = param_lookup.get(name)

            phys_vals = np.atleast_1d(phys_vals) if phys_vals is not None else None
            scale_vals = np.atleast_1d(scale_vals) if scale_vals is not None else None

            if phys_vals is not None and param_obj is not None:
                # If it's a true vector (e.g., N planets), loop through them
                if len(phys_vals) > 1:
                    for i in range(len(phys_vals)):
                        v_raw = phys_vals[i]
                        s_raw = scale_vals[i]

                        user_val = float(param_obj.from_internal(v_raw))
                        user_scale = float(param_obj.from_internal(s_raw))

                        u = param_obj.unit[i] if i < len(param_obj.unit) else param_obj.unit[0]
                        unit_str = u.to_string() if u.to_string() != 'dimensionless' else ""

                        row_name = f"{name}[{i}]"
                        print(
                            f"{row_name:>25} | {user_val:15.5f} | {user_scale:12.5f} | {unit_str:>12} | {logp_val:10.2f}")

                # If it's a scalar (or length 1), just print it
                else:
                    v_raw = phys_vals[0]
                    s_raw = scale_vals[0]

                    user_val = float(param_obj.from_internal(v_raw))
                    user_scale = float(param_obj.from_internal(s_raw))

                    u = param_obj.unit[0] if isinstance(param_obj.unit, list) else param_obj.unit
                    unit_str = u.to_string() if u.to_string() != 'dimensionless' else ""

                    print(f"{name:>25} | {user_val:15.5f} | {user_scale:12.5f} | {unit_str:>12} | {logp_val:10.2f}")

            else:
                # Derived/Obs/Potentials
                print(f"{name:>25} | {'Derived/Obs':>15} | {'N/A':>12} | {'---':>12} | {logp_val:10.2f}")
        print("-" * len(header))

        # if the starting model is bad, warn the user and stop
        if bad_nodes:
            raise ValueError(f"FATAL ERROR: The starting model is bad. "
                             f"Revise the parameter.yaml file.\nBad Nodes: {bad_nodes}")

        raw_start = pm.Point(transformed_inits, model=model)
        start_point = stellar_system.get_physical_point(model, raw_start)
        #start_point.update(ordered_inits)
        stellar_system.instruments.plot_model(stellar_system, stellar_system.planets, [start_point], filename_prefix=str(prefix) + "_start")

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
            ordered_scales = np.array(ordered_scales).flatten()
            step = pm.NUTS(scaling=ordered_scales, target_accept=target_accept)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                init=init,
                step=step,
                cores=cores,
                initvals=ordered_inits,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": False}
            )
            az.to_netcdf(idata, trace_path)

        # compute the loglikelihoods (super slow? I can't believe this can't be stored/recalled...
        #loglike = pm.compute_log_likelihood(idata)

    # populate the parameters with the posteriors
    stellar_system.distribute_posterior(idata)
    summary_path = Path(str(prefix) + "_summary.txt")
    summary_path.write_text(str(az.summary(idata)), encoding="utf-8")

    # Generate the corner plot
    import corner
    var_names = [v.name for v in model.free_RVs]
    data = az.extract(idata, var_names=var_names)

    minrank = 0.5-math.erf(1.0/math.sqrt(2))/2.0
    maxrank = 0.5+math.erf(1.0/math.sqrt(2))/2.0
    samples = np.array([data[v].values.flatten() for v in data.data_vars]).T
    n_samples = samples.shape[0]
    plot_contours = n_samples > 100

    try:
        fig = corner.corner(
            data,
            labels=var_names,
            quantiles=[minrank, 0.5, maxrank],
            show_titles=True,
            plot_contours=plot_contours,
            plot_density=plot_contours,
            title_kwargs={"fontsize": 12}
        )
        fig.savefig(str(prefix) + "_corner.pdf")
    except Exception as e:
        print(f"Warning: Corner plot failed (Sample size {n_samples}). Error: {e}")

    # Save a 1D trace plot (similar to EXOFASTv2 chain file
    save_multipage_trace(idata, var_names, str(prefix) + "_trace_detailed.pdf")

    #az.plot_trace(idata, var_names=var_names)
    #plt.savefig(str(prefix) + "_trace.pdf")
    #plt.close()

    # Generate latex table
    build_latex_output(stellar_system,
                       var_filename=str(prefix) + '_definitions.tex',
                       template_filename=str(prefix) + '_template.tex',
                       caption=r"Median and 68\% Confidence intervals for " + prefix.stem)

    # Generate final plots
    draws = get_draws(idata)
    stellar_system.instruments.plot_model(stellar_system, stellar_system.planets, draws, filename_prefix=str(prefix) + "_mcmc")

def save_multipage_trace(idata, var_names, filename, params_per_page=4):
    with PdfPages(filename) as pdf:
        # 1. Break the variable list into chunks of 4
        for i in range(0, len(var_names), params_per_page):
            chunk = var_names[i: i + params_per_page]

            # 2. Plot just this chunk
            # 'compact=False' gives each chain its own line (better for spotting drifts)
            axes = az.plot_trace(idata, var_names=chunk, compact=False)

            # 3. Clean up the formatting for readability
            fig = plt.gcf()
            fig.suptitle(f"Trace Plots: Page {i // params_per_page + 1}", fontsize=14)

            # Adjust layout so titles don't overlap axes
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 4. Save this specific figure as a new page in the PDF
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