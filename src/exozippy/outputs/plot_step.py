import matplotlib.pyplot as plt
import arviz as az
import numpy as np


def get_map_indices(idata):
    """Finds the draw that best fits the DATA + PRIORS."""
    if hasattr(idata, "sample_stats") and "lp" in idata.sample_stats:
        lp = idata.sample_stats.lp
        idx = lp.argmax(dim=["chain", "draw"])
        return int(idx["chain"]), int(idx["draw"])
    return 0, 0

def plot_step(idata, stellar_system, filename, ndraws=50, show_map=True):
    """
    OO-aware plotting:
    Relies on the RVInstrument to provide its own compiled physics.
    """
    # 1. Setup Samples
    stacked = az.extract(idata, combined=True)
    n_total_samples = stacked.sample.size
    ndraws = min(ndraws, n_total_samples)
    random_indices = np.random.choice(n_total_samples, size=ndraws, replace=False)

    # 2. Get MAP (Maximum A Posteriori) for the anchor line
    c_map, d_map = get_map_indices(idata)
    map_draw = idata.posterior.sel(chain=c_map, draw=d_map)

    # 3. Find the shortest period in the system to set the global sampling density
    all_periods = []
    for orbit in stellar_system.orbits:
        # Use the MAP period as our reference for 'smoothness'
        all_periods.append(float(map_draw[orbit.period.label]))
    p_min = min(all_periods) if all_periods else 1.0

    # 4. Plot per instrument
    for inst in stellar_system.instruments:
        plt.figure(figsize=(10, 6))

        # Determine the "Pretty" grid for this instrument's time range
        t_min, t_max = inst.time.min(), inst.time.max()
        # Ensure at least 50 points per orbital period for smoothness
        n_points = max(200, int((t_max - t_min) / (p_min / 50)))
        t_pretty = np.linspace(t_min, t_max, n_points)

        # --- COMPILED PHYSICS BRIDGE ---
        # Get the NumPy-ready function specifically for THIS instrument's graph
        compiled_rv, param_names = inst.get_compiled_model(stellar_system.planets)

        # Reference for Normalization (The MAP Gamma)
        gamma_map = float(map_draw[inst.gamma.label])

        # Plot Data (Centered by the MAP gamma)
        plt.errorbar(inst.time, inst.rv - gamma_map, yerr=inst.err,
                     fmt='ko', capsize=2, zorder=10, label=f'Data ({inst.name})')

        # 5. Plot Spaghetti (Random Draws)
        alpha = max(0.02, min(0.3, 2.0 / ndraws))
        for idx in random_indices:
            s = stacked.isel(sample=idx)
            # Match the order required by the compiled function
            # args = [time_vector, p1, p2, p3...]
            args = [t_pretty] + [float(s[name]) for name in param_names]

            # The compiled function returns a list of outputs; we want the first (the RV model)
            rv_draw = compiled_rv(*args)[0]

            # Note: The model includes Gamma, so we subtract our reference Gamma
            # to center the spaghetti around zero.
            plt.plot(t_pretty, rv_draw - gamma_map, color='royalblue', alpha=alpha, zorder=1)

        # 6. Plot MAP Anchor
        if show_map:
            map_args = [t_pretty] + [float(map_draw[name]) for name in param_names]
            rv_map = compiled_rv(*map_args)[0]
            plt.plot(t_pretty, rv_map - gamma_map, color='firebrick', lw=2, zorder=5, label='Max Posterior')

        plt.title(f"Instrument: {inst.name}")
        plt.ylabel("Relative RV [m/s]")
        plt.xlabel("Time [BJD]")
        plt.legend()
        plt.tight_layout()

        # Save a unique file per instrument
        inst_filename = filename.replace(".png", f"_{inst.name}.png")
        plt.savefig(inst_filename)
        print(f"Generated plot: {inst_filename}")
        plt.close()