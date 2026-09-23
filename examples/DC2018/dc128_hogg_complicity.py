"""Hogg complicity check on the bare 128 run (fitresults_mulens trace).

Question: did the Hogg mixture + err_scale absorb the finite-source
(caustic-crossing) residuals at the fitted mode, letting the event-rate /
kinematics priors drag rho from the LC-preferred 7.3e-3 down to ~9.4e-4?

Measurements, all at the max-lp posterior draw:
  1. Noise-model posture: err_scale, out_frac, out_scale per instrument.
  2. Per-point Hogg outlier probabilities (outlier_prob_at_data), and
     whether the high-probability points cluster in the anomaly window
     (window = where the fitted 2L1S model differs from its own PSPL).
  3. The decisive number: delta logl between the fitted mode and a
     rho-forced variant (lens mass and lens-source relative proper motion
     scaled together so rho -> 7.3e-3 while t_E is preserved; fluxes
     linearly refit), scored under
       (a) plain Gaussian with err_scale = 1     (what the LC alone says)
       (b) Gaussian with the mode's err_scale    (inflation only)
       (c) the mode's full Hogg mixture          (what the fit actually used)
     Complicity = the gap (a)-(c): nats of finite-source evidence the
     fitted noise model erased.
"""

import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "configs")
TRACE = os.path.join(CFG_DIR, "fitresults_mulens", "DC2018_128_trace.nc")
RHO_TARGET = 7.3e-3  # the d=7 free-rho LC measurement (log10 rho = -2.139)
LOG_2PI = float(np.log(2.0 * np.pi))


def gauss_logl(resid, sig):
    return float(
        np.sum(-0.5 * (resid / sig) ** 2 - np.log(sig) - 0.5 * LOG_2PI)
    )


def hogg_logl(resid, sig, out_frac, out_scale):
    core = -0.5 * (resid / sig) ** 2 - np.log(sig)
    wide_sig = np.sqrt(sig**2 + out_scale**2)
    wide = -0.5 * (resid / wide_sig) ** 2 - np.log(wide_sig)
    log_in = np.log(max(1.0 - out_frac, 1e-300)) + core
    log_out = np.log(max(out_frac, 1e-300)) + wide
    return float(np.sum(np.logaddexp(log_in, log_out) - 0.5 * LOG_2PI))


def pspl_mag(t, t0, u0, te):
    u = np.sqrt(u0**2 + ((t - t0) / te) ** 2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def main():
    import arviz as az

    from exozippy.system import System

    os.chdir(CFG_DIR)
    with open("DC2018_128_mulens.yaml") as f:
        config = yaml.safe_load(f)
    with open(config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    system.compile_plotter_functions(model)
    inst = system.mulensinstrument

    idata = az.from_netcdf(TRACE)
    post = idata.posterior
    lp = idata.sample_stats.lp.values
    ci, di = np.unravel_index(np.argmax(lp), lp.shape)
    print(f"max-lp draw: chain {ci}, draw {di}, lp = {lp[ci, di]:.2f}")

    point = {
        v: np.asarray(post[v].values[ci, di])
        for v in post.data_vars
        if not v.endswith("_raw")
    }
    missing = [p_.label for p_ in system.plot_params if p_.label not in point]
    if missing:
        print(f"plot_params falling back to initval (not in trace): {missing}")

    # Derived quantities at the draw: compile the Deterministic graphs with
    # the physical parameter nodes as inputs (the graph is cut there), the
    # same trick _compiled_flux uses.
    import pytensor

    param_symbols = [p_.value for p_ in system.plot_params]
    derived_fn = pytensor.function(
        inputs=param_symbols,
        outputs=[
            system.lens.rho.value,
            system.lens.t_E.value,
            system.lens.theta_E.value,
            system.lens.mu_rel_mag.value,
        ],
        on_unused_input="ignore",
    )
    vals_mode = inst._point_to_plot_params(point, system)
    rho_v, te_v, thetae_v, murel_v = [
        np.atleast_1d(np.asarray(x)) for x in derived_fn(*vals_mode)
    ]
    print(
        f"  derived at draw: rho={rho_v}, t_E={te_v}, "
        f"theta_E={thetae_v}, mu_rel={murel_v}"
    )

    rho_mode = float(rho_v[0])
    t0 = float(np.atleast_1d(point["lens.t_0"])[0])
    u0 = float(np.atleast_1d(point["lens.u_0"])[0])
    te = float(te_v[0])
    err_scale = np.atleast_1d(point["mulensinstrument.err_scale"])
    out_frac = np.atleast_1d(point["mulensinstrument.out_frac"])
    out_scale = np.atleast_1d(point["mulensinstrument.out_scale"])
    print(f"\nnoise posture at mode: err_scale={err_scale}")
    print(f"  out_frac={out_frac}\n  out_scale={out_scale}")

    # ---- model fluxes at the data, per instrument --------------------
    unique_observers, obs_to_inst, inst_obs_loc = inst._observer_groups()

    def model_flux_at_data(pt_dict):
        vals = inst._point_to_plot_params(pt_dict, system)
        fs_all, fb_all = inst._compiled_flux(*vals)
        out = np.empty_like(inst.flux)
        for i in range(inst.n_elements):
            mask = inst.inst_map == i
            t_i = inst.time[mask]
            obs_loc = inst_obs_loc[i]
            pos = inst._abs_to_delta(
                t_i, inst.get_observer_position(t_i, observer_location=obs_loc)
            )
            out[mask] = inst._compiled_model_flux(t_i, pos, i, *vals)
        return out, np.asarray(fs_all), np.asarray(fb_all)

    f_mode, fs_all, fb_all = model_flux_at_data(point)

    # ---- anomaly window: fitted 2L1S vs its own PSPL ------------------
    sig_data = inst.err  # unscaled
    a_2l1s = np.empty_like(inst.flux)
    a_pspl = pspl_mag(inst.time, t0, u0, te)
    for i in range(inst.n_elements):
        mask = inst.inst_map == i
        a_2l1s[mask] = (f_mode[mask] - fb_all[i]) / fs_all[i]
    dev = np.abs(a_2l1s - a_pspl) * fs_all[inst.inst_map] / sig_data
    hot = dev > 0.3
    if not np.any(hot):
        print("no anomaly window found (2L1S == PSPL at mode)!?")
        sys.exit(1)
    w_lo, w_hi = inst.time[hot].min(), inst.time[hot].max()
    window = (inst.time >= w_lo) & (inst.time <= w_hi)
    print(
        f"\nanomaly window (model-based): [{w_lo:.3f}, {w_hi:.3f}] "
        f"({window.sum()} points; deviation peak {dev.max():.1f} sigma)"
    )

    # ---- outlier probabilities ----------------------------------------
    probs = inst.outlier_prob_at_data(system, point)
    for thr in (0.5, 0.9):
        n_in = int(((probs > thr) & window).sum())
        n_out = int(((probs > thr) & ~window).sum())
        frac_in = n_in / max(window.sum(), 1)
        frac_out = n_out / max((~window).sum(), 1)
        print(
            f"outlier prob > {thr}: {n_in} in window ({100 * frac_in:.2f}% of "
            f"window), {n_out} outside ({100 * frac_out:.3f}% of rest); "
            f"concentration x{frac_in / max(frac_out, 1e-12):.1f}"
        )

    # ---- rho-forced variant -------------------------------------------
    # Scale theta_E down by k = rho_mode/rho_target while preserving the
    # LC-measured quantities t_E AND pi_E (the first attempt scaled only the
    # mass, which multiplied pi_E by 1/k and injected a parallax distortion
    # the comparison was not supposed to contain):
    #   pi_rel -> k * pi_rel   (move the lens toward the source)
    #   M      -> k * M        (theta_E = sqrt(kappa M pi_rel) -> k*theta_E)
    #   pm_rel -> k * pm_rel   (t_E = theta_E/mu_rel unchanged)
    # then pi_E = pi_rel/theta_E is unchanged and rho = theta*/theta_E
    # rises by 1/k to the target.
    k = rho_mode / RHO_TARGET  # < 1
    alt = {v: np.array(val, copy=True) for v, val in point.items()}
    lm = alt["star.logmass"]
    lm[0] = lm[0] + np.log10(k)
    d = alt["star.distance"]
    inv_dl_new = 1.0 / d[1] + k * (1.0 / d[0] - 1.0 / d[1])
    d[0] = 1.0 / inv_dl_new
    for pmc in ("star.pm_ra", "star.pm_dec"):
        v = alt[pmc]
        v[0] = v[1] + k * (v[0] - v[1])
    vals_alt = inst._point_to_plot_params(alt, system)
    rho_a, te_a, thetae_a, murel_a = [
        np.atleast_1d(np.asarray(x)) for x in derived_fn(*vals_alt)
    ]
    print(
        f"\nrho-forced point verified: rho={rho_a} (target {RHO_TARGET}), "
        f"t_E={te_a} (mode {te_v}), theta_E={thetae_a}, mu_rel={murel_a}"
    )
    assert abs(rho_a[0] / RHO_TARGET - 1.0) < 0.05, "rho forcing failed"
    assert abs(te_a[0] / te_v[0] - 1.0) < 0.02, "t_E not preserved"
    f_alt, fs_a, fb_a = model_flux_at_data(alt)

    # Linear flux refit per instrument (model is linear in fs, fb).
    f_alt_fit = np.empty_like(f_alt)
    for i in range(inst.n_elements):
        mask = inst.inst_map == i
        a_i = (f_alt[mask] - fb_a[i]) / fs_a[i]  # magnification curve
        A = np.column_stack([a_i, np.ones(a_i.size)])
        w = 1.0 / inst.err[mask]
        coef, *_ = np.linalg.lstsq(
            A * w[:, None], inst.flux[mask] * w, rcond=None
        )
        f_alt_fit[mask] = A @ coef
        print(
            f"  refit inst {i}: fs {fs_a[i]:.4f}->{coef[0]:.4f}, "
            f"fb {fb_a[i]:.4f}->{coef[1]:.4f}"
        )

    # ---- score both models under the three noise models ---------------
    def score(f_model, label):
        resid = inst.flux - f_model
        rows = {}
        for tag, sig_fn, use_hogg in (
            ("gauss_es1", lambda i, m: sig_data[m], False),
            ("gauss_esfit", lambda i, m: sig_data[m] * err_scale[i], False),
            ("hogg_fit", lambda i, m: sig_data[m] * err_scale[i], True),
        ):
            tot = win = 0.0
            for i in range(inst.n_elements):
                mask = inst.inst_map == i
                sig = sig_fn(i, mask)
                if use_hogg:
                    ll = hogg_logl(
                        resid[mask],
                        sig,
                        float(out_frac[i]),
                        float(out_scale[i]),
                    )
                    lw = hogg_logl(
                        resid[mask & window],
                        sig_fn(i, mask & window),
                        float(out_frac[i]),
                        float(out_scale[i]),
                    )
                else:
                    ll = gauss_logl(resid[mask], sig)
                    lw = gauss_logl(
                        resid[mask & window], sig_fn(i, mask & window)
                    )
                tot += ll
                win += lw
            rows[tag] = (tot, win)
        print(f"\n{label}:")
        for tag, (tot, win) in rows.items():
            print(
                f"  {tag:12s} total logl = {tot:14.2f}   "
                f"window logl = {win:12.2f}"
            )
        return rows

    r_mode = score(f_mode, "fitted mode (rho = %.3e)" % rho_mode)
    r_alt = score(
        f_alt_fit, "rho-forced (rho = %.3e, fluxes refit)" % RHO_TARGET
    )

    print("\n==== VERDICT ====")
    for tag, label in (
        ("gauss_es1", "clean Gaussian (err_scale=1)"),
        ("gauss_esfit", "Gaussian with fitted err_scale"),
        ("hogg_fit", "fitted Hogg mixture"),
    ):
        d_tot = r_mode[tag][0] - r_alt[tag][0]
        d_win = r_mode[tag][1] - r_alt[tag][1]
        pref = "mode" if d_tot > 0 else "RHO-FORCED"
        print(
            f"{label:36s}: mode - forced = {d_tot:+12.2f} nats "
            f"(window {d_win:+10.2f}) -> prefers {pref}"
        )
    for label, rows in (("mode", r_mode), ("rho-forced", r_alt)):
        forgiven = rows["hogg_fit"][1] - rows["gauss_es1"][1]
        print(f"nats forgiven in window by Hogg ({label}): {forgiven:+.1f}")
    print(
        "\nComplicity = how much the preference for the true-like rho "
        "shrinks going from clean Gaussian to the fitted noise model."
    )


if __name__ == "__main__":
    main()
