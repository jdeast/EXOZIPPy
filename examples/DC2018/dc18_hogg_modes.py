"""Score a run's posterior MODES under three noise models, to see whether
the Hogg mixture is what ranks them.

The question dc128_hogg_complicity.py asked of ONE mode against a forced
variant, asked here of the modes the pipeline itself reported: does the
ranking the fitted noise model produces survive a clean Gaussian?

  (a) gauss_es1    plain Gaussian, err_scale = 1      -- what the LC alone says
  (b) gauss_esfit  Gaussian with that mode's err_scale -- inflation only
  (c) hogg_fit     that mode's own Hogg mixture        -- what the fit used

COMPLICITY is a RANK INVERSION between (a) and (c): a geometry the light
curve prefers under clean errors losing once each mode is allowed its own
out_frac/out_scale.  Two guards against fooling ourselves:

  * Each mode is scored BOTH as-fitted and with its two flux parameters
    (f_source, f_blend) refit by weighted least squares under the noise
    model being used.  fs/fb were fitted under the mode's OWN mixture, so
    scoring them unrefit under a clean Gaussian is not a fair test.
  * Every mode is additionally scored under a COMMON noise model (mode
    `--common`'s), which separates "this geometry fits better" from "this
    mode bought itself a more forgiving likelihood".

Mode membership is by nearest reported centre in (log s, log q); the
representative point is the highest-lp draw in that cluster.
"""

import argparse
import os
import sys

import numpy as np
import yaml

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


def mode_centres(csv_path):
    """{mode: (s, q)} from the pipeline's own per-mode summaries."""
    import csv as _csv

    out = {}
    for r in _csv.reader(open(csv_path)):
        if not r or r[0].startswith("#") or r[1].startswith("rejected"):
            continue
        if r[0] == "lens.Companion.s":
            out.setdefault(r[1], {})["s"] = float(r[4])
        elif r[0] == "lens.Companion.q":
            out.setdefault(r[1], {})["q"] = float(r[4])
    return {
        k: (v["s"], v["q"]) for k, v in out.items() if "s" in v and "q" in v
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--modes", default="1,2,3,5")
    ap.add_argument(
        "--common",
        default="3",
        help="mode whose noise model is the shared one",
    )
    args = ap.parse_args()

    import arviz as az

    from exozippy.system import System

    run_dir = os.path.abspath(args.run_dir)
    pre = os.path.join(run_dir, f"DC2018_{args.event}")
    centres = mode_centres(pre + "_results.csv")
    want = [m for m in args.modes.split(",") if m in centres]
    print(f"modes reported: {sorted(centres)}; scoring {want}")
    for m in sorted(centres):
        print(f"  mode {m}: s = {centres[m][0]:.4g}, q = {centres[m][1]:.4g}")

    os.chdir(run_dir)
    config = yaml.safe_load(open(pre + ".yaml"))
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    system.compile_plotter_functions(model)
    inst = system.mulensinstrument

    idata = az.from_netcdf(pre + "_trace.nc")
    post = idata.posterior
    lp = idata.sample_stats.lp.values

    def last(v):
        a = np.asarray(post[v].values)
        return a[..., -1] if a.ndim == 3 else a

    s_draw = (
        last("lens.s")
        if "lens.s" in post.data_vars
        else 10.0 ** last("lens.log_s")
    )
    q_draw = 10.0 ** last("planet.log_q")
    log_s, log_q = np.log10(s_draw), np.log10(q_draw)

    keys = list(want)
    allc = np.array(
        [
            [np.log10(centres[m][0]), np.log10(centres[m][1])]
            for m in sorted(centres)
        ]
    )
    allk = sorted(centres)
    d2 = (log_s[..., None] - allc[:, 0]) ** 2 + (
        log_q[..., None] - allc[:, 1]
    ) ** 2
    assign = np.argmin(d2, axis=-1)

    unique_observers, obs_to_inst, inst_obs_loc = inst._observer_groups()

    def model_flux_at_data(pt):
        vals = inst._point_to_plot_params(pt, system)
        fs_all, fb_all = inst._compiled_flux(*vals)
        out = np.empty_like(inst.flux)
        for i in range(inst.n_elements):
            mask = inst.inst_map == i
            t_i = inst.time[mask]
            pos = inst._abs_to_delta(
                t_i,
                inst.get_observer_position(
                    t_i, observer_location=inst_obs_loc[i]
                ),
            )
            out[mask] = inst._compiled_model_flux(t_i, pos, i, *vals)
        return out, np.asarray(fs_all), np.asarray(fb_all)

    sig_data = inst.err
    picked = {}
    for m in keys:
        k = allk.index(m)
        sel = assign == k
        if not sel.any():
            print(f"mode {m}: no draws assigned, skipping")
            continue
        masked = np.where(sel, lp, -np.inf)
        ci, di = np.unravel_index(np.argmax(masked), lp.shape)
        point = {
            v: np.asarray(post[v].values[ci, di])
            for v in post.data_vars
            if not v.endswith("_raw")
        }
        f_mod, fs, fb = model_flux_at_data(point)
        picked[m] = dict(
            point=point,
            chain=int(ci),
            draw=int(di),
            lp=float(lp[ci, di]),
            n=int(sel.sum()),
            f=f_mod,
            fs=fs,
            fb=fb,
            err_scale=np.atleast_1d(point["mulensinstrument.err_scale"]),
            out_frac=np.atleast_1d(point["mulensinstrument.out_frac"]),
            out_scale=np.atleast_1d(point["mulensinstrument.out_scale"]),
            s=float(s_draw[ci, di]),
            q=float(q_draw[ci, di]),
        )
        p = picked[m]
        print(
            f"\nmode {m}: {p['n']} draws, best lp {p['lp']:.2f} "
            f"(chain {p['chain']}, draw {p['draw']}), s = {p['s']:.4g}, q = {p['q']:.4g}"
        )
        print(f"   err_scale = {np.round(p['err_scale'], 4)}")
        print(f"   out_frac  = {np.round(p['out_frac'], 4)}")
        print(f"   out_scale = {np.array2string(p['out_scale'], precision=6)}")
        try:
            probs = inst.outlier_prob_at_data(system, point)
            for i in range(inst.n_elements):
                mask = inst.inst_map == i
                print(
                    f"   inst {i}: outlier prob > 0.5 on {int((probs[mask] > 0.5).sum())}"
                    f"/{int(mask.sum())} points"
                )
        except Exception as exc:  # pragma: no cover
            print(f"   outlier probs unavailable: {exc}")

    def refit(f_mod, fs, fb, weights):
        """Linear refit of (f_source, f_blend) per instrument under `weights`."""
        out = np.empty_like(f_mod)
        for i in range(inst.n_elements):
            mask = inst.inst_map == i
            a_i = (f_mod[mask] - fb[i]) / fs[i]
            A = np.column_stack([a_i, np.ones(a_i.size)])
            w = weights[mask]
            coef, *_ = np.linalg.lstsq(
                A * w[:, None], inst.flux[mask] * w, rcond=None
            )
            out[mask] = A @ coef
        return out

    def score(f_model, es, of, os_, use_hogg):
        resid = inst.flux - f_model
        tot = 0.0
        for i in range(inst.n_elements):
            mask = inst.inst_map == i
            sig = sig_data[mask] * (es[i] if es is not None else 1.0)
            tot += (
                hogg_logl(resid[mask], sig, float(of[i]), float(os_[i]))
                if use_hogg
                else gauss_logl(resid[mask], sig)
            )
        return tot

    print("\n" + "=" * 78)
    print("PER-MODE, EACH UNDER ITS OWN NOISE MODEL (fluxes as fitted)")
    print(f"{'mode':<6}{'gauss_es1':>16}{'gauss_esfit':>16}{'hogg_fit':>16}")
    table = {}
    for m, p in picked.items():
        one = np.ones(inst.n_elements)
        row = (
            score(p["f"], one, p["out_frac"], p["out_scale"], False),
            score(
                p["f"], p["err_scale"], p["out_frac"], p["out_scale"], False
            ),
            score(p["f"], p["err_scale"], p["out_frac"], p["out_scale"], True),
        )
        table[m] = row
        print(f"{m:<6}" + "".join(f"{v:16.2f}" for v in row))

    print("\nSAME, WITH f_source/f_blend REFIT UNDER EACH NOISE MODEL")
    table_r = {}
    for m, p in picked.items():
        one = np.ones(inst.n_elements)
        w_clean = 1.0 / sig_data
        w_fit = 1.0 / (sig_data * p["err_scale"][inst.inst_map])
        f_clean = refit(p["f"], p["fs"], p["fb"], w_clean)
        f_fit = refit(p["f"], p["fs"], p["fb"], w_fit)
        row = (
            score(f_clean, one, p["out_frac"], p["out_scale"], False),
            score(f_fit, p["err_scale"], p["out_frac"], p["out_scale"], False),
            score(f_fit, p["err_scale"], p["out_frac"], p["out_scale"], True),
        )
        table_r[m] = row
        print(f"{m:<6}" + "".join(f"{v:16.2f}" for v in row))

    com = args.common
    if com in picked:
        print(f"\nEVERY MODE UNDER MODE {com}'s NOISE MODEL (fluxes refit)")
        pc = picked[com]
        w_fit = 1.0 / (sig_data * pc["err_scale"][inst.inst_map])
        print(f"{'mode':<6}{'gauss_es1':>16}{'common_hogg':>16}")
        for m, p in picked.items():
            f_clean = refit(p["f"], p["fs"], p["fb"], 1.0 / sig_data)
            f_fit = refit(p["f"], p["fs"], p["fb"], w_fit)
            print(
                f"{m:<6}{score(f_clean, np.ones(inst.n_elements), pc['out_frac'], pc['out_scale'], False):16.2f}"
                f"{score(f_fit, pc['err_scale'], pc['out_frac'], pc['out_scale'], True):16.2f}"
            )

    print("\n==== VERDICT ====")
    for tag, idx in (("clean Gaussian", 0), ("fitted Hogg", 2)):
        best = max(table_r, key=lambda m: table_r[m][idx])
        order = sorted(table_r, key=lambda m: -table_r[m][idx])
        print(f"{tag:16s} prefers mode {best}   ranking: {' > '.join(order)}")
    a = {m: table_r[m][0] for m in table_r}
    c = {m: table_r[m][2] for m in table_r}
    ref = max(a, key=a.get)
    print(
        f"\nnats each mode gains on the clean-Gaussian winner (mode {ref}) "
        f"by switching to its own Hogg mixture:"
    )
    for m in sorted(table_r):
        print(f"  mode {m}: {(c[m] - c[ref]) - (a[m] - a[ref]):+12.2f}")
    print(
        "\nA POSITIVE number is complicity: that mode's own noise model buys it"
        "\nlikelihood the light curve did not give it under clean errors."
    )


if __name__ == "__main__":
    main()
