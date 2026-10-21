"""Stage 1: per fit, offline from its trace -- normalized lnZ per mode,
the reliability numbers, and everything a submission needs.

    python 01_evidence.py 94 173 ... [--events-file list.txt]
                          [--fit-dir DIR ...] [--allow-free-ld]

For every finished fit of each event (events/<ev>/ and events/<ev>_*/ that
hold a trace; rmdc26.candidate_fit_dirs) this:

  1. stages the config + params into $RMDC26_REPORT/work/ (the campaign
     directory is another agent's live run and is never written to) and
     rebuilds the model with THIS tree, whitening restored from the trace's
     own <prefix>_whitening.json (never re-measured);
  2. re-checks that the rebuilt logp reproduces the trace's stored lp at 50
     draws (lpcheck.compare_lp); a fit that fails keeps its numbers but is
     UNRELIABLE;
  3. applies the wrap-up's own trimming -- fold_degeneracies, then
     convergence.analyze_idata (burn-in + stuck chains) -- and
     identify_modes, so only VALID draws reach the bridge (draws the mode
     pass rejects are labelled -1 and belong to no mode);
  4. runs estimate_mode_evidences(max_posterior_draws=20000,
     log_norm=System.evidence_log_norm()) over the modes: NORMALIZED lnZ,
     comparable across fits and architectures;
  5. per mode: posterior medians and 68% intervals of the submission
     parameters, the robust-likelihood audit (outputs/outliers.py) at the
     mode's highest-lp draw and at its median draw, err_scale, and the
     hogg / Gaussian log-likelihood and point count at the best draw;
  6. per fit: stuck fraction, lp Rhat, convergence / equilibration
     verdicts, invalid fraction, out_scale Rhat per band.

Writes $RMDC26_REPORT/evidence/<ev>/<fit>.json.  A fit whose limb
darkening is not pinned at zero is SKIPPED unless --allow-free-ld (the PI
ruled u1 = 0; e.g. events/094_pspl_backup is a free-u1 PSPL).

Heavy-ish: ~40k logp evaluations per mode at ~0.5 s each over 49k epochs.
Run it under sbatch -p test -c 32 (run_evidence.sbatch).
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lpcheck  # noqa: E402
import rmdc26 as R  # noqa: E402

# WORKAROUND (DC2018_069's gap, still open on master): outputs/evidence.py
# sizes its logp pool with mp.cpu_count(), which inside a Slurm allocation
# is the NODE's core count, not the grant.  Use the affinity mask.
multiprocessing.cpu_count = lambda: len(os.sched_getaffinity(0))


def _el(post, var, el, mask):
    """Draws of element `el` (negative counts from the end) where `mask`."""
    if var not in post.data_vars:
        return None
    a = np.asarray(post[var].values, dtype=float)
    if a.ndim == 3:
        a = a[..., el if el >= 0 else a.shape[2] + el]
    return a[mask]


def instrument_bands(config):
    """[(instrument name, band name)] in the mulensinstrument vector order."""
    return [(m["name"], m["band"]) for m in config.get("mulensinstrument", [])]


def t0_par_of(system):
    """The final t0_par (MulensInstrument writes it back into the event's
    length-1 list at stage 1; see mulensevent.py)."""
    for comp in system.active_components.values():
        t = getattr(comp, "t0_par", None)
        if isinstance(t, list) and t and t[0] is not None:
            return float(t[0])
    for comp in system.active_components.values():
        t = getattr(comp, "_t0_par", None)
        if t is not None:
            return float(np.ravel(t)[0])
    return float("nan")


def mode_summary(system, post, mask, arch, inst_bands):
    """{our name: (median, minus, plus)} over the draws in `mask`, user
    units except alpha, which is converted to RADIANS through the
    Parameter's own to_internal (the internal unit of lens.alpha is rad;
    asserted)."""
    lookup = system.get_parameter_lookup()
    s = {}
    for key, var, el in (
        ("t_0", "source.t_0", 0),
        ("u_0", "source.u_0", 0),
        ("t_E", "mulensevent.t_E", 0),
        ("rho", "source.rho", 0),
        ("pi_E_N", "mulensevent.pi_E_N", 0),
        ("pi_E_E", "mulensevent.pi_E_E", 0),
    ):
        v = _el(post, var, el, mask)
        if v is not None:
            s[key] = R.percentiles(v)
    if arch == R.ARCH_BINARY:
        for key, var in (("s", "lens.s"), ("q", "lens.q"),
                         ("log_q", "planet.log_q")):  # fmt: skip
            v = _el(post, var, -1, mask)
            if v is not None:
                s[key] = R.percentiles(v)
        par = lookup["lens.alpha"]
        if str(par.internal_unit) != "rad":
            raise ValueError(f"lens.alpha internal unit {par.internal_unit}")
        n_el = post["lens.alpha"].shape[-1]
        a = _el(post, "lens.alpha", -1, mask)
        s["alpha_rad"] = R.circ_percentiles(
            np.asarray(par.to_internal(a, index=n_el - 1), dtype=float)
        )
        s["alpha_deg_user"] = R.percentiles(a)
    for i, (_name, band) in enumerate(inst_bands):
        for var in ("f_source", "f_blend", "err_scale", "out_frac"):
            v = _el(post, "mulensinstrument." + var, i, mask)
            if v is not None:
                s["%s_%s" % (var, band)] = R.percentiles(v)
    return s


def best_draw(lp, mask):
    m = np.where(mask & np.isfinite(lp), lp, -np.inf)
    c, d = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(c), int(d)


def mode_subset(idata, mask):
    """An idata view holding only one mode's draws as a single 'chain'
    (median_draw_point needs (chain, draw); the order is irrelevant to it)."""
    import xarray as xr

    c, d = np.nonzero(mask)
    post = idata.posterior.to_dataset() if hasattr(
        idata.posterior, "to_dataset") else idata.posterior  # fmt: skip
    idx_c = xr.DataArray(c, dims="k")
    idx_d = xr.DataArray(d, dims="k")
    sub = post.isel(chain=idx_c, draw=idx_d)
    sub = sub.drop_vars(["chain", "draw"], errors="ignore")
    sub = sub.expand_dims(chain=[0]).rename({"k": "draw"})
    sub = sub.assign_coords(draw=np.arange(sub.sizes["draw"]))
    return xr.DataTree.from_dict({"posterior": sub})


def audit(system, point, label):
    from exozippy.outputs.outliers import audit_point

    rows = audit_point(system, point, label)
    for r in rows:
        r["flags"] = [list(f) for f in r["flags"]]
    return rows


def fit_wall_hours(fit_dir):
    """Sampling wall time of the fit, from the first timestamp of its log to
    the trace's mtime (both read-only; nan when either is missing)."""
    import datetime as dt

    logs = sorted((Path(fit_dir) / "fitresults").glob("RMDC26_*[0-9].log"))
    traces = sorted((Path(fit_dir) / "fitresults").glob("*_trace.nc"))
    if not logs or not traces:
        return float("nan")
    with open(logs[0], encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    try:
        start = dt.datetime.strptime(first[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return float("nan")
    end = dt.datetime.fromtimestamp(traces[0].stat().st_mtime)
    return (end - start).total_seconds() / 3600.0


def one_fit(fit_dir, max_draws, allow_free_ld, lp_n=50):
    import arviz as az
    import yaml

    from exozippy.outputs.evidence import estimate_mode_evidences
    from exozippy.outputs.modes import identify_modes
    from exozippy.outputs.summary_plot import median_draw_point, point_at_draw
    from exozippy.run import fold_degeneracies
    from exozippy.samplers import convergence

    fit_dir = Path(fit_dir)
    cfg_src = R.config_path(fit_dir)
    config0 = yaml.safe_load(cfg_src.read_text())
    params0 = yaml.safe_load((fit_dir / config0["parameter_file"]).read_text())
    arch = R.architecture(config0)
    ld0 = R.ld_pinned_zero(params0)
    rec = {
        "fit": fit_dir.name,
        "fit_dir": str(fit_dir),
        "arch": arch,
        "ld_pinned_zero": ld0,
        "config_sampler": config0.get("sampler", {}),
        "fit_wall_h": fit_wall_hours(fit_dir),
    }
    if not ld0 and not allow_free_ld:
        rec["skipped"] = "limb darkening not pinned at u1 = 0"
        return rec

    t0 = time.time()
    work = R.OUT / "work" / "evidence" / fit_dir.name
    system, model, config, prefix = lpcheck.build(fit_dir, work)
    rec["build_s"] = time.time() - t0
    log_norm = float(system.evidence_log_norm())
    rec["log_norm"] = log_norm
    inst_bands = instrument_bands(config)
    rec["bands"] = [b for _n, b in inst_bands]
    rec["t0_par"] = t0_par_of(system)
    rec["finite_source"] = bool(
        (config.get("mulensevent") or [{}])[0].get("finite_source", False)
    )

    idata = az.from_netcdf(prefix + "_trace.nc")
    attrs = dict(idata.attrs)
    rec["trace_git_commit"] = attrs.get("exozippy_git_commit")
    lp_full = np.asarray(idata.sample_stats["lp"].values, dtype=float)
    rec["n_chain_full"], rec["n_draw_full"] = map(int, lp_full.shape)
    rec["stuck_frac"] = R.stuck_fraction(lp_full)
    sc = config0.get("sampler", {})
    rec["hit_maxtime"] = bool(lp_full.shape[1] < int(sc.get("draws") or 0))
    lc = lpcheck.compare_lp(model, idata.posterior, lp_full, n=lp_n, seed=1)
    lc.pop("rows")
    rec["lpcheck"] = lc

    fold_degeneracies(system, model, idata)
    report_only = set(system.report_only_labels())
    _t, diag = convergence.analyze_idata(
        idata,
        exclude=report_only,
        min_ess=sc.get("min_ess"),
        max_rhat=sc.get("max_rhat"),
    )
    del _t
    # The burn-in is the WRAP-UP's own (summary.txt) when it recorded one,
    # never less than master's: the fits are multi-seed and the early draws
    # carry the seeds' transient, which master's max-ESS rule can keep (094:
    # wrap-up 75%, master 0%).  Chains: master's stuck-chain mask.
    wu = R.parse_wrapup(fit_dir)
    rec["wrapup"] = wu
    burnin = max(int(diag["burnin"]), int(wu.get("burnin") or 0))
    good_idx = np.nonzero(diag["good_mask"])[0]
    trimmed = convergence.trim_groups(idata, good_idx, burnin)
    if "mode" in trimmed.posterior.data_vars:
        del trimmed.posterior["mode"]
    lp_good = lp_full[good_idx]
    eq = convergence.check_equilibration(lp_good, burnin) or {}
    rec.update(
        burnin=burnin,
        burnin_master=int(diag["burnin"]),
        burnin_frac=float(burnin / lp_full.shape[1]),
        n_chain_kept=int(trimmed.posterior.sizes["chain"]),
        n_draw_kept=int(trimmed.posterior.sizes["draw"]),
        max_rhat=float(wu.get("max_rhat", diag["max_rhat"])),
        min_ess=float(wu.get("min_ess", diag["min_ess"])),
        worst_rhat_var=wu.get("worst_rhat_var", diag.get("worst_rhat_var")),
        converged=bool(diag["converged"]) and not wu.get("unconverged"),
        # Not equilibrated if EITHER the wrap-up said so or master's check
        # on the trimmed lp does.
        equilibrated=(
            None if not eq and "not_equilibrated" not in wu
            else not (wu.get("not_equilibrated") or
                      (eq and not eq.get("equilibrated")))
        ),
        equilibration={k: v for k, v in eq.items()
                       if isinstance(v, (int, float, bool))},
    )  # fmt: skip
    rep = identify_modes(trimmed, attach=True)
    n_tot = rep.n_valid + rep.n_invalid
    rec.update(
        n_modes=int(rep.n_modes),
        n_valid=int(rep.n_valid),
        n_invalid=int(rep.n_invalid),
        n_pileup=int(rep.n_pileup),
        invalid_frac=float(rep.n_invalid / max(n_tot, 1)),
        invalid_reasons=dict(rep.invalid_reason_counts),
        mode_provenance=rep.provenance,
    )
    post = trimmed.posterior
    lp = np.asarray(trimmed.sample_stats["lp"].values, dtype=float)
    rec["lp_rhat"] = R.split_rhat(lp)
    osr = {}
    if "mulensinstrument.out_scale" in post.data_vars:
        a = np.asarray(post["mulensinstrument.out_scale"].values, float)
        for i, (_n, band) in enumerate(inst_bands):
            osr[band] = R.split_rhat(a[..., i])
    rec["out_scale_rhat"] = osr

    t1 = time.time()
    ev = estimate_mode_evidences(
        model, trimmed, rep, max_posterior_draws=max_draws, log_norm=log_norm
    )
    rec["bridge_s"] = time.time() - t1

    labels = np.asarray(post["mode"].values, dtype=int)
    modes = []
    for e in ev:
        k = int(e.mode)
        mask = labels == k
        m = {
            "mode": k,
            "n_draws": int(mask.sum()),
            "n_chains": int(mask.any(axis=1).sum()),
            "occupancy": float(rep.weights[k]),
            "lnZ": float(e.lnZ),
            "lnZ_err": float(e.lnZ_err),
            "re2": float(e.re2),
            "bridge_refused": bool(e.refused),
            "bridge_state": e.state,
            "bridge_reason": e.reason,
            "lnZ_bridge": float(e.status.get("lnZ_bridge", np.nan)),
            "n_post": int(e.n_post),
            "lp_max": float(np.max(np.where(mask, lp, -np.inf))),
        }
        m["summary"] = mode_summary(system, post, mask, arch, inst_bands)
        m["err_scale"] = {
            b: m["summary"]["err_scale_" + b][0]
            for b in rec["bands"]
            if "err_scale_" + b in m["summary"]
        }
        c, d = best_draw(lp, mask)
        rows = audit(system, point_at_draw(system, trimmed, c, d), "MAP")
        m["log_likelihood"] = float(sum(r["hogg_logl"] for r in rows))
        m["gauss_loglike"] = float(sum(r["gauss_logl"] for r in rows))
        m["n_data_points"] = int(sum(r["n"] for r in rows))
        sub = mode_subset(trimmed, mask)
        rows += audit(system, median_draw_point(system, sub)[0], "median")
        m["audit"] = rows
        m["outlier_kinds"] = R.outlier_kinds(rows)
        modes.append(m)
    rec["modes"] = modes
    rec["wall_s"] = time.time() - t0
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", nargs="*", type=int)
    ap.add_argument("--events-file", default=None,
                    help="one event number per line (# comments)")  # fmt: skip
    ap.add_argument("--fit-dir", action="append", default=[],
                    help="an explicit fit directory (repeatable)")  # fmt: skip
    ap.add_argument("--allow-free-ld", action="store_true")
    ap.add_argument("--max-draws", type=int, default=R.MAX_POSTERIOR_DRAWS)
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the record exists")  # fmt: skip
    a = ap.parse_args(argv)
    evs = list(a.events)
    if a.events_file:
        for line in Path(a.events_file).read_text().splitlines():
            tok = line.split("#")[0].split()
            if tok:
                evs.append(int(tok[0]))
    jobs = [(int(Path(d).name.split("_")[0]), Path(d)) for d in a.fit_dir]
    for ev in evs:
        jobs += [(ev, d) for d in R.candidate_fit_dirs(ev)]
    rc = 0
    for ev, d in jobs:
        out = R.OUT / "evidence" / R.ev3(ev) / (d.name + ".json")
        if out.exists() and not a.force:
            print("%s: exists, skipping (--force to redo)" % out, flush=True)
            continue
        try:
            rec = one_fit(d, a.max_draws, a.allow_free_ld)
        except Exception as exc:  # recorded, never silently dropped
            rc = 1
            rec = {"fit": d.name, "fit_dir": str(d), "error": repr(exc),
                   "traceback": traceback.format_exc()}  # fmt: skip
        rec["event"] = ev
        R.dump_json(rec, out)
        if "modes" in rec:
            print(
                "%s %s (%s): lp check %s (max |dlp| %.2g); %d modes; %s; "
                "wall %.0f s (bridge %.0f s)"
                % (R.ev3(ev), d.name, rec["arch"], rec["lpcheck"]["verdict"],
                   rec["lpcheck"]["dlp_max_abs"], rec["n_modes"],
                   ", ".join("lnZ %.2f+/-%.2f %s" % (m["lnZ"], m["lnZ_err"],
                             m["bridge_state"]) for m in rec["modes"]),
                   rec["wall_s"], rec["bridge_s"]),
                flush=True,
            )  # fmt: skip
        else:
            print("%s %s: %s" % (R.ev3(ev), d.name,
                  rec.get("skipped") or rec.get("error")), flush=True)  # fmt: skip
    return rc


if __name__ == "__main__":
    sys.exit(main())
