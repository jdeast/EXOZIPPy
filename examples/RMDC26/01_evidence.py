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
        rad = np.asarray(par.to_internal(a, index=n_el - 1), dtype=float)
        # The conversion must be deg -> rad and nothing else (JDE: the VBM
        # convention as fitted; no +180, no sign flip).
        if (
            np.max(np.abs(R.wrap_pi(rad - R.alpha_submit(a))), initial=0)
            > 1e-9
        ):
            raise ValueError("lens.alpha to_internal is not a pure deg->rad")
        s["alpha_rad"] = R.circ_percentiles(rad)
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


def rawz_info(post, rep):
    """Which parameters the raw-z-rejected draws are out of range in.

    identify_modes reports its rejections only as counts by reason, so the
    robust z is recomputed here exactly as it computes it (median and
    1.4826 MAD per raw dimension over the finite draws, cut at
    DEFAULT_Z_MAX) for the draws it labelled -1.  A -1 draw with no
    dimension beyond the cut was dropped for another reason (a minor
    cluster, a non-finite value) and is not counted.
    """
    from exozippy.outputs import modes as M

    X, names = M._feature_matrix(post, rep.feature_vars)
    finite = np.isfinite(X).all(axis=1)
    med, mad = M._robust_center_scale(X[finite])
    scale = np.where(mad > 0, mad, 1.0)
    lab = np.asarray(post["mode"].values, dtype=int).reshape(-1)
    rej = (lab < 0) & finite
    z = np.abs((X[rej] - med) / scale)
    beyond = np.nan_to_num(z, nan=np.inf) > M.DEFAULT_Z_MAX
    beyond = beyond[beyond.any(axis=1)]
    counts = {}
    for j, d in enumerate(names):
        p = R.base_param(d)
        counts[p] = counts.get(p, 0)
    for p in counts:
        cols = np.array([R.base_param(d) == p for d in names])
        counts[p] = (
            int(beyond[:, cols].any(axis=1).sum()) if len(beyond) else 0
        )
    n_total = X.shape[0]
    return {
        "rawz_frac": float(len(beyond) / max(n_total, 1)),
        "rawz_param_counts": {k: v for k, v in counts.items() if v},
        "unlisted_mode_params": R.unlisted_mode_params(beyond, names, n_total),
    }


def audit(system, point, label):
    from exozippy.outputs.outliers import audit_point

    rows = audit_point(system, point, label)
    for r in rows:
        r["flags"] = [list(f) for f in r["flags"]]
    return rows


def fit_spec(fit_dir=None, config=None, name=None):
    """What one_fit needs to find a fit's files: {name, fit_dir, config,
    prefix (absolute, no suffix)}.  Either a legacy events/<NNN>/ directory
    (config RMDC26_<NNN>.yaml, outputs in fitresults/) or an explicit config
    file (the SED-tied layout: run.yaml / run_long.yaml, outputs under the
    config's own `prefix:`, e.g. out_long/009)."""
    import yaml

    cfg = Path(config) if config else R.config_path(fit_dir)
    d = cfg.parent
    c = yaml.safe_load(cfg.read_text()) or {}
    pre = Path(c["prefix"])
    if not pre.is_absolute():
        pre = d / pre
    pf = c.get("parameter_file")
    return {
        "name": name or Path(fit_dir or d).name,
        "fit_dir": d,
        "config": cfg,
        "params": (d / pf) if pf else None,
        "prefix": pre,
    }


def _f(spec, suffix):
    return Path(str(spec["prefix"]) + suffix)


def spec_staleness(spec):
    """rmdc26.staleness_from_mtimes on the spec's own files."""

    def mt(p):
        return p.stat().st_mtime if p is not None and p.exists() else None

    return R.staleness_from_mtimes(
        mt(_f(spec, "_trace.nc")), mt(spec["config"]), mt(spec["params"]),
        mt(_f(spec, "_whitening.json")),
    )  # fmt: skip


def trace_identity(spec):
    """(mtime, size) of the trace: a record is reused only while they hold."""
    st = _f(spec, "_trace.nc").stat()
    return [float(st.st_mtime), int(st.st_size)]


def fit_wall_hours(spec):
    """Sampling wall time of the fit, from the first timestamp of its log to
    the trace's mtime (both read-only; nan when either is missing)."""
    import datetime as dt

    logs = [p for p in [_f(spec, ".log")] if p.exists()]
    traces = [p for p in [_f(spec, "_trace.nc")] if p.exists()]
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


def mode_physical(post, mask, arch, idx):
    """Per-draw physical parameters of one mode (rmdc26.physical_draws),
    summarized; {} when the trace lacks what they need."""
    d = {}
    for v in R.PHYSICAL_VARS:
        if v in post.data_vars:
            d[v] = np.asarray(post[v].values, dtype=float)[mask]
    need = {"mulensevent.theta_E", "star.mass", "star.distance"}
    if arch == R.ARCH_BINARY:
        need.add("mulensevent.mlens_total")
    if not need <= set(d):
        return {}
    return R.physical_summary(R.physical_draws(d, arch, *idx))


def one_fit(spec, max_draws, allow_free_ld, lp_n=50, rawz_only=False):
    import arviz as az
    import yaml

    from exozippy.outputs.evidence import estimate_mode_evidences
    from exozippy.outputs.modes import identify_modes
    from exozippy.outputs.summary_plot import median_draw_point, point_at_draw
    from exozippy.run import fold_degeneracies
    from exozippy.samplers import convergence

    if not isinstance(spec, dict):
        spec = fit_spec(spec)
    fit_dir = Path(spec["fit_dir"])
    config0 = yaml.safe_load(Path(spec["config"]).read_text())
    params0 = yaml.safe_load(Path(spec["params"]).read_text())
    arch = R.architecture(config0)
    ld0 = R.ld_pinned_zero(params0)
    rec = {
        "fit": spec["name"],
        "fit_dir": str(fit_dir),
        "config": str(spec["config"]),
        "trace": str(_f(spec, "_trace.nc")),
        "arch": arch,
        "ld_pinned_zero": ld0,
        "config_sampler": config0.get("sampler", {}),
        "fit_wall_h": fit_wall_hours(spec),
    }
    if not ld0 and not allow_free_ld:
        rec["skipped"] = "limb darkening not pinned at u1 = 0"
        return rec
    stale = spec_staleness(spec)
    if stale:
        rec["skipped"] = "STALE trace: " + stale
        return rec
    rec["trace_identity"] = trace_identity(spec)

    t0 = time.time()
    work = R.OUT / "work" / "evidence" / spec["name"]
    system, model, config, prefix = lpcheck.build(spec["config"], work)
    rec["build_s"] = time.time() - t0
    log_norm = float(system.evidence_log_norm())
    rec["log_norm"] = log_norm
    inst_bands = instrument_bands(config)
    rec["bands"] = [b for _n, b in inst_bands]
    rec["t0_par"] = t0_par_of(system)
    rec["finite_source"] = bool(
        (config.get("mulensevent") or [{}])[0].get("finite_source", False)
    )

    rec["report_tree"], rec["report_tree_commit"] = report_tree()
    idata = az.from_netcdf(prefix + "_trace.nc")
    attrs = dict(idata.attrs)
    rec["trace_git_commit"] = attrs.get("exozippy_git_commit")
    lp_full = np.asarray(idata.sample_stats["lp"].values, dtype=float)
    rec["n_chain_full"], rec["n_draw_full"] = map(int, lp_full.shape)
    rec["stuck_frac"] = R.stuck_fraction(lp_full)
    sc = config0.get("sampler", {})
    rec["hit_maxtime"] = bool(lp_full.shape[1] < int(sc.get("draws") or 0))
    if not rawz_only:
        lc = lpcheck.compare_lp(
            model, idata.posterior, lp_full, n=lp_n, seed=1
        )
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
    wu = R.parse_wrapup(fit_dir, prefix=spec["prefix"])
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
    rec.update(rawz_info(post, rep))
    if rawz_only:
        return rec
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
    try:
        idx = R.body_indices(config0)
    except ValueError:
        idx = None
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
        m["physical"] = mode_physical(post, mask, arch, idx) if idx else {}
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


def _probe(arm_dir, ev, cands, manifest_config):
    """rmdc26.prefer_out_long's view of <arm>/<ev>/<name>/, plus the
    config that wrote it."""

    def probe(name):
        tr = arm_dir / name / ("%s_trace.nc" % R.ev3(ev))
        res = arm_dir / name / ("%s_results.csv" % R.ev3(ev))
        out = {"trace": tr.exists(), "results": res.exists(), "stale": "",
               "config": None}  # fmt: skip
        if not out["trace"]:
            return out
        cfg = R.pick_config(cands, name, ev, tr.stat().st_mtime,
                            manifest_config)  # fmt: skip
        if cfg is None:
            out["stale"] = (
                "no run*.yaml with prefix %s/%s that predates "
                "the trace" % (name, R.ev3(ev))
            )
            return out
        out["config"] = cfg
        out["stale"] = spec_staleness(fit_spec(config=cfg))
        return out

    return probe


def resolve_manifest_fit(entry):
    """(spec or None, notes) for one submitted manifest entry: out_long/
    preferred, the config that actually wrote the trace (run_long2.yaml is
    found even where the manifest names run_long.yaml)."""
    import yaml

    ev = entry["event"]
    tr = entry["paths"].get("trace")
    arm_dir = (Path(tr).parent.parent if tr else
               R.CAMPAIGN / "sed" / str(entry["arm"]) / R.ev3(ev))  # fmt: skip
    cands = []
    for c in sorted(arm_dir.glob("run*.yaml")):
        if ".params" in c.name:
            continue
        try:
            pre = (yaml.safe_load(c.read_text()) or {}).get("prefix")
        except Exception:  # an unreadable candidate is not a candidate
            continue
        if pre:
            cands.append((c, str(pre), c.stat().st_mtime))
    probe = _probe(arm_dir, ev, cands, entry["paths"].get("config"))
    seen = {}

    def cached(name):
        if name not in seen:
            seen[name] = probe(name)
        return seen[name]

    run_dir, note = R.prefer_out_long(entry["dir"], cached)
    notes = [note] if note else []
    if run_dir is None:
        return None, notes
    cfg = cached(run_dir)["config"]
    if entry["paths"].get("config") and str(cfg) != entry["paths"]["config"]:
        notes.append("config %s used (the manifest named %s)"
                     % (cfg, entry["paths"]["config"]))  # fmt: skip
    name = "%s_%s_%s" % (R.ev3(ev), arm_dir.parent.name, run_dir)
    return fit_spec(config=cfg, name=name), notes


def manifest_jobs(manifest):
    """[(ev, spec, entry, notes)] for every SUBMITTED event, and a resolved
    listing (also problems) that stage 2 reads.  The tier lists are read
    now, every time (they change)."""
    entries = R.parse_manifest(Path(manifest).read_text())
    full = R.read_event_list(R.TIER_FULL_FILE.read_text())
    scored = R.read_event_list(R.TIER_SCORED_FILE.read_text())
    sub, problems = R.select_submitted(entries, full, scored)
    jobs, resolved = [], []
    for e in sub:
        spec, notes = resolve_manifest_fit(e)
        row = {"event": e["event"], "tier": e["tier"], "arm": e["arm"],
               "manifest_dir": e["dir"], "verdict": e["verdict"],
               "notes": notes}  # fmt: skip
        if spec is None:
            problems.append("%03d (%s): no usable fit: %s"
                            % (e["event"], e["tier"], "; ".join(notes)))  # fmt: skip
            row["unusable"] = True
        else:
            row.update(fit=spec["name"], config=str(spec["config"]),
                       trace=str(_f(spec, "_trace.nc")),
                       run_dir=Path(str(spec["prefix"])).parent.name)  # fmt: skip
            jobs.append((e["event"], spec))
        resolved.append(row)
    listing = {"manifest": str(manifest), "tier_full": sorted(full),
               "tier_scored": sorted(scored), "problems": problems,
               "submitted": resolved}  # fmt: skip
    return jobs, listing


def report_tree():
    """(path, git HEAD) of the exozippy tree this process imports."""
    import subprocess

    import exozippy

    tree = os.path.dirname(os.path.dirname(os.path.dirname(exozippy.__file__)))
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()  # fmt: skip
    return tree, head or None


def record_is_current(out, spec):
    """An existing record is reused only for the SAME trace (mtime, size),
    evaluated by the SAME exozippy commit, and only if it did not error."""
    if not out.exists():
        return False
    try:
        old = json.loads(out.read_text())
    except ValueError:
        return False
    if "error" in old:
        return False
    if "modes" in old and old.get("report_tree_commit") != report_tree()[1]:
        return False
    if "trace_identity" not in old:
        return "skipped" in old and spec_staleness(spec) != ""
    return old["trace_identity"] == trace_identity(spec)


def run_workers(argv, n, jobs):
    """Run this script as `n` shard processes on disjoint core sets (the
    bridge pool sizes itself from the affinity mask), heaviest traces
    first, round-robin."""
    import subprocess

    cores = sorted(os.sched_getaffinity(0))
    per = max(1, len(cores) // n)
    procs = []
    for i in range(n):
        mine = (
            cores[i * per : (i + 1) * per] if i < n - 1 else cores[i * per :]
        )
        cmd = (
            [sys.executable, str(Path(__file__).resolve())]
            + argv
            + ["--shard", "%d/%d" % (i, n), "--cpus", ",".join(map(str, mine))]
        )
        procs.append(subprocess.Popen(cmd))
    return max(p.wait() for p in procs)


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", nargs="*", type=int)
    ap.add_argument("--events-file", default=None,
                    help="one event number per line (# comments)")  # fmt: skip
    ap.add_argument("--fit-dir", action="append", default=[],
                    help="an explicit fit directory (repeatable)")  # fmt: skip
    ap.add_argument(
        "--manifest", nargs="?", const=str(R.MANIFEST), default=None,
        help="process every SUBMITTED event of the campaign's "
        "SUBMISSION_MANIFEST.txt (tier lists read at run time; out_long/ "
        "preferred); default path %s" % R.MANIFEST,
    )  # fmt: skip
    ap.add_argument("--only", default=None,
                    help="with --manifest: only these events (e.g. 13,68)")  # fmt: skip
    ap.add_argument("--workers", type=int, default=1,
                    help="fits in parallel, each on its own core slice")  # fmt: skip
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--allow-free-ld", action="store_true")
    ap.add_argument("--max-draws", type=int, default=R.MAX_POSTERIOR_DRAWS)
    ap.add_argument(
        "--update-rawz",
        action="store_true",
        help="patch EXISTING records with the raw-z rejection breakdown "
        "(POSSIBLE_UNLISTED_MODE) without re-running the bridge",
    )
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the record exists")  # fmt: skip
    a = ap.parse_args(raw_argv)
    if a.cpus:
        os.sched_setaffinity(0, {int(c) for c in a.cpus.split(",")})
    evs = list(a.events)
    if a.events_file:
        evs += R.read_event_list(Path(a.events_file).read_text())
    jobs = [(int(Path(d).name.split("_")[0]), fit_spec(d)) for d in a.fit_dir]
    for ev in evs:
        jobs += [(ev, fit_spec(d)) for d in R.candidate_fit_dirs(ev)]
    if a.manifest:
        mj, listing = manifest_jobs(a.manifest)
        if a.only:
            keep = set(R.read_event_list(a.only.replace(",", "\n")))
            mj = [j for j in mj if j[0] in keep]
        jobs += mj
        if a.shard is None:
            R.dump_json(listing, R.OUT / "evidence" / "manifest_resolved.json")
            for p in listing["problems"]:
                print("MANIFEST: " + p, flush=True)
    if a.workers > 1 and a.shard is None:
        return run_workers(raw_argv, a.workers, jobs)
    if a.shard is not None:
        i, n = map(int, a.shard.split("/"))
        jobs.sort(key=lambda j: -_f(j[1], "_trace.nc").stat().st_size)
        jobs = jobs[i::n]
    rc = 0
    for ev, spec in jobs:
        out = R.OUT / "evidence" / R.ev3(ev) / (spec["name"] + ".json")
        if a.update_rawz:
            if not out.exists():
                continue
            old = json.loads(out.read_text())
            if "modes" not in old or "unlisted_mode_params" in old:
                continue
            new = one_fit(spec, a.max_draws, a.allow_free_ld, rawz_only=True)
            for k in ("rawz_frac", "rawz_param_counts",
                      "unlisted_mode_params"):  # fmt: skip
                old[k] = new[k]
            R.dump_json(old, out)
            print("%s %s: raw-z %.1f%%, unlisted %s"
                  % (R.ev3(ev), spec["name"], 100 * new["rawz_frac"],
                     new["unlisted_mode_params"]), flush=True)  # fmt: skip
            continue
        if not a.force and record_is_current(out, spec):
            print("%s: current, skipping (--force to redo)" % out, flush=True)
            continue
        try:
            rec = one_fit(spec, a.max_draws, a.allow_free_ld)
        except Exception as exc:  # recorded, never silently dropped
            rc = 1
            rec = {"fit": spec["name"], "fit_dir": str(spec["fit_dir"]),
                   "error": repr(exc),
                   "traceback": traceback.format_exc()}  # fmt: skip
        rec["event"] = ev
        R.dump_json(rec, out)
        if "modes" in rec:
            print(
                "%s %s (%s): lp check %s (max |dlp| %.2g); %d modes; %s; "
                "wall %.0f s (bridge %.0f s)"
                % (R.ev3(ev), spec["name"], rec["arch"],
                   rec["lpcheck"]["verdict"], rec["lpcheck"]["dlp_max_abs"],
                   rec["n_modes"],
                   ", ".join("lnZ %.2f+/-%.2f %s" % (m["lnZ"], m["lnZ_err"],
                             m["bridge_state"]) for m in rec["modes"]),
                   rec["wall_s"], rec["bridge_s"]),
                flush=True,
            )  # fmt: skip
        else:
            print("%s %s: %s" % (R.ev3(ev), spec["name"],
                  rec.get("skipped") or rec.get("error")), flush=True)  # fmt: skip
    return rc


if __name__ == "__main__":
    sys.exit(main())
