"""Why does ptde_async make ZERO ladder round trips on DC2018 event 128?

A round trip is one REPLICA carried T=1 -> T_max -> T=1.  It is temperature
transport, not mode transport: a replica never has to visit a second posterior
basin to complete one.  So event 128's 77-nat basin separation does not explain
the drought, and neither does the ladder shape -- the adaptation had equalized
per-pair swap acceptance to 0.504 +/- 0.019 and pulled Lambda from 25.6 to
19.8 (criterion 2L+1 = 41 against n_temps = 48, satisfied with margin) and
round trips were still exactly 0 at 21.4M evaluations.

The counter is sound: the async swap carries the direction tags with the state
(`direction[k][i], direction[k+1][j] = ...`) and _record_round_trips is
unit-tested in test_ptde_deo.py and test_mode_report.py.

ITERATION 1 measured swap_interval, n_chains and n_temps around the event-128
baseline and found 13 of 14 configurations at zero -- including one with 1916
swap attempts per replica.  Under DEO the round-trip rate should be
~1/(2+2*Lambda) per SWEEP, so that configuration's ~7660 sweep-equivalents
predicted ~180 round trips.  Getting 0 is too large a discrepancy for tuning.
Iteration 1 also failed to record swap acceptance or Lambda, which made every
zero uninterpretable, and included no configuration where transport was
guaranteed possible.

ITERATION 2 fixes both and adds the decisive comparison.  The SYNCHRONOUS
sampler swaps EVERY chain at EVERY DEO pair in one coherent sweep:

    deo_pairs = _deo_pairs(swap_round, n_temps)
    perm = rng.permutation(n_chains)
    for k, kp1 in deo_pairs:
        for i in range(n_chains):

while the ASYNC sampler picks one pair AND one random chain per swap event.  A
replica therefore participates only when its own index is drawn, so the chance
of riding a coherent sweep up the ladder is (1/n_chains)^(n_temps-1) -- i.e.
DEO's non-reversibility is destroyed and transport degrades to diffusion with
an extra 1/n_chains dilution.  A diffusive round trip over 47 rungs needs
~2*47^2 = 4400 accepted moves; nothing in iteration 1 came close.

If that is the cause, sync makes round trips where async makes none at
identical n_temps / n_chains / T_max.  Budget is held at a fixed number of
logp EVALUATIONS per configuration, since n_slots = n_temps x n_chains varies
across the sweep and evaluations are what cost money.
"""

import argparse
import io
import json
import logging
import re
import sys
import time
import traceback

import numpy as np


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _model(kind, d, sep):
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        x = pm.Normal("x", mu=0.0, sigma=1.0, shape=d)
        if kind == "bimodal":
            # Equal-weight mixture along axis 0, minus the base RV's own logp
            # on that axis so it is not double counted.  Getting this wrong
            # (subtracting the wrong term) yields a UNIMODAL target that looks
            # bimodal in the source -- it happened once already.
            base = pm.logp(pm.Normal.dist(mu=0.0, sigma=1.0), x[0])
            pm.Potential(
                "mix",
                pt.logaddexp(
                    base, pm.logp(pm.Normal.dist(mu=sep, sigma=1.0), x[0])
                )
                - base,
            )
    return model


def _run(cfg, target_evals, d, sep, cores):
    from exozippy.samplers.ptde import ptde_sample
    from exozippy.samplers.ptde_async import ptde_async_sample

    n_temps = cfg["n_temps"]
    n_chains = cfg["n_chains"]
    swap_interval = cfg["swap_interval"]
    kind = cfg["target"]
    n_slots = n_temps * n_chains
    # A configuration may override the global budget.  Required for the ELE
    # scan: raising swap_interval at FIXED evaluations starves the swaps
    # instead of adding exploration between them (swapint41472 got 14 swap
    # EVENTS in total, 0.0 per replica), so the budget has to grow with the
    # interval to hold swap events roughly constant.
    target_evals = int(cfg.get("target_evals") or target_evals)
    tune = max(20, target_evals // (4 * n_slots))
    draws = max(20, target_evals // n_slots - tune)

    model = _model(kind, d, sep)

    # Capture the sampler's own log: neither the per-pair swap acceptance nor
    # Lambda is stamped on the trace, and ladder_health_report already
    # computes Lambda exactly as Syed et al. define it, so parsing beats
    # recomputing.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    sampler_log = logging.getLogger("exozippy.samplers")
    prev_level = sampler_log.level
    sampler_log.addHandler(handler)
    sampler_log.setLevel(logging.INFO)

    common = dict(
        draws=draws,
        tune=tune,
        n_temps=n_temps,
        T_max=cfg.get("T_max", 8500.0),
        n_chains=n_chains,
        cores=cores,
        seed=cfg.get("seed", 17),
        # A handful of progress lines rather than none: gamma and the
        # per-rung DE acceptance appear ONLY there, and without them every
        # zero is unattributable -- the residual after the swap-schedule
        # deficit is supposed to be ELE violation (gamma far below
        # 2.38/sqrt(2d) with low acceptance), and that is exactly what these
        # two numbers say.
        log_interval=max(1, (tune + draws) // 4),
        min_ess=None,
        max_rhat=None,
    )
    t0 = time.time()
    try:
        if cfg.get("sampler", "async") == "sync":
            # swap_interval is STEPS for the sync sampler, and one sync step is
            # already a full sweep over every rung and chain, so 1 is the
            # analogue of the async default rather than an aggressive setting.
            idata = ptde_sample(
                model, _MinimalSystem(), swap_interval=1, **common
            )
        else:
            idata = ptde_async_sample(
                model,
                _MinimalSystem(),
                swap_interval=swap_interval,
                store_hot_chains=False,
                **common,
            )
    finally:
        # try/FINALLY: _run is called inside a try/except, so a bare sequence
        # would leave this handler attached on any failure and every later
        # configuration would inherit this one's captured text -- the leak
        # that fabricated a two-pass MMEXOFAST workflow in the architecture
        # post-hoc script.
        sampler_log.removeHandler(handler)
        sampler_log.setLevel(prev_level)
    wall = time.time() - t0

    text = log_buf.getvalue()
    m = re.findall(r"communication barrier Lambda=([0-9.]+)", text)
    lam = float(m[-1]) if m else float("nan")
    m = re.findall(r"swap(?:\(cum\))?=\[([^\]]*)\]", text)
    acc = float(np.mean([float(x) for x in m[-1].split(",")])) if m else np.nan
    # gamma and the per-rung DE acceptance: the ELE diagnostics.  ter Braak's
    # guideline is gamma = 2.38/sqrt(2d), so 0.32 at d=27; the microlensing
    # run adapted to 0.037 with acceptance 0.10-0.20, which is a severe ELE
    # violation AND self-inconsistent (at gamma that small acceptance should
    # be high, so the population covariance is not matching the posterior).
    m = re.findall(r"gamma=([0-9.]+)", text)
    gam = float(m[-1]) if m else float("nan")
    m = re.findall(r"accept=\[([^\]]*)\]", text)
    if m:
        ar = [float(x) for x in m[-1].split(",")]
        de_acc, de_acc_cold = float(np.mean(ar)), float(ar[0])
    else:
        de_acc = de_acc_cold = float("nan")

    a = idata.posterior.attrs
    rt = int(a.get("ptde_ladder_round_trips", 0))
    rounds = int(a.get("ptde_swap_rounds", 0))
    evals = n_slots * (tune + draws)
    out = dict(cfg)
    out.update(
        n_slots=n_slots,
        tune=tune,
        draws=draws,
        evals=evals,
        wall_s=round(wall, 1),
        round_trips=rt,
        swap_events=rounds,
        rt_per_1e6_evals=round(rt * 1e6 / max(evals, 1), 3),
        rt_per_1e5_swaps=round(rt * 1e5 / max(rounds, 1), 3),
        swaps_per_replica=round(rounds * 2.0 / max(n_slots, 1), 1),
        swap_accept=round(float(acc), 4),
        Lambda=round(lam, 2),
        gamma_final=round(gam, 5),
        de_accept_mean=round(de_acc, 4),
        de_accept_cold=round(de_acc_cold, 4),
        gamma_guideline=round(2.38 / np.sqrt(2 * 27), 4),
        sampler=cfg.get("sampler", "async"),
    )
    if kind == "bimodal":
        x0 = np.asarray(idata.posterior["x"])[..., 0].ravel()
        out["frac_far_mode"] = round(float(np.mean(x0 > sep / 2.0)), 4)
    return out


def _configs():
    base = dict(n_temps=48, n_chains=54, swap_interval=54, target="unimodal")
    cfgs = []
    # (a) TRIVIAL CONTROLS.  A short ladder over a small T_max MUST
    # communicate.  If these report no round trips the mechanism is broken and
    # nothing else in the sweep means anything.  Both samplers.
    for nt, tm in ((4, 4.0), (8, 16.0)):
        for smp in ("async", "sync"):
            cfgs.append(
                dict(
                    n_temps=nt,
                    n_chains=8,
                    swap_interval=8,
                    target="unimodal",
                    T_max=tm,
                    sampler=smp,
                    label=f"CONTROL-t{nt}-T{tm:.0f}-{smp}",
                )
            )
    # (a0) THE ELE SCAN, DONE PROPERLY.  At N=8 round trips DO occur (146
    # observed against 2774 predicted by 1/(2+2*sum(r/(1-r))) -- a 19x
    # shortfall), so the hypothesis is testable as ROUND TRIPS PER SWAP
    # EVENT: if correlated energies are the cause, adding exploration between
    # swaps should raise that ratio toward the prediction.
    #
    # The budget SCALES with swap_interval so the number of swap events stays
    # roughly fixed at ~2500 per replica; holding evaluations fixed instead
    # (the first attempt) simply removed the swaps.  n_chains=32 keeps
    # n_slots small enough to afford the largest interval, and stays above
    # n_params+2 = 29 so the DE difference vectors still span the space.
    #
    # Bittner, Nussbaumer & Janke (PRL 101, 130603, 2008) put the ideal at
    # n_local = tau_V (full energy decorrelation) and the CPU-optimal band at
    # tau_V/64..tau_V/8.  1, 8, 64, 256 brackets that for a plausible tau_V.
    for si in (1, 8, 64, 256):
        cfgs.append(
            dict(
                n_temps=8,
                n_chains=32,
                swap_interval=si,
                T_max=16.0,
                target="unimodal",
                target_evals=320_000 * si,
                label=f"ELE8-swapint{si}",
            )
        )
    # (a1) THE EXPLORATION-PER-SWAP SCAN -- the decisive test.
    #
    # Every configuration so far swaps after ONE DE proposal per chain, so
    # between swap attempts a chain makes ~0.3 accepted moves and its energy
    # V(X) barely changes.  That violates DEO's Efficient Local Exploration
    # assumption (A2: V(X) and V(X') independent across an inter-swap
    # interval) by construction, in BOTH samplers -- which is why sync, doing
    # proper full DEO sweeps with 231 attempts per replica, also made zero
    # round trips at 48 rungs where the finite-N formula
    # tau = 1/(2 + 2*sum(r/(1-r))) predicts 220.
    #
    # Note this REVERSES the swap_interval reasoning: swaps are free in model
    # evaluations, so the instinct was to do more of them (interval 2).  If
    # ELE is the binding constraint the fix is the opposite -- fewer swaps,
    # more exploration between them.  At fixed budget there is a tradeoff and
    # an optimum, and this scan is what finds it.
    #
    # swap_interval for async is EVALUATIONS per swap event, so larger =
    # more exploration per swap.  n_slots = 48*54 = 2592, so 2592 is one
    # full step of work per swap event and 10368 is four.
    for si in (2, 54, 2592, 10368, 41472):
        cfgs.append(
            dict(
                n_temps=48,
                n_chains=54,
                swap_interval=si,
                target="unimodal",
                label=f"ELE-swapint{si}",
            )
        )
    # (a2) UPWARD from the baseline -- the direction iteration 2 never tried.
    # Every earlier row was at or below n_temps=48 / n_chains=54, so "more
    # rungs do not help" was never actually tested.  If round trips appear at
    # 96 or 192 rungs then the DEO criterion N >= 2*Lambda+1 (= 41 here) is
    # merely far too optimistic and the theory is intact; if they do not, the
    # shortfall is about the within-rung kernel or the swap cadence instead.
    # swap_interval=2 throughout: that is the value that actually matches the
    # synchronous sampler's one-attempt-per-replica-per-step cadence, since
    # sync does (n_temps/2)*n_chains attempts per n_temps*n_chains
    # evaluations.  The old default of n_chains was 27x sparser than that.
    for nt in (96, 192, 384):
        cfgs.append(
            dict(
                n_temps=nt,
                n_chains=54,
                swap_interval=2,
                target="unimodal",
                label=f"UP-temps{nt}",
            )
        )
    for nc in (108,):
        cfgs.append(
            dict(
                n_temps=48,
                n_chains=nc,
                swap_interval=2,
                target="unimodal",
                label=f"UP-chains{nc}",
            )
        )
    # And the baseline ladder at the CORRECTED swap cadence, to separate
    # "more rungs" from "more swaps".
    cfgs.append(
        dict(
            n_temps=48,
            n_chains=54,
            swap_interval=2,
            target="unimodal",
            label="UP-swap2-baseline",
        )
    )
    # (b) One factor at a time from the event-128 baseline, plus sync.
    # n_chains stays >= n_params + 2 = 29: below that the DE difference
    # vectors cannot span the space, which confounds transport with a
    # degenerate proposal (iteration 1's chains4/chains16 rows both warned).
    # (g) EQUILIBRATION SCAN.  The production ladder on event 194 is
    # uniformly healthy -- 23 pairs at 0.44-0.50 swap acceptance, Lambda
    # 12.24, no bottleneck -- and makes ZERO round trips in 55,000 swap
    # rounds, while this Gaussian at the same T_max, n_temps and a
    # comparable Lambda makes 5 in 21.  The one sharp difference is
    # within-rung mobility: production's per-rung step acceptance is
    # 0.030-0.050 against this bench's 0.200.  DEO's transport theory
    # assumes a replica EQUILIBRATES at its rung between swap attempts; a
    # replica whose configuration is frozen carries a stale lp up and down
    # and cannot random-walk in temperature.  swap_interval is the number
    # of within-rung steps per swap round, so raising it buys equilibration
    # per swap.  If the mechanism is right, round trips PER SWAP ROUND rise
    # with swap_interval even as the number of rounds falls.
    # !! THIS GROUP MUST RUN ASYNC.  _run HARDCODES swap_interval=1 for the
    # sync sampler (see the comment at its call: one sync step is already a
    # full sweep over every rung and chain), so a sync scan over this knob
    # returns four byte-identical rows -- which is exactly what it did on
    # 2026-09-22 before anyone read the dispatch.  Async honours the knob,
    # but carries its own transport defect (2.4.9: sync 24-40 round trips
    # against async 0-1 on ob140939 at equal Lambda), so a positive result
    # here needs that controlled for before it means anything.
    for si in (1, 6, 54, 216):
        cfgs.append(
            dict(
                n_temps=24,
                n_chains=54,
                swap_interval=si,
                target="unimodal",
                sampler="async",
                T_max=200.0,
                label=f"EQUIL-swapint{si}",
            )
        )

    # (f) T_max SCAN AT A FIXED LADDER -- the experiment iteration 1 never
    # ran, and the reason its Lambda story does not hold.  Group (a) above
    # is labelled TRIVIAL CONTROLS and runs T_max 4 and 16; every other
    # configuration in this file runs the default T_max = 8500.  The
    # "round trips collapse with Lambda" table in
    # notes/pt_round_trip_collapse.txt is those two groups concatenated, so
    # Lambda and T_max are perfectly confounded across it -- and the data
    # already contains the counterexample: T_max=8500 with Lambda=3.50 made
    # ZERO round trips while T_max=16 with Lambda=4.14-4.79 made 90-1824.
    # Here n_temps and n_chains are HELD, so only the path length moves.
    for tm in (4.0, 16.0, 50.0, 200.0, 1000.0, 8500.0):
        for kind in ("unimodal", "bimodal"):
            cfgs.append(
                dict(
                    n_temps=24,
                    n_chains=54,
                    swap_interval=54,
                    target=kind,
                    sampler="sync",
                    T_max=tm,
                    label=f"TMAX-T{tm:.0f}-{kind}",
                )
            )

    for kind in ("unimodal", "bimodal"):
        b = dict(base, target=kind)
        cfgs.append(dict(b, label=f"baseline-{kind}"))
        cfgs.append(dict(b, sampler="sync", label=f"baseline-{kind}-sync"))
        cfgs.append(dict(b, swap_interval=1, label=f"swap1-{kind}"))
        cfgs.append(
            dict(b, n_chains=32, swap_interval=32, label=f"chains32-{kind}")
        )
        cfgs.append(dict(b, n_temps=24, label=f"temps24-{kind}"))
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-evals", type=int, default=600_000)
    ap.add_argument("--dim", type=int, default=27)
    ap.add_argument("--sep", type=float, default=8.0)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--out", default="pt_transport_bench.json")
    ap.add_argument("--only", default=None, help="comma-separated labels")
    args = ap.parse_args()

    cfgs = _configs()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        cfgs = [c for c in cfgs if c["label"] in want]

    print(
        f"{len(cfgs)} configurations, ~{args.target_evals} evals each, "
        f"d={args.dim}, cores={args.cores}",
        flush=True,
    )
    rows = []
    for c in cfgs:
        c = dict(c)
        label = c.pop("label")
        try:
            r = _run(c, args.target_evals, args.dim, args.sep, args.cores)
            r["label"] = label
            rows.append(r)
            print(
                f"  {label:28s} rt={r['round_trips']:6d}  "
                f"per1e5swap={r['rt_per_1e5_swaps']:8.3f}  "
                f"sw/repl={r['swaps_per_replica']:7.1f}  "
                f"acc={r['swap_accept']:.2f}  L={r['Lambda']:5.1f}  "
                f"gam={r['gamma_final']:.4f} deacc={r['de_accept_mean']:.2f}  "
                f"{r['wall_s']:6.1f}s"
                + (
                    f"  far={r.get('frac_far_mode')}"
                    if "frac_far_mode" in r
                    else ""
                ),
                flush=True,
            )
        except BaseException as exc:
            print(
                f"  {label:28s} FAILED {type(exc).__name__}: {exc}", flush=True
            )
            traceback.print_exc()
            rows.append({"label": label, "error": str(exc), **c})
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
