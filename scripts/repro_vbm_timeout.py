"""Reproduce VBM logp calls that hit PTDE's eval_timeout, from a run log.

ptde.py logs a "raw params: {...}" dict (keyed by free-RV name, in the raw/
unconstrained space) for every proposal that exceeds sampler.eval_timeout.
This script parses those dicts back out of a run's log file, rebuilds the
model from the corresponding YAML config, and replays each event's binary-
lens magnification call directly against VBMicrolensing so the pathological
(s, q, x, y, rho) can be inspected and timed outside the sampler.

Only the raw params are needed (the physical-space dict in the log is
usually incomplete: fully-derived parameters like rho, q, alpha, t_E, and
pi_E are not wrapped in pm.Deterministic unless something else forces it, so
they never make it into model.deterministics / the log's physical-param
dict -- they must be recomputed from the raw values through the model
graph, which is what this script does).

Usage:
    poetry run python scripts/repro_vbm_timeout.py <config.yaml> <run.log>
    poetry run python scripts/repro_vbm_timeout.py <config.yaml> <run.log> --epochs

--epochs additionally walks each event's full light curve epoch by epoch
(single VBM instance, like the sampler would use) and reports per-epoch
timing, so a slow call can be localized to a specific (s, q, x, y, rho).
Without --epochs, only the whole-curve VBMDirectMagOp evaluation is timed.
"""
import argparse
import ast
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TIMEOUT_RE = re.compile(
    r"logp call exceeded eval_timeout=([\d.]+)s at (.+?) \((.+?)\) — rejecting"
)
RAW_PARAMS_RE = re.compile(r"\s*raw params: (\{.*\})\s*$")


def parse_timeout_events(log_path):
    """Return a list of dicts: {step_label, who, timeout, raw_params}."""
    events = []
    pending = None
    with open(log_path) as f:
        for line in f:
            m = TIMEOUT_RE.search(line)
            if m:
                pending = {
                    "timeout": float(m.group(1)),
                    "step_label": m.group(2),
                    "who": m.group(3),
                }
                continue
            m = RAW_PARAMS_RE.match(line)
            if m and pending is not None:
                pending["raw_params"] = ast.literal_eval(m.group(1))
                events.append(pending)
                pending = None
    return events


def build_pvec_fn(config):
    """Compile raw free-RVs -> the VBMDirectMagOp param vector for source 0.

    Returns (fn, raw_var_names, coords, times, obs_pos_abs, bandpass) where
    fn(*raw_vals) -> [t0, u0, tE, piN, piE, rho, s, q, alpha, u1] (rho/u1
    included only if the event actually uses them).
    """
    import pytensor
    import pytensor.tensor as pt
    from exozippy.system import System

    system = System(config)
    system.prepare()
    model = system.build_model()

    lens = system.lens
    use_rho = lens.finite_source[0]
    sp = lens._get_safe_mm_params(0)
    param_exprs = [sp['t0'], sp['u0'], sp['tE'], sp['pi_N'], sp['pi_E']]
    labels = ['t_0', 'u_0', 't_E', 'pi_E_N', 'pi_E_E']
    if use_rho:
        param_exprs.append(lens.rho.value[0])
        labels.append('rho')
    param_exprs.append(lens.s.value[0])
    q_j = pt.clip(pt.nan_to_num(lens.q.value[0], nan=1e-9), 1e-9, 100.0)
    alpha_deg = pt.arctan2(lens.yalpha.value[0], lens.xalpha.value[0]) * (180.0 / np.pi)
    param_exprs.extend([q_j, alpha_deg])
    labels.extend(['s', 'q', 'alpha'])

    bandpass = None
    if use_rho and hasattr(system, 'band'):
        u1 = system.band.u1.value[0]
        param_exprs.append(u1)
        labels.append('u1')
        bandpass = system.band.names[0]

    p_stack = pt.stack(param_exprs)
    inputs = list(model.free_RVs)
    fn = pytensor.function(inputs, p_stack, on_unused_input='ignore')
    raw_var_names = [rv.name for rv in inputs]

    inst = system.mulensinstrument
    sndx = int(lens.source_map[0])
    ra_deg = float(system.star.ra.value[sndx].eval()) * 180.0 / np.pi
    dec_deg = float(system.star.dec.value[sndx].eval()) * 180.0 / np.pi
    coords = f"{ra_deg}d {dec_deg}d"

    return fn, raw_var_names, labels, coords, inst.time, inst.observer_pos_abs, bandpass


def trajectory(op, p, labels, times, obs_pos_abs):
    """Numpy trajectory (x, y) plus (s, q, rho, u1) for a binary-lens event."""
    vals = dict(zip(labels, p))
    dN, dE = op._deltas(times, obs_pos_abs)
    t0, u0, tE = vals['t_0'], vals['u_0'], vals['t_E']
    piN, piE = vals['pi_E_N'], vals['pi_E_E']
    u0 = np.sign(u0) * max(abs(u0), 1e-9) if u0 != 0 else 1e-9
    tE = max(tE, 1e-4)
    rho = max(vals.get('rho', 1e-9), 1e-9)
    s = max(vals['s'], 1e-6)
    q = float(np.clip(vals['q'], 1e-9, 100.0))
    alpha = vals['alpha']
    u1 = vals.get('u1', 0.0)

    tau = (times - t0) / tE + dN * piN + dE * piE
    uu = u0 - dN * piE + dE * piN
    a = np.radians(alpha)
    x = -tau * np.cos(a) + uu * np.sin(a)
    y = -tau * np.sin(a) - uu * np.cos(a)
    return x, y, s, q, rho, u1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML config for the event (e.g. DC2018_128.yaml)")
    ap.add_argument("log", help="Sampler run log to scan for eval_timeout events")
    ap.add_argument("--epochs", action="store_true",
                    help="Also walk each event epoch by epoch to localize the slow call")
    args = ap.parse_args()

    events = parse_timeout_events(args.log)
    print(f"found {len(events)} eval_timeout event(s) in {args.log}")
    if not events:
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.pop("sampler", None)

    from exozippy.components.mulensing.op import VBMDirectMagOp
    fn, raw_var_names, labels, coords, times, obs, bandpass = build_pvec_fn(config)
    op = VBMDirectMagOp(coords=coords, n_companions=1, use_rho=('rho' in labels),
                        bandpass=bandpass)
    op._deltas(times, obs)  # warm the parallax-offset cache

    for i, ev in enumerate(events):
        raw = ev["raw_params"]
        missing = [n for n in raw_var_names if n not in raw]
        if missing:
            print(f"[{i}] {ev['step_label']} ({ev['who']}): SKIP — "
                  f"log entry missing raw var(s) {missing} (model/config mismatch?)")
            continue
        vals = [np.asarray(raw[n], dtype=np.float64) for n in raw_var_names]
        p = fn(*vals)
        vals_named = dict(zip(labels, p))
        print(f"\n[{i}] {ev['step_label']} ({ev['who']}), "
              f"logged timeout={ev['timeout']:.0f}s")
        print("    " + "  ".join(f"{k}={v:.6g}" for k, v in vals_named.items()))

        x, y, s, q, rho, u1 = trajectory(op, p, labels, times, obs)
        r_inf = s + 1.0 / s + 2.0
        d = np.sqrt(x * x + y * y)
        far = d > (r_inf + 2.0 * rho)
        print(f"    R_inf={r_inf:.4g}  guard_radius={r_inf + 2*rho:.4g}  "
              f"source distance range=[{d.min():.4g}, {d.max():.4g}]  "
              f"epochs far-field={far.sum()}/{len(d)}")

        t0 = time.perf_counter()
        A = op._compute(p, times, obs)
        dt = time.perf_counter() - t0
        print(f"    current guarded Op: {dt:.3f}s for {len(times)} epochs  "
              f"(A range [{np.nanmin(A):.4g}, {np.nanmax(A):.4g}])")
        if dt * 5 < ev["timeout"]:
            print(f"    NOTE: reproduces >{ev['timeout']:.0f}x faster than the logged "
                  f"timeout in isolation. This event likely reflects worker-pool "
                  f"contention at the time (another proposal's genuinely expensive "
                  f"near-zone finite-source call sharing the box), not an intrinsic "
                  f"cost of THIS parameter set -- see the near-zone cost distribution "
                  f"discussion in vbm_fix.txt.")

        if args.epochs:
            vbm = op._vbm
            vbm.a1 = u1
            worst_dt, worst_i = 0.0, -1
            t0 = time.perf_counter()
            for k in range(len(x)):
                tk = time.perf_counter()
                vbm.BinaryMag2(s, q, float(x[k]), float(y[k]), rho)
                dtk = time.perf_counter() - tk
                if dtk > worst_dt:
                    worst_dt, worst_i = dtk, k
            print(f"    epoch walk: total={time.perf_counter()-t0:.3f}s  "
                  f"slowest epoch {worst_i} took {worst_dt:.3f}s  "
                  f"(x={x[worst_i]:.6g}, y={y[worst_i]:.6g}, d={d[worst_i]:.6g}, "
                  f"far={bool(far[worst_i])})")


if __name__ == "__main__":
    main()
