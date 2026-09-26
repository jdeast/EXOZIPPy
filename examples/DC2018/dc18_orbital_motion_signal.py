"""How much orbital motion does the DC2018 answer key's planet put in each
light curve, and can a STATIC 2L1S fit ignore it?

    python dc18_orbital_motion_signal.py 128            # one event, prints JSON
    python dc18_orbital_motion_signal.py 128 --out x.json

WHY.  Every bound planet in the 2018 Data Challenge orbits: the master file
gives a (AU), inclination, phase at t_0 and period, and the simulator moved
the lens.  A static binary fit pays for that in s and q -- on event 128,
whose planet turns 85 deg of orbital phase across +/-3 t_E, the static
posterior sat 1.6% low in s and 11% low in q at 40 / 21 sigma while the
prior-free likelihood preferred that biased solution over the truth by
5,900 chi2; with linear orbital motion the same likelihood recovers s, q
and rho to < 1% and dalpha/dt to 4% of the orbit's 360/P (review 2.4.14,
2026-09-22).  JDE's ruling: the STATIC sweep is restricted to events where
a static model is adequate, measured here, and the orbital-motion rung is a
roadmap item (notes/todo.txt) after static 2L1S works.

THE MEASUREMENT, optimizer-free so it cannot fall into a wrong basin (the
Nelder-Mead version of this did, on 107 and 186, and timed out on 12
events).  The projected separation of a circular orbit is

    s(t) = (a/rE) sqrt(cos^2 phi + sin^2 phi cos^2 inc),
    phi(t) = phase + 360 (t - t_0) / P

-- this reproduces the key's s at t_0 to four digits on all 43 planets, so
the convention is exact -- and its derivatives at t_0 give ds/dt and
dalpha/dt (128: -2.25/yr and 284 deg/yr against -2.32 and -293 fitted).
chi2 is then evaluated at the key's own (t_0, u_0, t_E, rho, s, q) with
alpha scanned (its convention is not ours, conventions.md C22), once static
and once moving at those rates, both signs of both rates tried (the phase
and alpha sense are conventions the key does not state).  static - moving
is the orbital-motion signal the data carry.  Above 25 (about 5 sigma for
two parameters) the event leaves the static sweep (events_moving.txt).

A NEGATIVE signal means the linear rate is a bad description of what is in
the data -- a 212 deg phase turn (208) is nowhere near linear, and 226's
static truth already fits at chi2/N = 1.00 -- not that the lens is static;
read it with chi2_static_truth / n_points beside it.  Event 131's static
truth fits at chi2/N = 1.12 and motion does not help: something else is
in that curve, and it stays in the sweep flagged.
"""

import argparse
import json
import sys
from pathlib import Path

import MulensModel as mm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402

ap = argparse.ArgumentParser(
    description="orbital-motion signal in one DC2018 event"
)
ap.add_argument("event", type=int)
ap.add_argument("--out", default=None, help="JSON path (default: print only)")
ap.add_argument("--data-dir", default=None)
args = ap.parse_args()
DATA = str(dc.data_dir_or_raise(args.data_dir))
ev = args.event
out = args.out or "/dev/null"
truth, cls = dc.load_truth(DATA, ev)
row, _ = dc.load_master_row(DATA, ev)
a, rE, inc, phase, P = (
    float(row[k]) for k in ("a", "rE", "inc", "phase", "period")
)
res = dict(
    event=ev,
    cls=cls,
    t_E=truth["t_E"],
    s_true=truth["s"],
    q_true=truth["q"],
    period_yr=P,
    inc=inc,
    phase=phase,
    a_over_rE=a / rE,
    phase_turn_deg=360.0 / P * 6 * truth["t_E"] / 365.25,
)
ci, ph = np.cos(np.radians(inc)), np.radians(phase)
proj = np.sqrt(np.cos(ph) ** 2 + np.sin(ph) ** 2 * ci * ci)
s_pred = a / rE * proj
dphi = 2 * np.pi / P  # rad/yr
ds_dt = a / rE * (np.sin(ph) * np.cos(ph) * (ci * ci - 1.0) / proj) * dphi
dalpha_dt = np.degrees(
    ci / (proj * proj) * dphi
)  # deg/yr, magnitude; sign convention tested below
res.update(
    s_pred_from_orbit=float(s_pred),
    ds_dt_pred=float(ds_dt),
    dalpha_dt_pred=float(dalpha_dt),
)
if not (np.isfinite(truth["s"]) and truth["q"] > 0) or cls.startswith("dccv"):
    res["status"] = "not 2L1S"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))
    sys.exit(0)
datasets = []
for b in ("W149", "Z087"):
    try:
        t, f, e = np.loadtxt(
            f"{DATA}/n20180816.{b}.WFIRST18.{ev:03d}.txt", unpack=True
        )
        datasets.append(mm.MulensData(data_list=[t, f, e], phot_fmt="flux"))
    except OSError:
        pass
win = (truth["t_0"] - 3 * truth["t_E"], truth["t_0"] + 3 * truth["t_E"])


def chi2(p):
    try:
        m = mm.Model(dict(p))
        m.set_magnification_methods([win[0], "VBBL", win[1]])
        return float(mm.Event(datasets=datasets, model=m).get_chi2())
    except Exception:
        return np.inf


T = {k: truth[k] for k in ("t_0", "u_0", "t_E", "rho", "s", "q")}


def scan(base):
    g = np.linspace(0, 360, 72, endpoint=False)
    c = np.array([chi2(dict(base, alpha=x)) for x in g])
    if not np.isfinite(c).any():
        return np.nan, np.inf
    a0 = g[int(np.nanargmin(c))]
    g2 = np.linspace(a0 - 6, a0 + 6, 61)
    c2 = np.array([chi2(dict(base, alpha=x)) for x in g2])
    i = int(np.nanargmin(c2))
    return float(g2[i]), float(c2[i])


a_s, c_s = scan(T)
res.update(
    alpha_static=a_s,
    chi2_static_truth=c_s,
    n_points=int(sum(len(d.time) for d in datasets)),
)
best = None
# Both signs of BOTH rates: the phase/node convention fixes |ds_dt| and
# |dalpha_dt| (128 checks the magnitudes) but their signs depend on which
# way phase and alpha are counted, and a wrong sign makes the moving model
# WORSE than static at the truth (six events came back negative).
for sgn_a in (-1.0, +1.0):
    for sgn_s in (-1.0, +1.0):
        a_m, c_m = scan(
            dict(
                T,
                ds_dt=sgn_s * ds_dt,
                dalpha_dt=sgn_a * dalpha_dt,
                t_0_kep=T["t_0"],
            )
        )
        res["chi2_motion_truth_dalpha%+d_ds%+d" % (int(sgn_a), int(sgn_s))] = (
            c_m
        )
        if best is None or c_m < best[1]:
            best = ((sgn_a, sgn_s), c_m, a_m)
res.update(
    dalpha_sign=best[0][0],
    ds_sign=best[0][1],
    chi2_motion_truth=best[1],
    alpha_motion=best[2],
    dchi2_motion_signal=c_s - best[1],
    status="ok",
)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
