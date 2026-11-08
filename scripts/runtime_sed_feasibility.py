"""Feasibility of a runtime spectral SED: interpolate the spectrum, redden
it with (A_V, R_V), integrate the bands -- against the shipped BC tables.

Accuracy half (this file; the speed half is runtime_sed_benchmark.py).
Run on a compute node: it loads whole [Fe/H] planes of the R = 20000
step-1 spectra (~0.7 GB each).

  E1 resolution: how coarse a runtime wavelength grid reproduces the
     full-resolution band fluxes, for two schemes --
       A  a generic binned spectrum (flux-conserving, log-uniform bins at
          resolution R) times a fixed per-filter matrix of bin-mean
          profiles: the error is the in-bin correlation of spectrum and
          profile;
       B  per-filter weighted bins, G_fb = int_bin F S dlam precomputed
          per node, reddened at runtime at the bin centre: the error is
          only the extinction's variation inside a bin.
  E2 off-node interpolation: leave-one-out cubes (the target node is
     dropped and rebuilt from its +/-1 neighbours on every axis, i.e. a
     cell twice the grid's) and 1-D Teff runs, comparing interpolation of
     the BC magnitude (what the tables do), of the linear flux and of the
     log flux; and the tables' A_V interpolation error, which the spectral
     path does not have.
  E3 R_V: how far BCs move between R_V = 2.5 and 3.1 (CCM89 + O'Donnell
     94 for both), and CCM(3.1) against the shipped extinction law.

Every number is a band-flux ratio in magnitudes: the BC's sigma Teff^4
term cancels between methods at one node.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np  # before pyarrow (system libstdc++ on hydra)

from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.bolometric_correction import (
    _filter_set_data,
    trapezoid,
)
from exozippy.models.NextGen.nextgen_spectra import WAVELENGTH_PTS

PROC = Path.home() / "spectra" / "BT-NextGen_AGSS2009_processed"
WAVE = np.asarray(WAVELENGTH_PTS, dtype=float)
LNW = np.log(WAVE)
GRID = gen.read_grid_yaml()
TEFFS = [float(t) for t in GRID["teff"]]
LOGGS = [float(g) for g in GRID["logg"]]
FEHS = [float(z) for z in GRID["feh"]]

# ---------------------------------------------------------------- filters


def load_filters():
    """Every shipped filter: weighted profile on its own support of the
    common grid, the support's index range, and the shipped law's
    A_lam/A_V there."""
    out = {}
    for fac, names in gen.FILTER_SETS.items():
        for name in names:
            d = _filter_set_data((name,), gen.FLUX_WEIGHTING)
            idx = np.flatnonzero(d["wave_mask"])
            out[d["filters_MIST"][0]] = {
                "facility": fac,
                "idx": idx,
                "prof": d["filter_profiles"][0],
                "ext_ship": d["extinction_modeled"] / d["V_band_extinction"],
            }
    return out


# ------------------------------------------------------------- extinction


def ccm89(wave_A, rv):
    """A_lam / A_V, Cardelli, Clayton & Mathis (1989) with the O'Donnell
    (1994) optical coefficients; x clamped to [0.1, 10] /um (the IR power
    law is extended beyond 3.3 um)."""
    x = np.clip(1e4 / np.asarray(wave_A, float), 0.1, 10.0)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = np.polyval(
        [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0], y
    )
    b[op] = np.polyval(
        [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0], y
    )
    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    fa = np.where(
        xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0
    )
    fb = np.where(
        xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0
    )
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb
    fu = x >= 8.0
    xf = x[fu] - 8.0
    a[fu] = -1.073 - 0.628 * xf + 0.137 * xf**2 - 0.070 * xf**3
    b[fu] = 13.670 + 4.257 * xf - 0.420 * xf**2 + 0.374 * xf**3
    return a + b / rv


# ----------------------------------------------------------------- spectra

_PLANES = {}


def plane(feh):
    if feh not in _PLANES:
        if len(_PLANES) >= 4:
            _PLANES.pop(next(iter(_PLANES)))
        _PLANES[feh] = gen._index_spectra(
            gen.load_processed_spectra(feh, PROC)
        )
    return _PLANES[feh]


def spectrum(teff, logg, feh):
    p = plane(feh)
    if (teff, logg) not in p.index:
        return None
    return np.asarray(p.loc[(teff, logg)]["flux"], dtype=float)


def band_flux(F, f, av, ext):
    """Full-resolution band flux through filter f at A_V = av with the
    A_lam/A_V array `ext` on f's support."""
    i = f["idx"]
    return trapezoid(WAVE[i], F[i] * np.exp(-av * ext / 1.086) * f["prof"])


def mag(a, b):
    return 2.5 * np.log10(a / b)


# --------------------------------------------------------------------- E1


def e1_resolution(
    filters, nodes, avs, R_A=(300, 1000, 3000, 10000), R_B=(10, 30, 100, 300)
):
    """Max |dmag| per filter and scheme/R, over nodes x avs."""
    cum_cache = {}
    res = {f"A{R}": {} for R in R_A} | {f"B{R}": {} for R in R_B}
    for t, g, z in nodes:
        F = spectrum(t, g, z)
        if F is None:
            continue
        for name, f in filters.items():
            i = f["idx"]
            w, S = WAVE[i], f["prof"]
            ext = f["ext_ship"]
            lo, hi = LNW[i[0]], LNW[i[-1]]
            truth = [band_flux(F, f, av, ext) for av in avs]
            # cumulative integrals on the support (trapezoid), for exact bin sums
            cF = np.concatenate(
                [[0], np.cumsum(0.5 * (F[i][1:] + F[i][:-1]) * np.diff(w))]
            )
            cS = np.concatenate(
                [[0], np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(w))]
            )
            FS = F[i] * S
            cFS = np.concatenate(
                [[0], np.cumsum(0.5 * (FS[1:] + FS[:-1]) * np.diff(w))]
            )
            for R in R_A:
                key = (name, R)
                if key not in cum_cache:
                    # global log-uniform bins at R (the generic grid), clipped to the support
                    k0, k1 = np.floor(lo * R), np.ceil(hi * R)
                    edges = np.exp(np.arange(k0, k1 + 1) / R)
                    edges = np.clip(edges, w[0], w[-1])
                    cum_cache[key] = edges
                edges = cum_cache[key]
                dl = np.diff(edges)
                ok = dl > 0
                Fbar = np.diff(np.interp(edges, w, cF))[ok] / dl[ok]
                Sint = np.diff(np.interp(edges, w, cS))[ok]  # = Sbar * dlam
                cen = np.sqrt(edges[:-1] * edges[1:])[ok]
                extc = np.interp(cen, w, ext)
                for av, tr in zip(avs, truth):
                    approx = np.sum(Fbar * Sint * np.exp(-av * extc / 1.086))
                    d = abs(mag(approx, tr))
                    res[f"A{R}"][name] = max(res[f"A{R}"].get(name, 0.0), d)
            for R in R_B:
                k0, k1 = np.floor(lo * R), np.ceil(hi * R)
                edges = np.clip(np.exp(np.arange(k0, k1 + 1) / R), w[0], w[-1])
                G = np.diff(np.interp(edges, w, cFS))
                cen = np.sqrt(edges[:-1] * edges[1:])
                extc = np.interp(cen, w, ext)
                for av, tr in zip(avs, truth):
                    approx = np.sum(G * np.exp(-av * extc / 1.086))
                    d = abs(mag(approx, tr))
                    res[f"B{R}"][name] = max(res[f"B{R}"].get(name, 0.0), d)
    return res


# --------------------------------------------------------------------- E2


def _bc_like(F, f, av):
    return -2.5 * np.log10(band_flux(F, f, av, f["ext_ship"]))


def e2_cubes(filters, targets, avs):
    """Leave-one-out cubes: rebuild the target from its 8 (+/-1) corners.
    Returns per filter the max |error| (mag) of each method."""
    out = {m: {} for m in ("table", "linflux", "logflux")}
    used = []
    for t, g, z in targets:
        ti, gi, zi = TEFFS.index(t), LOGGS.index(g), FEHS.index(z)
        corners = []
        for a in (ti - 1, ti + 1):
            for b in (gi - 1, gi + 1):
                for c in (zi - 1, zi + 1):
                    corners.append((TEFFS[a], LOGGS[b], FEHS[c]))

        # weights: trilinear at the target (uneven spacing allowed)
        def w1(v, lo, hi):
            return (hi - v) / (hi - lo), (v - lo) / (hi - lo)

        wt = w1(t, TEFFS[ti - 1], TEFFS[ti + 1])
        wg = w1(g, LOGGS[gi - 1], LOGGS[gi + 1])
        wz = w1(z, FEHS[zi - 1], FEHS[zi + 1])
        W = [
            wt[i] * wg[j] * wz[k]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
        Fs = [spectrum(*c) for c in corners]
        Ft = spectrum(t, g, z)
        if Ft is None or any(F is None for F in Fs):
            continue
        used.append((t, g, z))
        Flin = sum(wi * Fi for wi, Fi in zip(W, Fs))
        Flog = np.exp(
            sum(wi * np.log(np.maximum(Fi, 1e-300)) for wi, Fi in zip(W, Fs))
        )
        for name, f in filters.items():
            for av in avs:
                truth = _bc_like(Ft, f, av)
                # the tables interpolate the magnitude; sigma T^4 differs per
                # corner, so interpolate the BC (mag + 10 log T) and convert back
                tab = sum(
                    wi * (_bc_like(Fi, f, av) + 10 * np.log10(c[0]))
                    for wi, Fi, c in zip(W, Fs, corners)
                ) - 10 * np.log10(t)
                for m, v in (
                    ("table", tab),
                    ("linflux", _bc_like(Flin, f, av)),
                    ("logflux", _bc_like(Flog, f, av)),
                ):
                    out[m][name] = max(out[m].get(name, 0.0), abs(v - truth))
    return out, used


def e2_teff_runs(filters, logg, feh, av=0.0):
    """1-D leave-one-out along Teff: per (teff, filter) the error of each
    method rebuilding node i from i-1 and i+1."""
    rows = []
    for i in range(1, len(TEFFS) - 1):
        t0, t, t1 = TEFFS[i - 1], TEFFS[i], TEFFS[i + 1]
        F0, F, F1 = (
            spectrum(t0, logg, feh),
            spectrum(t, logg, feh),
            spectrum(t1, logg, feh),
        )
        if F0 is None or F is None or F1 is None:
            continue
        w = (t1 - t) / (t1 - t0)
        Flin = w * F0 + (1 - w) * F1
        Flog = np.exp(
            w * np.log(np.maximum(F0, 1e-300))
            + (1 - w) * np.log(np.maximum(F1, 1e-300))
        )
        for name, f in filters.items():
            truth = _bc_like(F, f, av)
            tab = (
                w * (_bc_like(F0, f, av) + 10 * np.log10(t0))
                + (1 - w) * (_bc_like(F1, f, av) + 10 * np.log10(t1))
                - 10 * np.log10(t)
            )
            rows.append(
                (
                    t,
                    name,
                    tab - truth,
                    _bc_like(Flin, f, av) - truth,
                    _bc_like(Flog, f, av) - truth,
                )
            )
    return rows


def e2_av(filters, nodes):
    """The tables' A_V interpolation error: exact BC at the midpoint of each
    A_V cell against the linear interpolation of its two nodes."""
    avg = [float(a) for a in GRID["av"]]
    out = {}
    for t, g, z in nodes:
        F = spectrum(t, g, z)
        if F is None:
            continue
        for name, f in filters.items():
            for a0, a1 in zip(avg[:-1], avg[1:]):
                am = 0.5 * (a0 + a1)
                err = 0.5 * (
                    _bc_like(F, f, a0) + _bc_like(F, f, a1)
                ) - _bc_like(F, f, am)
                k = (name, a0, a1)
                out[k] = max(out.get(k, 0.0), abs(err))
    return out


# --------------------------------------------------------------------- E3


def e3_rv(filters, nodes, avs=(1.0, 3.0, 8.0)):
    out = {}
    for t, g, z in nodes:
        F = spectrum(t, g, z)
        if F is None:
            continue
        for name, f in filters.items():
            w = WAVE[f["idx"]]
            e31, e25 = ccm89(w, 3.1), ccm89(w, 2.5)
            # normalize to A_V at 0.55 um as the shipped law is
            e31 /= ccm89(np.array([5500.0]), 3.1)[0]
            e25 /= ccm89(np.array([5500.0]), 2.5)[0]
            for av in avs:
                d_rv = mag(band_flux(F, f, av, e31), band_flux(F, f, av, e25))
                d_law = mag(
                    band_flux(F, f, av, f["ext_ship"]),
                    band_flux(F, f, av, e31),
                )
                k = (name, av)
                o = out.setdefault(k, {"rv": 0.0, "law": 0.0})
                o["rv"] = max(o["rv"], abs(d_rv))
                o["law"] = max(o["law"], abs(d_law))
    return out


# ------------------------------------------------------------------- main


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "out",
        nargs="?",
        default="runtime_sed_feasibility.json",
        help="JSON file the results are written to",
    )
    OUT = Path(ap.parse_args().out)
    t0 = time.time()
    filters = load_filters()
    print(
        f"{len(filters)} filters loaded ({time.time() - t0:.0f} s)", flush=True
    )

    # E1: a spread of nodes, low and high A_V
    e1_nodes = [
        (t, g, z)
        for z in (-2.0, 0.0)
        for g in (2.0, 4.5)
        for t in (2700.0, 3200.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0)
    ]
    e1 = e1_resolution(filters, e1_nodes, avs=(0.0, 2.0, 8.0, 20.0))
    print(f"E1 done ({time.time() - t0:.0f} s)", flush=True)

    # E2 cubes: interior targets with +/-1 neighbours on every axis
    targets = [
        (t, g, z)
        for z in (-1.0, -0.5)
        for g in (2.0, 4.5)
        for t in (3000.0, 3500.0, 4000.0, 5000.0, 5800.0, 7000.0, 8400.0)
    ]
    e2c, used = e2_cubes(filters, targets, avs=(0.0, 8.0))
    print(
        f"E2 cubes done on {len(used)} targets ({time.time() - t0:.0f} s)",
        flush=True,
    )
    runs = e2_teff_runs(filters, logg=4.5, feh=0.0)
    print(f"E2 teff runs done ({time.time() - t0:.0f} s)", flush=True)
    e2a = e2_av(
        filters, [(t, 4.5, 0.0) for t in (3000.0, 4500.0, 5800.0, 8000.0)]
    )
    print(f"E2 av done ({time.time() - t0:.0f} s)", flush=True)

    e3 = e3_rv(
        filters,
        [(t, 4.5, 0.0) for t in (3000.0, 4500.0, 5800.0, 8000.0)]
        + [(4500.0, 2.0, 0.0)],
    )
    print(f"E3 done ({time.time() - t0:.0f} s)", flush=True)

    fac = {n: f["facility"] for n, f in filters.items()}
    lam = {
        n: float(np.exp(np.average(LNW[f["idx"]], weights=f["prof"])))
        for n, f in filters.items()
    }
    json.dump(
        {
            "filters": {
                n: {"facility": fac[n], "lam_eff_A": lam[n]} for n in filters
            },
            "e1": e1,
            "e2_cubes": e2c,
            "e2_cube_targets": used,
            "e2_teff_runs": runs,
            "e2_av": [[k[0], k[1], k[2], v] for k, v in e2a.items()],
            "e3": [[k[0], k[1], v["rv"], v["law"]] for k, v in e3.items()],
        },
        open(OUT, "w"),
    )
    print("wrote", OUT, f"({time.time() - t0:.0f} s)")
    print("DONE")


if __name__ == "__main__":
    main()
