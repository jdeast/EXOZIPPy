"""Synthetic recovery through the sight-line dust profile (review 8.9.8;
slow).

Baade's window: a lens (an M dwarf at 3 kpc), a source (8.5 kpc) and the red
clump at the bar.  The TRUE extinction is the smooth layer (A_V = 2.0 at the
clump) plus a CLOUD of A_V = 1.5 at 2 kpc -- in front of the lens, which the
smooth layer alone (the literature standard, Bennett et al. 2015) cannot
represent: it would put the lens at s(d_L)/s(d_S) ~ 0.74 of the source's
A_V.  Photometry is generated through the SED's numpy twin (``appmag_np``,
scipy's interpolator on the tables, not the graph): the clump's calibrated
I and (V-I), and resolved (HST-like) V, I and H of the lens and the source,
with every stellar parameter but the extinction pinned at the truth -- the
question is the extinction alone.  The MAP of the default profile (K = 3
bins) recovers A_V(d_L) and A_V(d_S) with every object's scatter coordinate
at ~0 (measured: lens 2.9784 vs truth 2.9785, source 3.5129 vs 3.5129); the
smooth layer (dust_bins: 1) reaches the same A_V values only by pulling each
object's av_frac ~2.3 sigma (+/-0.07 against 0.03).
"""

import numpy as np
import pytensor
import pytest
import yaml
from scipy.optimize import minimize

from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.sed import dust_profile as dp
from exozippy.constants import LOG_L0_CONST, LOGG_CONST, LUM_CONST
from exozippy.system import System

pytestmark = pytest.mark.slow

V, I, H = "Generic/Johnson.V", "Generic/Cousins.I", "2MASS/2MASS.H"
RA, DEC = 270.9, -30.0
LAYER_RC, CLOUD, D_CLOUD = 2.0, 1.5, 2000.0
STARS = {
    # name: teff, logg, radius (Rsun), distance (pc)
    "Source": (5800.0, 4.4, 1.0, 8500.0),
    "Lens": (3700.0, 4.75, 0.45, 3000.0),
}
RC = (4684.0, 2.5, 0.0, -0.12)  # teff, logg, feh, M_I


def _sight():
    l_deg, b_deg = k21.line_of_sight_basis(np.radians(RA), np.radians(DEC))[:2]
    d_rc = 10 ** (0.2 * k21.bar_distance_moments(l_deg, b_deg)[0] + 1)
    return dp.sinb_of(b_deg), d_rc


def true_av(d):
    sinb, d_rc = _sight()
    s = dp.layer_fraction(np.asarray(d, float), sinb, 164.0, xp=np)
    s_rc = dp.layer_fraction(d_rc, sinb, 164.0, xp=np)
    return LAYER_RC * s / s_rc + CLOUD * (np.asarray(d) > D_CLOUD)


def _cfg(path, sed=None):
    return {
        "star": [{"name": n, "mist": False} for n in STARS],
        "sed": {"file": str(path), **(sed or {})},
        "galacticmodel": [{"anchor_idx": 0}],
        "redclump": [{"name": "RedClump", "bands": ["I"]}],
        "band": [{"name": "I", "filter": I, "star_constrains_ld": False}],
    }


def _params():
    p = {"sed.errscale": {"initval": 1.0, "sigma": 0}}
    for n, (t, g, r, d) in STARS.items():
        logm = np.log10(10 ** (g - LOGG_CONST) * r**2)
        p.update(
            {
                f"star.{n}.ra": {"initval": RA, "sigma": 0},
                f"star.{n}.dec": {"initval": DEC, "sigma": 0},
                f"star.{n}.teffsed": {"initval": t, "sigma": 0},
                f"star.{n}.radiussed": {"initval": r, "sigma": 0},
                f"star.{n}.logmass": {"initval": logm, "sigma": 0},
                f"star.{n}.feh": {"initval": 0.0, "sigma": 0},
                f"star.{n}.distance": {"initval": d, "sigma": 0},
            }
        )
    _, d_rc = _sight()
    t, g, feh, mi = RC
    p.update(
        {
            "redclump.RedClump.teff": {"initval": t, "sigma": 0},
            "redclump.RedClump.feh": {"initval": feh, "sigma": 0},
            "redclump.RedClump.mi": {"initval": mi, "sigma": 0},
            "redclump.RedClump.distance": {"initval": d_rc, "sigma": 0},
        }
    )
    return p


def _rows(sed):
    """The truth's photometry through the SED's numpy twin."""
    _, d_rc = _sight()
    rows = []
    for n, (t, g, r, d) in STARS.items():
        av = float(true_av(d))
        for f in (V, I, H):
            m = sed.appmag_np(f, t, g, 0.0, av, r, d)
            rows.append(
                {"name": f, "mag": m, "err": 0.02, "photType": {"pos": [n]}}
            )
    t, g, feh, mi = RC
    bc_i0 = sed.bc_np(I, t, g, feh, 0.0)
    r_rc = np.sqrt(
        10 ** (-0.4 * (mi + bc_i0 - LOG_L0_CONST)) / (LUM_CONST * t**4)
    )
    av_rc = float(true_av(d_rc))
    m_i = sed.appmag_np(I, t, g, feh, av_rc, r_rc, d_rc)
    m_v = sed.appmag_np(V, t, g, feh, av_rc, r_rc, d_rc)
    rows += [
        {
            "name": I,
            "mag": m_i,
            "err": 0.05,
            "photType": {"pos": ["RedClump"]},
        },
        {
            "name": V,
            "mag": m_v - m_i,
            "err": 0.03,
            "neg_filter": I,
            "photType": {"pos": ["RedClump"], "neg": ["RedClump"]},
        },
    ]
    return [{**r, "mag": float(r["mag"])} for r in rows]


def _build(tmp_path, rows, sed=None):
    path = tmp_path / "rec.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))
    system = System(_cfg(path, sed), _params())
    system.prepare()
    return system


def map_avs(tmp_path, sed=None):
    """(A_V lens, A_V source, A_V clump) at the MAP, and the truth."""
    seed_rows = [
        {"name": f, "mag": 15.0, "err": 0.05, "photType": {"pos": [n]}}
        for n in list(STARS) + ["RedClump"]
        for f in (V, I, H)
    ]
    probe = _build(tmp_path, seed_rows, sed)
    rows = _rows(probe.sed)
    system = _build(tmp_path, rows, sed)
    model = system.build_model()
    vv = model.value_vars
    ip = model.initial_point()
    ip.update(system.get_raw_start(model))
    shapes = [np.shape(ip[v.name]) for v in vv]
    sizes = [int(np.prod(s)) for s in shapes]
    lp, dlp = model.compile_logp(), model.compile_dlogp()

    def unpack(x):
        out, i = {}, 0
        for v, sh, n in zip(vv, shapes, sizes):
            out[v.name] = np.reshape(x[i : i + n], sh)
            i += n
        return out

    x0 = np.concatenate([np.ravel(ip[v.name]) for v in vv])
    res = minimize(
        lambda x: -lp(unpack(x)),
        x0,
        jac=lambda x: -dlp(unpack(x)),
        method="L-BFGS-B",
        options={"maxiter": 20000, "maxfun": 40000},
    )
    fn = pytensor.function(
        vv,
        model.replace_rvs_by_values(
            [
                system.star.av.value,
                system.redclump.av.value,
                system.star.av_frac.value,
                system.redclump.av_frac.value,
                system.sed.dust_anchor.value,
            ]
        ),
        on_unused_input="ignore",
    )
    av, av_rc, frac, frac_rc, anchor = fn(*[unpack(res.x)[v.name] for v in vv])
    names = list(system.star.names)
    _, d_rc = _sight()
    truth = {n: float(true_av(STARS[n][3])) for n in STARS}
    truth["RedClump"] = float(true_av(d_rc))
    got = {n: float(av[names.index(n)]) for n in STARS}
    got["RedClump"] = float(av_rc[0])
    fracs = np.concatenate([np.ravel(frac), np.ravel(frac_rc)])
    return got, truth, float(-res.fun), fracs


def test_the_profile_recovers_a_cloud_in_front_of_the_lens(tmp_path):
    """
    Given the cloud truth and the default profile (K = 3 bins),
    When the MAP is found,
    Then A_V(lens) and A_V(source) are recovered to 0.1 mag, and the lens's
      A_V is below the source's (monotone by construction).
    """
    got, truth, _, fracs = map_avs(tmp_path)
    assert np.all(np.abs(fracs) < 0.01)  # the bins hold the cloud
    assert got["Lens"] == pytest.approx(truth["Lens"], abs=0.1)
    assert got["Source"] == pytest.approx(truth["Source"], abs=0.1)
    assert got["Lens"] < got["Source"]


def test_the_smooth_layer_cannot_put_the_cloud_there(tmp_path):
    """
    Given the same data and the smooth layer alone (dust_bins: 1, the
    Bennett et al. 2015 form normalized at the clump),
    When the MAP is found,
    Then the layer reaches the lens's A_V only by pulling every object's
      own scatter coordinate av_frac about 2 sigma off zero (|av_frac| >
      0.05 against its 0.03 prior; measured +/-0.07): its fixed A_L/A_S
      cannot hold a cloud in front of the lens, and a larger cloud than the
      scatter can absorb is simply missed.
    """
    _got, _truth, _, fracs = map_avs(tmp_path, sed={"dust_bins": 1})
    assert np.max(np.abs(fracs)) > 0.05


if __name__ == "__main__":  # the numbers for the PR body
    import pathlib
    import sys
    import tempfile

    root = pathlib.Path(
        tempfile.mkdtemp(dir=sys.argv[1] if len(sys.argv) > 1 else None)
    )
    for tag, sed in (("K=3 profile", None), ("K=1 layer", {"dust_bins": 1})):
        (root / tag[:3]).mkdir()
        got, truth, lp, fracs = map_avs(root / tag[:3], sed)
        print(tag, got, truth, lp, fracs, flush=True)
