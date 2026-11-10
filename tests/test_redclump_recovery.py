"""Synthetic recovery through the red clump (review 8.9.11; slow).

A Baade's-window source and the clump at the bar's mean distance, truth
A_V = 4.0 and R_V = 2.0 for both (CCM89, the spectral path), photometry
from an INDEPENDENT numpy integration (``SpectralBC.bc_np``, not the
graph): the clump's calibrated I_RC and (V-I)_RC, Delta I (a same-filter
differential) and Delta(V-I) (a differential color row).  The MAP recovers
A_V, sed.rv_dust, A_I and E(V-I); with only the differential rows the
extinction is NOT measured (redclump.md, "What the clump measures").

The spectra are test_sed_fit_rv_dust's blackbody runtime grid, so the
numbers are about the machinery.  On the real R = 3000 grid (recorded in
the PR that added this) a NUTS run of the full arm gave A_V 4.01 +/- 0.16,
R_V 2.03 +/- 0.17, A_I 2.17 +/- 0.15 (truth 2.14), E(V-I) 1.730 +/- 0.067
(1.734), 0 divergences.
"""

import numpy as np
import pytensor
import pytest
import yaml
from scipy.optimize import brentq, minimize
from test_sed_fit_rv_dust import _bb_grid

from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.sed.spectral_bc import SpectralBC
from exozippy.components.star.star import Star
from exozippy.constants import LOG_L0_CONST, LOGG_CONST, LUM_CONST
from exozippy.models.NextGen import runtime_spectra as rs
from exozippy.system import System

pytestmark = pytest.mark.slow


def _grid():
    """Blackbodies on a 100 K Teff grid: the two-node default grid would
    interpolate a 4700 K clump between 2600 and 10000 K."""
    return _bb_grid(teffs=tuple(np.arange(2600.0, 10001.0, 100.0)))


V, I = "Generic/Johnson.V", "Generic/Cousins.I"
RA, DEC = 270.9, -30.0
AV, RV = 4.0, 2.0


def _mbol(r, t):
    return -2.5 * np.log10(LUM_CONST * r**2 * t**4) + LOG_L0_CONST


def _truth():
    spec = SpectralBC(_grid(), [V, I], law="ccm89")

    def bc(t, g, av):
        return spec.bc_np(t, g, 0.0, av, RV)  # [V, I]

    t_rc = brentq(
        lambda t: (bc(t, 2.5, 0)[1] - bc(t, 2.5, 0)[0]) - 1.06, 3000, 9000
    )
    l_deg, b_deg = k21.line_of_sight_basis(np.radians(RA), np.radians(DEC))[:2]
    d_rc = 10 ** (0.2 * k21.bar_distance_moments(l_deg, b_deg)[0] + 1)
    bc0 = bc(t_rc, 2.5, 0)
    r_rc = np.sqrt(
        10 ** (-0.4 * (-0.12 + bc0[1] - LOG_L0_CONST)) / (LUM_CONST * t_rc**4)
    )
    bcr = bc(t_rc, 2.5, AV)
    m_rc = _mbol(r_rc, t_rc) - bcr + 5 * np.log10(d_rc) - 5
    t_s, g_s, d_s = 5800.0, 4.4, 8300.0
    r_s = 10 ** ((LOGG_CONST - g_s) / 2)
    m_s = _mbol(r_s, t_s) - bc(t_s, g_s, AV) + 5 * np.log10(d_s) - 5
    return {
        "A_I": bc0[1] - bcr[1],
        "E_VI": (bc0[0] - bcr[0]) - (bc0[1] - bcr[1]),
        "rows": [
            (I, m_rc[1], 0.05, None, {"pos": ["RedClump"]}),
            (
                V,
                m_rc[0] - m_rc[1],
                0.03,
                ("neg_filter", I),
                {"pos": ["RedClump"], "neg": ["RedClump"]},
            ),
            (
                I,
                m_s[1] - m_rc[1],
                0.02,
                None,
                {"pos": ["Source"], "neg": ["RedClump"]},
            ),
            (
                V,
                (m_s[0] - m_s[1]) - (m_rc[0] - m_rc[1]),
                0.01,
                ("color_filter", I),
                {"pos": ["Source"], "neg": ["RedClump"]},
            ),
        ],
    }


def _map(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(
        rs, "load_runtime_grid", lambda model_root=None: _grid()
    )
    # the source is the survey-selected star the av tie applies to
    monkeypatch.setattr(Star, "selected_sources", lambda self, system: [0])
    filters = []
    for name, mag, err, extra, phot in rows:
        row = {"name": name, "mag": float(mag), "err": err, "photType": phot}
        if extra:
            row[extra[0]] = extra[1]
        filters.append(row)
    path = tmp_path / "rec.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": filters}))
    cfg = {
        "star": [{"name": "Source", "mist": False}],
        "sed": {"file": str(path), "fit_rv_dust": True},
        "galacticmodel": [{"anchor_idx": 0}],
        "redclump": [
            {"name": "RedClump", "bands": ["I"], "colors": [["V", "I"]]}
        ],
        "band": [
            {"name": "I", "filter": I, "star_constrains_ld": False},
            {"name": "V", "filter": V, "star_constrains_ld": False},
        ],
    }
    params = {
        "star.Source.ra": {"initval": RA, "sigma": 0},
        "star.Source.dec": {"initval": DEC, "sigma": 0},
        "sed.errscale": {"initval": 1.0, "sigma": 0},
        "star.Source.feh": {"initval": 0.0, "sigma": 0},
        "star.Source.logmass": {"initval": 0.0, "sigma": 0},
        "star.Source.distance": {"initval": 8300.0, "sigma": 0},
    }
    system = System(cfg, params)
    system.prepare()
    model = system.build_model()
    vv = model.value_vars
    ip = model.initial_point()
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
    rc, sed = system.redclump, system.sed
    outs = [
        rc.av.value[0],
        sed.rv_dust.value[0],
        rc.extinction.value[0],
        rc.color_excess.value[0],
    ]
    fn = pytensor.function(
        vv, model.replace_rvs_by_values(outs), on_unused_input="ignore"
    )
    point = unpack(res.x)
    return [float(x) for x in fn(*[point[v.name] for v in vv])]


def test_the_clump_measures_the_extinction_and_the_law(tmp_path, monkeypatch):
    """
    Given calibrated clump rows plus the source-minus-clump offsets,
    When the MAP is found,
    Then A_V, R_V, A_I and E(V-I) come back at the truth.
    """
    t = _truth()
    av, rv, a_i, e_vi = _map(tmp_path, monkeypatch, t["rows"])
    assert av == pytest.approx(AV, abs=0.3)
    assert rv == pytest.approx(RV, abs=0.3)
    assert a_i == pytest.approx(t["A_I"], abs=0.3)
    assert e_vi == pytest.approx(t["E_VI"], abs=0.1)


def test_differential_rows_alone_do_not_measure_it(tmp_path, monkeypatch):
    """With only Delta I and Delta(V-I) nothing calibrates the clump: its
    E(V-I) is not pinned at the truth (the extinction is prior/tie-held)."""
    t = _truth()
    _av, _rv, _a_i, e_vi = _map(tmp_path, monkeypatch, t["rows"][2:])
    assert abs(e_vi - t["E_VI"]) > 0.1
