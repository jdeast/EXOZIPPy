"""The SED's fitted-R_V mode (``sed: {fit_rv_dust: true}``; sed.md, "The
spectral mode").

The BCs come from the spectrum at every evaluation -- the R = 3000 runtime
grid (models/NextGen/runtime_spectra.py) interpolated in log flux, reddened
with CCM89 at each star's ``star.rv_dust``, integrated through each filter
exactly as the tables are (components/sed/spectral_bc.py).  The real grid
is 656 MB; these tests use a synthetic one of blackbody spectra on the
same R = 3000 bins (``_bb_grid``), so the numbers they pin are about the
machinery, not BT-NextGen.  The real grid's agreement with the shipped
tables is checked by scripts/validate_runtime_grid.py and recorded in
models/NextGen/README.md.
"""

import types

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.sed import extinction
from exozippy.components.sed.sed import fit_rv_dust_from_config
from exozippy.components.sed.spectral_bc import SpectralBC
from exozippy.models.NextGen import runtime_spectra as rs
from exozippy.models.NextGen.bolometric_correction import BolometricCorrection
from exozippy.models.NextGen.nextgen_spectra import WAVELENGTH_PTS
from exozippy.system import System

WAVE = np.asarray(WAVELENGTH_PTS, float)


def _bb(teff):
    """Surface F_lambda of a blackbody, erg/s/cm^2/A, on the step-1 grid."""
    h, c, k = 6.62607e-27, 2.99792e10, 1.380649e-16
    lam = WAVE * 1e-8
    with np.errstate(over="ignore"):
        return (
            np.pi * 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * teff))
        ) * 1e-8


def _bb_grid(teffs=(2600.0, 10000.0), loggs=(0.0, 5.5), fehs=(-4.0, 0.5)):
    """A RuntimeGrid stand-in: blackbodies, binned exactly as the real grid
    is, the same spectrum at every (logg, feh)."""
    edges = rs.bin_edges()
    lf = np.zeros(
        (len(teffs), len(loggs), len(fehs), len(edges) - 1), np.float32
    )
    for i, t in enumerate(teffs):
        fbar = rs.bin_spectrum(_bb(t), edges)
        with np.errstate(divide="ignore"):
            lf[i] = np.where(fbar > 0, np.log10(fbar), rs.LOG10_FLOOR)
    return types.SimpleNamespace(
        axes={
            "teff": np.asarray(teffs, float),
            "logg": np.asarray(loggs, float),
            "feh": np.asarray(fehs, float),
        },
        edges=edges,
        logflux=lf,
    )


FILTERS = [
    "2MASS/2MASS.J",
    "2MASS/2MASS.Ks",
    "GAIA/GAIA2r.G",
    "Generic/Bessell.V",
    "Generic/Bessell.B",
]


# ------------------------------------------------------------- the law


def test_ccm89_matches_the_published_ratios_and_is_linear_in_inverse_rv():
    """
    Given CCM89 + O'Donnell 94 at R_V = 3.1,
    When it is evaluated at V, B, J and Ks,
    Then A_lam/A_V is 1 at x = 1.82, ~1.32 at B, 0.282 at J and 0.113 at
      2.2 micron (CCM's Table 3), and A_lam/A_V at any R_V is a + b/R_V.
    """
    f = extinction.a_lambda_over_av_ccm
    assert f(np.array([1 / 1.82]), 3.1)[0] == pytest.approx(1.0, abs=1e-12)
    assert f(np.array([0.44]), 3.1)[0] == pytest.approx(1.322, abs=0.005)
    assert f(np.array([1.25]), 3.1)[0] == pytest.approx(0.282, abs=0.002)
    assert f(np.array([2.2]), 3.1)[0] == pytest.approx(0.113, abs=0.002)
    a, b = extinction.ccm89_ab(np.array([0.44, 1.25]))
    for rv in (2.5, 4.0):
        np.testing.assert_allclose(f(np.array([0.44, 1.25]), rv), a + b / rv)


def test_ccm89_extends_the_ir_power_law_past_3_3_micron():
    """Redward of CCM's range the extinction keeps falling (W2-W4, IRAC),
    rather than being held at its 3.3 micron value."""
    v = extinction.a_lambda_over_av_ccm(np.array([3.3, 4.6, 12.0, 22.0]), 3.1)
    assert np.all(np.diff(v) < 0)
    assert v[-1] < 0.01


# ---------------------------------------------------------- the binning


def test_bin_integrals_conserve_flux_exactly():
    """The bins partition the full-resolution trapezoid integral: they sum
    to it, and a bin whose edges are sample points holds exactly the
    trapezoid integral between them."""
    w = np.linspace(1.0, 10.0, 1000)
    y = w**2
    edges = np.array([w[0], w[300], w[701], w[-1]])
    got = rs.bin_integrals(w, y, edges)
    trap = lambda i, j: float(
        np.sum(0.5 * (y[i + 1 : j + 1] + y[i:j]) * np.diff(w[i : j + 1]))
    )
    np.testing.assert_allclose(
        got, [trap(0, 300), trap(300, 701), trap(701, 999)], rtol=1e-12
    )
    assert got.sum() == pytest.approx(trap(0, 999), rel=1e-12)


# ------------------------------------------------------- the BC itself


def test_the_shipped_law_reproduces_bolometric_correction_at_a_node():
    """
    Given a grid node's spectrum and the tables' own extinction law,
    When SpectralBC computes its BCs from the R = 3000 bins,
    Then they equal BolometricCorrection's from the R = 20000 spectrum (the
      generator of the shipped tables) to well under 1 mmag, A_V 0-20.
    """
    s = SpectralBC(_bb_grid(), FILTERS, law="shipped")
    for t in (2600.0, 10000.0):
        for av in (0.0, 8.0, 20.0):
            got = s.bc_np(t, 4.5, 0.0, av)
            ref = BolometricCorrection(
                FILTERS,
                {"teff": t, "logg": 4.5, "feh": 0.0, "av": av},
                spectrum=(0.0, _bb(t)),
            ).BC
            assert np.max(np.abs(got - ref)) < 1e-4, (t, av, got - ref)


def test_the_graph_matches_numpy_and_has_a_gradient_in_rv():
    """
    Given the CCM89 law,
    When the PyTensor node and the numpy twin are evaluated off-node,
    Then they agree, R_V moves every band, and d/dR_V is finite with the
      signs of a greyer law: at fixed A_V a larger R_V extincts less
      blueward of V (B brighter, BC up) and more redward (J fainter).
    """
    s = SpectralBC(_bb_grid(), FILTERS, law="ccm89")
    t, g, z, a, r = (pt.dvector(n) for n in "tgzar")
    out = s.bc_node(t, g, z, a, r)
    fn = pytensor.function(
        [t, g, z, a, r], [out, pt.grad(out[0, 4], r), pt.grad(out[0, 0], r)]
    )
    bc, dB, dJ = fn([5000.0], [4.3], [-0.2], [3.0], [3.1])
    np.testing.assert_allclose(
        bc[0], s.bc_np(5000.0, 4.3, -0.2, 3.0, 3.1), atol=1e-9
    )
    assert np.isfinite(dB).all() and dB[0] > 0
    assert np.isfinite(dJ).all() and dJ[0] < 0
    other = s.bc_np(5000.0, 4.3, -0.2, 3.0, 2.5)
    assert np.all(np.abs(other - bc[0]) > 1e-3)


def test_fit_rv_is_a_yaml_boolean():
    assert fit_rv_dust_from_config({}) is False
    assert fit_rv_dust_from_config({"fit_rv_dust": True}) is True
    assert fit_rv_dust_from_config([{"fit_rv_dust": False}]) is False
    with pytest.raises(ValueError, match="YAML boolean"):
        fit_rv_dust_from_config({"fit_rv_dust": "yes"})


# ------------------------------------------------------------ the fit

ROWS = [
    ("2MASS/2MASS.J", 8.000, 0.020),
    ("2MASS/2MASS.H", 7.800, 0.026),
    ("2MASS/2MASS.Ks", 7.750, 0.018),
    ("WISE/WISE.W1", 7.700, 0.030),
    ("WISE/WISE.W2", 7.720, 0.030),
    ("GAIA/GAIA2r.G", 9.300, 0.010),
]


def _system(
    tmp_path,
    monkeypatch,
    fit_rv_dust,
    n_rows=6,
    stars=("A",),
    pos=None,
    params=None,
):
    monkeypatch.setattr(
        rs, "load_runtime_grid", lambda model_root=None: _bb_grid()
    )
    lines = ["model: NextGen", "filters:"]
    for name, mag, err in ROWS[:n_rows]:
        lines += [
            f"  - name: {name}",
            f"    mag: {mag}",
            f"    err: {err}",
            "    magsys: Vega",
        ]
        if pos is not None:
            lines += ["    photType:", f"      pos: [{pos}]"]
    sed = tmp_path / "star.sed"
    sed.write_text("\n".join(lines) + "\n")
    config = {
        "star": [{"name": s, "mist": False} for s in stars],
        "sed": {
            "file": str(sed),
            **({"fit_rv_dust": True} if fit_rv_dust else {}),
        },
    }
    system = System(config, params or {})
    system.prepare()
    return system


def test_fit_rv_declares_rv_dust_and_table_mode_does_not(
    tmp_path, monkeypatch
):
    """
    Given the same SED with and without fit_rv_dust,
    When the model is built,
    Then only the fit_rv_dust one samples star.rv_dust, its log-probability and
      gradient are finite at the start, and its predicted magnitudes move
      with R_V.
    """
    table = _system(tmp_path, monkeypatch, fit_rv_dust=False)
    assert "rv_dust" not in table.star.manifest
    model = table.build_model()
    assert not any("rv_dust" in v.name for v in model.free_RVs)

    system = _system(tmp_path, monkeypatch, fit_rv_dust=True)
    assert "rv_dust" in system.star.manifest
    model = system.build_model()
    assert any("rv_dust" in v.name for v in model.free_RVs)
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))
    with model:
        m = system.sed._predicted_appmag_node(system)
    assert system.star.rv_dust.value in pytensor.graph.traversal.ancestors([m])


def test_one_sight_line_rv_is_counted_once_and_inactive_on_an_unread_star(
    tmp_path, monkeypatch
):
    """
    Given fit_rv_dust on a two-star system whose rows name only A,
    When the inventory is taken and the model built,
    Then R_V is ONE free coordinate, sed.rv_dust (counted once toward the
      errscale pin: 1 + 4 + 1 = 6 rows leave errscale free), not a per-star
      knob; A's star.rv_dust is derived from it and B's is inactive.
    """
    system = _system(
        tmp_path, monkeypatch, fit_rv_dust=True, stars=("A", "B"), pos="A"
    )
    inv = system.sed.data_free_inventory(system)
    assert "rv_dust" not in inv.star_knobs
    assert inv.n_free == 1 + 4 + 1
    assert not inv.pin_errscale
    assert inv.unread_stars == (1,)
    model = system.build_model()
    free = [v.name for v in model.free_RVs if "rv_dust" in v.name]
    assert free == ["sed.rv_dust_raw"]
    rv = system.star.rv_dust
    assert rv.element_is_derived(0) and not rv.element_is_active(1)
    assert system.sed.manifest["rv_dust"] is None  # sampled, on the SED


def test_two_read_stars_share_one_rv(tmp_path, monkeypatch):
    """Two stars the rows read: still exactly one R_V dimension, and both
    stars' star.rv_dust equal it."""
    system = _system(
        tmp_path, monkeypatch, fit_rv_dust=True, stars=("A", "B"), pos="A, B"
    )
    model = system.build_model()
    assert [v.name for v in model.free_RVs if "rv_dust" in v.name] == [
        "sed.rv_dust_raw"
    ]
    star_rv, sight = _at(
        model,
        [system.star.rv_dust.value, system.sed.rv_dust.value],
        model.initial_point(),
    )
    np.testing.assert_array_equal(star_rv, np.repeat(sight, 2))


def test_a_prior_on_star_rv_dust_raises_naming_sed_rv_dust(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="sed.rv_dust"):
        _system(
            tmp_path,
            monkeypatch,
            fit_rv_dust=True,
            params={"star.A.rv_dust": {"mu": 2.5, "sigma": 0.2}},
        )


def test_an_initval_on_a_star_seeds_the_sight_line(tmp_path, monkeypatch):
    system = _system(
        tmp_path,
        monkeypatch,
        fit_rv_dust=True,
        params={"star.A.rv_dust": {"initval": 2.2}},
    )
    cfg = system.config_manager.resolve("sed", "rv_dust", element=0)
    assert float(np.atleast_1d(cfg["initval"])[0]) == pytest.approx(2.2)


def test_a_repeated_filter_builds_and_reads_one_spectral_column(
    tmp_path, monkeypatch
):
    """
    Given fit_rv_dust and two rows in the SAME filter (a color row's second
    filter repeats a row's, the common clump case),
    When the model is built,
    Then the spectral BCs are integrated once per distinct filter and both
    rows read that one column (it used to raise: "names the same SVO
    filter twice").
    """
    monkeypatch.setattr(
        rs, "load_runtime_grid", lambda model_root=None: _bb_grid()
    )
    rows = [(n, m, e) for n, m, e in ROWS] + [("2MASS/2MASS.J", 8.01, 0.03)]
    lines = ["model: NextGen", "filters:"]
    for name, mag, err in rows:
        lines += [f"  - name: {name}", f"    mag: {mag}", f"    err: {err}"]
    sed = tmp_path / "rep.sed"
    sed.write_text("\n".join(lines) + "\n")
    system = System(
        {
            "star": [{"name": "A", "mist": False}],
            "sed": {"file": str(sed), "fit_rv_dust": True},
        },
        {},
    )
    system.prepare()
    assert system.sed._spectral_take is not None
    model = system.build_model()
    with model:
        m = system.sed._predicted_appmag_node(system)
    got = _at(model, m, model.initial_point())
    np.testing.assert_array_equal(got[:, 0], got[:, len(ROWS)])


def _at(model, outs, point):
    """Evaluate ``outs`` at a point dict of the model's value variables."""
    import pytensor

    fn = pytensor.function(model.value_vars, outs, on_unused_input="ignore")
    return fn(*[point[v.name] for v in model.value_vars])
