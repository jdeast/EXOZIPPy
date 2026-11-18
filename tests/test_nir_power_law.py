"""The SED's extinction law under ``fit_rv_dust``: CCM89 + O'Donnell (1994)
blueward of 0.909 micron, joined continuously to a near-infrared power law
A_lam ~ lambda**-alpha redward, with alpha the sight line's ``sed.nir_alpha``
(review 8.9.2; sed.md, "The near-infrared slope"; sed/extinction.py).

The law-level tests are exact; the band-integrated ones use
test_sed_fit_rv_dust's blackbody runtime grid, so they pin the machinery
(which ratio alpha moves and which it leaves alone), not BT-NextGen's
numbers -- those are measured on the real grid and recorded in sed.md.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from test_sed_fit_rv_dust import _bb_grid, _system

from exozippy.components.sed import extinction as ext
from exozippy.components.sed.spectral_bc import SpectralBC

LAM = np.geomspace(0.1, 30.0, 4001)


# ------------------------------------------------------------- the law


@pytest.mark.parametrize("rv", [1.5, 1.96, 3.1, 6.0])
def test_alpha_1p61_is_ccm89_up_to_ccms_own_break_at_the_join(rv):
    """
    Given the law at alpha = 1.61 (CCM89's own infrared exponent),
    When it is compared with CCM89 + O'Donnell at the same R_V,
    Then blueward of the join it IS CCM (bit for bit), and redward it is
      CCM times one constant: the ratio of CCM's optical polynomial to its
      infrared segment at x = 1.1, where CCM itself is discontinuous (the
      new law starts from the optical side so that it is continuous).
    """
    new = ext.a_lambda_over_av_nir(LAM, rv, ext.CCM_NIR_ALPHA)
    old = ext.a_lambda_over_av_ccm(LAM, rv)
    blue = LAM < ext.NIR_JOIN_MICRON
    np.testing.assert_array_equal(new[blue], old[blue])
    a_o, b_o = ext._odonnell_optical_ab(ext.NIR_JOIN_X)
    ir = ext.NIR_JOIN_X**ext.CCM_NIR_ALPHA * (0.574 - 0.527 / rv)
    np.testing.assert_allclose(
        new[~blue] / old[~blue], (a_o + b_o / rv) / ir, rtol=1e-12
    )
    # ... and that constant is small: CCM's break is 0.05% at R_V 3.1.
    assert abs((a_o + b_o / rv) / ir - 1) < 4e-3


@pytest.mark.parametrize("alpha", [1.2, 1.61, 1.85, 3.0])
@pytest.mark.parametrize("rv", [1.5, 3.1, 6.0])
def test_the_law_is_continuous_at_the_join(alpha, rv):
    """The power law starts from the optical side's value at 0.909 micron:
    a relative step across the join of 1e-9 moves A_lam/A_V by < 1e-8."""
    lam = ext.NIR_JOIN_MICRON * np.array([1 - 1e-9, 1.0, 1 + 1e-9])
    k = ext.a_lambda_over_av_nir(lam, rv, alpha)
    np.testing.assert_allclose(k, k[1], rtol=1e-8)


def test_alpha_moves_only_redward_of_the_join_and_steepens_the_nir():
    """
    Given two alphas,
    Then the law blueward of the join is unchanged, and redward a larger
      alpha extincts less (A ~ (x/1.1)**alpha with x < 1.1) -- by exactly
      (x/1.1)**Delta(alpha).
    """
    lo = ext.a_lambda_over_av_nir(LAM, 2.0, 1.61)
    hi = ext.a_lambda_over_av_nir(LAM, 2.0, 2.2)
    blue = LAM <= ext.NIR_JOIN_MICRON
    np.testing.assert_array_equal(lo[blue], hi[blue])
    x = 1 / LAM[~blue]
    np.testing.assert_allclose(
        hi[~blue] / lo[~blue], (x / ext.NIR_JOIN_X) ** (2.2 - 1.61), rtol=1e-12
    )
    assert np.all(hi[~blue] < lo[~blue])


def test_the_mix_of_two_layers_sharing_alpha_is_exact_in_inverse_rv():
    """The dust profile's near/far R_V mix (1/R_eff = f/R_near + (1-f)/R_far)
    stays exact under the power law, because the law is still linear in
    1/R_V at fixed alpha and both layers share the sight line's alpha."""
    from exozippy.components.sed import dust_profile as dp

    a, b, L = ext.ccm89_nir_basis(LAM)
    rng = np.random.default_rng(5)
    for _ in range(50):
        a_near, a_far = rng.uniform(0.01, 3.0, 2)
        r_near, r_far = rng.uniform(1.6, 5.5, 2)
        alpha = rng.uniform(1.2, 3.0)
        direct = a_near * ext.a_lambda_over_av_nir(
            LAM, r_near, alpha
        ) + a_far * ext.a_lambda_over_av_nir(LAM, r_far, alpha)
        r_eff = dp.mixed_rv(a_near / (a_near + a_far), r_near, r_far)
        np.testing.assert_allclose(
            direct,
            (a_near + a_far) * ext.a_lambda_over_av_nir(LAM, r_eff, alpha),
            rtol=1e-12,
        )


def test_a_non_positive_wavelength_raises():
    with pytest.raises(ValueError, match="finite and positive"):
        ext.ccm89_nir_basis(np.array([0.5, 0.0]))


# ------------------------------------------------------------- the graph

FILTERS = [
    "Generic/Johnson.V",
    "Generic/Cousins.I",
    "2MASS/2MASS.J",
    "2MASS/2MASS.Ks",
]


@pytest.mark.parametrize("mode", [None, "JAX"])
def test_value_and_gradient_are_finite_at_the_join_and_the_bounds(mode):
    """
    Given the spectral BCs of V, I, J and Ks (I's passband straddles the
      0.909 micron join) at alpha on its bounds 1.2 and 3.0, CCM's 1.61 and
      the prior center 1.85, and R_V on its bounds 1.5 and 6.0,
    When the BCs and their gradients in R_V, alpha and A_V are compiled in
      the default (C, with the grid lookup's Python fallback) and the JAX
      backends,
    Then every value and gradient is finite, the backends agree, and the
      numpy twin agrees with the graph (no pt.where anywhere: the law is
      one product, CLAUDE.md's where-trap).
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    s = SpectralBC(_bb_grid(), FILTERS, law="ccm89_nir")
    t, g, z, a, r = (pt.dvector(n) for n in "tgzar")
    al = pt.dscalar("alpha")
    out = s.bc_node(t, g, z, a, r, al)
    total = pt.sum(out)
    grads = pytensor.grad(total, [r, al, a])
    fn = pytensor.function([t, g, z, a, r, al], [out] + grads, mode=mode)
    ref = pytensor.function([t, g, z, a, r, al], [out] + grads)
    for alpha in (1.2, 1.61, 1.85, 3.0):
        for rv in (1.5, 6.0):
            args = ([4700.0], [2.5], [0.0], [3.0], [rv], alpha)
            vals = fn(*args)
            for v in vals:
                assert np.all(np.isfinite(v)), (alpha, rv, v)
            for v, w in zip(vals, ref(*args)):
                np.testing.assert_allclose(v, w, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(
                vals[0][0],
                s.bc_np(4700.0, 2.5, 0.0, 3.0, rv, alpha),
                atol=1e-9,
            )


def test_alpha_must_be_one_scalar_and_is_required():
    s = SpectralBC(_bb_grid(), FILTERS, law="ccm89_nir")
    v = pt.dvector("v")
    with pytest.raises(ValueError, match="scalar"):
        s.bc_node(v, v, v, v, v, v)
    with pytest.raises(ValueError, match="needs alpha"):
        s.bc_node(v, v, v, v, v)
    with pytest.raises(ValueError, match="needs rv and alpha"):
        s.bc_np(4700.0, 2.5, 0.0, 3.0, 2.0)


def test_alpha_carries_the_nir_ratios_and_leaves_the_optical_one():
    """
    Given a 4700 K clump-like spectrum at A_V = 3 and R_V = 1.96,
    When alpha goes from CCM's 1.61 to 2.0,
    Then A_I/E(V-I) (R_V's ratio) moves by < 0.5% -- only the tail of the
      I passband reaches past the join -- while A_V/A_Ks rises by > 25%:
      the two law parameters separate, as Nataf et al. (2016)'s two
      independent principal components do.
    """
    s = SpectralBC(_bb_grid(teffs=(4600.0, 4800.0)), FILTERS)

    def ratios(alpha):
        e = s.bc_np(4700.0, 2.5, 0.0, 0.0, 1.96, alpha) - s.bc_np(
            4700.0, 2.5, 0.0, 3.0, 1.96, alpha
        )
        a_v, a_i, _, a_k = e
        return a_i / (a_v - a_i), a_v / a_k

    (ai_lo, vk_lo), (ai_hi, vk_hi) = ratios(1.61), ratios(2.0)
    assert abs(ai_hi / ai_lo - 1) < 5e-3
    assert vk_hi / vk_lo > 1.25


# ------------------------------------------------------------ the system


def test_fit_rv_dust_samples_one_nir_alpha_and_the_tables_do_not(
    tmp_path, monkeypatch
):
    """
    Given the same SED with and without fit_rv_dust,
    When the model is built,
    Then only the fit_rv_dust one samples sed.nir_alpha (one element, the
      N(1.85, 0.25) prior of sed/defaults.yaml), the predicted magnitudes
      depend on it, and the start logp and gradient are finite.
    """
    table = _system(tmp_path, monkeypatch, fit_rv_dust=False)
    assert "nir_alpha" not in table.sed.manifest

    system = _system(tmp_path, monkeypatch, fit_rv_dust=True)
    assert system.sed.manifest["nir_alpha"] is None
    model = system.build_model()
    free = [v.name for v in model.free_RVs if "nir_alpha" in v.name]
    assert free == ["sed.nir_alpha_raw"]
    cfg = system.config_manager.resolve("sed", "nir_alpha", element=0)
    assert float(np.atleast_1d(cfg["mu"])[0]) == 1.85
    assert float(np.atleast_1d(cfg["sigma"])[0]) == 0.25
    assert system.sed.nir_alpha_start() == pytest.approx(1.85)
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))
    with model:
        m = system.sed._predicted_appmag_node(system)
    assert system.sed.nir_alpha.value in pytensor.graph.traversal.ancestors(
        [m]
    )


def test_nir_alpha_counts_once_toward_the_errscale_pin_unless_pinned(
    tmp_path, monkeypatch
):
    """sed.nir_alpha is one sight-line coordinate in the errscale count, as
    sed.rv_dust is; pinned (sigma: 0) it is not counted."""
    system = _system(tmp_path, monkeypatch, fit_rv_dust=True)
    n_free = system.sed.data_free_inventory(system).n_free
    pinned = _system(
        tmp_path,
        monkeypatch,
        fit_rv_dust=True,
        params={"sed.nir_alpha": {"initval": 1.61, "sigma": 0}},
    )
    assert pinned.sed.data_free_inventory(pinned).n_free == n_free - 1
    assert pinned.sed.nir_alpha_start() == pytest.approx(1.61)


@pytest.mark.slow  # two lens/source/clump builds: ~30 s on the dev box
def test_a_ks_anchor_converts_to_av_through_the_fitted_alpha(
    tmp_path, monkeypatch
):
    """
    Given the lens/source/clump sight line under fit_rv_dust with
      dust_anchor_band: 2MASS Ks (a near-IR map's value as the anchor),
    When sed.av_anchor is evaluated at the start, at alpha 1.85 and 1.61,
    Then each inverts the reference spectrum's band-integrated A_Ks AT THAT
      ALPHA to 1e-5, and the steeper law puts more A_V behind the same A_Ks
      (A_V/A_Ks grows with alpha): the map value needs no hand conversion.
    """
    from test_dust_profile import _lens_params
    from test_dust_profile import _system as _sight_line

    from exozippy.models.NextGen import runtime_spectra as rs

    monkeypatch.setattr(
        rs, "load_runtime_grid", lambda model_root=None: _bb_grid()
    )

    def av_anchor(alpha, sub):
        system = _sight_line(
            tmp_path / sub,
            sed={"fit_rv_dust": True, "dust_anchor_band": "2MASS/2MASS.Ks"},
            params=_lens_params(
                **{"sed.nir_alpha": {"initval": alpha, "sigma": 0}}
            ),
        )
        model = system.build_model()
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values(
                [system.sed.dust_anchor.value, system.sed.av_anchor.value]
            ),
            on_unused_input="ignore",
        )
        ip = model.initial_point()
        a_ks, av = fn(*[ip[v.name] for v in model.value_vars])
        got = system.sed.reference_band_extinction_np(
            "2MASS/2MASS.Ks", float(av[0]), system
        )
        np.testing.assert_allclose(got, float(a_ks[0]), rtol=1e-5)
        return float(av[0]) / float(a_ks[0])

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert av_anchor(1.85, "a") > 1.1 * av_anchor(1.61, "b")
