"""The microlensing zeropoint's magnitude system (issue #313) and the
zeropoint / structural-prior rules of review 2.2.21.

#313: ``mulensinstrument.zeropoint`` used to carry no magnitude system while
the SED's BC grid is Vega, so an AB zeropoint (the DC2018 challenge's 22.0)
was silently fitted as Vega -- wrong by m_AB - m_Vega in the band.  A light
curve now has a ``magsys:`` (exact "Vega"/"AB", the SED rows' parser);
unstated means the band filter's native system; the SED-predicted magnitude
is moved onto the zeropoint's system by the BC column record's
``ab_minus_vega`` in ONE place (``_predicted_mag_on_zp_system``).

2.2.21: the zeropoint has no default mu/sigma (the tie is the user's to
state).  The same review's structural priors are tests/test_structural_priors.py.

The end-to-end vehicle is ``examples/KMT-2019-BLG-1806`` (three Cousins-I
light curves, an SED block with no catalog rows), whose band filter is
Vega-native.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"
_INSTS = ("KMTC04", "KMTS04", "KMTA04")


def _kmt_inputs():
    with open(_KMT_DIR / "KMT-2019-BLG-1806.yaml") as f:
        config = yaml.safe_load(f)
    with open(_KMT_DIR / config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    return config, user_params


def _construct(config, user_params):
    """System.__init__ only (the SED reads its .sed file relative to cwd)."""
    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        return System(config, user_params=user_params)
    finally:
        os.chdir(cwd)


def _kmt_system(magsys=None, zp=None, build=True):
    """KMT-2019-BLG-1806 with every light curve's ``magsys`` set to
    ``magsys`` (None = unstated) and the broadcast zeropoint entry ``zp``
    (None = no entry)."""
    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    config, user_params = _kmt_inputs()
    if magsys is not None:
        for entry in config["mulensinstrument"]:
            entry["magsys"] = magsys
    if zp is not None:
        user_params["mulensinstrument.zeropoint"] = dict(zp)
    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model() if build else None
    finally:
        os.chdir(cwd)
    return system, model


def _eval(model, node, point):
    """Evaluate ``node`` at ``point`` (RVs replaced by their values, so the
    function reads the point instead of drawing from the prior)."""
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def _cousins_i_offset(system):
    """m_AB - m_Vega of the BC column the KMT band reads, from the record."""
    from exozippy.components.sed.magsys import read_magsys_table

    record = read_magsys_table(system.sed.model_root, system.sed.sedmodel)
    return float(record.loc["Cousins_I", "ab_minus_vega"])


# ---------------------------------------------------------------------------
# #313: the zeropoint's magnitude system
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ab_and_vega_twins():
    """Given the same event twice -- once with an AB zeropoint mu = 0, once
    with the SAME calibration written in Vega (unstated = Cousins I's native
    Vega, mu = 0 - (m_AB - m_Vega)) -- both tied with sigma 0.2."""
    ab_sys, ab_model = _kmt_system(magsys="AB", zp={"mu": 0.0, "sigma": 0.2})
    offset = _cousins_i_offset(ab_sys)
    vega_sys, vega_model = _kmt_system(
        magsys=None, zp={"mu": -offset, "sigma": 0.2}
    )
    return ab_sys, ab_model, vega_sys, vega_model, offset


def test_ab_zeropoint_is_converted_by_exactly_ab_minus_vega(
    ab_and_vega_twins,
):
    """
    Given the AB / Vega twins,
    When both zeropoint vectors are evaluated at ONE point,
    Then the AB zeropoint exceeds the Vega one by exactly the record's
    m_AB - m_Vega on every light curve (a zeropoint in AB reads the SED's
    Vega prediction plus the offset).  On master `magsys:` was read by
    nothing, so the difference was 0.
    """
    ab_sys, ab_model, vega_sys, vega_model, offset = ab_and_vega_twins
    assert offset > 0.3  # Cousins I: ~0.43 mag, so the test has teeth
    point = vega_model.initial_point()
    zp_ab = np.atleast_1d(
        _eval(ab_model, ab_model["mulensinstrument.zeropoint"], point)
    )
    zp_vega = np.atleast_1d(
        _eval(vega_model, vega_model["mulensinstrument.zeropoint"], point)
    )
    assert zp_ab.shape == (3,)
    np.testing.assert_allclose(zp_ab - zp_vega, offset, rtol=0, atol=1e-12)
    assert ab_sys.mulensinstrument.zp_magsys == ["AB"] * 3
    np.testing.assert_array_equal(
        ab_sys.mulensinstrument.zp_ab_minus_vega, [offset] * 3
    )


def test_ab_and_equivalent_vega_calibration_build_the_same_fit(
    ab_and_vega_twins,
):
    """
    Given the AB / Vega twins (one calibration, two spellings),
    When each model's start logp is evaluated at its own start,
    Then they agree: the conversion reaches the source-flux SEEDING (which
    reads the zeropoint mu) exactly as it reaches the tie, so the two
    spellings start the source at the same star and score it the same.
    """
    ab_sys, ab_model, vega_sys, vega_model, _ = ab_and_vega_twins
    for p in ("teff", "radius", "logmass"):
        np.testing.assert_allclose(
            np.atleast_1d(getattr(ab_sys.star, p).initval),
            np.atleast_1d(getattr(vega_sys.star, p).initval),
            rtol=1e-12,
        )
    lp_ab = float(ab_model.compile_logp()(ab_model.initial_point()))
    lp_vega = float(vega_model.compile_logp()(vega_model.initial_point()))
    assert np.isfinite(lp_ab)
    assert lp_ab == pytest.approx(lp_vega, rel=1e-12)


def test_unstated_magsys_is_the_band_filters_native_system(
    ab_and_vega_twins,
):
    """
    Given a light curve that states no magsys on a Cousins I band,
    When the zeropoint systems are resolved,
    Then each is the filter's native system (Vega) and nothing is converted.
    """
    _, _, vega_sys, _, _ = ab_and_vega_twins
    assert vega_sys.mulensinstrument.zp_magsys == ["Vega"] * 3
    np.testing.assert_array_equal(
        vega_sys.mulensinstrument.zp_ab_minus_vega, 0.0
    )


def test_startup_table_unit_names_the_zeropoint_system(ab_and_vega_twins):
    """
    Given the AB and Vega builds,
    When the zeropoint Parameter renders its unit for the startup table,
    Then the system is printed next to the unit on every element.
    """
    ab_sys, _, vega_sys, _, _ = ab_and_vega_twins
    zp_ab = ab_sys.mulensinstrument.zeropoint
    zp_vega = vega_sys.mulensinstrument.zeropoint
    assert [zp_ab.get_unit_str(i) for i in range(3)] == ["mag (AB)"] * 3
    assert [zp_vega.get_unit_str(i) for i in range(3)] == ["mag (Vega)"] * 3


def test_the_tie_and_its_ab_conversion_are_declared_in_the_draft(
    ab_and_vega_twins,
):
    """
    Given the AB / Vega twins (both tied),
    When the modeling-draft sentences are read,
    Then both declare the zeropoint-tie sentence, and only the AB build
    adds the AB conversion clause citing Oke & Gunn (1983).
    """
    ab_sys, _, vega_sys, _, _ = ab_and_vega_twins
    key = "mulensinstrument.zeropoint_tie"

    def text(system):
        (s,) = [x for x in system.prose.sentences() if x.key == key]
        return s.text

    assert "KMTC04" in text(ab_sys) and "Oke:1983" in text(ab_sys)
    assert "KMTC04" in text(vega_sys) and "AB system" not in text(vega_sys)


def test_an_untied_fit_declares_no_tie_sentence():
    """
    Given the KMT example as shipped (no zeropoint stated, so no tie),
    When the model is built,
    Then the draft carries no zeropoint-tie sentence.
    """
    system, _ = _kmt_system()
    assert "mulensinstrument.zeropoint_tie" not in system.prose


def test_resolution_is_logged_per_light_curve(caplog):
    """
    Given an AB-stated light curve set,
    When the system is prepared,
    Then each light curve logs its zeropoint system and the offset applied.
    """
    with caplog.at_level(logging.INFO):
        _kmt_system(magsys="AB", zp={"mu": 0.0, "sigma": 0.2}, build=False)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "zeropoint magnitude system" in r.getMessage()
    ]
    assert len(lines) == 3, lines
    for name in _INSTS:
        (line,) = [ln for ln in lines if f"mulensinstrument {name}:" in ln]
        assert "AB (stated" in line and "m_AB - m_Vega = +0.4" in line


@pytest.mark.parametrize("spelling, hint", [("ab", "AB"), ("VEGA", "Vega")])
def test_case_variant_magsys_raises_with_a_hint(spelling, hint):
    """
    Given a light curve whose magsys is a case variant,
    When the system is constructed,
    Then it raises naming the light curve, with a did-you-mean -- the
    boundary is case-sensitive, exactly as for an SED row.
    """
    config, user_params = _kmt_inputs()
    config["mulensinstrument"][1]["magsys"] = spelling
    with pytest.raises(ValueError, match=rf"KMTS04.*Did you mean '{hint}'"):
        _construct(config, user_params)


def test_unknown_magsys_raises():
    """
    Given a light curve whose magsys is not a supported system,
    When the system is constructed,
    Then it raises naming the light curve and the spelling.
    """
    config, user_params = _kmt_inputs()
    config["mulensinstrument"][0]["magsys"] = "ST"
    with pytest.raises(ValueError, match=r"KMTC04.*'ST'"):
        _construct(config, user_params)


def test_filter_with_no_native_system_and_none_stated_raises(monkeypatch):
    """
    Given a band filter whose record carries NO native system (as Kepler Kp
    and Gemini Zorro do) and a light curve that states none,
    When the system is prepared,
    Then it raises naming the light curve and telling the user to state it.
    """
    from exozippy.components.sed import sed as sed_mod

    real = sed_mod.read_magsys_table

    def no_native(model_root, model):
        record = real(model_root, model).copy()
        record.loc["Cousins_I", "native_system"] = ""
        return record

    monkeypatch.setattr(sed_mod, "read_magsys_table", no_native)
    with pytest.raises(
        ValueError, match=r"mulensinstrument 'KMTC04'.*no native"
    ):
        _kmt_system(zp={"mu": 0.0, "sigma": 0.2}, build=False)


# ---------------------------------------------------------------------------
# 2.2.21: the zeropoint tie is the user's statement, or nothing
# ---------------------------------------------------------------------------


def test_no_stated_zeropoint_means_no_tie():
    """
    Given the KMT example as shipped (no zeropoint entry),
    When the model is built,
    Then there is no zeropoint prior at all -- defaults.yaml supplies no
    mu/sigma -- while the zeropoint is still REPORTED as derived.
    """
    system, model = _kmt_system()
    assert "mulensinstrument.zeropoint" in model.named_vars
    assert "gaussian_prior.mulensinstrument.zeropoint" not in [
        p.name for p in model.potentials
    ]
    assert not np.any(system.mulensinstrument._zp_tied)


def test_zeropoint_mu_without_sigma_warns(caplog):
    """
    Given a zeropoint mu with no sigma,
    When the system is prepared,
    Then it warns, per light curve, that the entry does nothing.
    """
    with caplog.at_level(logging.WARNING):
        system, _ = _kmt_system(zp={"mu": 0.0}, build=False)
    hits = [
        r.getMessage()
        for r in caplog.records
        if "has a mu but no sigma" in r.getMessage()
    ]
    assert len(hits) == 3, hits
    assert all("DOES NOTHING" in h for h in hits)
    assert not np.any(system.mulensinstrument._zp_tied)


@pytest.mark.parametrize(
    "entry",
    [{"initval": 0.0}, {"initval": 0.0, "sigma": 0.2}],
    ids=["bare", "initval+sigma"],
)
def test_zeropoint_initval_raises(entry):
    """
    Given a zeropoint initval (or a bare `zeropoint: x`, which the params
    boundary translates to {initval: x}),
    When the system is prepared,
    Then it raises: the zeropoint is derived, so a start value does nothing;
    the user must give mu and sigma.
    """
    with pytest.raises(ValueError, match=r"zeropoint.*initval.*mu: <zp>"):
        _kmt_system(zp=entry, build=False)


def test_bare_zeropoint_value_raises_through_the_params_boundary():
    """
    Given the literal bare spelling `mulensinstrument.KMTC04.zeropoint: 0.0`,
    When the system is prepared,
    Then the same refusal fires for that light curve.
    """
    config, user_params = _kmt_inputs()
    user_params["mulensinstrument.KMTC04.zeropoint"] = 0.0
    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        system = System(config, user_params=user_params)
        with pytest.raises(ValueError, match=r"\['KMTC04'\].*initval"):
            system.prepare()
    finally:
        os.chdir(cwd)
