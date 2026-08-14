"""
Parity of the transit module against EXOFASTv2's exofast_tran.pro, both
unsmeared (2-minute cadence, no exptime/ninterp keys) and exposure-smeared
(30-minute cadence, exptime=30 / ninterp=10).

The reference fixture (tests/fixtures/exofast_tran_parity.json, generated
once by scripts/make_exofast_tran_reference.py under IDL) holds a typical
hot Jupiter around a sun-like star observed across one transit from one
exposure before ingress to one exposure after egress in TESS. It stores
the scenario (config + user params), the derived inputs EXOZIPPy fed to
exofast_tran (ar, inc, tp, omega, p, u1, u2, 2pi/n -- extracted from the
built model so both codes integrated the identical system), and the IDL
fluxes. Here the same System is rebuilt and must (a) still derive those
same inputs (guarding drift in constants or derivations) and (b) match
exofast_tran's light curves.

The generator reported max |EXOZIPPy - exofast_tran| of 4.6e-10
(unsmeared) and 1.4e-10 (smeared); the assertions allow 1e-8 (~20x
margin, still ~1e5x below the 1e-3 data errors of a typical light curve).
"""

import json
import os
from unittest import mock

import numpy as np
import pytensor
import pytest

from exozippy.components.transit.transit import Transit
from exozippy.system import System

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "exofast_tran_parity.json"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_FIXTURE_PATH),
    reason="exofast_tran reference fixture missing "
    "(run scripts/make_exofast_tran_reference.py)",
)

ATOL = 1e-8


def _write_lc(path, t):
    np.savetxt(
        path, np.column_stack([t, np.ones_like(t), np.full_like(t, 1e-3)])
    )
    return str(path)


def _initial_point_fn(model, tensors):
    ip = model.initial_point()
    givens = [
        (rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip
    ]
    return pytensor.function(
        [],
        tensors,
        givens=givens,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )


def _build_parity(tmp_path_factory):
    with open(_FIXTURE_PATH) as f:
        fix = json.load(f)

    d = tmp_path_factory.mktemp("exofast_parity")
    config = json.loads(json.dumps(fix["config"]))
    config["transit"][0]["file"] = _write_lc(
        d / "lc2min.dat", np.array(fix["unsmeared"]["time"])
    )
    config["transit"][1]["file"] = _write_lc(
        d / "lc30min.dat", np.array(fix["smeared"]["time"])
    )

    system = System(config, user_params=fix["user_params"])
    system.prepare()
    model = system.build_model()
    model_flux = np.asarray(
        _initial_point_fn(model, system.transit._model_flux_node)()
    )
    return fix, system, model, model_flux


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    """The fixture dict plus the rebuilt (system, model, model_flux)."""
    return _build_parity(tmp_path_factory)


@pytest.fixture(scope="module")
def parity_no_ltt(tmp_path_factory):
    """Same as `parity`, but with light-travel-time correction disabled.

    The exofast_tran.pro IDL fixture is pre-LTT by construction -- EXOFASTv2
    has no light-travel-time correction in this code path -- so comparing it
    against an LTT-on EXOZIPPy model is not apples to apples: LTT on has
    legitimately moved the model by ~a/c (~20s for this fixture's system),
    which shows up as an ingress/egress-weighted few-1e-4 relative
    discrepancy that is real physics, not a bug (confirmed: the shift
    matches this system's own a/c, and vanishes here). Disabling LTT here
    (the same Transit._LIGHT_TRAVEL_TIME_ACTIVE monkeypatch the OFF-
    equivalence test in test_transit_ltt.py uses) restores the
    apples-to-apples EXOFASTv2-parity check on the SHARED transit physics
    (Kepler solve, limb darkening, geometry) that this fixture exists to
    pin. LTT's own correctness (the 499s/AU amplitude, the 2a/c secondary
    offset, the wired accessors/units) is covered separately and on purpose
    by tests/test_ltt.py and tests/test_transit_ltt.py -- this fixture was
    never meant to validate LTT, only the physics EXOFASTv2 and EXOZIPPy
    both implement.
    """
    with mock.patch.object(Transit, "_LIGHT_TRAVEL_TIME_ACTIVE", False):
        return _build_parity(tmp_path_factory)


def test_derived_inputs_match_reference(parity):
    """
    Given the scenario stored in the fixture,
    When the System is rebuilt and the derived quantities the transit
    model consumes (ar, inc, tp, omega, p, u1, u2, period=2pi/n) are
    evaluated at the initial point,
    Then they match the values the generator fed to exofast_tran --
    i.e. the two light curves below really describe the same physical
    system, and no internal derivation or constant has drifted since
    the reference was made.
    """
    fix, system, model, _ = parity
    orbits, planets, band = system.orbit, system.planet, system.band
    fn = _initial_point_fn(
        model,
        [
            orbits.tp.value[0],
            orbits.n.value[0],
            orbits.ecc.value[0],
            orbits.cosw.value[0],
            orbits.sinw.value[0],
            orbits.inc.value[0],
            planets.ar.value[0],
            planets.p.value[0],
            band.u1.value[0],
            band.u2.value[0],
        ],
    )
    tp, n, ecc, cosw, sinw, inc, ar, p, u1, u2 = (float(v) for v in fn())
    derived = {
        "tp": tp,
        "period": 2.0 * np.pi / n,
        "e": ecc,
        "omega": np.arctan2(sinw, cosw),
        "inc": inc,
        "ar": ar,
        "p": p,
        "u1": u1,
        "u2": u2,
    }
    ref = fix["inputs"]
    assert set(derived) == set(ref)
    for key, value in derived.items():
        assert value == pytest.approx(ref[key], rel=1e-10, abs=1e-12), key


def test_unsmeared_matches_exofast_tran(parity_no_ltt):
    """
    Given the 2-minute-cadence instrument with no exptime/ninterp keys
    (the default, instantaneous model), and light-travel-time correction
    DISABLED (see parity_no_ltt: the IDL fixture is pre-LTT by
    construction, so this test compares EXOZIPPy and exofast_tran on the
    shared, LTT-independent transit physics only -- LTT itself is pinned
    separately by tests/test_ltt.py and tests/test_transit_ltt.py),
    When the per-observation model flux is evaluated at the initial point,
    Then it matches exofast_tran's light curve at every epoch.
    """
    fix, system, model, model_flux = parity_no_ltt
    rows = system.transit.inst_map == 0
    np.testing.assert_allclose(
        model_flux[rows], fix["unsmeared"]["flux"], atol=ATOL, rtol=0
    )


def test_smeared_matches_exofast_tran(parity_no_ltt):
    """
    Given the 30-minute-cadence instrument with exptime=30 / ninterp=10,
    and light-travel-time correction DISABLED (see parity_no_ltt),
    When the per-observation model flux is evaluated at the initial point,
    Then it matches EXOFASTv2's exposure-smeared light curve
    (exofast_chi2v2.pro's midpoint sub-exposure grid averaged with
    uniform 1/ninterp weights) at every epoch -- pinning both the
    sub-exposure offsets and the averaging against the reference
    implementation, not just against this module's own conventions.
    """
    fix, system, model, model_flux = parity_no_ltt
    rows = system.transit.inst_map == 1
    np.testing.assert_allclose(
        model_flux[rows], fix["smeared"]["flux"], atol=ATOL, rtol=0
    )

    # The smearing must matter at this cadence: the IDL smeared curve and
    # an instantaneous evaluation at the same epochs differ by far more
    # than the parity tolerance, so the test above cannot pass by
    # accidentally comparing two instantaneous models.
    t30 = np.array(fix["smeared"]["time"])
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    param_values = system.transit._point_to_plot_params(point, system)
    instantaneous = 1.0 + system.transit._compiled_full_lc(
        t30, 1, *param_values
    )
    assert np.max(np.abs(instantaneous - fix["smeared"]["flux"])) > 1e-5
