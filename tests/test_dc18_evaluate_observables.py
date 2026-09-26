"""t_E and rho were never scored on any DC2018 run, and nothing said so.

`OBSERVABLES` names six observables; `_pick` looked each up under a couple of
trace spellings and, finding none, did a bare `continue`.  `mulensevent.t_E`
and `source.log_rho` have never existed in a trace, so t_E and rho were
dropped from every score ever produced -- and `u_0` joined them once the sweep
config set `fitu0te: true` (which samples u_0*t_E).  A run missing half its
observables still printed a PASS.

Relations are quoted from the model's symbolic map; a first hand-derivation
used exp() where the model uses 10** and produced a confident 7-sigma pull on
a parameter that is actually correct to 1%.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"


@pytest.fixture(scope="module")
def ev():
    sys.path.insert(0, str(DC18))
    try:
        yield importlib.import_module("dc18_evaluate")
    finally:
        sys.path.remove(str(DC18))


def _post(**cols):
    """A 2-chain x 3-draw posterior; element-valued vars get a third axis."""
    data = {}
    for k, v in cols.items():
        a = np.asarray(v, dtype=float)
        dims = ["chain", "draw"] + (["element"] if a.ndim == 3 else [])
        data[k] = (dims, a)
    return xr.Dataset(data)


def _two_by_three(x):
    return np.full((2, 3), float(x))


def test_t_E_matches_the_symbolic_relation(ev):
    """t_E = theta_E / (mu_rel_mag / DAYS_PER_YEAR), theta_E = 10**log_theta_E."""
    from exozippy.constants import DAYS_PER_YEAR

    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "mulensevent.mu_ra_rel": _two_by_three(3.0),
            "mulensevent.mu_dec_rel": _two_by_three(4.0),
        }
    )
    got = ev._t_E(post)
    want = (10.0**-0.5) / (5.0 / DAYS_PER_YEAR)  # hypot(3,4) = 5
    assert np.allclose(got, want)


def test_t_E_is_not_computed_with_exp(ev):
    """The exact slip that produced a bogus 7-sigma pull; pin it shut."""
    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "mulensevent.mu_ra_rel": _two_by_three(3.0),
            "mulensevent.mu_dec_rel": _two_by_three(4.0),
        }
    )
    got = float(np.ravel(ev._t_E(post))[0])
    from exozippy.constants import DAYS_PER_YEAR

    wrong = np.exp(-0.5) / (5.0 / DAYS_PER_YEAR)
    assert not np.isclose(got, wrong)


def test_u_0_is_u0te_over_t_E(ev):
    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "mulensevent.mu_ra_rel": _two_by_three(3.0),
            "mulensevent.mu_dec_rel": _two_by_three(4.0),
            "source.u0te": _two_by_three(8.0),
        }
    )
    assert np.allclose(ev._u_0_from_u0te(post), 8.0 / ev._t_E(post))


def test_rho_is_returned_in_log10_like_its_stored_spelling(ev):
    """OBSERVABLES marks rho is_log=True (stored as `log_rho`), so the truth is
    log10'd.  A linear column here compared 0.0011 against log10(0.0011) and
    printed -3726 sigma."""
    from exozippy.constants import RSUN_TO_AU

    radius = np.zeros((2, 3, 2))
    dist = np.zeros((2, 3, 2))
    radius[..., ev.SOURCE_IDX] = 1.5
    dist[..., ev.SOURCE_IDX] = 8000.0
    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "star.radius": radius,
            "star.distance": dist,
        }
    )
    want = np.log10((1.5 * RSUN_TO_AU / 8000.0) * 1000.0 / (10.0**-0.5))
    assert np.allclose(ev._rho(post), want)
    assert float(np.ravel(ev._rho(post))[0]) < 0  # a log, not a linear rho


def test_rho_reads_the_SOURCE_element_not_the_lens(ev):
    """Non-vacuity: the lens element is filled with a different value, so an
    off-by-one on SOURCE_IDX changes the answer instead of passing quietly."""
    from exozippy.constants import RSUN_TO_AU

    radius = np.zeros((2, 3, 2))
    dist = np.zeros((2, 3, 2))
    radius[..., ev.SOURCE_IDX] = 1.5
    dist[..., ev.SOURCE_IDX] = 8000.0
    radius[..., 1 - ev.SOURCE_IDX] = 99.0
    dist[..., 1 - ev.SOURCE_IDX] = 10.0
    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "star.radius": radius,
            "star.distance": dist,
        }
    )
    want = np.log10((1.5 * RSUN_TO_AU / 8000.0) * 1000.0 / (10.0**-0.5))
    assert np.allclose(ev._rho(post), want)


def test_a_missing_observable_raises_instead_of_being_dropped(ev):
    """The bug itself: scoring must refuse, not quietly cover less."""
    post = _post(**{"source.t_0": _two_by_three(2459665.8)})
    # Non-vacuity: a deriver IS registered for t_E and this fixture genuinely
    # starves it, so the raise comes from the guard and not from a typo.
    assert ev.DERIVERS.get("t_E") is not None
    assert ev._t_E(post) is None
    with pytest.raises(ev.MissingObservable) as e:
        ev._pick_or_derive(post, "t_E", ("mulensevent.t_E",), required=True)
    assert "t_E" in str(e.value)


def test_pick_or_derive_returns_none_when_nothing_works(ev):
    post = _post(**{"source.t_0": _two_by_three(1.0)})
    assert ev._pick_or_derive(post, "t_E", ("mulensevent.t_E",)) == (
        None,
        None,
    )


def test_derived_values_are_flat_like_pick(ev):
    """_pick's contract is one value per (chain, draw), already flat.  Returning
    the 2-D array let np.vstack line 78 chains up against 3995 draws."""
    post = _post(
        **{
            "mulensevent.log_theta_E": _two_by_three(-0.5),
            "mulensevent.mu_ra_rel": _two_by_three(3.0),
            "mulensevent.mu_dec_rel": _two_by_three(4.0),
        }
    )
    name, v = ev._pick_or_derive(post, "t_E", ("mulensevent.t_E",))
    assert name == "<derived>"
    assert v.ndim == 1 and v.size == 6
