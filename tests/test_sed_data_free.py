"""The SED's data-free inventory and the errscale pin (review 2.9.11).

An SED with no photometric row (`filters: []`, the DC2018 sweep shape) still
declared ``sed.errscale`` and sampled it on its U(0.001, 1000) prior: nothing
reads it, since ``build_likelihood`` adds the photometric Normal only when
there are rows.  Four DC2018 arms put its median at 493-544 against the prior
median of 500 -- an unconstrained knob, once misread as model strain.  With a
FEW rows the same thing happens more quietly: the stars' own parameters
absorb the magnitudes and leave no residual scatter to measure an error
scale from (JDE 2026-09-10: "if we only have two (or 1?) band and we're
fitting an errscale, we don't have enough dof ... pin errscale to 1").

THE RULE: errscale is pinned at 1.0, through the manifest ``overrides``
channel (so a params entry still wins), whenever the SED has fewer usable
photometric rows than free SED parameters -- errscale itself plus, per star
a row names, the free ``SED.PHOTOMETRIC_DOF`` coordinates (the BC grid's
three sampled axes and one flux scale).  One star: 5, so four rows pin it
and five do not.

The inventory is DERIVED from the topology (``SED.photometric_readers``):
the same answer pins the unread stars' SED knobs in ``Star`` and drives the
no-grid fallback (2.9.15).
"""

import logging

import pymc as pm
import pytest

from exozippy.system import System

ROWS = [
    ("2MASS/2MASS.J", 8.000, 0.020),
    ("2MASS/2MASS.H", 7.800, 0.026),
    ("2MASS/2MASS.Ks", 7.750, 0.018),
    ("WISE/WISE.W1", 7.700, 0.030),
    ("WISE/WISE.W2", 7.720, 0.030),
]


def _sed_file(tmp_path, n_rows, pos=None):
    lines = ["model: NextGen", "filters:" if n_rows else "filters: []"]
    for name, mag, err in ROWS[:n_rows]:
        lines += [
            f"  - name: {name}",
            f"    mag: {mag}",
            f"    err: {err}",
            "    magsys: Vega",
        ]
        if pos is not None:
            lines += ["    photType:", f"      pos: [{pos}]"]
    path = tmp_path / "star.sed"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _system(tmp_path, n_rows, user_params=None, stars=("A",), pos=None):
    config = {
        "star": [{"name": s, "mist": False} for s in stars],
        # A band so a row-less SED still has a filter to build a grid for
        # (load_data refuses an SED with none at all).  Nothing reads it.
        "band": [{"name": "G", "filter": "GAIA/GAIA2r.G", "ld_law": "linear"}],
        "sed": {"file": _sed_file(tmp_path, n_rows, pos=pos)},
    }
    system = System(config, dict(user_params or {}))
    system.prepare()
    return system


def _free(model):
    return {rv.name for rv in model.free_RVs}


def _errscale_alone(system):
    """Build ONLY sed.errscale.  A row-less SED in a system where nothing
    else reads a predicted flux cannot build a whole model (a separate,
    pre-existing defect: compile_plotters asks for the prediction node
    outside the model context), and errscale depends on nothing."""
    with pm.Model() as model:
        system.sed.add_parameter(model, "errscale", system)
    return model


def test_no_rows_pins_errscale_and_says_why(tmp_path, caplog):
    """
    Given an SED with `filters: []`,
    When errscale is built,
    Then it is not a free RV, it sits at 1.0, and startup says it was
      pinned because no row reads it.
    """
    # ARRANGE / ACT
    with caplog.at_level(logging.INFO, logger="exozippy.components.sed.sed"):
        system = _system(tmp_path, 0)
        model = _errscale_alone(system)

    # ASSERT
    assert "sed.errscale_raw" not in _free(model)
    assert float(system.sed.errscale.initval[0]) == 1.0
    assert "errscale pinned at 1.0: no photometric rows" in caplog.text
    assert system.sed.data_free_inventory(system).data_free[
        "sed.errscale"
    ] == (0,)


@pytest.mark.parametrize("n_rows", [1, 2, 4])
def test_fewer_rows_than_free_parameters_pins_errscale(
    tmp_path, n_rows, caplog
):
    """
    Given one star and 1, 2 or 4 photometric rows (fewer than errscale +
      the star's four photometric coordinates = 5),
    When the model is built,
    Then errscale is pinned, and the warning carries the counts.
    """
    # ARRANGE / ACT
    with caplog.at_level(
        logging.WARNING, logger="exozippy.components.sed.sed"
    ):
        system = _system(tmp_path, n_rows)
        model = system.build_model()

    # ASSERT
    inv = system.sed.data_free_inventory(system)
    assert (inv.n_rows, inv.n_free) == (n_rows, 5)
    assert "sed.errscale_raw" not in _free(model)
    assert f"{n_rows} photometric row(s) against 5 free SED parameters" in (
        caplog.text
    )


def test_as_many_rows_as_free_parameters_fits_errscale(tmp_path):
    """
    Given one star and five rows -- exactly the threshold,
    When the model is built,
    Then errscale is sampled, as before this rule existed.
    """
    # ARRANGE / ACT
    system = _system(tmp_path, 5)
    model = system.build_model()

    # ASSERT
    assert system.sed.data_free_inventory(system).pin_errscale is False
    assert "sed.errscale_raw" in _free(model)


def test_a_user_pin_on_a_stellar_coordinate_lowers_the_count(tmp_path):
    """
    Given four rows and the star's feh pinned by the user (sigma: 0),
    When the inventory counts the free SED parameters,
    Then feh does not count (4 = 4), so errscale is fitted.
    """
    # ARRANGE / ACT
    system = _system(
        tmp_path, 4, user_params={"star.A.feh": {"initval": 0.0, "sigma": 0}}
    )
    model = system.build_model()

    # ASSERT
    inv = system.sed.data_free_inventory(system)
    assert (inv.n_rows, inv.n_free, inv.pin_errscale) == (4, 4, False)
    assert "sed.errscale_raw" in _free(model)


def test_a_user_errscale_entry_beats_the_pin(tmp_path):
    """
    Given no rows but a params-file errscale with bounds,
    When errscale is built,
    Then the user's entry wins (the pin rides the overrides channel, which
      layers UNDER the params file) and errscale samples.
    """
    # ARRANGE / ACT
    system = _system(
        tmp_path,
        0,
        user_params={"sed.errscale": {"initval": 2.0, "sigma": 0.5}},
    )
    model = _errscale_alone(system)

    # ASSERT
    assert "sed.errscale_raw" in _free(model)


def test_the_inventory_lists_what_no_data_reads(tmp_path):
    """
    Given two stars and rows naming only A,
    When the inventory is taken,
    Then B is unread: its SED knobs (av, radiussed, teffsed) are the
      data-free list, errscale is not (there are rows), and B's loggsed
      carries no grid barrier while A's keeps it.
    """
    # ARRANGE / ACT
    system = _system(tmp_path, 5, stars=("A", "B"), pos="A")
    system.build_model()
    inv = system.sed.data_free_inventory(system)

    # ASSERT
    assert inv.unread_stars == (1,)
    assert inv.readers[0] == ("sed photometry",)
    assert inv.data_free == {
        "sed.errscale": (),
        "star.av": (1,),
        "star.radiussed": (1,),
        "star.teffsed": (1,),
    }
    lo, hi = system.star.loggsed.lower, system.star.loggsed.upper
    assert (float(lo[0]), float(hi[0])) == (0.0, 5.0)
    assert (float(lo[1]), float(hi[1])) == (float("-inf"), float("inf"))
