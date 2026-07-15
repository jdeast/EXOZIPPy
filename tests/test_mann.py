"""Mann+ empirical mass/radius relations (components/mann)."""

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.components.mann import physics
from exozippy.components.mann.mann import Mann


def _call(fn, absks, feh=None):
    """Evaluate a relation on float64 symbolic inputs.

    Deliberately not pt.as_tensor_variable(<python float>): pytensor autocasts
    a bare Python float to the smallest dtype that represents it, so a round
    value silently becomes a float32 constant. These relations are pure
    polynomials and would upcast anyway, but the model always feeds float64
    nodes, so the tests do too (see tests/test_torres.py, where the same
    shortcut cost ~1e-7 through pt.log10).
    """
    ks = pt.dscalar("absks")
    point = {ks: float(absks)}
    if feh is None:
        out = fn(ks, None)
    else:
        f = pt.dscalar("feh")
        point[f] = float(feh)
        out = fn(ks, f)
    return float(out.eval(point))


# Reference values produced by the numpy translation of EXOFASTv2's
# massradius_mann.pro that used to live in
# exozippy/evolutionary_model/massradius_mann.py, captured before that
# orphan was removed in favour of the PyTensor implementation here.
# (absks, feh, expected_mass, expected_radius)
_REFERENCE = [
    (7.5, None, 0.2280342072000418, 0.2565000000000003),
    (7.5, -0.116, 0.22551544332725632, 0.2522662200000001),
    (6.0, 0.0, 0.43662992768383013, 0.4438199999999999),
    (9.0, 0.3, 0.11597242841928908, 0.1585439999999998),
]


@pytest.mark.parametrize("absks,feh,exp_mass,exp_radius", _REFERENCE)
def test_relations_match_the_idl_translation(absks, feh, exp_mass, exp_radius):
    """
    Given an absolute Ks (and optionally [Fe/H]) inside the calibration range,
    When the Mann mass/radius relations are evaluated on PyTensor nodes,
    Then they reproduce the IDL-derived reference values.
    """
    # Act
    mass = _call(physics.calc_mann_mass, absks, feh)
    radius = _call(physics.calc_mann_radius, absks, feh)

    # Assert
    assert mass == pytest.approx(exp_mass, rel=1e-12)
    assert radius == pytest.approx(exp_radius, rel=1e-12)


def test_feh_and_nofeh_forms_differ():
    """
    Given the same absolute Ks,
    When the [Fe/H]-dependent and [Fe/H]-free forms are evaluated,
    Then they give different answers (i.e. the feh switch is actually wired).
    """
    # Act
    m_nofeh = _call(physics.calc_mann_mass, 7.0)
    m_feh = _call(physics.calc_mann_mass, 7.0, 0.3)
    r_nofeh = _call(physics.calc_mann_radius, 7.0)
    r_feh = _call(physics.calc_mann_radius, 7.0, 0.3)

    # Assert
    assert m_nofeh != pytest.approx(m_feh, rel=1e-6)
    assert r_nofeh != pytest.approx(r_feh, rel=1e-6)


def test_relations_are_differentiable():
    """
    Given the relations sit inside a NUTS gradient graph,
    When d(mass)/d(absks) and d(radius)/d(absks) are taken,
    Then both are finite and negative (fainter star -> smaller mass/radius).
    """
    # Arrange
    ks = pt.dscalar("absks")
    feh = pt.dscalar("feh")
    point = {ks: 6.5, feh: -0.1}

    # Act
    dm = pt.grad(physics.calc_mann_mass(ks, feh), ks)
    dr = pt.grad(physics.calc_mann_radius(ks, feh), ks)
    dm_val = float(dm.eval(point))
    dr_val = float(dr.eval(point))

    # Assert
    assert np.isfinite(dm_val) and dm_val < 0
    assert np.isfinite(dr_val) and dr_val < 0


def test_published_scatter_floors():
    """
    Given the relations' published fractional scatter,
    When the floors are read,
    Then they match EXOFASTv2's rstaremfloor/mstaremfloor defaults.
    """
    # Assert
    assert physics.MSTAR_FLOOR[False] == 0.020
    assert physics.MSTAR_FLOOR[True] == 0.021
    assert physics.RSTAR_FLOOR[False] == 0.0289
    assert physics.RSTAR_FLOOR[True] == 0.027


# ----------------------------------------------------------------------
# Config parsing / validation
# ----------------------------------------------------------------------

class _FakeStar:
    names = ["A", "B", "C"]
    n_elements = 3


class _FakeSystem:
    star = _FakeStar()


def _mann(cfg):
    comp = Mann(cfg, config_manager=None)
    comp.load_data(_FakeSystem())
    return comp


def test_instances_resolve_stars_and_default_to_synthetic():
    """
    Given mann blocks naming stars by bare name and by star.X path,
    When the component loads,
    Then both resolve to star indices and default to the synthetic pathway.
    """
    # Arrange / Act
    comp = _mann([{"star": "B"}, {"star": "star.C"}])

    # Assert
    assert comp.star_indices == [1, 2]
    assert comp.names == ["B", "C"]          # named after their star
    assert comp.ks_synthetic == [True, True]
    assert comp.ks_err == [0.02, 0.02]       # EXOFASTv2's synthetic floor
    assert comp.constrain == [{"mass", "radius"}, {"mass", "radius"}]
    assert comp.use_feh == [True, True]


def test_observed_pathway_reads_ks_and_err():
    """
    Given a mann block with a numeric ks and a ks_err,
    When the component loads,
    Then it takes the observed pathway with those values.
    """
    # Arrange / Act
    comp = _mann([{"star": "B", "ks": 12.7, "ks_err": 0.03}])

    # Assert
    assert comp.ks_synthetic == [False]
    assert comp.ks_observed == [12.7]
    assert comp.ks_err == [0.03]


def test_floors_default_per_relation_form_and_can_be_overridden():
    """
    Given blocks that do and do not override the scatter floors,
    When the component loads,
    Then defaults follow the feh switch and overrides win.
    """
    # Arrange / Act
    comp = _mann([
        {"star": "B"},
        {"star": "C", "feh": False, "mstar_floor": 0.05},
    ])

    # Assert
    assert comp.mstar_floor == [0.021, 0.05]
    assert comp.rstar_floor == [0.027, 0.0289]


@pytest.mark.parametrize("cfg,match", [
    ([{"ks": "synthetic"}], "'star:' key is required"),
    ([{"star": "Z"}], "unknown star"),
    ([{"star": "B", "ks": 12.7}], "'ks_err:' is required"),
    ([{"star": "B", "ks": "sed"}], "must be either the string"),
    ([{"star": "B", "constrain": ["mass", "teff"]}], "unknown 'constrain:'"),
    ([{"star": "B", "constrain": []}], "is empty"),
    ([{"star": "B"}, {"star": "B"}], "Duplicate names"),
])
def test_config_errors_are_actionable(cfg, match):
    """
    Given a malformed mann block,
    When the component loads,
    Then it raises a ValueError naming the problem.
    """
    with pytest.raises(ValueError, match=match):
        _mann(cfg)
