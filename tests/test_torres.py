"""Torres+2010 empirical mass/radius relations (components/torres)."""

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.components.torres import physics
from exozippy.components.torres.torres import Torres


def _call(fn, teff, logg, feh):
    """Evaluate a relation on float64 symbolic inputs.

    Deliberately not pt.as_tensor_variable(<python float>): pytensor autocasts
    a bare Python float to the smallest dtype that represents it, so e.g.
    5778.0 becomes a float32 constant and pt.log10 of it is then computed in
    float32 -- losing ~1e-7 and hiding real errors. The model always feeds
    float64 nodes, so the tests do too.
    """
    t, g, f = pt.dscalar("teff"), pt.dscalar("logg"), pt.dscalar("feh")
    out = fn(t, g, f)
    return float(out.eval({t: float(teff), g: float(logg), f: float(feh)}))


# Cross-checked against EXOFASTv2's massradius_torres.pro run under IDL 9.0
# (Harvard-Smithsonian licence), not against a reimplementation:
#   IDL> massradius_torres, logg, teff, feh, mstar, rstar
# (teff, logg, feh, mstar, rstar)
_IDL_REFERENCE = [
    (5778.0, 4.438, 0.0, 1.05077482864747, 1.01764587819543),   # the Sun
    (6207.0, 4.10, -0.116, 1.26594776886753, 1.65376498791201),  # ~KELT-4A
    (4800.0, 4.50, 0.25, 0.82451403303288, 0.82705475886732),   # K dwarf
    (6500.0, 3.80, -0.5, 1.42345372564569, 2.51783507677170),   # subgiant
]


@pytest.mark.parametrize("teff,logg,feh,exp_mass,exp_radius", _IDL_REFERENCE)
def test_relations_match_idl(teff, logg, feh, exp_mass, exp_radius):
    """
    Given Teff/logg/[Fe/H] inside the calibration range,
    When the Torres relations are evaluated on PyTensor nodes,
    Then they reproduce massradius_torres.pro's IDL output.
    """
    # Act -- the relations return log10, so exponentiate to compare
    mass = 10.0 ** _call(physics.calc_torres_logmass, teff, logg, feh)
    radius = 10.0 ** _call(physics.calc_torres_logradius, teff, logg, feh)

    # Assert
    assert mass == pytest.approx(exp_mass, rel=1e-12)
    assert radius == pytest.approx(exp_radius, rel=1e-12)


def test_relations_return_log10_not_linear():
    """
    Given the component applies its penalty in log space,
    When the physics functions are called,
    Then they return log10(M) and log10(R), not M and R.
    """
    # Act
    logm = _call(physics.calc_torres_logmass, 5778.0, 4.438, 0.0)
    logr = _call(physics.calc_torres_logradius, 5778.0, 4.438, 0.0)

    # Assert
    assert logm == pytest.approx(np.log10(1.05077482864747), rel=1e-12)
    assert logr == pytest.approx(np.log10(1.01764587819543), rel=1e-12)


def test_relations_are_differentiable():
    """
    Given the relations sit inside a NUTS gradient graph,
    When gradients are taken w.r.t. every input,
    Then all of them are finite.
    """
    # Arrange
    t = pt.scalar("teff")
    g = pt.scalar("logg")
    f = pt.scalar("feh")
    point = {t: 5778.0, g: 4.438, f: 0.0}

    # Act
    grads = [pt.grad(physics.calc_torres_logmass(t, g, f), w) for w in (t, g, f)]
    grads += [pt.grad(physics.calc_torres_logradius(t, g, f), w) for w in (t, g, f)]
    vals = [float(gr.eval(point)) for gr in grads]

    # Assert
    assert all(np.isfinite(v) for v in vals)


def test_published_scatter_is_in_dex():
    """
    Given Torres' scatter is quoted in dex (unlike Mann's fractional floors),
    When the floors are read,
    Then they match EXOFASTv2's hardcoded ulogm/urstar.
    """
    # Assert
    assert physics.LOGM_FLOOR == 0.027
    assert physics.LOGR_FLOOR == 0.014
    assert physics.MSTAR_MIN == 0.6


# ----------------------------------------------------------------------
# Config parsing / validation
# ----------------------------------------------------------------------

class _FakeStar:
    names = ["A", "B", "C"]
    n_elements = 3


class _FakeSystem:
    star = _FakeStar()


def _torres(cfg):
    comp = Torres(cfg, config_manager=None)
    comp.load_data(_FakeSystem())
    return comp


def test_instances_resolve_stars_and_default_to_both():
    """
    Given torres blocks naming stars by bare name and by star.X path,
    When the component loads,
    Then both resolve, and mass+radius are constrained by default.
    """
    # Arrange / Act
    comp = _torres([{"star": "A"}, {"star": "star.C"}])

    # Assert
    assert comp.star_indices == [0, 2]
    assert comp.names == ["A", "C"]
    assert comp.constrain == [{"mass", "radius"}, {"mass", "radius"}]
    assert comp.logm_floor == [0.027, 0.027]
    assert comp.logr_floor == [0.014, 0.014]


def test_scatter_can_be_overridden_per_instance():
    """
    Given a block overriding the dex scatter,
    When the component loads,
    Then the override wins and the other instance keeps the default.
    """
    # Arrange / Act
    comp = _torres([{"star": "A", "logm_floor": 0.05}, {"star": "B"}])

    # Assert
    assert comp.logm_floor == [0.05, 0.027]
    assert comp.logr_floor == [0.014, 0.014]


def test_register_parameters_declares_no_parameters():
    """
    Given Torres reads only parameters the star already owns,
    When it registers,
    Then its manifest is empty -- it contributes potentials only.
    """
    # Arrange
    comp = _torres([{"star": "A"}])

    # Act
    comp.register_parameters(_FakeSystem())

    # Assert
    assert comp.manifest == {}


@pytest.mark.parametrize("cfg,match", [
    ([{"constrain": ["mass"]}], "'star:' key is required"),
    ([{"star": "Z"}], "unknown star"),
    ([{"star": "A", "constrain": ["mass", "teff"]}], "unknown 'constrain:'"),
    ([{"star": "A", "constrain": []}], "is empty"),
    ([{"star": "A", "logm_floor": 0}], "must be > 0 dex"),
    ([{"star": "A"}, {"star": "A"}], "Duplicate names"),
])
def test_config_errors_are_actionable(cfg, match):
    """
    Given a malformed torres block,
    When the component loads,
    Then it raises a ValueError naming the problem.
    """
    with pytest.raises(ValueError, match=match):
        _torres(cfg)
