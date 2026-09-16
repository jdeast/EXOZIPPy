"""The `population` component: between-subject variability (PK P4).

See ``src/exozippy/components/pharmacokinetics/README.md``: written by an
astrophysicist and an LLM with no domain reviewer. These tests pin the
wiring -- which coordinates become expressions of the population, the
non-centred form, the covariate model and the basis rules -- not that the
modelling choices are right.

Everything here runs on SYNTHETIC data in ``tmp_path``, for the reason the
sibling file gives: the real Theophylline example fetches its data over the
network, and a test must not.
"""

import numpy as np
import pytensor
import pytest

# tests/ is on sys.path via conftest; the synthetic assay file and the base
# config are the sibling module's, so the two files cannot drift about what a
# subject looks like.
from test_pharmacokinetics_components import (  # noqa: E402
    TRUTH,
    _config,
    synth_csv,  # noqa: F401  (a fixture, used by name)
)

from exozippy.components.factory import discover_components
from exozippy.components.pharmacokinetics.population import Population
from exozippy.system import System

LN10 = np.log(10.0)


def _system(csv, tmp_path, population=None, basis=None, user_params=None):
    """A prepared System, with an optional population over the subjects."""
    cfg = _config(csv, tmp_path)
    if basis is not None:
        flag = {"cl_v": "fitclv", "ke_v": "fitkev", "cl_ke": "fitclke"}
        for i, block in enumerate(cfg["subject"]):
            name = basis[i] if isinstance(basis, (list, tuple)) else basis
            block[flag[name]] = True
    if population is not None:
        cfg["population"] = [{"name": "adults", **population}]
    system = System(cfg, user_params=dict(user_params or {}))
    system.prepare()
    return system


def _together(*tensors):
    """Evaluate tensors in ONE call, so they share one draw of the RVs.

    Separate ``.eval()`` calls each draw from the prior afresh, which turns
    an identity check into a comparison of two unrelated points.
    """
    return [np.atleast_1d(a) for a in pytensor.function([], list(tensors))()]


# ---------------------------------------------------------------------------
# Discovery, and the no-population control
# ---------------------------------------------------------------------------


def test_the_population_component_is_auto_discovered():
    """Given the factory sweep, Then `population` is registered.

    A third component in an existing directory, found with no registration
    step -- the contract components.md states.
    """
    assert discover_components()["population"] is Population


def test_without_a_population_block_the_subjects_stay_independent(
    synth_csv, tmp_path
):
    """Given no population, Then nothing about `subject` changes.

    The hierarchy is genuinely optional rather than a default with a switch,
    and this is the assertion that says so: no eta, no weight parameter, and
    the log-coordinates still sampled.
    """
    system = _system(synth_csv, tmp_path)
    system.build_model()

    assert system.subject.log_cl.is_sampled.all()
    for absent in ("eta_cl", "eta_v", "eta_ka", "weight"):
        assert absent not in system.subject.manifest, absent


# ---------------------------------------------------------------------------
# What a population does
# ---------------------------------------------------------------------------


def test_a_population_turns_the_log_coordinates_into_its_own_expressions(
    synth_csv, tmp_path
):
    """Given a population, Then the subjects' coordinates are derived from it.

    The galacticmodel shape: a component that owns no data and whose whole
    effect is that another component's parameters stop being free.
    """
    system = _system(synth_csv, tmp_path, population={})
    subject, population = system.subject, system.population
    model = system.build_model()

    for quantity in ("cl", "v", "ka"):
        assert getattr(subject, f"log_{quantity}").is_derived.all(), quantity

    raw = {v.name for v in model.free_RVs}
    assert "subject.log_cl_raw" not in raw
    assert {
        "population.mu_log_cl_raw",
        "population.omega_cl_raw",
        "subject.eta_cl_raw",
    } <= raw
    assert population.basis == "cl_v"
    assert population.varying == ("cl", "v", "ka")


def test_the_hierarchy_is_non_centred(synth_csv, tmp_path):
    """Given a population, Then eta is sampled and carries the N(0,1).

    Not a style preference: the centred form gives each subject's coordinate
    a scale of omega, so a small omega closes a funnel NUTS cannot climb --
    and a small omega is what these data produce (the canonical nlme fit
    drives one between-subject SD to 1.9e-05).
    """
    system = _system(synth_csv, tmp_path, population={})
    model = system.build_model()

    eta = system.subject.eta_cl
    assert eta.is_sampled.all()
    assert "gaussian_prior.subject.eta_cl" in {
        p.name for p in model.potentials
    }
    # mu 0, sigma 1 -- the standardized deviation, not the deviation.
    np.testing.assert_allclose(np.atleast_1d(eta.mu), 0.0)
    np.testing.assert_allclose(np.atleast_1d(eta.sigma), 1.0)


def test_each_subject_is_its_typical_value_shifted_by_weight_and_eta(
    synth_csv, tmp_path
):
    """Given a population, Then log_cl is mu + beta*log10(WT/WTref) + omega*eta.

    The arithmetic, not merely the wiring. Every term is read from the model
    in one evaluation so they describe one point.
    """
    system = _system(synth_csv, tmp_path, population={})
    subject, population = system.subject, system.population
    system.build_model()

    log_cl, mu, beta, wt_ref, omega, eta, weight = _together(
        subject.log_cl.value,
        population.mu_log_cl.value,
        population.beta_cl.value,
        population.wt_ref.value,
        population.omega_cl.value,
        subject.eta_cl.value,
        subject.weight.value,
    )

    expected = mu + beta * np.log10(weight / wt_ref) + omega * eta
    np.testing.assert_allclose(log_cl, expected, rtol=1e-12)

    # The covariate is doing something: the subjects have different weights,
    # so an all-zero beta would make this test pass on a broken wiring.
    assert beta[0] == pytest.approx(0.75)
    assert len(set(np.round(weight, 6))) > 1


def test_the_weights_reach_the_model_as_configured(synth_csv, tmp_path):
    """Given a population, Then `weight` is the pinned covariate datum.

    Declared only when something reads it -- without a population, a weight
    is config that load_data uses to turn a per-kg dose into milligrams.
    """
    system = _system(synth_csv, tmp_path, population={})
    model = system.build_model()

    assert "subject.weight_raw" not in {v.name for v in model.free_RVs}
    np.testing.assert_allclose(
        system.subject.weight.initval,
        [wt for wt, *_rest in TRUTH.values()],
    )


def test_cv_percent_is_the_spread_in_the_unit_the_field_quotes(
    synth_csv, tmp_path
):
    """Given omega in dex, Then cv is 100*sqrt(exp((omega ln10)^2) - 1).

    omega here is the SD of the base-10 logarithm and a published CV comes
    from the SD of the NATURAL one -- a factor of 2.3026 between them, which
    is the whole reason this is computed rather than left to the reader.
    """
    system = _system(synth_csv, tmp_path, population={})
    population = system.population
    system.build_model()

    cv, omega = _together(population.cv_cl.value, population.omega_cl.value)
    np.testing.assert_allclose(
        cv, 100.0 * np.sqrt(np.expm1((omega * LN10) ** 2)), rtol=1e-12
    )
    # A derived parameter, so it arrives with an interval of its own rather
    # than as a point estimate in an extra column (pharmacokinetics.md).
    assert population.cv_cl.is_derived.all()


def test_variability_may_name_a_subset(synth_csv, tmp_path):
    """Given variability: [cl, ka], Then V has a typical value and no spread.

    A coordinate with no between-subject variability is the field's "no ETA
    on this parameter": every subject takes the typical value for its weight.
    A separate expression rather than an omega pinned to zero, which would
    leave one free eta per subject that no likelihood term reads.
    """
    system = _system(
        synth_csv, tmp_path, population={"variability": ["cl", "ka"]}
    )
    subject, population = system.subject, system.population
    model = system.build_model()

    assert population.varying == ("cl", "ka")
    assert "omega_v" not in population.manifest
    assert "cv_v" not in population.manifest
    assert "eta_v" not in subject.manifest
    # ...but V is still a population quantity, with a typical value.
    assert "mu_log_v" in population.manifest
    assert subject.log_v.is_derived.all()
    assert "subject.eta_v_raw" not in {v.name for v in model.free_RVs}


# ---------------------------------------------------------------------------
# The basis, and why it is not free of consequence here
# ---------------------------------------------------------------------------


def test_the_cl_ke_basis_samples_both_rates_and_derives_the_volume(
    synth_csv, tmp_path
):
    """Given fitclke, Then V is derived and log_v REPORTED.

    R's `SSfol` is parameterized in (lKe, lKa, lCl), so the canonical nlme
    fit's random effects are in that basis. It exists here because a
    diagonal set of omegas in one basis is not diagonal in another -- with
    (CL, V) varying independently, var(log ke) = var(log CL) + var(log V),
    and the collapsed lKe that fit reports is not representable at all.
    """
    system = _system(synth_csv, tmp_path, basis="cl_ke")
    subject = system.subject
    model = system.build_model()

    assert subject.log_cl.is_sampled.all()
    assert subject.log_ke.is_sampled.all()
    assert subject.v.is_derived.all()
    assert subject.log_v.is_reported.all()
    assert "subject.log_v_raw" not in {v.name for v in model.free_RVs}


def test_a_population_over_the_cl_ke_basis_varies_the_rates(
    synth_csv, tmp_path
):
    """Given cl_ke plus a population, Then the etas are on CL, ka and ke."""
    system = _system(synth_csv, tmp_path, basis="cl_ke", population={})
    model = system.build_model()

    assert system.population.varying == ("cl", "ka", "ke")
    raw = {v.name for v in model.free_RVs}
    assert {"subject.eta_cl_raw", "subject.eta_ke_raw"} <= raw
    assert "subject.eta_v_raw" not in raw


def test_a_population_cannot_span_two_bases(synth_csv, tmp_path):
    """Given subjects in different bases, Then the population raises.

    Not an unimplemented case: between-subject variability is defined IN a
    basis, so a population whose members disagree about the basis does not
    name a distribution.
    """
    with pytest.raises(ValueError, match="more than one coordinate basis"):
        _system(
            synth_csv,
            tmp_path,
            basis=["cl_v", "ke_v", "cl_v"],
            population={},
        )


def test_variability_must_name_a_coordinate_the_basis_samples(
    synth_csv, tmp_path
):
    """Given variability: [ke] in the cl_v basis, Then it raises."""
    with pytest.raises(ValueError, match="does not sample"):
        _system(synth_csv, tmp_path, population={"variability": ["ke"]})


def test_variability_rejects_an_unknown_quantity(synth_csv, tmp_path):
    """Given variability: [clearance], Then it names what is known."""
    with pytest.raises(ValueError, match="unknown quantities"):
        _system(synth_csv, tmp_path, population={"variability": ["clearance"]})


def test_cl_v_and_cl_ke_cannot_be_mixed_even_without_a_population(
    synth_csv, tmp_path
):
    """Given both bases, Then it raises rather than failing in the sort.

    `cl_v` derives ke from (cl, v) and `cl_ke` derives v from (cl, ke), so
    the per-parameter build order would need each to precede the other. The
    value graph is still acyclic per element; it is the sort that cannot be
    done, and there is no way to say so in a manifest.
    """
    with pytest.raises(ValueError, match="cannot be mixed"):
        _system(synth_csv, tmp_path, basis=["cl_v", "cl_ke", "cl_v"])


# ---------------------------------------------------------------------------
# What a population requires
# ---------------------------------------------------------------------------


def test_a_population_requires_a_weight_for_every_subject(synth_csv, tmp_path):
    """Given a subject with no weight, Then the population says so.

    The covariate model reads a body weight, so a missing one has to be an
    error rather than a silently invented 70 kg.
    """
    cfg = _config(synth_csv, tmp_path)
    for block in cfg["subject"]:
        # An absolute dose, so nothing else needs the weight either.
        block["dose"] = block["dose"] * block["weight"]
        block["dose_unit"] = "mg"
        block.pop("weight")
    cfg["population"] = [{"name": "adults"}]

    with pytest.raises(ValueError, match="every subject needs a 'weight:'"):
        System(cfg, user_params={}).prepare()


def test_only_one_population_is_supported_and_it_says_why(synth_csv, tmp_path):
    """Given two population blocks, Then it raises naming what is missing.

    Several populations over disjoint subjects (treatment arms) is a real
    model, and the message says what it would need rather than implying the
    config is malformed.
    """
    cfg = _config(synth_csv, tmp_path)
    cfg["population"] = [{"name": "arm_a"}, {"name": "arm_b"}]

    with pytest.raises(ValueError, match="exactly one 'population:' block"):
        System(cfg, user_params={}).prepare()
