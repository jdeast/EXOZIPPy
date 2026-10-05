"""Reported derived quantities are Deterministics, so the early-saved trace
carries them (review 2.6.14).

run.py writes ``<prefix>_trace.nc`` BEFORE wrap-up so that "an interrupt here
costs only what is not yet written".  That held for sampled parameters and
not for pure-expression derived ones -- the microlensing t_E, theta_E,
pi_rel, mu_rel_mag ... -- which ``Parameter.build_pymc`` gave no
``pm.Deterministic`` and which existed only as the ``Parameter.posterior``
that ``distribute_posterior`` rebuilt at report time.  Three OOM-killed
wrap-ups lost exactly those.  Every REPORTED derived parameter whose value
depends on a free RV is now a node (``Parameter.report_only_node``), and the
wrap-up consumers that enumerate the trace leave those nodes out, so they see
the variable set they always did.

Also here: a point-source event REPORTS rho = theta_star/theta_E (JDE
2026-09-14) without adding a logp term.
"""

import copy
import os
import shutil
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import yaml

from exozippy.samplers import convergence
from exozippy.system import System
from exozippy.trace_meta import (
    REPORT_ONLY_ATTR,
    report_only_vars,
    stamp_structural_metadata,
)

_OB_DIR = Path(__file__).parent.parent / "examples" / "ob140939"

# The quantities the item names: the event's best-measured observables, and
# the point-source rho that is determined by the source star and theta_E.
_MUST_SURVIVE = (
    "mulensevent.t_E",
    "mulensevent.theta_E",
    "mulensevent.mu_rel_mag",
    "source.rho",
)

# The coordinate configurations of the surgical plan: the default physical
# one and the four swaps.
_CONFIGS = {
    "default": {},
    "fitmurel": {"mulensevent": {"fitmurel": True}},
    "fitpirel": {"mulensevent": {"fitpirel": True}},
    "fitthetae": {"mulensevent": {"fitthetae": True}},
    "fitu0te": {"source": {"fitu0te": True}},
}


@pytest.fixture(scope="module")
def ob_workdir(tmp_path_factory):
    if not _OB_DIR.is_dir():
        pytest.skip("examples/ob140939 not present")
    work = tmp_path_factory.mktemp("ob140939") / "ob140939"
    shutil.copytree(_OB_DIR, work)
    return work


def _build(workdir, swaps):
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with open("ob140939.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        for comp, flags in swaps.items():
            config[comp][0].update(flags)
        system = System(copy.deepcopy(config), user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def built(ob_workdir):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = _build(ob_workdir, _CONFIGS[name])
        return cache[name]

    return get


def _fake_trace(system, model, n_chains=2, n_draws=6, seed=0):
    """A trace written exactly the way a PTDE run writes one: raw draws near
    the start, converted by the run's own compiled converter, assembled,
    unit-converted and stamped."""
    import logging

    from exozippy.run import _convert_posterior_to_user_units
    from exozippy.samplers._common import (
        assemble_inference_data,
        compile_conversions,
    )

    _, batched, raw_names, out_names = compile_conversions(model)
    ip = model.initial_point()
    rng = np.random.default_rng(seed)
    raw_start = {}
    stored = {}
    for v in model.free_RVs:
        val = np.asarray(ip[model.rvs_to_values[v].name], dtype=float)
        raw_start[v.name] = val
        stored[v.name] = val[None, None, ...] + 0.01 * rng.standard_normal(
            (n_chains, n_draws) + val.shape
        )
    idata = assemble_inference_data(
        stored,
        np.zeros((n_chains, n_draws)),
        n_draws,
        n_chains,
        raw_start,
        raw_names,
        out_names,
        batched,
        [0] * n_chains,
        "test",
        logging.getLogger("test"),
    )
    _convert_posterior_to_user_units(idata, system.get_parameter_lookup())
    stamp_structural_metadata(idata, system)
    return idata


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_CONFIGS))
def test_reported_derived_quantities_survive_a_dead_wrapup(
    built, name, tmp_path
):
    """
    Given a microlensing fit in each coordinate configuration,
    When its trace is written (the save that precedes wrap-up) and wrap-up
      then dies before distribute_posterior ever runs,
    Then t_E, theta_E, mu_rel_mag and the point-source rho are in the saved
      file's posterior group -- recoverable with no custom code.
    """
    system, model = built(name)
    det_names = {d.name for d in model.deterministics}
    missing = [n for n in _MUST_SURVIVE if n not in det_names]
    assert not missing, f"{name}: no Deterministic for {missing}"

    idata = _fake_trace(system, model)
    path = tmp_path / f"{name}_trace.nc"
    idata.to_netcdf(str(path))
    # ... and wrap-up dies here.  Only the file is left:
    reloaded = az.from_netcdf(str(path))
    for label in _MUST_SURVIVE:
        assert label in reloaded.posterior, f"{name}: {label} not in trace"
        assert np.all(np.isfinite(reloaded.posterior[label].values))


@pytest.mark.slow
def test_trace_values_equal_the_report_time_reconstruction(built):
    """
    Given the trace now carries a derived Deterministic,
    When distribute_posterior reads it,
    Then the value it reports equals what generate_posterior rebuilds from
      the sampled draws alone (the pre-2.6.14 path), so preferring the trace
      changes no reported number.
    """
    system, model = built("default")
    idata = _fake_trace(system, model)
    lookup = system.get_parameter_lookup()
    report_only = set(system.report_only_labels())
    assert {"mulensevent.t_E", "mulensevent.theta_E"} <= report_only

    posterior = az.extract(idata, keep_dataset=True)
    stripped = posterior.drop_vars(sorted(report_only))
    for label in sorted(report_only):
        param = lookup[label]
        from_trace = param.generate_posterior(posterior, param_lookup=lookup)
        rebuilt = param.generate_posterior(stripped, param_lookup=lookup)
        np.testing.assert_allclose(
            np.asarray(from_trace, dtype=float),
            np.asarray(rebuilt, dtype=float),
            rtol=1e-10,
            atol=0.0,
            err_msg=label,
        )


@pytest.mark.slow
def test_report_only_nodes_leave_the_convergence_scan_unchanged(built):
    """
    Given a trace that carries report-only Deterministics,
    When the burn-in / convergence scan picks its variables (run.py's
      analyze_idata via System.report_only_labels, mkparam's via the stamp),
    Then it sees exactly the variable set it saw before those nodes existed.
    """
    system, model = built("default")
    idata = _fake_trace(system, model)
    report_only = set(system.report_only_labels())
    assert report_only_vars(idata) == report_only
    assert idata.attrs[REPORT_ONLY_ATTR]

    before = convergence.default_var_names(
        idata.posterior.to_dataset().drop_vars(sorted(report_only))
    )
    after = convergence.default_var_names(
        idata.posterior, exclude=report_only_vars(idata)
    )
    assert after == before


@pytest.mark.slow
def test_point_source_rho_is_reported_with_no_logp_term(built):
    """
    Given a point-source event (finite_source: False),
    When the model is built,
    Then source.rho exists as a REPORTED (derived, consumed-by-nothing)
      parameter and no potential names it -- the defaults.yaml soft upper
      barrier is a finite-source statement and is not applied.
    """
    system, model = built("default")
    rho = system.source.rho
    assert np.all(rho.is_reported)
    assert rho.report_only_node
    assert not any("source.rho" in p.name for p in model.potentials)
    assert "source.rho" in {d.name for d in model.deterministics}


def test_a_trace_without_the_stamp_has_no_report_only_vars():
    """Every trace written before review 2.6.14 has none of these nodes, so
    an absent stamp means 'none' -- and an unreadable one raises."""
    idata = az.from_dict({"posterior": {"x": np.zeros((2, 3))}})
    assert report_only_vars(idata) == frozenset()
    idata.attrs[REPORT_ONLY_ATTR] = "not json"
    with pytest.raises(ValueError, match=REPORT_ONLY_ATTR):
        report_only_vars(idata)


def test_default_var_names_excludes_what_it_is_told():
    post = {
        "a": np.zeros((2, 3)),
        "a_raw": np.zeros((2, 3)),
        "b": np.zeros((2, 3)),
        "mode": np.zeros((2, 3)),
    }
    assert convergence.default_var_names(post) == ["a", "b"]
    assert convergence.default_var_names(post, exclude={"b"}) == ["a"]


def _scalar(label, **kwargs):
    from exozippy.components.parameter import Parameter

    defaults = dict(
        initval=0.5,
        init_scale=0.1,
        lower=0.0,
        upper=1.0,
        unit="",
        internal_unit="",
    )
    defaults.update(kwargs)
    return Parameter(label=label, **defaults)


def test_a_reported_derived_parameter_becomes_a_report_only_node():
    """
    Given a sampled parameter and three pure expressions of it,
    When they are built,
    Then the reported one is a Deterministic flagged report-only; one marked
      print_to_table: false is not a node (it is not reported); and a derived
      quantity of a PINNED input is a constant and gets no node either.
    """
    import pymc as pm

    with pm.Model() as model:
        a = _scalar("c.a")
        a.build_pymc()
        pinned = _scalar("c.pinned", sigma=0.0)
        pinned.build_pymc()
        reported = _scalar("c.twice", expression=lambda: 2.0 * a.value)
        reported.build_pymc()
        hidden = _scalar(
            "c.hidden", expression=lambda: 3.0 * a.value, print_to_table=False
        )
        hidden.build_pymc()
        const = _scalar("c.const", expression=lambda: 2.0 * pinned.value)
        const.build_pymc()

    names = {d.name for d in model.deterministics}
    assert "c.twice" in names and reported.report_only_node
    assert "c.a" in names and not a.report_only_node  # sampled: always a node
    assert "c.hidden" not in names and not hidden.report_only_node
    assert "c.const" not in names and not const.report_only_node
