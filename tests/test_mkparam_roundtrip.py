"""The mkparam round-trip on a real built system (design 8.6.17, section 6).

`test_mkparam.py` exercises the writer against hand-built configs and never
builds a System -- fast, and right for the writer's own logic.  This file
covers the join the mulensevent split moved: the writer names elements from
the CONFIG while the values come from a TRACE, so a component whose element
count or instance list changed can silently mis-file a start value with
nothing raising.

ob161003 is the case, because it is the only shipped 2S2L: two sources and
two lens bodies, so every vector length here is 2 and a wrong pairing lands
on a real parameter rather than an IndexError.  Pre-split, `lens.t_0`
element 0 was written as `lens.Lens.t_0` -- a SOURCE value under the LENS's
name -- because the lens component held one instance for a two-element
vector.  After the split each vector's length equals its own component's
instance count.

Two things here are load-bearing and neither is obvious:

  * THE MASKED LENS PRIMARY must receive no start value.  Lens element 0 is
    the primary and does not sample; mkparam gates on the trace's
    `element_roles["sampled"]` stamp and never looks at `["active"]`, so this
    is only safe because the split leaves `is_sampled` False there too.
    `test_the_roles_stamp_is_what_protects_the_masked_primary` strips the
    stamp and shows a start value DOES land on it -- the assertion has teeth
    and the stamp is what gives them.

  * ELEMENT NAMES depend on whether the config was mutated.  mkparam never
    builds a System, but `System.prepare()` mutates the config in place,
    filling `name:` from the new path-valued `body:` key.  Called with a
    prepared config it writes `source.SourceA.t_0`; with one read fresh off
    disk it writes `source.0.t_0`.  Both must reload -- numeric indices are
    the canonical internal spelling -- so both are tested.
"""

import copy
import json
import os
import shutil

import numpy as np
import pytest
import xarray as xr
import yaml

from exozippy import diagnostics, trace_meta
from exozippy.components.mulensing import mmexofast_support
from exozippy.mkparam import _get_instance_names, write_param_file
from exozippy.system import System

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "ob161003",
)
CONFIG = "ob161003.yaml"

# The nudge that makes a written start distinguishable from the start it was
# built at, so "the value landed" cannot pass by accident.
NUDGE = 0.01


def _build(work):
    """Build the example in `work`, returning (system, model, config)."""
    with open(os.path.join(work, CONFIG)) as fh:
        config = yaml.safe_load(fh)
    with open(os.path.join(work, config["parameter_file"])) as fh:
        params = yaml.safe_load(fh) or {}
    cwd = os.getcwd()
    try:
        os.chdir(work)
        system = System(config, params)
        system.prepare()
        return system, system.build_model(), config
    finally:
        os.chdir(cwd)


def _fabricate_trace(system, model, path, stamp_roles=True):
    """A one-draw trace over the REAL sampled vars at their REAL shapes.

    mkparam keeps only non-raw posterior vars that HAVE a `_raw` counterpart
    (mkparam.py:796), so a trace of raw vars alone makes it correctly write
    nothing -- which looks exactly like a broken writer.  Both names go in,
    at full vector length.
    """
    import arviz as az
    import pytensor

    start = system.get_raw_start(model)
    pairs = [
        (raw[:-4], raw, np.atleast_1d(np.asarray(val, dtype=float)))
        for raw, val in start.items()
        if raw.endswith("_raw") and raw[:-4] in model.named_vars
    ]
    assert pairs, "no sampled value/raw pairs -- the vehicle is broken"

    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(
            [model.named_vars[n] for n, _, _ in pairs]
        ),
        on_unused_input="ignore",
    )
    evaluated = fn(*[start[v.name] for v in model.value_vars])

    data_vars = {}
    for (value_name, raw_name, raw_arr), val in zip(pairs, evaluated):
        v = np.atleast_1d(np.asarray(val, dtype=float)) + NUDGE
        data_vars[value_name] = xr.DataArray(
            v[None, None, :], dims=["chain", "draw", value_name + "_dim"]
        )
        data_vars[raw_name] = xr.DataArray(
            raw_arr[None, None, :], dims=["chain", "draw", raw_name + "_dim"]
        )

    idata = az.from_dict(
        {
            "posterior": xr.Dataset(data_vars),
            "sample_stats": xr.Dataset(
                {
                    "lp": xr.DataArray(
                        np.array([[-10.0]]), dims=["chain", "draw"]
                    )
                }
            ),
        }
    )
    if stamp_roles:
        idata.attrs[trace_meta.ROLES_ATTR] = json.dumps(
            trace_meta.element_roles(system), sort_keys=True
        )
    idata.to_netcdf(str(path))
    return path


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    """Build once; write a restart file from a prepared and a raw config."""
    work = str(tmp_path_factory.mktemp("ob161003"))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(EXAMPLE, work)

    system, model, prepared_config = _build(work)
    trace = _fabricate_trace(system, model, os.path.join(work, "trace.nc"))

    out = {}
    for label, config in (
        ("prepared", prepared_config),
        # Fresh off disk: `body:` present, `name:` absent.
        ("raw", yaml.safe_load(open(os.path.join(work, CONFIG)))),
    ):
        path = write_param_file(
            config,
            base_dir=work,
            trace_path=trace,
            output_path=os.path.join(work, "restart_%s.params.yaml" % label),
        )
        with open(path) as fh:
            out[label] = yaml.safe_load(fh) or {}

    return {"work": work, "written": out, "trace": trace, "system": system}


# ---------------------------------------------------------------------------
# What the split moved
# ---------------------------------------------------------------------------


def test_no_source_value_is_filed_under_the_lens(roundtrip):
    """The pre-split defect: `lens.Lens.t_0` held source 0's t_0."""
    for label, written in roundtrip["written"].items():
        misfiled = [
            k
            for k in written
            if k.startswith("lens.")
            and any(p in k for p in (".t_0", ".u_0", ".rho"))
        ]
        assert not misfiled, "%s config: source values under lens.*: %s" % (
            label,
            misfiled,
        )


def test_the_sources_get_one_start_each(roundtrip):
    """Two sources, two t_0s -- under `source.*`, one per body."""
    for label, written in roundtrip["written"].items():
        t0 = sorted(k for k in written if k.endswith(".t_0"))
        assert len(t0) == 2, "%s config: expected 2 t_0 keys, got %s" % (
            label,
            t0,
        )
        assert all(k.startswith("source.") for k in t0), t0
        # Distinct values: one source's start must not be written twice.
        vals = {round(float(written[k]["initval"]), 6) for k in t0}
        assert len(vals) == 2, "both sources got the same t_0: %s" % vals


def test_nothing_is_written_onto_the_masked_lens_primary(roundtrip):
    """Lens element 0 is the primary and does not sample.

    A start value there is dropped at build and, before 2026-09, silently:
    it is the masked-primary off-by-one the split is most exposed to.
    """
    for label, written in roundtrip["written"].items():
        on_primary = [
            k
            for k in written
            if k.startswith("lens.0.") or k.startswith("lens.Lens.")
        ]
        assert not on_primary, (
            "%s config: wrote onto the masked primary: %s"
            % (
                label,
                on_primary,
            )
        )


def test_the_roles_stamp_is_what_protects_the_masked_primary(tmp_path):
    """The control for the test above -- proof it can fail.

    mkparam learns which elements sample from the trace's `element_roles`
    stamp; the raw variable's length says how many, never which.  Without the
    stamp it treats every element as sampled and writes a start onto the
    masked lens primary.  If this ever stops happening, the assertion above
    has gone vacuous and is no longer watching anything.
    """
    work = str(tmp_path / "ob161003")
    shutil.copytree(EXAMPLE, work)
    system, model, config = _build(work)
    trace = _fabricate_trace(
        system, model, os.path.join(work, "trace.nc"), stamp_roles=False
    )

    path = write_param_file(
        config,
        base_dir=work,
        trace_path=trace,
        output_path=os.path.join(work, "restart.params.yaml"),
    )
    with open(path) as fh:
        written = yaml.safe_load(fh) or {}

    on_primary = [
        k
        for k in written
        if k.startswith("lens.0.") or k.startswith("lens.Lens.")
    ]
    assert on_primary, (
        "stripping the element-roles stamp did NOT put a start value on the "
        "masked lens primary, so the guarded test above proves nothing"
    )


def test_a_raw_config_writes_canonical_index_spellings(roundtrip):
    """mkparam never builds a System, so `name:` may be absent.

    `System.prepare()` fills `name:` from the path-valued `body:` key in
    place.  Off disk that has not happened, and `_get_instance_names` falls
    back to the index -- legal, since indices are the canonical internal
    spelling, and the reload tests cover both.
    """
    prepared = roundtrip["written"]["prepared"]
    raw = roundtrip["written"]["raw"]

    assert "source.SourceA.t_0" in prepared, sorted(
        k for k in prepared if k.startswith("source.")
    )
    assert "source.0.t_0" in raw, sorted(
        k for k in raw if k.startswith("source.")
    )
    # Same information either way: one start per source, same values.
    assert len(prepared) == len(raw)


def test_the_direction_pair_is_written_as_an_angle(roundtrip):
    """`alpha` is sampled as an (xalpha, yalpha) pair and written collapsed.

    Writing the raw pair back would restate the angle as two Cartesian
    components whose length also carries the (fixed) magnitude, so mkparam
    emits the angle.  Pinned because a restart file naming `xalpha` would
    reload as a redundant constraint.
    """
    written = roundtrip["written"]["prepared"]
    angle = [k for k in written if k.endswith(".alpha")]
    assert angle, "no alpha written: %s" % sorted(
        k for k in written if k.startswith("lens.")
    )
    assert not [
        k for k in written if k.endswith(".xalpha") or k.endswith(".yalpha")
    ], "the raw direction pair was written alongside the angle"
    # The companion's angle, from the pair the trace actually carried.
    assert 0.0 <= float(written[angle[0]]["initval"]) < 360.0


# ---------------------------------------------------------------------------
# It has to reload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["prepared", "raw"])
def test_the_restart_file_needs_no_mmexofast_rerun(roundtrip, spelling):
    """The gate that regressed last time the spellings moved.

    `push_seed_hints` and `user_hints_sufficient` drifted apart once already:
    the probe reported "insufficient" on a perfectly good restart file, an
    expensive MMEXOFAST run followed, and its output was then discarded.
    """
    work = roundtrip["work"]
    config = yaml.safe_load(open(os.path.join(work, CONFIG)))
    config["parameter_file"] = None

    cwd = os.getcwd()
    try:
        os.chdir(work)
        system = System(config, copy.deepcopy(roundtrip["written"][spelling]))
        system.prepare()
        n_companions = system.lens.n_elements - 1
        assert mmexofast_support.user_hints_sufficient(
            system.config_manager,
            n_companions >= 1,
            bool(getattr(system.mulensevent, "finite_source", False)),
        ), "a restart file is not sufficient to skip an MMEXOFAST re-run"
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("spelling", ["prepared", "raw"])
def test_every_written_start_lands_on_its_parameter(roundtrip, spelling):
    """Reload, and check each written value against the built initval.

    Compared through `from_internal`: `Parameter.initval` is internal, the
    params file is user units, and `star.ra` differs by exactly the radian
    conversion (264.10513 deg -> 4.60950).  CLAUDE.md -- never hand-write a
    factor, the direction is in the name.
    """
    work = roundtrip["work"]
    written = roundtrip["written"][spelling]
    config = yaml.safe_load(open(os.path.join(work, CONFIG)))
    config["parameter_file"] = None

    cwd = os.getcwd()
    try:
        os.chdir(work)
        system = System(config, copy.deepcopy(written))
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)

    starts = {
        k: v["initval"]
        for k, v in written.items()
        if isinstance(v, dict) and "initval" in v
    }
    assert starts, "nothing was written -- the vehicle is broken"

    matched, wrong = set(), []
    for p in system.get_all_parameters():
        comp_key, _, param = p.label.rpartition(".")
        names = _get_instance_names(config, comp_key)
        internal = np.atleast_1d(np.asarray(p.initval, dtype=float))
        for i, raw_val in enumerate(internal):
            candidates = [
                "%s.%d.%s" % (comp_key, i, param),
                "%s.%s" % (comp_key, param),
            ]
            if i < len(names):
                candidates.insert(0, "%s.%s.%s" % (comp_key, names[i], param))
            key = next((c for c in candidates if c in starts), None)
            if key is None:
                continue
            matched.add(key)
            want = float(starts[key])
            got = float(p.from_internal(raw_val, index=i))
            if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                wrong.append((key, want, got))

    assert not wrong, "written starts did not land: %s" % wrong[:6]
    assert not sorted(set(starts) - matched), (
        "written keys matched no parameter: %s"
        % sorted(set(starts) - matched)[:8]
    )

    auditor = diagnostics.ModelAuditor(
        model, system, system.get_raw_start(model)
    )
    assert not auditor.check_unused_yaml(), (
        "the restart file it just wrote has keys it cannot read back: %s"
        % auditor.check_unused_yaml()
    )
