"""Source orbital motion (xallarap) for EVERY source (review 8.6.9, the
source twin of 8.6.33; conventions.md C25).

Until 2026-10 one event-level `mulensevent.source_orbit:` moved a single
luminous source, and a multi-source event could not have source motion at
all.  Now each source entry names its own orbit and sits alone on one side
of it: on the primary side its offset is the primary track
(a m_companion / M), on the companion side the opposite track
(-a m_primary / M).  Two sources on one orbit are the linked binary source.
Pinned here:

1. A linked binary source: the two sources' xallarap shifts are exactly
   opposite and in the mass ratio, dtau_2 = -(m_1/m_2) dtau_1, and the model
   builds with a finite logp.
2. A moving source beside a static one: the static source has no shift.
3. Each source reads its OWN orbit when two sources are on different orbits.
4. Refusals: a source inside a group of bodies (hierarchical), and a binary
   whose other source does not name the shared orbit.
"""

import numpy as np
import pytensor
import pytest
from test_xallarap import _T0_PAR, _write_lc

pytestmark = pytest.mark.slow

M1, M2 = 1.0, 0.6  # the two sources' masses


def _orbit(name, primary, companion):
    return {"name": name, "primary": [primary], "companion": [companion]}


def _system(tmp_path, stars, orbits, sources, extra_params=None):
    from exozippy.system import System

    tmp_path.mkdir(parents=True, exist_ok=True)
    lc = _write_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": nm} for nm in stars],
        "orbit": orbits,
        "mulensevent": [{"finite_source": False, "t0_par": _T0_PAR}],
        "lens": [{"body": "star.L1"}],
        "source": sources,
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "mulensevent.t_E": {"initval": 45.0},
        "mulensevent.pi_E_N": {"initval": 0.08},
        "mulensevent.pi_E_E": {"initval": 0.11},
        "star.L1.mass": {"initval": 0.6},
        "star.L1.distance": {"initval": 4000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for j, src in enumerate(sources):
        nm = src["body"].split(".")[1]
        params[f"source.{nm}.t_0"] = {"initval": _T0_PAR + 3.0 * j}
        params[f"source.{nm}.u_0"] = {"initval": 0.15 + 0.1 * j}
    for o in orbits:
        nm = o["name"]
        params[f"orbit.{nm}.period"] = {"initval": 120.0}
        params[f"orbit.{nm}.tc"] = {"initval": _T0_PAR - 20.0}
        params[f"orbit.{nm}.secosw"] = {"initval": 0.4}
        params[f"orbit.{nm}.sesinw"] = {"initval": 0.3}
        params[f"orbit.{nm}.cosi"] = {"initval": 0.4}
        params[f"orbit.{nm}.bigomega"] = {"initval": 70.0}
    for nm in stars:
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
        if nm != "L1":
            params[f"star.{nm}.distance"] = {"initval": 8000.0}
    params["star.S1.mass"] = {"initval": M1}
    params["star.S2.mass"] = {"initval": M2}
    params.update(extra_params or {})
    system = System(config, user_params=params)
    system.prepare()
    return system


def _shifts(system, model, times):
    ev = system.mulensevent
    nodes, slots = [], []
    for j in range(ev.n_sources):
        sh = ev._source_offset_series(times, system, j)
        slots.append(sh is not None)
        if sh is not None:
            nodes += list(sh)
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(nodes),
        on_unused_input="ignore",
    )
    vals = iter(fn(*[start[v.name] for v in model.value_vars]))
    return [(next(vals), next(vals)) if on else None for on in slots]


def _times():
    return np.linspace(_T0_PAR - 40.0, _T0_PAR + 40.0, 33)


def test_a_linked_binary_source_moves_both_sources(tmp_path):
    system = _system(
        tmp_path,
        ["L1", "S1", "S2"],
        [_orbit("B", "S1", "S2")],
        [
            {"body": "star.S1", "orbit": "B"},
            {"body": "star.S2", "orbit": "B"},
        ],
    )
    ev = system.mulensevent
    assert ev.source_orbit_side == [1, -1]
    model = system.build_model()
    (dt1, du1), (dt2, du2) = _shifts(system, model, _times())
    assert np.max(np.abs(dt1)) > 1e-6  # it really moves
    # opposite tracks, amplitudes in the mass ratio
    np.testing.assert_allclose(dt2, -(M1 / M2) * dt1, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(du2, -(M1 / M2) * du1, rtol=1e-9, atol=1e-15)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_a_static_source_beside_a_moving_one(tmp_path):
    system = _system(
        tmp_path,
        ["L1", "S1", "SComp", "S2"],
        [_orbit("A", "S1", "SComp")],
        [{"body": "star.S1", "orbit": "A"}, {"body": "star.S2"}],
        extra_params={"star.SComp.mass": {"initval": 0.3}},
    )
    model = system.build_model()
    moving, static = _shifts(system, model, _times())
    assert static is None
    assert np.max(np.abs(moving[0])) > 1e-6


def test_each_source_reads_its_own_orbit(tmp_path):
    system = _system(
        tmp_path,
        ["L1", "S1", "C1", "S2", "C2"],
        [_orbit("A", "S1", "C1"), _orbit("B", "S2", "C2")],
        [
            {"body": "star.S1", "orbit": "A"},
            {"body": "star.S2", "orbit": "B"},
        ],
        extra_params={
            "star.C1.mass": {"initval": 0.3},
            "star.C2.mass": {"initval": 0.3},
            # a different period on B, so the two tracks must differ
            "orbit.B.period": {"initval": 37.0},
        },
    )
    assert system.mulensevent.source_orbit_idx == [0, 1]
    model = system.build_model()
    (dt1, _), (dt2, _) = _shifts(system, model, _times())
    assert np.max(np.abs(dt1 - dt2)) > 1e-6


def test_a_source_inside_a_group_is_refused(tmp_path):
    orbit = {
        "name": "B",
        "primary": ["S1", "S2"],
        "companion": ["SComp"],
    }
    with pytest.raises(NotImplementedError, match="ALONE"):
        _system(
            tmp_path,
            ["L1", "S1", "S2", "SComp"],
            [orbit],
            [{"body": "star.S1", "orbit": "B"}, {"body": "star.S2"}],
            extra_params={"star.SComp.mass": {"initval": 0.3}},
        )


def test_a_binary_partner_must_name_the_orbit(tmp_path):
    with pytest.raises(ValueError, match="does not name that orbit"):
        _system(
            tmp_path,
            ["L1", "S1", "S2"],
            [_orbit("B", "S1", "S2")],
            [{"body": "star.S1", "orbit": "B"}, {"body": "star.S2"}],
        )
