"""The geocentric frame constants exist before any stage-1 probe.

The relaxation engine's t_E / pi_E relations are GEOCENTRIC (review
8.6.23): mu_geo = mu_helio - pi_rel * v_perp, with Earth's projected
velocity at t0_par supplied as two constants (ConfigManager.add_constant).
Until they are supplied the two mu_*_rel_geo relations are unusable, which
cuts the geocentric proper motion off from the stellar proper motions, the
only symbols its init_scale propagates from.  A config that seeds t_E AND
pi_E then makes a probe derive mu_rel_geo from those seeds alone, no symbol
of ``mu_rel_geo_mag**2 = mu_ra_rel_geo**2 + mu_dec_rel_geo**2`` has a
scale, and the engine raises.

The constants used to be supplied first by MulensInstrument.load_data, and
stage 1 runs components in the config's KEY ORDER, so any component listed
above ``mulensinstrument:`` that probes at stage 1 crashed prepare() --
OGLE-2011-BLG-0975 with ``evolutionarymodel:`` first, whose EEP seed probes
the source star.  MulensEvent now supplies the provisional constants when it
is CONSTRUCTED.  These tests probe straight after construction, which is
what any stage-1 reader that runs before MulensInstrument sees.

Fixture: the shipped DC2018_128 light curve (2L1S), on a scratch copy.
"""

import shutil
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent / "examples" / "DC2018_128"
DATA = "n20180816.Z087.WFIRST18.128.txt"

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / DATA).exists(), reason="DC2018_128 example data not present"
)

POSITION = {
    f"star.{s}.{c}": {"initval": v}
    for s in ("Lens", "Source")
    for c, v in (("ra", 267.595), ("dec", -28.982))
}

# The light-curve geometry as a user would seed it from a published
# solution: t_E and BOTH parallax components, no physical leaves.
SEEDS = {
    "source.Source.t_0": {"initval": 2458352.1},
    "source.Source.u_0": {"initval": 0.05},
    "mulensevent.t_E": {"initval": 18.2},
    "mulensevent.pi_E_N": {"initval": 0.3},
    "mulensevent.pi_E_E": {"initval": 0.2},
}

VPERP = ("mulensevent.0.earth_vperp_e", "mulensevent.0.earth_vperp_n")


def _construct(tmp_path, user_params, drop_data=False):
    from exozippy.system import System
    from exozippy.yamlio import load_system_config

    shutil.copy(EXAMPLE / "DC2018_128.yaml", tmp_path / "DC2018_128.yaml")
    (tmp_path / DATA).symlink_to(EXAMPLE / DATA)
    config = load_system_config(str(tmp_path / "DC2018_128.yaml"))
    config.pop("parameter_file", None)
    config["prefix"] = str(tmp_path / "fitresults" / "DC2018_128")
    config["mulensinstrument"][0]["file"] = str(tmp_path / DATA)
    if drop_data:
        del config["mulensinstrument"]
    return System(config, user_params=user_params)


def test_the_frame_constants_are_supplied_at_construction(tmp_path):
    """
    Given a microlensing config with microlensing data and a stated target
      position,
    When the System is constructed (no stage has run),
    Then Earth's projected velocity at the preliminary t0_par is already
      an engine constant, equal to what the event's own frame gives.
    """
    system = _construct(tmp_path, {**POSITION, **SEEDS})

    cm = system.config_manager
    assert set(VPERP) <= set(cm.constants)
    v_e, v_n = system.mulensevent._earth_vperp_en()
    assert cm.constants[VPERP[0]] == v_e
    assert cm.constants[VPERP[1]] == v_n
    # Earth moves ~6 AU/yr: a real projection, not the no-data zero
    assert (v_e**2 + v_n**2) ** 0.5 > 0.5


def test_a_probe_before_stage_1_judges_seeded_t_E_and_pi_E(tmp_path):
    """
    Given t_E, pi_E_N and pi_E_E seeded with every physical leaf left at
      its default,
    When a stage-1 reader probes a start before MulensInstrument has run
      (here: straight after construction),
    Then the probe answers.  Before the fix it raised "cannot judge
      'mulensevent.0.mu_rel_geo_mag**2 = ...': none of its symbols has an
      init_scale", because the geocentric relations' constants did not
      exist yet.
    """
    system = _construct(tmp_path, {**POSITION, **SEEDS})

    probed = system.config_manager.probe_start(
        ["star.1.distance", "mulensevent.0.t_E"]
    )

    assert probed["mulensevent.0.t_E"].user_value == pytest.approx(18.2)


def test_without_microlensing_data_the_constants_are_zero(tmp_path):
    """
    Given the same event with no microlensing photometry (nothing anchors
      t0_par),
    When the System is constructed,
    Then the constants are supplied as zero -- the two frames coincide, as
      register_frame_constants has them at stage 3 -- so a probe before
      stage 3 still has usable geocentric relations.
    """
    system = _construct(tmp_path, {**POSITION, **SEEDS}, drop_data=True)

    cm = system.config_manager
    assert [cm.constants[p] for p in VPERP] == [0.0, 0.0]
