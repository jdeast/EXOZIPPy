"""
Tests for seeding the SOURCE star's start from its own measured flux.

Without the seed, a microlensing-only source starts as a solar-mass
placeholder (mulensevent.py's logmass -0.5 hint) even though its apparent
magnitude is MEASURED: bootstrapped f_source through the zeropoint prior's
mu.  On DC2018 event 128 that placeholder start let the polish walk into a
swapped source/lens configuration.

The zeropoint mu is the USER's calibration statement: since review 2.2.21
defaults.yaml carries none, and a light curve seeds only when its zeropoint
is tied (mu AND sigma stated).  The KMT example's files are calibrated
magnitudes (flux = 10**(-0.4 m)), so a stated mu = 0.0 is correct there
(``_ZP_TIE`` below) and the config seeds; a wrong mu drives the measured
magnitude off the dwarf locus in one direction or the other and the guards
skip seeding with a warning that doubles as the miscalibration alarm.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


_WORKDIR = None

# The calibration the KMT files are in, stated: until review 2.2.21 this was
# the defaults.yaml zeropoint prior.
_ZP_TIE = {
    f"mulensinstrument.{n}.zeropoint": {"mu": 0.0, "sigma": 0.2}
    for n in ("KMTC04", "KMTS04", "KMTA04")
}


def _kmt_workdir():
    """A private copy of the example dir for THIS test module.

    Six xdist workers otherwise read and write (fitresults
    cache, whitening state) one shared examples/KMT-2019-BLG-1806
    concurrently -- the ezsuite-15362719 interference cluster.
    """
    global _WORKDIR
    if _WORKDIR is None:
        import shutil
        import tempfile

        _WORKDIR = Path(tempfile.mkdtemp(prefix="kmt_test_")) / "KMT"
        shutil.copytree(_KMT_DIR, _WORKDIR)
    return _WORKDIR


def _prepare_kmt(extra_params=None, drop_params=()):
    import os

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")

    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        # The shipped per-site zeropoint priors are removed: each test states
        # the tie it is about (or none).
        user_params = {
            k: v for k, v in user_params.items() if ".zeropoint" not in k
        }
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        for k in drop_params:
            user_params.pop(k, None)
        if extra_params:
            user_params.update(extra_params)
        system = System(config, user_params=user_params)
        system.prepare()
    finally:
        os.chdir(cwd)
    return system


def test_shipped_config_seeds_a_late_type_dwarf():
    """
    Given the shipped KMT config (calibrated magnitude files) with its
    zeropoint stated as 0 +/- 0.2 mag,
    When the system is prepared,
    Then the source (m_I ~ 21.3 at the 8 kpc bulge seed -> M_I ~ 6.8) is
    seeded as a K/M dwarf: the teff/radius hints exist (only this seeding
    writes them), logmass moved off mulensevent.py's -0.5 placeholder, and the
    resolved start carries the seed.

    The shipped params file's own Source teff/radius starts are dropped
    here: user entries outrank data hints by design (verified in this
    test's counterpart below), and this test is about the hint path.
    """
    system = _prepare_kmt(
        extra_params=_ZP_TIE,
        drop_params=("star.Source.radius", "star.Source.teff"),
    )
    cm = system.config_manager
    src = int(system.mulensevent.source_map[0])
    for param in ("logmass", "teff", "radius", "teffsed", "radiussed"):
        assert f"star.{src}.{param}" in cm.hints, param

    teff_seed = float(cm.hints[f"star.{src}.teffsed"])
    logmass_seed = float(cm.hints[f"star.{src}.logmass"])
    assert 3000.0 < teff_seed < 5300.0
    assert abs(logmass_seed - (-0.5)) > 0.01  # not the placeholder
    assert 0.2 < 10.0**logmass_seed < 0.9

    # The engine injects solved initvals back into user_params at finalize;
    # Parameters do not exist until build_model.
    resolved = cm.user_params[f"star.{src}.teffsed"]
    assert np.isclose(float(resolved["initval"]), teff_seed, rtol=1e-3)


def test_user_start_outranks_the_seed():
    """
    Given the shipped config, whose params file explicitly starts the
    Source at a bulge turnoff (teff 5800, radius 1.2),
    When the system is prepared,
    Then the hint is still computed but the USER start wins the resolve --
    data hints yield to explicit user values by design.
    """
    system = _prepare_kmt(extra_params=_ZP_TIE)
    cm = system.config_manager
    src = int(system.mulensevent.source_map[0])
    assert f"star.{src}.teffsed" in cm.hints  # seed computed...
    resolved = cm.user_params[f"star.{src}.teff"]
    assert np.isclose(float(resolved["initval"]), 5800.0)  # ...user won


def test_no_stated_zeropoint_does_not_seed():
    """
    Given the KMT config with no zeropoint entry, so no tie (review
    2.2.21) --
    When the system is prepared,
    Then no source-flux seed is written: with no calibration statement the
    baseline flux says nothing about the source's magnitude.  (The SAME
    config with the tie stated seeds, test_shipped_config_seeds_a_late_type_
    dwarf, so this is the stated tie's doing and not a broken seeder.)
    """
    system = _prepare_kmt(
        drop_params=("star.Source.radius", "star.Source.teff")
    )
    cm = system.config_manager
    src = int(system.mulensevent.source_map[0])
    for param in ("teff", "radius", "teffsed", "radiussed"):
        assert f"star.{src}.{param}" not in cm.hints, param


def test_faint_miscalibration_guard(caplog):
    """
    Given a zeropoint mu 22 mag too bright for these calibrated files,
    When the system is prepared,
    Then the measured source magnitude lands absurdly below the dwarf
    locus, seeding is skipped with the sub-stellar warning, and no
    teff/radius hints are written.
    """
    with caplog.at_level(logging.WARNING):
        system = _prepare_kmt(
            extra_params={
                f"mulensinstrument.{n}.zeropoint": {"mu": 22.0, "sigma": 0.02}
                for n in ("KMTC04", "KMTS04", "KMTA04")
            }
        )
    cm = system.config_manager
    src = int(system.mulensevent.source_map[0])
    assert f"star.{src}.teff" not in cm.hints
    assert any("far fainter" in r.message for r in caplog.records)


def test_bright_giant_guard(caplog):
    """
    Given a zeropoint mu that makes the source brighter than the whole
    dwarf locus (a giant source, or a badly wrong calibration),
    When the system is prepared,
    Then seeding is skipped with the giant warning and no teff/radius
    hints are written.
    """
    with caplog.at_level(logging.WARNING):
        system = _prepare_kmt(
            extra_params={
                f"mulensinstrument.{n}.zeropoint": {"mu": -8.0, "sigma": 0.02}
                for n in ("KMTC04", "KMTS04", "KMTA04")
            }
        )
    cm = system.config_manager
    src = int(system.mulensevent.source_map[0])
    assert f"star.{src}.teff" not in cm.hints
    assert any("BRIGHTER than the whole" in r.message for r in caplog.records)
