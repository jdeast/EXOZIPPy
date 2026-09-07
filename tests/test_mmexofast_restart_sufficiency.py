"""A restart file must not trigger an MMEXOFAST re-run (review 8.6.22).

`user_hints_sufficient` decides whether to launch MMEXOFAST -- minutes of
fitting, and on examples/DC2018_128 a crash (`ValueError: Parameter q has to
be larger than 0, not 0.0`) rather than a warning.  It requires
`mulensevent.0.t_E`, which mkparam NEVER writes because t_E is derived and
writing it would over-determine the next fit.  So every restart file depends
on t_E being DERIVABLE at the moment the probe asks.

It is not, in general.  t_E has one relation, `calc_t_E(theta_E,
mu_rel_geo_mag)`, and `mu_rel_geo_mag` needs `earth_vperp_e/n` -- context
constants injected in `add_parameter` at stage 6 -- while the probe runs at
stage 1, inside `MulensInstrument.load_data`.  The geocentric chain is
structurally closed when the question is put.

This is PRE-EXISTING, not caused by the mulensevent split: the geo deps
arrived with 51a130d9 (2026-08-08, an ancestor of master) and master's probe
requires `lens.0.t_E` through the same single relation.  The frame fix is
correct; the probe's timing is the problem.  Fix directions are recorded in
review 8.6.22 -- the leading one is to require the INGREDIENTS (theta_E and
mu_rel_mag) rather than the derived quantity -- and none is implemented,
because choosing when to launch an expensive user-visible fit is a ruling,
not a cleanup.

WHY THE VEHICLE IS "SHIPPED PARAMS MINUS t_E" and not a real restart file: a
restart file is precisely the shipped file without its derived entries, and
the literal `mulensevent.t_E` seed is what short-circuits the probe today.
Dropping just that key reproduces the production failure with no trace, no
fit and no fixture.  It also shows the defect is DATA-DEPENDENT rather than
universal, which is why it went unnoticed: ob161003's real restart file
carries all four stars' masses, distances and proper motions and does pass,
while DC2018_128's does not.
"""

import logging
import os
import pathlib
import shutil
import tempfile

import pytest
import yaml

from exozippy.system import System

EXAMPLES_DIR = pathlib.Path(__file__).parent / ".." / "examples"

pytestmark = pytest.mark.slow


class _Triggered(Exception):
    """Raised the instant the probe decides to fit, so no fit is run."""


class _Watch(logging.Handler):
    def emit(self, record):
        if "no sufficient user start values" in record.getMessage():
            raise _Triggered()


def _probe_triggers(example, config_name, drop_t_E):
    """True if building `example` launches MMEXOFAST.

    Aborts at the decision point rather than letting the fit start -- a test
    that actually ran MMEXOFAST would take minutes and, on DC2018_128, crash.
    """
    src = EXAMPLES_DIR / example
    work = pathlib.Path(tempfile.mkdtemp()) / example
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("fitresults"))

    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(config_name) as fh:
            config = yaml.safe_load(fh)
        with open(config["parameter_file"]) as fh:
            params = yaml.safe_load(fh) or {}

        if drop_t_E:
            dropped = [k for k in params if k.endswith(".t_E")]
            assert dropped, (
                f"{example} does not name t_E literally, so this vehicle "
                f"cannot show anything"
            )
            for key in dropped:
                params.pop(key)

        # An in-memory params dict, so the file on disk still names t_E.
        config["parameter_file"] = None

        handler = _Watch()
        logging.getLogger().addHandler(handler)
        previous = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        try:
            System(config, params).prepare()
            return False
        except _Triggered:
            return True
        finally:
            logging.getLogger().removeHandler(handler)
            logging.getLogger().setLevel(previous)
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "example,config_name",
    [
        ("DC2018_128", "DC2018_128.yaml"),
        ("ob161003", "ob161003.yaml"),
    ],
)
def test_a_literal_t_E_seed_is_sufficient(example, config_name):
    """The control: as shipped, neither example re-runs MMEXOFAST.

    Without this, the xfail below would also pass if the probe had simply
    become unconditionally triggering, which is a different bug.
    """
    assert not _probe_triggers(example, config_name, drop_t_E=False), (
        f"{example} launches MMEXOFAST from its own shipped params file, "
        f"which names every required observable outright"
    )


@pytest.mark.parametrize(
    "example,config_name",
    [
        ("DC2018_128", "DC2018_128.yaml"),
        ("ob161003", "ob161003.yaml"),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="review 8.6.22: t_E derives only through mu_rel_geo_mag, which "
    "needs earth_vperp_* from stage 6, while the probe asks at stage 1 -- so "
    "a params file without a literal t_E re-runs MMEXOFAST. Pre-existing "
    "(51a130d9). Fix is a ruling; see 8.6.22 (a)/(b)/(c).",
)
def test_a_derivable_t_E_is_also_sufficient(example, config_name):
    """The contract: a restart file's DERIVED t_E must count as sufficient.

    mkparam writes only sampled coordinates, so a restart file never names
    t_E.  If deriving it does not count, the documented
    fit-then-restart-from-the-MAP workflow re-runs MMEXOFAST every second
    iteration -- and on DC2018_128 dies inside it.
    """
    assert not _probe_triggers(example, config_name, drop_t_E=True), (
        f"{example} launches MMEXOFAST once its t_E is derived rather than "
        f"named, which is what every mkparam restart file looks like"
    )
