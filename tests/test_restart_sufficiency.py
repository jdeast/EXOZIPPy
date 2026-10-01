"""A restart file must not re-trigger a stage-1 seeder (review 8.6.22).

The seeder is the built-in peak finder (``MulensInstrument._peak_find_seeds``):
it runs at STAGE 1, inside ``MulensInstrument.load_data``, whenever
``peakfind.plan_peak_find`` finds t_0, u_0 or t_E without an INFORMED start
(provenance above PRECEDENCE_DEFAULT, directly or through a relation).  So the
question has to be asked by BUILDING, not by probing after ``prepare()`` has
pushed every hint.  The regression this pins was found on the seeder that
preceded it (an external fitter, removed 2026-10-01), where a re-run cost minutes and
on examples/DC2018_128 a crash; the same derivability question gates the peak
finder, and a restart file that reads as "nobody said so" would have it
replace a sampled solution's trajectory start with a fresh PSPL fit.

WHAT BROKE.  mkparam writes every SAMPLED parameter and no derived one, so a
restart file never names `t_E`; the engine has to derive it.  Provenance ranks
were not run to a fixed point, so on DC2018_128 `t_E` inherited rank 19 from
a `theta_E` that was itself still at the Condition A floor when the relation
fired, and kept 19 after `theta_E` reached 80.  The informed test is
`> PRECEDENCE_DEFAULT` (20), so a complete restart file read as "nobody told
us".  ob161003 escaped it only because the same relation happened to fire
after `theta_E` was promoted -- order, not physics.  Fixed by
`ConfigManager._propagate_provenance_to_fixed_point`.

THE VEHICLE IS A REAL RESTART FILE, and that matters: the first version of
this test used "the shipped params file minus its literal `t_E`", which is
NOT what mkparam produces.  Dropping `t_E` without adding the sampled
parameters a restart file carries leaves the mass/distance/proper-motion
chain unpinned, so `t_E` genuinely cannot be derived and the seeder is right
to run.  So build the system, fabricate a trace over its real sampled
variables, and let mkparam write the file -- the same path a
second-iteration fit takes.

BOTH TOPOLOGIES, because the bug was topology-dependent in a way one example
could not show: DC2018_128 is 2L1S with a PLANET companion and failed;
ob161003 is 2S2L with a star companion and passed. Testing only the one that
passed is what let this ship.
"""

import copy
import os
import pathlib
import shutil

import pytest
import yaml

# Reuse the round-trip file's builder and trace fabrication rather than
# keeping a second copy: a divergence between them would silently change
# what "a restart file" means in one of the two suites.
from test_mkparam_roundtrip import _build, _fabricate_trace

from exozippy.components.mulensing import peakfind
from exozippy.mkparam import write_param_file
from exozippy.system import System

EXAMPLES_DIR = pathlib.Path(__file__).parent / ".." / "examples"

pytestmark = pytest.mark.slow

CASES = [
    ("DC2018_128", "DC2018_128.yaml"),
    ("ob161003", "ob161003.yaml"),
]


class _Triggered(BaseException):
    """Raised the instant the seeder starts its search, so none is run.

    A BaseException because ``_peak_find_seeds`` deliberately catches every
    Exception from the search (a seeder failure is not fatal)."""


def _runs_the_seeder(work, config_name, params):
    """True if building `config_name` in `work` with `params` starts the
    peak finder's search (aborted at once rather than run)."""
    with open(os.path.join(work, config_name)) as fh:
        config = yaml.safe_load(fh)
    # An in-memory params dict, so the file on disk is untouched.
    config["parameter_file"] = None

    def _trip(*args, **kwargs):
        raise _Triggered()

    cwd = os.getcwd()
    real = peakfind.find_pspl_seed
    peakfind.find_pspl_seed = _trip
    try:
        os.chdir(work)
        System(config, copy.deepcopy(params)).prepare()
        return False
    except _Triggered:
        return True
    finally:
        peakfind.find_pspl_seed = real
        os.chdir(cwd)


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def restart(request, tmp_path_factory):
    """A REAL mkparam restart file for one example, plus its shipped params."""
    name, config_name = request.param
    work = str(tmp_path_factory.mktemp(name))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(str(EXAMPLES_DIR / name), work)

    system, model, config = _build(work, config_name)
    trace = _fabricate_trace(system, model, os.path.join(work, "trace.nc"))
    written = write_param_file(
        config,
        base_dir=work,
        trace_path=trace,
        output_path=os.path.join(work, "restart.params.yaml"),
    )
    with open(written) as fh:
        restart_params = yaml.safe_load(fh) or {}

    with open(os.path.join(work, config_name)) as fh:
        shipped_cfg = yaml.safe_load(fh)
    with open(os.path.join(work, shipped_cfg["parameter_file"])) as fh:
        shipped_params = yaml.safe_load(fh) or {}

    return {
        "name": name,
        "config_name": config_name,
        "work": work,
        "restart": restart_params,
        "shipped": shipped_params,
    }


def test_the_shipped_params_file_is_sufficient(restart):
    """The control: as shipped, no example runs the seeder.

    Without it the test below would also pass if the probe had simply stopped
    triggering altogether, which is a different bug.
    """
    assert not _runs_the_seeder(
        restart["work"], restart["config_name"], restart["shipped"]
    ), (
        f"{restart['name']} runs the peak finder from its own shipped params "
        f"file, which names every required observable outright"
    )


def test_a_restart_file_does_not_re_run_the_seeder(restart):
    """The contract: a restart file's DERIVED t_E must count as sufficient.

    mkparam writes only sampled coordinates, so a restart file never names
    t_E.  If deriving it does not count, the documented
    fit-then-restart-from-the-MAP workflow re-seeds the trajectory every
    second iteration.
    """
    assert not _runs_the_seeder(
        restart["work"], restart["config_name"], restart["restart"]
    ), (
        f"{restart['name']}'s own mkparam restart file ran the peak finder: "
        f"the derived t_E is not being credited to the sampled parameters "
        f"that determine it (review 8.6.22)"
    )


def test_the_restart_file_does_not_name_t_E(restart):
    """The premise of the test above, pinned so it cannot rot silently.

    If mkparam ever started writing a derived t_E, the contract test would
    pass for an uninteresting reason -- a literal start -- while the
    derivability path it exists to guard
    went unexercised.  JDE's ruling is that mkparam writes every SAMPLED
    parameter and nothing derived, so this is also that ruling's pin.
    """
    named = [k for k in restart["restart"] if k.endswith(".t_E")]
    assert not named, (
        f"{restart['name']}'s restart file names {named}; t_E is derived, so "
        f"writing it would over-determine the next fit and would hide "
        f"whether the derivability path still works"
    )
