"""A restart file must not trigger an MMEXOFAST re-run (review 8.6.22).

`user_hints_sufficient` decides whether to launch MMEXOFAST -- minutes of
fitting, and on examples/DC2018_128 a crash (`ValueError: Parameter q has to
be larger than 0, not 0.0`) rather than a warning.  It runs at STAGE 1,
inside `MulensInstrument.load_data`, so the question has to be asked by
BUILDING, not by calling the probe after `prepare()` has pushed every hint.

WHAT BROKE.  mkparam writes every SAMPLED parameter and no derived one, so a
restart file never names `t_E`; the probe has to derive it.  Provenance ranks
were not run to a fixed point, so on DC2018_128 `t_E` inherited rank 19 from
a `theta_E` that was itself still at the Condition A floor when the relation
fired, and kept 19 after `theta_E` reached 80.  `probe_derivable` tests
`> PRECEDENCE_DEFAULT` (20), so a complete restart file read as "nobody told
us" and an expensive fit re-ran and died.  ob161003 escaped it only because
the same relation happened to fire after `theta_E` was promoted -- order, not
physics.  Fixed by `ConfigManager._propagate_provenance_to_fixed_point`.

THE VEHICLE IS A REAL RESTART FILE, and that matters: the first version of
this test used "the shipped params file minus its literal `t_E`", which is
NOT what mkparam produces.  Dropping `t_E` without adding the sampled
parameters a restart file carries leaves the mass/distance/proper-motion
chain unpinned, so `t_E` genuinely cannot be derived and the probe is right
to re-run.  That vehicle stayed red after the real defect was fixed, which is
how it was caught.  So build the system, fabricate a trace over its real
sampled variables, and let mkparam write the file -- the same path a
second-iteration fit takes.

BOTH TOPOLOGIES, because the bug was topology-dependent in a way one example
could not show: DC2018_128 is 2L1S with a PLANET companion and failed;
ob161003 is 2S2L with a star companion and passed. Testing only the one that
passed is what let this ship.
"""

import copy
import logging
import os
import pathlib
import shutil

import pytest
import yaml

# Reuse the round-trip file's builder and trace fabrication rather than
# keeping a second copy: a divergence between them would silently change
# what "a restart file" means in one of the two suites.
from test_mkparam_roundtrip import _build, _fabricate_trace

from exozippy.mkparam import write_param_file
from exozippy.system import System

EXAMPLES_DIR = pathlib.Path(__file__).parent / ".." / "examples"

pytestmark = pytest.mark.slow

CASES = [
    ("DC2018_128", "DC2018_128.yaml"),
    ("ob161003", "ob161003.yaml"),
]


class _Triggered(Exception):
    """Raised the instant the probe decides to fit, so no fit is run."""


class _Watch(logging.Handler):
    def emit(self, record):
        if "no sufficient user start values" in record.getMessage():
            raise _Triggered()


def _launches_mmexofast(work, config_name, params):
    """True if building `config_name` in `work` with `params` starts a fit.

    Aborts at the decision point rather than letting MMEXOFAST run -- a test
    that actually ran it would take minutes and, on DC2018_128, crash.
    """
    with open(os.path.join(work, config_name)) as fh:
        config = yaml.safe_load(fh)
    # An in-memory params dict, so the file on disk is untouched.
    config["parameter_file"] = None

    handler = _Watch()
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    cwd = os.getcwd()
    try:
        os.chdir(work)
        System(config, copy.deepcopy(params)).prepare()
        return False
    except _Triggered:
        return True
    finally:
        os.chdir(cwd)
        root.removeHandler(handler)
        root.setLevel(previous)


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
    """The control: as shipped, no example re-runs MMEXOFAST.

    Without it the test below would also pass if the probe had simply stopped
    triggering altogether, which is a different bug.
    """
    assert not _launches_mmexofast(
        restart["work"], restart["config_name"], restart["shipped"]
    ), (
        f"{restart['name']} launches MMEXOFAST from its own shipped params "
        f"file, which names every required observable outright"
    )


def test_a_restart_file_does_not_re_run_mmexofast(restart):
    """The contract: a restart file's DERIVED t_E must count as sufficient.

    mkparam writes only sampled coordinates, so a restart file never names
    t_E.  If deriving it does not count, the documented
    fit-then-restart-from-the-MAP workflow re-runs MMEXOFAST every second
    iteration -- and on DC2018_128 dies inside it.
    """
    assert not _launches_mmexofast(
        restart["work"], restart["config_name"], restart["restart"]
    ), (
        f"{restart['name']}'s own mkparam restart file launched MMEXOFAST: "
        f"the derived t_E is not being credited to the sampled parameters "
        f"that determine it (review 8.6.22)"
    )


def test_the_restart_file_does_not_name_t_E(restart):
    """The premise of the test above, pinned so it cannot rot silently.

    If mkparam ever started writing a derived t_E, the contract test would
    pass for an uninteresting reason -- the literal short-circuit in
    `user_hints_sufficient` -- while the derivability path it exists to guard
    went unexercised.  JDE's ruling is that mkparam writes every SAMPLED
    parameter and nothing derived, so this is also that ruling's pin.
    """
    named = [k for k in restart["restart"] if k.endswith(".t_E")]
    assert not named, (
        f"{restart['name']}'s restart file names {named}; t_E is derived, so "
        f"writing it would over-determine the next fit and would hide "
        f"whether the derivability path still works"
    )
