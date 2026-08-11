"""Guard the goodness of fit AT THE SEED for every microlensing example.

The seed (``System.get_raw_start``) is where a production fit starts, so its
chi2 is the number
that says whether initialization is doing its job -- reproducing the solution
the observables came from.  Nothing checked it before, and that is exactly how a
change to the proper-motion start values passed 1272 tests while making two
examples' starts 60x and 2x worse:

    ob140939  chi2/N  3.04 -> 179.1
    ob161003  chi2/N  1.72 ->   3.9

A whole-suite pass means the model still *builds* and the physics still agrees
with its references.  It says nothing about starting somewhere sensible.

The bounds below are deliberately loose (a generous ceiling per example, not the
measured value to 4 decimals): they are here to catch a regression of that
magnitude, not to freeze the seeding.  Tighten only with a reason.

Note chi2/N > 1 at the seed is expected and is not a defect: MMEXOFAST fits with
pi_E = 0, so its seeded t_E/u_0/t_0 describe a no-parallax model while this one
applies a derived pi_E, and the published configs leave the lens mass, distance
and proper motion open for the engine to fill.
"""

import glob
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

EXAMPLES_DIR = pathlib.Path(__file__).parent / ".." / "examples"

pytestmark = pytest.mark.slow

# (example, ceiling on chi2/N at the seed).  Measured values as of 2026-08-11:
#   DC2018_128 1.213, ob08092 1.415, ob140939 3.039, ob161003 1.717
CEILINGS = [
    ("DC2018_128", 2.0),
    ("ob08092", 2.5),
    ("ob140939", 5.0),
    ("ob161003", 3.0),
]


def _seed_chi2(name):
    """chi2 and chi2/N of the mulens observation at the raw start point."""
    src = EXAMPLES_DIR / name
    cfg_path = [
        p
        for p in glob.glob(str(src / "*.yaml"))
        if "params" not in os.path.basename(p)
        and "hpc" not in os.path.basename(p)
    ][0]
    work = pathlib.Path(tempfile.mkdtemp()) / name
    shutil.copytree(
        src, work, ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#")
    )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(os.path.basename(cfg_path)) as f:
            config = yaml.safe_load(f)
        param_file = config.get("parameter_file")
        user_params = {}
        if param_file and os.path.exists(param_file):
            with open(param_file) as f:
                user_params = yaml.safe_load(f) or {}

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()

        # The REAL start, not an assumed all-zeros vector: get_raw_start is
        # 0 for logit-transformed elements but (initval - mu)/sigma for
        # Gaussian-path ones, which is nonzero whenever a prior mean differs
        # from the start value.
        point = system.get_raw_start(model)

        obs = [v for v in model.observed_RVs if "mulens" in v.name]
        assert obs, f"{name} has no mulens observation"
        node = obs[0]
        ins = node.owner.inputs
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([ins[-2], ins[-1]]),
            on_unused_input="ignore",
        )
        mu, sigma = [
            np.asarray(a, dtype=float).ravel()
            for a in fn(*[point[v.name] for v in model.value_vars])
        ]
        data = np.asarray(node.tag.observations.eval(), dtype=float).ravel()
        chi2 = float(np.sum(((data - mu) / sigma) ** 2))
        return chi2, chi2 / data.size, data.size
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("name,ceiling", CEILINGS)
def test_seed_is_a_sensible_starting_point(name, ceiling):
    """
    Given a microlensing example,
    When its model is evaluated at its raw start point,
    Then the light curve already fits to better than `ceiling` chi2 per point.

    This is the check that would have caught the proper-motion seeding
    regression; see the module docstring.
    """
    # Arrange / Act
    chi2, reduced, n = _seed_chi2(name)

    # Assert
    assert np.isfinite(chi2), f"{name}: chi2 at the seed is not finite"
    assert reduced < ceiling, (
        f"{name}: chi2/N at the seed is {reduced:.3f} (chi2={chi2:.1f}, "
        f"N={n}), above the {ceiling} ceiling.  The start values no longer "
        f"reproduce the solution the observables came from -- check what the "
        f"relaxation engine resolved for the lens mass, distance and proper "
        f"motion."
    )
