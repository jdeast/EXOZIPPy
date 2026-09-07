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

# (example, ceiling on chi2/N at the seed).  Re-measured 2026-09-07 after the
# mulensevent split and the multi-source galactic-pm gate's retirement:
#   DC2018_128 1.213, ob08092 1.406, ob140939 3.113, ob161003 1.702
# (2026-08-11, pre-split: 1.213, 1.415, 3.039, 1.717 -- the split moved every
# one of these by less than the ceilings' slack, which is the point.)
CEILINGS = [
    ("DC2018_128", 2.0),
    ("ob08092", 2.5),
    ("ob140939", 5.0),
    ("ob161003", 3.0),
]


def _seed_chi2(name, strip_prefixes=()):
    """chi2 and chi2/N of the mulens observation at the raw start point.

    `strip_prefixes` drops matching keys from the params file before
    building, which is how the multi-source proper-motion seeding is reached
    at all: ob161003 pins all four pm components, and the seeding's earlier
    blocker returns on any pm/parallax entry.
    """
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

        if strip_prefixes:
            for key in list(user_params):
                if key.startswith(tuple(strip_prefixes)):
                    user_params.pop(key)

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
        # The resolved proper motions come back too: chi2 alone cannot say
        # whether the galactic seeding fired or was refused upstream, and
        # "no change" from a gate that was never reached looks identical to
        # "no change" from a gate that did nothing.
        pms = {
            p.label: np.atleast_1d(np.asarray(p.initval, dtype=float))
            for p in system.get_all_parameters()
            if p.label.endswith((".pm_ra", ".pm_dec"))
        }
        return chi2, chi2 / data.size, data.size, pms
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
    chi2, reduced, n, _pms = _seed_chi2(name)

    # Assert
    assert np.isfinite(chi2), f"{name}: chi2 at the seed is not finite"
    assert reduced < ceiling, (
        f"{name}: chi2/N at the seed is {reduced:.3f} (chi2={chi2:.1f}, "
        f"N={n}), above the {ceiling} ceiling.  The start values no longer "
        f"reproduce the solution the observables came from -- check what the "
        f"relaxation engine resolved for the lens mass, distance and proper "
        f"motion."
    )


def test_a_multi_source_event_is_seeded_from_the_galactic_model():
    """The retired multi-source gate: seeding two sources is an improvement.

    The gate refused to seed proper motions for any event with more than one
    source, because pre-split each source carried its own mu_rel vector and
    one prior mean forced them together.  Under R1 there is one mu_rel per
    event, resolved through source body 0, so the same mean is information:
    source body 1's pm feeds only its own distance/rho chain.

    ob161003 cannot show this as shipped -- it pins all four pm components,
    and the seeding's earlier blocker returns on any pm/parallax entry, so
    the multi-source gate was unreachable for the only multi-source example
    and its retirement left every pinned number byte-identical.  Stripping
    those pins reaches it.  Measured 2026-09-07:

        gate present (no pm seeded)   chi2/N 10.91
        gate retired (pm seeded)      chi2/N  6.94

    Both halves are asserted, because chi2 alone cannot distinguish "seeded
    and better" from "refused upstream and unchanged".
    """
    # Arrange / Act -- remove the pm entries and NOTHING else, so the
    # galactic model is the only thing that can supply a proper motion while
    # every other published value stays.  (Stripping `star.Source`/
    # `star.Lens` wholesale would also drop the distances and radii, and the
    # numbers above would not be the numbers being measured.)
    chi2, reduced, n, pms = _seed_chi2(
        "ob161003",
        strip_prefixes=(
            "star.SourceA.pm_",
            "star.SourceB.pm_",
            "star.Lens.pm_",
            "star.LensB.pm_",
        ),
    )

    # Assert: the seeding fired, and both sources took the SAME bulge mean.
    # Element order is star config order: 0=Lens, 1=LensB, 2=SourceA,
    # 3=SourceB.
    for axis in ("pm_ra", "pm_dec"):
        vec = pms["star." + axis]
        assert vec.size == 4, f"expected 4 stars, got {vec.size}"
        assert vec[2] == pytest.approx(vec[3]), (
            f"the two sources got different {axis} seeds ({vec[2]} vs "
            f"{vec[3]}); under R1 one mu_rel per event means one bulge mean "
            f"for both source bodies"
        )
        assert vec[0] != pytest.approx(vec[2]), (
            f"the lens and the sources got the same {axis} seed; the lens is "
            f"seeded from the thin disk at 4 kpc and the sources from the "
            f"bulge at 8 kpc, so equal values mean neither was seeded"
        )

    # Assert: and it is better than refusing to seed at all (10.91).
    assert np.isfinite(chi2)
    assert reduced < 8.0, (
        f"chi2/N at the seed is {reduced:.3f} (chi2={chi2:.1f}, N={n}) with "
        f"no proper motion supplied.  Refusing to seed measured 10.91 and "
        f"seeding measured 6.94, so above 8.0 means the multi-source "
        f"seeding has stopped helping and its retirement needs revisiting."
    )
