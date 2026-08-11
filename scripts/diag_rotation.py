"""Is the DC2018_128 light curve blind to the DIRECTION of mu_rel?

Companion to diag_mulens.py, for the underdetermined-seeding issue (#93).

The relaxation engine seeds microlensing start values by inverting an
underdetermined system: the MMEXOFAST seed carries no pi_E, so splitting the
known ``mu_rel_mag`` into ``(mu_ra_rel, mu_dec_rel)`` is one equation in two
unknowns.  The engine picks a point on that circle.  The natural objection is
that a genuinely degenerate family should give the SAME light curve, which
would make any choice safe.

This script tests that.  It builds the model twice with the lens proper motion
rotated 90 degrees -- holding ``|mu_rel|`` fixed, so both builds sit in the same
family -- and evaluates both at the SEED (raw = 0), so no pinned raw offsets
confound the comparison.

    poetry run python scripts/diag_rotation.py

Measured on DC2018_128: ``|mu_rel|`` identical to seven digits, but chi2 differs
by ~77 over 870 points, because (a) ``mu_rel_geo = mu_rel_helio -`` an Earth
velocity term is a vector subtraction and so is not rotationally symmetric,
moving ``t_E``; and (b) ``pi_E`` is parallel to ``mu_rel``, so the rotation
rotates the parallax vector, which annual parallax makes observable.

Caveat when reading the absolute chi2: MMEXOFAST fits with pi_E = 0, so its
seeded t_E/u_0/t_0 describe a no-parallax model, while this model applies a
derived pi_E of order 0.2.  The seed is therefore not expected to reproduce
MMEXOFAST's chi2, and chi2/N > 1 here is not evidence of a bug on its own.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytensor
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "DC2018_128"
TEST_FILE = REPO_ROOT / "tests" / "test_runaway_logp_regression.py"

from exozippy.system import System  # noqa: E402

# The defaults.yaml proper motion, and the offset the engine derives on top of
# it.  mu_rel = lens_pm - source_pm, so pinning all four components explicitly
# is the only way to guarantee the two builds differ by a pure rotation: pinning
# just one lets the engine re-derive the other and |mu_rel| drifts (measured:
# 11.411 vs 11.616, a 1.8% contamination of the comparison).
DEFAULT = -3.0
OFFSET = -11.4113305      # == the engine's -14.4113305 hint minus DEFAULT


def _pin(mu_ra, mu_dec):
    """Pin both stars' proper motions so mu_rel is exactly (mu_ra, mu_dec)."""
    return {
        "star.0.pm_ra": {"initval": DEFAULT + mu_ra},
        "star.0.pm_dec": {"initval": DEFAULT + mu_dec},
        "star.1.pm_ra": {"initval": DEFAULT},
        "star.1.pm_dec": {"initval": DEFAULT},
    }

REPORT = (
    "lens.mu_ra_rel", "lens.mu_dec_rel", "lens.mu_rel_mag",
    "lens.mu_rel_geo_mag", "lens.t_E", "lens.theta_E",
    "lens.pi_E_N", "lens.pi_E_E",
)


def _good_raw():
    spec = importlib.util.spec_from_file_location(
        "_runaway_logp_regression", TEST_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GOOD_RAW


def build_and_measure(label, overrides, good_raw):
    work = Path(tempfile.mkdtemp()) / "DC2018_128"
    shutil.copytree(
        EXAMPLE_DIR, work,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open("DC2018_128.yaml") as f:
            config = yaml.safe_load(f)
        with open("DC2018_128.params.yaml") as f:
            user_params = yaml.safe_load(f)
        for entry in config.get("planet", []):
            entry.setdefault("mass_parameterization", "linear")
        user_params.update(overrides)

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()

        # Evaluate at the SEED: raw = 0 everywhere.
        point = {
            k: np.zeros_like(np.asarray(v, dtype=float))
            for k, v in good_raw.items()
        }
        total = float(np.asarray(model.compile_logp()(point)))

        params = {p.label: p for p in system.get_all_parameters()}
        labels, nodes = [], []
        for name in REPORT:
            node = getattr(params.get(name), "value", None)
            if isinstance(node, pytensor.graph.basic.Variable):
                labels.append(name)
                nodes.append(node)
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values(nodes),
            on_unused_input="ignore",
        )
        raw_args = [point[vv.name] for vv in model.value_vars]
        vals = {
            k: float(np.asarray(v, dtype=float).ravel()[0])
            for k, v in zip(labels, fn(*raw_args))
        }

        obs = [v for v in model.observed_RVs if "mulens" in v.name][0]
        ins = obs.owner.inputs
        f2 = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([ins[-2], ins[-1]]),
            on_unused_input="ignore",
        )
        mu_v, sig_v = [
            np.asarray(a, dtype=float).ravel() for a in f2(*raw_args)
        ]
        data = np.asarray(obs.tag.observations.eval(), dtype=float).ravel()
        chi2 = float(np.sum(((data - mu_v) / sig_v) ** 2))

        print(f"\n--- {label} ---")
        for k in labels:
            print(f"    {k:<24s} {vals[k]:+.6f}")
        mag = float(np.hypot(vals["lens.mu_ra_rel"], vals["lens.mu_dec_rel"]))
        print(f"    |mu_rel| from components  {mag:.6f}")
        print(f"    chi2 = {chi2:.2f}   chi2/N = {chi2 / data.size:.3f}")
        print(f"    total logp = {total:.4f}")
        return mag, chi2


    finally:
        os.chdir(cwd)


def main():
    good_raw = _good_raw()
    a_mag, a_chi2 = build_and_measure(
        "A: mu_rel entirely in RA", _pin(OFFSET, 0.0), good_raw
    )
    b_mag, b_chi2 = build_and_measure(
        "B: mu_rel entirely in Dec (rotated 90 deg)",
        _pin(0.0, OFFSET),
        good_raw,
    )

    print("\n=== VERDICT ===")
    print(f"  |mu_rel|:  A={a_mag:.6f}   B={b_mag:.6f}")
    if abs(a_mag - b_mag) > 1e-6 * max(a_mag, b_mag):
        print("  -> ABORT: the two builds are not the same family, so the chi2")
        print("     comparison below would be contaminated.  Fix the pinning.")
        return
    print("     (equal, so both builds sit in the same family)")
    print(f"  chi2    :  A={a_chi2:.2f}   B={b_chi2:.2f}   "
          f"delta={abs(a_chi2 - b_chi2):.2f}")
    if abs(a_chi2 - b_chi2) > 1.0:
        print("  -> the light curve DISTINGUISHES members of the family;")
        print("     an arbitrary choice is not safe.")
    else:
        print("  -> the light curve is blind to the direction.")


if __name__ == "__main__":
    main()
