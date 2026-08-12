"""End-to-end: run_fit driven by an IN-MEMORY user_params dict must produce a
restart file built from those params, not from whatever file happens to sit at
``config['parameter_file']``.

``run_fit(config, user_params=<dict>)`` is a documented entry point (no YAML
files needed), but mkprior used to re-read the parameter_file from disk to get
the constraints it merges the trace MAP into.  A dict-driven run therefore
emitted a restart file carrying priors and bounds that were never part of the
fit -- and silently, because the two inputs here differ only in the magnitudes
of mu/sigma, which ``evaluator.structural_hash`` deliberately does not cover,
so the trace-freshness check cannot see the substitution either.

Marked 'slow': it runs the whole pipeline once (kelt4 RV-only, 1 draw).
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from exozippy.run import run_fit

pytestmark = pytest.mark.slow

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"

# The value the fit actually ran with, and the decoy that sits on disk at
# config['parameter_file'].  Same parameter, same structure -- only the prior
# numbers differ, which is precisely the case nothing else in the pipeline
# catches.
FITTED_TEFF = {"initval": 6207.0, "mu": 6207.0, "sigma": 100.0}
STALE_TEFF = {"initval": 6207.0, "mu": 4000.0, "sigma": 900.0}


@pytest.fixture(scope="module")
def in_memory_result(tmp_path_factory):
    """Run the kelt4 RV-only example once through run_fit with an in-memory
    user_params dict, leaving a decoy params file on disk."""
    work_dir = tmp_path_factory.mktemp("mkprior_mem_work") / "kelt4"
    out_dir = tmp_path_factory.mktemp("mkprior_mem_out")

    shutil.copytree(
        EXAMPLE_DIR,
        work_dir,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("kelt4_rvonly.yaml") as f:
            config = yaml.safe_load(f)

        user_params = yaml.safe_load(open(config["parameter_file"]))
        user_params["star.A.teff"] = dict(FITTED_TEFF)
        user_params.pop("star.0.teff", None)

        # The decoy: same file the config still points at, different priors.
        stale = dict(user_params)
        stale["star.A.teff"] = dict(STALE_TEFF)
        with open(config["parameter_file"], "w") as f:
            yaml.dump(stale, f)

        config["prefix"] = str(out_dir / "KELT-4A")
        config["sampler"] = {
            "method": "nuts",
            "tune": 2,
            "draws": 1,
            "chains": 1,
            "cores": 1,
            "measure_scales": False,
            "recompute_trace": True,
        }

        run_fit(config, user_params=user_params)
    finally:
        os.chdir(orig_cwd)

    return out_dir, work_dir


def test_restart_file_is_written_for_an_in_memory_run(in_memory_result):
    """
    Given run_fit called with an in-memory user_params dict,
    When the run completes,
    Then mkprior still writes the next versioned restart file.
    """
    _, work_dir = in_memory_result
    assert (work_dir / "kelt4.params.2.yaml").exists(), (
        "mkprior wrote nothing; yaml files present: "
        f"{[f.name for f in work_dir.glob('*.yaml')]}"
    )


def test_restart_file_carries_the_in_memory_priors(in_memory_result):
    """
    Given a decoy params file on disk whose priors differ from the in-memory
    dict the fit ran with,
    When run_fit completes and mkprior writes the restart file,
    Then the restart file carries the FITTED prior, not the decoy's.
    """
    _, work_dir = in_memory_result
    result = yaml.safe_load(open(work_dir / "kelt4.params.2.yaml"))

    entry = result["star.A.teff"]
    assert entry["mu"] == pytest.approx(FITTED_TEFF["mu"])
    assert entry["sigma"] == pytest.approx(FITTED_TEFF["sigma"])
    assert entry["mu"] != pytest.approx(STALE_TEFF["mu"])
    assert entry["sigma"] != pytest.approx(STALE_TEFF["sigma"])
