"""The microlensing flux bootstrap reads the geometry the model STARTS at.

Review 8.6.23 / #455 regression: with fitpirel/fitthetae/fitmurel and no
pi_E entry, the stage-1 probe solved pi_E on the defaults.yaml distances and
masses (the event's kinematic hints arrived only at stage 3), and the
geocentric pi_E relation turned that into a parallax ~150x the model's start
(DC2018-069: (-1.46, -4.10) against ~(0.03, 0.02)).  The bootstrap fit its
flux split to point-lens columns carrying it, and W149's q_source started at
0.48 instead of 0.32.  Now the kinematic hints are pushed before the probe,
and MulensInstrument.reread_after_starts re-runs the bootstrap at the solved
start if a stage-3 hint still moved it; System.prepare raises if the extra
pass did not settle it.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


def _kmt_inputs():
    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    work = Path(tempfile.mkdtemp(prefix="kmt_bootgeom_")) / "KMT"
    shutil.copytree(_KMT_DIR, work)
    with open(work / "KMT-2019-BLG-1806.yaml") as f:
        config = yaml.safe_load(f)
    with open(work / config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    return work, config, user_params


@pytest.mark.slow
def test_bootstrap_pi_E_is_the_built_start_pi_E():
    """fitpirel + fitthetae + fitmurel, no pi_E entry: the pi_E the flux
    bootstrap used is the built model's start pi_E."""
    work, config, user_params = _kmt_inputs()
    ev = config["mulensevent"][0]
    ev.update(fitpirel=True, fitthetae=True, fitmurel=True)
    for k in ("mulensevent.pi_E_N", "mulensevent.pi_E_E"):
        user_params.pop(k)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)

    geometry = system.mulensinstrument._bootstrap_geometry
    used = [
        geometry("mulensevent.0.pi_E_N"),
        geometry("mulensevent.0.pi_E_E"),
    ]
    assert None not in used, "the bootstrap saw no informed pi_E"

    start = system.get_raw_start(model)
    nodes = model.replace_rvs_by_values(
        [model["mulensevent.pi_E_N"], model["mulensevent.pi_E_E"]]
    )
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    built = [
        float(np.ravel(v)[0])
        for v in fn(*[np.asarray(start[v.name]) for v in model.value_vars])
    ]
    # The bootstrap reads the engine's solved start; the model evaluates the
    # derived pi_E from the sampled leaves, so agreement is to the leaves'
    # round trip, not to the bit.
    np.testing.assert_allclose(used, built, rtol=1e-6)


@pytest.mark.slow
def test_unsettled_reread_raises(monkeypatch):
    """A component whose stage-1 estimate still disagrees with the solved
    start after the extra stage-3/4 pass makes prepare() RAISE, naming it,
    instead of iterating or keeping a start the estimate does not describe."""
    from exozippy.components.mulensing.mulensinstrument import (
        MulensInstrument,
    )

    calls = []

    def always_moved(self, system):
        calls.append(1)
        return [(f"[{self.prefix}] still moved", [])]

    monkeypatch.setattr(MulensInstrument, "reread_after_starts", always_moved)
    work, config, user_params = _kmt_inputs()
    cwd = os.getcwd()
    os.chdir(work)
    try:
        system = System(config, user_params=user_params)
        with pytest.raises(RuntimeError, match="still moved"):
            system.prepare()
    finally:
        os.chdir(cwd)
    assert len(calls) == 2
