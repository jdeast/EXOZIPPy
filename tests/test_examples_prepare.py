"""Every shipped example config must still reach a prepared System.

This is a rot guard, not a science test. The examples are the documentation
of the YAML surface, and nothing else in the suite touches most of them -- so
an API migration can silently leave one behind. That is exactly what happened
to examples/hat3: its `band:` block had an entry with no `name:` and its
transits still used the pre-Band `filter:` key, so `System.prepare()` died
with "Transit instrument(s) reference unknown band(s)" and nothing noticed.
examples/hd80606 had the same disease one rename earlier (`inst.` for
`rvinstrument.` in its params file).

Scope is deliberately `prepare()` only (stages 1-4: file I/O, index maps,
parameter registration, the relaxation engine). That is where a stale key,
a dangling cross-reference or a renamed prefix shows up, and it costs a
couple of seconds per example. Building the PyMC model, let alone sampling,
would cost minutes each and is covered per-feature elsewhere
(tests/test_integration_kelt4.py and friends).

Examples that genuinely cannot run here belong in ``_EXCLUDED`` with a
reason, never dropped silently.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.components.instrument import Instrument
from exozippy.system import System

_EXAMPLES = Path(__file__).parent.parent / "examples"

# Relative path (POSIX, from examples/) -> reason it cannot be prepared here.
# Empty today: every shipped config prepares. Add entries rather than
# narrowing the glob, so an excluded example stays visible.
_EXCLUDED: dict[str, str] = {}


def _system_configs():
    """Every examples/*/ YAML that is a system config, as (path, id) pairs.

    A system config is identified by content, not filename: it carries a
    `run:` or `sampler:` block. Parameter files, `.sed` files and the SED
    photometry YAMLs carry neither, and the naming is not consistent enough
    to key on (examples/hat3 spells its SED file `hat3_sed.yaml`, everyone
    else `<name>.sed.yaml`).
    """
    out = []
    if not _EXAMPLES.is_dir():
        return out
    for path in sorted(_EXAMPLES.glob("*/*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            # A config that will not even parse is a failure, not a
            # non-config: keep it so the test reports it.
            doc = {"sampler": None}
        if not isinstance(doc, dict):
            continue
        if "run" not in doc and "sampler" not in doc:
            continue
        out.append((path, path.relative_to(_EXAMPLES).as_posix()))
    return out


_CONFIGS = _system_configs()


# ~238 s across 25 cases, about 3.6% of the suite, and it pays that on every
# matrix combination. Reviewed 2026-08-25 during the CI runtime work and
# deliberately KEPT -- "expensive but worth it" (JDE). The BREADTH is the
# point: this is the canary that every shipped config still prepares, and
# narrowing it to a sample, or to one Python version, is what would let a
# broken example ship. It is a coverage trade, not a cleanup.
@pytest.mark.parametrize(
    "path,rel", _CONFIGS, ids=[rel for _, rel in _CONFIGS]
)
def test_shipped_example_prepares(path, rel, monkeypatch, caplog):
    """Given a shipped example config and its parameter file, when
    System.prepare() runs from the example's own directory, then it
    completes, every active component has a registered manifest, and no
    top-level YAML key was ignored as unrecognized."""
    # Arrange
    if rel in _EXCLUDED:
        pytest.skip(f"{rel}: {_EXCLUDED[rel]}")

    config = yaml.safe_load(path.read_text())
    monkeypatch.chdir(path.parent)

    param_file = config.get("parameter_file")
    user_params = None
    if param_file is not None:
        pf = Path(param_file)
        assert pf.is_file(), (
            f"{rel} names parameter_file '{param_file}', which does not "
            f"exist relative to {os.getcwd()}"
        )
        user_params = yaml.safe_load(pf.read_text())

    # Act
    with caplog.at_level("WARNING", logger="exozippy"):
        system = System(config, user_params)
        system.prepare()

    # Assert
    assert system.active_components, f"{rel} instantiated no components"
    for key, comp in system.active_components.items():
        assert hasattr(comp, "manifest"), (
            f"{rel}: component '{key}' registered no manifest in stage 3"
        )

    ignored = [
        r.getMessage()
        for r in caplog.records
        if "does not match any registered component" in r.getMessage()
    ]
    assert not ignored, f"{rel} has stale top-level YAML key(s): {ignored}"

    # Every instrument that concatenates its files must hand each element ONE
    # contiguous row range, in config order, in every concatenated array at
    # once: Instrument._build_block_detrend lays the detrend blocks on the
    # diagonal by walking the per-file counts in order, and mulensing's
    # observer_pos is addressed row-for-row against `time`. Both break
    # silently otherwise, so the shared accumulator's published row_ranges are
    # checked against inst_map on every shipped config -- for free, since the
    # systems are already prepared here.
    for key, comp in system.active_components.items():
        if not isinstance(comp, Instrument):
            continue
        if not getattr(comp, "row_ranges", None):
            continue  # per-file datasets (astrometryinstrument), not rows
        assert len(comp.row_ranges) == comp.n_elements
        assert comp.row_ranges[0][0] == 0, f"{rel}: {key} row_ranges"
        assert comp.row_ranges[-1][1] == comp.n_total_obs, (
            f"{rel}: {key} row_ranges do not cover n_total_obs"
        )
        for i, (lo, hi) in enumerate(comp.row_ranges):
            if i:
                assert lo == comp.row_ranges[i - 1][1], (
                    f"{rel}: {key} element {i} is not contiguous with {i - 1}"
                )
            assert np.array_equal(
                np.flatnonzero(comp.inst_map == i), np.arange(lo, hi)
            ), f"{rel}: {key} element {i} rows disagree with inst_map"
        for name in ("time", "err", "observer_pos", "detrend_matrix"):
            arr = getattr(comp, name, None)
            if arr is not None:
                assert len(arr) == comp.n_total_obs, (
                    f"{rel}: {key}.{name} is not row-aligned with time"
                )


def test_every_example_directory_has_a_config():
    """Given the examples/ tree, when its directories are enumerated, then
    each one either ships a system config this test collects or is listed
    as a data/workflow-only directory -- so a new example cannot be added
    in a way that this guard silently skips."""
    # Arrange: directories that hold data or a driver workflow, not a
    # config exozippy can be pointed at directly.
    data_only = {
        # Driver scripts only: one config per Roman Data Challenge event is
        # generated by the job pipeline, not committed (see the DC2018
        # README, examples/DC2018/README.md).
        "DC2018",
        # Inject-and-recover validation harness (7.13.1): a driver script
        # plus job files and result rows; configs are generated in tmp
        # workdirs per realization, never committed.
        "validation",
        # README only -- no data and no config committed yet.
        "KMT-2021-BLG-1122L",
        "ob161045",
        # In-progress event: data and outputs only, no config committed
        # yet (untracked local work as of 2026-08-13).
        "kb180087_obj3",
        # Same: raw params.dat/phot.dat only, no config committed yet
        # (untracked local work as of 2026-08-27).  This guard runs against
        # the WORKING TREE -- which is why an uncommitted in-progress event
        # directory trips it -- so the entry is here to keep the pre-push
        # hook usable while that example is being built up.  Drop it once
        # ob170114 ships a config.
        "ob170114",
    }
    covered = {rel.split("/")[0] for _, rel in _CONFIGS}

    # Act: dot/underscore directories are caches and local scratch, not
    # examples.
    dirs = {
        p.name
        for p in _EXAMPLES.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    }

    # Assert
    uncovered = dirs - covered - data_only
    assert not uncovered, (
        f"examples/ directories with no collected system config: "
        f"{sorted(uncovered)}. Add a config, or list the directory in "
        f"this test's data_only set with a reason."
    )
