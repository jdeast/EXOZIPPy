# tests/test_multiseed_mmexofast.py
"""Tests for the MMEXOFAST solutions-file loader (P4 layer b).

MMEXOFAST emits multiple lightly-optimized solutions spanning the standard
microlensing degeneracies (examples/DC2018_128/mmexofast.json).
mmexofast_support.load_json reads them and push_seed_hints pushes each fit's
observable-space values as a per-seed hint set feeding the layer-(a)
multi-seed relaxation engine -- once, from MulensInstrument._resolve_mmexofast
at stage 1 (MulensEvent's stage-3 re-push was deleted, reviews 1.6.15 and
2.1.12).  Post-split target paths: the trajectory offsets land on source.0,
the event chain on mulensevent.0, and the companion geometry on lens.1
(element 0 is the masked primary).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.components.mulensing import mmexofast_support

MMX_PATH = (
    Path(__file__).parent.parent / "examples" / "DC2018_128" / "mmexofast.json"
)


class _RecordingConfigManager:
    """Minimal config_manager stub: records add_seed_hints/add_scale_hint calls
    without touching the real relaxation engine (unit-tests the loader alone).
    """

    def __init__(self, system_config=None, user_params=None):
        self.system_config = system_config or {}
        self.user_params = user_params or {}
        self.seed_hint_sets = []
        self.scale_hints = {}

    def add_hint(self, *args, **kwargs):
        pass

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale

    def add_seed_hints(self, seed_dicts, *, source, replace=False):
        # Mirrors ConfigManager.add_seed_hints: one seeder per fit (review
        # 2.1.25) -- a second registration without replace=True raises.
        if self.seed_hint_sets and not replace:
            raise ValueError(f"second seed-set registration by {source!r}")
        self.seed_hint_sets = list(seed_dicts)


def _push(path, cfg_manager):
    """The explicit-file path of MulensInstrument._resolve_mmexofast, for a
    binary-lens finite-source event."""
    data = mmexofast_support.load_json(str(path))
    if data is None:
        return 0
    return mmexofast_support.push_seed_hints(
        data, cfg_manager, want_rho=True, is_binary=True, source=str(path)
    )


@pytest.mark.skipif(
    not MMX_PATH.exists(), reason="DC2018_128 fixture not present"
)
def test_mmexofast_loader_pushes_two_seeds_matching_json():
    """
    Given examples/DC2018_128/mmexofast.json (2 fits),
    When its seeds are pushed,
    Then 2 seed hint sets arrive whose t_0/u_0/t_E/rho/log_s/alpha/q match
    the json values (log_s = log10(s), alpha via the identity convention).
    """
    with open(MMX_PATH) as f:
        raw = json.load(f)

    cfg_manager = _RecordingConfigManager()
    assert _push(MMX_PATH, cfg_manager) == 2

    assert len(cfg_manager.seed_hint_sets) == 2
    for i, fit in enumerate(raw["fits"]):
        p = fit["parameters"]
        seed = cfg_manager.seed_hint_sets[i]
        assert np.isclose(seed["source.0.t_0"], p["t_0"])
        assert np.isclose(seed["source.0.u_0"], p["u_0"])
        assert np.isclose(seed["mulensevent.0.t_E"], p["t_E"])
        assert np.isclose(seed["source.0.rho"], p["rho"])
        assert np.isclose(seed["lens.1.q"], p["q"])
        # s is sampled as log_s (P2); the loader must push log10(s), not s.
        assert np.isclose(seed["lens.1.log_s"], np.log10(p["s"]))
        # Alpha convention: verified identity mapping (see
        # mmexofast_support.push_seed_hints and the convention test below).
        assert np.isclose(seed["lens.1.alpha"], p["alpha"])


def test_mmexofast_loader_missing_file_warns_and_noops(caplog):
    """
    Given a nonexistent mmexofast file,
    When it is loaded,
    Then a warning is logged and no seed is pushed.
    """
    cfg_manager = _RecordingConfigManager()
    with caplog.at_level("WARNING"):
        assert _push("/no/such/file.json", cfg_manager) == 0

    assert cfg_manager.seed_hint_sets == []
    assert any("mmexofast" in rec.message.lower() for rec in caplog.records)


def test_mmexofast_loader_corrupt_file_raises_rather_than_seeding_nothing(
    tmp_path,
):
    """
    Given an mmexofast file that exists but is truncated (a job killed
    mid-write),
    When it is loaded,
    Then CorruptMMEXOFASTFileError is raised and no seed is pushed, instead
    of warning once and letting the fit start from defaults.yaml. A
    user-named file is not exozippy's to regenerate -- only run_or_load's
    own cache is (tests/test_mmexofast_support.py covers that half).
    """
    from exozippy.components.mulensing.mmexofast_support import (
        CorruptMMEXOFASTFileError,
    )

    good = {
        "fits": [
            {
                "parameters": {
                    "t_0": 2458554.9,
                    "u_0": 0.14,
                    "t_E": 18.2,
                    "rho": 1e-3,
                    "s": 0.98,
                    "q": 1.1e-3,
                    "alpha": -52.0,
                },
                "sigmas": {},
            }
        ]
    }
    bad = tmp_path / "mmexofast.json"
    full = json.dumps(good, indent=4)
    bad.write_text(full[: len(full) // 2])

    cfg_manager = _RecordingConfigManager()
    with pytest.raises(CorruptMMEXOFASTFileError) as exc:
        _push(bad, cfg_manager)

    assert "mmexofast.json" in str(exc.value)
    assert cfg_manager.seed_hint_sets == []


# ---------------------------------------------------------------------------
# End to end through prepare(): ONE push per explicit file (1.6.15 / 2.1.12)
# ---------------------------------------------------------------------------
def _prepare_dc2018_128(tmp_path, **event_keys):
    """prepare() the shipped DC2018_128 example with an explicit
    `mmexofast:` file, in a scratch copy (prepare writes under the prefix),
    with no params file so every seed comes from the seeders."""
    from exozippy.system import System
    from exozippy.yamlio import load_system_config

    src = MMX_PATH.parent
    for name in ("DC2018_128.yaml", "mmexofast.json"):
        shutil.copy(src / name, tmp_path / name)
    data = "n20180816.Z087.WFIRST18.128.txt"
    (tmp_path / data).symlink_to(src / data)
    config = load_system_config(str(tmp_path / "DC2018_128.yaml"))
    config.pop("parameter_file", None)
    config["prefix"] = str(tmp_path / "fitresults" / "DC2018_128")
    config["mulensinstrument"][0]["file"] = str(tmp_path / data)
    config["mulensevent"][0].update(
        {"mmexofast": str(tmp_path / "mmexofast.json"), **event_keys}
    )
    system = System(config, user_params={})
    system.prepare()
    return system.config_manager


@pytest.mark.skipif(
    not MMX_PATH.exists(), reason="DC2018_128 fixture not present"
)
def test_explicit_file_is_pushed_exactly_once_through_prepare(tmp_path):
    """
    Given an explicit `mmexofast:` file with 2 fits and the default
    (absent) `peak_find:`,
    When prepare() runs,
    Then there are exactly 2 seed sets: the file's, once.  With the stage-3
    re-push still in place the second registration would raise (review
    2.1.25), and the default peak finder, one seeder per fit, stays out.
    """
    cm = _prepare_dc2018_128(tmp_path)
    with open(MMX_PATH) as f:
        raw = json.load(f)
    assert len(cm.seed_hint_sets) == len(raw["fits"]) == 2
    assert all("lens.1.q" in s for s in cm.seed_hint_sets)


@pytest.mark.skipif(
    not MMX_PATH.exists(), reason="DC2018_128 fixture not present"
)
def test_peak_find_true_replaces_an_explicit_file_through_prepare(tmp_path):
    """
    Given an explicit `mmexofast:` file AND `peak_find: true` (the A/B mode),
    When prepare() runs,
    Then the peak finder's ONE point-lens set is all that is left (review
    1.6.15: the stage-3 re-push used to put MMEXOFAST's two sets back after
    the finder had replaced them, so the A/B compared MMEXOFAST with itself).
    """
    cm = _prepare_dc2018_128(tmp_path, peak_find=True)
    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {
        "source.0.t_0",
        "source.0.u_0",
        "mulensevent.0.t_E",
    }


@pytest.mark.skipif(
    not MMX_PATH.exists(), reason="DC2018_128 fixture not present"
)
def test_mmexofast_alpha_convention_is_identity_not_180_minus():
    """
    Given the shipped examples/DC2018_128/DC2018_128.params.yaml (seeded from
    mmexofast.json by scripts/mmexofast_to_params.py, whose
    lens.Companion.alpha initval is list-valued -- one entry per MMEXOFAST
    solution, in file order; the key names the COMPANION lens body (lens
    element 1) after the mulensevent/lens/source split, where the pre-split
    file named the single lens instance, lens.Lens.alpha -- and which
    examples/DC2018_128/compare_results.py compares
    directly against MMEXOFAST/DC18 truth with no remapping),
    Then that params.yaml's seed-0 alpha initval equals the raw MMEXOFAST
    fit-0 alpha value -- confirming the IDENTITY convention (not the
    alpha_MM = 180 - alpha_paper relation recorded for a different event,
    ob161003, in project memory).
    """
    with open(MMX_PATH) as f:
        raw = json.load(f)
    fit0_alpha = raw["fits"][0]["parameters"]["alpha"]

    params_path = MMX_PATH.parent / "DC2018_128.params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    alpha_entry = params["lens.Companion.alpha"]
    initval = (
        alpha_entry["initval"]
        if isinstance(alpha_entry, dict)
        else alpha_entry
    )
    seed0_alpha = initval[0] if isinstance(initval, list) else initval

    # The params file pins seed-0 alpha to the MMEXOFAST fit-0 value (to the
    # precision scripts/mmexofast_to_params.py writes, 8 decimal places).
    assert seed0_alpha == pytest.approx(fit0_alpha, abs=1e-6)
