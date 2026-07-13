# tests/test_multiseed_mmexofast.py
"""Tests for the MMEXOFAST solutions-file loader (P4 layer b).

MMEXOFAST emits multiple lightly-optimized solutions spanning the standard
microlensing degeneracies (examples/DC2018_128/mmexofast.json). Lens.
_load_mmexofast_seeds reads them and pushes each fit's observable-space
values as a per-seed hint set feeding the layer-(a) multi-seed relaxation
engine.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.components.mulensing.lens import Lens

MMX_PATH = Path(__file__).parent.parent / "examples" / "DC2018_128" / "mmexofast.json"


class _RecordingConfigManager:
    """Minimal config_manager stub: records add_seed_hints/add_scale_hint calls
    without touching the real relaxation engine (unit-tests the loader alone)."""

    def __init__(self, system_config=None, user_params=None):
        self.system_config = system_config or {}
        self.user_params = user_params or {}
        self.seed_hint_sets = []
        self.seed_hint_rank = None
        self.scale_hints = {}

    def add_hint(self, *args, **kwargs):
        pass

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale

    def add_seed_hints(self, seed_dicts, rank=None):
        self.seed_hint_sets = seed_dicts
        if rank is not None:
            self.seed_hint_rank = rank


def _make_binary_lens(mmexofast_path, finite_source=True):
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
    }
    cfg_manager = _RecordingConfigManager(system_config=system_config)
    lens_config = [{
        "name": "Lens",
        "lenses": ["star.0", "planet.0"],
        "sources": ["star.1"],
        "finite_source": finite_source,
        "mmexofast": str(mmexofast_path),
    }]
    lens = Lens(lens_config, cfg_manager)
    return lens, cfg_manager


@pytest.mark.skipif(not MMX_PATH.exists(), reason="DC2018_128 fixture not present")
def test_mmexofast_loader_pushes_two_seeds_matching_json():
    """
    Given examples/DC2018_128/mmexofast.json (2 fits),
    When Lens._load_mmexofast_seeds runs,
    Then it pushes 2 seed hint sets whose t_0/u_0/t_E/rho/log_s/alpha/q match
    the json values (log_s = log10(s), alpha via the identity convention).
    """
    with open(MMX_PATH) as f:
        raw = json.load(f)

    lens, cfg_manager = _make_binary_lens(MMX_PATH)
    lens._load_mmexofast_seeds()

    assert len(cfg_manager.seed_hint_sets) == 2
    for i, fit in enumerate(raw["fits"]):
        p = fit["parameters"]
        seed = cfg_manager.seed_hint_sets[i]
        assert np.isclose(seed["lens.0.t_0"], p["t_0"])
        assert np.isclose(seed["lens.0.u_0"], p["u_0"])
        assert np.isclose(seed["lens.0.t_E"], p["t_E"])
        assert np.isclose(seed["lens.0.rho"], p["rho"])
        assert np.isclose(seed["lens.0.q"], p["q"])
        # s is sampled as log_s (P2); the loader must push log10(s), not s.
        assert np.isclose(seed["lens.0.log_s"], np.log10(p["s"]))
        # Alpha convention: verified identity mapping (see docstring in
        # Lens._load_mmexofast_seeds and the standalone convention test below).
        assert np.isclose(seed["lens.0.alpha"], p["alpha"])


@pytest.mark.skipif(not MMX_PATH.exists(), reason="DC2018_128 fixture not present")
def test_mmexofast_loader_missing_file_warns_and_noops(caplog):
    """
    Given a lens config with a nonexistent mmexofast file,
    When _load_mmexofast_seeds runs,
    Then it logs a warning and leaves seed_hint_sets empty rather than raising.
    """
    lens, cfg_manager = _make_binary_lens("/no/such/file.json")
    with caplog.at_level("WARNING"):
        lens._load_mmexofast_seeds()

    assert cfg_manager.seed_hint_sets == []
    assert any("mmexofast" in rec.message.lower() for rec in caplog.records)


def test_mmexofast_key_absent_is_a_noop():
    """
    Given a lens config with no 'mmexofast' key (the default, opt-in feature),
    When _load_mmexofast_seeds runs,
    Then nothing is pushed -- the ordinary single-start path is untouched.
    """
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
    }
    cfg_manager = _RecordingConfigManager(system_config=system_config)
    lens_config = [{
        "name": "Lens",
        "lenses": ["star.0", "planet.0"],
        "sources": ["star.1"],
    }]
    lens = Lens(lens_config, cfg_manager)
    lens._load_mmexofast_seeds()

    assert cfg_manager.seed_hint_sets == []


@pytest.mark.skipif(not MMX_PATH.exists(), reason="DC2018_128 fixture not present")
def test_mmexofast_alpha_convention_is_identity_not_180_minus():
    """
    Given the shipped examples/DC2018_128/DC2018_128.params.yaml (seeded from
    mmexofast.json by scripts/mmexofast_to_params.py, whose lens.Lens.alpha
    initval is list-valued -- one entry per MMEXOFAST solution, in file
    order -- and which examples/DC2018_128/compare_results.py compares
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

    alpha_entry = params["lens.Lens.alpha"]
    initval = alpha_entry["initval"] if isinstance(alpha_entry, dict) else alpha_entry
    seed0_alpha = initval[0] if isinstance(initval, list) else initval

    # The params file pins seed-0 alpha to the MMEXOFAST fit-0 value (to the
    # precision scripts/mmexofast_to_params.py writes, 8 decimal places).
    assert seed0_alpha == pytest.approx(fit0_alpha, abs=1e-6)
