"""
Tests for AstrometryInstrument.plot_data (GUI PlotSpec pathway) and its
plot() conversion to the generic PlotSpec renderer.

Pins, following tests/test_plot_data.py:
  * data-only mode (point=None) works after prepare() WITHOUT
    build_model() and returns data traces only;
  * every spec is JSON-serializable (json.dumps(spec.to_json()));
  * with a start point, each dataset's spec carries the expected model
    trace (gaia: scatter at the data times; abs: photocenter-orbit line;
    rel: model line + primary-star marker) with finite values;
  * meta file_tag reproduces the legacy per-dataset PDF filenames, and
    plot() still writes them (plus the hand-drawn *_sky.pdf diagnostics).

The simulated gaia+abs+rel datasets are reused from tests/test_astrometry.py.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from test_astrometry import _TRUTH, _simulate

from exozippy.system import System

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures (mirroring test_astrometry.astrometry_system, split so the
# data-only regime can be exercised before build_model)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def astro_prepared(tmp_path_factory):
    """gaia + abs + rel System, prepared but NOT built."""
    tmp_dir = tmp_path_factory.mktemp("astrom_plotdata")
    tc, epoch = _simulate(tmp_dir)
    T = _TRUTH

    config = {
        "name": "astromtest",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {
                "name": "GaiaSim",
                "file": str(tmp_dir / "sim.gaia.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": epoch,
            },
            {
                "name": "GroundAbs",
                "file": str(tmp_dir / "sim.abs.astrom"),
                "mode": "abs",
                "observer_location": "earth",
                "epoch": epoch,
            },
            {
                "name": "GroundRel",
                "file": str(tmp_dir / "sim.rel.astrom"),
                "mode": "rel",
            },
        ],
    }
    user_params = {
        "star.A.mass": {"initval": T["mstar"], "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5900, "sigma": 100},
        "star.A.feh": {"initval": -0.2, "sigma": 0.1},
        "star.A.ra": {"initval": T["ra0"]},
        "star.A.dec": {"initval": T["dec0"]},
        "star.A.pm_ra": {"initval": T["pmra"]},
        "star.A.pm_dec": {"initval": T["pmdec"]},
        "star.A.distance": {"initval": 1000.0 / T["plx"]},
        "planet.BH.mass": {"initval": T["mcomp"] * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": T["P"]},
        "orbit.BH.tc": {"initval": tc},
        "orbit.BH.secosw": {"initval": np.sqrt(T["ecc"]) * np.cos(T["w"])},
        "orbit.BH.sesinw": {"initval": np.sqrt(T["ecc"]) * np.sin(T["w"])},
        "orbit.BH.bigomega": {"initval": np.degrees(T["bigom"])},
        "orbit.BH.cosi": {"initval": np.cos(T["inc"])},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    return system


@pytest.fixture(scope="module")
def astro_built(astro_prepared):
    """The same System, built, with its start point."""
    system = astro_prepared
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plot_data_data_only_before_build(astro_prepared):
    """
    Given: a prepared (NOT built) system with gaia + abs + rel datasets
    When: plot_data is called with point=None
    Then: one data-only, JSON-serializable spec per dataset is returned,
          with empty param_deps and the legacy file_tag
    """
    system = astro_prepared
    comp = system.astrometryinstrument

    specs = comp.plot_data(system)

    assert len(specs) == 3
    for spec, name in zip(specs, ("GaiaSim", "GroundAbs", "GroundRel")):
        assert [t.role for t in spec.traces] == ["data"]
        assert spec.param_deps == []
        assert spec.meta["file_tag"] == f"astrometry_{name}"
        json.dumps(spec.to_json())
    # abs preview is the RAW offsets (no pm+plx subtraction without a point)
    abs_data = specs[1].traces[0]
    d = comp.datasets[1]
    np.testing.assert_array_equal(np.asarray(abs_data.x), d["dE_obs"])
    np.testing.assert_array_equal(np.asarray(abs_data.y), d["dN_obs"])
    assert abs_data.name == "data"


def test_plot_data_model_traces_at_start(astro_built):
    """
    Given: the built system and its start point
    When: plot_data is called with that point
    Then: each spec adds the expected finite model trace(s), carries
          non-empty param_deps, and still serializes to JSON
    """
    system, model, point = astro_built
    comp = system.astrometryinstrument

    specs = comp.plot_data(system, point)

    assert len(specs) == 3
    for spec in specs:
        assert any(t.role == "model" for t in spec.traces)
        assert spec.param_deps, f"{spec.id}: empty param_deps"
        for t in spec.traces:
            assert np.all(np.isfinite(np.asarray(t.y, dtype=float)))
        json.dumps(spec.to_json())

    # gaia: red-dot model sampled at the DATA times (kind "scatter")
    gaia_model = [t for t in specs[0].traces if t.role == "model"][0]
    assert gaia_model.kind == "scatter"
    np.testing.assert_array_equal(
        np.asarray(gaia_model.x), comp.datasets[0]["time"]
    )

    # abs: data trace is now offset-subtracted and labeled accordingly
    abs_data = [t for t in specs[1].traces if t.role == "data"][0]
    assert abs_data.name == "data - (pm+plx)"
    assert not np.allclose(np.asarray(abs_data.x), comp.datasets[1]["dE_obs"])
    abs_model = [t for t in specs[1].traces if t.role == "model"][0]
    assert abs_model.kind == "line"
    assert abs_model.name == "photocenter orbit"
    assert specs[1].meta["x_inverted"] and specs[1].meta["aspect_equal"]

    # rel: model line plus the single-point primary-star marker
    rel_models = [
        (t.name, t.kind) for t in specs[2].traces if t.role == "model"
    ]
    assert ("model", "line") in rel_models
    assert ("primary", "scatter") in rel_models
    marker = [t for t in specs[2].traces if t.name == "primary"][0]
    assert len(np.atleast_1d(marker.x)) == 1
    assert marker.style["marker"] == "*"


def test_plot_writes_legacy_pdf_filenames(astro_built, tmp_path):
    """
    Given: the built system and its start point
    When: plot() renders via the generic PlotSpec renderer
    Then: the per-dataset PDFs keep their exact legacy filenames, and the
          hand-drawn sky diagnostics are still written for gaia/abs
    """
    system, model, point = astro_built
    comp = system.astrometryinstrument
    prefix = str(tmp_path / "pf")

    comp.plot(system, point, filename_prefix=prefix)

    for name in ("GaiaSim", "GroundAbs", "GroundRel"):
        assert os.path.exists(f"{prefix}_astrometry_{name}.pdf")
    for name in ("GaiaSim", "GroundAbs"):
        assert os.path.exists(f"{prefix}_astrometry_{name}_sky.pdf")
    assert not os.path.exists(f"{prefix}_astrometry_GroundRel_sky.pdf")
