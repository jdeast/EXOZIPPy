"""The RGES-PIT container's pipeline function (containers/rges_pit).

Pins the input contract (README.md there): a malformed upstream event raises
naming the key, and a valid one builds a model for each architecture.  No
sampling: the fit itself is every other test's business.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from exozippy.system import System

HERE = Path(__file__).resolve().parent.parent / "containers" / "rges_pit"


@pytest.fixture(scope="module")
def cp():
    spec = importlib.util.spec_from_file_location(
        "rges_component_pipeline", HERE / "component_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.INPUT_ROOT = HERE / "input_data"
    return mod


@pytest.fixture
def event():
    with open(HERE / "input_events" / "sim_pspl_0001.json") as f:
        return json.load(f)


def test_example_event_is_valid(cp, event):
    assert cp.validate_input(event) is event


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda e: e.pop("ra"), "'ra' is missing"),
        (lambda e: e.update(architecture="2L1S"), "architecture must be"),
        (lambda e: e.update(architecture=None), "architecture must be"),
        (lambda e: e.update(event_id="a/b"), "event_id"),
        (lambda e: e.update(extra=1), "unknown key"),
        (lambda e: e.update(light_curves=[]), "non-empty list"),
        (lambda e: e["light_curves"][0].pop("time_frame"), "'time_frame'"),
        (
            lambda e: e["light_curves"][0].update(data_format="dia"),
            "data_format",
        ),
        (
            lambda e: e["light_curves"][0].update(columns={"time": "t"}),
            "only to a .parquet",
        ),
        (
            lambda e: e["light_curves"][0].update(path="lc.parquet"),
            "'columns' is missing",
        ),
        (
            lambda e: e["light_curves"].append(
                copy.deepcopy(e["light_curves"][0])
            ),
            "share the instrument name",
        ),
    ],
)
def test_malformed_event_raises(cp, event, mutate, match):
    mutate(event)
    with pytest.raises(cp.InputEventError, match=match):
        cp.validate_input(event)


def test_architecture_is_optional(cp, event):
    settings = cp.load_settings(HERE / "fit_settings.yaml")
    event.pop("architecture")
    cp.validate_input(event)
    assert cp.architecture(event, settings) == settings["default_architecture"]


def test_planned_inputs_are_accepted_and_reported(cp, event):
    event["lens_flux"] = {"path": "wg4/posterior.parquet"}
    event["field"] = {}
    event["light_curves"][0]["zeropoint"] = {"zp": 25.0}
    cp.validate_input(event)
    assert cp.inputs_not_used(event) == [
        "lens_flux",
        "field",
        "light_curves[0].zeropoint",
    ]


def test_parquet_light_curve_is_staged(cp, event, tmp_path):
    import numpy as np
    import pandas as pd

    src = np.loadtxt(HERE / "input_data" / "sim_pspl_F146.dat")
    df = pd.DataFrame(src, columns=["bjd", "mag", "mag_err"])
    df.to_parquet(tmp_path / "lc.parquet")
    lc = dict(
        event["light_curves"][0],
        path="lc.parquet",
        columns={"time": "bjd", "value": "mag", "error": "mag_err"},
    )
    old = cp.INPUT_ROOT
    cp.INPUT_ROOT = tmp_path
    try:
        out = cp.stage_light_curve(lc, tmp_path / "staged")
    finally:
        cp.INPUT_ROOT = old
    np.testing.assert_allclose(np.loadtxt(out), src, rtol=0, atol=1e-9)


@pytest.mark.parametrize("arch, n_free", [("1S1L", 12), ("1S2L", 18)])
def test_config_builds_a_model(cp, event, tmp_path, arch, n_free):
    event["architecture"] = arch
    settings = cp.load_settings(HERE / "fit_settings_quick.yaml")
    files = {
        cp._instrument_name(lc): cp.stage_light_curve(lc, tmp_path)
        for lc in event["light_curves"]
    }
    config = cp.build_config(event, files, tmp_path / "fit", settings)
    params = cp.build_user_params(event, cp.architecture(event, settings))
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    assert len(model.free_RVs) == n_free


def test_results_csv_header_variants(cp, tmp_path):
    p = tmp_path / "r.csv"
    p.write_text(
        "# parname, mode, weight, value, up_err, low_err\n"
        "# unimodal posterior: a note line\n"
        "lens.Source.t_E,0,1.0,25.1,0.2,0.3\n"
        "star.Lens.ra,0,1.0,268.0,,\n"
    )
    assert cp.read_results_csv(p) == [
        {
            "parname": "lens.Source.t_E",
            "mode": 0.0,
            "weight": 1.0,
            "value": 25.1,
            "up_err": 0.2,
            "low_err": 0.3,
        },
        {
            "parname": "star.Lens.ra",
            "mode": 0.0,
            "weight": 1.0,
            "value": 268.0,
            "up_err": None,
            "low_err": None,
        },
    ]
