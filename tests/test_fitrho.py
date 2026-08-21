"""
Tests for `fitrho: true` on the lens (severing rho = theta_star/theta_E).

With the flag on, rho becomes the light curve's own parameter -- sampled as
log_rho (mirroring log_s/s) -- and the stellar chain's prediction is
reported separately as rho_pred, so the pull between the light curve's rho
and the chain's rho is a published number instead of a silently resolved
tension.  With the flag off (the default) nothing changes: rho stays the
derived identity and neither log_rho nor rho_pred exists.

Vocabulary matches the planet component's fitbeam (sample the quantity
directly instead of deriving it from the physical chain).
"""

from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


def _load_kmt(fitrho=False, extra_params=None, build=False):
    import os

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")

    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        if fitrho:
            config["lens"][0]["fitrho"] = True
        if extra_params:
            user_params.update(extra_params)
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model() if build else None
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def kmt_fitrho():
    """Given the KMT example with fitrho: true and an explicit user rho
    start, when the system is prepared and built, provide
    (system, model, initial point)."""
    system, model = _load_kmt(
        fitrho=True,
        extra_params={"lens.Lens.rho": {"initval": 0.005}},
        build=True,
    )
    return system, model, model.initial_point()


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_default_off_no_new_parameters():
    """
    Given the shipped KMT config (no fitrho),
    When the system is prepared,
    Then rho stays the derived identity and neither log_rho nor rho_pred
    is declared.
    """
    system, _ = _load_kmt()
    manifest = system.lens.manifest
    assert "log_rho" not in manifest
    assert "rho_pred" not in manifest
    assert manifest["rho"].get("expr_key") == "default"


def test_fitrho_samples_log_rho(kmt_fitrho):
    """
    Given fitrho: true,
    When the model is built,
    Then log_rho is sampled, rho = 10**log_rho, and the user's rho initval
    back-solved to the log_rho start through the relaxation engine.
    """
    system, model, point = kmt_fitrho
    assert "lens.log_rho_raw" in [v.name for v in model.value_vars]

    # Derived Parameters do not emit named Deterministics; evaluate the
    # Parameter value nodes (graphs over the model's RVs) directly.
    log_rho = np.atleast_1d(_eval(model, system.lens.log_rho.value, point))
    rho = np.atleast_1d(_eval(model, system.lens.rho.value, point))
    assert np.allclose(rho, 10.0**log_rho, rtol=1e-12)
    # The engine's rho <-> log_rho relation turned the user rho start into
    # the sampled coordinate's start.
    assert np.isclose(log_rho[0], np.log10(0.005), atol=1e-6)


def test_rho_pred_reports_the_chain(kmt_fitrho):
    """
    Given fitrho: true,
    When the model is built,
    Then rho_pred carries the stellar chain's theta_star/theta_E -- the
    quantity rho used to BE -- computed from the same star radius/distance
    and theta_E, and it is decoupled from the sampled rho.
    """
    system, model, point = kmt_fitrho
    rho_pred = float(
        np.atleast_1d(_eval(model, system.lens.rho_pred.value, point))[0]
    )
    theta_e = float(
        np.atleast_1d(_eval(model, system.lens.theta_E.value, point))[0]
    )
    radius = np.atleast_1d(_eval(model, system.star.radius.value, point))
    distance = np.atleast_1d(_eval(model, system.star.distance.value, point))
    src = int(system.lens.source_map[0])
    RSUN_TO_AU = 0.0046505
    theta_star_mas = radius[src] * RSUN_TO_AU / distance[src] * 1000.0
    assert np.isclose(rho_pred, theta_star_mas / theta_e, rtol=1e-3)

    rho = float(np.atleast_1d(_eval(model, system.lens.rho.value, point))[0])
    assert np.isclose(rho, 0.005, rtol=1e-6)
    # At the START they coincide by design: the relaxation engine knows both
    # relations, so the user rho seeded the stellar chain consistently.
    # Decoupling means moving the sampled coordinate moves rho and NOT
    # rho_pred -- the severed identity in one perturbation.
    point2 = dict(point)
    point2["lens.log_rho_raw"] = point["lens.log_rho_raw"] + 3.0
    rho2 = float(np.atleast_1d(_eval(model, system.lens.rho.value, point2))[0])
    rho_pred2 = float(
        np.atleast_1d(_eval(model, system.lens.rho_pred.value, point2))[0]
    )
    assert not np.isclose(rho2, rho, rtol=1e-3)
    assert np.isclose(rho_pred2, rho_pred, rtol=1e-12)
