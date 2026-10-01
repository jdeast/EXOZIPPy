"""Structural priors (review 2.2.21, JDE 2026-09-30).

The direction latents' N(0, 1) priors (orbit xomega/yomega and
xbigomega/ybigomega, lens xalpha/yalpha) make their angles uniform, and
mann's ks_offset N(0, 1) IS the Ks-uncertainty prior; a user mu/sigma on
them -- and an initval on ks_offset -- is refused at ConfigManager
construction.  The flags are declarative (defaults.yaml
``structural_prior`` / ``structural_start``); see components/parameter.md,
"Structural priors".
"""

import copy

import pytest

from exozippy.config import (
    ConfigManager,
    load_base_defaults,
    structural_closed_fields,
)

_LENS_CFG = {
    "star": [{"name": "Lens"}, {"name": "Companion"}, {"name": "Source"}],
    "lens": [{"body": "Lens"}, {"body": "Companion"}],
}
_ORBIT_CFG = {"orbit": [{"name": "b"}]}
_MANN_CFG = {
    "star": [{"name": "A"}],
    "mann": [
        {"star": "A", "constrain": ["mass"], "ks": 8.782, "ks_err": 0.02}
    ],
}


@pytest.mark.parametrize(
    "key, field, cfg, physical",
    [
        ("orbit.b.xomega", "mu", _ORBIT_CFG, "orbit.<name>.omega"),
        ("orbit.b.yomega", "sigma", _ORBIT_CFG, "orbit.<name>.omega"),
        ("orbit.b.xbigomega", "sigma", _ORBIT_CFG, "orbit.<name>.bigomega"),
        ("orbit.ybigomega", "mu", _ORBIT_CFG, "orbit.<name>.bigomega"),
        ("lens.Companion.xalpha", "mu", _LENS_CFG, "lens.<name>.alpha"),
        ("lens.Companion.yalpha", "sigma", _LENS_CFG, "lens.<name>.alpha"),
        ("mann.A.ks_offset", "mu", _MANN_CFG, "ks_err"),
        ("mann.A.ks_offset", "sigma", _MANN_CFG, "ks_err"),
        ("mann.A.ks_offset", "initval", _MANN_CFG, "ks_err"),
    ],
)
def test_structural_field_raises_naming_the_physical_quantity(
    key, field, cfg, physical
):
    """
    Given a user mu or sigma on a structural N(0, 1) prior (or an initval on
    mann's ks_offset),
    When the ConfigManager is built,
    Then it raises naming the parameter and the field, and points at the
    physical quantity to constrain instead.
    """
    entry = {field: 0.5}
    if field == "sigma":
        # a centered prior, so only the structural rule fires
        entry["mu"] = 0.0
    with pytest.raises(ValueError, match="Structural prior") as exc:
        ConfigManager({key: entry}, system_config=copy.deepcopy(cfg))
    msg = str(exc.value)
    assert field in msg and physical in msg
    assert key.split(".")[-1] in msg


def test_structural_prior_refuses_a_linked_mu():
    """
    Given a link expression in a structural parameter's mu,
    When the ConfigManager is built,
    Then it is refused too -- extract_links removes the string from the
    entry, so the check must read the links.
    """
    with pytest.raises(ValueError, match="(?s)Structural prior.*xomega: mu"):
        ConfigManager(
            {
                "orbit.b.xomega": {"mu": "orbit.b.yomega + 1"},
                "orbit.b.yomega": {"initval": 1.0},
            },
            system_config=copy.deepcopy(_ORBIT_CFG),
        )


def test_direction_latents_still_accept_an_initval_and_bounds():
    """
    Given an initval or a bound on a direction latent (not a prior),
    When the ConfigManager is built,
    Then it is accepted: only mu/sigma are structural there.  ks_offset's
    bounds stay open too.
    """
    ConfigManager(
        {"orbit.b.xomega": {"initval": 0.3, "lower": -50.0}},
        system_config=copy.deepcopy(_ORBIT_CFG),
    )
    ConfigManager(
        {"mann.A.ks_offset": {"lower": -5.0}},
        system_config=copy.deepcopy(_MANN_CFG),
    )


def test_the_flags_are_declared_in_defaults_yaml_not_in_code():
    """
    Given the merged defaults.yaml tree,
    When the structural flags are read,
    Then exactly the RULED set carries them (the six direction latents and
    ks_offset), eta_* is untouched, and ks_offset alone closes initval.
    """
    defaults = load_base_defaults()
    found = {}
    for comp, block in defaults.items():
        if not isinstance(block, dict):
            continue
        for param, spec in block.items():
            if not isinstance(spec, dict):
                continue
            closed, remedy = structural_closed_fields(defaults, comp, param)
            if closed:
                assert remedy, f"{comp}.{param} declares no structural_remedy"
                found[f"{comp}.{param}"] = closed
    direction = ("mu", "sigma")
    assert found == {
        "orbit.xomega": direction,
        "orbit.yomega": direction,
        "orbit.xbigomega": direction,
        "orbit.ybigomega": direction,
        "lens.xalpha": direction,
        "lens.yalpha": direction,
        "mann.ks_offset": ("mu", "sigma", "initval"),
    }
