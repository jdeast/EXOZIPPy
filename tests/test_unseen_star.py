"""A star no photometric term sees has its SED-side parameters pinned, and
the empirical relations skip it.

The shipped DC2018_128 example has two stars -- a microlensing Source whose
flux the light curve ties through the zeropoint, and a Lens nobody
photometers -- but no SED.  The test adds the SED machinery the DC2018
sweep configs carry (an empty-filter `.sed` file, a 2MASS Ks band so the
grid has the filter Mann needs, Mann on the Lens with a synthetic Ks,
Torres on the Source), all from in-repo data.  With the blend tie off the
Lens's teff/av/radius/teffsed/radiussed are likelihood-free, and left free
their conditional widths depend on the lens MASS: marginalizing over them
tilted pi_rel 0.3-0.6 dex low and the lens mass 2-5x high across the
2026-09 sweep2 (notes 2026-09-25, "THE LENS-DISTANCE PULL").

Three behaviours, on the real provenance ledger:
  * default (blend tie off): the Lens's SED-side parameters are pinned, the
    Source's are sampled, and Mann adds no potential (a synthetic Ks of
    pinned placeholders is circular) while Torres on the seen Source does;
  * `sed_constrains_blend: true` on the light curve makes the Lens SEEN: its
    parameters sample and Mann's potentials come back;
  * a params entry with a prior frees a pinned parameter (the pin is the
    opt-in kind, layered under the params file).
"""

import copy
import logging
import pathlib

import pytest
import yaml

from exozippy.system import System

EXAMPLE_DIR = pathlib.Path(__file__).parent / ".." / "examples" / "DC2018_128"
PINNED = ("teff", "av", "radius", "radiussed", "teffsed")

pytestmark = pytest.mark.slow


def _inputs(tmp_path):
    with open(EXAMPLE_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "DC2018_128.params.yaml") as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    assert [s["name"] for s in config["star"]] == ["Lens", "Source"]
    assert "sed" not in config
    sedfile = tmp_path / "unseen.sed.yaml"
    sedfile.write_text("model: NextGen\nfilters: []\n")
    config["sed"] = {"file": str(sedfile)}
    config["band"].append(
        {"name": "Ks_bcgrid", "filter": "2MASS/2MASS.Ks", "ld_law": "linear"}
    )
    config["mann"] = [
        {"star": "Lens", "ks": "synthetic", "constrain": ["mass", "radius"]}
    ]
    config["torres"] = [{"star": "Source", "constrain": ["mass", "radius"]}]
    for c in config["mulensinstrument"]:
        c.pop("sed_constrains_blend", None)
    return config, user_params


def _build(config, user_params):
    system = System(
        copy.deepcopy(config), user_params=copy.deepcopy(user_params)
    )
    system.prepare()
    model = system.build_model()
    return system, model


def _sampled(system, param, idx):
    return getattr(system.star, param).element_is_sampled(idx)


def test_unseen_lens_is_pinned_and_mann_skips_it(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    with caplog.at_level(logging.WARNING):
        system, model = _build(config, user_params)
    for p in PINNED:
        assert not _sampled(system, p, 0), f"Lens {p} should be pinned"
        assert _sampled(system, p, 1), f"Source {p} should be sampled"
    pots = {p.name for p in model.potentials}
    assert not any(
        n.startswith("mann.") and n.endswith("_prior") for n in pots
    ), pots
    assert any(
        n.startswith("torres.") and n.endswith("mass_prior") for n in pots
    ), pots
    assert any("no photometry" in r.getMessage() for r in caplog.records)


def test_blend_tie_makes_the_lens_seen(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    for c in config["mulensinstrument"]:
        c["sed_constrains_blend"] = True
    system, model = _build(config, user_params)
    for p in PINNED:
        assert _sampled(system, p, 0), f"Lens {p} should sample under the tie"
    pots = {p.name for p in model.potentials}
    assert any(
        n.startswith("mann.") and n.endswith("mass_prior") for n in pots
    ), pots


def test_a_params_prior_frees_a_pinned_parameter(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    user_params = copy.deepcopy(user_params)
    user_params["star.Lens.teff"] = {"mu": 3800.0, "sigma": 300.0}
    system, _ = _build(config, user_params)
    assert _sampled(system, "teff", 0)
    assert not _sampled(system, "radius", 0)
