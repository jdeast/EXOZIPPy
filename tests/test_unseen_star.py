"""A star no photometric term sees: the SED reads only the stars it predicts a
flux for, so such a star's radius/teff/feh take the structural inactive tier
unless a relation or a user prior reads them, and its SED-side av/teffsed/
radiussed are pinned.

The shipped DC2018_128 example has two stars -- a microlensing Source and a
Lens nobody photometers -- but no SED.  The test adds the SED machinery the
DC2018 sweep configs carry (an empty-filter `.sed` file, the light curve's
band on the gridded Roman F087 filter with the challenge's AB zeropoint
STATED, so the source's flux is really tied to the SED; a 2MASS Ks band so
the grid has the filter Mann needs, Mann on the Lens with a synthetic Ks,
Torres on the Source), all from in-repo data.  (Until review 2.9.11 the
fixture kept the example's ungridded `Roman.WFI.Z087` label and stated no
zeropoint, so NOTHING read either star's SED flux -- the Source counted as
seen only because every source did, and the blend tie below was never
built.)  Left free, the Lens's Teff
had a conditional width that grew with the lens MASS (Ks is a weak
thermometer for a hotter star), and marginalizing over it tilted pi_rel
0.3-0.6 dex low and the lens mass 2-5x high across the 2026-09 sweep2
(notes 2026-09-25/27, "THE LENS-DISTANCE PULL").  The first cut (PR #337)
pinned radius/teff/feh too and SKIPPED the relations on the unseen star;
that was wrong -- Mann on the lens is a chain of overlapping constraints,
satisfied to 1.00 +/- 0.02 across sweep2 -- and is what this file now
guards against.

On the real provenance ledger:
  * default (blend tie off, Mann on the Lens): the Lens's av/teffsed/
    radiussed are pinned; radius and feh stay ACTIVE because Mann reads
    them and Mann's potentials are present; teff is inactive because
    nothing reads it; the Source's are all sampled;
  * `mamajek` on the Lens gives its teff a reader: it samples, and the
    Teff penalty is in the model;
  * `sed_constrains_blend: true` on the light curve makes the Lens SEEN:
    everything samples;
  * with NO stated zeropoint nothing reads the Source's SED flux either, so
    its av/teffsed/radiussed are pinned too (review 2.9.11's data-free
    inventory) while Torres keeps its radius/teff/feh;
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
SED_SIDE = ("av", "radiussed", "teffsed")
STRUCTURE = ("radius", "teff", "feh")

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
    # The light curve's band on a filter the BC grid has, and the
    # zeropoint stated, exactly as the sweep configs do: the tie that makes
    # the Source photometrically SEEN.
    (z087,) = [b for b in config["band"] if b["name"] == "Z087"]
    z087["filter"] = "Roman/WFI.F087"
    for c in config["mulensinstrument"]:
        c["magsys"] = "AB"
    user_params = dict(user_params)
    user_params["mulensinstrument.Roman_Z087.zeropoint"] = {
        "mu": 22.0,
        "sigma": 0.02,
    }
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


def _pots(model):
    return {p.name for p in model.potentials}


def test_unseen_lens_keeps_what_mann_reads_and_loses_the_rest(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    with caplog.at_level(logging.INFO):
        system, model = _build(config, user_params)
    for p in SED_SIDE:
        assert not _sampled(system, p, 0), f"Lens {p} should be pinned"
    for p in ("radius", "feh"):
        assert _sampled(system, p, 0), f"Lens {p} is read by Mann"
    assert not _sampled(system, "teff", 0), "nothing reads the Lens teff"
    for p in SED_SIDE + STRUCTURE:
        assert _sampled(system, p, 1), f"Source {p} should be sampled"
    pots = _pots(model)
    assert any(
        n.startswith("mann.") and n.endswith("mass_prior") for n in pots
    ), pots
    assert any(
        n.startswith("mann.") and n.endswith("radius_prior") for n in pots
    ), pots
    assert any(
        n.startswith("torres.") and n.endswith("mass_prior") for n in pots
    ), pots
    assert any(
        "nothing reads the SED flux of Lens" in r.getMessage()
        for r in caplog.records
    )
    # The source starts on the grid (3571 K), so the no-grid fallback of
    # tests/test_sed_no_grid_fallback.py does not fire.
    assert system.topology_revisions == []
    assert system.sed.severed_stars == frozenset()


def test_mamajek_gives_the_unseen_lens_teff_a_reader(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    config["mamajek"] = [{"star": "Lens"}]
    system, model = _build(config, user_params)
    assert _sampled(system, "teff", 0)
    for p in SED_SIDE:
        assert not _sampled(system, p, 0), f"Lens {p} should still be pinned"
    pots = _pots(model)
    assert "mamajek.teff_prior" in pots or any(
        n.startswith("mamajek.") and n.endswith("teff_prior") for n in pots
    ), pots
    assert not any(
        n.startswith("mamajek.") and n.endswith("radius_prior") for n in pots
    ), "radius is opt-in on mamajek"


def test_blend_tie_makes_the_lens_seen(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    for c in config["mulensinstrument"]:
        c["sed_constrains_blend"] = True
    system, model = _build(config, user_params)
    for p in SED_SIDE + STRUCTURE:
        assert _sampled(system, p, 0), f"Lens {p} should sample under the tie"
    pots = _pots(model)
    assert any(
        n.startswith("mann.") and n.endswith("mass_prior") for n in pots
    ), pots


def test_a_params_prior_frees_a_pinned_parameter(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    user_params = copy.deepcopy(user_params)
    user_params["star.Lens.av"] = {"mu": 0.5, "sigma": 0.3}
    system, _ = _build(config, user_params)
    assert _sampled(system, "av", 0)
    assert not _sampled(system, "teffsed", 0)


def test_an_unstated_zeropoint_leaves_the_source_unseen_too(
    monkeypatch, tmp_path
):
    """
    Given the same system with the zeropoint prior REMOVED,
    When it is built,
    Then nothing reads the Source's SED flux either (the zeropoint is only
      reported, review 2.2.21), so its av/teffsed/radiussed are pinned like
      the Lens's -- the 2.9.11 data-free inventory -- while Torres still
      reads its radius/teff/feh, and sed.errscale is pinned (no rows).
    """
    # ARRANGE
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs(tmp_path)
    del user_params["mulensinstrument.Roman_Z087.zeropoint"]

    # ACT
    system, model = _build(config, user_params)

    # ASSERT
    for idx in (0, 1):
        for p in SED_SIDE:
            assert not _sampled(system, p, idx), f"star {idx} {p} pinned"
    for p in STRUCTURE:
        assert _sampled(system, p, 1), f"Torres reads the Source {p}"
    free = {rv.name for rv in model.free_RVs}
    assert "sed.errscale_raw" not in free
    inv = system.sed.data_free_inventory(system)
    assert inv.unread_stars == (0, 1)
