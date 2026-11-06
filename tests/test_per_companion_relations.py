"""Every lens companion has its own relaxation-engine relations, and the total
lens mass is a sum over every body (review 8.6.13; mulensing.md, "Three or
more lens bodies").

Before, symbolic_physics mapped lens element 1 only and the mass sum existed
for exactly two bodies, so on a 3+ body lens `mulensevent.0.mlens_total` had
no relation to the body masses.  A seeded t_E was satisfied by moving
mlens_total -- a derived parameter whose start never reaches the model -- and
a seeded q for a second companion went nowhere: on examples/KMT-2021-BLG-1122L
(t_E 30 d, q 0.3 and 0.05, no body masses) the model started at t_E 124.7 d
with both companions at q = 3.16 (review 1.1.12; and t_E "not derivable" at
stage 1, review 2.6.34).  Pinned here:

1. The symbol maps: one per source, then one per companion, each companion's
   geometry and q mapped to its OWN lens element.
2. The extra relation sums every lens body's mass (and does not exist for a
   single lens).
3. On the shipped triple lens, direct t_E and per-companion q seeds are all
   delivered by the built model, and the body masses follow from them.
"""

import os
import shutil

import numpy as np
import pytest
import sympy as sp
import yaml

from exozippy.components.mulensing import symbolic_physics as sym

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "KMT-2021-BLG-1122L"
)


def _triple_config():
    return {
        "star": [{"name": "Lens"}, {"name": "B"}, {"name": "Source"}],
        "planet": [{"name": "c"}],
        "mulensevent": [{}],
        "lens": [
            {"body": "star.Lens"},
            {"body": "star.B"},
            {"body": "planet.c"},
        ],
        "source": [{"body": "star.Source"}],
    }


def test_each_companion_gets_its_own_map():
    cfg = _triple_config()
    maps = sym.get_symbol_map(cfg["mulensevent"][0], cfg)
    # one per source, then one per companion
    assert len(maps) == 3
    source_map, comp1, comp2 = maps
    # the event chain lives in the source map; no companion symbol does
    assert source_map["t_E"] == "mulensevent.0.t_E"
    assert not {"q_lens", "s", "alpha", "companion_mass"} & set(source_map)
    assert comp1["q_lens"] == "lens.1.q"
    assert comp1["companion_mass"] == "star.1.mass"
    assert comp1["log_s"] == "lens.1.log_s"
    assert comp2["q_lens"] == "lens.2.q"
    assert comp2["companion_mass"] == "planet.0.mass"
    assert comp2["xalpha"] == "lens.2.xalpha"
    # q_j is always relative to the PRIMARY
    assert comp1["primary_lens_mass"] == comp2["primary_lens_mass"]
    assert comp1["primary_lens_mass"] == "star.0.mass"
    # a companion map never maps the total mass: the binary-sum form would
    # otherwise instantiate once per companion and contradict itself
    assert "lens_mass_total" not in comp1 and "lens_mass_total" not in comp2


def test_the_total_mass_sums_every_body():
    cfg = _triple_config()
    (rel,) = sym.get_extra_relations(cfg["mulensevent"][0], cfg)
    assert rel.lhs == sp.Symbol("mulensevent.0.mlens_total")
    assert rel.rhs == sp.Add(
        sp.Symbol("star.0.mass"),
        sp.Symbol("star.1.mass"),
        sp.Symbol("planet.0.mass"),
    )


def test_a_single_lens_has_no_mass_sum():
    cfg = _triple_config()
    cfg["lens"] = cfg["lens"][:1]
    assert sym.get_extra_relations(cfg["mulensevent"][0], cfg) == []
    (source_map,) = sym.get_symbol_map(cfg["mulensevent"][0], cfg)
    assert source_map["lens_mass_total"] == "star.0.mass"


@pytest.mark.slow
def test_direct_seeds_on_a_triple_lens_are_delivered(tmp_path):
    """
    Given the shipped triple lens with its body-mass, distance and
      proper-motion seeds REMOVED, and t_E plus both companions' q seeded
      directly (the published-parameter style),
    When the model is built,
    Then every one of those starts is produced by the compiled graph, and the
      engine's body masses carry them: M_j = q_j M_1 and mlens_total is their
      sum.
    """
    from exozippy.diagnostics import ModelAuditor
    from exozippy.system import System

    work = tmp_path / "kb211122"
    shutil.copytree(
        EXAMPLE, work, ignore=shutil.ignore_patterns("fitresults*", "*.job")
    )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
        params = yaml.safe_load(open(cfg.pop("parameter_file")))
        cfg["parameter_file"] = None
        for key in list(params):
            if key.split(".")[-1] in (
                "mass",
                "logmass",
                "distance",
                "pm_ra",
                "pm_dec",
            ):
                params.pop(key)
        t_E, q_b, q_c = 30.0, 0.30, 0.05
        params["mulensevent.t_E"] = {"initval": t_E}
        params["lens.LensB.q"] = {"initval": q_b}
        params["lens.LensC.q"] = {"initval": q_c}

        system = System(cfg, params)
        system.prepare()
        led = system.config_manager._last_resolved
        logm = [led[f"star.{i}.logmass"] for i in range(3)]
        np.testing.assert_allclose(
            [logm[1] - logm[0], logm[2] - logm[0]],
            np.log10([q_b, q_c]),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            led["mulensevent.0.mlens_total"],
            sum(10.0**m for m in logm),
            rtol=1e-9,
        )

        model = system.build_model()
        start = system.get_raw_start(model)
        misses = ModelAuditor(model, system, start).check_user_starts()
        assert misses == [], misses
    finally:
        os.chdir(cwd)
