"""The no-grid fallback: sever a star no BC grid can host (review 2.9.15).

THE CASE (review 2.9.13).  A build-only pass over the 44 DC2018 events raised
on five, each because the source's data-derived start temperature (2350-2591
K) sat under NextGen's 2600 K floor -- and teffsed is SAMPLED through a logit
onto the grid box, so a start outside it has no raw coordinate.  The cool
starts come from the source-flux seeding: a faint source lands on the cool
end of the Mamajek dwarf locus, which runs to 2350 K at 0.078 Msun.  This
file reproduces it on the in-repo DC2018_128 fixture of
tests/test_unseen_star.py by stating a zeropoint 5 mag fainter than the
challenge's, which seeds the source at 2563 K.

JDE's rulings, both honoured:
  * "raising is the right move when we try to include the SED but we have no
    grid for it" (2026-09-14) -- so when SED PHOTOMETRY reads the star, the
    build still refuses the start;
  * the fallback when the star is read only through a cross-component tie
    (the microlensing zeropoint / blend ties): sever the link AND drop what
    severing leaves unconstrained, loudly and in the record.

The decision needs the SOLVED start, so it is made after stage 4
(Component.revise_after_starts) and System.prepare re-runs stages 3-4 from
the pre-solve inputs (ConfigManager.snapshot_solve_inputs: finalize is not
re-entrant -- it injects its solution into user_params).
"""

import copy

import pytest
from test_unseen_star import EXAMPLE_DIR, SED_SIDE, _build, _inputs, _sampled

pytestmark = pytest.mark.slow

FAINT_ZP = 27.0  # seeds the source at 2563 K, under NextGen's 2600 K floor


def _faint(tmp_path):
    config, user_params = _inputs(tmp_path)
    user_params = copy.deepcopy(user_params)
    user_params["mulensinstrument.Roman_Z087.zeropoint"]["mu"] = FAINT_ZP
    return config, user_params


def test_a_source_no_grid_hosts_is_severed_not_refused(monkeypatch, tmp_path):
    """
    Given a microlensing source whose data-derived start teffsed (2563 K)
      lies under the BC grid, read by the SED only through the zeropoint tie,
    When the system is prepared and built,
    Then it builds: the Source is severed, its SED knobs are pinned and its
      loggsed barrier lifted, the zeropoint tie is gone, the light curve's
      own constraints stay (Torres-read teff and radius sample), and the
      revision is in the record -- System.topology_revisions and a solve
      diagnostic -- with the start provenance unspoiled by the second solve.
    """
    # ARRANGE
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _faint(tmp_path)

    # ACT
    system, model = _build(config, user_params)

    # ASSERT
    sed = system.sed
    assert sed.severed_stars == frozenset({1})
    for p in SED_SIDE:
        assert not _sampled(system, p, 1), f"severed Source {p} is pinned"
    for p in ("radius", "teff"):
        assert _sampled(system, p, 1), f"Torres still reads the Source {p}"
    assert float(system.star.loggsed.lower[1]) == float("-inf")
    assert "zeropoint" not in system.mulensinstrument.manifest
    assert not system.mulensinstrument.sed_tie_mask(system).any()
    (rev,) = system.topology_revisions
    assert "SED link SEVERED for star 'Source'" in rev[0]
    assert "teffsed = 2562" in rev[0] and "(start from data)" in rev[0]
    assert "star.Source.teffsed" in rev[1]
    assert any(
        "SEVERED" in d["message"] for d in system.config_manager.diagnostics
    )
    # The second solve starts from the PRE-solve inputs: a solved start is
    # not mistaken for one the user wrote.
    cm = system.config_manager
    assert cm.initval_source("star", "teff", element=1) == "data"
    assert {rv.name for rv in model.free_RVs} >= {"star.teff_raw"}


def test_the_blend_tie_goes_with_a_severed_source(monkeypatch, tmp_path):
    """
    Given the same faint source with `sed_constrains_blend: true`,
    When it is built,
    Then the blend tie -- which compares the lens's predicted flux through
      the severed source's zeropoint -- is dropped too, so the Lens, read
      only through it, is unread again: its SED knobs are pinned.
    """
    # ARRANGE
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _faint(tmp_path)
    for c in config["mulensinstrument"]:
        c["sed_constrains_blend"] = True

    # ACT
    system, model = _build(config, user_params)

    # ASSERT
    assert system.sed.severed_stars == frozenset({1})
    for p in SED_SIDE:
        assert not _sampled(system, p, 0), f"Lens {p} is unread again"
    assert not any(
        p.name.endswith("sed_blend_prior") for p in model.potentials
    )
    assert "also on Lens" in system.topology_revisions[0][0]


def test_sed_photometry_on_the_star_keeps_the_raise(monkeypatch, tmp_path):
    """
    Given the same faint source, but with an SED photometry row naming it,
    When it is built,
    Then nothing is severed and the build refuses the start, as JDE ruled
      on 2.9.13: when the SED itself is included, no grid is an error.
    """
    # ARRANGE
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _faint(tmp_path)
    sedfile = tmp_path / "row.sed.yaml"
    sedfile.write_text(
        "model: NextGen\n"
        "filters:\n"
        "  - name: 2MASS/2MASS.Ks\n"
        "    mag: 18.0\n"
        "    err: 0.1\n"
        "    magsys: Vega\n"
        "    photType:\n"
        "      pos: [Source]\n"
    )
    config["sed"] = {"file": str(sedfile)}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="outside its hard bounds"):
        _build(config, user_params)
