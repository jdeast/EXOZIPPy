"""Component-computed values must not masquerade as the user's own.

Two components used to write straight into ``config_manager.user_params``:

  * ``SED._inject_grid_bounds`` -- the BC grid's coverage in (teff, feh, av),
    injected as ``lower``/``upper`` on ``star.teffsed``/``star.feh``/``star.av``;
  * ``AstrometryInstrument.register_parameters`` -- ``sigma: 0`` on the
    ``fluxfrac`` element of any file whose photocenter flux fraction the SED
    supplies, since nothing then reads the sampled one.

Both used ``setdefault``, so a user's own entry still won.  The defect was not
precedence, it was PROVENANCE: an entry in ``user_params`` is indistinguishable
from one the user wrote, so the relaxation engine's ledger, ``export_solution``,
``initval_source`` and the GUI all reported values the user never supplied.  The
SED site had a second consequence -- ``finalize_user_params`` registers every
unmapped ``user_params`` key as a leaf symbol, so bare ``star.av`` & co. became
relaxation-engine symbols with their own ledger rows (the "orphaned 2-part star
rows" ``export_solution`` filters out by hand).

Neither value is a start value, so neither is an ``add_hint``: a hint is a
ranked scalar feeding ``initval``, and ``lower``/``upper``/``sigma`` never enter
the provenance ledger at all.  Both are ``"overrides"``-channel items -- a grid
VALIDITY limit and a structural PIN -- applied under the user's params file:

  * astrometry owns ``fluxfrac``, so it uses the manifest ``"overrides"`` dict,
    exactly as ``Instrument._register_gp`` pins non-GP files;
  * the SED does NOT own ``star.*``, so it uses ``ConfigManager.add_override``,
    the same channel reached by path.

These tests pin the channel, the surviving user precedence, and the ledger
attribution.  Start logp is unchanged on every shipped example that exercises
either component (astrometry_sim, GaiaBH1, gj1214, hat3, hat3_staronly,
HIP1349, kelt4, kelt4_rv+transit+sed, KMT-2019-BLG-1806, wasp18).
"""

import numpy as np
import pytest

from exozippy.system import System

# The NextGen BC grid axes SED._inject_grid_bounds reads.  Asserted against the
# live grid in test_sed_grid_bounds_match_the_grid_axes rather than trusted.
_GRID = {
    "teffsed": (2600.0, 10000.0),
    "feh": (-4.0, 0.5),
    "av": (0.0, 6.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sed_config(tmp_path):
    """A minimal star + band + sed topology (no catalog rows needed)."""
    sed_file = tmp_path / "empty.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")
    return {
        "star": [{"name": "A", "mist": False}],
        "band": [{"name": "GaiaG", "filter": "GAIA2r.G", "ld_law": "linear"}],
        "sed": {"file": str(sed_file)},
    }


def _first(arr):
    return float(np.atleast_1d(arr)[0])


# ---------------------------------------------------------------------------
# SED grid bounds -> ConfigManager.add_override
# ---------------------------------------------------------------------------


def test_sed_grid_bounds_use_the_override_channel_not_user_params(tmp_path):
    """
    Given a system with a sed block,
    When the SED reads its BC grid axes at construction,
    Then the resulting bounds land on config_manager's cross-component
    override channel and NOT in user_params.
    """
    system = System(_sed_config(tmp_path), {})
    cm = system.config_manager

    for param, (lo, hi) in _GRID.items():
        assert cm.param_overrides[f"star.{param}"] == {
            "lower": lo,
            "upper": hi,
        }

    # The whole point: nothing the user did not write is in user_params.
    assert cm.user_params == {}


def test_sed_grid_bounds_match_the_grid_axes(tmp_path):
    """
    Given a system with a sed block,
    When resolve() is asked for the bounded star parameters,
    Then it returns the BC grid's own axis limits -- i.e. routing the bounds
    through the override channel still applies them.
    """
    system = System(_sed_config(tmp_path), {})
    cm = system.config_manager
    axes = system.sed.grid_axes

    for param, key in (
        ("teffsed", "teff_pts"),
        ("feh", "feh_pts"),
        ("av", "av_pts"),
    ):
        cfg = cm.resolve("star", param)
        assert _first(cfg["lower"]) == pytest.approx(float(axes[key].min()))
        assert _first(cfg["upper"]) == pytest.approx(float(axes[key].max()))
        assert _first(cfg["lower"]) == pytest.approx(_GRID[param][0])
        assert _first(cfg["upper"]) == pytest.approx(_GRID[param][1])


def test_sed_grid_bounds_are_not_attributed_to_the_user(tmp_path):
    """
    Given a system with a sed block and an empty params file,
    When the bounded star parameters are resolved,
    Then they are flagged as component-computed (auto_estimated), not as
    anything the user modified.
    """
    system = System(_sed_config(tmp_path), {})
    cm = system.config_manager

    for param in _GRID:
        cfg = cm.resolve("star", param)
        assert cfg["auto_estimated"] is True, param
        assert cfg["user_modified"] is False, param
        assert cfg["user_prior_modified"] is False, param


def test_sed_grid_bounds_leave_no_trace_in_the_provenance_ledger(tmp_path):
    """
    Given a prepared system with a sed block,
    When the relaxation engine has run,
    Then no bare `star.<param>` symbol, ledger row or user_params entry exists
    for the grid-bounded parameters, and the start value of each is reported
    as coming from defaults rather than from the user.
    """
    system = System(_sed_config(tmp_path), {})
    cm = system.config_manager
    system.prepare()

    for param in _GRID:
        path = f"star.{param}"
        # finalize_user_params registers every unmapped user_params key as a
        # relaxation-engine leaf symbol; an injected bound therefore invented
        # a symbol, a ledger row and (via the inject-back) a user_params
        # initval for a parameter instance that does not exist.
        assert path not in cm.user_params, path
        assert path not in cm.master_symbol_map, path
        assert path not in cm._last_provenance, path

        assert cm.initval_source("star", param, element=0, name="A") != "user"

    # No orphaned 2-part rows at all for a list-instanced component.
    assert [k for k in cm.master_symbol_map if k.count(".") == 1] == []


def test_a_tighter_user_bound_still_wins_over_the_sed_grid_bound(tmp_path):
    """
    Given a user who bounds star.A.teffsed inside the BC grid,
    When the parameter is resolved,
    Then the user's tighter bounds are what apply.
    """
    user_params = {"star.A.teffsed": {"lower": 4000.0, "upper": 5000.0}}
    system = System(_sed_config(tmp_path), user_params)
    cfg = system.config_manager.resolve("star", "teffsed")

    assert _first(cfg["lower"]) == pytest.approx(4000.0)
    assert _first(cfg["upper"]) == pytest.approx(5000.0)
    # ...and now it really IS the user's entry.
    assert cfg["user_modified"] is True
    assert cfg["user_prior_modified"] is True


def test_a_looser_user_bound_is_clipped_to_the_sed_grid_bound(
    tmp_path, caplog
):
    """
    Given a user who bounds star.A.av OUTSIDE the BC grid's av axis,
    When the parameter is resolved,
    Then the grid's validity limit clips it and the clip is logged.

    The old setdefault could not do this: it saw the user's key, left it
    alone, and never applied the grid bound at all -- so the sampler was free
    to walk off the interpolator, which is exactly what the injection existed
    to prevent.  apply_value's max(lower)/min(upper) is what makes the
    "overrides" channel the right one for a validity limit.
    """
    user_params = {"star.A.av": {"lower": -3.0, "upper": 50.0}}
    system = System(_sed_config(tmp_path), user_params)

    with caplog.at_level("WARNING"):
        cfg = system.config_manager.resolve("star", "av")

    assert _first(cfg["lower"]) == pytest.approx(_GRID["av"][0])
    assert _first(cfg["upper"]) == pytest.approx(_GRID["av"][1])
    assert "validity bound" in caplog.text
    assert "av" in caplog.text


# ---------------------------------------------------------------------------
# Astrometry fluxfrac pin -> manifest "overrides"
# ---------------------------------------------------------------------------


def _astrometry_sed_case(tmp_path, extra_user_params=None):
    """Host + luminous companion, a gaia-mode file with band +
    companion_star_ndx, and a sed block: the topology that pins fluxfrac.

    Mirrors test_sed_flux_constraints.test_astrometry_fluxfrac_derived_from_sed.
    """
    from test_astrometry import _TRUTH, _simulate

    tc, epoch = _simulate(tmp_path)
    T = _TRUTH

    sed_file = tmp_path / "empty.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    config = {
        "star": [{"name": "A", "mist": False}, {"name": "C", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "band": [{"name": "GaiaG", "filter": "GAIA2r.G", "ld_law": "linear"}],
        "astrometryinstrument": [
            {
                "name": "GaiaSim",
                "file": str(tmp_path / "sim.gaia.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": epoch,
                "band": "GaiaG",
                "companion_star_ndx": 1,
            },
        ],
        "sed": {"file": str(sed_file)},
    }
    user_params = {
        "star.A.ra": {"initval": T["ra0"]},
        "star.A.dec": {"initval": T["dec0"]},
        "star.A.pm_ra": {"initval": T["pmra"]},
        "star.A.pm_dec": {"initval": T["pmdec"]},
        "star.C.ra": {"initval": T["ra0"]},
        "star.C.dec": {"initval": T["dec0"]},
        "star.C.pm_ra": {"initval": T["pmra"]},
        "star.C.pm_dec": {"initval": T["pmdec"]},
        "planet.BH.mass": {"initval": T["mcomp"] * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": T["P"]},
        "orbit.BH.tc": {"initval": tc},
        "orbit.BH.secosw": {"initval": np.sqrt(T["ecc"]) * np.cos(T["w"])},
        "orbit.BH.sesinw": {"initval": np.sqrt(T["ecc"]) * np.sin(T["w"])},
        "orbit.BH.bigomega": {"initval": np.degrees(T["bigom"])},
        "orbit.BH.cosi": {"initval": np.cos(T["inc"])},
    }
    for s in ("A", "C"):
        user_params[f"star.{s}.mass"] = {"initval": 1.0, "sigma": 0.05}
        user_params[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        user_params[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        user_params[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        user_params[f"star.{s}.distance"] = {"initval": 1000.0 / T["plx"]}
    user_params.update(extra_user_params or {})

    system = System(config, user_params=user_params)
    system.prepare()
    return system


@pytest.fixture(scope="module")
def pinned_fluxfrac_system(tmp_path_factory):
    """The SED-fluxfrac topology, prepared (stage 1-3 only)."""
    return _astrometry_sed_case(tmp_path_factory.mktemp("fluxfrac_pin"))


def test_fluxfrac_pin_is_a_manifest_override_not_a_user_param(
    pinned_fluxfrac_system,
):
    """
    Given a gaia file whose fluxfrac comes from the SED,
    When parameters are registered,
    Then the sigma: 0 pin is declared on the manifest "overrides" channel and
    nothing is written into user_params.
    """
    system = pinned_fluxfrac_system
    inst = system.astrometryinstrument
    cm = system.config_manager

    assert inst._sed_fluxfrac == [True]
    assert inst.manifest["fluxfrac"]["overrides"]["sigma"] == [0.0]

    # user_params may carry the engine's injected-back initval for this path
    # (every mapped path gets one); what it must NOT carry is the pin.
    entry = cm.user_params.get("astrometryinstrument.0.fluxfrac") or {}
    assert "sigma" not in entry


def test_fluxfrac_pin_is_not_attributed_to_the_user(pinned_fluxfrac_system):
    """
    Given the same system,
    When fluxfrac is resolved WITHOUT the manifest overrides,
    Then nothing marks its PRIOR as user-modified -- the pin is the
    component's, and only the component's manifest carries it.

    ``user_modified`` is deliberately not asserted: finalize_user_params
    injects a solved ``initval`` back into user_params for every mapped path,
    so it is True here whatever this component does.  ``user_prior_modified``
    is the flag that requires a physics key (sigma/mu/lower/upper), which is
    exactly what the old injection wrote and this one does not -- and it is
    the flag run.py prints as ``*`` in the startup table.
    """
    cm = pinned_fluxfrac_system.config_manager
    cfg = cm.resolve("astrometryinstrument", "fluxfrac", shape=(1,))

    assert cfg["user_prior_modified"] is False
    assert cm.initval_source(
        "astrometryinstrument", "fluxfrac", element=0
    ) != ("user")


def test_fluxfrac_pin_still_fixes_the_parameter(pinned_fluxfrac_system):
    """
    Given the same system,
    When the model is built,
    Then the sampled fluxfrac is still fixed and absent from the free RVs --
    the behaviour the user_params write bought, preserved by the move.
    """
    system = pinned_fluxfrac_system
    model = system.build_model()

    assert _first(system.astrometryinstrument.fluxfrac.sigma) == 0.0
    # A sampled element's free RV is named "<label>_raw", so match the prefix
    # rather than the bare label (which is never a free RV name and so would
    # pass vacuously).
    assert not [
        rv.name
        for rv in model.free_RVs
        if rv.name.startswith("astrometryinstrument.fluxfrac")
    ]


def test_a_user_sigma_beats_the_fluxfrac_pin(tmp_path):
    """
    Given a user who gives the SED-fluxfrac file its own Gaussian sigma,
    When the model is built,
    Then the user's sigma applies and the element is sampled again -- the
    setdefault semantics the override channel has to preserve.
    """
    system = _astrometry_sed_case(
        tmp_path,
        # mu as well as sigma: validate_sigma_has_center refuses a Gaussian
        # prior with no center, which is orthogonal to what is tested here.
        {"astrometryinstrument.GaiaSim.fluxfrac": {"mu": 0.3, "sigma": 0.1}},
    )
    model = system.build_model()

    assert _first(system.astrometryinstrument.fluxfrac.sigma) == pytest.approx(
        0.1
    )
    assert "astrometryinstrument.fluxfrac_raw" in [
        rv.name for rv in model.free_RVs
    ]


# ---------------------------------------------------------------------------
# The MMEXOFAST run/skip decision
# ---------------------------------------------------------------------------


def test_neither_injection_can_reach_probe_derivable(tmp_path):
    """
    Given a system with a sed block,
    When probe_derivable builds its `flat` from user_params,
    Then the grid bounds contribute nothing to it either before or after the
    move -- they carry no initval and no mu, which is what probe_derivable
    reads.  So the MMEXOFAST run/skip decision (user_hints_sufficient, which
    tests provenance strictly above RANK_DEFAULT) is untouched by this change.
    """
    system = System(_sed_config(tmp_path), {})
    cm = system.config_manager

    # No value channel at all -- only bounds.
    for fields in cm.param_overrides.values():
        assert set(fields) <= {"lower", "upper"}

    system.prepare()
    derivable = cm.probe_derivable(
        [f"star.0.{p}" for p in _GRID] + ["star.0.teff"]
    )
    assert all(not p.endswith(".av") for p in derivable)
