"""Model-spectrum interpolation behind the SED figure (review 2.9.2).

``NextGenPlot._interp_spectra`` brackets each draw with the two nearest
spectra-grid nodes per axis and interpolates between them. Two things went
wrong there and both are exercised here without building a System: the
interpolation used scipy's default ``bounds_error=True``, so a draw off the
grid raised inside ``SED.plot`` (which run.py calls unguarded before
sampling); and a bracketing node absent from the grid left its flux at the
initialized 0.0 behind a ``warnings.warn``, i.e. interpolated the figure
against a fabricated zero.

The fixture is a hand-built 2x2x2 grid rather than the shipped 249 MB
NextGen table -- what is under test is the bracketing/interpolation logic,
which does not care how many nodes there are.
"""

import numpy as np
import pandas as pd
import pytest

from exozippy.models.NextGen.BCs.plot import NextGenPlot

_WAVE = np.array([5000.0, 6000.0])
_TEFF_PTS = (3000.0, 4000.0)
_LOGG_PTS = (4.0, 5.0)
_FEH_PTS = (0.0, 0.5)


def _spectra_frame(drop_node=None):
    """A complete 2x2x2 spectra table, optionally missing one node.

    ``flux`` is set to the node's teff so an interpolated result is easy to
    predict by hand.
    """
    rows = []
    for teff in _TEFF_PTS:
        for logg in _LOGG_PTS:
            for feh in _FEH_PTS:
                if drop_node is not None and (teff, logg, feh) == drop_node:
                    continue
                rows.append(
                    {
                        "filename": f"{teff}_{logg}_{feh}",
                        "teff": teff,
                        "logg": logg,
                        "feh": feh,
                        "alpha": 0.0,
                        "flux": np.full(len(_WAVE), teff),
                    }
                )
    return pd.DataFrame(rows)


def _plot_stub(teff, logg, feh, drop_node=None):
    """A NextGenPlot with only what ``_interp_spectra`` reads populated."""
    obj = object.__new__(NextGenPlot)
    obj.sedmodel = "NextGen"
    obj.df_spec = _spectra_frame(drop_node=drop_node)
    obj.df_wave = pd.DataFrame({"wavelength_angstrom": _WAVE})
    obj.grid_axes = {"teff", "logg", "feh"}
    obj.nstars = 1
    obj.ndraws = 1
    obj.draws = [
        {"star.teffsed": np.array([teff]), "star.feh": np.array([feh])}
    ]
    obj.logg_vals_draws = np.array([[logg]])
    return obj


def test_on_grid_draw_interpolates_between_the_bracketing_nodes():
    """
    Given a draw inside the spectra grid,
    When _interp_spectra runs,
    Then the flux is the linear interpolation of the bracketing nodes.

    The control for the two regressions below: the ordinary path must not
    move.
    """
    # ARRANGE
    plot = _plot_stub(teff=3250.0, logg=4.5, feh=0.25)

    # ACT
    plot._interp_spectra()

    # ASSERT -- flux == teff at every node, so the answer is the teff itself
    np.testing.assert_allclose(
        plot.model_spectrum_flux_draws[0, 0], np.full(len(_WAVE), 3250.0)
    )


def test_off_grid_draw_extrapolates_and_says_so_instead_of_raising():
    """
    Given a draw whose logg is above the whole spectra grid (the derived
      loggsed carries only a SOFT barrier, so this is a legal draw),
    When _interp_spectra runs,
    Then it returns a finite extrapolated spectrum and warns that it did.

    Regression: scipy's default bounds_error=True raised a ValueError here,
    inside SED.plot, inside run.py's unguarded pre-sampling plot loop -- so
    one off-grid start killed the fit before it began.
    """
    # ARRANGE -- logg 6.0 is a full cell above the grid's 5.0 ceiling
    plot = _plot_stub(teff=3500.0, logg=6.0, feh=0.25)

    # ACT
    with pytest.warns(UserWarning, match="logg=6"):
        plot._interp_spectra()

    # ASSERT
    flux = plot.model_spectrum_flux_draws[0, 0]
    assert np.all(np.isfinite(flux))
    np.testing.assert_allclose(flux, np.full(len(_WAVE), 3500.0))


def test_a_bracketing_node_missing_from_the_grid_raises():
    """
    Given a draw one of whose eight bracketing nodes is absent from the
      spectra grid (the real NextGen table holds 12796 of 23716 nodes),
    When _interp_spectra runs,
    Then it raises, naming the missing node.

    Regression: the missing corner kept its initialized 0.0 flux behind a
    warnings.warn, so the figure was interpolated against a fabricated zero
    -- a wrong published spectrum rather than a failure anyone could see.
    """
    # ARRANGE
    plot = _plot_stub(
        teff=3500.0, logg=4.5, feh=0.25, drop_node=(4000.0, 5.0, 0.5)
    )

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="no model spectrum at"):
        plot._interp_spectra()
