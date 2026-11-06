"""
The Claret limb-darkening tie's table side: which table a band reads, and
the one interpolation the model and the start-value seed share.

``Band`` owns the tie (``star_constrains_ld``); this module only knows
about tables. The tables themselves -- where they live, what they are
called, how they are read -- are ``models/Claret/claret_tables.py``'s. The
design and its rulings are in components/sed/sed.md ("The Claret
limb-darkening tie").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytensor.tensor as pt

from exozippy.components.sed.bc_grid import (
    AMBIGUOUS_FILTER_ALIASES,
    RegularGridInterpolator,
    _load_alias_table,
)
from exozippy.models.Claret.claret_tables import AXIS_ORDER, ClaretTable

# The alias table's marker for "no Claret table for this filter".
_UNSUPPORTED = "Unsupported"


def claret_band_lookup(filter_label: str) -> tuple[str | None, str | None]:
    """``(Claret table name, None)`` for a band's ``filter:`` label, or
    ``(None, why there is none)``.

    ``bc_grid.resolve_filter_name`` is deliberately not used here. It hands
    back the label unchanged for an unknown or ``Unsupported`` filter, which
    is right for the BC grid (the label may already be a canonical column
    name) and wrong for this tie: a Gaia G band would then look for a
    "Gaia.G" Claret table instead of being told there is none.  Returning the
    reason rather than raising lets Band decide: an explicit
    ``star_constrains_ld: true`` raises with it, the default skips the band
    with it.  A missing alias table is a broken install and always raises.
    """
    alias_df = _load_alias_table()
    if alias_df is None:
        raise RuntimeError(
            f"the filter alias table (filters/filternames.txt) is missing, so "
            f"filter {filter_label!r} cannot be mapped to a Claret table."
        )
    # A bare "I"/"R" names two rows; the project convention picks one
    # (AMBIGUOUS_FILTER_ALIASES), exactly as for the BC grid.
    disambiguated = AMBIGUOUS_FILTER_ALIASES.get(filter_label)
    if disambiguated is not None:
        rows = alias_df[alias_df["MIST"] == disambiguated]
    else:
        rows = alias_df[alias_df.eq(filter_label).any(axis=1)]
    if rows.empty:
        return None, (
            f"filter {filter_label!r} is not in the filter alias table "
            f"(filters/filternames.txt), so there is no Claret "
            f"limb-darkening table for it (spell the filter as the alias "
            f"table does if it should have one)"
        )
    names = {str(v) for v in rows["Claret"] if not pd.isna(v)}
    if len(names) > 1:
        return None, (
            f"filter {filter_label!r} names alias-table rows with different "
            f"Claret tables {sorted(names)} (write the filter out, its full "
            f"SVO id, to say which)"
        )
    if not names or names == {_UNSUPPORTED}:
        return None, (
            f"filter {filter_label!r} has no Claret limb-darkening table "
            f"(the Claret & Bloemen 2011 and Claret 2017 bands are U B V R I "
            f"J H K, SDSS u'g'r'i'z', Kepler, CoRoT, TESS, Spitzer IRAC and "
            f"Stromgren uvby)"
        )
    return names.pop(), None


# Effective wavelength (Angstrom, SVO WavelengthEff) of the filter each
# Claret table describes, through the alias table's rows for it: Cousins R/I
# and Johnson U/B/V for the Johnson-Cousins set, 2MASS for J H K, SDSS for
# Sloan, IRAC for Spitzer.  Used ONLY to name the nearest table in the error
# for a band with no table of its own (``nearest_claret_band``) -- a hint for
# the user's ``claret_filter:``, never applied by the code.  Hard-coded so
# the error never needs the network; CoRoT is absent because the alias table
# has no CoRoT row to measure.
CLARET_LAMBDA_EFF = {
    "U": 3551,
    "B": 4370,
    "V": 5468,
    "R": 6357,
    "I": 7829,
    "J": 12350,
    "H": 16620,
    "K": 21590,
    "Sloanu": 3608,
    "Sloang": 4672,
    "Sloanr": 6141,
    "Sloani": 7458,
    "Sloanz": 8923,
    "Kepler": 5978,
    "TESS": 7453,
    "Spit36": 35075,
    "Spit45": 44366,
    "Spit58": 56281,
    "Spit80": 75891,
    "Stromu": 3447,
    "Stromv": 4111,
    "Stromb": 4663,
    "Stromy": 5472,
}


def nearest_claret_band(filter_label: str, candidates) -> str:
    """A sentence naming the Claret table nearest ``filter_label`` in
    effective wavelength, among ``candidates`` (the tables that exist for
    the band's law) -- for the error that asks the user to choose.

    The filter's own effective wavelength comes from its SVO profile
    (``filters.filter.Filter``, shipped or cached, downloaded otherwise).
    This builds a MESSAGE: when the profile cannot be had, the sentence
    says why instead of naming a band, and the error it goes into is
    raised either way.
    """
    from exozippy.components.sed.bc_grid import resolve_filter_name
    from exozippy.filters.filter import Filter

    svo = resolve_filter_name(filter_label, _load_alias_table(), alias="SVO")
    if "/" not in svo:
        return (
            f"filter {filter_label!r} has no SVO identity, so no nearest "
            f"Claret band can be named"
        )
    try:
        lam = float(Filter(svo).WavelengthEff)
    except Exception as exc:  # network / SVO: the message says so
        return (
            f"the nearest Claret band could not be found: the profile of "
            f"{svo} is unavailable ({type(exc).__name__}: {exc})"
        )
    pool = {b: w for b, w in CLARET_LAMBDA_EFF.items() if b in candidates}
    best = min(pool, key=lambda b: abs(pool[b] - lam))
    return (
        f"the nearest Claret band in effective wavelength is {best} "
        f"({pool[best]:.0f} A, against {lam:.0f} A for {svo})"
    )


def claret_band_name(filter_label: str, where: str) -> str:
    """The Claret table name of a band's ``filter:`` label. Strict: every
    way of not having one RAISES, naming ``where``."""
    name, problem = claret_band_lookup(filter_label)
    if name is None:
        raise ValueError(
            f"{where}: {problem}. Set star_constrains_ld: false on this band "
            f"to fit its limb darkening from the data alone."
        )
    return name


def interpolate(
    table: ClaretTable, teff: float, logg: float, feh: float
) -> np.ndarray:
    """Coefficients at one point, in ``table.columns`` order -- numpy.

    The same trilinear interpolation as ``interpolate_node`` (bc_grid's
    RegularGridInterpolator), including its linear extrapolation off the edge
    cell, so a start value seeded from this is what the model computes at
    that start. Used for start values only.
    """
    point = {"teff": teff, "logg": logg, "feh": feh}
    idx, frac = [], []
    for ax in AXIS_ORDER:
        grid = table.axes[ax]
        i = int(
            np.clip(np.searchsorted(grid, point[ax]) - 1, 0, len(grid) - 2)
        )
        idx.append(i)
        frac.append((point[ax] - grid[i]) / (grid[i + 1] - grid[i]))
    out = np.zeros(table.values.shape[-1])
    for corner in np.ndindex(2, 2, 2):
        w = 1.0
        for c, f in zip(corner, frac):
            w *= f if c else 1.0 - f
        out += (
            w
            * table.values[
                idx[0] + corner[0], idx[1] + corner[1], idx[2] + corner[2]
            ]
        )
    return out


def interpolate_node(table: ClaretTable, teff, logg, feh):
    """Coefficients as a PyTensor vector, in ``table.columns`` order.

    ``teff``/``logg``/``feh`` are scalar tensors (one star). Off the box the
    interpolator extrapolates linearly from the edge cell (``fill_value``
    None, the SED's ruling); the star is held inside it by the bounds Band
    injects, and on logg -- derived, so a soft barrier -- by a restoring
    force rather than a wall.
    """
    interp = RegularGridInterpolator(
        points=[table.axes[ax] for ax in AXIS_ORDER],
        values=table.values,
    )
    coords = pt.stack([teff, logg, feh])[None, :]
    return interp.evaluate(coords)[0]


def kipping_from_quadratic(u1: float, u2: float) -> tuple[float, float]:
    """(q1, q2) of a quadratic (u1, u2), Kipping (2013) eqs. 15-16.

    The inverse of band/physics.py's calc_u1/u2_from_kipping. Only defined
    inside the physical triangle (u1 + u2 > 0); a point outside it has no
    (q1, q2) in [0, 1]^2, and the caller decides what to do with it.
    """
    s = u1 + u2
    return s * s, u1 / (2.0 * s)
