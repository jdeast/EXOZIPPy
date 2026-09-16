"""MIST evolutionary-track grid loader and pytensor interpolator.

Mirrors ``components/sed/bc_grid.py`` -- another precomputed-grid
interpolation problem -- closely enough that the actual N-D linear
interpolator is imported from there rather than re-implemented (see
``RegularGridInterpolator``; it already tolerates unevenly spaced axes,
which the MIST mass axis is).

Grid layout
-----------
``models/MIST/{model}/EEPs/afe_{token}_vvcrit{vvcrit}.grid.parquet`` holds
one row per (mass, initfeh, EEP) evolutionary-track point.  It is ~128 MB,
git-ignored, and **not** shipped in the package: :func:`load_mist_grid`
fetches it from Zenodo on first use through
``models/MIST/eep_grid.ensure_eep_grid``, which is the ONE reader of the
published grid (size- and md5-pinned; see ``utilities/zenodo.py`` for the
verify-and-cache mechanics, including the machine-level cache shared across
worktrees).  Building one locally instead needs ~186 GB of raw MIST tarballs
and hours of processing -- see ``models/MIST/README.md``.  Columns used here:

    mass, EEP, initfeh   -- the three grid axes (see below for units)
    feh_mist              -- present-day surface [Fe/H] (dex)
    radius_mist            -- stellar radius (solRad)
    teff_mist               -- effective temperature (K)
    age_mist                 -- age (yr)
    dEEP_dage                 -- d(EEP)/d(age[yr]), precomputed on the track
    here_be_dragons             -- 0 off-track; on-track it counts how many
                                    points past the first unreliable one this
                                    row is

The stellar-evolution component queries this grid at (log10(mass), initfeh,
EEP) -- log10(mass) because that is the coordinate the star component
actually samples (``star.logmass``) -- and reads back (feh_mist, radius_mist,
teff_mist, age_mist, dEEP_dage, here_be_dragons) as one vector-valued
interpolation, exactly like the SED's (teff, logg, feh, av) -> BC lookup.

Why "trim to a complete grid" is needed
----------------------------------------
The packaged grid is NOT a full (mass x initfeh) rectangle: MIST did not
compute every high-mass/very-metal-poor combination (see
``MissingGridPoints/missing_grid_points_afe_p0_vvcrit0.0.csv``).  Every track
that DOES exist is complete over the full EEP range (1-807; MIST considers
EEP 808+, where low-mass stars enter the TPAGB and mass loss invalidates the
"current mass == initial mass" assumption, out of scope -- see
:data:`MAX_VALID_EEP`), so the holes are whole missing (mass, initfeh)
tracks, not scattered NaNs within a track.
:class:`RegularGridInterpolator` needs a fully populated rectangular grid
(exactly like ``sed.bc_grid.build_bc_grid``'s hard "raise on any NaN" -- see
its module docstring), so :func:`trim_to_complete_grid` drops whole
rows/columns of the (mass, initfeh) plane until no holes remain.

The trim is a **single-axis cut**, not an iterative per-edge shave: it drops
every value of the chosen axis that appears in any missing pair.  For the
shipped afe_p0/vvcrit0.0 grid, ``cut="feh"`` (the default) removes exactly
the four lowest initfeh grid points (-4.0, -3.5, -3.0, -2.75) and none of the
mass axis: all 53 missing combinations lie at mass in [10, 38] solMass *and*
initfeh <= -2.75, so a handful of column drops clears every hole while
keeping the full 0.1-300 solMass mass range.  Note the flip side of a
single-axis cut -- ONE scattered hole at, say, initfeh = 0.0 would drop the
entire solar-metallicity column.  That is fine for the shipped grid, where
the holes are an edge block, and a trap for a regenerated one; the trim is
discovered at load time rather than hardcoded and logs exactly what it
dropped, and the component turns the surviving axis extents into parameter
bounds, so a user is told about the reduced range rather than silently
extrapolated past it.

``feh_mist == 30.0`` is DATA, not a sentinel -- do not "repair" it
-----------------------------------------------------------------
About 1.5% of rows carry exactly ``feh_mist = 30.0``.  That is not an error
marker and **must not be filtered, clamped or held across**.  [Fe/H] is
``log10(Z/X) - log10(Z/X)_sun``, so it diverges as the surface hydrogen mass
fraction X goes to zero; 30.0 is the stand-in the grid generator substitutes
to avoid the divide-by-zero.  It means "this photosphere has essentially no
hydrogen left", which really happens, and only in the highest-mass stars.
Those rows carry ``here_be_dragons > 0`` because a track is less trustworthy
in that regime, not because the number is wrong.

So a sampled star at [Fe/H] ~ 0 genuinely does not match such a track point,
and the enormous feh penalty it earns there is the correct answer.  An
earlier version of this module forward-filled the last "normal" ``feh_mist``
across those rows; that fabricated a solar-metallicity prediction for a
hydrogen-exhausted photosphere and is exactly the wrong fix.

One real limitation to know about, and it belongs to the GRID rather than to
this reader: within the single cell where the tabulated value steps from an
ordinary [Fe/H] to 30.0, linear interpolation returns a large intermediate
number (~15 at the midpoint) that the underlying track does not support.
Interpolating in a smooth variable instead -- surface X, or an uncapped
log10(Z/X) -- would fix that at the source.  Nothing is done about it here,
because every read-time workaround is a worse lie than the ramp.

Unresolved ages, and the one thing this module DOES rewrite
-----------------------------------------------------------
Where a star evolves fast enough that the stellar model cannot resolve the
timestep, the tabulated ages come out equal; the grid generator adds one year
to each successive duplicate to keep the column strictly increasing.  Two
things follow, and only the second is a problem this module can fix.

First, ``dEEP_dage`` in that stretch is meaningless -- it is 1/(1 yr) where
the true step is unresolvably small.  It is NOT repaired here.  It is also
genuinely huge in places that are perfectly fine (a 110 solMass star's EEP 1
is ~4e-3 yr old, so ~3e3 EEP/yr is real), so there is no threshold that
separates the two.  The consumer -- the component's EEP -> age Jacobian
potential -- clips ``|dEEP/dage|`` to a physical window instead, which
handles the huge values and the near-zero crossings between them in one
place.

Second, when such a run ends just BEFORE the end of a track, the +1 yr
bumping lifts those ages above the genuinely-later unique ages that follow,
so the column steps backwards on re-entry -- and the generator's
``here_be_dragons`` counter skips those unique rows, so the flag drops back
to 0 across exactly the worst part of the track.  Track (90 solMass, -2.25
dex) starts counting at EEP 615, falls to 0 for EEP 618-631 (whose
``dEEP_dage`` reaches 2e9), then resumes at 4.  Nothing was discouraging a
fit from sitting there.

:func:`_flag_from_first_unresolved_age` closes that gap: from a track's first
unresolved step to its end, ``here_be_dragons`` becomes a plain 1, 2, 3, ...
count.  Past that point the model is not tracking time properly whether or
not an individual row happens to be a duplicate, so flagging a few extra rows
costs nothing real.  It is applied as an element-wise maximum against what
the generator wrote, so it can only raise the flag and becomes a no-op the
day the generator does this itself -- which is where the fix belongs.
:func:`_unflagged_nonmonotone_tracks` then warns about any age decrease left
OUTSIDE a flagged tail, which should never happen and would mean something
this module does not understand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from ..sed.bc_grid import DEFAULT_MODEL_ROOT, RegularGridInterpolator

logger = logging.getLogger(__name__)

DEFAULT_MIST_MODEL_ROOT = DEFAULT_MODEL_ROOT / "MIST"

# Column order of the interpolator's trailing (output) axis. age_mist is
# converted yr -> Gyr before stacking (see _assemble_grid) to match
# star.age's unit; the rest are used exactly as tabulated.
OUTPUT_COLUMNS = (
    "feh_mist",
    "radius_mist",
    "teff_mist",
    "age_mist_gyr",
    "dEEP_dage",
    "here_be_dragons",
)

# Index of each output within the interpolator's trailing axis, for callers
# slicing the evaluated (ntest, n_outputs) result.
OUTPUT_INDEX = {name: i for i, name in enumerate(OUTPUT_COLUMNS)}

_YR_PER_GYR = 1.0e9

# The offset the grid generator adds to each successive duplicate age so a
# run of unresolvable timesteps stays strictly increasing. See
# _first_unresolved_age_index, which detects the run by this exact step.
_AGE_BUMP_YEARS = 1.0

# MIST considers EEP 808+ (low-mass TPAGB, where mass loss invalidates the
# initial-mass assumption every quantity here is computed under) out of
# scope. Every track in the shipped grid stops at 807; a regenerated grid
# that ran further is truncated here rather than silently extending the
# EEP axis into a regime the component's model does not cover.
MAX_VALID_EEP = 807

# The [Fe/H] the grid generator substitutes when the surface hydrogen mass
# fraction underflows, so log10(Z/X) does not divide by zero. A real value
# meaning "no hydrogen left in the photosphere", NOT an error marker -- see
# the module docstring. Named here only so tests can pin that it survives
# the load unchanged.
FEH_HYDROGEN_EXHAUSTED = 30.0

# Columns whose per-track values must stay contiguous for a `reshape` into
# the dense (mass, initfeh, EEP) array to be correct.
_AXIS_COLUMNS = ("mass", "initfeh", "EEP")


def _model_version(model: str) -> str:
    """The bare version string of a MIST model directory name.

    ``"MISTv2.5" -> "2.5"``.  The [alpha/Fe] filename token is spelled
    differently between MIST releases (see :func:`_afe_token_v12`), so the
    version has to come from somewhere; the model directory name is the only
    thing the caller supplies that carries it.
    """
    return str(model).removeprefix("MIST").lstrip("v")


def _afe_token_v12(alpha: float) -> str:
    """[alpha/Fe] value -> the MISTv1.2 filename's "afe_..." token.

    v1.2 spells it "p0.0", "m0.2", "p0.4"; v2.5 spells the same values "p0",
    "m2", "p4".  ONLY the v1.2 spelling lives here -- v2.5's comes from
    ``eep_grid.eep_grid_filename``, which is kept byte-identical to the
    authoring script's own ``_generate_alpha_vvcrit_filename_parts`` so a
    grid built locally and one fetched from Zenodo are interchangeable.  A
    third copy of that rule is a third thing to keep in agreement.
    """
    sign = "m" if alpha < 0 else "p"
    return f"{sign}{abs(alpha):0.1f}"


def _vvcrit_token(vvcrit: float) -> str:
    """v/vcrit value -> the grid filename's "vvcrit_..." token.

    E.g. 0.0 -> "vvcrit0.0", 0.4 -> "vvcrit0.4".  Matches the naming
    convention in the packaged MIST grid files; identical between MISTv1.2
    and MISTv2.5.
    """
    return f"vvcrit{vvcrit:0.1f}"


def _published_filename(alpha: float, vvcrit: float) -> str:
    """The MISTv2.5 grid filename, from the module that publishes it."""
    from ...models.MIST.eep_grid import eep_grid_filename

    return eep_grid_filename(alpha=alpha, vvcrit=vvcrit)


def grid_path(
    model: str = "MISTv2.5",
    alpha: float = 0.0,
    vvcrit: float = 0.0,
    model_root: Path | str = DEFAULT_MIST_MODEL_ROOT,
) -> Path:
    """Path to the grid.parquet for one (model, alpha, vvcrit).

    Where the file WOULD be; it may not exist yet (see
    :func:`load_mist_grid`, which fetches the published one on demand).

    ``model_root=None`` means the default, not a missing argument: a YAML
    ``model_root:`` with no value parses to None, and callers pass a config
    block's keys straight through (see `_needs_absent_mist_grid` in
    tests/test_examples_prepare.py), so the None arrives here rather than
    being defaulted away.
    """
    if model_root is None:
        model_root = DEFAULT_MIST_MODEL_ROOT
    model_root = Path(model_root)
    if _model_version(model) == "2.5":
        fname = _published_filename(alpha, vvcrit)
    else:
        token = _afe_token_v12(alpha)
        fname = f"afe_{token}_{_vvcrit_token(vvcrit)}.grid.parquet"
    return model_root / model / "EEPs" / fname


# ---------------------------------------------------------------------
# Trimming to a complete (mass, initfeh) rectangle
# ---------------------------------------------------------------------


def _missing_pairs(
    df: pd.DataFrame, mass_col: str, feh_col: str
) -> pd.DataFrame:
    """The (mass, initfeh) pairs in ``df``'s rectangular hull that are absent.

    The one implementation of "which holes are there", shared by
    :func:`_check_complete_grid` and :func:`trim_to_complete_grid` so the
    check and the cut cannot disagree about what a hole is.

    ``df`` carries one row per (mass, initfeh, EEP), i.e. every pair appears
    ``n_eep`` times.
    """
    mass_vals = np.sort(df[mass_col].unique())
    feh_vals = np.sort(df[feh_col].unique())

    expected = pd.MultiIndex.from_product(
        [mass_vals, feh_vals], names=[mass_col, feh_col]
    )
    present = pd.MultiIndex.from_arrays(
        [df[mass_col], df[feh_col]], names=expected.names
    )
    return expected.difference(present).to_frame(index=False)


def _check_complete_grid(
    df: pd.DataFrame, mass_col: str = "mass", feh_col: str = "initfeh"
) -> bool:
    """True when every (mass, initfeh) pair in ``df``'s hull is present."""
    return _missing_pairs(df, mass_col, feh_col).empty


def trim_to_complete_grid(
    df: pd.DataFrame,
    mass_col: str = "mass",
    feh_col: str = "initfeh",
    cut: Literal["greedy", "feh", "mass"] = "feh",
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Drop whole rows/columns until (mass, initfeh) is a full rectangle.

    ``cut`` selects the axis to cut along: ``"feh"`` (the default, and what
    the shipped grid wants -- see the module docstring), ``"mass"``, or
    ``"greedy"`` for whichever axis has fewer distinct values among the
    missing pairs.  Every value of that axis appearing in any missing pair is
    dropped, which is enough by construction: after the cut no surviving
    (mass, initfeh) pair can be one of the holes.

    Returns the trimmed frame and ``{"dropped_mass": [...],
    "dropped_initfeh": [...]}`` for the caller to log.  Raises ValueError
    if the result is somehow still incomplete.
    """
    missing = _missing_pairs(df, mass_col, feh_col)
    if missing.empty:
        return df.copy(), {"dropped_mass": [], "dropped_initfeh": []}

    if cut == "greedy":
        cut = (
            "feh"
            if missing[feh_col].nunique() <= missing[mass_col].nunique()
            else "mass"
        )

    if cut == "feh":
        drop_col, dropped_feh, dropped_mass = feh_col, None, []
    elif cut == "mass":
        drop_col, dropped_feh, dropped_mass = mass_col, [], None
    else:
        raise ValueError(
            f"Invalid cut option: {cut!r}. Must be one of 'greedy', 'feh' or "
            f"'mass'."
        )

    dropped_values = missing[drop_col].unique()
    df_trimmed = df[~df[drop_col].isin(dropped_values)].copy()
    if drop_col == feh_col:
        dropped_feh = dropped_values.tolist()
    else:
        dropped_mass = dropped_values.tolist()

    if df_trimmed.empty:
        raise ValueError(
            f"Trimming the MIST grid along '{drop_col}' removed every track: "
            f"the missing (mass, initfeh) pairs cover the whole {drop_col} "
            f"axis. Try cut='{'mass' if drop_col == feh_col else 'feh'}'."
        )

    if not _check_complete_grid(df_trimmed, mass_col, feh_col):
        raise ValueError(
            "Trimming did not result in a complete (mass, initfeh) rectangle."
        )

    return df_trimmed, {
        "dropped_mass": sorted(dropped_mass),
        "dropped_initfeh": sorted(dropped_feh),
    }


def _axis_weights(axis: np.ndarray, value: float) -> Tuple[int, float]:
    """Bracketing index and normalized distance of ``value`` along ``axis``.

    The same rule ``RegularGridInterpolator`` uses (``searchsorted`` minus
    one, clipped to the last cell), with ONE deliberate difference: the
    coordinate is clipped into the axis range first, so a query outside the
    grid is pinned to the edge rather than linearly extrapolated.  That is
    right for the callers here -- a seed search wants the nearest point the
    models actually cover, and an extrapolated start value is meaningless --
    and wrong for the likelihood, where extrapolating is what the component's
    grid bounds exist to prevent in the first place.
    """
    lo, hi = float(axis[0]), float(axis[-1])
    value = min(max(float(value), lo), hi)
    i = int(np.searchsorted(axis, value) - 1)
    i = max(0, min(i, len(axis) - 2))
    span = float(axis[i + 1] - axis[i])
    return i, (value - float(axis[i])) / span if span else 0.0


def interpolate_track(
    grid: Dict, logmass: float, initfeh: float
) -> np.ndarray:
    """One ``(n_eep, n_outputs)`` track, bilinear in (logmass, initfeh).

    The evolutionary tracks are not interchangeable between neighbouring
    grid points -- a 0.9 and a 1.0 solMass track differ everywhere, most of
    all in age -- so snapping a query to the nearest tabulated track (which
    is what the component's EEP seed search used to do) can start a fit on a
    track up to half a grid cell away in BOTH axes.  This blends the four
    surrounding tracks with the same weights the likelihood's interpolator
    uses, so the seed search and the fit see the same surface.

    The EEP axis is deliberately NOT interpolated: it is the axis being
    searched over, and every track shares it exactly.
    """
    mi, wm = _axis_weights(grid["logmass_pts"], logmass)
    fi, wf = _axis_weights(grid["initfeh_pts"], initfeh)
    values = grid["values"]
    return (
        values[mi, fi] * ((1.0 - wm) * (1.0 - wf))
        + values[mi + 1, fi] * (wm * (1.0 - wf))
        + values[mi, fi + 1] * ((1.0 - wm) * wf)
        + values[mi + 1, fi + 1] * (wm * wf)
    )


# ---------------------------------------------------------------------
# Grid-health diagnostics (reported, never repaired -- see the module
# docstring on why feh_mist == 30.0 is left exactly as tabulated)
# ---------------------------------------------------------------------


def _first_unresolved_age_index(d_age: np.ndarray) -> int:
    """Index of a track's first age step the stellar model could not resolve.

    ``d_age`` is the track's consecutive ``age_mist`` differences, in years,
    with ``d_age[0]`` NaN.  Two signatures, and the first one is why this is
    not simply a monotonicity test:

    * **exactly +1 yr.**  Where evolution is fast enough that the model
      cannot resolve the timestep, the tabulated ages come out equal, and the
      grid generator adds one year to each subsequent duplicate to keep them
      distinct.  So a resolved-away run of duplicates reaches us as a run of
      steps of exactly 1.0 yr, not as duplicates.
    * **<= 0.**  The consequence of the above, when a run of duplicates ends
      just BEFORE the end of a track: the +1 yr bumping lifts those ages
      above the genuinely-later unique ages that follow, so the column steps
      backwards on re-entry.

    The exact-1.0 test is safe against real fast evolution, which is dense in
    (0, 1) yr but essentially never lands on 1.0: in the shipped grid the
    two populations are 123518 steps at exactly 1.0 (EEP >= 203, all in the
    already-suspect tail) versus 69560 in (0, 1) (from EEP 2, i.e. the
    ordinary pre-main-sequence).
    """
    unresolved = (d_age == _AGE_BUMP_YEARS) | (d_age <= 0.0)
    hits = np.flatnonzero(unresolved)
    return int(hits[0]) if hits.size else -1


def _flag_from_first_unresolved_age(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Extend ``here_be_dragons`` over a track's whole unresolved-age tail.

    The grid generator flags the duplicated rows themselves, but its counter
    SKIPS the unique ages interleaved among them, so a track can come back to
    ``here_be_dragons == 0`` for a stretch and then resume counting.  Track
    (90 solMass, -2.25 dex) does exactly that: it starts counting at EEP 615,
    drops to 0 for EEP 618-631 -- the rows whose ages sit BELOW the bumped
    ones, carrying ``dEEP_dage`` up to 2e9 -- then resumes at 4.  Those unflagged
    rows are the worst in the track and nothing was discouraging the sampler
    from sitting on them.

    So: from the first unresolved step to the end of the track, the flag
    becomes a plain 1, 2, 3, ... count.  Past that point the model is not
    tracking time properly whether or not an individual row happens to be a
    duplicate, so flagging a few extra rows costs nothing real.  Applied as
    an element-wise ``maximum`` against what the generator wrote, so it can
    only ever raise the flag -- and so it becomes a no-op the day the
    generator does this itself.

    ``df`` must be sorted by (mass, initfeh, EEP).  Returns the frame and the
    number of rows whose flag was raised.
    """
    df = df.copy()
    grouped = df.groupby(["mass", "initfeh"], sort=False)
    d_age = grouped["age_mist"].diff().to_numpy()

    flags = df["here_be_dragons"].to_numpy(dtype=float).copy()
    raised = 0
    for _, index in grouped.indices.items():
        start = _first_unresolved_age_index(d_age[index])
        if start < 0:
            continue
        tail = index[start:]
        extended = np.arange(1, tail.size + 1, dtype=float)
        raised += int((extended > flags[tail]).sum())
        flags[tail] = np.maximum(flags[tail], extended)

    df["here_be_dragons"] = flags
    return df, raised


def _unflagged_nonmonotone_tracks(
    df: pd.DataFrame,
) -> List[Tuple[float, float]]:
    """Tracks whose age steps backwards where nothing is flagged.

    After :func:`_flag_from_first_unresolved_age` this must be empty: every
    decrease is a consequence of the +1 yr bumping and therefore sits inside
    an unresolved-age tail.  A non-empty result means a decrease arose some
    other way and the grid needs looking at -- which is worth a warning,
    unlike the decreases we now understand and flag.
    """
    d_age = df.groupby(["mass", "initfeh"], sort=False)["age_mist"].diff()
    suspect = (d_age < 0) & (df["here_be_dragons"] <= 0)
    bad = df.loc[suspect, ["mass", "initfeh"]].drop_duplicates()
    return [(float(m), float(f)) for m, f in bad.itertuples(index=False)]


# ---------------------------------------------------------------------
# Assembling the dense interpolation array
# ---------------------------------------------------------------------


def _assemble_grid(df: pd.DataFrame) -> Dict:
    """Pivot a complete (mass, initfeh, EEP) track table into a dense array.

    Requires ``df`` to already be a complete rectangle in (mass, initfeh)
    (see :func:`trim_to_complete_grid`) with every track spanning the exact
    same EEP array -- both are verified here rather than assumed, so a
    malformed/regenerated grid fails loudly instead of silently
    mis-indexing.
    """
    df = df[df["EEP"] <= MAX_VALID_EEP]

    mass_pts = np.sort(df["mass"].unique())
    feh_pts = np.sort(df["initfeh"].unique())
    eep_pts = np.sort(df["EEP"].unique())
    n_mass, n_feh, n_eep = len(mass_pts), len(feh_pts), len(eep_pts)

    expected = n_mass * n_feh * n_eep
    if len(df) != expected:
        raise ValueError(
            f"MIST grid is not a complete (mass, initfeh, EEP) rectangle "
            f"after trimming: expected {expected} rows "
            f"({n_mass} mass x {n_feh} initfeh x {n_eep} EEP), got "
            f"{len(df)}. Every remaining track must share the identical "
            f"EEP array."
        )

    df_sorted = df.sort_values(list(_AXIS_COLUMNS)).reset_index(drop=True)

    # Sanity check the sort actually reproduces the (mass, initfeh, EEP)
    # nesting the row-major reshape below assumes, in one cheap pass rather
    # than trusting the sort silently.
    if not np.array_equal(
        df_sorted["EEP"].to_numpy().reshape(n_mass * n_feh, n_eep),
        np.broadcast_to(eep_pts, (n_mass * n_feh, n_eep)),
    ):
        raise ValueError(
            "MIST grid: at least one track's EEP array differs from the "
            "others (order or values) after trimming -- cannot reshape "
            "into a dense (mass, initfeh, EEP) grid."
        )

    df_sorted, n_flagged = _flag_from_first_unresolved_age(df_sorted)
    nonmonotone = _unflagged_nonmonotone_tracks(df_sorted)

    df_sorted["age_mist_gyr"] = df_sorted["age_mist"] / _YR_PER_GYR

    # ~86 MB float64 for the shipped grid (171 x 13 x 807 x 6). It is baked
    # into the pytensor graph as one constant, shared by every component
    # instance through _GRID_CACHE -- think before adding a second grid.
    values = np.stack(
        [df_sorted[c].to_numpy(dtype=float) for c in OUTPUT_COLUMNS], axis=-1
    )
    values = values.reshape(n_mass, n_feh, n_eep, len(OUTPUT_COLUMNS))

    return {
        "logmass_pts": np.log10(mass_pts),
        "initfeh_pts": feh_pts,
        "eep_pts": eep_pts.astype(float),
        "values": values,
        "n_dragon_rows_added": n_flagged,
        "unflagged_nonmonotone_tracks": nonmonotone,
    }


# ---------------------------------------------------------------------
# Locating the parquet (fetching the published one if it is not here yet)
# ---------------------------------------------------------------------


def _resolve_grid_file(model, alpha, vvcrit, model_root) -> Path:
    """The grid parquet's local path, downloading the published one if needed.

    The grids are ~128 MB each and are neither checked into git nor shipped
    in the package, so "not on disk" is the NORMAL first-run state, not an
    error.  ``models/MIST/eep_grid`` is the one reader of the published
    grids: it fetches from Zenodo, verifies size and md5, and caches in
    place (and machine-wide, hard-linked across worktrees).  Nothing
    downloads at import -- this runs in the component's ``load_data``
    (stage 1a), on the first fit that actually asks for a track.

    The fetch is only offered for the DEFAULT model root.  An explicit
    ``model_root:`` means the caller is pointing at a tree of their own (a
    locally built grid, another MIST release, a test fixture), and silently
    populating the packaged tree instead would ignore what they asked for.
    """
    path = grid_path(
        model=model, alpha=alpha, vvcrit=vvcrit, model_root=model_root
    )
    if path.is_file():
        return path

    default_root = Path(model_root) == Path(DEFAULT_MIST_MODEL_ROOT)
    if default_root and _model_version(model) == "2.5":
        from ...models.MIST.eep_grid import ensure_eep_grid

        logger.info(
            f"MIST grid {path.name} is not present yet; fetching the "
            f"published copy (~128 MB, verified and cached -- see "
            f"utilities/zenodo.py). This happens once."
        )
        # Raises KeyError naming the published grids if this (alpha, vvcrit)
        # is not one of them, and RuntimeError if the download did not
        # verify. Both are better messages than anything this module could
        # write, so neither is caught.
        return ensure_eep_grid(alpha=alpha, vvcrit=vvcrit)

    raise FileNotFoundError(
        f"MIST evolutionary grid not found at {path}, and no published grid "
        f"is fetched for this request: model={model!r} "
        f"(only 'MISTv2.5' is published) with model_root={model_root}"
        + (
            "."
            if default_root
            else " (an explicit model_root is used as given, never "
            "auto-populated)."
        )
        + f" Build one from the raw MIST .eep files with the scripts in "
        f"{Path(model_root) / 'MIST'} -- see that directory's README.md."
    )


# ---------------------------------------------------------------------
# Public entry point (cached: the parquet is ~128 MB and every star
# instance of the component queries the same grid)
# ---------------------------------------------------------------------

_GRID_CACHE: Dict[tuple, Dict] = {}


def load_mist_grid(
    model: str = "MISTv2.5",
    alpha: float = 0.0,
    vvcrit: float = 0.0,
    model_root: Path | str = DEFAULT_MIST_MODEL_ROOT,
) -> Dict:
    """Load, trim and assemble the MIST grid for one (model, alpha, vvcrit).

    Returns a dict with keys:
        logmass_pts : (n_mass,)   log10(solMass), ascending
        initfeh_pts : (n_feh,)    dex, ascending
        eep_pts     : (n_eep,)    ascending (1..807 for the shipped grid)
        values      : (n_mass, n_feh, n_eep, len(OUTPUT_COLUMNS))
        interpolator: RegularGridInterpolator over (logmass, initfeh, EEP)
        dropped     : {"dropped_mass": [...], "dropped_initfeh": [...]}
                      from trim_to_complete_grid, for the caller to log.

    Cached at module level by (model, alpha, vvcrit, str(model_root)): the
    same grid is shared by every evolutionarymodel instance (one per
    constrained star) in a fit.

    The parquet is fetched from Zenodo on first use if it is not on disk yet
    -- see :func:`_resolve_grid_file`.
    """
    key = (model, float(alpha), float(vvcrit), str(model_root))
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    path = _resolve_grid_file(model, alpha, vvcrit, model_root)

    df = pd.read_parquet(path, engine="pyarrow")
    df_trimmed, dropped = trim_to_complete_grid(df)
    if dropped["dropped_mass"] or dropped["dropped_initfeh"]:
        logger.warning(
            f"MIST grid ({path.name}) is missing some (mass, initfeh) "
            f"tracks; trimmed to a complete rectangle by dropping "
            f"mass in {dropped['dropped_mass']} and initfeh in "
            f"{dropped['dropped_initfeh']}. The surviving axis extents "
            f"become the bounds on star.logmass/star.initfeh, so the fit is "
            f"kept inside the region the interpolator can evaluate. See "
            f"mist_grid.py's module docstring."
        )

    assembled = _assemble_grid(df_trimmed)

    if assembled["n_dragon_rows_added"]:
        logger.info(
            f"MIST grid ({path.name}): raised here_be_dragons on "
            f"{assembled['n_dragon_rows_added']} rows so every track's "
            f"unresolved-age tail is flagged continuously. The generator "
            f"flags the duplicated rows but skips the unique ages "
            f"interleaved among them, leaving unflagged gaps that carry the "
            f"worst dEEP_dage in the track. See mist_grid.py's module "
            f"docstring."
        )
    if assembled["unflagged_nonmonotone_tracks"]:
        tracks = assembled["unflagged_nonmonotone_tracks"]
        logger.warning(
            f"MIST grid ({path.name}): {len(tracks)} track(s) step BACKWARDS "
            f"in age outside any unresolved-age tail. Every decrease this "
            f"module understands is a consequence of the generator's +1 yr "
            f"duplicate bumping and is flagged; these are not, so the grid "
            f"needs looking at. (mass, initfeh): {tracks[:8]}"
            + (" ..." if len(tracks) > 8 else "")
        )

    interpolator = RegularGridInterpolator(
        points=[
            assembled["logmass_pts"],
            assembled["initfeh_pts"],
            assembled["eep_pts"],
        ],
        values=assembled["values"],
    )

    result = {**assembled, "interpolator": interpolator, "dropped": dropped}
    _GRID_CACHE[key] = result
    return result
