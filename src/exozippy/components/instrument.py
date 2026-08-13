"""
Shared base class for data (instrument) components.

rvinstrument, transit, mulensinstrument and astrometryinstrument are four
near-parallel data components that each re-implement the same scaffolding:
per-observation instrument maps, a per-instrument noise term, a jitter floor
derived from the smallest error bar, optional detrending against extra data
columns, and per-instrument plot styling.  ``Instrument`` extracts that
common machinery so it is written and tested once; the physics (RV curve vs
transit light curve vs magnification vs astrometric position) stays in each
child.

Reading a file is ``_read_data``; turning the per-file blocks into this
component's concatenated arrays is ``ConcatenatedData`` (below), the shared
template the three single-observable children drive from their own
``load_data``.  ``astrometryinstrument`` is deliberately not one of them: it
models TWO observables per epoch (dE/dN or sep/PA) in different units, so it
keeps one dict per file rather than concatenating -- the same asymmetry that
makes it set ``supports_gp = False`` and ``supports_robust_likelihood =
False``.

This class is deliberately NOT a discoverable component: it declares none of
``Component``'s abstract methods (``prefix``, ``register_parameters``,
``build_likelihood``), so it stays abstract and ``factory.discover_components``
skips it (see the ``inspect.isabstract`` guard there).

Two noise parameterizations are supported through the ``noise_model`` class
attribute so the base does not hardcode one:

  ``"jitter_variance"`` (default; rv/transit/astrometry)
      An additive per-instrument variance ``jitter_variance`` plus a reported
      ``jitter``; total sigma is ``sqrt(err**2 + jitter_variance[inst])``.
  ``"err_scale"`` (mulensinstrument)
      A multiplicative per-instrument error scale ``err_scale``; total sigma
      is ``err * err_scale[inst]``.

Bands are optional: this base never references a ``band:`` block, so children
that need one (transit, mulens, astrometry) resolve it themselves and children
that do not (rv) are unaffected.

Gaussian processes are optional too, and off unless asked for.  A data file
carrying a ``gp:`` key (``rotation``, ``sho``, or both -- see components/gp.py)
gets a celerite2 kernel instead of independent Gaussian errors; every other
file in the same component keeps the plain ``pm.Normal``.  Children opt in by
calling ``_prepare_gp`` at the end of ``load_data``, ``_register_gp`` in
``register_parameters``, and ``add_observation_likelihood`` in place of their
final ``pm.Normal``; a component that does none of these is unaffected, and a
component that does all three costs nothing extra when no file sets ``gp:``.

Robust observation likelihoods are optional in exactly the same shape.  A
data file carrying a ``likelihood:`` key (``hogg`` or ``studentt`` -- see
components/likelihood.py) swaps its plain ``pm.Normal`` for a marginalized
inlier/outlier mixture or a Student-t; every other file is untouched.  The
same three hooks carry it (``_prepare_robust``, ``_register_robust``, and the
shared ``add_observation_likelihood`` dispatcher), the parameters are
full-length pinned vectors like the GP hyperparameters, and a file cannot set
both ``gp:`` and ``likelihood:`` (celerite2's closed-form marginal is
Gaussian-only).  Off by default everywhere: with no ``likelihood:`` key the
model is byte-for-byte what it was before this feature existed.
"""

import logging

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import sympy as sp

from ..physics_registry import register_physics
from . import gp as gp_support
from . import likelihood as robust_support
from .component import Component

logger = logging.getLogger(__name__)

# Time-system vocabulary for the per-file time_scale/time_frame keys.
# Scales are astropy.time scale names ("ut" is accepted as an alias for
# ut1); frames name where the clock sits: jd = the observatory/geocenter,
# hjd = heliocenter, bjd = solar-system barycenter.  The astropy
# light_travel_time "kind" implementing each frame's correction is the
# mapped value (None = no light-travel correction).
_TIME_SCALES = ("utc", "tai", "tt", "tdb", "tcb", "tcg", "ut1")
_TIME_SCALE_ALIASES = {"ut": "ut1"}
_TIME_FRAMES = {"jd": None, "hjd": "heliocentric", "bjd": "barycentric"}
# Frame/scale conversion is only meaningful on absolute Julian Dates;
# anything below this is a truncated time (BJD-2450000, MJD, ...) that
# needs time_offset first.
_MIN_ABS_JD = 2_000_000.0

# Radicand floor for the reported jitter's square root, in the parameter's
# INTERNAL units.  Two knobs in one number: the reported jitter is quantized
# to sqrt(eps) = 1e-15 near zero (8e-12 m/s for rv, 1e-15 in relative flux for
# transit, 1e-15 mas for astrometry -- all far below any real error bar), and
# the slope is capped at 0.5/sqrt(eps).  Only |jitter_variance| < 1e-30 sees
# either, and at exactly zero the sign factor makes both the value and the
# gradient exactly zero.
_JITTER_EPS = 1e-30


@register_physics
def calc_jitter(jitter_variance):
    """Reported jitter: the SIGNED square root of ``jitter_variance``.

    ``sign(v) * sqrt(|v|)``, so the reported jitter stays in the data's own
    units, is monotonic in the sampled ``jitter_variance``, and has a finite
    gradient everywhere.

    ``jitter_variance`` is deliberately allowed to go negative -- down to
    ``Instrument._jitter_floor``'s validity limit, ``-0.95 * min(err)**2``,
    which is what keeps ``total_sigma``'s own ``sqrt`` real.  A negative
    jitter variance is a *result*, not a pathology: it says the quoted error
    bars are too large.  Clamping the report to zero there (what this did
    until 2026-08) cost two things: a Lucy-Sweeney-style upward bias on a
    marginally detected jitter, exactly as forcing e >= 0 biases
    eccentricity; and a zero-gradient plateau over the whole negative
    half-axis, so a user prior or link on ``jitter`` had nothing to push
    against over half of the parameter's legal range.

    The floor goes on the RADICAND, following ``calc_theta_E``: ``sqrt'(0)``
    is infinite, and clamping the result afterwards multiplies that infinity
    by ``pt.maximum``'s zero gradient to give NaN.  Flooring the argument
    instead keeps the derivative finite.  It has to be floored at all because
    ``jitter_variance``'s defaults.yaml initval is exactly 0.0, i.e. every
    fit starts on the cusp: the pre-fix ``pt.switch`` reported
    ``d jitter / d jitter_variance = inf`` there.

    DELIBERATE DEPARTURE FROM EXOFASTv2, which floors the jitter at zero.
    This is an upgrade, not a port bug -- do not "restore" the floor.  The
    argument is the one ``planet.mass`` in ``linear`` mode already makes (see
    the mass-parametrization section of CLAUDE.md): a positive-definite
    coordinate biases a marginal detection upward, because the half of the
    posterior that would have balanced it is folded onto the boundary.  Here
    the floor would also throw information away -- a negative jitter is the
    fit telling you the quoted error bars are too large, which is a result
    worth reporting.  And it would buy nothing in exchange: ``total_sigma``
    uses ``jitter_variance`` directly and NOTHING in any physics path consumes
    this function's output, so ``jitter`` is purely the reported form and a
    floor could only ever hide a negative variance the sampler had already
    visited.  The one cost of the sign is that a reader must know what it
    means, which the defaults.yaml ``description`` says and the ``latex``
    label carries: ``J``, not a ``sigma``, which would promise positivity.

    One function, three components (rv/transit/astrometry): it is pure
    algebra on the shared noise model this module owns, so it lives here
    rather than as three byte-identical copies under three registry names.
    Its sympy counterpart lives just below, for the same reason.
    """
    return pt.sign(jitter_variance) * pt.sqrt(
        pt.maximum(pt.abs(jitter_variance), _JITTER_EPS)
    )


# ---------------------------------------------------------------------------
# The symbolic counterpart of calc_jitter
# ---------------------------------------------------------------------------
# The relaxation engine needs the same relation in sympy so a user may seed
# EITHER side -- ``jitter`` in the data's own units (what anyone actually has a
# number for) or the sampled ``jitter_variance`` -- and get the other derived.
# It is the SIGNED square, the exact inverse of calc_jitter's signed square
# root, because ``jitter_variance`` is deliberately allowed to go negative down
# to ``Instrument._jitter_floor``.  Written as ``jitter**2`` it would fold a
# negative seed onto a POSITIVE variance: a silent sign flip on the one
# direction of this relation that matters.
#
# WHY THE DEFINITION LIVES HERE BUT THE REGISTRATION DOES NOT.  ConfigManager
# discovers relations by walking ``components/*/symbolic_physics.py`` and keys
# each file on its directory name (or its ``comp_key``), which must match a
# single YAML block; the symbol map it then applies is per component INSTANCE
# (``transit.0.jitter``, ...).  So there is no file this parent class could own
# that would register one relation against three different YAML blocks -- and
# it should not want to: ``mulensinstrument`` is an ``Instrument`` too, with
# ``noise_model = "err_scale"`` and no jitter at all, so the set of children
# that get this relation is a per-child fact.  Each additive-noise child
# therefore imports these two objects into its own ``symbolic_physics.py`` and
# registers them; the definition, like ``calc_jitter`` above, exists once.
JITTER_SYMBOLS = {
    "jitter": sp.Symbol("jitter", real=True),
    "jitter_variance": sp.Symbol("jitter_variance", real=True),
}

# Keys must equal the sympy symbol NAMES above: ConfigManager substitutes
# relation symbols by ``sym.name``, so a map key that does not match leaves the
# symbol unbound and the relation silently inert.  Sharing one dict across the
# three children is what makes that failure mode unreachable -- transit's copy
# named its symbol "jittervar" against a "jitter_variance" key and was inert
# from the day it was written until 2026-08.
JITTER_SYMBOL_MAP = {
    "jitter": "jitter",
    "jitter_variance": "jitter_variance",
}

JITTER_RELATIONS = [
    sp.Eq(
        JITTER_SYMBOLS["jitter_variance"],
        JITTER_SYMBOLS["jitter"] * sp.Abs(JITTER_SYMBOLS["jitter"]),
    )
]


class ConcatenatedData:
    """Per-file accumulator for the arrays an instrument concatenates.

    Every child that models ONE observable per epoch (rv, transit, mulens)
    builds the same things in ``load_data``: one concatenated time /
    observable / error array, the ``inst_map`` naming which file each row came
    from, the block-diagonal detrend design matrix, and -- last -- the GP and
    robust-likelihood indices derived from all of the above.  This class owns
    that template so it exists once.

    Usage (the child keeps only its own per-file physics)::

        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            df = self._read_data(i, roles=("time", "rv", "err"), detrend=True)
            ...per-file work...
            blocks.add(i, time=..., obs=..., err=..., df=df)
        blocks.finalize("rv", user_factor=...)

    WHY AN ACCUMULATOR AND NOT A TEMPLATE METHOD WITH A PER-FILE HOOK.  The
    three children disagree about *when* the files are read:
    ``mulensinstrument`` reads every file in a first pass because the Skowron
    reference epoch (``t0_par``) is resolved from ALL the times before the
    per-file observer positions can be computed in a second.  A template method
    owning the loop would have to grow a "read everything first" mode for that
    one caller; an accumulator the child feeds is indifferent to the loop
    structure, which is the part that genuinely differs.

    THE ROW-RANGE INVARIANT IS ENFORCED, NOT ASSUMED.  Each instrument's rows
    must be a single contiguous block, in config order, in every concatenated
    array simultaneously: ``Instrument._build_block_detrend`` lays its blocks
    on the diagonal by walking the per-file row counts in order, and
    ``mulensinstrument.observer_pos`` is addressed row-for-row against
    ``time``.  Both break silently if a file is added out of order or a side
    array disagrees in length, so ``add`` rejects both, and ``finalize``
    publishes the resulting ``(start, stop)`` ranges as ``owner.row_ranges``
    rather than leaving every consumer to re-derive them from ``inst_map``.
    """

    def __init__(self, owner, n_roles=3):
        self.owner = owner
        # Number of canonical (non-detrend) columns in the child's `roles`
        # tuple: the detrend columns are whatever follows them.
        self.n_roles = int(n_roles)
        self.times = []
        self.obs = []
        self.errs = []
        self.detrend = []
        self.counts = []
        self.sides = {}

    # -- per file --------------------------------------------------------
    def add(self, i, time, obs, err, df=None, detrend=None, **sides):
        """Append file ``i``'s block to every array.

        ``time``/``obs``/``err`` are the per-file arrays in the units the
        child wants concatenated (any unit conversion is the child's, applied
        before the call).  ``df`` is the DataFrame ``_read_data`` returned:
        its columns past ``n_roles`` become this file's detrend block.  Pass
        ``detrend`` explicitly to override that, or leave both unset for a
        file with no detrend columns.  Extra keyword arguments are per-epoch
        SIDE ARRAYS (mulensing's ``observer_pos``): they are concatenated
        along axis 0 and set on the owner under their keyword name, so they
        stay row-aligned with ``time`` by construction.
        """
        if i != len(self.counts):
            raise ValueError(
                f"[{self.owner.prefix}] data blocks must be added in config "
                f"order: expected element {len(self.counts)}, got {i}. The "
                f"concatenated arrays address each instrument as one "
                f"contiguous row range (block-diagonal detrending, per-epoch "
                f"side arrays), which out-of-order blocks break silently."
            )
        time = np.asarray(time)
        n_obs = len(time)
        for name, arr in (("obs", obs), ("err", err)):
            if len(np.asarray(arr)) != n_obs:
                raise ValueError(
                    f"[{self.owner.prefix}[{self.owner.names[i]}]] "
                    f"{name} has {len(np.asarray(arr))} rows but time has "
                    f"{n_obs}."
                )

        if detrend is None:
            if df is not None and df.shape[1] > self.n_roles:
                detrend = df.iloc[:, self.n_roles :].values.astype(float)
            else:
                detrend = np.empty((n_obs, 0))
        detrend = np.asarray(detrend)
        if detrend.shape[0] != n_obs:
            raise ValueError(
                f"[{self.owner.prefix}[{self.owner.names[i]}]] detrend block "
                f"has {detrend.shape[0]} rows but time has {n_obs}."
            )

        for name, arr in sides.items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_obs:
                raise ValueError(
                    f"[{self.owner.prefix}[{self.owner.names[i]}]] per-epoch "
                    f"array '{name}' has {arr.shape[0]} rows but time has "
                    f"{n_obs}; side arrays must stay row-aligned with the "
                    f"observations."
                )
            if i == 0:
                self.sides[name] = []
            elif name not in self.sides:
                raise ValueError(
                    f"[{self.owner.prefix}[{self.owner.names[i]}]] per-epoch "
                    f"array '{name}' was not supplied for earlier files; a "
                    f"side array must cover every element or none."
                )
            self.sides[name].append(arr)
        missing = set(self.sides) - set(sides)
        if missing:
            raise ValueError(
                f"[{self.owner.prefix}[{self.owner.names[i]}]] missing "
                f"per-epoch array(s) {sorted(missing)} supplied by earlier "
                f"files; a side array must cover every element or none."
            )

        self.times.append(time)
        self.obs.append(obs)
        self.errs.append(err)
        self.detrend.append(detrend)
        self.counts.append(n_obs)

    # -- after the loop --------------------------------------------------
    def finalize(self, observable, user_factor=1.0):
        """Concatenate, publish on the owner, and run the optional hooks.

        Sets ``time``, ``<observable>``, ``err``, ``inst_map``,
        ``n_total_obs``, ``row_ranges``, every side array, and the
        ``detrend_matrix`` / ``n_detrend_per_inst`` / ``total_detrend_cols``
        triple; then calls ``_prepare_gp`` and ``_prepare_robust``, which are
        no-ops unless a file set ``gp:`` / ``likelihood:``.

        ``user_factor`` converts the concatenated error from the internal unit
        it is held in to the user unit the GP amplitude and the robust
        ``out_scale`` are declared in (both are declared in the same unit as
        the data, so one factor serves both).
        """
        owner = self.owner
        if len(self.counts) != owner.n_elements:
            raise ValueError(
                f"[{owner.prefix}] {len(self.counts)} of {owner.n_elements} "
                f"elements contributed data blocks; every element must."
            )

        owner.time = np.concatenate(self.times).astype(float)
        setattr(owner, observable, np.concatenate(self.obs).astype(float))
        owner.err = np.concatenate(self.errs).astype(float)
        # Named `inst_map` so Component.build_tensor_maps auto-generates
        # `inst_map_tensor` in stage 4.
        owner.inst_map = np.repeat(
            np.arange(owner.n_elements), self.counts
        ).astype(int)
        owner.n_total_obs = int(owner.inst_map.size)
        owner.row_ranges = self.row_ranges()

        for name, blocks in self.sides.items():
            setattr(owner, name, np.concatenate(blocks, axis=0).astype(float))

        (
            owner.detrend_matrix,
            owner.n_detrend_per_inst,
            owner.total_detrend_cols,
        ) = owner._build_block_detrend(self.detrend, owner.n_total_obs)

        owner._prepare_gp(
            owner.time, owner.err, owner.inst_map, user_factor=user_factor
        )
        owner._prepare_robust(
            owner.err, owner.inst_map, user_factor=user_factor
        )

    def row_ranges(self):
        """``[(start, stop), ...]``: file ``i``'s rows in every array."""
        edges = np.concatenate(([0], np.cumsum(self.counts))).astype(int)
        return [
            (int(edges[i]), int(edges[i + 1])) for i in range(len(self.counts))
        ]


class Instrument(Component):
    # Noise parameterization: "jitter_variance" (additive) or "err_scale"
    # (multiplicative).  Subclasses override; the default matches the majority
    # (rv/transit/astrometry).
    noise_model = "jitter_variance"

    # Human noun for the modeling-draft prose ("radial velocity", "transit
    # photometry", ...).  Children override; the class-name fallback is only
    # legible to developers.
    prose_noun = None

    # Whether this child has wired up the GP hooks below.  Children that model
    # more than one observable per data file (astrometry: E/N or sep/PA, which
    # differ in units and so cannot share one amplitude) leave this False and
    # reject the ``gp:`` key outright rather than silently ignoring it.
    supports_gp = True

    # Same switch for the robust `likelihood:` hooks (hogg mixture needs a
    # single out_scale unit per file, so astrometry's two-observable files
    # opt out for the same reason they opt out of GPs).
    supports_robust_likelihood = True

    def __init__(self, component_config, config_manager):
        super().__init__(component_config, config_manager)
        # Every instrument reads its data from per-element files and tracks a
        # running observation count.
        self.files = [c.get("file") for c in self.config]
        self.n_total_obs = 0
        # Per-element (start, stop) row ranges into the concatenated arrays,
        # published by ConcatenatedData.finalize.  Empty for a child that does
        # not concatenate (astrometryinstrument keeps per-file datasets).
        self.row_ranges = []
        # Optional per-file exclusion mask (see _apply_mask). Parsed here so a
        # malformed spec fails at construction, not mid-load.
        self.mask_specs = [c.get("mask") for c in self.config]
        # Optional per-file time system (time_offset/time_scale/time_frame/
        # time_location/time_ephemeris) and column layout (columns:), both
        # applied by _read_data.  Parsed here so malformed specs fail at
        # construction too.
        self.time_specs = [
            self._parse_time_spec(c, i) for i, c in enumerate(self.config)
        ]
        self.column_specs = [
            self._parse_columns_spec(c, i) for i, c in enumerate(self.config)
        ]
        self._load_plot_styles()
        self._load_gp_config()
        self._load_likelihood_config()

    # ------------------------------------------------------------------
    # Per-instrument plot styling (config, not Parameters -- no sampling)
    # ------------------------------------------------------------------
    def _load_plot_styles(self):
        """Read an optional per-instrument ``plot: {color, marker}`` block.

        Populates ``self.plot_color[i]`` / ``self.plot_marker[i]`` (each None
        when unset).  These are user overrides for the categorical data-series
        style; the theme supplies the default by series index (see
        ``notes/gui_todo.txt`` precedence: user > theme).  A GUI can later
        expose a picker that writes this same key via the ruamel round-trip.
        """
        self.plot_color = []
        self.plot_marker = []
        for c in self.config:
            style = c.get("plot") or {}
            if not isinstance(style, dict):
                style = {}
            self.plot_color.append(style.get("color"))
            self.plot_marker.append(style.get("marker"))

    def _data_trace_style(self, i):
        """Style override dict for instrument ``i``'s data trace.

        Always carries the categorical ``series_index``; adds an explicit
        ``color`` / ``marker`` only when the user configured one (so the theme
        default by index still applies otherwise).  Suitable for
        ``plotspec.Trace.style``.
        """
        style = {"series_index": int(i)}
        if self.plot_color[i] is not None:
            style["color"] = self.plot_color[i]
        if self.plot_marker[i] is not None:
            style["marker"] = self.plot_marker[i]
        return style

    @staticmethod
    def _plot_style_config_schema():
        """The shared ``plot`` config-schema entry (per-instrument styling).

        Children append this to their ``config_schema()`` so introspection and
        a GUI discover the key generically.
        """
        return {
            "key": "plot",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                "Optional per-instrument plot styling: a mapping with "
                "'color' and/or 'marker' overriding the theme default for "
                "this data series (e.g. {color: '#1f77b4', marker: 's'})."
            ),
        }

    # ------------------------------------------------------------------
    # Shared data loading
    # ------------------------------------------------------------------
    def _concat_blocks(self, n_roles=3):
        """A fresh ``ConcatenatedData`` accumulator for this component.

        ``n_roles`` is the number of canonical columns in the child's
        ``_read_data(roles=...)`` tuple; anything past them is that file's
        detrend block.
        """
        return ConcatenatedData(self, n_roles=n_roles)

    @staticmethod
    def _sort_by_time(df, time_col=0):
        """Return ``df`` sorted ascending by its time column.

        Every instrument calls this on each data file right after reading it,
        BEFORE splitting out columns or deriving anything from the times, so
        one sort keeps the observable, the errors, the detrend columns and any
        per-epoch quantity computed downstream (observer positions, parallax
        factors) aligned by construction.

        Sorting per file rather than globally is deliberate: the concatenated
        arrays stay contiguous per instrument, which ``_build_block_detrend``
        relies on to place each instrument's columns on the block diagonal,
        and which keeps every row-aligned side array (mulens's observer
        positions) in step with its own file.  A global sort across files
        would interleave instruments and quietly break both.

        Stable sort, so files already in time order are untouched and repeated
        timestamps keep their relative order.
        """
        order = np.argsort(df.iloc[:, time_col].values, kind="stable")
        if np.all(order == np.arange(len(order))):
            return df  # already sorted: no copy
        return df.iloc[order].reset_index(drop=True)

    def _apply_mask(self, df, i):
        """Drop excluded rows of file ``i`` per its optional ``mask:`` config.

        Called on each data file right after reading and BEFORE
        ``_sort_by_time``, so every mask spec refers to the file's own row
        order as it is on disk (comment lines not counted).  Dropping whole
        rows here -- before any column is split out or anything is derived
        from the times -- keeps the observable, errors, detrend columns and
        every row-aligned side array (observer positions, parallax factors)
        consistent by construction, exactly like the sort.

        Accepted ``mask:`` forms on an instrument's config entry:

        - a path to a whitespace/newline-delimited file with ONE value per
          data row (0/1, true/false): nonzero/true means EXCLUDE that row.
          This is the shape a bad-data flag vector (e.g. from MMEXOFAST)
          saves to naturally.
        - a list of booleans, one per data row: True means EXCLUDE.
        - a list of integers: 0-based row indices to EXCLUDE.

        Absent or ``null`` keeps every point (the default: byte-for-byte the
        pre-feature behavior).
        """
        spec = self.mask_specs[i]
        if spec is None:
            return df
        label = f"{self.prefix}[{self.names[i]}]"
        n = len(df)

        if isinstance(spec, str):
            flags = np.loadtxt(spec, dtype=float, comments="#").ravel()
            if flags.size != n:
                raise ValueError(
                    f"[{label}] mask file '{spec}' has {flags.size} entries "
                    f"but the data file has {n} rows -- one flag per data "
                    f"row is required."
                )
            exclude = flags.astype(bool)
        elif isinstance(spec, (list, tuple, np.ndarray)):
            spec = list(spec)
            if all(isinstance(v, (bool, np.bool_)) for v in spec):
                if len(spec) != n:
                    raise ValueError(
                        f"[{label}] boolean mask has {len(spec)} entries but "
                        f"the data file has {n} rows."
                    )
                exclude = np.asarray(spec, dtype=bool)
            elif all(
                isinstance(v, (int, np.integer))
                and not isinstance(v, (bool, np.bool_))
                for v in spec
            ):
                idx = np.asarray(spec, dtype=int)
                if idx.size and (idx.min() < 0 or idx.max() >= n):
                    raise ValueError(
                        f"[{label}] mask indices must be 0-based row indices "
                        f"in [0, {n - 1}]; got min {idx.min()}, max "
                        f"{idx.max()}."
                    )
                exclude = np.zeros(n, dtype=bool)
                exclude[idx] = True
            else:
                raise ValueError(
                    f"[{label}] mask list must be all booleans (per-row "
                    f"flags) or all integers (row indices to exclude)."
                )
        else:
            raise ValueError(
                f"[{label}] mask must be a file path or a list; got "
                f"{type(spec).__name__}."
            )

        n_masked = int(exclude.sum())
        if n_masked == n:
            raise ValueError(
                f"[{label}] mask excludes every one of the {n} data points."
            )
        if n_masked:
            logger.info(f"[{label}] mask excluded {n_masked}/{n} data points.")
            df = df.loc[~exclude].reset_index(drop=True)
        return df

    @staticmethod
    def _mask_config_schema():
        """The shared ``mask`` config-schema entry (per-file point exclusion).

        Children append this to their ``config_schema()`` so introspection and
        a GUI discover the key generically.
        """
        return {
            "key": "mask",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                "Optional per-file point exclusion, applied to the data "
                "file's own row order before anything is derived from it: a "
                "path to a file with one 0/1 flag per data row (nonzero = "
                "exclude), a list of booleans (one per row, true = exclude), "
                "or a list of 0-based row indices to exclude."
            ),
        }

    # ------------------------------------------------------------------
    # Shared file reading: columns, mask, time system, sort
    # ------------------------------------------------------------------
    def _read_data(self, i, roles, detrend=False, shared_roles=()):
        """Read data file ``i`` into canonical column order, ready to use.

        One call replaces the read/mask/sort triplet every child used to
        write, and is the single place the user-facing file conveniences
        happen, in this order:

        1. ``columns:``   reorder/select columns into the canonical layout
                          (``_select_columns``);
        2. ``mask:``      drop excluded rows, in on-disk row order
                          (``_apply_mask``);
        3. ``time_*:``    add ``time_offset`` and convert the time column
                          to BJD_TDB (``_to_bjd_tdb``);
        4. sort ascending by (converted) time (``_sort_by_time``).

        ``roles`` names the canonical columns in order and must start with
        ``"time"``; with ``detrend=True`` any detrend columns follow them.
        The returned DataFrame is indexed positionally: column ``j`` is
        ``roles[j]``, columns ``len(roles):`` are the detrend columns.
        With none of the optional keys set this is byte-for-byte the old
        read (all columns, as-is on disk, sorted by column 0).

        ``shared_roles`` names groups of roles this component allows to
        read the SAME file column (see ``_check_no_duplicate_columns``);
        every other collision is an error.
        """
        if roles[0] != "time":
            raise ValueError(
                f"[{self.prefix}] _read_data roles must start with 'time'; "
                f"got {list(roles)}."
            )
        df = pd.read_csv(
            self.files[i], sep=r"\s+", engine="c", header=None, comment="#"
        )
        df = self._select_columns(df, i, roles, detrend, shared_roles)
        df = self._apply_mask(df, i)
        t = self._to_bjd_tdb(df.iloc[:, 0].values.astype(float), i)
        df[df.columns[0]] = t
        return self._sort_by_time(df)

    # ------------------------------------------------------------------
    # Optional per-file column layout (columns:)
    # ------------------------------------------------------------------
    def _parse_columns_spec(self, c, i):
        """Validate one config entry's optional ``columns:`` key (stage 0).

        The value is a mapping from role name to a 0-based column index in
        the data file, plus an optional ``detrend`` role mapping to a LIST
        of 0-based column indices.  Role-name validity is checked later, in
        ``_select_columns``, because only the child knows its roles (and
        astrometry's depend on the file's mode); the structure is checked
        here so a malformed spec fails at construction.
        """
        spec = c.get("columns")
        if spec is None:
            return None
        label = f"{self.prefix}[{self.names[i]}]"
        if not isinstance(spec, dict):
            raise ValueError(
                f"[{label}] columns must be a mapping of role name to "
                f"0-based column index (plus optional 'detrend: [i, ...]'); "
                f"got {type(spec).__name__}."
            )

        def _col(v):
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or v < 0
            ):
                raise ValueError(
                    f"[{label}] columns entries must be 0-based column "
                    f"indices (non-negative integers); got {v!r}."
                )
            return int(v)

        out = {}
        for k, v in spec.items():
            if k == "detrend":
                if not isinstance(v, (list, tuple)):
                    raise ValueError(
                        f"[{label}] columns.detrend must be a list of "
                        f"0-based column indices; got {v!r}."
                    )
                out[k] = [_col(x) for x in v]
            else:
                out[k] = _col(v)
        return out

    def _select_columns(self, df, i, roles, detrend, shared_roles=()):
        """Apply file ``i``'s ``columns:`` spec, returning the canonical layout.

        Without a spec the file is returned untouched (the on-disk order IS
        the canonical order, detrend columns are everything past the named
        roles).  With a spec, each role defaults to its canonical position
        and named roles override it; detrend columns are ONLY the ones the
        spec lists (an explicit layout leaves no 'rest of the columns' to
        guess at).
        """
        spec = self.column_specs[i]
        label = f"{self.prefix}[{self.names[i]}]"
        if spec is None:
            if df.shape[1] < len(roles):
                raise ValueError(
                    f"[{label}] data file has {df.shape[1]} columns but "
                    f"needs at least {len(roles)} ({', '.join(roles)})."
                )
            return df

        unknown = sorted(set(spec) - set(roles) - {"detrend"})
        if unknown:
            raise ValueError(
                f"[{label}] columns names unknown role(s) {unknown}; valid "
                f"roles are {list(roles)}"
                + (" plus 'detrend'." if detrend else ".")
            )
        if "detrend" in spec and not detrend:
            raise ValueError(
                f"[{label}] columns.detrend is not supported by this "
                f"component (it has no detrending)."
            )
        idx = [spec.get(role, j) for j, role in enumerate(roles)]
        det = spec.get("detrend", []) if detrend else []
        too_big = [c for c in idx + det if c >= df.shape[1]]
        if too_big:
            raise ValueError(
                f"[{label}] columns indices are 0-based and the data file "
                f"has {df.shape[1]} columns; got {sorted(set(too_big))}."
            )
        self._check_no_duplicate_columns(label, roles, idx, det, shared_roles)
        out = df.iloc[:, idx + det].copy()
        out.columns = range(out.shape[1])
        return out

    @staticmethod
    def _check_no_duplicate_columns(label, roles, idx, det, shared_roles=()):
        """Reject a ``columns:`` spec that points two roles at one column.

        A PARTIAL spec is the trap this closes: ``columns: {time: 1}``
        names only the time column, every other role keeps its canonical
        position, and ``rv`` is canonically column 1 -- so the RVs are
        silently the times, and the fit runs on a dataset nobody wrote.
        Nothing downstream can notice: ``df.iloc`` is happy to select a
        column twice.

        Two reuses are legitimate and survive:

        * The TIME column may ALSO be a detrend column.  Detrending
          against a linear trend in time is a real use case, and there is
          no other way to spell it (the detrend list may not name a
          column that is not in the file).
        * Roles the component declares interchangeable through
          ``shared_roles``, a sequence of role-name groups.  Astrometry's
          ``abs`` mode passes ``("err_e", "err_n")`` so one symmetric
          per-epoch uncertainty column can serve both sky axes -- a
          common catalog layout.

        Everything else is an error, including a ``detrend`` list that
        repeats a column (two identical basis vectors are an exactly
        degenerate pair of coefficients) and two roles in different units
        (astrometry ``rel``'s ``err_sep`` in mas vs ``err_pa`` in deg).
        """
        names = list(roles) + [f"detrend[{k}]" for k in range(len(det))]
        by_column = {}
        for name, col in zip(names, list(idx) + list(det)):
            by_column.setdefault(col, []).append(name)

        groups = [set(g) for g in shared_roles]
        for col, hits in sorted(by_column.items()):
            if len(hits) == 1:
                continue
            # time, reused as exactly one detrend column
            if (
                len(hits) == 2
                and "time" in hits
                and any(h.startswith("detrend[") for h in hits)
            ):
                continue
            if any(set(hits) <= g for g in groups):
                continue
            allowed = "".join(
                f"; '{sorted(g)}' may share one" for g in shared_roles
            )
            raise ValueError(
                f"[{label}] columns maps {hits} to the same file column "
                f"{col}. Each role needs its own column -- note that roles "
                f"the spec does not name keep their canonical position, so "
                f"a partial spec can collide with a default. (The only "
                f"reuse allowed is the time column doubling as a detrend "
                f"column{allowed}.)"
            )

    @staticmethod
    def _columns_config_schema(roles, detrend=True, note=""):
        """The shared ``columns`` config-schema entry for the given roles."""
        extra = (
            ", plus 'detrend' mapping to a list of column indices to "
            "detrend against (when columns: is given, detrend columns "
            "must be listed explicitly)"
            if detrend
            else ""
        )
        return {
            "key": "columns",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                f"Optional column layout: a mapping from role name to "
                f"0-based column index in the data file. Roles: "
                f"{', '.join(roles)}{extra}. Unnamed roles keep their "
                f"default position -- so two roles must not end up on the "
                f"same column, which a partial spec can cause (that is an "
                f"error; the time column may also be a detrend column). "
                f"Default: the documented column order."
                + (f" {note}" if note else "")
            ),
        }

    # ------------------------------------------------------------------
    # Optional per-file time system (time_offset / time_scale / time_frame)
    # ------------------------------------------------------------------
    def _parse_time_spec(self, c, i):
        """Validate one config entry's optional time-system keys (stage 0).

        Returns a dict with ``offset`` (days, added to the raw times
        first), ``scale`` (astropy time scale of the input), ``frame``
        (jd/hjd/bjd: where the input's clock sits), ``location`` (observer
        for the jd frame's light-travel and topocentric-scale terms),
        ``ephemeris`` (solar-system ephemeris for light_travel_time) and
        ``needs_conversion``.  The default -- offset 0, scale tdb, frame
        bjd -- is BJD_TDB in, BJD_TDB out, untouched.
        """
        label = f"{self.prefix}[{self.names[i]}]"

        offset = c.get("time_offset", 0.0)
        if isinstance(offset, bool) or not isinstance(
            offset, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"[{label}] time_offset must be a number (days, added to "
                f"every input time); got {offset!r}."
            )

        scale = c.get("time_scale", "tdb")
        if not isinstance(scale, str):
            raise ValueError(
                f"[{label}] time_scale must be one of {list(_TIME_SCALES)}; "
                f"got {scale!r}."
            )
        scale = _TIME_SCALE_ALIASES.get(scale.lower(), scale.lower())
        if scale not in _TIME_SCALES:
            raise ValueError(
                f"[{label}] time_scale must be one of {list(_TIME_SCALES)} "
                f"(or 'ut' for ut1); got {c.get('time_scale')!r}."
            )

        frame = c.get("time_frame", "bjd")
        if not isinstance(frame, str) or frame.lower() not in _TIME_FRAMES:
            raise ValueError(
                f"[{label}] time_frame must be one of "
                f"{list(_TIME_FRAMES)}; got {frame!r}."
            )
        frame = frame.lower()

        location = c.get("time_location")
        if location is not None:
            ok = isinstance(location, str) or (
                isinstance(location, (list, tuple))
                and len(location) in (2, 3)
                and all(
                    isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool)
                    for v in location
                )
            )
            if not ok:
                raise ValueError(
                    f"[{label}] time_location must be an observatory name "
                    f"(astropy EarthLocation.of_site) or [lon_deg, lat_deg"
                    f"(, height_m)]; got {location!r}."
                )

        ephemeris = c.get("time_ephemeris", "builtin")
        if not isinstance(ephemeris, str):
            raise ValueError(
                f"[{label}] time_ephemeris must be an astropy solar-system "
                f"ephemeris name ('builtin', 'jpl', 'de440', ...); got "
                f"{ephemeris!r}."
            )

        return {
            "offset": float(offset),
            "scale": scale,
            "frame": frame,
            "location": location,
            "ephemeris": ephemeris,
            "needs_conversion": scale != "tdb" or frame != "bjd",
        }

    @property
    def has_nontrivial_time_spec(self):
        """True when any file sets a time offset or a time-system conversion."""
        return any(
            s["offset"] != 0.0 or s["needs_conversion"]
            for s in self.time_specs
        )

    def _to_bjd_tdb(self, t, i):
        """Convert file ``i``'s raw times to BJD_TDB, per its time spec.

        ``time_offset`` is added first (so truncated times like
        BJD-2450000 or MJD become absolute JDs); the scale/frame
        conversion then runs on absolute JDs only.  The algorithm is the
        standard one (Eastman, Siverd & Gaudi 2010):

        1. strip the input frame's light-travel correction to recover the
           observer's JD in the input scale -- ``t = t_obs + ltt(t_obs)``
           is inverted by fixed-point iteration, which converges below a
           nanosecond in 3 passes because d(ltt)/dt <= v_earth/c ~ 1e-4;
        2. convert the time scale to TDB (astropy/erfa: leap seconds for
           UTC/TAI, the erfa TDB-TT model, IERS tables for UT1);
        3. add back the barycentric light-travel time in TDB.

        Input already in the bjd frame skips 1 and 3: the barycentric
        correction appears identically on both sides and cancels exactly,
        so a scale-only conversion (BJD_UTC -> BJD_TDB) needs no
        coordinates.

        Accuracy notes (why the remaining terms are out of scope):
        the observer's position enters through ``time_location`` (omitting
        it costs up to 21 ms of geocenter-vs-observatory Romer delay);
        the builtin (erfa) ephemeris is good to a few microseconds of
        light travel (``time_ephemeris: de440`` reaches ns, needs
        jplephem); a single float64 JD quantizes at ~40 microseconds
        anyway, which is the real floor here; TT(BIPM) (~30 us), the
        Shapiro delay (~us; ~100 us within ~1 deg of the Sun), and
        proper-motion/parallax evolution of the source direction (~us/yr
        for mas/yr motions) are all below that floor's usefulness and are
        not modeled.
        """
        spec = self.time_specs[i]
        if spec["offset"] != 0.0:
            t = t + spec["offset"]
        if not spec["needs_conversion"]:
            return t

        label = f"{self.prefix}[{self.names[i]}]"
        if t.min() < _MIN_ABS_JD:
            raise ValueError(
                f"[{label}] time_scale/time_frame conversion needs absolute "
                f"Julian Dates, but the smallest time after time_offset is "
                f"{t.min():.3f}. Set time_offset to restore full JDs (e.g. "
                f"2450000 for BJD-2450000 data, 2400000.5 for MJD)."
            )

        # astropy.coordinates is deliberately imported lazily: it is slow to
        # import and only needed when a file actually opts into conversion.
        from astropy.time import Time

        location = self._time_location(i)

        if spec["frame"] == "bjd":
            # Scale-only conversion: the barycentric light-travel term is
            # identical on both sides and cancels exactly (the TDB-vs-UTC
            # evaluation epoch of the correction matters at the 0.1 us
            # level), so no coordinates are needed at all.
            out = Time(
                t, format="jd", scale=spec["scale"], location=location
            ).tdb.jd
        else:
            coord = self._time_coord(i, label)
            kind = _TIME_FRAMES[spec["frame"]]
            ephemeris = spec["ephemeris"]

            t_obs = t
            if kind is not None:
                for _ in range(3):
                    ltt = Time(
                        t_obs,
                        format="jd",
                        scale=spec["scale"],
                        location=location,
                    ).light_travel_time(coord, kind=kind, ephemeris=ephemeris)
                    t_obs = t - ltt.jd
            t_tdb = Time(
                t_obs, format="jd", scale=spec["scale"], location=location
            ).tdb
            out = (
                t_tdb.jd
                + t_tdb.light_travel_time(
                    coord, kind="barycentric", ephemeris=ephemeris
                ).jd
            )
        logger.info(
            "[%s] converted %d times from %s_%s to BJD_TDB "
            "(median shift %+.3f s).",
            label,
            t.size,
            spec["frame"].upper(),
            spec["scale"].upper(),
            float(np.median(out - t)) * 86400.0,
        )
        return out

    def _time_coord(self, i, label):
        """The target ICRS direction for file ``i``'s light-travel terms.

        Reuses the star component's ra/dec exactly as astrometry and mulens
        do (``star_ndx`` on the file's config entry picks the star, default
        0; every star in one system is the same direction at the accuracy
        that matters here).  Requiring ``user_modified`` is deliberate: the
        defaults.yaml ra/dec are placeholders, and a conversion run against
        them would corrupt every BJD by up to +/-8 minutes with no error.
        """
        star_ndx = int(self.config[i].get("star_ndx", 0))
        ra = self.config_manager.resolve("star", "ra", element=star_ndx)
        dec = self.config_manager.resolve("star", "dec", element=star_ndx)
        if not (ra["user_modified"] and dec["user_modified"]):
            raise ValueError(
                f"[{label}] time_scale/time_frame conversion needs the "
                f"target's coordinates: set star.{star_ndx}.ra and "
                f"star.{star_ndx}.dec (deg) in the params file."
            )
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        # resolve() returns per-element arrays even for shape=(); take the
        # single element rather than float()-ing an ndarray (NumPy 2 error).
        return SkyCoord(
            ra=float(np.ravel(ra["initval"])[0]) * u.Unit(ra["unit"] or "deg"),
            dec=float(np.ravel(dec["initval"])[0])
            * u.Unit(dec["unit"] or "deg"),
        )

    def _time_location(self, i):
        """File ``i``'s observer EarthLocation (geocenter when unset)."""
        import astropy.units as u
        from astropy.coordinates import EarthLocation

        location = self.time_specs[i]["location"]
        if location is None:
            return EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
        if isinstance(location, str):
            return EarthLocation.of_site(location)
        lon, lat = float(location[0]), float(location[1])
        height = float(location[2]) if len(location) == 3 else 0.0
        return EarthLocation.from_geodetic(lon, lat, height)

    @staticmethod
    def _time_config_schema():
        """The shared time-system config-schema entries; children append them."""
        return [
            {
                "key": "time_offset",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Optional offset in days added to every input time "
                    "before anything else (e.g. 2450000 for BJD-2450000 "
                    "data, 2400000.5 for MJD). Default 0."
                ),
            },
            {
                "key": "time_scale",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Time scale of the input times: utc, tai, tt, tdb, "
                    "tcb, tcg or ut1 ('ut' is accepted for ut1; ut1 may "
                    "trigger an IERS table download). Default tdb."
                ),
            },
            {
                "key": "time_frame",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Reference frame of the input times: jd (observer/"
                    "geocenter), hjd (heliocentric) or bjd (barycentric). "
                    "Anything but the default bjd+tdb is converted to "
                    "BJD_TDB at load time, which requires absolute JDs "
                    "(see time_offset); jd/hjd also require user-set star "
                    "ra/dec. Default bjd."
                ),
            },
            {
                "key": "time_location",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Observer location for the time conversion: an astropy "
                    "observatory name (EarthLocation.of_site) or [lon_deg, "
                    "lat_deg(, height_m)]. Default geocenter (up to 21 ms "
                    "of Romer delay is unmodeled without it)."
                ),
            },
            {
                "key": "time_ephemeris",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Solar-system ephemeris for the light-travel terms: "
                    "'builtin' (erfa, ~us accuracy) or a JPL kernel like "
                    "'de440' (~ns, needs jplephem + download). Default "
                    "builtin."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Optional per-file Gaussian-process noise (celerite2)
    # ------------------------------------------------------------------
    def _load_gp_config(self):
        """Read the per-instrument ``gp:`` key (stage 0, in ``__init__``).

        Populates ``self.gp_terms[i]`` -- a (possibly empty) tuple of kernel
        names for element ``i`` -- and ``self.has_gp``.  Absent or ``none``
        means independent Gaussian errors, which is the default for every
        file and leaves the model byte-for-byte what it was before this
        feature existed.
        """
        self.gp_terms = []
        for i, c in enumerate(self.config):
            terms = gp_support.parse_gp_spec(
                c.get("gp"), context=f"{self.prefix}[{self.names[i]}]"
            )
            if terms and not self.supports_gp:
                raise NotImplementedError(
                    f"[{self.prefix}[{self.names[i]}]] Gaussian-process noise "
                    f"is not supported by this component: it models more than "
                    f"one observable per data file (with different units), so "
                    f"a single GP kernel is ambiguous. Remove the 'gp:' key."
                )
            self.gp_terms.append(terms)
        self.has_gp = any(self.gp_terms)

        # Element index -> indices into the concatenated observation arrays,
        # sorted by time.  Filled by _prepare_gp.
        self._gp_obs_index = {}
        self._gp_time = None
        # Linear values of the log-sampled hyperparameters, as Deterministics
        # (built once in add_observation_likelihood).
        self._gp_linear = {}
        # celerite2 GaussianProcess objects, keyed by element index, and the
        # observations each conditioned on -- both filled by
        # add_observation_likelihood and consumed by _compile_gp_plotters.
        self.gp_objects = {}
        self._gp_observed_node = {}
        self._gp_pred_at_data = {}
        self._gp_pred_on_grid = {}

    def _gp_elements(self, kind):
        """Element indices that requested GP term ``kind``."""
        return [i for i, terms in enumerate(self.gp_terms) if kind in terms]

    @staticmethod
    def _gp_config_schema():
        """The shared ``gp`` config-schema entry; children append it."""
        return gp_support.gp_config_schema_entry()

    @classmethod
    def shared_parameter_names(cls):
        """The GP and robust-likelihood parameters, declared at the root of
        components/defaults.yaml.

        Only the scale-carrying ones are redeclared per component (their unit
        is the data's), so introspection needs this list to report the rest.
        Components that opt out of a feature declare none of its names.
        """
        names = []
        if cls.supports_gp:
            names += [
                name
                for kind in gp_support.GP_TERMS
                for name in gp_support.GP_TERM_PARAMS[kind]
            ]
        if cls.supports_robust_likelihood:
            names += [
                name
                for kind in robust_support.LIKELIHOOD_KINDS
                for name in robust_support.LIKELIHOOD_PARAMS[kind]
            ]
        return names

    def _prepare_gp(self, time, err, inst_map, user_factor=1.0):
        """Stage 1a: index and seed the GP terms from the loaded data.

        Call at the end of ``load_data``, once the concatenated ``time``,
        error and ``inst_map`` arrays exist.  Two jobs:

        1. Record, per GP element, the time-sorted indices into the
           concatenated arrays.  celerite2's semiseparable solver requires
           ascending times and does not check, so every array handed to it
           (times, sigmas, model, data) goes through this one permutation.
        2. Push a data-driven amplitude hint: the median reported error bar,
           i.e. the file's white-noise level.  Anything derived from the
           *observations* instead (their scatter, or the point-to-point
           scatter) measures the signal the physical model is supposed to
           explain -- an eccentric RV orbit or a microlensing peak dominates
           both -- and seeding the GP there invites it to swallow the signal
           before the sampler has moved.  Starting at the white-noise level is
           conservative in the safe direction, and the logit transform lets
           the amplitude climb during tuning if the data want it.  A user who
           knows the activity amplitude should just set ``initval``.

        ``user_factor`` converts the error from the internal unit the caller
        holds it in to the user unit the amplitude parameter is declared in
        (hints are pushed in user units).
        """
        if not self.has_gp:
            return

        time = np.asarray(time, dtype=float)
        err = np.asarray(err, dtype=float)
        inst_map = np.asarray(inst_map, dtype=int)
        self._gp_time = time

        for i, terms in enumerate(self.gp_terms):
            if not terms:
                continue
            sel = np.flatnonzero(inst_map == i)
            if sel.size < 3:
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] a Gaussian process "
                    f"needs at least 3 observations; this file has "
                    f"{sel.size}."
                )
            self._gp_obs_index[i] = sel[np.argsort(time[sel], kind="stable")]

            amp = float(np.median(err[sel])) * user_factor
            if not np.isfinite(amp) or amp <= 0.0:
                # Degenerate (zero/absent) errors: fall back to the
                # defaults.yaml start rather than hinting a zero amplitude,
                # which would pin the logit transform against its lower bound.
                continue
            for kind in terms:
                path = (
                    f"{self.prefix}.{i}.{gp_support.GP_AMPLITUDE_PARAM[kind]}"
                )
                self.config_manager.add_hint(path, amp)
                self.config_manager.add_scale_hint(path, amp)

    def _register_gp(self, manifest):
        """Stage 2: add this component's GP hyperparameters to ``manifest``.

        Every GP parameter is a full-length (``n_elements``) vector so that a
        user path resolves the same way as any other instrument parameter --
        ``rvinstrument.HARPS.gp_rot_period`` means the HARPS element,
        whichever element that is.  Elements that did not opt into a term are
        pinned fixed (``sigma: 0``) through ``internal_overrides``, which sits
        below RANK_USER: they cost the sampler nothing, and a user who wants
        one back can still override it.

        Returns ``manifest`` for chaining, like ``_register_noise``.
        """
        if not self.has_gp:
            return manifest

        for kind in gp_support.GP_TERMS:
            on = set(self._gp_elements(kind))
            if not on:
                continue
            off = [i for i in range(self.n_elements) if i not in on]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            for param in gp_support.GP_TERM_PARAMS[kind]:
                manifest[param] = dict(entry)
        return manifest

    def _build_log10_deterministics(self, log_params):
        """Expose the linear value of every log10-sampled parameter.

        ``log_params`` is a ``{log_name: linear_name}`` table -- today
        ``gp.GP_LOG_PARAMS`` and ``likelihood.LIKELIHOOD_LOG_PARAMS``, which
        is why this is one function over two tables rather than two copies of
        it.  Both features sample a positive quantity in log10 (see the
        ``gp_*_log_q*`` convention) and both owe the posterior tables the
        linear quantity the model actually uses; a parameter the topology
        never built is skipped.  Returns the ``{linear_name: tensor}`` cache
        the feature's own code reads back.
        """
        linear = {}
        for log_name, lin_name in log_params.items():
            param = getattr(self, log_name, None)
            if param is None or param.value is None:
                continue
            linear[lin_name] = pm.Deterministic(
                f"{self.prefix}.{lin_name}", pt.power(10.0, param.value)
            )
        return linear

    def _build_gp_deterministics(self):
        """Record the linear value of every log-sampled hyperparameter.

        ``gp_rot_log_q0`` and friends are sampled in log10 (see
        components/gp.py); this exposes ``gp_rot_q0`` etc. as Deterministics so
        posterior tables and plots report the quantity the kernel actually
        uses, and caches the tensors for ``_gp_kernel``.
        """
        self._gp_linear = self._build_log10_deterministics(
            gp_support.GP_LOG_PARAMS
        )

    def _gp_kernel(self, i):
        """The summed celerite2 kernel for element ``i``."""
        params_by_kind = {}
        for kind in self.gp_terms[i]:
            if kind == "rotation":
                params_by_kind[kind] = {
                    "sigma": self.gp_rot_sigma.value[i],
                    "period": self.gp_rot_period.value[i],
                    "Q0": self._gp_linear["gp_rot_q0"][i],
                    "dQ": self._gp_linear["gp_rot_dq"][i],
                    "f": self.gp_rot_f.value[i],
                }
            else:  # "sho"
                params_by_kind[kind] = {
                    "sigma": self.gp_sho_sigma.value[i],
                    "rho": self.gp_sho_rho.value[i],
                    "Q": self._gp_linear["gp_sho_q"][i],
                }
        return gp_support.build_kernel(self.gp_terms[i], params_by_kind)

    # Kernel nouns for the modeling-draft prose; keys are components/gp.py's
    # GP_TERMS vocabulary.
    _GP_TERM_PROSE = {
        "rotation": "rotation kernel (two stochastically driven, damped "
        "harmonic oscillators at the period and its first harmonic)",
        "sho": "stochastically driven, damped simple-harmonic-oscillator "
        "kernel",
    }

    def _add_observation_prose(self, system):
        """Declare the modeling-draft sentences for this component's data.

        Called from ``add_observation_likelihood`` (with ``system``), so the
        prose describes exactly the likelihood being built: the data-set
        inventory, any time-system conversion, the noise parameterization,
        and the per-file GP kernels and robust likelihood families.
        Idempotent through the collector's keys, like every prose site.
        """
        from ..outputs.prose import get_collector, join_names, plural
        from ..outputs.texutils import latex_escape

        prose = get_collector(system)
        noun = self.prose_noun or self.__class__.__name__.lower()
        names = [latex_escape(str(n)) for n in self.names]

        prose.add(
            f"We fit {plural(len(names), f'{noun} dataset')} "
            f"({join_names(names)}; "
            f"{self.n_total_obs} observations in total).",
            section="data",
            key=f"{self.prefix}.data",
        )

        converted = [
            names[i]
            for i, s in enumerate(self.time_specs)
            if s["needs_conversion"]
        ]
        if converted:
            prose.add(
                "Times for " + join_names(converted) + " were converted "
                r"to $\rm BJD_{TDB}$ following \citet{Eastman:2010}.",
                section="data",
                key=f"{self.prefix}.time_conversion",
                rank=60,
            )

        if self.noise_model == "err_scale":
            prose.add(
                f"For each {noun} dataset we fit a multiplicative "
                "rescaling of its reported uncertainties.",
                section="noise",
                key=f"{self.prefix}.noise_model",
                rank=10,
            )
        else:
            prose.add(
                f"For each {noun} dataset we fit an additive noise "
                "(``jitter'') variance, added in quadrature to the reported "
                "uncertainties; a negative value indicates the reported "
                "uncertainties are overestimated.",
                section="noise",
                key=f"{self.prefix}.noise_model",
                rank=10,
            )

        by_kernel = {}
        for i, terms in enumerate(getattr(self, "gp_terms", []) or []):
            if terms:
                by_kernel.setdefault(tuple(terms), []).append(names[i])
        for terms, insts in by_kernel.items():
            desc = join_names(
                [f"a {self._GP_TERM_PROSE.get(k, k)}" for k in terms]
            )
            if len(terms) > 1:
                desc = "the sum of " + desc
            prose.add(
                "We modeled correlated noise in "
                + join_names(insts)
                + f" with a Gaussian process ({desc}), computed with "
                r"celerite2 \citep{ForemanMackey:2017, ForemanMackey:2018}.",
                section="noise",
                key=f"{self.prefix}.gp.{'_'.join(terms)}",
                rank=30,
            )
            prose.add_software("celerite2")

        robust = {"hogg": [], "studentt": []}
        for i, kind in enumerate(getattr(self, "likelihood_kinds", []) or []):
            if kind:
                robust[kind].append(names[i])
        if robust["hogg"]:
            prose.add(
                "For " + join_names(robust["hogg"]) + " we adopted a "
                "marginalized inlier/outlier Gaussian-mixture likelihood "
                r"\citep{Hogg:2010} in place of hard outlier rejection.",
                section="noise",
                key=f"{self.prefix}.robust.hogg",
                rank=40,
            )
        if robust["studentt"]:
            prose.add(
                "For " + join_names(robust["studentt"]) + " we adopted a "
                "Student's $t$ likelihood, the analytic marginalization "
                "of a per-point error inflation.",
                section="noise",
                key=f"{self.prefix}.robust.studentt",
                rank=40,
            )

    def add_observation_likelihood(
        self, name, mu, sigma, observed, system=None
    ):
        """Stage 6: the observational likelihood, with or without GPs and
        robust families.

        Drop-in replacement for the ``pm.Normal(name, mu, sigma, observed)``
        each child used to write.  With no ``gp:`` or ``likelihood:`` key
        anywhere it *is* that call, unchanged.  Otherwise the observations
        split by instrument: the plain files keep one shared Normal, each GP
        file gets a celerite2 marginal likelihood, and each robust file gets
        its mixture Potential or Student-t -- all around the same physical
        model ``mu`` and the same ``sigma`` (data error plus jitter/err_scale,
        which stays the inlier scale under the mixture).

        ``mu``, ``sigma`` and ``observed`` are (n_total_obs,) tensors in the
        concatenated order the child built; the per-file sort recorded by
        ``_prepare_gp`` is applied here.

        ``system``, when given, lets this one dispatcher declare the
        modeling-draft prose for exactly the likelihood it is building
        (data set inventory, noise model, GP kernels, robust families) --
        the declare-at-site rule with a single site per child.
        """
        if system is not None:
            self._add_observation_prose(system)
        if not (self.has_gp or self.has_robust_likelihood):
            return pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        if self.has_gp and self._gp_time is None:
            raise RuntimeError(
                f"[{self.prefix}] a 'gp:' key is configured but load_data "
                f"never called _prepare_gp; the GP times are unknown."
            )
        if self.has_robust_likelihood and not self._robust_obs_index:
            raise RuntimeError(
                f"[{self.prefix}] a 'likelihood:' key is configured but "
                f"load_data never called _prepare_robust; the per-file "
                f"observation indices are unknown."
            )

        special = np.zeros(self.n_total_obs, dtype=bool)
        for idx in self._gp_obs_index.values():
            special[idx] = True
        for idx in self._robust_obs_index.values():
            special[idx] = True

        plain = np.flatnonzero(~special)
        if plain.size:
            pm.Normal(
                name,
                mu=mu[plain],
                sigma=sigma[plain],
                observed=observed[plain],
            )

        self._add_robust_likelihoods(name, mu, sigma, observed)

        if not self.has_gp:
            return

        self._build_gp_deterministics()
        for i in sorted(self._gp_obs_index):
            idx = self._gp_obs_index[i]
            t = gp_support.check_sorted(
                self._gp_time[idx], context=f"{self.prefix}[{self.names[i]}]"
            )
            self.gp_objects[i] = gp_support.marginal_likelihood(
                f"{name}.gp.{self.names[i]}",
                self._gp_kernel(i),
                t=t,
                yerr=sigma[idx],
                mean=mu[idx],
                observed=observed[idx],
            )
            # Kept for the plotting path: predicting the GP component needs
            # the same observations the likelihood conditioned on, in the same
            # sorted order.
            self._gp_observed_node[i] = observed[idx]
            logger.info(
                "[%s] %s: %s GP on %d observations.",
                self.prefix,
                self.names[i],
                "+".join(self.gp_terms[i]),
                idx.size,
            )

    # ------------------------------------------------------------------
    # GP conditional prediction, for plotting
    # ------------------------------------------------------------------
    def _compile_gp_plotters(self, system):
        """Compile per-file evaluators of the GP conditional mean.

        Call from a child's ``compile_plotters``.  Builds two compiled
        functions per GP file, both of the *pure GP component*
        (``include_mean=False``, so celerite2 subtracts the physical model the
        likelihood conditioned on and returns only the correlated part):

          ``_gp_pred_at_data[i]``  the GP at that file's own observation
              times -- what plots subtract from the data.  Reuses the
              likelihood's factorization, so it costs about one extra logp
              evaluation.
          ``_gp_pred_on_grid[i]``  the GP on an arbitrary (sorted) time grid
              -- what plots add to a model curve.  Needs celerite2's
              general_matmul ops, O((N+M) J^2).

        Both take the same positional ``system.plot_params`` the other
        compiled plotters take, so a caller can evaluate them per posterior
        draw exactly like the physical model curves.
        """
        if not self.gp_objects:
            return
        import pytensor

        t_input = pt.vector("gp_t_input")
        param_symbols = [p.value for p in system.plot_params]

        self._gp_pred_at_data = {}
        self._gp_pred_on_grid = {}
        for i, gp in self.gp_objects.items():
            y = self._gp_observed_node[i]
            self._gp_pred_at_data[i] = pytensor.function(
                inputs=param_symbols,
                outputs=gp.predict(y, include_mean=False),
                on_unused_input="ignore",
            )
            self._gp_pred_on_grid[i] = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=gp.predict(y, t=t_input, include_mean=False),
                on_unused_input="ignore",
            )

    def has_gp_plotters(self):
        """True when _compile_gp_plotters has produced evaluators."""
        return bool(getattr(self, "_gp_pred_at_data", None))

    def gp_mean_at_data(self, system, point):
        """GP component at every observation, in the concatenated data order.

        Returns an ``(n_total_obs,)`` array that is zero for observations from
        files without a GP, so a caller can subtract it unconditionally.  In
        the internal units of the observable (the same units ``mu`` was in).
        """
        out = np.zeros(self.n_total_obs, dtype=float)
        if not self.has_gp_plotters():
            return out
        values = self._point_to_plot_params(point, system)
        for i, fn in self._gp_pred_at_data.items():
            out[self._gp_obs_index[i]] = np.asarray(fn(*values), dtype=float)
        return out

    def gp_mean_on_grid(self, system, point, i, t_grid):
        """GP component for file ``i`` on ``t_grid`` (must be sorted ascending).

        Zeros when that file has no GP, so a caller can add it
        unconditionally.  Internal units of the observable.
        """
        t_grid = np.asarray(t_grid, dtype=float)
        if not self.has_gp_plotters() or i not in self._gp_pred_on_grid:
            return np.zeros_like(t_grid)
        values = self._point_to_plot_params(point, system)
        return np.asarray(
            self._gp_pred_on_grid[i](t_grid, *values), dtype=float
        )

    # ------------------------------------------------------------------
    # Optional per-file robust likelihood (hogg mixture / Student-t)
    # ------------------------------------------------------------------
    def _load_likelihood_config(self):
        """Read the per-instrument ``likelihood:`` key (stage 0, in __init__).

        Populates ``self.likelihood_kinds[i]`` -- ``""`` (plain Gaussian, the
        default for every file) or a key from
        ``likelihood.LIKELIHOOD_KINDS`` -- and ``self.has_robust_likelihood``.
        Runs after ``_load_gp_config`` so the gp/likelihood conflict is
        caught at construction.
        """
        self.likelihood_kinds = []
        for i, c in enumerate(self.config):
            label = f"{self.prefix}[{self.names[i]}]"
            kind = robust_support.parse_likelihood_spec(
                c.get("likelihood"), context=label
            )
            if kind and not self.supports_robust_likelihood:
                raise NotImplementedError(
                    f"[{label}] a robust likelihood is not supported by this "
                    f"component: it models more than one observable per data "
                    f"file (with different units), so a single outlier scale "
                    f"is ambiguous. Remove the 'likelihood:' key."
                )
            if kind and self.gp_terms[i]:
                raise ValueError(
                    f"[{label}] 'gp:' and 'likelihood:' cannot be combined "
                    f"on one file: celerite2's closed-form marginal "
                    f"likelihood is Gaussian-only, so a robust observation "
                    f"model cannot be marginalized through it. Choose one."
                )
            self.likelihood_kinds.append(kind)
        self.has_robust_likelihood = any(self.likelihood_kinds)

        # Element index -> indices into the concatenated observation arrays
        # (filled by _prepare_robust), the per-file symbolic outlier log-odds
        # (filled by _add_robust_likelihoods), their lazily compiled
        # evaluators, and the linear values of log-sampled parameters.
        self._robust_obs_index = {}
        self._hogg_logodds = {}
        self._hogg_prob_fns = None
        self._robust_linear = {}

    def _robust_elements(self, kind):
        """Element indices that requested likelihood family ``kind``."""
        return [i for i, k in enumerate(self.likelihood_kinds) if k == kind]

    @staticmethod
    def _likelihood_config_schema():
        """The shared ``likelihood`` config-schema entry; children append it."""
        return robust_support.likelihood_config_schema_entry()

    def _prepare_robust(self, err, inst_map, user_factor=1.0):
        """Stage 1a: index the robust files and seed the mixture scale.

        Call at the end of ``load_data``, right next to ``_prepare_gp``.
        Records each opted-in element's indices into the concatenated
        observation arrays (no sort needed -- both families are per-point
        independent), and pushes a data-driven hint for the hogg background
        scale: ``10 x median(err)``, i.e. well clear of the inlier scatter,
        so the two mixture components start separated and cannot swap roles
        during tuning.  Seeding from the observations' own scatter would
        instead measure the physical signal (the same reasoning as the GP
        amplitude seed).

        ``user_factor`` converts the error from the internal unit the caller
        holds it in to the user unit ``out_scale`` is declared in.
        """
        if not self.has_robust_likelihood:
            return

        err = np.asarray(err, dtype=float)
        inst_map = np.asarray(inst_map, dtype=int)

        for i, kind in enumerate(self.likelihood_kinds):
            if not kind:
                continue
            sel = np.flatnonzero(inst_map == i)
            if sel.size == 0:
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] a 'likelihood:' key "
                    f"is set but the file contributed no observations."
                )
            self._robust_obs_index[i] = sel

            scale_param = robust_support.LIKELIHOOD_SCALE_PARAM.get(kind)
            if scale_param is None:
                continue
            scale = 10.0 * float(np.median(err[sel])) * user_factor
            if not np.isfinite(scale) or scale <= 0.0:
                # Degenerate (zero/absent) errors: keep the defaults.yaml
                # start rather than pinning the logit against its bound.
                continue
            path = f"{self.prefix}.{i}.{scale_param}"
            self.config_manager.add_hint(path, scale)
            self.config_manager.add_scale_hint(path, scale)

    def _register_robust(self, manifest):
        """Stage 2: add this component's robust-likelihood parameters.

        Same shape as ``_register_gp``: every parameter is a full-length
        (``n_elements``) vector so user paths resolve by instrument name, and
        elements that did not opt into a family are pinned fixed
        (``sigma: 0``) through ``internal_overrides`` -- free to the sampler,
        still user-overridable.  Returns ``manifest`` for chaining.
        """
        if not self.has_robust_likelihood:
            return manifest

        for kind in robust_support.LIKELIHOOD_KINDS:
            on = set(self._robust_elements(kind))
            if not on:
                continue
            off = [i for i in range(self.n_elements) if i not in on]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            for param in robust_support.LIKELIHOOD_PARAMS[kind]:
                manifest[param] = dict(entry)
        return manifest

    def _build_robust_deterministics(self):
        """Record the linear value of every log-sampled robust parameter.

        ``t_log_nu`` is sampled in log10 (see components/likelihood.py); this
        exposes ``t_nu`` as a Deterministic so posterior tables report the
        degrees of freedom the likelihood actually uses.
        """
        self._robust_linear = self._build_log10_deterministics(
            robust_support.LIKELIHOOD_LOG_PARAMS
        )

    def _add_robust_likelihoods(self, name, mu, sigma, observed):
        """Add each robust file's likelihood term (from the dispatcher)."""
        if not self.has_robust_likelihood:
            return

        self._build_robust_deterministics()
        for i in sorted(self._robust_obs_index):
            idx = self._robust_obs_index[i]
            kind = self.likelihood_kinds[i]
            if kind == "hogg":
                resid = observed[idx] - mu[idx]
                logp = robust_support.hogg_logp(
                    resid,
                    sigma[idx],
                    self.out_frac.value[i],
                    self.out_scale.value[i],
                )
                pm.Potential(f"{name}.hogg.{self.names[i]}", pt.sum(logp))
                self._hogg_logodds[i] = robust_support.hogg_outlier_logodds(
                    resid,
                    sigma[idx],
                    self.out_frac.value[i],
                    self.out_scale.value[i],
                )
            else:  # "studentt"
                pm.StudentT(
                    f"{name}.t.{self.names[i]}",
                    nu=self._robust_linear["t_nu"][i],
                    mu=mu[idx],
                    sigma=sigma[idx],
                    observed=observed[idx],
                )
            logger.info(
                "[%s] %s: %s likelihood on %d observations.",
                self.prefix,
                self.names[i],
                kind,
                idx.size,
            )

    def outlier_prob_at_data(self, system, point):
        """Posterior outlier probability of every observation, at ``point``.

        Returns an ``(n_total_obs,)`` array in the concatenated data order:
        ``sigmoid`` of the hogg mixture's per-point log-odds for observations
        from hogg files, and zero everywhere else, so a caller can threshold
        or color by it unconditionally.  This is the auditable replacement
        for a hard bad-data mask -- evaluate it per posterior draw (it is
        compiled lazily against ``system.plot_params``, like the GP
        evaluators) and report the average.
        """
        out = np.zeros(self.n_total_obs, dtype=float)
        if not self._hogg_logodds:
            return out
        if self._hogg_prob_fns is None:
            import pytensor

            param_symbols = [p.value for p in system.plot_params]
            self._hogg_prob_fns = {
                i: pytensor.function(
                    inputs=param_symbols,
                    outputs=pt.sigmoid(node),
                    on_unused_input="ignore",
                )
                for i, node in self._hogg_logodds.items()
            }
        values = self._point_to_plot_params(point, system)
        for i, fn in self._hogg_prob_fns.items():
            out[self._robust_obs_index[i]] = np.asarray(
                fn(*values), dtype=float
            )
        return out

    # NOTE: a GP imposes NO sampler constraint, so this class deliberately
    # does not override sampler_requirements().  celerite2 ships a JAX
    # implementation of every op it uses and registers it with PyTensor's JAX
    # linker (``@jax_funcify.register(_CeleriteOp)`` in celerite2/pymc/ops.py,
    # mapping factor/solve/matmul and their _rev gradients onto
    # celerite2.jax.ops primitives), so the JAX-funcifying samplers work.
    # Verified 2026-07-24 on celerite2 0.3.3 + pymc 6.1 + pytensor 3.1: the
    # JAX logp matches the C backend to 1e-14, its gradient is finite, and
    # nuts_sampler='numpyro' samples the full RV+GP model on 1, 2 and 4
    # chains.  ('blackjax' is separately broken by pymc passing a progress_bar
    # kwarg blackjax 1.6.2's kernel does not accept, but that fails
    # identically with no GP in the model, so it is not this component's
    # constraint to declare -- see exozippy/compat/blackjax_progressbar.py.)
    #
    # Before adding a sampler exclusion anywhere, SAMPLE with the sampler.
    # This class briefly carried one written by analogy with transit's, which
    # was wrong for celerite2; transit's own turned out to be fixable rather
    # than fundamental, and is gone too (exoplanet-core >= 0.4.0 makes the
    # limb-darkening op JAX-differentiable).

    # ------------------------------------------------------------------
    # Shared noise machinery
    # ------------------------------------------------------------------
    @staticmethod
    def _jitter_floor(err, factor=1.0):
        """Additive jitter-variance floor: ``-0.95 * min(err*factor)**2``.

        Keeps the total variance ``err**2 + jitter_variance`` strictly
        positive.  ``factor`` converts the raw error column to the internal
        unit first (rv scales to m/s; transit/astrometry use factor 1).

        This is a **validity** bound, not a preference: at exactly
        ``-min(err)**2`` the best-measured point's sigma is zero (infinite
        logp), and below it ``total_sigma``'s ``sqrt`` is NaN for that point
        and its gradient with it.  The 0.95 keeps 5% of the variance as
        margin.  Hence it is applied as a floor the user may tighten but not
        loosen -- see ``_register_noise``.
        """
        scaled_min = np.min(np.asarray(err, dtype=float)) * factor
        return -0.95 * scaled_min**2

    def _register_noise(self, manifest, jittervar_lower=None):
        """Add the shared per-instrument noise term to ``manifest``.

        ``"jitter_variance"``: additive variance (needs a per-instrument
        ``jittervar_lower`` floor) plus a reported ``jitter``.
        ``"err_scale"``: a single multiplicative scale.
        Returns ``manifest`` for chaining.

        The computed floor goes through the ``"overrides"`` channel, NOT the
        plain manifest options: options are merged as ``{**cfg, **options}``
        and so REPLACE the resolved array, which silently discarded a user's
        explicit ``jitter_variance: {lower: ...}``.  ``"overrides"`` are
        applied inside ``ConfigManager.resolve``, which special-cases bounds
        (``lower`` -> ``max``, ``upper`` -> ``min``), so the resolved lower is
        ``max(defaults.yaml, floor, user)`` regardless of application order:
        the user can tighten the bound freely, and can only be clipped by the
        floor, which is a hard validity limit (see ``_jitter_floor``) -- below
        it the likelihood is NaN, so there is nothing there to sample.  That
        clip is warned about, not silent (ConfigManager.resolve).
        """
        if self.noise_model == "err_scale":
            manifest["err_scale"] = None
        else:
            if jittervar_lower is None:
                raise ValueError(
                    f"[{self.prefix}] jitter_variance noise model requires a "
                    f"jittervar_lower floor."
                )
            manifest["jitter_variance"] = {
                "overrides": {"lower": jittervar_lower}
            }
            manifest["jitter"] = "default"
        return manifest

    def total_sigma(self, err_tensor):
        """Per-observation sigma combining data error with the instrument's
        noise term, mapped over ``inst_map_tensor``.

        Requires the per-observation ``inst_map`` machinery (rv/transit/
        mulens); astrometry, whose data are stored per dataset, forms its
        sigma inline instead.
        """
        if self.noise_model == "err_scale":
            return err_tensor * self.err_scale.value[self.inst_map_tensor]
        return pt.sqrt(
            pt.sqr(err_tensor)
            + self.jitter_variance.value[self.inst_map_tensor]
        )

    # ------------------------------------------------------------------
    # Shared optional detrending against extra data columns
    # ------------------------------------------------------------------
    @staticmethod
    def _build_block_detrend(all_detrend, n_total_obs):
        """Block-diagonal detrend design matrix from per-instrument blocks.

        ``all_detrend`` is one ``(n_obs_i, n_cols_i)`` array per instrument
        (``n_cols_i == 0`` when that instrument has no detrend columns).
        Returns ``(matrix, n_detrend_per_inst, total_detrend_cols)`` where the
        matrix is ``(n_total_obs, total_detrend_cols)`` with each instrument's
        columns placed on the block diagonal so coefficients never mix across
        instruments.
        """
        n_per = [d.shape[1] for d in all_detrend]
        total = sum(n_per)
        matrix = np.zeros((n_total_obs, total))
        r, c = 0, 0
        for block in all_detrend:
            n_r, n_c = block.shape
            if n_c > 0:
                matrix[r : r + n_r, c : c + n_c] = block
            r, c = r + n_r, c + n_c
        return matrix, n_per, total
