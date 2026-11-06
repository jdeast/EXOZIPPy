"""The Hipparcos-to-Gaia frame tie of one astrometric dataset (`fithipframe`).

Epoch-level Hipparcos astrometry (the Intermediate Astrometric Data, fit in
``mode: gaia`` as examples/HIP1349 does) and Gaia epoch astrometry
(``data_format: gaia_epoch``) of ONE star fit jointly give the 25-year
Hipparcos-Gaia lever arm the Hipparcos-Gaia Catalog of Accelerations
(Brandt 2018, 2021) is built on -- done from the epochs rather than from
three catalog proper motions.  The two missions' reference frames are not
identical, and a frame difference reads exactly as a position or
proper-motion change of the star between 1991 and 2017, i.e. as the
acceleration the joint fit exists to measure.  So the Hipparcos dataset
carries the frame difference as parameters with priors set by its measured
size, and every other dataset defines the frame.

The model.  A frame differs from another by an orientation offset and a
spin: ``Delta u = eps(t) x u``, ``eps(t) = eps0 + omega (t - t0)``.  For ONE
star the rotation projects onto the local tangent plane as a position offset
that is linear in time (Lindegren et al. 2021b, Sect. 4.5), so on the flagged
dataset every modeled position moves by

    dE(t) += frame_dE + frame_dpmE * (t - J1991.25)/yr
    dN(t) += frame_dN + frame_dpmN * (t - J1991.25)/yr

-- the Hipparcos frame minus the frame of the other datasets, AT the star,
East (true arc) and North, at the Hipparcos catalog epoch J1991.25 (where
the Hipparcos frame was aligned to the ICRS; Kovalevsky et al. 1997).  The
four local terms, not the six global rotation parameters: for one star the
two are the same model (the projection of an isotropic rotation prior is
isotropic on the tangent plane: the rows of the projection are orthonormal),
and two of the six global components are unconstrained by one star anyway.
A GLOBAL rotation shared by several stars is the multi-star extension; it
needs the 3+3 rotation vector projected per star instead.

The default priors (defaults.yaml; a params entry overrides them per
dataset) are zero-mean Gaussians:

  * frame_dE, frame_dN: sigma 0.3 mas.  Gaia EDR3's bright reference frame
    was aligned to the Hipparcos positions at J1991.25 (Lindegren et al.
    2021b, Sect. 4.5), so the GLOBAL orientation difference is ~0 by
    construction, and Brandt (2021, Sect. 6) finds no positional error
    inflation needed on the Hipparcos positions, with an extra 0.3 mas
    indistinguishable from none -- the local systematic floor adopted here.
    If a Gaia release is NOT so aligned, the bound is the Hipparcos frame's
    own +/-0.6 mas per axis at J1991.25 (Kovalevsky et al. 1997).
  * frame_dpmE, frame_dpmN: sigma 0.28 mas/yr, the quadrature sum of the
    local corrections Brandt (2021, Fig. 3) applies to the Hipparcos proper
    motions to bring them into the Gaia EDR3 frame (standard deviation 0.2
    mas/yr per coordinate) and the 0.2 mas/yr additive error inflation
    Brandt (2018, Sect. 6; adopted again in 2021, Sect. 4.3) finds the
    Hipparcos proper motions need.  For one star an additive proper-motion
    error is indistinguishable from a proper-motion frame offset, so both
    belong in this one prior.  The global Hipparcos-vs-ICRS spin is +/-0.25
    mas/yr per axis (ESA 1997).

These are calibrations of the HGCA's 60/40 mixture of the two Hipparcos
reductions; the 1997 IAD this codebase reads (getdata --hipparcos-iad) is
one of the two, and its own per-star calibration is not published.  A
per-epoch error inflation is the dataset's ordinary ``jitter_variance``.

Legal only on a ``mode: gaia`` or ``mode: abs`` dataset that is not Gaia
itself (``data_format: gaia_epoch`` and ``observer_location: gaia`` raise:
the Gaia datasets are the reference).  INACTIVE on every other dataset (the
fluxfrac pattern), and declared only when some dataset has the terms, so a
fit without them has no such rows and an unchanged graph.

The default.  An explicit ``fithipframe: true``/``false`` always wins.
Without one, the terms are ON for a dataset that IS Hipparcos epoch
astrometry and is fit together with another absolute dataset that defines
the frame, and OFF otherwise.  "Is Hipparcos" is decided from the data,
not from a file name or a comment line:

  * ``mode: gaia`` (one-dimensional along-scan abscissae with scan angles),
  * not observed by Gaia (``observer_location`` is not ``gaia``; a
    gaia_epoch file is Gaia by construction), and
  * EVERY epoch inside the Hipparcos mission, 1989 Nov 1 - 1993 Apr 1
    (JD 2447831.5 - 2449078.5; observations ran 1989 Nov - 1993 Mar,
    ESA 1997 Vol. 1).

Hipparcos was the only mission that delivered along-scan abscissae before
Gaia, so nothing else satisfies all three; ground-based astrometry of the
same years is two-dimensional (``mode: abs``) and is never defaulted on.
"Another absolute dataset that defines the frame" is any gaia/abs/catalog
dataset that is not itself Hipparcos by that test.  A Hipparcos dataset
fit alone (examples/HIP1349) has nothing to tie to, so the default leaves
it off (and says so); an explicit ``true`` there is honored with the
warning in ``check_frame_reference``.
"""

import logging

import numpy as np

from exozippy.constants import DAYS_PER_YEAR

logger = logging.getLogger(__name__)

# The Hipparcos catalog epoch J1991.25 (TT), as a JD: the epoch of the
# frame terms' position offset.
HIPPARCOS_EPOCH_JD = 2448349.0625

# The per-dataset parameters, in the order (position E, N; rate E, N).
FRAME_PARAMS = ("frame_dE", "frame_dN", "frame_dpmE", "frame_dpmN")

# The flag that turns them on.  A frame tie to some OTHER catalog's frame
# would be a sibling boolean (exclusive with this one, the repo's rule for
# n-way choices), with its own priors.
FRAME_FLAG = "fithipframe"
FRAME_MODES = ("gaia", "abs")
# The Hipparcos mission, padded to whole months: the data window that makes
# a mode-gaia dataset "Hipparcos" for the default (see the module docstring).
HIPPARCOS_WINDOW_JD = (2447831.5, 2449078.5)


def parse_frame_flags(inst):
    """``fithipframe:`` per dataset of AstrometryInstrument ``inst``
    (at construction, after its data formats are parsed).  Returns a list
    of True / False / None (unset: resolved from the data by
    ``resolve_frame_defaults`` once the epochs are read); raises, naming the
    dataset, on a non-boolean value or a dataset that cannot be in the
    Hipparcos frame."""
    flags = []
    for i, c in enumerate(inst.config):
        where = f"[{inst.prefix}.{inst.names[i]}]"
        flag = c.get(FRAME_FLAG)
        if flag is None:
            flags.append(None)
            continue
        if not isinstance(flag, bool):
            raise ValueError(
                f"{where} {FRAME_FLAG} must be true or false; got {flag!r}."
            )
        if flag:
            if inst.modes[i] not in FRAME_MODES:
                raise ValueError(
                    f"{where} {FRAME_FLAG}: true puts this dataset's "
                    f"positions in the Hipparcos frame, a statement about "
                    f"epoch astrometry (mode gaia or abs); this dataset "
                    f"is mode '{inst.modes[i]}'."
                )
            if inst.data_formats[i] == "gaia_epoch":
                raise ValueError(
                    f"{where} {FRAME_FLAG}: true on Gaia epoch astrometry: "
                    f"the Gaia datasets DEFINE the frame the Hipparcos one "
                    f"is tied to.  Set it on the Hipparcos dataset."
                )
            if inst.observers[i] == "gaia":
                raise ValueError(
                    f"{where} {FRAME_FLAG}: true on a dataset observed by "
                    f"the Gaia spacecraft (observer_location: gaia): Gaia "
                    f"defines the frame.  Hipparcos epoch astrometry is "
                    f"observer_location: earth."
                )
        flags.append(flag)
    return flags


def is_hipparcos(inst, i):
    """Whether dataset ``i`` (loaded) is Hipparcos epoch astrometry by the
    module docstring's three tests."""
    if inst.modes[i] != "gaia" or inst.observers[i] == "gaia":
        return False
    if inst.data_formats[i] == "gaia_epoch":
        return False
    t = np.asarray(inst.datasets[i]["time"], dtype=float)
    lo, hi = HIPPARCOS_WINDOW_JD
    return bool(len(t)) and bool(np.all((t >= lo) & (t <= hi)))


def resolve_frame_defaults(inst):
    """Replace each unset (None) flag by the default (module docstring),
    after ``load_data`` has read every dataset's epochs.  Logs each
    decision the default made."""
    hip = [is_hipparcos(inst, i) for i in range(inst.n_elements)]
    frame_defined = [
        m in ("gaia", "abs", "catalog") and not hip[i]
        for i, m in enumerate(inst.modes)
    ]
    out = []
    for i, flag in enumerate(inst.fithipframe):
        if flag is not None:
            out.append(flag)
            continue
        on = hip[i] and any(frame_defined)
        where = f"[{inst.prefix}.{inst.names[i]}]"
        if on:
            logger.info(
                f"{where} Hipparcos epoch astrometry (mode gaia, every epoch "
                f"in the Hipparcos mission) fit with "
                f"{[inst.names[j] for j, f in enumerate(frame_defined) if f]}"
                f": {FRAME_FLAG} defaults to true, the Hipparcos frame's "
                f"offset and spin from theirs are fit.  Set {FRAME_FLAG}: "
                f"false to take the frames as identical."
            )
        elif hip[i]:
            logger.info(
                f"{where} Hipparcos epoch astrometry with no other absolute "
                f"dataset to define a frame: {FRAME_FLAG} defaults to false "
                f"(the frame terms would be fixed by their priors alone)."
            )
        out.append(on)
    return out


def check_frame_reference(inst):
    """Warn when every absolute dataset carries the frame terms: then
    nothing defines the frame they tie to, and the terms are constrained
    by their priors alone (degenerate with the star's position and proper
    motion)."""
    flags = inst.fithipframe
    if not any(flags):
        return
    others = [
        i
        for i, m in enumerate(inst.modes)
        if m in ("gaia", "abs", "catalog") and not flags[i]
    ]
    if not others:
        logger.warning(
            f"[{inst.prefix}] {FRAME_FLAG}: true on "
            f"{[inst.names[i] for i, f in enumerate(flags) if f]}, and no "
            f"other absolute dataset defines the frame they are tied to: "
            f"the frame terms are then fixed by their priors alone and "
            f"degenerate with the star's position and proper motion."
        )


def frame_offset(t, dE0, dN0, dpmE, dpmN):
    """The frame terms' (dE, dN) [mas] at BJD_TDB times ``t``, numpy or
    pytensor alike (the likelihood and the plots call this one formula)."""
    dt_yr = (t - HIPPARCOS_EPOCH_JD) / DAYS_PER_YEAR
    return dE0 + dpmE * dt_yr, dN0 + dpmN * dt_yr


def add_frame_prose(inst, system):
    """Modeling-draft sentence for the frame tie, only when a dataset
    carries it (outputs.md: declared next to the model)."""
    tied = [i for i, f in enumerate(inst.fithipframe) if f]
    if not tied:
        return
    from exozippy.outputs.prose import get_collector, join_names
    from exozippy.outputs.texutils import latex_escape

    names = join_names([latex_escape(inst.names[i]) for i in tied])
    get_collector(system).add(
        rf"The {names} astrometry is in the Hipparcos reference frame, "
        r"which differs from the Gaia frame of the other astrometry by an "
        r"orientation offset and a spin \citep{Kovalevsky:1997, "
        r"Lindegren:2021b}; for one star these project onto a sky-plane "
        r"position offset at J1991.25 and a proper-motion offset, which we "
        r"fit with Gaussian priors of 0.3~mas and 0.28~mas~yr$^{-1}$ per "
        r"coordinate set by the Hipparcos--Gaia cross-calibration of "
        r"\citet{Brandt:2018, Brandt:2021}.",
        section="data",
        key=f"{inst.prefix}.frame_tie",
        rank=41,
    )


def numpy_frame_offset(inst, d, t, point):
    """``frame_offset`` from a point's values, for the numpy plot path
    (``AstrometryInstrument._linear_terms``); zeros on a dataset without
    the flag."""
    i = d["index"]
    if not inst.fithipframe[i]:
        z = np.zeros(np.shape(t), dtype=float)
        return z, z.copy()
    vals = [
        inst._point_value(point, getattr(inst, p), i) for p in FRAME_PARAMS
    ]
    return frame_offset(np.asarray(t, dtype=float), *vals)
