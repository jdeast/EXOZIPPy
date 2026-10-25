"""A star's sky position as a stage 1-3 reader must see it.

Several components need a star's (ra, dec) before stage 4 -- the BJD_TDB
conversion's line of sight, the microlensing frame, the astrometry tangent
point.  ``ConfigManager.resolve()`` is the wrong reader for that: it shows
only what was WRITTEN for the path, so a position given as a LINK
(``star.Source.ra: {initval: star.Lens.ra}``) -- links are applied at stage
4 -- read as the defaults.yaml placeholder (180, 0) deg, with
``user_modified`` True because the link's remaining fields stay in
``user_params``.  Every reader then ran silently against the placeholder
(review 1.1.10).

WHY NOT ``probe_start``.  config.md's sanctioned stage 1-3 reader runs the
whole relaxation engine, and the first of these readers runs before the
engine CAN run: the microlensing line of sight is what
``MulensEvent.register_frame_constants`` needs to hand the engine Earth's
projected velocity, without which the geocentric proper-motion relations
have no scale and the probe raises (every microlensing acceptance fixture
did).  A position is never DERIVED by a relation at stage 1 -- it is a
number the user wrote, or a link to other such numbers -- so this module
reads exactly that: the user's own initval (or mu), else the link
expression evaluated on its dependencies' starts, recursively.
"""

import numpy as np

from exozippy.config import user_entry

# A start the user states.  resolve() starts a mu-only entry at its mu.
_START_FIELDS = ("initval", "mu")


def _user_start(config_manager, path, chain=()):
    """`path`'s start in its USER unit, or None when nothing informs it.

    The user's numeric initval/mu for the path, else a link on one of those
    fields evaluated on its dependencies' own starts (links store their
    expression in the dependencies' USER units, ``linking.ParamLink``).
    """
    if path in chain:
        raise ValueError(
            f"star position links form a cycle: {' -> '.join(chain + (path,))}"
        )
    comp, idx, param = path.split(".")
    entry = user_entry(config_manager.user_params, path)
    links = config_manager.links.get(path, {})
    for fld in _START_FIELDS:
        if entry is not None and fld in entry:
            # resolve() applies the user's spelling precedence and the
            # seed-0 rule, in the user's unit; it returns a one-element
            # array even for one element.
            r = config_manager.resolve(comp, param, element=int(idx))
            return float(np.ravel(r["initval"])[0])
        link = links.get(fld)
        if link is not None:
            subs = {}
            for dep in link.dep_paths:
                val = _user_start(config_manager, dep, chain + (path,))
                if val is None:
                    return None
                subs[dep] = val
            symbols = {s.name: s for s in link.expr.free_symbols}
            value = link.expr.subs({symbols[d]: v for d, v in subs.items()})
            return float(value)
    return None


def probe_star_radec(config_manager, indices, prefix="star"):
    """Each star's start (ra, dec) in RADIANS, or None when uninformed.

    Returns ``{index: (ra_rad, dec_rad)}`` for a star whose ra AND dec are
    stated by the user (a number, or a link to stated numbers), and
    ``{index: None}`` for one whose position would be the defaults.yaml
    placeholder.  The caller decides what an uninformed position means for
    it -- raise, or skip an optional seed -- and never uses the placeholder.
    """
    out = {}
    for i in sorted({int(i) for i in indices}):
        pos = []
        for c in ("ra", "dec"):
            path = f"{prefix}.{i}.{c}"
            val = _user_start(config_manager, path)
            if val is None:
                break
            # INTERNAL units are radians (star/defaults.yaml internal_unit)
            pos.append(config_manager.to_internal(path, val))
        out[i] = tuple(pos) if len(pos) == 2 else None
    return out


def require_star_radec(config_manager, index, who, prefix="star"):
    """One star's informed (ra, dec) in radians.

    Raises ValueError, starting with `who` (e.g. ``"[KMTC01] time
    conversion"``), when the position is uninformed.
    """
    pos = probe_star_radec(config_manager, [index], prefix=prefix)[int(index)]
    if pos is None:
        raise ValueError(
            f"{who} needs the sky position of {prefix}.{int(index)}, but "
            f"its ra/dec are not set (only the defaults.yaml placeholder "
            f"exists).  Give {prefix}.{int(index)}.ra and "
            f"{prefix}.{int(index)}.dec (deg) in the params file, or link "
            f"them to a star that has them."
        )
    return pos
