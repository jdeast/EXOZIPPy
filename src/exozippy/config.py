# src/exozippy/config.py
import copy
import dataclasses
import importlib
import logging
import math
import signal
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import sympy as sp
import yaml

logger = logging.getLogger(__name__)
import re

from exozippy.constants import SYMPY_SOLVE_TIMEOUT_S
from exozippy.linking import extract_links

# --- The per-parameter sub-key vocabulary --------------------------------
#
# These are THE sub-keys a params.yaml entry may carry, i.e. exactly what
# ConfigManager.resolve() below absorbs from a user override dict.  They live
# at module scope because three places need the same answer and used to
# restate it: resolve() itself, diagnostics.ModelAuditor.check_unused_yaml
# (which warns that anything else "did not match any model parameter"), and
# introspect._parameter_entry (which exposes the numeric fields to the GUI).
# The copies drifted -- diagnostics was missing latex/description/
# print_to_table/debug_print and so reported four legitimate keys as typos,
# and introspect was missing bound_scale.  A false "ignored" warning is worse
# than silence: it teaches users to disbelieve the startup check.  Add a new
# sub-key here and to the loop in resolve() that consumes it, and every
# consumer follows.  tests/test_known_keys.py cross-checks the constants
# against what resolve() really reads, in both directions.

# Preliminary conditioning only, never a posterior term.  init_scale is no
# longer user-facing (ConfigManager strips it with a warning), but it stays in
# the vocabulary: resolve() still fills it from defaults/hints, and pre-2026-07
# mkprior restart files name it, so flagging it as a typo would be wrong.
TUNING_KEYS = ("initval", "init_scale")

# The two INTERVAL fields.  They are the one pair resolve() does not simply
# assign -- apply_value keeps the strictest of the two -- so the loops that
# have to treat them differently from every other field name them from here
# rather than spelling the pair out again (review 1.1.5).
BOUND_KEYS = ("lower", "upper")

# Keys that change the posterior.  bound_scale is one of them: it sets
# soft-bound barrier steepness, a real posterior term (unlike init_scale).
# A user entry touching any of these marks the parameter prior-modified.
PHYSICS_KEYS = BOUND_KEYS + ("mu", "sigma", "bound_scale")

# Every field resolve() reads as a number and scales by the element's unit.
NUMERIC_KEYS = TUNING_KEYS + PHYSICS_KEYS

# Passed through verbatim; per element when the parameter is a vector.
STRING_KEYS = ("unit", "latex", "description")

# Reporting switches; whole-parameter, not per element.
BOOL_KEYS = ("print_to_table", "debug_print")

# What justifies an entry's prior (e.g. the catalog a Gaia parallax came
# from); per element.  Each item is a references.bib key, rendered as a
# \citet, or any other text, passed through verbatim for the user to turn
# into a citation by hand.  Metadata, never a posterior term, so
# deliberately NOT in PHYSICS_KEYS: citing a prior does not modify it.
# Parameter renders it as a Prior-column table note.
CITATION_KEYS = ("citation",)

# The union: the complete set of legal sub-keys.
USER_PARAM_KEYS = NUMERIC_KEYS + STRING_KEYS + BOOL_KEYS + CITATION_KEYS


def normalize_citation(value, where):
    """Normalize a params-file ``citation`` to a tuple of citations.

    The one spelling inside the codebase is a tuple of stripped, non-empty
    strings.  A user writes one citation as a string or several as a list;
    a string is never split, since free text ("Smith, Jones & Lee 2023")
    carries commas.  Anything else raises naming ``where`` (the parameter
    path).
    """
    if isinstance(value, str):
        items = [value.strip()]
    elif isinstance(value, (list, tuple)) and all(
        isinstance(k, str) for k in value
    ):
        items = [k.strip() for k in value]
    else:
        raise ValueError(
            f"{where}: 'citation' must be a string or a list of strings "
            f"(references.bib keys or free text), got {value!r}"
        )
    if not items or not all(items):
        raise ValueError(f"{where}: 'citation' has an empty entry: {value!r}")
    return tuple(items)


class SymbolicTimeout(Exception):
    pass


def parse_unit(unit_str, where):
    """Parse a unit string strictly, or raise naming the offending string.

    An unparseable ``unit:`` used to be swallowed here (the conversion
    factor silently fell back to 1.0), which does not mean "no conversion"
    -- it means the user's number is reinterpreted in whatever the internal
    unit happens to be.  ``planet.b.mass: {initval: 1.0, unit: earthMasses}``
    (note the typo) became one SOLAR mass, a factor of 333000, with no
    message anywhere.  Same policy as ``rvinstrument._parse_rv_unit`` and
    ``Parameter.__post_init__``: an unrecognized unit is an error.
    """
    try:
        return u.Unit(unit_str)
    except Exception as exc:
        raise ValueError(
            f"[{where}] unit: {unit_str!r} is not a unit astropy can parse "
            f"(e.g. 'earthMass', 'jupiterMass', 'deg', 'm/s', 'd')."
        ) from exc


def unit_conversion(from_str, to_str, where):
    """Multiplier converting a value in ``from_str`` to ``to_str``.

    Both strings must parse and must be mutually convertible; either
    failure raises, naming ``where`` (the parameter path the unit came
    from).  Log-space units short-circuit to 1.0 exactly as they always
    have -- ``dex`` conversions are handled by the physics, not here -- but
    the strings are still validated first.
    """
    if from_str == to_str:
        return 1.0

    from_u = parse_unit(from_str, where)
    to_u = parse_unit(to_str, where)

    if "dex" in str(from_str) or "dex" in str(to_str):
        return 1.0

    try:
        return float(from_u.to(to_u))
    except Exception as exc:
        raise ValueError(
            f"[{where}] cannot convert '{from_str}' to '{to_str}': the two "
            f"units are not compatible.  Check the unit: key on this "
            f"parameter."
        ) from exc


import contextlib

# SIGALRM is POSIX-only -- Windows has no such signal, and touching
# signal.SIGALRM there raises AttributeError at import-adjacent call time.
# Guarding on the attribute rather than on sys.platform keeps this honest on
# any other platform that lacks it.
#
# The consequence on Windows is real and worth stating: the symbolic solver
# runs WITHOUT a wall-clock timeout. A pathological equation/target pair that
# would be abandoned after 2s on Linux can hang instead. The alternative --
# a thread-based timeout -- cannot actually interrupt sympy once it is down
# in C, so it would provide the appearance of a limit rather than a limit.
_HAS_SIGALRM = hasattr(signal, "SIGALRM")


def _arm_alarm(seconds, handler):
    """Arm a SIGALRM timeout. No-op (returns None) where SIGALRM is absent."""
    if not _HAS_SIGALRM:
        return None
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    return old_handler


def _disarm_alarm(old_handler=None):
    """Cancel a pending SIGALRM and optionally restore the prior handler."""
    if not _HAS_SIGALRM:
        return
    signal.alarm(0)
    if old_handler is not None:
        signal.signal(signal.SIGALRM, old_handler)


@contextlib.contextmanager
def _sympy_time_limit(seconds=SYMPY_SOLVE_TIMEOUT_S):
    """Hard wall-clock limit for a block of symbolic work.

    sp.solve (and evalf on its solutions) can hang effectively forever on
    certain equation/target pairs, and which pairs get attempted depends on
    hash-seed-sensitive iteration order -- so an unguarded call is a latent
    intermittent hang.  Raises SymbolicTimeout when the limit is hit.

    The handler re-arms the alarm before raising: if the exception fires
    while the interpreter is inside a C-level frame that discards it (seen
    in practice with JAX's gc callback -- "Exception ignored in
    _xla_gc_callback" -- after which the guarded solve ran unbounded), the
    next alarm gets another chance to land in interpretable bytecode.
    """

    def handler(signum, frame):
        signal.alarm(1)
        raise SymbolicTimeout()

    old_handler = _arm_alarm(seconds, handler)
    try:
        yield
    finally:
        _disarm_alarm(old_handler)


# Instance names appear as the middle part of dotted parameter paths
# (star.MyName.teff), so they must be safe to split on "." and must not
# collide with the internal index notation (star.0.teff).
_VALID_INSTANCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_instance_names(system_config):
    """Fatal-error check on user-supplied component instance names.

    Rejects names that would corrupt parameter-path parsing:
      - non-string names (YAML ``name: 128`` arrives as an int)
      - characters outside [A-Za-z0-9_-] ("." splits paths; whitespace,
        brackets, etc. break resolve()/mkparam key matching)
      - all-digit names, which alias the internal index notation
        (a component named "1" would silently resolve to instance index 1)
    """
    for comp_key, entries in (system_config or {}).items():
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            name = entry["name"]
            where = f"{comp_key}[{i}]"
            if not isinstance(name, str):
                raise ValueError(
                    f"Invalid name {name!r} for {where}: names must be strings. "
                    f'Quote it in your config YAML (name: "{name}").'
                )
            if not _VALID_INSTANCE_NAME.match(name):
                raise ValueError(
                    f"Invalid name '{name}' for {where}: names may only contain "
                    f"letters, digits, underscores, and hyphens. Characters like "
                    f"'.' or spaces would break parameter-path parsing "
                    f"(e.g. '{comp_key}.{name}.param')."
                )
            if name.isdigit():
                raise ValueError(
                    f"Invalid name '{name}' for {where}: purely numeric names "
                    f"collide with the internal index notation "
                    f"({comp_key}.0, {comp_key}.1, ...). Add a non-digit "
                    f'character (e.g. name: "{comp_key}_{name}").'
                )


def _sigma_is_zero(value):
    """True only if ``value`` is definitively zero (scalar or all-zero list).

    Anything unparseable -- a link expression string, None-free junk -- returns
    False, i.e. "treat as a real prior width", which is the conservative answer
    for validate_sigma_has_center.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        try:
            return all(float(v) == 0.0 for v in value)
        except (TypeError, ValueError):
            return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _reject_renamed_arsun(user_params):
    """Raise, with the fix, on the pre-rename ``arsun`` parameter spelling.

    The semi-major axis was renamed ``arsun`` -> ``a`` (2026-08: the name
    dated from EXOFASTv2's fixed internal units; with unit handling the
    user-facing value is AU and the name lied).  An unknown parameter path
    in a params file is otherwise SILENTLY ignored, so without this check
    an old file's ``orbit.b.arsun`` seed would simply stop doing anything
    -- a silent behavior change, the exact failure mode pointed errors on
    renames exist to prevent (compare planet.log_q's stale-restart-file
    raise).  Checked on the raw params (before standardization, so the
    message shows the user's own spelling) and inside link expressions.
    """
    for key, spec in (user_params or {}).items():
        if str(key) == "arsun" or str(key).endswith(".arsun"):
            raise ValueError(
                f"'{key}': the semi-major axis parameter was renamed "
                f"'arsun' -> 'a' (reported in AU; internally solRad). "
                f"Rename the entry to "
                f"'{str(key)[: -len('arsun')]}a'."
            )
        if isinstance(spec, dict):
            for field, value in spec.items():
                if isinstance(value, str) and "arsun" in value:
                    raise ValueError(
                        f"'{key}.{field}' links to '{value}': the "
                        f"semi-major axis parameter was renamed 'arsun' "
                        f"-> 'a' (reported in AU; internally solRad). "
                        f"Update the expression."
                    )


def _reject_list_valued_fields(user_params, source=None):
    """Raise on a list in a numeric field that has no per-seed meaning.

    A LIST means exactly one thing in a params file: per-seed start values for
    the multi-seed sampler, which is an ``initval`` concept only -- seeds move
    the START, never the bounds or the prior (``_build_seed_overrides`` reads
    initval lists and nothing else).  Everywhere else a list is a
    misunderstanding, most often "one value per element", which is spelled
    with one KEY per element (``star.0.teff`` / ``star.A.teff``), not with a
    list.

    It used to crash instead of explaining: the numeric loop in ``resolve()``
    tolerates lists by taking element 0 (the per-seed convention), so a
    ``{sigma: [10, 20]}`` sailed past the sigma application and died several
    frames later in ``apply_value``'s ``float(list)`` -- a bare TypeError
    naming no parameter, no field and no file.  Checked on the raw params,
    before standardization, so the message quotes the user's own spelling.

    ``init_scale`` is deliberately exempt: it is stripped with a warning a few
    lines later (it is no longer user-facing), and pre-2026-07 restart files
    name it, so turning an inert entry into a fatal one would break files that
    work today.
    """
    where = f" in {source}" if source else ""
    for key, spec in (user_params or {}).items():
        if not isinstance(spec, dict):
            continue
        for fld in PHYSICS_KEYS:
            val = spec.get(fld)
            if isinstance(val, (list, tuple)):
                parts = str(key).split(".")
                per_element = f"{parts[0]}.<instance>.{parts[-1]}"
                raise ValueError(
                    f"\n!!! LIST IN A NON-SEED FIELD !!!\n"
                    f"'{key}'{where} sets {fld}: {list(val)}.  A list is "
                    f"per-SEED start values, which only 'initval' takes -- "
                    f"seeds move the start, never the bounds or the prior.\n"
                    f"For one value per element, write one entry per element "
                    f"('{per_element}'); for one value everywhere, write the "
                    f"number itself."
                )


# ---------------------------------------------------------------------------
# Reserved NON-PARAMETER keys in a params file
# ---------------------------------------------------------------------------

#: Top-level keys a params file may carry that are NOT parameters.  They are
#: split off (``split_reserved_param_keys``) before any other reader sees the
#: mapping, because ``finalize_user_params`` registers every unmapped
#: ``user_params`` key as a LEAF SYMBOL in the relaxation engine -- see the
#: "never write into user_params" invariant in CLAUDE.md.  Measured, not
#: assumed: a bare ``overdisperse: false`` became the symbol ``overdisperse``,
#: entered ``_last_resolved``, and the engine INJECTED BACK
#: ``{'initval': 0.0, 'derived': True}`` over the user's own boolean.
#:
#: Value contract, one entry per key:
#:   overdisperse -- bool.  Do these seeds still want scattering before the
#:     chains start?  An ABSENT key means True (the safe direction for a
#:     hand-written file).  Read by samplers/_common.py's ``_make_starts``
#:     and by polish.resolve_polish_steps (a declared posterior-draw set is
#:     never auto-polished, review 2.4.14 (d)); the reasoning is in
#:     samplers/samplers.md, "Chain starts".
RESERVED_PARAM_KEYS = frozenset({"overdisperse"})


def split_reserved_param_keys(user_params, source=None):
    """Split a params mapping into (parameters, reserved non-parameter keys).

    Returns two NEW dicts and never mutates the caller's: ``System`` keeps the
    file exactly as read (``structural_payload`` and ``mkparam`` both read it
    again), so popping in place would quietly edit their input.

    An unknown top-level key with a SCALAR value is warned about BY NAME here
    rather than being carried on.  Every parameter path has a dot in it, so a
    dot-less scalar entry is either a reserved key or a mistake -- and the
    mistake is otherwise perfectly silent: it becomes a leaf symbol with its
    own ledger row and its own inject-back initval, for a parameter that does
    not exist.  A dot-less entry whose value is a DICT is left alone; only the
    scalar form is diagnosable this cheaply.
    """
    where = f" in {source}" if source else ""
    params, reserved = {}, {}
    for key, val in (user_params or {}).items():
        skey = str(key)
        if skey in RESERVED_PARAM_KEYS:
            reserved[skey] = val
            continue
        if "." not in skey and not isinstance(val, dict):
            logger.warning(
                f"'{skey}'{where} is not a parameter (a parameter path has a "
                f"dot in it: 'star.A.teff') and is not one of the reserved "
                f"params-file keys {sorted(RESERVED_PARAM_KEYS)}. It is "
                f"IGNORED. Check the spelling, or delete the line."
            )
            continue
        params[skey] = val

    od = reserved.get("overdisperse")
    if od is not None and not isinstance(od, bool):
        raise ValueError(
            f"'overdisperse'{where} is {od!r}; it must be a boolean "
            f"(`overdisperse: true` / `overdisperse: false`). It declares "
            f"whether the seeds in this file still want scattering before the "
            f"chains start -- an absent key means true."
        )
    return params, reserved


def _reject_bare_string_values(user_params, source=None):
    """Raise on a bare (non-dict) parameter value that is not a number.

    A params entry may be a dict of fields, a number, or a list of numbers
    (per-seed starts).  A bare STRING is none of those, and the two ways to
    write one both used to crash without saying where:

        star.A.teff: star.B.teff      # a link in the bare spelling
        star.A.teff: "hot"            # any other typo

    ``extract_links`` inspects dict entries only, so neither is recognized as
    a link and both reach ``finalize_user_params``' bare ``float(val)`` --
    a ValueError naming no parameter and no file.

    A bare link is REFUSED rather than rewritten to ``{initval: ...}``, per
    the house rule: we do not add special cases to rescue arbitrary user
    syntax.  The bare spelling is also genuinely ambiguous about the thing
    that matters most about a link -- with ``sigma: 0`` it is a hard link,
    with a sigma a soft one, with neither a seed -- so guessing would pick a
    model for the user.  The error names the dict spelling instead.
    """
    where = f" in {source}" if source else ""
    for key, val in (user_params or {}).items():
        if "." not in str(key) or not isinstance(val, str):
            continue
        try:
            float(val)
        except ValueError:
            pass
        else:
            continue  # "5800" is a number that happens to be quoted
        link_like = bool(re.search(r"[A-Za-z_]\w*\.[A-Za-z_]", val))
        advice = (
            f"To LINK it, write the field explicitly, and say which kind of "
            f"link you mean:\n"
            f"  {key}: {{initval: {val}, sigma: 0}}   # hard link (derived)\n"
            f"  {key}: {{initval: {val}, sigma: 0.5}} # soft link (penalty)\n"
            f"  {key}: {{initval: {val}}}             # seed only"
            if link_like
            else "A parameter entry is a number, a list of per-seed numbers, "
            "or a dict of fields (initval, mu, sigma, lower, upper, unit)."
        )
        raise ValueError(
            f"\n!!! NON-NUMERIC PARAMETER VALUE !!!\n"
            f"'{key}'{where} is set to the bare string '{val}', which is not "
            f"a number.\n{advice}"
        )


def as_field_entry(val):
    """One user entry in the ONE internal spelling: a dict of fields.

    A params file may write a parameter three ways -- a dict of fields, a bare
    number (``star.0.teff: 5800``) or a bare per-seed list
    (``star.0.teff: [5000, 6000]``) -- and the two bare forms mean exactly
    ``{initval: <value>}``.  This is the boundary translation the ONE-SPELLING
    rule (config.md) asks for, and it must be TOTAL: before review 1.1.7 the
    bare shape survived standardization, and each downstream reader re-guessed
    it differently -- ``_build_seed_overrides`` skipped it outright, so a bare
    per-seed list silently kept only seed 0.

    Always a fresh object (deepcopied), because downstream code writes through
    these entries in place -- see ``standardize_param_names``.
    """
    if isinstance(val, dict):
        return copy.deepcopy(val)
    return {"initval": copy.deepcopy(val)}


def standardize_entries(user_params):
    """Every parameter entry as a field dict; a null entry RAISES.

    The entry-shape half of the boundary, shared by
    ``ConfigManager.standardize_param_names`` (every pass) and
    ``ConfigManager.__init__``'s no-system_config branch, so both leave
    ``user_params`` holding one shape.  Keys are not touched.

    A NULL entry (``star.0.teff:`` with nothing after it) states no field at
    all.  Every reader used to skip it -- ``resolve()`` and
    ``finalize_user_params`` both ``continue`` on ``None`` -- except the
    key-presence checks, which counted it as a user value.  It RAISES, naming
    every such key (JDE 2026-09-30, review 1.1.7): it is ambiguous user input
    -- a value the user meant to write and did not, or a line they meant to
    delete -- not an unambiguous shortcut the boundary may translate, so
    guessing "no entry" would be exactly the silent repair the ONE-SPELLING
    rule forbids.
    """
    out = {}
    nulls = []
    for key, val in (user_params or {}).items():
        if val is None:
            nulls.append(str(key))
            continue
        out[key] = as_field_entry(val)
    if nulls:
        raise ValueError(
            f"params entries with no value: {sorted(nulls)}. Give each one a "
            f"value (a number, a list of per-seed numbers, or a dict of "
            f"fields), or delete the line."
        )
    return out


def user_entry(user_params, key):
    """The standardized entry under ``key``, or None when there is none.

    The one accessor for a ``ConfigManager.user_params`` entry.  After
    construction every value there is a field dict (``standardize_entries``),
    so a reader never has to guess the shape.  A non-dict here is an upstream
    bookkeeping bug -- something wrote a bare value past the boundary -- and it
    RAISES, naming the key, rather than being re-guessed (CLAUDE.md, "No
    fallbacks for internal invariants"; review 1.1.7).
    """
    return require_entry(key, user_params.get(key))


def require_entry(key, entry):
    """``entry`` itself if it is a field dict or None; TypeError otherwise.

    For a reader that already holds the value (a ``.items()`` loop); see
    ``user_entry``.
    """
    if entry is None or isinstance(entry, dict):
        return entry
    raise TypeError(
        f"user_params['{key}'] is a bare {type(entry).__name__} ({entry!r}), "
        f"not a field dict. ConfigManager translates every params entry to "
        f"{{initval: ...}} form at construction (standardize_entries), so a "
        f"bare value here means something wrote into user_params past that "
        f"boundary. Fix the writer; do not re-guess the shape here."
    )


def validate_sigma_has_center(user_params, links=None, source=None):
    """Fatal-error check: a Gaussian prior must have an explicit center.

    ``sigma > 0`` on a user parameter asks for a Gaussian prior.  When neither
    ``mu`` nor ``initval`` is given, Parameter.build_pymc centers that prior on
    whatever start value the system resolved (``prior_mus = np.where(~isnan(mus),
    mus, inits)``) -- and that start is frequently DERIVED FROM THE DATA: a
    component's PRECEDENCE_DERIVED_DATA hint, a relaxation-engine solution, or a
    start value mkparam seeded from a previous fit's MAP.  A prior centered on
    the data's own best fit double-counts the data, so there is no
    configuration in which it is what the user meant.  We refuse to run rather
    than silently produce it.

    Legitimate, and NOT flagged:
      - ``sigma: 0`` (any all-zero form) -- a fixed pin, not a prior.  It means
        "hold this at whatever it resolves to", which double-counts nothing.
      - a LINK expression in ``mu`` or ``initval`` -- a center is specified, it
        is just computed from another parameter.

    A ``sigma`` that is itself a link expression still requires a center: its
    width is dynamic but its center is no less obliged to be independent.

    Parameters
    ----------
    user_params : dict
        User params, standardized or RAW.  A bare (non-dict) entry is
        skipped: it states an initval and nothing else, so it cannot carry a
        sigma.  Only a raw params file can hold one -- mkparam validates the
        file it read this way -- since ConfigManager translates every entry
        to a field dict at construction (review 1.1.7).
    links : dict, optional
        ``{target_path: {field: ParamLink}}`` from ``extract_links``, which
        DELETES the link string from the entry -- so a linked mu/sigma is
        invisible in ``user_params`` and must be read from here instead.
    source : str, optional
        File the params came from, quoted in the error message.
    """
    links = links or {}
    offenders = []
    for key, entry in (user_params or {}).items():
        if not isinstance(entry, dict):
            continue
        entry_links = links.get(key, {})
        sigma_linked = "sigma" in entry_links
        sigma_val = entry.get("sigma")
        if not sigma_linked and ("sigma" not in entry or sigma_val is None):
            continue  # no sigma at all (an absent or null sigma is not a prior)
        if not sigma_linked and _sigma_is_zero(sigma_val):
            continue  # fixed pin, not a Gaussian prior
        if {"mu", "initval"} & (set(entry) | set(entry_links)):
            continue  # a center is specified (numerically or via a link)
        offenders.append(key)

    if offenders:
        where = f" in {source}" if source else ""
        raise ValueError(
            f"Gaussian prior with no center{where}: "
            f"{', '.join(sorted(offenders))}. "
            f"A 'sigma' greater than 0 asks for a Gaussian prior, but with "
            f"neither 'mu' nor 'initval' given the prior is centered on "
            f"whatever start value the system resolves -- and that start is "
            f"frequently derived FROM THE DATA (a component's data hint, a "
            f"relaxation-engine solution, or a start value mkparam seeded "
            f"from a previous fit's MAP). A prior "
            f"centered on the data's own best fit double-counts that data, "
            f"so it can never be justified. "
            f"Fix: give an explicit 'mu' (the independent prior center you "
            f"actually mean), or use 'sigma: 0' to hold the parameter fixed "
            f"at its resolved value (a pin, which applies no prior)."
        )


# defaults.yaml-only declaration that a parameter's prior is STRUCTURAL --
# part of the parameterization, not a statement a user may replace (review
# 2.2.21, JDE 2026-09-30).  `structural_prior: true` closes these fields to
# the params file.  The N(0, 1) on a direction latent (orbit xomega/yomega and
# xbigomega/ybigomega, lens xalpha/yalpha) is what makes the angle
# arctan2(y, x) uniform; on mann's non-centered ks_offset it IS the
# Ks-uncertainty prior.  Re-centering or re-widening it changes the meaning of
# the parameterization, not a prior.  An initval stays open on every one of
# them (JDE 2026-10-01 on ks_offset: "Losing the start bothers me").
#
# A parameter carrying the flag also declares `structural_remedy:`, the
# sentence the refusal ends with -- which PHYSICAL quantity to constrain
# instead.  Enforced in exactly one place, `validate_structural_fields`, at
# ConfigManager construction; documented in components/parameter.md.
STRUCTURAL_PRIOR_FIELDS = ("mu", "sigma")


def structural_closed_fields(base_defaults, comp, param):
    """The params-file fields closed on ``comp.param`` by a defaults.yaml
    ``structural_prior: true``, and its ``structural_remedy`` (or None).

    ``base_defaults`` is the merged defaults.yaml tree; the component's own
    block is layered over a root-level entry exactly as ``resolve()`` does.
    """
    spec = dict(base_defaults.get(param) or {})
    comp_block = base_defaults.get(comp)
    if isinstance(comp_block, dict) and isinstance(
        comp_block.get(param), dict
    ):
        spec.update(comp_block[param])
    closed = (
        STRUCTURAL_PRIOR_FIELDS if spec.get("structural_prior") is True else ()
    )
    return closed, spec.get("structural_remedy")


def validate_structural_fields(
    user_params, base_defaults, links=None, source=None
):
    """Fatal-error check: no user mu/sigma on a STRUCTURAL prior.

    ``base_defaults`` is the merged defaults.yaml tree (component blocks plus
    root-level parameters, as ``ConfigManager`` loads it).  Every params
    entry is matched to its parameter by its first segment (the component)
    and its last (the parameter) -- the same reading every one of the three
    spellings shares -- and refused when the parameter's defaults declare a
    ``structural_prior`` flag and the entry states mu or sigma, numerically
    or as a link (``extract_links`` deletes a linked field from the entry,
    so ``links`` is read too).  All offenders are named at once.
    """
    links = links or {}
    offenders = []
    for key in sorted(set(user_params or {}) | set(links)):
        parts = key.split(".")
        if len(parts) < 2:
            continue
        closed, remedy = structural_closed_fields(
            base_defaults, parts[0], parts[-1]
        )
        if not closed:
            continue
        entry = (user_params or {}).get(key) or {}
        stated = set(entry if isinstance(entry, dict) else ()) | set(
            links.get(key, {})
        )
        hit = [f for f in closed if f in stated]
        if hit:
            offenders.append((key, hit, remedy))
    if offenders:
        where = f" in {source}" if source else ""
        lines = []
        for key, hit, remedy in offenders:
            line = f"  {key}: {', '.join(hit)}"
            if remedy:
                line += f" -- {remedy}"
            lines.append(line)
        raise ValueError(
            f"Structural prior{where}: these parameters' priors are part "
            f"of their parameterization, not a statement a params file may "
            f"replace, so these fields are refused:\n" + "\n".join(lines)
        )


def _raise_duplicate_spelling(key_a, key_b, element, source=None):
    """Refuse a config that names one element under two spellings.

    The message names both keys verbatim, says they are the same element,
    lists the three legal spellings and says to keep exactly one -- the
    house style of the other pointed errors in this module.
    """
    where = f" in {source}" if source else ""
    comp = element.split(".")[0]
    param = element.split(".")[-1]
    raise ValueError(
        f"\n!!! DUPLICATE PARAMETER SPELLING !!!\n"
        f"'{key_a}' and '{key_b}'{where} are two spellings of the SAME "
        f"parameter element ('{element}').\n"
        f"An element has three legal spellings: the broadcast "
        f"'{comp}.{param}' (which covers every element no more specific "
        f"entry claims), the index form '{element}', and the name form "
        f"'{comp}.<name>.{param}'.  The last two are equally specific -- "
        f"each names exactly one element -- so nothing decides which of "
        f"them wins, and the resolver's own two passes do not agree: the "
        f"first-hit lookups (a 'unit:', an init_scale) take the index form "
        f"while the apply-every-match loops (initval, bounds, mu, sigma) "
        f"take the name form.\n"
        f"Fix: keep exactly one of the two entries and delete the other, "
        f"merging any fields you need from both.  (Combining the BROADCAST "
        f"form with one specific entry is fine and always means 'the most "
        f"specific wins'.)"
    )


def _reject_duplicate_spellings(user_params, system_config, source=None):
    """Fatal-error check: one element, one spelling.

    Runs at ConfigManager construction, on the RAW params (before
    ``standardize_param_names``, so the message shows the user's own
    spellings) -- and it has to run there, because standardization is where
    the evidence is destroyed: pass 1 files ``star.A.teff`` and
    ``star.0.teff`` under the one key ``star.0.teff``, so whichever the YAML
    listed second silently overwrote the other and nothing downstream could
    ever tell.  Same argument as ``_reject_renamed_arsun``: an entry that is
    quietly discarded may carry a value, a bound or a PRIOR.

    Only the two SPECIFIC spellings collide.  Broadcast + specific is a
    well-defined and useful idiom (set every element, refine one) and is
    deliberately untouched: pass 2 expands a 2-part key only into the indices
    no 3-part key claimed, which IS "the most specific wins".

    A name that is also an index string cannot produce a false positive: the
    two spellings are then the same string and a dict holds one of them.
    (``validate_instance_names`` bans all-digit names outright, so this
    cannot arise from a real config at all -- but the check is written not to
    depend on that.)

    Content is not consulted.  Two entries that happen to agree are still one
    element addressed twice, and exempting them would be a second rule to
    maintain for a config that is no less confusing to read.
    """
    seen = {}
    for raw_key in user_params or {}:
        key = str(raw_key)
        if len(key.split(".")) != 3:
            continue
        canon = canonical_param_key(key, system_config)
        prev = seen.get(canon)
        if prev is not None and prev != key:
            first, second = sorted((prev, key))
            _raise_duplicate_spelling(first, second, canon, source)
        seen[canon] = key


def canonical_param_key(key, system_config):
    """Canonical (index-form) spelling of a user-facing parameter key.

    ``star.A.mass`` -> ``star.0.mass`` when the config's ``star`` list has an
    entry named ``A``.  This is the ONE place the config-scanning name ->
    index translation lives: ``standardize_param_names`` uses it to store
    ``user_params``, ``ConfigManager._translate_and_scale`` uses it for every
    component hint, and anything looking a key up in ``user_params`` must go
    through it too -- otherwise the lookup silently depends on whether the
    user named their instances.  (``_index_path`` is the same translation
    against a prebuilt map, for the relaxation engine's inner loops.)

    Keys that are not 3-part, that name a flat-dict (non-list) component, or
    that name an instance the config does not define are returned unchanged,
    which is exactly how ``standardize_param_names`` stores them.
    """
    parts = key.split(".", 2)
    if len(parts) != 3:
        return key

    comp_type, comp_name, param_name = parts
    comp_list = (system_config or {}).get(comp_type)
    if not isinstance(comp_list, list):
        return key

    for idx, entry in enumerate(comp_list):
        if isinstance(entry, dict) and entry.get("name") == comp_name:
            return f"{comp_type}.{idx}.{param_name}"
    return key


def _index_path(path, name_to_index):
    """Index-form spelling of ``path`` under a prebuilt name -> index map.

    The relaxation engine builds ``{(comp_key, name): index}`` once per solve
    from the raw config (see ``resolve_and_validate_parameters``) and then
    translates many paths against it, so it uses this rather than
    ``canonical_param_key``, which rescans the config list on every call.
    Same answer, different cost model -- and this is the ONE implementation of
    the map-driven form, shared by the flat_params pass and
    ``_to_symbol_path``.

    Paths that are not 3-part, or whose (component, name) pair is not in the
    map -- including index-form paths, which need no translation -- come back
    unchanged.
    """
    parts = path.split(".")
    if len(parts) != 3:
        return path
    comp_type, name, param = parts
    if (comp_type, name) not in name_to_index:
        return path
    return f"{comp_type}.{name_to_index[(comp_type, name)]}.{param}"


def _declared_instance_names(system_config):
    """Every ``name:`` declared by any list-instanced component in the config.

    This is the universe of legal instance names in a 3-part parameter key.
    It is deliberately NOT per component: a component's per-element names are
    a manifest option that may borrow another component's instance names --
    the source component's instances are named after their body STARS
    (``source.SourceA.t_0`` with the star declared as ``star: [name:
    SourceA]``), for instance.  Used by ``standardize_param_names`` to
    reject typo'd instance names.
    """
    names = set()
    for entries in (system_config or {}).values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.add(entry["name"])
    return names


# ---------------------------------------------------------------------------
# THE PRECEDENCE LADDER (reviews 3.14.14, 3.14.22).  ONE ordered table that
# decides WHICH statement about a parameter wins, and the ONE place its levels
# are written: every PRECEDENCE_* constant below is read off it, and a
# defaults.yaml `rank:` names one of its keys rather than restating a number.
# That is what lets an ordering be permuted for a measurement by editing one
# dict (JDE 2026-10-14: the ordering sweep of the KS engine).
#
# What the levels MEAN under the KS rule (relaxation.py; config.md,
# "Precedence, and which value absorbs a discrepancy"): only a SEED carries a
# level -- a user value, a component hint, a defaults.yaml value on a sampled
# element.  The engine fixes seeds tier by tier from the top down, and a lower
# tier ABSORBS whatever a higher one implies; no level is ever traded against
# another.  A value the engine DERIVES carries no level of its own: the
# ledger still REPORTS one (the min of its inputs', capped at DERIVED_USER)
# for the provenance labels, but nothing decides on it.  So DERIVED_USER and
# DERIVED_MIXED are mostly REPORTING levels; a component may still push a
# hint at either (a start it computed from user values, or from a mix), and
# DERIVED_MIXED is the level a standalone solver's answer is held at.
#
#   USER                 a params.yaml value.  On a SAMPLED element it is
#                        produced exactly; on a derived one it is kept in
#                        order, or dropped and reported.
#   DERIVED_USER         reported for a value derived from user values only.
#   DERIVED_DATA         a data-derived component hint (an RV offset, a
#                        peak-finder seed, a galactic-model proper motion).
#   LIKELIHOOD_BACKED    a parameter whose start a stiff likelihood makes
#                        expensive to move: an evolutionary-model mass, age
#                        and EEP, the SED's own temperature, a mass
#                        or radius an empirical relation predicts.  The
#                        component says so (ConfigManager.
#                        declare_likelihood_backed) and that parameter's
#                        start is promoted to this level for the DECISION
#                        only -- it still reports the rank it came with (a
#                        defaults.yaml value still reports "default").  JDE
#                        2026-10-14 (R-b): "a data level provenance ...
#                        between user and default".  The slot is the one just
#                        below DERIVED_DATA rather than equal to it: a data
#                        hint is a measurement of the value itself and should
#                        still move a backed default, never the reverse, while
#                        every lower seed -- the microlensing distance
#                        guesses, the defaults -- absorbs a contradiction
#                        before a backed parameter moves.  The exact slot is
#                        one axis of the ordering sweep.
#   DERIVED_MIXED        reported for a value derived from a mix; the level
#                        of a standalone solver's answer and of the 3+-body
#                        lens seeds.
#   MULENS_SOURCE_DISTANCE / MULENS_LENS_DISTANCE
#                        the microlensing distance guesses (~8 kpc bulge
#                        source, ~4 kpc lens).  Both beat the 10 pc
#                        defaults.yaml backstop and yield to the user; the
#                        lens yields to the source, so a relative parallax
#                        the other values imply is absorbed by the LENS
#                        distance (review 4.6.2 named them).
#   DEFAULT              a defaults.yaml value.
#   DEFAULT_WEAK         a defaults.yaml value written `rank: DEFAULT_WEAK`
#                        (the stellar proper motions, the planet mass and
#                        radius): absorbs before any ordinary default.
#
# THE RENAME STOPS AT TWO CONTRACT BOUNDARIES (review 3.14.14).  The FIELD
# keeps its historical name `rank`: the defaults.yaml key (whose VALUE is a
# ladder NAME since review 3.14.22, never a number), and the exported
# provenance dict key that `introspect.py`'s field allowlist and the GUI read.
# Local variables holding that field's value are still `rank`, deliberately.
# ---------------------------------------------------------------------------
PRECEDENCE_LADDER = {
    "USER": 100,
    "DERIVED_USER": 80,
    "DERIVED_DATA": 60,
    "LIKELIHOOD_BACKED": 55,
    "DERIVED_MIXED": 40,
    "MULENS_SOURCE_DISTANCE": 30,
    "MULENS_LENS_DISTANCE": 25,
    "DEFAULT": 20,
    "DEFAULT_WEAK": 10,
}

PRECEDENCE_USER = PRECEDENCE_LADDER["USER"]
PRECEDENCE_DERIVED_USER = PRECEDENCE_LADDER["DERIVED_USER"]
PRECEDENCE_DERIVED_DATA = PRECEDENCE_LADDER["DERIVED_DATA"]
PRECEDENCE_LIKELIHOOD_BACKED = PRECEDENCE_LADDER["LIKELIHOOD_BACKED"]
PRECEDENCE_DERIVED_MIXED = PRECEDENCE_LADDER["DERIVED_MIXED"]
PRECEDENCE_MULENS_SOURCE_DISTANCE = PRECEDENCE_LADDER["MULENS_SOURCE_DISTANCE"]
PRECEDENCE_MULENS_LENS_DISTANCE = PRECEDENCE_LADDER["MULENS_LENS_DISTANCE"]
PRECEDENCE_DEFAULT = PRECEDENCE_LADDER["DEFAULT"]
PRECEDENCE_DEFAULT_WEAK = PRECEDENCE_LADDER["DEFAULT_WEAK"]


def ladder_level(name, where):
    """The level a defaults.yaml `rank:` names.

    A NAME of the ladder, never a number: the bare 5s and 10s restated the
    table in a second place, which is exactly what made an ordering
    impossible to change in one edit (review 3.14.22)."""
    if isinstance(name, str) and name in PRECEDENCE_LADDER:
        return PRECEDENCE_LADDER[name]
    raise ValueError(
        f"{where}: rank {name!r} is not a level of the precedence ladder.  "
        f"`rank:` takes one of its names ({', '.join(PRECEDENCE_LADDER)}); "
        f"a bare number is no longer accepted (review 3.14.22)."
    )


# THE COARSE LABEL IS DERIVED FROM THE SCALE, never stored beside it -- that
# is 3.14.14's part (b), and this set is the whole of the derivation.  Which
# levels mean "a component pushed this from the DATA", as opposed to "the
# relaxation engine derived it".
#
# A SET and not a numeric band, because the data levels do not bracket a
# contiguous range: PRECEDENCE_DERIVED_MIXED sits between them and
# PRECEDENCE_DERIVED_USER above them, and both are solver levels.  So the
# obvious `>= 30 and < PRECEDENCE_USER` test would report every
# engine-derived value as data-derived -- the mirror of the bug this replaces,
# which hardcoded the two literals and so reported any NEW data level as
# "solved".  Add a level here in the same commit that introduces it.
# PRECEDENCE_LIKELIHOOD_BACKED is deliberately NOT here: it is a DECISION
# level only and is never reported (a backed value reports the rank it came
# with).  PRECEDENCE_MULENS_LENS_DISTANCE is not here either: that is the
# historical behavior, and whether those seeds should report as data at all
# was review item 3.14.6 -- CLOSED, premise wrong (see the 2026-08-24
# review's appendix A): those sites are computed exclusively from
# user_params entries, so there was never a provenance lie to remove.
DATA_PRECEDENCES = frozenset(
    {PRECEDENCE_DERIVED_DATA, PRECEDENCE_MULENS_SOURCE_DISTANCE}
)


@dataclasses.dataclass(frozen=True)
class ProbedStart:
    """One path's answer from ``ConfigManager.probe_start``.

    ``value`` is in INTERNAL units (the relaxation engine's native unit, the
    unit ``add_hint`` stores and a Parameter's ``to_internal`` produces);
    ``user_value`` is the same number in the path's USER unit (honoring a
    user ``unit:`` override), converted by ``ConfigManager.from_internal``.
    Pick the one your consumer is written in -- never convert by hand.

    ``rank`` is the engine's provenance rank for the value and ``source`` its
    coarse label (``ConfigManager._provenance_label``).  A path the engine
    genuinely cannot reach -- no user entry, no hint, no defaults.yaml
    ``initval``, and no relation that solves it from those -- comes back with
    ``value``/``user_value``/``rank`` all None and ``source`` ==
    ``"not derivable"``; it is an answer, not an error.
    """

    path: str
    value: float | None
    user_value: float | None
    rank: int | None
    source: str

    @property
    def derivable(self):
        """True when the engine produced ANY start, a defaults.yaml one included."""
        return self.value is not None

    @property
    def informed(self):
        """True when the start traces back to a user entry or a component
        hint, directly or through a relation -- the ``probe_derivable`` test
        (rank above PRECEDENCE_DEFAULT).  A bare defaults.yaml value is
        derivable but NOT informed."""
        return self.rank is not None and self.rank > PRECEDENCE_DEFAULT


"""
ConfigManager is the reconciliation engine that decides the sole source of
truth for every parameter's start, bounds and preliminary scale.  It

1) determines each parameter's limits, the strictest of the defaults.yaml
   bounds, a component's validity limits and the user's;

2) decides every START value with the relaxation engine (relaxation.py):
   seeds -- user values, component hints, defaults.yaml values on sampled
   elements -- are fixed tier by tier down the precedence ladder, each tier
   absorbing what the ones above it imply, and the symbolic relations of
   every component's symbolic_physics.py carry each fixed value to the
   symbols it determines (config.md, "Precedence, and which value absorbs a
   discrepancy");

3) carries a PRELIMINARY init_scale for every engine symbol -- declared, or
   propagated from the sampled parameters through the relations -- which
   seeds the whitening probe (exozippy/whitening.py measures the real ones)
   and is the unit the engine splits a correction in;

4) reports what it could not honor: user values that contradict each other,
   and user values on derived elements that had to be dropped
   (ModelAuditor.check_user_starts words both).

The logic of the config manager is independent of any specific component.
"""

# The change test lives with the engine (relaxation.py); re-exported here,
# where the tests and config.md name it (review 1.1.9).
from .relaxation import CHANGE_TOL_IN_SCALES, value_moved  # noqa: E402,F401


def _element_flag(table, key, element, fallback, absent):
    """One element's boolean from a per-element mask table, a set, or nothing.

    `export_solution`'s two role tables come in three shapes and all three have
    to answer for a single exported path:

      * ``None``            -- no table: use `fallback` (the pre-table guess).
      * a set of keys       -- whole-parameter answer (`System.derived_params`).
      * ``{key: mask}``     -- per element (`System.derived_elements`), which is
        the shape to prefer; `element` is the index, or ``None`` for a 2-part
        broadcast path that stands for the whole vector.  A broadcast path is
        reported True only when EVERY element agrees, because its consumers read
        the flag as "this applies to all of it".

    `absent` is the answer when the table exists but does not mention the key at
    all -- False for derived (a parameter no manifest derives) and True for
    active (a parameter nothing masked).
    """
    if table is None:
        return fallback
    if not hasattr(table, "get"):
        return key in table
    mask = table.get(key)
    if mask is None:
        return absent
    arr = np.atleast_1d(mask)
    if element is None:
        return bool(np.all(arr))
    if element < arr.size:
        return bool(arr[element])
    return bool(np.all(arr))


class ConfigManager:
    def __init__(self, user_params, system_config=None):
        self.custom_solvers = {}
        self.standalone_solvers = set()

        # FIRST, before any other reader: a reserved key is not a parameter,
        # and everything below -- the validators, standardize_param_names'
        # pass 3, and above all finalize_user_params' leaf-symbol fallback --
        # would treat it as one.  See RESERVED_PARAM_KEYS.
        user_params, self.reserved_params = split_reserved_param_keys(
            user_params
        )

        _reject_renamed_arsun(user_params)
        _reject_list_valued_fields(user_params)
        _reject_bare_string_values(user_params)

        # Path of the params FILE these entries were read from, set by System
        # only when it actually read one -- it stays None when the caller
        # passed user_params in memory, even if the config happens to name a
        # parameter_file it did not use.  Metadata only: error messages that
        # ask the user to edit an entry quote it so they know which file to
        # open.
        self.param_file = None

        # User-defined parameter links (expression strings in numeric fields).
        # Populated by extract_links, which also strips the strings from
        # user_params so downstream numeric code never sees them.
        self.links = {}

        # The keys the USER actually wrote, before any standardization and
        # before the relaxation engine injects its solution back.  Read only
        # by resolve()'s duplicate-spelling check, which must not mistake an
        # injected index-form entry for a second user spelling (it is exactly
        # that on examples/ob161003, by design -- see the note there).
        self._raw_user_param_keys = {str(k) for k in (user_params or {})}

        # The (component, parameter) pairs the user wrote in the 2-part
        # BROADCAST form.  Pass 2 below expands them into indexed keys sized
        # by the config list and the evidence is gone, so record it here and
        # check it in resolve(), which is the first place the parameter's real
        # element count exists.  See _check_broadcast_covers_vector.
        # {internal "comp.<e>.param": user-facing "comp.<owner>.param_<k>"}
        # for the elements of OWNED vectors (adopt_element_owners), so
        # export_solution labels them the way the params file spells them.
        self._owned_display = {}

        self._broadcast_param_keys = {
            tuple(str(k).split("."))
            for k in self._raw_user_param_keys
            if len(str(k).split(".")) == 2
        }

        # If config is provided, validate names then standardize right away
        if system_config is not None:
            validate_instance_names(system_config)
            # BEFORE standardization: pass 1 folds the name form into the
            # index form, so a collision is invisible (and one of the two
            # entries silently gone) by the time it returns.
            _reject_duplicate_spellings(user_params, system_config)
            self.user_params = self.standardize_param_names(
                user_params, system_config
            )
            self._strip_user_init_scales()
            self.links = extract_links(self.user_params, system_config)
        else:
            # DEEPCOPY, because `finalize_user_params` INJECTS BACK into
            # `self.user_params` (solved values, `derived: True` markers), and
            # aliasing the caller's dict makes those writes land in their
            # object.  `standardize_param_names` above returns a fresh dict
            # and documents non-aliasing as a contract (the #76 lesson); this
            # branch silently lacked it, so the same ConfigManager built two
            # ways behaved differently.  Production always passes a
            # system_config -- this bit tests and direct drivers, which is
            # exactly where a mutated input is hardest to see (review 2.1.9).
            # standardize_entries deepcopies every entry AND translates a bare
            # value to {initval: ...}, the same entry shape the standardizer
            # emits (review 1.1.7): the keys here stay as the caller wrote
            # them, but no reader downstream ever sees a bare entry.
            self.user_params = standardize_entries(user_params)
            self._strip_user_init_scales()

        # Must run AFTER extract_links: that call deletes the link string from
        # the entry, so a linked mu/initval is only visible in self.links.  In
        # the no-system_config branch extract_links never ran and the link
        # strings are still in the entries, which the same check accepts.
        validate_sigma_has_center(self.user_params, self.links)

        self.system_config = system_config or {}
        self.base_defaults = {}
        self.all_relations = []
        self.master_symbol_map = {}
        # The subset of master_symbol_map that came from the components' own
        # get_symbol_map discovery -- i.e. paths some RELATION genuinely
        # knows.  master_symbol_map itself is NOT that set: finalize's
        # fallback registers every unmapped user key as a leaf symbol there,
        # so testing membership in the full map to ask "does the engine
        # consume this key" is vacuously true for any typo.  diagnostics'
        # check_unused_yaml reads this instead (the DC2018_128 rho-seed
        # false positive).
        self.relation_symbol_paths = set()

        # Storage for hints passed by components during Registration Sweep
        self.hints = {}
        self.hint_ranks = {}

        # Cross-component parameter overrides (see add_override).  Same
        # channel as a manifest entry's "overrides" dict -- layered UNDER the
        # user's params.yaml -- for a component that must constrain a
        # parameter another component owns.  path -> {field: value}.
        self.param_overrides = {}

        # Multi-seed sampling (P4).  seed_resolved holds K fully-solved start
        # points (list of {internal_path: internal_value} dicts) after
        # finalize_user_params runs the relaxation engine once per seed; it
        # stays None for the ordinary single-start case (K == 1).  seed_hint_sets
        # is a per-seed observable channel that components (the microlensing
        # peak finder) push into; it feeds the relaxation engine at
        # PRECEDENCE_DERIVED_DATA -- a seeder is a derivation FROM THE DATA, not
        # a user statement, so it sits in the same tier as any other data-driven hint
        # and every user entry outranks it.
        self.seed_resolved = None
        self.seed_hint_sets = []
        # Who registered the current seed_hint_sets (add_seed_hints' required
        # `source` label), so a second registration can name both callers
        # when it refuses (review 2.1.25).
        self.seed_hint_source = None
        self.scale_hints = {}  # path -> init_scale in internal units
        # path -> init_scale (internal) as the LAST relaxation solve left it:
        # the declared scale (defaults, a component scale hint, a user sigma)
        # where there is one, else the one propagated from the sampled
        # parameters through the relations (relaxation.Engine.scale).
        self.propagated_scales = {}
        self.symbolic_blacklist = set()
        # (equation, target symbol) -> the solutions sp.solve returned.
        # sp.solve is a pure function of (equation, target) -- it sees no
        # values -- so memoizing it is safe across seeds and probes, and the
        # engine relaxes the same inversions many times per prepare().
        # relaxation.py keeps its compiled (lambdified) forms the same way.
        self._symbolic_solve_cache = {}

        # Structured diagnostics collected by the relaxation engine (e.g.
        # over-constrained contradictions).  Each entry is a dict
        # {severity, message, param_paths}.  Consumed by the solve/validate
        # API (solve_api.py) without parsing log text.  Populated only when
        # the engine detects a contradiction; empty for a clean solve.
        self.diagnostics = []
        # Data-derived starts a component withheld because a user value
        # states the same quantity (supersede_hint); re-emitted into
        # `diagnostics` on every solve.
        self.superseded_hints = []

        # Snapshots of the last relaxation solve (seed 0, which is solved
        # last in finalize_user_params).  Exposed via export_solution() so a
        # caller can report each parameter's solved value, scale, and
        # provenance without rebuilding the PyMC model.
        self._last_provenance = {}  # internal_path -> rank
        self._last_scale_provenance = {}  # internal_path -> rank
        self._last_resolved = {}  # internal_path -> internal value
        self._last_solved_by = {}  # internal_path -> relation string
        # The keep-in-order and tier report of the last solve: which user
        # values on derived elements were kept or dropped, which relations
        # the rigid user values contradict, which seeds were absorbed or
        # moved.  relaxation.Engine.keep_report; read by the measurement
        # harness and the tests, reported to the user through diagnostics.
        self._last_keep_report = {}

        # Which engine paths are SAMPLED, DERIVED or INACTIVE elements of the
        # model (System._engine_roles, set before each finalize).  None
        # before stage 4 -- a stage 1-3 probe cannot know the manifests --
        # and then every defaults.yaml value is a seed and every user value
        # rigid, which is the documented probe behavior (config.md).
        self.engine_roles = None
        # Set by System: () -> (roles, prefixes, unknown paths) for the
        # components that have declared a manifest so far, read by a stage
        # 1-3 probe (roles None).  System._declared_engine_roles.
        self.role_provider = None
        # Exact numbers a component supplies for the relations to use --
        # never parameters, never seeds, never reported (add_constant).
        self.constants = {}
        # Paths a component's stiff likelihood backs (evolutionary model,
        # SED, empirical relations): their starts decide at
        # PRECEDENCE_LIKELIHOOD_BACKED.  See declare_likelihood_backed.
        self.likelihood_backed = set()

        components_dir = Path(__file__).parent / "components"

        # Sorted: rglob yields entries in filesystem directory order, which is
        # stable on one machine but differs between machines (ext4's hashed
        # btree vs xfs/NFS), so it survives PYTHONHASHSEED randomization and
        # looks perfectly reproducible until you compare two boxes.  This order
        # sets all_relations order, which sets the order the relaxation engine
        # visits equations, which decides WHICH member of a symmetric pair it
        # solves for -- e.g. mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2 and
        # pi_rel = KAPPA*m*(pi_E_N**2 + pi_E_E**2) are both symmetric under
        # swapping the pair, so nothing in the equation itself breaks the tie.
        # Unsorted, an Ubuntu 26.04 box transposed (pm_ra, pm_dec),
        # (mu_ra_rel, mu_dec_rel) and (pi_E_N, pi_E_E) relative to a RHEL8 box:
        # t_E came out 11.54 d instead of 18.29 d and DC2018_128's logp at the
        # pinned GOOD_RAW went -945.57 -> -113614.65.
        for py_file in sorted(components_dir.rglob("symbolic_physics.py")):
            module_name = (
                f"exozippy.components.{py_file.parent.name}.symbolic_physics"
            )

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register_solvers"):
                module.register_solvers(self)

            yaml_key = getattr(module, "comp_key", py_file.parent.name)

            # A relations file may be CONDITIONAL on the topology.  It is
            # keyed on one YAML block (comp_key), but that block is not
            # always the one whose presence makes the relations mean
            # anything: sed/symbolic_physics.py piggybacks on `star` so its
            # relations instantiate once per star, yet what they describe
            # (teffsed = teff, radiussed = radius) only exists when there is
            # a `sed:` block -- and star.py's manifest already declares
            # teffsed/radiussed only then.  Registering them regardless put
            # phantom leaf symbols, and their ledger rows, into every
            # non-SED config (review 1.9.4).
            #
            # A module opts in by defining relations_apply(system_config).
            # Absent, the relations always apply: that is the historical
            # behaviour and the right default for a file keyed on its own
            # block.  This is the symbolic-side analogue of the per-child
            # JITTER_RELATIONS registration (instrument.md) -- WHICH configs
            # a relation applies to is a fact only the relation's own file
            # knows, so it is the file that states it.
            applies = getattr(module, "relations_apply", None)
            if applies is not None and not applies(self.system_config):
                continue

            if (
                hasattr(module, "get_symbol_map")
                and yaml_key in self.system_config
            ):
                comp_section = self.system_config[yaml_key]

                if not isinstance(comp_section, list):
                    comp_section = [comp_section]

                for i, entry_cfg in enumerate(comp_section):
                    if not isinstance(entry_cfg, dict):
                        continue

                    # A component may return a single symbol map, or a list of
                    # maps when one config entry instantiates the relations
                    # multiple times (e.g. a lens with N sources instantiates
                    # the per-source parameter chain once per source).
                    #
                    # THE WHOLE SYSTEM CONFIG IS PASSED, not just this
                    # component's block.  A map builder may need to see
                    # another component's instances: the `mulensevent` split
                    # moves the lens and source BODY LISTS off the component
                    # whose relations reference them, so answering from
                    # `entry_cfg` alone stops being possible.  The signature
                    # is a hard contract -- every implementor takes both
                    # arguments, there is no introspection and no fallback,
                    # because all nine live in this repo and an internal
                    # interface carries exactly one spelling.
                    raw_maps = module.get_symbol_map(
                        entry_cfg, self.system_config
                    )
                    if not isinstance(raw_maps, list):
                        raw_maps = [raw_maps]

                    for raw_map in raw_maps:
                        instance_map = {}
                        # A symbol mapped to None names a quantity THIS
                        # instance does not have (a planet with no Keplerian
                        # orbit has no orbital period): every relation that
                        # reads it is left out of this instance, rather than
                        # bound to some other instance's path or left as a
                        # bare symbol shared across instances.
                        absent = {
                            sym_name
                            for sym_name, path in raw_map.items()
                            if path is None
                        }
                        for sym_name, path in raw_map.items():
                            if path is None:
                                continue
                            if "." in str(path):
                                instance_map[sym_name] = path
                            else:
                                instance_map[sym_name] = (
                                    f"{yaml_key}.{i}.{path}"
                                )

                        for _, full_path in instance_map.items():
                            self.master_symbol_map[full_path] = sp.Symbol(
                                full_path
                            )
                            self.relation_symbol_paths.add(full_path)

                        # Extract the exact SymPy objects (with all their assumptions) from the equations
                        module_symbols = set()
                        for rel in getattr(module, "RELATIONS", []):
                            module_symbols.update(rel.free_symbols)

                        # Sorted: module_symbols is a set, so its walk order
                        # is PYTHONHASHSEED-dependent, and `subs` below is
                        # applied as an unordered mapping.  The renaming is
                        # order-independent today (disjoint symbols), but the
                        # cost of stating the order is zero.
                        subs = {}
                        for sym in sorted(module_symbols, key=str):
                            if sym.name in instance_map:
                                subs[sym] = sp.Symbol(instance_map[sym.name])

                        for rel in getattr(module, "RELATIONS", []):
                            if any(
                                sym.name in absent for sym in rel.free_symbols
                            ):
                                continue
                            rel_inst = rel.subs(subs)
                            # Maps sharing symbols (e.g. per-source maps with a
                            # common lens mass) produce identical instances of
                            # the shared relations; keep one copy.
                            if rel_inst not in self.all_relations:
                                self.all_relations.append(rel_inst)

                    # Relations whose SHAPE depends on the config (a sum over
                    # however many bodies a lens has, review 8.6.13) cannot
                    # live in the fixed RELATIONS list, so a module may also
                    # define get_extra_relations(entry_cfg, system_config):
                    # equations already written in FULL-PATH symbols, no
                    # renaming.  A bare (pathless) symbol there is a bug in
                    # that module, not a spelling, so it raises.
                    extra = getattr(module, "get_extra_relations", None)
                    if extra is not None:
                        for rel in extra(entry_cfg, self.system_config):
                            for sym in sorted(rel.free_symbols, key=str):
                                if "." not in sym.name:
                                    raise ValueError(
                                        f"{yaml_key} get_extra_relations: "
                                        f"symbol '{sym.name}' in {rel} is not "
                                        f"a full parameter path."
                                    )
                                self.master_symbol_map[sym.name] = sp.Symbol(
                                    sym.name
                                )
                                self.relation_symbol_paths.add(sym.name)
                            if rel not in self.all_relations:
                                self.all_relations.append(rel)

        # Sorted for the same reason as the symbolic_physics walk above.  No
        # root-level key is currently defined by two components (a test pins
        # that), so today this is future-proofing rather than a live fix --
        # _deep_merge is last-writer-wins, so the first such collision would
        # otherwise resolve by walk order.
        for defaults_file in sorted(components_dir.rglob("defaults.yaml")):
            with open(defaults_file, "r") as f:
                comp_defaults = yaml.safe_load(f) or {}
                self._deep_merge(self.base_defaults, comp_defaults)

        # Needs the defaults (a parameter declares its own structural flags)
        # and the user's entries as WRITTEN: finalize_user_params injects the
        # engine's solved starts back into user_params later, so this is the
        # last point at which every initval there is the user's.
        validate_structural_fields(
            self.user_params,
            self.base_defaults,
            self.links,
            self.param_file,
        )

        # Add this inside ConfigManager.__init__ after filling all_relations
        for rel in self.all_relations:
            logger.debug(f"Relation: {rel}")

    @property
    def overdisperse(self):
        """Do this params file's seeds still want scattering? (default True.)

        The WRITER declares it, because the reader cannot tell the two cases
        apart at run time: K seeds may be K modes a user is trying (each of
        which wants dispersing) or K posterior draws off a finished fit
        (already dispersed, by construction).  ABSENT means True -- the safe
        direction for a hand-written file, since over-dispersing a good seed
        set costs some burn-in while under-dispersing a bad one makes Rhat
        read ~1.00 on chains that never mixed.  Consumed by
        samplers/_common.py's ``_make_starts``; see samplers/samplers.md.
        """
        return bool(self.reserved_params.get("overdisperse", True))

    def register_custom_solver(
        self, target_str, solver_func, standalone=False
    ):
        """
        Register a shortcut solver for 'comp.param' targets.

        By default a custom solver replaces the symbolic inversion of any
        relation the relaxation engine solves for the target
        (relaxation.Engine._solve_for).  standalone=True additionally makes
        it a one-way derivation of its own (relaxation.Engine.solve):
        required when the target's defining relation always holds a second
        derived unknown (e.g. orbit.m_total in the Kepler relation, whose
        other side is the equally-unknown semi-major axis), so the equation path can
        never get down to one unknown by itself.
        """
        self.custom_solvers[target_str] = solver_func
        if standalone:
            self.standalone_solvers.add(target_str)

    # ----------------------------
    # User-defined parameter links
    # ----------------------------

    def _link_internal_expr(self, plink):
        """
        Convert a ParamLink's user-unit expression to internal units.

        The user writes f(deps) where each dep contributes its value in its
        own user unit and the result is in the target's user unit.  The
        relaxation engine works in internal units, so substitute
        dep -> dep_internal / factor_dep and multiply by factor_target.
        """
        subs = {}
        for dep in plink.dep_paths:
            parts = dep.split(".")
            f = self.get_conversion_factor(parts[0], parts[-1], full_path=dep)
            if f != 1.0:
                subs[sp.Symbol(dep)] = sp.Symbol(dep) / f
        tparts = plink.target_path.split(".")
        tf = self.get_conversion_factor(
            tparts[0], tparts[-1], full_path=plink.target_path
        )
        expr = plink.expr.subs(subs) if subs else plink.expr
        return tf * expr if tf != 1.0 else expr

    def add_link(self, target_path, expr_str, origin):
        """A COMPONENT's hard link: ``target_path.initval = expr_str``.

        The component channel for a tie the topology implies (a star in an
        orbit group shares its anchor's distance, proper motion, rv,
        initfeh, age and av -- ``Star._link_orbit_members``).  It lands in
        ``self.links`` beside the user's, so the engine, the build graph and
        ``Component._wire_user_links`` treat it exactly as they treat a
        params-file link, but it is NEVER a ``user_params`` entry
        (CLAUDE.md), and its ``origin`` keeps it out of the user's column of
        the provenance ledger: the engine ranks its target as a propagated
        value, not a USER one.  The link is hard only where the target's
        resolved ``sigma`` is 0, which the caller supplies through its
        manifest ``"overrides"`` channel.  Stage 3 only: the engine reads
        the links at stage 4.

        Both paths in the index form.  A target that already carries an
        ``initval`` or ``mu`` link raises -- the caller decides precedence
        against the user's links before calling, so reaching here with one
        is a bookkeeping bug.
        """
        from .linking import ParamLink, parse_link_expression

        existing = self.links.get(target_path, {})
        if "initval" in existing or "mu" in existing:
            raise ValueError(
                f"add_link({target_path!r}, {expr_str!r}, origin={origin!r}): "
                f"the target already carries a link "
                f"({sorted(existing)}); the caller must not add a second one."
            )
        expr, deps = parse_link_expression(
            expr_str, self.system_config, f" (on {target_path}.initval)"
        )
        self.links.setdefault(target_path, {})["initval"] = ParamLink(
            target_path=target_path,
            field="initval",
            expr_str=expr_str,
            expr=expr,
            dep_paths=list(deps),
            origin=origin,
        )

    def get_element_links(self, comp_type, param_name):
        """
        Return the user-defined links targeting elements of one parameter:
        {field: {element_index: ParamLink}}.  Used by Component.add_parameter
        to wire dynamic (runtime) links into the PyMC graph.
        """
        out = {}
        for target, fields in self.links.items():
            parts = target.split(".")
            if (
                len(parts) == 3
                and parts[0] == comp_type
                and parts[2] == param_name
            ):
                for fld, plink in fields.items():
                    out.setdefault(fld, {})[int(parts[1])] = plink
        return out

    def _reject_broadcast_hint_path(self, path, translated_path):
        """Refuse a 2-part broadcast path on an INTERNAL hint channel.

        Only where the broadcast form is genuinely a broadcast, i.e. where
        the component is a LIST.  For a flat-dict component (``sed.av``) the
        2-part spelling IS the canonical one and names a single instance, so
        it stays legal -- which is also why the check is on the component's
        shape rather than on the number of dots.

        Permissive when there is no ``system_config``: that path cannot tell
        a broadcast from a canonical 2-part key, and it is the tests-and-
        direct-drivers mode, not production.
        """
        parts = translated_path.split(".")
        if len(parts) != 2 or not isinstance(self.system_config, dict):
            return
        if not isinstance(self.system_config.get(parts[0]), list):
            return
        spelled = (
            f"{path!r}"
            if path == translated_path
            else f"{path!r} (translated to {translated_path!r})"
        )
        raise ValueError(
            f"\n!!! BROADCAST PATH ON AN INTERNAL HINT CHANNEL !!!\n"
            f"{spelled} is the 2-part broadcast form, and "
            f"'{parts[0]}' is a list component with "
            f"{len(self.system_config[parts[0]])} instance(s).\n"
            f"Broadcasting is a USER-FACING convenience: a params-file entry "
            f"is expanded into one key per element at construction, and "
            f"nothing downstream sees the broadcast form again.  A hint, a "
            f"scale hint or a seed is pushed BY A COMPONENT, which knows "
            f"which element it means -- so push one per element:\n"
            f"    {parts[0]}.<i>.{parts[1]}\n"
            f"This is refused rather than expanded because one scalar cannot "
            f"answer for every element: since review 1.1.5 the elements may "
            f"carry different `unit:` overrides, so there is no single unit "
            f"the value could be read in."
        )

    def _translate_and_scale(self, path, value, unit=None):
        """Standardize a human-readable path to internal-index form and convert
        its value to internal units.  Returns (translated_path, internal_value).

        ``unit=None`` reads ``value`` in the parameter's resolved USER unit
        (a user ``unit:`` override, else the defaults.yaml ``unit``).  A
        ``unit`` string states the unit the number is actually in, and the
        value is converted straight to the parameter's ``internal_unit`` --
        the user's display unit plays no part (see ``add_hint``).

        The ONE implementation shared by add_hint / add_scale_hint /
        add_seed_hints / seed_start_value, so they cannot drift apart on
        nomenclature or unit handling.  ``add_scale_hint`` used to carry a
        line-for-line copy of this body despite the docstring claiming the
        sharing; a divergence there would have silently sent a component's
        scale hint to a path ``resolve()`` never looks up.
        """
        translated_path = self.canonical_key(path)

        # Scale to Internal Units, asking about the TRANSLATED path.  The
        # comment here used to say the opposite -- that the ORIGINAL path is
        # what carries a user `unit:` override -- and that is precisely
        # backwards: `standardize_param_names` folds the name form into the
        # index form at construction, so after it runs a LIST component's
        # `user_params` cannot contain a name-form key (with the two
        # exceptions named below), and a hint pushed as `star.B.distance`
        # looked up a spelling that does not exist and got the DEFAULTS
        # unit.  Measured: with `star.B.distance: {unit: kpc}`,
        # pushing the name form scaled by 1.0 and the index form by 1000.0 --
        # the same element of the same parameter, differing only in spelling
        # (review 2.14.6).
        #
        # Safe because `canonical_key` is the identity wherever `user_params`
        # keeps the original spelling: it rewrites only the name form, and
        # only for a LIST component.  So the translated path equals the
        # original everywhere the original already worked.
        #
        # AND THE 2-PART BROADCAST SPELLING IS REFUSED OUTRIGHT (review
        # 2.14.8).  It would otherwise miss in the same way and for the same
        # reason -- `canonical_key` leaves `star.distance` alone while
        # standardization expanded that entry into
        # `star.0.distance`/`star.1.distance` -- but the fix cannot be
        # another index translation, because a broadcast hint is ONE scalar
        # for every element while the elements may legitimately carry
        # DIFFERENT units (since review 1.1.5 a specific `unit:` overrides a
        # broadcast one per element).  There is no forced answer to "which
        # unit is this scalar in", so the channel refuses the question.
        #
        # JDE's ruling, 2026-09-04: broadcasting is a USER-FACING
        # CONVENIENCE.  It is translated once, at construction, and never
        # thought about again -- so an INTERNAL channel must never be handed
        # one.  The guard lives here rather than in `add_hint` because this
        # helper is the single implementation behind add_hint,
        # add_scale_hint, add_seed_hints and seed_start_value, and all four
        # are internal.
        self._reject_broadcast_hint_path(path, translated_path)

        final_parts = translated_path.split(".")
        if unit is not None:
            return translated_path, float(value) * self._stated_unit_factor(
                translated_path, unit
            )
        if len(final_parts) >= 2:
            c_type, p_name = final_parts[0], final_parts[-1]
            factor = self.get_conversion_factor(
                c_type, p_name, full_path=translated_path
            )
            internal_value = float(value) * factor
        else:
            internal_value = float(value)

        return translated_path, internal_value

    def _stated_unit_factor(self, translated_path, unit):
        """user -> internal multiplier for a value a COMPONENT computed in
        the stated ``unit`` (review 1.5.10).

        The parameter must declare an ``internal_unit`` in its defaults.yaml:
        a component stating a unit for a parameter that has none is a
        bookkeeping bug in that component, and is raised, naming both.
        """
        parts = translated_path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"[{translated_path}] a hint with unit={unit!r} needs a "
                f"<component>.<param> path."
            )
        c_type, p_name = parts[0], parts[-1]
        i_str = (self.base_defaults.get(c_type, {}).get(p_name, {}) or {}).get(
            "internal_unit"
        )
        if not i_str:
            raise ValueError(
                f"[{translated_path}] a hint was pushed with unit={unit!r}, "
                f"but {c_type}.{p_name} declares no internal_unit in its "
                f"defaults.yaml, so the stated unit cannot be converted."
            )
        return unit_conversion(unit, i_str, translated_path)

    def add_hint(self, path, value, rank=PRECEDENCE_DERIVED_DATA, unit=None):
        """Register a component's ranked, data-driven START VALUE.

        This is the one correct channel for a start value a component
        computes, and it has TWO consumers that must agree:

        * the relaxation engine (``resolve_and_validate_parameters`` step
          1.5), which solves from the number and files ``rank`` in the
          provenance ledger that ``initval_source``, ``export_solution``,
          the startup table and the GUI all report; and
        * ``resolve()``, which layers it as ``initval`` under the user's
          params and over defaults, so the stage-2/3 readers that ask for a
          start value BEFORE the engine runs -- ``Orbit``'s ``tc`` window,
          ``Orbit._seeded_period`` -- see the same number.  ``resolve()``
          did not, until review 3.14.3, which is why
          ``globalsearch.seed_start`` wrote every seed a SECOND time
          through ``add_override``.

        Converts human-readable paths to strict indices and scales to
        internal units (``_translate_and_scale``, shared with every other
        hint channel).

        STATE THE UNIT OF A NUMBER YOU COMPUTED (``unit=``).  Without it the
        value is read in the parameter's resolved USER unit, which honors a
        user ``unit:`` override -- right for a value that came out of the
        user's own params (or out of ``resolve()``), wrong for one a
        component computed in a fixed unit.  RVInstrument's gamma start, the
        mean RV in m/s, used to be pushed bare, so ``rvinstrument.0.gamma:
        {unit: km/s}`` read 12002 m/s as 12002 km/s and started gamma 1000x
        off (review 1.5.10); Planet's K start did the same.  With ``unit=``
        the value goes straight from that unit to ``internal_unit`` through
        ``unit_conversion`` and the user's display unit plays no part.  A hint is one scalar feeding ``initval`` and nothing
        else: a bound, a prior or a structural pin goes through
        ``add_override``, a preliminary whitening scale through
        ``add_scale_hint``.

        PUSH PER-ELEMENT PATHS.  The engine matches a hint by its EXACT
        path while ``resolve()`` matches it under all three spellings, so a
        2-part broadcast hint -- which nothing writes today -- would reach
        every element through ``resolve()`` at the parameter's DEFAULTS rank
        while reaching the ledger under no symbol at all (and leaving an
        orphaned 2-part row behind, the kind ``export_solution`` filters out
        by hand).
        """
        translated_path, internal_value = self._translate_and_scale(
            path, value, unit=unit
        )

        # Store the fully processed, ready-to-use hint
        self.hints[translated_path] = internal_value
        self.hint_ranks[translated_path] = rank

    def add_override(self, path, **fields):
        """Register a component-computed override on a parameter it does NOT own.

        This is the cross-component spelling of a manifest entry's
        ``"overrides"`` dict, and it behaves identically: the fields are
        applied inside ``resolve()`` through ``apply_value`` *before* the
        user's params.yaml, so the user still wins -- with the one deliberate
        exception ``apply_value`` builds in, that competing bounds combine as
        ``max(lower)`` / ``min(upper)`` order-independently.  That is what
        makes this the right channel for a **validity limit** (a range past
        which the likelihood is NaN or meaningless, e.g. the coverage of an
        interpolation grid) and the wrong one for a preference: a user bound
        it clips is applied and logged, not silently dropped.

        It exists because ``add_hint`` cannot express any of this.  A hint is
        a ranked *start value*: one scalar, feeding ``initval`` only, competing
        in the relaxation engine's provenance ledger.  A bound or a structural
        pin is neither a start value nor ranked -- ``lower``/``upper``/``sigma``
        never enter the ledger at all.  The corollary runs the other way too:
        ``resolve()`` layers hints ABOVE this channel, so an ``initval``
        written here is a fallback that a hint on the same path replaces.
        Do not reach for it to make a start value visible -- that is what
        ``add_hint`` already does.

        Writing into ``user_params`` instead (what the SED did until this was
        added) is what this replaces: those entries are indistinguishable from
        the user's own, so the ledger, ``export_solution``, ``initval_source``
        and the GUI all report a value the user never wrote, and
        ``finalize_user_params`` additionally registers the path as a leaf
        symbol in the relaxation engine.

        ``path`` may be any of the three spellings ``resolve()`` accepts --
        ``comp.param`` (broadcast to every element), ``comp.<i>.param`` or
        ``comp.<name>.param``.  Values are in the parameter's **defaults.yaml
        unit**, like every other override (a user ``unit:`` rescales them, the
        same way it rescales the defaults).  Per-element lists are accepted
        with the same conventions as a manifest override: ``NaN`` means "leave
        this element alone", ``+/-inf`` is a real bound.
        """
        self.param_overrides.setdefault(path, {}).update(fields)

    def add_seed_hints(self, seed_dicts, *, source, replace=False):
        """Register K per-seed observable sets for multi-seed sampling (P4).

        `seed_dicts` is a list of length K; each entry maps a parameter path
        (human-readable or index form) to a value in that parameter's user
        unit.  These feed the relaxation engine as one complete start point per
        seed (see finalize_user_params), at PRECEDENCE_DERIVED_DATA -- the same tier
        as ``add_hint``'s default, because a seeder (the built-in peak
        finder) is a derivation from the data, not a user statement.
        Every user entry therefore outranks a seed.  Paths absent from a given
        seed fall back to the base (defaults/hints/user) solution for that
        seed.

        ONE SEEDER PER FIT (review 2.1.25, JDE ruling 2026-09-30).  A second
        registration without ``replace=True`` RAISES, naming the caller that
        registered the existing sets and the new one.  It used to ASSIGN
        (silently discarding the first caller's seeds, review 2.1.12) and then
        to APPEND -- but a user per-seed ``initval: [a, b, c]`` list must match
        the total K, and a K assembled from two unrelated seeders is a number
        the user cannot know.  If a real two-seeder case ever appears, the
        upgrade is to PAIR set k with set k, not to concatenate.
        ``replace=True`` is the explicit override: the existing sets are
        discarded (``peak_find: true`` asks for it; with one seeder in the
        tree no live call replaces anything).

        `source` is a REQUIRED label for the caller (e.g.
        ``"peak finder (mulensinstrument)"``); it is recorded so the
        refusal can name who got there first.
        """
        if not source:
            raise ValueError(
                "add_seed_hints: `source` must name the caller registering "
                "the seed sets."
            )
        if self.seed_hint_sets and not replace:
            raise ValueError(
                f"add_seed_hints: {source!r} tried to register "
                f"{len(seed_dicts)} seed set(s), but {self.seed_hint_source!r} "
                f"already registered {len(self.seed_hint_sets)}.  One seeder "
                f"per fit (review 2.1.25): a user per-seed initval list must "
                f"match K, and a K summed over unrelated seeders is not one "
                f"the user can know.  Pass replace=True to discard the "
                f"existing sets on purpose."
            )
        processed = []
        for d in seed_dicts:
            pd = {}
            for path, value in d.items():
                tpath, ival = self._translate_and_scale(path, value)
                pd[tpath] = ival
            processed.append(pd)
        self.seed_hint_sets = processed
        self.seed_hint_source = source

    def seeded_paths(self):
        """Index-form paths seeded by ANY registered seed set.

        The "has something already seeded this?" question.  Seed hints live
        only in ``seed_hint_sets`` -- not in ``user_params`` and not in what
        ``probe_derivable`` sees at stage 1 -- so a check built on those two
        alone answers False even after a successful seeder push (the second
        half of review 2.1.12's trap).
        """
        return {p for s in self.seed_hint_sets for p in s}

    def seed_start_value(self, path, seed=0):
        """Seed-hint start value for ``path`` in USER units, or None.

        ``add_seed_hints`` stores values in index-form paths and INTERNAL
        units (via ``_translate_and_scale``).  Stage-1 consumers that read
        raw ``user_params`` entries -- which are still in user units before
        ``finalize_user_params`` runs -- use this as the matching-unit
        fallback (e.g. the mulens flux bootstrap needs alpha in degrees,
        the MulensModel convention, not the internal radians).
        """
        sets = self.seed_hint_sets or []
        if seed >= len(sets):
            return None
        tpath, _ = self._translate_and_scale(path, 0.0)
        if tpath not in sets[seed]:
            return None
        parts = tpath.split(".")
        factor = self.get_conversion_factor(
            parts[0], parts[-1], full_path=path
        )
        val = float(sets[seed][tpath])
        return val / factor if factor else val

    def add_scale_hint(self, path, scale, unit=None):
        """
        Register a context-appropriate PRELIMINARY init_scale for a parameter.

        Overrides the defaults.yaml init_scale.  Use this to set physically
        meaningful scales that differ from the generic stellar defaults (e.g.
        bulge distances need ~500 pc, not 0.1 pc).  Sampled parameters get
        their real scale from the whitening probe at startup regardless; the
        hint seeds that probe.  (It no longer sets any derived parameter's
        soft-bound barrier steepness -- the sympy forward Jacobian pass that
        did is deleted, and whitening.measure_barrier_scales measures those
        numerically instead.)

        The most specific spelling wins: a 3-part path here beats a 2-part
        broadcast one for the element it names, exactly as it does for every
        other field ``resolve()`` layers.

        ``unit=`` states the unit a COMPUTED scale is in, exactly as for
        ``add_hint``; without it the scale is read in the user's unit.
        """
        translated_path, internal_scale = self._translate_and_scale(
            path, scale, unit=unit
        )
        self.scale_hints[translated_path] = internal_scale

    def _deep_merge(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _check_broadcast_covers_vector(
        self, component_type, param_name, shape, n_elements, names=None
    ):
        """Refuse a 2-part broadcast key that does not cover the whole vector.

        ``standardize_param_names`` Pass 2 expands ``comp.param`` into one
        indexed key per CONFIG ENTRY, because that is all it can see: it runs
        at ConfigManager construction, long before any manifest exists.  But a
        parameter's element count is a manifest option (``shape``) and need not
        equal the config-list length, in EITHER direction:

          * LONGER than the config list -- HISTORICAL, and the reason this
            check was written: the pre-split ``lens`` had ONE config entry
            while its per-source vectors (``t_0``, ``u_0``, ``rho``, ...)
            carried one element per SOURCE, so ``lens.t_0: 2450000`` expanded
            to ``lens.0.t_0`` only and element 1 silently fell back to the
            defaults.yaml backstop.  Not a start-value-only defect: a
            broadcast ``sigma:``/``mu:``/``lower:`` applied the PRIOR to
            element 0 alone, i.e. a silent posterior change on a 2S2L fit.
            The 8.6.17 split removed that mismatch at the root -- ``source``
            now has one entry per source body and ``lens`` one per lens body,
            so those vectors' lengths equal their own config lists.  The
            branch is kept because nothing STOPS a component declaring a
            ``shape`` longer than its config list; it simply no longer has a
            shipped instance.
          * SHORTER than the config list -- ``detrend_coeffs`` has shape
            (total detrend columns,), which for two files with one column
            between them is 1 while the config list is 2.  Pass 2 then writes
            indexed keys for elements that do not exist.

        So raise, rather than teach Pass 2 to fill: the count is unknowable
        where the expansion happens and known here, and there is no filling
        rule that is right for both surfaces anyway (element j of a
        pre-split lens vector was a SOURCE, element j of ``detrend_coeffs``
        is a COLUMN -- neither is "instance j", which is the only thing a
        broadcast key can possibly mean).  Costs users nothing: no shipped
        example writes a 2-part broadcast on any such parameter
        (``examples/ob161003`` spells every per-source entry by the source
        star's name).

        Only checked when the caller passed an explicit vector ``shape``.  The
        single-element modes -- ``shape=()`` with or without ``element=`` --
        legitimately resolve one element of a longer vector (the relaxation
        engine and ``Instrument._time_coord`` both do), and comparing 1 against
        the config-list length there would reject every ordinary broadcast.

        Residual case, deliberately not covered: a mismatch the COUNTS agree
        about (two instruments with two detrend columns between them).  A
        length test cannot see that element i means something other than
        instance i; naming the columns individually is the fix if it ever
        bites.
        """
        if shape == () or not self._broadcast_param_keys:
            return
        if (component_type, param_name) not in self._broadcast_param_keys:
            return

        comp_list = self.system_config.get(component_type)
        if not isinstance(comp_list, list) or len(comp_list) == n_elements:
            return

        bcast = f"{component_type}.{param_name}"
        spellings = [
            f"{component_type}.{i}.{param_name}" for i in range(n_elements)
        ]
        if names:
            spellings = [
                f"{component_type}.{names[i]}.{param_name}"
                if i < len(names)
                else spellings[i]
                for i in range(n_elements)
            ]
        where = f" in {self.param_file}" if self.param_file else ""
        raise ValueError(
            f"\n!!! BROADCAST KEY DOES NOT COVER THE VECTOR !!!\n"
            f"'{bcast}'{where} is the 2-part broadcast form, which addresses "
            f"one element per '{component_type}' config entry "
            f"({len(comp_list)} of them), but '{bcast}' has "
            f"{n_elements} element(s): its length comes from the component's "
            f"manifest, not from the config list.\n"
            f"Broadcasting it would silently cover only part of the vector "
            f"(or address elements that do not exist), taking the value, "
            f"bounds and PRIOR with it.\n"
            f"Write one entry per element instead:\n  "
            + "\n  ".join(spellings)
        )

    @staticmethod
    def _owned_element_slots(
        component_type, param_name, n_elements, names, element_owners
    ):
        """``[(j, k), ...]`` per element of a vector with ``element_owners``:
        the owning instance's INDEX j and the element's position k among that
        instance's elements.  Raises on owners that do not match the vector
        or the instance names -- both are the component's own bookkeeping."""
        owners = [str(o) for o in element_owners]
        names = [str(n) for n in (names or [])]
        where = f"{component_type}.{param_name}"
        if len(owners) != n_elements:
            raise ValueError(
                f"{where}: element_owners names {len(owners)} elements but "
                f"the vector has {n_elements}"
            )
        unknown = sorted(set(owners) - set(names))
        if unknown:
            raise ValueError(
                f"{where}: element_owners {unknown} are not instances of "
                f"'{component_type}' ({names})"
            )
        return [
            (names.index(o), owners[:e].count(o)) for e, o in enumerate(owners)
        ]

    def adopt_element_owners(
        self, component_type, param_name, names, element_owners
    ):
        """Translate the params-file spelling of an OWNED vector's elements
        into the internal element-index form, after stage 3 and before the
        solve.  Called by ``System.prepare`` for every manifest entry that
        declares ``element_owners`` (Parameter.element_owners).

        A user names element e of such a vector -- ``detrend_coeffs``, one
        element per detrend COLUMN over every file -- as
        ``comp.<owner>.param_<k>`` (folded to ``comp.<j>.param_<k>`` by
        ``standardize_param_names``), because "instance j, its column k" is
        what it is.  Internally it is ``comp.<e>.param``, the form every
        reader of ``user_params`` -- ``resolve()``, the relaxation engine's
        unit/default/bound lookups, the inject-back -- already handles as
        element e.  Only the user-facing edges carry the owner spelling: the
        params file (here), ``Parameter.get_display_label`` (results.csv, the
        startup table, the diagnostics), ``export_solution`` and mkparam.

        The owners are unknown when ``ConfigManager`` is constructed (they
        come from the data files' columns, stage 1), which is why this is a
        separate step rather than a branch of ``standardize_param_names``.
        Idempotent: a second registration pass (a topology revision) finds
        nothing left to translate.
        """
        n_elements = len(element_owners)
        slots = self._owned_element_slots(
            component_type, param_name, n_elements, names, element_owners
        )
        self._check_owned_spellings(component_type, param_name, names, slots)
        for e, (j, k) in enumerate(slots):
            user_key = f"{component_type}.{j}.{param_name}_{k}"
            internal = f"{component_type}.{e}.{param_name}"
            self._owned_display[internal] = (
                f"{component_type}.{names[j]}.{param_name}_{k}"
            )
            for store in (self.user_params, self.links):
                if user_key not in store:
                    continue
                if internal in store:
                    raise ValueError(
                        f"{component_type}.{param_name}: both {user_key!r} "
                        f"and the internal {internal!r} are present -- this "
                        f"ConfigManager's own bookkeeping, not the params "
                        f"file, is wrong"
                    )
                store[internal] = store.pop(user_key)

    def _check_owned_spellings(
        self, component_type, param_name, names, owned_slots
    ):
        """Refuse every params-file key on an owned vector that does not
        name exactly one of its elements as ``comp.<instance>.param_<k>``.

        ``detrend_coeffs`` has one element per detrend COLUMN over every file
        (block-diagonal, in file order), so element j is not instance j.  The
        spelling ``comp.<name>.param`` used to fold to ``comp.<i>.param`` --
        element i, i.e. column i of the whole vector -- whatever file
        ``<name>`` was: a key naming one file set another file's coefficient
        (or, past the last column, nothing), and mkparam wrote restart files
        that way because the round trip came back to the same slot.  Every
        spelling that cannot say which file and which column is refused, and
        the message lists the ones that can (#440):

          * ``comp.<name|i>.param``  -- the old spelling;
          * ``comp.param`` and ``comp.param_<k>`` -- a broadcast: element j
            is not instance j, so "every instance" is not "every element";
          * ``comp.<name|i>.param_<k>`` for a column that instance does not
            have -- it would otherwise be silently ignored, prior and all.

        Read off the RAW keys, because ``standardize_param_names`` has
        already folded a name into an index and expanded a broadcast by the
        time ``user_params`` exists.
        """
        raw_keys = getattr(self, "_raw_user_param_keys", set())
        valid = {
            f"{component_type}.{names[j]}.{param_name}_{k}"
            for j, k in owned_slots
        }
        valid_idx = set(owned_slots)
        name_to_idx = {str(n): i for i, n in enumerate(names or [])}
        suffix = re.compile(rf"{re.escape(param_name)}_(\d+)")

        bad = []
        for key in sorted(raw_keys):
            parts = key.split(".")
            if parts[0] != component_type:
                continue
            last = parts[-1]
            m = suffix.fullmatch(last)
            if last != param_name and m is None:
                continue
            if len(parts) != 3:
                bad.append(key)
                continue
            inst = parts[1]
            j = int(inst) if inst.isdigit() else name_to_idx.get(inst)
            if m is None or (j, int(m.group(1))) not in valid_idx:
                bad.append(key)
        if not bad:
            return

        where = f" in {self.param_file}" if self.param_file else ""
        listed = (
            "\n  ".join(sorted(valid))
            if valid
            else f"(none -- no '{component_type}' instance has one)"
        )
        raise ValueError(
            f"\n!!! '{component_type}.{param_name}' IS NOT ONE ELEMENT PER "
            f"INSTANCE !!!\n"
            f"{', '.join(repr(k) for k in bad)}{where} cannot be resolved.  "
            f"'{component_type}.{param_name}' does not have one element per "
            f"'{component_type}' entry (detrend coefficients: one per "
            f"column, over every file), so an element is "
            f"named by its instance AND its position among that instance's "
            f"elements: '{component_type}.<instance>.{param_name}_<k>' "
            f"(the index form '{component_type}.<i>.{param_name}_<k>' works "
            f"too).  The spelling '{component_type}.<instance>.{param_name}' "
            f"used to address the column whose global index was the "
            f"instance's position -- usually another file's -- and a "
            f"broadcast cannot cover these elements one per instance.\n"
            f"The elements of this fit are:\n  {listed}"
        )

    def resolve(
        self,
        component_type,
        param_name,
        shape=(),
        internal_overrides=None,
        names=None,
        element=None,
    ):
        # `element` targets a single specific instance when resolving with
        # shape=(): the per-element user-param keys are built with that index
        # instead of the local loop index.  Without it, resolving star.1.age
        # one element at a time would read star.0.age's user entry and bleed
        # element-0 overrides (e.g. sigma: 0) into its siblings.

        # 1. Grab an isolated copy of the global root default blueprint
        base = copy.deepcopy(self.base_defaults.get(param_name, {}))

        # 2. Grab an isolated copy of the component-specific override block if it exists
        comp_defaults = self.base_defaults.get(component_type, {})
        comp_override = copy.deepcopy(comp_defaults.get(param_name, {}))

        if comp_override:
            # Safely layer in the math blueprint without mutating base_defaults
            if "expressions" not in comp_override and "expressions" in base:
                comp_override["expressions"] = base["expressions"]
            self._deep_merge(base, comp_override)

        # --- EXPLICIT ROOT LINEAGE BACKUP ---
        # If both base and child overrides are missing expressions due to startup file load order,
        # reach out directly to check the root-level dictionary block.
        if "expressions" not in base or not base["expressions"]:
            root_cfg = self.base_defaults.get(param_name, {})
            if "expressions" in root_cfg:
                base["expressions"] = copy.deepcopy(root_cfg["expressions"])

        n_elements = int(np.prod(shape)) if shape != () else 1

        def _eff_idx(i):
            return element if (element is not None and n_elements == 1) else i

        def _element_keys(i):
            """The user-facing spellings of element ``i`` of this parameter.

            The three forms ``resolve()`` accepts, in the order the loops that
            APPLY EVERY MATCH scan them: the 2-part broadcast form, the index
            form, and (only when the caller supplied per-element ``names``)
            the name form -- least specific first, so the most specific entry
            lands last and wins.  One list, five call sites, because the ORDER
            is the precedence rule and five copies of it is five chances to
            disagree about which spelling of a parameter wins.

            The lookups that pick a single winner (``unit:``, propagated
            scales, scale hints) go through ``_lookup_keys`` below, which
            traverses this same list so that the same entry wins there.
            """
            # BOTH per-element spellings index through `_eff_idx`, which is
            # the whole point of that helper: in single-element mode
            # (shape=(), element=j) the loop variable `i` is always 0 while
            # the element being resolved is `j`.  The index form used
            # `_eff_idx(i)` and the name form used `names[i]`, so the two
            # named DIFFERENT elements -- `star.1.distance` alongside
            # `star.A.distance` -- and a name-form entry for the element
            # actually being resolved was never looked up (review 2.1.8).
            ndx = _eff_idx(i)
            keys = [
                f"{component_type}.{param_name}",
                f"{component_type}.{ndx}.{param_name}",
            ]
            if names and ndx < len(names):
                keys.append(f"{component_type}.{names[ndx]}.{param_name}")
            return keys

        def _lookup_keys(i):
            """``_element_keys(i)`` with the broadcast spelling demoted to LAST.

            THE MOST SPECIFIC ENTRY WINS, for every field.  The loops that
            apply every match get that for free by ordering the list
            least-specific-first; a lookup that stops at the first hit has to
            be handed the other traversal, so this rotates the 2-part
            broadcast key to the end.  Same rule, opposite traversal -- and
            the reason to invert here rather than to keep two lists is that
            ``_element_keys`` is then still the one place the order lives.

            Until 2026-08 the first-hit lookups scanned the list as written,
            so a broadcast ``star.teff: {unit: K}`` BEAT a specific
            ``star.0.teff: {unit: ...}`` for the unit and for an init_scale,
            while LOSING to it for every numeric field -- in one config.

            Only the broadcast/specific tiers are reordered.  The index and
            name forms both name exactly ONE element and are equally specific,
            so nothing can decide which of those two wins -- and a config
            that sets both spellings of one element now RAISES rather than
            being adjudicated (see _reject_duplicate_spellings and the check
            just below).  The two traversals therefore never disagree: at
            most one of the two specific keys exists.
            """
            keys = _element_keys(i)
            return keys[1:] + keys[:1]

        # THE GUARANTEE AND ITS TWO EXCEPTIONS (review 2d-1/2d-4).  Since
        # System calls `normalize_config_block` on every component's config
        # BEFORE building the ConfigManager, every derived instance name
        # exists when standardization runs, so a LIST component's
        # `user_params` really cannot hold a name-form key.  That was NOT
        # true until 2026-09: Mann and Torres derived `name:` in their own
        # __init__, after this object was built, so `mann.A.ks_offset`
        # survived name-form forever and this very comment was wrong.
        # Still outstanding, deliberately:
        #   * a FLAT-DICT component keeps all three spellings verbatim --
        #     `canonical_key` is the identity there and `sed.B.av` has no
        #     index to fold into (review 3.14.17);
        #   * `system_config=None` skips standardization entirely (tests and
        #     direct drivers).
        #
        # ONE ELEMENT, ONE SPELLING -- the residual half of the check that
        # runs at construction.  That one sees every collision whose name
        # form is a CONFIG INSTANCE name, because standardize_param_names has
        # already folded such a key into the index form by the time we get
        # here.  What only this site can see is the case it cannot: per-
        # element `names` handed in by a component's manifest that are NOT
        # its own config instances' names.  HISTORICAL: the pre-split lens
        # did exactly that, labelling its per-source vectors with the
        # SOURCE STARS' names (`lens.SourceA.t_0`), rewritten to slot form
        # by a lens-private `_rewrite_source_param_keys` pass in its
        # __init__.  The 8.6.17 split removed the borrowed-name machinery
        # at the root -- source/lens instances are now NAMED after their
        # bodies (normalize_config_block, the Mann/Torres idiom), so the
        # name form folds in standardize_param_names like every other
        # component's.  This check is reached via `_raw_user_param_keys`
        # below -- the point of reading the raw keys rather than
        # `user_params`.
        #
        # Only keys the user WROTE count.  finalize_user_params injects the
        # engine's solved start values back under the index form, and on
        # ob161003 those legitimately sit alongside the user's own name-form
        # entries -- reading self.user_params here would fail every such fit
        # at stage 6.
        if names:
            raw_keys = getattr(self, "_raw_user_param_keys", set())
            for i in range(n_elements):
                keys = _element_keys(i)
                if len(keys) < 3 or keys[1] == keys[2]:
                    continue
                if keys[1] in raw_keys and keys[2] in raw_keys:
                    _raise_duplicate_spelling(
                        keys[1], keys[2], keys[1], self.param_file
                    )

        # A BROADCAST KEY MUST COVER THE WHOLE VECTOR OR NOT BE WRITTEN.
        self._check_broadcast_covers_vector(
            component_type, param_name, shape, n_elements, names
        )

        base_unit_str = base.get("unit", "")

        # The `unit:` override is resolved PER ELEMENT.  It used to be a
        # single global scan that stopped at the first element carrying one
        # and applied that element's unit -- and its scaling -- to the whole
        # vector: `planet.b.mass: {unit: earthMass}` silently relabeled
        # planet.c as earthMass too, so its defaults.yaml bounds (the actual
        # uniform prior range) came out 318x too wide and its start value
        # disagreed with what get_conversion_factor, which IS per element,
        # told the relaxation engine.
        #
        # This scan sets the SCALING while the user-params loop below rewrites
        # the reported unit STRING, so the two could in principle read
        # different entries -- numbers scaled as Kelvin, labelled deg_C in the
        # table and the CSV.  The duplicate-spelling raise makes that
        # unreachable, and by exhaustion rather than by luck: at most one
        # SPECIFIC entry exists per element, so either it carries a `unit:`
        # and both this scan (specific first, after _lookup_keys' rotation)
        # and the loop (specific last, apply-every-match) pick it, or it does
        # not -- in which case it cannot overwrite the label either and both
        # fall through to the broadcast entry.  Do not add a tie-break here.
        elem_units = []
        elem_scaling = np.ones(n_elements, dtype=float)
        for i in range(n_elements):
            u_str = None
            u_src = None
            for k in _lookup_keys(i):
                entry = user_entry(self.user_params, k)
                if entry is not None and "unit" in entry:
                    u_str = entry["unit"]
                    u_src = k
                    break

            if u_str:
                # A user `unit:` that astropy cannot parse, or that is not
                # convertible to the parameter's own unit, RAISES.  Falling
                # back to 1.0 silently reinterpreted the user's number in
                # the wrong unit -- see unit_conversion's docstring.  A
                # dimensionless parameter (base_unit_str == "") is included
                # deliberately: a unit on it cannot mean anything.
                elem_scaling[i] = unit_conversion(base_unit_str, u_str, u_src)
                elem_units.append(u_str)
            else:
                elem_units.append(base_unit_str)

        # Keep the scalar spelling when every element agrees, so the common
        # case is byte-for-byte what it always was; Parameter accepts either.
        unit_field = (
            elem_units[0] if len(set(elem_units)) == 1 else list(elem_units)
        )

        resolved = {
            "shape": shape,
            "user_modified": False,
            "user_prior_modified": False,
            "unit": unit_field,
            "internal_unit": base.get("internal_unit"),
            "latex": base.get("latex", ""),
            "description": base.get("description", ""),
            "expressions": base.get("expressions", {}),
            "print_to_table": base.get("print_to_table", True),
            "debug_print": base.get("debug_print", None),
            # Params-file only (see CITATION_KEYS); per element when a vector.
            "citation": (),
            # Component-declared, defaults.yaml only (not a params-file key,
            # so deliberately absent from STRING_KEYS): the sentence the
            # near-bound warnings append when this parameter's start or
            # posterior lands against a wall.  See parameter.md.
            "near_bound_remedy": base.get("near_bound_remedy"),
            # Component-declared, defaults.yaml only, the start-time twin of
            # near_bound_remedy: the sentence check_user_starts appends when
            # a user initval on this parameter is not what the model is
            # built at -- how to SEED it instead (review 8.6.15).  See
            # parameter.md.
            "seed_remedy": base.get("seed_remedy"),
            # Component-declared, defaults.yaml only: this parameter is
            # defined modulo a period ({value, unit} or {param: sibling}),
            # so its posterior is recentered about its mode before it is
            # summarized.  Parameter validates the shape; see parameter.md.
            "periodic": base.get("periodic"),
            # Component-declared, defaults.yaml only: this parameter is an
            # EPOCH of an event repeating with a sibling parameter's period,
            # and its reported posterior is moved by the whole number of
            # periods that minimizes its covariance with that period
            # (EXOFASTv2's T_0).  See parameter.md.
            "optimal_epoch": base.get("optimal_epoch"),
            # Component-declared, defaults.yaml only: the restart file
            # (mkparam) writes this SAMPLED parameter's start value as the
            # named derived sibling instead, because the sibling is what a
            # user writes and what the next fit must re-read to mean the same
            # thing (orbit.tc_sampled -> tc).  See run.md.
            "restart_as": base.get("restart_as"),
        }

        # The sub-key vocabulary is declared once at module scope (see
        # USER_PARAM_KEYS); diagnostics.py and introspect.py read the same
        # constants instead of restating them.
        physics_keys = PHYSICS_KEYS
        all_numeric = NUMERIC_KEYS

        for key in all_numeric:
            val = base.get(key)
            if val is not None:
                resolved[key] = float(val) * elem_scaling
            else:
                resolved[key] = None

        def apply_value(key, current_arr, idx, new_val):
            """Layer one value onto element `idx`; bounds keep the STRICTEST.

            The max()/min() on `lower`/`upper` is the documented
            defaults-and-component-validity rule stated in the class
            docstring above: a user bound never WIDENS a physics or grid
            limit (e.g. Instrument._register_noise's jitter-variance floor).

            It is deliberately NOT a rule between two USER spellings of one
            parameter.  Order-independent max/min defeated the ordering of
            `_element_keys`, so a broadcast bound beat a more specific one --
            the single field where "most specific wins" did not hold (review
            1.1.5).  That contest is now settled by specificity in the
            user-params loop BEFORE anything reaches here, so by the time a
            user bound arrives it is the only user bound in play and this
            clip can only ever be user-vs-defaults.
            """
            if new_val is None:
                return current_arr
            if current_arr is None:
                current_arr = np.full(n_elements, np.nan, dtype=float)
                resolved[key] = current_arr
            v = float(new_val)
            if np.isnan(current_arr[idx]):
                current_arr[idx] = v
            elif key == "lower":
                current_arr[idx] = max(current_arr[idx], v)
            elif key == "upper":
                current_arr[idx] = min(current_arr[idx], v)
            else:
                current_arr[idx] = v
            return current_arr

        resolved["auto_estimated"] = False
        # Component-computed bounds that were actually applied, so a user bound
        # they later clip can be reported instead of silently overridden (the
        # clip itself is deliberate: these are validity limits, e.g.
        # Instrument._register_noise's jitter-variance floor).
        override_bounds = {}

        def apply_overrides(od, indices):
            """Layer one component override dict onto elements `indices`."""
            for key in all_numeric:
                if key not in od:
                    continue
                val = od[key]
                for i in indices:
                    # A per-element override list is indexed by the
                    # PARAMETER's element, which is `_eff_idx(i)`, not by the
                    # local loop index: in the single-element mode (shape=(),
                    # element=j) the loop runs once with i = 0 while the
                    # element being resolved is j, so reading val[0] there
                    # applied element 0's pin to every element in turn.  The
                    # whole-vector mode is unaffected (element is None, so
                    # _eff_idx is the identity), which is why this went
                    # unnoticed until export_solution began passing manifest
                    # overrides through this path.
                    src = _eff_idx(i)
                    if isinstance(val, (list, np.ndarray)):
                        if src >= len(val):
                            continue
                        v = val[src]
                    else:
                        v = val
                    if v is None:
                        continue
                    v = float(v)
                    # NaN means "leave this element alone".  Component-supplied
                    # overrides are frequently per-element and sparse (e.g.
                    # Instrument._register_gp pins only the files that did not
                    # opt into a GP term); NaN lets one array express that
                    # without inventing a value for the others.  +/-inf is a
                    # legitimate bound and is NOT skipped.
                    if np.isnan(v):
                        continue
                    resolved["auto_estimated"] = True
                    apply_value(key, resolved[key], i, v * elem_scaling[i])
                    if key in ("lower", "upper"):
                        override_bounds[(key, i)] = v * elem_scaling[i]

        if internal_overrides:
            apply_overrides(internal_overrides, range(n_elements))

        # Cross-component overrides (ConfigManager.add_override): the same
        # channel, for a component constraining a parameter it does not own --
        # today the SED, whose model grid bounds star.teffsed/feh/av.  Keyed by
        # path rather than reached through the owner's manifest, and read here
        # under the same three spellings resolve() accepts everywhere else, so
        # a broadcast `star.av` covers every element and a per-element
        # `star.0.av` refines it.  Applied BEFORE the user's params below, so
        # the user still wins (bounds combine, per apply_value).
        if self.param_overrides:
            for i in range(n_elements):
                for k in _element_keys(i):
                    od = self.param_overrides.get(k)
                    if od:
                        apply_overrides(od, [i])

        # propagated_scales and scale_hints are stored in INTERNAL units and
        # resolve() returns USER units, so both are divided by the ELEMENT's
        # own user -> internal factor -- the same factor add_scale_hint used
        # to store them (_translate_and_scale -> get_conversion_factor with
        # full_path, which honors a user `unit:`), and the same one
        # Parameter._get_conversion_factors re-applies on the way back in.
        #
        # THAT FACTOR IS NOT `internal_factor` ALONE (review 1.14.1).
        # internal_factor is the DEFAULTS-unit -> internal multiplier;
        # elem_scaling[i] is defaults -> the element's USER unit, which every
        # other number in this dict already carries.  So user -> internal is
        #     internal_factor / elem_scaling[i]
        # and the recovery multiplies by elem_scaling[i] / internal_factor.
        # Dividing by internal_factor alone made the two halves of the round
        # trip disagree by exactly the unit factor whenever an element carried
        # a `unit:` override: 318x for planet.b.mass in earthMass, 57x for a
        # deg -> rad angle.  Not cosmetic -- init_scale seeds the whitening
        # probe and, for an unbounded element with no sigma, IS the prior
        # width.  The hints loop below does the same arithmetic for the same
        # reason; keep all three spellings identical.
        internal_factor = (
            self.get_conversion_factor(component_type, param_name) or 1.0
        )

        # Apply the previous solve's propagated scales (lowest priority after
        # defaults, overridden by scale hints and user params below).
        if self.propagated_scales:
            for i in range(n_elements):
                for k in _lookup_keys(i):
                    if k in self.propagated_scales:
                        apply_value(
                            "init_scale",
                            resolved["init_scale"],
                            i,
                            self.propagated_scales[k]
                            * elem_scaling[i]
                            / internal_factor,
                        )
                        break

        # Apply scale hints: context-appropriate init_scales from components
        # (e.g. bulge distances). They override defaults.yaml but yield to the
        # user's explicit init_scale below.
        for i in range(n_elements):
            for k in _lookup_keys(i):
                if k in self.scale_hints:
                    apply_value(
                        "init_scale",
                        resolved["init_scale"],
                        i,
                        self.scale_hints[k]
                        * elem_scaling[i]
                        / internal_factor,
                    )
                    break

        # Apply component START-VALUE hints (ConfigManager.add_hint).
        #
        # WHY resolve() has to see them at all: a hint is the one correct
        # channel for a component-supplied start value, but the relaxation
        # engine that consumes it does not run until stage 4, and several
        # components ask resolve() for a start value at stage 2/3 -- Orbit
        # builds tc's HARD window as tc_init +/- P/2 there, so an epoch it
        # cannot see does not merely start the chain in the wrong place, it
        # makes the right place unreachable.  Until this existed,
        # globalsearch.seed_start wrote every searched period and epoch a
        # SECOND time through add_override purely to be visible here: one
        # number, two channels, two chances to drift (review 3.14.3).
        #
        # THE PRECEDENCE IS THE POINT -- under the user's params, over
        # everything else.  Do not "simplify" the order:
        #   * OVER defaults.yaml, because a hint is a measurement (an RV
        #     median, a BLS epoch, median(err) for a GP amplitude) and the
        #     defaults.yaml number is the generic backstop it exists to
        #     replace.  The relaxation engine ranks it exactly that way --
        #     step 1.5 of resolve_and_validate_parameters overwrites the
        #     rank-20 default armor -- and resolve() disagreeing with the
        #     engine about the same number is the bug this whole block is
        #     avoiding.
        #   * OVER both override channels, whose numeric payload is
        #     component-computed DEFAULTS and validity BOUNDS.  Hints touch
        #     only `initval`, which no shipped override writes, so nothing
        #     moves today; the order states which wins if one ever does, and
        #     a ranked measurement beating an unranked fallback is the same
        #     rule as the line above.
        #   * UNDER the user, because every hint rank is below PRECEDENCE_USER and
        #     the engine enforces precisely that (step 1.5's `< PRECEDENCE_USER`
        #     guard).  resolve() has no ledger to consult, so HERE THE ORDER
        #     IS THE RULE: putting this loop after the user-params loop
        #     below would let a component's guess silently overwrite a
        #     number the user typed, in the one place the ledger cannot see.
        #
        # Only `initval`: a hint is one scalar start value and can never be
        # a bound, a prior or a scale (add_scale_hint is the scale channel,
        # applied just above).  It deliberately does NOT set
        # `auto_estimated` either -- that flag belongs to the override
        # channel, and a hint's provenance is already carried by its rank in
        # the ledger, which is what initval_source and export_solution read.
        if self.hints:
            for i in range(n_elements):
                for k in _lookup_keys(i):
                    if k in self.hints:
                        # Hints are stored in INTERNAL units (add_hint ->
                        # _translate_and_scale) and resolve() returns USER
                        # units, so divide by the user -> internal factor.
                        # internal_factor is base(defaults) -> internal and
                        # elem_scaling[i] is base -> user, so user ->
                        # internal is internal_factor / elem_scaling[i] --
                        # the element's own `unit:` override has to be in
                        # there, or a hint on a planet mass the user asked
                        # for in earthMass comes back 318x wrong.
                        apply_value(
                            "initval",
                            resolved["initval"],
                            i,
                            self.hints[k] * elem_scaling[i] / internal_factor,
                        )
                        break

        for i in range(n_elements):
            # A SPECIFIC USER BOUND IS THE EXCEPTION TO A BROADCAST ONE, NOT
            # AN ADDITION TO IT (review 1.1.5).  `_element_keys` is in
            # specificity order and the loop below applies every match, so
            # for every field the last write wins and the most specific
            # spelling therefore wins -- which is what config.md promises
            # "for every field".  Bounds were the silent exception, because
            # apply_value keeps the strictest instead of assigning: a
            # broadcast `sed.av: {lower: 1.0}` beat a specific
            # `sed.0.av: {lower: 0.5}` and nothing said so.
            #
            # So pick the most specific spelling that STATES each bound and
            # let only that one through.  Doing it here, rather than by
            # weakening apply_value, keeps the two rules apart: user-vs-user
            # is decided by specificity, user-vs-defaults is still decided by
            # strictness.  It is also what lets a "the defaults clipped you"
            # warning be written correctly (2.1.6) -- today the same clip
            # fires user-vs-user and would mis-blame the wrong entry.
            #
            # Reachable only where BOTH spellings survive standardization:
            # for a LIST component `standardize_param_names` expands a 2-part
            # key solely into the indices no 3-part key claimed, so the
            # element never sees the broadcast.  A FLAT-DICT component's
            # params (`sed.av` and `sed.0.av`) pass through verbatim, and
            # that is where this bit.
            bound_source = {}
            for k in _element_keys(i):
                scanned = user_entry(self.user_params, k)
                if scanned is not None:
                    for bkey in BOUND_KEYS:
                        if bkey in scanned:
                            bound_source[bkey] = k

            for k in _element_keys(i):
                if k in self.user_params:
                    ov = user_entry(self.user_params, k)

                    resolved["user_modified"] = True
                    if any(pk in ov for pk in physics_keys):
                        resolved["user_prior_modified"] = True

                    for key in all_numeric:
                        if key in ov:
                            if bound_source.get(key, k) != k:
                                # A more specific spelling of this element
                                # states this bound; that one is the whole
                                # user bound, not a second opinion to be
                                # combined with this one.
                                continue
                            v_ov = ov[key]
                            # List-valued initvals are per-seed start points
                            # (P4 multi-seed sampling); bounds/scales and the
                            # canonical single start derive from seed 0.
                            if isinstance(v_ov, (list, tuple)):
                                v_ov = v_ov[0]
                            apply_value(key, resolved[key], i, v_ov)
                            # apply_value keeps the TIGHTER of the two bounds,
                            # so the user's entry may simply not apply.  EITHER
                            # WAY THEY HEAR ABOUT IT (review 2.1.6).  The clip
                            # itself is documented design and stays -- a user
                            # bound never widens a physics or grid limit -- but
                            # the SILENCE was not designed: an override-bound
                            # clip warned and a defaults-bound clip said
                            # nothing, so the same user mistake was announced
                            # or swallowed depending on which layer happened
                            # to own the tighter number.  Rope, not gates: the
                            # user hears that what they wrote did not take
                            # effect.
                            #
                            # Reachable as a clean user-vs-defaults statement
                            # only since review 1.1.5.  Before that the same
                            # clip also fired user-vs-USER, so this message
                            # would have blamed "the defaults" for a broadcast
                            # entry the user had written themselves.  1.1.5
                            # settles broadcast-vs-specific by specificity
                            # first, which is what makes this warning
                            # truthful.
                            if not np.isclose(resolved[key][i], float(v_ov)):
                                if (key, i) in override_bounds:
                                    logger.warning(
                                        f"[{component_type}.{_eff_idx(i)}."
                                        f"{param_name}] user "
                                        f"{key}={float(v_ov):g}"
                                        f" is outside the component-computed "
                                        f"validity bound "
                                        f"{override_bounds[(key, i)]:g}; "
                                        f"using {resolved[key][i]:g}."
                                    )
                                else:
                                    logger.warning(
                                        f"[{component_type}.{_eff_idx(i)}."
                                        f"{param_name}] user "
                                        f"{key}={float(v_ov):g} WIDENS the "
                                        f"defaults.yaml {key} of "
                                        f"{resolved[key][i]:g} and was NOT "
                                        f"applied; the stricter bound wins. "
                                        f"Edit the component's defaults.yaml "
                                        f"if the limit itself is wrong."
                                    )

                    for str_key in STRING_KEYS:
                        if str_key in ov:
                            if n_elements > 1:
                                if not isinstance(resolved[str_key], list):
                                    resolved[str_key] = [
                                        resolved[str_key]
                                    ] * n_elements
                                resolved[str_key][i] = ov[str_key]
                            else:
                                resolved[str_key] = ov[str_key]

                    for bool_key in BOOL_KEYS:
                        if bool_key in ov:
                            resolved[bool_key] = ov[bool_key]

                    if "citation" in ov and not any(
                        pk in ov for pk in physics_keys
                    ):
                        # A citation justifies a PRIOR; on an entry that
                        # states none (an initval is a start value) it would
                        # render as "Prior from ..." on the defaults' uniform,
                        # a false statement in a published table.
                        logger.warning(
                            f"[{component_type}.{_eff_idx(i)}.{param_name}] "
                            f"'citation' ignored: the entry states no prior "
                            f"({', '.join(physics_keys)}), so there is "
                            f"nothing for it to justify."
                        )
                    elif "citation" in ov:
                        keys = normalize_citation(
                            ov["citation"],
                            f"{component_type}.{_eff_idx(i)}.{param_name}",
                        )
                        if n_elements > 1:
                            if not isinstance(resolved["citation"], list):
                                resolved["citation"] = [
                                    resolved["citation"]
                                ] * n_elements
                            resolved["citation"][i] = keys
                        else:
                            resolved["citation"] = keys

                    # A user-supplied Gaussian prior width doubles as the
                    # best preliminary scale (user init_scale entries were
                    # stripped at construction, so sigma is the only user
                    # signal left).
                    if "sigma" in ov and resolved["sigma"] is not None:
                        apply_value(
                            "init_scale",
                            resolved.get("init_scale"),
                            i,
                            ov["sigma"],
                        )

                    # If user gave mu but not initval, start the chain at mu rather
                    # than the defaults.yaml value — the user's prior center is always
                    # a better starting point than an arbitrary global default.
                    if "mu" in ov and "initval" not in ov:
                        apply_value(
                            "initval", resolved["initval"], i, ov["mu"]
                        )

        return resolved

    def get_conversion_factor(
        self, component_type, param_name, full_path=None
    ):
        u_str = None
        user_supplied = False
        # 1. Check if the user explicitly provided a unit in their config
        entry = user_entry(self.user_params, full_path) if full_path else None
        if entry is not None:
            u_str = entry.get("unit")
            user_supplied = bool(u_str)

        # 2. Fallback to defaults
        comp_cfg = self.base_defaults.get(component_type, {})
        param_cfg = comp_cfg.get(param_name, {})
        if not u_str:
            u_str = param_cfg.get("unit", "")

        i_str = param_cfg.get("internal_unit", "")

        if not u_str or not i_str:
            # No declared pair -> genuinely nothing to convert.  But a unit
            # the USER supplied on a parameter that declares no
            # internal_unit cannot be honored, and returning 1.0 would read
            # their number as if it had been written in the default unit.
            if user_supplied and not i_str:
                raise ValueError(
                    f"[{full_path}] unit: {u_str!r} was given, but "
                    f"{component_type}.{param_name} declares no "
                    f"internal_unit, so the value cannot be converted.  "
                    f"Remove the unit: key or give the parameter an "
                    f"internal_unit in its defaults.yaml."
                )
            return 1.0

        # Multiplier to convert FROM user TO internal.  Raises on an
        # unparseable or incompatible unit rather than silently using 1.0.
        return unit_conversion(
            u_str, i_str, full_path or f"{component_type}.{param_name}"
        )

    def _assert_canonical_engine_paths(self, seed_hints=None):
        """Refuse a non-canonical path at the relaxation engine's door.

        THE STANDING RULE (JDE 2026-09-04): internal syntax is strict, ONE
        SPELLING ONLY.  Only the USER boundary is permissive, and what it
        accepts it translates immediately -- `standardize_param_names` folds
        the name form into the index form and expands a broadcast into every
        index it covers, and `_translate_and_scale` does the same for every
        component hint channel.

        This is the assertion that the translation was TOTAL.  It is cheap
        (about ninety paths, once per seed) and it is the check that would
        have caught the whole family of defects this exists because of --
        1.1.5, 1.1.6, 2.1.8, 2.14.6 and 2.14.8 were all one thing: a
        user-facing spelling surviving past the boundary, with something
        downstream left to guess which spelling it had.  Each was found
        separately, by hand, months apart.  One assertion here fails loudly
        the first time any of them recurs.

        CANONICAL means:
          * `comp.<i>.param` -- the index form, for a LIST component;
          * `comp.param` -- for a FLAT-DICT component, where the 2-part form
            IS the canonical spelling and there is no list to broadcast over
            (that asymmetry is itself review 3.14.17).

        Deliberately NOT checked here: membership in `master_symbol_map`.
        That was the obvious formulation and it is WRONG -- measured across
        all twenty shipped examples, 25 of 72 hint paths are legitimately
        absent from the map (`mulensinstrument.<i>.f_source`/`f_blend`/
        `q_flux`), because `resolve()` reads hints too and a hinted parameter
        need not be an engine symbol.  Gating on the map would have refused
        every microlensing example.  Shape is the invariant; symbol
        membership is not.
        """
        if not isinstance(self.system_config, dict):
            # No system config: the tests-and-direct-drivers mode, which
            # cannot tell a broadcast from a flat-dict component's canonical
            # key.  Same reason `_reject_broadcast_hint_path` is permissive
            # there.
            return

        for channel, paths in (
            ("add_hint", self.hints),
            ("add_scale_hint", self.scale_hints),
            ("add_seed_hints", seed_hints or {}),
        ):
            for path in paths:
                parts = str(path).split(".")
                comp = self.system_config.get(parts[0]) if parts else None
                if comp is None:
                    # A component this config does not declare at all.  Not
                    # this check's business -- standardize_param_names
                    # already raises on an undeclared prefix, and refusing
                    # here would only duplicate it in a worse place.
                    continue
                if isinstance(comp, list):
                    ok = len(parts) == 3 and parts[1].isdigit()
                    canonical = f"{parts[0]}.<i>.{parts[-1]}"
                else:
                    ok = len(parts) == 2
                    canonical = f"{parts[0]}.{parts[-1]}"
                if ok:
                    continue
                raise ValueError(
                    f"\n!!! NON-CANONICAL PATH AT THE ENGINE !!!\n"
                    f"{path!r}, pushed through {channel}, reached the "
                    f"relaxation engine without being translated to the "
                    f"canonical internal spelling {canonical!r}.\n"
                    f"Internal paths carry exactly ONE spelling.  The user "
                    f"boundary accepts three and translates immediately "
                    f"(standardize_param_names for params.yaml, "
                    f"_translate_and_scale for every hint channel), so a "
                    f"non-canonical path here means a translation was "
                    f"skipped rather than that a new spelling is legal.\n"
                    f"This is the guard for the defect class of reviews "
                    f"1.1.5, 1.1.6, 2.1.8, 2.14.6 and 2.14.8; if you are "
                    f"adding a channel, translate at its entry point, not "
                    f"here."
                )

    def canonical_key(self, key):
        """Index-form spelling of ``key`` under THIS manager's system config.

        Thin instance wrapper around :func:`canonical_param_key`, for callers
        that hold a ConfigManager and need to look a user-facing (possibly
        name-form) path up in ``self.user_params``, which is stored in index
        form.
        """
        return canonical_param_key(key, self.system_config)

    @staticmethod
    def standardize_param_names(user_params, config):
        """
        Translate all user-facing parameter keys to canonical internal index form.

        Three input forms are accepted, processed in two passes so that explicit
        per-instance values always win over broadcast values regardless of file order:

          Pass 1 — 3-part keys (highest precedence):
            star.A.teff   →  star.0.teff   (name looked up in config list)
            star.0.teff   →  star.0.teff   (already indexed; stored as-is)

          Pass 2 — 2-part keys (broadcast to all instances):
            star.teff     →  star.0.teff, star.1.teff, …
            sed.errscale  →  sed.errscale  (flat-dict component; kept as-is)

        After this function, self.user_params contains only indexed or flat-dict
        keys internally.  The 2-part form is purely a user convenience.

        PASS 2 MERGES PER FIELD, NOT PER KEY (review 1.1.6).  It used to skip
        any index a Pass-1 key had already claimed -- `if indexed_key not in
        standardized` -- which made "explicit beats broadcast" a statement
        about whole ENTRIES rather than about the fields they set.  So a
        specific entry that mentioned some OTHER field silently cancelled the
        broadcast for that element, and it fell back to defaults.yaml:

            star.radius: 3.0                        -> [3.0, 3.0]
            star.radius: 3.0
              + star.B.radius: {lower: 0.1}         -> [3.0, 1.0]   B lost 3.0

        A user who put a BOUND on one star silently lost their broadcast
        START VALUE for that star.  Now the broadcast supplies the base entry
        and the specific one overrides it field by field, so "the most
        specific spelling wins, for every field" -- which is what config.md
        promises in those words -- is true of the standardizer too, and not
        only of `resolve()`'s later layering.

        EVERY pass deepcopies the entry.  The returned dict must share no object
        with the caller's, because downstream code writes through these entries
        in place: extract_links deletes the link-expression fields, and
        finalize_user_params' inject-back sets initval/derived.  Aliasing them
        would (a) strip a caller's link strings out of their own dict, so a
        second ConfigManager built from it sees no links and a hard link
        silently degrades to a fixed parameter, and (b) feed the previous
        solve's answer back in as PRECEDENCE_USER input.  Pass 2 additionally needs
        the copy per broadcast instance, so the last instance's write does not
        clobber all the others (e.g. per-source radii solved from rho).

        EVERY ENTRY LEAVES AS A FIELD DICT (review 1.1.7).  A bare value
        (`star.0.teff: 5800`, or the per-seed list `star.0.teff: [5000,
        6000]`) is translated to `{initval: ...}` before any pass runs, via
        `standardize_entries`, so the keys and the entry shape are both in
        their one internal spelling on the way out and no reader downstream
        has to re-guess either.  The user's FILE keeps its own spelling; this
        is a translated copy.
        """
        if not user_params:
            return {}

        user_params = standardize_entries(user_params)
        standardized = {}

        # Pass 1: resolve 3-part keys to index form.
        for key, val in user_params.items():
            parts = key.split(".", 2)
            if len(parts) != 3:
                continue

            comp_type, comp_name, param_name = parts

            if comp_type not in config:
                raise ValueError(
                    f"\n!!! STRICT NAMING ERROR !!!\n"
                    f"Parameter '{key}' uses the prefix '{comp_type}', but '{comp_type}' "
                    f"is not defined in your system configuration.\n"
                    f"Ensure your YAML block names match your parameter prefixes exactly."
                )

            comp_list = config[comp_type]
            if not isinstance(comp_list, list):
                # flat-dict component 3-part key: keep the key as-is
                standardized[key] = copy.deepcopy(val)
                continue

            # An UNKNOWN INSTANCE NAME is as fatal as an unknown component
            # prefix, and for the same reason.  canonical_param_key returns
            # such a key unchanged (deliberately -- it is also the lookup
            # helper behind ConfigManager.canonical_key, which must stay
            # total), so it used to be stored verbatim, registered as an inert
            # leaf symbol by finalize_user_params, and never reach any
            # parameter: `star.Aa.teff` silently dropped the user's value,
            # bounds, mu and sigma.  A typo'd PRIOR that changes the posterior
            # without a word is the worst version of this, so refuse to run.
            #
            # The accepted name set is every `name:` DECLARED ANYWHERE in the
            # config, not just this component's own list, and that width is
            # load-bearing -- a component's per-element names need not be its
            # config entries' names:
            #   * `source.SourceA.t_0` addresses a source element named
            #     after its body STAR (SourceA is a star.name, echoed onto
            #     the source instance by normalize_config_block); before the
            #     8.6.17 split the same width covered the pre-split lens's
            #     borrowed source-star names.  The per-parameter `names`
            #     list is a manifest option and is not known until stage 3,
            #     long after this runs.
            #   * `mann.B.ks_offset` names a mann block that has no `name:`
            #     yet: this ConfigManager is built BEFORE the component loop
            #     in System.__init__, and mann/torres derive their name from
            #     their `star:` key inside their own __init__.
            # Both are covered because the borrowed name is always some other
            # component's instance name.  A genuine typo is a string that
            # appears nowhere, which is what we reject.  Numeric index forms
            # (`star.0.teff`) are pass-through by design.
            if (
                not comp_name.isdigit()
                and canonical_param_key(key, config) == key
                and comp_name not in _declared_instance_names(config)
            ):
                own = [
                    e["name"]
                    for e in comp_list
                    if isinstance(e, dict) and e.get("name") is not None
                ]
                own_msg = (
                    f"'{comp_type}' instances are: "
                    f"{', '.join(repr(n) for n in own)}."
                    if own
                    else f"No '{comp_type}' entry declares a 'name:', so its "
                    f"instances are addressable by index only."
                )
                raise ValueError(
                    f"\n!!! STRICT NAMING ERROR !!!\n"
                    f"Parameter '{key}' names the instance '{comp_name}', "
                    f"which is not declared by any component in your system "
                    f"configuration.\n"
                    f"{own_msg} "
                    f"Indices 0-{len(comp_list) - 1} also work "
                    f"(e.g. '{comp_type}.0.{param_name}'), as does the "
                    f"instance-less broadcast form "
                    f"'{comp_type}.{param_name}'.\n"
                    f"Fix the spelling, or delete the entry: it is otherwise "
                    f"ignored outright, silently discarding its value, bounds "
                    f"and prior."
                )

            # Numeric index or resolved name: canonical_param_key returns the
            # index form.  deepcopy so broadcast instances never share one
            # dict (see the aliasing fix in #76).
            standardized[canonical_param_key(key, config)] = copy.deepcopy(val)

        # Pass 2: expand 2-part keys for list components, merging PER FIELD
        # into whatever Pass 1 already wrote for that index (review 1.1.6).
        for key, val in user_params.items():
            parts = key.split(".", 2)
            if len(parts) != 2:
                continue

            comp_type, param_name = parts
            comp_list = config.get(comp_type)

            if not isinstance(comp_list, list):
                # flat-dict or unknown component: keep the key as-is
                standardized[key] = copy.deepcopy(val)
                continue

            for i in range(len(comp_list)):
                indexed_key = f"{comp_type}.{i}.{param_name}"
                if indexed_key not in standardized:
                    standardized[indexed_key] = copy.deepcopy(val)
                    continue

                # This index already carries a specific entry.  The specific
                # one wins FIELD BY FIELD; every field it does not mention is
                # inherited from the broadcast instead of being discarded.
                # Both are field dicts already (standardize_entries above).
                merged = copy.deepcopy(val)
                overriding = standardized[indexed_key]

                # AMBIGUOUS, so it is refused rather than guessed at: the
                # specific entry redefines `unit` while inheriting a NUMBER
                # from the broadcast.  A broadcast `{initval: 3, unit: solRad}`
                # under a specific `{unit: jupiterRad}` leaves "3" with no
                # determinate meaning -- 3 solRad restated in jupiterRad, or 3
                # jupiterRad?  Broadcasting is a convenience for the
                # unambiguous case; this is the ambiguous one, and the user
                # can say exactly what they mean by spelling the value out
                # per element.
                if "unit" in overriding and overriding["unit"] != merged.get(
                    "unit"
                ):
                    inherited = sorted(
                        f
                        for f in NUMERIC_KEYS
                        if f in merged and f not in overriding
                    )
                    if inherited:
                        raise ValueError(
                            f"\n!!! BROADCAST VALUE WITH A PER-ELEMENT UNIT "
                            f"!!!\n"
                            f"'{indexed_key}' overrides unit: "
                            f"{overriding['unit']!r}, but "
                            f"{', '.join(inherited)} would be inherited from "
                            f"the broadcast '{key}'"
                            + (
                                f", written in {merged['unit']!r}"
                                if "unit" in merged
                                else " (in the default unit)"
                            )
                            + f".\n"
                            f"A broadcast number has no determinate meaning "
                            f"once one element reads it in a different unit, "
                            f"so it is refused rather than guessed at.  "
                            f"Write {', '.join(inherited)} explicitly on "
                            f"'{indexed_key}' (and on the other elements, or "
                            f"leave the broadcast to cover them)."
                        )

                merged.update(overriding)
                standardized[indexed_key] = merged

        # Pass 3: 1-part and other unhandled keys (e.g. 'run').
        for key, val in user_params.items():
            if "." not in key and key not in standardized:
                standardized[key] = copy.deepcopy(val)

        return standardized

    def finalize_user_params(self):
        """
        Called by the System object AFTER all components have registered their hints.
        """
        # Reset structured diagnostics for this solve.  resolve_and_validate
        # runs once per seed and appends contradictions here; _record_diagnostic
        # dedupes so repeated seeds do not multiply identical entries.
        self.diagnostics = []
        for path, value, by, why in self.superseded_hints:
            self._record_diagnostic(
                "info",
                f"{path}: the data-derived start {value:.6g} was not used "
                f"-- {by} supersedes it ({why}).",
                [path, by],
                kind="superseded",
            )

        flat_params = {}
        name_to_index = {}

        # Build index mapping directly from the config keys (No translations!)
        for comp_key, entries in self.system_config.items():
            if isinstance(entries, list):
                for i, c in enumerate(entries):
                    if isinstance(c, dict) and "name" in c:
                        name_to_index[(comp_key, c["name"])] = i

        logger.debug("=" * 50 + "\nSYMBOL MAP DEBUGGING\n" + "=" * 50)

        def _user_start(data):
            """The start value a user entry states, `initval` else `mu`.

            THE FALLBACK IS THE POINT (review 1.1.3).  `resolve()` promotes a
            lone `mu` to the start -- deliberately, and pinned by
            tests/test_hint_resolution.py: a user's prior centre is a better
            start than an arbitrary defaults.yaml value.  So `mu` IS the start
            the model uses.  The engine used to read only `initval`, which
            meant a numeric `mu` reached it solely through the default-armor
            pass at PRECEDENCE_DEFAULT (20) -- so the value was right but its
            PRECEDENCE was wrong by 80, and any relation touching that symbol
            was free to solve it away from the number the user wrote.
            Downstream, the ledger, export_solution and initval_source all
            reported the hint or default rather than the user's own prior
            centre, and build_pymc errors blamed "data" for it.

            `probe_derivable` already falls back to `mu`, so before this the
            same input got two different answers at stage 1a and stage 4.
            Three readers, one rule.  `data` is always a field dict: a bare
            params value was translated to `{initval: ...}` at construction
            (review 1.1.7), so there is no shape to guess here.
            """
            val = data.get("initval")
            return data.get("mu") if val is None else val

        for path, data in self.user_params.items():
            val = _user_start(require_entry(path, data))
            # A list-valued initval is a set of per-seed start points (P4
            # multi-seed sampling).  The base flat_params below seeds the
            # relaxation engine with seed 0; _build_seed_overrides re-injects
            # each seed's element as a PRECEDENCE_USER override in the K-solve loop.
            if isinstance(val, (list, tuple)):
                val = val[0]
            if val is not None:
                translated_path = _index_path(path, name_to_index)

                sym = self.master_symbol_map.get(translated_path)
                if sym:
                    c_type = translated_path.split(".")[0]
                    p_name = translated_path.split(".")[-1]
                    factor = self.get_conversion_factor(
                        c_type, p_name, full_path=path
                    )
                    internal_val = float(val) * factor
                    logger.debug(
                        f"Mapped: {path} -> {translated_path} -> {sym} = {internal_val} (internal)"
                    )
                    flat_params[sym] = internal_val
                else:
                    if "." in path:
                        logger.debug(
                            f"Unmapped: {path} (tried translated: {translated_path})"
                        )

        # --- REGISTER LINK TARGETS AND DEPENDENCIES AS SYMBOLS ---
        # A link target/dep may not appear in any component's symbol map
        # (e.g. star.age has no symbolic relations); registering it here lets
        # the relaxation engine seed it from defaults.yaml and solve the
        # directed link assignments.
        for target, fields in self.links.items():
            for plink in fields.values():
                for path in [target] + list(plink.dep_paths):
                    if path not in self.master_symbol_map:
                        self.master_symbol_map[path] = sp.Symbol(path)
                        logger.debug(f"Registered link symbol: {path}")

        # --- FALLBACK TO LEAFS ---
        for path in list(self.user_params.keys()):
            if path not in self.master_symbol_map:
                self.master_symbol_map[path] = sp.Symbol(path)
                logger.debug(f"Registered as leaf: {path}")

                # Push the user's value into the solver's initial state
                data = user_entry(self.user_params, path)
                val = _user_start(data)  # initval else mu; review 1.1.3
                if isinstance(val, (list, tuple)):
                    val = val[0]  # seed 0; see per-seed handling below
                if val is not None:
                    c_type = path.split(".")[0]
                    p_name = path.split(".")[-1]
                    factor = self.get_conversion_factor(
                        c_type, p_name, full_path=path
                    )
                    flat_params[self.master_symbol_map[path]] = (
                        float(val) * factor
                    )

        base_flat = {str(k): v for k, v in flat_params.items()}

        # --- MULTI-SEED SOLVE (P4) ---
        # Build the K per-seed override sets in their two provenance channels
        # (user initval lists at PRECEDENCE_USER, component seed hints at
        # PRECEDENCE_DERIVED_DATA; both fall back to the shared base_flat for any
        # path they do not touch), then run the relaxation engine once per
        # seed inside this single prepare() call so every seed shares one symbol
        # environment and one relation ordering.
        # Bounds/scales are taken from seed 0 only -- seeds move the START, never
        # the bounds -- so self.propagated_scales is restored to seed 0's after
        # the loop.
        K, user_overrides, seed_hint_overrides = self._build_seed_overrides(
            name_to_index
        )

        # Solve seed 0 LAST so the final self.propagated_scales (and the
        # ledger) reflect seed 0 -- the seed whose bounds/scales the model
        # actually uses.  Only start positions vary between seeds;
        # bounds/scales do not.
        seed_resolved = [None] * K
        for k in list(range(1, K)) + [0]:
            flat_k = dict(base_flat)
            flat_k.update(user_overrides[k])
            seed_resolved[k] = self.resolve_and_validate_parameters(
                flat_k, seed_hints=seed_hint_overrides[k]
            )

        # seed 0 remains the canonical single start injected back into
        # user_params below; the full K-set is stored for get_raw_starts.
        resolved_flat = seed_resolved[0]
        self.seed_resolved = seed_resolved if K > 1 else None
        if K > 1:
            logger.info(f"Multi-seed sampling: solved {K} seed start points.")

        logger.debug("Solver finished.")

        # 4. INJECT BACK SAFELY
        for sym_node, val in resolved_flat.items():
            path = str(sym_node)
            parts = path.split(".")
            final_path = path

            if len(parts) == 3:
                comp_type, idx, param = parts
                for (c_type, c_name), i in name_to_index.items():
                    if c_type == comp_type and str(i) == str(idx):
                        final_path = f"{comp_type}.{c_name}.{param}"
                        break
                # `path` (index form), not `final_path` (name form): a user
                # `unit:` override lives under the standardized index key, so
                # looking it up by name silently falls back to the default
                # unit and the injected start is off by that factor.
                factor = self.get_conversion_factor(
                    comp_type, param, full_path=path
                )
                user_val = val / factor
            else:
                user_val = val

            # standardize_param_names stores entries under the index form
            # (star.0.teff) while final_path uses the name form (star.A.teff).
            # Check both so we don't create a spurious duplicate entry.
            #
            # Every entry is a field dict (review 1.1.7), so presence is the
            # whole test.  It used to also require `isinstance(..., dict)`,
            # which made a user's BARE entry look absent: it was overwritten
            # with a fresh `{initval, derived}` instead of being updated.
            existing_key = None
            for try_key in (final_path, path):
                if user_entry(self.user_params, try_key) is not None:
                    existing_key = try_key
                    break

            if existing_key is None:
                # A NEW entry is written in the INDEX form (`path`), never the
                # config-instance-name form (`final_path`).  resolve() looks a
                # per-element entry up under three keys -- `comp.param`,
                # `comp.<i>.param` and `comp.<names[i]>.param` -- and only the
                # index one is guaranteed to address element i: `names` is a
                # manifest option, so a component may label its elements with
                # something other than its own config instances' names.
                #
                # `lens` does exactly that (examples/ob161003): its per-source
                # vectors are named for the SOURCE STARS ("SourceA",
                # "SourceB"), while the lens block has a single entry named
                # "Lens" at index 0.  So the engine's answer for source slot 0
                # -- `lens.0.theta_E` -- used to be filed as
                # `lens.Lens.theta_E`, which matches none of element 0's three
                # keys, and the solved value was silently dropped.  With no
                # `initval` in mulensing/defaults.yaml (theta_E is derived) and
                # element 1 filed readably as `lens.1.theta_E`, apply_value
                # allocated a NaN-filled vector and wrote only element 1:
                # `lens.theta_E.initval == [nan, 0.839]`.  Element 0 was the
                # only one affected because index 0 is the only index a lens
                # instance name collides with.
                #
                # The index form is also the documented internal spelling (see
                # standardize_param_names) and is what get_conversion_factor,
                # propagated_scales, scale_hints and the engine's own symbol
                # paths already use.
                self.user_params[path] = {
                    "initval": user_val,
                    "derived": True,
                }
            else:
                existing = self.user_params[existing_key]
                # Don't clobber a user-specified Gaussian prior: if the user gave
                # mu but no initval, resolve() will use mu as the starting point.
                # Injecting the default-derived initval here would undo that.
                if "mu" in existing and "initval" not in existing:
                    existing["derived"] = True
                else:
                    existing["initval"] = user_val
                    existing["derived"] = True

        # --- SNAPSHOT USER-LINK EXPRESSIONS FOR STATIC FIELDS ---
        # sigma links are static by design; lower / upper links
        # additionally need a numeric snapshot so the logit transform can be
        # set up (the dynamic tensor bound replaces it at runtime).
        for target, fields in self.links.items():
            entry = user_entry(self.user_params, target)
            if entry is None:
                entry = {}
                self.user_params[target] = entry
            for fld, plink in fields.items():
                if fld in ("initval", "mu"):
                    continue  # handled by the directed relaxation pass
                if not all(d in resolved_flat for d in plink.dep_paths):
                    logger.warning(
                        f"Link '{target}.{fld} = {plink.expr_str}' could not be "
                        f"snapshot: unresolved dependencies "
                        f"{[d for d in plink.dep_paths if d not in resolved_flat]}."
                    )
                    continue
                try:
                    val_int = float(
                        self._link_internal_expr(plink).evalf(
                            subs=resolved_flat
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Link '{target}.{fld} = {plink.expr_str}' snapshot "
                        f"evaluation failed: {e}"
                    )
                    continue
                tparts = target.split(".")
                tf = self.get_conversion_factor(
                    tparts[0], tparts[-1], full_path=target
                )
                entry[fld] = val_int / tf
                logger.debug(
                    f"Link snapshot: {target}.{fld} = {entry[fld]:.6g} (user units)"
                )

    def _build_seed_overrides(self, name_to_index):
        """Assemble the per-seed override sets for multi-seed sampling.

        Two sources feed the seeds, and they are kept in SEPARATE channels
        because they carry different provenance:
          1. User initval lists in params.yaml (`initval: [v0, v1, ...]`) --
             PRECEDENCE_USER, merged into the engine's user_provided_params.
          2. Component seed hints (config_manager.seed_hint_sets), e.g. the
             peak finder -- PRECEDENCE_DERIVED_DATA, passed to the engine as
             `seed_hints` and layered in with the other data-driven hints.

        Merging them into one PRECEDENCE_USER dict (as this did until the 2.1.2
        review fix) had two consequences: a seed silently clobbered a user's
        *scalar* initval for the same path -- the scalar lives in base_flat,
        which the merged override dict overwrote -- and a seed that disagreed
        with a genuine user entry made every symbol of the connecting relation
        PRECEDENCE_USER, tripping the "over-constrained" contradiction clause.

        Returns (K, user_overrides, seed_hint_overrides) where K is the seed
        count and each override list has length K of {internal_path_str:
        internal_value} dicts.  Every initval list must have length K or 1
        (length-1 broadcasts to all seeds).  When no list initvals and no seed
        hints exist, K == 1 and both lists are [{}], exactly reproducing the
        legacy single-solve behavior.
        """
        # 1. User initval lists -> {sym_path: [internal values]}
        user_lists = {}
        #    Every entry is a field dict, so a bare per-seed list
        #    (`star.0.teff: [5000, 6000]`) arrives here as `{initval: [...]}`.
        #    This loop used to `continue` past any non-dict entry, and that
        #    bare spelling -- documented as legal -- silently kept only seed 0
        #    (review 1.1.7).
        for path, data in self.user_params.items():
            iv = require_entry(path, data).get("initval")
            if not isinstance(iv, (list, tuple)):
                continue
            sym_path = self._to_symbol_path(path, name_to_index)
            if sym_path is None:
                logger.warning(
                    f"List initval on '{path}' is not a known parameter path; "
                    f"multi-seed override ignored."
                )
                continue
            c_type, p_name = sym_path.split(".")[0], sym_path.split(".")[-1]
            factor = self.get_conversion_factor(c_type, p_name, full_path=path)
            user_lists[sym_path] = [float(x) * factor for x in iv]

        mm_sets = self.seed_hint_sets or []

        # 2. Determine K and validate list lengths.
        Ku = max((len(v) for v in user_lists.values()), default=1)
        Km = len(mm_sets)
        K = max(Ku, Km, 1)

        for p, v in user_lists.items():
            if len(v) not in (1, Ku):
                raise ValueError(
                    f"Inconsistent seed count: initval list for '{p}' has "
                    f"length {len(v)}, expected {Ku} or 1. All initval lists in "
                    f"a params file must share one length K (or be length 1)."
                )
        if Ku > 1 and Km > 1 and Ku != Km:
            raise ValueError(
                f"Seed-count mismatch: {Ku} user initval seeds vs {Km} "
                f"component seed hints. Provide matching counts (or length 1)."
            )

        # 3. Split per seed.  No merge: the two channels enter the engine at
        #    different ranks, and a path in both is resolved by rank, not by
        #    dict order (a user list is PRECEDENCE_USER and wins).
        user_overrides = []
        seed_hint_overrides = []
        for k in range(K):
            if mm_sets:
                src = mm_sets[k] if len(mm_sets) > 1 else mm_sets[0]
                seed_hint_overrides.append(dict(src))
            else:
                seed_hint_overrides.append({})
            user_overrides.append(
                {
                    p: (vals[k] if len(vals) > 1 else vals[0])
                    for p, vals in user_lists.items()
                }
            )

        return K, user_overrides, seed_hint_overrides

    def _to_symbol_path(self, path, name_to_index):
        """Translate a user_params key to the internal-index path string used by
        the relaxation engine (e.g. 'lens.Lens.t_0' -> 'lens.0.t_0').  Returns
        None if the path does not correspond to a registered symbol."""
        translated = _index_path(path, name_to_index)
        if translated in self.master_symbol_map:
            return translated
        if path in self.master_symbol_map:
            return path
        return None

    def _strip_user_init_scales(self):
        """Warn about and drop any init_scale entries in the user's params.

        init_scale is obsolete as a user-facing knob: the whitening scale is
        measured directly from the data at startup (see exozippy/whitening.py)
        and any user value would be overwritten anyway.  Old params files keep
        working -- the key is simply ignored with a warning.  Entries are
        removed via a rebuilt per-parameter dict; the caller's original dict is
        already insulated by standardize_param_names, which deepcopies every
        entry, so this is belt and braces for the no-config path that skips
        standardization.
        """
        stripped = []
        for k, v in list(self.user_params.items()):
            if "init_scale" in require_entry(k, v):
                self.user_params[k] = {
                    kk: vv for kk, vv in v.items() if kk != "init_scale"
                }
                stripped.append(k)
        if stripped:
            logger.warning(
                f"'init_scale' is obsolete and was ignored for: {stripped}. "
                f"Whitening scales are now measured directly from the data at "
                f"startup; remove init_scale from your params file."
            )

    def supersede_hint(self, path, value, by, why):
        """Record that a component did NOT push its data-derived start for
        `path` (value `value`, user unit) because the user's `by` states
        the same quantity (``Planet._push_rv_mass_hints``: an RV
        semi-amplitude estimate against a user planet mass).  It is the
        same-element rule -- a higher tier replaces a lower one outright --
        applied by the component that knows two paths are one quantity;
        the relaxation engine never sees the estimate.  Reported in the
        solve diagnostics (kind "superseded"), which finalize_user_params
        resets, so the record lives here and is re-emitted on every solve.
        """
        entry = (path, value, by, why)
        if entry not in self.superseded_hints:
            self.superseded_hints.append(entry)

    def record_diagnostic(self, severity, message, param_paths):
        """Public face of ``_record_diagnostic`` for a caller outside the
        solve -- ``System.prepare``'s topology revisions (review 2.9.15),
        recorded AFTER the last ``finalize_user_params``, which resets the
        list."""
        self._record_diagnostic(severity, message, param_paths, kind=None)

    def _record_diagnostic(self, severity, message, param_paths, kind=None):
        """Append a structured diagnostic (deduped) for the solve/validate API.

        severity is one of "error" | "warning" | "info"; param_paths is a list
        of the parameter paths involved.  Duplicate entries (same severity,
        message, and paths) -- e.g. the same contradiction seen once per seed
        -- collapse to a single record.  ``kind`` names what the engine found
        ("contradiction": user values that break a relation among themselves;
        "dropped": a user value on a derived element that could not be kept;
        "unsettled": a tier that could not absorb a relation inside its
        bounds), so ``ModelAuditor.check_user_starts`` can word each without
        parsing the message.  The first path of a "dropped" entry is the
        value that was dropped.
        """
        entry = {
            "severity": severity,
            "message": message,
            "param_paths": list(param_paths),
        }
        if kind is not None:
            entry["kind"] = kind
        if entry not in self.diagnostics:
            self.diagnostics.append(entry)

    def set_engine_roles(self, roles):
        """Tell the engine which paths are SAMPLED, DERIVED or INACTIVE.

        ``roles`` is ``{internal_path: "sampled" | "derived" | "inactive"}``
        for every element a manifest declares (``System._engine_roles``); a
        path it does not name is not a parameter of this model and is
        treated as derived.  Set by ``System.prepare`` immediately before
        each ``finalize_user_params`` -- after the solve-input snapshot, so
        that a revision's second registration pass probes with the roles
        unknown, exactly as the first did.  The roles decide what is a SEED
        (relaxation.Anchor): a defaults.yaml value seeds only a sampled
        element, and a user value is produced exactly only there.
        """
        self.engine_roles = dict(roles) if roles is not None else None

    def add_constant(self, path, value):
        """Supply an exact NUMBER the relations may use (review 8.6.23).

        Some relations need a per-fit constant that is not a parameter and
        is known only once the data are read: Earth's projected velocity at
        t0_par, which turns the heliocentric relative proper motion into the
        geocentric one t_E and pi_E are defined in.  The component maps the
        symbol in its ``get_symbol_map`` (so it is an engine path) and calls
        this with the value in INTERNAL units; the relaxation engine then
        substitutes it into every relation, holds it fixed, never solves
        for it, never seeds anything with it and never reports it (it is in
        no ledger, no inject-back and no export).  A relation whose constant
        has not been supplied yet -- a stage-1 probe before the component
        knows it -- is simply not usable, which is what an unknown symbol
        already means.  A constant is a component input, like a hint:
        re-supplied by a second registration pass, not rolled back.
        Canonical index-form path only.
        """
        parts = str(path).split(".")
        if len(parts) != 3 or not parts[1].isdigit():
            raise ValueError(
                f"add_constant: '{path}' is not a canonical index-form path "
                f"(<component>.<index>.<name>)."
            )
        if path not in self.master_symbol_map:
            raise KeyError(
                f"add_constant: '{path}' is not an engine symbol; map it in "
                f"the component's get_symbol_map first."
            )
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(
                f"add_constant: '{path}' = {value} is not finite."
            )
        self.constants[path] = value

    def declare_likelihood_backed(self, path):
        """Declare that a stiff likelihood term backs `path`'s start.

        The start of that parameter -- whatever seeds it below
        PRECEDENCE_LIKELIHOOD_BACKED: a defaults.yaml value, a distance
        guess -- then DECIDES at that level, so poorly constrained seeds
        absorb a contradiction before it does (review 3.14.22, Q3: "a crazy
        stellar mass might be a tough one to recover if we constrain it by
        evolutionary models").  It still REPORTS the rank it came with.  A
        user value or a higher hint is untouched.  Called from a component's
        register_parameters (stage 3) for the parameters its own likelihood
        constrains: the evolutionary model's mass, EEP, initial [Fe/H] and
        age; the SED's own temperature; an empirical relation's
        mass and radius.  Any of the three spellings; stored in index form.
        """
        translated, _ = self._translate_and_scale(path, 0.0)
        self.likelihood_backed.add(translated)

    def _provenance_label(self, rank):
        """Map a numeric provenance rank to a coarse source label."""
        if rank is None:
            return "default"
        if rank >= PRECEDENCE_USER:
            return "user"
        # Every rank a component pushes a hint at -- see DATA_PRECEDENCES for why
        # membership and not a numeric band.  Hardcoding the two literals
        # here meant any NEW intermediate data rank silently reported
        # "solved", which is a wrong provenance in the startup table, in
        # export_solution and in the GUI.
        if rank in DATA_PRECEDENCES:
            return "data"
        if rank > PRECEDENCE_DEFAULT:
            return "solved"
        return "default"

    def initval_source(self, component, param, element=None, name=None):
        """Where did this element's START VALUE come from?

        Returns one of the ``_provenance_label`` strings -- "user" (written in
        the params file), "data" (a component's data-derived hint), "solved"
        (the relaxation engine derived it from other inputs) or "default"
        (defaults.yaml) -- for the parameter ``component.param`` at element
        index ``element``.

        This exists so an error about a start value can say WHOSE start value
        it is: blaming a user's params file for a number the engine derived is
        worse than saying nothing.  Parameter.build_pymc is the only caller
        (Component.add_parameter hands it this bound method).

        Two known limits, both of which only soften the wording of a message
        and never change a decision:
          - it reports the provenance of the engine's LAST solve, so a
            parameter the engine rewrote after reading a user value reports
            the rewrite ("solved"), not "user";
          - a parameter with no symbol in the master symbol map has no rank at
            all, so it falls back to looking for the user's own numeric
            ``initval`` in the params file (safe there: the inject-back only
            writes into mapped paths).
        """
        # Index form first (what _last_provenance and a standardized
        # user_params are keyed on), then the instance-name form (which
        # survives when no system_config was attached, e.g. a unit test or a
        # component driven directly), then the 2-part broadcast form.
        paths = []
        if element is not None:
            paths.append(f"{component}.{element}.{param}")
        if name is not None:
            paths.append(f"{component}.{name}.{param}")
        paths.append(f"{component}.{param}")

        for path in paths:
            rank = (self._last_provenance or {}).get(path)
            if rank is not None:
                return self._provenance_label(rank)

        for path in paths:
            entry = user_entry(self.user_params, path)
            if entry is not None and entry.get("initval") is not None:
                return "user"
            linked = (self.links or {}).get(path, {})
            for fld in ("initval", "mu"):
                if fld in linked:
                    # A component link (``add_link``) is a tie the topology
                    # implies, not a value the user wrote.
                    return "user" if linked[fld].origin == "user" else "solved"
        return "default"

    def export_solution(
        self,
        derived_params=None,
        active_elements=None,
        manifest_overrides=None,
    ):
        """Export the resolved parameter solution as JSON-friendly dicts.

        `derived_params`, when given, says which parameters the built manifests
        actually derive.  Pass it whenever a System is in scope: the fallback --
        "this parameter has an expressions: block in its defaults.yaml" -- is
        only an approximation, since a component may declare the same parameter
        free in one topology and derived in another (see planet.mass's linear vs
        log_q coordinate).  Two spellings are accepted:

          * `System.derived_elements()` -- `{(prefix, param): boolean mask}`,
            the PER-ELEMENT answer, and the one to prefer: a component whose
            instances chose different parameterizations derives only some
            elements, and reporting the whole vector either way mislabels the
            rest.  Consumers act on the label (`_bounds_diagnostics` skips
            derived parameters entirely), so a mislabelled element is a check
            that silently does not run.
          * `System.derived_params()` -- a set of `(prefix, param)` pairs, kept
            for callers that have no element context.

        `active_elements` (`System.active_elements()`) marks elements that are
        not parameters of their instance's parameterization at all; they export
        with `"active": False` so a reporting consumer can leave them out, as
        the tables do.

        `manifest_overrides` (`System.manifest_overrides()`) is the third
        after-prepare() table, and it is needed for the same reason as the
        other two: what the BUILD does is decided by the manifest, and a
        report that re-derives it from the config alone disagrees.  Component
        `"overrides"` are the channel a component pins an element through
        (`sigma: 0` for a GP hyperparameter of a file that did not ask for a
        GP, a robust-likelihood parameter of a file with no `likelihood:`, an
        unread limb-darkening coefficient), and `resolve()` only applies them
        when it is handed them -- so without this table every such element
        exported `fixed: False` and the GUI's Tune tab drew a slider for a
        knob the sampler never moves.  Reporting-only: the build has always
        been correct, since `Component.add_parameter` passes them.

        Returns a dict with:
          - "parameters": {user_path: {value, unit, internal_unit, lower,
            upper, init_scale, sigma, mu, fixed, derived, active, provenance}}
            where provenance is {rank, label, relation}.  All numeric fields
            are in the parameter's user unit (as reported by resolve()).
          - "seeds": a list of {user_path: value} start points, present only
            when multi-seed sampling produced more than one seed.

        Reads only in-memory state left behind by finalize_user_params; it does
        NOT build the PyMC model.  Must be called after System.prepare().

        Note: the exported values for solved quantities are one valid
        solution, not a canonical one -- a system of relations can admit
        several and the engine picks by a fixed rule.  It is nonetheless
        REPRODUCIBLE: the cross-build nondeterminism this note used to warn
        about (unsorted free_symbols / rglob walks) is fixed.  Bounds were
        never part of it either way; the engine only reads them.  See the
        solve_api module docstring.
        """

        def _clean(x):
            if x is None:
                return None
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(xf):
                return None
            return xf

        # Build an index -> name map per component for readable paths.
        idx_to_name = {}
        for comp_key, entries in (self.system_config or {}).items():
            if isinstance(entries, list):
                for i, c in enumerate(entries):
                    if isinstance(c, dict) and "name" in c:
                        idx_to_name[(comp_key, str(i))] = str(c["name"])

        def _display_path(internal_path):
            if internal_path in self._owned_display:
                return self._owned_display[internal_path]
            parts = internal_path.split(".")
            if len(parts) == 3 and (parts[0], parts[1]) in idx_to_name:
                return f"{parts[0]}.{idx_to_name[(parts[0], parts[1])]}.{parts[2]}"
            return internal_path

        parameters = {}
        for internal_path in self.master_symbol_map:
            if internal_path in self.constants:
                continue  # an exact number the relations use, no parameter
            parts = internal_path.split(".")
            c_type = parts[0]
            p_name = parts[-1]
            # Skip instance-less (2-part) symbol entries for components that are
            # instanced as a list. These are abstract relaxation-engine symbols
            # (e.g. a bare "star.feh" that SED physics references) with no owning
            # instance; the real parameters are the per-instance "star.A.feh".
            # Exporting them surfaced orphaned "star" rows in the GUI tree that
            # correspond to no actual star.
            if len(parts) == 2 and isinstance(
                (self.system_config or {}).get(c_type), list
            ):
                continue
            el = (
                int(parts[1])
                if len(parts) == 3 and parts[1].isdigit()
                else None
            )
            cfg = self.resolve(
                c_type,
                p_name,
                element=el,
                internal_overrides=(manifest_overrides or {}).get(
                    (c_type, p_name)
                ),
            )

            def _first(key):
                arr = cfg.get(key)
                if arr is None:
                    return None
                try:
                    return _clean(np.atleast_1d(arr)[0])
                except (IndexError, TypeError):
                    return _clean(arr)

            sigma = _first("sigma")
            # Prefer the engine's solved value (index-form paths that match the
            # symbol map exactly, in internal units) over resolve()'s initval:
            # finalize injects derived initvals under the name-form path, which
            # a nameless resolve() call cannot see.
            if internal_path in self._last_resolved:
                factor = (
                    self.get_conversion_factor(
                        c_type, p_name, full_path=internal_path
                    )
                    or 1.0
                )
                value = _clean(self._last_resolved[internal_path] / factor)
            else:
                value = _first("initval")
            derived = _element_flag(
                derived_params,
                (c_type, p_name),
                el,
                fallback=bool(cfg.get("expressions")),
                absent=False,
            )
            active = _element_flag(
                active_elements,
                (c_type, p_name),
                el,
                fallback=True,
                absent=True,
            )
            # A parameter is fixed when sigma == 0 -- the ONE documented pin
            # (see Parameter.build_pymc).  This used to read
            # `cfg.get("value") is not None or ...`, which described a channel
            # that does not exist: resolve() never emits a "value" key, so the
            # disjunct was dead and its only effect was to invite someone to
            # "fix" a pin bug by populating it.
            fixed = sigma is not None and sigma == 0
            # An INACTIVE element is held at a bookkeeping value whatever its
            # resolved sigma says (the pin is the role, not a `sigma: 0` in the
            # config), so it must not be reported as free: a consumer that draws
            # a slider per free parameter would offer a knob that moves nothing.
            fixed = fixed or not active

            rank = self._last_provenance.get(internal_path)
            relation = self._last_solved_by.get(internal_path)
            label = self._provenance_label(rank)

            parameters[_display_path(internal_path)] = {
                "value": value,
                "unit": cfg.get("unit"),
                "internal_unit": cfg.get("internal_unit"),
                "lower": _first("lower"),
                "upper": _first("upper"),
                "init_scale": _first("init_scale"),
                "sigma": sigma,
                "mu": _first("mu"),
                "fixed": bool(fixed),
                "derived": derived,
                "active": active,
                "provenance": {
                    "rank": rank,
                    "label": label,
                    "relation": relation if label == "solved" else None,
                },
            }

        result = {"parameters": parameters}

        # Multi-seed start points, converted to user units and readable paths.
        if self.seed_resolved and len(self.seed_resolved) > 1:
            seeds = []
            for seed in self.seed_resolved:
                seed_out = {}
                for internal_path, internal_val in seed.items():
                    parts = internal_path.split(".")
                    c_type = parts[0]
                    p_name = parts[-1]
                    factor = (
                        self.get_conversion_factor(
                            c_type, p_name, full_path=internal_path
                        )
                        or 1.0
                    )
                    seed_out[_display_path(internal_path)] = _clean(
                        internal_val / factor
                    )
                seeds.append(seed_out)
            result["seeds"] = seeds

        return result

    # Every mutable ConfigManager attribute the relaxation engine writes to.
    # probe_derivable deep-copies each one before running the engine and puts
    # the copy back in a finally block, which is what makes a probe genuinely
    # read-only.  If the engine ever starts writing somewhere new, it belongs
    # in this tuple -- a mutation missing from it silently survives the probe.
    _PROBE_SNAPSHOT_ATTRS = (
        "user_params",  # finalize injects the solved starts back
        "diagnostics",  # _record_diagnostic appends contradictions
        "propagated_scales",  # refreshed at the end of every solve
        "symbolic_blacklist",  # a 2 s sp.solve timeout adds the target
        "_last_provenance",
        "_last_scale_provenance",
        "_last_resolved",
        "_last_solved_by",
        "_last_keep_report",
        # Not written by the engine, but by System.prepare right before each
        # finalize; in the snapshot so restore_solve_inputs returns a
        # revision's second registration pass to "roles unknown", the state
        # the first pass's stage-3 probes ran in.
        "engine_roles",
    )
    # _symbolic_solve_cache is deliberately NOT here.  Everything above is a
    # RESULT the probe must not leave behind -- a solved value, a rank, a
    # blacklist entry earned under the probe's own inputs.  The solve cache is
    # a memo of sp.solve, which is a pure function of (equation, target) and
    # sees no values at all, so what the probe learns there is equally true
    # afterwards; keeping it is the point (the probe runs at stage 1a and the
    # real solve repeats the same inversions at stage 3).  Contrast the
    # blacklist, which IS rolled back precisely because it is input-dependent:
    # a 2 s timeout under the probe's inputs says nothing about stage 3's.

    def snapshot_solve_inputs(self):
        """What a SECOND ``finalize_user_params`` must start from.

        ``finalize_user_params`` is not re-entrant: it injects the engine's
        solved starts back into ``user_params``, so a second call would read
        every solved start as one the user WROTE -- PRECEDENCE_USER in the
        ledger, "user" in ``initval_source``, and in every stage-3 reader of
        ``user_params``.  ``System.prepare`` takes this snapshot before the
        first solve and hands it to ``restore_solve_inputs`` before a
        topology revision's second one (review 2.9.15).  Same attributes the
        probe rolls back, plus the multi-seed result and the symbol map.
        Hints are deliberately NOT included: they are component inputs, which
        the second registration pass re-pushes (or, for a tie the revision
        dropped, keeps from the first -- a measurement does not stop being
        one because a prior that used it was cut).
        """
        saved = {
            attr: copy.deepcopy(getattr(self, attr))
            for attr in self._PROBE_SNAPSHOT_ATTRS + ("seed_resolved",)
        }
        return saved, dict(self.master_symbol_map)

    def restore_solve_inputs(self, snapshot):
        """Undo everything ``finalize_user_params`` wrote since
        ``snapshot_solve_inputs``.  See there."""
        saved, symbols = snapshot
        for attr, value in saved.items():
            setattr(self, attr, copy.deepcopy(value))
        self.master_symbol_map.clear()
        self.master_symbol_map.update(symbols)

    def probe_derivable(self, paths):
        """Which of `paths` the relaxation engine can pin down from what is
        known now, as opposed to falling back on a bare defaults.yaml value.

        Runs the engine on a snapshot and rolls every mutation back, so this
        is a read-only question: `finalize_user_params` still does the real
        solve later, from the same inputs plus whatever stages 1-3 add.

        The test is on **provenance**, not on presence: the engine's "default
        armor" step seeds every mapped path from defaults.yaml, so almost
        everything comes back resolved.  A rank above PRECEDENCE_DEFAULT means the
        value traces back to a user entry or a component hint rather than to
        a default -- i.e. someone actually told us, directly or through a
        relation.

        Called at stage 1 (before most hints exist), so a False here means
        "not derivable *yet*"; callers that must decide early -- notably
        globalsearch's seed gate -- get the conservative answer.

        An engine failure RAISES (review 2.1.17).  It used to be swallowed as
        "not derivable", which flipped a seeder's trigger on an engine or
        bookkeeping bug; the probe solves a subset of stage 4's inputs, so a
        failure here is never an expected outcome of knowing less.
        """
        _, ranks = self._probe_solve(paths)
        return {p for p in paths if ranks.get(p, 0) > PRECEDENCE_DEFAULT}

    def probe_start(self, paths):
        """The start value the relaxation engine would give each of `paths`
        from what is known NOW, with its provenance -- read-only.

        THE READER TO USE AT STAGES 1-3 (config.md, "Reading a start before
        stage 4").  ``resolve()`` and raw ``user_params`` show only what was
        WRITTEN for a path; a start the user's entries imply through a
        relation (``star.mass: 0.92`` -> ``star.logmass``; masses, distances
        and proper motions -> ``mulensevent.t_E``) does not exist until
        ``finalize_user_params`` runs the engine at stage 4, so an earlier
        reader silently sees the defaults.yaml value, or nothing.  This runs
        the same engine on the same snapshot ``probe_derivable`` uses and
        returns the solved values instead of discarding them.

        Inputs are the user entries (seed 0 of an ``initval`` list), every
        hint pushed so far, and seed 0 of ``seed_hint_sets`` -- what stage 4's
        seed-0 solve layers.  THE LIMIT: it sees only hints pushed so far.  A
        hint a later component pushes, or the caller's own, can still move the
        stage-4 start; the answer is "the start given everything known now",
        which is the most a stage 1-3 reader can have.

        Args:
            paths: canonical INDEX-form paths (``star.0.logmass``,
                ``mulensevent.0.t_E``).  A name-form or 2-part path is a
                caller bug and raises ``ValueError``; a path naming no
                parameter the defaults declare raises ``KeyError``.  A
                parameter no relation mentions is answered as a leaf (its
                start is whatever resolve() and the hints give it, at the
                provenance they carry), which is what stage 4 does with it.

        Returns:
            ``{path: ProbedStart}`` -- ``value`` in INTERNAL units,
            ``user_value`` in the path's USER units, ``rank``/``source`` the
            provenance.  A path the engine cannot reach at all comes back with
            ``source == "not derivable"`` and None values (see
            ``ProbedStart``); a reader must handle that case explicitly.

        Raises:
            RuntimeError: the engine itself failed; names the paths.  Never
                swallowed -- see ``probe_derivable`` (review 2.1.17).

        Not cached.  The inputs (user_params, hints and their ranks, seed
        hints, overrides, links, the symbol map) are all mutable and are read
        by ``resolve()`` deep inside the solve, so no cheap key is safe from
        going stale; a caller instead asks for every path it needs in ONE
        call (the EEP seed once per star, the flux bootstrap once per
        load_data).
        """
        paths = list(paths)
        for p in paths:
            parts = p.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(
                    f"probe_start: '{p}' is not a canonical index-form path "
                    f"(<component>.<index>.<param>)."
                )
            c_type, p_name = parts[0], parts[2]
            if (
                p not in self.master_symbol_map
                and p_name not in self.base_defaults.get(c_type, {})
            ):
                raise KeyError(
                    f"probe_start: '{p}' names no parameter -- it is not an "
                    f"engine symbol and {c_type}/defaults.yaml declares no "
                    f"'{p_name}'."
                )
        sets = self.seed_hint_sets or []
        resolved, ranks = self._probe_solve(
            paths, seed_hints=dict(sets[0]) if sets else None
        )
        out = {}
        for p in paths:
            if p not in resolved:
                out[p] = ProbedStart(p, None, None, None, "not derivable")
                continue
            val = float(resolved[p])
            rank = ranks[p]
            out[p] = ProbedStart(
                p,
                val,
                self.from_internal(p, val),
                rank,
                self._provenance_label(rank),
            )
        return out

    def from_internal(self, path, value):
        """Convert an INTERNAL-unit value of `path` to its USER unit.

        The ConfigManager-side twin of ``Parameter.from_internal``.
        ``get_conversion_factor`` is the user -> internal multiplier (the
        RECIPROCAL of Parameter's factor; CLAUDE.md invariants), so this
        divides by it.  `path` is the canonical index form, which is where a
        user ``unit:`` override lives after standardize_param_names.
        """
        parts = path.split(".")
        factor = self.get_conversion_factor(
            parts[0], parts[-1], full_path=path
        )
        return float(value) / factor

    def to_internal(self, path, value):
        """Convert a USER-unit value of `path` to its INTERNAL unit.

        The inverse of ``from_internal``: ``get_conversion_factor`` is the
        user -> internal multiplier, so this multiplies by it.
        """
        parts = path.split(".")
        factor = self.get_conversion_factor(
            parts[0], parts[-1], full_path=path
        )
        return float(value) * factor

    def _probe_solve(self, paths, seed_hints=None):
        """Run the engine on a snapshot and roll every mutation back.

        The ONE snapshot/rollback shared by ``probe_derivable`` and
        ``probe_start``.  Returns ``(resolved, provenance)`` -- the engine's
        {internal_path: internal value} and {internal_path: rank} -- as copies
        that outlive the rollback.

        A requested path that is not an engine symbol is registered as a LEAF
        for the duration of the probe, as finalize_user_params' "fallback to
        leafs" pass does for a user entry, so it enters the default-armor pass
        and a user entry on it keeps PRECEDENCE_USER.  The symbol map is
        restored afterwards.

        An exception from the engine is re-raised naming `paths`; see
        ``probe_derivable``'s docstring for why it is never swallowed.
        """
        saved_symbols = dict(self.master_symbol_map)
        # The engine writes back init_scale into user_params, appends
        # diagnostics, blacklists any inversion whose sp.solve hits the 2 s
        # alarm, and refreshes the export snapshots.  None of that may leak
        # out of a probe -- least of all the blacklist, which is consulted
        # for the rest of the process: one slow inversion during this
        # throwaway stage-1a probe would otherwise disable that relation for
        # the real stage-3 solve, which has different inputs and might well
        # have solved it in time.
        saved = {
            attr: copy.deepcopy(getattr(self, attr))
            for attr in self._PROBE_SNAPSHOT_ATTRS
        }
        prev_level = logger.level
        try:
            logger.setLevel(logging.WARNING)
            for p in paths:
                if p not in self.master_symbol_map:
                    self.master_symbol_map[p] = sp.Symbol(p)

            flat = {}
            for upath, data in self.user_params.items():
                sym = self.master_symbol_map.get(upath)
                if sym is None:
                    continue
                data = require_entry(upath, data)
                val = data.get("initval")
                if val is None:
                    val = data.get("mu")
                if isinstance(val, (list, tuple)):
                    val = val[0] if len(val) else None
                if val is None:
                    continue
                c_type, p_name = upath.split(".")[0], upath.split(".")[-1]
                factor = (
                    self.get_conversion_factor(c_type, p_name, full_path=upath)
                    or 1.0
                )
                flat[str(sym)] = float(val) * factor

            try:
                resolved = self.resolve_and_validate_parameters(
                    flat, seed_hints=seed_hints
                )
            except Exception as e:
                raise RuntimeError(
                    f"The relaxation engine failed while probing the start of "
                    f"{sorted(paths)} ({type(e).__name__}: {e}). The probe "
                    f"solves a subset of stage 4's inputs, so this is an "
                    f"engine or bookkeeping bug, not a missing input."
                ) from e
            return (
                {str(k): v for k, v in resolved.items()},
                dict(self._last_provenance),
            )
        finally:
            logger.setLevel(prev_level)
            for attr, value in saved.items():
                setattr(self, attr, value)
            self.master_symbol_map.clear()
            self.master_symbol_map.update(saved_symbols)

    def resolve_and_validate_parameters(
        self, user_provided_params, seed_hints=None
    ):
        """Run the relaxation engine (``relaxation.Engine``) once.

        ``user_provided_params`` is {internal_path: internal_value}, the
        user's own starts (seed 0 or one seed's set); ``seed_hints`` is this
        seed's hint set (see ``_build_seed_overrides``).  Returns
        {internal_path: internal value} and leaves behind the ledger
        (``_last_resolved``, ``_last_provenance``, ``_last_solved_by``), the
        preliminary scales (``propagated_scales``), the keep-in-order report
        (``_last_keep_report``) and the diagnostics.

        This method only builds the engine's inputs -- one ANCHOR per path
        (``relaxation.Anchor``), the declared scales, the roles -- and reads
        the outputs; the rule itself is ``relaxation.py``'s.
        """
        from .relaxation import KEEP_ORDER, Anchor, Engine

        # THE TRANSLATION WAS TOTAL, or we stop here (review 3.14.15c).
        # Everything below indexes by PATH, so a non-canonical key does not
        # fail -- it quietly creates a phantom second entry.
        self._assert_canonical_engine_paths(seed_hints)

        L = PRECEDENCE_LADDER
        roles = self.engine_roles
        partial_roles, registered, unknown = {}, set(), set()
        if roles is None and self.role_provider is not None:
            partial_roles, registered, unknown = self.role_provider()
        user = {str(k): float(v) for k, v in user_provided_params.items()}
        seed_hints = dict(seed_hints or {})

        def to_scalar(val):
            return val.item() if hasattr(val, "item") else float(val)

        # Every engine symbol, plus every hinted path (resolve() reads hints
        # too, and a hinted parameter need not be an engine symbol -- the
        # microlensing flux hints -- but its start passes through the ledger
        # the same way).
        paths = list(self.master_symbol_map)
        seen = set(paths)
        for extra in list(self.hints) + list(seed_hints):
            if extra not in seen:
                paths.append(extra)
                seen.add(extra)

        anchors = {}
        declared = {}
        scale_rank = {}
        periodic = {}
        # The DECLARED scales are read with the previous solve's propagated
        # scales switched off: resolve() layers those as an init_scale
        # default, which would make one seed's (or one probe's) solve an
        # input to the next.
        saved_scales = self.propagated_scales
        self.propagated_scales = {}
        try:
            for path in paths:
                parts = path.split(".")
                c_type, p_name = parts[0], parts[-1]
                el = (
                    int(parts[1])
                    if len(parts) == 3 and parts[1].isdigit()
                    else None
                )
                cfg = self.resolve(c_type, p_name, element=el)
                factor = self.get_conversion_factor(
                    c_type, p_name, full_path=path
                )
                pdef = self.base_defaults.get(c_type, {}).get(p_name) or {}
                rank_name = pdef.get("rank")
                if rank_name is None:
                    rank_name = (self.base_defaults.get(p_name) or {}).get(
                        "rank"
                    )
                default_rank = (
                    L["DEFAULT"]
                    if rank_name is None
                    else ladder_level(
                        rank_name, f"{c_type}/defaults.yaml '{p_name}'"
                    )
                )
                # resolve()'s init_scale, unless it is the user's `sigma: 0`
                # (resolve() syncs a sigma into init_scale, and a pin is not
                # a scale): then the defaults.yaml one it overwrote.
                s_val = cfg.get("init_scale")
                s_val = None if s_val is None else to_scalar(s_val)
                if s_val is None or not s_val > 0:
                    s_val = pdef.get("init_scale")
                if s_val is not None and float(s_val) > 0:
                    declared[path] = float(s_val) * factor
                    scale_rank[path] = default_rank
                per = pdef.get("periodic")
                if isinstance(per, dict) and "value" in per:
                    i_unit = pdef.get("internal_unit") or ""
                    pf = (
                        unit_conversion(per["unit"], i_unit, path)
                        if per.get("unit") and i_unit
                        else 1.0
                    )
                    periodic[path] = float(per["value"]) * pf

                if roles is not None:
                    role = roles.get(path, "derived")
                elif c_type in registered and path not in unknown:
                    # a stage 1-3 probe, but this component has already
                    # declared its manifest (System._declared_engine_roles)
                    role = partial_roles.get(path, "derived")
                else:
                    # A stage 1-3 probe: no manifest exists yet, so the role
                    # is read off defaults.yaml through the component
                    # contract (components.md): a SAMPLED parameter declares
                    # lower, upper and an init_scale, so one with an
                    # `expressions:` block that lacks any of them can only be
                    # derived (t_E, theta_E, the relative proper motions, a
                    # lens q).  One with all of them is ambiguous -- distance,
                    # logmass and the proper motions carry alternative
                    # parameterizations -- and is read as sampled: the common
                    # topology, and a seed the probe keeps is the
                    # conservative answer.  A parameter with no expression
                    # is sampled everywhere.
                    samplable = all(
                        pdef.get(k) is not None
                        for k in ("lower", "upper", "init_scale")
                    )
                    role = (
                        "derived"
                        if cfg.get("expressions") and not samplable
                        else "sampled"
                    )
                sampled = role == "sampled"
                if path in user:
                    anchors[path] = Anchor(
                        user[path],
                        "rigid" if sampled else "candidate",
                        L["USER"],
                        L["USER"],
                        sampled,
                        "user",
                    )
                    continue
                hint_rank = (
                    self.hint_ranks.get(path, L["DERIVED_DATA"])
                    if path in self.hints
                    else None
                )
                # A seed set (the peak finder's) wins a tie with an ordinary
                # component hint: a per-seed fit of the light curve is more
                # informative than a guess pushed for every seed alike.  A
                # hint ABOVE the data level (a start derived from the user's
                # own numbers) still beats it.
                if path in seed_hints and (
                    hint_rank is None or hint_rank <= L["DERIVED_DATA"]
                ):
                    val, rank, src = (
                        seed_hints[path],
                        L["DERIVED_DATA"],
                        "seed hint",
                    )
                elif hint_rank is not None:
                    val, rank, src = self.hints[path], hint_rank, "hint"
                elif cfg.get("initval") is not None:
                    val, rank, src = (
                        to_scalar(cfg["initval"]) * factor,
                        default_rank,
                        "default",
                    )
                else:
                    continue
                # A defaults.yaml value seeds only a SAMPLED element: on a
                # derived one it can never reach the model, so it is a fill
                # (a value for a symbol nothing else determines), never a
                # statement that can move another value.
                kind = "seed" if (sampled or src != "default") else "fill"
                tier = rank
                if (
                    kind == "seed"
                    and path in self.likelihood_backed
                    and tier < L["LIKELIHOOD_BACKED"]
                ):
                    tier = L["LIKELIHOOD_BACKED"]
                anchors[path] = Anchor(
                    float(val), kind, tier, rank, sampled, src
                )
        finally:
            self.propagated_scales = saved_scales

        # Scale hints (exact, already-normalized paths) and user sigmas: a
        # user's prior width is the best preliminary scale there is.
        for hint_path, hint_scale in self.scale_hints.items():
            if hint_path in seen:
                declared[hint_path] = hint_scale
                scale_rank[hint_path] = L["DERIVED_DATA"]
        for path in self.master_symbol_map:
            up = user_entry(self.user_params, path)
            if up is None:
                continue
            sig = up.get("sigma")
            if sig is not None and not isinstance(sig, str) and float(sig) > 0:
                parts = path.split(".")
                factor = self.get_conversion_factor(
                    parts[0], parts[-1], full_path=path
                )
                declared[path] = float(sig) * factor
                scale_rank[path] = L["USER"]

        # Directed user links (initval/mu): target := expr(deps), USER.
        directed_links = []
        for link_target, link_fields in self.links.items():
            for link_field, plink in link_fields.items():
                if link_field in ("initval", "mu"):
                    directed_links.append(
                        (
                            link_target,
                            link_field,
                            plink,
                            self._link_internal_expr(plink),
                        )
                    )

        file_order = {}
        for i, key in enumerate(self.user_params):
            file_order.setdefault(key, i)

        logger.info(
            "Solving for starting values/scales of sampled parameters given "
            "user/data/default initialization...."
        )
        engine = Engine(
            self,
            anchors,
            declared,
            file_order,
            directed_links,
            periodic,
            ladder=L,
            keep_order_mode=KEEP_ORDER,
            constants=self.constants,
        )
        resolved = engine.solve()
        logger.info(
            f"Done solving ({engine.keep_report.get('wall_s', 0.0):.2f} s)."
        )

        provenance = {p: r for p, r in engine.ranks.items() if p in resolved}
        solved_by = {}
        for st in engine.plan.steps:
            solved_by[st.target] = st.label
        for m, (a_m, x_m, tier_name) in engine.moved.items():
            solved_by[m] = f"split among the {tier_name} starts"
        for p, v in engine.solver_seeds.items():
            solved_by.setdefault(p, "standalone solver")
        from .diagnostics import USER_START_ATOL, USER_START_RTOL

        for severity, message, paths_, kind in engine.diagnostics:
            if kind == "dropped":
                # The rule drops a value that does not hold EXACTLY; the
                # user hears about it only where that is material, on the
                # same footing check_user_starts reports a miss.  A
                # published t_E rounded to 173 against a chain that implies
                # 172.9987 is honored to 8e-6 and is not news.
                p = paths_[0]
                want = anchors[p].value
                got = resolved.get(p)
                if got is not None:
                    diff = got - want
                    per = periodic.get(p)
                    if per:
                        diff = (diff + 0.5 * per) % per - 0.5 * per
                    if abs(diff) <= max(
                        USER_START_ATOL, USER_START_RTOL * abs(want)
                    ):
                        continue
            self._record_diagnostic(severity, message, paths_, kind=kind)

        # The preliminary scales the rest of the build reads (resolve()'s
        # lowest-priority init_scale, the whitening probe's seed, and the
        # soft barriers' preliminary steepness on derived bounds).  This
        # decides nothing in the solve -- the split and the change test use
        # engine.scale -- but the polish and the probe start from it, so it
        # follows the Jacobian propagation the old engine exported:
        #
        # * a SAMPLED or pinned element exports its own scale (declared,
        #   else propagated), except where the solve DERIVED it from inputs
        #   whose DECLARED scales are better sourced (a user sigma on
        #   star.mass, carried to the sampled logmass it determines);
        # * a DERIVED element exports the scale propagated to it, at the
        #   solved values, from its inputs' EXPORTED scales -- its own
        #   defaults.yaml init_scale describes it only where it is sampled
        #   (orbit.vcve's 0.1, against the 0.0034 its secosw/sesinw give),
        #   and a scale propagated from the leaves at their ANCHORS (the
        #   leaf pass) describes the defaults, not this start.
        #
        # Measured on kelt4_rvonly (tests/test_polish.py, the ulp test): the
        # one-sided, anchor-based and declared-on-derived export changed
        # the planet mass's scale by 15% and several barrier scales 4x, and
        # one ulp-perturbed L-BFGS polish then reached the cos i ~ 0.98
        # basin on a CI BLAS.  Only inputs with DECLARED scales count toward
        # "better sourced": a propagated input scale carries no rank, and
        # counting it let the RV K hint (absorbed by the planet mass)
        # re-export the mass's scale from K's.
        from .relaxation import _secant

        plan = engine.plan
        scales = {}
        srank = {}
        for p in resolved:
            s_p = engine.scale(p)
            if s_p is not None:
                scales[p] = s_p
                srank[p] = scale_rank.get(p, 0)

        def exported(st):
            """RSS of st's response to each input's EXPORTED scale."""
            base = plan.values.get(st.target)
            if base is None:
                return None
            var = 0.0
            for inp in st.inputs:
                s_i = scales.get(inp)
                if s_i is None:
                    s_i = engine.scale(inp, plan)
                if s_i is None or not s_i > 0:
                    return None
                d = _secant(
                    st.fn,
                    plan.values,
                    inp,
                    s_i,
                    base,
                    engine.periodic.get(st.target),
                )
                if not math.isfinite(d):
                    return None
                var += d * d
            return math.sqrt(var) if var > 0.0 else None

        for st in plan.steps:
            if not st.inputs:
                continue
            derived = (
                roles is not None
                and roles.get(st.target, "derived") == "derived"
            )
            if not derived:
                ins = [i for i in st.inputs if i in scale_rank]
                if not ins or len(ins) != len(st.inputs):
                    continue
                mean_rank = sum(scale_rank[i] for i in ins) / len(ins)
                if mean_rank <= srank.get(st.target, 0):
                    continue
            prop = exported(st)
            if prop is not None:
                scales[st.target] = prop
                if not derived:
                    srank[st.target] = mean_rank
        self.propagated_scales = scales
        self._last_scale_provenance = srank
        self._last_provenance = provenance
        self._last_resolved = dict(resolved)
        self._last_solved_by = solved_by
        self._last_keep_report = engine.keep_report
        return resolved
