# src/exozippy/config.py
import copy
import importlib
import logging
import signal
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import sympy as sp
import yaml

logger = logging.getLogger(__name__)
import re

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

# Keys that change the posterior.  bound_scale is one of them: it sets
# soft-bound barrier steepness, a real posterior term (unlike init_scale).
# A user entry touching any of these marks the parameter prior-modified.
PHYSICS_KEYS = ("lower", "upper", "mu", "sigma", "bound_scale")

# Every field resolve() reads as a number and scales by the element's unit.
NUMERIC_KEYS = TUNING_KEYS + PHYSICS_KEYS

# Passed through verbatim; per element when the parameter is a vector.
STRING_KEYS = ("unit", "latex", "description")

# Reporting switches; whole-parameter, not per element.
BOOL_KEYS = ("print_to_table", "debug_print")

# The union: the complete set of legal sub-keys.
USER_PARAM_KEYS = NUMERIC_KEYS + STRING_KEYS + BOOL_KEYS


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
def _sympy_time_limit(seconds=2):
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


def validate_sigma_has_center(user_params, links=None, source=None):
    """Fatal-error check: a Gaussian prior must have an explicit center.

    ``sigma > 0`` on a user parameter asks for a Gaussian prior.  When neither
    ``mu`` nor ``initval`` is given, Parameter.build_pymc centers that prior on
    whatever start value the system resolved (``prior_mus = np.where(~isnan(mus),
    mus, inits)``) -- and that start is frequently DERIVED FROM THE DATA: a
    component's RANK_DERIVED_DATA hint, a relaxation-engine solution, or a
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
        Standardized user params.  Entries that are not dicts are skipped.
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
    the lens's per-source vectors are addressed by the SOURCE STAR's name
    (``lens.SourceA.t_0``), for instance.  Used by
    ``standardize_param_names`` to reject typo'd instance names.
    """
    names = set()
    for entries in (system_config or {}).values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.add(entry["name"])
    return names


# Provenance Ranks
RANK_USER = 100  # Explicitly in params.yaml
RANK_DERIVED_USER = 80  # Solved using ONLY Rank 100s
RANK_DERIVED_DATA = 60  # auto-estimated (e.g., K-band mass, RV offsets)
RANK_DERIVED_MIXED = 40  # Solved using a mix of User and Defaults
RANK_DEFAULT = 20  # From defaults.yaml


class ProvenanceState:
    def __init__(self):
        self.values = {}
        self.ranks = {}

    def set(self, path, value, rank):
        if rank > self.ranks.get(path, 0):
            self.values[path] = value
            self.ranks[path] = rank
            return True
        return False


""" 
This class is the reconciliation engine to determine the sole source of truth 
with which to initialize the parameters (values, scales, bounds) of the model. Specifically, it

1) determines the appropriate limits of each parameter, such that the bounds are the strictest among 
  a) the default bounds (typically very conservatively limited by physics, e.g., abs(velocity) < c), 
  b) any component specific bounds (e.g., bounds of the underlying grids), 
  c) and any user supplied bounds.

2) merges conflicting constraints, respecting a hierarchy of trust, such that

user-specified values (from user_params.yaml) > data-derived estimates > global defaults (from defaults.yaml)

It makes use of each component's symbolic_physics.py to derive the sampled parameter initvals from the set of values 
above, iteratively replacing the lowest ranked parameter with the re-derived value, and updating its weight the 
average of its constituent weights. 

It repeats the process for init_scale, automatically differentiating each relation as it solves it
(step 6 of _execute_solve) to propagate uncertainties.  These scales are only PRELIMINARY: they seed the
whitening probe, which measures the real per-parameter scales from the data (see exozippy/whitening.py),
and they set only the PRELIMINARY soft-bound barrier steepness, which whitening.measure_barrier_scales
re-measures numerically from the whitened model.  The separate sympy forward and backward Jacobian scale
PASSES that used to run after the solve loop -- filling derived-parameter scales for good, and
back-propagating a user scale on a derived parameter to its sampled parents -- are deleted; see the note
at the end of _execute_solve.  init_scale is NOT user-facing: entries in a user params file are stripped
with a warning at construction.

3) warns the user if there are conflicting user-specified constraints that cannot be reconciled.

The logic of the config manager is independent of any specific components.
"""


def _meaningful_change(
    new_val,
    old_val,
    new_rank,
    old_rank,
    tolerance,
    provenance,
    target_str,
):
    """Return True iff _execute_solve should apply this update and signal progress.

    Propagates a rank improvement silently (updates provenance but returns False)
    when the value itself hasn't changed.  Returning False when the value is
    unchanged prevents the relaxation loop from running to max_iter on systems
    that have converged but still have two competing derivation paths for the
    same variable.
    """
    if old_val is None:
        return True  # Condition A: variable was previously unknown — always an update
    ref = max(abs(new_val), abs(old_val), 1e-9)
    if abs(new_val - old_val) / ref >= tolerance:
        return True  # value changed meaningfully
    # Value unchanged; propagate rank silently if it improved
    if new_rank > old_rank:
        provenance[target_str] = new_rank
    return False


class ConfigManager:
    def __init__(self, user_params, system_config=None):
        self.custom_solvers = {}
        self.standalone_solvers = set()

        _reject_renamed_arsun(user_params)

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
            self.user_params = user_params
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
        # is a per-seed observable channel that components (e.g. the MMEXOFAST
        # loader) push into; it feeds the relaxation engine at RANK_DERIVED_DATA
        # -- MMEXOFAST is a (very fancy) derivation FROM THE DATA, not a user
        # statement, so it sits in the same tier as any other data-driven hint
        # and every user entry outranks it.
        self.seed_resolved = None
        self.seed_hint_sets = []
        self.scale_hints = {}  # path -> init_scale in internal units
        # path -> init_scale (internal) as the LAST relaxation solve left it:
        # defaults, component hints, user sigmas and the engine's own
        # solved-value scale sync, refreshed at the end of _execute_solve.
        # (It is not "from the Jacobian forward pass" -- the sympy forward and
        # backward scale passes are deleted; see the note there.)
        self.propagated_scales = {}
        self.symbolic_blacklist = set()

        # Structured diagnostics collected by the relaxation engine (e.g.
        # over-constrained contradictions).  Each entry is a dict
        # {severity, message, param_paths}.  Consumed by the solve/validate
        # API (solve_api.py) without parsing log text.  Populated only when
        # the engine detects a contradiction; empty for a clean solve.
        self.diagnostics = []

        # Snapshots of the last relaxation solve (seed 0, which is solved
        # last in finalize_user_params).  Exposed via export_solution() so a
        # caller can report each parameter's solved value, scale, and
        # provenance without rebuilding the PyMC model.
        self._last_provenance = {}  # internal_path -> rank
        self._last_scale_provenance = {}  # internal_path -> rank
        self._last_resolved = {}  # internal_path -> internal value
        self._last_solved_by = {}  # internal_path -> relation string

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
                    raw_maps = module.get_symbol_map(entry_cfg)
                    if not isinstance(raw_maps, list):
                        raw_maps = [raw_maps]

                    for raw_map in raw_maps:
                        instance_map = {}
                        for sym_name, path in raw_map.items():
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
                            rel_inst = rel.subs(subs)
                            # Maps sharing symbols (e.g. per-source maps with a
                            # common lens mass) produce identical instances of
                            # the shared relations; keep one copy.
                            if rel_inst not in self.all_relations:
                                self.all_relations.append(rel_inst)

        # Sorted for the same reason as the symbolic_physics walk above.  No
        # root-level key is currently defined by two components (a test pins
        # that), so today this is future-proofing rather than a live fix --
        # _deep_merge is last-writer-wins, so the first such collision would
        # otherwise resolve by walk order.
        for defaults_file in sorted(components_dir.rglob("defaults.yaml")):
            with open(defaults_file, "r") as f:
                comp_defaults = yaml.safe_load(f) or {}
                self._deep_merge(self.base_defaults, comp_defaults)

        # Add this inside ConfigManager.__init__ after filling all_relations
        for rel in self.all_relations:
            logger.debug(f"Relation: {rel}")

    def register_custom_solver(
        self, target_str, solver_func, standalone=False
    ):
        """
        Register a shortcut solver for 'comp.param' targets.

        By default a custom solver only runs when the relaxation engine
        attacks an equation containing the target (see _execute_solve).
        standalone=True additionally runs it once per iteration on its own:
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

    def _translate_and_scale(self, path, value):
        """Standardize a human-readable path to internal-index form and convert
        its value to internal units.  Returns (translated_path, internal_value).

        The ONE implementation shared by add_hint / add_scale_hint /
        add_seed_hints / seed_start_value, so they cannot drift apart on
        nomenclature or unit handling.  ``add_scale_hint`` used to carry a
        line-for-line copy of this body despite the docstring claiming the
        sharing; a divergence there would have silently sent a component's
        scale hint to a path ``resolve()`` never looks up.
        """
        translated_path = self.canonical_key(path)

        # Scale to Internal Units.  The ORIGINAL path is what carries a user
        # `unit:` override in user_params, so that is what get_conversion_factor
        # is asked about; the translated path only supplies the defaults.yaml
        # lookup pair (component type, parameter name).
        final_parts = translated_path.split(".")
        if len(final_parts) >= 2:
            c_type, p_name = final_parts[0], final_parts[-1]
            factor = self.get_conversion_factor(c_type, p_name, full_path=path)
            internal_value = float(value) * factor
        else:
            internal_value = float(value)

        return translated_path, internal_value

    def add_hint(self, path, value, rank=RANK_DERIVED_DATA):
        """
        API for components to register their data-driven guesses.
        Converts human-readable paths to strict indices and scales to internal units.
        """
        translated_path, internal_value = self._translate_and_scale(
            path, value
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
        never enter the ledger at all.

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

    def add_seed_hints(self, seed_dicts):
        """Register K per-seed observable sets for multi-seed sampling (P4).

        `seed_dicts` is a list of length K; each entry maps a parameter path
        (human-readable or index form) to a value in that parameter's user
        unit.  These feed the relaxation engine as one complete start point per
        seed (see finalize_user_params), at RANK_DERIVED_DATA -- the same tier
        as ``add_hint``'s default, because the MMEXOFAST loader (the primary
        caller) is a derivation from the data, not a user statement.  Every
        user entry therefore outranks a seed.  Paths absent from a given seed
        fall back to the base (defaults/hints/user) solution for that seed.
        """
        processed = []
        for d in seed_dicts:
            pd = {}
            for path, value in d.items():
                tpath, ival = self._translate_and_scale(path, value)
                pd[tpath] = ival
            processed.append(pd)
        self.seed_hint_sets = processed

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

    def add_scale_hint(self, path, scale):
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
        """
        translated_path, internal_scale = self._translate_and_scale(
            path, scale
        )
        self.scale_hints[translated_path] = internal_scale

    def _deep_merge(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

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
            keys = [
                f"{component_type}.{param_name}",
                f"{component_type}.{_eff_idx(i)}.{param_name}",
            ]
            if names and i < len(names):
                keys.append(f"{component_type}.{names[i]}.{param_name}")
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

        # ONE ELEMENT, ONE SPELLING -- the residual half of the check that
        # runs at construction.  That one sees every collision whose name
        # form is a CONFIG INSTANCE name, because standardize_param_names has
        # already folded such a key into the index form by the time we get
        # here.  What only this site can see is the case it cannot: per-
        # element `names` handed in by a component's manifest that are NOT
        # its own config instances' names.  `lens` does exactly that
        # (examples/ob161003), labelling its per-source vectors with the
        # SOURCE STARS' names, so `lens.SourceA.t_0` survives standardization
        # verbatim and coexists with `lens.0.t_0` as two live user keys.
        #
        # Only keys the user WROTE count.  finalize_user_params injects the
        # engine's solved start values back under the index form, and on
        # ob161003 those legitimately sit alongside the user's own name-form
        # entries -- reading self.user_params here would fail every such fit
        # at stage 5.
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
                entry = self.user_params.get(k)
                if isinstance(entry, dict) and "unit" in entry:
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
                    v = val[i] if isinstance(val, (list, np.ndarray)) else val
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

        # propagated_scales and scale_hints are stored in internal units.
        # Divide by get_conversion_factor (user→internal) to recover user units
        # before passing to Parameter, which will re-apply the same factor.
        # This is distinct from unit_scaling (base→user), which only applies to
        # default values read from defaults.yaml.
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
                            self.propagated_scales[k] / internal_factor,
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
                        self.scale_hints[k] / internal_factor,
                    )
                    break

        for i in range(n_elements):
            for k in _element_keys(i):
                if k in self.user_params:
                    ov = self.user_params[k]
                    if ov is None:
                        continue
                    if not isinstance(ov, dict):
                        ov = {"initval": ov}

                    resolved["user_modified"] = True
                    if any(pk in ov for pk in physics_keys):
                        resolved["user_prior_modified"] = True

                    for key in all_numeric:
                        if key in ov:
                            v_ov = ov[key]
                            # List-valued initvals are per-seed start points
                            # (P4 multi-seed sampling); bounds/scales and the
                            # canonical single start derive from seed 0.
                            if isinstance(v_ov, (list, tuple)):
                                v_ov = v_ov[0]
                            apply_value(key, resolved[key], i, v_ov)
                            # apply_value keeps the TIGHTER of the two bounds.
                            # When the winner is a component-computed one, say
                            # so: these are validity limits (the likelihood is
                            # NaN past them), but the user asked for something
                            # else and deserves to hear that it did not apply.
                            if (key, i) in override_bounds and not np.isclose(
                                resolved[key][i], float(v_ov)
                            ):
                                logger.warning(
                                    f"[{component_type}.{_eff_idx(i)}."
                                    f"{param_name}] user {key}={float(v_ov):g}"
                                    f" is outside the component-computed "
                                    f"validity bound "
                                    f"{override_bounds[(key, i)]:g}; using "
                                    f"{resolved[key][i]:g}."
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
        if (
            full_path
            and full_path in self.user_params
            and isinstance(self.user_params[full_path], dict)
        ):
            u_str = self.user_params[full_path].get("unit")
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

        EVERY pass deepcopies the entry.  The returned dict must share no object
        with the caller's, because downstream code writes through these entries
        in place: extract_links deletes the link-expression fields, and
        finalize_user_params' inject-back sets initval/derived.  Aliasing them
        would (a) strip a caller's link strings out of their own dict, so a
        second ConfigManager built from it sees no links and a hard link
        silently degrades to a fixed parameter, and (b) feed the previous
        solve's answer back in as RANK_USER input.  Pass 2 additionally needs
        the copy per broadcast instance, so the last instance's write does not
        clobber all the others (e.g. per-source radii solved from rho).
        """
        if not user_params:
            return {}

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
            #   * `lens.SourceA.t_0` (examples/ob161003) addresses element j
            #     of the lens's per-source vectors by the SOURCE STAR's name;
            #     the lens block itself has one entry, named "Lens".  The
            #     per-parameter `names` list is a manifest option and is not
            #     known until stage 2, long after this runs.
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

        # Pass 2: expand 2-part keys for list components.
        # Indexed entries written by Pass 1 are never overwritten (explicit beats broadcast).
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

        flat_params = {}
        name_to_index = {}

        # Build index mapping directly from the config keys (No translations!)
        for comp_key, entries in self.system_config.items():
            if isinstance(entries, list):
                for i, c in enumerate(entries):
                    if isinstance(c, dict) and "name" in c:
                        name_to_index[(comp_key, c["name"])] = i

        logger.debug("=" * 50 + "\nSYMBOL MAP DEBUGGING\n" + "=" * 50)

        for path, data in self.user_params.items():
            if data is None:
                continue
            val = data.get("initval") if isinstance(data, dict) else data
            # A list-valued initval is a set of per-seed start points (P4
            # multi-seed sampling).  The base flat_params below seeds the
            # relaxation engine with seed 0; _build_seed_overrides re-injects
            # each seed's element as a RANK_USER override in the K-solve loop.
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
                data = self.user_params[path]
                val = data.get("initval") if isinstance(data, dict) else data
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
        # (user initval lists at RANK_USER, component/MMEXOFAST seed hints at
        # RANK_DERIVED_DATA; both fall back to the shared base_flat for any
        # path they do not touch), then run the relaxation engine once per
        # seed inside this single prepare() call so every seed shares one symbol
        # environment and one relation ordering.
        # Bounds/scales are taken from seed 0 only -- seeds move the START, never
        # the bounds -- so self.propagated_scales is restored to seed 0's after
        # the loop.
        K, user_overrides, seed_hint_overrides = self._build_seed_overrides(
            name_to_index
        )

        # Solve seed 0 LAST so the final self.propagated_scales and any
        # init_scale synced back into self.user_params by _execute_solve both
        # reflect seed 0 -- the seed whose bounds/scales the model actually
        # uses.  Only start positions vary between seeds; bounds/scales do not.
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
            existing_key = None
            for try_key in (final_path, path):
                if try_key in self.user_params and isinstance(
                    self.user_params[try_key], dict
                ):
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
            entry = self.user_params.get(target)
            if not isinstance(entry, dict):
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
             RANK_USER, merged into the engine's user_provided_params.
          2. Component seed hints (config_manager.seed_hint_sets), e.g. the
             MMEXOFAST loader -- RANK_DERIVED_DATA, passed to the engine as
             `seed_hints` and layered in with the other data-driven hints.

        Merging them into one RANK_USER dict (as this did until the 2.1.2
        review fix) had two consequences: a seed silently clobbered a user's
        *scalar* initval for the same path -- the scalar lives in base_flat,
        which the merged override dict overwrote -- and a seed that disagreed
        with a genuine user entry made every symbol of the connecting relation
        RANK_USER, tripping the "over-constrained" contradiction clause.

        Returns (K, user_overrides, seed_hint_overrides) where K is the seed
        count and each override list has length K of {internal_path_str:
        internal_value} dicts.  Every initval list must have length K or 1
        (length-1 broadcasts to all seeds).  When no list initvals and no seed
        hints exist, K == 1 and both lists are [{}], exactly reproducing the
        legacy single-solve behavior.
        """
        # 1. User initval lists -> {sym_path: [internal values]}
        user_lists = {}
        for path, data in self.user_params.items():
            if not isinstance(data, dict):
                continue
            iv = data.get("initval")
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
        #    dict order (a user list is RANK_USER and wins).
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
            if isinstance(v, dict) and "init_scale" in v:
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

    def _record_diagnostic(self, severity, message, param_paths):
        """Append a structured diagnostic (deduped) for the solve/validate API.

        severity is one of "error" | "warning" | "info"; param_paths is a list
        of the parameter paths involved.  Duplicate entries (same severity,
        message, and paths) -- e.g. the same contradiction seen once per seed
        -- collapse to a single record.
        """
        entry = {
            "severity": severity,
            "message": message,
            "param_paths": list(param_paths),
        }
        if entry not in self.diagnostics:
            self.diagnostics.append(entry)

    def _provenance_label(self, rank):
        """Map a numeric provenance rank to a coarse source label."""
        if rank is None:
            return "default"
        if rank >= RANK_USER:
            return "user"
        # Microlensing distance hint (rank 30) and data-derived estimates
        # (RANK_DERIVED_DATA = 60) both come from the data channel.
        if rank == RANK_DERIVED_DATA or rank == 30:
            return "data"
        if rank > RANK_DEFAULT:
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
            entry = (self.user_params or {}).get(path)
            if isinstance(entry, dict) and entry.get("initval") is not None:
                return "user"
            if path in (self.links or {}) and (
                "initval" in self.links[path] or "mu" in self.links[path]
            ):
                return "user"
        return "default"

    def export_solution(self, derived_params=None):
        """Export the resolved parameter solution as JSON-friendly dicts.

        `derived_params`, when given, is the set of `(component_prefix,
        param_name)` pairs the built manifests actually derive
        (`System.derived_params()`).  Pass it whenever a System is in scope:
        the fallback -- "this parameter has an expressions: block in its
        defaults.yaml" -- is only an approximation, since a component may
        declare the same parameter free in one topology and derived in
        another (see planet.mass's linear vs log_q coordinate).

        Returns a dict with:
          - "parameters": {user_path: {value, unit, internal_unit, lower,
            upper, init_scale, sigma, mu, fixed, derived, provenance}} where
            provenance is {rank, label, relation}.  All numeric fields are in
            the parameter's user unit (as reported by resolve()).
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
            parts = internal_path.split(".")
            if len(parts) == 3 and (parts[0], parts[1]) in idx_to_name:
                return f"{parts[0]}.{idx_to_name[(parts[0], parts[1])]}.{parts[2]}"
            return internal_path

        parameters = {}
        for internal_path in self.master_symbol_map:
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
            cfg = self.resolve(c_type, p_name, element=el)

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
            if derived_params is None:
                derived = bool(cfg.get("expressions"))
            else:
                derived = (c_type, p_name) in derived_params
            # A parameter is fixed when it has a hardcoded value or sigma == 0.
            fixed = (cfg.get("value") is not None) or (
                sigma is not None and sigma == 0
            )

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
        "user_params",  # _execute_solve syncs solved init_scale back
        "diagnostics",  # _record_diagnostic appends contradictions
        "propagated_scales",  # refreshed at the end of every solve
        "symbolic_blacklist",  # a 2 s sp.solve timeout adds the target
        "_last_provenance",
        "_last_scale_provenance",
        "_last_resolved",
        "_last_solved_by",
    )

    def probe_derivable(self, paths, tolerance=1e-3):
        """Which of `paths` the relaxation engine can pin down from what is
        known now, as opposed to falling back on a bare defaults.yaml value.

        Runs the engine on a snapshot and rolls every mutation back, so this
        is a read-only question: `finalize_user_params` still does the real
        solve later, from the same inputs plus whatever stages 1-2 add.

        The test is on **provenance**, not on presence: the engine's "default
        armor" step seeds every mapped path from defaults.yaml, so almost
        everything comes back resolved.  A rank above RANK_DEFAULT means the
        value traces back to a user entry or a component hint rather than to
        a default -- i.e. someone actually told us, directly or through a
        relation.

        Called at stage 1a (before most hints exist), so a False here means
        "not derivable *yet*"; callers that must decide early -- notably the
        MMEXOFAST trigger -- get the conservative answer.
        """
        flat = {}
        for upath, data in self.user_params.items():
            sym = self.master_symbol_map.get(upath)
            if sym is None:
                continue
            val = data.get("initval") if isinstance(data, dict) else data
            if val is None and isinstance(data, dict):
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
            self.resolve_and_validate_parameters(flat, tolerance)
            ranks = dict(self._last_provenance)
        except Exception as e:
            # A probe must never break a run that would otherwise work; the
            # caller's fallback is simply "not derivable".
            logger.debug(f"Derivability probe failed ({e}); assuming unknown.")
            ranks = {}
        finally:
            logger.setLevel(prev_level)
            for attr, value in saved.items():
                setattr(self, attr, value)

        return {p for p in paths if ranks.get(p, 0) > RANK_DEFAULT}

    def resolve_and_validate_parameters(
        self, user_provided_params, tolerance=1e-3, seed_hints=None
    ):
        """Run the relaxation engine.

        ``user_provided_params`` is {internal_path: internal_value} at
        RANK_USER.  ``seed_hints`` is the optional per-seed hint set for this
        seed (see ``_build_seed_overrides``), layered in at RANK_DERIVED_DATA
        alongside ``self.hints``.
        """
        resolved = {str(k): float(v) for k, v in user_provided_params.items()}
        provenance = {str(k): RANK_USER for k in user_provided_params.keys()}
        resolved_scales = {}
        scale_provenance = {}
        # Per-solve record of which relation last set each variable (seed 0's
        # values win because it is solved last).  Reset each call.
        self._last_solved_by = {}

        # 1. Initialize Default Armor (Rank 20)
        def to_scalar(val):
            return val.item() if hasattr(val, "item") else float(val)

        for path_str, sym in self.master_symbol_map.items():
            parts = path_str.split(".")
            c_type, p_name = parts[0], parts[-1]
            el = (
                int(parts[1])
                if len(parts) == 3 and parts[1].isdigit()
                else None
            )
            cfg = self.resolve(c_type, p_name, element=el)

            # Read rank directly from base_defaults (not from resolve() return dict)
            param_rank = (
                self.base_defaults.get(c_type, {}).get(p_name, {}).get("rank")
                or self.base_defaults.get(p_name, {}).get("rank")
                or RANK_DEFAULT
            )

            if path_str not in resolved and cfg.get("initval") is not None:
                # resolve() returns the initval in USER units, so the factor
                # must honor a user `unit:` override -- pass full_path, the
                # same way the init_scale conversion below does.
                factor = self.get_conversion_factor(
                    c_type, p_name, full_path=path_str
                )
                resolved[path_str] = to_scalar(cfg["initval"]) * factor
                provenance[path_str] = param_rank

            if cfg.get("init_scale") is not None:
                factor = self.get_conversion_factor(
                    c_type, p_name, full_path=path_str
                )
                resolved_scales[path_str] = (
                    to_scalar(cfg["init_scale"]) * factor
                )
                scale_provenance[path_str] = param_rank

        # 1.5 LAYER IN COMPONENT HINTS
        for path_str, val in self.hints.items():
            if provenance.get(path_str, 0) < RANK_USER:
                resolved[path_str] = val
                provenance[path_str] = self.hint_ranks.get(
                    path_str, RANK_DERIVED_DATA
                )

        # 1.5b LAYER IN THIS SEED'S HINT SET (RANK_DERIVED_DATA)
        # Same tier as the component hints above: an MMEXOFAST solution is a
        # derivation from the data, not a user statement.  The guard is `<=`
        # rather than `<` so a seed WINS a tie with an ordinary component
        # hint -- a per-seed fit of the actual light curve is strictly more
        # informative than the generic guess a component pushes for every
        # seed alike.  (Today no real path is in both channels: the seeds
        # carry lens.0.{t_0,u_0,t_E,rho,log_s,alpha,q}, and the only
        # component hint that touches one of those, lens.0.alpha, is rank
        # 20.  The rule is stated so a future overlap has a defined answer.)
        # Anything above RANK_DERIVED_DATA -- every user entry, in particular
        # a scalar initval, which lives in user_provided_params -- wins.
        for path_str, val in (seed_hints or {}).items():
            if provenance.get(path_str, 0) <= RANK_DERIVED_DATA:
                resolved[path_str] = val
                provenance[path_str] = RANK_DERIVED_DATA

        # 1.6 LAYER IN SCALE HINTS (correct indexed paths)
        # The initialization loop above calls resolve() without an index, so a hint
        # for star.0.logmass can bleed into star.1.logmass.  Apply scale_hints
        # directly using their already-normalized full paths to fix that.
        for hint_path, hint_scale in self.scale_hints.items():
            if hint_path in self.master_symbol_map:
                resolved_scales[hint_path] = hint_scale
                scale_provenance[hint_path] = RANK_DERIVED_DATA

        # 1.7 LAYER IN USER-SPECIFIED SIGMAS AS SCALES (RANK_USER)
        # A user-supplied Gaussian prior width is also the best available
        # preliminary scale for that parameter, and the engine's per-relation
        # Jacobian propagation (step 6 of _execute_solve) carries it into the
        # scale of whatever gets solved from it.  The separate forward
        # Jacobian PASS that used to run after the loop, filling every derived
        # parameter's scale so it could set that parameter's soft-bound
        # barrier steepness, is deleted -- those steepnesses are measured
        # numerically now (whitening.measure_barrier_scales).  User
        # init_scale entries no longer exist at this point -- they are
        # stripped with a warning at construction (whitening scales are
        # measured from the data instead; see exozippy/whitening.py).
        for path_str in self.master_symbol_map:
            up = self.user_params.get(path_str)
            if not isinstance(up, dict):
                continue
            parts = path_str.split(".")
            c_type, p_name = parts[0], parts[-1]
            if (
                "sigma" in up
                and up.get("sigma") is not None
                and float(up["sigma"]) > 0
            ):
                factor = self.get_conversion_factor(
                    c_type, p_name, full_path=path_str
                )
                resolved_scales[path_str] = float(up["sigma"]) * factor
                scale_provenance[path_str] = RANK_USER

        # 1.8 PREPARE USER-DEFINED LINK ASSIGNMENTS (directed, RANK_USER)
        # initval/mu links are directed: the target is defined in terms of its
        # dependencies, never the reverse.  They are re-asserted every
        # iteration so downstream physics relations always see the linked
        # value, and RANK_USER provenance protects it from being overridden.
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

        # 2. The Relaxation Engine
        logger.info(
            "Solving for starting values/scales of sampled parameters given user/data/default initialization...."
        )
        iteration = 0
        max_iter = 100  # Failsafe
        _CYCLE_HIST = 6  # how many recent values to keep per variable
        value_history = {}  # {var_name: [recent rounded values]}
        pinned_vars = (
            set()
        )  # variables locked out of further updates due to cycle

        while iteration < max_iter:
            iteration += 1
            resolved_snapshot = dict(resolved)

            self._apply_directed_links(
                directed_links,
                resolved,
                provenance,
                resolved_scales,
                scale_provenance,
                tolerance,
                pinned_vars,
            )

            self._run_standalone_solvers(
                resolved, provenance, tolerance, pinned_vars
            )

            for eq in self.all_relations:
                self._relax_equation(
                    eq,
                    resolved,
                    provenance,
                    resolved_scales,
                    scale_provenance,
                    tolerance,
                    pinned_vars,
                )

            # Convergence check: compare end-of-iteration state to start-of-iteration.
            # This correctly handles intra-iteration oscillation (two equations fighting
            # over the same variable within one pass): individual updates may fire on
            # each equation, but if the net state is unchanged the loop should stop.
            net_changed = False
            for k, v in resolved.items():
                old = resolved_snapshot.get(k)
                if old is None:
                    net_changed = True
                    break
                ref = max(abs(v), abs(old), 1e-9)
                if abs(v - old) / ref >= tolerance:
                    net_changed = True
                    break
            if not net_changed:
                break

            # Cycle detection: track per-variable history; pin any that oscillate.
            # Pinning lets other variables keep converging instead of stopping the loop.
            for k, v in resolved.items():
                if k in pinned_vars:
                    continue
                if k not in resolved_snapshot or resolved_snapshot[k] != v:
                    hist = value_history.setdefault(k, [])
                    hist.append(round(v, 8))
                    if len(hist) > _CYCLE_HIST:
                        hist.pop(0)

            for k, hist in value_history.items():
                if k in pinned_vars:
                    continue
                if (
                    len(hist) >= 4
                    and hist[-1] == hist[-3]
                    and hist[-2] == hist[-4]
                    and hist[-1] != hist[-2]
                ):
                    logger.warning(
                        f"Cycle: '{k}' oscillates between {hist[-2]:.6g} and {hist[-1]:.6g} — "
                        f"pinned to {hist[-1]:.6g} (conflicting equal-rank constraints)."
                    )
                    pinned_vars.add(k)

        if iteration == max_iter:
            logger.warning(
                "Relaxation engine reached max iterations — check for unstable circular dependencies."
            )

        logger.info(f"Done solving after {iteration} iterations.")

        # (The old sympy scale passes are gone.  Backward -- inverse-Jacobian
        # propagation of a user scale on a derived parameter to its sampled
        # parents -- because sampled scales are only PRELIMINARY now: the
        # whitening probe measures the real ones from the data, prior sigmas
        # included.  Forward -- filling derived-parameter scales for their
        # soft-bound barrier steepness -- because those are now measured
        # numerically from the whitened model's actual unit-step response
        # (whitening.measure_barrier_scales), which is both cheaper and more
        # reliable than per-relation sp.solve with timeouts.  What remains in
        # resolved_scales: defaults, component hints, user sigmas, and the
        # engine's own solved-value scale sync.)

        # Expose final scales for resolve() to use as a low-priority default
        # (component scale hints still win; see resolve()).
        self.propagated_scales = dict(resolved_scales)

        # Snapshot provenance/scales/values for export_solution().  In the
        # multi-seed loop seed 0 is solved last, so these end up reflecting the
        # canonical seed-0 solution whose bounds/scales the model uses.
        self._last_provenance = dict(provenance)
        self._last_scale_provenance = dict(scale_provenance)
        self._last_resolved = dict(resolved)

        return resolved

    def _run_standalone_solvers(
        self, resolved, provenance, tolerance, pinned_vars=None
    ):
        """
        Run standalone-registered custom solvers once per relaxation
        iteration, for every instance path of their target in the symbol
        map.  A solver that raises (missing dependencies) is retried next
        iteration; results carry RANK_DERIVED_MIXED so user values and
        data-derived hints always win, and re-fire as inputs refine.

        "Always win" has to be enforced here, not just asserted: unlike the
        equation path (``_attempt_solve``), which picks the LOWEST-ranked
        symbol of a violated relation, a standalone solver writes its target
        unconditionally.  ``_meaningful_change`` compares values, not ranks,
        so before the guard below an explicit ``orbit.b.m_total`` in
        params.yaml (RANK_USER) was overwritten every iteration by the body
        mass sum and its provenance DOWNGRADED to RANK_DERIVED_MIXED --
        exactly the inversion the ranking system exists to prevent.  A path
        already held at a rank the solver cannot beat is therefore skipped;
        one held at RANK_DERIVED_MIXED or below (including the solver's own
        previous answer) still re-fires as its inputs refine.
        """
        # Sorted: standalone_solvers is a set, and each solver both reads and
        # writes `resolved`, so with two of them registered the walk order
        # decides whether one sees the other's answer from this iteration or
        # the last -- feeding _meaningful_change, the provenance ranks and
        # the cycle history.  Only orbit.m_total is registered today.
        for lookup_key in sorted(self.standalone_solvers):
            solver_func = self.custom_solvers[lookup_key]
            comp, param = lookup_key.split(".")[0], lookup_key.split(".")[-1]
            for path in list(self.master_symbol_map):
                parts = path.split(".")
                if (
                    len(parts) != 3
                    or parts[0] != comp
                    or parts[2] != param
                    or not parts[1].isdigit()
                ):
                    continue
                if pinned_vars and path in pinned_vars:
                    continue
                if provenance.get(path, 0) > RANK_DERIVED_MIXED:
                    logger.debug(
                        f"Standalone solver for {path} skipped: already held "
                        f"at rank {provenance.get(path)} > "
                        f"{RANK_DERIVED_MIXED}."
                    )
                    continue
                try:
                    val = float(
                        solver_func(
                            resolved, self.system_config, int(parts[1])
                        )
                    )
                except Exception as e:
                    logger.debug(f"Standalone solver for {path} deferred: {e}")
                    continue
                if not _meaningful_change(
                    val,
                    resolved.get(path),
                    RANK_DERIVED_MIXED,
                    provenance.get(path, 0),
                    tolerance,
                    provenance,
                    path,
                ):
                    continue
                resolved[path] = val
                provenance[path] = RANK_DERIVED_MIXED
                self._last_solved_by[path] = (
                    f"{lookup_key} (standalone solver)"
                )
                logger.debug(f"Updated {path} = {val:.4g} (standalone solver)")

    def _apply_directed_links(
        self,
        directed_links,
        resolved,
        provenance,
        resolved_scales,
        scale_provenance,
        tolerance,
        pinned_vars=None,
    ):
        """
        Assert user-defined link assignments (target := f(deps), internal units).

        An 'initval' link IS the user's value for the target, so it always
        wins (RANK_USER).  A 'mu' link only seeds the starting point, so it
        yields to an explicit numeric initval (which also carries RANK_USER).
        Scales propagate through the link Jacobian at RANK_DERIVED_USER.
        """
        for target, fld, plink, expr_int in directed_links:
            if pinned_vars and target in pinned_vars:
                continue
            if not all(d in resolved for d in plink.dep_paths):
                continue
            if fld == "mu" and provenance.get(target, 0) >= RANK_USER:
                continue
            try:
                val = float(expr_int.evalf(subs=resolved))
            except (TypeError, ValueError) as e:
                logger.debug(
                    f"Link '{target} := {plink.expr_str}' not evaluable yet: {e}"
                )
                continue

            if _meaningful_change(
                val,
                resolved.get(target),
                RANK_USER,
                provenance.get(target, 0),
                tolerance,
                provenance,
                target,
            ):
                resolved[target] = val
                provenance[target] = RANK_USER
                logger.debug(
                    f"Updated {target} = {val:.4g} (user link: {plink.expr_str})"
                )

            # Scale propagation through the link Jacobian
            if scale_provenance.get(target, 0) >= RANK_USER:
                continue
            var = 0.0
            any_input = False
            for dep in plink.dep_paths:
                dep_scale = resolved_scales.get(dep)
                if dep_scale is None:
                    continue
                try:
                    d = float(
                        sp.diff(expr_int, sp.Symbol(dep)).evalf(subs=resolved)
                    )
                except (TypeError, ValueError):
                    continue
                if np.isfinite(d) and abs(d) < 1e15:
                    var += (d * dep_scale) ** 2
                    any_input = True
            if (
                any_input
                and var > 0
                and RANK_DERIVED_USER > scale_provenance.get(target, 0)
            ):
                resolved_scales[target] = float(np.sqrt(var))
                scale_provenance[target] = RANK_DERIVED_USER

    def _relax_equation(
        self,
        eq,
        resolved,
        provenance,
        resolved_scales,
        scale_provenance,
        tolerance,
        pinned_vars=None,
    ):

        if not isinstance(eq, sp.Eq):
            return False

        # Sorted: free_symbols is a set of Symbols whose hashes include the
        # PYTHONHASHSEED-randomized name string, so bare iteration order
        # varies per process.  The July-2026 init_scale flakiness came from
        # exactly this in the (since-deleted) scale passes.
        symbols_in_eq = sorted(str(s) for s in eq.free_symbols)

        # 1. Gatekeeper: Skip if equation contains undefined variables
        if not all(s in self.master_symbol_map for s in symbols_in_eq):
            return False

        def get_rank(s):
            return provenance.get(s, 0)

        unknowns = [s for s in symbols_in_eq if s not in resolved]
        target = None
        is_contradiction = False

        # 2. Trigger Condition A: Missing Information
        if len(unknowns) == 1:
            target = unknowns[0]

        # 3. Trigger Condition B: Physics Verification
        elif len(unknowns) == 0:
            try:
                lhs = float(eq.lhs.evalf(subs=resolved))
                rhs = float(eq.rhs.evalf(subs=resolved))
                error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-9)
            except TypeError:
                return False

            if error <= tolerance:
                return False  # Equation is perfectly satisfied. Stop here.

            # Equation is broken. Find the weakest armor.
            # Sort by Rank (Ascending), then Alphabetically (Deterministic Tie-Breaker)
            candidates = sorted(symbols_in_eq, key=lambda s: (get_rank(s), s))
            target = candidates[0]

            # 4. The Contradiction Clause
            if get_rank(target) >= RANK_USER:
                is_contradiction = True

        if not target or len(unknowns) > 1:
            return False

        # Skip pinned variables so other variables can keep converging
        if pinned_vars and target in pinned_vars:
            return False

        # 5. Calculate New Armor
        # Use min(input_ranks) so a chain is only as strong as its weakest link.
        inputs = [s for s in symbols_in_eq if s != target]
        min_input_rank = (
            min(get_rank(s) for s in inputs) if inputs else RANK_DEFAULT
        )
        # Condition A floor: RANK_DEFAULT-1 so an indirectly-derived value
        #   (e.g. ecc from K when secosw/sesinw are unavailable) always yields
        #   to an expression-path derivation or a default in Condition B.
        # Condition B floor: 0, NOT RANK_DEFAULT -- a duplicate of this comment
        #   sat directly above and claimed RANK_DEFAULT, which the line below
        #   has never done.  0 lets low-rank inputs (e.g. pm defaults at rank
        #   10) produce low-rank results, so they cannot block higher-rank
        #   derivations from the t_E/theta_E/pi_rel chain.
        rank_floor = RANK_DEFAULT - 1 if len(unknowns) == 1 else 0
        new_rank = max(rank_floor, min(RANK_DERIVED_USER, min_input_rank))

        if is_contradiction:
            # All variables in this equation were explicitly set by the user.
            # Overriding any of them would silently discard user intent — this
            # commonly happens with "default identity" relations like
            # Eq(radiussed, radius) when the user intentionally gave the two
            # parameters different MAP values on a second-iteration run.
            # Leave every value untouched and let the sampler and likelihood
            # sort out any inconsistency.
            logger.debug(
                f"Over-constrained: all variables in '{eq}' have user rank "
                f"but equation is violated (error={error:.4g}). "
                f"Leaving all user values unchanged."
            )
            self._record_diagnostic(
                "error",
                f"Over-constrained relation '{eq.lhs} = {eq.rhs}' is violated "
                f"(relative error {error:.4g}): every parameter it links was "
                f"set explicitly, so no value can be adjusted to satisfy it.",
                symbols_in_eq,
            )
            return False

        return self._execute_solve(
            eq,
            target,
            resolved,
            provenance,
            new_rank,
            resolved_scales,
            scale_provenance,
            inputs,
            pinned_vars=pinned_vars,
            tolerance=tolerance,
        )

    def _execute_solve(
        self,
        eq,
        target_str,
        resolved,
        provenance,
        new_rank,
        resolved_scales,
        scale_provenance,
        inputs,
        pinned_vars=None,
        tolerance=1e-3,
    ):
        if pinned_vars and target_str in pinned_vars:
            return False

        # 1. ALWAYS check custom solvers FIRST
        parts = target_str.split(".")
        lookup_key = f"{parts[0]}.{parts[-1]}"
        idx = int(parts[1]) if len(parts) >= 3 else 0

        if lookup_key in self.custom_solvers:
            solver_func = self.custom_solvers[lookup_key]
            try:
                valid_val = float(
                    solver_func(resolved, self.system_config, idx)
                )
                if not _meaningful_change(
                    valid_val,
                    resolved.get(target_str),
                    new_rank,
                    provenance.get(target_str, 0),
                    tolerance,
                    provenance,
                    target_str,
                ):
                    return False
                resolved[target_str] = valid_val
                provenance[target_str] = new_rank
                self._last_solved_by[target_str] = f"{eq.lhs} = {eq.rhs}"
                logger.debug(
                    f"Updated {target_str} = {valid_val:.4g} (custom solver)"
                )
                return True
            except Exception as e:
                # A custom solver is a shortcut for one specific relation
                # (e.g. K -> companion mass).  If it can't run (missing
                # dependencies), fall through to the generic symbolic solver
                # so OTHER equations targeting this parameter (e.g.
                # q * M_primary = M_companion) still get their chance.
                logger.debug(
                    f"Custom solver failed for {target_str}: {e}; "
                    f"falling back to symbolic solve."
                )

        # 2. Skip equations whose symbolic inversion has previously timed out
        if target_str in self.symbolic_blacklist:
            return False

        # 3. Diagnostic Timing for the Symbolic Solver
        logger.debug(f"Attempting to solve: {eq} for target: {target_str}")

        # Print the equation with substituted numerical values ---
        try:
            # Format as "lhs = rhs" instead of "Eq(lhs, rhs)"
            eq_str = f"{eq.lhs} = {eq.rhs}"

            # Replace only the known symbols so the math structure is
            # preserved.  Sorted for the same reason as _execute_solve's walk
            # (free_symbols is a set of Symbols whose hashes include the
            # string hash): successive re.sub calls are not commutative when
            # one symbol's name is a substring of another's, so an unsorted
            # walk could render this line differently in two processes.  It
            # is only a debug string today -- but a diagnostic that varies
            # run to run is the one thing a diagnostic must not do.
            for s in sorted(eq.free_symbols, key=str):
                s_str = str(s)
                if s_str in resolved:
                    # Format to 5 sig figs (handles scientific notation automatically)
                    val_str = f"{float(resolved[s_str]):.5g}"
                    # Use regex with word boundaries to replace exact variable names
                    eq_str = re.sub(
                        rf"\b{re.escape(s_str)}\b", val_str, eq_str
                    )

            logger.debug(f"  Substituted: {eq_str}")
        except Exception as e:
            pass

        start_time = time.time()

        def handler(signum, frame):
            raise TimeoutError("Symbolic solver timed out!")

        _arm_alarm(2, handler)  # 2-second hard limit (POSIX only)

        solutions = []
        used_nsolve = False

        try:
            target_sym = next(
                s for s in eq.free_symbols if str(s) == target_str
            )
            # simplify=False + check=False: sp.solve's default post-simplify and
            # its checksol() verification pass together dominate prepare()
            # (sympy.simplify + sympy.checksol) and are redundant here -- every
            # candidate solution is evaluated numerically, bounds-checked, and
            # (for multiple roots) scored against the other relations below, so
            # the code already does its own root validation. simplify only
            # changes the expression's form (identical numeric value); check
            # only pre-filters roots the numeric bounds/scoring pass re-filters.
            solutions = sp.solve(
                eq, target_sym, dict=False, simplify=False, check=False
            )
            elapsed = time.time() - start_time
            logger.debug(
                f"sp.solve finished in {elapsed:.4f}s for {target_str}"
            )
        except TimeoutError:
            logger.debug(
                f"sp.solve timed out for {target_str} — blacklisting."
            )
            self.symbolic_blacklist.add(target_str)
            _disarm_alarm()
            return False
        except Exception as e:
            logger.debug(f"sp.solve exception for {target_str}: {e}")
            _disarm_alarm()
            return False
        finally:
            _disarm_alarm()

        # 3. Fallback to nsolve if analytical failed
        if not solutions:
            try:
                with _sympy_time_limit(2):
                    guess = float(resolved.get(target_str, 1.0))
                    sub_dict = {s: resolved[str(s)] for s in inputs}
                    expr = (eq.lhs - eq.rhs).subs(sub_dict).evalf()
                    solutions = [sp.nsolve(expr, target_sym, guess)]
                    used_nsolve = True
            except Exception:
                return False

        # 4. Validation — collect all in-bounds solutions
        cfg = self.resolve(parts[0], parts[-1], shape=(), element=idx)
        lower = cfg["lower"][0] if cfg.get("lower") is not None else -np.inf
        upper = cfg["upper"][0] if cfg.get("upper") is not None else np.inf

        valid_solutions = []
        for sol in solutions:
            try:
                val = (
                    float(sol.evalf(subs=resolved))
                    if not used_nsolve
                    else float(sol)
                )
                if lower <= val <= upper:
                    valid_solutions.append((val, sol))
            except (TypeError, ValueError):
                continue

        if not valid_solutions:
            return False

        # When multiple roots exist (e.g. ± from a quadratic), pick the one that
        # best satisfies other equations sharing this variable.  This prevents
        # sign-ambiguity failures (e.g. mu_ra_rel getting the wrong sign and
        # making pi_E_E irreconcilable).
        if len(valid_solutions) == 1:
            valid_val, valid_sol = valid_solutions[0]
        else:
            # Sorted by value: sp.solve's root order is not contractual, and
            # the strict < below means equal-score roots go to whichever came
            # first -- break that tie by value, not arrival order.
            best_val, best_sol, best_score = None, None, float("inf")
            for val, sol in sorted(valid_solutions, key=lambda vs: vs[0]):
                temp = {**resolved, target_str: val}
                score = 0.0
                for other_eq in self.all_relations:
                    other_syms = [str(s) for s in other_eq.free_symbols]
                    if target_str not in other_syms or other_eq is eq:
                        continue
                    if not all(s in temp for s in other_syms):
                        continue
                    try:
                        lhs = float(other_eq.lhs.evalf(subs=temp))
                        rhs = float(other_eq.rhs.evalf(subs=temp))
                        ref = max(abs(lhs), abs(rhs), 1e-9)
                        score += ((lhs - rhs) / ref) ** 2
                    except Exception:
                        pass
                if score < best_score:
                    best_score = score
                    best_val, best_sol = val, sol
            valid_val, valid_sol = best_val, best_sol

        # 5. Apply Value and Armor
        if valid_val is not None:
            if not _meaningful_change(
                valid_val,
                resolved.get(target_str),
                new_rank,
                provenance.get(target_str, 0),
                tolerance,
                provenance,
                target_str,
            ):
                return False
            # 5b. Jacobian-filtered rank refinement.
            # Replace raw min(input_ranks) with min(active_input_ranks), where
            # "active" means the input has a non-negligible Jacobian contribution.
            # This prevents incidentally-zero inputs (e.g. mu_ra_rel ≈ 0 in the
            # mu_dec_rel = sqrt(mu_rel_mag² - mu_ra_rel²) equation) from dragging
            # down the trustworthiness of the result.
            jac_active_inputs = []
            if (
                not used_nsolve
                and hasattr(valid_sol, "free_symbols")
                and inputs
            ):
                for parent_str in inputs:
                    parent_sym = sp.Symbol(parent_str)
                    if not valid_sol.has(parent_sym):
                        continue
                    try:
                        d = float(
                            sp.diff(valid_sol, parent_sym).evalf(subs=resolved)
                        )
                        if np.isfinite(d) and abs(d) > 1e-6:
                            jac_active_inputs.append(parent_str)
                    except Exception:
                        jac_active_inputs.append(
                            parent_str
                        )  # conservative: include on error

            if jac_active_inputs:
                min_jac_rank = min(
                    provenance.get(s, RANK_DEFAULT) for s in jac_active_inputs
                )
                # Refine upward only: never reduce a rank that was already determined
                # by a valid floor (e.g. Condition A floor of RANK_DEFAULT-1).
                new_rank = max(new_rank, min(RANK_DERIVED_USER, min_jac_rank))

            resolved[target_str] = valid_val
            provenance[target_str] = new_rank
            self._last_solved_by[target_str] = f"{eq.lhs} = {eq.rhs}"
            logger.debug(
                f"Updated {target_str} = {valid_val:.4g} (rank: {new_rank})"
            )

            # 6. Independent Scale Propagation via Jacobian
            if (
                not used_nsolve
                and hasattr(valid_sol, "free_symbols")
                and inputs
            ):
                scale_variance = 0.0
                valid_scale_inputs = []

                for parent_str in inputs:
                    parent_sym = sp.Symbol(parent_str)
                    if not valid_sol.has(parent_sym):
                        continue

                    parent_scale = resolved_scales.get(parent_str, 1e-9)
                    valid_scale_inputs.append(parent_str)

                    try:
                        derivative = sp.diff(valid_sol, parent_sym)
                        sensitivity = float(derivative.evalf(subs=resolved))

                        # Only update variance if the sensitivity is a sane number
                        if (
                            np.isfinite(sensitivity)
                            and abs(sensitivity) < 1e15
                        ):
                            scale_variance += (sensitivity * parent_scale) ** 2

                    except (OverflowError, FloatingPointError, TypeError):
                        # If the derivative is explosive, we have reached a physical regime
                        # where variance propagation is numerically invalid.
                        # Treat the variance contribution as undefined/maxed out.
                        scale_variance = np.inf
                        break

                if valid_scale_inputs:
                    new_scale_rank = sum(
                        scale_provenance.get(s, 0) for s in valid_scale_inputs
                    ) / len(valid_scale_inputs)
                    new_scale = float(np.sqrt(scale_variance))
                    if new_scale_rank > scale_provenance.get(target_str, 0):
                        resolved_scales[target_str] = new_scale
                        scale_provenance[target_str] = new_scale_rank

                    # Sync the solved scale back into user_params -- but only
                    # when there IS one.  init_scale is optional at every
                    # source (defaults.yaml, component hints, user sigmas), so
                    # a target whose parents are ALL scale-less scores
                    # new_scale_rank == 0, fails the guard above, and has no
                    # entry from the default-armor pass either: reading
                    # resolved_scales[target_str] here used to raise KeyError
                    # straight out of prepare().  (Reproduced by naming
                    # `orbit.<name>.a` in a params file -- it is solved
                    # from m_total and period, and none of the three carries
                    # an init_scale default.)  Skipping leaves the parameter
                    # with no preliminary scale, which is the documented and
                    # handled state: build_pymc falls back to a fraction of
                    # the bound span, and the startup whitening probe measures
                    # the real scale from the data regardless.
                    if target_str in resolved_scales and isinstance(
                        self.user_params.get(target_str), dict
                    ):
                        factor = self.get_conversion_factor(
                            parts[0], parts[-1], full_path=target_str
                        )
                        self.user_params[target_str]["init_scale"] = (
                            resolved_scales[target_str] / factor
                        )

            return True

        return False

    def _attempt_rank_upgrade(
        self, eq, resolved, provenance, resolved_scales, scale_provenance
    ):
        # Sorted for the same reason as _execute_solve's walk: free_symbols is a
        # set of Symbols whose hashes include the PYTHONHASHSEED-randomized name
        # string, so bare iteration order varies per process.
        symbols_in_eq = sorted(str(s) for s in eq.free_symbols)

        def get_rank(s):
            return provenance.get(s, RANK_DEFAULT)

        # 1. We need ALL symbols to be known except at most one
        # If any symbol is NOT in master_symbol_map, we can't solve this equation.
        if not all(s in self.master_symbol_map for s in symbols_in_eq):
            return False

        # 2. Identify the target:
        # Pick the symbol with the lowest provenance, breaking rank ties
        # alphabetically -- a bare min() returns the first minimum it meets, so
        # on tied ranks the winner would follow iteration order.
        target = min(symbols_in_eq, key=lambda s: (get_rank(s), s))

        # 3. Check dependencies: Are all inputs known?
        inputs = [s for s in symbols_in_eq if s != target]
        if not all(s in resolved for s in inputs):
            return False

        # 4. Calculate bottleneck rank
        input_ranks = [get_rank(s) for s in inputs]
        new_rank = min(input_ranks) if input_ranks else RANK_DEFAULT

        # 5. Monotonic upgrade check
        if new_rank > get_rank(target):
            logger.debug(
                f"Rank upgrade: {target} ({get_rank(target)} -> {new_rank}) via {eq}"
            )
            return self._solve_and_update(
                eq,
                target,
                resolved,
                provenance,
                new_rank,
                resolved_scales,
                scale_provenance,
            )

        return False

    def _solve_and_update(
        self,
        eq,
        target_str,
        resolved,
        provenance_map,
        new_rank,
        resolved_scales,
        scale_provenance,
    ):
        target_sym = next(s for s in eq.free_symbols if str(s) == target_str)

        # 1. Setup bounds and solver lookups
        parts = target_str.split(".")
        el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
        cfg = self.resolve(parts[0], parts[-1], shape=(), element=el)
        lower = cfg["lower"][0] if cfg.get("lower") is not None else -np.inf
        upper = cfg["upper"][0] if cfg.get("upper") is not None else np.inf

        # 2. Logic to isolate target and solve
        solutions = []
        valid_val = None
        valid_sol = None

        # Trivial Isolation
        if eq.lhs == target_sym:
            solutions = [eq.rhs]
        elif eq.rhs == target_sym:
            solutions = [eq.lhs]
        else:
            try:
                solutions = sp.solve(eq, target_sym, dict=False)
            except Exception:
                pass

        # Fallback to nsolve
        if not solutions:
            try:
                guess = float(resolved.get(target_str, 1.0))
                # Sorted: sympy applies an unordered subs mapping in its own
                # canonical order, but the dict's own insertion order comes
                # straight off a hash-randomized set -- do not rely on a
                # third party to launder that for us.
                sub_dict = {
                    s: resolved[str(s)]
                    for s in sorted(eq.free_symbols, key=str)
                    if str(s) != target_str
                }
                expr = (eq.lhs - eq.rhs).subs(sub_dict).evalf()
                solutions = [sp.nsolve(expr, target_sym, guess)]
            except Exception:
                return False

        # 3. Validate Solution
        for sol in solutions:
            try:
                val = float(sol.evalf(subs=resolved))
                if lower <= val <= upper:
                    valid_val = val
                    valid_sol = sol
                    break
            except (TypeError, ValueError):
                continue

        # 4. Final Update Guard (The "Provenance Engine")
        if valid_val is not None:
            # Update Value
            resolved[target_str] = valid_val
            provenance_map[target_str] = new_rank

            # Propagate Scale (Calculus-based)
            if hasattr(valid_sol, "free_symbols"):
                scale_variance = 0.0
                # Sorted for the same reason as _execute_solve's walk (whose
                # `inputs` list is already derived from a sorted
                # symbols_in_eq): free_symbols is a set of Symbols whose
                # hashes include the PYTHONHASHSEED-randomized name string.
                # Here the order is doubly load-bearing -- it is the
                # SUMMATION order of the float accumulation below, so an
                # unsorted walk makes init_scale differ in its last bits
                # between two processes running identical code.
                for parent_sym in sorted(valid_sol.free_symbols, key=str):
                    parent_str = str(parent_sym)
                    # Fallback to a small epsilon if parent scale is missing
                    parent_scale = resolved_scales.get(parent_str, 1e-9)

                    derivative = sp.diff(valid_sol, parent_sym)
                    sensitivity = float(derivative.evalf(subs=resolved))
                    scale_variance += (sensitivity * parent_scale) ** 2

                # Apply scale update only if rank allows
                if new_rank >= scale_provenance.get(target_str, 0):
                    resolved_scales[target_str] = float(
                        np.sqrt(scale_variance)
                    )
                    scale_provenance[target_str] = new_rank

                    # Sync scale back to user_params if necessary
                    if target_str in self.user_params and isinstance(
                        self.user_params[target_str], dict
                    ):
                        factor = self.get_conversion_factor(
                            parts[0], parts[-1], full_path=target_str
                        )
                        self.user_params[target_str]["init_scale"] = (
                            resolved_scales[target_str] / factor
                        )

            return True

        return False

    def print_contradiction_warning(self, eq, error):
        logger.warning(
            "!" * 60 + "\n"
            "WARNING: PHYSICAL CONTRADICTION DETECTED\n"
            f"Relation: {eq}\n"
            f"Relative Error: {error:.2%}\n" + "-" * 60 + "\n"
            "The parameters provided in your config do not satisfy this equation.\n"
            "Verify your starting values; a bad initialization will destroy NUTS efficiency.\n"
            + "!"
            * 60
        )
