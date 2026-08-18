# tests/test_known_keys.py
"""Anti-drift cover for the hand-maintained "known key" vocabularies.

EXOZIPPy warns about unrecognized keys in exactly three places:

  * ``System.__init__`` -- top-level YAML keys that are neither a registered
    component nor in ``system.RESERVED_CONFIG_KEYS``;
  * ``run.warn_unknown_sampler_keys`` -- keys inside the ``sampler:`` block
    that are not in ``run.KNOWN_SAMPLER_KEYS``;
  * ``diagnostics.ModelAuditor.check_unused_yaml`` -- per-parameter SUB-keys
    inside a params.yaml entry that are not in ``diagnostics.VALID_SUBKEYS``.

All three warnings say the key "will be ignored". When the declared set falls
behind the set the code actually consumes, that statement becomes false --
worse than silence, because it teaches users to disbelieve the warnings.
That is exactly what happened to ``modes``/``mkparam``/``gui`` (honored by
run.py, mkparam.py and gui/status.py) and to ``jitter`` (the key
``sample_jax_nuts``'s own comment tells users to opt back in with).

The cross-checks below are deliberately NOT a second copy of the same
strings: a test that re-listed the keys would have been edited in lockstep
with the bug and caught nothing. Instead they parse the shipped source with
``ast`` and collect the string keys the code really reads off the config
dicts, then compare in BOTH directions -- consumed-but-undeclared (false
"ignored" warnings) and declared-but-unconsumed (dead vocabulary).
"""

import ast
import pathlib

import pytest

from exozippy import config as config_mod
from exozippy import diagnostics as diagnostics_mod
from exozippy import introspect as introspect_mod
from exozippy.components.factory import discover_components
from exozippy.run import KNOWN_SAMPLER_KEYS, warn_unknown_sampler_keys
from exozippy.system import RESERVED_CONFIG_KEYS, System

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "exozippy"
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"

# Expressions that denote THE top-level system-config dict. A component's own
# ``self.config`` is its YAML block (a list), not the system config, so bare
# ``config`` is only trusted outside components/ (a component-local ``config``
# is a per-instance block -- see planet/symbolic_physics.py) and ``self.config``
# only inside system.py, where System owns it.
_TOP_LEVEL_RECEIVERS = {
    "system.config",
    "system_config",
    "self.system_config",
    "cm.system_config",
    "self.config_manager.system_config",
}

# Reserved keys that are deliberately accepted and then ignored, so no
# ``config["<key>"]`` read exists for the scanner to find. Keep this list
# tiny and justified -- an entry here is an exemption from the reverse check.
_INERT_RESERVED_KEYS = {
    # A free-form documentation block (``run: {name: ...}`` in every example
    # config). Nothing reads it; evaluator._NON_STRUCTURAL_CONFIG_KEYS only
    # excludes it from the structural hash.
    "run",
}


def _expr_src(node):
    """Source text of an AST expression, e.g. 'self.config_manager.system_config'."""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return ""


def _literal_keys_accessed(tree, receivers):
    """Every string literal used as a dict key on one of ``receivers``.

    Matches both ``recv["key"]`` and ``recv.get("key", ...)``. Non-literal
    keys (``config.get(comp_type)``) are invisible to this scan by design:
    they cannot be checked statically, and every key in the two vocabularies
    under test is spelled literally at its point of use.
    """
    found = {}
    for node in ast.walk(tree):
        recv = key = None
        if isinstance(node, ast.Subscript) and isinstance(
            node.slice, ast.Constant
        ):
            if isinstance(node.slice.value, str):
                recv, key = _expr_src(node.value), node.slice.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            recv, key = _expr_src(node.func.value), node.args[0].value
        if recv in receivers:
            found.setdefault(key, set()).add(node.lineno)
    return found


def _scan_top_level_config_keys():
    """{key: {"module:line", ...}} for every literal read of the system config."""
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        receivers = set(_TOP_LEVEL_RECEIVERS)
        if rel.parts[0] != "components":
            receivers.add("config")
        if rel.as_posix() == "system.py":
            receivers.add("self.config")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for key, lines in _literal_keys_accessed(tree, receivers).items():
            for line in lines:
                found.setdefault(key, set()).add(f"{rel.as_posix()}:{line}")
    return found


def _scan_sampler_cfg_keys():
    """{key: {"run.py:line", ...}} for every literal read of the sampler block."""
    tree = ast.parse((SRC / "run.py").read_text(encoding="utf-8"))
    return {
        key: {f"run.py:{line}" for line in lines}
        for key, lines in _literal_keys_accessed(tree, {"sampler_cfg"}).items()
    }


def _resolve_fn():
    """The ``ConfigManager.resolve`` FunctionDef, parsed from config.py."""
    tree = ast.parse((SRC / "config.py").read_text(encoding="utf-8"))
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ConfigManager":
            for fn in cls.body:
                if (
                    isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and fn.name == "resolve"
                ):
                    return fn
    raise AssertionError("ConfigManager.resolve not found in config.py")


def _scan_resolved_subkeys():
    """Every per-parameter sub-key ``ConfigManager.resolve`` reads.

    Structural, in three steps, so the answer follows a refactor instead of
    having to be re-typed after one:

      1. find the locals resolve() binds from ``self.user_params`` -- those
         hold ONE user override entry (``entry``, ``ov``);
      2. collect every sub-key tested or subscripted against them, i.e.
         ``"unit" in entry``, ``ov[key]``, ``ov.get(...)``;
      3. where the sub-key is a loop variable, resolve it back through any
         local alias to the module-level constant it iterates over and read
         that constant's real value off the imported module.

    Returns (keys, unresolved) -- the second is every name step 3 could not
    resolve, which the test asserts is empty rather than silently shrinking
    the vocabulary.
    """
    fn = _resolve_fn()

    # (1) locals bound from self.user_params.
    def _touches_user_params(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript):
                if _expr_src(sub.value) == "self.user_params":
                    return True
            if isinstance(sub, ast.Call) and isinstance(
                sub.func, ast.Attribute
            ):
                if _expr_src(sub.func.value) == "self.user_params":
                    return True
        return False

    entry_names = set()
    aliases = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if _touches_user_params(node.value):
            entry_names.add(target.id)
        elif isinstance(node.value, ast.Name):
            # e.g. `all_numeric = NUMERIC_KEYS`
            aliases[target.id] = node.value.id

    assert entry_names, "no local in resolve() reads self.user_params"

    # (3) loop variable -> the iterable it walks.
    loop_iter = {}
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.comprehension)):
            if isinstance(node.target, ast.Name) and isinstance(
                node.iter, ast.Name
            ):
                loop_iter[node.target.id] = node.iter.id

    def _constant_value(name):
        seen = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return getattr(config_mod, name, None)

    keys = set()
    unresolved = set()

    def _record(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.add(node.value)
        elif isinstance(node, ast.Name):
            value = _constant_value(loop_iter.get(node.id, node.id))
            if isinstance(value, (list, tuple, set, frozenset)):
                keys.update(value)
            else:
                unresolved.add(node.id)
        else:
            unresolved.add(_expr_src(node))

    # (2) sub-keys read off those entries.
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if (
                isinstance(node.ops[0], ast.In)
                and _expr_src(node.comparators[0]) in entry_names
            ):
                _record(node.left)
        elif isinstance(node, ast.Subscript):
            if _expr_src(node.value) in entry_names:
                _record(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and _expr_src(node.func.value) in entry_names
        ):
            _record(node.args[0])

    return keys, unresolved


# One probe value per declared sub-key, used to prove resolve() really
# applies it (test_no_dead_subkey_vocabulary). The target is star.distance:
# defaults.yaml gives it initval 10, lower 0.001, upper 1e5, unit pc and no
# mu/sigma/bound_scale, so 42 moves every numeric field -- apply_value keeps
# max(lower) and min(upper), and 42 is strictly inside that bracket, so one
# value serves both bounds. This is a table of VALUES, not of the
# vocabulary: the vocabulary is still iterated off config.USER_PARAM_KEYS,
# and a new key with no probe here fails the test by name.
_SUBKEY_PROBES = {
    "initval": 42.0,
    "lower": 42.0,
    "upper": 42.0,
    "mu": 42.0,
    "sigma": 42.0,
    "bound_scale": 42.0,
    "unit": "AU",
    "latex": "D_{probe}",
    "description": "probe",
    "print_to_table": False,
    "debug_print": True,
}

# Declared, legal in a params file, and deliberately NOT absorbed from a user
# entry -- so the probe above cannot apply to them.
_SUBKEYS_INERT_FROM_ENTRY = {
    # ConfigManager._strip_user_init_scales deletes it at construction (with
    # a warning) because the whitening scale is measured at startup now. It
    # stays in the vocabulary so old mkprior restart files are not called
    # typos, and resolve() still fills the field from defaults and hints.
    "init_scale",
}

# Companion keys a probe needs to be a legal entry at all. The baseline is
# resolved with the same context, so what the test measures is still the
# effect of the probed key alone.
_SUBKEY_PROBE_CONTEXT = {
    # config.validate_sigma_has_center refuses a sigma with no center.
    "sigma": {"mu": 5.0},
}


def _resolve_star_distance(entry):
    """Resolve star.distance with ``entry`` as the user's params override."""
    from exozippy.config import ConfigManager

    user_params = {"star.0.distance": dict(entry)} if entry else {}
    cm = ConfigManager(user_params, system_config={"star": [{"name": "A"}]})
    resolved = cm.resolve("star", "distance", shape=(1,))
    # numpy arrays do not compare with ==; repr is enough to spot a change.
    return {k: repr(v) for k, v in resolved.items()}


def _example_config_files():
    """Every examples/ YAML that is a system config (not a params/aux file)."""
    yaml = pytest.importorskip("yaml")
    out = []
    for path in sorted(EXAMPLES.rglob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        keys = {k for k in data if isinstance(k, str)}
        # 'parameter_file' is the decisive marker: every runnable system
        # config names one, and no params file or component input file does.
        if "parameter_file" in keys:
            out.append((path, keys))
    return out


# ---------------------------------------------------------------------------
# The two reported bugs (these fail on pre-fix src/).
# ---------------------------------------------------------------------------


def test_honored_global_blocks_are_not_warned_about(caplog):
    """
    Given a config using the global blocks run.py/mkparam.py/gui honor
      (`modes:`, `mkparam:`, `gui:`),
    When a System is constructed from it,
    Then no "will be ignored" warning is emitted for them.

    Regression (review 2.2.1): reserved_keys listed only run/parameter_file/
    prefix/sampler/name/logger_level, so every DC2018 run warned that 'modes'
    would be ignored -- about a key run.py reads twice.
    """
    # ARRANGE
    config = {
        "star": [{"name": "A"}],
        "modes": {"weights": "evidence", "ledger": False},
        "mkparam": {"n_seeds": 4},
        "gui": {"snapshot": True},
    }

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.system"):
        System(config, user_params={})

    # ASSERT
    ignored = [
        r.getMessage() for r in caplog.records if "ignored" in r.getMessage()
    ]
    assert ignored == [], f"honored keys reported as ignored: {ignored}"


def test_an_unreserved_top_level_key_is_still_reported(caplog):
    """
    Given a config with a top-level key that is neither a component nor
      reserved (here a plausible typo of `mkparam:`),
    When a System is constructed from it,
    Then it is warned about by name as "will be ignored", and nothing on the
      System answers to it.

    The mirror of the test above, and the top-level counterpart of
    test_a_genuine_typo_is_still_reported: widening RESERVED_CONFIG_KEYS to
    silence the false warnings must not silence the true ones. Both halves
    matter -- the warning is only worth reading if the key really is inert,
    which is asserted here too.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}], "mkparm": {"n_seeds": 4}}

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.system"):
        system = System(config, user_params={})

    # ASSERT
    assert any(
        "mkparm" in r.getMessage() and "will be ignored" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert "mkparm" not in system.active_components
    assert not hasattr(system, "mkparm")


def test_jitter_is_a_recognized_sampler_key(caplog):
    """
    Given a sampler block that opts back into jittered JAX starts,
    When the unknown-sampler-key check runs,
    Then nothing is reported as unrecognized.

    Regression (review 2.2.2): 'jitter' is consumed by the numpyro/blackjax
    branch (sample_jax_nuts(jitter=...)), and its own comment tells users to
    "Opt back in with 'jitter: true'" -- but it was missing from
    KNOWN_SAMPLER_KEYS, so following that instruction fired a warning saying
    the key was unknown and would be ignored.
    """
    # ARRANGE
    sampler_cfg = {"method": "numpyro", "jitter": True}

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.run"):
        unknown = warn_unknown_sampler_keys(sampler_cfg)

    # ASSERT
    assert unknown == []
    assert not [r for r in caplog.records if "sampler block" in r.getMessage()]


def test_a_genuine_typo_is_still_reported(caplog):
    """
    Given a sampler block with a misspelled key,
    When the unknown-sampler-key check runs,
    Then it is still reported -- widening the vocabulary must not mute it.
    """
    # ARRANGE / ACT
    with caplog.at_level("WARNING", logger="exozippy.run"):
        unknown = warn_unknown_sampler_keys({"jittr": True, "draws": 10})

    # ASSERT
    assert unknown == ["jittr"]
    assert any("will be ignored" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Anti-drift: the declared vocabularies vs the consumed ones.
# ---------------------------------------------------------------------------


def test_every_consumed_sampler_key_is_declared():
    """
    Given every `sampler_cfg["..."]` / `.get("...")` read in run.py's source,
    When compared against KNOWN_SAMPLER_KEYS,
    Then no consumed key is missing from it.

    Not a restatement of the set: the expected side is parsed out of run.py
    itself, so adding a `sampler_cfg.get("new_knob")` without declaring it
    fails here. This is how 'jitter' would have been caught.
    """
    # ARRANGE
    consumed = _scan_sampler_cfg_keys()
    # Sanity: the scanner must actually be finding reads, or the test is vacuous.
    assert len(consumed) > 10

    # ACT
    undeclared = {
        k: sorted(v)
        for k, v in consumed.items()
        if k not in KNOWN_SAMPLER_KEYS
    }

    # ASSERT
    assert undeclared == {}, (
        "sampler keys consumed by run.py but missing from KNOWN_SAMPLER_KEYS "
        f"(users get a false 'will be ignored' warning): {undeclared}"
    )


def test_no_dead_sampler_vocabulary():
    """
    Given KNOWN_SAMPLER_KEYS,
    When compared against the keys run.py actually reads,
    Then every declared key is consumed somewhere.

    The mirror direction: a declared-but-unread key silently accepts a knob
    that does nothing, which is the failure mode the review's sibling items
    (IMF: Kroupa, a typo'd ld_law) all shared.
    """
    # ARRANGE
    consumed = set(_scan_sampler_cfg_keys())

    # ACT
    dead = sorted(KNOWN_SAMPLER_KEYS - consumed)

    # ASSERT
    assert dead == [], f"declared sampler keys nothing reads: {dead}"


def test_every_consumed_top_level_key_is_a_component_or_reserved():
    """
    Given every literal read of the system-config dict across src/exozippy,
    When each key is checked against the component registry plus
      RESERVED_CONFIG_KEYS,
    Then all of them are recognized.

    Again parsed from the source rather than re-listed, so a new global block
    ("gui:", "modes:", ...) that is honored but not reserved fails here
    instead of shipping a warning that lies about it.
    """
    # ARRANGE
    registry = set(discover_components())
    consumed = _scan_top_level_config_keys()
    assert len(consumed) > 5  # scanner sanity

    # ACT
    unrecognized = {
        k: sorted(v)
        for k, v in consumed.items()
        if k not in registry and k not in RESERVED_CONFIG_KEYS
    }

    # ASSERT
    assert unrecognized == {}, (
        "top-level config keys the code honors but System warns are ignored: "
        f"{unrecognized}"
    )


def test_no_dead_reserved_config_vocabulary():
    """
    Given RESERVED_CONFIG_KEYS,
    When compared against the keys the source actually reads,
    Then every reserved key is either consumed or explicitly listed as inert.
    """
    # ARRANGE
    consumed = set(_scan_top_level_config_keys())

    # ACT
    dead = sorted(RESERVED_CONFIG_KEYS - consumed - _INERT_RESERVED_KEYS)

    # ASSERT
    assert dead == [], (
        "reserved top-level keys nothing reads; either wire them up, drop "
        f"them, or document them in _INERT_RESERVED_KEYS: {dead}"
    )


def test_no_example_config_triggers_the_ignored_warning():
    """
    Given every runnable system config under examples/,
    When its top-level keys are classified the way System.__init__ does,
    Then none of them would be warned about as ignored.

    A user-facing cross-check that does not depend on the AST scan: the
    shipped examples are the vocabulary we promise works.
    """
    # ARRANGE
    registry = set(discover_components())
    configs = _example_config_files()
    assert len(configs) > 5  # the examples tree must not have moved

    # ACT
    offenders = {}
    for path, keys in configs:
        bad = sorted(keys - registry - RESERVED_CONFIG_KEYS)
        if bad:
            offenders[path.name] = bad

    # ASSERT
    assert offenders == {}, (
        f"example configs using unreserved keys: {offenders}"
    )


# ---------------------------------------------------------------------------
# The third vocabulary: per-parameter SUB-keys inside a params.yaml entry.
#
# check_unused_yaml warns "did not match any model parameter" for anything it
# does not recognize. Its set had drifted the dangerous way -- it was missing
# latex/description/print_to_table/debug_print, four keys resolve() really
# absorbs, so setting a LaTeX label on a parameter earned a typo warning about
# a key that had been applied. There is also a THIRD copy of the same
# vocabulary in introspect._NUMERIC_FIELDS, which had lost bound_scale.
# ---------------------------------------------------------------------------


def test_every_subkey_resolve_consumes_is_declared():
    """
    Given the sub-keys ConfigManager.resolve() really reads off a user
      override entry, scanned out of config.py's own source,
    When they are compared with diagnostics.VALID_SUBKEYS,
    Then every one of them is declared.

    The consumed-but-undeclared direction: an omission here is a FALSE
    "ignored" warning about a key that was applied, which is worse than
    silence. latex/description/print_to_table/debug_print were all missing.
    """
    # ARRANGE
    consumed, unresolved = _scan_resolved_subkeys()

    # ACT
    undeclared = sorted(consumed - set(diagnostics_mod.VALID_SUBKEYS))

    # ASSERT
    assert unresolved == set(), (
        f"sub-keys the scan could not resolve statically: {sorted(unresolved)}"
        " -- the scan, not the vocabulary, needs updating"
    )
    assert undeclared == [], (
        "resolve() absorbs these sub-keys but check_unused_yaml calls them "
        f"typos: {undeclared}"
    )


def test_no_dead_subkey_vocabulary():
    """
    Given every sub-key config.USER_PARAM_KEYS declares,
    When one is added to a params entry and star.distance is resolved,
    Then the resolved parameter really changes.

    The declared-but-unconsumed direction, and it has to be BEHAVIORAL: the
    AST scan resolves ``for key in NUMERIC_KEYS`` to the constant's live
    value, so it would agree with the constant whatever the constant said.
    Actually resolving is what a declared-but-never-read key fails. Widening
    the set to kill the false positives above must not blind the check --
    accepting a key nothing reads means a real typo that happens to collide
    with it passes silently.
    """
    # ARRANGE
    vocabulary = set(config_mod.USER_PARAM_KEYS) - _SUBKEYS_INERT_FROM_ENTRY
    missing_probe = sorted(vocabulary - set(_SUBKEY_PROBES))
    assert missing_probe == [], (
        f"no probe value declared for new sub-key(s) {missing_probe}; add "
        "one to _SUBKEY_PROBES (and if the key cannot be probed because "
        "nothing reads it off a user entry, that is the bug this test hunts)"
    )

    # ACT
    inert = []
    for key in sorted(vocabulary):
        context = _SUBKEY_PROBE_CONTEXT.get(key, {})
        baseline = _resolve_star_distance(context)
        probed = _resolve_star_distance({**context, key: _SUBKEY_PROBES[key]})
        if probed == baseline:
            inert.append(key)

    # ASSERT
    assert inert == [], (
        "sub-keys declared in config.USER_PARAM_KEYS that resolve() never "
        f"applies; either wire them up or drop them: {inert}"
    )


def test_the_inert_subkeys_really_are_inert():
    """
    Given diagnostics._INERT_SUBKEYS -- the exemptions to the test above,
    When each is compared with what resolve() consumes,
    Then none of them is a key resolve() actually reads.

    An exemption list is the one way to launder a genuine key into "we do
    not check that one", so the list itself is checked: an entry here must
    be inert, not merely unaudited.
    """
    # ARRANGE
    consumed, _ = _scan_resolved_subkeys()

    # ACT
    live = sorted(set(diagnostics_mod._INERT_SUBKEYS) & consumed)

    # ASSERT
    assert live == [], f"exempted as inert but resolve() reads them: {live}"


def test_the_three_subkey_vocabularies_share_one_owner():
    """
    Given config.USER_PARAM_KEYS, diagnostics.VALID_SUBKEYS and
      introspect._NUMERIC_FIELDS,
    When their provenance is inspected,
    Then the two consumers are built FROM the config constants rather than
      restating them.

    Structural on purpose: a test that re-listed the strings would be edited
    in lockstep with the next drift and catch nothing. Object identity for
    introspect and a set equation for diagnostics are what a copy-paste
    reintroduction fails.
    """
    # ARRANGE
    owner = config_mod

    # ACT / ASSERT -- the union is exactly its declared parts.
    assert set(owner.USER_PARAM_KEYS) == (
        set(owner.NUMERIC_KEYS) | set(owner.STRING_KEYS) | set(owner.BOOL_KEYS)
    )
    assert set(owner.NUMERIC_KEYS) == (
        set(owner.TUNING_KEYS) | set(owner.PHYSICS_KEYS)
    )

    # introspect must be the constant itself, not a copy of its contents.
    assert introspect_mod._NUMERIC_FIELDS is owner.NUMERIC_KEYS

    # diagnostics must be the config vocabulary plus its documented inerts.
    assert set(diagnostics_mod.VALID_SUBKEYS) == (
        set(owner.USER_PARAM_KEYS) | set(diagnostics_mod._INERT_SUBKEYS)
    )


def test_a_valid_subkey_is_not_reported_but_a_typo_still_is():
    """
    Given a params entry carrying every legal sub-key plus one typo,
    When check_unused_yaml classifies them,
    Then only the typo is reported.

    The user-facing half, independent of the AST scan. Both directions in
    one assertion: on pre-fix src/ this reports latex, description,
    print_to_table and debug_print alongside 'intival'.
    """

    # ARRANGE
    class _FakeParam:
        label = "star.teff"
        shape = ()

        def get_display_label(self, i):
            return "star.A.teff"

    entry = {k: 1 for k in config_mod.USER_PARAM_KEYS}
    entry["intival"] = 5800

    class _FakeSystem:
        user_params = {"star.A.teff": entry}

        def get_parameter_lookup(self):
            return {}

        def get_all_parameters(self):
            return [_FakeParam()]

    auditor = diagnostics_mod.ModelAuditor.__new__(
        diagnostics_mod.ModelAuditor
    )
    system = _FakeSystem()
    auditor.system = system
    auditor.user_params = system.user_params
    auditor.all_params = system.get_all_parameters()

    # ACT
    reported = auditor.check_unused_yaml()

    # ASSERT
    assert reported == ["star.A.teff -> 'intival'"]
