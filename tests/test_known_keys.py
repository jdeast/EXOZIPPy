# tests/test_known_keys.py
"""Anti-drift cover for the two hand-maintained "known key" vocabularies.

EXOZIPPy warns about unrecognized keys in exactly two places:

  * ``System.__init__`` -- top-level YAML keys that are neither a registered
    component nor in ``system.RESERVED_CONFIG_KEYS``;
  * ``run.warn_unknown_sampler_keys`` -- keys inside the ``sampler:`` block
    that are not in ``run.KNOWN_SAMPLER_KEYS``.

Both warnings say the key "will be ignored". When the declared set falls
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
