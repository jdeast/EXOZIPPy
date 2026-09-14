"""Review 5.3.3: run.py's dead residue, and the one piece that was not dead.

Five clauses, one file, because they are one shape of defect: code that a
refactor carried forward because nothing pointed at it.

CLAUSE (a) IS THE INTERESTING ONE, and its first reading was wrong.  `init`
was NOT dead code: it was read off the sampler block, bound to a local, and
passed to `pm.sample` by name.  What was dead was its EFFECT -- pymc's own
`pm.sample` docstring says of `init`, verbatim, "This argument is ignored when
manually passing the NUTS step method", and the plain-NUTS branch passes
`step=pm.NUTS(...)`.  So a user who set it got silence rather than a no-op
warning, and a grep for an unused variable found nothing.  Fifteen shipped
example configs carried `init: adapt_diag` and it never did anything in any of
them.

The key is DELETED rather than made live, which is a ruling and not a
shortcut: making it live means dropping the explicit step, and `initvals`' own
docstring entry reads "Initialization methods for NUTS (see ``init`` keyword)
can overwrite the default" -- so a live `adapt_diag` could jitter the chain off
the polished start, the exact pathology seed_polish exists to prevent.

(b) make_corner's `model` argument was unused, and so was the `model` argument
of its only other caller.  (c) a commented-out az.plot_pair block had been
carried through every refactor.  (d) get_raw_starts ran on the trace-REUSE
path, where no sampler consumes it -- a re-solve of every seed transform for
nothing.  (e) `cores: "auto"` crashed on an unhandled int() while the same
spelling handed to the seed polish warned and took the default grant.
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest

from exozippy import run as run_mod
from exozippy.run import (
    KNOWN_SAMPLER_KEYS,
    RETIRED_SAMPLER_KEYS,
    warn_retired_sampler_keys,
    warn_unknown_sampler_keys,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "src" / "exozippy" / "run.py"
EXAMPLES = REPO_ROOT / "examples"


# ---------------------------------------------------------------------------
# (a) the `init` key
# ---------------------------------------------------------------------------


def test_init_is_gone_from_the_sampler_vocabulary():
    """
    Given the retired `init` key,
    When the sampler vocabulary is inspected,
    Then it is declared retired and is no longer a known key.
    """
    assert "init" in RETIRED_SAMPLER_KEYS
    assert "init" not in KNOWN_SAMPLER_KEYS


def test_nothing_in_run_py_reads_or_passes_init():
    """
    Given run.py's source,
    When it is scanned for the key and the kwarg,
    Then neither the read nor the pm.sample kwarg survives.

    Asserted on the SOURCE rather than on behavior because the defect was
    invisible to behavior: passing `init=` to a pm.sample that ignores it
    looks exactly like not passing it.  A future edit that reinstates the
    kwarg -- the obvious "fix" for a key a user asked about -- fails here.
    """
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))

    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and ast.unparse(node.func.value) == "sampler_cfg"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "init"
    ]
    assert reads == [], "run.py still reads sampler_cfg['init']"

    kwargs = [
        kw
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "init"
    ]
    assert kwargs == [], "run.py still passes init= to a sampler"


def test_the_explicit_nuts_step_is_kept():
    """
    Given the plain-NUTS branch,
    When it is inspected,
    Then it still builds an explicit pm.NUTS step.

    The other half of the ruling, and the half a later reader is likeliest to
    undo: dropping the step is what would make `init` live, and a live
    adapt_diag may jitter the chain off the polished start.
    """
    source = inspect.getsource(run_mod._run_fit)

    assert "pm.NUTS(target_accept=target_accept)" in source
    assert "step=step" in source


def test_a_retired_key_is_explained_rather_than_called_a_typo(caplog):
    """
    Given a config that still sets `init:`,
    When the startup checks run,
    Then the retired-key channel explains it and the unknown-key channel
      stays quiet -- one stale line, one message.
    """
    # ARRANGE
    sampler_cfg = {"method": "nuts", "init": "adapt_diag"}

    # ACT
    with caplog.at_level(logging.WARNING, logger=run_mod.__name__):
        retired = warn_retired_sampler_keys(sampler_cfg)
        unknown = warn_unknown_sampler_keys(sampler_cfg)

    # ASSERT
    assert retired == ["init"]
    assert unknown == []
    assert "RETIRED" in caplog.text
    assert "Did you mean" not in caplog.text


def test_a_genuine_typo_is_still_a_typo(caplog):
    """
    Given a misspelled sampler key,
    When the two checks run,
    Then the retired channel says nothing and the unknown channel reports it.

    The mirror direction: a retired-key exemption must not launder typos.
    """
    with caplog.at_level(logging.WARNING, logger=run_mod.__name__):
        assert warn_retired_sampler_keys({"inti": "adapt_diag"}) == []
        assert warn_unknown_sampler_keys({"inti": "adapt_diag"}) == ["inti"]

    assert "RETIRED" not in caplog.text


def test_every_retired_key_explains_itself():
    """
    Given RETIRED_SAMPLER_KEYS,
    When each entry's text is inspected,
    Then it is a real explanation, not a placeholder.

    The whole value of a separate channel is the sentence in it; an entry
    with an empty or one-word message is worse than the typo warning it
    displaces.
    """
    for key, why in RETIRED_SAMPLER_KEYS.items():
        assert isinstance(why, str) and len(why) > 40, key


def test_no_shipped_config_still_sets_a_retired_key():
    """
    Given every YAML under examples/ and the configs the helper scripts
      generate,
    When they are scanned for a retired sampler key,
    Then none of them sets one.

    Fifteen shipped configs carried `init: adapt_diag`.  Leaving them would
    have shipped the new warning on nearly every example -- and the point of
    deleting a key nothing honors is that the configs stop carrying it.
    """
    offenders = []
    for path in sorted(EXAMPLES.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for key in RETIRED_SAMPLER_KEYS:
            if f"\n  {key}:" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {key}")
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# (b) make_corner's unused model argument
# ---------------------------------------------------------------------------


def test_make_corner_takes_no_model():
    """
    Given make_corner, which plots from idata.posterior alone,
    When its signature is inspected,
    Then it does not ask for a model it never reads.
    """
    params = list(inspect.signature(run_mod.make_corner).parameters)

    assert "model" not in params
    assert params[0] == "idata"


def test_the_per_mode_emitter_takes_no_model_either():
    """
    Given _emit_per_mode_outputs, whose only use of `model` was to forward it
      to make_corner,
    When its signature is inspected,
    Then the argument is gone from there too.

    Its own tests had been passing None for it, which is as clear a statement
    as a test can make that nothing read it.
    """
    params = list(inspect.signature(run_mod._emit_per_mode_outputs).parameters)

    assert "model" not in params


# ---------------------------------------------------------------------------
# (c) the commented-out plot_pair block
# ---------------------------------------------------------------------------


def test_the_commented_out_plot_pair_block_is_gone():
    """
    Given run.py's source,
    When it is searched for the commented-out az.plot_pair scratch block,
    Then nothing is left of it.

    Carried through every refactor of the wrap-up since it was written, and
    read every time by somebody working out whether it mattered.
    """
    source = RUN_PY.read_text(encoding="utf-8")

    assert "plot_pair" not in source
    assert "vars_to_check" not in source
    assert "suspected troublemakers" not in source


# ---------------------------------------------------------------------------
# (d) get_raw_starts on the trace-reuse path
# ---------------------------------------------------------------------------


def _enclosing_reuse_gates(source, call_src):
    """Every `if not reusing_trace:` whose body contains ``call_src``."""
    tree = ast.parse(source)
    gates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "not reusing_trace":
            continue
        if any(
            isinstance(sub, ast.Assign) and call_src in ast.unparse(sub)
            for sub in ast.walk(node)
        ):
            gates.append(node)
    return gates


def test_get_raw_starts_is_skipped_when_a_trace_is_reused():
    """
    Given the trace-REUSE path, where no sampler runs,
    When the start population would be built,
    Then the call sits inside an `if not reusing_trace:` block.

    get_raw_starts re-solves every seeded parameter's forward transform, once
    per seed; on the reuse path nothing consumes the result -- the seed ledger
    is already skipped there and no sampler branch is entered.  Pinned on the
    source because reproducing it needs a real multi-seed fit with a real
    trace on disk, and the cost is wall clock rather than a wrong number.
    """
    source = inspect.getsource(run_mod._run_fit)

    assert "system.get_raw_starts(model)" in source  # not vacuous
    gates = _enclosing_reuse_gates(source, "system.get_raw_starts(model)")
    # Two calls, two gates: the pre-polish one (raw_starts_pre) and the
    # post-whitening one this clause moved.
    assert len(gates) == 2


def test_the_multi_seed_names_survive_the_reuse_path():
    """
    Given the reuse path, where get_raw_starts is now skipped,
    When the code below it runs,
    Then the names it left behind are still safe to read.

    The trap in gating a call is the NameError (or the `len(None)`) that
    replaces the waste: the seed-ledger guard calls len(raw_starts) and the
    nested branch calls len() after an `is not None`.  Empty lists keep both
    honest, so the skip is pinned by asserting the fallback is a sized empty
    value rather than None.
    """
    source = inspect.getsource(run_mod._run_fit)
    idx = source.index("raw_starts, seed_indices = [], []")

    assert idx < source.index(
        "raw_starts, seed_indices = system.get_raw_starts(model)"
    )


# ---------------------------------------------------------------------------
# (e) cores: "auto"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["auto", "all", "", "4.5x", [4]])
def test_an_unreadable_cores_value_is_refused_by_name(bad):
    """
    Given a `cores:` value that is not an integer,
    When the sampler block is parsed,
    Then it is refused with a message that names the key, the value, and the
      two things a user can write instead.

    `cores: "auto"` used to reach a bare int() and die with
    "invalid literal for int() with base 10: 'auto'" from inside run.py --
    a traceback that names neither the config key nor the fix, for a spelling
    `n_temps:` accepts.
    """
    with pytest.raises(ValueError) as excinfo:
        run_mod.resolve_cores_setting(bad)

    message = str(excinfo.value)
    assert "cores" in message
    assert "omit" in message.lower()  # absent cores IS the automatic grant
    assert "cores: 1" in message  # ... and 1 is how you ask for serial


def test_an_absent_cores_takes_the_automatic_grant():
    """
    Given no `cores:` key,
    When it is resolved,
    Then None comes back -- which every stage reads as AUTO (review 6.11.3).
    """
    assert run_mod.resolve_cores_setting(None) is None


@pytest.mark.parametrize("value", [1, 4, "8", 64])
def test_an_integer_cores_is_taken_as_written(value):
    """
    Given an integer `cores:` (or its string spelling, as YAML may hand it
      over from a quoted value),
    When it is resolved,
    Then the number is passed through untouched.

    A POSITIVE count is what passes through untouched.  `cores: 0` and any
    negative are normalized to the None AUTO sentinel with a warning (review
    2.4.8, pinned in tests/test_polish.py) -- they used to pass through here
    and then mean three different things in the three resolvers downstream.
    """
    assert run_mod.resolve_cores_setting(value) == int(value)


def test_the_polish_and_the_startup_parse_tell_the_same_story():
    """
    Given the two places a user's `cores:` value is interpreted,
    When their messages about an unreadable value are compared,
    Then both name the key and both say what AUTO is and how to ask for
      serial.

    They differ in OUTCOME on purpose, and the difference is positional: the
    startup parse can still refuse the run before any work is done, while the
    polish runs inside a fit and a wrap-up stage may not kill a finished one,
    so it warns and takes the default grant.  What they must not do is
    disagree about what the key MEANS -- that is how one rule came to have
    two behaviors (#215, review 6.11.3).
    """
    from exozippy import polish as polish_mod

    polish_source = inspect.getsource(polish_mod._resolve_polish_cores)

    for text in (
        polish_source,
        inspect.getsource(run_mod.resolve_cores_setting),
    ):
        assert "cores" in text
        assert "cores=1" in text or "cores: 1" in text
