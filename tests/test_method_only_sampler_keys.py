"""Review 2.4.2 / 2.3.6: a sampler key some method ignores must not be silent.

2.4.2 landed the mechanism for the three keys it had traced:
`store_hot_chains` is forwarded only to ptde_async and `rung_thin_factor` /
`rung_thin_start` only to ptde, but all three are in KNOWN_SAMPLER_KEYS -- so
warn_unknown_sampler_keys says nothing and the feature simply never runs.

2.3.6 is that finding on the full list: at least a dozen more keys are read by
exactly one branch or family.  THE HEADLINE IS `chains`, which is forwarded to
the HMC branches and to demc/demcz and to nothing else -- so under
method: ptde / ptde_async, the recommended default for every microlensing fit,
a user's `sampler: {chains: 16}` was silently ignored and the samplers sized
their population from the parameter count instead.  Measured end to end on
examples/kelt4 RV-only, 2026-09-11: `chains: 5` gave 5 chains under
method: nuts and 30 under method: ptde.

Two properties of the mechanism this file pins, because both are what keeps a
warning worth reading:

* it reports only keys the user EXPLICITLY set (every one of them has a
  default, and warning about a default nobody wrote would fire every run);
* it scores the key against the branch that will REALLY run, so an
  unrecognized `method:` value -- which falls through to nuts, as it always
  has -- is not told that `chains` is ignored when the nuts branch reads it.
"""

import pytest

from exozippy.run import (
    ALL_METHOD_SAMPLER_KEYS,
    KNOWN_SAMPLER_KEYS,
    METHOD_ONLY_SAMPLER_KEYS,
    SAMPLER_METHODS,
    warn_method_only_sampler_keys,
)


@pytest.mark.parametrize(
    "key,bad_method",
    [
        # The three 2.4.2 originals.
        ("store_hot_chains", "ptde"),
        ("rung_thin_factor", "ptde_async"),
        ("rung_thin_start", "ptde_async"),
        # 2.3.6's headline, in both spellings of the recommended default.
        ("chains", "ptde"),
        ("chains", "ptde_async"),
        ("chains", "nested"),
        # The PTDE family's knobs under an HMC method.
        ("n_temps", "nuts"),
        ("T_max", "nuts"),
        ("n_chains", "numpyro"),
        ("adapt_ladder", "nuts"),
        ("de_mode_hop", "nuts"),
        ("eval_timeout", "nuts"),
        ("swap_schedule", "nuts"),
        ("collect_rung_timing", "nuts"),
        # nested-only knobs under the default sampler.
        ("nested_backend", "nuts"),
        ("nlive", "nuts"),
        ("dlogz", "nuts"),
        ("walks", "nuts"),
        ("checkpoint_dir", "nuts"),
        # JAX-only knobs under PyMC NUTS.
        ("chain_method", "nuts"),
        ("jitter", "nuts"),
        # A Hamiltonian knob under a gradient-free sampler.
        ("target_accept", "ptde_async"),
        ("target_accept", "demc"),
        # Chain length, against the one method whose length is its own.
        ("tune", "nested"),
        ("draws", "nested"),
    ],
)
def test_a_key_the_chosen_method_ignores_is_reported(key, bad_method, caplog):
    """
    Given a sampler key that only some methods consume,
    When the fit selects one that does NOT,
    Then a warning names the key and how to make it count.
    """
    # ARRANGE / ACT
    warned = warn_method_only_sampler_keys({key: 1}, bad_method)

    # ASSERT
    assert warned == [(key, bad_method)]
    assert key in caplog.text and "IGNORED" in caplog.text


@pytest.mark.parametrize(
    "key,good_method",
    [
        ("store_hot_chains", "ptde_async"),
        ("rung_thin_factor", "ptde"),
        # `chains` under every method that really forwards it.
        ("chains", "nuts"),
        ("chains", "numpyro"),
        ("chains", "blackjax"),
        ("chains", "nutpie"),
        ("chains", "demc"),
        ("chains", "demcz"),
        ("n_temps", "ptde"),
        ("n_temps", "ptde_async"),
        ("swap_schedule", "ptde_async"),
        ("eval_timeout", "ptde_async"),
        ("nlive", "nested"),
        ("checkpoint_dir", "nested"),
        ("jitter", "numpyro"),
        ("chain_method", "blackjax"),
        ("target_accept", "nuts"),
        ("target_accept", "nutpie"),
        ("draws", "ptde"),
        ("tune", "demcz"),
    ],
)
def test_the_consuming_method_is_silent(key, good_method):
    """
    Given the same key,
    When the chosen method is one that consumes it,
    Then nothing is warned.
    """
    assert warn_method_only_sampler_keys({key: 1}, good_method) == []


def test_an_unset_key_is_silent_even_under_the_wrong_method():
    """
    Given a sampler block that does not mention the key at all,
    When any method runs,
    Then nothing is warned -- these all have defaults, and warning about a
    default nobody wrote would fire on every run.
    """
    assert warn_method_only_sampler_keys({"method": "nuts"}, "nuts") == []


def test_a_shipped_default_sampler_block_is_silent():
    """
    Given the shape of a real config -- several keys the chosen method does
      consume,
    When the check runs,
    Then it says nothing at all.

    The cry-wolf guard: a table that over-claims turns every run's log into
    noise, which is the failure that makes the honest warnings unreadable.
    """
    cfg = {
        "method": "ptde_async",
        "n_temps": 8,
        "T_max": 200,
        "tune": 5000,
        "draws": 50000,
        "nthin": 1,
        "recompute_trace": True,
        "eval_timeout": 10,
        "cores": 64,
        "seed_polish": "on",
        "store_hot_chains": True,
    }
    assert warn_method_only_sampler_keys(cfg, "ptde_async") == []


def test_an_unrecognized_method_is_scored_as_the_nuts_fallthrough():
    """
    Given a misspelled `method:` value, which falls through to the nuts
      branch as it always has,
    When a key the nuts branch DOES consume is set,
    Then nothing is warned -- and a key nuts really ignores still is.

    Without this the generalized table would have started lying: the branch
    that runs consumes `chains`, so "chains is IGNORED" would be false, and a
    warning that cries wolf is precisely what this mechanism exists to avoid.
    """
    assert warn_method_only_sampler_keys({"chains": 8}, "nutts") == []
    assert warn_method_only_sampler_keys({"nlive": 500}, "nutts") == [
        ("nlive", "nutts")
    ]


def test_the_message_names_the_shorter_side_of_the_split():
    """
    Given a key only ONE method ignores (`draws` under nested),
    When the warning is emitted,
    Then it names the method that ignores it rather than listing the eight
      that do not.

    "only nuts / numpyro / blackjax / nutpie / ptde / ptde_async / demc /
    demcz consume it" is a list the reader has to diff by hand to find the
    one method that matters.
    """
    import logging

    from exozippy import run as run_mod

    logger = logging.getLogger(run_mod.__name__)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        warn_method_only_sampler_keys({"draws": 5000}, "nested")
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    text = records[0].getMessage()
    assert "except nested" in text
    assert "ptde_async" not in text


def test_every_method_only_key_is_a_known_sampler_key():
    """
    Given the METHOD_ONLY_SAMPLER_KEYS table,
    When it is compared against KNOWN_SAMPLER_KEYS,
    Then every entry is known -- an unknown key is the OTHER warning's job,
    and a key in neither table would be silent twice over.
    """
    assert set(METHOD_ONLY_SAMPLER_KEYS) <= KNOWN_SAMPLER_KEYS


def test_every_consumer_named_is_a_real_dispatch_branch():
    """
    Given every method named as a consumer,
    When it is checked against the dispatch's own method list,
    Then all of them exist.

    A typo'd consumer ("ptde_ascyn") would make the key look method-only for
    a method nothing dispatches to, i.e. warn under the method that really
    reads it -- the cry-wolf direction, and invisible without this.
    """
    unknown = {
        key: sorted(set(consumers) - set(SAMPLER_METHODS))
        for key, consumers in METHOD_ONLY_SAMPLER_KEYS.items()
        if set(consumers) - set(SAMPLER_METHODS)
    }
    assert unknown == {}


def test_the_two_tables_partition_the_sampler_vocabulary():
    """
    Given METHOD_ONLY_SAMPLER_KEYS and ALL_METHOD_SAMPLER_KEYS,
    When their union and intersection are compared with KNOWN_SAMPLER_KEYS,
    Then they partition it exactly.

    This is what makes 2.3.6 stay fixed.  The defect was not any one key; it
    was that a key could join the vocabulary with nobody ruling on whether
    some method silently ignores it.  A new `sampler:` key now fails HERE
    until it is classified, and a key that moves from one table to the other
    has to be moved rather than copied.
    """
    method_only = set(METHOD_ONLY_SAMPLER_KEYS)
    all_method = set(ALL_METHOD_SAMPLER_KEYS)

    assert method_only & all_method == set(), (
        "a key cannot be both method-specific and read on every path: "
        f"{sorted(method_only & all_method)}"
    )
    unclassified = sorted(KNOWN_SAMPLER_KEYS - method_only - all_method)
    assert unclassified == [], (
        "sampler keys in neither table -- rule on whether some method "
        "ignores them, then add them to METHOD_ONLY_SAMPLER_KEYS or to "
        f"ALL_METHOD_SAMPLER_KEYS: {unclassified}"
    )
    stale = sorted((method_only | all_method) - KNOWN_SAMPLER_KEYS)
    assert stale == [], f"classified keys that are no longer known: {stale}"
