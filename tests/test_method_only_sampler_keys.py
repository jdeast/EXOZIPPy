"""Review 2.4.2: a sampler key only one method consumes must not be silent.

`store_hot_chains` is forwarded only to ptde_async and `rung_thin_factor` /
`rung_thin_start` only to ptde, but all three are in KNOWN_SAMPLER_KEYS -- so
warn_unknown_sampler_keys says nothing and the feature simply never runs.
"""

import pytest

from exozippy.run import (
    KNOWN_SAMPLER_KEYS,
    METHOD_ONLY_SAMPLER_KEYS,
    warn_method_only_sampler_keys,
)


@pytest.mark.parametrize(
    "key,bad_method",
    [
        ("store_hot_chains", "ptde"),
        ("rung_thin_factor", "ptde_async"),
        ("rung_thin_start", "ptde_async"),
    ],
)
def test_a_key_the_chosen_method_ignores_is_reported(key, bad_method, caplog):
    """
    Given a sampler key that only one method consumes,
    When the fit selects a DIFFERENT method,
    Then a warning names the key and the methods that would consume it.
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


def test_every_method_only_key_is_a_known_sampler_key():
    """
    Given the METHOD_ONLY_SAMPLER_KEYS table,
    When it is compared against KNOWN_SAMPLER_KEYS,
    Then every entry is known -- an unknown key is the OTHER warning's job,
    and a key in neither table would be silent twice over.
    """
    assert set(METHOD_ONLY_SAMPLER_KEYS) <= KNOWN_SAMPLER_KEYS
