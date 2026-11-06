"""blackjax's parallel chains need one JAX device per chain (#506).

pymc maps blackjax's `chain_method: parallel` onto `jax.pmap` with no device
check.  Importing `pymc.sampling.jax` forces 100 host devices, but only if the
JAX backend has not started; afterwards there is one device and two chains
crash with an opaque "cannot select an axis to squeeze out".  run.py warns and
runs the chains vectorized instead.
"""

import logging

import pytest

from exozippy.run import _blackjax_chain_method


def test_too_few_devices_switches_parallel_to_vectorized_and_warns(caplog):
    """
    Given parallel chains, two chains and one JAX device,
    When the blackjax chain method is chosen,
    Then it is vectorized and the warning names both counts.
    """
    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.run"):
        got = _blackjax_chain_method("parallel", 2, 1)

    # Assert
    assert got == "vectorized"
    assert "2 chains" in caplog.text
    assert "1 device(s)" in caplog.text


@pytest.mark.parametrize(
    "chain_method, chains, n_devices",
    [
        ("parallel", 2, 100),
        ("parallel", 4, 4),
        ("vectorized", 2, 1),
    ],
)
def test_a_runnable_chain_method_is_kept_silently(
    chain_method, chains, n_devices, caplog
):
    """
    Given a chain method the devices can run (enough devices, or vectorized),
    When the blackjax chain method is chosen,
    Then it is returned unchanged with no warning.
    """
    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.run"):
        got = _blackjax_chain_method(chain_method, chains, n_devices)

    # Assert
    assert got == chain_method
    assert caplog.text == ""
