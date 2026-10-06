"""The raw-start size invariant raises instead of repairing (2.14.15).

A Parameter's ``<label>_raw`` variable, its ``raw_initval`` and every raw
start built for it hold one entry per SAMPLED element, because
``Parameter.build_pymc`` sets all three in one call.  Sites that used to
zero, pad or skip on a mismatch made every consumer of the start agree on
the same wrong start, so no equality test could see it.
``Parameter.check_raw_size`` is the one raising check; these tests break
the invariant on purpose and assert the raise names the offender.

These guards are dead on correct code by construction: the acceptance for
the change is a bit-identical start logp on the shipped examples, not these
tests.
"""

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter


def _toy_vector_model():
    """A 3-element parameter with element 1 pinned: 2 sampled elements."""
    p = Parameter(
        label="toy.x",
        initval=np.array([2.0, 5.0, 3.0]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([10.0, 10.0, 10.0]),
        sigma=np.array([np.nan, 0.0, np.nan]),
        shape=(3,),
    )
    with pm.Model() as model:
        p.build_pymc()
    return model, p


# ---------------------------------------------------------------------------
# Parameter.check_raw_size (2.14.15)
# ---------------------------------------------------------------------------


def test_check_raw_size_passes_a_correct_vector_through_flat():
    """
    Given a parameter with two sampled elements,
    When check_raw_size is handed a 2-entry raw vector of any shape,
    Then it returns it as a flat float64 array, unchanged in value.
    """
    # ARRANGE
    _, p = _toy_vector_model()
    assert len(p._raw_transform["sampled_idx"]) == 2

    # ACT
    out = p.check_raw_size(np.array([[0.5], [-1.0]]), "test")

    # ASSERT
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0.5, -1.0])


@pytest.mark.parametrize("bad", [np.zeros(1), np.zeros(3), np.zeros(0)])
def test_check_raw_size_raises_on_a_mis_sized_vector(bad):
    """
    Given a parameter with two sampled elements,
    When check_raw_size is handed a vector of any other size,
    Then ValueError names the site, the parameter and both sizes.
    """
    # ARRANGE
    _, p = _toy_vector_model()

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        p.check_raw_size(bad, "my_site[toy.x_raw]")
    msg = str(exc.value)
    assert "my_site[toy.x_raw]" in msg
    assert "'toy.x'" in msg
    assert f"has {bad.size} entries" in msg
    assert "raw variable has 2" in msg


def test_check_raw_size_raises_on_none():
    """
    Given a parameter with two sampled elements,
    When check_raw_size is handed None (a raw_initval never written),
    Then ValueError names the parameter rather than np.asarray(None)
      quietly becoming a 1-entry NaN vector.
    """
    # ARRANGE
    _, p = _toy_vector_model()

    # ACT / ASSERT
    with pytest.raises(
        ValueError, match=r"site: Parameter 'toy.x' raw vector is None"
    ):
        p.check_raw_size(None, "site")


def test_check_raw_size_raises_without_a_raw_transform():
    """
    Given a parameter build_pymc never ran on (no raw transform),
    When check_raw_size is asked about a raw vector for it,
    Then ValueError names the parameter: it has no raw variable at all.
    """
    # ARRANGE
    p = Parameter(label="toy.y", initval=1.0, lower=0.0, upper=2.0)

    # ACT / ASSERT
    with pytest.raises(
        ValueError, match=r"site: Parameter 'toy.y' has no raw transform"
    ):
        p.check_raw_size(np.zeros(1), "site")
