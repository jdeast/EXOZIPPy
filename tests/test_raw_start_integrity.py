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
from exozippy.system import System


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


# ---------------------------------------------------------------------------
# System.get_raw_start (2.3.22)
# ---------------------------------------------------------------------------


class _StubSystem:
    """Duck-typed stand-in for System: parameter lookup + seed storage."""

    def __init__(self, params, seed_resolved=None):
        self._params = params

        class _CM:
            pass

        self.config_manager = _CM()
        self.config_manager.seed_resolved = seed_resolved

    def get_all_parameters(self):
        return self._params


def test_get_raw_start_reads_raw_initval_on_correct_code():
    """
    Given a built parameter with a displaced raw_initval,
    When get_raw_start builds the start,
    Then the start holds raw_initval for that raw variable (the non-vacuity
      check for the raising tests below).
    """
    # ARRANGE
    model, p = _toy_vector_model()
    p.raw_initval = np.array([0.25, -0.5])

    # ACT
    start = System.get_raw_start(_StubSystem([p]), model)

    # ASSERT
    np.testing.assert_array_equal(start["toy.x_raw"], [0.25, -0.5])


def test_get_raw_start_raises_on_a_raw_variable_with_no_parameter():
    """
    Given a model whose free variable 'toy.x_raw' has no Parameter in the
      system (a label / raw-name mismatch),
    When get_raw_start builds the start,
    Then KeyError names the raw variable, instead of the start silently
      becoming raw = 0 for it.
    """
    # ARRANGE
    model, _ = _toy_vector_model()
    other = Parameter(label="toy.other", initval=1.0, lower=0.0, upper=2.0)

    # ACT / ASSERT
    with pytest.raises(
        KeyError, match=r"raw variable 'toy.x_raw' has no Parameter"
    ):
        System.get_raw_start(_StubSystem([other]), model)


@pytest.mark.parametrize(
    "bad, fragment",
    [
        (np.zeros(3), "raw vector has 3 entries but its raw variable has 2"),
        (None, "raw vector is None"),
    ],
)
def test_get_raw_start_raises_on_a_mis_sized_or_missing_raw_initval(
    bad, fragment
):
    """
    Given a built parameter whose raw_initval was overwritten with the
      wrong size, or erased,
    When get_raw_start builds the start,
    Then ValueError names the raw key, the parameter and the sizes, instead
      of the start silently becoming raw = 0 for it.
    """
    # ARRANGE
    model, p = _toy_vector_model()
    p.raw_initval = bad

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        System.get_raw_start(_StubSystem([p]), model)
    msg = str(exc.value)
    assert "System.get_raw_start[toy.x_raw] raw_initval" in msg
    assert "'toy.x'" in msg
    assert fragment in msg


# ---------------------------------------------------------------------------
# System.apply_polished_starts (2.3.23)
# ---------------------------------------------------------------------------


def test_apply_polished_starts_adopts_every_seed_on_correct_code():
    """
    Given polished starts for seeds 0 and 2 of the right size, and a
      seed_resolved entry for seed 2,
    When apply_polished_starts adopts them,
    Then seed 0 lands in raw_initval and seed 2 in seed_resolved[2] (the
      non-vacuity check for the raising tests below).
    """
    # ARRANGE
    model, p = _toy_vector_model()
    seed_resolved = [{}, {}, {}]
    stub = _StubSystem([p], seed_resolved=seed_resolved)
    polished = [
        {"toy.x_raw": np.array([0.1, 0.2])},
        {"toy.x_raw": np.array([0.3, 0.4])},
    ]

    # ACT
    System.apply_polished_starts(stub, polished, [0, 2])

    # ASSERT
    np.testing.assert_array_equal(p.raw_initval, [0.1, 0.2])
    phys = p.phys_from_raw(np.array([0.3, 0.4]))
    assert seed_resolved[2] == {
        "toy.0.x": pytest.approx(phys[0]),
        "toy.2.x": pytest.approx(phys[2]),
    }


def test_apply_polished_starts_raises_on_a_key_with_no_parameter():
    """
    Given a polished start whose raw key matches no Parameter,
    When apply_polished_starts adopts it,
    Then KeyError names the key instead of the polish being dropped.
    """
    # ARRANGE
    _, p = _toy_vector_model()
    polished = [{"toy.nope_raw": np.zeros(2)}]

    # ACT / ASSERT
    with pytest.raises(KeyError, match="raw variable 'toy.nope_raw' has no"):
        System.apply_polished_starts(_StubSystem([p]), polished, [0])


def test_apply_polished_starts_raises_on_a_parameter_with_no_raw_transform():
    """
    Given a polished start for a Parameter that has no raw transform,
    When apply_polished_starts adopts it,
    Then ValueError names the key, the seed and the parameter.
    """
    # ARRANGE
    p = Parameter(label="toy.y", initval=1.0, lower=0.0, upper=2.0)
    polished = [{"toy.y_raw": np.zeros(1)}]

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        System.apply_polished_starts(_StubSystem([p]), polished, [0])
    msg = str(exc.value)
    assert "System.apply_polished_starts[toy.y_raw, seed 0]" in msg
    assert "'toy.y' has no raw transform" in msg


@pytest.mark.parametrize("seed_index", [0, 2])
def test_apply_polished_starts_raises_on_a_mis_sized_polished_start(
    seed_index,
):
    """
    Given a polished start of the wrong size, for seed 0 or a later seed,
    When apply_polished_starts adopts it,
    Then ValueError names the key, the seed and both sizes, instead of the
      UNPOLISHED start silently surviving for that parameter.
    """
    # ARRANGE
    _, p = _toy_vector_model()
    raw_before = np.array(p.raw_initval, copy=True)
    seed_resolved = [{}, {}, {}]
    polished = [{"toy.x_raw": np.zeros(2)}] if seed_index else []
    polished.append({"toy.x_raw": np.zeros(3)})
    seed_indices = [0, seed_index] if seed_index else [0]

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        System.apply_polished_starts(
            _StubSystem([p], seed_resolved=seed_resolved),
            polished,
            seed_indices,
        )
    msg = str(exc.value)
    assert f"System.apply_polished_starts[toy.x_raw, seed {seed_index}]" in msg
    assert "'toy.x'" in msg
    assert "raw vector has 3 entries but its raw variable has 2" in msg
    if not seed_index:
        np.testing.assert_array_equal(p.raw_initval, raw_before)


@pytest.mark.parametrize(
    "seed_resolved",
    [None, [{}, {}], [{}, {}, None]],
    ids=["no-seed-list", "short-seed-list", "none-entry"],
)
def test_apply_polished_starts_raises_on_a_seed_with_no_seed_resolved_entry(
    seed_resolved,
):
    """
    Given a polished start for seed 2 and a seed_resolved that is absent,
      too short, or holds None at index 2,
    When apply_polished_starts adopts it,
    Then ValueError names the key and the seed instead of the polished
      seed being discarded.
    """
    # ARRANGE
    _, p = _toy_vector_model()
    polished = [
        {"toy.x_raw": np.zeros(2)},
        {"toy.x_raw": np.array([0.3, 0.4])},
    ]

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        System.apply_polished_starts(
            _StubSystem([p], seed_resolved=seed_resolved), polished, [0, 2]
        )
    msg = str(exc.value)
    assert "System.apply_polished_starts[toy.x_raw, seed 2]" in msg
    assert "has no config_manager.seed_resolved entry" in msg
