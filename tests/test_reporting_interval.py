"""The reporting credible-interval width (``exozippy.reporting``).

Median + a 68.27% (1-sigma) interval is an ASTRONOMY convention.  It was
hardcoded at four sites and was right for every component shipped until now,
because every component shipped until now was astronomy.  A component from a
field whose universal convention is 95% (pharmacometrics, epidemiology,
anything clinical) cannot be reported at 68% and merely annotated: a reader
whose field has one convention does not check the caption, so the number is
read as 95% and the uncertainty is understated by about a factor of two.

The two properties that matter, and that these tests pin:

* the DEFAULT is bit-unchanged, so no shipped astronomy config moves;
* a cached summary belongs to the WIDTH it was computed at as well as to the
  draws it came from, so a re-report at a second width cannot publish the
  first width's intervals under the second width's caption.
"""

import math

import numpy as np
import pytest

from exozippy import evaluator as evaluator_mod
from exozippy import reporting
from exozippy.components.parameter import Parameter
from exozippy.system import RESERVED_CONFIG_KEYS

# The literals the code carried before reporting.py existed.  Spelled out here
# rather than imported so this file would still catch a change to
# constants.SIGMA_1 itself -- importing the constant under test is how a
# regression pin becomes a tautology.
LEGACY_SIGMA_1 = math.erf(1.0 / math.sqrt(2.0))
LEGACY_LOW = 0.5 - LEGACY_SIGMA_1 / 2.0
LEGACY_HIGH = 0.5 + LEGACY_SIGMA_1 / 2.0


@pytest.fixture(autouse=True)
def _restore_default_width():
    """The width is process-wide, so a test that changes it must put it back.

    Autouse and unconditional: a leaked 95% setting would not fail the test
    that leaked it, it would fail some unrelated table test later in the same
    worker, which is the expensive way to find out.
    """
    yield
    reporting.reset()


@pytest.fixture
def normal_draws():
    """200k draws from N(10, 2), enough that a quantile is good to ~0.01."""
    return np.random.default_rng(0).normal(10.0, 2.0, 200_000)


# ---------------------------------------------------------------------------
# The default does not move
# ---------------------------------------------------------------------------


def test_default_width_is_the_historical_one_sigma():
    """Given a fresh process, When nothing configures a width, Then it is 1 sigma.

    The regression pin for every shipped astronomy config: the default
    quantiles must be EXACTLY the two numbers the tables reported before the
    setting existed, not merely close to them.
    """
    assert reporting.get_credible_interval() == LEGACY_SIGMA_1

    low, high = reporting.quantiles()
    assert low == LEGACY_LOW
    assert high == LEGACY_HIGH


def test_default_summary_is_bit_identical_to_the_legacy_quantiles(
    normal_draws,
):
    """Given draws, When summarized at the default, Then it matches nanquantile.

    Pins the NUMBERS rather than the code path: the summary must equal a
    hand-computed median and the two legacy quantiles to the last bit.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws
    summary = p.ensure_summary()

    med = float(np.nanquantile(normal_draws, 0.5))
    assert summary.median == med
    assert summary.err_minus == med - float(
        np.nanquantile(normal_draws, LEGACY_LOW)
    )
    assert (
        summary.err_plus
        == float(np.nanquantile(normal_draws, LEGACY_HIGH)) - med
    )


# ---------------------------------------------------------------------------
# A width that is set is the width that is reported
# ---------------------------------------------------------------------------


def test_ninety_five_percent_reports_1p96_sigma(normal_draws):
    """Given N(10, 2) draws, When the width is 0.95, Then the interval is +/-1.96 sigma.

    The arithmetic, not the path: 0.95 must produce the 2.5%/97.5% quantiles,
    which for a normal are 1.96 sigma either side -- 3.92 for sigma = 2.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws

    with reporting.use_credible_interval(0.95):
        summary = p.ensure_summary()

    assert summary.median == pytest.approx(10.0, abs=0.02)
    assert summary.err_minus == pytest.approx(1.96 * 2.0, abs=0.03)
    assert summary.err_plus == pytest.approx(1.96 * 2.0, abs=0.03)


def test_quantiles_are_equal_tailed_about_the_median():
    """Given any width, When quantiles are taken, Then they are central, not HDI.

    A median belongs with an equal-tailed interval; the two tails must carry
    the same mass and the pair must bracket exactly the requested probability.
    """
    for width in (0.5, LEGACY_SIGMA_1, 0.9, 0.95, 0.99):
        low, high = reporting.quantiles(width)
        assert high - low == pytest.approx(width)
        assert low + high == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A cached summary belongs to the width it was computed at
# ---------------------------------------------------------------------------


def test_cached_summary_recomputes_when_the_width_changes(normal_draws):
    """Given a summary cached at 68%, When the width becomes 95%, Then it recomputes.

    This is the staleness the module exists to prevent.  Without it a second
    report off one live System -- exozippy-modes, a GUI re-solve, any script
    that fits and then re-reports -- would publish the FIRST width's intervals
    under the SECOND width's caption, and an interval is plausible at any
    width, so nothing downstream could notice.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws

    narrow = p.ensure_summary().err_plus
    assert p.summary_is_current()

    with reporting.use_credible_interval(0.95):
        assert not p.summary_is_current(), "stale-width cache reported current"
        wide = p.ensure_summary().err_plus
        assert p.summary_is_current()

    assert wide > narrow
    # ...and back the other way: restoring the width restores the summary.
    assert not p.summary_is_current()
    assert p.ensure_summary().err_plus == narrow


def test_mode_summaries_recompute_when_the_width_changes(normal_draws):
    """Given cached per-mode summaries, When the width changes, Then they are stale.

    The per-mode cache is checked on TWO properties -- the mode count (review
    2.11.3) and now the width.  Both must be able to invalidate it alone, so
    this pins the width half at a FIXED mode count, where the length check
    cannot be what fires.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws
    labels = np.zeros(normal_draws.size, dtype=int)
    labels[normal_draws.size // 2 :] = 1

    p.compute_mode_summaries(labels, 2)
    assert p.mode_summaries_are_current(2)
    # The length guard still works and is independent of the width one.
    assert not p.mode_summaries_are_current(3)

    with reporting.use_credible_interval(0.95):
        assert not p.mode_summaries_are_current(2)


def test_the_two_caches_carry_independent_width_stamps(normal_draws):
    """Given a summary at 68%, When only the MODE summaries are rebuilt at 95%,
    Then the scalar summary is still reported stale.

    A single shared stamp passes every other test in this file and is still
    wrong: ``compute_mode_summaries`` would overwrite it with the new width
    and thereby mark a ``summary`` computed at the OLD width current.  No
    shipped call site reaches that ordering today -- ``build_csv_output``
    refreshes the scalar summary first -- so this pins the property rather
    than a live bug, and it is the test that fails if the two stamps are ever
    merged back into one.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws
    p.ensure_summary()

    labels = np.zeros(normal_draws.size, dtype=int)
    with reporting.use_credible_interval(0.95):
        p.compute_mode_summaries(labels, 1)

        assert p.mode_summaries_are_current(1)
        assert not p.summary_is_current(), (
            "the scalar summary was computed at 68% and must not be reported "
            "current merely because the mode summaries were rebuilt at 95%"
        )


def test_new_draws_still_clear_the_width_stamp(normal_draws):
    """Given a cached summary, When new draws arrive, Then the stamp clears too.

    The pre-existing invariant (a summary belongs to its draws, review 3.14.7)
    must keep holding: the width stamp is part of that cache and has to be
    dropped with it, or the next ensure_summary would see a matching width and
    a summary computed from the OLD posterior.
    """
    p = Parameter(label="x")
    p.posterior = normal_draws
    p.ensure_summary()

    p.posterior = normal_draws + 5.0

    assert p.summary is None
    assert p.mode_summaries is None
    assert not p.summary_is_current()
    assert not p.mode_summaries_are_current(1)
    assert p.ensure_summary().median == pytest.approx(15.0, abs=0.02)


def test_ensure_summary_is_a_noop_without_draws():
    """Given a parameter with no posterior, When asked, Then it stays None.

    The normal state of a fixed element and of every parameter before the fit;
    it must not raise and must not fabricate a summary.
    """
    p = Parameter(label="x")
    assert p.posterior is None
    assert p.ensure_summary() is None


# ---------------------------------------------------------------------------
# The caption cannot disagree with the numbers
# ---------------------------------------------------------------------------


def test_caption_percentage_is_exact_for_a_round_probability():
    """Given a round-probability width, When the caption is built, Then it is exact.

    0.95 IS exactly 95% and 0.9 exactly 90%, so these must be rendered
    exactly; rounding them would be wrong rather than conventional.  A
    property, not a literal: a caption asserting a fixed string would keep
    passing if the width plumbing broke, which is the exact failure the
    caption is supposed to make visible.
    """
    for width in (0.5, 0.9, 0.95, 0.99):
        with reporting.use_credible_interval(width):
            assert float(reporting.label()) == pytest.approx(100.0 * width)
            assert reporting.label() in reporting.caption_phrase()


def test_default_caption_still_says_68_not_68_point_3():
    """Given the 1-sigma default, When the caption is built, Then it reads "68".

    68.3 is not more correct than 68 -- it is one digit more precise about a
    number whose exact value the "(1-$\\sigma$)" qualifier already carries, and
    the caption has said 68 for years.  Widening the gloss would be gratuitous
    churn in the output of every existing astronomy run, so this pins it.
    """
    assert reporting.label() == "68"
    assert reporting.caption_phrase() == (
        r"median and 68\% (1-$\sigma$) credible intervals"
    )


@pytest.mark.parametrize("n, gloss", [(1, "68"), (2, "95"), (3, "99.7")])
def test_sigma_multiples_use_their_conventional_gloss(n, gloss):
    """Given an n-sigma width, When labelled, Then it reads as the field writes it.

    68 / 95 / 99.7 is neither a fixed decimal count nor a fixed significant
    figure count, which is why SIGMA_GLOSS is a table and not a rounding rule.
    """
    width = reporting.SIGMA_WIDTHS[n]
    assert reporting.sigma_multiple(width) == n
    assert reporting.label(width) == gloss
    assert rf"({n}-$\sigma$)" in reporting.caption_phrase(width)


def test_the_medical_95_percent_is_not_two_sigma():
    """Given 0.95, When asked for a sigma multiple, Then it is None, not 2.

    The distinction this module turns on.  The medical/regulatory 95% is an
    EXACT probability descending from the 0.05 significance level (normal
    quantile 1.959964); 2 sigma is 0.954500 (quantile 2).  They differ by 0.45
    percentage points, so labelling one as the other would put a false
    "(2-$\\sigma$)" on a pharmacometrics table -- and a false "95%" on an
    astronomy one.
    """
    assert reporting.sigma_multiple(0.95) is None
    assert reporting.sigma_multiple(reporting.SIGMA_WIDTHS[2]) == 2
    assert reporting.SIGMA_WIDTHS[2] == pytest.approx(0.9544997, abs=1e-6)

    # Both render "95"; only the sigma one carries the qualifier that says so.
    assert reporting.label(0.95) == reporting.label(reporting.SIGMA_WIDTHS[2])
    assert r"$\sigma$" not in reporting.caption_phrase(0.95)
    assert r"(2-$\sigma$)" in reporting.caption_phrase(
        reporting.SIGMA_WIDTHS[2]
    )


def test_caption_says_n_sigma_only_for_a_sigma_width():
    """Given a non-sigma width, When the caption is built, Then it omits the qualifier.

    "1-$\\sigma$" is true of 68.27% and false of everything else; carrying it
    to a 95% table would be a wrong statement in a published caption.
    """
    assert r"1-$\sigma$" in reporting.caption_phrase()
    with reporting.use_credible_interval(0.95):
        assert r"$\sigma$" not in reporting.caption_phrase()


def test_caption_capitalization_leaves_the_rest_of_the_string_alone():
    """Given a phrase with LaTeX in it, When capitalized, Then only char 0 changes.

    ``str.capitalize`` lower-cases the remainder, which would silently rewrite
    a macro name or a proper noun the moment one entered the phrase.
    """
    plain = reporting.caption_phrase()
    capped = reporting.caption_phrase(capitalized=True)
    assert capped[0] == plain[0].upper()
    assert capped[1:] == plain[1:]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_percentage_spelling_raises_and_says_how_to_write_it():
    """Given `credible_interval: 95`, When validated, Then it raises with the fix.

    95 is a far more natural thing to type than 0.95 and is off by exactly
    100x, so it must not reach np.nanquantile -- which would raise naming
    neither the key nor the file.
    """
    with pytest.raises(ValueError) as exc:
        reporting.validate_credible_interval(95)
    message = str(exc.value)
    assert "reporting.credible_interval" in message
    assert "0.95" in message


@pytest.mark.parametrize(
    "bad", [0.0, 1.0, -0.5, 1.5, float("nan"), "0.95", None, True]
)
def test_invalid_widths_raise(bad):
    """Given a value outside (0, 1) or not a number, When validated, Then it raises.

    ``True`` is in the list deliberately: bool is an int subclass, so an
    unguarded numeric check would accept it as 1.0.
    """
    with pytest.raises(ValueError):
        reporting.validate_credible_interval(bad)


def test_absent_block_and_absent_key_leave_the_default():
    """Given no reporting block, When configured, Then the default is untouched.

    Every config that predates this feature must keep reporting exactly what
    it always has.
    """
    assert reporting.configure_from(None) == LEGACY_SIGMA_1
    assert reporting.configure_from({}) == LEGACY_SIGMA_1
    assert reporting.configure_from({"something_else": 1}) == LEGACY_SIGMA_1


def test_configure_from_applies_the_key():
    """Given a block naming the key, When configured, Then the width is set."""
    assert reporting.configure_from({"credible_interval": 0.95}) == 0.95
    assert reporting.get_credible_interval() == 0.95


def test_non_dict_block_raises():
    """Given `reporting: 0.95`, When configured, Then it raises showing the block."""
    with pytest.raises(ValueError) as exc:
        reporting.configure_from(0.95)
    assert "credible_interval" in str(exc.value)


def test_reporting_is_a_reserved_key():
    """Given the top-level vocabulary, Then `reporting` is in it.

    Otherwise System warns that the block "will be ignored" while run.py
    honors it -- the false-warning failure tests/test_known_keys.py exists to
    prevent.
    """
    assert "reporting" in RESERVED_CONFIG_KEYS


def test_reporting_does_not_stale_a_trace():
    """Given two configs differing only in width, Then the structural hash matches.

    Re-reporting an existing trace at a different width is the whole point of
    the setting, so it must not invalidate the trace that would be re-read.
    """
    base = {"star": [{"name": "A"}], "prefix": "out/x"}
    wide = dict(base, reporting={"credible_interval": 0.95})

    assert evaluator_mod.structural_hash(
        base
    ) == evaluator_mod.structural_hash(wide)


def test_corner_quantiles_follow_the_width():
    """Given a width, When the corner triple is built, Then it is [low, 0.5, high].

    The corner plot and the table must not be able to disagree: a run at 95%
    showing 68% contours beside a 95% table is worse than either alone.
    """
    from exozippy.corner_utils import _quantile_triple

    assert _quantile_triple() == [LEGACY_LOW, 0.5, LEGACY_HIGH]
    with reporting.use_credible_interval(0.95):
        low, med, high = _quantile_triple()
        assert med == 0.5
        assert (low, high) == pytest.approx((0.025, 0.975))


def test_use_credible_interval_restores_on_exception():
    """Given an exception inside the context, When it propagates, Then the width resets.

    A leaked width would corrupt every later report in the process, and the
    test that leaked it would still be green.
    """
    with pytest.raises(RuntimeError):
        with reporting.use_credible_interval(0.95):
            raise RuntimeError("boom")
    assert reporting.get_credible_interval() == LEGACY_SIGMA_1
