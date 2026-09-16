"""The credible-interval convention every report is written in.

One setting, read by every consumer that summarizes a posterior: the LaTeX
table, ``<prefix>_results.csv``, the mode report, the corner plots and the
table caption.  It is set once per run from the config's ``reporting:`` block
and is a RUN-level choice, deliberately not a per-component or per-parameter
one -- a table whose rows carried different interval widths would be
unreadable, and its caption could not describe it.

WHY THIS EXISTS
---------------

The historical default -- median plus a 68.27% (1-sigma) interval -- is an
ASTRONOMY convention.  It was hardcoded at four sites and was correct for
every component shipped until now, because every component shipped until now
was astronomy.  It is not universal:

* pharmacometrics reports 95% (NONMEM, Monolix, Stan/Torsten, nlmixr2), and a
  regulated bioequivalence analysis reports a 90% interval whose bounds are
  written into FDA/EMA guidance;
* clinical and epidemiological work is 95% essentially without exception.

Stating the width in the caption does not rescue an unidiomatic number.  A
reader whose field has exactly one convention does not check the caption; they
read the interval they expect, and a 68% interval read as a 95% one understates
the uncertainty by about a factor of two.  So a code that emits a result for a
field reports it in that field's convention, and this module is how.

PROVENANCE OF THE NON-ASTRONOMY CLAIMS ABOVE
--------------------------------------------

The statements about pharmacometrics, clinical and epidemiological practice
were written by an astrophysicist and an LLM.  NO domain expert has reviewed them.
They are the reason this setting exists and they are believed accurate, but
they are secondary knowledge, not expertise: treat a specific claim (which
regulator asks for which width, what a given field's default is) as a starting
point to verify, not as settled.

The MECHANISM does not depend on any of it.  This module says only "the width
is a setting with a default of 1 sigma"; which width a given analysis should
use is the user's call, and getting the motivating anecdote wrong would not
make the setting wrong.

NOT A KNOB, AND THEN A KNOB
---------------------------

``Parameter.compute_summary`` used to carry an ``nsigma`` argument that nothing
read, so ``compute_summary(nsigma=2)`` silently returned 1 sigma.  Deleting it
was right, and the docstring that replaced it ("the interval width is not a
knob") described the code accurately at the time.  This module is not that
argument restored: the width now reaches ALL of the consumers or none of them,
and a cached summary knows which width it was computed at (see
``Parameter.ensure_summary``), so the failure mode that made the old argument a
lie -- a caller asking for a width and being handed another -- cannot recur.

WHY A MODULE-LEVEL SETTING
--------------------------

The alternative is threading the width through every consumer.  Two of them
(``Parameter._summarize_array`` and ``compute_summary``) are reached lazily,
from three call sites that each say "compute this if it is missing", and one
(``corner_utils``) has no System in scope at all.  Threading would put the
width in five signatures that otherwise have no interest in it.

The cost of a module-level setting is staleness, and that cost is paid rather
than ignored: ``Parameter`` records the width each cached summary was computed
at and recomputes when it no longer matches.  That is the same shape as the
rule in ``parameter.md`` that a summary belongs to the draws it came from --
now also to the width it was computed at.
"""

import math
from contextlib import contextmanager

from .constants import SIGMA_1

# The astronomy convention, and the default: a 68.27% interval, i.e. +/- 1
# Gaussian sigma.  Taken from the ONE definition of SIGMA_1 in constants.py
# (review 4.2.6) rather than respelled here, so the default quantiles are
# EXACTLY the ones the tables have always reported.
DEFAULT_CREDIBLE_INTERVAL = SIGMA_1

# THE TWO FIELDS DO NOT JUST USE DIFFERENT WIDTHS, THEY USE DIFFERENT KINDS OF
# NUMBER, and conflating them is the subtler of the two errors this module
# exists to prevent.
#
# Astronomy and physics quote SIGMA MULTIPLES: the width is erf(n/sqrt(2)) for
# integer n, an irrational number conventionally glossed as 68%, 95%, 99.7%.
# The gloss is a rounding of the real quantity, and the real quantity is
# "n sigma".
#
# Medicine, pharmacometrics and regulatory statistics quote EXACT ROUND
# PROBABILITIES.  A 95% interval there is exactly 0.95 -- it descends from the
# 0.05 significance level, and its normal quantile is 1.959964, NOT 2.  It is
# not 2 sigma and was never meant to be: 2 sigma is 0.954500, which differs by
# 0.45 percentage points.  Bioequivalence's 90% is exact for the same reason
# (it pairs with two one-sided 5% tests).
#
# So `credible_interval: 0.95` means the medical 95%, and a user who wants
# 2 sigma writes 0.9545 (or SIGMA_WIDTHS[2]).  `label` renders a recognized
# sigma multiple with its CONVENTIONAL gloss and anything else at its own
# precision, and `caption_phrase` appends the "(n-sigma)" qualifier only for
# the former -- so the caption distinguishes the two 95%s that are not the
# same number.
SIGMA_WIDTHS = {n: math.erf(n / math.sqrt(2.0)) for n in (1, 2, 3)}

# How each sigma multiple is conventionally written as a percentage.  A TABLE
# rather than a rounding rule: the field writes 68, 95 and 99.7, which is
# neither a fixed number of decimals nor a fixed number of significant
# figures, and inventing a rule that happened to reproduce three values would
# be a rule nobody could check.  A wider gloss is also gratuitous churn -- the
# caption said 68 percent for years, and 68.3 is not more correct than 68,
# only one digit more precise about a number whose exact value the
# "(1-sigma)" qualifier already carries.
SIGMA_GLOSS = {1: "68", 2: "95", 3: "99.7"}

# How close a width must be to a sigma multiple to be rendered as one.  Loose
# enough that a user typing 0.6827 or 0.9545 is recognized, tight enough that
# the medical 0.95 is NOT mistaken for 2 sigma (they differ by 4.5e-3).
SIGMA_MATCH_TOL = 5e-4

# The config key, named once so the error messages and the schema cannot drift
# from the reader.
CONFIG_BLOCK = "reporting"
CONFIG_KEY = "credible_interval"

_credible_interval = DEFAULT_CREDIBLE_INTERVAL


def validate_credible_interval(value, where=None):
    """Return ``value`` as a float in (0, 1), or raise naming ``where``.

    The percent spelling is the error worth catching by hand: ``95`` is a far
    more natural thing to type than ``0.95``, it is off by a factor of 100
    rather than being nonsense, and left alone it would sail into
    ``np.nanquantile`` and raise there naming neither the key nor the file.
    """
    where = where or f"{CONFIG_BLOCK}.{CONFIG_KEY}"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"[{where}] must be a number strictly between 0 and 1 (a "
            f"probability, not a percentage): 0.95 for a 95% interval, "
            f"0.6827 for the 1-sigma default.  Got {value!r}."
        )

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"[{where}] must be a finite number strictly between 0 and 1; "
            f"got {value!r}."
        )
    if not 0.0 < value < 1.0:
        hint = ""
        if 1.0 < value < 100.0:
            hint = (
                f"  It looks like a percentage -- write it as a probability, "
                f"{value / 100.0:g}."
            )
        raise ValueError(
            f"[{where}] must be strictly between 0 and 1; got {value!r}.{hint}"
        )
    return value


def get_credible_interval():
    """The interval width every report currently uses, as a probability."""
    return _credible_interval


def set_credible_interval(value):
    """Set the reporting interval width.  Returns the PREVIOUS value.

    The previous value is returned so a caller that changes it temporarily can
    put it back without reaching for the module global itself; the
    ``use_credible_interval`` context manager below is the usual way.
    """
    global _credible_interval
    previous = _credible_interval
    _credible_interval = validate_credible_interval(value)
    return previous


@contextmanager
def use_credible_interval(value):
    """Temporarily report at ``value``.  Restores on exit, exceptions included."""
    previous = set_credible_interval(value)
    try:
        yield get_credible_interval()
    finally:
        set_credible_interval(previous)


def configure_from(block):
    """Apply a config's ``reporting:`` BLOCK.  Returns the resulting width.

    Takes the block rather than the whole config so the ``"reporting"`` key is
    read at the call site, off the top-level config dict, where
    ``tests/test_known_keys.py``'s AST scan can see it -- a key looked up
    through a module constant is invisible to that scan and would be reported
    as dead vocabulary.

    An absent block, or a block that does not name the key, leaves the default
    in place -- so every existing config keeps reporting exactly what it always
    has.  A block that IS present is validated, because a width that silently
    fell back to 1 sigma is the failure the old ``nsigma`` argument had.
    """
    block = block or {}
    if not isinstance(block, dict):
        raise ValueError(
            f"[{CONFIG_BLOCK}] must be a block of settings, e.g.\n"
            f"  {CONFIG_BLOCK}:\n    {CONFIG_KEY}: 0.95\n"
            f"Got {block!r}."
        )
    if CONFIG_KEY not in block:
        return get_credible_interval()
    set_credible_interval(block[CONFIG_KEY])
    return get_credible_interval()


def reset():
    """Restore the default width.  For tests and for a fresh in-process run."""
    return set_credible_interval(DEFAULT_CREDIBLE_INTERVAL)


def quantiles(value=None):
    """The ``(low, high)`` quantiles bracketing the interval, centred on the median.

    A CENTRAL (equal-tailed) interval, not a highest-density one: it is what
    the tables have always reported, it is what a median belongs with, and it
    is what both conventions this module exists to serve actually mean.
    """
    p = (
        get_credible_interval()
        if value is None
        else validate_credible_interval(value)
    )
    return 0.5 - p / 2.0, 0.5 + p / 2.0


def label(value=None):
    """The width as a percentage string for a caption, e.g. ``"68"``, ``"95"``.

    A recognized sigma multiple gets its conventional gloss (``SIGMA_GLOSS``),
    so the 1-sigma default reads ``"68"`` exactly as this caption has always
    read.  Any other width is rendered at its own precision with trailing
    zeros trimmed: 0.95 is ``"95"`` and 0.9 is ``"90"``, and those ARE exactly
    95% and 90%, so rounding them would be wrong rather than conventional.
    """
    p = (
        get_credible_interval()
        if value is None
        else validate_credible_interval(value)
    )
    n = sigma_multiple(p)
    if n is not None:
        return SIGMA_GLOSS[n]
    return f"{100.0 * p:.2f}".rstrip("0").rstrip(".")


def sigma_multiple(value=None):
    """``n`` when the width is n-sigma within ``SIGMA_MATCH_TOL``, else ``None``.

    The caption says "n-sigma" only for the widths that word belongs to; for
    any other width it would be wrong, and silently so.  In particular the
    medical 0.95 returns None rather than 2 -- see the note by
    ``SIGMA_WIDTHS``.
    """
    p = (
        get_credible_interval()
        if value is None
        else validate_credible_interval(value)
    )
    for n, width in SIGMA_WIDTHS.items():
        if abs(p - width) < SIGMA_MATCH_TOL:
            return n
    return None


def caption_phrase(value=None, capitalized=False):
    """The noun phrase a table caption uses for this interval.

    One implementation, so the LaTeX caption and any other reader cannot
    disagree about what the numbers in the table mean.  ``capitalized`` upper-
    cases the first letter only -- ``str.capitalize`` would lower-case the
    rest, which is wrong the moment a phrase carries a LaTeX macro or a proper
    noun.
    """
    pct = label(value)
    n = sigma_multiple(value)
    if n is not None:
        phrase = rf"median and {pct}\% ({n}-$\sigma$) credible intervals"
    else:
        phrase = rf"median and {pct}\% credible intervals"
    return phrase[0].upper() + phrase[1:] if capitalized else phrase
