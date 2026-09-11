# DC2018 sweep evaluation: what counts as "the most sensible result"

The sweep's headline number must answer one question -- *did we produce the
most sensible result?* -- and that is not the same as "did we recover the
truth". A fit can recover truth and still be wrong (it reported the wrong
solution as favoured), and it can fail to recover truth and still be right
(the data did not support it, and we said so).

Ruled by JDE 2026-09-11. Classes (a)-(d) are his; (e), (f) and (h) were
added after five events exposed outcomes with no home, and he ruled on how
each scores.

## Tier 1 -- one number

    PASS / TOTAL

A PASS means the report was *defensible given the data*. Everything else is
a FAIL, including the soft failures below: a pipeline that knows it failed
is better than one that does not, but it still did not deliver a result.

## Tier 2 -- the breakdown, and whose fault it is

### (a) TRUTH_RECOVERED
Truth lies within the stated uncertainties **and** the interval is
informative (see UNINFORMATIVE under (b) -- an interval as wide as the prior
does not count as recovery).

* **PASS** -- the mode weights are right: we found the evidence in *all*
  modes and named the correct favourite. The correct favourite is decided
  **by evidence, not by the truth table**: if the data genuinely prefer a
  non-truth solution, reporting it with honest weights is a PASS.
* **FAIL** -- the weights are wrong, or the wrong solution is reported as
  favoured.

### (b) TRUTH_UNRECOVERABLE
The data could not support the truth: too low SNR, no detectable anomaly, or
the posterior is simply the prior. **UNINFORMATIVE folds in here** -- an
interval that contains truth only because it contains everything is a
non-detection wearing a measurement's clothes.

* **PASS** -- an accurate statement of what *is* detectable. Both of these
  qualify: an immediate report that no anomaly was detected, and a returned
  uninformative prior.
* **FAIL** -- a confidently wrong claim.

### (c) TRUTH_NOT_RECOVERED
The truth was detectable and we did not detect it. **Always a FAIL**, and
the hardest one -- the data were there.

### (d) WRONG_ARCHITECTURE
We never fitted the architecture the event actually has.

* **PASS** -- the wrong architecture genuinely has higher evidence, *and* we
  surfaced the correct one as a candidate mode.
* **FAIL** -- we never considered it at all.

### (e) ABSTAINED  -- soft
The pipeline declined to answer: non-convergence, too many numerically
invalid draws, a refused mode report. **FAIL** at tier 1, qualified as soft
here, because declining beats emitting a confident wrong answer -- which
would have been a (b) FAIL.
**Sweep action: automatically trigger a longer run.** This is the one class
with a mechanical remedy.

### (f) INFRASTRUCTURE_FAILURE -- soft
Our code broke: a crash, a seeding failure, a parser bug. Same "wrong but we
know it" family as (e) and deliberately kept separate, so an afternoon of
bad plumbing is never scored as a physics result.
**Sweep action: fix and rerun.** Reported outside the science denominator as
well as inside the tier-1 count.

### (h) DEGENERATE_COUNTERPART -- soft
We found and reported a known-degenerate image of truth -- close/wide
s <-> 1/s, or the +/-u_0 pair -- as though it were unique. **FAIL**, softly:
the physics is a real solution, the error is presenting one of a pair as the
answer. Reporting *both* with sane weights is not this class; it is (a).

## What the classifier needs that we do not yet have

**A detectability discriminator.** (b) and (c) differ only by whether the
truth was *detectable*, and nothing in a fit's own output answers that. The
cheap discriminator is one extra **logp evaluation at truth** (not a fit):

* truth's logp comparable to or better than the favoured solution's, yet
  truth's mode absent from the posterior -> **(c)**, a search failure.
* truth's logp far worse -> **(b)**, the data really do disfavour it.

That costs one model build per event rather than a second fit. It relies on
injected *sampled* parameters round-tripping, which they do (review 2.3.17:
49 of 52 took exactly; only *derived* parameters are silently overridden --
so the injected set must be the sampled observables).

**Per-mode evidence.** (a)-PASS is defined on evidence, and the current mode
report gives occupancy weights whose error bars are understated 8-23x
(review 1.11.3), and on DC2018-128 refused evidence weighting outright. So
(a)-PASS is **not yet measurable**; until it is, a recovered truth with
plausible weights should be reported as (a) with the weight check marked
UNVERIFIED rather than silently passed.

**Architecture selection.** Nothing fits more than one architecture, so
every non-2L1S event is structurally (d)-FAIL. That is expected and is the
point of tracking it: the tier-2 line for (d) is the size of the prize for
building escalation.
