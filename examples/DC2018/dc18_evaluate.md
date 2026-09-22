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

### How "truth within the stated uncertainties" is judged (JDE 2026-09-22)

Not by a tolerance table. Until 2026-09-22 the classifier asked for one
draw inside FIXED tolerances (t_0 0.05 d, u_0 0.005, log q 0.030, ...) in
every observable at once, and on a peak that fell in a season gap the
honest posterior -- 3.6 d wide in t_0, 0.5 in u_0 -- had no such draw, so a
posterior that CONTAINED the truth at 0.1-2 sigma scored TRUTH_NOT_RECOVERED,
the same class as a wrong-topology fit 1,356 tolerances away (experiment A
vs the blind fit on 226, review 2.4.14). The rule is now:

1. **Dispersion gate first** (class (g) below). Is the posterior's own
   uncertainty believable? Per mode, the drop from the best lp to the median
   lp is chi2_k/2 for a k-dimensional near-Gaussian mode; its z-score
   against k/2 says whether the mode is under-dispersed (stuck chains) or
   over-dispersed (unconverged, hot-contaminated). Internal to the fit, no
   truth needed. A failure here is the PRIMARY failure and nothing
   downstream is read.
2. **Mode-aware pull.** The truth's pull against the width of the mode it
   is nearest, per observable, judged jointly as a chi2 p-value at 3 sigma.
   The truth "is in" that mode when the joint pull is not rejected. A
   30-day-wide t_0 that contains the truth passes exactly as a 0.05-day one
   does: a wide honest interval is a correct statement, not a failure.
3. **Weights on evidence.** The truth in a minor mode is a clean (a) PASS
   when the reported weights are trustworthy -- bridge evidence, or an
   occupancy the mode report itself validated -- and favour another basin:
   the data+prior topology is being reported honestly and the truth basin
   is identified within its uncertainties. Only an explanation is owed.
   It is (i) WRONG_BASIN when the weights favour another mode although the
   evidence lies in the truth's basin. With UNTRUSTED weights the best lp
   of the run is the only basin arbiter available: the truth's mode holding
   it while occupancy favours another mode is WRONG_BASIN; the favourite
   holding it is a provisional (a) PASS with the weight check UNVERIFIED.
4. **Information, not a threshold.** Every core observable reports its
   posterior sd over its prior sd (the prior read from the fit's own
   resolved bounds). About 1 means the prior came back; the event folds
   into (b) when every core observable did, and the sweep says so. Anything
   smaller is sculpting and is worth having, however wide.
5. **Coverage.** Across the sweep the pulls themselves are the dispersion
   test: honest posteriors give standard-normal pulls, and the summary
   reports the fraction inside 1/2/3 sigma against 68/95/99.7, over gated
   events and informative parameters only (a prior-width posterior always
   contains the truth and would inflate it).

### (a) TRUTH_RECOVERED
Truth lies within the stated uncertainties of one identified mode (rule 2)
**and** the posterior is informative (rule 4).

* **PASS** -- the truth's mode is the favourite, or the weights are
  trustworthy and favour another basin on the evidence (rule 3). The
  correct favourite is decided **by evidence, not by the truth table**.
* **FAIL** -- see (i) WRONG_BASIN.

### (b) TRUTH_UNRECOVERABLE
The data could not support the truth: too low SNR, no detectable anomaly, or
the posterior is simply the prior. **The prior coming back folds in here**
(rule 4) -- an interval that contains truth only because it contains
everything is a non-detection wearing a measurement's clothes, and saying
so is the correct report.

* **PASS** -- an accurate statement of what *is* detectable. Both of these
  qualify: an immediate report that no anomaly was detected, and a returned
  prior.
* **FAIL** -- a confidently wrong claim.

### (c) TRUTH_NOT_RECOVERED
The truth was detectable and we did not detect it: it lies in no mode by
rule 2, and the detectability test says the data supported it. **Always a
FAIL**, and the hardest one -- the data were there. The detail line carries
the truth's worst pull against its nearest mode, so a near miss and a
wrong topology are never the same number.

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

### (g) UNRELIABLE_POSTERIOR -- soft
The dispersion gate (rule 1) failed: a mode's lp drop is far from its
chi2_k/2 expectation, or more than a tenth of the draws are numerically
invalid. **FAIL**, softly, and reported as the primary failure -- the pulls
of an under- or over-dispersed posterior mean nothing, so none are judged.
**Sweep action: longer run or ladder tuning.**

### (i) WRONG_BASIN
The truth is recovered inside a minor mode, but the reported weights favour
another basin against the evidence (rule 3). **FAIL**, hard: the physics
was found and the report points the reader away from it.

### (h) DEGENERATE_COUNTERPART -- retired 2026-09-22
A close/wide or +/-u_0 partner reported as favoured is now either an (a)
PASS with an explanation (trusted weights genuinely prefer the partner) or
(i) WRONG_BASIN (they do not). Presenting one image of a pair as unique is
a mode-report question, not a classification.

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
(review 1.11.3), and on DC2018-128 and -226 refused evidence weighting
outright (bridge iteration did not converge). Until the bridge converges
routinely, the weight check reads the report's own provenance line: it
trusts "evidence" and "occupancy (validated ...)", and otherwise falls back
to the best-lp arbiter of rule 3 with the check marked UNVERIFIED.

**Architecture selection.** Nothing fits more than one architecture, so
every non-2L1S event is structurally (d)-FAIL. That is expected and is the
point of tracking it: the tier-2 line for (d) is the size of the prize for
building escalation.
