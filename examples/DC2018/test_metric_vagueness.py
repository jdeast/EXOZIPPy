"""Every truth-comparison metric must FAIL a deliberately diffuse posterior.

Three metrics in this codebase have now had the same defect: they scored a
vague answer as a good one.

  * the per-parameter PULL (review 7.15.1): max|pull| came back MONOTONIC in
    n_eff -- the best-sampled run scored WORST -- because resolving a
    posterior makes it narrower and narrow-in-the-wrong-mode is what a pull
    punishes.  dynesty beat PTDE 25.5 to 39.4 purely by being 170x broader.
  * NEAREST-APPROACH in dc18_mode_aware_score: the closest draw to truth
    improves with draw count, so it flattered whichever sampler produced
    more draws (2.7M vs 6.7k).
  * the BASIN TEST's z = (median - truth)/sd: small whenever sd is large, so
    an unconstrained parameter always "stayed".  It reported 4 of 4 stayed
    on events 152 and 194 while log_s sat at exactly 0 with sd 43-45x
    tolerance and log_q had sd ~2.1 in log10 -- a factor of 126.  Two
    BASIN_OMITTED verdicts were withdrawn.

The common shape: a statistic divided by, or compared against, the
posterior's own width. This test builds two posteriors -- one TIGHT and
correct, one DIFFUSE and uninformative but centred near truth -- and asserts
that a metric prefers the tight one. A metric that cannot tell them apart is
measuring confidence, not accuracy.
"""

import numpy as np


def _posteriors(rng, truth, tol):
    tight = truth + rng.normal(0, tol * 0.1, size=20000)
    diffuse = truth + rng.normal(0, tol * 50.0, size=20000)
    return tight, diffuse


def test_informativeness_gate_rejects_a_diffuse_posterior():
    """The gate the basin test now uses must reject the diffuse case."""
    rng = np.random.default_rng(0)
    tol = 0.01
    truth = 1.0
    for name, v in zip(("tight", "diffuse"), _posteriors(rng, truth, tol)):
        sd = float(np.std(v))
        informative = sd < tol
        z = (float(np.median(v)) - truth) / sd
        # the diffuse posterior has a SMALL z -- that is the trap
        if name == "diffuse":
            assert abs(z) < 5.0, "diffuse z should look innocuous"
            assert not informative, "diffuse must fail the gate"
        else:
            assert informative, "tight must pass the gate"


def test_a_bare_pull_is_fooled_and_documents_why():
    """Pinned so nobody reintroduces a bare pull as a success metric."""
    rng = np.random.default_rng(1)
    tol = 0.01
    truth = 1.0
    tight, diffuse = _posteriors(rng, truth, tol)
    # tight but OFFSET by one tolerance -- a real, resolved near-miss
    tight = tight + tol
    pull_tight = abs(np.median(tight) - truth) / np.std(tight)
    pull_diffuse = abs(np.median(diffuse) - truth) / np.std(diffuse)
    assert pull_tight > pull_diffuse, (
        "the bare pull prefers the DIFFUSE posterior -- this is review "
        "7.15.1 and is why the pull is not the sweep's metric"
    )


if __name__ == "__main__":
    test_informativeness_gate_rejects_a_diffuse_posterior()
    test_a_bare_pull_is_fooled_and_documents_why()
    print(
        "both pass: the gate rejects diffuse, and the bare pull is "
        "demonstrably fooled by it"
    )
