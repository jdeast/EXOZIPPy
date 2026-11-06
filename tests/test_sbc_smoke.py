"""CI-sized simulation-based calibration of the cheapest shipped model (7.16.4).

The full runs (a few hundred replicates per model) live on an HPC cluster and
are driven by examples/validation/sbc.py.  This is their smoke version: the
one-subject pharmacokinetic model, 20 replicates, small NUTS budgets, the same
code path end to end -- prior draws from the built model with its likelihood
removed, data from the independent closed-form curve, the per-replicate
invariants (same prior parameterization, same prior density at the truth,
forward models agree), a fit from the truth, and the rank of the truth among
the thinned posterior draws.

Loose but meaningful.  With N = 20 a rank test only sees a gross failure --
a missing Jacobian on a coordinate, a likelihood whose sigma is off by a
factor of a few, a posterior centred on the wrong value.  The negative
control is what proves the test can fail: the SAME posteriors ranked against
the WRONG replicate's truth (the truth of replicate i + 1) put every rank of a
data-dominated parameter in the tails, and the statistic must reject that.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "examples" / "validation"
sys.path.insert(0, str(VALIDATION))

N_REP = 20


@pytest.fixture(scope="module")
def sbc_rows(tmp_path_factory):
    import sbc

    out = tmp_path_factory.mktemp("sbc")
    spec = sbc.PKSpec()
    spec.tune, spec.draws, spec.chains, spec.cores = 300, 300, 2, 2
    spec.prior_tune, spec.prior_chains = 600, 2
    common = dict(model="pk", outdir=str(out), seed=7164, cores=2)
    sbc.cmd_prior(
        argparse.Namespace(**common, n=N_REP, raw_per_draw=40), spec=spec
    )
    sbc.cmd_run(
        argparse.Namespace(**common, start=0, count=N_REP, maxtime=0.0),
        spec=spec,
    )
    rows = []
    for fp in sorted((out / "pk" / "rows").glob("rep_*.json")):
        with open(fp) as f:
            rows.append(json.load(f))
    with open(out / "pk" / "prior_report.json") as f:
        report = json.load(f)
    return rows, report


SAMPLED = [
    "subject.log_cl[0]",
    "subject.log_ka[0]",
    "subject.log_ke[0]",
    "assay.sigma_add[0]",
    "assay.sigma_prop[0]",
]


@pytest.mark.slow
def test_every_replicate_passes_the_invariants(sbc_rows):
    """Given 20 PK replicates, then none was refused: the prior build and every
    replicate build share one prior, and the independent curve agrees with
    exozippy's at the truth."""
    rows, report = sbc_rows
    assert len(rows) == N_REP
    errors = [r["error"] for r in rows if "error" in r]
    assert not errors, errors
    assert max(r["mu_check_sigma"] for r in rows) < 1e-6
    assert report["divergences"] == 0
    assert set(SAMPLED) <= set(report["diagnostics"])


@pytest.mark.slow
def test_pk_ranks_are_not_grossly_miscalibrated(sbc_rows):
    """Given the converged replicates, then no sampled coordinate's ranks
    leave the 99.9% simultaneous ECDF band."""
    import sbc_stats

    rows, _ = sbc_rows
    good = [
        r
        for r in rows
        if r["divergences"] == 0
        and all(
            q["rhat"] < 1.05 and q["ess_bulk"] > 100
            for q in r["quantities"].values()
        )
    ]
    assert len(good) >= 16, f"only {len(good)} of {N_REP} converged"
    L = good[0]["L"]
    gamma = sbc_stats.ecdf_band_gamma(len(good), L, alpha=0.001)
    for name in SAMPLED:
        r = np.array([row["quantities"][name]["rank"] for row in good])
        _, p, inside = sbc_stats.ecdf_diff(r, L, gamma, alpha=0.001)
        assert inside, (name, sorted(r.tolist()), p)


@pytest.mark.slow
def test_negative_control_wrong_truth_is_rejected(sbc_rows):
    """Ranking each posterior against the NEXT replicate's truth must fail:
    without this, a pass above could mean the statistic sees nothing."""
    import sbc_stats

    rows, _ = sbc_rows
    rows = [r for r in rows if "error" not in r]
    L = rows[0]["L"]
    gamma = sbc_stats.ecdf_band_gamma(len(rows), L, alpha=0.001)
    rejected = []
    for name in SAMPLED:
        r = np.array(
            [
                int(
                    np.sum(
                        np.asarray(row["quantities"][name]["thin"])
                        < rows[(i + 1) % len(rows)]["quantities"][name][
                            "truth"
                        ]
                    )
                )
                for i, row in enumerate(rows)
            ]
        )
        _, _, inside = sbc_stats.ecdf_diff(r, L, gamma, alpha=0.001)
        rejected.append(not inside)
    # The clearance is the best-measured coordinate; it must be rejected, and
    # so must most of the rest.
    assert rejected[0], "log_cl with the wrong truth stayed inside the band"
    assert sum(rejected) >= 3, dict(zip(SAMPLED, rejected))
