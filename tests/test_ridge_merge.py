"""Tests for the lp-barrier ridge merge in identify_modes (_lp_ridge_merge).

A flat likelihood ridge -- an unconstrained degeneracy direction, stretched
out by the raw-space transform -- separates in occupancy DENSITY without
separating in LIKELIHOOD, so the density-dip merge keeps it as a spurious
extra "mode".  The motivating case is a converged (16k-draw) RV-only kelt4
fit: the m--cos i degeneracy tail at cos i -> 1 sat ~1500 raw units from
the bulk and was reported as a 0.6% second mode, while the draws' max-lp
was flat to ~6 nats along the whole connecting path (no barrier at all).

The merge criterion has two halves, and each test pins one:

* a populated, lp-flat path between two clusters means ONE basin (merge);
* an lp-suppressed path, or an EMPTY path (absence of evidence -- also
  what a curved banana projects to), keeps the clusters separate.
"""

import arviz as az
import numpy as np
import pytest

from exozippy.outputs.modes import identify_modes

N_CHAIN = 4
BULK, BAND, LUMP = 6000, 60, 300
SEP = 20.0  # raw-space separation of the two density lumps


def _idata(rng, band_lp_offset=0.0, with_band=True):
    """Two density lumps at 0 and SEP along one raw dimension, optionally
    bridged by a sparse uniform band of draws; two more isotropic-noise
    dimensions; flat lp everywhere except the band's ``band_lp_offset``.
    """
    parts = [
        rng.normal(0.0, 1.0, BULK),
        rng.normal(SEP, 1.0, LUMP),
    ]
    lp_parts = [
        rng.normal(-50.0, 2.5, BULK),
        rng.normal(-50.0, 2.5, LUMP),
    ]
    if with_band:
        parts.append(rng.uniform(0.0, SEP, BAND))
        lp_parts.append(rng.normal(-50.0 + band_lp_offset, 2.5, BAND))
    x0 = np.concatenate(parts)
    lp = np.concatenate(lp_parts)

    n = x0.size - (x0.size % N_CHAIN)
    order = rng.permutation(x0.size)[:n]
    x0, lp = x0[order], lp[order]

    posterior = {
        "a_raw": x0.reshape(N_CHAIN, -1),
        "b_raw": rng.normal(0, 1, n).reshape(N_CHAIN, -1),
        "c_raw": rng.normal(0, 1, n).reshape(N_CHAIN, -1),
    }
    return az.from_dict(
        {
            "posterior": posterior,
            "sample_stats": {"lp": lp.reshape(N_CHAIN, -1)},
        }
    )


def test_flat_lp_ridge_merges_to_one_mode():
    """
    Given two density lumps bridged by a sparse band of draws whose lp is
      just as high as both lumps' (a flat likelihood ridge),
    When modes are identified,
    Then the lumps merge into ONE mode and the report's notes say why --
      density separated them, likelihood does not.
    """
    idata = _idata(np.random.default_rng(3), band_lp_offset=0.0)

    report = identify_modes(idata, attach=False)

    assert report.n_modes == 1
    assert any("flat likelihood ridge" in n for n in report.notes)


def test_lp_barrier_keeps_two_modes():
    """
    Given the same geometry but with the bridging draws' lp suppressed by
      40 nats (a real likelihood barrier between two basins),
    When modes are identified,
    Then both modes survive: a populated valley is only merged when its
      lp reaches the peaks.
    """
    idata = _idata(np.random.default_rng(3), band_lp_offset=-40.0)

    report = identify_modes(idata, attach=False)

    assert report.n_modes == 2
    assert not any("flat likelihood ridge" in n for n in report.notes)


def test_empty_valley_keeps_two_modes():
    """
    Given two lumps with NO draws at all between them (the classic
      well-separated bimodal posterior -- chains essentially never cross),
    When modes are identified,
    Then both modes survive: an empty path is absence of evidence, never
      a merge.
    """
    idata = _idata(np.random.default_rng(3), with_band=False)

    report = identify_modes(idata, attach=False)

    assert report.n_modes == 2
    assert not any("flat likelihood ridge" in n for n in report.notes)


def test_missing_lp_skips_the_ridge_merge():
    """
    Given a trace with no sample_stats['lp'] (the ridge merge's evidence),
    When modes are identified on the flat-ridge geometry,
    Then the merge is skipped (two modes stand, as before this feature)
      rather than crashing or merging on no evidence.
    """
    idata = _idata(np.random.default_rng(3), band_lp_offset=0.0)
    del idata.sample_stats["lp"]

    report = identify_modes(idata, attach=False)

    assert report.n_modes == 2
    assert not any("flat likelihood ridge" in n for n in report.notes)
