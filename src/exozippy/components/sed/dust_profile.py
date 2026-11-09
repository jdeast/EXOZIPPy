"""The sight-line dust profile: a monotone A_V(d) shared by every SED object
on one sight line, and the near/far R_V mixing (review 8.9.8; sed.md, "The
sight-line dust profile").

THE COORDINATE.  The smooth exponential dust layer (scale height H, the
microlensing standard: Bennett et al. 2015 eq. 19, Nataf et al. 2013 eq. 21)
puts a fraction

    s(d) = 1 - exp(-d sin|b| / H)          (computed as -expm1, exact on the
                                             plane's side)

of the total column in front of distance d.  The profile's K bins are EQUAL
INTERVALS IN s -- each carries the same share of the smooth layer's dust --
so the bins are dense where the dust is and none is wasted beyond the layer:

    phi_k(d) = clip(K s(d) - k, 0, 1),      k = 0 .. K-1
    C(d)     = sum_k 10**z_k phi_k(d)

With every z_k = 0 (the prior center), C(d) = K s(d): the smooth layer
EXACTLY, not a piecewise-linear approximation of it.  A z_k away from zero
puts more (or less) dust in bin k, so a cloud between two objects is
representable; C is monotone by construction (10**z > 0, ramps
non-decreasing), continuous, and differentiable in d almost everywhere (a
finite slope jump at each edge, like any piecewise-linear interpolator; clip's
gradient is finite on both sides).  No pt.where anywhere (CLAUDE.md).

THE ANCHOR.  A(d) = A_anchor C(d) / C(d_anchor): the sampled anchored total
carries the AMOUNT, the z_k only the SHAPE (their common shift cancels).
d_anchor is the red clump's sampled distance when a clump is in the fit, else
infinity (C(inf) = sum_k 10**z_k: the whole column, what a 2D map states).

NO DIVISION BY ZERO, NO FLOOR.  C(d) >= 10**z_0 K s(d) > 0 for d > 0, and
every distance's lower bound is strictly positive; s(d) > 0 needs sin|b| > 0,
which ``plan`` refuses to give up (it raises on the plane itself).  The near
fraction C(min(d, d_b)) / C(d) is in (0, 1] for the same reason.

R_V MIXING.  CCM is linear in 1/R_V (A_lambda/A_V = a + b/R_V), so dust from
two populations with different R_V sums EXACTLY to one CCM curve at

    1/R_eff(d) = f(d)/R_near + (1 - f(d))/R_far,   f(d) = A(min(d, d_b))/A(d)

the A_V-weighted mean of 1/R_V in front of d (Nataf et al. 2016's ">= 2
degrees of freedom" caveat: this makes the law consistent along the line of
sight within the CCM family, not right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

#: The dust layer's default scale height (pc): Nataf et al. (2013) eq. 21,
#: genulens's koshimoto21.DUST_SCALE_HEIGHT.  Only the binned prior's CENTER
#: (JDE 2026-11-09, ruling 6).
DEFAULT_SCALE_HEIGHT_PC = 164.0

#: Each bin's log10 increment scatters about the smooth layer by this many
#: dex (Green et al. 2014's log-normal increments; JDE's 0.7-1 dex).
Z_SIGMA_DEX = 0.8

#: The default bin count is one per SED object with its own distance,
#: between these (JDE 2026-11-09, ruling 14).
MIN_BINS = 3
MAX_BINS = 6

#: The reference spectrum an anchor band other than V is integrated on: the
#: bulge red clump (F1 of the design: (V-I)_0 = 1.06 through our tables),
#: because the 2D extinction maps are built from clump giants.
REFERENCE_SPECTRUM = (4700.0, 2.5, 0.0)

#: Fixed-point iterations inverting the band-integrated A_band(A_V) at the
#: reference spectrum (convergence factor ~ the elasticity of A_band/A_V,
#: ~0.01-0.05: three iterations from the A_V = 1 ratio leave < 1e-5).
ANCHOR_ITERATIONS = 3

#: The near/far boundary's broad prior (pc): where disk dust gives way to
#: inner-Galaxy dust toward the bulge (JDE 2026-11-09, ruling 13).
BOUNDARY_RANGE_PC = (3000.0, 6000.0)

#: The near (disk) dust's R_V prior in OUR band-integrated CCM89 +
#: O'Donnell family: Schlafly et al. (2016)'s diffuse-ISM R'(V) = 3.32 +/-
#: 0.18 (their eq. 8 proxy, calibrated on Fitzpatrick 1999) mapped through
#: the proxy computed on their reference giant (4500 K, logg 2.5, [Fe/H] 0)
#: at A_V = 2 with our spectral path: R_V = 2.88 +/- 0.19 (sed.md).
NEAR_RV_MU = 2.88
NEAR_RV_SIGMA = 0.19


def default_bins(n_groups):
    """One bin per SED object with its own distance, clipped to
    [MIN_BINS, MAX_BINS]: n objects have n extinctions, one of which the
    anchor fixes, and K bins carry K - 1 free shape coordinates (their
    common shift cancels) -- so K = n is the most the data can resolve, and
    finer bins add only prior-dominated degeneracy.  The floor keeps a
    lens-source pair able to put most of the dust between them."""
    return int(min(max(int(n_groups), MIN_BINS), MAX_BINS))


def layer_fraction(d, sinb, h, xp=pt):
    """s(d) = 1 - exp(-d sin|b| / H), via expm1."""
    return -xp.expm1(-d * (sinb / h))


def ramps(s, n_bins, xp=pt):
    """(..., K) bin ramps phi_k = clip(K s - k, 0, 1)."""
    k = np.arange(n_bins, dtype=float)
    if xp is pt:
        return pt.clip(n_bins * s[..., None] - k, 0.0, 1.0)
    return np.clip(n_bins * np.asarray(s, float)[..., None] - k, 0.0, 1.0)


def cumulative(d, z, sinb, h, n_bins, xp=pt):
    """C(d) = sum_k 10**z_k phi_k(d): shape (...,) for d of shape (...,);
    ``n_bins`` is K (static: the ramps' offsets are constants)."""
    phi = ramps(layer_fraction(d, sinb, h, xp), n_bins, xp)
    if xp is pt:
        return pt.sum(phi * 10.0**z, axis=-1)
    return np.sum(phi * 10.0 ** np.asarray(z, float), axis=-1)


def total(z, xp=pt):
    """C(inf) = sum_k 10**z_k (every ramp at 1)."""
    return xp.sum(10.0**z)


def bin_edges_pc(n_bins, sinb, h):
    """The K - 1 finite bin edges in distance (pc): s = k/K."""
    k = np.arange(1, n_bins, dtype=float)
    return -(h / sinb) * np.log1p(-k / n_bins)


def near_fraction(d, d_b, z, sinb, h, n_bins, xp=pt):
    """f(d) = C(min(d, d_b)) / C(d): the share of the dust in front of d that
    lies in front of the boundary.  min of two clips: a finite gradient."""
    lo = xp.minimum(d, d_b)
    return cumulative(lo, z, sinb, h, n_bins, xp) / cumulative(
        d, z, sinb, h, n_bins, xp
    )


def mixed_rv(f, rv_near, rv_far):
    """1/R_eff = f/R_near + (1 - f)/R_far (CCM is linear in 1/R_V)."""
    return 1.0 / (f / rv_near + (1.0 - f) / rv_far)


@dataclass(frozen=True)
class DustPlan:
    """What the SED's dust profile is on this sight line (``SED.dust_plan``).

    ``on``            the profile ties every read object's A_V to one A(d).
    ``reason``        one sentence, for the log, saying why on or off.
    ``n_bins``        K.
    ``sinb``          sin|b| of the sight line.
    ``h_pc``          the layer's scale height.
    ``clump``         the anchor is the red clump's distance (else infinity).
    ``anchor_band``   None for A_V itself, else the filter label.
    ``split``         near/far R_V mixing on.
    ``groups``        list of tuples of read star indices sharing a distance.
    ``near_stars``    read stars outside the far group (lens light).
    """

    on: bool
    reason: str
    n_bins: int = 0
    sinb: float = float("nan")
    b_deg: float = float("nan")
    h_pc: float = DEFAULT_SCALE_HEIGHT_PC
    clump: bool = False
    anchor_band: str | None = None
    split: bool = False
    groups: tuple = field(default_factory=tuple)
    near_stars: tuple = field(default_factory=tuple)

    def shape_np(self, d, z=None, d_anchor=None):
        """numpy A(d) / A_anchor at the z prior center (or ``z``)."""
        if z is None:
            z = np.zeros(self.n_bins)
        z = np.asarray(z, float)
        k = self.n_bins
        c = cumulative(np.asarray(d, float), z, self.sinb, self.h_pc, k, xp=np)
        if self.clump:
            ca = cumulative(
                np.asarray(d_anchor, float), z, self.sinb, self.h_pc, k, xp=np
            )
        else:
            ca = total(z, xp=np)
        return c / ca


def sinb_of(b_deg):
    return abs(math.sin(math.radians(float(b_deg))))
