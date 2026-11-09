"""The N-lens finite-source hexadecapole shortcut (review 8.6.43; mulensing.md,
"N-lens finite source: the hexadecapole shortcut").

Inside the far-field radius ``max(s + 1/s) + 2`` the vbm_direct Op used to
send every N-lens epoch to MultiMag2.  A tight pair inflates that radius to
tens of Einstein radii, and there MultiMag2 is slow and wrong where nothing
is happening: OGLE-2013-BLG-0341's close seeds (the stellar pair at
s ~ 0.03) sent epochs 25-28 theta_E from every caustic to it at 140-290 s
each, returning A = 0.99-1.34 for a point-source 1.000004.  The Op now takes
Gould's (2008) hexadecapole from 13 MultiMag0 calls where the quadrupole
term is within the accuracy goal max(Tol, RelTol A), the hexadecapole term a
tenth of it, and no caustic lies within 3 rho of the source.  Pinned here:

1. The disk moments reproduce Gould's linear-law factors and a direct
   quadrature of the quadratic law.
2. Far from every caustic but inside the far-field radius (a tight pair),
   the Op answers without MultiMag2 and equals the point-source value.
3. Along a trajectory past the caustics, every epoch the shortcut accepts
   matches a Tol = 1e-5 MultiMag2 within the goal, and the epochs near the
   caustics still go to MultiMag2.
4. Next to a caustic much smaller than rho -- a close pair's secondary
   caustic -- the series test alone would accept, and the caustic guard is
   what sends the epoch to MultiMag2.
5. The accuracy goal follows the Op's Tol.
6. The hexadecapole centroid matches MultiMag2's.
7. A uniform_rotation pair carried through zero (far side at |s|): the far
   epochs take the shortcut and equal the point source.
8. Computing the caustics does not perturb the next MultiMag2.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import VBMicrolensing

from exozippy.components.mulensing.op import (
    _HEX_OFFSETS_LIST,
    HEXADECAPOLE_CAUSTIC_CLEARANCE,
    VBMDirectMagOp,
    _hexadecapole,
    ld_disk_moments,
)

_V = VBMicrolensing.VBMicrolensing
_COORDS = "268.0d -29.0d"
_T0 = 2456400.0
_TE = 30.0
_RHO = 3e-3
_U1 = 0.6
# OGLE-2013-BLG-0341's close layout (Gould et al. 2014): companion 0 the
# planet (s, q of the primary, alpha deg), companion 1 the stellar companion
# (q 2.42 of the primary) at separation s_star.
_PLANET = (0.95, 1.7e-4, 175.0)
_STAR_Q, _STAR_ALPHA = 2.42, 190.0


class _CountingVBM:
    """The Op's VBM instance with every MultiMag2 call's (x, y) logged."""

    def __init__(self, vbm):
        object.__setattr__(self, "_v", vbm)
        object.__setattr__(self, "mag2_xy", [])

    def __getattr__(self, name):
        attr = getattr(self._v, name)
        if name != "MultiMag2":
            return attr

        def logged(x, y, rho):
            self.mag2_xy.append((x, y))
            return attr(x, y, rho)

        return logged

    def __setattr__(self, name, val):
        setattr(self._v, name, val)


def _layout(s_star):
    m = np.array([1.0, _PLANET[1], _STAR_Q])
    m /= m.sum()
    al = np.radians([_PLANET[2], _STAR_ALPHA])
    pos = np.array(
        [
            [0.0, 0.0],
            [_PLANET[0] * np.cos(al[0]), -_PLANET[0] * np.sin(al[0])],
            [s_star * np.cos(al[1]), -s_star * np.sin(al[1])],
        ]
    )
    pos -= m @ pos
    return pos, m


def _vbm(s_star, tol):
    v = _V()
    v.Tol = tol
    v.RelTol = 0.0
    v.minannuli = 2
    v.SetLDprofile(_V.LDlinear)
    v.a1 = _U1
    v.SetMethod(_V.Method.Multipoly)
    pos, m = _layout(s_star)
    v.SetLensGeometry(np.column_stack([pos, m]).ravel().tolist())
    return v


def _op(accuracy=1e-3, astrometry=False, orbital_motion=False):
    op = VBMDirectMagOp(
        coords=_COORDS,
        n_companions=2,
        use_rho=True,
        bandpass="I",
        accuracy=accuracy,
        astrometry=astrometry,
        orbital_motion=orbital_motion,
    )
    counter = _CountingVBM(op._vbm)
    op._vbm = counter
    p, t, o = pt.dvector("p"), pt.dvector("t"), pt.dmatrix("o")
    ins = [p, t, o]
    if orbital_motion:
        ins += [pt.dmatrix("s"), pt.dmatrix("a")]
    return op, counter, pytensor.function(ins, op(*ins))


def _params(s_star, u_0):
    return np.array(
        [_T0, u_0, _TE, 0.0, 0.0, _RHO]
        + list(_PLANET)
        + [s_star, _STAR_Q, _STAR_ALPHA, _U1]
    )


def _at(points):
    """Times and u_0 putting the source at trajectory-frame (x, y) points
    with one shared y (the Op's source sits at (-tau, -u_0))."""
    xs = np.array([p[0] for p in points])
    ys = {p[1] for p in points}
    assert len(ys) == 1
    return _T0 - xs * _TE, -ys.pop()


# ---------------------------------------------------------------------------
# 1. the disk moments
# ---------------------------------------------------------------------------


def test_disk_moments_reproduce_gould_and_quadrature():
    """
    Given: VBM's limb-darkening profiles,
    When:  ld_disk_moments gives <r^2> and <r^4> over the disk,
    Then:  uniform is (1/2, 1/3); linear is Gould's (2008)
           ((1 - Gamma/5)/2, (1 - 11 Gamma/35)/3), Gamma = 2a/(3 - a); and
           a quadratic law matches a direct quadrature.
    """
    np.testing.assert_allclose(
        ld_disk_moments(0.0), (0.5, 1.0 / 3.0), rtol=1e-14
    )
    for a in (0.3, 0.649, 0.9):
        gam = 2.0 * a / (3.0 - a)
        np.testing.assert_allclose(
            ld_disk_moments(a),
            ((1 - gam / 5) / 2, (1 - 11 * gam / 35) / 3),
            rtol=1e-13,
        )
    a1, a2 = 0.4, 0.25
    r = np.linspace(0.0, 1.0, 400001)
    mu = np.sqrt(1.0 - r * r)
    w = (1.0 - a1 * (1.0 - mu) - a2 * (1.0 - mu) ** 2) * r
    m1 = np.trapezoid(w * r**2, r) / np.trapezoid(w, r)
    m2 = np.trapezoid(w * r**4, r) / np.trapezoid(w, r)
    np.testing.assert_allclose(ld_disk_moments(a1, a2), (m1, m2), rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. far from everything, inside the inflated far-field radius
# ---------------------------------------------------------------------------


def test_tight_pair_far_epochs_take_the_shortcut_and_equal_the_point_source():
    """
    Given: the close layout with the stellar pair at s = 0.03 (far-field
           radius s + 1/s + 2 ~ 35) and source positions 15-28 theta_E
           from the center of mass, away from the pair's secondary
           caustics,
    When:  the Op evaluates them,
    Then:  no epoch calls MultiMag2, and each equals MultiMag0 at the
           center to 1e-6 (the finite-source correction there is ~1e-12).
    """
    s_star = 0.03
    xs = [-28.0, -20.0, -15.0, 15.0, 20.0]
    times, u_0 = _at([(x, 0.4) for x in xs])
    op, counter, f = _op()
    A = f(_params(s_star, u_0), times, np.zeros((len(times), 3)))
    assert counter.mag2_xy == []
    ref = _vbm(s_star, 1e-3)
    point = np.array([ref.MultiMag0(x, 0.4) for x in xs])
    np.testing.assert_allclose(A, point, rtol=0.0, atol=1e-6)
    assert np.all(np.abs(A - 1.0) < 1e-4)


# ---------------------------------------------------------------------------
# 3. along a trajectory: accepted epochs vs a tight MultiMag2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_star", [0.19, 0.6])
def test_accepted_epochs_match_tight_multimag2_within_the_goal(s_star):
    """
    Given: a trajectory at u_0 = 0.05 across the close layout's caustics,
           81 epochs over +/-2 t_E,
    When:  the Op evaluates it at the default Tol = 1e-3,
    Then:  some epochs take the shortcut and the ones nearest the peak go
           to MultiMag2; every shortcut epoch matches MultiMag2 at
           Tol = 1e-5 to the 1e-3 goal (in fact far better, recorded).
    """
    times = np.linspace(_T0 - 2 * _TE, _T0 + 2 * _TE, 81)
    op, counter, f = _op()
    A = f(_params(s_star, 0.05), times, np.zeros((len(times), 3)))
    x = -(times - _T0) / _TE
    mag2 = {round(px, 12) for px, _ in counter.mag2_xy}
    took_mag2 = np.array([round(px, 12) in mag2 for px in x])
    accepted = ~took_mag2
    assert accepted.sum() >= 10 and took_mag2.sum() >= 3
    # the epochs nearest the caustics are the MultiMag2 ones
    assert took_mag2[np.argmin(np.abs(x))]
    tight = _vbm(s_star, 1e-5)
    ref = np.array([tight.MultiMag2(px, -0.05, _RHO) for px in x[accepted]])
    err = np.abs(A[accepted] - ref)
    assert err.max() < 1e-3
    assert np.max(err / ref) < 1e-4


# ---------------------------------------------------------------------------
# 4. the caustic guard
# ---------------------------------------------------------------------------


def _secondary_caustic(s_star):
    v = _vbm(s_star, 1e-3)
    curves = [(np.asarray(c[0]), np.asarray(c[1])) for c in v.Multicaustics()]
    sizes = [max(np.ptp(cx), np.ptp(cy)) for cx, cy in curves]
    cx, cy = curves[int(np.argmin(sizes))]
    return float(cx.mean()), float(cy.mean()), min(sizes)


@pytest.mark.parametrize("offset_rho", [1.3, 2.5])
def test_guard_sends_an_epoch_beside_a_tiny_caustic_to_multimag2(offset_rho):
    """
    Given: the close pair at s = 0.03, whose secondary caustics are ~1.6e-5
           theta_E across at ~33 theta_E (<< rho = 3e-3), and the source
           centered offset_rho * rho from one, on a line between the
           13 sample points,
    When:  the Op evaluates that epoch,
    Then:  the series test on the 13 samples would ACCEPT it (the samples
           miss the caustic), and the epoch still goes to MultiMag2,
           because the caustic is closer than 3 rho.
    """
    s_star = 0.03
    ccx, ccy, size = _secondary_caustic(s_star)
    assert size < 0.01 * _RHO
    # between the axis (rho/2, rho) and diagonal samples: 22.5 deg
    ang = np.radians(22.5)
    sx = ccx - offset_rho * _RHO * np.cos(ang)
    sy = ccy - offset_rho * _RHO * np.sin(ang)
    v = _vbm(s_star, 1e-3)
    a0 = v.MultiMag0(sx, sy)
    samp = np.array(
        [
            v.MultiMag0(sx + _RHO * ox, sy + _RHO * oy)
            for ox, oy in _HEX_OFFSETS_LIST
        ]
    )
    m1, m2 = ld_disk_moments(_U1)
    _, series_ok = _hexadecapole(a0, samp, m1, m2, 1e-3)
    assert series_ok
    assert offset_rho < HEXADECAPOLE_CAUSTIC_CLEARANCE
    op, counter, f = _op()
    times, u_0 = _at([(sx, sy)])
    A = f(_params(s_star, u_0), times, np.zeros((1, 3)))
    assert len(counter.mag2_xy) == 1
    np.testing.assert_allclose(A[0], v.MultiMag2(sx, sy, _RHO), rtol=1e-9)


def test_guard_clears_an_epoch_beyond_the_clearance():
    """
    Given: the same tiny caustic and the source 6 rho from it,
    When:  the Op evaluates that epoch,
    Then:  it takes the shortcut (no MultiMag2) and equals a brute-force
           limb-darkened disk average of MultiMag0 to 1e-7 -- where the
           finite-source correction itself is ~1e-6 (the caustic's
           curvature, which the expansion resolves).
    """
    s_star = 0.03
    ccx, ccy, _ = _secondary_caustic(s_star)
    sx, sy = ccx - 6.0 * _RHO, ccy
    op, counter, f = _op()
    times, u_0 = _at([(sx, sy)])
    A = f(_params(s_star, u_0), times, np.zeros((1, 3)))
    assert counter.mag2_xy == []
    v = _vbm(s_star, 1e-3)
    disk = _disk_average(v, sx, sy)
    assert abs(disk - v.MultiMag0(sx, sy)) > 3e-7
    assert abs(A[0] - disk) < 1e-7


def _disk_average(v, x, y, n_r=60, n_th=96):
    """The linear-law disk average of MultiMag0 by midpoint quadrature in
    (r^2, theta) -- a reference with no expansion in it."""
    r = np.sqrt((np.arange(n_r) + 0.5) / n_r)
    th = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
    w = 1.0 - _U1 * (1.0 - np.sqrt(1.0 - r * r))
    ring = np.array(
        [
            np.mean(
                [
                    v.MultiMag0(
                        x + _RHO * ri * np.cos(t), y + _RHO * ri * np.sin(t)
                    )
                    for t in th
                ]
            )
            for ri in r
        ]
    )
    return float(np.sum(w * ring) / np.sum(w))


# ---------------------------------------------------------------------------
# 5. the goal follows Tol
# ---------------------------------------------------------------------------


def test_goal_follows_the_ops_tol():
    """
    Given: epochs a few tenths of theta_E from the central caustic, which
           the default Tol = 1e-3 lets take the shortcut,
    When:  the same Op is built with Tol = 1e-7,
    Then:  more of them go to MultiMag2 -- the shortcut answers to the
           accuracy the event asked for (vbm_tol / vbm_reltol).
    """
    times = np.linspace(_T0 - 1.5 * _TE, _T0 - 0.3 * _TE, 25)
    p = _params(0.19, 0.3)
    _, loose, f_loose = _op(accuracy=1e-3)
    f_loose(p, times, np.zeros((len(times), 3)))
    _, tight, f_tight = _op(accuracy=1e-7)
    f_tight(p, times, np.zeros((len(times), 3)))
    assert len(tight.mag2_xy) > len(loose.mag2_xy)


# ---------------------------------------------------------------------------
# 6. the centroid
# ---------------------------------------------------------------------------


def test_hexadecapole_centroid_matches_multimag2():
    """
    Given: the astrometric Op on the close layout (s = 0.19) along
           u_0 = 0.3,
    When:  it evaluates 41 epochs,
    Then:  every shortcut epoch's centroid shift matches a Tol = 1e-5
           MultiMag2's to 1e-4 theta_E, and its A to 1e-4 relative.
    """
    s_star = 0.19
    times = np.linspace(_T0 - 2 * _TE, _T0 + 2 * _TE, 41)
    op, counter, f = _op(astrometry=True)
    A, dtau, dbeta = f(_params(s_star, 0.3), times, np.zeros((len(times), 3)))
    x = -(times - _T0) / _TE
    mag2 = {round(px, 12) for px, _ in counter.mag2_xy}
    acc = np.array([round(px, 12) not in mag2 for px in x])
    assert acc.sum() >= 10
    v = _vbm(s_star, 1e-5)
    v.astrometry = True
    for i in np.flatnonzero(acc):
        a = v.MultiMag2(x[i], -0.3, _RHO)
        # trajectory frame: shift = centroid - source, source at (x, -u_0)
        assert abs(A[i] - a) / a < 1e-4
        assert abs(dtau[i] - (v.astrox1 - x[i])) < 1e-4
        assert abs(dbeta[i] - (v.astrox2 + 0.3)) < 1e-4


# ---------------------------------------------------------------------------
# 7. a uniform_rotation pair through zero
# ---------------------------------------------------------------------------


def test_pair_through_zero_far_epochs_take_the_shortcut():
    """
    Given: the moving path with the stellar pair's polar line carried
           through zero (|s| = 0.05 -> 0.002 -> 0.04 on the far side,
           alpha + 180 deg past the crossing, as the orbit's Cartesian
           state hands it over; review 8.6.44) and the source 12-20 theta_E
           away, far from the pair's secondary caustics at ~1/|s|,
    When:  the Op evaluates those epochs,
    Then:  none calls MultiMag2 and each equals the point source to 1e-6.
    """
    n = 6
    xs = np.linspace(-20.0, -12.0, n)
    times, u_0 = _at([(x, 0.5) for x in xs])
    s_line = np.linspace(0.05, -0.04, n)
    s_t = np.vstack([np.full(n, _PLANET[0]), np.abs(s_line)])
    a_t = np.vstack(
        [
            np.full(n, _PLANET[2]),
            np.where(s_line < 0, _STAR_ALPHA + 180.0, _STAR_ALPHA),
        ]
    )
    op, counter, f = _op(orbital_motion=True)
    A = f(_params(0.05, u_0), times, np.zeros((n, 3)), s_t, a_t)
    assert counter.mag2_xy == []
    for i in range(n):
        v = _vbm(1.0, 1e-3)
        m = np.array([1.0, _PLANET[1], _STAR_Q])
        m /= m.sum()
        pos = np.zeros((3, 2))
        for j, (s, a) in enumerate(
            ((s_t[0, i], a_t[0, i]), (s_t[1, i], a_t[1, i]))
        ):
            ar = np.radians(a)
            pos[j + 1] = (s * np.cos(ar), -s * np.sin(ar))
        pos -= m @ pos
        v.SetLensGeometry(np.column_stack([pos, m]).ravel().tolist())
        assert abs(A[i] - v.MultiMag0(xs[i], 0.5)) < 1e-6


# ---------------------------------------------------------------------------
# 8. the caustics leave VBM's state alone
# ---------------------------------------------------------------------------


def test_multicaustics_does_not_perturb_multimag2():
    """
    Given: a geometry and a near-caustic epoch,
    When:  MultiMag2 is called before and after Multicaustics(),
    Then:  the two values are identical.
    """
    v = _vbm(0.19, 1e-3)
    before = v.MultiMag2(0.01, -0.02, _RHO)
    v.Multicaustics()
    after = v.MultiMag2(0.01, -0.02, _RHO)
    assert after == pytest.approx(before, rel=1e-12)
