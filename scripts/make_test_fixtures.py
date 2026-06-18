"""Generate pre-computed test fixtures that avoid network / ephemeris downloads.

Run once (or after changing the test time range) from the repo root:
    poetry run python scripts/make_test_fixtures.py
"""
import os
import numpy as np
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u

os.makedirs("tests/fixtures", exist_ok=True)

# --- earth_parallax_test.npz ---
# Simulated satellite observer positions for test_pspl_symbolic_vs_op_with_earth_parallax.
#
# xyz_abs is Earth + a constant satellite displacement (~Spitzer-like, 1 AU from Sun).
# Using a displaced satellite rather than Earth itself is essential: for a pure
# Earth observer the Op computes geocentric = obs_abs - earth_actual = 0 (no
# parallax), while the symbolic path computes obs_abs - linearized_earth ≠ 0
# (non-linear orbital residual).  With a satellite offset both paths compute the
# same geocentric deviation and agree to < 1e-3 over ±25 days.
#
# Uses Astropy 'builtin' ephemeris — bundled, no network required, accurate to ~1 km
# over 50 days (well below the 1e-3 tolerance of that test).

solar_system_ephemeris.set('builtin')

t0, t0_par = 2460025.0, 2460025.0
t_vals = np.linspace(t0 - 25.0, t0 + 25.0, 200)

earth_xyz = (get_body_barycentric('earth', Time(t_vals, format='jd', scale='tdb'))
             .xyz.to(u.au).value.T)

# Constant displacement in the ecliptic plane (x, y, z in AU).
# Magnitude ~0.05 AU gives a clear parallax signal at the test pi_E values.
satellite_offset = np.array([0.04, 0.03, 0.01])
xyz_abs = earth_xyz + satellite_offset[np.newaxis, :]

dt = 0.5
def _earth(t):
    return (get_body_barycentric('earth', Time([t], format='jd', scale='tdb'))
            .xyz.to(u.au).value.T[0])

earth_pos_ref = _earth(t0_par)          # Earth reference (no satellite offset)
earth_vel_ref = (_earth(t0_par + dt) - _earth(t0_par - dt)) / (2.0 * dt)

np.savez('tests/fixtures/earth_parallax_test.npz',
         t_vals=t_vals,
         xyz_abs=xyz_abs,
         earth_pos_ref=earth_pos_ref,
         earth_vel_ref=earth_vel_ref,
         t0_par=np.array([t0_par]))

print(f"Written tests/fixtures/earth_parallax_test.npz  (xyz_abs shape: {xyz_abs.shape})")
