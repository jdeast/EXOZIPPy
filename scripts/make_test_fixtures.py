"""Generate pre-computed test fixtures that avoid network / ephemeris downloads.

Run once (or after changing the test time range) from anywhere:
    poetry run python scripts/make_test_fixtures.py

NOTE (2026-08-12 review): nothing in tests/ currently reads the one fixture
this writes.  ``test_pspl_symbolic_vs_op_with_earth_parallax``, named below,
no longer exists -- tests/test_pspl_symbolic_vs_op.py now builds its
deviations analytically (``_skowron_deviations``) and its satellite case is
``test_pspl_symbolic_vs_op_with_satellite_offset``.  Left in place, not
deleted, pending a decision; do not assume regenerating this changes any
test's behaviour.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# --- earth_parallax_test.npz ---
# Simulated satellite observer positions for a symbolic-vs-Op parallax test.
#
# xyz_abs is Earth + a constant satellite displacement (~Spitzer-like, 1 AU from Sun).
# Using a displaced satellite rather than Earth itself is essential: for a pure
# Earth observer the Op computes geocentric = obs_abs - earth_actual = 0 (no
# parallax), while the symbolic path computes obs_abs - linearized_earth != 0
# (non-linear orbital residual).  With a satellite offset both paths compute the
# same geocentric deviation and agree to < 1e-3 over +/-25 days.
#
# Uses Astropy 'builtin' ephemeris -- bundled, no network required, accurate to
# ~1 km over 50 days (well below the 1e-3 tolerance of that test).

T0 = 2460025.0
T0_PAR = 2460025.0
DT = 0.5

# Constant displacement in the ecliptic plane (x, y, z in AU).
# Magnitude ~0.05 AU gives a clear parallax signal at the test pi_E values.
SATELLITE_OFFSET = np.array([0.04, 0.03, 0.01])


def _earth(t):
    return (
        get_body_barycentric("earth", Time([t], format="jd", scale="tdb"))
        .xyz.to(u.au)
        .value.T[0]
    )


def main():
    solar_system_ephemeris.set("builtin")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    t_vals = np.linspace(T0 - 25.0, T0 + 25.0, 200)
    earth_xyz = (
        get_body_barycentric("earth", Time(t_vals, format="jd", scale="tdb"))
        .xyz.to(u.au)
        .value.T
    )
    xyz_abs = earth_xyz + SATELLITE_OFFSET[np.newaxis, :]

    earth_pos_ref = _earth(T0_PAR)  # Earth reference (no satellite offset)
    earth_vel_ref = (_earth(T0_PAR + DT) - _earth(T0_PAR - DT)) / (2.0 * DT)

    out = FIXTURE_DIR / "earth_parallax_test.npz"
    np.savez(
        out,
        t_vals=t_vals,
        xyz_abs=xyz_abs,
        earth_pos_ref=earth_pos_ref,
        earth_vel_ref=earth_vel_ref,
        t0_par=np.array([T0_PAR]),
    )
    print(f"Written {out}  (xyz_abs shape: {xyz_abs.shape})")


if __name__ == "__main__":
    main()
