"""``data_format: dia`` -- native difference-imaging photometry.

Difference imaging reports a flux relative to a reference image, which is
NOT the modeled observable: the model is ``F = f_s*A + f_b`` in total flux,
and a difference curve has its baseline at zero, so ``f_total = f_s + f_b``
is ~0 while ``f_s`` is large.  That drives ``q_source = f_s/f_total`` far
outside its ``[0, 2]`` bound -- measured at 181 on OGLE-2016-BLG-1045 --
and the bound is correct, so the data must be converted instead.

The conversion needs the reference flux, which a native pySIS file carries
implicitly: its five columns are ``time dflux dflux_err mag mag_err``, and
``mag = zp - 2.5*log10(ref - dflux)`` determines both ``ref`` and ``zp``.
These tests pin that recovery, and pin the refusal to proceed without it --
a three-column difference file must RAISE rather than have a reference
guessed for it, because every guess silently changes the science (offsetting
to zero blending imposes a blending fraction, corrupting any theta_E or lens
mass taken from the source flux).
"""

import numpy as np
import pytest

from exozippy.components.mulensing.mulensinstrument import MulensInstrument

# A synthetic native pySIS file, built from the relation the reader inverts
# so the expected answer is known exactly rather than fitted from real data.
_REF = 1584.893
_ZP = 28.0


def _probe(files, config):
    """A MulensInstrument with only the attributes these helpers read.

    __init__ wants a whole System; the two methods under test read exactly
    ``files``, ``config`` and ``names``.  ``prefix`` is a read-only property
    and is left alone.
    """
    obj = object.__new__(MulensInstrument)
    obj.files = files
    obj.config = config
    obj.names = [c.get("name", "site") for c in config]
    return obj


def _write_native(path, n=120, ref=_REF, zp=_ZP):
    """Write a five-column pySIS file whose reference flux is ``ref``."""
    rng = np.random.default_rng(7)
    t = 8168.0 + np.arange(n) * 0.05
    # A peak: total flux rises well above the reference, so dflux (which is
    # ref - total) goes strongly NEGATIVE, the pySIS convention.
    amp = 1.0 + 40.0 * np.exp(-0.5 * ((t - t[n // 2]) / 0.6) ** 2)
    total = ref * amp
    dflux = ref - total
    dflux_err = np.full(n, 150.0) + rng.normal(0, 1e-9, n)
    mag = zp - 2.5 * np.log10(total)
    mag_err = (np.log(10.0) / 2.5) * dflux_err / total
    np.savetxt(
        path,
        np.column_stack([t, dflux, dflux_err, mag, mag_err]),
        fmt="%.8f",
        header="HJD dflux dflux_err mag mag_err",
    )
    return t, dflux, mag


def test_native_five_column_file_recovers_the_reference_flux(tmp_path):
    """Given a native five-column pySIS file, when the reference flux is
    solved from its mag and dflux columns, then both the reference and the
    zeropoint come back at the values the file was built with."""
    # Arrange
    path = tmp_path / "native.pysis"
    _, dflux, mag = _write_native(path)
    inst = _probe([str(path)], [{"name": "KMTC14", "data_format": "dia"}])

    # Act
    ref, zp = inst._dia_reference(0, dflux=dflux, mag=mag)

    # Assert
    assert ref == pytest.approx(_REF, rel=1e-6)
    assert zp == pytest.approx(_ZP, abs=1e-6)


def test_reconstructed_total_flux_is_positive_and_inverts_the_mag_column(
    tmp_path,
):
    """Given a native pySIS file, when the total flux is reconstructed as
    ref - dflux, then it is positive everywhere and reproduces the file's
    own mag column -- which is the check that the subtraction (not an
    addition) is the right sign convention."""
    # Arrange
    path = tmp_path / "native.pysis"
    _, dflux, mag = _write_native(path)
    inst = _probe([str(path)], [{"name": "KMTC14", "data_format": "dia"}])

    # Act
    ref, zp = inst._dia_reference(0, dflux=dflux, mag=mag)
    total = ref - dflux

    # Assert
    assert np.all(total > 0.0)
    assert np.allclose(zp - 2.5 * np.log10(total), mag, atol=1e-6)


def test_three_column_difference_file_raises_rather_than_guessing(tmp_path):
    """Given a difference file stripped to three columns, when it is read as
    dia, then it raises and the message names both ways out -- the reference
    flux is absent and no choice of it is derivable, so guessing one would
    silently change the science."""
    # Arrange
    native = tmp_path / "native.pysis"
    _write_native(native)
    stripped = tmp_path / "stripped.dat"
    np.savetxt(stripped, np.loadtxt(native)[:, :3], fmt="%.8f")
    inst = _probe([str(stripped)], [{"name": "KMTC14", "data_format": "dia"}])

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        inst._require_dia_columns(0)
    message = str(excinfo.value)
    assert "has 3" in message
    assert "reference_flux" in message
    assert "reference_mag" in message


def test_explicit_reference_flux_lets_a_three_column_file_be_read(tmp_path):
    """Given a three-column difference file and an explicit reference_flux,
    when it is read as dia, then the guard passes and that value is used
    verbatim instead of being solved."""
    # Arrange
    native = tmp_path / "native.pysis"
    _, dflux, mag = _write_native(native)
    stripped = tmp_path / "stripped.dat"
    np.savetxt(stripped, np.loadtxt(native)[:, :3], fmt="%.8f")
    inst = _probe(
        [str(stripped)],
        [{"name": "KMTC14", "data_format": "dia", "reference_flux": 1600.0}],
    )

    # Act
    inst._require_dia_columns(0)
    ref, _ = inst._dia_reference(0, dflux=dflux, mag=mag)

    # Assert
    assert ref == pytest.approx(1600.0)


def test_reference_mag_is_converted_with_the_zeropoint(tmp_path):
    """Given reference_mag instead of reference_flux, when the override is
    resolved, then it is converted as 10**(-0.4*(mag - zp)) -- so mag 20 at
    the pySIS zeropoint of 28 is the same 1584.893 the solver returns."""
    # Arrange
    path = tmp_path / "stripped.dat"
    np.savetxt(path, np.zeros((3, 3)), fmt="%.1f")
    inst = _probe(
        [str(path)],
        [{"name": "KMTC14", "data_format": "dia", "reference_mag": 20.0}],
    )

    # Act
    ref = inst._reference_flux_override(0)

    # Assert
    assert ref == pytest.approx(1584.893, rel=1e-6)


def test_a_reference_below_the_largest_dflux_is_refused(tmp_path):
    """Given data whose solved reference would leave a non-positive total
    flux, when the reference is resolved, then it raises -- ref must exceed
    every dflux or some epoch's total flux ref - dflux is <= 0, which is not
    a light curve."""
    # Arrange: mag constant, so the relation cannot pin a sensible ref.
    path = tmp_path / "flat.pysis"
    n = 40
    dflux = np.linspace(-100.0, 5000.0, n)
    mag = np.full(n, 18.0)
    np.savetxt(
        path,
        np.column_stack(
            [np.arange(n, dtype=float), dflux, np.ones(n), mag, np.ones(n)]
        ),
        fmt="%.6f",
    )
    inst = _probe([str(path)], [{"name": "KMTC14", "data_format": "dia"}])

    # Act / Assert
    with pytest.raises(ValueError, match="reference|solve"):
        inst._dia_reference(0, dflux=dflux, mag=mag)
