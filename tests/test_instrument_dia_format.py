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

from pathlib import Path

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


# ---------------------------------------------------------------------------
# An explicit reference replaces the mag column (review 2.2.22b)
# ---------------------------------------------------------------------------


def _write_native_bjd(path):
    """``_write_native`` with full BJD times (the fit's ephemeris needs a
    real date), returned as the path."""
    _write_native(path)
    d = np.loadtxt(path)
    d[:, 0] += 2450000.0
    np.savetxt(path, d, fmt="%.8f")
    return path


def _prepare_pspl(tmp_path, entry):
    """Stages 1-4 of a one-light-curve PSPL event reading ``entry``."""
    import os

    from exozippy.system import System

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [{"finite_source": False, "t0_par": 2458171.0}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [entry],
    }
    params = {
        "source.Source.t_0": {"initval": 2458171.0},
        "source.Source.u_0": {"initval": 0.1},
        "mulensevent.t_E": {"initval": 1.0},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        system = System(config, user_params=params)
        system.prepare()
    finally:
        os.chdir(cwd)
    return system


def test_three_column_file_with_reference_mag_loads(tmp_path):
    """Given a three-column difference file and reference_mag, when the
    system is prepared, then it loads -- it used to raise "data file has 3
    columns; got [3, 4]", because the roles and the default column spec
    still named mag/mag_err -- and reads exactly what the native
    five-column file with the same reference_mag reads."""
    # Arrange
    native = _write_native_bjd(tmp_path / "native.pysis")
    stripped = tmp_path / "stripped.dat"
    np.savetxt(stripped, np.loadtxt(native)[:, :3], fmt="%.8f")
    entry = {"name": "KMTC14", "data_format": "dia", "reference_mag": 20.0}

    # Act
    three = _prepare_pspl(tmp_path, dict(entry, file="stripped.dat"))
    five = _prepare_pspl(tmp_path, dict(entry, file="native.pysis"))

    # Assert
    a, b = three.mulensinstrument, five.mulensinstrument
    assert a.column_specs[0] == {"time": 0, "dflux": 1, "err": 2}
    assert b.column_specs[0] == {"time": 0, "dflux": 1, "err": 2}
    np.testing.assert_array_equal(a.flux, b.flux)
    np.testing.assert_array_equal(a.err, b.err)
    assert a.flux_system_zp[0] == pytest.approx(20.0)


def test_a_column_spec_naming_mag_with_an_explicit_reference_raises(
    tmp_path,
):
    """Given an explicit reference AND a columns: map naming mag, when the
    file is read, then it raises: the explicit reference replaces the mag
    column, so the spec states something nothing reads."""
    # Arrange
    _write_native_bjd(tmp_path / "native.pysis")
    entry = {
        "name": "KMTC14",
        "file": "native.pysis",
        "data_format": "dia",
        "reference_mag": 20.0,
        "columns": {"time": 0, "dflux": 1, "err": 2, "mag": 3, "mag_err": 4},
    }

    # Act / Assert
    with pytest.raises(ValueError, match="replaces the mag column"):
        _prepare_pspl(tmp_path, entry)


# ---------------------------------------------------------------------------
# A solved pySIS reference is not the star's baseline (review 2.2.22c)
# ---------------------------------------------------------------------------

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


def _kmt_as_dia(tmp_path, entry_extra):
    """KMT-2019-BLG-1806 (an SED block, three I curves) with every file
    rewritten as a native pySIS dia file of the same photometry, each entry
    updated with ``entry_extra``; returns (config, user_params, workdir)."""
    import shutil

    import yaml

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    workdir = tmp_path / "KMT"
    shutil.copytree(_KMT_DIR, workdir)
    with open(workdir / "KMT-2019-BLG-1806.yaml") as f:
        config = yaml.safe_load(f)
    with open(workdir / config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    for entry in config["mulensinstrument"]:
        d = np.loadtxt(workdir / entry["file"])
        f_native = 10 ** (-0.4 * (d[:, 1] - _ZP))
        np.savetxt(
            workdir / (entry["file"] + ".dia"),
            np.column_stack(
                [
                    d[:, 0],
                    _REF - f_native,
                    np.log(10.0) / 2.5 * f_native * d[:, 2],
                    d[:, 1],
                    d[:, 2],
                ]
            ),
            fmt="%.17g",
        )
        entry["file"] = entry["file"] + ".dia"
        entry["data_format"] = "dia"
        entry.update(entry_extra)
    return config, user_params, workdir


def _prepare_in(workdir, config, user_params):
    import os

    from exozippy.system import System

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        system = System(config, user_params=user_params)
        system.prepare()
    finally:
        os.chdir(cwd)
    return system


def test_blend_tie_on_a_solved_dia_reference_raises(tmp_path):
    """Given native pySIS dia files with NO explicit reference and
    sed_constrains_blend: true, when the system is prepared, then it raises
    naming the light curve: the solved reference is a pipeline number
    (pySIS's fixed 20.0 mag here), not the star's baseline, so the f_blend
    the tie constrains is
    fictitious.  (f_source is unaffected, which is why the zeropoint tie
    alone stays legal.)"""
    config, params, workdir = _kmt_as_dia(
        tmp_path, {"sed_constrains_blend": True}
    )
    with pytest.raises(ValueError, match="'KMTC04'.*pipeline reference"):
        _prepare_in(workdir, config, params)


@pytest.mark.slow
def test_blend_tie_with_a_measured_reference_is_accepted(tmp_path):
    """Given the same files with an explicit reference_mag (the user's
    measured baseline), the blend tie is accepted and reads it."""
    config, params, workdir = _kmt_as_dia(
        tmp_path, {"sed_constrains_blend": True, "reference_mag": 20.0}
    )
    system = _prepare_in(workdir, config, params)
    assert system.mulensinstrument.flux_system_zp == pytest.approx(
        [20.0, 20.0, 20.0]
    )


def test_a_solved_dia_reference_with_no_sed_block_is_accepted(tmp_path):
    """Rope, not gates: with no SED block sed_constrains_blend reads
    nothing, and a solved reference only shifts the blend, which nothing
    reads -- so the file loads."""
    native = _write_native_bjd(tmp_path / "native.pysis")
    system = _prepare_pspl(
        tmp_path,
        {
            "name": "KMTC14",
            "file": native.name,
            "data_format": "dia",
            "sed_constrains_blend": True,
        },
    )
    assert system.mulensinstrument.flux_system_zp[0] == pytest.approx(
        _ZP - 2.5 * np.log10(_REF), abs=1e-4
    )
