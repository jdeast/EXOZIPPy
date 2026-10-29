"""`mulensevent.vbm_tol` / `vbm_reltol` reach VBMicrolensing's Tol / RelTol.

JDE ruling: VBM's accuracy goals are user-facing floats on the event block
(mulensing.md, "VBM accuracy").  Absent, every VBMDirectMagOp carries the
values it always did (Tol = 1e-3, RelTol = 0), so no shipped example moves;
present, the number reaches the VBM instance the magnification is computed
on; anything that is not a number in range raises naming the key.
"""

import pytensor.tensor as pt
import pytest
from test_band_autopin_ld import _mulens_config, _mulens_params, _write_pspl_lc

from exozippy.components.mulensing.mulensevent import parse_vbm_accuracy
from exozippy.components.mulensing.op import (
    VBM_RELTOL_DEFAULT,
    VBM_TOL_DEFAULT,
    VBMDirectMagOp,
)
from exozippy.system import System

RHO = 0.02
U0 = 0.01


def _fs_op(tmp_path, **event_keys):
    """The VBMDirectMagOp a finite-source single-lens event builds, with
    `event_keys` written into its mulensevent block."""
    lc = _write_pspl_lc(tmp_path / "lc.dat")
    params = _mulens_params(finite_source=True)
    params["source.Source.rho"] = {"initval": RHO}
    params["source.Source.u_0"] = {"initval": U0}
    config = _mulens_config(
        lc,
        bands=[{"name": "I", "filter": "I", "ld_law": "linear"}],
        finite_source=True,
    )
    config["mulensevent"][0].update(event_keys)
    system = System(config, user_params=params)
    system.prepare()
    system.build_model()
    u1, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    node = system.mulensevent.get_magnification_op(
        pt.dvector("t"),
        pt.dmatrix("obs"),
        system,
        index=0,
        u1=u1,
        u2=u2,
        bandpass=bandpass,
    )
    op = node.owner.op
    assert isinstance(op, VBMDirectMagOp)
    return op


def test_vbm_accuracy_defaults_are_unchanged(tmp_path):
    """
    Given a finite-source event that names neither key,
    When its VBM Op is built,
    Then VBM runs at Tol = 1e-3 and RelTol = 0, the values every Op carried
      before the keys existed.
    """
    op = _fs_op(tmp_path)
    assert (VBM_TOL_DEFAULT, VBM_RELTOL_DEFAULT) == (1e-3, 0.0)
    assert op._vbm.Tol == 1e-3
    assert op._vbm.RelTol == 0.0


def test_vbm_accuracy_keys_reach_vbm_tol_and_reltol(tmp_path):
    """
    Given `vbm_tol: 1.0e-5` and `vbm_reltol: 1.0e-4` on the event,
    When its VBM Op is built,
    Then the VBMicrolensing instance the magnification is computed on
      carries exactly those numbers.
    """
    op = _fs_op(tmp_path, vbm_tol=1.0e-5, vbm_reltol=1.0e-4)
    assert op._vbm.Tol == 1.0e-5
    assert op._vbm.RelTol == 1.0e-4


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("vbm_tol", 0.0, r"mulensevent\.vbm_tol must be in \(0, 1\)"),
        ("vbm_tol", -1.0e-3, r"mulensevent\.vbm_tol must be in \(0, 1\)"),
        ("vbm_tol", 10.0, r"mulensevent\.vbm_tol must be in \(0, 1\)"),
        ("vbm_tol", float("nan"), r"mulensevent\.vbm_tol must be in"),
        ("vbm_tol", "1e-4", r"mulensevent\.vbm_tol must be a number.*1\.0e-4"),
        ("vbm_tol", True, r"mulensevent\.vbm_tol must be a number"),
        ("vbm_tol", [1.0e-3], r"mulensevent\.vbm_tol must be a number"),
        (
            "vbm_reltol",
            -1.0e-3,
            r"mulensevent\.vbm_reltol must be in \[0, 1\)",
        ),
        ("vbm_reltol", 1.0, r"mulensevent\.vbm_reltol must be in \[0, 1\)"),
        ("vbm_reltol", "0", r"mulensevent\.vbm_reltol must be a number"),
    ],
)
def test_a_bad_vbm_accuracy_raises_naming_the_key(key, value, match):
    """
    Given a vbm_tol / vbm_reltol that is not a number in its range (PyYAML's
      string `1e-4` included, with the spelling that works in the message),
    When the event block is read,
    Then it raises naming the key -- nothing is coerced.
    """
    with pytest.raises(ValueError, match=match):
        parse_vbm_accuracy({key: value}, key)


def test_a_bad_vbm_accuracy_raises_at_system_construction(tmp_path):
    """
    Given `vbm_tol: 0` in a real config,
    When the System is built,
    Then construction raises naming mulensevent.vbm_tol.
    """
    with pytest.raises(ValueError, match=r"mulensevent\.vbm_tol"):
        _fs_op(tmp_path, vbm_tol=0)
