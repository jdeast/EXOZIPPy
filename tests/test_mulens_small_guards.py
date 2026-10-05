"""Small microlensing robustness fixes (reviews 2.6.7, 2.6.8, 2.6.9, 4.6.4).

Each test builds the synthetic system in tests/mulens_synthetic.py with the
topology its item needs.
"""

import numpy as np
import pytest
from mulens_synthetic import build, mulens_config, mulens_params, write_flat_lc

from exozippy.diagnostics import ModelAuditor
from exozippy.system import System

# ---------------------------------------------------------------------------
# 2.6.7: microlensing photometry without its event raises a CONFIG error
# naming the missing block (it used to die with a bare AttributeError on
# system.source deep inside stage 1).
# ---------------------------------------------------------------------------


def test_mulens_photometry_without_an_event_names_the_missing_blocks(
    tmp_path,
):
    """
    Given: a config with a mulensinstrument light curve and stars, but no
      mulensevent / lens / source block,
    When: prepare() runs stage 1,
    Then: MulensInstrument.load_data raises a ValueError naming all three
      missing blocks -- not an AttributeError on `system.source`.
    """
    # Arrange
    full = mulens_config(write_flat_lc(tmp_path / "lc.dat"))
    params = mulens_params(full)
    config = {k: full[k] for k in ("star", "mulensinstrument")}
    system = System(config, user_params=params)

    # Act / Assert
    with pytest.raises(ValueError) as err:
        system.prepare()
    msg = str(err.value)
    for block in ("'mulensevent'", "'lens'", "'source'"):
        assert block in msg, msg


# ---------------------------------------------------------------------------
# 2.6.8: _validate_pspl_start checks SAMPLED elements only -- under fitu0te
# the derived u_0 is skipped and u0te is checked instead.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_fitu0te(tmp_path_factory):
    """Two sources: S1 samples u0te (u_0 derived), S2 samples u_0."""
    lc = write_flat_lc(tmp_path_factory.mktemp("mixed") / "lc.dat")
    config = mulens_config(lc, sources=(("S1", {"fitu0te": True}), ("S2", {})))
    return build(config, mulens_params(config))


@pytest.mark.slow
def test_pspl_start_check_skips_a_derived_u_0_and_checks_u0te(
    mixed_fitu0te,
):
    """
    Given: a built 2-source system, source 0 under fitu0te (u_0 derived from
      the sampled u0te) and source 1 plain (u_0 sampled; its u0te element
      INACTIVE, pinned at 0),
    When: each start is forced to NaN in turn and _validate_pspl_start runs,
    Then: a NaN in the DERIVED u_0 (bookkeeping) or the INACTIVE u0te passes;
      a NaN in the sampled u_0 or the sampled u0te raises naming it.  Before
      2.6.8 the derived u_0's NaN raised and the u0te was never checked.
    """
    # Arrange
    system, _ = mixed_fitu0te
    src = system.source
    assert src.u_0.element_is_derived(0) and not src.u_0.element_is_derived(1)
    assert src.u0te.element_is_active(0) and not src.u0te.element_is_active(1)
    u_0, u0te = src.u_0.initval, src.u0te.initval
    nan = float("nan")
    try:
        # Act / Assert: bookkeeping NaNs are not starts.
        src.u_0.initval = np.array([nan, 0.3])
        src.u0te.initval = np.array([5.0, nan])
        src._validate_pspl_start()

        # The sampled u_0 (source 1) is a start.
        src.u_0.initval = np.array([0.3, nan])
        with pytest.raises(ValueError, match=r"source\.u_0"):
            src._validate_pspl_start()

        # The sampled u0te (source 0) is a start.
        src.u_0.initval = np.array([nan, 0.3])
        src.u0te.initval = np.array([nan, 0.0])
        with pytest.raises(ValueError, match=r"source\.u0te"):
            src._validate_pspl_start()
    finally:
        src.u_0.initval, src.u0te.initval = u_0, u0te


@pytest.mark.slow
def test_u_0_floor_warning_ignores_a_derived_u_0(mixed_fitu0te, caplog):
    """
    Given: the same mixed system, with the DERIVED u_0 (source 0) at 0 --
      a value the fit never starts at (it starts at u0te/t_E),
    When: _validate_pspl_start runs,
    Then: no floor warning: it would name a start the fit does not use.
      A sampled u_0 at 0 still warns.
    """
    system, _ = mixed_fitu0te
    src = system.source
    u_0 = src.u_0.initval
    try:
        src.u_0.initval = np.array([0.0, 0.3])
        caplog.clear()
        src._validate_pspl_start()
        assert "floor on |u_0|" not in caplog.text

        src.u_0.initval = np.array([0.3, 0.0])
        caplog.clear()
        src._validate_pspl_start()
        assert "floor on |u_0|" in caplog.text
    finally:
        src.u_0.initval = u_0


# ---------------------------------------------------------------------------
# 2.6.9 (backend: mulensmodel's auto_vbbl bracket covers the plot grid) and
# 4.6.4(c) (a companion's geometry is addressable by the body's name), on
# one binary-lens build.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_mulensmodel(tmp_path_factory):
    lc = write_flat_lc(tmp_path_factory.mktemp("binary") / "lc.dat")
    config = mulens_config(
        lc, lenses=("L1", "C"), event={"backend": "mulensmodel"}
    )
    params = mulens_params(config, {"lens.C.log_s": {"initval": 0.1}})
    system, model = build(config, params)
    return system, model, params


@pytest.mark.slow
def test_auto_vbbl_bracket_covers_the_plot_grid(binary_mulensmodel):
    """
    Given: a binary lens on backend: mulensmodel with the default
      mag_method (auto_vbbl),
    When: the model is built,
    Then: the resolved method list brackets EVERY epoch the event may be
      evaluated at -- the data, the plot grid (t_0 +/- 5 t_E) and epochs far
      outside both.  The plot grid used to outrun the data-span +/- 1 d
      bracket, so MulensModel drew the plotted curve beyond the data with
      its default point-source method.
    """
    system, _, _ = binary_mulensmodel
    inst = system.mulensinstrument
    method = system.mulensevent.mag_method[0]
    assert method[1] == "VBM", method
    t_model, _, _ = inst._model_time_grid()
    # The plot grid here happens to sit inside the data (the flat curve
    # seeds a short t_E), so also demand a grid far outside it: the bracket
    # must not depend on which grid a caller -- a plot, the GUI -- picks.
    # The old data-span +/- 1 d bracket fails this one.
    t_far = np.array([np.min(inst.time) - 1000.0, np.max(inst.time) + 1000.0])
    for t in (inst.time, t_model, t_far):
        assert method[0] < np.min(t) and np.max(t) < method[2], (
            method,
            np.min(t),
            np.max(t),
        )


@pytest.mark.slow
def test_a_companion_is_addressable_by_its_body_name(binary_mulensmodel):
    """
    Given: lens.C.log_s set in the params file, C being the companion
      body's star name (lens element 1),
    When: the model is built,
    Then: the value lands on element 1 and the unused-yaml audit does not
      call the key unmatched -- companions are addressable by name exactly
      as sources are (review 4.6.4(c); the instance naming of the 8.6.17
      split delivers it).
    """
    system, model, _ = binary_mulensmodel
    assert np.isclose(np.atleast_1d(system.lens.log_s.initval)[1], 0.1)
    unused = ModelAuditor(model, system, {}).check_unused_yaml()
    assert not [k for k in unused if str(k).startswith("lens.")], unused
