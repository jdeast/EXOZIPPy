"""The plotted microlensing curve must be the curve the likelihood fits.

Two independent ways the plot path had drifted from the likelihood path
(review 1.6.1 and 1.6.4):

  * ``compile_plotters`` called ``get_magnification_op`` with no u1/bandpass,
    so ``effective_bandpass`` came out None and every plotted/GUI model curve
    was the UNIFORM-source magnification while the fit used the limb-darkened
    one.  The discrepancy is largest exactly where these plots are read.
  * the plotted curve kept a ``maximum(A_eff, 1e-30)`` clamp inside the graph,
    so a draw whose model flux went non-positive (heavy negative blending in
    difference imaging) drew a ~75 mag spike where the DATA path had already
    switched to a NaN gap.

Both are pinned here by evaluating the compiled plot function, not by
inspecting the source.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from test_band_autopin_ld import _mulens_config, _mulens_params, _write_pspl_lc

from exozippy.system import System


@pytest.fixture(scope="module")
def finite_source_system(tmp_path_factory):
    """A finite-source PSPL fit whose light curve declares a band.

    Finite source + a band is the one topology in which limb darkening
    reaches the magnification, so it is the only one in which the plot/
    likelihood parity claim has any content.
    """
    lc = _write_pspl_lc(tmp_path_factory.mktemp("mulens_parity") / "lc.dat")
    params = _mulens_params(finite_source=True)
    # A RESOLVED source: rho must be comparable to the impact parameter or the
    # source is a point as far as the magnification is concerned and limb
    # darkening changes nothing, which would make the parity assertion below
    # pass for the wrong reason.
    params["lens.Lens.rho"] = {"initval": 0.05}
    params["lens.Lens.u_0"] = {"initval": 0.01}
    system = System(
        _mulens_config(
            lc, bands=[{"name": "I", "filter": "I"}], finite_source=True
        ),
        user_params=params,
    )
    system.prepare()
    model = system.build_model()
    system.compile_plotter_functions(model)
    return system


def _plot_magnification(system, times):
    """Magnification the PLOT path draws, recovered from its model flux.

    The compiled plotter returns ``fs*A + fb`` in instrument 0's own flux
    system, so inverting it with that same point's fitted fluxes gives back
    the magnification curve the chart is showing.
    """
    inst = system.mulensinstrument
    values = inst._point_to_plot_params({}, system)
    obs_pos = inst._abs_to_delta(times, inst.get_observer_position(times))
    flux = np.asarray(
        inst._compiled_model_flux(times, obs_pos, 0, *values), dtype=float
    )
    fs_vec, fb_vec = inst._compiled_flux(*values)
    return (
        (flux - float(np.atleast_1d(fb_vec)[0]))
        / float(np.atleast_1d(fs_vec)[0]),
        values,
        obs_pos,
    )


def _likelihood_magnification(
    system, times, obs_pos, values, u1, bandpass, u2=None
):
    """Magnification the LIKELIHOOD path builds, compiled the same way."""
    t_in = pt.dvector("t")
    obs_in = pt.dmatrix("obs")
    node = system.lens.get_magnification_op(
        t_in, obs_in, system, index=0, u1=u1, u2=u2, bandpass=bandpass
    )
    fn = pytensor.function(
        inputs=[t_in, obs_in] + [p.value for p in system.plot_params],
        outputs=node,
        on_unused_input="ignore",
    )
    return np.asarray(fn(times, obs_pos, *values), dtype=float)


def test_finite_source_ld_resolver_finds_the_band(finite_source_system):
    """
    Given a finite-source lens whose light curve declares a band,
    When the shared (u1, u2, bandpass) resolver is asked,
    Then it returns them -- the precondition for the parity claim below.
    """
    # Arrange / Act
    u1, u2, bandpass = (
        finite_source_system.mulensinstrument._finite_source_limb_darkening(
            finite_source_system
        )
    )

    # Assert
    assert bandpass == "I"
    assert u1 is not None
    # u2 tracks the band's declared law: present for the quadratic default,
    # absent from the manifest entirely when every band is linear.
    band = finite_source_system.band
    assert (u2 is not None) == ("u2" in band.manifest)


def test_plotted_magnification_is_the_limb_darkened_one(finite_source_system):
    """
    Given a finite-source fit with a band,
    When the compiled plot curve and the likelihood's magnification are
      evaluated at the same point and times,
    Then they agree -- and the plot curve is measurably NOT the uniform-source
      magnification, which is what it silently was (review 1.6.1).
    """
    # Arrange
    system = finite_source_system
    inst = system.mulensinstrument
    u1, u2, bandpass = inst._finite_source_limb_darkening(system)
    times = inst.time[:40]

    # Act
    a_plot, values, obs_pos = _plot_magnification(system, times)
    a_fit = _likelihood_magnification(
        system, times, obs_pos, values, u1, bandpass, u2=u2
    )
    a_uniform = _likelihood_magnification(
        system, times, obs_pos, values, None, None
    )

    # Assert
    np.testing.assert_allclose(a_plot, a_fit, rtol=1e-10)
    # The regression guard: limb darkening must actually change the curve
    # here, or the parity assertion above would pass for the wrong reason.
    assert np.max(np.abs(a_fit - a_uniform)) > 1e-6


def test_plot_curve_is_nan_not_a_75_mag_spike(finite_source_system):
    """
    Given a point whose blend flux drives the model flux negative (the
      heavy-negative-blending difference-imaging case),
    When plot_data builds the model trace,
    Then the affected samples come back NaN -- a gap -- rather than the ~75
      mag spike the old in-graph 1e-30 clamp drew (review 1.6.4).
    """
    # Arrange
    system = finite_source_system
    inst = system.mulensinstrument
    # f_source/f_blend are DERIVED from (log_f_total, q_source):
    #   f_source = 10**log_f_total * q_source
    #   f_blend  = 10**log_f_total * (1 - q_source)
    # so a negative q_source is the sampled spelling of "the blend outshines a
    # negative source", i.e. the heavy-negative-blending corner where
    # f_s*A + f_b goes through zero (A >= 1 always, so no POSITIVE q_source
    # can get there).
    q_source = next(
        p for p in system.plot_params if p.label.endswith("q_source")
    )
    point = {q_source.label: np.full(inst.n_elements, -10.0)}

    # Act
    specs = inst.plot_data(system, point=point)
    model_y = np.concatenate(
        [
            np.asarray(tr.y, dtype=float)
            for tr in specs[0].traces
            if tr.role == "model"
        ]
    )

    # Assert
    assert np.any(np.isnan(model_y))
    finite = model_y[np.isfinite(model_y)]
    assert finite.size == 0 or np.max(np.abs(finite)) < 60.0
