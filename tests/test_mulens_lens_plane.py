"""The microlensing zooms and the MulensModel caustic/trajectory figure.

* ``mmplot.anomaly_windows`` groups the epochs where a binary lens departs
  from its effective single lens, pads them, and drops a window that
  repeats the peak zoom.
* ``mmplot.single_lens_magnification`` puts that single lens where the
  binary acts as one, in MulensModel's frame: the anomaly detector's
  reference light curve is only as good as that frame.
* ``backend: mulensmodel`` scores a POINT-source binary: ``resolve_auto_vbbl``
  used to emit a VBM method list for it, which MulensModel refuses, so every
  proposal scored NaN.
* On examples/DC2018_128 the plots write the full light curve, the peak
  zoom, an anomaly zoom over the planetary perturbation, and the lens-plane
  figure, and every posterior draw lands in the reference point's windows.
"""

import os
import shutil
from pathlib import Path

import MulensModel as mm
import numpy as np
import pytest
from mulens_synthetic import (
    T0,
    build,
    mulens_config,
    mulens_params,
    write_flat_lc,
)

from exozippy.components.mulensing import mmplot

REPO = Path(__file__).resolve().parents[1]


def test_anomaly_windows_group_pad_and_drop_the_peak_repeat():
    """
    Given anomalous epochs in three clusters -- one filling a peak window
      with no stronger structure inside it, two well apart -- plus one
      isolated deviant epoch,
    When anomaly_windows groups them,
    Then the isolated epoch and the peak repeat are dropped and the two
      clusters come back padded by half their width, in time order.
    """
    # Arrange
    t = np.arange(0.0, 100.0, 0.1)
    z = np.zeros_like(t)
    z[(t >= 20.0) & (t <= 21.0)] = 10.0  # anomaly A
    z[(t >= 60.0) & (t <= 62.0)] = 5.0  # anomaly B
    z[np.argmin(np.abs(t - 40.0))] = 50.0  # one deviant epoch
    z[(t >= 80.0) & (t <= 89.0)] = 8.0  # fills the peak window
    departure = np.full_like(t, 0.005)  # below every refinement level
    peak = (79.0, 90.0)

    # Act
    windows = mmplot.anomaly_windows(t, z, departure, 20.0, peak_window=peak)

    # Assert
    assert len(windows) == 2
    (a_lo, a_hi), (b_lo, b_hi) = windows
    assert a_lo == pytest.approx(20.0 - 0.5, abs=0.06)
    assert a_hi == pytest.approx(21.0 + 0.5, abs=0.06)
    assert b_lo == pytest.approx(60.0 - 1.0, abs=0.06)
    assert b_hi == pytest.approx(62.0 + 1.0, abs=0.06)


def test_anomaly_windows_descend_into_a_stellar_binarys_crossings():
    """
    Given a departure over the whole event (a stellar binary: a few percent
      everywhere, so one z > 3 window spanning the peak) with two sharp
      caustic crossings inside it, and epoch errors that make z jump about
      where the model is smooth,
    When anomaly_windows groups them,
    Then the crossings come back as two narrow windows -- located by the
      model's departure, not split on the error bars.
    """
    # Arrange
    rng = np.random.default_rng(3)
    t = np.arange(70.0, 100.0, 0.02)
    departure = np.full_like(t, 0.05)
    departure[np.abs(t - 84.0) < 0.1] = 2.0  # caustic entry
    departure[np.abs(t - 86.0) < 0.1] = 2.0  # caustic exit
    sigma = rng.choice([0.002, 0.05], size=t.size)  # two instruments
    z = 0.1 * departure / sigma

    # Act
    windows = mmplot.anomaly_windows(
        t, z, departure, 30.0, peak_window=(75, 95)
    )

    # Assert
    assert len(windows) == 2
    for (lo, hi), crossing in zip(windows, (84.0, 86.0)):
        assert lo < crossing < hi
        assert hi - lo < 1.0


def test_anomaly_windows_are_none_without_departures():
    t = np.linspace(0.0, 10.0, 50)
    assert mmplot.anomaly_windows(t, np.zeros(50), np.zeros(50), 5.0) == []


def test_peak_half_width_follows_u0_and_rho():
    """A few u_0 t_E, widened by a resolved source, floored and capped."""
    assert mmplot.peak_half_width(20.0, 0.1) == pytest.approx(6.0)
    assert mmplot.peak_half_width(20.0, 0.01, rho=0.05) == pytest.approx(3.0)
    assert mmplot.peak_half_width(20.0, 1e-5) == pytest.approx(
        20.0 * mmplot.PEAK_HALF_MIN_TE
    )
    assert mmplot.peak_half_width(20.0, 2.0) == pytest.approx(
        20.0 * mmplot.PEAK_HALF_MAX_TE
    )


@pytest.mark.parametrize(
    "s, u_0",
    [(10.0, 0.02), (5.0, 0.05), (0.2, 0.02), (0.1, 0.02)],
    ids=["wide10", "wide5", "close0.2", "close0.1"],
)
def test_single_lens_magnification_is_the_binary_away_from_caustics(s, u_0):
    """
    Given a planet (q = 1e-3) well away from the Einstein ring and a
      high-magnification trajectory that passes no caustic,
    When the effective single lens is evaluated along MulensModel's own
      trajectory,
    Then it reproduces MulensModel's binary magnification to 2e-3 through
      the peak -- while the bare primary (no Chang-Refsdal shift, or no
      center-of-mass for a close companion) misses by 3-10 times that, and
      the primary mirrored through the center of mass by tens of percent on
      the wide side, so the frame is really being tested.
    """
    # Arrange
    q = 1e-3
    params = dict(t_0=0.0, u_0=u_0, t_E=20.0, s=s, q=q, alpha=40.0)
    model = mm.Model(params)
    model.set_magnification_methods([-np.inf, "point_source", np.inf])
    t = np.linspace(-1.0, 1.0, 201) * u_0 / 0.02
    A_binary = model.get_magnification(t)
    traj = mm.Trajectory(t, parameters=model.parameters)
    named = {"s": s, "q": q, "rho": None}

    # Act
    A_single = mmplot.single_lens_magnification(traj.x, traj.y, named)
    u = np.hypot(traj.x + s * q / (1 + q), traj.y)
    A_bare = (u**2 + 2) / np.sqrt(u**2 * (u**2 + 4))

    # Assert
    err = np.max(np.abs(A_single / A_binary - 1))
    assert err < 2e-3
    assert np.max(np.abs(A_bare / A_binary - 1)) > 3 * err
    if s > 1:
        mirrored = mmplot.single_lens_magnification(
            traj.x + 2 * s * q / (1 + q), traj.y, named
        )
        assert np.max(np.abs(mirrored / A_binary - 1)) > 0.05


@pytest.fixture(scope="module")
def binary_lc(tmp_path_factory):
    return write_flat_lc(tmp_path_factory.mktemp("lens_plane") / "lc.dat")


def _binary_logp(lc, backend):
    config = mulens_config(lc, lenses=("L1", "C"), event={"backend": backend})
    params = mulens_params(
        config,
        {
            "source.S1.t_0": {"initval": T0},
            "source.S1.u_0": {"initval": 0.1},
            "lens.C.s": {"initval": 1.0},
            "lens.C.alpha": {"initval": 30.0},
        },
    )
    system, model = build(config, params)
    return system, float(model.compile_logp()(model.initial_point()))


def test_mulensmodel_backend_scores_a_point_source_binary(binary_lc):
    """
    Given a point-source binary lens,
    When it is built on backend: mulensmodel and on vbm_direct,
    Then the method list is MulensModel's point-source solver and both
      backends score the same finite logp (it was NaN on mulensmodel).
    """
    # Act
    system_mm, lp_mm = _binary_logp(binary_lc, "mulensmodel")
    _, lp_vbm = _binary_logp(binary_lc, "vbm_direct")

    # Assert
    assert system_mm.mulensevent.mag_method[0] == [
        -np.inf,
        "point_source",
        np.inf,
    ]
    assert np.isfinite(lp_mm)
    assert lp_mm == pytest.approx(lp_vbm, rel=1e-9)


@pytest.mark.slow
def test_dc2018_128_plots_zoom_onto_the_planetary_anomaly(tmp_path):
    """
    Given examples/DC2018_128 (a planetary binary with its perturbation
      2-3 days after the peak) at its start,
    When the microlensing plots are drawn for the start and two jittered
      draws,
    Then the light curve, the peak zoom, one anomaly zoom covering the
      perturbation (t ~ 2458556-2458558) and the lens-plane figure are
      written; the zooms carry their own y range; and every draw's zoom
      chart sits in the reference point's window.
    """
    # Arrange
    from exozippy.system import System
    from exozippy.yamlio import load_system_config

    work = tmp_path / "DC2018_128"
    shutil.copytree(REPO / "examples" / "DC2018_128", work)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        system = System(load_system_config("DC2018_128.yaml"))
        system.prepare()
        model = system.build_model()
        system.compile_plotter_functions(model)
        inst = system.mulensinstrument
        prefix = str(tmp_path / "dc128")

        # Act
        windows = inst.zoom_windows(system, {})
        inst.plot(system, [{}, {}], filename_prefix=prefix)
        specs = inst.plot_data(system, {}, zoom_windows=windows)
    finally:
        os.chdir(cwd)

    # Assert
    assert len(windows["anomalies"]) == 1
    lo, hi = windows["anomalies"][0]
    assert lo < 2458556.5 and hi > 2458557.8
    peak_lo, peak_hi = windows["peak"]
    assert hi - lo < mmplot.ANOMALY_MAX_PEAK_COVER * (peak_hi - peak_lo)
    for tag in ("mulens", "mulens_zoom", "mulens_anomaly1", "mulens_caustics"):
        assert Path(f"{prefix}_{tag}.pdf").is_file(), tag
    by_id = {s.id: s for s in specs}
    zoom = by_id["mulensinstrument.lightcurve_anomaly1"]
    assert zoom.x_range == [lo, hi]
    assert zoom.y_range is not None and zoom.y_range[0] < zoom.y_range[1]
    model_x = np.concatenate(
        [np.asarray(t.x) for t in zoom.traces if t.role == "model"]
    )
    assert model_x.min() >= lo and model_x.max() <= hi
