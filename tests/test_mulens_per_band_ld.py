"""Finite-source limb darkening is per BAND, not per instrument (review 8.6.10).

`MulensInstrument._finite_source_limb_darkening` used to resolve ONE
(u1, u2, bandpass) for every light curve -- the first band's -- and warn
"Multiple bands for finite-source instruments; using first band's u1."  So a
V-band light curve crossed the caustic with band I's source profile, across
exactly the epochs that carry the finite-source information (+40-55 chi2 per
solution at the start of a real OGLE-2013-BLG-0341 fit).

`_model_flux` now makes one magnification call per (source, band), each on
that band's rows only, reassembled with `pt.set_subtensor` (never `pt.where`,
CLAUDE.md).  Pinned here by EVALUATING the graph against a direct per-band
VBM call, on a two-band caustic-crossing binary whose two bands have very
different (pinned) u1:

  * every row's magnification is its OWN band's (and measurably not the other
    band's at the caustic epochs, so the comparison has content);
  * the cost is one VBM evaluation per epoch: one Op per band, whose inputs
    partition the data;
  * the plotted curve for each light curve is drawn with its own band's
    profile (the plot = likelihood rule, review 1.6.1).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from mulens_synthetic import T0, build, mulens_config, mulens_params
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import applys_between, graph_inputs

from exozippy.components.mulensing.op import VBMDirectMagOp

U1 = {"I": 0.70, "V": 0.15}
RHO = 0.01


def _write_lc(path, offset):
    """A light curve densely sampled across the caustic region.  The two
    bands sample interleaved epochs (``offset``), so the per-band row sets
    are not contiguous blocks and a wrong partition cannot pass.  The values
    are a rough point-lens bump (u_0 = 0.05, t_E ~ 21 d, the start's) so the
    flux bootstrap finds a real source flux for both files; their fit is not
    what is tested."""
    t = np.linspace(T0 - 20.0, T0 + 20.0, 300) + offset
    u = np.hypot(0.05, (t - T0) / 21.0)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    mag = 17.0 - 2.5 * np.log10(0.8 * amp + 0.2)
    err = np.full_like(t, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


@pytest.fixture(scope="module")
def two_band_binary(tmp_path_factory):
    d = tmp_path_factory.mktemp("per_band_ld")
    config = mulens_config(
        _write_lc(d / "I.dat", 0.0),
        lenses=("L1", "C"),
        event={"finite_source": True, "backend": "vbm_direct"},
    )
    config["mulensinstrument"] = [
        {"name": "OGLE_I", "file": _write_lc(d / "I.dat", 0.0), "band": "I"},
        {"name": "OGLE_V", "file": _write_lc(d / "V.dat", 0.067), "band": "V"},
    ]
    config["band"] = [
        {"name": "I", "filter": "Generic/Cousins.I", "ld_law": "linear"},
        {"name": "V", "filter": "Generic/Bessell.V", "ld_law": "linear"},
    ]
    params = mulens_params(
        config,
        {
            "source.S1.t_0": {"initval": T0},
            # Through the resonant caustic of an s = 1, q = 1/6 binary.
            "source.S1.u_0": {"initval": 0.05},
            "source.S1.rho": {"initval": RHO},
            "lens.C.s": {"initval": 1.0},
            "lens.C.alpha": {"initval": 30.0},
            "band.I.u1": {"initval": U1["I"], "sigma": 0},
            "band.V.u1": {"initval": U1["V"], "sigma": 0},
        },
    )
    system, model = build(config, params)
    system.compile_plotter_functions(model)
    return system, model


def _at_start(model, nodes):
    fn = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
    ip = model.initial_point()
    return fn(*[ip[v.name] for v in model.free_RVs])


def _applys(node):
    return list(applys_between(list(graph_inputs([node])), [node]))


def _vbm_ops(node):
    """The VBMDirectMagOp applications in ``node``'s graph."""
    return [a for a in _applys(node) if isinstance(a.op, VBMDirectMagOp)]


def _magnification(system, flux_node):
    """``flux_node`` with every f_source = 1 and f_blend = 0: the
    magnification itself, exactly, with no (F - f_b)/f_s inversion to lose
    digits in."""
    inst = system.mulensinstrument
    n = len(inst.names)
    return graph_replace(
        flux_node,
        {inst.f_source.value: pt.ones(n), inst.f_blend.value: pt.zeros(n)},
        strict=False,
    )


def _likelihood_flux(system):
    """The likelihood's own model-flux expression (build_likelihood's call)."""
    inst = system.mulensinstrument
    return inst._model_flux(
        system,
        pt.as_tensor_variable(inst.time),
        inst.observer_pos,
        inst.inst_map_tensor,
        inst.inst_map,
    )


def _direct_vbm(system, model, band, t, obs):
    """ONE direct VBM call at ``band``'s pinned u1: a fresh VBMDirectMagOp
    fed the event's numeric parameter vector (the geometry is read off the
    likelihood graph's own Op input, its trailing u1 replaced by a literal),
    so nothing of `_model_flux`'s band bookkeeping is reused."""
    op_inputs = _vbm_ops(_likelihood_flux(system))[0].inputs
    (p,) = _at_start(model, [op_inputs[0]])
    p = np.array(p, dtype=float)
    p[-1] = U1[band]
    op = VBMDirectMagOp(
        coords=system.mulensevent.mm_coords(system, 0),
        n_companions=1,
        use_rho=True,
        bandpass=band,
    )
    out = [[None]]
    op.perform(None, [p, np.asarray(t, float), np.asarray(obs, float)], out)
    return np.asarray(out[0][0], dtype=float)


def test_each_band_gets_its_own_profile(two_band_binary):
    """
    Given a caustic-crossing binary with an I and a V light curve whose u1
      differ (0.70 vs 0.15),
    When the likelihood's model flux is evaluated at the start,
    Then each row's magnification matches a direct VBM call at its OWN
      band's u1, and at the caustic epochs measurably differs from the other
      band's -- the first band's profile was applied to both until 8.6.10.
    """
    # Arrange
    system, model = two_band_binary
    inst = system.mulensinstrument
    a_node = _magnification(system, _likelihood_flux(system))

    # Act
    a_model, u1 = _at_start(model, [a_node, system.band.u1.value])

    # Assert -- the pins reached the band nodes (the test's premise).
    np.testing.assert_allclose(u1, [U1["I"], U1["V"]], rtol=0, atol=1e-12)
    n_caustic = 0
    for i, band, other in ((0, "I", "V"), (1, "V", "I")):
        rows = inst.rows(i)
        t, obs = inst.time[rows], inst.observer_pos[rows]
        own = _direct_vbm(system, model, band, t, obs)
        wrong = _direct_vbm(system, model, other, t, obs)
        np.testing.assert_allclose(a_model[rows], own, rtol=1e-10, atol=0)
        # The caustic epochs: where the two profiles disagree by > 1 mmag.
        caustic = np.abs(2.5 * np.log10(wrong / own)) > 1e-3
        n_caustic += int(caustic.sum())
        assert caustic.sum() >= 5, f"{band}: no caustic epochs sampled"
        assert np.max(np.abs(a_model[rows] / wrong - 1)) > 1e-2
    assert n_caustic >= 10


def test_one_vbm_evaluation_per_epoch(two_band_binary):
    """
    Given the two-band event,
    When the likelihood's magnification graph is inspected,
    Then there is one VBM Op per band, and their time inputs partition the
      data: the per-band split costs no extra VBM evaluations.
    """
    # Arrange
    system, model = two_band_binary
    inst = system.mulensinstrument

    # Act
    ops = _vbm_ops(_likelihood_flux(system))
    times = _at_start(model, [a.inputs[1] for a in ops])

    # Assert
    assert sorted(a.op.bandpass for a in ops) == ["I", "V"]
    np.testing.assert_array_equal(
        np.sort(np.concatenate(times)), np.sort(inst.time)
    )


def test_plotted_curve_uses_its_own_band(two_band_binary):
    """
    Given the two-band event,
    When each light curve's compiled plot curve is drawn on a common grid,
    Then it is that light curve's own band's magnification (the plotted =
      fitted rule, review 1.6.1), not the first band's for both.
    """
    # Arrange
    system, model = two_band_binary
    inst = system.mulensinstrument
    values = inst._point_to_plot_params({}, system)
    fs, fb = inst._compiled_flux(*values)
    grid = np.linspace(T0 - 10.0, T0 + 10.0, 200)
    obs = inst._abs_to_delta(grid, inst.get_observer_position(grid))

    for i, band, other in ((0, "I", "V"), (1, "V", "I")):
        # Act
        flux = np.asarray(
            inst._compiled_model_flux(grid, obs, i, *values), dtype=float
        )

        # Assert -- forward (f_s A + f_b), with f_s a real share of the
        # flux, so the comparison is sensitive to A.
        own = _direct_vbm(system, model, band, grid, obs)
        wrong = _direct_vbm(system, model, other, grid, obs)
        assert fs[i] > 0.25 * (fs[i] + fb[i]) and fb[i] > 0
        np.testing.assert_allclose(
            flux, fs[i] * own + fb[i], rtol=1e-10, atol=0
        )
        assert np.max(np.abs(flux / (fs[i] * wrong + fb[i]) - 1)) > 1e-3


def test_single_band_event_is_one_call_on_every_row(two_band_binary):
    """
    Given the same event with only ONE band present in the rows,
    When the model flux is built,
    Then it is a single VBM Op over all rows -- the pre-8.6.10 graph, so
      single-band fits are bit-identical (the acceptance fixtures pin the
      numbers; this pins the structure that makes it so).
    """
    # Arrange
    system, _ = two_band_binary
    inst = system.mulensinstrument
    rows = inst.rows(0)

    t = pt.as_tensor_variable(inst.time[rows])

    # Act
    node = inst._model_flux(
        system, t, inst.observer_pos[rows], 0, inst.inst_map[rows]
    )

    # Assert -- one Op, fed the time vector itself (no row selection).
    ops = _vbm_ops(node)
    assert [a.op.bandpass for a in ops] == ["I"]
    assert ops[0].inputs[1] is t
