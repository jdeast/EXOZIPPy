"""MulensModel drawings of the fitted lens: the anomaly windows the light-curve
zooms frame, and the caustic + source-trajectory figure.

Everything here is built from ``MulensEvent.mm_param_list`` -- the parameter
vector the ``mulensmodel`` backend's Ops take -- through op.py's own model
builders, so the lens drawn is the lens the parameters describe, with the
same parallax convention: the Skowron+2011 geocentric deviations fed as
MulensModel's satellite channel (``op._dev_skycoord``).  Nothing here enters
the likelihood.  ``MulensEvent.mulensmodel_unsupported`` names the
topologies MulensModel cannot carry; the callers skip those.

The frame is MulensModel's (and VBM's): origin at the lens center of mass,
the binary axis along +x with the companion at ``+s/(1+q)`` and the primary
at ``-s q/(1+q)`` -- measured, MulensModel's trajectory reproduces VBM's
``BinaryMag0`` at those positions to 4e-15, and its caustics are drawn in
the same frame.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import MulensModel as mm
import numpy as np
import VBMicrolensing

from .op import (
    _build_binary_model,
    _build_pspl_model,
    _clear_mm_satellite_cache,
    _dev_skycoord,
)

logger = logging.getLogger(__name__)

# An epoch is anomalous when the lens model departs from the single-lens
# model by more than this many of that epoch's own photometric errors.
ANOMALY_Z = 3.0
# A window needs at least this many anomalous epochs (one deviant point is
# not a feature worth a figure).
ANOMALY_MIN_POINTS = 3
# Two anomalous epochs belong to different windows when at least this many
# NON-anomalous epochs lie between them, or when they are further apart than
# ANOMALY_MAX_GAP_TE * t_E with no data at all in between.
ANOMALY_SPLIT_POINTS = 2
ANOMALY_MAX_GAP_TE = 0.25
# The figures stay legible: the windows with the largest summed z^2 win.
ANOMALY_MAX_WINDOWS = 6
# Each window is padded by this fraction of its width on either side (at
# least ANOMALY_MIN_PAD_DAYS), so the zoom shows the curve returning to the
# single-lens shape.
ANOMALY_PAD_FRACTION = 0.5
ANOMALY_MIN_PAD_DAYS = 0.1
# A window that spans this much of the peak zoom would show the same picture
# and is dropped.  A narrower one inside it is kept: on DC2018_128 the
# planetary anomaly's padded window is 64% of the peak zoom and is the one
# view in which the perturbation fills the panel.
ANOMALY_MAX_PEAK_COVER = 0.8
# A stellar binary departs from ANY single lens over the whole event
# (examples/ob09020/ob09020_linear.yaml: 1316 of 2837 epochs above 3 sigma), so at
# ANOMALY_Z its anomaly is one window spanning the peak and nothing is left
# to zoom on.  Such a window is re-examined at these successively larger
# RELATIVE departures |A - A_1|/A_1 of the model itself, until its strongest
# departures -- the caustic crossings -- come apart as windows of their
# own.  The levels are in the model's departure, not in z: z is the
# departure over each epoch's error, so with several instruments
# interleaved it jumps from epoch to epoch where the model is smooth, and a
# z ladder split ob09020's light curve on its error bars.
ANOMALY_DEPARTURE_LEVELS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

# Peak zoom half-width, in t_E: 3 u_0 (the PSPL FWHM is ~3.5 u_0 t_E for
# u_0 << 1), widened to 3 rho for a resolved source, and kept between these.
PEAK_HALF_MIN_TE = 0.02
PEAK_HALF_MAX_TE = 1.5

# Layout of mm_param_list's vector (op.py's positional layout).
_N_BASE = 5


def peak_half_width(t_E, u_0, rho=None):
    """Half-width [days] of the peak zoom about t_0."""
    half = 3.0 * abs(float(u_0))
    if rho is not None:
        half = max(half, 3.0 * float(rho))
    return float(t_E) * min(max(half, PEAK_HALF_MIN_TE), PEAK_HALF_MAX_TE)


def unpack(pvec, finite_source, n_lenses):
    """Name the entries of an ``mm_param_list`` vector."""
    p = np.asarray(pvec, dtype=float)
    out = dict(zip(("t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"), p[:_N_BASE]))
    idx = _N_BASE
    out["rho"] = None
    if finite_source:
        out["rho"] = float(p[idx])
        idx += 1
    if n_lenses == 2:
        out["s"], out["q"], out["alpha"] = (float(v) for v in p[idx : idx + 3])
        idx += 3
    out["linear_motion"] = len(p) > idx
    return out


def build_model(pvec, coords, mag_method, finite_source, n_lenses, t_0_kep):
    """The MulensModel ``Model`` of one source, built by op.py's builder."""
    if n_lenses == 2:
        named = unpack(pvec, finite_source, n_lenses)
        return _build_binary_model(
            pvec,
            coords,
            mag_method,
            use_rho=finite_source,
            orbital_motion=named["linear_motion"],
            t_0_kep=t_0_kep,
        )
    return _build_pspl_model(pvec, coords, mag_method, use_rho=finite_source)


def trajectory(model, times, obs_pos, cache):
    """Source position ``(x, y)`` in the lens frame at ``times`` seen from
    the observer whose Skowron deviations are ``obs_pos``."""
    _clear_mm_satellite_cache()
    traj = model.get_trajectory(
        np.asarray(times, dtype=float),
        satellite_skycoord=_dev_skycoord(obs_pos, cache),
    )
    return np.asarray(traj.x, dtype=float), np.asarray(traj.y, dtype=float)


def magnification(model, times, obs_pos, cache):
    """The model's magnification at ``times`` for that observer (no limb
    darkening: these windows only need to know WHERE the lens departs)."""
    _clear_mm_satellite_cache()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(
            model.get_magnification(
                np.asarray(times, dtype=float),
                satellite_skycoord=_dev_skycoord(obs_pos, cache),
            ),
            dtype=float,
        )


def single_lens_magnification(x, y, named, vbm=None):
    """The binary's EFFECTIVE single lens along the binary model's own
    trajectory: the light curve the event shows away from its anomalies.

    Far from the caustics a binary acts as one point lens, but not the bare
    primary.  A CLOSE companion (s < 1) sits inside the Einstein ring, so the
    whole mass lenses from the center of mass, in theta_E units.  A WIDE one
    (s > 1) leaves the primary's own Einstein radius theta_E/sqrt(1+q), and
    its constant deflection at the primary moves the effective center by
    ``q/((1+q) s)`` toward the companion (the Chang-Refsdal shift; the same
    offset that places the central caustic).  Both are one expression: the
    center is ``x_1 + q/(1+q) min(s, 1/s)`` from the primary at
    ``x_1 = -s q/(1+q)``.  Measured against MulensModel's binary
    magnification for q = 1e-3, along u_0 = 0.02-0.05 trajectories: 1e-4 to
    6e-4 at s = 0.1-0.2 and 5-10, against 3e-3 to 1e-2 for the bare primary,
    whose systematic offset over the whole peak is larger than Roman-grade
    errors and would flag it as an "anomaly".  Near s ~ 1-3 the companion's
    shear still leaves a few 1e-3 -- which is the anomaly being looked for.

    Point source by Paczynski, finite (uniform) source by VBM's
    ``ESPLMag2``, rho rescaled into the same Einstein units.
    """
    s, q = named["s"], named["q"]
    x_eff = -s * q / (1.0 + q) + q / (1.0 + q) * min(s, 1.0 / s)
    scale = np.sqrt(1.0 + q) if s > 1.0 else 1.0
    u = np.hypot(x - x_eff, y) * scale
    if named["rho"] is None:
        u2 = u * u
        return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))
    vbm = vbm or VBMicrolensing.VBMicrolensing()
    rho = named["rho"] * scale
    return np.array([vbm.ESPLMag2(float(ui), rho) for ui in u])


def anomaly_windows(t, z, departure, t_E, peak_window=None):
    """Group anomalous epochs into time windows.

    ``t`` are the data epochs (any order); ``z`` each epoch's departure from
    the effective single-lens model in units of its own error, and
    ``departure`` the model's relative departure ``|A - A_1|/A_1`` there.
    An epoch is anomalous when ``z > ANOMALY_Z``: the data can see the
    companion.  Returns ``[(t_lo, t_hi), ...]`` sorted in time, padded,
    with at most ANOMALY_MAX_WINDOWS kept (largest summed z^2 first).

    The windows form a tree over ANOMALY_DEPARTURE_LEVELS: a window is
    replaced by the windows its own anomalous epochs form when only those
    departing by more than the next level count, when it spans most of
    ``peak_window`` (ANOMALY_MAX_PEAK_COVER -- it would only repeat the
    peak zoom) or when it holds two or more separate features at that level
    (a stellar binary's caustic crossings inside one broad departure).  A
    window holding ONE feature keeps its wider context: DC2018_128's
    planetary dip-and-bump is shown whole.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    departure = np.asarray(departure, dtype=float)
    order = np.argsort(t)
    t, z, departure = t[order], z[order], departure[order]
    seen = np.isfinite(z) & (z > ANOMALY_Z)

    def covers_peak(w):
        return (
            peak_window is not None
            and _peak_cover(w, peak_window) >= ANOMALY_MAX_PEAK_COVER
        )

    def descend(window, level):
        """The windows to show for ``window``; ``level`` indexes the
        departure that selected its epochs (-1: z alone)."""
        children = []
        if level + 1 < len(ANOMALY_DEPARTURE_LEVELS):
            inside = (t >= window[0]) & (t <= window[1])
            selected = seen & (departure > ANOMALY_DEPARTURE_LEVELS[level + 1])
            children = _windows_of(t[inside], z[inside], selected[inside], t_E)
        if covers_peak(window) or len(children) >= 2:
            return [w for c in children for w in descend(c, level + 1)]
        return [window]

    kept = [
        w for top in _windows_of(t, z, seen, t_E) for w in descend(top, -1)
    ]
    kept.sort(key=lambda w: -w[2])
    kept = sorted(kept[:ANOMALY_MAX_WINDOWS], key=lambda w: w[0])
    return [(float(lo), float(hi)) for lo, hi, _ in kept]


def _windows_of(t, z, selected, t_E):
    """Padded ``(lo, hi, sum z^2)`` windows of the ``selected`` epochs of
    the time-sorted ``t``."""
    idx = np.flatnonzero(selected)
    if idx.size == 0:
        return []
    groups = [[idx[0]]]
    for a, b in zip(idx[:-1], idx[1:]):
        between = b - a - 1
        if (
            between >= ANOMALY_SPLIT_POINTS
            or t[b] - t[a] > ANOMALY_MAX_GAP_TE * t_E
        ):
            groups.append([])
        groups[-1].append(b)

    # Pad every group, then merge the ones whose padded windows overlap:
    # with several instruments interleaved, an epoch with a larger error
    # can fall under the threshold in the middle of one caustic crossing
    # and cut it into pieces (examples/ob09020 came back with the same
    # crossing as four windows).  The point count is checked on the
    # merged window, so a fragment of a real feature is not lost.
    padded = []
    for g in groups:
        lo, hi = t[g[0]], t[g[-1]]
        pad = max(ANOMALY_PAD_FRACTION * (hi - lo), ANOMALY_MIN_PAD_DAYS)
        padded.append([lo - pad, hi + pad, float(np.sum(z[g] ** 2)), len(g)])
    merged = [padded[0]]
    for w in padded[1:]:
        last = merged[-1]
        if w[0] <= last[1]:
            last[1] = max(last[1], w[1])
            last[2] += w[2]
            last[3] += w[3]
        else:
            merged.append(w)
    return [
        (lo, hi, score)
        for lo, hi, score, n in merged
        if n >= ANOMALY_MIN_POINTS
    ]


def _peak_cover(window, peak):
    """The fraction of the peak window that ``window`` spans."""
    inter = max(0.0, min(window[1], peak[1]) - max(window[0], peak[0]))
    return inter / (peak[1] - peak[0])


def _frame(xs, ys, pad=0.15, min_half=None):
    """Square limits (equal aspect) around the given points."""
    x = np.concatenate([np.ravel(v) for v in xs])
    y = np.concatenate([np.ravel(v) for v in ys])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    half = 0.5 * max(x.max() - x.min(), y.max() - y.min()) * (1.0 + 2 * pad)
    if min_half is not None:
        half = max(half, min_half)
    half = max(half, 1e-6)
    return (cx - half, cx + half), (cy - half, cy + half)


def _approach_frame(x, y, caustic_xy, lens_xy, rho):
    """Square limits on the source's closest approach to the lens structure
    (the caustic of a binary, the lens itself for a point lens) along the
    trajectory stretch ``(x, y)``: centered between the two closest points,
    wide enough to show both, the source disk, and a quarter of the
    stretch."""
    if caustic_xy is not None:
        sx, sy = caustic_xy
    else:
        sx = np.array([p[0] for p in lens_xy])
        sy = np.array([p[1] for p in lens_xy])
    d2 = (x[:, None] - sx[None, :]) ** 2 + (y[:, None] - sy[None, :]) ** 2
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    dmin = float(np.sqrt(d2[i, j]))
    length = float(np.hypot(np.ptp(x), np.ptp(y)))
    half = max(1.5 * dmin, 0.25 * length, 3.0 * (rho or 0.0), 1e-4)
    cx, cy = 0.5 * (x[i] + sx[j]), 0.5 * (y[i] + sy[j])
    return (cx - half, cx + half), (cy - half, cy + half)


def draw_lens_plane(
    path,
    models,
    named,
    *,
    caustic_epoch,
    observers,
    data_tracks,
    peak_window,
    windows,
    title,
):
    """Write the caustic + trajectory figure.

    ``models``: one MulensModel per source; ``named``: their unpacked
    parameter dicts.  ``observers``: ``[(label, t_grid, obs_pos_grid)]``,
    one trajectory per observer location.  ``data_tracks``:
    ``[(label, color, t, obs_pos)]`` per instrument, drawn as its epochs'
    source positions.  ``windows``: the anomaly windows, each of which gets
    its own panel framed on the source's closest approach to the caustic
    during that window.  The caustic and the source trajectory are drawn by
    MulensModel itself (``Model.plot_caustics`` / ``Model.plot_trajectory``).
    """
    cache = {}
    binary = "s" in named[0]
    rho = named[0]["rho"]

    caustic_xy = None
    if binary:
        models[0].update_caustics(epoch=caustic_epoch)
        cx, cy = models[0].caustics.get_caustics(n_points=CAUSTIC_POINTS)
        caustic_xy = (np.asarray(cx), np.asarray(cy))
        s, q = named[0]["s"], named[0]["q"]
        lens_xy = [(-s * q / (1 + q), 0.0), (s / (1 + q), 0.0)]
    else:
        lens_xy = [(0.0, 0.0)]

    # Each instrument's epochs on each source's trajectory.
    data_xy = [
        [(t,) + trajectory(model, t, obs, cache) for model in models]
        for _, _, t, obs in data_tracks
    ]

    # Frames.  The main panel: the caustics (or the lens) and every
    # source's path through the peak zoom's window.
    xs = [np.array([p[0] for p in lens_xy])]
    ys = [np.array([p[1] for p in lens_xy])]
    if caustic_xy is not None:
        xs.append(caustic_xy[0])
        ys.append(caustic_xy[1])
    for model in models:
        for _, t_grid, obs_grid in observers:
            sel = (t_grid >= peak_window[0]) & (t_grid <= peak_window[1])
            if np.any(sel):
                x, y = trajectory(model, t_grid[sel], obs_grid[sel], cache)
                xs.append(x)
                ys.append(y)
    frames = [_frame(xs, ys, pad=0.1)]
    labels = [title]
    # One panel per anomaly window: where the source passed the caustic
    # while the data in that window were taken (source 0, first observer).
    for lo, hi in windows:
        _, t_grid, obs_grid = observers[0]
        sel = (t_grid >= lo) & (t_grid <= hi)
        x, y = trajectory(models[0], t_grid[sel], obs_grid[sel], cache)
        frames.append(_approach_frame(x, y, caustic_xy, lens_xy, rho))
        labels.append(f"anomaly {len(labels)}: {lo:.2f} - {hi:.2f}")

    n_panels = len(frames)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 5.5), squeeze=False
    )
    axes = axes[0]
    for k, (ax, (xlim, ylim), label) in enumerate(zip(axes, frames, labels)):
        # Limits FIRST: MulensModel sizes the trajectory arrow from the
        # current axis limits, and nothing drawn after may rescale them.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale(False)
        plt.sca(ax)
        if binary:
            models[0].plot_caustics(
                n_points=CAUSTIC_POINTS,
                epoch=caustic_epoch,
                color="red",
                s=0.5,
            )
        for lx, ly in lens_xy:
            ax.plot(lx, ly, marker="+", color="k", ms=8, mew=1.2, ls="none")
        for j, model in enumerate(models):
            for m, (obs_label, t_grid, obs_grid) in enumerate(observers):
                _clear_mm_satellite_cache()
                color = OBSERVER_COLORS[m % len(OBSERVER_COLORS)]
                model.plot_trajectory(
                    times=t_grid,
                    satellite_skycoord=_dev_skycoord(obs_grid, cache),
                    arrow=k == 0,
                    arrow_kwargs={"color": color} if k == 0 else None,
                    color=color,
                    lw=0.8,
                    ls="-" if j == 0 else "--",
                    label=(
                        obs_label
                        if j == 0 and k == 0 and len(observers) > 1
                        else None
                    ),
                )
        for (label_i, color, t, _), tracks in zip(data_tracks, data_xy):
            for j, (_, x, y) in enumerate(tracks):
                ax.plot(
                    x,
                    y,
                    ls="none",
                    marker="o",
                    ms=2.5,
                    color=color,
                    alpha=0.7,
                    label=label_i if j == 0 and k == 0 else None,
                )
        if rho is not None and k > 0:
            # The source disk at the closest approach, so the
            # finite-source smoothing of a crossing can be judged by eye.
            lo, hi = windows[k - 1]
            _, t_grid, obs_grid = observers[0]
            sel = (t_grid >= lo) & (t_grid <= hi)
            x, y = trajectory(models[0], t_grid[sel], obs_grid[sel], cache)
            mid = int(
                np.argmin(np.hypot(x - np.mean(xlim), y - np.mean(ylim)))
            )
            ax.add_patch(
                plt.Circle(
                    (x[mid], y[mid]), rho, fill=False, color="0.3", lw=0.8
                )
            )
        ax.set_xlabel(r"$x$ [$\theta_E$]")
        ax.set_ylabel(r"$y$ [$\theta_E$]")
        ax.set_title(label, fontsize="medium")
    if len(observers) > 1 or data_tracks:
        axes[0].legend(loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# Points along the caustic curve (MulensModel's own default).
CAUSTIC_POINTS = 5000
# One trajectory per observer location: the first (usually Earth) black,
# the others in grays.
OBSERVER_COLORS = ("k", "0.45", "0.65", "0.8")
