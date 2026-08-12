"""Shared helpers for building corner plots from an ArviZ posterior.

Components decide *which* variables and labels go into a corner plot
(component-specific); this module only handles the generic mechanics of
flattening posterior variables into a sample matrix and rendering it.
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _flatten_arrays(items):
    """Shared flattening core for both collectors below.

    ``items`` is a list of (arr, fallback_name, element_labels), where arr
    has the sample dimension last (ArviZ/Parameter.posterior convention);
    every other dimension is flattened into individual columns.
    """
    samples_list = []
    labels = []

    for arr, fallback_name, element_labels in items:
        arr = np.asarray(arr, dtype=float)
        n_samples = arr.shape[-1]

        if arr.ndim == 1:
            samples_list.append(arr)
            labels.append(
                element_labels[0] if element_labels else fallback_name
            )
        else:
            arr_flat = arr.reshape(-1, n_samples)
            n_elem = arr_flat.shape[0]
            if n_elem > 100:
                logger.warning(
                    f"corner_utils: {fallback_name} has {n_elem} "
                    "elements -- possible shape mismatch"
                )
            for i in range(n_elem):
                samples_list.append(arr_flat[i])
                if element_labels and i < len(element_labels):
                    labels.append(element_labels[i])
                else:
                    labels.append(f"{fallback_name}[{i}]")

    if not samples_list:
        return None, []
    return np.array(samples_list).T, labels


def collect_corner_samples(idata, var_specs):
    """Flatten posterior variables into a (n_samples, n_dims) array + labels.

    ``var_specs`` is a list of (var_name, element_labels) pairs. var_name is
    looked up in idata.posterior; missing variables are silently skipped
    (e.g. rho when the event is not finite-source). element_labels is either
    None (falls back to "var_name" / "var_name[i]") or an explicit list with
    one label per vector element.

    Note: a parameter that is a pure physics expression with no sampled
    elements and no user links never gets a pm.Deterministic node (see
    Parameter.build_pymc's ``track_node`` logic), so it never appears in
    idata.posterior at all -- e.g. microlensing's t_E is *always* in this
    category. Use collect_parameter_corner_samples for those.

    Returns (samples, labels), or (None, []) if nothing matched.
    """
    posterior = idata["posterior"]
    items = []
    for var_name, element_labels in var_specs:
        if var_name not in posterior.data_vars:
            continue
        arr = posterior[var_name].stack(sample=("chain", "draw")).values
        items.append((arr, var_name, element_labels))
    return _flatten_arrays(items)


def collect_parameter_corner_samples(param_specs):
    """Flatten Parameter.posterior arrays into a (n_samples, n_dims) array + labels.

    ``param_specs`` is a list of (Parameter, element_labels) pairs. Unlike
    collect_corner_samples (which reads idata.posterior directly), this
    reads each Parameter's already-reconstructed ``.posterior`` attribute,
    populated by System.distribute_posterior(idata) -- the mechanism that
    works for pure-expression parameters with no pm.Deterministic node (see
    collect_corner_samples' docstring) by evaluating the physics expression
    over the tracked posterior. Must be called after
    ``system.distribute_posterior(idata)``; parameters whose ``.posterior``
    is still unset are silently skipped.

    Returns (samples, labels), or (None, []) if nothing matched.
    """
    items = []
    for param, element_labels in param_specs:
        post = getattr(param, "posterior", None)
        if post is None:
            continue
        arr = getattr(post, "values", post)
        items.append((arr, param.label, element_labels))
    return _flatten_arrays(items)


def _label_at(labels, j):
    return labels[j] if j < len(labels) else f"column {j}"


def _drop_undrawable(samples, labels, filename):
    """Drop what corner.corner() cannot draw, naming every omission.

    corner builds its panels from np.histogram, so it raises outright -- and
    takes the WHOLE corner plot down -- on a column with no dynamic range
    ("It looks like the parameter(s) in column(s) N have no dynamic range")
    and on one holding any non-finite value at all ("Axis limits cannot be
    NaN or Inf", since its default range is the column's raw min/max).  Both
    shapes are things a real fit produces: an element pinned with
    ``sigma: 0`` inside an otherwise sampled vector is exactly constant
    across every draw, and a short or stopped run leaves variables that never
    moved.

    Three passes, in this order so each terminates:

      1. columns with no finite draw at all -- else pass 2 would delete
         every row;
      2. rows (draws) still carrying a non-finite value -- such a draw
         cannot be placed in any 2-D panel involving that column anyway;
      3. columns that are constant over the surviving rows.

    Dropping beats passing an artificial ``range``: corner's ``range``
    argument also sets ``force_range``, which changes the axis limits of the
    panels that were fine, and a widened range around a constant would draw
    a spike that reads as a measured posterior.
    """
    finite = np.isfinite(samples)
    keep = [j for j in range(samples.shape[1]) if finite[:, j].any()]
    for j in range(samples.shape[1]):
        if j not in keep:
            logger.warning(
                f"corner plot ({filename}): omitting "
                f"{_label_at(labels, j)} (no finite draws); the remaining "
                "parameters are still plotted"
            )
    samples = samples[:, keep]
    labels = [_label_at(labels, j) for j in keep]
    if samples.shape[1] == 0:
        return samples, labels

    good_rows = np.isfinite(samples).all(axis=1)
    n_bad = int((~good_rows).sum())
    if n_bad:
        logger.warning(
            f"corner plot ({filename}): dropping {n_bad} of "
            f"{samples.shape[0]} draws that hold a non-finite value"
        )
        samples = samples[good_rows]
    if samples.shape[0] == 0:
        return samples, labels

    keep = []
    for j in range(samples.shape[1]):
        col = samples[:, j]
        if col.min() == col.max():
            logger.warning(
                f"corner plot ({filename}): omitting {labels[j]} (constant "
                f"at {float(col.min()):.10g}); the remaining parameters are "
                "still plotted"
            )
        else:
            keep.append(j)
    return samples[:, keep], [labels[j] for j in keep]


def save_corner_plot(samples, labels, filename, max_samples=1000):
    """Render a corner plot of ``samples`` (n_samples, n_dims) to ``filename``.

    Thins to at most ``max_samples`` rows (1000 is enough for visual quality
    and keeps corner.corner()'s memory use bounded). No-op if there is
    nothing to plot.
    """
    import corner

    if samples is None or samples.shape[0] == 0 or samples.shape[1] == 0:
        logger.warning(
            f"save_corner_plot: no samples to plot for {filename}; skipping"
        )
        return

    if samples.shape[0] > max_samples:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
        idx.sort()
        samples = samples[idx]

    samples, labels = _drop_undrawable(samples, labels, filename)
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        logger.warning(
            f"save_corner_plot: nothing left to plot for {filename} after "
            "dropping flat / non-finite parameters and draws; skipping"
        )
        return

    minrank = 0.5 - math.erf(1.0 / math.sqrt(2)) / 2.0
    maxrank = 0.5 + math.erf(1.0 / math.sqrt(2)) / 2.0

    try:
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[minrank, 0.5, maxrank],
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        # PNG (Agg) rasterizes to a fixed pixel buffer bounded by DPI x size,
        # unlike the PDF vector backend which holds every scatter marker path
        # in memory during serialization.
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(
            f"Corner plot failed (sample size {samples.shape[0]}, "
            f"file {filename}): {e}"
        )
