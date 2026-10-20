"""The post-fit robust-likelihood audit: ``<prefix>_outliers.txt``.

For every data file fitted with the hogg mixture (``likelihood: hogg``) it
reports, at the highest-lp draw ("MAP") and at the median draw:

  * ``Gaussian - hogg`` -- the file's summed log-likelihood under a plain
    Gaussian minus under the mixture, the same residuals and the same
    err_scale-inflated sigma.  Strongly negative means the mixture is
    forgiving residuals a Gaussian would charge for.
  * ``n_out`` -- the points with posterior outlier membership > 0.5.
  * ``enrich`` -- ``P(in event window | membership > 0.5) / P(in event
    window)``: how concentrated those points are where the signal lives.
    A missed planetary anomaly sits AT the event (DC2026 validation: 18.6);
    bad data and stellar variability are spread over the baseline (<= 0.9).
  * a FLAG, with its reason, when ``enrich > 3`` with at least 3 such points
    or when the mixture forgives more than 10 nats (thresholds in
    ``components/likelihood.py``).

Why (measured on DC2018 traces, 2026-10): on a no-planet "plateau" fit the
background width pinned at its cap and 2-6 planetary-anomaly points took
membership ~1; each cost ~11 nats instead of ~600, subsidizing the simpler,
wrong model by up to ~1200 nats.  The mixture must stay available (DC2026's
real starspot variability needs it), so this audit is what tells the two
apart after the fit.

Component-agnostic: the per-file numbers come from any component's
``robust_fit_summary(system, point)`` (``Instrument`` implements it), and
the event window from any component's ``event_windows(system, point)``
(``MulensEvent`` implements it).  Nothing here names either.  With no
window-owning component ``enrich`` is reported as ``nan`` and only the
forgiven-nats flag can fire.
"""

import logging

import numpy as np

from ..components import likelihood as robust_support

logger = logging.getLogger(__name__)


def _robust_components(system):
    return [
        c
        for c in system.active_components.values()
        if callable(getattr(c, "robust_fit_summary", None))
    ]


def event_windows(system, point):
    """Every component's event windows at ``point``, concatenated."""
    windows = []
    for comp in system.active_components.values():
        hook = getattr(comp, "event_windows", None)
        if callable(hook):
            windows.extend(hook(system, point) or [])
    return windows


def audit_point(system, point, label):
    """The audit rows (one per hogg file) at one point."""
    windows = event_windows(system, point)
    rows = []
    for comp in _robust_components(system):
        for f in comp.robust_fit_summary(system, point):
            delta = f["gauss_logl"] - f["hogg_logl"]
            if f["times"] is None:
                enrich, n_high = (
                    float("nan"),
                    int(
                        np.sum(
                            f["membership"]
                            > robust_support.MEMBERSHIP_THRESHOLD
                        )
                    ),
                )
            else:
                enrich, n_high = robust_support.enrichment_ratio(
                    f["membership"], f["times"], windows
                )
            rows.append(
                {
                    "point": label,
                    "component": comp.prefix,
                    "name": f["name"],
                    "n": f["n"],
                    "gauss_logl": f["gauss_logl"],
                    "hogg_logl": f["hogg_logl"],
                    "delta_logl": float(delta),
                    "n_high": n_high,
                    "enrich": enrich,
                    "flags": robust_support.robust_flags(
                        delta, enrich, n_high
                    ),
                }
            )
    return rows


def audit_points(system, idata):
    """``{"MAP": point, "median": point}`` from a trimmed, mode-labelled
    trace: the highest-lp valid draw and ``median_draw_point``'s draw.
    "MAP" is the best draw the sampler visited, not an optimizer's; it is
    omitted when the trace carries no ``lp``."""
    from .summary_plot import median_draw_point, point_at_draw

    points = {}
    posterior = idata.posterior
    ss = getattr(idata, "sample_stats", None)
    if ss is not None and "lp" in ss.data_vars:
        lp = np.asarray(ss["lp"].values, dtype=float)
        if "mode" in posterior:
            lp = np.where(
                np.asarray(posterior["mode"].values) >= 0, lp, -np.inf
            )
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        if np.isfinite(lp).any():
            chain, draw = np.unravel_index(int(np.argmax(lp)), lp.shape)
            points["MAP"] = point_at_draw(system, idata, chain, draw)
    points["median"] = median_draw_point(system, idata)[0]
    return points


def has_robust_audit(system):
    """True when any component fitted a file with the hogg mixture."""
    return any(
        "hogg" in (getattr(c, "likelihood_kinds", None) or ())
        for c in _robust_components(system)
    )


def format_audit(rows):
    """The ``<prefix>_outliers.txt`` text."""
    lines = [
        "# Robust (hogg mixture) likelihood audit, per data file.",
        "# dlogL = Gaussian logL - hogg logL (same residuals, same "
        "err_scale); n_out = points with outlier membership > "
        f"{robust_support.MEMBERSHIP_THRESHOLD:g};",
        "# enrich = P(in event window | membership > "
        f"{robust_support.MEMBERSHIP_THRESHOLD:g}) / P(in event window).  "
        f"FLAG when enrich > {robust_support.ENRICH_FLAG:g} with n_out >= "
        f"{robust_support.ENRICH_MIN_POINTS}, or dlogL < "
        f"-{robust_support.FORGIVEN_NATS_FLAG:g}.",
        "# See src/exozippy/components/instrument.md, 'Robust observation "
        "likelihoods'.",
        f"{'point':<7} {'file':<32} {'n':>7} {'gaussL':>14} {'hoggL':>14} "
        f"{'dlogL':>10} {'n_out':>6} {'enrich':>8}  flag",
    ]
    for r in rows:
        name = f"{r['component']}.{r['name']}"
        lines.append(
            f"{r['point']:<7} {name:<32} {r['n']:>7d} "
            f"{r['gauss_logl']:>14.3f} {r['hogg_logl']:>14.3f} "
            f"{r['delta_logl']:>10.3f} {r['n_high']:>6d} "
            f"{r['enrich']:>8.2f}  "
            + ("FLAG: " + "; ".join(r["flags"]) if r["flags"] else "-")
        )
    return "\n".join(lines) + "\n"


def write_robust_audit(system, idata, prefix, log=logger):
    """Run the audit and write ``<prefix>_outliers.txt``; WARN per flag.

    Returns the rows, or None (and writes nothing) when no file used the
    mixture -- a fit without ``likelihood: hogg`` gets no new output.
    """
    if not has_robust_audit(system):
        return None
    rows = []
    for label, point in audit_points(system, idata).items():
        rows.extend(audit_point(system, point, label))
    path = str(prefix) + "_outliers.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_audit(rows))
    for r in rows:
        if r["flags"]:
            log.warning(
                "ROBUST LIKELIHOOD (%s draw): %s.%s -- %s.  The mixture may "
                "be absorbing a real signal: inspect the residuals in the "
                "event window, compare against a fit with "
                "'out_frac_prior: beta' or without the mixture (see %s).",
                r["point"],
                r["component"],
                r["name"],
                "; ".join(r["flags"]),
                path,
            )
    return rows
