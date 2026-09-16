"""Aggregate inject-and-recover rows into the 7.13.1 pull table.

Reads rows/*.json written by inject_recover.py and reports, per (pipeline,
parameter) and optionally split by SNR or epoch count:

    pull mean  +/- sem      -- tests for BIAS      (consistent with 0?)
    pull width +/- its own  -- tests CALIBRATION   (consistent with 1?)

Both uncertainties matter and the item says so explicitly: "a pull width of
0.5 is as much a finding as a pull mean of 2".

WIDTH ESTIMATOR.  The width is the sample standard deviation of the pulls
with ddof=1, and its uncertainty is sigma/sqrt(2(N-1)) -- the large-N error
on a standard deviation.  With N = 5 per cell that error is 35%, which is
why the default report aggregates over SNR and cadence and the split views
are labelled as indicative only.  Reporting a per-cell width from 5
realizations as if it were measured would be the same overconfidence 1.11.3
filed against the mode weights.

ROBUSTNESS.  A mean and an SD are not robust, and one stuck fit (a
non-converged chain, a mode swap) can dominate both.  So the median and the
normalized median absolute deviation (MAD/0.6745, which estimates the same
sigma for a Gaussian) are printed alongside.  When the two disagree, the
outlier list at the bottom says which rows are responsible; that disagreement
is a finding about the fits, not a nuisance to be smoothed away.

Convergence hygiene: rows carrying ess/rhat for a check are flagged when
rhat > 1.05 or ess < 200, and can be dropped with --require-converged.  A
pull from a chain that did not converge is not a measurement of the
pipeline, and silently keeping it would let a sampler problem masquerade as
a calibration problem.

Usage:
    python3 aggregate_pulls.py                       # everything, pooled
    python3 aggregate_pulls.py --split snr           # by SNR
    python3 aggregate_pulls.py --require-converged
    python3 aggregate_pulls.py --pipeline transit
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

RHAT_MAX = 1.05
ESS_MIN = 200.0


def load_rows(rows_dir, pipeline=None):
    rows = []
    for path in sorted(Path(rows_dir).glob("*.json")):
        try:
            with open(path) as fh:
                row = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print("  SKIP %s: %r" % (path.name, exc))
            continue
        if pipeline and row.get("pipeline") != pipeline:
            continue
        row["_file"] = path.name
        rows.append(row)
    return rows


def _mean_sem(v):
    n = len(v)
    if n == 0:
        return None, None
    m = sum(v) / n
    if n < 2:
        return m, None
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return m, math.sqrt(var / n)


def _sd(v):
    n = len(v)
    if n < 2:
        return None, None
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    # Large-N error on a standard deviation.
    return sd, sd / math.sqrt(2.0 * (n - 1))


def _median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _nmad(v):
    med = _median(v)
    if med is None or len(v) < 2:
        return None
    return _median([abs(x - med) for x in v]) / 0.6745


def collect(rows, require_converged):
    """-> {(pipeline, param, cell): [pulls]}, plus flags and dropped rows."""
    cells = defaultdict(list)
    flagged = []
    dropped = 0
    for row in rows:
        for key, val in row.items():
            if not isinstance(val, dict) or "pull" not in val:
                continue
            pull = val["pull"]
            if pull is None:
                continue
            bad = []
            if val.get("rhat") is not None and val["rhat"] > RHAT_MAX:
                bad.append("rhat=%.3f" % val["rhat"])
            if val.get("ess") is not None and val["ess"] < ESS_MIN:
                bad.append("ess=%.0f" % val["ess"])
            if bad:
                flagged.append((row["_file"], key, ", ".join(bad), pull))
                if require_converged:
                    dropped += 1
                    continue
            cells[(row["pipeline"], key)].append(
                (pull, row.get("snr"), row.get("n_epochs"), row["_file"])
            )
    return cells, flagged, dropped


def report(cells, split):
    print(
        "%-13s %-26s %-9s %4s  %-16s %-16s  %-8s %-8s"
        % (
            "pipeline",
            "parameter",
            "cell",
            "N",
            "pull mean +/- sem",
            "pull width +/- er",
            "median",
            "nMAD",
        )
    )
    print("-" * 118)
    verdicts = []
    for (pipe, param), entries in sorted(cells.items()):
        if split == "snr":
            groups = defaultdict(list)
            for pull, snr, nep, f in entries:
                groups["snr=%g" % snr if snr is not None else "snr=?"].append(
                    pull
                )
        elif split == "epochs":
            groups = defaultdict(list)
            for pull, snr, nep, f in entries:
                groups["n=%s" % nep].append(pull)
        else:
            groups = {"all": [e[0] for e in entries]}

        for cell, pulls in sorted(groups.items()):
            m, sem = _mean_sem(pulls)
            sd, sderr = _sd(pulls)
            med = _median(pulls)
            nmad = _nmad(pulls)
            print(
                "%-13s %-26s %-9s %4d  %-16s %-16s  %-8s %-8s"
                % (
                    pipe,
                    param,
                    cell,
                    len(pulls),
                    "%+.3f +/- %.3f" % (m, sem)
                    if sem is not None
                    else ("%+.3f +/- ?" % m if m is not None else "-"),
                    "%.3f +/- %.3f" % (sd, sderr) if sd is not None else "-",
                    "%+.3f" % med if med is not None else "-",
                    "%.3f" % nmad if nmad is not None else "-",
                )
            )
            if split is None and sem is not None and sderr is not None:
                # Bias: is the mean consistent with 0?  Calibration: is the
                # width consistent with 1?  Both in units of their own error.
                verdicts.append(
                    (pipe, param, len(pulls), m / sem, (sd - 1.0) / sderr)
                )
    if verdicts:
        print(
            "\n%-13s %-26s %4s %12s %14s   %s"
            % (
                "pipeline",
                "parameter",
                "N",
                "bias (sigma)",
                "width-1 (sigma)",
                "reading",
            )
        )
        print("-" * 104)
        for pipe, param, n, zb, zw in verdicts:
            notes = []
            notes.append(
                "BIASED"
                if abs(zb) > 3
                else "bias?"
                if abs(zb) > 2
                else "unbiased"
            )
            notes.append(
                "UNDER-DISPERSED (errors too small)"
                if zw > 3
                else "OVER-DISPERSED (errors too big)"
                if zw < -3
                else "width?"
                if abs(zw) > 2
                else "calibrated"
            )
            print(
                "%-13s %-26s %4d %12.2f %14.2f   %s"
                % (pipe, param, n, zb, zw, "; ".join(notes))
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(HERE / "rows"))
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--split", choices=["snr", "epochs"], default=None)
    ap.add_argument("--require-converged", action="store_true")
    ap.add_argument(
        "--outliers",
        type=float,
        default=3.0,
        help="list any row whose |pull| exceeds this",
    )
    args = ap.parse_args()

    rows = load_rows(args.rows, args.pipeline)
    print("loaded %d rows from %s" % (len(rows), args.rows))
    if not rows:
        return
    by_pipe = defaultdict(int)
    for r in rows:
        by_pipe[r.get("pipeline")] += 1
    print("per pipeline: %s\n" % dict(by_pipe))

    cells, flagged, dropped = collect(rows, args.require_converged)
    if args.require_converged:
        print(
            "dropped %d non-converged check(s) "
            "(rhat > %.2f or ess < %.0f)\n" % (dropped, RHAT_MAX, ESS_MIN)
        )
    report(cells, args.split)

    big = []
    for (pipe, param), entries in sorted(cells.items()):
        for pull, snr, nep, fname in entries:
            if abs(pull) > args.outliers:
                big.append((abs(pull), pipe, param, pull, fname))
    if big:
        print("\n=== |pull| > %g ===" % args.outliers)
        for _, pipe, param, pull, fname in sorted(big, reverse=True):
            print("  %-13s %-26s pull=%+8.2f  %s" % (pipe, param, pull, fname))

    if flagged:
        print("\n=== convergence-flagged checks (%d) ===" % len(flagged))
        for fname, key, why, pull in flagged[:40]:
            print("  %-40s %-26s %-22s pull=%+.2f" % (fname, key, why, pull))
        if len(flagged) > 40:
            print("  ... and %d more" % (len(flagged) - 40))


if __name__ == "__main__":
    main()
