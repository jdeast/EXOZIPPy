"""Stage 5: dedupe the runs and write the blind mode report.

    python 05_report.py [--truth] [--copy-to results/]

Reads every $DC069_RUNS/evidence/*.json (04_evidence.py), collapses runs
that converged to the same basin (same sign u_0, Mahalanobis < 3 on log s,
log q, alpha; the lower-re2 run is kept and duplicate lnZ is NEVER added),
and writes $DC069_RUNS/report/DC2018_069_blind_report.{json,txt}: per mode
s, q, alpha, t_0, u_0, t_E, rho (median, 68%), lnL_max, normalized lnZ +/-
err, the relative probability (softmax over the unique 2L1S modes, and over
those plus the 1L1S_fs baseline), and the reliability flags.

The procedure is blind.  `--truth` appends a SEPARATE, clearly labelled
validation section from the answer key; nothing above it reads the key.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--truth",
        action="store_true",
        help="append the answer-key VALIDATION section",
    )
    ap.add_argument("--data-dir", default=B.DEFAULT_DATA)
    ap.add_argument(
        "--copy-to", default=None, help="also copy the two report files here"
    )
    a = ap.parse_args(argv)

    runs, baseline = [], None
    for p in sorted(B.run_dir("evidence").glob("*.json")):
        r = json.load(open(p))
        if r["arch"] == B.ARCH_2L1S:
            runs.append(r)
        elif r["arch"] == B.ARCH_BASELINE:
            baseline = r
    if not runs:
        raise SystemExit("no 2L1S evidence records; run 04_evidence.py")
    rep = B.build_report(runs, baseline)

    truth = None
    if a.truth:
        B.dc18_path()
        import dc18_common as C

        truth, _cls = C.load_truth(a.data_dir, B.EVENT)
        rep["validation_only_truth"] = {
            k: truth[k] for k in ("s", "q", "t_0", "u_0", "t_E", "rho")
        }
    out = B.run_dir("report")
    out.mkdir(parents=True, exist_ok=True)
    jp = out / ("%s_blind_report.json" % B.NAME)
    tp = out / ("%s_blind_report.txt" % B.NAME)
    B.dump_json(rep, jp)
    tp.write_text(B.report_text(rep, truth))
    print(tp.read_text())
    if a.copy_to:
        Path(a.copy_to).mkdir(parents=True, exist_ok=True)
        for p in (jp, tp):
            shutil.copy(p, a.copy_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
