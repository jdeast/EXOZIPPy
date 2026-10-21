"""Does THIS code tree reproduce a trace's stored lp?  (run it first)

    python lpcheck.py <fit_dir> [--n 50] [--seed 1] [--out result.json]

The RMDC26 traces were sampled by an older tree (wt/launch1015, plus PRs
#467 and #472) while the evidence normalization this report needs
(System.evidence_log_norm, PR #465) exists only on master, whose config
engine was rewritten in between.  A bridge lnZ integrates the model logp
evaluated by WHICHEVER tree runs the report, at the trace's raw draws, so
it is only an evidence for the sampled posterior if that tree's logp is the
sampler's logp at those draws.  This script checks exactly that and nothing
else:

  1. rebuild the System from the fit's own config (and params file) with
     the tree on PYTHONPATH,
  2. restore the whitening the trace was sampled under (<prefix>_whitening
     .json; never re-measured -- raw draws only decode under it),
  3. evaluate model.compile_logp(jacobian=True) -- the density the sampler
     targets in raw space -- at `--n` random stored draws (T = 1 chains,
     second half), and
  4. compare with the trace's own sample_stats.lp at the same draws.

The verdict is PASS when max |dlp| < 0.01 nat (the task asks for << 0.1).
It imports only APIs present in both trees, so the same file runs under
the old tree as a control: PYTHONPATH=<tree>/src python lpcheck.py ...

Light work, but it builds a model: run it on `srun -p test`.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def config_path(fit_dir):
    """The fit's system config: RMDC26_<ev>.yaml (a stem with no dot; the
    params file and the run's restart file carry one)."""
    d = Path(fit_dir)
    cands = [p for p in sorted(d.glob("RMDC26_*.yaml")) if "." not in p.stem]
    if len(cands) != 1:
        raise SystemExit(f"{d}: expected one RMDC26_*.yaml, found {cands}")
    return cands[0]


def stage(fit_dir, work_dir):
    """Copy a fit's config + params into `work_dir` and link its trace and
    whitening there, so nothing this process does can write into the
    campaign's directory (it is another agent's live run).  Returns the
    staged config path.  `prefix:` is pointed at `work_dir`; it is not part
    of the structural fingerprint, and the data paths are absolute."""
    import yaml

    cfg = config_path(fit_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_load(cfg.read_text())
    src_prefix = Path(text["prefix"])
    if not src_prefix.is_absolute():
        src_prefix = cfg.parent / src_prefix
    new_prefix = work / src_prefix.name
    raw = cfg.read_text().replace(str(text["prefix"]), str(new_prefix))
    (work / cfg.name).write_text(raw)
    pf = text.get("parameter_file")
    if pf:
        src = Path(pf) if Path(pf).is_absolute() else cfg.parent / pf
        (work / Path(pf).name).write_text(src.read_text())
    for suffix in ("_trace.nc", "_whitening.json"):
        link = Path(str(new_prefix) + suffix)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(str(src_prefix) + suffix)
    return work / cfg.name


def build(fit_dir, work_dir):
    """(system, model, config, prefix) for a fit directory, staged into
    `work_dir`, with the whitening restored from the trace's own file."""
    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace
    from exozippy.yamlio import load_system_config

    cfg = stage(fit_dir, work_dir)
    os.chdir(cfg.parent)
    config = load_system_config(cfg.name)
    prefix = config["prefix"]
    system = System(config)
    system.prepare()
    model = system.build_model()
    state = restore_whitening_for_trace(
        system, prefix + "_whitening.json", prefix + "_trace.nc"
    )
    if state != "restored":
        raise SystemExit(f"whitening not restored ({state}): cannot check")
    return system, model, config, prefix


def draw_point(post, names, shapes, c, d):
    return {
        n: np.asarray(post[n].values[c, d], dtype=float).reshape(shapes[n])
        for n in names
    }


def compare_lp(model, post, lp, n=50, seed=1):
    """Evaluate the model's raw-space logp at `n` random finite stored draws
    (second half of the T = 1 chains) and compare with the stored lp.

    `post` is the posterior Dataset (or DataTree node) and `lp` the
    (chain, draw) sample_stats lp.  Returns a dict with the per-draw rows
    and max |dlp|; the caller decides what tolerance means PASS.
    """
    ip = model.initial_point()
    names = list(ip)
    missing = [k for k in names if k not in post.data_vars]
    if missing:
        raise ValueError(f"trace lacks model value vars: {missing}")
    shapes = {k: np.shape(ip[k]) for k in names}
    with model:
        f = model.compile_logp(jacobian=True)
    rng = np.random.default_rng(seed)
    lp = np.asarray(lp, dtype=float)
    n_chain, n_draw = lp.shape
    rows = []
    tries = 0
    while len(rows) < n and tries < 100 * n:
        tries += 1
        c = int(rng.integers(n_chain))
        d = int(rng.integers(n_draw // 2, n_draw))
        if not np.isfinite(lp[c, d]):
            continue
        v = float(f(draw_point(post, names, shapes, c, d)))
        rows.append((c, d, float(lp[c, d]), v, v - float(lp[c, d])))
    dlp = np.array([r[4] for r in rows])
    return {
        "n_value_vars": len(names),
        "n": len(rows),
        "lp_trace_median": float(np.median([r[2] for r in rows])),
        "dlp_max_abs": float(np.max(np.abs(dlp))),
        "dlp_median": float(np.median(dlp)),
        "dlp_std": float(np.std(dlp)),
        "verdict": "PASS" if np.max(np.abs(dlp)) < 0.01 else "FAIL",
        "rows": rows,
    }


def check(fit_dir, work_dir, n=50, seed=1):
    import xarray as xr

    t0 = time.time()
    system, model, config, prefix = build(fit_dir, work_dir)
    t_build = time.time() - t0
    tree = os.path.dirname(os.path.dirname(sys.modules["exozippy"].__file__))
    dt = xr.open_datatree(prefix + "_trace.nc")
    attrs = dict(dt.attrs)
    out = {
        "fit_dir": str(fit_dir),
        "tree": tree,
        "trace_git_commit": attrs.get("exozippy_git_commit"),
        "trace_git_describe": attrs.get("exozippy_git_describe"),
        "trace_git_dirty": attrs.get("exozippy_git_dirty"),
        "build_s": t_build,
    }
    out.update(
        compare_lp(
            model,
            dt["posterior"].to_dataset(),
            dt["sample_stats"]["lp"].values,
            n,
            seed,
        )
    )
    if hasattr(system, "evidence_log_norm"):
        out["log_norm"] = float(system.evidence_log_norm())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_dir")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--work",
        default="/n/holylabs/jjohnson_lab/Lab/rmdc26_report/work/lpcheck",
        help="staging directory (a subdirectory per fit is made)",
    )
    a = ap.parse_args(argv)
    fit_dir = Path(a.fit_dir).resolve()
    res = check(fit_dir, Path(a.work) / fit_dir.name, a.n, a.seed)
    txt = json.dumps(res, indent=1)
    if a.out:
        with contextlib.suppress(FileExistsError):
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")
    print(
        "%s [%s]: |dlp| max %.3g, median %+.3g, sd %.3g over %d draws -> %s"
        % (fit_dir, res["tree"], res["dlp_max_abs"], res["dlp_median"],
           res["dlp_std"], res["n"], res["verdict"])
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
