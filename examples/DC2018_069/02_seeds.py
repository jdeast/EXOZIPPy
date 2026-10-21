"""Stage 2: grid basins -> candidate modes -> one 2L1S config per mode.

    python 02_seeds.py [--n-temps auto --tune 300 --draws 3000 \\
                        --maxtime 23400] [--baseline]

Reads $DC069_RUNS/grid/DC2018_069_grid.npz (01_grid.py) and writes

    modes/candidates.json         the pre-deduped candidate list
    modes/m<k>/seed.yaml          that mode's grid seed (seed_fragment, K=1)
    modes/m<k>/DC2018_069_2L1S.yaml (+ .params.yaml, .sed.yaml)
    modes/1L1S_fs/...             --baseline: the no-companion architecture
    modes/configs.txt             one config per line, for 03_fit_modes

Every config comes from dc18_arch_config.build, so all of them -- every
mode and the baseline -- carry the IDENTICAL canonical prior set (checked
with dc18_arch_config.check_shared here); a seed contributes initvals only.

Run settings per mode (PROVISIONAL, pending the temperature-count study on
event 194): n_temps auto, tune 300, draws 3000, maxtime 23400 s (6.5 h)
under an 8 h wall, seed_polish false, overdisperse true.  n_temps 1 is
refused: on 194 a single temperature did not mix even inside the basin
(Rhat ~1.8 after 30k draws).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402

B.dc18_path()


def main(argv=None):
    import dc18_arch_config as A
    import yaml
    from dc18_grid_seed import NAMES

    from exozippy.components.mulensing.gridsearch import (
        GridResult,
        seed_fragment,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-temps", type=A.n_temps_arg, default="auto")
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--maxtime", type=int, default=23400)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--dchi2-max", type=float, default=B.DCHI2_MAX)
    ap.add_argument("--max-modes", type=int, default=B.MAX_MODES)
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also write the 1L1S_fs (no companion) config",
    )
    ap.add_argument("--data-dir", default=B.DEFAULT_DATA)
    a = ap.parse_args(argv)
    if a.n_temps == 1:
        ap.error(
            "n_temps 1 does not mix on these posteriors (event 194: "
            "Rhat ~1.8 after 30k draws); use auto or >= 4"
        )

    grid = GridResult.from_npz(str(B.run_dir("grid", B.NAME + "_grid.npz")))
    cands = B.candidate_modes(grid, a.dchi2_max, a.max_modes)
    ev = A.Event(a.data_dir, B.EVENT)
    run = dict(
        n_temps=a.n_temps,
        tune=a.tune,
        draws=a.draws,
        maxtime=a.maxtime,
        cores=a.cores,
        t_max=200.0,
        hogg=False,
    )

    print(
        "%d candidate mode(s) (dchi2 <= %g, grid min-sep %s nodes):"
        % (len(cands), a.dchi2_max, grid.default_min_sep())
    )
    paths = []
    for c in cands:
        d = B.run_dir("modes", c["mode"])
        d.mkdir(parents=True, exist_ok=True)
        frag = seed_fragment(
            [c], NAMES, seed_polish=False,
            header=["# %s %s: %s of the blind grid" % (B.NAME, c["mode"],
                                                       c["origin"])],
        )  # fmt: skip
        (d / "seed.yaml").write_text(frag)
        seed = yaml.safe_load(frag)
        paths.append(A.build(ev, B.ARCH_2L1S, str(d), run, seed))
        print(
            "  %s  %-22s s=%.4f log q=%.2f alpha=%6.1f u_0=%+.4f "
            "dchi2=%.1f"
            % (
                c["mode"],
                c["origin"],
                c["s"],
                c["log_q"],
                c["alpha"],
                c["u_0"],
                c["dchi2"],
            )
        )
    if a.baseline:
        d = B.run_dir("modes", B.ARCH_BASELINE)
        paths.append(A.build(ev, B.ARCH_BASELINE, str(d), run))
    digest = A.check_shared(paths)
    print(
        "shared priors identical across %d configs: %s"
        % (len(paths), digest[:16])
    )
    B.dump_json(
        {"grid": str(B.run_dir("grid", B.NAME + "_grid.npz")),
         "d_log_s": grid.d_log_s, "d_log_q": grid.d_log_q,
         "dchi2_max": a.dchi2_max, "run": run, "candidates": cands},
        B.run_dir("modes", "candidates.json"),
    )  # fmt: skip
    B.run_dir("modes", "configs.txt").write_text("\n".join(paths) + "\n")
    print("wrote %s" % B.run_dir("modes", "configs.txt"))


if __name__ == "__main__":
    main()
