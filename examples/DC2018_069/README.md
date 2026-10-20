# DC2018_069: a blind binary-lens (2L1S) fit, end to end

This example fits event 069 of the 2018 Roman (WFIRST) Data Challenge with
the complete blind procedure:

    grid search over (s, q, alpha)  ->  candidate modes
      ->  one sampling run per mode  ->  dedupe
      ->  normalized evidence per mode  ->  mode report with weights

The event is a clean planetary recovery. Its anomaly is a central-caustic
crossing, which suffers the close/wide (s <-> 1/s) degeneracy: the
solution and its mirror image both fit. That makes 069 a good test of the
procedure, because the grid's best node is the MIRROR of the truth (see
`components/mulensing/gridsearch.py`). A search that reported only its best
fit would get the topology wrong without any warning. This example is also
the acceptance test for the blind-pipeline driver: the driver has to
reproduce what these scripts decide (`blind.py` holds every decision, and
`tests/test_dc2018_069_blind.py` pins them).

**Blind** means that no stage reads the answer key. The truth appears in
exactly one place, a separate section at the end of the report labelled
VALIDATION ONLY, which is written only when you pass `--truth` to stage 5.

## Before you start

Read `examples/DC2018/README.md`. Its "Architecture configs" section gives
the canonical prior set that every config here is built with, and
`src/exozippy/components/mulensing/mulensing.md` ("The binary mode
finder") explains why the topology has to be found before the fit.

- Environment: `source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh;
  conda activate exozippy`, with `PYTHONPATH=<repo>/src`.
- Data: the full challenge release (`lc/ulwdc1_069_{W149,Z087}.txt`).
  Set `$DC18_DATA` to point at it; on Cannon it is
  `/n/holylabs/jjohnson_lab/Lab/dc18_full/data-challenge-1`.
- Run output: everything goes under `$DC069_RUNS`, default
  `/n/holylabs/jjohnson_lab/Lab/dc069_runs`. Each trace is about 1 GB, so
  never put this in `$HOME`. Only the small final report is copied back
  into `results/`.
- Light stages (1 when reusing the grid, 2, 4 and 5) go on
  `srun -p test`, never on a login node. The fits (stage 3) go through
  `sbatch`.

## The walkthrough

    cd examples/DC2018_069
    export DC18_DATA=/n/holylabs/jjohnson_lab/Lab/dc18_full/data-challenge-1
    T="srun -p test -c 16 --mem=64G -t 2:00:00"

### Stage 1 -- the grid (`01_grid.py`)

    $T python 01_grid.py                 # reuse grid/sqgrid30_069.npz (8 KB)

This writes `$DC069_RUNS/grid/DC2018_069_grid.npz` and the chi2 surface
`DC2018_069_chi2.png` (a copy is in `results/`). The local minima are
circled and the added mirrors are squares.

The shipped grid comes from the 30-event sweep: 21 x 8 nodes over
log s in [-0.7, 0.7] and log q in [-6, -1], with 12 alpha starts per node.
At every node, t_0, u_0, t_E and alpha are re-minimized, and rho is
optimized separately in [1e-4, 1e-2]. The grid uses W149 only and took
**42.7 core-h** (16 cores x 2.67 h).

To compute the grid from scratch instead, use the package path
(`gridsearch.curves_from_system -> pspl_start -> grid_search`). This uses
both bands, read the way the fit reads them, and the same box:

    sbatch -p sapphire -c 64 --mem=64G -t 4:00:00 --wrap \
      "python 01_grid.py --compute --ncpu 64 --maxtime 12000"

Budget roughly 45-130 core-h per event at 64 cores. The low end is the
measured W149-only cost above. Both bands add epochs, and the cost per
node scales with the epoch count. Always pass `--maxtime` below the wall
time: the grid evaluates its nodes in shuffled order, so a grid that
stops early is a uniform subsample and is still usable. Do not coarsen
the grid to save time (see mulensing.md).

### Stage 2 -- candidate modes and configs (`02_seeds.py`)

    $T python 02_seeds.py --baseline

This takes the grid's distinct local minima within `DCHI2_MAX` (500) of
the best node, plus the s <-> 1/s mirror of each one
(`GridResult.mirror_of`). The list is then de-duplicated: a candidate
inside the grid's own basin-suppression radius of an accepted one is
dropped. At most `MAX_MODES` (5) are kept.

For each mode the script writes a seed (`seed_fragment`, K = 1) and a 2L1S
config, both through `dc18_arch_config.build`. Every mode therefore
carries the identical canonical prior, and the script runs `check_shared`
to confirm it. A seed sets initvals only, never a prior. `--baseline`
also writes the no-companion `1L1S_fs` architecture as an evidence
baseline. All the configs are listed in `modes/configs.txt`.

### Stage 3 -- one PTDE run per mode (`03_fit_modes.sbatch`)

    sbatch --array=1-$(wc -l < $DC069_RUNS/modes/configs.txt) 03_fit_modes.sbatch

Each array task runs one config on 64 cores on sapphire, with an 8 h wall
and `--signal=B:TERM@1800`. The script forwards SIGTERM to the fit, so
the sampler can reach its save path. The task copies the config with
`sampler.cores` set from the cores Slurm actually granted.

**Run settings (PROVISIONAL).** `n_temps: auto`, `tune: 300`,
`draws: 3000`, `sampler.maxtime: 23400` (6.5 h, which leaves 1.5 h for
the wrap-up), `seed_polish: false`, `overdisperse: true`. These come from
the temperature-count study on event 194, which is still running:

- 24 temperatures held the basin and converged at tune 300 + draws 3000
  (about 5 h).
- 1 temperature did not mix even inside the basin (Rhat ~1.8 after 30k
  draws, acceptance 0.02). `02_seeds.py` refuses `--n-temps 1`.
- 4 temperatures are being tested.

All of these are arguments to `02_seeds.py` (`--n-temps`, `--tune`,
`--draws`, `--maxtime`). Revisit them when the study reports.

### Stage 4 -- per-mode normalized evidence and flags (`04_evidence.py`)

    $T python 04_evidence.py m0 m1 ... 1L1S_fs

This works offline from each trace. For each run it rebuilds the model,
restores the run's whitening (it is never re-measured), drops the first
half of the draws, and labels every remaining draw as belonging to that
run's mode. If `identify_modes` splits the run into sub-modes, each
sub-mode gets its own bridge and the results are combined with a
logsumexp. The evidence is **normalized** lnZ, from
`estimate_mode_evidences(log_norm=system.evidence_log_norm())`, so it can
be compared across runs and architectures.

The script also computes these reliability flags:

- stuck chains: more than 10% of chains sitting more than 20 nats below
  the best chain;
- lp split-Rhat above 1.1;
- bridge status (refused or partial) and re2 above 0.1;
- drift out of the seeded basin: more than one grid node in log s or
  log q, a crossing of s = 1, or a flip in the sign of u_0;
- the run hit maxtime.

### Stage 5 -- the report (`05_report.py`)

    $T python 05_report.py --truth --copy-to results/

Runs that converged to the same basin are de-duplicated. Two runs count as
the same basin when sign(u_0) agrees and the Mahalanobis distance between
their (log s, log q, alpha) posteriors is under 3. The run with the lower
re2 is kept, and the duplicate's lnZ is never added.

The report is written as `DC2018_069_blind_report.{json,txt}`. For each
mode it gives:

- s, q, alpha, t_0, u_0, t_E and rho (median and 68% interval);
- lnL_max;
- lnZ +/- err;
- the relative probability: a softmax over the unique 2L1S modes, and a
  second softmax that also includes the 1L1S_fs baseline;
- the flags.

## Results for 069

**Stages 1-2 (done).** From the shipped grid (`results/DC2018_069_chi2.png`,
`results/candidates.json`), the four candidate modes are:

| mode | origin | s | log q | alpha | grid dchi2 |
|---|---|---|---|---|---|
| m0 | minimum 0 | 0.851 | -3.86 | 288.7 | 0 |
| m1 | minimum 1 | 1.175 | -3.86 | 288.7 | 17.6 |
| m2 | minimum 2 | 0.525 | -3.14 | 287.6 | 272.8 |
| m3 | mirror of minimum 2 | 1.905 | -3.14 | 287.6 | 280.7 |

m0 and m1 are the close/wide pair. `mirror_of` returns None for both: at
|log s| = 0.07 the mirror window overlaps the seed's own basin, so the
mirror was not added. It did not need to be, because each side is a grid
minimum in its own right. The next basin (s = 3.63, dchi2 992) falls
outside the cut. Every config, including the 1L1S_fs baseline, carries
one shared prior (sha256 e2ccca2a966572b0).

**Stages 3-5: PENDING.** The array is Slurm job 54191630 (5 tasks: m0-m3
and 1L1S_fs) and is queued on sapphire. The mode table, the weights and
the comparison with truth will be added to `results/` when the runs
finish.

## Gaps found (for the blind-pipeline driver)

- `outputs/evidence.py` sizes its logp pool with `mp.cpu_count()`, which
  inside a Slurm allocation is the node's core count, not the grant.
  `04_evidence.py` works around this by patching it to the affinity
  mask, which is the same fix `gridsearch.default_grid_cores` already
  uses.
- The wrap-up computes per-mode evidence only for a multimodal posterior,
  and only over that run's own modes. A single-mode run therefore gets
  no lnZ, which is why stage 4 is a separate offline pass. The driver
  should own this pass.
- On the earlier single 069 fit (24 temperatures, 8000 draws, hogg on),
  the bridge refused both sub-modes that `identify_modes` found
  (not-converged; re2 = 1.3). The Gaussian proposal fits this posterior
  poorly. Refused estimates are kept only as `lnZ_indicative` and are
  never used for a weight. Whether the Gaussian runs here pass is the
  open question for stage 4.
- `dc18_highmag_grid2.py` (the script that made the shipped grid) reads
  the answer key in order to print a truth comparison. The search itself
  is blind, but the driver should use the package path that
  `01_grid.py --compute` shows.

## Files

| file | what |
|---|---|
| `blind.py` | paths, thresholds and every decision (pure functions) |
| `01_grid.py` ... `05_report.py`, `03_fit_modes.sbatch` | the stages |
| `grid/sqgrid30_069.npz` | the reused 168-node grid (8 KB) |
| `results/` | the chi2 PNG, candidate list and final blind report |
| `tests/test_dc2018_069_blind.py` | stage 2/4/5 logic, no sampler |
