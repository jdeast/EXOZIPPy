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

**Stage 3: the runs.** Slurm array 54191630, 64 cores each. Settings:
n_temps auto (23 rungs for 2L1S, 22 for 1L1S_fs), tune 300, draws 3000.
All five tasks finished well inside the 6.5 h maxtime.

| run | wall | wrap-up verdict |
|---|---|---|
| m0 | 2.7 h | CONVERGENCE NOT REACHED: burn-in 75%, max Rhat 1.038, min ESS 1230 (both on `log_pi_rel`), 74 chains kept |
| m1 | 2.7 h | the same: Rhat 1.040, ESS 1365 (`log_pi_rel`) |
| m2 | 2.6 h | the same: Rhat 1.038, ESS 1270 (`log_pi_rel`) |
| m3 | 3.0 h | **NOT EQUILIBRATED**: lp still trending, +191 nats between the last two quarters; stuck fraction 0.99; lp Rhat 4.56 |
| 1L1S_fs | 0.7 h | CONVERGENCE NOT REACHED: Rhat 1.034, ESS 1426 (`log_pi_rel`) |

The worst-mixed parameter in m0, m1, m2 and 1L1S_fs is the prior-dominated
`log_pi_rel`. On the last 25% of draws their lp split-Rhat is 1.03-1.05,
with no stuck chains. Every ladder logged "communication-limited" (the
2 Lambda + 1 criterion asks for 26-37 rungs). Stage 4 therefore uses
`--burn-frac 0.75`, the burn-in the wrap-up itself measured.

**Stages 4-5: the blind report** (`results/DC2018_069_blind_report.txt`).
lnZ is normalized. The bridge uses 20000 posterior and 20000 proposal
draws (`04_evidence.py --max-draws`, default 20000; the reason is below).
The bridge limit is re2 <= 0.25.

| mode | seed (s, log q) | posterior s | q | bridge | re2 | lnZ | weight (2L1S) | flags |
|---|---|---|---|---|---|---|---|---|
| m1 | 1.175, -3.86 | 1.168 +0.037/-0.034 | 1.88e-4 | ok | 0.037 | 923259.09 +/- 0.19 | 0.775 | none |
| m2 | 0.525, -3.14 | 0.846 +0.028/-0.030 | 1.94e-4 | ok | 0.045 | 923257.86 +/- 0.21 | 0.225 | drift: log s +0.207, into m0's basin |
| m0 | 0.851, -3.86 | 0.846 | 1.94e-4 | ok | 0.046 | 923260.03 +/- 0.21 | -- | duplicate of m2 (Mahalanobis 0.13): lnZ not added |
| m3 | 1.905, -3.14 | 1.694 | 6.2e-4 | REFUSED (not-converged) | -- | -- | 0 | stuck 0.99, lp Rhat 4.56, not equilibrated |
| 1L1S_fs | -- | -- | -- | ok | 0.052 | 919814.97 +/- 0.23 | 0 (all architectures) | none |

- **2L1S over 1L1S_fs: dlnZ = 3444.1 nats.** The companion is detected
  decisively. lnL_max agrees: 923408.9 against 919950.0.
- **Close vs wide: NOT decided, even though both bridges were accepted.**
  The 0.775 / 0.225 split depends on which duplicate the dedupe keeps. m0
  and m2 are two runs of the same close basin, and their lnZ differ by
  2.2 nats against errors of 0.2. With m0 kept instead, the split would
  be 0.28 wide / 0.72 close. "Why two runs of one basin disagree", below,
  shows that this is posterior coverage, not normalization. Until the
  latent block converges, the ranking of the pair is not trustworthy.
- **m3** did not converge and is excluded. Its refused estimate sits ~6500 nats below m1.

**Why the bridge refused at 4000 draws.** At 4000 draws, every run was
refused (re2 0.37-0.63). The diagnosis (scratch scripts under
`$DC069_RUNS/tools/`):

- re2 is dominated by the PROPOSAL-side term, which falls as 1/N
  (relative variance 200-750 per draw). The posterior-side term is only
  0.013-0.03 (IACT tau ~2-3), so longer chains would not help.
- `estimate_mode_evidences` caps the proposal count at
  `max_posterior_draws`.
- At N = 4000 the re2 estimate is itself heavy-tailed: the same
  configuration gave 0.12 in one realization and 0.37 in another.
- At 20000, re2 is 0.03-0.05 on every converged run. This costs ~40k
  logp evaluations per mode (~5 min on 16 cores).
- Student-t and Warp-III proposals agree with the Gaussian to 0.3 nats
  and do no better at this N. The Laplace approximation is +10-13 nats
  off and unusable.
- The worst raw marginals are `neighbor_flux` (skew -3, excess
  kurtosis +17, piled at its bound) and the source star's teff/feh/av.

**Why two runs of one basin disagree (m0 vs m2, 2.2 nats).**

- **Not the soft barriers.** Each run measures its own barrier steepness,
  and those scales differ up to 10x between runs. But the barrier
  potentials touch almost no posterior mass (mean -3e-4 nats), and
  putting m0's barrier scales into m2's model moves lnZ by 0.0000.
- **Not the normalization.** I mapped 3000 of m2's draws through physical
  space into m0's raw coordinates. The normalized densities agree at
  every point (D = 0.0000 +/- 0.0012) while the whitening Jacobians
  differ by 1.58 nats, so `evidence_log_norm` makes the density exactly
  whitening-invariant.
- **The posterior samples differ.** s, q and alpha are identical (sd of
  log s 0.0153 vs 0.0147), but the prior-dominated source-star block is
  not. Source teff is 3481 +/- 390 in m0 and 3101 +/- 286 in m2 (0.8
  sigma); feh, av and the luminosity differ by 0.4-0.6 sigma. The bridge
  fits its proposal to the draws it is given, so a run that covered the
  latent block differently reports a different lnZ. The wrap-up's own
  verdict agrees: "CONVERGENCE NOT REACHED", with Rhat 1.04 on
  `log_pi_rel`.

**Validation only (answer key; not used above).** The truth is s = 1.138,
log q = -3.763 (q = 1.72e-4), with mirror 1/s = 0.878.

- m1 is on the truth side: s = 1.168, so dlog s = +0.011 and
  dlog q = +0.037.
- m2 (and its duplicate m0) is on the mirror side: s = 0.846, so
  dlog s = -0.016 from the mirror and dlog q = +0.051.

Both members of the degenerate pair were found, each close to its true
value. The accepted weights put 0.775 on the truth side, but as shown
above that split is not robust; the honest verdict is that the procedure
does not rank the pair. The only
non-degenerate alternative (m3) is ruled out. The grid's best node was
the mirror. Seeding both sides is what recovered the truth-side solution.

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
- **The bridge's default draw count is too small.**
  `estimate_mode_evidences` caps the proposal draws at
  `max_posterior_draws` = 4000, which refused every 069 run. 20000
  passes. `04_evidence.py` now defaults to 20000. The library default, or
  a separate `n_proposal` that is not capped by the posterior count,
  should follow.
- **Run-to-run lnZ scatter, from the latent block.** Two runs of one
  basin differ by 2.2 nats against bridge errors of 0.2. The bridge error
  describes one run's draws, not the run-to-run sampling of the
  prior-dominated stellar block. Either more runs per basin with the
  scatter quoted as the error, or longer or better-mixed runs in that
  block, is needed before close vs wide can be ranked.
- **Dedupe rule.** Keeping the lower-re2 duplicate hides that scatter.
  The driver should report every duplicate's lnZ and take their spread as
  a systematic. Pooling the duplicates' draws for one bridge is also an
  option, because the normalization is whitening-invariant (verified
  above).
- A seed can drift into another mode's basin (m2 -> m0), so the dedupe has
  to run on the posteriors, not on the seeds. It does.
- The wrap-up reports "CONVERGENCE NOT REACHED" on `log_pi_rel`, a
  prior-dominated latent. The flags in this pipeline use lp instead.
- 04_evidence.py first selected the run's restart file
  (`*.params.2.yaml`) as its config, because the run writes that file
  into the same directory. Fixed here: the config is the yaml whose stem
  has no dot.
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
