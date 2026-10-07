# KMT-2018-BLG-0087 ("obj3")

A binary-lens, point-source (2L1S) event observed by KMTNet, fit here from six
I-band light curves: two fields (BLG14, BLG15) at each of the three sites
(CTIO, SAAO, SSO).

`obj3` is an index into A. Stoneew's event list, **not** a solution index --
the sibling directories in that list are `ob171146_obj1`, `kb160696_obj2`,
`kb180173_obj4`, `kb180030_obj5`, `ob181428_obj6`. The two competing
solutions here are named `Close` and `Close_alt`.

Event position: 17:37:18.48 -27:49:55.42 (J2000) = 264.32700, -27.83206 deg,
galactic (l, b) = (359.97, +2.14).

## Two solutions, and which one leads

Both are static (no parallax was fit), from the MMEXOFAST run
`20260624170138` after error renormalization, over 3094 data points:

| solution    | t_0 (HJD')  | u_0   | t_E (d) | s     | q        | alpha (deg) | chi2    |
|-------------|-------------|-------|---------|-------|----------|-------------|---------|
| `Close_alt` | 8281.7275   | 0.524 | 4.554   | 0.910 | 2.368e-3 | -105.71     | 3045.54 |
| `Close`     | 8281.7365   | 0.509 | 4.631   | 0.634 | 2.878e-3 | -106.42     | 3050.43 |
| PSPL        | 8281.7500   | 0.526 | 4.544   | --    | --       | --          | 3203.61 |

`Close_alt` is preferred by `delta chi2 = 4.9`, which is not decisive, so
both are seeded. `source.Source.*` and friends carry list-valued `initval`s
(P4 multi-seed sampling), ordered `[Close_alt, Close]`. That **reverses** the
order in the source JSON, deliberately: bounds are not per-seed and always
come from seed 0, so the preferred solution should anchor them.

The binary model is worth 158 chi2 over the point lens, so the anomaly is
real; the ambiguity is only in which close-topology solution describes it.

## What is fit, and what is deliberately not

**Point source.** The MMEXOFAST fit carried `rho` and used VBBL across the
anomaly, but these data do not measure it: `log_rho` has a marginal sigma of
**3.3 dex in both solutions** (-6.12 -3.41/+3.28 and -6.47 -3.34/+3.34), so
the posterior simply fills its prior and both point estimates sit near the
floor. Fitting a finite source would add an unconstrained dimension rather
than measure anything, so this ships as `finite_source: False` -- the same
call `DC2018_128` makes, for the same reason. Do not read the ~1.7 dex gap
between the two seeded `rho` values as a difference between the solutions.

The `rho` seed is still kept and is not dead weight: with `finite_source:
False` the relaxation engine back-solves it through `rho = theta_star/theta_E`
into the stellar chain, so it informs the starting `theta_E` and through it
`t_E`.

**Limb darkening is declared but unused.** The band block declares a linear
law; with a point source nothing reads a limb-darkening coefficient, and the
model has no LD free parameter (confirmed: the built model's 14 free RVs
contain none). For anyone turning `finite_source: True` back on, the value
MMEXOFAST used is in its `ModelConfig` (`limb_darkening_coeffs_gamma =
{'I': -0.497}`, with `limb_darkening_coeffs_u = None`).

**Do not copy that number.** The sign is wrong: `band.u1` is the ordinary
linear coefficient `u`, related to `gamma` by `u = 3*gamma/(2 + gamma)`, so
`gamma = -0.497` gives `u = -0.99` -- a source that brightens toward the
limb, which is unphysical. The usual I-band value is `gamma = +0.5103`
(`u = +0.610`), which is what MMEXOFAST's own
`examples/use_case_03_ob1045.py` passes and what Shin et al. 2018 Table 1
quotes for the same band. The stored negative is either a sign slip in the
driver or a deliberate test; the solutions in `results.tex` were fit with
it either way, which is harmless here only because this is a point-source
fit where nothing reads it. Declare the positive value if finite source is
turned on.

**Error renormalization.** The per-dataset `errfacs` are mapped one-to-one
onto `err_scale` (`sigma = err * err_scale`). They are
**initvals, not pins**: `err_scale` stays free, as in every other shipped
fit. MMEXOFAST derived them on `Close_alt` with the anomaly-window points
protected, which is why one set serves both seeds.

| KMTC14 | KMTS14 | KMTA14 | KMTC15 | KMTS15 | KMTA15 |
|--------|--------|--------|--------|--------|--------|
| 1.2303 | 1.2519 | 1.4598 | 1.2599 | 1.3422 | 1.4300 |

## Conventions

Both of the standard traps were checked against
`src/exozippy/components/mulensing/conventions.md` rather than assumed:

- **`alpha` transfers with no 180 degree shift.** MMEXOFAST calls
  MulensModel, and C18 records that MulensModel's `alpha` is identical to
  ours -- the "shifted by 180 deg" note in MulensModel's own `Trajectory`
  docstring does not hold. Verified empirically here, not just read: building
  this model with the published `alpha = -105.71` gives a start logp **155.5
  nats better** than the same model with `alpha + 180`.
- **`q` is not inverted.** C14's inversion, and the 180 degree `alpha` shift
  that accompanies it, applies only when the lighter body is listed first.
  `q = 2.4e-3 << 1` with `star.Lens` leading the lens block, so the published
  value transfers as is.

## Data

`L_data/n20260623.I.KMT{A,C,S}{14,15}.pys` are KMTNet pySIS output: three
whitespace columns, **HJD-2450000** / I mag / mag err, already sorted. Every
other shipped example ships full Julian Dates, so these are read with
`time_offset: 2450000.0`, which the loader applies before anything else.

No `time_frame`/`time_scale` conversion is requested. The published solution
is quoted in the files' own time system, and MMEXOFAST treats pySIS HJD as
plain HJD, so converting here would silently move `t_0` away from the value
being reproduced. `ob09020` reads its `.rv` files the same way.

There is no `mmexofast:` key: MMEXOFAST was stripped from the codebase in
PR #361 and the key now raises at the boundary. Nothing is lost here -- the
params file supplies the whole solution, and the numbers below were taken
from MMEXOFAST output that is already committed in this directory.

## Provenance

Everything above comes from files committed in this directory, not from an
external run:

- `test_output/ob180087_20260624170138_renorm_exozippy_init.json` --
  parameters, sigmas, `errfacs`, `mag_methods`
- `test_output/ob180087_20260624170138_renorm_results.tex` -- the same
  solution tabulated, plus per-dataset source/blend I magnitudes
- `test_output/*.pkl`, `pkl_xtra/*.pkl` -- MMEXOFAST restart pickles; the
  coordinates and the limb-darkening gamma were read from the `ModelConfig`
  they carry

The logs in `test_output/` point at `/scratch/public/sao/astoneew/` on the
Hydra cluster, under the directory's older name `ob180087_obj3`. That path is
not needed to run this example.

## Running it

    cd examples/kb180087_obj3 && poetry run exozippy kb180087_obj3.yaml

The sampler is `ptde_async`: the binary magnification backend
(`VBMDirectMagOp`) has **no gradient**, so this event needs PTDE or another
gradient-free sampler. `eval_timeout: 10` is set because VBMicrolensing can
hang for some `(s, q)`.
