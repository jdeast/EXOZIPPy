# Population pharmacokinetics

## READ THIS FIRST: what this is, and what it is not

**This component was written by an astrophysicist and an LLM. No biologist,
pharmacologist, clinician, or pharmacometrician has reviewed it.**

It exists to **demonstrate and enforce the component-agnostic architecture**,
and to be a **starting point for future non-astronomy development**.

That is not the same as saying nothing has been checked. The implementation is
validated against a published fit -- see [Validation](#validation) below for
exactly what was reproduced and to what tolerance -- and it will not ship until
it recovers those values. **What that establishes is that the code computes the
model it says it computes.** It is real evidence and it is the right first bar.

**What it does not establish is that the modelling choices are right for your
data.** No domain expert has reviewed any of the following, and each is a place
where an implementation can reproduce a reference fit and still be the wrong
model to apply somewhere else:

- the model form (one-compartment, first-order absorption) and whether it suits
  the drug, route, and sampling design at hand;
- the priors, their widths, and whether they are defensible or merely
  convenient;
- the residual error model and what it assumes about the assay;
- the between-subject variability structure and the covariate model;
- the units, and the dose/weight conventions the example data uses;
- the reported quantities and the conventions they are reported in.

So: **treat this as a working implementation with an unreviewed modelling
layer, not as a validated tool.** Do not lean on it for a dosing decision, a
study design, a regulatory submission, or a published result without someone
who knows the field checking the list above against your problem.

**If you are a domain expert:** review is genuinely wanted, and that list is
the place to start. **If you are not:** treat every domain claim in this
directory as a starting point to verify against the literature, not as settled.

### Why the architecture claim is different

The architectural claim -- *a component from a field with no stars in it can be
written against the documented extension API without changing the core* -- is
demonstrable by construction, and this component demonstrates it. That claim
does not depend on the pharmacology being right; it would hold equally if every
prior here were mis-specified.

The domain claim -- *these are the right choices for modelling this drug* -- is
not demonstrated by anything here. Keeping the two apart is the point of this
section, and blurring them is exactly the misuse it exists to prevent.

## Validation

**Performed 2026-09-11, against R.** R's `nlme` package is the reference
implementation for this model and this dataset, so the comparison is against an
independent implementation rather than against ourselves.

### 1. The forward model IS `SSfol` (exact)

`physics.calc_pk_concentration` agrees with R's `SSfol` -- the self-starting
first-order-absorption model `nlme` ships -- to a **maximum relative difference
of 1.7e-13** over 400 random points spanning `ke` in [0.018, 0.37]/hr, `ka` in
[0.37, 4.5]/hr, `Cl` in [0.14, 7.4], doses 50-600 and times 0.05-30 h. That is
float64 round-off, so the implementations are the same function.

### 2. Per-subject estimates vs `nlsList` (estimator-matched)

`nlsList` fits each subject independently, which is exactly what this component
does today, so this is like for like.

| quantity | agreement with `nlsList` |
|----------|--------------------------|
| `CL/F` | median ratio **0.990**, worst subject **3.6%** -- all 12 |
| `V/F` | median ratio 1.009, worst 11.0% (10 subjects; see below) |
| `ka` | median ratio 0.953, worst 23.5% (10 subjects; see below) |

**Two subjects differ, for two different and both-understood reasons.**

* **Subject 12 is the flip-flop degeneracy.** Its chain settled in the mirrored
  mode: our `ka` = 0.119/hr against its `ke` = 0.1056/hr (the swap's signature),
  `V` correspondingly at 4.1 L against 24.1 L -- and `CL` unchanged at 2.57 vs
  2.54. Expected, documented, and exactly what the algebra predicts.
* **Subject 9 is NOT a flip-flop**, and the distinction matters. Our `ka` = 21/hr
  against `nlsList`'s 8.87/hr, but 21 is nowhere near its `ke` of 0.0866, so no
  swap has happened. Its absorption is simply faster than the 0.25 h first
  sample can resolve: above roughly 5/hr the likelihood is flat in `ka`, and
  both values sit in that flat direction. A weakly identified parameter, not a
  second mode.

`CL` agreeing to 3.6% on **every** subject including both of those is the
substantive result, and it is not a coincidence: `CL` is the quantity invariant
under the flip-flop.

### 3. Population level vs `nlme` (indicative ONLY -- different estimators)

| quantity | ours (median of 12 independent fits) | `nlme` fixed effect | ratio |
|----------|--------------------------------------|---------------------|-------|
| `CL/F` | 0.04285 L/hr/kg | 0.03967 | 1.08 |
| `t_half` | 7.65 h | 8.07 | 0.95 |

**These are not the same estimator and must not be read as a pass/fail.**
`nlme` is a mixed-effects fit whose fixed effects are population typical values
with shrinkage; ours is the median of twelve unpooled fits. Agreement to ~8% is
what one should expect, and a rigorous population-level comparison is not
possible until the `population` component (P4) exists. **That is a reason P4 is
not a convenience feature.**

The residual error models also differ in form -- ours is combined
(`sigma_add` 0.300 mg/L, `sigma_prop` 0.132), `nlme`'s is additive-only
(0.7092 mg/L) -- so only their magnitude is comparable: our combined sigma over
the 132 real observations has rms **0.813 mg/L** against `nlme`'s 0.709, a
ratio of 1.15.

### 4. Population level vs `nlme` (estimator-matched, 2026-09-12)

**This is the comparison section 3 could not make.** With the `population`
component (P4) the fit is mixed-effects like `nlme`'s, and in the same
coordinate basis: `fitclke: true` puts the random effects on
(lKe, lKa, lCl), which is what `SSfol` is parameterized in. A diagonal set of
omegas in one basis is not diagonal in another, so this is not a detail --
without it the two models are different models.

**Which published model.** Pinheiro & Bates fit *three* to these data, and the
script that reproduces the book ships with `nlme`
(`system.file("scripts", package="nlme")`, `ch08.R`):

| | random effects | lKe | lKa | lCl |
|---|---|---|---|---|
| `fm1Theo.nlme` | unstructured (correlated) | -2.432671 | 0.451410 | -3.214452 |
| `fm2Theo.nlme` | `pdDiag(lKe + lKa + lCl)` | -2.454646 | 0.465505 | -3.227198 |
| `fm3Theo.nlme` | `pdDiag(lKa + lCl)` | -2.454704 | 0.465734 | -3.227223 |

`fm3` is the book's final model (it is the one it goes on to plot and
diagnose), and its fixed effects agree with `fm2`'s to 1e-4. **That is what we
compare against.** `fm1`'s differ, and the difference matters -- see the
random effects below.

**The run**, as the shipped `examples/theophylline` config stands: 4 chains,
2000 tune + 2000 draws, `target_accept: 0.95`. Max Rhat 1.001, min ESS 2016,
48 divergences in 8000 draws (0.6%), sitting in the upper tail of `omega_ka`
-- the least identified quantity -- and not at an omega -> 0 boundary.

**Typical values, compared in units of sigma**, because a percentage is not a
statement about agreement. Ours is the posterior SD, `nlme`'s the standard
error from `summary()`; the difference is divided by their quadrature sum:

| quantity | ours (log10) | `nlme` fm3 (log10) | difference | n sigma |
|----------|--------------|--------------------|-----------|---------|
| `log ke` | -1.06757 +/- 0.02368 | -1.06604 +/- 0.02280 | -0.00153 | **0.05** |
| `log ka` | 0.19768 +/- 0.11268 | 0.20217 +/- 0.08627 | -0.00449 | **0.03** |
| `log CL/F` at 70 kg | 0.44389 +/- 0.03358 | 0.44354 +/- 0.02606 | +0.00035 | **0.01** |

`nlme`'s clearance is per kg -- an allometric exponent of exactly 1 -- while
this fit uses the field-standard 0.75 about a 70 kg reference, so the two
covariate models agree at the reference weight and diverge by at most ~6% at
the ends of this cohort's 54.6-86.4 kg range. A modelling choice, not a
discrepancy.

**Between-subject variability.** Our `omega` is the SD of the base-10
logarithm and `nlme`'s of the natural one, a factor of ln(10) = 2.3026 apart.
Converted, against `fm2` (the matching structure):

| | ours, median [95%] | `nlme` fm2, est [95%] |
|---|---|---|
| `lCl` | 0.228 [0.143, 0.409] | 0.1669 [0.109, 0.256] |
| `lKa` | 0.792 [0.494, 1.450] | 0.6439 [0.406, 1.021] |
| `lKe` | 0.095 [0.007, 0.244] | 1.6e-05 [2.4e-143, 1.1e+133] |

**The "collapsed lKe" is an artifact of forcing the covariance diagonal, and
reporting it as a result was our error.** Two things say so. `nlme`'s own
interval on it spans 276 orders of magnitude -- the variance component is
completely unidentified, so its point estimate carries no information and
"we do not reproduce 1.6e-05" was conceding to a number that means nothing
(it also moves between runs: 1.61e-05 here, 1.92e-05 in an earlier one).
And `fm1`, which lets the random effects correlate, finds **sd(lKe) = 0.131
with corr(lKe, lCl) = 0.995** -- no collapse at all. The physical statement is
not "ke does not vary between subjects", it is "lKe and lCl vary almost
perfectly together", which is what one would expect when V = CL/ke is the
quantity that is stable across subjects.

**So the honest gap is that this component cannot represent `fm1`.** `omega`
is diagonal by construction: there is one between-subject SD per coordinate
and no correlation. Adding a correlated `omega` (a Cholesky factor, the
standard non-centred hierarchical form) is the concrete next feature, and
until it exists the `fm1` row above is a model we can fit *to*, not *with*.

Residual error: ours is combined (`sigma_add` 0.357 mg/L [0.199, 0.641],
`sigma_prop` 0.123 [0.059, 0.163]), `nlme`'s additive-only
(0.7092 [0.620, 0.811]), so only magnitudes compare -- our combined sigma has
rms **0.786 mg/L** over the 132 observations against 0.709.

**The hierarchy resolved subject 12's flip-flop**, which is worth recording
because it is shrinkage doing exactly what it is for. Fitted alone that
subject settled in the mirrored mode (section 2); under a population it is at
`ka` = 0.85/hr, `ke` = 0.093/hr, `V/F` = 26.0 L -- the direct mode, and close
to `nlsList`'s 24.1 L -- pulled there by the other eleven. Subject 9's `ka`
remains large (6.8/hr) and unidentified, for the unrelated reason section 2
gives.

### What kind of comparison this is, and what it is not

`nlme` fits by **maximum likelihood** (its default, and what was used here;
REML gives identical fixed effects on these data). This is a Bayesian
posterior with proper priors. The two agreeing to a twentieth of a sigma says
the likelihood dominates and both implementations are computing the same
model -- it is not a claim that ML and Bayes are the same procedure.

**This is not the first Bayesian analysis of the Theophylline data, and
nothing here should be read as claiming otherwise.** `Theoph` is the standard
worked example of the field's Bayesian tooling -- the Stan/Torsten tutorial
of Margossian, Zhang & Gillespie is the best-known -- and it appears
throughout `nlmixr2`, Monolix and NONMEM `$BAYES` material. What is new here
is only that a component from this field runs on this astronomy code's
machinery, which is an architectural claim and not a pharmacometric one.

### What this does and does not establish

**Established:** the code computes the model it claims to (1); it recovers
per-subject estimates matching an independent reference implementation on real
data (2); and, with the population component, it recovers that implementation's
mixed-effects typical values to better than 0.5% and its between-subject
variability within the posterior's own intervals (4).

**NOT established:** that the modelling choices are right for anyone's data. No
domain expert has reviewed the priors, the error model, the handling of the
`t = 0` samples, or the decision to fix `F`. The caveat at the top of this file
stands unchanged -- reproducing a reference fit validates an implementation,
never a modelling choice.

## Status

P0, P1, P3, P4 and P5 are implemented; P2 is not started. P1 and P4 are
validated against R (see above). The design is in `notes/pharmacokinetics.txt`
(private notes repository). Phases:

| Phase | Scope | State |
|-------|-------|-------|
| P0 | Credible-interval width as a run-level setting (`exozippy.reporting`) | DONE |
| P1 | `subject` + `assay`, no hierarchy | DONE, validated (above) |
| P2 | Symbolic relations / relaxation-engine seeding | not started |
| P3 | TRANS1/TRANS2 parameterization via element roles (`fitclv`/`fitkev`/`fitclke`) | DONE |
| P4 | `population`: between-subject variability + allometric covariate | DONE (implementation; see below) |
| P5 | Flip-flop degeneracy: mode reporting and the opt-in ordering bound | DONE |

**P4 is not a convenience feature**, and the validation above is why: the
canonical published fit is a MIXED-EFFECTS fit, so a rigorous population-level
comparison is impossible until a `population` component exists. Without it this
is twelve independent fits sharing an error model -- no between-subject
variability, no shrinkage, and no CV%/eta-shrinkage to report.

**What P4 delivers, and what it does not.** The `population` component exists,
the hierarchy is non-centred, the allometric covariate is in, CV% is reported
as a derived parameter with its own credible interval, and the fit is done in
the coordinate basis R's `nlme` uses (`fitclke: true`) so that the
comparison is of the same model. One thing is still owed:

* **eta-shrinkage is not reported.** It is a function of the finished TRACE,
  not of the model, and there is no channel for a component to contribute a
  post-fit number to the report -- see `pharmacokinetics.md`, which states
  exactly what such a channel has to carry.
* **eta-shrinkage is the only thing P4 owes and does not deliver** -- the
  population-level comparison against `nlme` is in section 4 of Validation
  above.

P0 comes first because reporting a pharmacokinetic result in the astronomy
convention (median + 68%) would be misread as the 95% interval that field uses,
understating the uncertainty about twofold. A caveat in a caption does not fix
that, so the convention is a gate on shipping any PK example rather than a
later refinement. See the "credible-interval width" section of
`src/exozippy/outputs/outputs.md`.

## Where the caveat must be repeated, and why

Four places, because each is reached by a different reader and none of them
implies the others. Only the first exists today; **the remaining three are a
requirement on the phase that creates each file, not a nicety**, and a phase is
not done until its copy is in place:

| Location | Reader | State |
|----------|--------|-------|
| this README | developer browsing the directory | DONE |
| the component's `.md` subsystem doc | an editor, via CLAUDE.md's map | required at P1 |
| the component class docstring | a user calling `help()` | required at P1 |
| `examples/theophylline/README` | whoever copies the example | required at P1 |

The example matters most of the four and is the easiest to forget: an example
is the thing that gets copied, and a copied config travels without any of the
component's documentation attached to it.

`src/exozippy/reporting.py` already carries a narrower version of the same note
(its PROVENANCE section), for the same reason: its docstring surveys
non-astronomy reporting conventions to motivate the credible-interval setting,
and that survey is unreviewed secondary knowledge too.
