# Theophylline: population pharmacokinetics

## READ THIS FIRST

**This example was written by an astrophysicist and an LLM. No biologist,
pharmacologist, clinician, or pharmacometrician has reviewed it.**

It exists to demonstrate that a component from a field with no stars in it runs
on exactly the same machinery as every astronomy fit in this repository -- the
same config, the same relaxation engine, the same sampler, the same tables --
and to be a starting point for non-astronomy development.

**It is not a validated pharmacokinetic analysis and must not be used as one.**
Do not lean on it for a dosing decision, a study design, a regulatory
submission, or a published result.

This caveat is repeated here, rather than only in the component directory,
because **an example is the thing people copy** -- and a copied config travels
without any of the component's documentation attached to it. If you are
adapting this file, bring this paragraph with you.

The fuller statement of what is and is not validated, and the list of things a
domain expert would need to check, is in
`src/exozippy/components/pharmacokinetics/README.md`.

## Running it

The data are **not redistributed** with EXOZIPPy (licence compatibility -- see
`src/exozippy/components/pharmacokinetics/fetch_theoph.py`). Fetch them first:

```bash
cd examples/theophylline
exozippy-fetch-theoph          # writes theoph.csv here, md5-pinned
exozippy theophylline.yaml
```

`theoph.csv` is gitignored deliberately. Do not `git add -f` it.

## The data

Theophylline pharmacokinetics: 12 subjects given a single oral dose, 11 serum
concentrations each over about 25 hours, with body weight recorded.

> Boeckmann, A. J., Sheiner, L. B., and Beal, S. L. (1994),
> *NONMEM Users Guide: Part V*, NONMEM Project Group, UCSF.

It is the canonical nonlinear mixed-effects example, and ships with R as
`datasets::Theoph`. Columns: `Subject` (1-12), `Wt` (kg), `Dose` (**mg/kg**),
`Time` (hr), `conc` (mg/L).

## The model

One compartment, first-order absorption, single oral dose -- NONMEM's ADVAN2,
nlme's `SSfol`. Each subject has its own apparent clearance `CL/F`, absorption
rate `ka` and elimination rate `ke`, sampled in log10, with the apparent volume
`V/F = CL/ke` derived. That basis (`parameterization: "cl_ke"`) is `SSfol`'s,
and is chosen deliberately: see the config's own comment and "Validation"
below.

The twelve subjects are drawn from a **population** -- a typical value, an
allometric scaling with body weight, and a log-normal between-subject
deviation per subject, written non-centred. That is what makes this population
PK rather than twelve separate fits, and it is what the canonical published
fit of these data does.

Bioavailability `F` is not identifiable from oral dosing alone, so it is fixed
at 1 and every clearance and volume here is an **apparent** value. A reader who
takes `CL/F` for `CL` is wrong by `1/F`.

## Three things a domain expert would have an opinion about

Named explicitly, because they are exactly the kind of choice that looks
settled in a config file and is not:

1. **The covariate model is allometric and its exponents are fixed.**
   Clearance scales as `WT^0.75` and volume as `WT^1` about a 70 kg
   reference, which is the field's default and not a measurement on this
   cohort. They are pinned parameters, so freeing one (`population.beta_cl:
   {sigma: 0.2}`) or switching it off (`{initval: 0.0}`) is a line in the
   params file -- but as shipped they encode a convention. The canonical
   `nlme` fit of these data uses no covariate at all and an implied exponent
   of 1, and the two agree at the reference weight and differ by up to ~6% at
   the ends of this cohort's weight range.

   Related: the between-subject variability is **diagonal**. Real popPK
   models frequently estimate a correlation between CL and V, and this one
   cannot.
2. **The `t = 0` samples are kept.** Nine subjects have exactly `conc = 0` at
   `t = 0`, and three have 0.15, 0.24 and 0.74 mg/L. The model predicts
   *exactly* zero at `t = 0`, so those three contribute residuals that the
   additive error term absorbs by inflating. Dropping them, or modelling a
   pre-dose baseline, are both defensible and neither is done here. Silently
   dropping data would be worse than the inflation, so they are kept and
   flagged.
3. **The priors are wide, not informed.** The bounds in the component's
   `defaults.yaml` are chosen not to exclude a plausible drug, not to encode
   knowledge about this one. They are not a considered prior for theophylline.

## Reporting at 95%

The config sets:

```yaml
reporting:
  credible_interval: 0.95
```

This is not cosmetic. EXOZIPPy defaults to median + 68.3% (1 sigma), which is
astronomy's convention; pharmacometrics, clinical and regulatory work report
95% essentially without exception. A 68% interval in a table a
pharmacometrician reads is not read as a 68% interval -- it is read as the 95%
one their field always uses, which understates the uncertainty by about a
factor of two. A note in the caption does not fix that.

Note also that `0.95` here means **exactly 95%**, the round probability, not
2 sigma (which is 95.45%). The two fields use different kinds of number, not
just different widths; see `src/exozippy/reporting.py`.

## Validation

**Performed 2026-09-11 against R's `nlme`** -- an independent implementation of
this model on this dataset. Headlines:

* our forward model agrees with R's `SSfol` to **1.7e-13** (float64 round-off,
  i.e. it is the same function);
* per-subject `CL/F` agrees with R's `nlsList` -- the estimator-matched
  comparison -- to **3.6% on every one of the 12 subjects**, median ratio 0.990;
* with the population model, the mixed-effects typical values match `nlme`'s
  to better than **0.5%** on all four of `ke`, `t_half`, `ka` and `CL/F`, and
  the between-subject spreads sit inside the posterior's own 95% intervals --
  including reproducing `nlme`'s finding that `ke` has essentially no
  between-subject variability.

The full table, including why subject 12 differs (the flip-flop) and why
subject 9 differs for an unrelated reason (`ka` unidentifiable above ~5/hr with
a first sample at 0.25 h), is in
`src/exozippy/components/pharmacokinetics/README.md`.

**This validates the implementation, not the modelling choices.** The caveat at
the top of this file is unchanged.
