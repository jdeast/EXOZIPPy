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
nlme's `SSfol`. Each subject has its own apparent clearance `CL/F`, apparent
volume `V/F`, and absorption rate `ka`, sampled in log10.

Bioavailability `F` is not identifiable from oral dosing alone, so it is fixed
at 1 and every clearance and volume here is an **apparent** value. A reader who
takes `CL/F` for `CL` is wrong by `1/F`.

## Three things a domain expert would have an opinion about

Named explicitly, because they are exactly the kind of choice that looks
settled in a config file and is not:

1. **Every subject is independent.** This is *not* population PK in the field's
   sense. There is no between-subject variability model, no shrinkage, and no
   covariate model -- so a subject with uninformative data is constrained only
   by the wide priors in `defaults.yaml`, not pulled toward a population mean.
   That component (`population`) is P4 and does not exist yet.
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

## What a first run shows (NOT validation)

Recorded because it is informative, and labelled because it is **not** the
validation this component still owes.

Fitting all 12 subjects and comparing to an independent per-subject
least-squares fit of the same closed-form model (scipy `curve_fit`, nothing to
do with EXOZIPPy):

| quantity | agreement |
|----------|-----------|
| `CL/F` | within **3.6%** on every subject, median ratio 0.994 |
| `V/F` | agrees except subject 7 |
| `ka` | agrees except subjects 7 and 9 |

The population medians come out at `CL/F` ~ 0.043 L/hr/kg and `t_half` ~ 7.7 h,
which sit in the range usually quoted for theophylline in adults.

**Subject 7 is the flip-flop degeneracy, caught in the wild.** Its chain
settled in the mirrored mode, and the signature is exactly the one the algebra
predicts: EXOZIPPy's `ka` (0.112 /hr) equals the conventional fit's `ke`
(0.102 /hr), `V` has moved toward the predicted `V*ke/ka`, and **`CL` is
unchanged** (3.296 vs 3.333 L/hr). That is the degeneracy behaving as derived,
not a bug -- and it is why the mode report warns that the chains do not mix
between modes.

It is also the practical lesson the degeneracy carries: the clearance-derived
quantities are trustworthy here and the volume-derived ones are not, for that
subject, without outside information to pick the branch.

Subject 9's `ka` is a different matter -- simply poorly determined, because the
absorption phase is faster than the sampling times can resolve.

## Validation

**Not yet performed.** The comparison above is against *another fit of the same
model by us*, which checks the implementation's internal consistency and
nothing else. It is not a comparison against a published population fit, it
does not test the modelling choices, and no domain expert has reviewed any of
it.

Until this section carries a named publication and real numbers, treat what
this example produces as unverified as well as unreviewed. The bar and the gate
are in `src/exozippy/components/pharmacokinetics/README.md`.
