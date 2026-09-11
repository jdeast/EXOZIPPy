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

**Not yet performed -- this section is a placeholder and a gate.** The component
does not ship until it is filled in with real numbers.

The bar, which is a condition on P1 rather than an aspiration:

- **Recover a published fit.** The acceptance example must reproduce the
  published population parameters for its dataset, and this section must state
  which publication, which parameters, and the agreement achieved -- as numbers,
  not as "agrees well". Where the published fit is maximum-likelihood and this
  one is Bayesian, say so and compare like with like (posterior median against
  point estimate, and the interval against the reported standard error).
- **State what was NOT compared.** A reference fit pins some quantities and not
  others; the ones it does not pin are unvalidated and must be named here rather
  than left to look covered.
- **Keep the caveat above regardless.** Reproducing a reference fit validates
  the implementation, not the modelling choices, and it does not substitute for
  domain review. A green acceptance test is not permission to delete the first
  section of this README.

Until this section carries numbers, assume the implementation is unverified as
well as unreviewed.

## Status

**Not yet implemented.** The design is in `notes/pharmacokinetics.txt` (private
notes repository). Phases:

| Phase | Scope | State |
|-------|-------|-------|
| P0 | Credible-interval width as a run-level setting (`exozippy.reporting`) | in progress |
| P1 | `subject` + `assay`, no hierarchy | not started |
| P2 | Symbolic relations / relaxation-engine seeding | not started |
| P3 | TRANS1/TRANS2 parameterization via element roles | not started |
| P4 | `population`: between-subject variability + allometric covariate | not started |
| P5 | Flip-flop degeneracy: mode reporting and the opt-in ordering bound | not started |

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
