# How to Store the Results of Microlensing Fits?

## The Problem

In standard microlensing practice (which we will use for seeding EXOZIPPy
fits and as a comparison methodology), fits are done by successively adding 
more parameters. At some point in the process, the data errorbars are also
renormalized and the fits are re-evaluated, which is necessary for fit 
comparison by chi2. 

How should we store these successive fits?

How do we keep track of which fit is best or should be referenced as the 
baseline fit for seeding a more complex fit?

## Some Example Workflows

### PSPL

1. Fit static PSPL
2. Fit u0+ parallax PSPL
3. Fit u0- parallax PSPL
4. Evaluate models 1, 2, 3 and choose the best one
5. Renormalize errors rel to model selected in 4.
6. Re-run fits 1-3 with renormalized errors starting from prev best-fit for 
   each individual model.

Modification for satellite parallax: there may be up to 4 distinct 
solutions.

### FSPL

Same as PSPL, EXCEPT:

0. Fit static PSPL

Replace "PSPL" with "FSPL" in Steps 1-3.
Expand Step 6 to include the PSPL model (4 models total.)

### Planet - Basic

1. Excise planetary signal.
2. Fit static PSPL to excised data.
3. Renormalize errors relative to model from Step 2.
4. Conduct planet search seeded with model from Step 3.
5. Fit planet + parallax for u0+ and u0- cases.
6. Double-check that error renormalization is still appropriate.

### Planet with Large Parallax

1. Excise planetary signal.
2. Follow PSPL Steps 1-5 for excised data.
3. Optimize model selected in PSPL Step 4.
4. Conduct planet search seeded with model from PSPL Step 4.
5. Fit planet + parallax for alternate u0 case.
6. Double-check that error renormalization is still appropriate.

Hope that the signal is really parallax and not 3L1S.

### Binary with Caustic Crossings

1. Excise caustic crossings.
2. Fit static PSPL to excised data.
3. Seed a binary lens search starting with best-fit PSPL from Step 2.
4. Renormalize errors relative to best-fit model found in Step 3.
5. Optimize minima from Step 3.
6. Evaluate minima and select "survivors."
7. Add parallax to survivors.
8. Add orbital motion to parallax fits.

### Binary with Asymmetry Only

1. Follow PSPL Steps 1-6.
2. Seed static binary lens search starting with best-fit static PSPL model.
3. Add parallax to static binary lens models.
4. Add orbital motion to binary lens + parallax models.
5. Double-check that error renormalization is still appropriate. Re-optimize if 
   necessary.

## Use Cases

1. Track the results of every step in the workflow.

2. Output "final" (with renormalized errors) fits for the major model classes
   (in the proper order) for comparison in a table.

3. Output some custom table with just the models I think are important.

Maybe we need to look at some tables in recent papers for ideas about how
fit information is organized for different events.

## Proposed Solutions

Assume that at some point, someone will want to print out the results of each
step in the workflow.

Goal is to remove all the if statements.

Questions:
- Are there different properties or options that we would want for 2L1S fits?

- Is it important to track the order the fits were done in? (argues against a 
  dictionary)

- Which reference fits do we need to keep track of (i.e. do we use to seed 
  new fits)?

### Dictionary

results = {'best': fitter.best, 'results': fitter.results, 'parameters_to_fit': fitter.parameters_to_fit}

example:
    fit_results = {'static_PSPL_raw': results, 'static_PSPL_renorm': results}
    base_pspl = 'static_PSPL_renorm'

### Fit Classes

Advantages:
- Possible to have a nice print function for all cases using class inheritance.
- Standardized properties.
- Can build in the correct parameters_to_fit to minimize errors.

Isn't this sort of already built into the classes in fitters? Is it just that 
we don't want to store the entire fitter?

# Current EXOZIPPy Architecture

Everything has a read/write interface to an Event Structure.

Different solutions have different objects associated with them.
