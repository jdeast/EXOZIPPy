# The red clump as a reference object (`redclump`)

`src/exozippy/components/redclump/`: the field red clump of a microlensing sight line, as ONE
reference object the SED evaluates like a star. Read this before touching the clump, the
interim source-extinction tie, or how a published clump extinction reaches a fit. Related:
`src/exozippy/components/sed/sed.md` ("SED flux sources", "Differential color rows", "The
spectral mode"), `src/exozippy/components/galacticmodel/galacticmodel.md` ("The sight-line
queries"). Review 8.9.11; the dust stack it belongs to is color rows (#600) -> this clump ->
the monotone A_V(d) profile (review 8.9.8).

## What it is, and what it is not

The CENTROID of the field clump on the event's sight line: a population object with a known
intrinsic color and luminosity at the bar's distance. It is NOT a `Star`: a star brings a
mass and through the galactic model the IMF, the density and kinematic mixture, linked-star
groups, structure activity, Claret ties and tracks -- all wrong for a population centroid, and
each a place it would have to be special-cased out. It is read only through the SED, by the
generic flux-source seam (`Component.sed_flux_sources`): an SED row names it in its photType
by its block `name` (default spelling `RedClump`; it must not be a star's name), and its
magnitudes come out of the SAME bolometric-correction evaluation as the stars'.

```yaml
redclump:
  - name: RedClump
    clump_constrains_source_av: true   # the interim tie (below); default true
    bands: [OGLE_I, OGLE_V]            # band blocks whose clump extinction is reported
    colors: [[OGLE_V, OGLE_I]]         # [blue, red] pairs whose color excess is reported
```

One block (a second raises). It RAISES without an `sed:` block, without a `galacticmodel:`
block (the sight line is the galactic model's anchor, and the clump's distance prior is its
bar: one definition of "the sight line", no second anchor key; JDE 2026-11-09), on a sight line
outside the bulge box (|l|, |b| < 10 deg: no clump there), and when nothing reads it (no SED
row names it and no prior is stated on one of its extinction elements: its parameters would
be sampled with nothing constraining them).

## Its parameters

| param | role | prior | why |
|---|---|---|---|
| `teff` | sampled | flat [4000, 5500] K, the SED grid overlaid | the color prior acts through `vi0` |
| `logg` | pinned 2.5 | a params entry frees it | moves (V-I)_0 by 0.002 mag per 0.5 dex |
| `feh` | sampled | N(0.0, 0.3) | bulge clump-giant MDF, two ~equal components near -0.3 and +0.3 dex (Hill et al. 2011) |
| `mi` | sampled | N(-0.12, 0.10) mag | M_I,RC = -0.12 (Nataf et al. 2013); 0.10 a provisional CENTROID systematic (the 0.09 Nataf quotes is the clump's star-to-star dispersion); JDE 2026-11-09: keep, and verify both against the paper's text once the dust stack is done (TODO in defaults.yaml) |
| `distance` | sampled | N(d_bar, 0.1 mag) | the galactic model's bar on the sight line |
| `av` | sampled | flat [0.05, 20] | lower bound strictly positive: the tie divides by it |
| `radius` | derived | -- | what makes M_I = `mi` at A_V = 0 through the SED's own BC_I |
| `vi0` | derived | N(1.06, 0.06) | (V-I)_0 = BC_I - BC_V at A_V = 0 (Johnson V, Cousins I); Bensby et al. (2013) |
| `extinction_<k>` | derived | none by default | A_band per `bands:` entry |
| `color_excess_<k>` | derived | none by default | E(a - b) per `colors:` pair |
| `av_ratio_<k>` | derived | N(1, 0.09) | the interim tie, one element per tied star |

- **The color prior is stated in the observable.** `vi0`'s Gaussian (the derived-with-sigma
  idiom of `band.du1` and `mulensinstrument.zeropoint`) makes (V-I)_0 = 1.06 +/- 0.06 come out
  of OUR bolometric corrections, at whatever logg/[Fe/H] the clump samples and through
  whichever path (table or spectral) the fit uses; a regenerated table retunes it. Through the
  shipped NextGen tables it is Teff ~ N(4684, 124) K at logg 2.5, [Fe/H] 0 (+140 K per dex of
  [Fe/H]), and the tables' (J-Ks)_0 = 0.669 there matches Nataf's clump 0.66. No Jacobian is
  owed: teff is flat and vi0 a smooth monotone function of it.
- **`mi` rather than a radius** because M_I,RC is what the literature states, so the prior is
  the published one with no log-radius tilt; `radius` is derived (M_bol = M_I + BC_I(A_V = 0),
  L from the IAU zero point, R from L and Teff -- the inverses of `calc_absbolmag` and
  `calc_luminosity`).
- **`mi` and `distance` are exactly degenerate in the photometry** (only mi + mu enters). Both
  are kept: they are independent statements (population luminosity vs bar geometry), the ridge
  is linear, and the dust profile of review 8.9.8 needs the distance alone.
- **The distance prior** is the bar's: `GalacticModel.bar_distance_moments` (the
  `koshimoto21.bar_distance_moments` query: moments in distance modulus of the bar branch's
  n(d) d^2 dd, plus the NSD inside genulens's box), centered on the MEAN with 0.1 mag of
  distance modulus -- the centroid's systematic (the port's bar and genulens's coded clump
  formula agree to <= 0.08 mag; a disagreement over 0.15 mag on the sight line warns). The
  bar's depth (0.17-0.30 mag) is the spread of individual clump stars, not the centroid's
  uncertainty; it is logged and stated in the draft. It is set through the manifest
  `overrides` channel, so a params entry on `redclump.distance` replaces it.
- **The A_V start** comes from the clump's own calibrated rows when it has them (a color row
  naming the clump on both sides -- its (V-I)_RC, distance-free -- else an absolute row naming
  only the clump), solved numerically through the SED's grid (`SED.bc_np` / `appmag_np`), as a
  component hint below any user initval.
- `teff` and `av` are `declare_likelihood_backed`; the six sampled coordinates are relaxation
  engine leaves (`symbolic_physics.py`). `radius`, `vi0` and the extinctions read the BC grid
  and are in no relation.
- Default citations (Hill:2011, Nataf:2013, Bensby:2013) are set on the clump's own
  parameters as manifest options unless the params file states one.

## What the clump measures (and what it does not)

Differential rows -- Delta I (a same-filter differential, Source minus clump) and Delta(V-I)
(a differential color row, `color_filter:`, sed.md) -- are calibration-free and fix the
SOURCE's intrinsic color and magnitude relative to the clump given the shared reddening. They
do not measure A_I or E(V-I). The extinction is measured only when something carries an
absolute calibration: the clump's calibrated I_RC and (V-I)_RC (an absolute row and a color
row naming the clump, what a calibrated CMD or a 2D map gives), or the source's calibrated
flux (a stated zeropoint tie, an absolute source row). The forward model needs no special case
for any of these; a clump read only by differential rows logs at INFO that its A_V is held by
what calibrates the source and the tie.

## The interim soft A_V tie (`clump_constrains_source_av`)

`A_V(source) = A_V(clump) (1 +/- 9%)`, Nataf et al. (2013)'s field-scale extinction
granularity: the derived `redclump.<name>.av_ratio_<k> = star.av[tied] / redclump.av` with its
defaults.yaml Gaussian N(1, 0.09) (the derived-with-sigma idiom, so the width is an ordinary
params field, `redclump.RedClump.av_ratio_0: {sigma: 0.05}`, and the element is reported).
Not a hand-built potential and not `potentials.py` (whose soft bounds are one-sided); no
`pt.where` anywhere -- a plain ratio on a static map, its denominator floored strictly
positive. Tied: every star a component selects as an observed source
(`Component.selected_sources`, which may be asked from stage 3 on: the microlensing event
answers from its stage-2 `source_map`), plus a star whose distance is hard-linked to one (a
second source), less orbit-group members (their av is their anchor's) and stars whose SED
flux nothing reads (their av is pinned). A YAML boolean, default true; absent with nothing to
tie is skipped at INFO, explicit `true` with nothing to tie RAISES; `false` is the case of a
DISK source in front of the bar. Review 8.9.8's A_V(d) profile replaces it with geometry.

## Its extinction elements: where a published clump extinction goes

`bands:` gives `redclump.<name>.extinction_<k> = m_clump(band, A_V) - m_clump(band, 0)`, and
`colors:` `redclump.<name>.color_excess_<k> = extinction(a) - extinction(b)` -- integrated
through the passband on the clump's own spectrum at the fit's law (the tables, or the fitted
`sed.rv_dust`), off the SED's one grid evaluation (`SED.bc_node`, injected as context nodes).
A published value is a params-file prior on that element, with its citation:
`redclump.RedClump.extinction_0: {mu: 1.80, sigma: 0.05, citation: ...}` (the owner spelling,
`element_owners`). Better still, when the paper gives the calibrated I_RC and (V-I)_RC, write
them as SED rows naming the clump: that is the measurement, not a derived number.

This REPLACES the monochromatic inversions (review 2.9.16(b)): `extinction.
av_from_band_extinction` / `av_from_colour_excess` and mkticsed's `band_extinction` (which now
raises with this migration) inverted a band extinction at one wavelength into a star.av prior,
~15% off for a W149-width filter (conventions.md C29); the DC2018 generators put the
challenge's clump color excess on `color_excess_0` instead of an inverted `star.Source.av`
prior. The challenge's own monochromatic convention is still not reproduced (JDE 2026-09-24: a
caveat, not a feature).

## Interactions

- **fit_rv_dust: false (tables) WARNS** (JDE 2026-11-09; the message is kept short, "a
  fixed R_V may bias Teff/R_star"), and this is why: on the
  table path the law is fixed, so the clump's two measurements (A_I and E(V-I)) face one knob
  (A_V) and their ratio is forced to the table law's (~1.7 at the clump, against the ~1.2
  bulge fields measure, Nataf et al. 2013); A_V settles where it fits neither and the tension
  leaks into the source's Teff and radius (theta_star). The draft says so. A clump with only an I-band
  magnitude is still useful on the tables. Under `fit_rv_dust: true` the clump reads the sight
  line's `sed.rv_dust` (sed.md). In the band-integrated CCM89 + O'Donnell family the bulge mean
  A_I/E(V-I) = 1.217 is R_V ~ 1.98, NOT Nataf's 2.5, and no single R_V matches both the
  optical and the NIR clump colors (E(J-Ks)/E(V-I) 0.30 at R_V = 2.0 against Nataf's 0.34):
  with V, I, J and Ks clump rows the residual lands in errscale and the stars' Teff.
- **No R_V map prior beside a clump**: Nataf's R_I map is built from clump colors, so a sight-
  line prior from it on `sed.rv_dust` counts the clump twice.
- **galacticmodel `source_selection: luminosity_function`** keeps its static `source_ai_rc`
  (JDE 2026-11-09); with a clump the clump's A_I is also fitted, and a mismatch above 0.2 mag
  warns at the start and again at the posterior median (`check_at_point`), giving the clump's
  A_I as the value to put in `galacticmodel.source_ai_rc` before rerunning -- the remedy is the
  user's.
- PHOENIX limb darkening reads `star.av`, the mulensing zeropoint and the background read star
  predictions only: unaffected.

## Tests

`tests/test_redclump.py` (the seam, the tie, the raises, the bar query, the clump physics),
`tests/test_sed_differential_color.py` (the four-flux row), `tests/test_sed_fit_rv_dust.py`
(one sight-line R_V); the synthetic recovery is `tests/test_redclump_recovery.py` (slow).
