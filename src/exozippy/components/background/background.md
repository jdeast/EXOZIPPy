# The unresolved background of a microlensing sightline

`src/exozippy/components/background/`: the light inside a point-spread function that no
star block models, as ONE object per sightline. Read this before giving the blended light
of a microlensing target any parameter of its own, before adding a prior on the background's
fluxes or colors, and before letting another component (the SED) read it. Related:
`src/exozippy/components/mulensing/mulensing.md` ("The blended photocenter moves"), C30 in
`src/exozippy/components/mulensing/conventions.md`.

## What it is, and what it is not

A ground-based (or Gaia) PSF around a microlensing target holds the source's images, the
luminous lens, the occasional resolved neighbor, and the stars below any high-resolution
catalog's completeness limit summed over the PSF. The first three are star blocks: the lens
and the source always, a resolved neighbor once it is given its own star block (with MIST
structure and pm pinned at the field mean -- the planned resolved-blends follow-up). The last
is not a star and has no star's parameters; it is a STATISTICAL background, and this
component is its one spelling (review 8.10.3, JDE 2026-10-26: the "unmodeled remainder" of
the blend split and the planned follow-up's "unresolved background" are the same light and
must be one object). Modeled stars stay separate from it: a dataset lists them in
`psf_stars:` and the background in `background:`.

```yaml
background:
  - name: Bg
    bands: [G, Ks]        # the band names datasets' `band:` keys use
astrometryinstrument:
  - name: VIRAC2
    mode: catalog
    band: Ks
    psf_stars: [Lens]
    background: Bg
```

## Its parameters

| parameter | elements | meaning |
|---|---|---|
| `fluxratio` | one per (instance, band), `background.<name>.fluxratio_<k>`, k the band's position in `bands:` | flux relative to the primary source's baseline flux in that band, F_bg / F_S |
| `offset_e`, `offset_n` | one per instance | photocenter at the astrometric reference epoch relative to the primary source's position there, mas; pinned at 0 by default (nearly degenerate with the source position under a catalog solution) |
| `pm_ra`, `pm_dec` | one per instance | that photocenter's proper motion, mas/yr; no parallax of its own |

- **The flux is sampled per band, explicitly.** One element per band the instance lists,
  shared by every dataset that reads that band, uniform on [0, 1000] unless the params file
  states a prior. A dataset reading a band the instance does not list RAISES. Relative to the
  SOURCE, not absolute, because the astrometric weighting needs only ratios and because a
  band the SED cannot predict (VISTA Ks before its BC columns) still has a ratio.
- **The pm has a field prior by default.** On a bulge sightline (|l|, |b| < 10 deg) the
  manifest `"overrides"` channel gives `pm_ra`/`pm_dec` a Gaussian at the galactic model's
  mean proper motion of a bulge star at the Galactic center's distance on this sightline
  (`galacticmodel.physics.expected_proper_motion`, the same mean the microlensing pm seeds
  use) with the bulge velocity dispersion there (100 km/s, Koshimoto & Bennett 2020's
  intermediate axis: ~2.6 mas/yr), logged at stage 3 and replaced by any params entry. Off
  the bulge the pm is uniform on +/-50 mas/yr, with a warning naming the fix. A free flat pm
  is honest but uninformative: with any background flux it absorbs the share of a catalog
  proper motion the background can carry.
- **Anchored on the primary source.** The background exists only on a microlensing sightline
  (no `mulensevent:` RAISES), and its photocenter and fluxes are referred to the primary
  source, which every dataset reading it measures.

## The seams (what a later component extends, without renaming anything here)

Every reader goes through two methods, and those are where the planned resolved-blends
follow-up attaches:

- **`Background.flux_ratio(i, band, where)`** -- the one reader of the per-band flux. The
  follow-up adds a PRIOR on these elements (or on the magnitudes derived from them and the
  source's SED prediction): the THOR luminosity function integrated below completeness over
  the PSF-weighted area, a mean and a Poisson covariance across bands (so the colors are
  constrained jointly). It is a potential on existing elements; no coordinate changes.
- **SED rows naming the background** in `photType` read the same element times the
  source's predicted flux in that row's band -- again a new reader of `flux_ratio`, not a new
  parameter.
- **`Background.track(i, source_e, source_n, dt_yr)`** -- the one reader of the photocenter.

Neither is implemented yet; both are follow-ups.

## Known limits

- One flux per band per instance: two datasets of very different PSF size in one band see
  different amounts of unresolved light. Give each PSF its own instance (`background:` takes
  a name) when that matters.
- The light curve's `f_blend` is not tied to the members plus the background in the light
  curve's own band; the split's astrometric weights use the sampled ratios only.
- No parallax: the background's photocenter is treated as distant (a bulge background's
  ~0.12 mas parallax times its flux share is below current catalog parallax errors).

Tests: `tests/test_catalog_astrometry.py` (one object shared by two catalogs, the field
prior, the photocenter track inside the simulated catalog solutions, and the configuration
raises).
