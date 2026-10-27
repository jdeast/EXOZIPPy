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
structure; a bound companion is an orbit group instead, which ties its distance and motion
to its host's; a neighbor's pm is tied where the params file links it, free under a
galacticmodel or when an astrometry dataset names it in `psf_stars:`, and pinned at the
field mean only when none of these constrains it, `Star._pin_unmeasured_neighbor_pm` and
`components/star/star.md` -- the planned resolved-blends follow-up). The last
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

- **The flux: one element per band, sampled -- or DERIVED where a light curve covers the
  band.** One element per band the instance lists, shared by every dataset that reads that
  band. A dataset reading a band the instance does not list RAISES. Relative to the SOURCE,
  not absolute, because the astrometric weighting needs only ratios and because a band the
  SED cannot predict (VISTA Ks before its BC columns) still has a ratio. In a band no light
  curve covers it is sampled, uniform on [0, 1000] unless the params file states a prior. In
  a band a microlensing light curve covers, see the next section.

## The light-curve tie (JDE ruling on PR #497)

A light curve measures the blend of its PSF: `f_blend/f_source` IS the summed flux of every
non-source star in it plus the unresolved background, relative to the source, in its band.
So in that band the background's flux is not a free number:

    background.<name>.fluxratio_<k> = f_blend/f_source - sum_{j in psf_stars} F_j/F_S

where `psf_stars` is EVERY modeled non-source star in that PSF -- the lens always (below) --
so the unresolved background is, identically, the part of the light curve's blend no star
block supplies (JDE 2026-10-27: "f_neighbor = f_blend - f_lens - sum(f_modeled stars)").

**The form chosen: DERIVED, not a second sampled parameter tied by a potential.** The
element takes the per-element `derived` role (`expr_key: {from_lightcurve: <tied
elements>}`, `physics.calc_background_fluxratio`), so it keeps its one spelling
(`background.<name>.fluxratio_<k>`, in the tables and the params file alike) whether derived
or sampled, adds no sampled dimension, and the identity holds exactly rather than to a
chosen width. The light curve's `f_blend` and `f_source` are its manifest deps
(`mulensinstrument.f_blend[lc_map]`), and the members' fluxes `F_j/F_S` are the SED's
predictions in that band, injected as the `member_flux` context node (the zeropoint
idiom). The SIGN problem -- modeled stars brighter than the light curve's blend allows, so
the derived flux goes negative -- is not gated: the element's defaults.yaml bounds [0, 1000]
become, on a derived element, the `potentials.soft_lower_bound`/`soft_upper_bound` barrier
(`components/parameter.md`, "a derived element gets one"), whose gradient acts on what the
element derives from (`f_blend`, `f_source`, the members' SED parameters). A tension is a
pull, never a NaN, and no `pt.where` is involved.

- **Which light curve.** Every light curve whose `band:` is in the instance's `bands:` is a
  candidate; two in one band RAISE, because each has its own blend (KMT's three I-band
  sites, different seeing). `light_curves: [name, ...]` names the ones the background is;
  `light_curves: []` ties none (every band then sampled).
- **Which stars: `psf_stars:` is the complete list, and the event's own members are
  required in it** (JDE 2026-10-27; `Background.event_blend_stars`,
  `_check_tie_membership`, stage 1). Some stars are in every light curve's blend by the
  event's own geometry, whatever the PSF:
  - every lens body that is a star -- the primary lens (always: `_validate_bodies` makes it a
    star) and a luminous stellar lens companion (2L1S, 3L1S; its separation is ~theta_E);
  - every star on the far side of a source's own orbit (xallarap): an unlensed companion
    whose orbit the light curve resolves in time is far below any PSF.

  A tied band whose `psf_stars` omits one of them RAISES naming it, and so does a
  `psf_stars` naming a source (its light is the f_source half, lensed). Not required, and
  so the user's statement: a resolved neighbor, and a star merely BOUND to a lens or source
  body in a static (or RV) orbit group -- a bound companion's separation is not fixed by
  the event, and a ground PSF may or may not hold it. A planet companion is dark here (the
  SED predicts no planet flux). Every member's flux in the tied band is the SED's
  prediction relative to the source's, and because the lens is always a member there is
  no member-free tie: a tied band needs an `sed:` block, a `band:` block naming the band
  and a filter the BC grid has, and a member the SED severed (no BC grid hosts its start)
  has no prediction -- each RAISES, naming the alternative (leave that light curve out of
  `light_curves:`, so the band is sampled). A dark lens (a free-floating planet, a
  remnant) is therefore not tieable today; untie its band. The tie also makes its members
  and the primary source SEEN by the SED (`SED.photometric_readers`, via
  `Background.sed_read_stars`), so their SED-side parameters are sampled, not pinned.
- **Why explicit rather than implicit.** Three spellings were possible: the lens implicit
  (psf_stars lists only the OTHER stars), the lens auto-added, or omitting it raises. The
  first two make `psf_stars: [Lens]` and `psf_stars: []` mean the same light -- two
  spellings for one statement (config.md's one-spelling rule) -- and the first would make
  the background's list differ from an astrometry dataset's for the very same PSF, where
  the lens must be named to put it on its own track. Explicit-and-required keeps ONE
  meaning for `psf_stars:` on both components ("every modeled non-source star in this
  PSF") and makes the identity hold in every model that builds.
- **Consistency with the astrometry.** A dataset reading the background in a tied band must
  name the same `psf_stars` as the background (else a star is counted twice or not at
  all) and may not set `fitpsfflux` there (the derived flux already uses the SED's member
  fluxes) -- both RAISE. Because the background's list now holds every required member,
  so must that dataset's; a luminous lens COMPANION therefore cannot be read by an
  astrometry dataset in a tied band (only the primary lens's track is derived, review
  8.10.4, and that dataset RAISES saying so) -- tie a band no astrometry dataset reads, or
  leave the light curve untied.
- **Not with `sed_constrains_blend: true` on the same light curve (RAISES).** That flag
  states the same decomposition a second time: `f_blend = f_S * sum_{non-sources} F_j/F_S
  + neighbor_flux`, as a `sed_blend_sigma` (0.2 mag) Gaussian on the magnitude, with
  `neighbor_flux >= 0` the unmodeled light. Its lens term is the background's exactly (the
  zeropoint enters as `f_S * 10^(-0.4 (m_L - m_S))` whether stated or derived from
  f_source), so with the tie it would hold `neighbor_flux` -- an extra sampled dimension --
  at `f_S * fluxratio` within 0.2 mag: two parameters for one light, which is what this
  component exists to prevent, and the "blend holds at least the lens" bound counted twice
  (the background's `[0, 1000]` barrier already is that bound, exact rather than to 0.2
  mag). Its member list is also every non-source star in the system, not the PSF's, so a
  neighbor outside the PSF would sit in one sum and not the other. The tie supersedes it:
  drop the flag, or untie that light curve.
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
- One light curve per band ties the background; the others in that band keep their own
  `f_blend` untied.
- The required-member rule is the LIGHT-CURVE tie's. An astrometry dataset reading the
  background in an UNTIED band is not (yet) made to name the lens in its own `psf_stars:`;
  there the background's flux is sampled, so an omitted lens is absorbed by it at the
  background's photocenter rather than on its own track.
- A dark lens (no BC grid hosts it) cannot be tied: its band must stay sampled.
- No parallax: the background's photocenter is treated as distant (a bulge background's
  ~0.12 mas parallax times its flux share is below current catalog parallax errors).

Tests: `tests/test_catalog_astrometry.py` (one object shared by two catalogs, the field
prior, the photocenter track inside the simulated catalog solutions, and the configuration
raises; the tie's identity to 1e-12 at random points with the lens and a neighbor
subtracted, a luminous lens companion subtracted and SEEN, the lens or a source-orbit
companion left out raising, the SED/BC-grid/sed_constrains_blend raises, and C and JAX
gradients of the tied model agreeing).
