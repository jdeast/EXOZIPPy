# Dust-map priors: `exozippy-mkdust`

`src/exozippy/utilities/mkdust.py`. Given a sky position, it reads every extinction map it can
reach and writes `<name>.dust.params.yaml`: candidate priors on `sed.dust_anchor`, ONE live and
every other commented out (the way `exozippy-mkticsed` comments out non-default catalog rows).
Read this before changing a map, a conversion ratio, a width, or which map is the default.

```
exozippy-mkdust 17:53:48.01 -29:15:47.48 --name KB242087 --outpath .
```

RA/Dec in decimal degrees, or sexagesimal (RA in hours) with colons. A negative sexagesimal Dec
needs no quoting or `--`.

## What it feeds, and why no new key

The SED's sight-line dust profile (`components/sed/sed.md`, "The sight-line dust profile")
samples one total, `sed.dust_anchor`: the extinction at the profile's anchor, in A_V unless the
system config names `sed: {dust_anchor_band: <filter>}`. A red-clump map states the extinction to
the clump, at the bar, which is where the anchor sits when the fit has a red clump or a galactic
model (the bar's mean distance); with neither, the anchor is the whole column, a few percent more
at |b| ~ 3-4 deg. So every map value is an ordinary `{mu, sigma, citation}` prior on
`sed.dust_anchor`. The prior needs the profile on: two or more SED objects at different
distances, or `sed: {distance_constrains_av: true}` for one.

A line written in a band other than V (marked `[band X]` in the file) also needs
`sed: {dust_anchor_band: X}` in the system config; without it the number is read as A_V. The
file says so above the lines. The live line is always A_V, so it can never be misread that way.

With calibrated red-clump rows in the fit (`components/redclump/redclump.md`), use none of the
bulge maps: they are built from the same clump stars, so the clump would be counted twice.

## The default

Following Nataf's 2026 recommendation for bulge sight lines, the live line is the FIRST of these
that gives a usable value:

1. Surot et al. (2020), VVV E(J-Ks) (VizieR J/A+A/644/A140; the nearest node in a 1' cone).
2. Zelakiewicz et al. (2025), UKIRT A_Ks (Zenodo record 14919461, fetched md5-checked on first
   use; the 2' x 2' pixel holding the target). A pixel the authors flag as a lower limit is
   written, marked, and never the default.
3. Nataf et al. (2013), OGLE-III A_I and E(V-I) (Table 3 from the OGLE archive, fetched size- and
   md5-checked; the nearest reliable point within 600", as the OGLE calculator does).
4. Schlafly & Finkbeiner (2011)'s recalibrated SFD E(B-V) (IRSA), only at |b| >= 5 deg: it is the
   whole column, and not valid toward the inner bulge.

Surot's map covers Zelakiewicz's, so Zelakiewicz is the default only when Surot is unreachable;
the file then says which preferred map it passed over and why (a failed query is listed as
FAILED with the error, and logged as a WARNING).

### Why the default is A_V converted by hand, not Ks

`sed.md` lets a map's value enter in its own band (`dust_anchor_band`), converted by the fit's
law. That is the right channel for a Ks map ONLY once the law's near-infrared slope is right. The
fit's law today is CCM89 + O'Donnell, tabulated at R_V = 3.1 or with a fitted R_V, and its
near-infrared slope is fixed: at R_V = 3.1, A_V/A_Ks ~ 8.4, while the bulge measures 13-15
(Nataf et al. 2016's median 13.44; 15.3 from the Nataf et al. 2013 means below). A Ks anchor would
put ~40% too little dust in V and I, the bands most microlensing data are in. So the live line
converts the map to A_V with Nataf et al. (2013)'s bulge ratios, and the Ks-native lines are
written commented. They become the better choice when the law's near-infrared slope is fitted.

## The conversions and widths

- **Nataf et al. (2013) bulge means.** A_I = 1.217 E(V-I) (their eq. 29 at the mean R_JKVI) and
  R_JKVI = E(J-Ks)/E(V-I) = 0.3433 (their Sec. 8.1). Hence A_V = A_I + E(V-I) = 2.217 E(V-I), and

      A_V / E(J-Ks) = (1.217 + 1) / 0.3433 = 6.458.

- **The ratio's sight-line scatter, 7%.** Over the reliable Table 3 sight lines with E(V-I) > 0.5,
  (A_I + E(V-I)) / (R_JKVI E(V-I)) has mean 6.465 (agreeing with 6.458) and a fractional standard
  deviation of 0.070 (0.070 again for |b| < 3). That is the width a mean ratio carries on one
  sight line; it is added in quadrature to the map's own error.
- **Surot et al. (2020):** A_V = 6.458 E(J-Ks); sigma from the node's E(J-Ks) error and the 7%.
  The catalog's error is the error of the node MEAN (the star-to-star dispersion is sqrt(20) times
  larger); the anchor is a pencil-beam value whose per-object scatter the profile models
  separately (`star.av_frac`, `redclump.av_frac`), so the mean's error is the right one. The
  Ks line is A_Ks = 0.422167 E(J-Ks), Surot's own measured VVV bulge vector (their eq. 1), in
  `Paranal/VISTA.Ks`.
- **Zelakiewicz et al. (2025):** A_Ks to E(J-Ks) through the same Surot vector, then the Nataf
  ratio: A_V = 6.458 / 0.422167 A_Ks = 15.30 A_Ks. The width adds a 10% term for the A_Ks to
  E(J-Ks) step, the spread of published bulge A_Ks/E(J-Ks) (0.422 Surot et al. 2020, 0.428
  Alonso-Garcia et al. 2017, 0.423 Nagatomo et al. 2019, 0.49 Wang & Chen 2019), to the map error
  (the mean of the asymmetric errors) and the 7%. The Ks line is written in `2MASS/2MASS.Ks`:
  UKIRT WFCAM K has no bolometric-correction column.
- **Nataf et al. (2013) Table 3:** A_V = A_I + E(V-I) directly, no NIR step. The width is the
  listed differential reddening S_E(V-I) propagated, (A_I/E(V-I) + 1) S_E(V-I); a point that lists
  S_E(V-I) = 0 gets Nataf's 9% field-scale granularity (their eq. 20), and the note says so. The
  A_I line is in `Generic/Cousins.I`.
- **SFD:** A_V = 3.1 E(B-V) on the Schlafly & Finkbeiner (2011) recalibration, sigma from the 5'
  standard deviation IRSA reports.

On KMT-2024-BLG-2087 ((l, b) = (0.64, -1.70)) the two independent routes agree: Zelakiewicz's
A_Ks = 0.216 gives A_V = 3.30 +/- 0.86 and Nataf's OGLE-III point gives A_V = 3.36 +/- 0.35.

## Maps it does not read

Listed in every output with the reason, and never approximated:

- Zucker et al. (2025), 3D: a multi-GB cube read through the `dustmaps` package, not a web
  query; it saturates at A_V ~ 12 and does not reach the bulge itself.
- Green et al. (2019), Bayestar19: the `dustmaps` package (multi-GB); dec > -30 only.
- Gonzalez et al. (2012), BEAM: an interactive web calculator only; Surot et al. (2020)
  supersedes it over the same VVV area.
- VVV-ExMaps (the BaSTI web service of Surot et al. 2020): needs a user ID; the same map is read
  from VizieR.
- Marshall et al. (2006): a 15', Besancon-model 3D map; not read.

## Tests

`tests/test_mkdust.py`: each reader on a fixture in the map's own format (no network), the
default order and a failed query, the SFD latitude rule, the CLI's negative Dec, every written
line parsing as YAML with references.bib citations, and the live line resolving as the
`sed.dust_anchor` prior of a dust-profile system.
