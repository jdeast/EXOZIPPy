"""Generate ONE identically-shaped fit config per DC2018 event.

THE POINT IS THAT NOTHING HERE IS PER-EVENT TUNED.  Every number this
writes comes from a rule applied the same way to every event, and the rules
are named below.  A sweep whose configs were each nudged into working is a
sweep whose results mean nothing, so the only per-event inputs are
measurements: the coordinates, the line-of-sight extinction, the data
files, and the PSPL seed our own peak finder produced.

THE RECIPE IS v7's, which is the only microlensing configuration in this
repo that has been validated against truth end to end (review 8.6.7): the
three observable coordinate swaps (fitmurel / fitpirel / fitthetae) on the
event, the SED relinked to rho (`star_constrains_rho: True`), stellar teff
and radius UNPINNED with feh at N(0, 0.5), and the Hogg mixture on both
light curves.  On DC2018-128 that combination moved mu_rel from 10.83 to
1.469 against a truth of 1.814 and put five of seven physical quantities
inside their 95% intervals.

WHAT DELIBERATELY IS NOT CARRIED OVER FROM v7:
  * `mulensevent.log_theta_E: {initval: -1.27}`.  That start was read off
    v5's fitted theta_star / rho -- i.e. it used the answer.  A blind sweep
    cannot, so it is dropped and the relaxation engine plus the seed polish
    have to find theta_E themselves.  b88d791a (whitening re-centred on the
    polished start) landed for exactly this kind of case, so this run also
    tests it.  IF THE SWEEP FAILS ON 128, WHERE v7 SUCCEEDED, THE START IS
    THE FIRST SUSPECT.
  * MMEXOFAST.  `mmexofast:` points at dc18_seed.py's output instead --
    same JSON contract, a PSPL peak fit rather than a binary-lens search,
    and s/q/alpha left at defaults.yaml generics so the sampler has to find
    the planet on its own.
  * The solar pins run_event.py still writes (`star.Lens.teff: sigma: 0`
    and friends).  2.9.10 showed those force the lens mass through the SED
    floor potentials; v5 removed them and the lens mass went from 1.98x
    truth to 0.85x.

ALSO CARRIED, ADDED 2026-09-17 ("update the sweep to follow all currently
known best practices -- ptde, reparameterizations, data derived tight
bounds, and the correct av prior").  Each of the four was measured on event
194 and each was measured SEPARATELY; this is the first configuration that
combines them, which is the one thing about it that is not yet validated.
  * `ptde`, not `ptde_async`.  Seven runs of 194: ptde put all 78 chains in
    the good-likelihood region 3/3, under three different configurations;
    every ptde_async run put at most 40 there and usually under three.
  * `fitu0te` on the source.  Best of the three ptde arms by ESS(core) --
    30-32k against 23-25k (tight bounds) and 9.9k (control), Rhat 1.00.
  * Data-derived tight bounds (8.2.2 path 1): t_0 to the observing span,
    |u_0| <= 3, log_f_total to the measured baseline +/-2 dex.  Worth
    9.9k -> 23-25k ESS on their own.  `err_scale` is NOT the tight arm's
    [0.1, 10] but the LATER and tighter [0.5, 2.0] (JDE 2026-09-15).
  * The colour-anchored av prior, which is what AV_COLS is about below.
AND ONE THING NO ARM HAS HAD: all seven predate #293, so `ptde` ignored
store_hot_chains in every sync arm -- no hot-chain mode detector.  That is
fixed, so this sweep gets the good chains AND mode discovery together.

THE RULES, one line each:
  ra, dec           event_info.txt columns 3-4.
  star.Source.av    the red-clump (A_W149, A_Z087) of event_info.txt
                    columns 6-9, inverted for A_V through OUR OWN BC grid
                    and anchored on the COLOUR.  v7 put A_W149 itself here,
                    which is ~5x too small; see AV_COLS below for why the
                    colour is the right anchor and why the two quoted
                    dispersions are one correlated error.
  out_scale         upper = 10x the per-LC median flux error, initval = 1x,
                    in the file's flux system (review 8.6.3, RULED).
  zeropoint         N(22.0, 0.02) -- the simulation's, identical for every
                    event because it is the same instrument.  Deliberately
                    NOT widened for the C29 residual: with `filters: []`
                    these two are the only colour information in the fit.
                    See the comment at the zeropoint for the full reason.
  source.t_0        the observing span of the light curves themselves.
  source.u_0        [-3, 3]: the source was magnified.  A soft barrier
                    rather than hard support, since fitu0te makes u_0
                    derived.
  log_f_total       log10(median flux of that light curve) +/- 2 dex.
  err_scale         [0.5, 2.0] (JDE 2026-09-15, 8.2.2) -- these are
                    simulated curves with honest error bars, so err_scale
                    is a check, not a fit.
  u1                pinned to 0: no limb darkening is simulated.
  everything else   defaults.yaml.
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402

# $DC18_DATA, matching run_event.py's spelling.  Hardcoding this made every
# generated config machine-locked to one home directory: 36 tracked files
# pinned their seed JSON, both light curves, `prefix`, `sed.file` and
# `parameter_file` to /home/jeastman, and on a second machine not one of
# them resolved -- the fit died in load_data before sampling (2.4.19).  The
# rest of the generator was already portable (everything else is
# os.path.abspath against the CWD), so this constant was the whole defect,
# and it is INVISIBLE on the machine that wrote it.
DATA = os.environ.get(
    "DC18_DATA", "/home/jeastman/python/MMEXOFAST/data/2018DataChallenge"
)
BANDS = [("W149", "Roman/WFI.F146"), ("Z087", "Roman/WFI.F087")]

# event_info.txt columns, 0-based: 5 = A_W149 and 6 its dispersion,
# 7 = A_Z087 and 8 its dispersion.
#
# THE OLD RULE PUT A_W149 STRAIGHT INTO `av` AND THAT WAS SIMPLY WRONG: the
# SED's parameter is A_V, and A_W149 is roughly a fifth of it, so every
# event ran with a prior ~5x too small.  It survived because it MASKED a
# second defect -- the BC grid's Av axis stopped at 6.0 mag, so a correct
# prior would have piled against a hard bound instead (7.7.3).  Both are
# fixed now; the axis reaches 20 mag.
#
# WE ANCHOR THE COLOUR, NOT EITHER BAND.  Our model integrates a reddened
# spectrum through the passband; the simulation reddened monochromatically
# at each filter's effective wavelength (conventions.md C29, demonstrated to
# 0.1% on the band ratio and across 293 events).  For a filter as wide as
# W149 those differ, so NO single av reproduces both simulated band
# extinctions in our model -- on event 194 the anchors span av = 9.01
# (colour) to 11.75 (A_W149).  Anchoring the colour leaves both bands off
# by the SAME amount, and a common-mode grey offset is degenerate with
# distance and radius, which the SED already fits; anchoring either band
# instead puts the residual in COLOUR, where only teffsed can absorb it.
#
# WE DO NOT ENGINEER AROUND IT, WE BUDGET FOR IT (JDE 2026-09-17: "raise
# our zero point error and/or our av prior error to accommodate their
# error, compute the impact, and list the systematic disagreement as a
# caveat").  The leftover grey term is handed to the zeropoint, whose prior
# was N(22.0, 0.02) -- two orders of magnitude too tight to absorb it, which
# is why it was being absorbed by theta_star instead.
#
# THE TWO DISPERSIONS ARE ONE FRACTIONAL ERROR, FULLY CORRELATED, and
# treating them as independent overstates the colour error by ~2.4x.
# Measured across all six events: sigma_Z/sigma_W = 1.885-2.000 against
# A_Z/A_W = 1.918-1.924, and sigma_W/A_W = sigma_Z/A_Z to three digits
# (event 194: 0.1200 vs 0.1191).  So event_info quotes ONE fractional
# uncertainty on the line-of-sight extinction, applied to both bands, and
# the colour excess carries that same fraction rather than the quadrature
# sum.
AV_COLS = (5, 6)
AZ_COLS = (7, 8)

# Fractional uncertainty on the reddening LAW's shape, added in quadrature
# to the clump's own.  The shipped extinction_law.ascii runs ~6% high of
# CCM89 in J, H and Ks (measured, review 2.9.16), consistently enough to
# look like a different NIR power law rather than scatter.  It also keeps
# event 008 off a zero-width prior: both its dispersions are quoted as 0.01,
# which rounds the correlated colour error to exactly 0.
LAW_FRAC_SIGMA = 0.06


def av_from_clump_colour(a_w149, sig_w149, a_z087, sig_z087):
    """
    Invert the clump's (A_W149, A_Z087) for `av` THROUGH OUR OWN BC GRID.

    Returns (av, sigma_av, grey_residual), where grey_residual is the
    common-mode band-extinction offset our integrated model still carries at
    that av -- the piece the zeropoint prior has to be wide enough to absorb.

    This inverts the INTEGRATED band extinction the model actually uses
    (BC(Av=0) - BC(Av) off the shipped Roman table), NOT the monochromatic
    law in components.sed.extinction.  Using the law here would reproduce the
    simulation's own convention and so hide the disagreement rather than
    budget for it (C29).

    The cell is the red clump (Teff 4800 K, logg 2.5, solar feh).  That
    choice is cheap: measured across 36 cells spanning Teff 4500-5500,
    logg 2.0-3.0 and feh -0.5 to +0.3, the colour-anchored av has a standard
    deviation of 0.10 and is independent of logg and feh to +/-0.01,
    depending on Teff alone at 0.28 per 1000 K.
    """
    bc = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(C.__file__))),
        "..",
        "src",
        "exozippy",
        "models",
        "NextGen",
        "BCs",
        "Roman",
        "feh+0.0_afe+0.0.Roman",
    )
    bc = os.path.normpath(bc)
    if not os.path.exists(bc):
        raise SystemExit(
            "cannot find the Roman BC table at %s; the colour-anchored av "
            "prior needs it (regenerate with components/sed/make_bc.py)" % bc
        )
    header = next(ln for ln in io.open(bc) if "lgTef" in ln)
    cols = header.lstrip("#").split()
    tab = np.genfromtxt(bc, comments="#", skip_header=1)
    i146 = 6 + cols[6:].index("WFI_F146")
    i087 = 6 + cols[6:].index("WFI_F087")

    lgt = tab[:, 0]
    cell = np.isclose(
        lgt, lgt[np.argmin(abs(lgt - np.log10(4800.0)))]
    ) & np.isclose(tab[:, 1], 2.5)
    sub = tab[cell]
    sub = sub[np.argsort(sub[:, 4])]
    av_pts = sub[:, 4]
    a146 = sub[0, i146] - sub[:, i146]
    a087 = sub[0, i087] - sub[:, i087]

    excess = a_z087 - a_w149
    if not a087[-1] - a146[-1] > excess:
        raise SystemExit(
            "colour excess %.3f exceeds the grid's reach (%.3f at av=%g); "
            "extend the Av axis rather than extrapolating"
            % (excess, a087[-1] - a146[-1], av_pts[-1])
        )
    av = float(np.interp(excess, a087 - a146, av_pts))

    # ONE fractional error, fully correlated (see AV_COLS), so the colour
    # carries the same fraction and av scales with it; the law's own shape
    # uncertainty goes in quadrature.
    frac_clump = 0.5 * (sig_w149 / a_w149 + sig_z087 / a_z087)
    sigma = av * float(np.hypot(frac_clump, LAW_FRAC_SIGMA))

    # AND THE CONVENTION AMBIGUITY ITSELF, AS A WIDTH ON av.  The anchors
    # disagree -- colour gives this av, A_W149 alone gives a larger one --
    # and that spread IS the systematic, so half of it belongs in the prior
    # width.  Putting it HERE rather than on the zeropoint is deliberate and
    # measured: with `filters: []` the SED carries no photometry of its own,
    # so the ONLY colour information in the fit is the two-band zeropoint
    # tie, and widening both zeropoints to the grey residual would inject
    # sqrt(2)*grey of COLOUR slack -- 0.82 mag on event 194 -- destroying the
    # constraint the colour anchor exists to exploit.  `av` is the correctly
    # CORRELATED nuisance: moving it reddens both bands together in the ratio
    # the law dictates, which is mostly grey with exactly the colour term the
    # physics implies, so it buys freedom along the reddening direction
    # without opening the differential.
    av_band = float(np.interp(a_w149, a146, av_pts))
    sigma = float(np.hypot(sigma, 0.5 * abs(av_band - av)))

    grey = float(np.interp(av, av_pts, a146)) - a_w149
    return av, sigma, grey


def event_info_row(event):
    rows = np.genfromtxt(
        os.path.join(DATA, "event_info.txt"), dtype=None, encoding="utf-8"
    )
    for r in rows:
        if int(r[1]) == int(event):
            return r
    raise ValueError("event %s not in event_info.txt" % event)


def median_flux_err(path):
    e = np.loadtxt(path)[:, 2]
    return float(np.median(e[np.isfinite(e) & (e > 0)]))


def median_flux(path):
    """Median of the flux column, in the file's own flux system."""
    d = np.loadtxt(path)
    f, e = d[:, 1], d[:, 2]
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0)
    return float(np.median(f[ok]))


def observing_span(paths):
    """(first, last) finite epoch across every light curve, in the files' JD."""
    lo, hi = None, None
    for path in paths:
        d = np.loadtxt(path)
        t, e = d[:, 0], d[:, 2]
        t = t[np.isfinite(t) & np.isfinite(e) & (e > 0)]
        if not len(t):
            continue
        lo = t.min() if lo is None else min(lo, t.min())
        hi = t.max() if hi is None else max(hi, t.max())
    if lo is None:
        raise SystemExit("no finite epochs in any light curve")
    return float(lo), float(hi)


def build(event, outdir, draws, tune, cores, t_max):
    ev3 = "%03d" % int(event)
    name = "DC2018_%s" % ev3
    row = event_info_row(event)
    ra, dec = float(row[2]), float(row[3])
    av_mu, av_sd, av_grey = av_from_clump_colour(
        float(row[AV_COLS[0]]),
        float(row[AV_COLS[1]]),
        float(row[AZ_COLS[0]]),
        float(row[AZ_COLS[1]]),
    )

    files = {
        b: os.path.join(DATA, "n20180816.%s.WFIRST18.%s.txt" % (b, ev3))
        for b, _ in BANDS
    }
    for b, p in files.items():
        if not os.path.exists(p):
            raise SystemExit("missing light curve: %s" % p)

    seed = os.path.abspath("events/%s/%s_seed.json" % (ev3, name))
    if not os.path.exists(seed):
        raise SystemExit(
            "no seed for %s -- run: python dc18_seed.py %s" % (ev3, int(event))
        )

    base = os.path.abspath(os.path.join(outdir, ev3))
    os.makedirs(base, exist_ok=True)

    # The SED block: no catalog photometry.  These are simulated light
    # curves and the challenge released none, so the SED constrains the
    # source ONLY through the calibrated microlensing baseline flux (the
    # per-lightcurve zeropoint hook) with the Roman BC grids pulled in via
    # each band's `filter:`.  This is JDE's axis 3 -- the absolute source
    # flux constraining its radius -- and it is what v7 validated.
    sed_path = os.path.join(base, "%s.sed.yaml" % name)
    io.open(sed_path, "w", encoding="utf-8").write(
        "model: NextGen\nfilters: []\n"
    )

    cfg = {
        "run": {"name": name},
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
        "mulensevent": [
            {
                "finite_source": True,
                "fitmurel": True,
                "fitpirel": True,
                "fitthetae": True,
                "mmexofast": seed,
            }
        ],
        "lens": [{"body": "star.Lens"}, {"body": "planet.Companion"}],
        "source": [
            {
                "body": "star.Source",
                "star_constrains_rho": True,
                # THE BEST-PERFORMING COORDINATE MEASURED SO FAR.  Sampling
                # u_0*t_E instead of u_0 attacks the u_0/t_E/f_blend
                # degeneracy that makes the heavily blended events hard, and
                # on event 194 it was the best of the three ptde arms:
                # ESS(core) 30-32k against 23-25k for tight bounds and 9.9k
                # for the control, at Rhat 1.00 and 78/78 chains.
                # NOTE this makes `u_0` a DERIVED parameter ("from_u0te",
                # source.py), with `u0te` sampled on [-5000, 5000], so the
                # |u_0| <= 3 bound below lands on a derived parameter and
                # acts as a SOFT barrier rather than as hard support.  That
                # is the intended behaviour and is why it is still worth
                # writing: it says "this is a magnified event", which is
                # true whichever coordinate samples it.
                "fitu0te": True,
            }
        ],
        "galacticmodel": [{"name": name, "anchor_idx": 1}],
        "band": (
            [{"name": b, "filter": f, "ld_law": "linear"} for b, f in BANDS]
            + [
                {
                    "name": "Ks_bcgrid",
                    "filter": "2MASS/2MASS.Ks",
                    "ld_law": "linear",
                }
            ]
        ),
        "mulensinstrument": [
            {
                "name": "Roman_%s" % b,
                "file": files[b],
                "data_format": "flux",
                "observer_location": "roman_simulated_2018dc",
                "band": b,
                "likelihood": "hogg",
            }
            for b, _ in BANDS
        ],
        "prefix": os.path.join(base, name),
        "sed": {"file": sed_path},
        "torres": [{"star": "Source", "constrain": ["mass", "radius"]}],
        "mann": [
            {
                "star": "Lens",
                "ks": "synthetic",
                "constrain": ["mass", "radius"],
            }
        ],
        "sampler": {
            # ptde, NOT ptde_async.  Measured on event 194, seven runs
            # (notes/supercomputer_queue.txt): ptde put ALL 78 chains in the
            # good-likelihood region three times out of three, under three
            # different configurations (control, tight bounds, fitu0te),
            # while every ptde_async run of the same event put at most 40
            # there and usually fewer than three.  ESS followed: u0te_sync
            # reached ~30,000 from 3,595 draws at 36 GB where its async twin
            # needed 26,984 draws and 385 GB.  Those arms all predate #293,
            # so `ptde` ignored store_hot_chains in every one of them; that
            # is fixed, so the sweep gets the good chains AND the hot-chain
            # mode detector, which no arm has had together.
            "method": "ptde",
            "cores": cores,
            "n_temps": "auto",
            "T_max": t_max,
            "tune": tune,
            "draws": draws,
            "nthin": 1,
            "recompute_trace": True,
            "eval_timeout": 10,
            # Retain the hot rungs: outputs.ledger.discover_hot_modes
            # clusters them, and finding ALL the modes -- not just the one
            # the cold chains settled in -- is half of what this sweep is
            # for.  Explicit rather than "auto" so the sweep does not
            # silently lose mode discovery on an event where auto declines.
            "store_hot_chains": True,
        },
        "modes": {"weights": "evidence"},
        "parameter_file": os.path.join(base, "%s.params.yaml" % name),
    }

    t_lo, t_hi = observing_span(files.values())

    params = {
        # BOTH stars carry the event's line of sight.  The galactic model
        # anchors on the Source (anchor_idx 1 below) and the magnification Op
        # freezes the SOURCE's coordinates for the parallax projection, so a
        # Source left at defaults.yaml's (180, 0) put every prior and the
        # parallax geometry of the 2026-09 sweep at Galactic (l 276, b +60)
        # -- the R_source/2 pull (notes 2026-09-24).  GalacticModel now
        # refuses an unpositioned anchor; this is the position it wants.
        "star.Lens.ra": {"initval": ra, "sigma": 0},
        "star.Lens.dec": {"initval": dec, "sigma": 0},
        "star.Source.ra": {"initval": ra, "sigma": 0},
        "star.Source.dec": {"initval": dec, "sigma": 0},
        # Generic, NOT solar: teff and radius stay FREE (2.9.10), and feh
        # gets the broad prior JDE specified rather than a pin.
        "star.Lens.feh": {"mu": 0.0, "sigma": 0.5},
        # Not a microlensing observable and not used by anything here;
        # pinning it keeps it out of the sampled dimension count.
        "planet.Companion.radius": {"sigma": 0},
        "star.Source.av": {"mu": av_mu, "sigma": av_sd},
        # No limb darkening is present in the simulation.
        "band.W149.u1": {"initval": 0.0, "sigma": 0},
        "band.Z087.u1": {"initval": 0.0, "sigma": 0},
        # 2.9.11: with `filters: []` there is no photometry for errscale to
        # scale, so left free it samples its U(0.001, 1000) prior and
        # reports a median of ~500 that reads like a measurement.  Pin it.
        "sed.errscale": {"initval": 1.0, "sigma": 0},
        # DATA-DERIVED TIGHT BOUNDS (8.2.2 path 1), as the ab194 `tight`
        # arm applied them by hand.  On event 194 they took ESS(core) from
        # 9.9k to 23-25k at Rhat 1.00 under ptde, so they are not free --
        # they buy real mixing by keeping the hot rungs and the start
        # dispersion out of volume the data have already excluded.
        #
        # t_0 CANNOT LIE OUTSIDE THE OBSERVATIONS.  A peak before the first
        # epoch or after the last is not a detection of anything; this is
        # the span of the light curves themselves, not a guess.
        "source.Source.t_0": {"lower": t_lo, "upper": t_hi},
        # |u_0| <= 3 says the source was MAGNIFIED (A - 1 < 0.7% at u_0 = 3).
        # Under fitu0te this is a soft barrier on a derived parameter, which
        # is the right strength for a statement this weak.
        "source.Source.u_0": {"lower": -3.0, "upper": 3.0},
    }
    for b, _ in BANDS:
        inst = "mulensinstrument.Roman_%s" % b
        med = median_flux_err(files[b])
        # LEFT AT THE SIMULATION'S OWN WIDTH, and that is a decision, not an
        # oversight.  The C29 convention residual is a GREY band-extinction
        # offset (-0.07 mag on event 008 up to -0.58 on 194) and the
        # zeropoint is where a grey term would naturally be absorbed -- but
        # these configs run `filters: []`, so the SED has no photometry of
        # its own and this pair of zeropoints is the ONLY colour information
        # in the fit.  Two independent priors of width 0.58 admit
        # sqrt(2)*0.58 = 0.82 mag of COLOUR slack, which is an order of
        # magnitude more than the colour signal separating plausible source
        # temperatures; it would buy an honest error bar on theta_star by
        # throwing away teffsed.  The format cannot express the one prior
        # that would be right here -- a single term shared between the bands
        # -- and building that is the engineering JDE ruled out.
        # So the convention width goes on `av` instead (see
        # av_from_clump_colour), which is the correctly correlated nuisance,
        # and the leftover theta_star bias is REPORTED rather than absorbed:
        # dc18_truth_table.py prints it beside the recovery table.
        params["%s.zeropoint" % inst] = {"mu": 22.0, "sigma": 0.02}
        params["%s.out_scale" % inst] = {"upper": 10.0 * med, "initval": med}
        # THE BASELINE FLUX IS MEASURED, SO BOUND THE TOTAL FLUX BY IT.
        # +/-2 dex around the median of this light curve's own flux column:
        # 100x either side of the observed baseline is generous for a
        # blended source and still excludes the decades of flux space the
        # sampler otherwise explores.  Same rule the `tight` arm used
        # (verified: its numbers reproduce as log10(median flux) +/- 2).
        lf = float(np.log10(median_flux(files[b])))
        params["%s.log_f_total" % inst] = {
            "lower": lf - 2.0,
            "upper": lf + 2.0,
        }
        # These are SIMULATED curves with honest error bars, so err_scale is
        # a check, not a fit: 0.5-2 (JDE 2026-09-15, review 8.2.2), tighter
        # than defaults.yaml's 0.01-100.  On 226 the point-lens basin had
        # inflated both bands 300-460x and turned the anomaly into noise;
        # with this bound the fit has to explain the data or sit against
        # the wall where the near-bound warning names the real remedy.
        params["%s.err_scale" % inst] = {"lower": 0.5, "upper": 2.0}

    io.open(cfg["parameter_file"], "w", encoding="utf-8").write(
        yaml.safe_dump(params, sort_keys=True, default_flow_style=False)
    )
    cfg_path = os.path.join(base, "%s.yaml" % name)
    io.open(cfg_path, "w", encoding="utf-8").write(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    )

    s = json.load(io.open(seed))["fits"][0]["parameters"]
    print(
        "%s  ra=%.4f dec=%.4f  av=%.2f+/-%.2f (grey %+.2f)  "
        "seed t_0=%.3f u_0=%.4f "
        "t_E=%.2f\n    -> %s"
        % (
            name,
            ra,
            dec,
            av_mu,
            av_sd,
            av_grey,
            s["t_0"],
            s["u_0"],
            s["t_E"],
            cfg_path,
        )
    )
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+", type=int)
    ap.add_argument("--outdir", default="sweep")
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--tune", type=int, default=5000)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--t-max", type=float, default=200.0)
    a = ap.parse_args()
    for ev in a.events:
        build(ev, a.outdir, a.draws, a.tune, a.cores, a.t_max)


if __name__ == "__main__":
    main()
