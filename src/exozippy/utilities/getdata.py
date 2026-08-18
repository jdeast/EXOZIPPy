"""Download TESS/Kepler light curves and format them for EXOFASTv2/EXOZIPPy.

This is the importable home of the former scripts/getdata.py. The CLI is
defined by build_parser() and driven by main(argv=None); scripts/getdata.py
is now a thin wrapper that calls main().

The heavy lightkurve import is deferred into run() so that build_parser()
and the introspection layer stay importable without the download stack.

Please acknowledge Lightkurve when using this tool:

This research made use of Lightkurve, a Python package for Kepler and TESS
data analysis (Lightkurve Collaboration 2018).
"""

import argparse
import datetime
import glob
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# --- product selection tables -------------------------------------------------
#
# When one sector offers several products, exactly one is kept: the first
# exposure time present in EXPTIME_PRIORITY, and among those the first author
# present in AUTHOR_PRIORITY. Both are ordered best-first and both are read by
# the same _highest_priority() helper, so the tables are the documentation.
#
# AUTHOR_PRIORITY is also the set of authors run() asks MAST for, so the ranked
# set and the searched set cannot drift apart. Every entry must be readable --
# in AUTHOR_MISSION -- or listed in UNSUPPORTED_AUTHORS and ranked below every
# readable one, or the tie-break can hand run() a product it then discards.
# "TESS" used to head this table and was neither: it is a lightkurve *mission*
# ("Kepler", "K2", "TESS"), not an author. Official TESS pipeline products are
# authored "SPOC" (lightkurve's own search_lightcurve docstring), MAST has no
# observation with provenance_name = "TESS" (author is a synonym for
# provenance_name), and lightkurve's AUTHOR_LINKS / result-sort tables do not
# list it either. So it ranked a value that cannot be returned, first.

# Short cadence first (120, 200, 20 s), then the long-cadence FFI products.
EXPTIME_PRIORITY = (120, 200, 20, 300, 600, 1800)

AUTHOR_PRIORITY = (
    "SPOC",
    "TESS-SPOC",
    "QLP",
    "Kepler",
    "K2SFF",
    "EVEREST",
    "K2",
    # readable authors above this line; the rest are recognized but unreadable
    # (UNSUPPORTED_AUTHORS) and so must stay last -- they are ranked at all
    # only so that a sector offering nothing else is skipped with their own
    # message rather than the ambiguous-tie one.
    "CDIPS",
    "TASOC",
)


@dataclass(frozen=True)
class Mission:
    """How to read one mission's products: epoch, bandpass and sector label.

    ``bjd_offset`` is added to the light curve's times to recover BJD_TDB.
    ``sector`` turns the lightkurve mission string ("TESS Sector 14",
    "Kepler Quarter 07", "K2 Campaign 13") into the tag that goes in the
    filename. ``undeblend`` says whether -u is supported: only the TESS
    products carry the CROWDSAP the correction inverts.
    """

    bjd_offset: float
    filter: str
    telescope: str
    sector: Callable[[str], str]
    undeblend: bool


# Kepler and K2 share an epoch and a bandpass and differ only in how their
# observing unit is labelled; both take the last two characters of the mission
# string, while TESS parses its trailing sector number and zero-pads it.
MISSIONS = {
    "Kepler": Mission(
        bjd_offset=2454833.0,
        filter="Kepler",
        telescope="Kepler",
        sector=lambda mission: "Q" + mission[-2:],
        undeblend=False,
    ),
    "K2": Mission(
        bjd_offset=2454833.0,
        filter="Kepler",
        telescope="Kepler",
        sector=lambda mission: "C" + mission[-2:],
        undeblend=False,
    ),
    "TESS": Mission(
        bjd_offset=2457000.0,
        filter="TESS",
        telescope="TESS",
        sector=lambda mission: "S" + str(int(mission.split()[-1])).zfill(2),
        undeblend=True,
    ),
}

# Which mission each pipeline's products belong to. An author that is absent
# is not supported: CDIPS and TASOC put their flux (and errors) somewhere else,
# and anything else is unrecognized.
AUTHOR_MISSION = {
    "Kepler": "Kepler",
    "K2": "K2",
    "EVEREST": "K2",
    "K2SFF": "K2",
    "K2VARCAT": "K2",
    "TESS-SPOC": "TESS",
    "QLP": "TESS",
    "SPOC": "TESS",
}

# Recognized, but their products are not in a format this tool can read.
UNSUPPORTED_AUTHORS = ("CDIPS", "TASOC")


def _highest_priority(values, priority):
    """Return the indices of the best-ranked value present in ``values``.

    ``priority`` is ordered best-first; the first of its entries that appears
    in ``values`` wins, and every index holding it is returned (the caller
    breaks a remaining tie with the next table, or gives up). An empty result
    means ``values`` holds nothing the table ranks at all.
    """
    for want in priority:
        match = np.where(values == want)[0]
        if len(match) > 0:
            return match
    return np.array([], dtype=int)


def build_parser():
    """Return the argparse parser for the getdata utility."""
    parser = argparse.ArgumentParser(
        prog="getdata.py",
        description="Downloads TESS/Kepler data and formats it for EXOFASTv2",
    )
    parser.add_argument("id", help="SIMBAD-resolvable star name")
    parser.add_argument(
        "-d",
        "--depth",
        default=0.03,
        type=float,
        dest="depth",
        help="Fractional transit depth. Flux >= 1-depth will not be clipped.",
    )
    parser.add_argument(
        "-n",
        "--nsigma",
        default=5.0,
        type=float,
        dest="nsigma",
        help="N sigma clipping. Negative values will skip clipping.",
    )
    parser.add_argument(
        "-p", "--path", default=".", dest="path", help="path to output files"
    )
    parser.add_argument(
        "-u",
        "--undeblend",
        default=False,
        action="store_true",
        dest="undeblend",
        help="Undo deblending from lightcurves",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        dest="verbose",
        help="Display clipping stats",
    )
    parser.add_argument(
        "-a",
        "--all",
        default=False,
        action="store_true",
        dest="download_all",
        help="Download all lightcurves",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        dest="overwrite",
        help="Overwrite previously downloaded files",
    )
    return parser


def tic_contamination_ratio(target_id, verbose=False):
    """Return the TIC-8.2 contamination ratio Rcont for ``target_id``.

    Rcont = (flux from contaminating neighbors) / (flux from the target),
    measured in the TIC's own fixed aperture. Returns 0.0 when the TIC has
    no contamination entry for the star. Only used as a fallback when the
    light curve itself does not carry CROWDSAP (see crowding_fraction).
    """
    # this feature probably won't be used often. don't make it a
    # dependency if not used
    from astroquery.vizier import Vizier

    v = Vizier(
        columns=["TIC", "Ncont", "Rcont", "_r"], catalog="IV/39/tic82"
    ).query_object(target_id)[0]
    ndx = np.argmin(v["_r"])
    v = v[ndx]

    if str(v["Rcont"]) == "--":
        if verbose:
            print("No contamination listed in the TIC for " + str(target_id))
        return 0.0

    if verbose:
        print(
            "TIC lists "
            + str(v["Ncont"])
            + " contaminating stars (Rcont = "
            + str(v["Rcont"])
            + ")"
        )
    return float(v["Rcont"])


def crowding_fraction(lc, contratio=0.0):
    """Return (crowdsap, source) for a light curve about to be re-blended.

    CROWDSAP is the fraction of the flux in the photometric aperture that
    belongs to the target. The SPOC/PDC pipeline used exactly this number to
    subtract the contaminating flux, so taking it from the light curve's own
    header is the exact inverse of the correction we are undoing (it is
    per-sector and measured for the aperture actually used, unlike the TIC's
    single all-sky number).

    When the header does not carry it -- older products, QLP and other
    non-SPOC pipelines, or a light curve that has already been un-corrected
    -- fall back on the TIC contamination ratio, which is the same quantity
    in a different form: Rcont = F_contam / F_target, so

        crowdsap = F_target / (F_target + F_contam) = 1 / (1 + Rcont)

    FLFRCSAP (the fraction of the target's flux that lands inside the
    aperture) is deliberately unused: the pipeline divides the flux by it,
    and a multiplicative constant cancels exactly under normalization.

    Returns (1.0, "none") when there is nothing to undo.
    """
    meta = getattr(lc, "meta", None) or {}
    raw = None
    for key in ("CROWDSAP", "crowdsap"):
        if key in meta:
            raw = meta[key]
            break

    if raw is not None:
        try:
            crowdsap = float(raw)
        except (TypeError, ValueError):
            crowdsap = np.nan
        if np.isfinite(crowdsap) and 0.0 < crowdsap <= 1.0:
            return crowdsap, "CROWDSAP"
        print(
            "WARNING: ignoring unusable CROWDSAP = "
            + str(raw)
            + " in the light curve header"
        )

    if contratio > 0.0:
        print(
            "WARNING: no usable CROWDSAP in the light curve header; falling "
            "back on the TIC contamination ratio (Rcont = "
            + str(contratio)
            + "), which was measured in the TIC aperture, not this "
            "pipeline's"
        )
        return 1.0 / (1.0 + contratio), "Rcont"

    print(
        "WARNING: no CROWDSAP in the light curve header and no TIC "
        "contamination ratio -- leaving the light curve as delivered "
        "(no deblending undone)"
    )
    return 1.0, "none"


def reblend_lightcurve(lc, crowdsap):
    """Put back the contaminating flux the pipeline subtracted.

    SPOC/PDC removes crowding *additively*, then divides by the aperture
    throughput (Kepler Data Processing Handbook, KSCI-19081):

        F_pdc = (F_sap - (1 - CROWDSAP) * median(F_sap)) / FLFRCSAP

    Taking the median of both sides gives

        median(F_sap) = FLFRCSAP * median(F_pdc) / CROWDSAP

    and substituting back inverts the correction exactly:

        F_sap = FLFRCSAP * (F_pdc + (1 - CROWDSAP)/CROWDSAP * median(F_pdc))

    The FLFRCSAP factor is a constant, so it cancels under normalize(); the
    additive term does not. In normalized units the whole operation is

        f_blended = CROWDSAP * f + (1 - CROWDSAP)

    so a transit of undiluted depth d comes back at depth CROWDSAP * d, and
    the errors scale by CROWDSAP with it (the transit S/N is unchanged).
    That dilution is the whole point of the -u flag: the fit then models it
    with its own dilution parameter.

    The old code multiplied the flux by the constant (1 + Rcont) instead,
    which normalize() divided straight back out -- a no-op that left users
    with undiluted depths while their file said otherwise.
    """
    if not np.isfinite(crowdsap) or crowdsap <= 0.0 or crowdsap > 1.0:
        raise ValueError("crowdsap must be in (0, 1]; got " + repr(crowdsap))
    if crowdsap == 1.0:
        # no contaminating flux in the aperture: nothing to put back
        return lc

    median = np.nanmedian(np.asarray(lc.flux.value, dtype=float))
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError(
            "cannot undo deblending: the median flux is "
            + repr(median)
            + ", so the contaminating flux level is undefined"
        )

    excess = (1.0 - crowdsap) / crowdsap * median
    unit = getattr(lc.flux, "unit", None)
    if unit is not None:
        excess = excess * unit
    return lc + excess


def run(args):
    """Download and write light curves for the parsed argparse namespace."""
    import lightkurve as lk

    # undo the deblending applied to TESS/Kepler lightcurves.
    # contratio is None for a lightcurve that must be left as delivered;
    # otherwise it is the TIC contamination ratio, used only as a fallback
    # when the lightcurve header has no CROWDSAP.
    if args.undeblend:
        og_contratio = tic_contamination_ratio(args.id, verbose=args.verbose)
        file_ext = ".undeblended.dat"
    else:
        file_ext = ".dat"
        contratio = None

    t0 = datetime.datetime(2000, 1, 1)
    jd0 = 2451544.5

    # The ranked set IS the searched set: asking for an author nothing ranks
    # would return products the tie-break then drops, and ranking one nothing
    # asks for is a dead row. lightkurve does not care about the order here.
    search_results = lk.search_lightcurve(args.id, author=AUTHOR_PRIORITY)

    if len(search_results) == 0:
        print(
            "No light curves found for "
            + args.id
            + ". Name must be SIMBAD-resolveable"
        )
        return

    # sometimes IDs match to multiple TIC IDs
    # warn the user to select by TIC ID
    unique_ids = list(set(search_results.target_name))

    # K2 has unique names (ktwoEPICID) for the same target
    # Kepler has unique names (kplrKICID) for the same target
    # download them
    match = []
    for id in unique_ids:
        if "ktwo" in id:
            match.append(id)
        if "kplr" in id:
            match.append("KIC" + id[4:])

    if len(match) == 1 and len(unique_ids) <= 2:
        pass
    else:
        if len(unique_ids) > 1 and "TIC" in args.id:
            match = np.where(search_results.target_name == args.id[3:])[0]
            if len(match) == 0:
                raise ValueError(
                    "No light curves match the requested TIC ID "
                    + args.id
                    + ". The search returned these target names: "
                    + ", ".join(sorted(str(u) for u in unique_ids))
                    + ". Specify the target by one of those TIC IDs."
                )
            search_results = search_results[match]
            unique_ids = list(set(search_results.target_name))

        if len(unique_ids) > 1:
            print("Multiple IDs match " + args.id)
            print(unique_ids)
            print("Specify target by TIC ID")
            return

    unique_sectors = list(set(search_results.mission))

    if args.download_all:
        # download everything
        to_download = list(range(len(search_results)))
    else:
        # only get unique lightcurves (prioritized by cadence and pipeline)
        to_download = []
        for sector in unique_sectors:
            match = np.where(search_results.mission == sector)[0]
            if len(match) == 1:
                to_download.append(match[0])
            if len(match) > 1:
                match2 = _highest_priority(
                    search_results[match].exptime.value, EXPTIME_PRIORITY
                )
                if len(match2) == 1:
                    to_download.append(match[match2[0]])
                if len(match2) > 1:
                    match3 = _highest_priority(
                        search_results[match[match2]].author, AUTHOR_PRIORITY
                    )
                    if len(match3) == 1:
                        to_download.append(match[match2[match3[0]]])
                    else:
                        # Every other skip in this loop prints; this one
                        # dropped a whole sector from the download in
                        # silence, so the light curve came back short with
                        # no indication that anything was missing.
                        print(
                            f"WARNING: {len(match3)} products remain for "
                            f"sector {sector} after author priority "
                            f"filtering; skipping this sector.  Authors: "
                            f"{list(search_results[match[match2]].author)}"
                        )
                if len(match2) == 0:
                    print(
                        f"WARNING: no product with a recognized exposure "
                        f"time for sector {sector}; skipping it. "
                        f"Exposure times offered: "
                        f"{list(search_results[match].exptime.value)}"
                    )

    for search_result in search_results[to_download]:
        author = search_result.author[0]  # SPOC, QLP, etc
        exptime = str(int(search_result.exptime[0].value)).zfill(4)
        ticid = "TIC" + search_result.target_name[0]

        if author in UNSUPPORTED_AUTHORS:
            # they don't have flux in the same place. Or errors.
            print("WARNING: CDIPS and TASOC LCs are not supported (yet?)")
            continue
        if author not in AUTHOR_MISSION:
            print(
                "WARNING: Skipping lightcurve with unrecognized author: "
                + author
            )
            continue

        mission = MISSIONS[AUTHOR_MISSION[author]]
        bjd_offset = mission.bjd_offset
        sector = mission.sector(str(search_result.mission[0]))
        filter = mission.filter
        telescope = mission.telescope
        if args.undeblend:
            if mission.undeblend:
                contratio = og_contratio
                file_ext = ".undeblended.dat"
            else:
                contratio = None
                file_ext = ".dat"
                print(
                    "WARNING: undeblending not supported for Kepler -- ignoring -u option"
                )
        file_suffix = (
            "."
            + filter
            + "."
            + telescope
            + "."
            + args.id
            + "."
            + sector
            + "."
            + exptime
            + "."
            + author
            + file_ext
        )

        # skip if I've already got it
        if not args.overwrite:
            files = glob.glob(os.path.join(args.path, "*" + file_suffix))
            if len(files) != 0:
                continue

        lc = search_result.download()
        lc = lc.remove_nans()
        if contratio is not None:
            # must happen before normalize(): the correction is additive,
            # and normalize() would divide any multiplicative one back out
            crowdsap, source = crowding_fraction(lc, contratio)
            lc = reblend_lightcurve(lc, crowdsap)
            if crowdsap < 1.0:
                print(
                    "Undoing deblending of "
                    + sector
                    + " with crowdsap = "
                    + "%.6f" % crowdsap
                    + " (from "
                    + source
                    + "); transit depths are diluted by that factor"
                )
        lc = lc.normalize()

        time = np.array(lc.time.value) + bjd_offset
        flux = np.array(lc.flux.value)
        err = np.array(lc.flux_err.value)

        # replace nans in err with median absolute deviation
        nan = np.where(np.isnan(err) | np.isinf(err))[0]
        maderr = np.median(abs(flux[1:] - flux[0:-1])) * 1.48 / np.sqrt(2.0)
        err[nan] = maderr

        # lopsided 5 sigma clipping (keeping values that are low by transit depth)
        # Should not clip transits or stellar variability
        if args.nsigma < 0:
            nbad = 0
            ngood = len(flux)
        else:
            nbad = 1

        while nbad != 0:
            rms = np.std(flux)
            median = np.median(flux)
            good = np.where(
                (flux > (median - args.depth - args.nsigma * rms))
                & (flux < (median + args.nsigma * rms))
            )[0]
            ngood = len(good)
            nbad = len(flux) - ngood

            time = time[good]
            flux = flux[good] / median
            err = err[good] / median
            if args.verbose:
                print((nbad, rms, median))

        # are they all bad?
        if ngood == 0:
            print("WARNING: no good points after sigma clipping")
            print(
                "Skipping lightcurve " + sector + " " + exptime + " " + author
            )
            continue

        datestr = (t0 + datetime.timedelta(days=time[0] - jd0)).strftime(
            "n%Y%m%d"
        )

        # create the filename in EXOFASTv2 format
        filename = os.path.join(args.path, datestr + file_suffix)

        np.savetxt(filename, np.column_stack([time, flux, err]))
        print("Downloaded " + filename)


def main(argv=None):
    """CLI entry point. Parses argv (or sys.argv) and runs the download."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
