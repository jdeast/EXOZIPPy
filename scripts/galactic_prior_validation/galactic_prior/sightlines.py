"""Sight lines and genulens's line-of-sight discretization (numpy only)."""

import numpy as np

# Galactic (l, b) in degrees.  ``ogle`` names the published
# efficiency-corrected t_E sample for the field the line sits in
# (Mroz et al. 2017 high-cadence fields, binned; Mroz et al. 2019
# low-cadence fields, per event); None for lines without one.
#
# * baade: Baade's window; inside OGLE-IV field BLG513.
# * blg500 ... blg611: the centers of Mroz et al. 2017's nine high-cadence
#   fields (Extended Data Table 4; centers from Mroz et al. 2019 Table 7).
# * blg521: a low-cadence field at b = -5.1 (69 events, Mroz 2019).
# * ob140939: OGLE-2014-BLG-0939, in field BLG630 (Mroz 2019, Table 3
#   lists the event as BLG630.01).
# * nsd_pt, gbtds_w, nsd_lowb: three more lines inside genulens's
#   nuclear-stellar-disk box (|l| < 5, |b| < 2), in the Roman Galactic
#   Bulge Time Domain Survey footprint; nsd_lowb is at |b| < 1.
SIGHTLINES = {
    "baade": {"l": 1.0, "b": -3.9, "ogle": ("mroz2019", "BLG513")},
    "blg500": {"l": 0.9999, "b": -1.0293, "ogle": ("mroz2017", "BLG500")},
    "blg501": {"l": -0.0608, "b": -1.6400, "ogle": ("mroz2017", "BLG501")},
    "blg504": {"l": 2.1491, "b": -1.7747, "ogle": ("mroz2017", "BLG504")},
    "blg505": {"l": 1.0870, "b": -2.3890, "ogle": ("mroz2017", "BLG505")},
    "blg506": {"l": 0.0103, "b": -2.9974, "ogle": ("mroz2017", "BLG506")},
    "blg511": {"l": 3.2835, "b": -2.5219, "ogle": ("mroz2017", "BLG511")},
    "blg512": {"l": 2.2154, "b": -3.1355, "ogle": ("mroz2017", "BLG512")},
    "blg534": {"l": -1.1356, "b": -2.2547, "ogle": ("mroz2017", "BLG534")},
    "blg611": {"l": 0.3282, "b": 2.8242, "ogle": ("mroz2017", "BLG611")},
    "blg521": {"l": 1.1905, "b": -5.1086, "ogle": ("mroz2019", "BLG521")},
    "ob140939": {"l": 6.6480, "b": 3.6113, "ogle": ("mroz2019", "BLG630")},
    "nsd_pt": {"l": 0.3, "b": -1.6, "ogle": None},
    "gbtds_w": {"l": -0.4, "b": -1.2, "ogle": None},
    "nsd_lowb": {"l": 0.0, "b": -0.6, "ogle": None},
}

# Lines evaluated with unresolved binary lenses on both sides
# (galacticmodel ``binaries: true`` / genulens ``BINARY 1``).
BINARY_LINES = ("baade", "nsd_pt")


def run_name(line, binaries=False):
    return f"{line}_bin" if binaries else line


def parse_run(name):
    """'baade_bin' -> ('baade', True)."""
    if name.endswith("_bin"):
        return name[:-4], True
    return name, False


def genulens_nsd_active(l_deg, b_deg):
    """genulens switches its nuclear stellar disk on for |l| < 5, |b| < 2.

    sampler.cpp; the same rule as ``koshimoto21.nsd_active`` (a test holds
    the two together).
    """
    l_signed = (l_deg + 180.0) % 360.0 - 180.0
    return abs(l_signed) < 5.0 and abs(b_deg) < 2.0


def genulens_bin_width(l_deg, b_deg, dmax_pc=16000.0):
    """genulens's line-of-sight grid step dD in pc.

    los_density_grid.cpp: nbin = 0.20/0.10/0.04 Dmax (NSD on and within
    0.05 / 0.10 deg of the center / NSD on) or 0.01 Dmax, rounded, and
    dD = Dmax / nbin -- 100 pc off the NSD box and 25 pc inside it, for
    both Dmax = 16 and 20 kpc.
    """
    nsd = genulens_nsd_active(l_deg, b_deg)
    if nsd and abs(l_deg) < 0.05 and abs(b_deg) < 0.05:
        frac = 0.20
    elif nsd and abs(l_deg) < 0.10 and abs(b_deg) < 0.10:
        frac = 0.10
    elif nsd:
        frac = 0.04
    else:
        frac = 0.01
    nbin = int(frac * dmax_pc + 0.5)
    return dmax_pc / nbin


def genulens_support_mask(d_l, d_s, dd_pc, dmax_pc=None):
    """Draws inside the support genulens's discretization gives an event.

    (e) genulens draws the lens only from the line-of-sight bins wholly in
    front of the source's bin (get_lens_importance_bins: bins
    0 .. int(D_S / dD) - 1), so a lens needs D_L < int(D_S / dD) dD.
    (f) with ``dmax_pc`` set, genulens's grid ends at Dmax, so the source
    (and with it the lens) needs D_S < Dmax.

    Applied to the PORT's weighted draws, this reproduces genulens's support
    at analysis time, without touching genulens.
    """
    d_l = np.asarray(d_l, float)
    d_s = np.asarray(d_s, float)
    edge = np.floor(d_s / dd_pc) * dd_pc
    mask = d_l < edge
    if dmax_pc is not None:
        mask &= d_s < dmax_pc
    return mask
