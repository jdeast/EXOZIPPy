"""One DC2018 event, four architectures, ONE prior: configs for evidence.

    python dc18_arch_config.py --event 194 --arch 2L1S --out sweep_arch/194
    python dc18_arch_config.py --event 194 --arch all --out sweep_arch/194
    python dc18_arch_config.py --check sweep_arch/194/*.yaml

writes `<out>/DC2018_<NNN>_<arch>.yaml` and `.params.yaml` (plus the shared
`DC2018_<NNN>.sed.yaml`) for

    1L1S     point source, point lens (PSPL)
    1L1S_fs  finite source, point lens (FSPL)
    1L2S     binary source, point lens (finite source)
    2L1S     binary lens, finite source (the static sweep's model)

THE POINT IS THE EVIDENCE RATIO.  ln Z_A - ln Z_B is a statement about the
DATA only when the two models put the same prior on every parameter they
share; any difference in a shared prior's support or normalization adds an
arbitrary constant to the ratio.  So everything shared -- the stars, the
event chain, the flux and noise nuisances, the SED, the run settings -- is
built by functions that take NO architecture argument (shared_config,
shared_priors), and each architecture only ADDS its own parameters
(arch_config, arch_priors).  `check_shared` then verifies the WRITTEN files:
the shared view of every config + params file must serialize to the same
bytes (sha256 printed in each params file's header).  `--arch all` runs it
on what it wrote.

THE CANONICAL PRIOR SET (examples/DC2018/README.md, "Architecture configs"):

  shared, every architecture
    source.Source.t_0    uniform, observing span +/- T0_PAD_DAYS (90 d)
    source.Source.u_0    fitu0te everywhere (u0te uniform on [-5000, 5000]
                         d, the 1/t_E Jacobian applied); |u_0| <= 3 soft
                         barrier on the derived u_0
    log_f_total          log10(median data flux) +/- 2 dex, per band
    err_scale            uniform [0.5, 2]
    likelihood           Gaussian (no hogg mixture: the challenge's noise is
                         simulated Gaussian); --hogg turns the mixture back
                         on, with out_scale capped at 10x the median error
    neighbor_flux        neighbor_fraction uniform [0, 1], neighbor_flux =
                         neighbor_fraction * f_blend -- the physical
                         support, set by the component itself (no flux
                         scale, so nothing a seed or topology can move)
    zeropoint            AB (magsys: AB), mu = the layout's zeropoint,
                         sigma hypot(0.02, grey) on W149, Z087 linked to it
                         with 0.02*sqrt(2) on the difference
    stars / SED          dc18_sweep_config.py's recipe (av from the clump
                         colour, feh N(0, 0.5) on the lens, torres on the
                         source, mann + mamajek on the lens, galactic model
                         anchored on the source)
  1L1S, 1L1S_fs          nothing more: rho keeps defaults.yaml's soft upper
                         barrier at 10 on 1L1S_fs (real FFPs can exceed 1),
                         and is report-only on 1L1S
  2L1S                   log_q uniform [-6, 0] (stellar binaries are in the
                         sample), log_s uniform [-1, 1], alpha uniform;
                         rho <= 1 (soft barrier, bound_scale 0.01: ~440 nats
                         at rho = 1.01, i.e. hard in practice)
  1L2S                   a second source star Source2, paired with Source
                         by a `type: static` orbit (one bound pair: the
                         orbit group ties distance, proper motion, rv and
                         av; feh is hard-linked by hand), its own t_0 / u_0
                         with Source's bounds; log_q_flux uniform [-3, 3]
                         per band (fitlogqflux); rho <= 1 on both sources,
                         as 2L1S

WHAT CONFIG CANNOT MAKE IDENTICAL (and the README says so):
  * 1L2S has THREE star bodies, the others two.  The galactic model's
    per-star density prior is unnormalized (galacticmodel.py).  With
    Source2's distance, proper motion and rv all tied to Source's by the
    static orbit, Source2 joins Source's linked group and adds no kinematic
    term (star.md, "Linked stars"); the extra star's SED/torres terms remain.
  * rho's prior differs between 1L1S_fs (soft 10) and 2L1S/1L2S (soft 1) by
    the instruction that set them; the barrier removes prior mass above the
    cap without renormalizing.
  * A 2L1S companion is a `planet` body: it carries mass (through log_q)
    but no light, so a stellar-binary lens's companion flux is absorbed by
    neighbor_flux rather than predicted.

RUN SETTINGS (per mode, seeded from ONE grid mode): ptde, seed_polish false,
overdisperse true, n_temps / tune / draws from the command line (4 / 300 /
8000 by default; being tuned on event 194; --n-temps auto lets the sampler
size the ladder), and sampler.maxtime ALWAYS
written, with headroom below the job's wall time.

SEEDS ARE STARTS, NEVER PRIORS: only the `initval` field of a --seed file is
read, and only for parameters the architecture has (a 2L1S grid seed's
log_s / log_q / alpha are dropped from 1L1S, for example).
"""

import argparse
import copy
import hashlib
import io
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402
from dc18_sweep_config import (  # noqa: E402
    AV_COLS,
    AZ_COLS,
    BANDS,
    av_from_clump_colour,
    t0_bounds,
)

ARCHES = ("1L1S", "1L1S_fs", "1L2S", "2L1S")

# Architecture-specific prior constants (see the module docstring).
LOG_Q_BOUNDS = (-6.0, 0.0)
LOG_S_BOUNDS = (-1.0, 1.0)
LOG_Q_FLUX_BOUNDS = (-3.0, 3.0)
RHO_CAP = 1.0
RHO_CAP_BOUND_SCALE = 0.01
U0_BOUND = 3.0
ERR_SCALE_BOUNDS = (0.5, 2.0)
LOG_F_TOTAL_DEX = 2.0

# The parameters a SHARED view leaves out: they exist only in some
# architectures, or (rho) differ by design.  check_shared drops these and
# compares everything else byte for byte.
ARCH_PARAM_PREFIXES = (
    "planet.Companion.",
    "lens.Companion.",
    "star.Source2.",
    "source.Source2.",
)
ARCH_PARAM_SUFFIXES = (".q_flux", ".log_q_flux")
ARCH_PARAM_KEYS = ("source.Source.rho",)


def event_info_row(data_dir, event):
    rows = np.genfromtxt(
        os.path.join(data_dir, "event_info.txt"), dtype=None, encoding="utf-8"
    )
    for r in np.atleast_1d(rows):
        if int(r[1]) == int(event):
            return r
    raise SystemExit("event %s not in event_info.txt" % event)


class Event:
    """Everything measured from the data that the priors read, once."""

    def __init__(self, data_dir, event):
        self.data_dir = str(C.data_dir_or_raise(data_dir))
        self.event = int(event)
        self.ev3 = "%03d" % self.event
        self.name = "DC2018_%s" % self.ev3
        self.layout = C.data_layout(self.data_dir)
        self.data_format, self.zp_ab = C.LAYOUT_FORMAT[self.layout]
        row = event_info_row(self.data_dir, self.event)
        self.ra, self.dec = float(row[2]), float(row[3])
        self.av_mu, self.av_sd, self.av_grey = av_from_clump_colour(
            float(row[AV_COLS[0]]),
            float(row[AV_COLS[1]]),
            float(row[AZ_COLS[0]]),
            float(row[AZ_COLS[1]]),
        )
        self.files = C.light_curve_files(
            self.data_dir, self.event, bands=[b for b, _ in BANDS]
        )
        self.curves = {
            b: C.read_light_curve(p, self.data_format)
            for b, p in self.files.items()
        }
        t_all = np.concatenate([c[0] for c in self.curves.values()])
        self.t_lo, self.t_hi = float(t_all.min()), float(t_all.max())

    def median_flux(self, band):
        f = self.curves[band][1]
        return float(np.median(f[f > 0]))

    def median_flux_err(self, band):
        return float(np.median(self.curves[band][2]))

    def brightest_epoch(self, band="W149", nbin=1.0):
        """The epoch of the brightest 1-day bin: a START for a second
        source's t_0 when no seed names one (never a prior)."""
        t, f, _ = self.curves[band]
        edges = np.arange(t.min(), t.max() + nbin, nbin)
        idx = np.digitize(t, edges)
        best, t_best = -np.inf, float(np.median(t))
        for k in np.unique(idx):
            sel = idx == k
            if sel.sum() >= 3:
                m = float(np.median(f[sel]))
                if m > best:
                    best, t_best = m, float(np.median(t[sel]))
        return t_best


# ---------------------------------------------------------------------------
# Shared: NO architecture argument, so identical by construction
# ---------------------------------------------------------------------------


def shared_config(ev, run):
    """Config blocks every architecture has, exactly as written."""
    inst = []
    for b, _ in BANDS:
        entry = {
            "name": "Roman_%s" % b,
            "file": ev.files[b],
            "data_format": ev.data_format,
            "observer_location": "roman_simulated_2018dc",
            # The blend must hold at least the lens's light: the SED's lens
            # flux plus a non-negative neighbor_flux is tied to f_blend
            # (dc18_sweep_config.py, JDE 2026-09-26).
            "sed_constrains_blend": True,
            "band": b,
            # The challenge's photometry is AB (dc18_photometric_leg.py);
            # Roman WFI records no native system, so it must be stated.
            "magsys": "AB",
        }
        if run["hogg"]:
            entry["likelihood"] = "hogg"
        inst.append(entry)
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [
            {"fitmurel": True, "fitpirel": True, "fitthetae": True}
        ],
        "source": [
            {
                "body": "star.Source",
                "star_constrains_rho": True,
                # ONE u_0 coordinate in every architecture: the u0te span's
                # normalization and its 1/t_E Jacobian differ from sampling
                # u_0 directly by ~5.6 nats (source.py, build_likelihood).
                "fitu0te": True,
            }
        ],
        "galacticmodel": [{"name": ev.name, "anchor_idx": 1}],
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
        "mulensinstrument": inst,
        "torres": [{"star": "Source", "constrain": ["mass", "radius"]}],
        "mann": [
            {
                "star": "Lens",
                "ks": "synthetic",
                "constrain": ["mass", "radius"],
            }
        ],
        "mamajek": [{"star": "Lens", "constrain": ["teff"]}],
        "sampler": {
            "method": "ptde",
            "cores": run["cores"],
            "n_temps": run["n_temps"],
            "T_max": run["t_max"],
            "tune": run["tune"],
            "draws": run["draws"],
            "nthin": 1,
            # Always written: a fit that hits the wall without it loses
            # everything (pilot 194 lost 48 h).
            "maxtime": run["maxtime"],
            "recompute_trace": True,
            "eval_timeout": 10,
            "store_hot_chains": True,
            # Seeded from ONE grid mode: no polish.
            "seed_polish": False,
        },
        "modes": {"weights": "evidence"},
    }


def shared_priors(ev, run):
    """Params-file PRIORS every architecture has (no starts)."""
    p = {
        "star.Lens.ra": {"initval": ev.ra, "sigma": 0},
        "star.Lens.dec": {"initval": ev.dec, "sigma": 0},
        "star.Source.ra": {"initval": ev.ra, "sigma": 0},
        "star.Source.dec": {"initval": ev.dec, "sigma": 0},
        "star.Lens.feh": {"mu": 0.0, "sigma": 0.5},
        "star.Source.av": {"mu": ev.av_mu, "sigma": ev.av_sd},
        "band.W149.u1": {"initval": 0.0, "sigma": 0},
        "band.Z087.u1": {"initval": 0.0, "sigma": 0},
        "sed.errscale": {"initval": 1.0, "sigma": 0},
        "source.Source.t_0": t0_bounds(ev.t_lo, ev.t_hi),
        "source.Source.u_0": {"lower": -U0_BOUND, "upper": U0_BOUND},
        # K basins -> K modes; a single grid mode still wants scattering.
        "overdisperse": True,
    }
    first = BANDS[0][0]
    for b, _ in BANDS:
        inst = "mulensinstrument.Roman_%s" % b
        if b == first:
            p["%s.zeropoint" % inst] = {
                "mu": float(ev.zp_ab),
                "sigma": round(float(np.hypot(0.02, abs(ev.av_grey))), 4),
            }
        else:
            # Both bands AB on the same zeropoint: the difference is 0.
            p["%s.zeropoint" % inst] = {
                "mu": "mulensinstrument.Roman_%s.zeropoint + 0.0" % first,
                "sigma": round(float(0.02 * np.sqrt(2.0)), 4),
            }
        lf = float(np.log10(ev.median_flux(b)))
        p["%s.log_f_total" % inst] = {
            "lower": lf - LOG_F_TOTAL_DEX,
            "upper": lf + LOG_F_TOTAL_DEX,
        }
        p["%s.err_scale" % inst] = {
            "lower": ERR_SCALE_BOUNDS[0],
            "upper": ERR_SCALE_BOUNDS[1],
        }
        if run["hogg"]:
            med = ev.median_flux_err(b)
            p["%s.out_scale" % inst] = {"upper": 10.0 * med, "initval": med}
    return p


# ---------------------------------------------------------------------------
# Per architecture: only ADDS
# ---------------------------------------------------------------------------


def arch_config(arch, cfg):
    cfg = copy.deepcopy(cfg)
    finite = arch != "1L1S"
    cfg["mulensevent"][0]["finite_source"] = finite
    if arch == "2L1S":
        cfg["planet"] = [{"name": "Companion"}]
        cfg["lens"] = [{"body": "star.Lens"}, {"body": "planet.Companion"}]
    else:
        cfg["lens"] = [{"body": "star.Lens"}]
    if arch == "1L2S":
        cfg["star"].append({"name": "Source2"})
        cfg["source"].append(
            {
                "body": "star.Source2",
                "star_constrains_rho": True,
                "fitu0te": True,
            }
        )
        cfg["torres"].append(
            {"star": "Source2", "constrain": ["mass", "radius"]}
        )
        for entry in cfg["mulensinstrument"]:
            entry["fitlogqflux"] = True
        # One bound pair that does not move: the static orbit ties Source2's
        # distance, proper motion, rv and av to Source's
        # (components/orbit/orbit.md, "type: static").
        cfg["orbit"] = [
            {
                "name": "Sources",
                "type": "static",
                "primary": ["Source"],
                "companion": ["Source2"],
            }
        ]
    return cfg


def arch_priors(arch, ev, shared):
    p = {}
    rho_cap = {"upper": RHO_CAP, "bound_scale": RHO_CAP_BOUND_SCALE}
    if arch == "2L1S":
        p["planet.Companion.radius"] = {"sigma": 0}
        p["planet.Companion.log_q"] = {
            "lower": LOG_Q_BOUNDS[0],
            "upper": LOG_Q_BOUNDS[1],
        }
        p["lens.Companion.log_s"] = {
            "lower": LOG_S_BOUNDS[0],
            "upper": LOG_S_BOUNDS[1],
        }
        p["source.Source.rho"] = dict(rho_cap)
    elif arch == "1L2S":
        p["star.Source2.ra"] = {"initval": ev.ra, "sigma": 0}
        p["star.Source2.dec"] = {"initval": ev.dec, "sigma": 0}
        # Distance, space motion and av come from the static orbit in
        # arch_config; the orbit group does not tie the present-day [Fe/H],
        # so the pair's composition is linked here.
        p["star.Source2.feh"] = {"initval": "star.Source.feh", "sigma": 0}
        p["source.Source2.t_0"] = dict(shared["source.Source.t_0"])
        p["source.Source2.u_0"] = dict(shared["source.Source.u_0"])
        p["source.Source.rho"] = dict(rho_cap)
        p["source.Source2.rho"] = dict(rho_cap)
        for b, _ in BANDS:
            p["mulensinstrument.Roman_%s.log_q_flux" % b] = {
                "lower": LOG_Q_FLUX_BOUNDS[0],
                "upper": LOG_Q_FLUX_BOUNDS[1],
            }
    return p


def is_arch_key(key):
    return (
        key.startswith(ARCH_PARAM_PREFIXES)
        or key.endswith(ARCH_PARAM_SUFFIXES)
        or key in ARCH_PARAM_KEYS
    )


def arch_has_key(arch, key):
    """Whether a params path names a parameter this architecture has."""
    if key.startswith(("planet.Companion.", "lens.Companion.")):
        return arch == "2L1S"
    if key.startswith(("star.Source2.", "source.Source2.")):
        return arch == "1L2S"
    if key.endswith(ARCH_PARAM_SUFFIXES):
        return arch == "1L2S"
    return True


def seed_starts(arch, seed, ev):
    """Per-seed `initval`s from a --seed file, filtered to the architecture.

    Only initval is read: a seed is a start, and a bound or prior in it
    would make the prior depend on the seed.  `lens.Companion.q` is dropped
    (derived from log_q and the masses; dc18 grid seeds carry log_q).
    """
    starts = {}
    for key, entry in (seed or {}).items():
        if not isinstance(entry, dict) or "initval" not in entry:
            continue
        if key == "lens.Companion.q" or not arch_has_key(arch, key):
            continue
        starts[key] = {"initval": entry["initval"]}
    lens = {
        len(v["initval"])
        for v in starts.values()
        if isinstance(v["initval"], list)
    }
    if len(lens) > 1:
        raise SystemExit(
            "seed file mixes per-seed lists of lengths %s; every list must "
            "have the same length K" % sorted(lens)
        )
    if arch == "2L1S" and "planet.Companion.log_q" not in starts:
        # Unseeded, the engine starts log_q against its lower bound (the
        # defaults' planet mass against the lens mass); a mid-range start.
        starts["planet.Companion.log_q"] = {"initval": -3.0}
    if arch == "1L2S" and "source.Source2.t_0" not in starts:
        # The peak finder seeds source 0 only, and a t_0 with no start
        # would sit at defaults.yaml's backstop.  The brightest epoch is a
        # data-derived start; it is not a prior.
        starts["source.Source2.t_0"] = {"initval": ev.brightest_epoch()}
    if arch == "1L2S" and "source.Source2.u_0" not in starts:
        # An informed u_0 too, so the flux bootstrap decomposes two
        # sources rather than folding the second into the first.
        starts["source.Source2.u_0"] = {"initval": 1.0}
    return starts


def merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def shared_view(cfg, params, sed_text):
    """The part of one architecture's (config, params) every other must
    share, in a canonical form.  Arch-only parameters and blocks are dropped;
    starts (initval on anything not pinned) are dropped; paths that differ
    only by output directory (prefix, parameter_file, sed.file) are replaced
    by what they hold."""
    c = copy.deepcopy(cfg)
    for k in ("run", "prefix", "parameter_file", "lens", "planet", "orbit"):
        c.pop(k, None)
    c["sed"] = sed_text
    c["star"] = [s for s in c["star"] if s.get("name") != "Source2"]
    c["source"] = [s for s in c["source"] if s.get("body") != "star.Source2"]
    c["torres"] = [t for t in c["torres"] if t.get("star") != "Source2"]
    for m in c["mulensevent"]:
        m.pop("finite_source", None)
    for inst in c["mulensinstrument"]:
        inst.pop("fitlogqflux", None)
    p = {}
    for key, entry in params.items():
        if not isinstance(entry, dict):
            p[key] = entry
            continue
        if is_arch_key(key):
            continue
        e = dict(entry)
        if e.get("sigma") != 0:
            e.pop("initval", None)
        if e:
            p[key] = e
    return yaml.safe_dump(
        {"config": c, "params": p}, sort_keys=True, default_flow_style=False
    ).encode("utf-8")


def shared_digest(cfg, params, sed_text):
    return hashlib.sha256(shared_view(cfg, params, sed_text)).hexdigest()


def _load(cfg_path):
    with io.open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with io.open(cfg["parameter_file"], encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    with io.open(cfg["sed"]["file"], encoding="utf-8") as fh:
        sed_text = fh.read()
    return cfg, params, sed_text


def check_shared(cfg_paths):
    """Verify the WRITTEN files share one prior.  Returns the digest.

    Raises SystemExit naming the first differing line otherwise.  Also
    checks the architecture-specific entries that must agree among the
    architectures that have them (the rho cap; Source2's t_0/u_0 bounds
    equal to Source's).
    """
    views = {}
    loaded = {}
    for path in cfg_paths:
        cfg, params, sed_text = _load(path)
        loaded[path] = (cfg, params)
        views[path] = shared_view(cfg, params, sed_text)
    ref_path = cfg_paths[0]
    for path, view in views.items():
        if view != views[ref_path]:
            a = views[ref_path].decode().splitlines()
            b = view.decode().splitlines()
            diff = next(
                ((i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y),
                (min(len(a), len(b)), "<end>", "<end>"),
            )
            raise SystemExit(
                "shared priors differ between %s and %s at line %d:\n  %s\n"
                "  %s" % (ref_path, path, diff[0], diff[1], diff[2])
            )
    rhos = {
        tuple(sorted(p["source.Source.rho"].items()))
        for _, p in loaded.values()
        if "source.Source.rho" in p
    }
    if len(rhos) > 1:
        raise SystemExit(
            "the rho cap differs between architectures: %s" % rhos
        )
    for path, (_, p) in loaded.items():
        for par in ("t_0", "u_0"):
            k2 = "source.Source2.%s" % par
            if k2 in p:
                b1 = {
                    f: p["source.Source.%s" % par].get(f)
                    for f in ("lower", "upper")
                }
                b2 = {f: p[k2].get(f) for f in ("lower", "upper")}
                if b1 != b2:
                    raise SystemExit(
                        "%s: %s bounds %s differ from Source's %s"
                        % (path, k2, b2, b1)
                    )
    return hashlib.sha256(views[ref_path]).hexdigest()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def build(ev, arch, outdir, run, seed=None):
    """Write one architecture's config + params; return the config path."""
    if arch not in ARCHES:
        raise SystemExit("unknown --arch %r; one of %s" % (arch, ARCHES))
    base = os.path.abspath(outdir)
    os.makedirs(base, exist_ok=True)
    tag = "%s_%s" % (ev.name, arch)

    sed_path = os.path.join(base, "%s.sed.yaml" % ev.name)
    with io.open(sed_path, "w", encoding="utf-8") as fh:
        fh.write("model: NextGen\nfilters: []\n")

    shared_p = shared_priors(ev, run)
    cfg = arch_config(arch, shared_config(ev, run))
    cfg = merge(
        {"run": {"name": tag}},
        cfg,
        {
            "prefix": os.path.join(base, tag),
            "sed": {"file": sed_path},
            "parameter_file": os.path.join(base, "%s.params.yaml" % tag),
        },
    )
    params = merge(
        shared_p, arch_priors(arch, ev, shared_p), seed_starts(arch, seed, ev)
    )

    with io.open(sed_path, encoding="utf-8") as fh:
        digest = shared_digest(cfg, params, fh.read())
    header = (
        "# DC2018 event %s, architecture %s, written by dc18_arch_config.py.\n"
        "# Shared priors sha256 %s -- identical across 1L1S/1L1S_fs/1L2S/"
        "2L1S\n# of this event (dc18_arch_config.py --check).\n"
        % (ev.ev3, arch, digest)
    )
    with io.open(cfg["parameter_file"], "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(params, fh, sort_keys=True, default_flow_style=False)
    cfg_path = os.path.join(base, "%s.yaml" % tag)
    with io.open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    print(
        "%s  %s  shared %s\n    -> %s" % (ev.name, arch, digest[:12], cfg_path)
    )
    return cfg_path


def n_temps_arg(text):
    """`--n-temps`: a positive integer, or the sampler's own "auto".

    "auto" is passed through as the string, so the written config says
    `n_temps: auto` and the ladder is sized at run time from the model's
    dimension (samplers/ladder.py, resolve_n_temps) -- the generator does
    not know that dimension and must not guess it.
    """
    if str(text).strip().lower() == "auto":
        return "auto"
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--n-temps must be a positive integer or 'auto', got %r" % text
        )
    if n < 1:
        raise argparse.ArgumentTypeError("--n-temps must be >= 1, got %d" % n)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--event", type=int)
    ap.add_argument("--arch", choices=ARCHES + ("all",), default="all")
    ap.add_argument("--seed", default=None, help="params YAML of starts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--n-temps",
        type=n_temps_arg,
        default=4,
        help="an integer rung count, or 'auto' (samplers/ladder.py "
        "resolve_n_temps sizes the ladder from the model's dimension)",
    )
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--draws", type=int, default=8000)
    ap.add_argument(
        "--maxtime",
        type=int,
        default=72000,
        help="sampler.maxtime [s]; keep headroom below the job wall time",
    )
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--t-max", type=float, default=200.0)
    ap.add_argument(
        "--hogg",
        action="store_true",
        help="turn the hogg outlier mixture back on (off by default: the "
        "challenge's noise is simulated Gaussian)",
    )
    ap.add_argument(
        "--allow-cv",
        action="store_true",
        help="write configs for a CV-class event (skipped by default)",
    )
    ap.add_argument(
        "--check",
        nargs="+",
        metavar="CONFIG",
        help="verify these written configs share their priors, and exit",
    )
    a = ap.parse_args(argv)

    if a.check:
        digest = check_shared(a.check)
        print(
            "shared priors identical across %d configs: sha256 %s"
            % (len(a.check), digest)
        )
        return 0
    if a.event is None or a.out is None:
        ap.error("--event and --out are required (or --check)")

    ev = Event(a.data_dir or C.DEFAULT_DATA_DIR, a.event)
    master = os.path.join(ev.data_dir, "Answers", "master_file.txt")
    if os.path.exists(master) and not a.allow_cv:
        cls = C.event_class(ev.data_dir, ev.event)
        if cls == C.CV_CLASS:
            raise SystemExit(
                "event %s is a cataclysmic variable (class %s): skipped; "
                "--allow-cv to write it anyway" % (ev.ev3, cls)
            )
    seed = None
    if a.seed:
        with io.open(a.seed, encoding="utf-8") as fh:
            seed = yaml.safe_load(fh) or {}
    run = {
        "n_temps": a.n_temps,
        "tune": a.tune,
        "draws": a.draws,
        "maxtime": a.maxtime,
        "cores": a.cores,
        "t_max": a.t_max,
        "hogg": a.hogg,
    }
    arches = ARCHES if a.arch == "all" else (a.arch,)
    paths = [build(ev, arch, a.out, run, seed) for arch in arches]
    if len(paths) > 1:
        digest = check_shared(paths)
        print("check: shared priors identical, sha256 %s" % digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
