"""
Doppler Tomography (DT) -- the time-resolved 2D sibling of the RM anomaly.

During transit the planet blocks a strip of the rotating stellar surface; the
missing light appears as a moving bump (the "Doppler shadow") in the line-
profile residuals.  A DT dataset is a 2D image ``ccf2d[ntime, nvel]`` of those
residuals; the shadow's trajectory across it directly constrains the projected
obliquity lambda, vsini and the transit geometry.

Model (port of EXOFASTv2 ``dopptom_chi2.pro``, Beatty 2015 / Eastman 2017):
for each exposure the shadow is the rotation half-ellipse of the occulted
strip, centred on the subplanet velocity ``vsini * up`` with half-width
``vsini * p``, convolved with a Gaussian of width sqrt(vline^2 + (c/R)^2)
(intrinsic line + instrument), scaled by the blocked flux ``beta`` from the
quadratic-limb-darkened light curve.  Here the ellipse (x) Gaussian convolution
is evaluated by Chebyshev-Gauss (2nd kind) quadrature, whose weights carry the
sqrt(1-t^2) ellipse factor exactly -- no discrete convolution, exact unit
normalization, and differentiable end to end (NUTS / numpyro ready).

Likelihood: Normal residuals with sigma = rms * dtscale, with the WHOLE
log-likelihood divided by ``IndepVels = (c/R / 2.355) / dv`` -- the
supersampling factor of the velocity grid relative to the true spectral
resolution -- exactly as EXOFASTv2 tempers its DT chi^2 for correlated pixels.

Data format (EXOFASTv2 DTPATH convention): a FITS file whose primary HDU is
``ccf2d[ntime, nvel]``, HDU 1 the BJD_TDB of each exposure, HDU 2 the velocity
grid in km/s; the instrument resolving power is parsed from the filename
(``nYYYYMMDD.<planet>.<instrument>.<R>.fits``) or given as ``resolution:``.

DT velocity grids are typically delivered heavily supersampled relative to
the spectral resolution (the IndepVels tempering exists precisely because
adjacent pixels are not independent).  ``velbin:`` block-averages the grid at
load time -- ``auto`` (default) bins down to ~3 pixels per instrumental sigma
(IndepVels ~ 3), which loses no information (no structure narrower than the
resolution element exists in the data) while cutting the likelihood and
gradient cost by the same factor.  The tempering is recomputed on the binned
grid, so the statistical weight of the dataset is unchanged.

Geometry (subplanet coordinate, blocked flux) reuses ``components/rm.py``; the
spin-orbit parameterization is the shared sqrt(vsini)cos/sin(lambda) pair on
``orbit`` and the Gaussian line width ``vline`` lives on ``star``.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.limbdark import quad_limb_darkened_flux
from exozippy.outputs.texutils import latex_escape

C_KMS = 299792.458
FWHM2SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def cheb_gauss2(n):
    """Chebyshev-Gauss (2nd kind) nodes t_k and ellipse-normalized weights:
    sum_k wtil_k f(t_k) ~ (2/pi) int_-1^1 f(t) sqrt(1-t^2) dt, with
    sum wtil = 1 exactly -- the unit-area rotation half-ellipse."""
    kk = np.arange(1, n + 1)
    t_k = np.cos(kk * np.pi / (n + 1.0))
    w_k = (np.pi / (n + 1.0)) * np.sin(kk * np.pi / (n + 1.0)) ** 2
    return t_k, (2.0 / np.pi) * w_k


def choose_quadrature_order(width_ratio, floor=16, cap=192):
    """Chebyshev-Gauss order resolving a Gaussian of sigma against an
    ellipse of half-width ``width_ratio`` sigmas.

    The rule's node spacing at the strip centre is ~pi/(n+1) in t, i.e.
    ~pi * halfwidth / (n+1) in velocity; requiring >= ~4 nodes per
    Gaussian sigma across the ellipse (n ~ 4 * pi/4 * ratio, plus a
    floor for the wide-line regime) keeps the quadrature error well
    under 1% of the profile peak at every ratio the cap admits -- a
    fixed 16 was accurate only for ratio <~ 3 and off by ~half the
    peak at ratio ~ 20 (PR #323 review).  The cap bounds the graph
    cost for pathological inputs; 192 nodes covers ratio ~ 60, beyond
    any plausible spectrograph/rotator combination.
    """
    n = int(np.ceil(3.2 * max(float(width_ratio), 1.0))) + 8
    return int(min(cap, max(floor, n)))


def dt_shadow(v_kms, subx, vsini_kms, p, sigma_g_kms, n_gl=16):
    """Normalized Doppler-shadow profiles, (ntime, nvel).

    The occulted strip's rotation half-ellipse (centre ``vsini*subx``,
    half-width ``vsini*p``) convolved with a Gaussian of dispersion
    ``sigma_g_kms``, integrated by Chebyshev-Gauss quadrature so the ellipse
    weight is exact and each profile integrates to 1 analytically.  All
    symbolic inputs may be tensors; ``v_kms`` is the (fixed) velocity grid.
    """
    t_k, wtil = cheb_gauss2(n_gl)
    centers = (
        vsini_kms * subx[:, None]
        + (vsini_kms * p) * pt.as_tensor_variable(t_k)[None, :]
    )  # (ntime, n_gl)
    dvc = (
        pt.as_tensor_variable(np.asarray(v_kms, dtype=float))[None, :, None]
        - centers[:, None, :]
    )  # (ntime, nvel, n_gl)
    gauss = pt.exp(-0.5 * pt.sqr(dvc / sigma_g_kms)) / (
        pt.sqrt(2.0 * np.pi) * sigma_g_kms
    )
    return pt.dot(gauss, pt.as_tensor_variable(wtil))  # (ntime, nvel)


def dt_orbits_in_system(system):
    """Orbit names referenced by any dopptom `orbit:` key (topology helper,
    mirrors rm.rm_orbits_in_system)."""
    names = set()
    cfg = getattr(system, "config", None) or {}
    for entry in cfg.get("dopptom", []) or []:
        o = entry.get("orbit")
        if o:
            names.add(o)
    return names


def dt_enabled(system):
    return len(dt_orbits_in_system(system)) > 0


def dt_primary_star_indices(system):
    """Star indices that are the primary of a DT-targeted orbit.

    Each DT dataset reads ONE star's local line width (its orbit's
    primary), so `vline` is declared for exactly this set -- a
    system-wide declaration would hand every other star a
    likelihood-free sampled dimension.  Topology helper: reads the
    config and the orbit component's construction-time body maps only,
    so it is safe from stage 2 on.
    """
    orbit_comp = getattr(system, "orbit", None)
    if orbit_comp is None:
        return set()
    targets = dt_orbits_in_system(system)
    idx = set()
    for oidx, name in enumerate(orbit_comp.names):
        if name not in targets:
            continue
        star_idx = next(
            (
                i
                for (ctype, i) in orbit_comp.primary_bodies[oidx]
                if ctype == "star"
            ),
            None,
        )
        if star_idx is None:
            # Fail HERE (stage 3, with a dopptom label), not later at
            # build time through rm_primary_star_index's "[rm] ..."
            # message in a fit that has no rm: key (deep review finding).
            raise ValueError(
                f"[dopptom] orbit '{name}' has no star in its primary "
                f"body group, so there is no transited star whose line "
                f"profile the Doppler-tomography model can use. Put the "
                f"transited star in the orbit's `primary:` group."
            )
        idx.add(int(star_idx))
    return idx


class Dopptom(Component):
    # Per-build node caches System.build_model clears before stage 5
    # (see Component.per_build_caches): compiled plot functions and the
    # symbolic nodes they were compiled against belong to ONE model; a
    # rebuilt live System (the GUI Solve path) must never call them with
    # the new model's plot-param values.
    per_build_caches = (
        "_model_nodes",
        "_plot_fns",
        "_subvel_nodes",
    )

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Doppler Tomography"
        self.files = [c.get("file") for c in self.config]
        self.orbit_names = [c.get("orbit") for c in self.config]
        self.band_names = [c.get("band") for c in self.config]
        self.resolutions = [c.get("resolution") for c in self.config]
        self.velbins = [c.get("velbin", "auto") for c in self.config]
        # Light-travel-time on the occultation seam, per file, default on
        # -- the same key, default and gating as transit.py and rm.py, so
        # a joint transit+RM+DT fit places the occultation at ONE time.
        self._light_travel_time_active = np.array(
            [bool(c.get("light_travel_time", True)) for c in self.config]
        )
        for i, (f, o) in enumerate(zip(self.files, self.orbit_names)):
            if not f or not o:
                raise ValueError(
                    f"[dopptom.{self.names[i]}] both `file:` and `orbit:` are "
                    f"required."
                )

    @property
    def prefix(self):
        return "dopptom"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.fits",
                "required": True,
                "doc": (
                    "EXOFASTv2-convention DT FITS file: primary HDU "
                    "ccf2d[ntime, nvel] line-profile residuals, HDU 1 BJD_TDB, "
                    "HDU 2 velocity grid [km/s]. The filename "
                    "nYYYYMMDD.<pl>.<inst>.<R>.fits encodes the resolving "
                    "power R unless `resolution:` is given."
                ),
            },
            {
                "key": "orbit",
                "kind": "ref",
                "accepts": ["orbit"],
                "required": True,
                "doc": "Name of the transiting orbit producing the shadow.",
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": False,
                "doc": (
                    "Band providing the quadratic limb darkening of the "
                    "occulted strip (default: first band)."
                ),
            },
            {
                "key": "resolution",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Spectrograph resolving power R (overrides the value "
                    "encoded in the filename)."
                ),
            },
            {
                "key": "velbin",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Velocity-grid binning factor applied at load time. "
                    "'auto' (default) block-averages the typically "
                    "supersampled grid down to ~3 pixels per instrumental "
                    "sigma (IndepVels ~ 3) -- no information loss, "
                    "proportionally cheaper likelihood. An integer forces "
                    "that factor; 1 disables binning."
                ),
            },
            {
                "key": "light_travel_time",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Apply the light-travel-time (Roemer delay) "
                    "correction to this dataset's exposure times on the "
                    "occultation seam (default true) -- the same key, "
                    "default and factor as transit's and rm's, so a "
                    "joint fit places the occultation at one time."
                ),
            },
        ]

    # ------------------------------------------------------------------
    def load_data(self, system):
        """Stage 1a: read the DT FITS cubes and derive data-driven scales."""
        from astropy.io import fits

        self.ccf2d, self.bjd, self.vel = [], [], []
        self.med, self.rms, self.indep_vels = [], [], []
        for i, fname in enumerate(self.files):
            with fits.open(fname) as hdul:
                ccf2d = np.asarray(hdul[0].data, dtype=float)  # (ntime, nvel)
                bjd = np.asarray(hdul[1].data, dtype=float)
                vel = np.asarray(hdul[2].data, dtype=float)  # km/s
            if ccf2d.shape != (bjd.size, vel.size):
                raise ValueError(
                    f"[dopptom.{self.names[i]}] ccf2d shape {ccf2d.shape} "
                    f"does not match (n_bjd={bjd.size}, n_vel={vel.size})."
                )
            R = self.resolutions[i]
            if R is None:
                import os

                tokens = os.path.basename(fname).split(".")
                try:
                    R = float(tokens[3])
                except (IndexError, ValueError):
                    R = 0.0
                if R <= 0:
                    raise ValueError(
                        f"[dopptom.{self.names[i]}] the filename does not "
                        f"encode the resolving power "
                        f"(nYYYYMMDD.<pl>.<inst>.<R>.fits); set `resolution:`."
                    )
            # An explicit resolution: gets the same gate as the parsed
            # path: R feeds a division (instrumental sigma = c/R) and the
            # lnL tempering, so 0, a negative, or a non-finite value would
            # surface far from here as a divide-by-zero or nonsense
            # broadening.
            R = float(R)
            if not np.isfinite(R) or not (1e2 <= R <= 1e7):
                raise ValueError(
                    f"[dopptom.{self.names[i]}] resolution must be a "
                    f"finite resolving power in a plausible spectrograph "
                    f"range (1e2..1e7); got {R!r}. A filename outside "
                    f"the nYYYYMMDD.<pl>.<inst>.<R>.fits convention can "
                    f"parse a stray numeric token as R -- set "
                    f"`resolution:` explicitly."
                )
            self.resolutions[i] = R

            # velocity binning (see module docstring): the grid is usually
            # supersampled ~10-30x vs the resolution element; block-average
            # before anything downstream sees the data.
            dv_native = float(np.median(np.abs(np.diff(vel))))
            indep_native = (C_KMS / float(R) / FWHM2SIGMA) / dv_native
            vb = self.velbins[i]
            if vb is None or vb == "auto":
                nbin = max(1, int(np.floor(indep_native / 3.0)))
                if nbin >= vel.size:
                    # auto self-limits: keep at least 2 bins
                    nbin = max(1, vel.size // 2)
                    logger.warning(
                        f"[dopptom.{self.names[i]}] velbin auto clamped "
                        f"to {nbin} on a {vel.size}-pixel grid."
                    )
            else:
                nbin = max(1, int(vb))
                if nbin >= vel.size:
                    # reduceat over zero edges would silently EMPTY the
                    # cube and then crash on counts[-1] far from the
                    # velbin: key that caused it (deep review finding).
                    raise ValueError(
                        f"[dopptom.{self.names[i]}] velbin={vb} on a "
                        f"{vel.size}-pixel velocity grid -- at least 2 "
                        f"bins must survive. Lower velbin: (or use "
                        f"'auto')."
                    )
            if nbin > 1:
                # Bin edges every nbin pixels; a trailing remainder is
                # FOLDED INTO THE LAST BIN (one slightly wider bin at one
                # edge of the grid) rather than silently dropped from the
                # likelihood, the rms estimate and n_tot.
                n_bins = vel.size // nbin
                edges = nbin * np.arange(n_bins)
                counts = np.diff(np.append(edges, vel.size))
                ccf2d = np.add.reduceat(ccf2d, edges, axis=1) / counts[None, :]
                vel = np.add.reduceat(vel, edges) / counts
                rem = int(counts[-1] - nbin)
                logger.info(
                    f"[dopptom.{self.names[i]}] velocity grid binned x{nbin} "
                    f"(dv {dv_native:.3f} -> {dv_native * nbin:.3f} km/s; "
                    f"velbin: {vb}"
                    + (
                        f"; last bin absorbs the {rem}-pixel remainder)"
                        if rem
                        else ")"
                    )
                )

            med = float(np.median(ccf2d))
            # EXOFASTv2 exofast_readdt rough per-pixel uncertainty
            rms0 = float(ccf2d.std())
            errscale = np.sqrt(
                np.sum(((ccf2d - med) / rms0) ** 2) / (ccf2d.size - 3.0)
            )
            rms = rms0 * errscale
            dv = float(np.median(np.abs(np.diff(vel))))
            rvel = C_KMS / float(R)
            # supersampling of the velocity grid vs the spectral resolution;
            # the whole lnL is divided by this (EXOFASTv2 correlated-pixel
            # tempering). Constant: depends only on the grid and R.
            indep = (rvel / FWHM2SIGMA) / dv
            if indep < 1.0:
                logger.warning(
                    f"[dopptom.{self.names[i]}] velocity grid is coarser "
                    f"than the spectral resolution (IndepVels={indep:.2f}; "
                    f"velbin too aggressive?) -- clamping the tempering to 1."
                )
                indep = 1.0

            self.ccf2d.append(ccf2d)
            self.bjd.append(bjd)
            self.vel.append(vel)
            self.med.append(med)
            self.rms.append(rms)
            self.indep_vels.append(indep)
            logger.info(
                f"[dopptom.{self.names[i]}] {ccf2d.shape[0]} exposures x "
                f"{ccf2d.shape[1]} velocities, R={R:.0f}, rms={rms:.3e}, "
                f"IndepVels={indep:.1f}"
            )

    # ------------------------------------------------------------------
    def register_parameters(self, system):
        """Stage 2: one error-scaling parameter per DT dataset."""
        self.manifest = {"dtscale": None}

    # ------------------------------------------------------------------
    # Static-window headroom: the sampler may wander to vsini values up to
    # ~2x the initval before the shadow could leave the window; the Gaussian
    # tails add 5 sigma and a 10 km/s floor guards small-vsini cases.
    _WINDOW_VSINI_HEADROOM = 1.95
    _WINDOW_GAUSS_NSIGMA = 5.0
    _WINDOW_PAD_KMS = 10.0

    def _shadow_window(self, i, vsini_init, vline_init):
        """Static velocity-pixel window that can ever contain the shadow.

        The shadow centre satisfies |vsini*up| <= ~1.2*vsini during transit;
        outside the window the model is exactly the constant baseline, so
        those pixels enter the likelihood through a precomputed (constant)
        sum of squares. The window is fixed at graph-build time from the
        initval vsini, widened by _WINDOW_VSINI_HEADROOM so the sampled vsini
        can move well above its starting value without the shadow leaving it.
        """
        sigma0 = np.sqrt(vline_init**2 + (C_KMS / self.resolutions[i]) ** 2)
        half = (
            self._WINDOW_VSINI_HEADROOM * vsini_init
            + self._WINDOW_GAUSS_NSIGMA * sigma0 / FWHM2SIGMA
            + self._WINDOW_PAD_KMS
        )
        self._window_half_kms[i] = float(half)
        return np.abs(self.vel[i]) <= half

    # The shadow centre reaches ~1.2 * vsini during transit (subplanet
    # |x_p| can exceed 1 at ingress/egress for a grazing chord); the
    # post-fit check below asks whether any posterior draw needed more
    # window than the initvals bought.
    _SHADOW_EXCURSION = 1.2

    def _check_shadow_window(self, points):
        """Post-fit rope (CLAUDE.md rope-not-gates): the window is FIXED
        from initvals at graph-build time, so a fit whose posterior vsini
        grew past the headroom has its shadow silently clipped against
        the precomputed out-of-window baseline -- warn loudly instead of
        letting the vsini/lambda posterior lean on the window edge."""
        if isinstance(points, dict):
            points = [points]
        if not points:
            return
        if "orbit.vsini" not in points[0] or "star.vline" not in points[0]:
            # A point layout without the tracked nodes cannot be
            # checked; say so instead of silently skipping (this used to
            # be a bare return INSIDE the per-dataset loop, skipping the
            # check for every dataset -- deep review finding).
            logger.warning(
                f"[{self.prefix}] shadow-window check skipped: the "
                f"point dict carries no tracked orbit.vsini/star.vline "
                f"nodes."
            )
            return
        for i, nd in enumerate(self._model_nodes):
            half = self._window_half_kms[i]
            if half is None:
                continue
            oidx, star_idx = nd["oidx"], nd["star_idx"]
            worst = -np.inf
            for pnt in points:
                vsini = float(np.atleast_1d(pnt["orbit.vsini"])[oidx]) / 1e3
                vline = float(np.atleast_1d(pnt["star.vline"])[star_idx]) / 1e3
                sigma = np.sqrt(vline**2 + (C_KMS / self.resolutions[i]) ** 2)
                worst = max(
                    worst,
                    self._SHADOW_EXCURSION * vsini
                    + self._WINDOW_GAUSS_NSIGMA * sigma / FWHM2SIGMA,
                )
            if worst > half:
                logger.warning(
                    f"[dopptom.{self.names[i]}] the sampled vsini/vline "
                    f"need a shadow window of ~{worst:.1f} km/s but the "
                    f"build-time window is {half:.1f} km/s (set from the "
                    f"start values): the shadow is being clipped against "
                    f"the frozen out-of-window baseline and the "
                    f"vsini/lambda posterior may lean on the window "
                    f"edge. Restart the fit with a start vsini near the "
                    f"posterior value."
                )

    def build_likelihood(self, model, system):
        from .. import ltt
        from .. import rm as rm_mod

        # band via getattr: a DT config with no band: block must reach
        # resolve_rm_indices, whose error names the fix ("add a band:
        # block naming the filter..."), instead of dying here on a raw
        # AttributeError (System only setattrs configured components).
        orbit, star, planet = system.orbit, system.star, system.planet
        band = getattr(system, "band", None)
        cm = self.config_manager

        # initval estimates for the static window (see _shadow_window).
        # No try/except: these parameters are declared whenever DT is
        # enabled, so a resolve failure is a config error to surface, not
        # a condition to paper over with a default that would silently
        # size the window for the wrong star (review: a 120 km/s rotator
        # behind a swallowed resolve error would be clipped at the
        # 50 km/s fallback's window with only a log line).
        sv_shape = (len(orbit.names),)
        sc = np.atleast_1d(
            cm.resolve("orbit", "svcoslam", shape=sv_shape, names=orbit.names)[
                "initval"
            ]
        )
        ss_ = np.atleast_1d(
            cm.resolve("orbit", "svsinlam", shape=sv_shape, names=orbit.names)[
                "initval"
            ]
        )
        vsini_init_all = (sc**2 + ss_**2) / 1e3  # km/s
        vline_init_all = (
            np.atleast_1d(
                cm.resolve(
                    "star",
                    "vline",
                    shape=(len(star.names),),
                    names=star.names,
                )["initval"]
            )
            / 1e3
        )
        p_init_all = np.atleast_1d(
            cm.resolve(
                "planet",
                "p",
                shape=(len(planet.names),),
                names=planet.names,
            )["initval"]
        )
        # retained for the post-fit window check (_check_shadow_window)
        self._window_half_kms = [None] * self.n_elements

        self._model_nodes = []
        for i in range(self.n_elements):
            oidx, pidx, bidx = rm_mod.resolve_rm_indices(
                system, self.orbit_names[i], self.band_names[i]
            )
            # The transited star, with no star-0 default: rm.py removed
            # exactly that fallback (rm_primary_star_index's docstring),
            # and here it would additionally read a vline element
            # mode_manifest never declared for that star.
            star_idx = rm_mod.rm_primary_star_index(orbit, oidx)

            ecc = orbit.ecc.value[oidx]
            omega = orbit.omega.value[oidx]
            inc = orbit.inc.value[oidx]
            lam = orbit.lam.value[oidx]
            ar = planet.ar.value[pidx]
            p = planet.p.value[pidx]
            u1 = band.u1.value[bidx]
            # With ld_law: linear the Band manifest has no u2 (same guard
            # as rm.py/transit.py); quad_limb_darkened_flux's Green's-basis
            # coefficients are exact at u2 = 0, so no separate branch.
            if "u2" in band.manifest:
                u2 = band.u2.value[bidx]
            else:
                u2 = pt.zeros_like(u1)
            vsini_kms = orbit.vsini.value[oidx] / 1e3
            vline_kms = star.vline.value[star_idx] / 1e3

            # Light-travel-time on the true-anomaly seam, exactly as
            # rm.py does for the same occultation geometry (same per-file
            # key, same OFF gating when the orbit's bodies did not
            # resolve, same mass-DIFFERENCE factor): without it a joint
            # transit+RM+DT fit places this dataset's occultation
            # ~a(M1-M2)/(Mc) away from the others' and pulls tc.
            ltt_on = bool(self._light_travel_time_active[i])
            if ltt_on and not ltt.orbit_supports_ltt(orbit):
                logger.warning(
                    "[dopptom.%s] light-travel-time correction disabled "
                    "-- the orbit does not define %s (its bodies did not "
                    "resolve). Set light_travel_time: false on this "
                    "dataset to silence this.",
                    self.names[i],
                    ", ".join(ltt.REQUIRED_ORBIT_PARAMS),
                )
                ltt_on = False
            time_i = pt.as_tensor_variable(self.bjd[i])
            if ltt_on:
                ltt_factor = (
                    orbit.m_primary.value[oidx] - orbit.m_companion.value[oidx]
                ) / orbit.m_total.value[oidx]
                time_i, _ = ltt.retarded_time(
                    time_i,
                    orbit.tp_target.value[oidx],
                    orbit.n.value[oidx],
                    ecc,
                    orbit.sinw.value[oidx],
                    orbit.cosw.value[oidx],
                    pt.sin(inc),
                    orbit.a.value[oidx],  # physical, R_sun -- NOT ar
                    factor=ltt_factor,
                    z0=0.0,
                    circular=orbit._all_circular([oidx]),
                )

            # geometry at the exposure midtimes (shared RM helpers);
            # orbit_idx solves Kepler for this orbit alone (review 6.8.1)
            f = orbit.get_true_anomaly(time_i, orbit_idx=oidx)
            x, y, z = rm_mod.rm_planet_xyz(f, ecc, omega, ar, inc, lam)
            rho = pt.sqrt(x * x + y * y)
            # shared Green's-basis LD flux (review 1.7: feeding mu-power
            # coefficients straight into quad_solution_vector is wrong for
            # any u2 != 0)
            flux = quad_limb_darkened_flux(rho, p, u1, u2)
            beta = pt.switch(pt.ge(z, 0.0), 1.0 - flux, 0.0)  # blocked frac

            # shadow: ellipse (x) Gaussian by quadrature over the strip
            sigma_g = (
                pt.sqrt(pt.sqr(vline_kms) + (C_KMS / self.resolutions[i]) ** 2)
                / FWHM2SIGMA
            )
            win = self._shadow_window(
                i, vsini_init_all[oidx], vline_init_all[star_idx]
            )
            v_win = self.vel[i][win]  # (nwin,)
            # Static quadrature order from the worst-case ellipse/Gaussian
            # width ratio this dataset can reach: the instrument-only
            # sigma floor (vline can sample toward 0) and the same vsini
            # headroom as the window.  Graph-build-time constant, like the
            # window itself (PR #323 review: a fixed 16 is wrong for
            # narrow-line fast rotators).
            sigma_floor = (C_KMS / self.resolutions[i]) / FWHM2SIGMA
            ratio0 = (
                self._WINDOW_VSINI_HEADROOM
                * vsini_init_all[oidx]
                * p_init_all[pidx]
            ) / sigma_floor
            n_gl_i = choose_quadrature_order(ratio0)
            logger.info(
                f"[dopptom.{self.names[i]}] quadrature order "
                f"n_gl={n_gl_i} (worst-case width ratio {ratio0:.1f})"
            )
            shadow = dt_shadow(v_win, x, vsini_kms, p, sigma_g, n_gl=n_gl_i)
            # EXOFASTv2 normalizes the bump in v/vsini units
            # (int bump d(v/vsini) = beta), i.e. vsini times the per-km/s
            # unit-area profile dt_shadow returns.
            model_win = beta[:, None] * shadow * vsini_kms

            # tempered Gaussian likelihood (EXOFASTv2 chi2 / IndepVels)
            dtscale_i = self.dtscale.value[i]
            sigma = self.rms[i] * dtscale_i
            resid_win = (
                pt.as_tensor_variable(self.ccf2d[i][:, win] - self.med[i])
                - model_win
            )
            ssq_out = float(
                np.sum((self.ccf2d[i][:, ~win] - self.med[i]) ** 2)
            )
            n_tot = self.ccf2d[i].size
            chi2 = (pt.sum(pt.sqr(resid_win)) + ssq_out) / pt.sqr(sigma)
            logl = (
                -0.5
                * (chi2 + n_tot * (2.0 * pt.log(sigma) + np.log(2.0 * np.pi)))
                / self.indep_vels[i]
            )
            pm.Potential(f"{self.prefix}.{self.names[i]}.loglike", logl)
            self._model_nodes.append(
                dict(
                    win=win,
                    model_win=model_win,
                    vsini=vsini_kms,
                    period=orbit.period.value[oidx],
                    tc=orbit.tc.value[oidx],
                    ar=ar,
                    p=p,
                    cosi=pt.cos(inc),
                    ecc=ecc,
                    omega=omega,
                    lam=lam,
                    inc=inc,
                    # The shadow's centre velocity vsini * x_p at each
                    # exposure -- the 1D "shadow trajectory" the GUI chart
                    # draws (plot_data), retained symbolically so
                    # param_deps can be walked from it.
                    subvel=vsini_kms * x,
                    tfwhm=planet.tfwhm.value[pidx],
                    t14=planet.t14.value[pidx],
                    oidx=oidx,
                    star_idx=star_idx,
                )
            )

    # ------------------------------------------------------------------
    def compile_plotters(self, model, system):
        """Compile per-dataset functions returning the shadow image, the
        1D shadow-trajectory velocities (plot_data's model trace), and the
        orbit scalars needed to annotate the EXOFASTv2-style figure."""
        import pytensor

        param_symbols = [p.value for p in system.plot_params]
        self._plot_fns = []
        self._subvel_nodes = []
        for nd in self._model_nodes:
            outs = [
                nd["model_win"],
                nd["vsini"],
                nd["period"],
                nd["tc"],
                nd["ar"],
                nd["p"],
                nd["cosi"],
                nd["ecc"],
                nd["omega"],
                nd["lam"],
                nd["inc"],
                nd["subvel"],
                nd["tfwhm"],
            ]
            self._subvel_nodes.append(nd["subvel"])
            self._plot_fns.append(
                pytensor.function(
                    param_symbols, outs, on_unused_input="ignore"
                )
            )

    # ------------------------------------------------------------------
    def _shadow_centroid_data(self, i):
        """Per-exposure flux-weighted centroid velocity of the observed
        line-profile residuals [km/s], NaN where an exposure carries no
        significant shadow signal (out of transit).  Pure numpy on the
        loaded cube: usable with point=None, before any model exists."""
        resid = self.ccf2d[i] - self.med[i]
        w = np.clip(resid, 0.0, None)
        denom = w.sum(axis=1)
        cent = np.full(denom.shape, np.nan)
        strong = denom > 0.25 * denom.max() if denom.max() > 0 else denom > 0
        np.divide(w @ self.vel[i], denom, out=cent, where=strong)
        return cent

    def plot_data(self, system, point=None):
        """GUI charts: one shadow-trajectory chart per DT dataset.

        The 2D phase-velocity image triple stays a bespoke matplotlib
        figure (``plot``), as the plot contract allows for image-like
        diagnostics; this chart is the 1D projection both renderers can
        draw with the shared scatter/line vocabulary -- the observed
        shadow centroid velocity per exposure against the model's
        subplanet velocity vsini * x_p(t), the trace whose slope and
        extent carry lambda and vsini.
        """
        from exozippy.chart import Chart, Trace

        specs = []
        for i in range(self.n_elements):
            name = self.names[i]
            traces = [
                Trace(
                    name=name,
                    role="data",
                    kind="scatter",
                    x=self.bjd[i],
                    y=self._shadow_centroid_data(i),
                    style={"series_index": i},
                )
            ]
            deps = []
            fns = getattr(self, "_plot_fns", None)
            if point is not None and fns:
                args = self._point_to_plot_params(point, system)
                # subvel is outs[-2]: tfwhm was appended AFTER it for the
                # bespoke figure's duration markers, and grabbing [-1]
                # here handed the model trace a 0-d duration scalar (deep
                # review finding: matplotlib then aborts every DT plot).
                subvel = np.asarray(fns[i](*args)[-2], dtype=float)
                if subvel.shape != np.shape(self.bjd[i]):
                    raise RuntimeError(
                        f"[dopptom.{name}] plot outputs out of order: "
                        f"expected the (n_exposure,) subplanet-velocity "
                        f"vector, got shape {subvel.shape}."
                    )
                node = self._subvel_nodes[i]
                deps = self._model_trace_param_deps(node, system)
                traces.append(
                    Trace(
                        name="model",
                        role="model",
                        kind="line",
                        x=self.bjd[i],
                        y=subvel,
                        node=node,
                    )
                )
            specs.append(
                Chart(
                    id=f"dopptom-{name}",
                    component={"yaml_key": "dopptom", "instance": name},
                    title=f"Doppler shadow trajectory: {name}",
                    xlabel="Time [BJD_TDB]",
                    ylabel="Shadow velocity [km/s]",
                    traces=traces,
                    param_deps=deps,
                    meta={
                        "file_tag": f"DT_trace_{name}",
                        "figsize": (12, 5),
                        "caption": (
                            r"Doppler-shadow trajectory of "
                            + latex_escape(name)
                            + r": the flux-weighted centroid velocity "
                            r"of the observed line-profile residuals "
                            r"per exposure (points) against the "
                            r"model subplanet velocity "
                            r"$v\sin{i_*}\,x_p(t)$ (line)."
                        ),
                    },
                )
            )
        return specs

    def plot(self, system, points, filename_prefix="debug"):
        """The standard chart PDFs (plot_via_specs over plot_data), plus
        the bespoke EXOFASTv2 dopptom_chi2-style figure per dataset:
        Data / Model / Residuals as grayscale phase-velocity images with
        +-vsini and ingress/egress markers and a 'Fractional Variation'
        colorbar (an image plot, outside the Chart mark vocabulary, kept
        matplotlib-only as the plot contract allows for such
        diagnostics)."""
        import matplotlib.pyplot as plt

        if not getattr(self, "_plot_fns", None):
            return
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)
        if isinstance(points, dict):
            points = [points]
        if len(points) == 0:
            logger.warning("No points provided for DT plotting.")
            return
        # The REFERENCE draw, same convention as plot_via_specs (data and
        # decorations from points[0]): a parameter-wise median across
        # draws is not a point on the posterior -- in a bimodal lambda
        # posterior the sv medians give a vsini no draw has.
        point = points[0]
        self._check_shadow_window(points)
        args = self._point_to_plot_params(point, system)

        for i, fn in enumerate(self._plot_fns):
            (
                mwin,
                vsini,
                period,
                tc,
                ar,
                p,
                cosi,
                ecc,
                omega,
                lam,
                inc,
                _subvel,
                tfwhm,
            ) = fn(*args)
            vsini, period, tc = float(vsini), float(period), float(tc)
            ar, p, cosi, ecc, omega, lam, inc, tfwhm = (
                float(ar),
                float(p),
                float(cosi),
                float(ecc),
                float(omega),
                float(lam),
                float(inc),
                float(tfwhm),
            )
            win = self._model_nodes[i]["win"]
            ccf, vel, bjd = self.ccf2d[i], self.vel[i], self.bjd[i]
            model = np.full_like(ccf, 0.0)
            model[:, win] = np.asarray(mwin)
            resid = (ccf - self.med[i]) - model

            # orbital phase about the local transit centre (EXOFASTv2)
            nper = np.round((bjd.mean() - tc) / period)
            phase = (bjd - (tc + period * nper)) / period

            # ingress/egress markers at +-Tfwhm/2, read from the MODEL's
            # own planet.tfwhm node (compiled into the plotter outputs)
            # rather than a hand copy of the Winn 2010 formula: the model
            # node carries the grazing-geometry handling and can never
            # drift from the reported durations.
            egress_phase = 0.5 * tfwhm / period

            sigma = resid.std()
            vr = 5.0 * sigma
            vmax_plot = np.max(np.abs(vel))
            phmax = np.max(np.abs(phase))
            extent = [vel.min(), vel.max(), phase.min(), phase.max()]

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(7, 12),
                gridspec_kw={"height_ratios": [1, 1, 1, 0.08]},
            )
            # EXOFASTv2 model title: lambda and inclination in degrees
            model_title = (
                rf"Model ($\lambda$={np.degrees(lam):.1f}$^\circ$, "
                rf"i={np.degrees(inc):.1f}$^\circ$, "
                rf"vsini={vsini:.1f} km/s)"
            )
            panels = [
                (-(ccf - self.med[i]), f"{self.names[i]} Doppler Data"),
                (-model, model_title),
                (-resid, "Residuals"),
            ]
            for ax, (img, title) in zip(axes[:3], panels):
                ax.imshow(
                    img,
                    aspect="auto",
                    origin="lower",
                    cmap="gray",
                    extent=extent,
                    vmin=-vr,
                    vmax=vr,
                )
                for vv in (-vsini, vsini):
                    ax.axvline(vv, color="tab:blue", lw=1.5)
                for pp in (-egress_phase, egress_phase):
                    if np.isfinite(pp):
                        ax.axhline(pp, color="tab:blue", lw=1.5)
                ax.set_xlim(-vmax_plot, vmax_plot)
                ax.set_ylim(-phmax, phmax)
                ax.set_ylabel("Orbital Phase")
                ax.set_title(title, fontsize=10)
            axes[2].set_xlabel("Velocity (km/s)")
            grad = np.linspace(-vr, vr, 256)[None, :]
            axes[3].imshow(
                grad,
                aspect="auto",
                cmap="gray",
                extent=[-vr, vr, -1, 1],
            )
            axes[3].set_yticks([])
            axes[3].set_xlabel("Fractional Variation")
            fig.tight_layout()
            # PDF like every sibling component's saved figures (the
            # images inside stay rasters; PDF is only the container).
            # NOTE the modeling draft pairs figures by Chart file_tag
            # (outputs/modeling.py collect_figures), so this bespoke
            # image is NOT auto-collected -- the DT_trace chart is the
            # one that reaches the draft; include this file by hand.
            out = f"{filename_prefix}_DT_{self.names[i]}.pdf"
            fig.savefig(out)
            plt.close(fig)
            logger.info(f"[dopptom.{self.names[i]}] wrote {out}")
