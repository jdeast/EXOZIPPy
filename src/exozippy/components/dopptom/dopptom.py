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


class Dopptom(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Doppler Tomography"
        self.files = [c.get("file") for c in self.config]
        self.orbit_names = [c.get("orbit") for c in self.config]
        self.band_names = [c.get("band") for c in self.config]
        self.resolutions = [c.get("resolution") for c in self.config]
        self.velbins = [c.get("velbin", "auto") for c in self.config]
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
            self.resolutions[i] = float(R)

            # velocity binning (see module docstring): the grid is usually
            # supersampled ~10-30x vs the resolution element; block-average
            # before anything downstream sees the data.
            dv_native = float(np.median(np.abs(np.diff(vel))))
            indep_native = (C_KMS / float(R) / FWHM2SIGMA) / dv_native
            vb = self.velbins[i]
            if vb is None or vb == "auto":
                nbin = max(1, int(np.floor(indep_native / 3.0)))
            else:
                nbin = max(1, int(vb))
            if nbin > 1:
                nkeep = (vel.size // nbin) * nbin
                ccf2d = ccf2d[:, :nkeep].reshape(
                    ccf2d.shape[0], nkeep // nbin, nbin
                ).mean(axis=2)
                vel = vel[:nkeep].reshape(nkeep // nbin, nbin).mean(axis=1)
                logger.info(
                    f"[dopptom.{self.names[i]}] velocity grid binned x{nbin} "
                    f"(dv {dv_native:.3f} -> "
                    f"{dv_native * nbin:.3f} km/s; velbin: {vb})"
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
        return np.abs(self.vel[i]) <= half

    def build_likelihood(self, model, system):
        from .. import rm as rm_mod

        orbit, star, planet, band = (
            system.orbit,
            system.star,
            system.planet,
            system.band,
        )
        cm = self.config_manager

        # initval estimates for the static window (see _shadow_window)
        sv_shape = (len(orbit.names),)
        try:
            sc = np.atleast_1d(
                cm.resolve(
                    "orbit", "svcoslam", shape=sv_shape, names=orbit.names
                )["initval"]
            )
            ss_ = np.atleast_1d(
                cm.resolve(
                    "orbit", "svsinlam", shape=sv_shape, names=orbit.names
                )["initval"]
            )
            vsini_init_all = (sc**2 + ss_**2) / 1e3  # km/s
        except Exception as exc:
            logger.warning(
                f"[{self.prefix}] could not resolve svcoslam/svsinlam "
                f"initvals ({exc}); using 50 km/s for the shadow window."
            )
            vsini_init_all = np.full(len(orbit.names), 50.0)
        try:
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
        except Exception as exc:
            logger.warning(
                f"[{self.prefix}] could not resolve vline initval ({exc}); "
                f"using 5 km/s for the shadow window."
            )
            vline_init_all = np.full(len(star.names), 5.0)

        self._model_nodes = []
        for i in range(self.n_elements):
            oidx, pidx, bidx = rm_mod.resolve_rm_indices(
                system, self.orbit_names[i], self.band_names[i]
            )
            star_idx = next(
                (
                    idx
                    for (ctype, idx) in orbit.primary_bodies[oidx]
                    if ctype == "star"
                ),
                0,
            )

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

            # geometry at the exposure midtimes (shared RM helpers);
            # orbit_idx solves Kepler for this orbit alone (review 6.8.1)
            f = orbit.get_true_anomaly(self.bjd[i], orbit_idx=oidx)
            x, y, z = rm_mod.rm_planet_xyz(f, ecc, omega, ar, inc, lam)
            rho = pt.sqrt(x * x + y * y)
            # shared Green's-basis LD flux (review 1.7: feeding mu-power
            # coefficients straight into quad_solution_vector is wrong for
            # any u2 != 0)
            flux = quad_limb_darkened_flux(rho, p, u1, u2)
            beta = pt.switch(pt.ge(z, 0.0), 1.0 - flux, 0.0)  # blocked frac

            # shadow: ellipse (x) Gaussian by quadrature over the strip
            sigma_g = (
                pt.sqrt(
                    pt.sqr(vline_kms) + (C_KMS / self.resolutions[i]) ** 2
                )
                / FWHM2SIGMA
            )
            win = self._shadow_window(
                i, vsini_init_all[oidx], vline_init_all[star_idx]
            )
            v_win = self.vel[i][win]  # (nwin,)
            shadow = dt_shadow(v_win, x, vsini_kms, p, sigma_g)
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
                )
            )

    # ------------------------------------------------------------------
    def compile_plotters(self, model, system):
        """Compile per-dataset functions returning the shadow image and the
        orbit scalars needed to annotate the EXOFASTv2-style figure."""
        import pytensor

        param_symbols = [p.value for p in system.plot_params]
        self._plot_fns = []
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
            ]
            self._plot_fns.append(
                pytensor.function(
                    param_symbols, outs, on_unused_input="ignore"
                )
            )

    def plot(self, system, points, filename_prefix="debug"):
        """EXOFASTv2 dopptom_chi2-style figure per dataset: Data / Model /
        Residuals as grayscale phase-velocity images with +-vsini and
        ingress/egress markers and a 'Fractional Variation' colorbar."""
        import matplotlib.pyplot as plt

        if not getattr(self, "_plot_fns", None):
            return
        if isinstance(points, dict):
            points = [points]
        if len(points) == 0:
            logger.warning("No points provided for DT plotting.")
            return
        if len(points) == 1:
            point = points[0]
        else:  # posterior median across draws
            point = {
                k: np.median(
                    [np.asarray(pnt[k], dtype=float) for pnt in points],
                    axis=0,
                )
                for k in points[0]
            }
        args = self._point_to_plot_params(point, system)

        for i, fn in enumerate(self._plot_fns):
            (mwin, vsini, period, tc, ar, p, cosi, ecc, omega, lam, inc) = fn(
                *args
            )
            vsini, period, tc = float(vsini), float(period), float(tc)
            ar, p, cosi, ecc, omega, lam, inc = (
                float(ar),
                float(p),
                float(cosi),
                float(ecc),
                float(omega),
                float(lam),
                float(inc),
            )
            win = self._model_nodes[i]["win"]
            ccf, vel, bjd = self.ccf2d[i], self.vel[i], self.bjd[i]
            model = np.full_like(ccf, 0.0)
            model[:, win] = np.asarray(mwin)
            resid = (ccf - self.med[i]) - model

            # orbital phase about the local transit centre (EXOFASTv2)
            nper = np.round((bjd.mean() - tc) / period)
            phase = (bjd - (tc + period * nper)) / period

            # ingress/egress phases (EXOFASTv2 t14/t23 -> Tfwhm/2)
            sini = np.sqrt(max(1.0 - cosi**2, 0.0))
            esinw = ecc * np.sin(omega)
            bp = ar * cosi * (1.0 - ecc**2) / (1.0 + esinw)
            fac = np.sqrt(1.0 - ecc**2) / (1.0 + esinw)
            with np.errstate(invalid="ignore"):
                t14 = (
                    period / np.pi
                    * np.arcsin(np.sqrt((1.0 + p) ** 2 - bp**2) / (sini * ar))
                    * fac
                )
                t23 = (
                    period / np.pi
                    * np.arcsin(np.sqrt((1.0 - p) ** 2 - bp**2) / (sini * ar))
                    * fac
                )
            tfwhm = t14 - (t14 - t23) / 2.0
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
            out = f"{filename_prefix}_DT_{self.names[i]}.png"
            fig.savefig(out, dpi=130)
            plt.close(fig)
            logger.info(f"[dopptom.{self.names[i]}] wrote {out}")
