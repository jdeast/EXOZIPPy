import logging

import numpy as np
import pymc as pm
import pytensor

logger = logging.getLogger(__name__)
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops

from exozippy.components.instrument import Instrument
from exozippy.components.limbdark import quad_limb_darkened_flux

from . import physics


class Transit(Instrument):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Transit Parameters"
        # Filter identity and limb darkening live on the Band component;
        # each instrument references a band block by name.
        self.band_names = [c.get("band") for c in self.config]
        deprecated_filters = [
            c.get("filter") for c in self.config if c.get("filter")
        ]
        if deprecated_filters:
            logger.warning(
                "transit 'filter:' is deprecated and ignored; reference a "
                "band: block instead (bands carry the filter identity and "
                "limb darkening)."
            )
        self.total_detrend_cols = 0
        # SED depth-dilution node, built once by build_likelihood and
        # reused by compile_plotters.
        self._dilution_node = None

    @property
    def prefix(self):
        return "transit"

    @classmethod
    def get_utilities(cls):
        from ...utilities import getdata
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="getdata",
                label="Download TESS/Kepler light curves",
                description=(
                    "Fetch TESS/Kepler light curves via lightkurve and write "
                    "them in EXOFASTv2/EXOZIPPy format."
                ),
                component_keys=["transit"],
                available=True,
                build_parser=getdata.build_parser,
                run=argparse_subprocess_runner("exozippy.utilities.getdata"),
            ),
            UtilitySpec(
                name="bls",
                label="BLS period search",
                description=(
                    "Box Least Squares transit-period search (not yet "
                    "implemented)."
                ),
                component_keys=["transit"],
                available=False,
            ),
        ]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.dat",
                "required": True,
                "doc": (
                    "Whitespace-delimited light curve; columns are time, "
                    "flux, flux_err, then optional detrend columns. Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": True,
                "doc": (
                    "Name of the band: block carrying this light curve's "
                    "filter identity and limb darkening."
                ),
            },
            {
                "key": "filter",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Deprecated and ignored; reference a band: block "
                    "instead (bands carry the filter identity)."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(("time", "flux", "err")),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
        ]

    def load_data(self, system):
        """Stage 1a: Load CSVs and generate data-driven bounds/inits."""
        self.baseline_init = [1.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        # Fixed instrument properties (not fitted parameters), read straight
        # off the config like self.files/self.band_names: exptime is the
        # exposure duration in minutes, ninterp the number of sub-samples
        # used to smear the model over that exposure (EXOFASTv2 parity).
        # Invalid combinations (ninterp < 1, a non-positive exptime, or
        # ninterp > 1 without an exptime to smear over) warn and fall back
        # to the inert defaults (ninterp=1, exptime=1: no smearing) rather
        # than crashing in the grid builder or silently smearing over a
        # made-up exposure time.
        self.exptime_min = [1.0] * self.n_elements
        self.ninterp = [1] * self.n_elements
        for i, c in enumerate(self.config):
            exptime = float(c.get("exptime", 1.0))
            ninterp = int(c.get("ninterp", 1))
            if (
                ninterp < 1
                or exptime <= 0
                or (ninterp > 1 and "exptime" not in c)
            ):
                logger.warning(
                    f"transit {self.names[i]}: invalid exposure smearing "
                    f"config (exptime={c.get('exptime', '<unset>')} min, "
                    f"ninterp={c.get('ninterp', '<unset>')}): needs "
                    f"ninterp >= 1 and, when ninterp > 1, an explicit "
                    f"exptime > 0. Ignoring -- defaulting to ninterp=1, "
                    f"exptime=1 (no smearing)."
                )
                exptime, ninterp = 1.0, 1
            self.exptime_min[i] = exptime
            self.ninterp[i] = ninterp

        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            # Shared reader: columns:, mask:, time_* conversion, then one
            # sort per file before anything is derived from it, keeping the
            # flux, errors and detrend columns aligned by construction.
            df = self._read_data(
                i, roles=("time", "flux", "err"), detrend=True
            )
            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = self._jitter_floor(df.iloc[:, 2].values)

            blocks.add(
                i,
                time=df.iloc[:, 0].values,
                obs=df.iloc[:, 1].values,
                err=df.iloc[:, 2].values,
                df=df,
            )

        # Shared accumulator: concatenation (time/flux/err), inst_map, the
        # per-file row ranges, the block-diagonal detrend matrix, and the
        # optional GP / robust-likelihood hooks.  No user_factor: the errors
        # are already in the amplitude parameters' unit (relative flux).
        blocks.finalize("flux")

        self._build_oversample_grid()

    def _build_oversample_grid(self):
        """
        Build per-ninterp observation groups (self._oversample_groups) used
        to smear the model over each instrument's exposure time (EXOFASTv2
        exofast_chi2v2.pro parity: a sub-exposure time grid collapsed with
        a weighted mean per observation).

        exptime/ninterp are per-instrument, but self.time is concatenated
        across instruments, so instruments may disagree on both. Rather
        than padding every observation out to the largest ninterp among
        the active instruments (which would make build_likelihood evaluate
        the transit model at that many sub-samples even for observations
        whose own instrument needs only 1), observations are partitioned
        by their own instrument's ninterp value into groups: each group's
        sub-exposure time grid is exactly as wide as that group's own
        ninterp, so a likelihood evaluation costs exactly the sub-samples
        each observation's instrument needs, never another instrument's
        larger ninterp.

        build_likelihood processes groups in np.unique(ninterp) order, not
        row order, so self._oversample_inverse_order (the argsort of the
        concatenated group row-indices) is recorded here to gather
        per-group results back into original observation order afterward
        with a plain index, instead of a scatter.
        """
        exptime_days = np.asarray(self.exptime_min, dtype=float) / 1440.0
        ninterp_per_inst = np.asarray(self.ninterp, dtype=int)

        ninterp_obs = ninterp_per_inst[self.inst_map]  # (N_obs,)
        exptime_obs = exptime_days[self.inst_map]  # (N_obs,)

        self._oversample_groups = []
        row_order = []
        for kk in np.unique(ninterp_obs):
            rows = np.nonzero(ninterp_obs == kk)[0]
            if kk == 1:
                # A lone sample sits at the timestamp itself; exptime is
                # irrelevant (matches the original instantaneous model).
                grid = self.time[rows][:, None]
                weights = np.ones(1)
            else:
                # Midpoint Riemann sub-times: kk equal-width cells across
                # [-exptime/2, +exptime/2], sampled at each cell's center
                # (offsets (j + 0.5)/kk - 0.5, never the exposure edges).
                # This is exactly EXOFASTv2's grid (exofast_chi2v2.pro:
                # dindgen(ninterp)/ninterp - (ninterp-1)/(2*ninterp)),
                # chosen there over trapezoid/Simpson: with uniform 1/kk
                # weights, edge samples would overweight the exposure
                # boundaries and systematically over-smear.
                j = np.arange(kk)
                frac = (j + 0.5) / kk - 0.5
                grid = (
                    self.time[rows][:, None]
                    + frac[None, :] * exptime_obs[rows][:, None]
                )
                weights = np.full(kk, 1.0 / kk)
            self._oversample_groups.append((rows, grid, weights))
            row_order.append(rows)
        self._oversample_inverse_order = np.argsort(np.concatenate(row_order))

    def register_parameters(self, system):
        """Stage 2: Embed data-driven hints into the PyMC manifest."""
        self._hint_baseline()
        self.manifest = {"baseline": None}
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {
                "shape": (self.total_detrend_cols,)
            }

        # Limb darkening comes from the Band component; map each
        # instrument (and each observation) to its band. A transit model
        # cannot be computed without LD, so a missing/unknown band
        # reference is an error, not a warning.
        if not hasattr(system, "band"):
            raise ValueError(
                "Transit instruments require band: blocks (bands carry the "
                "filter identity and limb-darkening parameters)."
            )
        name_to_idx = {name: i for i, name in enumerate(system.band.names)}
        missing = [
            (self.names[i], n)
            for i, n in enumerate(self.band_names)
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"Transit instrument(s) reference unknown band(s): "
                f"{missing}. Available bands: {list(name_to_idx)}."
            )
        self.band_map = np.array(
            [name_to_idx[n] for n in self.band_names], dtype=int
        )
        self.obs_band_map = self.band_map[self.inst_map]

    def _hint_baseline(self):
        """Push each light curve's median flux as a RANK_DERIVED_DATA hint.

        The median is measured in ``load_data`` (stage 1a), so it is ready
        by the time this runs at stage 2 -- which is what lets it go through
        the provenance pipeline at all.

        It used to be a plain manifest option (``{"baseline": {"initval":
        ...}}``), and options are merged as ``{**cfg, **options}`` AFTER
        ``resolve()``: they beat the user's params file outright and never
        acquire a rank.  For a data-derived START value that is backwards --
        an explicit ``transit.<name>.baseline`` in a params file (a restart
        file, say) was silently discarded.  As a hint it sits at
        RANK_DERIVED_DATA (60), the tier this channel exists for: above the
        defaults.yaml 1.0 (20) and below the user (100), exactly like
        ``rvinstrument``'s gamma (median RV) and ``mulensinstrument``'s
        f_source/log_f_total.

        A non-finite median (a file with no usable flux) is skipped rather
        than hinted: the defaults.yaml 1.0 is then what the engine resolves,
        which is also the no-data fallback ``load_data`` seeds
        ``baseline_init`` with, so the two agree by construction.
        """
        baseline_init = getattr(self, "baseline_init", None)
        if baseline_init is None:
            return  # register_parameters without load_data (bare harness)
        arr = np.atleast_1d(baseline_init)
        for i in range(self.n_elements):
            val = float(arr[i])
            if not np.isfinite(val):
                logger.warning(
                    f"transit {self.names[i]}: median flux is not finite; "
                    f"leaving baseline at its defaults.yaml start value."
                )
                continue
            self.config_manager.add_hint(f"{self.prefix}.{i}.baseline", val)

    def _build_dilution(self, system):
        """
        Per-instrument SED-predicted depth dilution factor
        F_host / sum_j F_j in the instrument's band, as a (n_elements,)
        tensor (Deterministic "transit.dilution" for diagnostics), or
        None if no instrument's band filter is in the SED's BC grid.
        Instruments whose band filter is unavailable get dilution 1.
        """
        if getattr(self, "_dilution_node", None) is not None:
            return self._dilution_node

        sed = system.sed
        band = system.band
        dils = []
        any_diluted = False
        for i in range(self.n_elements):
            band_idx = int(self.band_map[i])
            filter_key = band.filter_mist[band_idx]
            if filter_key and sed.has_filter(filter_key):
                host = int(band.star_indices[band_idx])
                dils.append(
                    sed.predict_flux_fraction(host, filter_key, system)
                )
                any_diluted = True
            else:
                logger.warning(
                    f"transit {self.names[i]}: band filter '{filter_key}' "
                    f"is not in the SED's BC grid; no depth deblending "
                    f"applied for this instrument."
                )
                dils.append(pt.constant(1.0))
        if not any_diluted:
            return None
        self._dilution_node = pm.Deterministic(
            f"{self.prefix}.dilution", pt.stack(dils)
        )
        return self._dilution_node

    def build_likelihood(self, model, system):
        time = pm.Data("transit_time", self.time)
        flux = pm.Data("transit_data", self.flux)
        err = pm.Data("transit_err", self.err)

        orbits = system.orbit
        planets = system.planet

        # 1. Start with the photometric baseline
        lc_model = self.baseline.value[self.inst_map_tensor]

        # 1b. Per-planet transit/occultation geometry (impact parameter & durations),
        # exposed as Deterministics for diagnostics and plotting (e.g. phased-plot xlim).
        ecc_p = orbits.ecc.value[planets.orbit_map]  # (N_planets,)
        esinw_p = orbits.esinw.value[planets.orbit_map]
        inc_p = orbits.inc.value[planets.orbit_map]
        period_p = orbits.period.value[planets.orbit_map]
        tc_p = orbits.tc.value[planets.orbit_map]
        ar_p = planets.ar.value
        p_p = planets.p.value

        # Numerical-stability floor for the geometry below. Keeps arcsin arguments
        # strictly inside (-1, 1) (where its derivative is finite) and denominators
        # away from 0, so a transient excursion during NUTS leapfrog steps (e.g.
        # inc away from 90 deg, or ecc/esinw near 1) can't produce a NaN/inf
        # gradient. Values at the actual posterior mode are far from these floors,
        # so the reported b/t14/tau are unaffected.
        _GEOM_EPS = 1e-6

        sini_p = pt.sin(inc_p)
        cosi_p = pt.cos(inc_p)
        ecc_factor = pt.sqrt(pt.clip(1.0 - pt.sqr(ecc_p), _GEOM_EPS, 1.0))

        denom_minus = pt.clip(1.0 - esinw_p, _GEOM_EPS, np.inf)
        denom_plus = pt.clip(1.0 + esinw_p, _GEOM_EPS, np.inf)
        sini_ar = pt.clip(pt.abs(sini_p * ar_p), _GEOM_EPS, np.inf)

        # Winn 2010 eqs 7-8: the primary transit happens at true anomaly
        # pi/2 - omega (see calc_tp), where r = a(1-e^2)/(1 + esinw); the
        # secondary sits at the opposite conjunction, r = a(1-e^2)/(1 - esinw).
        dur_b = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_plus
        dur_bs = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_minus

        def _arcsin_term(p_offset_sq, dur_bx):
            radicand = pt.clip(p_offset_sq - pt.sqr(dur_bx), 0.0, np.inf)
            arg = pt.clip(
                pt.sqrt(radicand) / sini_ar, -1.0 + _GEOM_EPS, 1.0 - _GEOM_EPS
            )
            return pt.arcsin(arg)

        # Winn 2010 eqs 14-16: the duration's eccentricity correction is
        # sqrt(1-e^2)/(1 + esinw) for the primary and sqrt(1-e^2)/(1 - esinw)
        # for the secondary.
        dur_t14 = (
            (period_p / np.pi)
            * _arcsin_term(pt.sqr(1.0 + p_p), dur_b)
            * ecc_factor
            / denom_plus
        )
        dur_t14s = (
            (period_p / np.pi)
            * _arcsin_term(pt.sqr(1.0 + p_p), dur_bs)
            * ecc_factor
            / denom_minus
        )

        # The (1-p)^2 arcsin term is Winn 2010's t23 (full-occultation
        # duration, 2nd to 3rd contact); the FWHM is (t14 + t23)/2 and the
        # ingress/egress duration tau is (t14 - t23)/2 (EXOFASTv2
        # derivepars.pro convention).
        dur_t23 = (
            (period_p / np.pi)
            * _arcsin_term(pt.sqr(1.0 - p_p), dur_b)
            * ecc_factor
            / denom_plus
        )
        dur_t23s = (
            (period_p / np.pi)
            * _arcsin_term(pt.sqr(1.0 - p_p), dur_bs)
            * ecc_factor
            / denom_minus
        )

        dur_tfwhm = (dur_t14 + dur_t23) / 2.0
        dur_tfwhms = (dur_t14s + dur_t23s) / 2.0
        dur_tau = (dur_t14 - dur_t23) / 2.0
        dur_taus = (dur_t14s - dur_t23s) / 2.0

        pm.Deterministic(f"{self.prefix}.b", dur_b)
        pm.Deterministic(f"{self.prefix}.bs", dur_bs)
        pm.Deterministic(f"{self.prefix}.t14", dur_t14)
        pm.Deterministic(f"{self.prefix}.t14s", dur_t14s)
        pm.Deterministic(f"{self.prefix}.tfwhm", dur_tfwhm)
        pm.Deterministic(f"{self.prefix}.tfwhms", dur_tfwhms)
        pm.Deterministic(f"{self.prefix}.tau", dur_tau)
        pm.Deterministic(f"{self.prefix}.taus", dur_taus)

        # 2. Orbital elements per planet. These don't depend on the
        # observation/sub-exposure grid, so they're computed once and
        # reused by every ninterp group below (see 4.).
        tp = orbits.tp.value[planets.orbit_map][
            None, None, :
        ]  # (1, 1, N_planets)
        n = orbits.n.value[planets.orbit_map][None, None, :]
        ecc = ecc_p[None, None, :]
        cosw = orbits.cosw.value[planets.orbit_map][None, None, :]
        sinw = orbits.sinw.value[planets.orbit_map][None, None, :]
        inc = inc_p[None, None, :]
        a_rstar = ar_p[None, None, :]
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        # 3. Limb Darkening Setup (per observation, mapped from each
        # instrument's Band). When every band uses the linear law, Band's
        # manifest has no u2; the quadratic term is then zero.
        band = system.band

        # Secondary-eclipse thermal emission (fitthermal): gate the whole
        # branch on the resolved parameter state, not the fitthermal
        # flags -- the manifest "overrides" pin is layered UNDER user
        # params, so params.yaml can free or set thermal without
        # fitthermal. When every element is pinned at exactly 0 the
        # thermal math (a second quad_solution_vector per planet, as
        # expensive as the transit itself) is skipped entirely, matching
        # exofast_tran.pro's `if thermal ne 0d0` runtime gate.
        thermal_active = band.thermal_may_be_nonzero()
        # BEER (PR 1.b): reflection is smeared on the same sub-exposure
        # grid as thermal (see the group loop below) and shares the same
        # resolved-state gate. Ellipsoidal is smooth on-orbit and is not
        # smeared -- it stays a flat per-observation array, applied once
        # after the group loop.
        reflect_active = band.reflect_may_be_nonzero()
        ellip_mapped = band.ellipsoidal.value[self.obs_band_map_tensor]

        # 3b. SED deblending (EXOFASTv2 parity): with more than one
        # modeled star, only the host contributes the transit, so the
        # observed depth is diluted by dil = F_host / sum_j F_j in the
        # instrument's band (host = the band's star_ndx).
        dil_inst = None
        if hasattr(system, "sed") and system.star.n_elements > 1:
            dil_inst = self._build_dilution(system)
        # Flat per-observation dilution for the post-loop beam term below
        # (EXOFASTv2 parity -- see the beam comment past the group loop).
        dil_obs_flat = None
        if dil_inst is not None:
            dil_obs_flat = dil_inst[self.inst_map_tensor]  # (N_obs,)

        # 4. Exoplanet-core Transit Model, evaluated once per distinct
        # ninterp group (see _build_oversample_grid) instead of once for
        # the whole component at the largest ninterp: each group's sub-exposure
        # axis is exactly that group's own ninterp wide, so a
        # short-cadence (ninterp=1) observation is never evaluated at
        # another instrument's larger ninterp. With ninterp==1 everywhere
        # there is exactly one group of width 1, identical to the
        # original (pre-oversampling) computation.
        planet_group_decrement = [[] for _ in range(planets.n_elements)]
        for rows, time_grid_np, weights_np in self._oversample_groups:
            t_grid = pt.constant(time_grid_np)[:, :, None]  # (n_g, k_g, 1)
            w_g = pt.constant(weights_np)  # (k_g,)
            time_g = t_grid[:, :, 0]  # (n_g, k_g), for calc_reflect_term

            M = (t_grid - tp) * n
            sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

            sin_wf = sinw * cosf + cosw * sinf
            cos_wf = cosw * cosf - sinw * sinf

            # (n_g, k_g, N_planets)
            b = pt.sqrt(
                pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i)
            )
            Z = r_norm * sin_wf * sin_i

            u1_mapped = band.u1.value[self.obs_band_map[rows]]  # (n_g,)
            if "u2" in band.manifest:
                u2_mapped = band.u2.value[self.obs_band_map[rows]]  # (n_g,)
            else:
                u2_mapped = pt.zeros_like(u1_mapped)

            thermal_g = None
            if thermal_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                thermal_g = band.thermal.value[self.obs_band_map[rows]]

            reflect_g = None
            if reflect_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                reflect_g = band.reflect.value[self.obs_band_map[rows]]

            dil_obs = None
            if dil_inst is not None:
                dil_obs = dil_inst[self.inst_map[rows]]  # (n_g,)

            for p_idx in range(planets.n_elements):
                b_p = b[
                    :, :, p_idx
                ]  # (n_g, k_g) sky-plane separation in units of R_*
                Z_p = Z[
                    :, :, p_idx
                ]  # (n_g, k_g) line-of-sight coord (+ = planet in front of star)
                r_p = planets.p.value[p_idx]  # scalar R_p/R_*

                # Limb-darkened flux fraction: 1.0 off-disk, <1.0 during transit.
                # u1/u2_mapped are (n_g,); add the sub-exposure axis so they
                # broadcast against b_p.
                flux_frac = quad_limb_darkened_flux(
                    b_p, r_p, u1_mapped[:, None], u2_mapped[:, None]
                )  # (n_g, k_g)

                # Fraction of stellar flux blocked (0 off-disk, ~ r^2 at disk centre)
                blocked = 1.0 - flux_frac

                # Primary transit only; secondary eclipse (planet behind star) has Z < 0
                blocked = pt.where(Z_p > 0.0, blocked, 0.0)

                # Secondary eclipse / constant thermal emission (fitthermal)
                # and BEER (PR 1.b) reflection. Both come off the planet's
                # disk and share planetvisible; both live inside this group
                # loop so they get exposure-smeared on the same sub-exposure
                # grid as the transit (EXOFASTv2 averages the full model,
                # thermal included). Beam/ellipsoidal are smooth on-orbit
                # and are handled separately, un-smeared, after this loop.
                net = blocked
                if thermal_g is not None or reflect_g is not None:
                    visible = physics.calc_planet_visible(
                        b_p, Z_p, r_p
                    )  # (n_g, k_g)
                    if thermal_g is not None:
                        net = net - 1e-6 * thermal_g[:, None] * visible
                    if reflect_g is not None:
                        reflect_term_g = physics.calc_reflect_term(
                            time_g,
                            tc_p[p_idx],
                            period_p[p_idx],
                            reflect_g[:, None],
                            visible,
                        )
                        net = net - reflect_term_g

                if dil_obs is not None:
                    # One blended-aperture dilution for the whole
                    # perturbation from unity: EXOFASTv2's
                    # f0*(model*(1-dilute)+dilute) scales the transit dip
                    # and the planet's extra flux by the same factor.
                    net = net * dil_obs[:, None]

                # Weighted mean over this group's own ninterp sub-samples
                # (weights sum to 1). With ninterp==1 this is a no-op
                # identity (single column, weight 1).
                net_avg_g = pt.sum(net * w_g[None, :], axis=1)  # (n_g,)
                planet_group_decrement[p_idx].append(net_avg_g)

        for p_idx in range(planets.n_elements):
            # Groups were visited in np.unique(ninterp) order, not row
            # order; _oversample_inverse_order restores the original
            # per-observation order after concatenation. This decrement
            # already includes the transit, thermal, and (BEER, PR 1.b)
            # reflection terms, all computed and exposure-smeared in the
            # group loop above.
            net_avg = pt.concatenate(planet_group_decrement[p_idx])[
                self._oversample_inverse_order
            ]
            lc_model = lc_model - net_avg

            tc_this = tc_p[p_idx]  # scalar, this planet's time of conjunction
            period_this = period_p[p_idx]

            # Beaming is diluted the same way thermal/reflect are above --
            # EXOFASTv2 parity: exofast_chi2v2.pro:1517/1556 pass both beam
            # and dilute into exofast_tran, which adds beam at
            # exofast_tran.pro:146 and applies the dilution scaling at
            # exofast_tran.pro:157 (after beam, so beam is diluted too).
            # Ellipsoidal is NOT diluted -- it's a multiplicative factor on
            # the running lc_model (baseline + this planet's transit/
            # eclipse/thermal/reflect/beam so far), not an additive flux
            # term, so the dilution scaling doesn't apply to it the same
            # way. Neither term is gated by planetvisible: both are stellar
            # effects (the star's own RV motion / tidal shape), present
            # regardless of the planet's occultation state. Both are smooth
            # on-orbit and are deliberately NOT exposure-smeared -- they are
            # evaluated at the flat per-observation time, unlike thermal/
            # reflect above.
            beam_p = planets.beam.value[p_idx]  # scalar, ppm
            beam_term = physics.calc_beam_term(
                time, tc_this, period_this, beam_p
            )
            if dil_obs_flat is not None:
                beam_term = beam_term * dil_obs_flat
            lc_model = lc_model + beam_term

            # Ellipsoidal is multiplicative (exofast_tran.pro:143), applied
            # to the running lc_model (baseline + this planet's transit/
            # eclipse/thermal/reflect/beam so far). With >1 planet sharing
            # a band, each planet's factor multiplies in turn -- order-
            # dependent for N>1, exact for the single-planet case this PR
            # targets.
            ellip_factor = physics.calc_ellipsoidal_factor(
                time, tc_this, period_this, ellip_mapped
            )
            lc_model = lc_model * ellip_factor

        if self.total_detrend_cols > 0:
            detrend = pm.Data("transit_detrend", self.detrend_matrix)
            lc_model += pt.dot(detrend, self.detrend_coeffs.value)

        # Full per-observation model prediction (baseline + detrend +
        # exposure-averaged transit decrement). Kept as a plain attribute,
        # not a Deterministic: at (N_obs,) this would add N_obs * draws *
        # chains floats to every trace (tens of thousands x the size of the
        # other diagnostics here, which are all (N_planets,)). Tests compile
        # a one-off pytensor.function from this node directly instead.
        self._model_flux_node = lc_model

        # 5. Likelihood (shared base helper: sqrt(err^2 + jitter_variance)).
        # add_observation_likelihood is the plain Normal unless a light curve
        # asked for a GP, in which case that curve gets a celerite2 marginal
        # likelihood around this same transit model.
        sigma = self.total_sigma(err)
        self.add_observation_likelihood(
            "transit_likelihood", mu=lc_model, sigma=sigma, observed=flux
        )

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions for generating plotting lightcurves."""
        t_input = pt.vector("lc_t_input")
        inst_idx = pt.iscalar("lc_inst_idx")

        param_symbols = [p.value for p in system.plot_params]
        planets = getattr(system, "planet", None)
        orbits = getattr(system, "orbit", None)

        if planets is not None and orbits is not None:
            t_grid = t_input[:, None]
            tp = orbits.tp.value[planets.orbit_map][None, :]
            n = orbits.n.value[planets.orbit_map][None, :]
            ecc = orbits.ecc.value[planets.orbit_map][None, :]
            cosw = orbits.cosw.value[planets.orbit_map][None, :]
            sinw = orbits.sinw.value[planets.orbit_map][None, :]
            inc = orbits.inc.value[planets.orbit_map][None, :]

            M = (t_grid - tp) * n
            sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

            a_rstar = planets.ar.value[None, :]
            p_ratio = planets.p.value[None, :]
            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

            sin_wf = sinw * cosf + cosw * sinf
            cos_wf = cosw * cosf - sinw * sinf
            sin_i = pt.sin(inc)
            cos_i = pt.cos(inc)

            b = pt.sqrt(
                pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i)
            )
            Z = r_norm * sin_wf * sin_i

            band = system.band
            band_idx = self.band_map_tensor[inst_idx]
            u1_inst = band.u1.value[band_idx]  # scalar for this instrument
            if "u2" in band.manifest:
                u2_inst = band.u2.value[band_idx]
            else:
                u2_inst = pt.zeros_like(u1_inst)
            # Same resolved-state gates as build_likelihood: no thermal or
            # reflect graph at all when every band's value is pinned at 0.
            thermal_inst = None
            if band.thermal_may_be_nonzero():
                thermal_inst = band.thermal.value[band_idx]  # scalar ppm
            reflect_inst = None
            if band.reflect_may_be_nonzero():
                reflect_inst = band.reflect.value[band_idx]  # scalar ppm
            ellip_inst = band.ellipsoidal.value[
                band_idx
            ]  # scalar, 0 unless fitellip
            baseline_inst = self.baseline.value[inst_idx]  # scalar

            decrement_matrix_list = []
            for p_idx in range(planets.n_elements):
                b_p = b[:, p_idx]  # (N_times,)
                Z_p = Z[:, p_idx]
                r_p = planets.p.value[p_idx]
                tc_this = orbits.tc.value[planets.orbit_map][p_idx]
                period_this = orbits.period.value[planets.orbit_map][p_idx]

                flux_frac = quad_limb_darkened_flux(
                    b_p, r_p, u1_inst, u2_inst
                )  # (N_times,)
                # Negative so that _compiled_full_lc output + baseline gives a transit dip
                blocked = pt.where(Z_p > 0.0, 1.0 - flux_frac, 0.0)
                # match the likelihood's SED depth dilution (built there first)
                dil_node = getattr(self, "_dilution_node", None)
                if dil_node is not None:
                    blocked = blocked * dil_node[inst_idx]

                # Secondary eclipse / constant thermal emission + reflection
                # -- same shared helpers build_likelihood uses (physics.py),
                # same resolved-state gates. Both are pre-dilution terms,
                # like the transit depth above.
                additive_term = pt.zeros_like(b_p)
                if thermal_inst is not None or reflect_inst is not None:
                    planetvisible = physics.calc_planet_visible(b_p, Z_p, r_p)
                    if reflect_inst is not None:
                        reflect_term = physics.calc_reflect_term(
                            t_input,
                            tc_this,
                            period_this,
                            reflect_inst,
                            planetvisible,
                        )
                        additive_term = additive_term + reflect_term
                    if thermal_inst is not None:
                        thermal_term = 1e-6 * thermal_inst * planetvisible
                        additive_term = additive_term + thermal_term
                if dil_node is not None:
                    additive_term = additive_term * dil_node[inst_idx]

                # Beaming is diluted like thermal/reflect above (EXOFASTv2
                # parity: exofast_chi2v2.pro:1517/1556, exofast_tran.pro:157
                # -- see build_likelihood). Not gated by planetvisible --
                # same placement as build_likelihood.
                beam_p = planets.beam.value[p_idx]
                beam_term = physics.calc_beam_term(
                    t_input, tc_this, period_this, beam_p
                )
                if dil_node is not None:
                    beam_term = beam_term * dil_node[inst_idx]

                # Ellipsoidal is multiplicative, applied to the running
                # total *including baseline* (exofast_tran.pro:143). Since
                # this function's contract is "decrement from baseline"
                # (baseline is added back separately by callers -- see
                # _eval_unphased_lc), fold baseline in locally so the
                # multiplication is exact, then subtract it back out:
                #   decrement = (baseline + additive)*ellip_factor - baseline
                # Reduces to the plain additive decrement when ellip_factor
                # == 1 (fitellip off). Only exact for a single planet per
                # band; with >1 planet sharing a band, each gets its own
                # fold-in, same simplification noted in build_likelihood.
                ellip_factor = physics.calc_ellipsoidal_factor(
                    t_input, tc_this, period_this, ellip_inst
                )
                planet_decrement = -blocked + additive_term + beam_term
                decrement_matrix_list.append(
                    (baseline_inst + planet_decrement) * ellip_factor
                    - baseline_inst
                )

            lc_matrix = pt.stack(
                decrement_matrix_list, axis=1
            )  # (N_times, N_planets)
            lc_full_node = pt.sum(lc_matrix, axis=1)

            # Retain the symbolic nodes and their non-param inputs so
            # plot_data can derive param_deps (graph walk) and hand G5 the
            # symbolic tensors behind the model traces. Unused by plot().
            self._lc_t_input = t_input
            self._lc_inst_idx = inst_idx
            self._lc_matrix_node = lc_matrix
            self._lc_full_node = lc_full_node

            self._compiled_full_lc = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_full_node,
                on_unused_input="ignore",
            )
            self._compiled_lc_matrix = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_matrix,
                on_unused_input="ignore",
            )

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        self._compile_gp_plotters(system)

    def _oversample_offsets(self, inst_idx):
        """Sub-exposure time offsets (days) and averaging weights for this
        instrument's own ninterp/exptime -- the same midpoint-Riemann grid
        across [-exptime/2, +exptime/2] that _build_oversample_grid uses
        for build_likelihood, so a plot reproduces the smeared model the
        fit actually optimized against rather than the instantaneous one."""
        ninterp = int(self.ninterp[inst_idx])
        if ninterp <= 1:
            return np.zeros(1), np.ones(1)
        exptime_days = float(self.exptime_min[inst_idx]) / 1440.0
        j = np.arange(ninterp)
        frac = (j + 0.5) / ninterp - 0.5
        return frac * exptime_days, np.full(ninterp, 1.0 / ninterp)

    def _smeared_full_lc(self, t, inst_idx, *param_values):
        """Exposure-smeared counterpart of _compiled_full_lc: averages that
        same compiled (instantaneous) function over this instrument's own
        sub-exposure offsets, matching build_likelihood's oversampling. A
        ninterp=1 instrument short-circuits to the plain instantaneous call."""
        offsets, weights = self._oversample_offsets(inst_idx)
        if len(offsets) == 1:
            return self._compiled_full_lc(t, inst_idx, *param_values)
        acc = np.zeros_like(t, dtype=float)
        for off, w in zip(offsets, weights):
            acc += w * self._compiled_full_lc(t + off, inst_idx, *param_values)
        return acc

    def _smeared_lc_matrix(self, t, inst_idx, *param_values):
        """Exposure-smeared counterpart of _compiled_lc_matrix (per-planet
        decrement columns); see _smeared_full_lc."""
        offsets, weights = self._oversample_offsets(inst_idx)
        if len(offsets) == 1:
            return self._compiled_lc_matrix(t, inst_idx, *param_values)
        acc = None
        for off, w in zip(offsets, weights):
            contrib = w * self._compiled_lc_matrix(
                t + off, inst_idx, *param_values
            )
            acc = contrib if acc is None else acc + contrib
        return acc

    # ------------------------------------------------------------------
    # Shared data preparation. The matplotlib plot() path and the GUI
    # plot_data() path both go through these helpers, so the two paths
    # always draw the exact same arrays (see plotspec.PlotSpec).
    # ------------------------------------------------------------------
    def _baseline_for(self, point, i):
        """Baseline flux for instrument i, in internal units.

        The value comes from the point when it is there, else from the
        baseline Parameter's own initval -- the same fallback
        _point_to_plot_params uses for every other plotted parameter.

        A ``point.get(label, 1.0)`` here silently substituted UNITY for any
        parameter absent from the draws, and pinned (``sigma: 0``)
        parameters are always absent (an all-fixed vector never becomes a
        pm.Deterministic, so it is in neither model.deterministics nor the
        posterior).  Unity is not a neutral default: load_data seeds each
        baseline with the light curve's own median flux, so on an
        un-normalized light curve (raw counts) a pinned baseline plotted
        the model curve and the phased panel's cleaned flux off by the
        entire flux scale.
        """
        vals = point.get(self.baseline.label)
        if vals is None:
            vals = self.baseline.initval
        base_vals = np.atleast_1d(vals)
        return float(base_vals[i] if i < len(base_vals) else base_vals[0])

    def _eval_unphased_lc(self, system, point, i):
        """Full model light curve for instrument i: baseline + transit + GP.

        The unphased panel shows the model the likelihood actually fits, so
        any GP this light curve requested is included; the phased panels take
        it back out of the data instead (see _phased_lc_arrays). The GP term
        is zero for a light curve without a gp: key. The transit decrement
        goes through _smeared_full_lc (not _compiled_full_lc directly) so
        this shared panel reflects the same exposure-time smearing
        build_likelihood fit against, not the instantaneous model.
        """
        mask = self.inst_map == i
        t_data = self.time[mask]
        t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(
            np.float64
        )
        param_values = self._point_to_plot_params(point, system)
        y_decrement = self._smeared_full_lc(t_pretty, i, *param_values)
        y_gp = self.gp_mean_on_grid(system, point, i, t_pretty)
        return t_pretty, self._baseline_for(point, i) + y_decrement + y_gp

    def _phased_lc_arrays(self, system, point, p_idx, i):
        """
        One-period phase grid, isolated model decrement for planet p_idx,
        and the baseline-subtracted, other-planet-cleaned flux at the
        observed times -- used by plot_data() (and via it plot()). Uses
        _smeared_lc_matrix (see _eval_unphased_lc) so the phased panel
        matches the exposure-smeared model as well.
        """
        planets = system.planet
        P_ref = float(
            np.atleast_1d(point.get(system.orbit.period.label))[p_idx]
        )
        tc_ref = float(np.atleast_1d(point.get(system.orbit.tc.label))[p_idx])

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000
        ).astype(np.float64)
        phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
        time_from_center_model = phase_model * P_ref
        sort_m = np.argsort(phase_model)

        param_values = self._point_to_plot_params(point, system)
        lc_matrix = self._smeared_lc_matrix(t_model, i, *param_values)
        y_planet = lc_matrix[:, p_idx]

        mask = self.inst_map == i
        data_lc_matrix = self._smeared_lc_matrix(
            self.time[mask], i, *param_values
        )
        other_mask = np.ones(planets.n_elements, dtype=bool)
        other_mask[p_idx] = False
        other_decrements = np.sum(data_lc_matrix[:, other_mask], axis=1)

        baseline = self._baseline_for(point, i)
        # Remove the correlated component along with the other planets', so
        # the phased panel is not smeared by it. Zero without a gp: key.
        gp_signal = self.gp_mean_at_data(system, point)[mask]
        cleaned_flux = (
            self.flux[mask] - baseline - other_decrements - gp_signal
        )
        data_phases = ((self.time[mask] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "x_model": time_from_center_model[sort_m],
            "y_model": y_planet[sort_m],
            "x_data": data_phases * P_ref,
            "y_data": cleaned_flux,
        }

    def plot(self, system, points, filename_prefix="debug"):
        """Render the unphased + phased LC PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)

    def plot_data(self, system, point=None):
        """
        GUI plot specs for the transit photometry: per instrument an
        unphased flux-vs-time chart, and (with a point) one phased chart
        per planet/instrument. point=None returns only the raw data
        traces. See Component.plot_data and plotspec.PlotSpec.
        """
        from exozippy.plotspec import PlotSpec, Trace

        specs = []
        full_deps = self._model_trace_param_deps(
            getattr(self, "_lc_full_node", None), system
        )
        matrix_deps = self._model_trace_param_deps(
            getattr(self, "_lc_matrix_node", None), system
        )

        # The baseline enters both panels in numpy (_baseline_for), not
        # through the symbolic nodes, so the graph walk cannot see it --
        # without this dep a baseline slider would never refresh these
        # charts in the GUI.
        baseline_label = getattr(
            getattr(self, "baseline", None), "label", None
        )
        if baseline_label:
            if baseline_label not in full_deps:
                full_deps = full_deps + [baseline_label]
            if baseline_label not in matrix_deps:
                matrix_deps = matrix_deps + [baseline_label]

        # ---- Unphased: flux vs time, per instrument -------------------
        for i in range(self.n_elements):
            mask = self.inst_map == i
            traces = []
            deps = []
            if point is not None:
                # A failed model eval keeps the data-only panel (matching the
                # per-point tolerance of the old hand-drawn loop).
                try:
                    t_pretty, y_full = self._eval_unphased_lc(system, point, i)
                    deps = full_deps
                    traces.append(
                        Trace(
                            name="model",
                            role="model",
                            kind="line",
                            x=t_pretty,
                            y=y_full,
                            node=getattr(self, "_lc_full_node", None),
                        )
                    )
                except Exception as e:  # noqa: BLE001 - bad point/draw
                    logger.warning(f"LC model eval failed: {e}")
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=self.flux[mask],
                    yerr=self.err[mask],
                    # Black-dot default (the historical PDF look); a user
                    # plot: color/marker still wins via _data_trace_style.
                    style={
                        "color": "k",
                        "marker": ".",
                        **self._data_trace_style(i),
                    },
                )
            )
            specs.append(
                PlotSpec(
                    id=f"{self.prefix}.unphased.{self.names[i]}",
                    component={
                        "yaml_key": self.prefix,
                        "instance": self.names[i],
                    },
                    title=f"Transit Photometry: {self.names[i]}",
                    xlabel="Time [BJD]",
                    ylabel="Relative Flux",
                    traces=traces,
                    param_deps=deps,
                    meta={
                        "phase_folded": False,
                        "instrument": self.names[i],
                        "file_tag": f"LC_unphased_{self.names[i]}",
                        "figsize": (12, 5),
                    },
                )
            )

        # ---- Phased: one chart per planet/instrument (needs a model) --
        if point is not None:
            planets = system.planet
            for p_idx in range(planets.n_elements):
                for i in range(self.n_elements):
                    # A failed prep skips this panel, exactly as the old
                    # hand-drawn loop skipped its figure.
                    try:
                        prep = self._phased_lc_arrays(system, point, p_idx, i)
                    except Exception as e:  # noqa: BLE001 - bad point/draw
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue
                    mask = self.inst_map == i
                    traces = [
                        Trace(
                            name="model",
                            role="model",
                            kind="line",
                            x=prep["x_model"],
                            y=prep["y_model"],
                            node=getattr(self, "_lc_matrix_node", None),
                            style={"lw": 1.0},
                        ),
                        Trace(
                            name=self.names[i],
                            role="data",
                            kind="scatter",
                            x=prep["x_data"],
                            y=prep["y_data"],
                            yerr=self.err[mask],
                            style={
                                "color": "k",
                                "marker": ".",
                                **self._data_trace_style(i),
                            },
                        ),
                    ]
                    pname = planets.names[p_idx]
                    meta = {
                        "phase_folded": True,
                        "planet": pname,
                        "instrument": self.names[i],
                        "period": prep["P_ref"],
                        "tc": prep["tc_ref"],
                        "file_tag": (f"LC_phased_{self.names[i]}_{pname}"),
                        "figsize": (10, 6),
                        "hline_y": 0.0,
                        # The phased DATA re-folds with tc/P and its cleaning
                        # subtracts the baseline, other planets and any GP --
                        # all point-dependent, so live evals must re-ship it.
                        "dynamic_data": True,
                    }
                    # Zoom to +/- t14 around mid-transit when the point
                    # carries a transit duration for this planet.
                    t14_raw = point.get(f"{self.prefix}.t14")
                    if t14_raw is not None:
                        t14_ref = float(np.atleast_1d(t14_raw)[p_idx])
                        meta["x_range"] = [-t14_ref, t14_ref]
                    specs.append(
                        PlotSpec(
                            id=f"{self.prefix}.phased.{self.names[i]}.{pname}",
                            component={
                                "yaml_key": self.prefix,
                                "instance": self.names[i],
                            },
                            title=f"Phased LC: {pname} -- {self.names[i]}",
                            xlabel=f"Time from Mid-Transit [d] (P = {prep['P_ref']:.5f} d)",
                            ylabel="Flux - Baseline",
                            traces=traces,
                            param_deps=matrix_deps,
                            meta=meta,
                        )
                    )

        return specs
