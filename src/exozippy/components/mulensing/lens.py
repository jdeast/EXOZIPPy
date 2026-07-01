import logging
import pytensor.tensor as pt
import pymc as pm
import numpy as np

from exozippy.components.component import Component
from exozippy.constants import KAPPA
from exozippy.potentials import soft_lower_bound
from .op import MulensMagOp, BinaryLensMagOp

logger = logging.getLogger(__name__)


def _parse_body_ref(ref):
    """Parse 'star.0' → ('star', 0), 'planet.1' → ('planet', 1)."""
    parts = str(ref).split(".")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Invalid body reference '{ref}': expected '<component>.<index>', "
            f"e.g. 'star.0' or 'planet.0'."
        )
    return (parts[0], int(parts[1]))


class Lens(Component):
    """Microlensing lens component.

    Supports PSPL (1 lens body) and binary lens (2 lens bodies via planet or star).
    Bodies are specified in the YAML config as:
        lenses:  ["star.0", "planet.0"]   # 2-body binary
        sources: ["star.1"]

    Backward-compatible shorthand (single-star PSPL):
        lens_ndx:   0
        source_ndx: 1
    """

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"

        # One event at a time: a single (t_0, u_0, t_E, ...) geometry. Multiple
        # lenses, sources, or instruments all belong to that one event.
        if self.n_elements > 1:
            raise ValueError(
                "Only one lensing event may be modeled at a time. Define a single "
                "lens block and list all bodies in 'lenses'/'sources' "
                "(e.g. lenses: ['star.0', 'planet.0', 'planet.1'])."
            )

        # Parse lens / source body lists per event
        self.lens_bodies = []    # list of lists of (comp_type, ndx) per event
        self.source_bodies = []  # list of lists of (comp_type, ndx) per event

        for c in self.config:
            if "lenses" in c:
                lb = [_parse_body_ref(r) for r in c["lenses"]]
            else:
                lb = [("star", int(c.get("lens_ndx", 0)))]

            if "sources" in c:
                sb = [_parse_body_ref(r) for r in c["sources"]]
            else:
                sb = [("star", int(c.get("source_ndx", 1)))]

            self.lens_bodies.append(lb)
            self.source_bodies.append(sb)

        self.n_lens_bodies = [len(b) for b in self.lens_bodies]
        self.n_source_bodies = [len(b) for b in self.source_bodies]

        # Companions: every lens body beyond the primary. Each carries its own
        # separation s and trajectory angle alpha; mass ratios come from the
        # bodies' masses.
        self.n_companions = self.n_lens_bodies[0] - 1

        if any(n > 1 for n in self.n_source_bodies):
            raise NotImplementedError("Binary source is not yet supported.")

        # Convenience maps: primary lens and source (index 0 of each list)
        self.finite_source = [c.get("finite_source", False) for c in self.config]
        self.t0_par = [self._resolve_t0_par(i, c, config_manager)
                       for i, c in enumerate(self.config)]

        self.mag_method = [
            c.get("mag_method",
                  "auto_vbbl" if (c.get("finite_source", False)
                                  or self.n_lens_bodies[i] > 1)
                  else "point_source")
            for i, c in enumerate(self.config)
        ]

        # use_op: force the MulensModel Op even for point-source PSPL.
        # Default False for PSPL (symbolic is NUTS-friendly); True forces the Op
        # (useful for testing or when MulensModel's parallax handling is needed).
        self.use_op = [c.get("use_op", False) for c in self.config]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_t0_par(i, c, config_manager):
        if 't0_par' in c:
            return float(c['t0_par'])
        entry = config_manager.user_params.get(f"lens.{i}.t_0")
        if isinstance(entry, dict):
            val = entry.get('initval')
        else:
            val = entry
        return float(val) if val is not None else 2450000.0

    @property
    def prefix(self):
        return "lens"

    def _primary_lens(self, event_idx):
        """Return (comp_type, star_ndx) for the primary lens of event i."""
        return self.lens_bodies[event_idx][0]

    def _primary_source(self, event_idx):
        """Return (comp_type, star_ndx) for the primary source of event i."""
        return self.source_bodies[event_idx][0]

    def _body_mass(self, system, comp_type, ndx):
        """Return the PyTensor mass node for a given body."""
        return getattr(system, comp_type).mass.value[ndx]

    def _validate_bodies(self, system):
        """Fail at registration time if a body reference points to a component
        or instance that does not exist (instead of an AttributeError deep in
        the model build)."""
        for i in range(self.n_elements):
            for role, bodies in (("lens", self.lens_bodies[i]),
                                 ("source", self.source_bodies[i])):
                for comp_type, ndx in bodies:
                    comp = getattr(system, comp_type, None)
                    if comp is None:
                        raise ValueError(
                            f"lens.{i}: {role} body '{comp_type}.{ndx}' refers to "
                            f"component '{comp_type}', but no '{comp_type}' block "
                            f"exists in the config."
                        )
                    if ndx >= comp.n_elements:
                        raise ValueError(
                            f"lens.{i}: {role} body '{comp_type}.{ndx}' is out of "
                            f"range: only {comp.n_elements} '{comp_type}' "
                            f"instance(s) are configured."
                        )

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 1b: Build integer index arrays for primary lens and source."""
        _, l_ndxs = zip(*[self._primary_lens(i) for i in range(self.n_elements)])
        _, s_ndxs = zip(*[self._primary_source(i) for i in range(self.n_elements)])
        self.lens_map = np.array(l_ndxs, dtype=int)
        self.source_map = np.array(s_ndxs, dtype=int)

        if self.n_companions >= 1:
            # Scalar maps (length-1) so the bracket-slice dep yields a scalar
            # mass rather than a full-component mass array.
            _, p_ndx = self.lens_bodies[0][0]
            self.primary_lens_map = np.array([p_ndx], dtype=int)
            _, c_ndx = self.lens_bodies[0][1]
            self.companion_mass_map = np.array([c_ndx], dtype=int)

    def register_parameters(self, system):
        """Stage 2: Declare the manifest."""
        self._validate_bodies(system)

        self.manifest = {
            "t_0": None, "u_0": None,
            "pi_rel": "default", "theta_E": "default",
            "mu_ra_rel": "default", "mu_dec_rel": "default",
            "mu_rel_mag": "default", "t_E": "default",
            "pi_E_N": "default", "pi_E_E": "default",
        }

        # Companion geometry: one (s, alpha) pair per lens body beyond the
        # primary. The shape override sizes these by companion count rather
        # than by component element count.
        if self.n_companions >= 1:
            companion_shape = (self.n_companions,)
            self.manifest["s"] = {"shape": companion_shape}
            self.manifest["cosalpha"] = {"shape": companion_shape}
            self.manifest["sinalpha"] = {"shape": companion_shape}
            # alpha derived from cosalpha/sinalpha; internal unit = rad, display = deg
            self.manifest["alpha"] = {"expr_key": "default", "shape": companion_shape}
            # q = M_companion / M_primary; companion component type varies by config
            companion_type = self.lens_bodies[0][1][0]
            self.manifest["q"] = {
                "expr_key": "default",
                "shape": companion_shape,
                "deps": [f"{companion_type}.mass[companion_mass_map]",
                         "star.mass[primary_lens_map]"],
            }

        if any(self.finite_source):
            self.manifest["rho"] = "default"

        # Seed alpha hint (degrees, user unit) so inspect_start can display it
        # even before the expression graph is built.
        inst = self.names[0] if self.names else "0"
        ca_entry = (self.config_manager.user_params.get(f"lens.{inst}.cosalpha")
                    or self.config_manager.user_params.get(f"lens.0.cosalpha") or {})
        sa_entry = (self.config_manager.user_params.get(f"lens.{inst}.sinalpha")
                    or self.config_manager.user_params.get(f"lens.0.sinalpha") or {})
        ca = ca_entry.get("initval")
        sa = sa_entry.get("initval")
        if ca is not None and sa is not None:
            alpha_deg = float(np.arctan2(float(sa), float(ca)) * 180.0 / np.pi)
            self.config_manager.add_hint(f"lens.0.alpha", alpha_deg, rank=20)

        # Inject per-event physical hints
        for i in range(self.n_elements):
            l_type, l_idx = self._primary_lens(i)
            s_type, s_idx = self._primary_source(i)

            # Rank 25: overrides the 10 pc defaults.yaml default (rank 20) but yields
            # to any value the relaxation engine derives from pi_rel+d_S (rank 30).
            # This breaks the d_L↔parallax cycle: pi_rel drives d_L to rank 30 via
            # Condition B, then parallax (rank 25) is corrected as the weaker symbol.
            self.config_manager.add_hint(f"star.{l_idx}.distance", 4000.0, rank=25)
            self.config_manager.add_scale_hint(f"star.{l_idx}.distance", 5.0)
            self.config_manager.add_hint(f"star.{l_idx}.logmass", -0.5)
            self.config_manager.add_scale_hint(f"star.{l_idx}.logmass", 0.001)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_ra", 3.0)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_dec", 3.0)
            self.config_manager.add_scale_hint(f"star.{l_idx}.rv", 1e5)

            self.config_manager.add_hint(f"star.{s_idx}.distance", 8000.0, rank=30)
            self.config_manager.add_scale_hint(f"star.{s_idx}.distance", 5.0)
            self.config_manager.add_hint(f"star.{s_idx}.logmass", -0.5)
            self.config_manager.add_scale_hint(f"star.{s_idx}.logmass", 0.3)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_ra", 3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_dec", 3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.rv", 1e5)

            # Companion lens bodies (everything beyond the primary)
            for l2_type, l2_idx in self.lens_bodies[i][1:]:
                if l2_type == "star":
                    self.config_manager.add_hint(f"star.{l2_idx}.distance", 4000.0, rank=25)
                    self.config_manager.add_scale_hint(f"star.{l2_idx}.distance", 5.0)

        # Tighten lens logmass scale when satellite parallax is available
        if hasattr(system, 'mulensinstrument') and hasattr(system.mulensinstrument, 'inst_ref_pos'):
            ref_pos = system.mulensinstrument.inst_ref_pos
            max_sep = max(
                (float(np.linalg.norm(ref_pos[ii] - ref_pos[jj]))
                 for ii in range(len(ref_pos)) for jj in range(ii + 1, len(ref_pos))),
                default=0.0
            )
            if max_sep > 0.5:
                scale = 0.0005
            elif max_sep > 1e-5:
                scale = 0.00075
            else:
                scale = None
            if scale is not None:
                for i in range(self.n_elements):
                    _, l_idx = self._primary_lens(i)
                    self.config_manager.add_scale_hint(f"star.{l_idx}.logmass", scale)

    def build_likelihood(self, model, system):
        """Stage 6: Observational penalties on the lensing geometry."""
        mu_rel = self.mu_rel_mag.value
        theta_E = self.theta_E.value

        pm.Potential(f"{self.prefix}.event_rate_prior",
                     pt.sum(pt.log(mu_rel) + pt.log(theta_E)))

        # Shared log-sigmoid barriers (see exozippy.potentials): smooth and
        # asymptotically linear, so the sampler feels a restoring gradient
        # instead of a -1e6 cliff. scale=440 pc preserves the previous ~1/pc
        # slope; scale=1e-5 puts the singularity turn-on at ~1e-7 (mas or
        # mas/yr), matching the previous steepness.
        d_l = system.star.distance.value[self.lens_map]
        d_s = system.star.distance.value[self.source_map]
        pm.Potential(f"{self.prefix}.source_behind_lens",
                     pt.sum(soft_lower_bound(d_s - d_l, 10.0, scale=440.0)))

        pm.Potential(f"{self.prefix}.mu_rel_singularity",
                     pt.sum(soft_lower_bound(mu_rel, 1e-6, scale=1e-5)))

        pm.Potential(f"{self.prefix}.theta_E_singularity",
                     pt.sum(soft_lower_bound(theta_E, 1e-6, scale=1e-5)))

    # ------------------------------------------------------------------
    # Magnification
    # ------------------------------------------------------------------

    def _get_safe_mm_params(self, index=0):
        tE_raw = self.t_E.value[index]
        u0_raw = self.u_0.value[index]
        theta_E_raw = self.theta_E.value[index]
        pi_N_raw = self.pi_E_N.value[index]
        pi_E_raw = self.pi_E_E.value[index]

        tE_scrubbed = pt.nan_to_num(tE_raw, nan=100.0)
        u0_scrubbed = pt.nan_to_num(u0_raw, nan=1.0)
        theta_E_scrubbed = pt.nan_to_num(theta_E_raw, nan=0.0)
        pi_N_scrubbed = pt.nan_to_num(pi_N_raw, nan=0.0)
        pi_E_scrubbed = pt.nan_to_num(pi_E_raw, nan=0.0)

        tE_safe = pt.maximum(tE_scrubbed, 1e-4)
        u0_safe = pt.sign(u0_scrubbed) * pt.maximum(pt.abs(u0_scrubbed), 1e-6)
        is_physical = pt.gt(theta_E_scrubbed, 1e-6)

        return {
            't0': self.t_0.value[index],
            'u0': u0_safe,
            'tE': tE_safe,
            'pi_N': pt.switch(is_physical, pi_N_scrubbed, 0.0),
            'pi_E': pt.switch(is_physical, pi_E_scrubbed, 0.0),
        }

    def _get_binary_mm_params(self, system, index=0):
        """Build corrected params for a binary lens event (total mass theta_E)."""
        s = self._get_safe_mm_params(index)

        l2_type, l2_idx = self.lens_bodies[index][1]
        m1 = self._body_mass(system, *self.lens_bodies[index][0])
        m2 = self._body_mass(system, l2_type, l2_idx)
        q = m2 / pt.maximum(m1, 1e-10)

        pi_rel = self.pi_rel.value[index]
        mu = self.mu_rel_mag.value[index]
        mu_safe = pt.maximum(pt.nan_to_num(mu, nan=0.0), 1e-10)

        theta_E_tot = pt.sqrt(KAPPA * (m1 + m2) * pi_rel)
        theta_E_tot_safe = pt.maximum(pt.nan_to_num(theta_E_tot, nan=0.0), 1e-10)
        is_physical = pt.gt(theta_E_tot_safe, 1e-6)

        t_E_tot = pt.maximum(theta_E_tot_safe / (mu_safe / 365.25), 1e-4)
        pi_E_N_tot = pt.switch(is_physical,
            pi_rel / theta_E_tot_safe * self.mu_dec_rel.value[index] / mu_safe, 0.0)
        pi_E_E_tot = pt.switch(is_physical,
            pi_rel / theta_E_tot_safe * self.mu_ra_rel.value[index] / mu_safe, 0.0)

        # s/cosalpha/sinalpha are indexed by companion (binary = companion 0),
        # not by event.
        alpha_deg = pt.arctan2(self.sinalpha.value[0],
                               self.cosalpha.value[0]) * (180.0 / np.pi)

        return {
            't0': s['t0'], 'u0': s['u0'],
            'tE': t_E_tot, 'pi_N': pi_E_N_tot, 'pi_E': pi_E_E_tot,
            's': self.s.value[0], 'q': q, 'alpha': alpha_deg,
        }

    def get_magnification(self, times, obs_pos_abs, system, index=0):
        """Symbolic Paczynski magnification including parallax (PSPL only).

        obs_pos_abs : (N, 3) absolute barycentric positions in AU — the same
        convention as MulensModel's satellite_skycoord, so this function and
        the MulensModel Op are interchangeable callers.

        Internally converts to Skowron+2011 geocentric deviations using the
        reference constants stored on the MulensInstrument.  When no
        instrument is present (e.g. unit tests with zero positions) the
        positions are treated as already-centered deviations (no parallax).
        """
        source_ndx = self.source_map[index]
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        instr = getattr(system, 'mulensinstrument', None)
        if instr is not None:
            # Convert absolute barycentric → Skowron+2011 geocentric deviations:
            #   delta(t) = xyz_obs(t) - [xyz_earth(t0_par) + v_earth(t0_par)*(t - t0_par)]
            t_delta = times - instr._t0_par          # (N,)
            ref = (instr._earth_pos_ref[None, :]     # (1, 3) constant
                   + instr._earth_vel_ref[None, :] * t_delta[:, None])  # (N, 3)
            obs_pos = obs_pos_abs - ref
        else:
            obs_pos = obs_pos_abs

        x, y, z = obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2]
        delta_e = -x * pt.sin(ra) + y * pt.cos(ra)
        delta_n = (-x * pt.cos(ra) * pt.sin(dec)
                   - y * pt.sin(ra) * pt.sin(dec)
                   + z * pt.cos(dec))

        p = self._get_safe_mm_params(index)
        # MulensModel convention: delta_tau = -delta_N*pi_E_N - delta_E*pi_E_E
        # (negative on both N and E, matching Skowron+2011 via MulensModel's
        # sign choice). MMEXOFAST calls MulensModel, so published pi_E values
        # are calibrated to this convention.
        tau_p = ((times - p['t0']) / p['tE']
                 - delta_n * p['pi_N'] - delta_e * p['pi_E'])
        u_p = p['u0'] + delta_n * p['pi_E'] - delta_e * p['pi_N']

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def uses_op(self, index=0):
        """Return True if get_magnification_op will dispatch to the MulensModel Op.

        Callers use this to decide which obs_pos convention to pass:
        - True  → absolute barycentric AU (MulensModel satellite_skycoord)
        - False → Skowron+2011 geocentric deviations (symbolic get_magnification)
        """
        n_lenses = self.n_lens_bodies[index]
        use_rho = self.finite_source[index]
        forced = self.use_op[index]
        return forced or (n_lenses > 1) or use_rho

    def sampler_requirements(self):
        """Declare sampler constraints for this lens configuration.

        Binary/finite-source lenses use the MulensModel Op, which is not
        differentiable.  Gradient-based samplers (NUTS, numpyro, blackjax)
        will produce invalid results; PTDE is required.

        PSPL lenses use a symbolic PyTensor formula and are NUTS-compatible,
        so no constraints are returned.
        """
        if any(self.uses_op(i) for i in range(len(self.n_lens_bodies))):
            return {
                'incompatible': {'nuts', 'numpyro', 'blackjax'},
                'recommended': 'ptde',
                'reason': (
                    "binary/finite-source microlensing uses the MulensModel Op, "
                    "which is not differentiable — gradient-based samplers produce "
                    "invalid results"
                ),
            }
        return {}

    def get_magnification_op(self, times, obs_pos, system, index=0, u1=None, bandpass=None):
        """Magnification dispatcher.

        For point-source PSPL (n_lenses==1, finite_source=False, use_op=False)
        falls back to the symbolic PyTensor formula so NUTS can differentiate
        through it without the O(N_params) numerical-gradient overhead of
        _MagGradOp.

        obs_pos convention is caller's responsibility and must match:
        - Symbolic path: Skowron+2011 geocentric deviations (AU)
        - Op path:       absolute barycentric positions (AU); the Op
          converts them to geocentric (satellite - earth_actual) before
          passing to MulensModel, which expects geocentric input.

        u1/bandpass: when finite_source is True and a Band component is wired,
        u1 (a PyTensor scalar) and bandpass (str) are passed so the Op can call
        set_limb_coeff_u and get_magnification(bandpass=...).  Passing neither
        falls back to uniform-source finite-source magnification.

        Set ``use_op: true`` in the lens YAML block to force the Op (e.g. for
        testing or when MulensModel's finite-source parallax is needed).
        """
        if self.n_lens_bodies[index] > 2:
            raise NotImplementedError(
                f"{self.n_lens_bodies[index]}-lens magnification is not yet "
                "available: the MulensModel backend supports at most 2 lens "
                "bodies. The N-lens parameters (per-companion s/alpha, "
                "per-body masses) are wired; a triple+ backend (e.g. "
                "VBMicrolensing) still needs to be integrated here."
            )

        if not self.uses_op(index):
            return self.get_magnification(times, obs_pos, system, index)

        source_ndx = self.source_map[index]
        ra_deg = float(system.star.ra.value[source_ndx].eval()) * (180.0 / np.pi)
        dec_deg = float(system.star.dec.value[source_ndx].eval()) * (180.0 / np.pi)
        coords = f"{ra_deg}d {dec_deg}d"

        use_rho = self.finite_source[index]
        n_lenses = self.n_lens_bodies[index]

        # Apply LD only for finite-source and when a band is connected.
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None

        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        if n_lenses == 2:
            bp = self._get_binary_mm_params(system, index)
            param_list = [bp['t0'], bp['u0'], bp['tE'], bp['pi_N'], bp['pi_E']]
            if use_rho:
                param_list.append(self.rho.value[index])
            param_list.extend([bp['s'], bp['q'], bp['alpha']])
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = BinaryLensMagOp(coords=coords, mag_method=self.mag_method[index],
                                     use_rho=use_rho, bandpass=effective_bandpass)
        else:
            sp = self._get_safe_mm_params(index)
            param_list = [sp['t0'], sp['u0'], sp['tE'], sp['pi_N'], sp['pi_E']]
            if use_rho:
                param_list.append(self.rho.value[index])
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = MulensMagOp(coords=coords, mag_method=self.mag_method[index],
                                 use_rho=use_rho, bandpass=effective_bandpass)

        return mag_op(pt.stack(param_list), times_tensor, obs_tensor)

    # ------------------------------------------------------------------
    # Auto method brackets
    # ------------------------------------------------------------------

    def _get_initval(self, param, event_idx=0):
        """Look up a resolved initval from config_manager, checking name and index forms."""
        cm = self.config_manager
        name = self.names[event_idx] if event_idx < len(self.names) else str(event_idx)
        for key in [f"lens.{name}.{param}", f"lens.{event_idx}.{param}"]:
            entry = cm.user_params.get(key)
            if entry is not None:
                val = entry.get("initval") if isinstance(entry, dict) else entry
                if val is not None:
                    return float(val)
        return None

    def resolve_auto_vbbl(self, times_np, index=0, threshold=0.001,
                          buffer=0.1, max_eval=2000):
        """Replace 'auto_vbbl' with a concrete bracket list computed at initvals.

        Runs hexadecapole and VBM on a time grid, finds intervals where they
        differ by more than `threshold` (fractional), pads each interval by
        `buffer` days, and stores the result in self.mag_method[index].

        `buffer` covers two effects: (1) the sharp transition zone where
        hexadecapole begins to fail (~rho*t_E wide, typically 0.05-0.2 d), and
        (2) gaps in the evaluation grid when data is subsampled (max_eval).
        0.1 d is appropriate for a well-initialised model; increase to 0.2-0.5 d
        if initvals are rough.

        Intentionally skips parallax — we only need approximate caustic-crossing
        timing for method selection, not a precise fit.
        """
        if self.mag_method[index] != "auto_vbbl":
            return

        import MulensModel as mm

        g = self._get_initval
        t_0   = g("t_0",   index)
        u_0   = g("u_0",   index)
        t_E   = g("t_E",   index)
        s_val = g("s",     index)
        q_val = g("q",     index)
        # alpha in degrees; try direct value first, fall back to cosalpha/sinalpha
        alpha_deg = g("alpha", index)
        if alpha_deg is None:
            ca = g("cosalpha", index)
            sa = g("sinalpha", index)
            if ca is not None and sa is not None:
                alpha_deg = float(np.degrees(np.arctan2(sa, ca)))

        if any(v is None for v in [t_0, u_0, t_E, s_val, q_val, alpha_deg]):
            logger.warning("auto_vbbl: missing initvals for bracket computation; "
                           "falling back to VBM everywhere.")
            self.mag_method[index] = [float(times_np.min()) - 1.0, "VBM",
                                      float(times_np.max()) + 1.0]
            return

        params = {
            "t_0":   t_0,
            "u_0":   max(abs(u_0), 1e-9) * (1 if u_0 >= 0 else -1),
            "t_E":   max(t_E, 1e-4),
            "s":     max(s_val, 1e-6),
            "q":     float(np.clip(q_val, 1e-9, 1.0)),
            "alpha": alpha_deg,
        }
        if self.finite_source[index]:
            rho = g("rho", index)
            if rho is not None:
                params["rho"] = max(rho, 1e-9)

        # Build eval grid: subsample if data is very dense
        if len(times_np) > max_eval:
            step = len(times_np) // max_eval
            eval_times = np.sort(times_np[::step])
        else:
            eval_times = np.sort(times_np)

        t_lo = float(eval_times[0])
        t_hi = float(eval_times[-1])

        try:
            model_hex = mm.Model(parameters=params)
            model_vbm = mm.Model(parameters=params)
            model_hex.set_magnification_methods([t_lo - 1.0, "hexadecapole", t_hi + 1.0])
            model_vbm.set_magnification_methods([t_lo - 1.0, "VBM",          t_hi + 1.0])

            A_hex = np.asarray(model_hex.get_magnification(eval_times))
            A_vbm = np.asarray(model_vbm.get_magnification(eval_times))

            diff = np.abs(A_hex - A_vbm) / np.maximum(np.abs(A_vbm), 1.0)
            needs_vbm = diff > threshold

        except Exception as e:
            logger.warning(f"auto_vbbl: bracket computation failed ({e}); "
                           "falling back to VBM everywhere.")
            self.mag_method[index] = [t_lo - 1.0, "VBM", t_hi + 1.0]
            return

        if not np.any(needs_vbm):
            self.mag_method[index] = [t_lo - 1.0, "hexadecapole", t_hi + 1.0]
            logger.info("auto_vbbl: hexadecapole sufficient everywhere "
                        f"(max diff {diff.max():.4f} < threshold {threshold})")
            return

        # Build contiguous VBM intervals with buffer
        brackets = []
        in_vbm = False
        for i, t in enumerate(eval_times):
            if needs_vbm[i] and not in_vbm:
                t_start = max(t_lo - 1.0, t - buffer)
                if brackets:
                    brackets.append(t_start)
                    brackets.append("VBM")
                else:
                    brackets = [t_lo - 1.0, "hexadecapole", t_start, "VBM"]
                in_vbm = True
            elif not needs_vbm[i] and in_vbm:
                t_end = min(t_hi + 1.0, t + buffer)
                brackets.append(t_end)
                brackets.append("hexadecapole")
                in_vbm = False

        if in_vbm:
            brackets.append(t_hi + 1.0)

        # Ensure the bracket list ends with a sentinel time
        if not brackets:
            brackets = [t_lo - 1.0, "hexadecapole", t_hi + 1.0]
        elif not isinstance(brackets[-1], float):
            brackets.append(t_hi + 1.0)

        n_vbm = sum(1 for b in brackets if b == "VBM")
        logger.info(f"auto_vbbl: {n_vbm} VBM interval(s) "
                    f"(threshold={threshold}, buffer={buffer} d, "
                    f"max diff={diff.max():.4f})")
        self.mag_method[index] = brackets

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
