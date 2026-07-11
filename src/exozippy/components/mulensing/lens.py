import logging
import pytensor.tensor as pt
import pymc as pm
import numpy as np

from exozippy.components.component import Component
from exozippy.constants import KAPPA
from exozippy.corner_utils import collect_parameter_corner_samples, save_corner_plot
from exozippy.potentials import soft_lower_bound
from .op import MulensMagOp, BinaryLensMagOp, VBMDirectMagOp

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

    Supports N sources and up to 2 lens bodies (NSNL; the MulensModel backend
    caps the lens side at binary for now).  Bodies are specified in the YAML
    config as:
        lenses:  ["star.0", "planet.0"]   # 2-body binary
        sources: ["star.1", "star.2"]     # binary source (2S)

    Each source follows its own trajectory: t_0, u_0, rho and the derived
    chain (t_E, theta_E, pi_rel, pi_E_*, mu_*) are vectors with one element
    per source, sharing the lens-side parameters (masses, s, alpha).  In the
    params file, address element j either by slot index (lens.1.t_0) or by
    the source star's instance name (lens.SourceB.t_0).

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

        # Sources: single event ⇒ one flat list of source bodies; per-source
        # parameters (t_0, u_0, rho, ...) are vectors of this length.
        self.n_sources = self.n_source_bodies[0]

        # Translate lens.<SourceStarName>.<param> user keys to the canonical
        # slot-index form lens.<j>.<param> so resolve() and the relaxation
        # engine see one naming scheme.  Must happen before any stage-1 code
        # (e.g. MulensInstrument.load_data) reads user_params.
        self._rewrite_source_param_keys(config_manager)

        # Convenience maps: primary lens and source (index 0 of each list)
        self.finite_source = [c.get("finite_source", False) for c in self.config]
        self.t0_par = [self._resolve_t0_par(i, c, config_manager)
                       for i, c in enumerate(self.config)]

        # One magnification method per source (each source has its own
        # trajectory and caustic-crossing times); all sources start from the
        # event-level config value, and resolve_auto_vbbl refines each slot.
        event_method = self.config[0].get(
            "mag_method",
            "auto_vbbl" if (self.finite_source[0] or self.n_lens_bodies[0] > 1)
            else "point_source")
        self.mag_method = [event_method] * self.n_sources

        # use_op: force the MulensModel Op even for point-source PSPL.
        # Default False for PSPL (symbolic is NUTS-friendly); True forces the Op
        # (useful for testing or when MulensModel's parallax handling is needed).
        self.use_op = [c.get("use_op", False) for c in self.config]

        # backend: which magnification engine the multi-lens Op path uses.
        #   vbm_direct  — call VBMicrolensing directly (default; ~5x faster,
        #                 supports 2+ lens bodies)
        #   mulensmodel — rebuild an mm.Model per call (A/B reference; binary only)
        self.backend = self.config[0].get("backend", "vbm_direct")
        if self.backend not in ("vbm_direct", "mulensmodel"):
            raise ValueError(
                f"lens.backend must be 'vbm_direct' or 'mulensmodel', "
                f"got '{self.backend}'."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rewrite_source_param_keys(self, config_manager):
        """Rewrite lens.<SourceStarName>.<param> → lens.<j>.<param>.

        The generic standardize_param_names pass only knows the lens event's
        own instance name; addressing a per-source element by the source
        star's name (lens.SourceB.t_0) is lens-specific knowledge, so the
        translation lives here.  Keys already in index or event-name form are
        untouched (event-name form was standardized to lens.0.* and refers to
        source slot 0).
        """
        system_config = getattr(config_manager, "system_config", None) or {}
        slot_by_name = {}
        for j, (comp_type, ndx) in enumerate(self.source_bodies[0]):
            entries = system_config.get(comp_type, [])
            if ndx < len(entries) and isinstance(entries[ndx], dict):
                name = entries[ndx].get("name")
                if name is not None:
                    slot_by_name[str(name)] = j

        up = config_manager.user_params
        for key in list(up.keys()):
            parts = key.split(".")
            if len(parts) == 3 and parts[0] == self.prefix and parts[1] in slot_by_name:
                new_key = f"{self.prefix}.{slot_by_name[parts[1]]}.{parts[2]}"
                if new_key in up:
                    logger.warning(
                        f"Parameter '{key}' duplicates '{new_key}'; keeping '{new_key}'."
                    )
                    del up[key]
                else:
                    up[new_key] = up.pop(key)

    def _source_instance_names(self):
        """Display names for per-source vector elements (source star names)."""
        system_config = getattr(self.config_manager, "system_config", None) or {}
        names = []
        for comp_type, ndx in self.source_bodies[0]:
            entries = system_config.get(comp_type, [])
            if ndx < len(entries) and isinstance(entries[ndx], dict) and entries[ndx].get("name"):
                names.append(str(entries[ndx]["name"]))
            else:
                names.append(f"{comp_type}{ndx}")
        return names

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

    def _mass_initval(self, comp_type, ndx):
        """Best-effort mass initval (solMass) for a body at stage 2, from
        user_params mass or logmass entries; None when neither is given."""
        up = self.config_manager.user_params
        entry = up.get(f"{comp_type}.{ndx}.mass")
        val = entry.get("initval") if isinstance(entry, dict) else entry
        if val is not None:
            return float(val)
        entry = up.get(f"{comp_type}.{ndx}.logmass")
        val = entry.get("initval") if isinstance(entry, dict) else entry
        return float(10.0 ** float(val)) if val is not None else None

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
        """Stage 1b: Build integer index arrays for lens and source bodies.

        source_map has one entry per SOURCE BODY (not per event): it drives the
        shapes of the per-source parameter chain (pi_rel, t_E, rho, ...) via the
        star.<param>[source_map] dependency slices.
        """
        _, l_ndxs = zip(*[self._primary_lens(i) for i in range(self.n_elements)])
        self.lens_map = np.array(l_ndxs, dtype=int)
        self.source_map = np.array([ndx for (_, ndx) in self.source_bodies[0]],
                                   dtype=int)

        if self.n_companions >= 1:
            # Scalar maps (length-1) so the bracket-slice dep yields a scalar
            # mass rather than a full-component mass array.  One map per
            # companion: companions may live in different component types
            # (star vs planet), so each mass needs its own bracket dep.
            _, p_ndx = self.lens_bodies[0][0]
            self.primary_lens_map = np.array([p_ndx], dtype=int)
            for j, (_, c_ndx) in enumerate(self.lens_bodies[0][1:]):
                setattr(self, f"companion{j}_mass_map",
                        np.array([c_ndx], dtype=int))

    def register_parameters(self, system):
        """Stage 2: Declare the manifest."""
        self._validate_bodies(system)

        # Per-source vector parameters: one element per source body.  Elements
        # are displayed and addressed by the source star's instance name
        # (lens.SourceB.t_0) or slot index (lens.1.t_0).
        src_shape = (self.n_sources,)
        src_names = self._source_instance_names() if self.n_sources > 1 else None

        def per_source(expr_key=None):
            entry = {"shape": src_shape}
            if expr_key is not None:
                entry["expr_key"] = expr_key
            if src_names is not None:
                entry["names"] = src_names
            return entry

        self.manifest = {
            "t_0": per_source(), "u_0": per_source(),
            "pi_rel": per_source("default"), "theta_E": per_source("default"),
            "mu_ra_rel": per_source("default"), "mu_dec_rel": per_source("default"),
            "mu_rel_mag": per_source("default"), "t_E": per_source("default"),
            "pi_E_N": per_source("default"), "pi_E_E": per_source("default"),
        }

        # Companion geometry: one (s, alpha) pair per lens body beyond the
        # primary. The shape override sizes these by companion count rather
        # than by component element count.
        if self.n_companions >= 1:
            companion_shape = (self.n_companions,)
            self.manifest["s"] = {"shape": companion_shape}
            self.manifest["xalpha"] = {"shape": companion_shape}
            self.manifest["yalpha"] = {"shape": companion_shape}
            # alpha derived from xalpha/yalpha via arctan2; internal unit = rad, display = deg
            self.manifest["alpha"] = {"expr_key": "default", "shape": companion_shape}
            # q_j = M_companion_j / M_primary; companion component types vary
            # by config, hence one scalar bracket dep per companion.
            companion_mass_deps = [
                f"{c_type}.mass[companion{j}_mass_map]"
                for j, (c_type, _) in enumerate(self.lens_bodies[0][1:])
            ]
            self.manifest["q"] = {
                "expr_key": "default",
                "shape": companion_shape,
                "deps": companion_mass_deps + ["star.mass[primary_lens_map]"],
            }
            # Multi-lens convention: theta_E (and hence t_E, rho, pi_E) is
            # referenced to the TOTAL lens mass, matching the published
            # parameterization.  mlens_total sums the body masses and replaces
            # the primary mass in the theta_E dependency chain.
            self.manifest["mlens_total"] = {
                "expr_key": "default",
                "shape": (1,),
                "deps": ["star.mass[primary_lens_map]"] + companion_mass_deps,
            }
            theta_entry = dict(self.manifest["theta_E"])
            theta_entry["deps"] = ["mlens_total", "pi_rel"]
            self.manifest["theta_E"] = theta_entry

        if self.n_companions >= 2:
            # The symbolic relaxation engine only knows the binary mass-sum
            # and q relations (see symbolic_physics.get_symbol_map), so for
            # 3+ lens bodies the mlens_total and per-slot q initvals are
            # seeded from the per-body mass initvals instead — body masses
            # (or logmass) must be supplied in the params file; a user q
            # cannot back-propagate to a companion mass here.  Rank 40
            # (derived-mixed): overrides defaults, yields to explicit user
            # values.
            body_masses = [self._mass_initval(c_type, c_ndx)
                           for c_type, c_ndx in self.lens_bodies[0]]
            if any(m is None for m in body_masses):
                missing = [f"{ct}.{cn}" for (ct, cn), m
                           in zip(self.lens_bodies[0], body_masses) if m is None]
                logger.info(
                    f"No mass initval for lens body/bodies {missing}; cannot "
                    "seed lens.0.mlens_total or per-companion q — supply body "
                    "masses (or logmass) in the params file for 3+ body lenses."
                )
            else:
                self.config_manager.add_hint(
                    "lens.0.mlens_total", float(sum(body_masses)), rank=40)
                for j, m_c in enumerate(body_masses[1:]):
                    q_j = m_c / body_masses[0]
                    self.config_manager.add_hint(f"lens.{j}.q", q_j, rank=40)
                    self.config_manager.add_scale_hint(f"lens.{j}.q", 0.1 * q_j)

        if any(self.finite_source):
            self.manifest["rho"] = per_source("default")

        # Seed alpha hint (degrees, user unit) so inspect_start can display it
        # even before the expression graph is built.
        inst = self.names[0] if self.names else "0"
        ca_entry = (self.config_manager.user_params.get(f"lens.{inst}.xalpha")
                    or self.config_manager.user_params.get(f"lens.0.xalpha") or {})
        sa_entry = (self.config_manager.user_params.get(f"lens.{inst}.yalpha")
                    or self.config_manager.user_params.get(f"lens.0.yalpha") or {})
        ca = ca_entry.get("initval")
        sa = sa_entry.get("initval")
        if ca is not None and sa is not None:
            alpha_deg = float(np.arctan2(float(sa), float(ca)) * 180.0 / np.pi)
            self.config_manager.add_hint(f"lens.0.alpha", alpha_deg, rank=20)

        # Inject per-event physical hints
        for i in range(self.n_elements):
            l_type, l_idx = self._primary_lens(i)

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

            # Every source body gets the same bulge-source seeding: each source
            # has its own trajectory chain (distance, pm) to initialize.
            for s_type, s_idx in self.source_bodies[i]:
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
        """Sanitized single-source params.  ``index`` is the SOURCE slot: the
        per-source vector parameters (t_0, u_0, t_E, pi_E_*) hold one element
        per source body of the single event."""
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
        """Params for a binary lens.  ``index`` is the SOURCE slot; the lens
        bodies are shared by all sources (single event ⇒ event index 0).

        The derived chain (theta_E, t_E, rho, pi_E) is already referenced to
        the TOTAL lens mass via mlens_total, so the safe single-source params
        pass straight through — only the companion geometry (s, q, alpha) is
        added here.
        """
        s = self._get_safe_mm_params(index)

        l2_type, l2_idx = self.lens_bodies[0][1]
        m1 = self._body_mass(system, *self.lens_bodies[0][0])
        m2 = self._body_mass(system, l2_type, l2_idx)
        q = m2 / pt.maximum(m1, 1e-10)
        q_safe = pt.clip(pt.nan_to_num(q, nan=1e-9), 1e-9, 100.0)

        # s/xalpha/yalpha are indexed by companion (binary = companion 0),
        # not by event or source.
        alpha_deg = pt.arctan2(self.yalpha.value[0],
                               self.xalpha.value[0]) * (180.0 / np.pi)

        return {
            **s,
            's': self.s.value[0], 'q': q_safe, 'alpha': alpha_deg,
        }

    def get_magnification(self, times, obs_pos_abs, system, index=0):
        """Symbolic Paczynski magnification including parallax (PSPL only).

        ``index`` is the SOURCE slot (one trajectory per source body).

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

        Event-level property (the lens bodies and finite_source flag are shared
        by all sources), so ``index`` is ignored beyond backward compatibility.

        Callers use this to decide which obs_pos convention to pass:
        - True  → absolute barycentric AU (MulensModel satellite_skycoord)
        - False → Skowron+2011 geocentric deviations (symbolic get_magnification)
        """
        n_lenses = self.n_lens_bodies[0]
        use_rho = self.finite_source[0]
        forced = self.use_op[0]
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

        ``index`` is the SOURCE slot: each source body has its own trajectory
        (t_0, u_0, rho, ...) but shares the lens bodies.  Multi-source callers
        (MulensInstrument) invoke this once per source and combine the returned
        magnifications with per-source fluxes.

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
        if self.n_lens_bodies[0] > 2 and self.backend != "vbm_direct":
            raise NotImplementedError(
                f"{self.n_lens_bodies[0]}-lens magnification requires "
                "backend: vbm_direct (VBMicrolensing MultiMag2); the "
                "MulensModel backend supports at most 2 lens bodies."
            )

        if not self.uses_op(index):
            return self.get_magnification(times, obs_pos, system, index)

        source_ndx = self.source_map[index]
        ra_deg = float(system.star.ra.value[source_ndx].eval()) * (180.0 / np.pi)
        dec_deg = float(system.star.dec.value[source_ndx].eval()) * (180.0 / np.pi)
        coords = f"{ra_deg}d {dec_deg}d"

        use_rho = self.finite_source[0]
        n_lenses = self.n_lens_bodies[0]

        # Apply LD only for finite-source and when a band is connected.
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None

        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        if n_lenses >= 2 and self.backend == "vbm_direct":
            sp = self._get_safe_mm_params(index)
            param_list = [sp['t0'], sp['u0'], sp['tE'], sp['pi_N'], sp['pi_E']]
            if use_rho:
                param_list.append(self.rho.value[index])
            for j in range(self.n_companions):
                q_j = pt.clip(pt.nan_to_num(self.q.value[j], nan=1e-9),
                              1e-9, 100.0)
                alpha_deg_j = pt.arctan2(self.yalpha.value[j],
                                         self.xalpha.value[j]) * (180.0 / np.pi)
                param_list.extend([self.s.value[j], q_j, alpha_deg_j])
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = VBMDirectMagOp(coords=coords, n_companions=self.n_companions,
                                    use_rho=use_rho, bandpass=effective_bandpass)
        elif n_lenses == 2:
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

    def _get_initval(self, param, slot=0):
        """Look up a resolved initval from config_manager, checking name and index forms.

        ``slot`` is the element index within the parameter's own vector:
        source slot for per-source params (t_0, u_0, rho, ...), companion slot
        for per-companion params (s, alpha, q).
        """
        cm = self.config_manager
        name = self.names[slot] if slot < len(self.names) else str(slot)
        for key in [f"lens.{name}.{param}", f"lens.{slot}.{param}"]:
            entry = cm.user_params.get(key)
            if entry is not None:
                val = entry.get("initval") if isinstance(entry, dict) else entry
                if val is not None:
                    return float(val)
        return None

    def resolve_auto_vbbl(self, times_np, index=0):
        """Replace 'auto_vbbl' with a concrete method list for multi-body lenses.

        Historically this computed hexadecapole-vs-VBM brackets on a time
        grid, but MulensModel implements binary-lens hexadecapole as 13
        python-level VBM.BinaryMag0 calls per epoch while VBM's BinaryMag2
        runs the equivalent quadrupole safety test internally in C++ and
        short-circuits to point-source when safe.  Measured on DC2018_128:
        hexadecapole 32.9 ms vs VBM-everywhere 7.7 ms per 870-point call, at
        equal or better accuracy — so the bracket machinery optimized for the
        wrong cost model and was removed (see hpc_optimization.txt, P1).

        Single-lens events are left untouched: 'auto_vbbl' is resolved inside
        the PSPL model builder (point_source + finite-source window), and the
        VBM/VBBL methods emitted here are binary-lens-only.

        Only the mulensmodel backend consumes the resulting method list; the
        default vbm_direct backend always calls BinaryMag2/MultiMag2.
        """
        if self.mag_method[index] != "auto_vbbl":
            return
        if self.n_lens_bodies[0] < 2:
            return

        t_lo = float(np.min(times_np))
        t_hi = float(np.max(times_np))
        method = "VBM" if self.finite_source[0] else "VBBL"
        self.mag_method[index] = [t_lo - 1.0, method, t_hi + 1.0]
        logger.info(f"auto_vbbl: using {method} everywhere for source {index} "
                    "(hexadecapole bracketing removed — VBM's internal C++ "
                    "point-source test is faster; see hpc_optimization.txt P1)")

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

    def _companion_instance_names(self):
        """Display names for per-companion vector elements (companion lens bodies)."""
        system_config = getattr(self.config_manager, "system_config", None) or {}
        names = []
        for comp_type, ndx in self.lens_bodies[0][1:]:
            entries = system_config.get(comp_type, [])
            if ndx < len(entries) and isinstance(entries[ndx], dict) and entries[ndx].get("name"):
                names.append(str(entries[ndx]["name"]))
            else:
                names.append(f"{comp_type}{ndx}")
        return names

    def plot_corner(self, idata, filename_prefix="debug"):
        """Corner plot of the fitted lensing geometry: t_0, u_0, t_E, s, q,
        alpha, rho -- whichever of these the event actually has (rho only for
        finite-source events; s/q/alpha only when there is at least one lens
        companion). Only meaningful with the full posterior, so this is
        called once, after sampling, via plot_corner (not the twice-called
        plot() hook, which also runs pre-flight on a single point).

        t_E (and, for multi-body lenses, q and alpha) are pure physics
        expressions with no sampled elements of their own, so they never get
        a pm.Deterministic node and never appear in idata.posterior directly
        (see Parameter.build_pymc's ``track_node`` logic) -- this reads each
        Parameter's ``.posterior`` instead, which System.distribute_posterior
        (already called earlier in run_fit, before this hook) reconstructs
        for both tracked and pure-expression parameters alike.
        """
        src_names = self._source_instance_names() if self.n_sources > 1 else None
        comp_names = self._companion_instance_names() if self.n_companions > 1 else None

        def per_source_labels(param):
            return [f"{param}[{name}]" for name in src_names] if src_names else None

        def per_companion_labels(param):
            return [f"{param}[{name}]" for name in comp_names] if comp_names else None

        param_specs = [
            (self.t_0, per_source_labels("t_0")),
            (self.u_0, per_source_labels("u_0")),
            (self.t_E, per_source_labels("t_E")),
        ]
        if hasattr(self, "rho"):
            param_specs.append((self.rho, per_source_labels("rho")))
        if self.n_companions >= 1:
            param_specs.append((self.s, per_companion_labels("s")))
            param_specs.append((self.q, per_companion_labels("q")))
            param_specs.append((self.alpha, per_companion_labels("alpha")))

        samples, labels = collect_parameter_corner_samples(param_specs)
        save_corner_plot(samples, labels, f"{filename_prefix}_lens_corner.png")
