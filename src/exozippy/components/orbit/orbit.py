import logging

import numpy as np
import astropy.units as u

logger = logging.getLogger(__name__)

# pymc/exoplanet imports
import pytensor.tensor as pt
import pymc as pm
from exoplanet_core.pymc import ops as ops

# local imports
from exozippy.components.parameter import Parameter
from exozippy.components.component import Component
from .bodies import parse_orbit_bodies
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

class Orbit(Component):
    """
    Two-body Keplerian orbit between a primary and a companion body group
    (see bodies.py for the group syntax).  Alongside the timing/geometry
    elements, each orbit derives its own physical scale -- m_primary,
    m_companion, m_total, arsun, K -- from the masses of its member bodies,
    so hierarchical systems (e.g. B orbits C, B+C orbits A, planet b orbits
    A) stay mass-consistent automatically: every orbit touching a body
    reads the same star.mass/planet.mass nodes.  Each group is treated as a
    point mass at its barycenter (standard hierarchical approximation).
    """
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Orbital Parameters"

        self.primary_bodies, self.companion_bodies = parse_orbit_bodies(
            self.config, getattr(config_manager, "system_config", None))
        self.i180 = [c.get("i180",False) for c in self.config]
        self.fitvcve = [c.get("fitvcve",False) for c in self.config]

    @property
    def prefix(self):
        return "orbit"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "primary",
                "kind": "ref",
                "accepts": ["star", "planet"],
                "required": False,
                "doc": (
                    "Body group forming the primary of this two-body "
                    "Keplerian arc: a list of star/planet instance names or "
                    "star.X/planet.X paths (a multi-body group is treated as "
                    "a point mass at its barycenter). Omit both primary and "
                    "companion to use the legacy implicit host/planet "
                    "topology."
                ),
            },
            {
                "key": "companion",
                "kind": "ref",
                "accepts": ["star", "planet"],
                "required": False,
                "doc": (
                    "Body group forming the companion of this two-body "
                    "Keplerian arc (see primary)."
                ),
            },
            {
                "key": "i180",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Reflect the inclination about 90 deg (retrograde branch "
                    "of the transit/RV inclination degeneracy). Default false."
                ),
            },
            {
                "key": "fitvcve",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Parametrize eccentricity via V_c/V_e instead of "
                    "sqrt(e)cos(omega)/sqrt(e)sin(omega). Default false."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Body groups
    # ------------------------------------------------------------------
    def bodies(self, i):
        """All (comp_type, index) bodies of orbit i (both groups)."""
        return self.primary_bodies[i] + self.companion_bodies[i]

    def star_membership(self, star_idx):
        """
        Orbits containing star star_idx, as [(orbit_index, role), ...] with
        role 'primary' or 'companion'.  Used by instruments to decide which
        orbits move (or blend with) a given star.
        """
        out = []
        key = ("star", int(star_idx))
        for i in range(self.n_elements):
            if key in self.primary_bodies[i]:
                out.append((i, "primary"))
            elif key in self.companion_bodies[i]:
                out.append((i, "companion"))
        return out

    def build_maps(self):
        """Stage 1b: 0/1 weight matrices mapping body masses into groups.

        _group_w[side][comp_type] is an (n_orbits, n_<comp_type>) float
        matrix; the group mass is its product with the component's mass
        vector.  Matrices are built for every component type referenced by
        at least one group.
        """
        sys_cfg = getattr(self.config_manager, "system_config", None) or {}
        self._group_w = {"primary": {}, "companion": {}}
        for side, groups in (("primary", self.primary_bodies),
                             ("companion", self.companion_bodies)):
            types = {t for g in groups for (t, _) in g}
            for ctype in types:
                section = sys_cfg.get(ctype) or []
                if not isinstance(section, list):
                    section = [section]
                n_cols = max([len(section)] +
                             [idx + 1 for g in groups for (t, idx) in g
                              if t == ctype])
                W = np.zeros((self.n_elements, n_cols))
                for i, g in enumerate(groups):
                    for (t, idx) in g:
                        if t == ctype:
                            W[i, idx] = 1.0
                self._group_w[side][ctype] = W

    def register_parameters(self, system):
        """Stage 2: Calculate window constraints and declare the manifest."""
        shape = (self.n_elements,)

        # 1. Peer into the config (Pre-flight windows)
        logP_cfg = self.config_manager.resolve(self.prefix, "logP", shape=shape, names=self.names)
        tc_cfg = self.config_manager.resolve(self.prefix, "tc", shape=shape, names=self.names)

        logP_init = np.atleast_1d(logP_cfg["initval"])
        tc_init = np.atleast_1d(tc_cfg["initval"])
        half_period = (10 ** logP_init) / 2.0

        self.manifest = {
            "logP": None,
            "period": {"force_node": True, "expr_key": "default"},
            "n": "default",
            "tc": {
                "force_node": True,
                "lower": tc_init - half_period,
                "upper": tc_init + half_period
            }
        }

        fitvcve_mask = np.atleast_1d(getattr(self, 'fitvcve', False)).astype(bool)
        hk_mask = ~fitvcve_mask

        if any(self.fitvcve):
            raise NotImplementedError("VCVE parameterization not yet migrated to manifest.")
        else:
            self.manifest.update({
                "secosw": {"mask": hk_mask},
                "sesinw": {"mask": hk_mask},
                "cosi": {"mask": hk_mask},
                "ecc": "default",
                "omega": "default",
                "inc": "default",
                "sini": "default",
                "sinw": "default",
                "cosw": "default",
                "esinw": "default",
                "ecosw": "default",
                "tp": "default",
            })

        # Physical scale of every orbit, from the member bodies' masses
        # (see class docstring).  Group-mass deps name the mass vectors of
        # whichever component types the groups actually reference; the
        # weighted per-group sums are injected in add_parameter below.
        # Bare orbits whose implicit default bodies do not exist (test
        # harnesses, geometry-only systems) skip the scale parameters.
        if self._validate_bodies(system):
            body_types = sorted({t for i in range(self.n_elements)
                                 for (t, _) in self.bodies(i)})
            group_deps = [f"{t}.mass" for t in body_types]
            self.manifest.update({
                "m_primary": {"expr_key": "default", "deps": group_deps},
                "m_companion": {"expr_key": "default", "deps": group_deps},
                "m_total": "default",
                "arsun": "default",
                "K": "default",
            })

        # Astrometry constrains the longitude of the ascending node and
        # breaks the i <-> 180-i degeneracy, so sample the node direction
        # vector (xbigomega, ybigomega; each N(0,1) -> uniform marginal on
        # bigomega, like the microlensing trajectory angle alpha) and allow
        # the full inclination range when an astrometry component is active.
        topology_keys = []
        if hasattr(system, 'config') and hasattr(system.config, 'keys'):
            topology_keys = list(system.config.keys())
        has_astrometry = (hasattr(system, 'astrometryinstrument')
                          or 'astrometryinstrument' in topology_keys)
        if has_astrometry:
            self.manifest["xbigomega"] = None
            self.manifest["ybigomega"] = None
            self.manifest["bigomega"] = "default"

            # The (bigomega, omega) <-> (bigomega+180, omega+180)
            # transformation is a reflection through the sky plane
            # (z -> -z): invisible to ANY astrometry, absolute or relative.
            # Only radial information (RVs) identifies the ascending node.
            has_rv = (hasattr(system, 'rvinstrument')
                      or 'rvinstrument' in topology_keys)
            if not has_rv:
                self._restrict_bigomega_halfplane(shape)

        i180_arr = np.atleast_1d(getattr(self, 'i180', False)) | has_astrometry
        derived_lowers = np.where(i180_arr, -1.0, 0.0)
        self.manifest["cosi"] = {"lower": derived_lowers}

    def _restrict_bigomega_halfplane(self, shape):
        """Astrometry without RVs: restrict bigomega to [0, 180] deg.

        (bigomega, omega_*) and (bigomega+180, omega_*+180) is a reflection
        through the sky plane, so it produces identical astrometry of every
        kind (absolute, epoch, and relative); only RVs identify which node
        is ascending.  Bounding ybigomega >= 0 selects the bigomega in
        [0, 180] mode.  Seeds in (180, 360) are remapped to the equivalent
        solution -- which flips (xbigomega, ybigomega) AND (secosw,
        sesinw), and shifts tc so the orbit's position-vs-time is
        unchanged.  A table note documents the artificial boundary on
        omega_* and bigomega.
        """
        note = (r"With astrometry but no RVs, $(\Omega, \omega_*)$ and "
                r"$(\Omega+180^\circ, \omega_*+180^\circ)$ are exactly "
                r"degenerate (which node is ascending is unknown); "
                r"$\Omega$ is artificially restricted to "
                r"$[0^\circ, 180^\circ]$ to select one mode.")

        # NOTE: this runs at stage 2, BEFORE the relaxation engine, so only
        # user-provided initvals (and defaults) are visible here.  The x/y
        # direction vector is therefore derived directly from the user's
        # bigomega initval; the manifest initvals set below override the
        # relaxation-engine seeds at build time.
        cm = self.config_manager
        n_el = int(np.prod(shape))

        def rslv(name):
            val = cm.resolve(self.prefix, name, shape=shape, names=self.names)["initval"]
            if val is None:
                return np.full(n_el, np.nan)
            return np.atleast_1d(val).astype(float).copy()

        factor_bo = cm.get_conversion_factor(self.prefix, "bigomega") or 1.0
        bo = rslv("bigomega") * factor_bo   # rad; NaN where unseeded

        # Unseeded elements start at bigomega = 90 deg (center of the
        # allowed half-plane; y = 0 would sit exactly on the new bound).
        x_init = np.where(np.isnan(bo), 0.0, np.cos(bo))
        y_init = np.where(np.isnan(bo), 1.0, np.sin(bo))

        # Seeds with bigomega in (180, 360): remap to the degenerate
        # partner (bigomega - 180, omega_* + 180, and tc shifted so the
        # position-vs-time model is unchanged).
        flip = y_init < 0.0
        if np.any(flip):
            logger.warning(
                f"[{self.prefix}] bigomega initval(s) in (180, 360) deg but no "
                f"RVs are present; remapping element(s) {np.where(flip)[0]} to "
                f"the degenerate (bigomega-180, omega+180) solution.")

            # Orientation in the user's own terms: prefer explicit ecc+omega
            # initvals; otherwise secosw/sesinw (user or defaults).
            sc0 = rslv("secosw")
            ss0 = rslv("sesinw")
            factor_om = cm.get_conversion_factor(self.prefix, "omega") or 1.0
            om = rslv("omega") * factor_om
            e_u = rslv("ecc")
            have_ew = ~np.isnan(om) & ~np.isnan(e_u)
            sc0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.cos(om), sc0)
            ss0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.sin(om), ss0)

            tc0 = rslv("tc")
            # The relaxation engine has not run yet, so a user-supplied
            # 'period' has not been propagated to logP; prefer it directly.
            period_user = rslv("period")
            period = np.where(~np.isnan(period_user), period_user,
                              10.0 ** rslv("logP"))

            def _M_c(ecc, w):
                E_c = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * (1.0 - np.sin(w)),
                                       np.sqrt(1.0 + ecc) * np.cos(w))
                return E_c - ecc * np.sin(E_c)

            ecc0 = np.clip(sc0 ** 2 + ss0 ** 2, 0.0, 0.9999)
            w0 = np.arctan2(ss0, sc0)
            n_mm = 2.0 * np.pi / period
            tp = tc0 - _M_c(ecc0, w0) / n_mm
            tc_new = tp + _M_c(ecc0, w0 + np.pi) / n_mm

            x_init = np.where(flip, -x_init, x_init)
            y_init = np.where(flip, -y_init, y_init)
            sc_init = np.where(flip, -sc0, sc0)
            ss_init = np.where(flip, -ss0, ss0)
            tc_init = np.where(flip, tc_new, tc0)

            self.manifest["secosw"] = {**self.manifest["secosw"], "initval": sc_init}
            self.manifest["sesinw"] = {**self.manifest["sesinw"], "initval": ss_init}
            half_period = period / 2.0
            self.manifest["tc"] = {**self.manifest["tc"], "initval": tc_init,
                                   "lower": tc_init - half_period,
                                   "upper": tc_init + half_period}

        # Keep seeded boundary values (bigomega exactly 0 or 180) strictly
        # inside the ybigomega >= 0 bound.
        y_init = np.maximum(y_init, 1e-6)

        self.manifest["xbigomega"] = {"initval": x_init}
        self.manifest["ybigomega"] = {"initval": y_init, "lower": 0.0}
        self.manifest["bigomega"] = {"expr_key": "default", "table_note": note}
        self.manifest["omega"] = {"expr_key": "default", "table_note": note}

    def _validate_bodies(self, system):
        """Check body references against the live system topology.

        Returns True when every body resolves to an active component
        element, so the mass/scale parameters can be built.  Unresolvable
        bodies raise if the user declared the groups explicitly; implicit
        defaults (bare orbit in a test harness or geometry-only system)
        just disable the scale parameters.
        """
        if not hasattr(system, "active_components"):
            return False
        for i in range(self.n_elements):
            explicit = ("primary" in self.config[i]
                        or "companion" in self.config[i])
            for (ctype, idx) in self.bodies(i):
                comp = getattr(system, ctype, None)
                bad = (comp is None or not isinstance(comp, Component)
                       or idx >= comp.n_elements)
                if bad and explicit:
                    n = comp.n_elements if isinstance(comp, Component) else 0
                    raise ValueError(
                        f"[{self.prefix}.{self.names[i]}] references body "
                        f"'{ctype}.{idx}', but the active system has only "
                        f"{n} '{ctype}' instance(s).")
                if bad:
                    logger.info(
                        f"[{self.prefix}.{self.names[i]}] implicit body "
                        f"'{ctype}.{idx}' is not in the system; orbit "
                        f"mass/scale parameters (m_total, arsun, K) are "
                        f"disabled.")
                    return False
        # A planet in a companion group should point its orbit_ndx here
        # (transit/planet geometry reads the orbit through that map).
        planet_cfgs = (getattr(self.config_manager, "system_config", None)
                       or {}).get("planet") or []
        for i in range(self.n_elements):
            for (ctype, idx) in self.companion_bodies[i]:
                if ctype != "planet" or idx >= len(planet_cfgs):
                    continue
                o_ndx = int((planet_cfgs[idx] or {}).get("orbit_ndx", 0))
                if o_ndx != i:
                    logger.warning(
                        f"[{self.prefix}.{self.names[i]}] companion planet."
                        f"{idx} has orbit_ndx={o_ndx}, not {i}; the planet's "
                        f"transit/RV geometry will follow orbit {o_ndx} "
                        f"while its mass moves this orbit.")
        return True

    _GROUP_MASS_SIDE = {"m_primary": "primary", "m_companion": "companion"}

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """
        The group masses are weighted sums over other components' mass
        vectors -- a matrix product the generic dep parser cannot express.
        Intercept them here: build each referenced component's mass node,
        pre-compute the per-group weighted sums, and hand them to the
        generic machinery as context nodes keyed by the dep names.
        """
        side = self._GROUP_MASS_SIDE.get(param_name)
        if side is not None and not context_nodes:
            if not hasattr(self, "_group_w"):
                self.build_maps()   # standalone use outside the lifecycle
            context_nodes = dict(context_nodes or {})
            for ctype, W in self._group_w[side].items():
                comp = getattr(system, ctype, None)
                if comp is None:
                    # Standalone harness (validated systems raised at stage
                    # 2): absent components contribute zero mass.
                    context_nodes[f"{ctype}.mass"] = pt.zeros((self.n_elements,))
                    continue
                if not isinstance(getattr(comp, "mass", None), Parameter):
                    comp.add_parameter(model, "mass", system)
                context_nodes[f"{ctype}.mass"] = pt.dot(
                    pt.as_tensor_variable(W), comp.mass.value)
            # A group side may reference only a subset of the body types
            # named in the shared deps list; the missing type contributes
            # zero mass.
            for ctype in ("star", "planet"):
                dep = f"{ctype}.mass"
                if dep not in context_nodes and any(
                        d == dep for d in self.manifest[param_name]["deps"]):
                    context_nodes[dep] = pt.zeros((self.n_elements,))
        return super().add_parameter(model, param_name, system, context_nodes)

    def build_likelihood(self, model, system):
        pass

    def get_true_anomaly(self, t):
        """Returns the true anomaly f for all planets at all times."""
        t_grid = t[:, None]
        tp = self.tp.value[None, :]
        n = self.n.value[None, :]
        ecc = self.ecc.value[None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        return pt.arctan2(sinf, cosf)

    def get_sky_position(self, t, a_scale, orbit_map, relative=False):
        """
        Vectorized sky-plane offsets of an orbiting body.

        t: (N_obs,) vector of times [BJD_TDB]
        a_scale: (N_planets,) amplitude scaling, e.g. the photocenter or
                 relative semimajor axis in mas; sets the output units
        orbit_map: integer map from planet slots to orbit elements
        relative: False -> the primary/photocenter orbit around the
                  barycenter (uses omega_*); True -> the companion's orbit
                  relative to the primary (omega_* + 180 deg)

        Returns (dE, dN), each (N_obs, N_planets): offsets toward East and
        North in the units of a_scale.

        Conventions (EXOFASTv2): omega is the argument of periastron of the
        PRIMARY's orbit (omega_*). bigomega is the position angle of the
        ascending node, measured East of North, where the ascending node is
        the node at which the body recedes from the observer -- consistent
        with the sign of get_radial_velocity (the primary crosses its
        ascending node at omega_* + f = 0, where its RV is maximal).
        Without RVs, (bigomega, omega) and (bigomega+180, omega+180) are
        exactly degenerate for astrometry of every kind (a reflection
        through the sky plane); see _restrict_bigomega_halfplane.
        """
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        cosi = self.cosi.value[orbit_map][None, :]
        bigomega = self.bigomega.value[orbit_map][None, :]
        cosO = pt.cos(bigomega)
        sinO = pt.sin(bigomega)

        if relative:
            # The companion's argument of periastron is omega_* + pi
            cosw = -cosw
            sinw = -sinw

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # Separation from the barycenter (or primary) in units of a_scale
        r = a_scale[None, :] * (1.0 - ecc ** 2) / (1.0 + ecc * cosf)

        # cos/sin(omega + f)
        coswf = cosw * cosf - sinw * sinf
        sinwf = sinw * cosf + cosw * sinf

        # Thiele-Innes projection (North, East), PA measured East of North:
        # at omega + f = 0 (ascending node) the body sits at PA = bigomega.
        dN = r * (cosO * coswf - sinO * sinwf * cosi)
        dE = r * (sinO * coswf + cosO * sinwf * cosi)
        return dE, dN

    def get_radial_velocity(self, t, K, orbit_map):
        """
        The optimized vectorized reflex RV signal.
        t: (N_obs,) vector of times
        K: (N_planets,) vector of semi-amplitudes
        """
        # 1. Broadcast time and orbital parameters into 2D grids
        # Shape: (N_obs, N_planets)
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        K_grid = K[None, :]

        # 2. Calculate Mean Anomaly (M)
        # M = n * (t - tp)
        M = (t_grid - tp) * n

        # 3. Solve Kepler's Equation
        # ops.kepler handles the (N_obs, N_planets) grid efficiently
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # 4. Calculate RV per planet
        # Using the identity: cos(w + f) = cos(w)cos(f) - sin(w)sin(f)
        rv_matrix = K_grid * (cosw*cosf - sinw*sinf + ecc*cosw)

        return rv_matrix