"""The SED side of the sight-line dust profile (review 8.9.8; sed.md, "The
sight-line dust profile").  A mixin of ``SED``: the config keys, the one
plan every reader asks (``dust_plan``), the SED-level parameters, the graph
builders the stars and flux sources read their A_V and R_V from, and the
start seeding.  The arithmetic is ``dust_profile.py``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component, in_topology
from exozippy.config import PRECEDENCE_DERIVED_USER, user_entry
from exozippy.logger import first_time

from . import dust_profile as dp

logger = logging.getLogger(__name__)

#: The SED-level parameters of the profile.
DUST_ANCHOR = "dust_anchor"
AV_ANCHOR = "av_anchor"
DUST_Z = "dust_z"
RV_DUST_NEAR = "rv_dust_near"
DUST_BOUNDARY = "dust_boundary"

#: Read stars farther apart on the sky than this are not on ONE sight line.
SIGHT_LINE_TOLERANCE_ARCSEC = 60.0

#: sin|b| below which the layer has no vertical extent to speak of.
MIN_SINB = 1e-4


class SightLineDust:
    """Mixed into ``SED``."""

    DUST_CONFIG_KEYS = (
        "distance_constrains_av",
        "fit_rv_dust_near",
        "dust_bins",
        "dust_scale_height",
        "dust_anchor_band",
    )

    # ------------------------------------------------------------------
    # Config (construction time)
    # ------------------------------------------------------------------
    def _parse_dust_config(self):
        cfg = self.config

        def _bool(key):
            v = cfg.get(key)
            if v is not None and not isinstance(v, bool):
                raise ValueError(
                    f"sed.{key} must be true or false (a YAML boolean), not "
                    f"{v!r}."
                )
            return v

        self.distance_constrains_av = _bool("distance_constrains_av")
        self.fit_rv_dust_near = _bool("fit_rv_dust_near")
        bins = cfg.get("dust_bins")
        if bins is not None and (
            isinstance(bins, bool) or not isinstance(bins, int) or bins < 1
        ):
            raise ValueError(
                f"sed.dust_bins must be a positive integer, not {bins!r}."
            )
        self.dust_bins = bins
        h = cfg.get("dust_scale_height", dp.DEFAULT_SCALE_HEIGHT_PC)
        if (
            isinstance(h, bool)
            or not isinstance(h, (int, float))
            or not math.isfinite(h)
            or h <= 0
        ):
            raise ValueError(
                f"sed.dust_scale_height must be a positive number of pc, not "
                f"{h!r}."
            )
        self.dust_scale_height = float(h)
        band = cfg.get("dust_anchor_band")
        if band is not None and not isinstance(band, str):
            raise ValueError(
                f"sed.dust_anchor_band must be a filter name (a string), not "
                f"{band!r}; leave it out to anchor A_V itself."
            )
        self.dust_anchor_band = band
        if self.distance_constrains_av is False:
            stray = [k for k in self.DUST_CONFIG_KEYS[1:] if k in cfg]
            if stray:
                raise ValueError(
                    f"sed.distance_constrains_av: false turns the sight-line "
                    f"dust profile off, so {stray} describe nothing; remove "
                    f"them, or the false."
                )
        if self.fit_rv_dust_near and not self.fit_rv_dust:
            raise ValueError(
                "sed.fit_rv_dust_near: true needs sed.fit_rv_dust: true -- "
                "the near/far split mixes two CCM R_V values, and the "
                "tabulated law has no R_V."
            )
        self._dust_seed_distances = None
        self._dust_parts_cache = None

    # ------------------------------------------------------------------
    # The plan (stage 3; asked by Star, flux sources and this SED)
    # ------------------------------------------------------------------
    def _distance_groups(self, system, stars):
        """Tuples of ``stars`` that share ONE distance: an orbit group's
        members are their anchor's (Star.orbit_anchors), and a params-file
        hard link ``star.<k>.distance: {initval: star.<j>.distance, sigma:
        0}`` ties k to j (a microlensing second source)."""
        star = system.star
        n = int(star.n_elements)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        for member, anchor in star.orbit_anchors(system).items():
            union(int(member), int(anchor))
        cm = self.config_manager
        for k in range(n):
            link = (cm.links.get(f"star.{k}.distance") or {}).get("initval")
            entry = user_entry(cm.user_params or {}, f"star.{k}.distance")
            if (
                link is None
                or (entry or {}).get("sigma") != 0
                or len(link.dep_paths) != 1
            ):
                continue
            parts = link.dep_paths[0].split(".")
            if (
                len(parts) == 3
                and parts[0] == "star"
                and parts[2] == "distance"
                and parts[1].isdigit()
            ):
                union(k, int(parts[1]))
        groups = {}
        for i in stars:
            groups.setdefault(find(int(i)), []).append(int(i))
        return [tuple(v) for _, v in sorted(groups.items())]

    def _sight_line_lb(self, system, stars):
        """(l, b) in degrees: the galactic model's sight line when there is
        one (its anchor star's position root), else the read stars' own
        stated positions, which must agree to SIGHT_LINE_TOLERANCE_ARCSEC.
        None when a position it needs is not stated (``probe_star_radec``:
        the user's own numbers, never a placeholder)."""
        from exozippy.components.galacticmodel import koshimoto21 as k21
        from exozippy.components.star.position import probe_star_radec

        cm = self.config_manager
        gm = in_topology(system, "galacticmodel")
        if isinstance(gm, Component):
            from exozippy.components.star.star import position_root_of

            root = position_root_of(system, int(gm.anchor_idx))
            if probe_star_radec(cm, [root])[root] is None:
                return None
            return gm.sight_line(system)
        star = system.star
        got = probe_star_radec(cm, [int(i) for i in stars])
        if any(got[int(i)] is None for i in stars):
            return None
        pos = [got[int(i)] for i in stars]
        ra0, dec0 = pos[0]  # radians
        for (ra, dec), i in zip(pos[1:], stars[1:]):
            c = math.sin(dec0) * math.sin(dec) + math.cos(dec0) * math.cos(
                dec
            ) * math.cos(ra - ra0)
            sep = math.degrees(math.acos(min(1.0, max(-1.0, c)))) * 3600.0
            if sep > SIGHT_LINE_TOLERANCE_ARCSEC:
                raise ValueError(
                    f"[{self.prefix}] the stars the SED reads are not on one "
                    f"sight line: {star.names[int(stars[0])]} and "
                    f"{star.names[int(i)]} are {sep:.0f} arcsec apart (> "
                    f"{SIGHT_LINE_TOLERANCE_ARCSEC:g}).  The dust profile is "
                    f"one A_V(d) per sight line; set "
                    f"sed.distance_constrains_av: false."
                )
        l_deg, b_deg, _m, _v0 = k21.line_of_sight_basis(ra0, dec0)
        return float(l_deg), float(b_deg)

    def dust_plan(self, system):
        """What the dust profile is on this sight line (``DustPlan``): ONE
        answer every reader asks (Star, the flux sources, this SED), from
        the topology and the config, never from a start.

        ON by default when the objects the SED reads -- the read stars and
        the flux sources -- sit at two or more DIFFERENT distances (a
        distance hard-linked between stars counts once); ``sed.
        distance_constrains_av: false`` returns every A_V to a free
        parameter, ``true`` turns it on for a single object too (a map's
        total as the anchor).  JDE 2026-11-09."""
        if self.distance_constrains_av is False:
            return dp.DustPlan(False, "sed.distance_constrains_av: false")
        star = system.star
        seen = self.seen_star_mask(system)
        read = [i for i in range(int(star.n_elements)) if seen[i]]
        groups = self._distance_groups(system, read)
        # A flux source at a STAR's distance (its path names one) adds no
        # distance of its own.
        own = [
            p
            for p in self.providers
            if not self._split_path(p.paths["distance"])[0] == "star"
        ]
        n_groups = len(groups) + len(own)
        explicit = self.distance_constrains_av is True
        if n_groups == 0:
            if explicit:
                raise ValueError(
                    f"[{self.prefix}] distance_constrains_av: true, but the "
                    f"SED reads no object: there is nothing to put on a dust "
                    f"profile."
                )
            return dp.DustPlan(False, "the SED reads no object")
        if n_groups < 2 and not explicit:
            return dp.DustPlan(
                False,
                "every object the SED reads is at one distance, so one "
                "free A_V describes the sight line",
            )
        anchors = [p for p in self.providers if p.extinction_anchor]
        if len(anchors) > 1:
            raise ValueError(
                f"[{self.prefix}] SED flux sources "
                f"{[p.name for p in anchors]} each claim to anchor the "
                f"sight line's extinction; one sight line has one anchor."
            )
        lb = self._sight_line_lb(system, read)
        if lb is None:
            # No stated sky position, so no Galactic latitude and no layer:
            # the profile cannot be placed.  Off by default (a transit
            # blend with no positions keeps its free A_V values); asked for
            # explicitly, it raises.
            if explicit:
                raise ValueError(
                    f"[{self.prefix}] distance_constrains_av: true needs the "
                    f"sight line's sky position (the Galactic latitude sets "
                    f"the dust layer): give the read stars' ra/dec (or the "
                    f"galactic model's anchor's) in the params file."
                )
            return dp.DustPlan(
                False,
                "no sky position is stated for the read stars, so the "
                "sight line's Galactic latitude (the dust layer) is unknown",
            )
        l_deg, b_deg = lb
        sinb = dp.sinb_of(b_deg)
        if sinb < MIN_SINB:
            raise ValueError(
                f"[{self.prefix}] the sight line is on the Galactic plane (b "
                f"= {b_deg:.4f} deg): the dust layer of the profile has no "
                f"vertical extent there.  Set sed.distance_constrains_av: "
                f"false."
            )
        n_bins = (
            int(self.dust_bins)
            if self.dust_bins is not None
            else dp.default_bins(n_groups)
        )
        clump = bool(anchors)
        far = set()
        for comp in system.active_components.values():
            far |= {int(k) for k in comp.selected_sources(system)}
        far_groups = [g for g in groups if far & set(g)]
        near = tuple(i for g in groups if g not in far_groups for i in g)
        has_far = bool(far_groups) or clump
        if self.fit_rv_dust_near is None:
            # Default ON only where an object is KNOWN to be nearer than the
            # far group: a read star outside the group of a selected source
            # (a microlensing lens).  A lone source sees only the integrated
            # column, where one R_V describes it.
            split = bool(self.fit_rv_dust and near and far_groups)
        else:
            split = bool(self.fit_rv_dust_near)
        if (
            split
            and not (near and has_far)
            and first_time("dust-split-no-near")
        ):
            logger.warning(
                f"[{self.prefix}] fit_rv_dust_near: true, but no read object "
                f"is nearer than the far group, so sed.rv_dust_near is held "
                f"by its prior alone.\nRemove fit_rv_dust_near from the sed "
                f"block, or read a nearer object (a lens) in an SED row."
            )
        names = star.names
        reason = (
            f"{n_groups} SED object(s) at distinct distances "
            f"({', '.join('/'.join(names[i] for i in g) for g in groups)}"
            + (
                f"{', ' if groups else ''}"
                f"{', '.join(p.name for p in self.providers)}"
                if self.providers
                else ""
            )
            + f") on the sight line at b = {b_deg:.3f} deg"
        )
        return dp.DustPlan(
            True,
            reason,
            n_bins=n_bins,
            sinb=sinb,
            b_deg=float(b_deg),
            h_pc=self.dust_scale_height,
            clump=clump,
            anchor_band=self.dust_anchor_band,
            split=split,
            groups=tuple(groups),
            near_stars=near,
        )

    def anchor_provider(self):
        """The flux source the profile is anchored at, or None."""
        for p in self.providers:
            if p.extinction_anchor:
                return p
        return None

    # ------------------------------------------------------------------
    # Stage 3: the SED-level parameters
    # ------------------------------------------------------------------
    def _register_dust(self, system):
        plan = self.dust_plan(system)
        self._dust_plan = plan
        if not plan.on:
            if self.distance_constrains_av is not False and (
                self.fit_rv_dust_near
                or self.dust_bins is not None
                or self.dust_anchor_band is not None
            ):
                raise ValueError(
                    f"[{self.prefix}] the dust profile is off ({plan.reason}),"
                    f" so fit_rv_dust_near/dust_bins/dust_anchor_band "
                    f"describe nothing; set distance_constrains_av: true to "
                    f"turn it on, or remove them."
                )
            logger.debug(
                f"[{self.prefix}] no sight-line dust profile: {plan.reason}."
            )
            return plan
        if self.fit_rv_dust_near and not plan.split:
            raise AssertionError("dust_plan dropped an explicit split")
        band = plan.anchor_band or "V"
        where = (
            f"the {self.anchor_provider().name}'s distance"
            if plan.clump
            else "the far end of the sight line (the whole column)"
        )
        self.manifest[DUST_ANCHOR] = {
            "description": f"Extinction in {band} at {where}",
            "latex": (
                r"A_{V,\rm anchor}"
                if plan.anchor_band is None
                else r"A_{\rm anchor}"
            ),
        }
        self.manifest[DUST_Z] = {
            "shape": (plan.n_bins,),
            "names": [f"bin{k}" for k in range(plan.n_bins)],
        }
        if plan.anchor_band is not None:
            self.manifest[AV_ANCHOR] = "default"
        if plan.split:
            self.manifest[RV_DUST_NEAR] = None
            self.manifest[DUST_BOUNDARY] = None
        self._seed_dust_anchor(system, plan)
        edges = dp.bin_edges_pc(plan.n_bins, plan.sinb, plan.h_pc)
        logger.debug(
            f"[{self.prefix}] sight-line dust profile ON: {plan.reason}. "
            f"A(d) = A_anchor C(d)/C(anchor) with {plan.n_bins} bins equal in "
            f"the H = {plan.h_pc:g} pc layer's dust (edges "
            f"{', '.join(f'{e:.0f}' for e in edges)} pc), anchored in {band} "
            f"at {where}"
            + (
                f"; near/far R_V split (near stars "
                f"{[system.star.names[i] for i in plan.near_stars]})"
                if plan.split
                else ""
            )
            + ".  sed.distance_constrains_av: false frees every A_V."
        )
        return plan

    # ------------------------------------------------------------------
    # Seeding the anchor from a user's A_V start (stage 3)
    # ------------------------------------------------------------------
    def _seed_dust_anchor(self, system, plan):
        """A user's A_V start on a read star becomes the anchor's start: the
        star's A_V is DERIVED now, so the value it asks for is produced by
        the anchor A_V / (C(d)/C(anchor)) at the start distances and the
        bins' prior center.  A component hint at DERIVED_USER (above the
        clump's data hint), never a user_params write; ``reread_after_starts``
        redoes it once if the solve moved the distances it read."""
        star = system.star
        seen = self.seen_star_mask(system)
        cands = []
        for i in range(int(star.n_elements)):
            if not seen[i]:
                continue
            entry = star._user_entry_for(i, "av")
            if entry is None:
                continue
            val = entry.get("initval", entry.get("mu"))
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                cands.append((i, float(val)))
        self._dust_seed = None
        if not cands:
            return
        far = set()
        for comp in system.active_components.values():
            far |= {int(k) for k in comp.selected_sources(system)}
        cands.sort(key=lambda c: (c[0] not in far, c[0]))
        i, av = cands[0]
        paths = [f"star.{i}.distance"]
        prov = self.anchor_provider()
        if plan.clump:
            paths.append(prov.paths["distance"])
        dist = self._dust_seed_distances
        if dist is None or set(dist) != set(paths):
            probed = self.config_manager.probe_start(paths)
            dist = {}
            for p in paths:
                v = probed[p].user_value
                if v is None:
                    logger.debug(
                        f"[{self.prefix}] {p} has no start yet, so "
                        f"star.{star.names[i]}.av's start does not seed the "
                        f"dust anchor."
                    )
                    return
                dist[p] = float(v)
        shape = float(
            plan.shape_np(
                dist[paths[0]],
                d_anchor=dist[paths[1]] if plan.clump else None,
            )
        )
        av_anchor = av / shape
        value = av_anchor
        if plan.anchor_band is not None:
            value = self.reference_band_extinction_np(
                plan.anchor_band, av_anchor, system
            )
        self._dust_seed = (paths, dist, i, av)
        self.config_manager.add_hint(
            f"{self.prefix}.{DUST_ANCHOR}", value, PRECEDENCE_DERIVED_USER
        )
        logger.debug(
            f"[{self.prefix}] the dust anchor starts at {value:.4g} mag, so "
            f"that star.{star.names[i]}.av starts at the {av:g} asked for "
            f"(distance {dist[paths[0]]:.0f} pc, layer shape {shape:.4g})."
        )

    def reread_after_starts(self, system):
        """The anchor seed read the distances at stage 3 (``probe_start``);
        where the solve moved them, recompute it at the solved start (one
        extra pass, config.md "An estimate that READ a start too early")."""
        seed = getattr(self, "_dust_seed", None)
        if not seed:
            return []
        paths, read, i, av = seed
        solved = self.config_manager.solved_start(paths)
        moved = [
            p
            for p in paths
            if solved[p].user_value is not None
            and not np.isclose(
                float(solved[p].user_value), read[p], rtol=1e-6, atol=0.0
            )
        ]
        if not moved:
            return []
        self._dust_seed_distances = {
            p: float(solved[p].user_value) for p in paths
        }
        what = ", ".join(
            f"{p} {read[p]:.6g} -> {self._dust_seed_distances[p]:.6g}"
            for p in moved
        )
        return [
            (
                f"[{self.prefix}] the dust anchor's start was computed from "
                f"distances the start solve then moved ({what}); recomputed "
                f"so star.{system.star.names[i]}.av starts at {av:g}.",
                [f"{self.prefix}.{DUST_ANCHOR}"] + moved,
            )
        ]

    # ------------------------------------------------------------------
    # The reference spectrum (an anchor band other than V)
    # ------------------------------------------------------------------
    def reference_band_extinction_np(self, band, av, system, rv=None):
        """A_band on the reference spectrum at A_V = ``av`` (numpy)."""
        teff, logg, feh = dp.REFERENCE_SPECTRUM
        if self.fit_rv_dust and rv is None:
            from .sed import RV_DUST

            rv = float(
                np.atleast_1d(
                    self.config_manager.resolve("sed", RV_DUST, element=0)[
                        "initval"
                    ]
                )[0]
            )
        return self.bc_np(band, teff, logg, feh, 0.0, rv) - self.bc_np(
            band, teff, logg, feh, av, rv
        )

    def _anchor_av_from_band(self, a_band, rv):
        """A_V on the reference spectrum whose A_band is ``a_band`` (graph):
        the band-integrated A_band(A_V) inverted by ANCHOR_ITERATIONS
        fixed-point steps A_V <- A_V a_band / A_band(A_V), started from the
        ratio at A_V = 1 (dust_profile.py)."""
        teff, logg, feh = (
            pt.as_tensor_variable(np.array([v], dtype=float))
            for v in dp.REFERENCE_SPECTRUM
        )
        col = self.filter_column(self._dust_plan.anchor_band)
        rv_vec = None if rv is None else pt.reshape(rv, (1,))
        bc0 = self.bc_node(teff, logg, feh, pt.zeros((1,)), rv_vec)[0, col]

        def a_of(av):
            return (
                bc0
                - self.bc_node(teff, logg, feh, pt.reshape(av, (1,)), rv_vec)[
                    0, col
                ]
            )

        av = a_band / a_of(pt.as_tensor_variable(np.float64(1.0)))
        for _ in range(dp.ANCHOR_ITERATIONS):
            av = av * a_band / a_of(av)
        return av

    # ------------------------------------------------------------------
    # Stage 6: the graph builders
    # ------------------------------------------------------------------
    def _dust_param(self, system, name):
        model = pm.modelcontext(None)
        if not Component._parameter_is_current(self, name, model):
            self.add_parameter(model, name, system)
        return getattr(self, name).value

    def _dust_parts(self, system):
        """The profile's shared nodes for THIS build: z, the anchor
        distance's C, and the far R_V (cached per build)."""
        model = pm.modelcontext(None)
        cached = self._dust_parts_cache
        if cached is not None and cached[0] is model:
            return cached[1]
        plan = self._dust_plan
        z = self._dust_param(system, DUST_Z)
        parts = {"z": z}
        if plan.clump:
            prov = self.anchor_provider()
            d_a = self.source_input_node(system, prov.paths["distance"])
            parts["d_anchor"] = d_a
            parts["c_anchor"] = dp.cumulative(
                d_a, z, plan.sinb, plan.h_pc, plan.n_bins
            )
        else:
            parts["d_anchor"] = None
            parts["c_anchor"] = dp.total(z)
        if self.fit_rv_dust:
            from .sed import RV_DUST

            parts["rv_far"] = self._dust_param(system, RV_DUST)[0]
        if plan.split:
            parts["rv_near"] = self._dust_param(system, RV_DUST_NEAR)[0]
            parts["d_boundary"] = self._dust_param(system, DUST_BOUNDARY)[0]
        self._dust_parts_cache = (model, parts)
        return parts

    def anchor_av_node(self, system):
        """A_V at the anchor (scalar): the sampled anchor itself, or under an
        anchor band its A_V on the reference spectrum (``sed.av_anchor``)."""
        if self._dust_plan.anchor_band is None:
            return self._dust_param(system, DUST_ANCHOR)[0]
        return self._dust_param(system, AV_ANCHOR)[0]

    def profile_av_node(self, system, distance):
        """A_V(d) for a vector of distances (pc): the ONE expression every
        star, the flux sources and the figure read."""
        plan = self._dust_plan
        p = self._dust_parts(system)
        c = dp.cumulative(distance, p["z"], plan.sinb, plan.h_pc, plan.n_bins)
        return self.anchor_av_node(system) * c / p["c_anchor"]

    def profile_rv_node(self, system, distance):
        """R_V,eff(d) of the dust in front of each distance: the near/far
        mix under the split, else the sight line's sed.rv_dust broadcast."""
        plan = self._dust_plan
        p = self._dust_parts(system)
        if not plan.split:
            return pt.ones_like(distance) * p["rv_far"]
        f = dp.near_fraction(
            distance,
            p["d_boundary"],
            p["z"],
            plan.sinb,
            plan.h_pc,
            plan.n_bins,
        )
        return dp.mixed_rv(f, p["rv_near"], p["rv_far"])

    def _anchor_rv(self, system):
        """R_V,eff at the anchor (the reference spectrum's law), or None on
        the table path."""
        if not self.fit_rv_dust:
            return None
        p = self._dust_parts(system)
        if not self._dust_plan.split:
            return p["rv_far"]
        if p["d_anchor"] is None:
            # The whole column: f = C(d_b) / C(inf).
            plan = self._dust_plan
            f = (
                dp.cumulative(
                    p["d_boundary"],
                    p["z"],
                    plan.sinb,
                    plan.h_pc,
                    plan.n_bins,
                )
                / p["c_anchor"]
            )
            return dp.mixed_rv(f, p["rv_near"], p["rv_far"])
        return self.profile_rv_node(system, p["d_anchor"])

    def dust_context(self, param_name, system):
        """Context nodes for the SED's own derived dust parameters."""
        if param_name == AV_ANCHOR:
            a_band = self._dust_param(system, DUST_ANCHOR)[0]
            av = self._anchor_av_from_band(a_band, self._anchor_rv(system))
            return {"anchor_av": pt.reshape(av, (1,))}
        return {}

    # ------------------------------------------------------------------
    # The figure: A_V(d) with each SED object's distance marked
    # ------------------------------------------------------------------
    #: The distance grid the figure evaluates the profile on (pc).
    DUST_PLOT_GRID_PC = np.geomspace(30.0, 20000.0, 300)

    def _compile_dust_plotter(self, system, param_symbols, source_vectors):
        """Compile the profile at the plot grid and every object's (d, A_V)
        off the SAME nodes the forward model reads (components.md, "A plot
        never recomputes the model")."""
        import pytensor

        self._compiled_dust = None
        plan = self._dust_plan
        if plan is None or not plan.on:
            return
        grid = pt.as_tensor_variable(self.DUST_PLOT_GRID_PC)
        outs = [
            self.profile_av_node(system, grid),
            source_vectors["distance"],
            source_vectors["av"],
        ]
        if plan.split:
            outs.append(self.profile_rv_node(system, grid))
        self._dust_profile_node = outs[0]
        self._compiled_dust = pytensor.function(
            inputs=param_symbols, outputs=outs, on_unused_input="ignore"
        )

    def _dust_objects(self, system):
        """(label, index) of every object the profile places: the read
        stars, then the flux sources (the source axis of the forward
        model)."""
        seen = self.seen_star_mask(system)
        out = [
            (system.star.names[i], i)
            for i in range(int(system.star.n_elements))
            if seen[i]
        ]
        n = int(system.star.n_elements)
        out += [(p.name, n + k) for k, p in enumerate(self.providers)]
        return out

    def dust_profile_charts(self, system, points):
        """The A_V(d) chart (and, under the near/far split, R_V(d)): the
        posterior median and 68% band of the profile over ``points``, and
        each object's median distance and A_V with their 68% ranges.  One
        point (the GUI) draws its own curve and objects."""
        from exozippy.chart import Chart, Trace

        plan = self._dust_plan
        if plan is None or not plan.on or self._compiled_dust is None:
            return []
        if isinstance(points, dict):
            points = [points]
        evals = [
            self._compiled_dust(*self._point_to_plot_params(p, system))
            for p in points
        ]
        kpc = self.DUST_PLOT_GRID_PC / 1000.0
        a = np.array([np.asarray(e[0], float) for e in evals])
        d = np.array([np.asarray(e[1], float) for e in evals])
        av = np.array([np.asarray(e[2], float) for e in evals])
        deps = self._model_trace_param_deps(self._dust_profile_node, system)
        comp = {"yaml_key": self.prefix, "instance": None}

        def band_traces(y, name, ylabel):
            if len(points) == 1:
                return [
                    Trace(
                        name=name,
                        role="model",
                        kind="line",
                        x=kpc,
                        y=y[0],
                        node=self._dust_profile_node,
                    )
                ]
            lo, med, hi = np.percentile(y, [15.865, 50.0, 84.135], axis=0)
            return [
                Trace(
                    name=f"{name} (median)",
                    role="model",
                    kind="line",
                    x=kpc,
                    y=med,
                    alpha=1.0,
                    style={"color": "k", "lw": 2.0, "legend": True},
                ),
                Trace(
                    name="68% range",
                    role="model",
                    kind="line",
                    x=kpc,
                    y=lo,
                    alpha=0.6,
                    style={"color": "0.4", "lw": 0.8, "legend": True},
                ),
                Trace(
                    name="68% range (upper)",
                    role="model",
                    kind="line",
                    x=kpc,
                    y=hi,
                    alpha=0.6,
                    style={"color": "0.4", "lw": 0.8},
                ),
            ]

        traces = band_traces(a, "A_V(d)", "A_V")
        for k, (label, j) in enumerate(self._dust_objects(system)):
            dj, aj = d[:, j] / 1000.0, av[:, j]
            if len(points) == 1:
                xe = ye = None
                x, y = dj, aj
            else:
                ql, qm, qh = np.percentile(dj, [15.865, 50.0, 84.135])
                al, am, ah = np.percentile(aj, [15.865, 50.0, 84.135])
                x, y = np.array([qm]), np.array([am])
                xe = np.array([[qm - ql], [qh - qm]])
                ye = np.array([[am - al], [ah - am]])
            traces.append(
                Trace(
                    name=label,
                    role="data",
                    kind="scatter",
                    x=x,
                    y=y,
                    xerr=xe,
                    yerr=ye,
                    style={"series_index": k},
                )
            )
        band = plan.anchor_band or "V"
        charts = [
            Chart(
                id=f"{self.prefix}.dust_profile",
                component=comp,
                title="Sight-line dust profile",
                xlabel="Distance [kpc]",
                ylabel="A_V [mag]",
                traces=traces,
                param_deps=deps,
                x_log=True,
                meta={
                    "file_tag": "dust_profile",
                    "caption": (
                        r"The sight-line extinction profile $A_V(d)$ "
                        r"(posterior median and 68\% range), "
                        rf"{plan.n_bins} bins of equal dust in an "
                        rf"exponential layer of scale height "
                        rf"{plan.h_pc:g}~pc, anchored in {band} at "
                        + (
                            "the red clump"
                            if plan.clump
                            else "the whole column"
                        )
                        + r"; each point is an SED object at its distance."
                    ),
                },
            )
        ]
        if plan.split:
            r = np.array([np.asarray(e[3], float) for e in evals])
            charts.append(
                Chart(
                    id=f"{self.prefix}.dust_rv",
                    component=comp,
                    title="R_V of the dust in front of distance d",
                    xlabel="Distance [kpc]",
                    ylabel="R_V,eff",
                    traces=band_traces(r, "R_V,eff(d)", "R_V"),
                    param_deps=deps,
                    x_log=True,
                    meta={"file_tag": "dust_rv"},
                )
            )
        return charts

    def plot_dust_profile(self, system, points, filename_prefix):
        """Write ``<prefix>_dust_profile.pdf`` (and ``_dust_rv.pdf`` under
        the split) through plotrender, the PDFs' one renderer."""
        from exozippy.plotrender import render_spec_groups

        charts = self.dust_profile_charts(system, points)
        if charts:
            render_spec_groups([charts], filename_prefix=filename_prefix)

    def _add_dust_prose(self, system):
        """The draft's sentences for the profile (and the split)."""
        from exozippy.outputs.prose import get_collector

        plan = self._dust_plan
        if plan is None or not plan.on:
            return
        band = plan.anchor_band or "V"
        anchor = (
            "the red clump's distance"
            if plan.clump
            else "the far end of the sight line (the whole column)"
        )
        prose = get_collector(system)
        prose.add(
            r"The extinction of every object in the SED is one monotone "
            r"profile $A_V(d)$ of distance along the sight line, in the "
            rf"spirit of \citet{{Green:2014}}: {plan.n_bins} bins, each "
            rf"holding an equal share of the dust of an exponential layer "
            rf"of scale height {plan.h_pc:g}~pc \citep{{Nataf:2013}}, whose "
            r"log increments scatter about that layer by 0.8~dex, anchored "
            rf"by the extinction in {band} at {anchor}, which is sampled "
            r"directly.  Each object's $A_V$ is the profile at its distance "
            r"times $1 + \delta$, a per-object scatter "
            r"\citep{Sale:2012} of 3\%"
            + (
                " (9\\% for the red clump's centroid, \\citealt{Nataf:2013})"
                if plan.clump
                else ""
            )
            + ". ",
            section="priors",
            key=f"{self.prefix}.dust_profile",
            rank=20.5,
        )
        if plan.split:
            prose.add(
                r"The dust in front of a sampled boundary (3--6~kpc) has its "
                r"own $R_V$, with the diffuse-ISM prior of "
                r"\citet{Schlafly:2016} mapped into the extinction law used "
                r"here ($2.88 \pm 0.19$); because the law is linear in "
                r"$1/R_V$, each object sees the $A_V$-weighted mean of "
                r"$1/R_V$ in front of it.  This makes the law consistent "
                r"along the sight line within one law family, which cannot "
                r"match the bulge's optical and near-infrared extinction "
                r"ratios at once. ",
                section="priors",
                key=f"{self.prefix}.dust_split",
                rank=20.55,
            )
