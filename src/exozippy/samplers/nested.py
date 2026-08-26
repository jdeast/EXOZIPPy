"""Nested sampling on the full EXOZIPPy model (sampler config `method: nested`).

Why this exists, measured on DC2018 event 128 (27 sampled elements, binary
microlensing, no usable gradients): blind nested sampling on the 7-parameter
profile likelihood found the full solution -- including the s <-> 1/s
geometry -- from generic priors at ~43 core-hours, where the PTDE run that
was seeded AT the solution cost ~1,100 core-hours and delivered mode weights
its own report flagged as initialization artifacts (PT round-trip transport
collapses for Lambda >~ 5; notes/pt_round_trip_collapse.txt).  Nested
sampling needs no temperature transport: it sweeps one live-point population
from the prior to the posterior, discovers basins as clusters on the way,
weights them by MASS (volume included, not peak height), and its dead points
are a native record of explored-and-rejected structure.

THE BRIDGE.  A nested sampler wants a unit-cube prior transform and a
likelihood.  Every sampled element of an EXOZIPPy build is logit-bounded
("every sampled parameter must have lower and upper"), i.e.

    physical = lower + span * sigmoid(c + s * raw)

so uniform-in-physical is exactly q = u, and raw(u) = (logit(u) - c) / s.
The constants are recovered NUMERICALLY per element from the model's own
compiled raw -> physical graph -- probes at raw = 0, +/-1, +/-40, one
element at a time -- never from Parameter internals, and the recovered
transform is verified by round-trip at random u before any sampling.  An
element that is not pure static-bounds logit (e.g. a dynamic linked bound,
whose interval is another parameter's value) fails that verification and
raises with the element named: such configs cannot use `method: nested` yet.

The sampler runs in u-space: prior = the unit cube, and the "likelihood" is
the u-space density of the FULL posterior kernel,

    log L_u(u) = model.logp(raw(u)) + sum_k [ -log|s_k| - log(u_k(1-u_k)) ]

(the |d raw / d u| Jacobian; the span factors are constants).  The
per-element uniform priors are the instrumental prior; everything else the
model states -- Gaussian priors, galactic-model potentials, the volume
prior, robust likelihoods -- rides along inside L_u.  That decomposition is
exact over the bounded support, so logZ and relative mode masses are the
model's own.  Whitening never enters: it only defines the raw coordinates
the bridge converts through, self-consistently by construction.

BACKENDS.  `nested_backend: dynesty` (default) or `ultranest`.  dynesty uses
its own worker pool (the compiled logp is installed in module globals BEFORE
the fork, the same copy-on-write trick _common.set_worker_globals documents).
ultranest is driven in vectorized mode, with the batch fanned across the
same pool.  MultiNest is deliberately not wired: it needs a Fortran build
and offers nothing these two do not.
"""

import logging
import multiprocessing as mp
import time

import numpy as np

from . import _common

logger = logging.getLogger(__name__)

# Worker-inherited state (copy-on-write through fork; see
# _common.set_worker_globals for the pattern and why it must be set BEFORE
# the pool is created).
_NB = {}


class NestedBridgeError(RuntimeError):
    """The model has an element the unit-cube bridge cannot represent."""


class UnitCubeBridge:
    """Numeric unit-cube <-> raw bridge for an EXOZIPPy-built model."""

    def __init__(self, model, sampled_masks=None):
        (
            self._raw_to_phys,
            self.raw_to_phys_batched,
            self.raw_names,
            self.out_names,
        ) = _common.compile_conversions(model)
        ip = model.initial_point()
        self.ip_shapes = {v: np.asarray(ip[v]).shape for v in self.raw_names}
        self.sizes = [int(np.asarray(ip[v]).size) for v in self.raw_names]
        self.ndim = int(sum(self.sizes))
        self.flat_names = []
        for v, n in zip(self.raw_names, self.sizes):
            self.flat_names += [f"{v}[{j}]" for j in range(n)]

        def phys_dict(raw_flat):
            args, ofs = [], 0
            for v, n in zip(self.raw_names, self.sizes):
                args.append(
                    np.asarray(raw_flat[ofs : ofs + n], dtype=float).reshape(
                        self.ip_shapes[v]
                    )
                )
                ofs += n
            outs = self._raw_to_phys(*args)
            return dict(zip(self.out_names, outs))

        # RAW -> PHYSICAL ELEMENT MAP, probed numerically.  A raw vector
        # holds only the SAMPLED elements of its parameter; the physical
        # vector holds ALL elements (pinned, derived, inactive included), so
        # flat raw index k is NOT physical flat index k.  The first version
        # assumed it was and read back pinned Source-star constants for the
        # Lens's kinematics ("non-positive span" on star.pm_ra).  Instead:
        # kick each raw element by +3 and find which element of its base
        # physical vector moved.  A kick that moves NO element, or more than
        # one (a same-parameter ordering link), is a model this bridge
        # cannot represent -- refuse it by name.
        z0 = np.zeros(self.ndim)
        P0 = phys_dict(z0)
        self._col_map = []
        ofs = 0
        for v, n in zip(self.raw_names, self.sizes):
            base = v[:-4] if v.endswith("_raw") else v
            if base not in P0:
                base = v
            ref = np.asarray(P0[base], dtype=float).ravel()
            for j in range(n):
                e = z0.copy()
                e[ofs + j] = 3.0
                moved = np.abs(
                    np.asarray(phys_dict(e)[base], dtype=float).ravel() - ref
                )
                hits = np.where(moved > 1e-9 * np.maximum(np.abs(ref), 1.0))[0]
                if hits.size > 1 and sampled_masks and base in sampled_masks:
                    # Same-parameter derived elements (the surgical swaps:
                    # pm[lens] = pm[source] + mu_rel) legitimately co-move
                    # with the sampled element that drives them.  The raw
                    # coordinate CONTROLS its sampled element; the derived
                    # movers are reconstructed by the physical graph either
                    # way, so the map keeps the sampled one.  Role masks
                    # come from the caller (nested_sample has the System);
                    # a hand-built model without them keeps the strict
                    # exactly-one contract below.
                    mask = np.asarray(sampled_masks[base], dtype=bool).ravel()
                    sampled_hits = hits[mask[hits]]
                    if sampled_hits.size == 1:
                        hits = sampled_hits
                if hits.size != 1:
                    raise NestedBridgeError(
                        f"raw element {v}[{j}] moves {hits.size} elements "
                        f"of '{base}' (need exactly 1); this parameter has "
                        "linked or coupled elements the unit-cube bridge "
                        "cannot represent"
                    )
                self._col_map.append((base, int(hits[0])))
            ofs += n

        def phys_at(raw_flat):
            d = phys_dict(raw_flat)
            return np.array(
                [
                    float(np.asarray(d[b], dtype=float).ravel()[i])
                    for b, i in self._col_map
                ]
            )

        self._phys_at = phys_at
        z = np.zeros(self.ndim)
        p0 = phys_at(z)

        # ELEMENT CLASSIFICATION, probed numerically.  Not every sampled
        # element is logit-bounded: the build also carries raw-Normal-prior
        # elements -- Gaussian/unbounded kinematics (pm_ra, pm_dec, rv) and
        # log-normal error scales -- whose physical transform never
        # saturates.  Fitting a logit to those is what produced c ~ 0,
        # s ~ 0.05 and raw(u) of +/-50 on the first d=27 pilot: a wrong
        # prior, caught by verify().  The discriminator is SATURATION: a
        # logit element is frozen between raw = 40 and raw = 80; a
        # Normal-prior element keeps moving.  A Normal-prior element needs
        # NO recovered constants at all -- its prior IS the raw N(0,1), so
        # raw = Phi^-1(u) exactly, whatever the physical transform is.
        probes = {}
        for r in (-80.0, -40.0, 40.0, 80.0):
            vals = np.empty(self.ndim)
            for k in range(self.ndim):
                e = z.copy()
                e[k] = r
                vals[k] = phys_at(e)[k]
            probes[r] = vals
        inner = np.abs(probes[40.0] - probes[-40.0])
        outer = np.abs(probes[80.0] - probes[40.0]) + np.abs(
            probes[-40.0] - probes[-80.0]
        )
        if not np.all(inner > 0):
            bad = [self.flat_names[k] for k in np.where(~(inner > 0))[0]]
            raise NestedBridgeError(
                f"transform is flat over raw +/-40 on {bad[:5]}"
            )
        self.is_logit = outer < 1e-9 * inner

        lower = np.where(
            self.is_logit,
            np.minimum(probes[-40.0], probes[40.0]),
            np.nan,
        )
        upper = np.where(
            self.is_logit,
            np.maximum(probes[-40.0], probes[40.0]),
            np.nan,
        )
        span = upper - lower
        c = np.zeros(self.ndim)
        s_arr = np.ones(self.ndim)
        for k in np.where(self.is_logit)[0]:
            q0 = np.clip((p0[k] - lower[k]) / span[k], 1e-12, 1 - 1e-12)
            c[k] = np.log(q0 / (1 - q0))
            e = z.copy()
            e[k] = 1.0
            qk = np.clip(
                (phys_at(e)[k] - lower[k]) / span[k], 1e-12, 1 - 1e-12
            )
            s_arr[k] = np.log(qk / (1 - qk)) - c[k]
            if not np.isfinite(s_arr[k]) or s_arr[k] == 0:
                raise NestedBridgeError(
                    f"degenerate logit scale on {self.flat_names[k]}"
                )
        self.lower, self.span, self.c, self.s = lower, span, c, s_arr
        n_logit = int(self.is_logit.sum())
        logger.info(
            f"UnitCubeBridge: {n_logit} logit-bounded and "
            f"{self.ndim - n_logit} raw-Normal-prior element(s)"
        )

    def raw_from_u(self, u):
        from scipy.special import ndtri

        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
        raw = ndtri(u)  # Normal-prior elements: raw = Phi^-1(u), exact
        m = self.is_logit
        raw[m] = (np.log(u[m] / (1 - u[m])) - self.c[m]) / self.s[m]
        return raw

    def point_from_raw(self, raw):
        point, ofs = {}, 0
        for v, n in zip(self.raw_names, self.sizes):
            point[v] = raw[ofs : ofs + n].reshape(self.ip_shapes[v])
            ofs += n
        return point

    def log_jac(self, u):
        """log |d raw / d u| at u, per element class.

        Logit: 1 / (s * u(1-u)).  Normal-prior: 1 / phi(Phi^-1(u)) =
        raw^2/2 + log sqrt(2 pi) -- exactly what cancels the raw N(0,1)
        prior inside model.logp, leaving that element's likelihood share.
        """
        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
        raw = self.raw_from_u(u)
        out = 0.5 * raw**2 + 0.5 * np.log(2 * np.pi)
        m = self.is_logit
        out[m] = -np.log(np.abs(self.s[m])) - (np.log(u[m]) + np.log1p(-u[m]))
        return float(np.sum(out))

    def verify(self, n=5, seed=1, rtol=1e-6):
        """Round-trip the LOGIT elements at random u; raise on mismatch.

        Normal-prior elements are exact by construction (raw = Phi^-1(u)
        uses no recovered constants), so only the logit elements' (c, s,
        lower, span) recovery is checkable -- and only there can a coupled
        or non-static support hide.
        """
        if not np.any(self.is_logit):
            return
        rng = np.random.default_rng(seed)
        m = self.is_logit
        for _ in range(n):
            u = rng.uniform(0.02, 0.98, self.ndim)
            got = self._phys_at(self.raw_from_u(u))[m]
            want = (self.lower + self.span * u)[m]
            rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-9)
            if np.max(rel) > rtol:
                k = np.where(m)[0][int(np.argmax(rel))]
                raise NestedBridgeError(
                    f"transform round-trip failed (rel err "
                    f"{np.max(rel):.2e}, worst element "
                    f"{self.flat_names[k]}); an element's support is not "
                    "static, so `method: nested` cannot sample this model "
                    "yet"
                )


def _identity_transform(u):
    return np.asarray(u, dtype=float)


def _loglike_u(u):
    """u-space posterior kernel density (module-level: pool workers call it
    with the bridge + compiled logp inherited through fork)."""
    bridge = _NB["bridge"]
    logp_fn = _NB["logp_fn"]
    raw = bridge.raw_from_u(u)
    try:
        lp = float(logp_fn(bridge.point_from_raw(raw)))
    except Exception:
        return -1e300
    if not np.isfinite(lp):
        return -1e300
    return lp + bridge.log_jac(u)


def _loglike_u_batch(U):
    pool = _NB.get("pool")
    rows = list(np.asarray(U, dtype=float))
    if pool is None:
        return np.array([_loglike_u(r) for r in rows])
    return np.array(pool.map(_loglike_u, rows))


def _transform_batch(U):
    return np.asarray(U, dtype=float)


def nested_sample(
    model,
    system,
    backend="dynesty",
    nlive=500,
    dlogz=0.5,
    walks=None,
    cores=None,
    seed=None,
    maxiter=None,
    n_pseudo_chains=4,
    checkpoint_dir=None,
):
    """Run nested sampling and return an arviz.InferenceData.

    The posterior group holds EQUAL-WEIGHT resampled draws converted to
    physical space through the same machinery the PT samplers use, arranged
    as `n_pseudo_chains` pseudo-chains so every downstream consumer (mode
    report, tables, plots, trace persistence) works unchanged.  The full
    WEIGHTED run -- u-space samples, log-likelihoods, log-weights -- is
    attached as an extra `nested` group, because the equal-weight resample
    is lossy exactly where nested sampling is strongest: dead points in
    rejected basins carry the explored-and-rejected record and the mass
    integrals.  posterior attrs carry logz/logzerr/ncall/backend.
    """
    t0 = time.time()
    sampled_masks = None
    if system is not None:
        # Per-parameter sampled-element masks, so the raw->physical probe
        # can disambiguate same-parameter derived elements (see _col_map).
        sampled_masks = {}
        for comp in system.active_components.values():
            for pname in comp.manifest or {}:
                param = getattr(comp, pname, None)
                if param is None or not hasattr(param, "element_is_sampled"):
                    continue
                n = param._n_elements()
                sampled_masks[f"{comp.prefix}.{pname}"] = np.array(
                    [bool(param.element_is_sampled(i)) for i in range(n)]
                )
    bridge = UnitCubeBridge(model, sampled_masks=sampled_masks)
    bridge.verify()
    logp_fn = model.compile_logp()
    _NB.update(bridge=bridge, logp_fn=logp_fn, pool=None)

    phys_cores = mp.cpu_count()
    # default_cores(), not a third hand-written 0.75: this copy had dropped
    # the `phys - 1` arm, so an unconfigured nested run took every core the
    # OS and the user's shell were meant to keep one of.
    actual = max(1, min(cores or _common.default_cores(), phys_cores))
    pool = mp.Pool(actual) if actual > 1 else None
    _NB["pool"] = pool
    logger.info(
        f"nested[{backend}]: ndim={bridge.ndim} nlive={nlive} "
        f"dlogz={dlogz} cores={actual}"
    )

    try:
        if backend == "dynesty":
            import dynesty

            sampler = dynesty.NestedSampler(
                _loglike_u,
                _identity_transform,
                ndim=bridge.ndim,
                nlive=nlive,
                sample="rwalk",
                walks=walks or max(25, 2 * bridge.ndim),
                rstate=np.random.default_rng(seed),
                pool=pool,
                queue_size=actual if pool is not None else None,
            )
            dy_kwargs = {}
            if checkpoint_dir:
                import os as _os
                from pathlib import Path as _Path

                _os.makedirs(str(checkpoint_dir), exist_ok=True)
                dy_kwargs = {
                    "checkpoint_file": str(
                        _Path(checkpoint_dir) / "dynesty.ckpt"
                    )
                }
            sampler.run_nested(
                **dy_kwargs, dlogz=dlogz, maxiter=maxiter, print_progress=False
            )
            res = sampler.results
            U = np.asarray(res.samples)
            logl = np.asarray(res.logl)
            logwt = np.asarray(res.logwt)
            logz = float(res.logz[-1])
            logzerr = float(res.logzerr[-1])
            ncall = int(np.sum(res.ncall))
        elif backend == "ultranest":
            import ultranest
            import ultranest.stepsampler

            un_kwargs = {}
            if checkpoint_dir:
                # log_dir + resume makes SIGTERM survivable: a resubmitted
                # job continues from the stored live points, and a
                # truncated run still holds a valid logZ lower bound and
                # the dead-point record.  The first d=27 pilot predated
                # this and burned ~2.9 days unrecoverably.
                un_kwargs = {"log_dir": str(checkpoint_dir), "resume": True}
            sampler = ultranest.ReactiveNestedSampler(
                bridge.flat_names,
                _loglike_u_batch,
                transform=_transform_batch,
                vectorized=True,
                **un_kwargs,
            )
            if bridge.ndim >= 15:
                # Region sampling degrades in high d; slice stepping is
                # ultranest's own recommendation there -- but the plain
                # SliceSampler is SEQUENTIAL (each step depends on the
                # previous accept/reject), so it hands the vectorized
                # likelihood batches of size ~1 and starves the worker
                # pool: the first d=27 pilot ran at 1/64 = 1.6% CPU on a
                # 64-slot node for 2.9 days.  The population variant
                # advances popsize walkers concurrently, producing real
                # batches sized to keep every pool worker busy.
                import ultranest.popstepsampler as _pss

                sampler.stepsampler = _pss.PopulationSliceSampler(
                    popsize=max(2 * actual, 8),
                    nsteps=2 * bridge.ndim,
                    generate_direction=(
                        _pss.generate_mixture_random_direction
                    ),
                )
            res = sampler.run(
                min_num_live_points=nlive,
                dlogz=dlogz,
                max_iters=maxiter,
                show_status=False,
                viz_callback=False,
            )
            wp = res["weighted_samples"]
            U = np.asarray(wp["points"])
            logl = np.asarray(wp["logl"])
            w = np.asarray(wp["weights"])
            logwt = np.log(np.maximum(w, 1e-300))
            logz = float(res["logz"])
            logzerr = float(res["logzerr"])
            ncall = int(res.get("ncall", -1))
        else:
            raise ValueError(
                f"unknown nested_backend '{backend}' (dynesty | ultranest)"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _NB["pool"] = None

    wall = time.time() - t0
    logger.info(
        f"nested[{backend}]: logZ = {logz:.2f} +/- {logzerr:.2f}  "
        f"ncall={ncall}  wall={wall / 3600:.2f} h"
    )

    # TRUTH-FREE PEAK-MISS DIAGNOSTIC.  For any d-dimensional posterior with
    # a quadratic peak, max(logl) stands ~ H + d/2 above logZ (H = the
    # prior-to-posterior compression the run itself estimates as
    # iterations/nlive).  A gap far BELOW d/2 is impossible for a genuinely
    # explored peak: it means the live points plateaued on a shoulder, the
    # dlogz criterion saw nothing above them, and the run terminated cleanly
    # around a peak it never entered.  Measured on DC2018 event 128 at
    # d = 27: gap = 7.2 nats against a 13.5-nat floor and an ~80-nat
    # expectation -- the run had silently stopped 1,319 nats below the known
    # optimum, with a plausible-looking logZ and error bar.  The healthy
    # d = 7 benchmark gives gap = 54.8 vs H + d/2 ~ 55.
    gap = float(np.max(logl) - logz)
    n_iter = max(len(logl) - nlive, 1)
    h_est = n_iter / max(nlive, 1)
    if gap < 0.5 * (h_est + 0.5 * bridge.ndim) or gap < 0.5 * bridge.ndim:
        logger.warning(
            f"nested[{backend}]: max(logl) - logZ = {gap:.1f} nats, far "
            f"below the ~H + d/2 = {h_est:.0f} + {bridge.ndim / 2:.1f} a "
            "resolved peak implies. The sampler most likely terminated on a "
            "SHOULDER without entering the posterior's core: treat logZ as "
            "a lower bound and the posterior moments as suspect. Remedies: "
            "a step sampler that penetrates sharp cores "
            "(nested_backend: ultranest), more walks, or larger nlive."
        )

    # Equal-weight resample -> pseudo-chains -> physical posterior group.
    rng = np.random.default_rng(seed)
    w = np.exp(logwt - logwt.max())
    w /= w.sum()
    n_eff = int(1.0 / np.sum(w**2))
    draws = max(100, n_eff // n_pseudo_chains)
    idx = rng.choice(U.shape[0], size=n_pseudo_chains * draws, p=w)
    raw_rows = np.array([bridge.raw_from_u(U[i]) for i in idx])
    jacs = np.array([bridge.log_jac(U[i]) for i in idx])
    lp_rows = logl[idx] - jacs  # back to the model's own logp scale

    stored_raw = {}
    ofs = 0
    for v, n in zip(bridge.raw_names, bridge.sizes):
        stored_raw[v] = raw_rows[:, ofs : ofs + n].reshape(
            (n_pseudo_chains, draws) + bridge.ip_shapes[v]
        )
        ofs += n
    stored_lp = lp_rows.reshape(n_pseudo_chains, draws)
    raw_start = {v: np.zeros(bridge.ip_shapes[v]) for v in bridge.raw_names}
    idata = _common.assemble_inference_data(
        stored_raw,
        stored_lp,
        draws,
        n_pseudo_chains,
        raw_start,
        bridge.raw_names,
        bridge.out_names,
        bridge.raw_to_phys_batched,
        np.zeros(n_pseudo_chains, dtype=int),
        f"nested[{backend}]",
        logger,
    )
    idata.posterior.attrs["nested_backend"] = backend
    idata.posterior.attrs["nested_logz"] = logz
    idata.posterior.attrs["nested_logzerr"] = logzerr
    idata.posterior.attrs["nested_ncall"] = ncall
    idata.posterior.attrs["nested_nlive"] = int(nlive)
    idata.posterior.attrs["nested_n_eff"] = n_eff

    # The full weighted run, for mass-accurate mode reporting and the
    # rejected-mode ledger (dead points ARE the explored-and-rejected set).
    import xarray as xr

    # DataTree-style group attach, same idiom as ptde_async's posterior_hot.
    idata["nested"] = xr.Dataset(
        {
            "samples_u": (("point", "dim"), U),
            "logl": (("point",), logl),
            "logwt": (("point",), logwt),
        },
        coords={
            "point": np.arange(U.shape[0]),
            "dim": bridge.flat_names,
        },
    )
    return idata
