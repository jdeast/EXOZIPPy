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

    def __init__(self, model):
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
        # Bounds probed PER ELEMENT (others at the raw origin): pushing every
        # element to +/-40 at once inverts intervals whenever a transform is
        # decreasing or elements couple, which is exactly how the first
        # version of this died.
        lower = np.empty(self.ndim)
        upper = np.empty(self.ndim)
        for k in range(self.ndim):
            e = z.copy()
            e[k] = -40.0
            a = phys_at(e)[k]
            e[k] = 40.0
            b = phys_at(e)[k]
            lower[k], upper[k] = min(a, b), max(a, b)
        span = upper - lower
        if not np.all(span > 0) or not np.all(np.isfinite(span)):
            bad = [
                self.flat_names[k]
                for k in np.where(~(span > 0) | ~np.isfinite(span))[0]
            ]
            raise NestedBridgeError(
                f"non-positive or non-finite support on {bad[:5]}; these "
                "elements are not static-bounds logit and `method: nested` "
                "cannot sample them yet"
            )
        q0 = np.clip((p0 - lower) / span, 1e-12, 1 - 1e-12)
        c = np.log(q0 / (1 - q0))
        s = np.empty(self.ndim)
        for k in range(self.ndim):
            e = z.copy()
            e[k] = 1.0
            qk = np.clip(
                (phys_at(e)[k] - lower[k]) / span[k], 1e-12, 1 - 1e-12
            )
            s[k] = np.log(qk / (1 - qk)) - c[k]
        if not np.all(np.isfinite(s)) or np.any(s == 0):
            bad = [
                self.flat_names[k]
                for k in range(self.ndim)
                if not np.isfinite(s[k]) or s[k] == 0
            ]
            raise NestedBridgeError(f"degenerate logit scale on {bad[:5]}")
        self.lower, self.span, self.c, self.s = lower, span, c, s

    def raw_from_u(self, u):
        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
        return (np.log(u / (1 - u)) - self.c) / self.s

    def point_from_raw(self, raw):
        point, ofs = {}, 0
        for v, n in zip(self.raw_names, self.sizes):
            point[v] = raw[ofs : ofs + n].reshape(self.ip_shapes[v])
            ofs += n
        return point

    def log_jac(self, u):
        """log |d raw / d u| at u (span factors are constants; dropped)."""
        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
        return float(
            -np.sum(np.log(np.abs(self.s))) - np.sum(np.log(u) + np.log1p(-u))
        )

    def verify(self, n=5, seed=1, rtol=1e-6):
        """Round-trip the transform at random u; raise on any mismatch.

        This is the guard that catches coupled or non-logit elements the
        per-element probes cannot see (a dynamic bound read from another
        parameter changes when that parameter moves).
        """
        rng = np.random.default_rng(seed)
        for _ in range(n):
            u = rng.uniform(0.02, 0.98, self.ndim)
            got = self._phys_at(self.raw_from_u(u))
            want = self.lower + self.span * u
            err = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9))
            if err > rtol:
                k = int(
                    np.argmax(
                        np.abs(got - want) / np.maximum(np.abs(want), 1e-9)
                    )
                )
                raise NestedBridgeError(
                    f"transform round-trip failed (rel err {err:.2e}, worst "
                    f"element {self.flat_names[k]}); an element's support "
                    "is not static, so `method: nested` cannot sample this "
                    "model yet"
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
    bridge = UnitCubeBridge(model)
    bridge.verify()
    logp_fn = model.compile_logp()
    _NB.update(bridge=bridge, logp_fn=logp_fn, pool=None)

    phys_cores = mp.cpu_count()
    actual = max(1, min(cores or max(1, int(phys_cores * 0.75)), phys_cores))
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
            sampler.run_nested(
                dlogz=dlogz, maxiter=maxiter, print_progress=False
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

            sampler = ultranest.ReactiveNestedSampler(
                bridge.flat_names,
                _loglike_u_batch,
                transform=_transform_batch,
                vectorized=True,
            )
            if bridge.ndim >= 15:
                # Region sampling degrades in high d; slice steps are
                # ultranest's own recommendation there.
                sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=2 * bridge.ndim,
                    generate_direction=(
                        ultranest.stepsampler.generate_mixture_random_direction
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
