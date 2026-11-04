"""Importance sampling of the EXOZIPPy prior on one sight line.

WHY NOT MCMC.  The rate-weighted lens mass function has two modes in log
mass: the luminous stars and white dwarfs below ~2 Msun and the black holes
above ~4 Msun, separated by a trough near 3 Msun that is ~19 dex deep (the
present-day mass function has no objects there; it is floored at 1e-20 of
its peak).  NUTS cannot cross it: on Baade's window, 64 chains x 3000 draws
(R-hat 1.015) held NOT ONE draw above 3 Msun, where the prior puts 1.5% of
the event rate, and that missing tail shortens t_E's q90 / q95 by ~4% / 7%.
R-hat cannot see a mode no chain finds.  Importance sampling evaluates the
prior's density directly, so it has no such blind spot, and its error is
checkable (effective sample size, largest weight, Pareto k).

THE ESTIMATE.  Target p(x): the model's own compiled logp in its raw
coordinates (every prior term, potential and Jacobian -- exactly the density
a fit samples when the light curve is uninformative).  Proposal q(x) =
q_rest(x without the lens log mass) x q_m(lens log mass):

* q_rest: a mixture of multivariate Student-t components (df 5, the fitted
  Gaussian covariances inflated by 1.5^2), fit by EM to a short NUTS run,
  then refit to the importance-resampled draws for a few adaptive rounds
  (the pilot's fit -- itself 20% one broad component with the pilot's
  covariance x 3^2 -- keeps a 20% defensive share in every round; at most
  one component per 50 points fit);
* q_m: piecewise constant on a log-mass grid spanning the whole mass
  function (10^-3.3 to 10^2.3 Msun; K21's IMF spans 1e-3 - 120 Msun and
  its most massive remnants are ~16 Msun, so outside the grid the prior is
  at its 1e-20 floor and is not proposed), mixing (85%) the average over pilot
  draws of the normalized conditional profile p(m | rest) -- which sees the
  black-hole mode because it is computed on the grid, not sampled -- with
  (15%) a density uniform in log mass, so no part of the support is
  under-covered.

After the first round, half of the proposal is instead a Student-t mixture
over ALL coordinates fit to the resampled draws, which carries the
mass-velocity correlation of kicked neutron-star lenses that the product
form cannot; and in every round 20% of the proposal draws the lens's
distance and velocity coordinates three times wider, for the kicked neutron
stars (350 km/s) and black holes (100 km/s) and for the nearest lenses.
An under-covered region shows up in no diagnostic -- it simply has no
draws -- so these defensive shares, not the diagnostics, are what guard the
tails; the comparison with genulens on 15 lines is the check that they do.  The final draw uses the proposal of the
adaptive round with the highest ESS fraction.

Weights w = p / q; every quantile is a weighted quantile, and its error a
weighted bootstrap (``stats.bootstrap_quantile_se``).  The weights are NOT
Pareto-smoothed: smoothing caps exactly the heavy draws that mark an
under-covered region (on one BLG521 run it moved D_L's q05 by +2.5%), so it
trades a visible variance for an invisible bias.
"""

import multiprocessing as mp
import time

import numpy as np

from . import stats

DF = 5.0
INFLATE = 1.5
LOGM_GRID = (-3.3, 2.3, 0.01)  # log10 Msun: lower, upper, step
UNIFORM_SHARE = 0.15
KICK_SHARE, KICK_WIDEN = 0.2, 3.0  # the kicked-lens proposal share
BROAD_SHARE, BROAD_WIDEN = 0.2, 3.0  # the broad component in the base
POINTS_PER_COMPONENT = 50
DEFENSIVE = 0.2  # share of the pilot-fit mixture kept in later rounds

_PRIOR = None  # the PriorModel, inherited by forked workers


def _eval_chunk(chunk):
    return np.array([_PRIOR.evaluate(v) for v in chunk])


def evaluate_many(prior, X, cores):
    """Rows [logp, derived...] for every row of X, over ``cores`` processes."""
    global _PRIOR
    _PRIOR = prior
    if cores <= 1:
        return _eval_chunk(X)
    chunks = np.array_split(X, max(cores * 8, 1))
    with mp.get_context("fork").Pool(cores) as pool:
        out = pool.map(_eval_chunk, chunks)
    return np.concatenate(out, axis=0)


class TMixture:
    """Mixture of multivariate Student-t densities (common df)."""

    def __init__(self, weights, means, covs, df=DF):
        self.weights = np.asarray(weights, float) / np.sum(weights)
        self.means = np.asarray(means, float)
        self.chol = np.linalg.cholesky(np.asarray(covs, float))
        self.df = float(df)
        self.dim = self.means.shape[1]

    def sample(self, n, rng):
        k = rng.choice(self.weights.size, size=n, p=self.weights)
        z = rng.standard_normal((n, self.dim))
        u = rng.chisquare(self.df, size=n) / self.df
        x = np.empty((n, self.dim))
        for j in np.unique(k):
            sel = k == j
            x[sel] = (
                self.means[j]
                + (z[sel] @ self.chol[j].T) / np.sqrt(u[sel])[:, None]
            )
        return x

    def logpdf(self, x):
        from scipy.special import gammaln, logsumexp

        x = np.atleast_2d(x)
        d, nu = self.dim, self.df
        const = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
        )
        comp = np.empty((x.shape[0], self.weights.size))
        for j in range(self.weights.size):
            sol = np.linalg.solve(self.chol[j], (x - self.means[j]).T)
            maha = np.sum(sol**2, axis=0)
            logdet = np.sum(np.log(np.diag(self.chol[j])))
            comp[:, j] = (
                np.log(self.weights[j])
                + const
                - logdet
                - 0.5 * (nu + d) * np.log1p(maha / nu)
            )
        return logsumexp(comp, axis=1)


def fit_tmixture(X, n_components, rng, max_fit=40000):
    """EM Gaussian mixture on (standardized) X -> Student-t mixture.

    At most one component per POINTS_PER_COMPONENT points: a component fit
    to a handful of points is a spike, and a proposal made of spikes has
    worse coverage than a broad one.
    """
    from sklearn.mixture import GaussianMixture

    n_components = int(
        np.clip(X.shape[0] // POINTS_PER_COMPONENT, 1, n_components)
    )
    if X.shape[0] > max_fit:
        X = X[rng.choice(X.shape[0], max_fit, replace=False)]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=int(rng.integers(2**31)),
        max_iter=500,
    ).fit((X - mu) / sd)
    means = mu + gm.means_ * sd
    covs = gm.covariances_ * np.outer(sd, sd)[None] * INFLATE**2
    return TMixture(gm.weights_, means, covs)


class GridProposal:
    """Piecewise-constant density on raw-coordinate bin edges."""

    def __init__(self, edges, probs):
        self.edges = np.asarray(edges, float)
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("grid edges must increase")
        self.probs = np.asarray(probs, float) / np.sum(probs)
        self.logdens = np.log(self.probs) - np.log(np.diff(self.edges))

    def sample(self, n, rng):
        k = rng.choice(self.probs.size, size=n, p=self.probs)
        return rng.uniform(self.edges[k], self.edges[k + 1])

    def logpdf(self, x):
        x = np.asarray(x, float)
        k = np.searchsorted(self.edges, x, side="right") - 1
        inside = (k >= 0) & (k < self.probs.size)
        out = np.full(x.shape, -np.inf)
        out[inside] = self.logdens[k[inside]]
        return out


def logmass_proposal(prior, pilot, cores, rng, n_profile=256):
    """q_m: averaged conditional profile + uniform in log mass."""
    lo, hi, step = LOGM_GRID
    logm_edges = np.arange(lo, hi + 0.5 * step, step)
    edges = prior.logmass_raw_from_phys(logm_edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)
    rest = pilot[rng.choice(pilot.shape[0], n_profile, replace=False)]
    X = np.repeat(rest, centers.size, axis=0)
    X[:, prior.i_logmass] = np.tile(centers, n_profile)
    lp = evaluate_many(prior, X, cores)[:, 0].reshape(n_profile, centers.size)
    lp = lp + np.log(widths)[None]
    lp -= lp.max(axis=1, keepdims=True)
    prof = np.exp(lp)
    prof /= prof.sum(axis=1, keepdims=True)
    profile = prof.mean(axis=0)
    uniform = np.full(centers.size, 1.0 / centers.size)  # equal mass per dex
    probs = (1 - UNIFORM_SHARE) * profile + UNIFORM_SHARE * uniform
    return GridProposal(edges, probs), {
        "logm_edges": logm_edges.tolist(),
        "profile_bh_share": float(
            profile[logm_edges[:-1] >= np.log10(3.0)].sum()
        ),
    }


class ProductProposal:
    def __init__(self, mixture, grid, i_m, dim):
        self.mixture, self.grid, self.i_m, self.dim = mixture, grid, i_m, dim
        self.rest = [j for j in range(dim) if j != i_m]

    def sample(self, n, rng):
        x = np.empty((n, self.dim))
        x[:, self.rest] = self.mixture.sample(n, rng)
        x[:, self.i_m] = self.grid.sample(n, rng)
        return x

    def logpdf(self, x):
        return self.mixture.logpdf(x[:, self.rest]) + self.grid.logpdf(
            x[:, self.i_m]
        )


class Blend:
    """sum_k share_k q_k(x) for proposals with sample / logpdf."""

    def __init__(self, parts):
        self.shares = np.array([s for s, _ in parts], float)
        self.shares /= self.shares.sum()
        self.parts = [q for _, q in parts]

    def sample(self, n, rng):
        counts = rng.multinomial(n, self.shares)
        x = np.concatenate(
            [q.sample(c, rng) for q, c in zip(self.parts, counts)], axis=0
        )
        return x[rng.permutation(n)]

    def logpdf(self, x):
        from scipy.special import logsumexp

        return logsumexp(
            [np.log(s) + q.logpdf(x) for s, q in zip(self.shares, self.parts)],
            axis=0,
        )


def widen(mix, dims, factor):
    """A copy of ``mix`` with the coordinates ``dims`` scaled by ``factor``."""
    scale = np.ones(mix.dim)
    scale[dims] = factor
    out = TMixture.__new__(TMixture)
    out.weights, out.df, out.dim = mix.weights, mix.df, mix.dim
    out.means = mix.means
    out.chol = mix.chol * scale[None, :, None]
    return out


def resample(X, log_w, n, rng):
    """n DISTINCT draws, chosen with probability ~ weight.

    Without replacement, so a heavy draw does not enter the EM fit as many
    identical points (which collapses a component onto it).
    """
    w = np.exp(log_w - np.max(log_w))
    n = min(n, int(np.count_nonzero(w)))
    return X[rng.choice(X.shape[0], size=n, replace=False, p=w / w.sum())]


def broad(X, factor):
    """One Student-t with X's mean and covariance x factor^2."""
    cov = np.cov(X, rowvar=False) * factor**2
    return TMixture([1.0], [X.mean(axis=0)], [cov])


def combine(a, b, share_b):
    """One TMixture holding a's components (1 - share_b) and b's (share_b)."""
    out = TMixture.__new__(TMixture)
    out.weights = np.concatenate(
        [(1 - share_b) * a.weights, share_b * b.weights]
    )
    out.means = np.concatenate([a.means, b.means])
    out.chol = np.concatenate([a.chol, b.chol])
    out.df, out.dim = a.df, a.dim
    return out


def run(
    prior,
    *,
    n_final,
    n_adapt=200_000,
    rounds=3,
    n_components=40,
    pilot_draws=500,
    pilot_tune=500,
    pilot_chains=4,
    cores=4,
    seed=7163,
    log=print,
):
    """Importance-sample the prior; returns (columns, log_w, info)."""
    rng = np.random.default_rng(seed)
    info = {"seed": seed, "cores": cores}
    t0 = time.time()
    pilot, div = prior.pilot_nuts(
        draws=pilot_draws, tune=pilot_tune, chains=pilot_chains, seed=seed
    )
    info["pilot"] = {
        "draws": int(pilot.shape[0]),
        "divergences": div,
        "wall_s": time.time() - t0,
    }
    log(f"pilot NUTS: {pilot.shape[0]} draws, {div} divergent")
    grid, ginfo = logmass_proposal(prior, pilot, cores, rng)
    info["logmass_proposal"] = ginfo
    log(
        f"log-mass profile: share above 3 Msun {ginfo['profile_bh_share']:.4f}"
    )
    dim = prior.layout.total
    rest = [j for j in range(dim) if j != prior.i_logmass]
    # The pilot's fit stays in every round's product proposal as a broad
    # defensive share, so adaptation can sharpen it but never drop coverage.
    # A single broad component (the pilot's covariance x 3^2) keeps a share
    # of the base: the pilot is a short NUTS run whose chains need not have
    # mixed, so the fit to it alone can miss whole regions.
    base = combine(
        fit_tmixture(pilot[:, rest], n_components, rng),
        broad(pilot[:, rest], BROAD_WIDEN),
        BROAD_SHARE,
    )
    # A share of the proposal draws the LENS's distance and velocity
    # coordinates 3x wider.  Kicked lenses -- neutron stars (350 km/s natal
    # kick) and black holes (100 km/s) -- move much faster than their
    # populations' stars; and the nearest lenses (the D_L tail below ~2-3
    # kpc) are a few percent of the rate that a short pilot, and so every
    # refit after it, can under-cover without any heavy weight to show it.
    kick_dims = [
        rest.index(prior.layout.index(f"star.{c}_raw", 0))
        for c in ("distance", "pm_ra", "pm_dec", "rv")
    ]

    def kicked(mix):
        return ProductProposal(
            widen(mix, kick_dims, KICK_WIDEN), grid, prior.i_logmass, dim
        )

    prop = Blend(
        [
            (
                1 - KICK_SHARE,
                ProductProposal(base, grid, prior.i_logmass, dim),
            ),
            (KICK_SHARE, kicked(base)),
        ]
    )
    info["rounds"] = []
    best = (-1.0, prop)
    for r in range(rounds + 1):
        final = r == rounds
        n = n_final if final else n_adapt
        if final:
            # Adaptation is noisy (an EM refit can land badly); the final
            # draw uses the proposal that did best on its own round.
            prop = best[1]
        X = prop.sample(n, rng)
        t1 = time.time()
        rows = evaluate_many(prior, X, cores)
        lq = prop.logpdf(X)
        lp = rows[:, 0]
        n_nan = int(np.isnan(lp).sum())
        log_w = np.where(np.isfinite(lp), lp - lq, -np.inf)
        d = stats.importance_diagnostics(log_w)
        d["n_nan_logp"] = n_nan
        d["wall_s"] = time.time() - t1
        info["rounds"].append(d)
        log(
            f"round {r}: n {n}  ESS {d['kish_ess']:.0f} "
            f"({100 * d['ess_fraction']:.1f}%)  max w "
            f"{100 * d['max_weight_fraction']:.3f}%  khat "
            f"{d['pareto_khat']:.2f}  NaN {n_nan}  {d['wall_s']:.0f} s"
        )
        if not final and d["ess_fraction"] > best[0]:
            best = (d["ess_fraction"], prop)
        if not final:
            fit_on = resample(X, log_w, int(d["kish_ess"]), rng)
            # Half the next proposal is a JOINT mixture over every
            # coordinate, log mass included: it carries the correlations
            # the product cannot -- e.g. kicked neutron-star lenses, whose
            # mass (~1.3-2 Msun) comes with a much broader velocity spread.
            rest_mix = combine(
                fit_tmixture(fit_on[:, rest], n_components, rng),
                base,
                DEFENSIVE,
            )
            share = 0.5 * (1 - KICK_SHARE)
            prop = Blend(
                [
                    (
                        share,
                        ProductProposal(rest_mix, grid, prior.i_logmass, dim),
                    ),
                    (share, fit_tmixture(fit_on, n_components, rng)),
                    (KICK_SHARE, kicked(rest_mix)),
                ]
            )
    info["final"] = info["rounds"][-1]
    info["wall_s"] = time.time() - t0
    return prior.to_user_units(rows), log_w, info
