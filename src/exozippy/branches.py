"""Reporting a branch-marginalized posterior: draw the branch, never average it.

A many-to-one parameterization (today the V_c/V_e eccentricity, whose inversion
is a quadratic; ``System.register_branch_alternative``) makes the likelihood a
MIXTURE over a discrete branch indicator ``z`` that the sampler never sees:
``System._add_branch_mixtures`` adds ``logsumexp_c(log w_c + L_c(theta))`` over
the ``2^k`` combinations ``c`` of the ``k`` declared branches.  The chains
therefore sample ``theta`` from its MARGINAL posterior, and every Deterministic
in the trace was evaluated on the as-built (primary) branch -- which is not the
posterior of anything branch-dependent.  On examples/gj1214 with
``fitvcve: true`` the trace's ``orbit.ecc`` was the upper root on every draw
(median 0.04, 84th percentile 0.9999) while the mixture's own eccentricity was
0.005 (review 1.8.14).

THE RULE (JDE 2026-10-01).  Exact posterior draws of the joint ``(theta, z)``
come from drawing ``z`` per draw from its conditional

    P(z = c | theta)  proportional to  exp(log w_c + L_c(theta)),

the very per-combination terms the logaddexp combines, and then evaluating
EVERY branch-dependent quantity under that one combination.  Three things that
look equivalent and are not:

* NOT a responsibility-weighted average of the branches per draw.  With 50/50
  responsibilities on e = 0.1 and e = 0.6 that reports 0.35, a value NEITHER
  branch supports, and it smears every quantile and interval.  A weighted
  average is right only for an EXPECTATION -- the Rao-Blackwellized branch
  probability ``mean_draws P(z = c | theta)`` -- and that is what is reported
  as a number (`branch_probabilities`), never as a draw.
* NOT each branch drawn independently from its own marginal responsibility.
  With several branches the draw is of the JOINT combination: shared
  parameters (the star's density, the RVs, a TTV) can correlate the branches
  of different orbits, and independent draws would pair branches the
  posterior never pairs.
* NOT per-quantity.  One ``z`` per draw, applied to every Deterministic that
  depends on a substituted node -- ``orbit.ecc`` and everything derived from it
  -- so a draw's quantities are mutually consistent.

``z`` is drawn with an RNG seeded from the run's own ``sampler: seed`` (stamped
on the trace as ``posterior.attrs["random_seed"]``), so the report is
reproducible from the trace alone.  It is stored per draw in
``sample_stats["branch_combination"]`` (with the per-combination log-weights
beside it in ``sample_stats["branch_log_weight"]``), not in the posterior:
it is a reporting device, not a sampled parameter, and the posterior group
feeds the corner plot, the trace plots and the mode finder.

Component-agnostic: nothing here knows which component declared a branch.
"""

import logging
import multiprocessing as mp

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors

logger = logging.getLogger(__name__)

# Mixed into the run's seed so the branch draw is its own stream: reusing the
# bare seed would correlate z with the sampler's first random numbers.
_BRANCH_STREAM = 0x6272
_LABEL_SEP = " | "

# Forked workers inherit these (the compiled function is not picklable); see
# run._compute_lp_from_model, whose pattern this is.
_EVAL_FN = None
_EVAL_NAMES = None


def _apply_combination(nodes, sequence):
    """``nodes`` with one combination's substitutions applied IN ORDER.

    In sequence, not merged, for the reason ``_add_branch_mixtures`` gives:
    two branches routinely replace the same node (two V_c/V_e orbits are two
    elements of one ``ecc`` vector) and each replacement is written relative
    to the node it replaces.
    """
    for replacements in sequence:
        nodes = graph_replace(nodes, replacements, strict=False)
    return nodes


def branch_dependent_deterministics(system, model):
    """The model's Deterministics that read any node some branch replaces."""
    keys = {
        key
        for branch in system._branch_alternatives
        for key in branch["replacements"]
    }
    return [d for d in model.deterministics if keys & set(ancestors([d]))]


def _compile_evaluator(system, model):
    """f(value vars...) -> [log weight per combination, deterministics per combo]."""
    terms = system._branch_combination_terms
    sequences = system._branch_combination_replacements
    if len(terms) != 2 ** len(system._branch_alternatives) or len(
        sequences
    ) != len(terms):
        raise RuntimeError(
            "[branches] the branch mixture's per-combination terms are missing "
            f"or incomplete ({len(terms)} terms, {len(sequences)} substitution "
            f"sequences for {len(system._branch_alternatives)} declared "
            "branches): System._add_branch_mixtures must have built them."
        )
    dets = branch_dependent_deterministics(system, model)
    outputs = [pt.stack([pt.as_tensor_variable(t) for t in terms])]
    for sequence in sequences:
        outputs += (
            list(_apply_combination(list(dets), sequence)) if dets else []
        )
    outputs = model.replace_rvs_by_values(outputs)
    inputs = [model.rvs_to_values[rv] for rv in model.free_RVs]
    fn = pytensor.function(inputs, outputs, on_unused_input="ignore")
    names = [rv.name for rv in model.free_RVs]
    return fn, names, dets


def _eval_chain(args):
    chain_data, chain_idx, n_draws = args
    rows = []
    for d in range(n_draws):
        rows.append(_EVAL_FN(*[chain_data[n][d] for n in _EVAL_NAMES]))
    return chain_idx, rows


def _raw_inputs(posterior, names, param_lookup):
    """Per-chain arrays of every free RV, in INTERNAL units.

    The free RVs are the raw sampled coordinates; in this codebase they are
    unitless and never converted.  A free RV that IS a Parameter label (the
    one case conversion would apply) is converted through its own
    ``to_internal`` rather than assumed away.
    """
    missing = [n for n in names if n not in posterior]
    if missing:
        raise KeyError(
            f"[branches] the trace has no draws of the free variable(s) "
            f"{missing}; the branch combination of a draw cannot be computed "
            "without every coordinate the likelihood reads."
        )
    out = []
    for c in range(posterior.sizes["chain"]):
        chain = {}
        for n in names:
            vals = np.asarray(posterior[n].values[c], dtype=float)
            if param_lookup is not None and n in param_lookup:
                vals = np.asarray(param_lookup[n].to_internal(vals))
            chain[n] = vals
        out.append(chain)
    return out


def _draw_combinations(log_w, seed):
    """One combination per draw from its conditional; (chain, draw) ints."""
    rng = np.random.default_rng([int(seed), _BRANCH_STREAM])
    shifted = log_w - np.max(log_w, axis=-1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(axis=-1, keepdims=True)
    u = rng.uniform(size=log_w.shape[:-1])
    cdf = np.cumsum(prob, axis=-1)
    return np.minimum(
        (cdf < u[..., None]).sum(axis=-1), log_w.shape[-1] - 1
    ), prob


def branch_probabilities(prob, n_branches):
    """Rao-Blackwellized P(branch b takes its alternative), per declared branch.

    ``prob`` is (..., 2^k) per-draw combination responsibilities; bit ``b`` of
    a combination index says whether branch ``b`` took its alternative.  This
    is the ONE place a responsibility-weighted average is right: it is an
    expectation, not a draw.
    """
    combos = np.arange(prob.shape[-1])
    flat = prob.reshape(-1, prob.shape[-1])
    return [
        float(flat[:, (combos >> b) & 1 == 1].sum(axis=1).mean())
        for b in range(n_branches)
    ]


def is_branch_resolved(idata):
    """True when this trace's draws are already branch-resolved.

    `resolve_branch_draws` stamps `sample_stats["branch_combination"]`, and
    run.py rewrites the trace on disk with it, so a REUSED trace
    (`recompute_trace: false`) arrives resolved: its Deterministics are the
    assigned branches and must be read as they are, not resolved again.
    """
    ss = idata.get("sample_stats")
    return ss is not None and "branch_combination" in ss


def resolve_branch_draws(system, model, idata, param_lookup=None, cores=None):
    """Draw each posterior draw's branch combination and re-derive, in place.

    Returns the set of posterior variable NAMES it regenerated; they come back
    in INTERNAL units (as ``System.fold_degenerate_draws``' do) and the caller
    converts them.  Empty -- and nothing touched -- for a model with no
    declared branch.
    """
    if not system._branch_alternatives:
        return set()
    if is_branch_resolved(idata):
        # Its Deterministics are no longer the primary branch, so resolving
        # again would treat assigned branches as if they were the as-built
        # ones.  The caller asks is_branch_resolved first; reaching here is a
        # bookkeeping bug.
        raise RuntimeError(
            "[branches] this trace is already branch-resolved "
            "(sample_stats['branch_combination'] exists); it must not be "
            "resolved twice."
        )
    posterior = idata.posterior
    seed = posterior.attrs.get("random_seed")
    if seed is None:
        # A trace written before the seed was stamped (review 2.14.4).  The
        # trace_meta "unverifiable" precedent: say so, and stay reproducible.
        logger.warning(
            "[branches] the trace carries no 'random_seed'; drawing the branch "
            "combinations with seed 0, which is reproducible but is not this "
            "run's seed."
        )
        seed = 0

    fn, names, dets = _compile_evaluator(system, model)
    chain_arrays = _raw_inputs(posterior, names, param_lookup)
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    n_comb = len(system._branch_combination_terms)
    n_dets = len(dets)
    labels = [b["label"] for b in system._branch_alternatives]
    logger.info(
        "[branches] drawing the branch combination of %d x %d draws from the "
        "mixture's own per-combination weights (%d combination(s): %s)",
        n_chains,
        n_draws,
        n_comb,
        ", ".join(labels),
    )

    global _EVAL_FN, _EVAL_NAMES
    _EVAL_FN, _EVAL_NAMES = fn, names
    try:
        from .samplers._common import default_cores

        grant = default_cores() if cores is None else max(1, int(cores))
        n_workers = min(n_chains, grant)
        jobs = [(arr, c, n_draws) for c, arr in enumerate(chain_arrays)]
        if n_workers > 1:
            with mp.get_context("fork").Pool(n_workers) as pool:
                results = pool.map(_eval_chain, jobs)
        else:
            results = [_eval_chain(j) for j in jobs]
    finally:
        _EVAL_FN, _EVAL_NAMES = None, None

    log_w = np.full((n_chains, n_draws, n_comb), np.nan)
    det_vals = [[None] * n_draws for _ in range(n_chains)]
    for c, rows in results:
        for d, row in enumerate(rows):
            log_w[c, d] = np.asarray(row[0], dtype=float)
            det_vals[c][d] = row[1:]
    if not np.all(np.isfinite(log_w).any(axis=-1)):
        bad = int((~np.isfinite(log_w).any(axis=-1)).sum())
        raise FloatingPointError(
            f"[branches] {bad} draw(s) have no finite branch-combination "
            "weight at all; their branch cannot be drawn."
        )
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    z, prob = _draw_combinations(log_w, seed)

    regenerated = set()
    for j, det in enumerate(dets):
        if det.name not in posterior:
            continue
        target = posterior[det.name]
        stacked = np.empty(target.shape, dtype=target.dtype)
        for c in range(n_chains):
            for d in range(n_draws):
                stacked[c, d] = np.reshape(
                    det_vals[c][d][int(z[c, d]) * n_dets + j], target.shape[2:]
                )
        posterior[det.name] = target.copy(data=stacked)
        regenerated.add(det.name)

    ss = idata.get("sample_stats")
    if ss is None:
        idata["sample_stats"] = xr.Dataset()
        ss = idata.sample_stats
    coords = {"chain": posterior.chain, "draw": posterior.draw}
    ss["branch_combination"] = xr.DataArray(
        z.astype("int64"), dims=["chain", "draw"], coords=coords
    )
    ss["branch_log_weight"] = xr.DataArray(
        log_w,
        dims=["chain", "draw", "branch_combination_dim"],
        coords={**coords, "branch_combination_dim": np.arange(n_comb)},
    )
    probs = branch_probabilities(prob, len(labels))
    # netCDF attributes: one string and one float array (a list of strings
    # is not portable across netCDF engines).
    ss["branch_combination"].attrs["branch_labels"] = _LABEL_SEP.join(labels)
    ss["branch_combination"].attrs["branch_probability"] = np.asarray(
        probs, dtype=float
    )
    ss["branch_combination"].attrs["seed"] = int(seed)
    for label, p_alt in zip(labels, probs):
        logger.info(
            "[branches] %s: posterior probability %.4f (Rao-Blackwellized; "
            "%d of %d draws took it)",
            label,
            p_alt,
            int(((z >> labels.index(label)) & 1).sum()),
            z.size,
        )
    return regenerated


def branch_summary_lines(idata):
    """Human-readable branch probabilities for the summary file, or []."""
    ss = idata.get("sample_stats")
    if ss is None or "branch_combination" not in ss:
        return []
    attrs = ss["branch_combination"].attrs
    lines = [
        "Branch probabilities (Rao-Blackwellized mean responsibility; each "
        "draw's quantities use ONE branch combination drawn from it, never an "
        "average):"
    ]
    labels = str(attrs["branch_labels"]).split(_LABEL_SEP)
    probs = np.atleast_1d(attrs["branch_probability"])
    for label, p in zip(labels, probs):
        lines.append(f"  {label}: {float(p):.4f}")
    return lines
