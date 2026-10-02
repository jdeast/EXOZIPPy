"""An INDEPENDENT Bayesian fit of the Theophylline hierarchy.

Written directly against numpyro -- no EXOZIPPy component machinery, no shared
model-assembly code, a different sampler implementation -- with the same data,
the same structural model and the same priors as
examples/theophylline/theophylline.yaml in the `fitclke` basis. It is the
numpyro cross-check quoted in src/exozippy/components/pharmacokinetics/
README.md, published so the comparison there can be re-run.

WHAT IT CAN AND CANNOT TEST.  It cannot be a logp-level match: EXOZIPPy puts a
soft barrier on each subject's derived log-coordinates, with a transition
width measured by the whitening probe at runtime, and there is no way to
restate that here without copying the machinery this is supposed to be
independent of.  What it tests is the POSTERIOR, which is what a
component-assembly error would move: the forward model, the non-centred
hierarchy, the allometric covariate, the units, and the basis.

The forward model here is the TEXTBOOK form, deliberately -- the 0/0 at
ka == ke that physics.py goes to such lengths to remove.  ka and ke are well
separated on these data, so it is safe, and using it means this reference
shares no algebra with the implementation it is checking.

REQUIREMENTS.  numpyro (and its jax) and arviz.  numpyro is NOT an EXOZIPPy
dependency; install it yourself into the same environment
(`pip install numpyro`).  The data are not redistributed: fetch them first
with `exozippy-fetch-theoph` from this directory (see README.md).

USAGE (from this directory):

    exozippy-fetch-theoph
    python numpyro_reference.py                 # 4 chains x 2000 + 2000
    python numpyro_reference.py --data /path/to/theoph.csv --samples 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

WEIGHT = np.array(
    [79.6, 72.4, 70.5, 72.7, 54.6, 80.0, 64.6, 70.5, 86.4, 58.2, 65.0, 60.5]
)
DOSE_PER_KG = np.array(
    [4.02, 4.4, 4.53, 4.4, 5.86, 4.0, 4.95, 4.53, 3.1, 5.5, 4.92, 5.3]
)
DOSE = DOSE_PER_KG * WEIGHT  # mg, as Subject.load_data does
WT_REF, BETA_CL, BETA_KE, BETA_KA = 70.0, 0.75, -0.25, 0.0

SUMMARY_NAMES = (
    "mu_log_cl",
    "mu_log_ke",
    "mu_log_ka",
    "omega_cl",
    "omega_ke",
    "omega_ka",
    "sigma_add",
    "sigma_prop",
)


def load_data(path):
    """Return (subject index 0-11, time [hr], conc [mg/L]) from theoph.csv."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found.  The Theophylline data are not redistributed "
            "with EXOZIPPy; run `exozippy-fetch-theoph` in "
            "examples/theophylline/ first, or pass --data."
        )
    df = pd.read_csv(path)
    subj = df["Subject"].to_numpy().astype(int) - 1
    if sorted(set(subj)) != list(range(len(WEIGHT))):
        raise ValueError(
            f"{path}: expected subjects 1-{len(WEIGHT)}, got {sorted(set(subj + 1))}"
        )
    return subj, df["Time"].to_numpy(float), df["conc"].to_numpy(float)


def make_model(subj, t, obs):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model():
        # Flat priors over exactly the defaults.yaml bounds.
        mu_cl = numpyro.sample("mu_log_cl", dist.Uniform(-2.0, 2.0))
        mu_ke = numpyro.sample("mu_log_ke", dist.Uniform(-4.0, 2.0))
        mu_ka = numpyro.sample("mu_log_ka", dist.Uniform(-2.0, 2.0))
        om_cl = numpyro.sample("omega_cl", dist.Uniform(0.0, 1.0))
        om_ke = numpyro.sample("omega_ke", dist.Uniform(0.0, 1.0))
        om_ka = numpyro.sample("omega_ka", dist.Uniform(0.0, 1.0))
        # eta ~ N(0,1) on [-10, 10]: the non-centred latent, truncated as the
        # component's logit bounds truncate it.
        tn = dist.TruncatedNormal(0.0, 1.0, low=-10.0, high=10.0)
        with numpyro.plate("subject", len(WEIGHT)):
            e_cl = numpyro.sample("eta_cl", tn)
            e_ke = numpyro.sample("eta_ke", tn)
            e_ka = numpyro.sample("eta_ka", tn)

        wt = jnp.log10(jnp.asarray(WEIGHT) / WT_REF)
        log_cl = mu_cl + BETA_CL * wt + om_cl * e_cl
        log_ke = mu_ke + BETA_KE * wt + om_ke * e_ke
        log_ka = mu_ka + BETA_KA * wt + om_ka * e_ka
        cl, ke, ka = 10.0**log_cl, 10.0**log_ke, 10.0**log_ka
        v = cl / ke  # the `fitclke` basis

        s_add = numpyro.sample("sigma_add", dist.Uniform(0.0, 100.0))
        s_prop = numpyro.sample("sigma_prop", dist.Uniform(0.0, 2.0))

        d, k_a, k_e, vv = DOSE[subj], ka[subj], ke[subj], v[subj]
        conc = (
            (d * k_a)
            / (vv * (k_a - k_e))
            * (jnp.exp(-k_e * t) - jnp.exp(-k_a * t))
        )
        sigma = jnp.sqrt(s_add**2 + (s_prop * conc) ** 2)
        numpyro.sample("obs", dist.Normal(conc, sigma), obs=obs)

    return model


def _print_summary(post, mask=None):
    for name in SUMMARY_NAMES:
        x = np.asarray(post[name]).ravel()
        if mask is not None:
            x = x[mask]
        lo, hi = np.percentile(x, [2.5, 97.5])
        print(
            f"{name:12s} mean {x.mean():9.5f}  sd {x.std(ddof=1):8.5f}  "
            f"median {np.median(x):9.5f}  95% [{lo:9.5f}, {hi:9.5f}]"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(HERE / "theoph.csv"))
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    subj, t, obs = load_data(args.data)

    import numpyro

    numpyro.set_host_device_count(args.chains)
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(
        NUTS(make_model(subj, t, obs), target_accept_prob=0.95),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(args.seed))
    post = mcmc.get_samples()

    print("### numpyro reference, Theophylline, fitclke basis")
    _print_summary(post)

    # THE CHAINS VISIT BOTH FLIP-FLOP MODES, which is the point rather than a
    # failure: an implementation that shares no code with EXOZIPPy, given no
    # hint that the degeneracy exists, finds it.  A marginal summary over both
    # modes is meaningless for ka and ke (their roles swap), so condition on
    # the direct mode -- ka > ke -- which is exactly what
    # `assume_fast_absorption` imposes.
    direct = (
        np.asarray(post["mu_log_ka"]) > np.asarray(post["mu_log_ke"])
    ).ravel()
    print(
        f"\n### conditioned on the direct mode (ka > ke): {direct.mean():.3f} of draws"
    )
    _print_summary(post, direct)

    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=list(SUMMARY_NAMES))
    print("\n### convergence (marginal, i.e. across BOTH modes)")
    print(summ[["r_hat", "ess_bulk"]].to_string())


if __name__ == "__main__":
    main()
