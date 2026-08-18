"""Probe which JAX samplers actually work in the current environment.

The house rule is that sampler compatibility is verified by SAMPLING, never
by analogy with another component's Op and never by
``compile_logp(mode="JAX")`` -- a funcified logp that evaluates can still
have no differentiation rule, which is exactly how ``nuts_sampler="numpyro"``
died at HMC init on every transit model through exoplanet-core 0.4.0rc1
(exoplanet-dev/exoplanet-core#144).  This script is that rule, automated.

It exists because the answer is PLATFORM DEPENDENT.  jaxlib's last macOS
x86_64 wheel is 0.4.38 and there is no sdist, so an Intel Mac cannot have
the jax the rest of the tree is pinned to -- and measurement (CI run
31742944005) showed that installing 0.4.38 there is worse than installing
none: exoplanet-core reaches ``jax.ffi``, public only since jax 0.5.0, and
raises AttributeError past its own ImportError guard, so ``import exozippy``
fails outright.  Intel Mac therefore ships without jax and both samplers
report "not installed" here, which is the correct answer rather than a gap
in the probe.

The probe stays useful everywhere else: it is what catches an upstream
release quietly removing a differentiation rule.

Three probe models, cheapest first, each a strictly larger slice of the
stack, so a failure localizes itself:

    plain      a bare pm.Normal.  Exercises only pymc's JAX sampling wiring
               (pymc.sampling.jax.sample_jax_nuts) against whatever numpyro
               or blackjax is installed.  If this fails, the sampler binding
               is broken and the two below say nothing new.
    kepler     exoplanet_core.pymc.ops.kepler.  First of the two
               exoplanet-core Ops the tree uses, and the first thing that
               needs the JAX FFI (jax.ffi.ffi_call, public since jax 0.5.0)
               to have both a conversion AND a differentiation rule.
    limbdark   exoplanet_core.pymc.ops.quad_solution_vector.  The Op from
               issue #144, the one whose missing JAX grad rule is why
               exoplanet-core>=0.4.0 is a load-bearing floor in pyproject.

Every (model, sampler) pair is attempted independently and a failure is
recorded rather than raised, because the useful output is the whole matrix:
"numpyro works except under limb darkening" and "numpyro is broken outright"
call for completely different fixes.

Usage::

    python scripts/diag_jax_samplers.py
    python scripts/diag_jax_samplers.py --samplers numpyro
    python scripts/diag_jax_samplers.py --summary "$GITHUB_STEP_SUMMARY"
    python scripts/diag_jax_samplers.py --strict     # exit 1 on any failure

Exit status is 0 even when probes fail unless --strict is given.  That is
deliberate: in CI this runs on a platform where some failures are the
EXPECTED finding, and a red job would bury the table that is the point.
"""

import argparse
import importlib.metadata
import io
import logging
import os
import platform
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

# Keep the probe honest about size: these are correctness checks, not
# convergence checks.  Two chains rather than one because chain_method
# ="parallel" over a single chain has historically hidden vmap/pmap bugs.
DRAWS = 5
TUNE = 5
CHAINS = 2

# Reported in the version table.  Anything absent prints "-- not installed",
# which is itself a finding (numpyro missing on an Intel Mac is the whole
# story), so this must never raise.
_REPORTED = (
    "exozippy",
    "jax",
    "jaxlib",
    "numpyro",
    "blackjax",
    "pymc",
    "pytensor",
    "arviz",
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "exoplanet-core",
    "celerite2",
    "nutpie",
)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def describe_environment():
    """Return (lines, versions) describing the interpreter and the stack."""
    lines = [
        f"python           {platform.python_version()} "
        f"({platform.python_implementation()})",
        f"platform         {platform.platform()}",
        f"machine          {platform.machine()}",
    ]
    versions = {}
    for name in _REPORTED:
        v = _version(name)
        versions[name] = v
        lines.append(f"{name:16s} {v if v else '-- not installed'}")
    return lines, versions


# --- probe models ---------------------------------------------------------
#
# Each builds a pymc model whose logp genuinely depends on the Op under
# test, so that a missing JAX differentiation rule fails at HMC init rather
# than being rewritten away.  They are deliberately tiny; a probe that takes
# minutes is a probe nobody runs.


def model_plain():
    import pymc as pm

    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Normal("obs", mu=x, sigma=1.0, observed=[0.1, -0.2, 0.3])
    return model


def model_kepler():
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    from exoplanet_core.pymc import ops

    t = np.linspace(0.0, 1.0, 32)
    with pm.Model() as model:
        ecc = pm.Uniform("ecc", lower=0.0, upper=0.8, initval=0.1)
        # Mean anomaly over one full period, so the solver is exercised
        # across the whole branch structure rather than near M = 0.
        M = pt.as_tensor_variable(2.0 * np.pi * t)
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))
        pm.Normal("obs", mu=sinf + cosf, sigma=1.0, observed=np.zeros_like(t))
    return model


def model_limbdark():
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt

    from exozippy.components.limbdark import quad_limb_darkened_flux

    # Impact parameters sweeping across the limb, so the occultation is
    # partial for most points -- the regime where the Green's-basis solution
    # vector is nontrivial.
    b = np.linspace(0.0, 1.2, 32)
    with pm.Model() as model:
        u1 = pm.Uniform("u1", lower=0.0, upper=1.0, initval=0.3)
        u2 = pm.Uniform("u2", lower=0.0, upper=1.0, initval=0.2)
        flux = quad_limb_darkened_flux(pt.as_tensor_variable(b), 0.1, u1, u2)
        pm.Normal("obs", mu=flux, sigma=1.0, observed=np.ones_like(b))
    return model


PROBES = {
    "plain": model_plain,
    "kepler": model_kepler,
    "limbdark": model_limbdark,
}


def run_probe(build, sampler):
    """Sample one probe model with one JAX sampler.

    Returns (ok, detail).  Mirrors run.py's JAX branch exactly -- x64
    enabled, the blackjax progress-bar patch applied, and
    ``sample_jax_nuts`` called directly rather than through
    ``pm.sample(nuts_sampler=...)`` -- so that a pass here means the code
    path production uses actually works, not merely a neighbouring one.
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", True)

        if sampler == "blackjax":
            from exozippy.compat import patch_blackjax_progress_bar

            patch_blackjax_progress_bar()

        from pymc.sampling.jax import sample_jax_nuts

        model = build()
        # pymc and the JAX samplers are chatty and the progress bars are
        # noise in a log; the traceback is what matters on failure.
        #
        # Three separate muzzles are needed, and none of them is redundant.
        # DRAWS is 5 by design, so pymc's convergence check ("The number of
        # samples is too small to check convergence reliably") fires on
        # EVERY probe -- and it arrives through `logging`, whose
        # StreamHandler captured the real sys.stderr when it was
        # constructed, so redirect_stderr does not touch it and it lands
        # mid-line between the progress dots and the OK.  logging.disable
        # is what silences that; catch_warnings only covers what comes
        # through the warnings module.  Both are restored on the way out.
        buf = io.StringIO()
        previous_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with model, redirect_stdout(buf), redirect_stderr(buf):
                    idata = sample_jax_nuts(
                        draws=DRAWS,
                        tune=TUNE,
                        chains=CHAINS,
                        target_accept=0.8,
                        # jitter=False matches run.py: chains start where
                        # we put them, not one whitening scale away.
                        jitter=False,
                        chain_method="parallel",
                        nuts_sampler=sampler,
                    )
        finally:
            logging.disable(previous_disable)
        n = int(idata.posterior.sizes.get("draw", 0))
        if n != DRAWS:
            return False, f"expected {DRAWS} draws, got {n}"
        return True, f"{n} draws x {CHAINS} chains"
    except Exception as exc:  # noqa: BLE001 -- reporting, not handling
        detail = f"{type(exc).__name__}: {exc}"
        return False, detail.replace("\n", " ")[:300]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samplers",
        default="numpyro,blackjax",
        help="comma-separated subset of numpyro,blackjax",
    )
    parser.add_argument(
        "--probes",
        default=",".join(PROBES),
        help=f"comma-separated subset of {','.join(PROBES)}",
    )
    parser.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="append a markdown report to this file (default: "
        "$GITHUB_STEP_SUMMARY when set)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any probe fails (default: always exit 0)",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="print the full traceback for each failure",
    )
    args = parser.parse_args(argv)

    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    unknown = set(probes) - set(PROBES)
    if unknown:
        parser.error(f"unknown probe(s): {', '.join(sorted(unknown))}")

    env_lines, versions = describe_environment()
    print("=" * 68)
    print("environment")
    print("=" * 68)
    for line in env_lines:
        print(line)
    print()

    results = {}
    for sampler in samplers:
        if versions.get(sampler) is None:
            for probe in probes:
                results[(probe, sampler)] = (
                    None,
                    "not installed",
                )
            continue
        for probe in probes:
            print(f"-- {probe} x {sampler} ... ", end="", flush=True)
            ok, detail = run_probe(PROBES[probe], sampler)
            results[(probe, sampler)] = (ok, detail)
            print("OK" if ok else "FAIL")
            if not ok:
                print(f"   {detail}")
                if args.traceback:
                    traceback.print_exc()
    print()

    # --- report -----------------------------------------------------------
    def cell(entry):
        ok, detail = entry
        if ok is None:
            return "n/a"
        return "OK" if ok else "FAIL"

    header = f"| {'probe':10s} | " + " | ".join(f"{s:9s}" for s in samplers)
    print(header + " |")
    print("|" + "-" * 12 + "|" + "|".join("-" * 11 for _ in samplers) + "|")
    for probe in probes:
        row = f"| {probe:10s} | " + " | ".join(
            f"{cell(results[(probe, s)]):9s}" for s in samplers
        )
        print(row + " |")

    if args.summary:
        md = ["## JAX sampler probe", ""]
        md.append("| package | version |")
        md.append("| --- | --- |")
        for name in _REPORTED:
            md.append(f"| {name} | {versions[name] or '*not installed*'} |")
        md.append("")
        md.append(f"`{platform.platform()}` / `{platform.machine()}`")
        md.append("")
        md.append("| probe | " + " | ".join(samplers) + " |")
        md.append("| --- |" + " --- |" * len(samplers))
        for probe in probes:
            md.append(
                f"| {probe} | "
                + " | ".join(cell(results[(probe, s)]) for s in samplers)
                + " |"
            )
        failures = [
            (p, s, d) for (p, s), (ok, d) in results.items() if ok is False
        ]
        if failures:
            md += ["", "<details><summary>Failures</summary>", ""]
            for probe, sampler, detail in failures:
                md.append(f"- **{probe} x {sampler}** -- `{detail}`")
            md += ["", "</details>"]
        md.append("")
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")

    failed = any(ok is False for ok, _ in results.values())
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
