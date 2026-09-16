"""Compare the DC2018_128 mulens start values and logp across two machines.

Diagnostic for the class of bug fixed in #92: the relaxation engine inverts an
underdetermined system to build microlensing start values (see #93), so two
machines could resolve the same config into two different physical models.  The
raw point is FIXED (the test's ``GOOD_RAW``), so any divergence in the physical
block localizes the cause.

Run on both machines from anywhere and diff the two outputs:

    poetry run python scripts/diag_mulens.py > diag_$(hostname).txt

Prints, in order: environment and versions, every resolved initval and
init_scale, every parameter's physical value at ``GOOD_RAW``, and the mulens
residual summary (chi2, per-time-bin z, worst points).

``GOOD_RAW`` is a coordinate in the WHITENED space, so the test's whitening
fixture has to be restored onto the build before it decodes to the state it
came from -- exactly as ``tests/test_runaway_logp_regression.py`` does.  This
script refuses to print anything if that restore fails, because the numbers
would otherwise look ordinary and be wrong: without it, measured on
DC2018_128, the total logp reads -4.77e5 instead of 3401.59 and the mulens
chi2/N reads 1088 instead of 1.002.
"""

import importlib.metadata as md
import importlib.util
import os
import platform
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytensor
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "DC2018_128"
TEST_FILE = REPO_ROOT / "tests" / "test_runaway_logp_regression.py"

from exozippy import whitening  # noqa: E402
from exozippy.system import System  # noqa: E402


def _load_pinned_point():
    """Import the pinned point from the test that owns it.

    Returns (GOOD_RAW, GOOD_EXPECTED_LP, WHITENING_FIXTURE).  Loaded by path
    rather than by adding tests/ to sys.path, so this script works from any
    cwd and does not shadow anything named like a test module.  All three
    come from the same module on purpose: the raw point, the lp it is
    expected to produce and the whitening it was sampled under are one
    inseparable set, and reading two of the three is how this script started
    printing an unrelated state.
    """
    spec = importlib.util.spec_from_file_location(
        "_runaway_logp_regression", TEST_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.GOOD_RAW,
        module.GOOD_EXPECTED_LP,
        module.WHITENING_FIXTURE,
    )


def _fmt(a, prec=8):
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    return np.array2string(a[:4], precision=prec, floatmode="maxprec")


def main():
    good_raw, expected_lp, whitening_fixture = _load_pinned_point()

    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"  host          {socket.gethostname()}")
    print(f"  python        {sys.version.split()[0]}")
    print(f"  platform      {platform.platform()}")
    print(f"  glibc         {platform.libc_ver()}")
    for pkg in (
        "numpy",
        "scipy",
        "pytensor",
        "pymc",
        "vbmicrolensing",
        "astropy",
        "sympy",
        "exoplanet-core",
        "celerite2",
        "numba",
    ):
        try:
            print(f"  {pkg:<13} {md.version(pkg)}")
        except Exception:
            print(f"  {pkg:<13} MISSING")
    print(f"  blas__ldflags {pytensor.config.blas__ldflags!r}")
    print(f"  pytensor cxx  {pytensor.config.cxx!r}")
    print(f"  GOOD_EXPECTED_LP (pinned in the test) = {expected_lp}")

    work = Path(tempfile.mkdtemp()) / "DC2018_128"
    shutil.copytree(
        EXAMPLE_DIR,
        work,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )
    orig = os.getcwd()
    os.chdir(work)
    try:
        with open("DC2018_128.yaml") as f:
            config = yaml.safe_load(f)
        with open("DC2018_128.params.yaml") as f:
            user_params = yaml.safe_load(f)
        # The pinned draws were recorded under linear planet mass; a lens body
        # now defaults to log_q, and the raw values do not carry across.
        for entry in config.get("planet", []):
            entry.setdefault("mass_parameterization", "linear")

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()
        # build_model leaves the PRELIMINARY whitening scales in place;
        # run.py measures the real ones at startup.  GOOD_RAW came from a
        # run, so restore that run's whitening or it decodes elsewhere.
        if not whitening.load_whitening(system, whitening_fixture):
            raise SystemExit(
                f"ERROR: could not restore the whitening fixture "
                f"{whitening_fixture}.  GOOD_RAW only means anything under "
                f"the whitening it was sampled with, so every number this "
                f"script would print describes a different physical state "
                f"-- and looks perfectly ordinary while doing so.  Refusing "
                f"to emit a cross-machine comparison that cannot be "
                f"compared."
            )
        point = {k: np.asarray(v, dtype=float) for k, v in good_raw.items()}

        total = float(np.asarray(model.compile_logp()(point)))
        print(f"  TOTAL logp at GOOD_RAW = {total:.4f}")

        params = system.get_all_parameters()

        print()
        print("=" * 78)
        print("RESOLVED initval / init_scale / bounds  (stage 4 output)")
        print("=" * 78)
        for p in params:
            iv = "SYMBOLIC" if hasattr(p.initval, "owner") else _fmt(p.initval)
            sc = "None" if p.init_scale is None else _fmt(p.init_scale, 6)
            lo = "None" if p.lower is None else _fmt(p.lower, 6)
            up = "None" if p.upper is None else _fmt(p.upper, 6)
            print(f"  {p.label:<38s} iv={iv} sc={sc} lo={lo} up={up}")

        print()
        print("=" * 78)
        print("PHYSICAL value of every parameter at GOOD_RAW")
        print("=" * 78)
        labels, nodes = [], []
        for p in params:
            v = getattr(p, "value", None)
            if isinstance(v, pytensor.graph.basic.Variable):
                labels.append(p.label)
                nodes.append(v)
        exprs = model.replace_rvs_by_values(nodes)
        fn = pytensor.function(
            model.value_vars, exprs, on_unused_input="ignore"
        )
        vals = fn(*[point[vv.name] for vv in model.value_vars])
        for lab, val in zip(labels, vals):
            print(f"  {lab:<38s} {_fmt(val)}")

        print()
        print("=" * 78)
        print("MULENS RESIDUALS at GOOD_RAW")
        print("=" * 78)
        obs = [v for v in model.observed_RVs if "mulens" in v.name][0]
        ins = obs.owner.inputs
        mu, sigma = ins[-2], ins[-1]
        data = np.asarray(obs.tag.observations.eval(), dtype=float).ravel()
        e = model.replace_rvs_by_values([mu, sigma])
        f2 = pytensor.function(model.value_vars, e, on_unused_input="ignore")
        mu_v, sig_v = [
            np.asarray(a, dtype=float).ravel()
            for a in f2(*[point[vv.name] for vv in model.value_vars])
        ]
        inst = next(
            c
            for c in system.get_all_components()
            if c.prefix == "mulensinstrument"
        )
        t = np.asarray(inst.time, dtype=float).ravel()
        r = data - mu_v
        z = r / sig_v
        print(
            f"  N={data.size}  chi2={np.sum(z**2):.2f}  "
            f"chi2/N={np.sum(z**2) / data.size:.3f}"
        )
        print(
            f"  data  min={data.min():.6f} max={data.max():.6f} "
            f"mean={data.mean():.6f}"
        )
        print(
            f"  model min={mu_v.min():.6f} max={mu_v.max():.6f} "
            f"mean={mu_v.mean():.6f}"
        )
        print(
            f"  sigma min={sig_v.min():.6f} med={np.median(sig_v):.6f} "
            f"max={sig_v.max():.6f}"
        )
        print(
            f"  resid mean={r.mean():+.6f} std={r.std():.6f}  "
            f"z mean={z.mean():+.4f} std={z.std():.4f}"
        )
        print("  per-bin (sorted by time, 8 equal bins):")
        for b in np.array_split(np.argsort(t), 8):
            print(
                f"    t=[{t[b].min():.2f},{t[b].max():.2f}] n={b.size:4d} "
                f"mean_z={z[b].mean():+9.3f} max|z|={np.abs(z[b]).max():9.2f}"
            )
        print("  worst 8 points:")
        for i in np.argsort(-np.abs(z))[:8]:
            print(
                f"    t={t[i]:.5f} data={data[i]:+.6f} model={mu_v[i]:+.6f} "
                f"sig={sig_v[i]:.6f} z={z[i]:+.2f}"
            )
    finally:
        os.chdir(orig)


if __name__ == "__main__":
    main()
