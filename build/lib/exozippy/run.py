from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Optional


from exozippy.config_loader import load_model_from_fit_yaml
from exozippy.data.rv import load_rv_data
from exozippy.inference.pymc_backend import build_rv_only_model


def _default_run_dir(fit_yaml_path: Path, fit_cfg: dict) -> Path:
    run_name = (fit_cfg.get("run", {}) or {}).get("name")
    stem = run_name if run_name else fit_yaml_path.stem
    ts = time.strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stem}_{ts}"


def _print_dry_run_summary(lm, rv_data):
    print("=== exozippy dry-run ===")
    print(f"Planets: {lm.planets}")
    print(f"Instruments: {lm.instruments}")
    print("Ecc parameterization:")
    for pl in lm.planets:
        print(f"  planet {pl}: {lm.ecc_kind_by_planet[pl]}")

    print("\nRV data summary:")
    for inst, d in rv_data.items():
        print(f"  {inst}: N={d.t.size}  t=[{d.t.min():.6f},{d.t.max():.6f}]  rv=[{d.y.min():.3f},{d.y.max():.3f}]  err~{float(d.yerr.mean()):.3f}")

    # show parameter list after aliasing
    canonical = {}
    for name, canon in lm.aliases.items():
        canonical.setdefault(canon, []).append(name)

    print("\nSampled/base parameters (after alias groups):")
    for canon, names in sorted(canonical.items()):
        spec = lm.specs[names[0]]
        pr = spec.prior.dist if spec.prior else "None"
        print(f"  {canon}  (aliases: {names})  prior={pr} init={spec.init} bounds={spec.bounds} fixed={spec.fixed}")

    if lm.constraints:
        print("\nConstraints:")
        for c in lm.constraints:
            print(f"  {c.name}: {c.expr} ({c.kind})")
    else:
        print("\nConstraints: none")

    print("\nDry-run complete (no sampling).")


def run_fit(
    fit_yaml_path: Path,
    outdir_override: Optional[str] = None,
    seed: Optional[int] = None,
    make_plots: bool = True,
    dry_run: bool = False,
) -> Path:

    lm = load_model_from_fit_yaml(fit_yaml_path)
    rv_data = load_rv_data(lm.fit, base_dir=fit_yaml_path.parent)

    if dry_run:
        _print_dry_run_summary(lm, rv_data)
        # no outdir created on dry-run
        return Path(".")

    # avoid importing for dry runs (?)
    import pymc as pm
    import arviz as az

    outdir = Path(outdir_override).expanduser().resolve() if outdir_override else _default_run_dir(fit_yaml_path, lm.fit)
    outdir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(fit_yaml_path, outdir / "fit.yaml")
    shutil.copy2((fit_yaml_path.parent / lm.fit["parameters"]["file"]).resolve(), outdir / "params_obsolete.yaml")

    pymc_cfg = ((lm.fit.get("sampler", {}) or {}).get("pymc", {}) or {})

    with pm.Model() as model:
        build_rv_only_model(lm, rv_data, user_params=(lm.fit.get("parameters", {}) or {}).get("user_params"))
        idata = pm.sample(
            draws=int(pymc_cfg.get("draws", 2000)),
            tune=int(pymc_cfg.get("tune", 2000)),
            chains=int(pymc_cfg.get("chains", 4)),
            target_accept=float(pymc_cfg.get("target_accept", 0.9)),
            init=str(pymc_cfg.get("init", "adapt_diag")),
            random_seed=seed,
        )

    az.to_netcdf(idata, outdir / "idata.nc")
    (outdir / "summary.txt").write_text(str(az.summary(idata)), encoding="utf-8")

    # plots omitted in this minimal bundle
    _ = make_plots
    return outdir
