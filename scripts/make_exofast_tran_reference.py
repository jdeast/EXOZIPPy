"""Generate tests/fixtures/exofast_tran_parity.json: an EXOFASTv2
exofast_tran reference light curve for tests/test_transit_exofast_parity.py.

A typical hot Jupiter around a sun-like star (P = 3 d, Rp/R* ~ 0.1,
b ~ 0.44, e = 0, quadratic TESS limb darkening u1 = 0.4 / u2 = 0.2),
observed across one transit from one exposure before ingress to one
exposure after egress at two cadences:

  - 2-minute cadence, unsmeared (no exptime/ninterp keys: the default,
    instantaneous model)
  - 30-minute cadence, smeared with exptime=30 / ninterp=10 (EXOFASTv2's
    midpoint-Riemann sub-exposure grid, exofast_chi2v2.pro)

To guarantee both codes integrate the same physical system, the scenario
is first built with EXOZIPPy and the *derived* quantities the transit
model actually consumes (ar, inc, tp, omega, p, u1, u2, 2pi/n) are
extracted at the model's initial point and fed verbatim to exofast_tran.
The JSON stores the scenario (config + user params), those inputs, the
time grids, and the IDL fluxes; the test rebuilds the same System,
asserts its derived inputs still match (guarding constants drift), and
compares light curves.

Run once from the repo root on a machine with IDL and EXOFASTv2 in
IDL_PATH (regenerate only if the scenario changes):

    poetry run python scripts/make_exofast_tran_reference.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytensor

from exozippy.system import System

FIXTURE = os.path.join("tests", "fixtures", "exofast_tran_parity.json")

# Which EXOFASTv2 checkout produced the reference numbers. Recorded in the
# fixture's "provenance" string, which is the only thing that says WHICH
# EXOFASTv2 the parity test is a parity test against -- so a failure to read
# it is loud (see _exofastv2_commit) rather than a quiet "unknown".
EXOFASTV2_DIR = os.environ.get(
    "EXOFASTV2_DIR", os.path.expanduser("~/old_home/scratch/EXOFASTv2")
)

# ---------------------------------------------------------------------------
# Scenario: the config/params the test will rebuild. File paths are filled
# in at build time (here and in the test) from the time grids below.
# ---------------------------------------------------------------------------

TC = 2459000.0
PERIOD = 3.0

CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b"}],
    "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
    "transit": [
        {"name": "TESS2min", "band": "TESS"},
        {"name": "TESS30min", "band": "TESS", "exptime": 30.0, "ninterp": 10},
    ],
}

USER_PARAMS = {
    "star.0.mass": {"initval": 1.0, "sigma": 0.05},
    "star.0.radius": {"initval": 1.0, "sigma": 0.05},
    "star.0.teff": {"initval": 5800, "sigma": 100},
    "star.0.feh": {"initval": 0.0, "sigma": 0.08},
    "orbit.0.period": {"initval": PERIOD},
    "orbit.0.tc": {"initval": TC},
    "orbit.0.cosi": {"initval": 0.05},
    "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
    "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
    "planet.0.radius": {"initval": 1.0},
    # Kipping (q1, q2) chosen for exact quadratic u1 = 0.4, u2 = 0.2.
    "band.TESS.q1": {"initval": 0.36, "sigma": 0.0},
    "band.TESS.q2": {"initval": 1.0 / 3.0, "sigma": 0.0},
}

EXPTIME30_MIN = 30.0
NINTERP30 = 10


def _time_grid(cadence_min, exptime_min, t14_days):
    """Centered grid at `cadence_min` sampling covering one exposure before
    ingress to one exposure after egress (timestamps are mid-exposure)."""
    cadence = cadence_min / 1440.0
    half_span = t14_days / 2.0 + exptime_min / 1440.0
    n = int(np.ceil(2.0 * half_span / cadence)) + 1
    return TC + (np.arange(n) - (n - 1) / 2.0) * cadence


def _write_lc(path, t):
    np.savetxt(
        path, np.column_stack([t, np.ones_like(t), np.full_like(t, 1e-3)])
    )
    return str(path)


def _build_system(workdir, t2, t30):
    config = json.loads(json.dumps(CONFIG))  # deep copy
    config["transit"][0]["file"] = _write_lc(
        os.path.join(workdir, "lc2min.dat"), t2
    )
    config["transit"][1]["file"] = _write_lc(
        os.path.join(workdir, "lc30min.dat"), t30
    )
    system = System(config, user_params=json.loads(json.dumps(USER_PARAMS)))
    system.prepare()
    model = system.build_model()
    return system, model


def _initial_point_fn(system, model, tensors):
    ip = model.initial_point()
    givens = [
        (rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip
    ]
    return pytensor.function(
        [],
        tensors,
        givens=givens,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )


def extract_inputs(system, model):
    """The exact scalars Transit.build_likelihood consumes, evaluated at
    the model's initial point -- fed verbatim to exofast_tran so both
    codes integrate the identical system regardless of internal constants
    (G, Rsun/AU, ...)."""
    orbits, planets, band = system.orbit, system.planet, system.band
    fn = _initial_point_fn(
        system,
        model,
        [
            orbits.tp.value[0],
            orbits.n.value[0],
            orbits.ecc.value[0],
            orbits.cosw.value[0],
            orbits.sinw.value[0],
            orbits.inc.value[0],
            planets.ar.value[0],
            planets.p.value[0],
            band.u1.value[0],
            band.u2.value[0],
        ],
    )
    tp, n, ecc, cosw, sinw, inc, ar, p, u1, u2 = (float(v) for v in fn())
    return {
        "tp": tp,
        # exofast_tran phases with (t - tp)*2pi/period; EXOZIPPy with
        # (t - tp)*n. Pass 2pi/n as the period so they agree exactly.
        "period": float(2.0 * np.pi / n),
        "e": ecc,
        "omega": float(np.arctan2(sinw, cosw)),
        "inc": inc,
        "ar": ar,
        "p": p,
        "u1": u1,
        "u2": u2,
    }


def eval_model_flux(system, model):
    fn = _initial_point_fn(system, model, system.transit._model_flux_node)
    return np.asarray(fn())


IDL_TEMPLATE = """
pro exofast_tran_parity_gen
  compile_opt idl2
  inc = {inc:.17g}d0
  ar = {ar:.17g}d0
  tp = {tp:.17g}d0
  period = {period:.17g}d0
  e = {e:.17g}d0
  omega = {omega:.17g}d0
  p = {p:.17g}d0
  u1 = {u1:.17g}d0
  u2 = {u2:.17g}d0

  readcol, '{t2_file}', t2, format='D', /silent
  readcol, '{t30_file}', t30, format='D', /silent

  ;; unsmeared, 2-minute cadence
  f2 = exofast_tran(t2, inc, ar, tp, period, e, omega, p, u1, u2, 1d0)

  ;; smeared, 30-minute cadence: EXOFASTv2's sub-exposure grid and
  ;; average, transcribed from exofast_chi2v2.pro
  ninterp = {ninterp}
  exptime = {exptime:.17g}d0 ; minutes
  frac = dindgen(ninterp)/ninterp - (ninterp-1d0)/(2d0*ninterp)
  n30 = n_elements(t30)
  f30 = dblarr(n30)
  for i=0, n30-1 do begin
     tgrid = t30[i] + frac*exptime/1440d0
     f30[i] = total(exofast_tran(tgrid, inc, ar, tp, period, e, omega, $
                                 p, u1, u2, 1d0))/ninterp
  endfor

  openw, lun, '{out2_file}', /get_lun
  for i=0, n_elements(t2)-1 do printf, lun, f2[i], format='(e26.18)'
  free_lun, lun
  openw, lun, '{out30_file}', /get_lun
  for i=0, n30-1 do printf, lun, f30[i], format='(e26.18)'
  free_lun, lun
end
"""


def run_idl(workdir, inputs, t2, t30):
    np.savetxt(os.path.join(workdir, "t2.txt"), t2, fmt="%.17g")
    np.savetxt(os.path.join(workdir, "t30.txt"), t30, fmt="%.17g")
    paths = {
        "t2_file": os.path.join(workdir, "t2.txt"),
        "t30_file": os.path.join(workdir, "t30.txt"),
        "out2_file": os.path.join(workdir, "f2.txt"),
        "out30_file": os.path.join(workdir, "f30.txt"),
    }
    pro = os.path.join(workdir, "exofast_tran_parity_gen.pro")
    with open(pro, "w") as f:
        f.write(
            IDL_TEMPLATE.format(
                ninterp=NINTERP30, exptime=EXPTIME30_MIN, **inputs, **paths
            )
        )
    cmds = f'.compile "{pro}"\nexofast_tran_parity_gen\nexit\n'
    result = subprocess.run(
        ["idl", "-quiet"],
        input=cmds,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0 or not os.path.exists(paths["out2_file"]):
        raise RuntimeError(
            f"IDL failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    f2 = np.loadtxt(paths["out2_file"])
    f30 = np.loadtxt(paths["out30_file"])
    return f2, f30


def _exofastv2_commit():
    """Short commit of the EXOFASTv2 checkout, or '' with a loud warning.

    The path is machine-specific, so this used to fail silently and stamp
    the fixture "EXOFASTv2 unknown" -- which is the one field a parity
    fixture exists to carry.  Point EXOFASTV2_DIR at the checkout.
    """
    result = subprocess.run(
        ["git", "-C", EXOFASTV2_DIR, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"WARNING: could not read the EXOFASTv2 commit from "
            f"{EXOFASTV2_DIR} (rc={result.returncode}): "
            f"{result.stderr.strip()}\n"
            f"         The fixture's provenance will say 'unknown'. Set "
            f"EXOFASTV2_DIR to the checkout that IDL is running from.",
            file=sys.stderr,
        )
        return ""
    return result.stdout.strip()


def main():
    # Approximate T14 (window construction only; exactness is irrelevant
    # to parity -- both codes get the same final grids).
    a_rsun = (PERIOD / 365.25) ** (2.0 / 3.0) * 215.032
    cosi = USER_PARAMS["orbit.0.cosi"]["initval"]
    b = a_rsun * cosi
    p_approx = 0.1028  # 1 Rjup / 1 Rsun
    sini = np.sqrt(1.0 - cosi**2)
    t14 = (PERIOD / np.pi) * np.arcsin(
        np.sqrt((1.0 + p_approx) ** 2 - b**2) / (a_rsun * sini)
    )

    t2 = _time_grid(2.0, 2.0, t14)
    t30 = _time_grid(30.0, EXPTIME30_MIN, t14)
    print(
        f"T14 ~ {t14 * 1440:.1f} min; grids: {len(t2)} x 2 min, "
        f"{len(t30)} x 30 min"
    )

    with tempfile.TemporaryDirectory() as workdir:
        system, model = _build_system(workdir, t2, t30)
        inputs = extract_inputs(system, model)
        print("derived inputs:", json.dumps(inputs, indent=2, sort_keys=True))

        f2_idl, f30_idl = run_idl(workdir, inputs, t2, t30)

        # Preview the achieved agreement (the test asserts this).
        model_flux = eval_model_flux(system, model)
        inst_map = system.transit.inst_map
        diff2 = np.max(np.abs(model_flux[inst_map == 0] - f2_idl))
        diff30 = np.max(np.abs(model_flux[inst_map == 1] - f30_idl))
        print(
            f"max |EXOZIPPy - exofast_tran|: unsmeared {diff2:.3e}, "
            f"smeared {diff30:.3e}"
        )

    git_hash = _exofastv2_commit()

    fixture = {
        "provenance": (
            f"exofast_tran.pro, EXOFASTv2 {git_hash or 'unknown'}, IDL; "
            f"generated by scripts/make_exofast_tran_reference.py"
        ),
        "config": CONFIG,
        "user_params": USER_PARAMS,
        "inputs": inputs,
        "unsmeared": {"time": t2.tolist(), "flux": f2_idl.tolist()},
        "smeared": {
            "time": t30.tolist(),
            "flux": f30_idl.tolist(),
            "exptime": EXPTIME30_MIN,
            "ninterp": NINTERP30,
        },
    }
    with open(FIXTURE, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
