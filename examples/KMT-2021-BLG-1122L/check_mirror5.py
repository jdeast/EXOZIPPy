"""Where does the mirror build's stage-1 t_E = 2.65 d come from?  Print the
probe's full ProbedStart (value, user_value, rank, source) for the t_E
chain in the build named on the command line."""

import copy
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = [
    "mulensevent.0.t_E",
    "mulensevent.0.theta_E",
    "mulensevent.0.pi_rel",
    "mulensevent.0.mu_rel_mag",
    "mulensevent.0.mlens_total",
    "mulensevent.0.pi_E_N",
    "mulensevent.0.pi_E_E",
    "source.0.rho",
    "source.0.u_0",
    "source.0.t_0",
    "star.0.mass",
    "star.1.mass",
    "star.2.mass",
    "star.0.distance",
    "star.3.distance",
    "star.0.pm_ra",
    "star.3.pm_ra",
    "star.0.pm_dec",
    "star.3.pm_dec",
    "star.3.radius",
    "lens.1.alpha",
    "lens.1.q",
]


def main(tag):
    from exozippy.components.mulensing import mulensinstrument as mi
    from exozippy.system import System

    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open("KMT-2021-BLG-1122L.params.yaml"))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    a2, a3, u0, t0 = 48.68, 127.92, -0.4213, 2459370.609
    sign = 1 if tag == "winner" else -1
    p = copy.deepcopy(prm)
    p["source.Source.t_0"] = {"initval": t0}
    p["source.Source.u_0"] = {"initval": sign * u0}
    p["lens.LensB.alpha"] = {"initval": sign * a2}
    p["lens.LensC.log_s"] = {"initval": float(np.log10(1.601))}
    p["lens.LensC.xalpha"] = {"initval": float(np.cos(np.radians(sign * a3)))}
    p["lens.LensC.yalpha"] = {"initval": float(np.sin(np.radians(sign * a3)))}
    orig = mi.MulensInstrument._probe_bootstrap_geometry
    done = []

    def spy(self, system):
        if not done:
            done.append(1)
            ok = []
            for path in PATHS:
                try:
                    r = self.config_manager.probe_start([path])[path]
                    print(
                        f"PROBE[{tag}] {path:28s} value {r.value!r:>22}  user {r.user_value!r:>22}  rank {r.rank!r:>5}  source {r.source!r}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"PROBE[{tag}] {path:28s} RAISED {type(exc).__name__}: {str(exc)[:150]}",
                        flush=True,
                    )
        return orig(self, system)

    mi.MulensInstrument._probe_bootstrap_geometry = spy
    System(copy.deepcopy(cfg), user_params=p).prepare()


if __name__ == "__main__":
    main(sys.argv[1])
