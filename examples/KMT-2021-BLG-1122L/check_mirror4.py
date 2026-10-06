"""Is the stage-1 bootstrap probe ORDER-dependent (state leaking between
System builds in one process) or SIGN-dependent?  Build the configurations
named on the command line, in that order, in this one process, and print
the probe geometry each build saw."""

import copy
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main(order):
    from exozippy.components.mulensing import mulensinstrument as mi
    from exozippy.system import System

    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open("KMT-2021-BLG-1122L.params.yaml"))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    a2, a3, u0, t0 = 48.68, 127.92, -0.4213, 2459370.609
    seen = []
    orig = mi.MulensInstrument._probe_bootstrap_geometry

    def spy(self, system):
        g = orig(self, system)
        seen.append(
            {
                p: g(p)
                for p in (
                    "mulensevent.0.t_E",
                    "mulensevent.0.pi_E_E",
                    "source.0.rho",
                    "source.0.u_0",
                    "lens.1.alpha",
                )
            }
        )
        return g

    mi.MulensInstrument._probe_bootstrap_geometry = spy
    for tag in order:
        sign = 1 if tag == "winner" else -1
        p = copy.deepcopy(prm)
        p["source.Source.t_0"] = {"initval": t0}
        p["source.Source.u_0"] = {"initval": sign * u0}
        p["lens.LensB.alpha"] = {"initval": sign * a2}
        p["lens.LensC.log_s"] = {"initval": float(np.log10(1.601))}
        p["lens.LensC.xalpha"] = {
            "initval": float(np.cos(np.radians(sign * a3)))
        }
        p["lens.LensC.yalpha"] = {
            "initval": float(np.sin(np.radians(sign * a3)))
        }
        seen.clear()
        System(copy.deepcopy(cfg), user_params=p).prepare()
        g = seen[0] if seen else {}
        print(
            f"ORDER {'>'.join(order)} BUILD {tag}: t_E {g.get('mulensevent.0.t_E')} pi_E_E {g.get('mulensevent.0.pi_E_E')} rho {g.get('source.0.rho')} u_0 {g.get('source.0.u_0')} alpha {g.get('lens.1.alpha')}  (probes {len(seen)})",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
