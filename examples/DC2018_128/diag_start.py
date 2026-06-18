"""Diagnostic: evaluate the binary-lens Op inputs at the raw starting point."""
import numpy as np
import yaml
import pytensor
import pytensor.tensor as pt

from exozippy.system import System

config = yaml.safe_load(open("DC2018_128.yaml"))
user_params = yaml.safe_load(open("DC2018_128.params.yaml"))

system = System(config, user_params)
system.prepare()
model = system.build_model()

lens = system.lens
params = lens._get_binary_mm_params(system, 0)

raw_start = system.get_raw_start(model)

names = ["t0", "u0", "tE", "pi_N", "pi_E", "s", "q", "alpha"]
tensors = [params[k] for k in names]
names.append("rho")
tensors.append(lens.rho.value[0])

fn = pytensor.function(model.free_RVs, tensors, on_unused_input="ignore")
inputs = [raw_start[v.name] for v in model.free_RVs]
vals = fn(*inputs)

expected = {"t0": 2458554.8942, "u0": 0.14297, "tE": 18.170,
            "pi_N": 0.0, "pi_E": 0.0, "s": 0.97875, "q": 1.106e-3,
            "alpha": -52.151, "rho": 5.38e-3}
print("Binary params passed to Op at raw start:")
for n, v in zip(names, vals):
    v = float(np.asarray(v).ravel()[0])
    print(f"  {n:8s} = {v:.6f}   (expect ~{expected[n]:.6g})")

# Regenerate the start plots with the corrected raw start
internal_start = system.get_internal_point(model, raw_start)
system.compile_plotter_functions(model)
for comp in system.active_components.values():
    comp.plot(system, [internal_start], filename_prefix="fitresults/DC2018_128_start")
print("start plots written")
