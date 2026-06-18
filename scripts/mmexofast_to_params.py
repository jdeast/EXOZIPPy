#!/usr/bin/env python3
"""
Convert an MMEXOFAST output JSON to an EXOZIPPy params.yaml seed file.

Usage:
    python scripts/mmexofast_to_params.py examples/DC2018_128/mmexofast.json \\
        --lens-name Lens --out examples/DC2018_128/DC2018_128.params.yaml

MMEXOFAST provides initvals and estimated uncertainties from a quick
optimization fit to the data.  The uncertainties are mapped to EXOZIPPy
init_scales, which only sets the sampler's initial step size without adding any
logp penalty. They are NOT used as priors, which would double-count the data and
artificially shrink the posterior.

MMEXOFAST gives uncertainties in log space for s, q, rho (log_s, log_q,
log_rho).  Physical init_scale is recovered via first-order propagation:
    sigma_x_physical = x * sigma_ln_x
"""

import argparse
import json
import math
from pathlib import Path


def mmexofast_to_params(json_path, lens_name="Lens", solution_index=0, out_path=None):
    with open(json_path) as f:
        data = json.load(f)

    fits = data["fits"]
    n = len(fits)
    if solution_index >= n:
        raise ValueError(f"Solution {solution_index} requested but file has only {n} solution(s)")

    fit = fits[solution_index]
    params = fit["parameters"]
    sigmas = fit["sigmas"]

    def log_scale(log_key, phys_key):
        """Convert log-space sigma to physical init_scale: x * sigma_ln_x."""
        return params[phys_key] * sigmas[log_key]


    q = params.get("q", None)
    rho = params.get("rho", 0.0)
    use_rho = rho > 1e-10

    lines = [
        f"# Seeded from MMEXOFAST solution {solution_index} (0-indexed)",
        f"# Source: {json_path}",
        f"# n_solutions in file: {n}",
        f"#",
        f"# MMEXOFAST uncertainties are mapped to init_scale, NOT sigma.",
        f"# init_scale = sampling hint only (no logp penalty).",
        f"# sigma = Gaussian prior (would double-count the data).",
        f"",
        f"lens.{lens_name}.t_0:",
        f"    initval: {params['t_0']:.8f}",
        f"    init_scale: {sigmas['t_0']:.8f}",
        f"",
        f"lens.{lens_name}.u_0:",
        f"    initval: {params['u_0']:.8f}",
        f"    init_scale: {sigmas['u_0']:.8f}",
        f"",
        f"# t_E is derived in EXOZIPPy from stellar masses/distances/proper motions.",
        f"# Provided here as an initval hint to seed the relaxation engine.",
        f"lens.{lens_name}.t_E:",
        f"    initval: {params['t_E']:.8f}",
        f"    init_scale: {sigmas['t_E']:.8f}",
        f"",
        f"lens.{lens_name}.s:",
        f"    initval: {params['s']:.8f}",
        f"    init_scale: {log_scale('log_s', 's'):.8f}",
        f"",
        f"# alpha: relaxation engine propagates initval/init_scale to cosalpha/sinalpha.",
        f"lens.{lens_name}.alpha:",
        f"    initval: {params['alpha']:.8f}",
        f"    init_scale: {sigmas['alpha']:.8f}",
    ]

    if use_rho:
        lines += [
            f"",
            f"lens.{lens_name}.rho:",
            f"    initval: {rho:.8e}",
            f"    init_scale: {log_scale('log_rho', 'rho'):.8e}",
        ]
    else:
        lines += [
            f"",
            f"# rho ~ 0 in this solution; finite_source: False in YAML is appropriate",
        ]

    if q is not None:
        scale_q = log_scale("log_q", "q")
        lines += [
            f"",
            f"# q = M_companion / M_primary.  EXOZIPPy's relaxation engine propagates",
            f"# this through the symbolic relation  q * M_primary = M_companion",
            f"# to set the companion's mass initval automatically.",
            f"lens.{lens_name}.q:",
            f"    initval: {q:.8e}",
            f"    init_scale: {scale_q:.8e}",
        ]

    text = "\n".join(lines) + "\n"

    if out_path:
        Path(out_path).write_text(text)
        print(f"Wrote {out_path}")
    else:
        print(text)

    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="Path to mmexofast.json")
    ap.add_argument("--lens-name", default="Lens", help="Lens component name in YAML (default: Lens)")
    ap.add_argument("--solution", type=int, default=0,
                    help="Which solution to use, 0-indexed (default: 0)")
    ap.add_argument("--out", help="Output params.yaml path (default: stdout)")
    args = ap.parse_args()
    mmexofast_to_params(args.json, args.lens_name, args.solution, args.out)
