#!/usr/bin/env python3
"""
Convert an MMEXOFAST output JSON to an EXOZIPPy params.yaml seed file.

This is the importable home of the former scripts/mmexofast_to_params.py. The
CLI is defined by build_parser() and driven by main(argv=None);
scripts/mmexofast_to_params.py is now a thin wrapper that calls main().

Usage:
    # All solutions in the file (default) -> list-valued initvals, one
    # mutually-consistent start point per MMEXOFAST fit (P4: multi-seed
    # sampling, config.py's list-initval relaxation-engine extension).
    python scripts/mmexofast_to_params.py examples/DC2018_128/mmexofast.json \\
        --lens-name Lens --out examples/DC2018_128/DC2018_128.params.yaml

    # A single solution -> plain scalar initvals (legacy single-start mode).
    python scripts/mmexofast_to_params.py examples/DC2018_128/mmexofast.json \\
        --lens-name Lens --solution 1 --out examples/DC2018_128/DC2018_128.params.yaml

MMEXOFAST provides initvals and, optionally, estimated uncertainties from a
quick optimization fit to the data.  The uncertainties are mapped to EXOZIPPy
init_scales, which only sets the sampler's initial step size without adding any
logp penalty. They are NOT used as priors, which would double-count the data and
artificially shrink the posterior.

Uncertainties are optional: MMEXOFAST omits the per-fit ``sigmas`` block for
solutions that are initial estimates rather than optimized fits, and is
expected to stop emitting sigmas altogether. Any parameter whose sigma is
absent simply gets no ``init_scale`` line, leaving EXOZIPPy's own default step
size in place. Only ``initval`` is required.

MMEXOFAST gives uncertainties in log space for s, q, rho (log_s, log_q,
log_rho).  Physical init_scale is recovered via first-order propagation:
    sigma_x_physical = x * sigma_ln_x

With multiple solutions, ``initval`` becomes a list (one entry per solution,
in file order) so the relaxation engine solves one mutually-consistent start
point per entry inside a single prepare() call (see config.py's
finalize_user_params / _build_seed_overrides). Bounds and init_scale are NOT
per-seed -- they resolve once, from the first (seed 0) solution -- so this
script always emits a single scalar init_scale even when initval is a list.
"""

import argparse
import json
from pathlib import Path


def _fmt(values, spec):
    """A scalar for one value, a YAML list literal for several."""
    if len(values) == 1:
        return format(values[0], spec)
    return "[" + ", ".join(format(v, spec) for v in values) + "]"


def _scale(fit, key):
    """Linear-space init_scale for ``key``, or None when it is unavailable.

    MMEXOFAST omits ``sigmas`` entirely for solutions that are initial
    estimates rather than optimized fits, and is expected to stop emitting
    them altogether. init_scale is only a sampler hint, so a missing sigma
    means "let EXOZIPPy pick the step size", not an error.
    """
    return (fit.get("sigmas") or {}).get(key)


def _log_scale(fit, log_key, phys_key):
    """Convert a log-space sigma to a physical init_scale: x * sigma_ln_x.

    Returns None when the sigma is unavailable; see :func:`_scale`.
    """
    sigma = _scale(fit, log_key)
    if sigma is None:
        return None

    return fit["parameters"][phys_key] * sigma


def _param_block(path, initval, scale, scale_spec):
    """YAML lines for one parameter, omitting init_scale when it is None."""
    lines = [f"{path}:", f"    initval: {initval}"]
    if scale is not None:
        lines.append(f"    init_scale: {format(scale, scale_spec)}")

    return lines


def mmexofast_to_params(
    json_path, lens_name="Lens", solution_index=None, out_path=None
):
    """Build a params.yaml text seeding ``lens.<lens_name>`` from MMEXOFAST fits.

    ``solution_index=None`` (default) uses every solution in the file, one
    per list entry, in file order (P4 multi-seed sampling). Pass an int to
    restrict to a single solution (legacy scalar-initval behavior).
    """
    with open(json_path) as f:
        data = json.load(f)

    fits = data["fits"]
    n = len(fits)

    if solution_index is None:
        chosen = fits
        indices = list(range(n))
    else:
        if solution_index >= n:
            raise ValueError(
                f"Solution {solution_index} requested but file has only {n} solution(s)"
            )
        chosen = [fits[solution_index]]
        indices = [solution_index]

    multi = len(chosen) > 1

    lines = [
        f"# Seeded from MMEXOFAST solution(s) {indices} (0-indexed)",
        f"# Source: {json_path}",
        f"# n_solutions in file: {n}",
        f"#",
        f"# MMEXOFAST uncertainties are mapped to init_scale, NOT sigma.",
        f"# init_scale = sampling hint only (no logp penalty).",
        f"# sigma = Gaussian prior (would double-count the data).",
    ]
    if multi:
        lines += [
            f"#",
            f"# initval is list-valued: one mutually-consistent start point per",
            f"# solution above (P4 multi-seed sampling -- see config.py's",
            f"# finalize_user_params). init_scale/bounds are NOT per-seed and",
            f"# always come from the first (seed 0) solution.",
        ]
    lines.append("")

    lines += _param_block(
        f"lens.{lens_name}.t_0",
        _fmt([fit["parameters"]["t_0"] for fit in chosen], ".8f"),
        _scale(chosen[0], "t_0"),
        ".8f",
    )
    lines.append("")
    lines += _param_block(
        f"lens.{lens_name}.u_0",
        _fmt([fit["parameters"]["u_0"] for fit in chosen], ".8f"),
        _scale(chosen[0], "u_0"),
        ".8f",
    )
    lines += [
        f"",
        f"# t_E is derived in EXOZIPPy from stellar masses/distances/proper motions.",
        f"# Provided here as an initval hint to seed the relaxation engine.",
    ]
    lines += _param_block(
        f"lens.{lens_name}.t_E",
        _fmt([fit["parameters"]["t_E"] for fit in chosen], ".8f"),
        _scale(chosen[0], "t_E"),
        ".8f",
    )
    lines.append("")
    lines += _param_block(
        f"lens.{lens_name}.s",
        _fmt([fit["parameters"]["s"] for fit in chosen], ".8f"),
        _log_scale(chosen[0], "log_s", "s"),
        ".8f",
    )
    lines += [
        f"",
        f"# alpha: relaxation engine propagates initval/init_scale to xalpha/yalpha.",
    ]
    lines += _param_block(
        f"lens.{lens_name}.alpha",
        _fmt([fit["parameters"]["alpha"] for fit in chosen], ".8f"),
        _scale(chosen[0], "alpha"),
        ".8f",
    )

    rhos = [fit["parameters"].get("rho", 0.0) for fit in chosen]
    use_rho = any(r > 1e-10 for r in rhos)
    if use_rho:
        lines.append("")
        lines += _param_block(
            f"lens.{lens_name}.rho",
            _fmt(rhos, ".8e"),
            _log_scale(chosen[0], "log_rho", "rho"),
            ".8e",
        )
    else:
        lines += [
            f"",
            f"# rho ~ 0 in {'every' if multi else 'this'} solution; finite_source: False in YAML is appropriate",
        ]

    qs = [fit["parameters"].get("q") for fit in chosen]
    if all(q is not None for q in qs):
        lines += [
            f"",
            f"# q = M_companion / M_primary.  EXOZIPPy's relaxation engine propagates",
            f"# this through the symbolic relation  q * M_primary = M_companion",
            f"# to set the companion's mass initval automatically.",
        ]
        lines += _param_block(
            f"lens.{lens_name}.q",
            _fmt(qs, ".8e"),
            _log_scale(chosen[0], "log_q", "q"),
            ".8e",
        )

    text = "\n".join(lines) + "\n"

    if out_path:
        Path(out_path).write_text(text)
        print(f"Wrote {out_path}")
    else:
        print(text)

    return text


def build_parser():
    """Return the argparse parser for the mmexofast_to_params utility."""
    ap = argparse.ArgumentParser(
        prog="mmexofast_to_params.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("json", help="Path to mmexofast.json")
    ap.add_argument(
        "--lens-name",
        default="Lens",
        help="Lens component name in YAML (default: Lens)",
    )
    ap.add_argument(
        "--solution",
        type=int,
        default=None,
        help="Restrict to a single solution, 0-indexed (default: use "
        "every solution in the file as list-valued initvals)",
    )
    ap.add_argument("--out", help="Output params.yaml path (default: stdout)")
    return ap


def main(argv=None):
    """CLI entry point. Parses argv (or sys.argv) and runs the conversion."""
    args = build_parser().parse_args(argv)
    mmexofast_to_params(args.json, args.lens_name, args.solution, args.out)


if __name__ == "__main__":
    main()
