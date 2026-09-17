"""Copy a run config and set `sampler.cores` to what the scheduler granted.

WHY THIS EXISTS.  Review 2.4.11's ruling is that EXOZIPPy respects the
user's `cores:` and does NOT probe the scheduler -- correctly, since most
machines have no scheduler to probe.  The consequence is that the JOB has
to fill it in, because only the job knows what it was granted.  With a
fixed `-pe mthread 64` that was harmless; with a RANGE it is essential:
SGE may grant 32, and a config still claiming 64 oversubscribes the node by
2x, which is exactly the defect 2.4.11 is about.

WHY IT COPIES RATHER THAN EDITS IN PLACE.  The configs under sweep/ and
ab194/ are committed artifacts that other array tasks -- and other
concurrently running jobs -- read at startup.  Rewriting one while a
neighbour is reading it is a race for no benefit, and it would also make
the committed file differ from what was actually run.  The copy lands in
the job's own scratch, so what ran is reproducible from the committed
config plus the granted core count, and nothing shared is touched.

This is safe to point at any config whose internal paths are ABSOLUTE,
which is what dc18_sweep_config.py writes (prefix, parameter_file, the SED
file, the light curves and the seed JSON).  A config with relative paths
would break when run from a different directory, so it refuses those
rather than producing a fit that silently cannot find its data.
"""

import argparse
import io
import os
import sys

import yaml

# Keys whose values are paths the fit must be able to open from wherever it
# is run.  `file:` is checked per light curve inside mulensinstrument.
_PATH_KEYS = ("prefix", "parameter_file")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--cores", type=int, required=True)
    a = ap.parse_args()

    if a.cores < 1:
        sys.exit("--cores must be >= 1 (got %d); is NSLOTS set?" % a.cores)

    cfg = yaml.safe_load(io.open(a.config, encoding="utf-8"))

    # An empty or comment-only config parses to None, and every access below
    # then dies as "'NoneType' object has no attribute 'get'" -- which names
    # neither the file nor the problem.  Measured 2026-09-16: ab194's av_true
    # arm was a 0-byte DC2018_194.yaml (truncated when the arm was set up),
    # and the array task burned a 64-slot allocation to print that traceback.
    if not isinstance(cfg, dict):
        sys.exit(
            "%s holds no YAML mapping (parsed as %s). An arm config that was "
            "truncated or never written looks exactly like this; rebuild it "
            "from a sibling arm and remember that prefix, parameter_file AND "
            "sed.file all name the arm." % (a.config, type(cfg).__name__)
        )

    bad = [
        k
        for k in _PATH_KEYS
        if isinstance(cfg.get(k), str) and not os.path.isabs(cfg[k])
    ]
    for inst in cfg.get("mulensinstrument") or []:
        f = inst.get("file")
        if isinstance(f, str) and not os.path.isabs(f):
            bad.append("mulensinstrument.file=%s" % f)
    sed = cfg.get("sed")
    if (
        isinstance(sed, dict)
        and isinstance(sed.get("file"), str)
        and not os.path.isabs(sed["file"])
    ):
        bad.append("sed.file=%s" % sed["file"])
    if bad:
        sys.exit(
            "refusing to relocate a config with relative paths (%s): the "
            "copy would run from a different directory and silently fail to "
            "find its inputs.  Regenerate with dc18_sweep_config.py, which "
            "writes absolute paths." % ", ".join(bad)
        )

    old = (cfg.get("sampler") or {}).get("cores")
    cfg.setdefault("sampler", {})["cores"] = int(a.cores)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    io.open(a.out, "w", encoding="utf-8").write(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    )
    print(
        "cores %s -> %d   %s -> %s" % (old, a.cores, a.config, a.out),
        flush=True,
    )


if __name__ == "__main__":
    main()
