"""Choose how many xdist workers this machine can actually afford.

Why this is computed rather than written down
---------------------------------------------
``pyproject.toml`` pins ``-n 6`` for a workstation and CI used to override it
with a hardcoded ``-n 2``. That 2 was chosen when a GitHub-hosted Linux
runner had two cores, and it was never revisited: the runners have grown
since, so the number bought headroom nobody needed while the suite's wall
clock tripled. Measured on 2026-08-24, the four matrix jobs did the SAME
amount of work per worker as this 36-core workstation does -- ubuntu 3.12
spent 5202 worker-seconds against the workstation's 5682 -- so at that point
CI's whole 43-minute gap over a local 15:47 was worker COUNT, with the
compile cache warm on both.

The wrong fix is a new hardcoded number, because the reason the old one went
stale is that it was hardcoded. Both inputs are cheap to read at startup, so
this reads them.

The rule
--------
``min(cpu_count, memory_gb // _GB_PER_WORKER)``, floored at 1.

The memory term is the binding one and it is not conservatism. Each worker
peaks at roughly 1-2 GB while building a System and compiling PyTensor
graphs, and the failure when that is exceeded is not an assertion but
``worker 'gwN' crashed``, landing on whichever heavy test drew the short
straw -- it moved between test_integration_kelt4, test_gp and
test_sed_plot_data across runs and vanished on others. A memory ceiling
presents as a flaky, wandering test failure, so the headroom is worth more
than the last worker.

``_GB_PER_WORKER = 3`` is that 1-2 GB peak plus room for the interpreter,
the restored compile cache's page cache, and the fact that the peak is a
measured typical rather than a bound.

Deliberately NOT used to pick the local default. A developer's box is shared
with editors, browsers and hours-long interactive fits, and this cannot see
any of that; ``addopts`` keeps its explicit ``-n 6``, and this script exists
for the one caller that has a machine to itself.
"""

from __future__ import annotations

import argparse
import os
import sys

# See the module docstring: measured per-worker peak is 1-2 GB, and the
# failure mode for getting this wrong is a wandering "worker crashed".
_GB_PER_WORKER = 3


def cpu_count() -> int:
    """Cores this process may actually use, not cores the machine has.

    ``os.process_cpu_count()`` honours the CPU affinity mask, which is what a
    container-scheduled CI runner sets; ``os.cpu_count()`` reports the host's
    total and would over-subscribe. The fallback is for pre-3.13.
    """
    getter = getattr(os, "process_cpu_count", None)
    if getter is not None:
        return getter() or 1
    return os.cpu_count() or 1


def memory_gb() -> float | None:
    """Total physical memory in GB, or None if it cannot be determined.

    Two probes because the two platforms this runs on report it differently
    and neither has a portable answer in the standard library. Returning None
    rather than a guess is deliberate: the caller then falls back to the CPU
    term alone, which is the behaviour we had before this existed, instead of
    silently choosing a worker count from a fabricated memory figure.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    else:
        if pages > 0 and page_size > 0:
            return pages * page_size / 1024**3

    # macOS reports SC_PHYS_PAGES on recent versions, but has not always, and
    # sysctl is the documented interface there.
    if sys.platform == "darwin":
        try:
            import subprocess  # noqa: PLC0415 -- only needed on this path

            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return int(out.stdout.strip()) / 1024**3
        except Exception:
            return None
    return None


def choose_workers(
    cpus: int, mem_gb: float | None, gb_per_worker: int = _GB_PER_WORKER
) -> int:
    """The rule from the module docstring, floored at 1."""
    if mem_gb is None:
        return max(1, cpus)
    by_memory = int(mem_gb // gb_per_worker)
    return max(1, min(cpus, by_memory))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print the number of pytest-xdist workers this machine can "
            "afford, from its core count and its physical memory."
        )
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "also print the inputs on stderr. CI uses this so the chosen "
            "number is recorded in the log rather than inferred later from "
            "a runner-spec guess."
        ),
    )
    args = parser.parse_args(argv)

    cpus = cpu_count()
    mem = memory_gb()
    workers = choose_workers(cpus, mem)
    if args.explain:
        shown = "unknown" if mem is None else f"{mem:.1f} GB"
        print(
            f"pytest workers: {workers} "
            f"(cpus={cpus}, memory={shown}, {_GB_PER_WORKER} GB/worker)",
            file=sys.stderr,
        )
    print(workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
