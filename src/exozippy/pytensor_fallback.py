"""Detect a broken PyTensor C toolchain at startup and fall back gracefully.

PyTensor compiles C code at runtime, so ``pip install`` succeeds on a machine
that cannot actually run a fit: a missing g++ or missing Python development
headers (the RHEL-family ``python3.12-devel`` package) only surfaces as a
``CompileError`` deep inside the first ``pytensor.function`` call, long after
the model has been built. ``ensure_usable_backend()`` probes the C backend
with a trivial compile up front and, when it is broken, prints a loud banner
naming the fix and switches PyTensor to its pure-Python backend in-process.

The pure-Python backend alone is not enough: PyMC joins the model's logp
factors with a single variadic ``add`` node, and a model with more than ~31
factors exceeds numpy's 32-operand ufunc limit (``NPY_MAXARGS``), which
``Elemwise.perform`` refuses (inputs + outputs > 32; the C backend has no such
limit). The fallback therefore also registers a graph rewrite that splits any
too-wide variadic ``Add``/``Mul`` into a tree of narrower ones. PyTensor's own
fusion pass already self-caps at the same limit, so pre-existing variadic
nodes are the only offenders.
"""

import logging
import textwrap

logger = logging.getLogger(__name__)

# numpy ufuncs segfault or raise above 32 operands, where operands means
# inputs + outputs. Add/Mul have one output, so at most 31 inputs per node.
_MAX_UFUNC_OPERANDS = 32
_MAX_FANIN = _MAX_UFUNC_OPERANDS - 1

_REWRITE_REGISTERED = False


def _split_variadic(fn, inputs):
    """Rebuild fn(*inputs) as a tree of fn nodes with fan-in <= _MAX_FANIN."""
    while len(inputs) > _MAX_FANIN:
        inputs = [
            fn(*chunk) if len(chunk) > 1 else chunk[0]
            for chunk in (
                inputs[i : i + _MAX_FANIN]
                for i in range(0, len(inputs), _MAX_FANIN)
            )
        ]
    return fn(*inputs)


def register_wide_elemwise_split():
    """Register the >32-operand Add/Mul splitting rewrite (idempotent).

    Registered under the standard fast_run/fast_compile tags so every mode
    picks it up. It fires only on nodes the Python backend cannot execute at
    all, and the split tree is numerically identical (same pairwise-upcast
    rules), so registering it is harmless when the C backend works -- but we
    only bother when falling back, to leave healthy setups byte-for-byte
    untouched.
    """
    global _REWRITE_REGISTERED
    if _REWRITE_REGISTERED:
        return

    import pytensor.scalar as ps
    import pytensor.tensor as pt
    from pytensor.compile import optdb
    from pytensor.graph.rewriting.basic import node_rewriter, out2in
    from pytensor.tensor.elemwise import Elemwise

    @node_rewriter([Elemwise])
    def local_split_wide_variadic(fgraph, node):
        if len(node.inputs) + len(node.outputs) <= _MAX_UFUNC_OPERANDS:
            return None
        if isinstance(node.op.scalar_op, ps.Add):
            fn = pt.add
        elif isinstance(node.op.scalar_op, ps.Mul):
            fn = pt.mul
        else:
            return None
        return [_split_variadic(fn, list(node.inputs))]

    # Position 48.5: after add_mul_fusion (48), which flattens nested adds and
    # so can itself create wide nodes, and before elemwise_fusion (49), which
    # self-caps at the operand limit and creates no new offenders.
    optdb.register(
        "exozippy_split_wide_variadic",
        out2in(local_split_wide_variadic, ignore_newtrees=False),
        "fast_run",
        "fast_compile",
        position=48.5,
    )
    _REWRITE_REGISTERED = True


def check_c_backend():
    """Try a trivial C-backend compile; return None if it works, else the error.

    On a healthy machine the probe module is in PyTensor's compile cache after
    the first ever run, so this costs milliseconds. When the toolchain is
    broken (no g++, or g++ without Python.h) the compile fails fast.
    """
    import pytensor
    import pytensor.tensor as pt

    if not pytensor.config.cxx:
        # PyTensor already found no compiler at import (it warns and clears
        # cxx itself), or the user forced the Python backend via
        # PYTENSOR_FLAGS. Either way there is nothing to probe.
        return "PyTensor has no C compiler configured (config.cxx is empty)"
    try:
        x = pt.dscalar("exozippy_c_backend_probe")
        probe = pytensor.function([x], x + 1.0)
        probe(0.0)
    except Exception as exc:
        return str(exc)
    return None


def activate_python_fallback(reason):
    """Switch PyTensor to the pure-Python backend and make it usable."""
    import pytensor

    pytensor.config.cxx = ""
    try:
        pytensor.config.blas__ldflags = ""
    except Exception:
        pass  # only relevant to C-code BLAS probing; never fatal
    register_wide_elemwise_split()

    hint = ""
    if "Python.h" in reason:
        hint = (
            "\nThe Python development headers are missing. To install them:\n"
            "  RHEL / Rocky / Alma / Fedora:  sudo dnf install gcc-c++ python3.X-devel\n"
            "  Debian / Ubuntu:               sudo apt install g++ python3.X-dev\n"
            "  no root:                       use a conda Python (headers included)\n"
            "(match the package version to your Python; see README 'Runtime\n"
            "requirements')"
        )
    banner = textwrap.dedent(
        """
        {sep}
        WARNING: PyTensor's C backend is unusable on this machine:

        {reason}
        {hint}
        Falling back to PyTensor's pure-Python backend. This works, but it is
        ORDERS OF MAGNITUDE slower -- fine as a smoke test, hopeless for a
        real fit. Fix the toolchain above before fitting anything real.
        {sep}
        """
    ).format(
        sep="!" * 75, reason=textwrap.indent(reason.strip(), "    "), hint=hint
    )
    print(banner, flush=True)
    logger.warning(
        "PyTensor C backend unusable; using pure-Python fallback: %s", reason
    )


def ensure_usable_backend():
    """Probe the C backend; fall back to pure Python (loudly) if broken.

    Called once at the top of ``run_fit``, before anything compiles. Returns
    True when the C backend works, False when the fallback was activated.
    """
    reason = check_c_backend()
    if reason is None:
        return True
    activate_python_fallback(reason)
    return False
