"""Staleness metadata for saved traces (``<prefix>_trace.nc``).

A trace only means anything under the model it was sampled from: its raw
draws decode to physical values through THIS build's bounds, links and
whitening.  Reloading one -- ``sampler: {recompute_trace: false}`` with an
existing trace, or the ``exozippy-modes`` reprocessing CLI -- under an
edited config therefore does not fail; it silently relabels the old draws
with the new parameter names and regenerates every report (posteriors,
LaTeX tables, plots) from them.  Nothing downstream can notice, because
everything downstream succeeds.

So the structural fingerprint of the config + params that produced a trace
is stamped into the file's root attrs when it is written, and checked on
every reload.  The fingerprint is :func:`exozippy.evaluator.structural_hash`
-- the same function the GUI uses to decide "the config changed, re-Solve"
-- covering the component set, per-parameter bounds / fixed-ness / link
wiring, and the data file list.  A second attr carries the payload the hash
was taken over, so a mismatch can name WHAT changed, not only that
something did.

Policy:

  * hash matches       -> silent.
  * hash mismatches    -> :class:`StaleTraceError`.  The trace cannot be
    repaired at load time, and the project is under active development, so
    "probably stale" is in practice "broken"; re-sampling is cheap
    (``recompute_trace: true``) and regenerating a report from foreign
    draws is not recoverable after the fact.
  * hash absent        -> one warning, worded as UNVERIFIABLE (not stale),
    and the load proceeds.  Traces written before this check existed are
    legitimate and must keep working.

The neighboring reload in ``whitening`` splits along the same line, and for
the same reason.  On a run that is about to sample, a whitening mismatch
merely costs a probe (``load_whitening`` warns and re-measures): the
coordinates are still a free choice.  On the REUSE path they are not -- the
draws being decoded were sampled in the old coordinates -- so
``whitening.restore_whitening_for_trace`` raises ``StaleWhiteningError``,
exactly as this module raises here.  The draws are already drawn either way.

The code that produced the trace is recorded alongside the fingerprint (the
package version, and the git commit / describe / dirty flag of the source
tree it ran from), purely so a StaleTraceError can say WHICH code made those
draws and print the git incantation to get back to it.  It is DIAGNOSTIC
ONLY: nothing here ever compares versions, and a version or commit
difference never raises.  A user on newer code whose model is structurally
unchanged must keep being able to reuse a trace -- only the structural hash
decides staleness.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._version import __version__

logger = logging.getLogger(__name__)

# Root-group attrs on the InferenceData.  Prefixed so they can never
# collide with ArviZ's or a sampler's own metadata.
HASH_ATTR = "exozippy_structural_hash"
PAYLOAD_ATTR = "exozippy_structural_payload"
VERSION_ATTR = "exozippy_version"
COMMIT_ATTR = "exozippy_git_commit"
DESCRIBE_ATTR = "exozippy_git_describe"
DIRTY_ATTR = "exozippy_git_dirty"
# Per-element role masks of the parameters whose vectors are NOT uniform in a
# role (see element_roles).  Read by mkparam, which has no System; absent on
# every trace written before per-element roles existed, and on every model whose
# vectors are uniform -- so a reader must treat "missing" as "all sampled",
# which is what it always meant.
ROLES_ATTR = "exozippy_element_roles"

# Which unit system the posterior's sampled variables are stored in.  run.py
# runs _convert_posterior_to_user_units on the way to netCDF, so every trace
# this code writes says "user" -- but a trace written before that conversion
# existed holds INTERNAL-unit draws that are numerically plausible and
# indistinguishable from user-unit ones (radians against degrees, solar
# against jupiter masses).  Nothing downstream can detect that, so the stamp
# is what makes the difference statable at all.
UNITS_ATTR = "exozippy_posterior_units"
POSTERIOR_UNITS = "user"

# The payload is a debugging aid, not the check itself; a pathological
# config (thousands of per-parameter entries) should not bloat the trace
# file.  Past this many characters only the hash is stored, and a mismatch
# reports "no detail available".
_MAX_PAYLOAD_CHARS = 200_000

_BANNER = "!" * 70


class StaleTraceError(RuntimeError):
    """A saved trace was sampled under a structurally different model."""


def _attrs(idata) -> Dict[str, Any]:
    """Root-group attrs of an InferenceData, or an empty dict."""
    attrs = getattr(idata, "attrs", None)
    return attrs if isinstance(attrs, dict) else {}


def _fingerprint_of(source) -> tuple:
    """Accept a System (anything with structural_fingerprint()) or a ready
    ``(hash, payload)`` pair.  The pair form lets a caller that already has a
    built System -- run.py handing its fingerprint to mkparam -- pass it
    straight through instead of recomputing it from a config dict that stage
    1-2 may since have written into."""
    if hasattr(source, "structural_fingerprint"):
        return source.structural_fingerprint()
    fingerprint, payload = source
    return fingerprint, payload


# ---------------------------------------------------------------------------
# Code provenance (diagnostic only -- never a staleness criterion)
# ---------------------------------------------------------------------------

_provenance_cache: Optional[Dict[str, Any]] = None


def _git(args: List[str], cwd) -> Optional[str]:
    """Run a git command, returning its stripped stdout or None.

    Never raises and never blocks: git may be absent (installed wheel, slim
    container), the source may not be in a repo, or the repo may be on a
    stalled network mount.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd)] + args,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def code_provenance(refresh: bool = False) -> Dict[str, Any]:
    """Identify the code that is running, as far as it can be identified.

    Returns ``{version, commit, describe, dirty}``.  ``commit``/``describe``
    are None when the package is not running from a git checkout (an
    installed wheel, or git unavailable) -- that is a normal, supported
    state, not an error.  ``dirty`` reports UNCOMMITTED TRACKED changes: a
    dirty tree means the commit alone does not reproduce the code that ran.
    Untracked files are ignored on purpose -- example data, scratch outputs
    and notes sit untracked in a working EXOZIPPy tree and would otherwise
    flag every run dirty.

    Cached: the git calls run once per process.
    """
    global _provenance_cache
    if _provenance_cache is not None and not refresh:
        return _provenance_cache

    src_dir = Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], src_dir)
    describe = (
        _git(["describe", "--tags", "--always", "--dirty"], src_dir)
        if commit
        else None
    )
    dirty = describe.endswith("-dirty") if describe else None
    _provenance_cache = {
        "version": __version__,
        "commit": commit,
        "describe": describe,
        "dirty": dirty,
    }
    return _provenance_cache


def _repo_root() -> Optional[str]:
    """Top level of the git checkout this code is running from, if any."""
    return _git(
        ["rev-parse", "--show-toplevel"], Path(__file__).resolve().parent
    )


def describe_trace_provenance(attrs: Dict[str, Any]) -> List[str]:
    """Say which code produced a trace, and how to get back to it.

    Degrades in three steps: full detail for a trace stamped from a git
    checkout; version-only for one stamped from an installed package; and a
    plain "not recorded" statement -- never a broken git command -- for a
    trace written before this metadata existed.
    """
    version = attrs.get(VERSION_ATTR)
    commit = attrs.get(COMMIT_ATTR)
    describe = attrs.get(DESCRIBE_ATTR)
    dirty = str(attrs.get(DIRTY_ATTR, "")).lower() == "true"

    if not version and not commit:
        return [
            "The trace records no exozippy version or commit (it predates "
            "that metadata), so the code that produced it cannot be "
            "identified from the file."
        ]

    who = (
        f"exozippy {version}" if version else "an unrecorded exozippy version"
    )
    if describe:
        who += f" (git {describe})"
    lines = [f"The trace was produced with {who}."]

    if not commit:
        lines.append(
            "No git commit was recorded -- it ran from an installed package, "
            "not a checkout -- so that exact source cannot be checked out."
        )
        return lines

    root = _repo_root() or "/path/to/your/EXOZIPPy"
    short = commit[:12]
    lines.append("To run that exact code without disturbing your tree:")
    lines.append(
        f"    git -C {root} worktree add ../exozippy-{short} {commit}"
    )
    lines.append(f"    cd ../exozippy-{short} && poetry install")
    if dirty:
        lines.append(
            "    CAVEAT: that tree had uncommitted changes when the trace was "
            "written, so the commit alone does not reproduce the code that ran."
        )
    return lines


def element_roles(system) -> Dict[str, Dict[str, List[bool]]]:
    """Per-element role masks of every built Parameter, as plain JSON types.

    ``{label: {"sampled": [...], "derived": [...], "active": [...]}}``, and only
    for parameters whose vector is not uniform in a role -- a fully sampled
    vector says nothing a reader cannot infer.

    This exists for ``mkparam``, which writes the next params.yaml from a trace
    plus a config and deliberately never builds a System (see its module
    docstring).  Without the roles it cannot tell WHICH elements of a partially
    derived vector are sampled: the raw variable's length says how many, not
    which, so it would emit a start value for an element whose value is an
    expression -- a redundant constraint on the next fit, which is exactly what
    its "only include physically sampled variables" filter exists to prevent.
    """
    out: Dict[str, Dict[str, List[bool]]] = {}
    get_params = getattr(system, "get_all_parameters", None)
    if not callable(get_params):
        return out
    for p in get_params():
        roles = {}
        for key, attr, default in (
            ("sampled", "is_sampled", True),
            ("derived", "is_derived", False),
            ("active", "is_active", True),
        ):
            mask = np.atleast_1d(getattr(p, attr, default))
            if mask.dtype != bool or mask.size <= 1:
                continue
            if bool(np.all(mask)) or not bool(np.any(mask)):
                continue  # uniform: nothing a reader needs told
            roles[key] = [bool(b) for b in mask]
        if roles:
            out[p.label] = roles
    return out


def stamp_structural_metadata(idata, source) -> None:
    """Record the structural fingerprint + code provenance in root attrs.

    ``source`` is a System (or a ``(hash, payload)`` pair).  Called
    immediately before the trace is written to netCDF, so every trace this
    code writes can be checked when it is read back.
    """
    fingerprint, payload = _fingerprint_of(source)
    attrs = _attrs(idata)
    attrs[HASH_ATTR] = fingerprint
    # The save path converts the posterior to user units immediately before
    # this call, so the stamp is a statement of fact about the array being
    # written -- not a promise about a future format.
    attrs[UNITS_ATTR] = POSTERIOR_UNITS
    roles = element_roles(source)
    if roles:
        attrs[ROLES_ATTR] = json.dumps(roles, sort_keys=True)
    else:
        attrs.pop(ROLES_ATTR, None)
    blob = json.dumps(payload, sort_keys=True, default=str)
    if len(blob) <= _MAX_PAYLOAD_CHARS:
        attrs[PAYLOAD_ATTR] = blob
    else:
        attrs.pop(PAYLOAD_ATTR, None)

    # Diagnostic context for a future mismatch; never compared on load.
    prov = code_provenance()
    attrs[VERSION_ATTR] = str(prov["version"])
    if prov["commit"]:
        attrs[COMMIT_ATTR] = prov["commit"]
        attrs[DESCRIBE_ATTR] = prov["describe"] or ""
        attrs[DIRTY_ATTR] = "true" if prov["dirty"] else "false"


def _diff_mapping(old: dict, new: dict, label: str) -> List[str]:
    """Human-readable added/removed/changed lines for two flat-ish dicts."""
    lines = []
    for key in sorted(set(new) - set(old)):
        lines.append(f"{label} added: {key}")
    for key in sorted(set(old) - set(new)):
        lines.append(f"{label} removed: {key}")
    for key in sorted(set(old) & set(new)):
        if old[key] != new[key]:
            lines.append(
                f"{label} changed: {key}: {old[key]!r} -> {new[key]!r}"
            )
    return lines


def describe_structural_diff(
    old_payload: Optional[dict], new_payload: dict, max_lines: int = 12
) -> List[str]:
    """Describe how two structural payloads differ, as short text lines.

    Returns a "no detail" line when the saved payload is unavailable (an
    over-size payload was dropped at save time, or the attr was lost).
    """
    if not isinstance(old_payload, dict):
        return [
            "(the trace stores no structural detail, only its hash, so the "
            "specific difference cannot be shown)"
        ]

    lines: List[str] = []
    lines += _diff_mapping(
        old_payload.get("components", {}) or {},
        new_payload.get("components", {}) or {},
        "component",
    )
    old_files = set(old_payload.get("files", []) or [])
    new_files = set(new_payload.get("files", []) or [])
    for path in sorted(new_files - old_files):
        lines.append(f"data file added: {path}")
    for path in sorted(old_files - new_files):
        lines.append(f"data file removed: {path}")
    lines += _diff_mapping(
        old_payload.get("params", {}) or {},
        new_payload.get("params", {}) or {},
        "parameter",
    )

    if not lines:
        # The hashes disagree but every section compares equal: the payload
        # was written by a different version of structural_payload.
        lines = [
            "(the hashes disagree but the stored detail compares equal -- the "
            "trace was written by a different exozippy version)"
        ]
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... and {extra} more difference(s)"]
    return lines


def check_posterior_units(attrs, trace_path) -> str:
    """Report which unit system a reloaded trace's posterior is stored in.

    Returns ``"user"``, ``"unverifiable"`` (no stamp) or ``"unknown"`` (a
    stamp this code does not recognize).  Warns for the latter two; never
    raises, and never converts.

    **Warn, not raise, and deliberately not the same call as staleness.** A
    stale trace is a DETECTED mismatch and cannot be repaired, so it raises.
    A missing units stamp is the ``trace_meta`` unverifiable case: it means
    the trace predates the stamp, and the overwhelming majority of those are
    perfectly fine -- the posterior conversion is older than the stamp, so
    only a genuinely pre-2026 trace holds internal-unit draws. Refusing them
    all would invalidate working traces to catch a shrinking population.

    **And never convert.** Multiplying by the conversion factor on a guess
    would corrupt every trace whose draws were already in user units, which
    is nearly all of them; there is no way to tell the two apart from the
    numbers (radians against degrees, solar against jupiter masses are all
    plausible), which is precisely why the stamp had to exist.  The remedy
    is re-sampling, and the warning says so.
    """
    stamped = attrs.get(UNITS_ATTR)
    if stamped == POSTERIOR_UNITS:
        return POSTERIOR_UNITS
    if stamped:
        logger.warning(
            f"{_BANNER}\n"
            f"UNKNOWN POSTERIOR UNITS: {trace_path} declares its posterior "
            f"is stored in '{stamped}' units, which this version of EXOZIPPy "
            f"does not know how to read (it writes '{POSTERIOR_UNITS}'). "
            f"Every reported value may be wrong by a unit conversion factor. "
            f"Upgrade EXOZIPPy, or re-sample with "
            f"'sampler: {{recompute_trace: true}}'.\n{_BANNER}"
        )
        return "unknown"
    logger.warning(
        f"{_BANNER}\n"
        f"UNVERIFIABLE POSTERIOR UNITS: {trace_path} carries no unit stamp, "
        f"so it cannot be confirmed to hold USER-unit draws. Traces written "
        f"before the posterior conversion existed (pre-2026) hold INTERNAL-"
        f"unit draws -- radians rather than degrees, solar rather than "
        f"jupiter masses -- which are numerically plausible and cannot be "
        f"told apart from the numbers, so nothing here converts or refuses. "
        f"This is NOT a detected mismatch -- proceeding. If the reported "
        f"values look wrong by a unit conversion factor, re-sample with "
        f"'sampler: {{recompute_trace: true}}'.\n{_BANNER}"
    )
    return "unverifiable"


def check_trace_freshness(idata, source, trace_path) -> str:
    """Verify a reloaded trace was sampled from this model.

    ``source`` is a System (or a ``(hash, payload)`` pair).  Returns
    ``"match"`` or ``"unverifiable"``; raises :class:`StaleTraceError` when
    the stored fingerprint disagrees with the current config + params.  Only
    that fingerprint is compared -- the recorded version/commit are printed
    in the error but never tested, so newer code with an unchanged model
    reloads its trace silently.

    Also runs :func:`check_posterior_units`, whose result is advisory and
    does not change the return value: which units the draws are stored in is
    a different question from whether they came from this model, it can only
    ever warn, and every reload should ask it -- which is what makes this
    single choke point the right place for it.
    """
    fingerprint, payload = _fingerprint_of(source)
    attrs = _attrs(idata)
    stored = attrs.get(HASH_ATTR)
    check_posterior_units(attrs, trace_path)

    if not stored:
        logger.warning(
            f"{_BANNER}\n"
            f"UNVERIFIABLE TRACE: {trace_path} carries no structural "
            f"fingerprint, so it cannot be checked against the model this "
            f"config builds. It predates the check (or was written by "
            f"another tool). This is NOT a detected mismatch -- proceeding. "
            f"If the config or parameter file has been edited since the "
            f"trace was sampled, re-sample with "
            f"'sampler: {{recompute_trace: true}}'.\n{_BANNER}"
        )
        return "unverifiable"

    if stored == fingerprint:
        logger.debug(
            f"Trace {trace_path} matches the current model "
            f"(structural hash {fingerprint[:12]})."
        )
        return "match"

    try:
        old_payload = json.loads(attrs.get(PAYLOAD_ATTR, "null"))
    except (TypeError, ValueError):
        old_payload = None
    detail = "\n".join(
        f"  - {line}"
        for line in describe_structural_diff(old_payload, payload)
    )
    provenance = "\n".join(describe_trace_provenance(attrs))
    raise StaleTraceError(
        f"{_BANNER}\n"
        f"STALE TRACE: {trace_path} was sampled under a structurally "
        f"DIFFERENT model than the one this config builds.\n"
        f"  trace structural hash:  {stored}\n"
        f"  current model's hash:   {fingerprint}\n"
        f"What changed since the trace was sampled:\n{detail}\n"
        f"Reusing it would relabel those draws with the current parameter "
        f"names and regenerate every posterior, LaTeX table and plot from "
        f"them -- silently wrong numbers. The draws cannot be repaired at "
        f"load time.\n"
        f"{provenance}\n"
        f"REMEDY: re-sample with 'sampler: {{recompute_trace: true}}' in the "
        f"config (or revert the config/parameter-file edits listed above to "
        f"match the trace).\n{_BANNER}"
    )
