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

Note the deliberate asymmetry with the neighboring reload in
``whitening.load_whitening``, which detects a model mismatch of its own and
merely falls back: whitening can honestly be re-measured from scratch at
load time, so a mismatch there costs a probe.  A stale trace has no such
repair -- the draws are already drawn.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Root-group attrs on the InferenceData.  Prefixed so they can never
# collide with ArviZ's or a sampler's own metadata.
HASH_ATTR = "exozippy_structural_hash"
PAYLOAD_ATTR = "exozippy_structural_payload"

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


def stamp_structural_metadata(idata, system) -> None:
    """Record ``system``'s structural fingerprint in ``idata``'s root attrs.

    Called immediately before the trace is written to netCDF, so every trace
    this code writes can be checked when it is read back.
    """
    fingerprint, payload = system.structural_fingerprint()
    attrs = _attrs(idata)
    attrs[HASH_ATTR] = fingerprint
    blob = json.dumps(payload, sort_keys=True, default=str)
    if len(blob) <= _MAX_PAYLOAD_CHARS:
        attrs[PAYLOAD_ATTR] = blob
    else:
        attrs.pop(PAYLOAD_ATTR, None)


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


def check_trace_freshness(idata, system, trace_path) -> str:
    """Verify a reloaded trace was sampled from ``system``'s model.

    Returns ``"match"`` or ``"unverifiable"``; raises
    :class:`StaleTraceError` when the stored fingerprint disagrees with the
    current config + params.
    """
    fingerprint, payload = system.structural_fingerprint()
    attrs = _attrs(idata)
    stored = attrs.get(HASH_ATTR)

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
        f"REMEDY: re-sample with 'sampler: {{recompute_trace: true}}' in the "
        f"config (or revert the config/parameter-file edits listed above to "
        f"match the trace).\n{_BANNER}"
    )
