"""One-off: move an MMEXOFAST JSON's content into the config + params files.

WHY THIS EXISTS.  The `mmexofast:` config key was removed (JDE 2026-10-01:
"We're stripping mmexofast from the repo ... Do the dc2018 rely on its Json
input though?  If so, that should be refactored to use the param file
input").  A config that still names the key now RAISES at the user
boundary.  Every DC2018 config that named a JSON is converted by this
script, which reproduces EXACTLY what the removed
`MulensInstrument._resolve_mmexofast` did with the file, but as user input:

  fits[k].parameters  -> per-seed `initval: [...]` lists in the params file
                         (one entry per fit, in file order), on the same
                         paths `push_seed_hints` seeded: source t_0 (minus
                         the JSON's `jd_offset`), u_0, rho (finite source
                         only), mulensevent t_E, and on a binary lens the
                         companion's log_s = log10(s), alpha (identity
                         convention) and q.  A path the params file already
                         starts (initval, or mu) is left alone -- every
                         user entry outranked the seeds, so it was the
                         user's value the fit started from all along.
  excluded_points     -> `mask:` (0-based on-disk row indices, offset-free)
                         on the mulensinstrument entry whose file has that
                         basename -- skipped, as before, for a file with a
                         robust `likelihood:` or its own `mask:`.
  errfacs             -> `mulensinstrument.<name>.err_scale: {initval: f}`
                         unless the params file already starts it.
  > 1 fit             -> `sampler: {seed_polish: true}` unless the config
                         sets seed_polish.  `seed_polish: auto` polishes a
                         multi-seed set only when it came from a component
                         seeder; the same starts given as user lists are
                         read as posterior-draw restarts and would NOT be
                         polished, so this keeps the run doing what it did.
  fits[0].sigmas      -> dropped.  They were init_scale hints for the
                         whitening probe only, and a user `init_scale` is
                         warn-stripped (whitening scales are measured from
                         the data, src/exozippy/whitening.md).  Reported.
  coords, mag_methods -> dropped; nothing ever read them.

The `mmexofast:` (and `mmexofast_options:`) key is then deleted from the
config.  A config whose JSON does not exist is reported and left
UNTOUCHED, so it keeps raising at the boundary rather than silently
starting from the peak finder: the JSON exists only where the fit ran (the
`*_mmexofast.json` caches are gitignored), so run this there.

Several configs may share one params file (configs/, events/152); all edits
are computed against the ORIGINAL files first and a shared params file is
written once, after checking that every config sharing it asks for the same
edits -- otherwise the script stops and names them.

    python convert_mmexofast_json.py CONFIG.yaml [CONFIG.yaml ...] \\
        [--repo-prefix /home/jeastman/python/EXOZIPPy] [--dry-run]

`--repo-prefix P` rewrites absolute paths under P (the configs' params,
JSON and seed paths) to this checkout, so a worktree edits its own files.
Paths are resolved relative to the config's directory, which is the
working directory the fits run from.
"""

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

import yaml as pyyaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

REPO = Path(__file__).resolve().parents[2]

# The note a cluster-only config carries above its `mmexofast:` key until it
# is converted (5 comment lines); apply_config removes it with the key.
_MARKER = "# REMOVED KEY (2026-10-01):"
_MARKER_LINES = 5

yaml = YAML()

yaml.preserve_quotes = True
yaml.width = 4096


def _flow(values):
    seq = CommentedSeq(values)
    seq.fa.set_flow_style()
    return seq


class Unconvertible(RuntimeError):
    pass


def _rebase(path, prefix):
    if prefix and str(path).startswith(prefix.rstrip("/") + "/"):
        return str(REPO) + str(path)[len(prefix.rstrip("/")) :]
    return str(path)


def _resolve(path, cfg_dir, prefix):
    p = Path(_rebase(path, prefix))
    return p if p.is_absolute() else (cfg_dir / p).resolve()


def _names(block):
    """Element names the way the components derive them (name:, else the
    body ref's trailing segment -- bodies.derive_body_names)."""
    out = []
    for e in block or []:
        n = e.get("name")
        if n is None and e.get("body") is not None:
            n = str(e["body"]).split(".")[-1]
        out.append(None if n is None else str(n))
    return out


def _spellings(comp, i, name, param):
    keys = [f"{comp}.{param}", f"{comp}.{i}.{param}"]
    if name is not None:
        keys.append(f"{comp}.{name}.{param}")
    return keys


def _starts(params, keys):
    """True when any spelling already carries a start (initval or mu)."""
    for k in keys:
        if k not in params:
            continue
        v = params[k]
        if not isinstance(v, dict):
            return True  # a bare number / per-seed list IS an initval
        if v.get("initval") is not None or v.get("mu") is not None:
            return True
    return False


def _target_key(params, keys):
    """The most specific existing spelling, else the name (or index) form."""
    for k in reversed(keys):
        if k in params:
            return k
    return keys[-1]


def _load_rt(path):
    """Round-trip load (comments kept), naming the file on a parse error."""
    try:
        return yaml.load(Path(path).read_text())
    except Exception as e:  # noqa: BLE001 -- re-raised with the file named
        raise Unconvertible(f"{path}: cannot round-trip this YAML: {e}") from e


def plan(cfg_path, prefix):
    """The edits one config needs, or None when it names no JSON."""
    cfg_path = Path(cfg_path).resolve()
    cfg_dir = cfg_path.parent
    # Plain load first: a config whose JSON is absent is reported and never
    # written, so it need not survive a round trip (one cluster-only config
    # carries a duplicate top-level key that PyYAML tolerates).
    plain = pyyaml.safe_load(cfg_path.read_text()) or {}
    if "mulensevent" not in plain and any(
        "mmexofast" in (e or {}) for e in plain.get("lens") or []
    ):
        # The pre-v0.1.0 schema (event options on the lens block): such a
        # config has not built since the lens/source/mulensevent split, with
        # or without this key, so there is nothing to preserve.
        return {"cfg_path": cfg_path, "presplit": True}
    events = plain.get("mulensevent") or []
    if events and isinstance(events[0].get("mmexofast"), str):
        json_path = _resolve(events[0]["mmexofast"], cfg_dir, prefix)
        if not json_path.exists():
            return {"cfg_path": cfg_path, "missing": json_path}
    cfg = _load_rt(cfg_path)
    events = cfg.get("mulensevent") or []
    if not events or "mmexofast" not in events[0]:
        if events and "mmexofast_options" in events[0]:
            return {"cfg_path": cfg_path, "cfg": cfg, "json": None}
        return None
    ev = events[0]
    spec = ev["mmexofast"]
    if spec is False or spec is None:
        return {"cfg_path": cfg_path, "cfg": cfg, "json": None}
    if spec is True:
        raise Unconvertible(
            f"{cfg_path}: `mmexofast: true` ran MMEXOFAST at fit time; there "
            f"is no file to convert.  Drop the key (the peak finder seeds "
            f"t_0/u_0/t_E) or point it at the run's cached JSON first."
        )
    json_path = _resolve(spec, cfg_dir, prefix)

    insts = cfg.get("mulensinstrument") or []
    for e in insts:
        if any(k in e for k in ("time_offset", "time_scale", "time_frame")):
            raise Unconvertible(
                f"{cfg_path}: instrument {e.get('name')!r} has a time spec; "
                f"MMEXOFAST seeds are in the raw file time system and the "
                f"removed loader refused this combination too."
            )
    data = json.loads(json_path.read_text())
    if not isinstance(data, dict) or "fits" not in data:
        raise Unconvertible(f"{json_path}: not an MMEXOFAST JSON (no 'fits').")
    fits = data["fits"] or []
    # An EMPTY `fits` is a real case, not a malformed file: MMEXOFAST found
    # no solution for DC2018 062 (dc18_seed.py's peak finder did), and the
    # removed loader then seeded nothing but still applied the JSON's
    # excluded_points and errfacs.  Convert those, seed nothing, and say so.

    pf = cfg.get("parameter_file")
    if pf is None:
        raise Unconvertible(
            f"{cfg_path}: no parameter_file to write seeds to."
        )
    params_path = _resolve(pf, cfg_dir, prefix)
    params = _load_rt(params_path) if params_path.exists() else None
    params = params if params is not None else CommentedMap()

    want_rho = bool(ev.get("finite_source", False))
    lens_names = _names(cfg.get("lens"))
    src_names = _names(cfg.get("source"))
    is_binary = len(lens_names) >= 2
    jd_offset = float(data.get("jd_offset", 0.0) or 0.0)

    # (observable in the JSON, component, element, param, transform)
    rows = [
        ("t_0", "source", 0, "t_0", lambda v: float(v) - jd_offset),
        ("u_0", "source", 0, "u_0", float),
        ("t_E", "mulensevent", 0, "t_E", float),
    ]
    if want_rho:
        rows.append(("rho", "source", 0, "rho", float))
    if is_binary:
        rows += [
            ("s", "lens", 1, "log_s", lambda v: float(math.log10(float(v)))),
            ("alpha", "lens", 1, "alpha", float),
            ("q", "lens", 1, "q", float),
        ]
    names = {"source": src_names, "lens": lens_names, "mulensevent": [None]}

    param_edits, skipped, notes = {}, [], []
    if not fits:
        notes.append(
            "JSON 'fits' is empty (MMEXOFAST produced no solution): no "
            "seeds written; the peak finder seeds t_0/u_0/t_E as before"
        )
    for key, comp, i, param, fn in rows if fits else []:
        name = names[comp][i] if i < len(names[comp]) else None
        spell = _spellings(comp, i, name, param)
        if comp == "lens" and param == "log_s":
            spell_all = spell + _spellings(comp, i, name, "s")
        else:
            spell_all = spell
        vals = []
        for k, fit in enumerate(fits):
            p = fit.get("parameters", {})
            if key not in p or (key == "s" and not float(p[key]) > 0):
                raise Unconvertible(
                    f"{json_path}: fit {k} has no usable {key!r}; the removed "
                    f"loader seeded such a fit PARTIALLY, which a per-seed "
                    f"list cannot express.  Convert by hand."
                )
            vals.append(fn(p[key]))
        if _starts(params, spell_all):
            skipped.append(spell[-1])
            continue
        param_edits[_target_key(params, spell)] = (
            vals if len(fits) > 1 else vals[0]
        )

    inst_names = _names(insts)
    by_base = {
        os.path.basename(str(e.get("file"))): j for j, e in enumerate(insts)
    }
    mask_edits = {}
    for label, info in (data.get("excluded_points") or {}).items():
        j = by_base.get(os.path.basename(str(label)))
        if j is None:
            raise Unconvertible(
                f"{json_path}: excluded_points '{label}' matches no "
                f"mulensinstrument file of {cfg_path}."
            )
        idx = [int(x) for x in (info.get("indices") or [])]
        if not idx:
            continue
        if insts[j].get("likelihood") or insts[j].get("mask") is not None:
            notes.append(
                f"{len(idx)} excluded point(s) of '{label}' not masked "
                f"(robust likelihood or user mask, as before)"
            )
            continue
        mask_edits[j] = idx

    for label, fac in (data.get("errfacs") or {}).items():
        j = by_base.get(os.path.basename(str(label)))
        if j is None:
            raise Unconvertible(
                f"{json_path}: errfacs '{label}' matches no mulensinstrument "
                f"file of {cfg_path}."
            )
        fac = float(fac)
        if not (math.isfinite(fac) and fac > 0):
            continue
        spell = _spellings("mulensinstrument", j, inst_names[j], "err_scale")
        if _starts(params, spell):
            skipped.append(spell[-1])
            continue
        param_edits[_target_key(params, spell)] = fac

    if fits and (fits[0].get("sigmas") or {}):
        notes.append(
            "fit 0 sigmas dropped (init_scale hints; the whitening probe "
            "measures scales)"
        )
    sampler = cfg.get("sampler") or {}
    polish = len(fits) > 1 and "seed_polish" not in sampler

    return {
        "cfg_path": cfg_path,
        "cfg": cfg,
        "json": json_path,
        "params_path": params_path,
        "param_edits": param_edits,
        "mask_edits": mask_edits,
        "polish": polish,
        "skipped": skipped,
        "notes": notes,
        "n_fits": len(fits),
    }


def apply_params(params_path, edits):
    params = _load_rt(params_path) if params_path.exists() else None
    params = params if params is not None else CommentedMap()
    for key, val in edits.items():
        entry = params.get(key)
        if entry is None:
            entry = CommentedMap()
            params[key] = entry
        entry["initval"] = _flow(val) if isinstance(val, list) else val
    with open(params_path, "w") as fh:
        yaml.dump(params, fh)


def apply_config(p):
    cfg = p["cfg"]
    ev = cfg["mulensevent"][0]
    for k in ("mmexofast", "mmexofast_options"):
        if k in ev:
            del ev[k]
    for j, idx in (p.get("mask_edits") or {}).items():
        cfg["mulensinstrument"][j]["mask"] = _flow(idx)
    if p.get("polish"):
        if cfg.get("sampler") is None:
            cfg["sampler"] = CommentedMap()
        cfg["sampler"]["seed_polish"] = True
    buf = io.StringIO()
    yaml.dump(cfg, buf)
    # Drop the "REMOVED KEY" note the cluster-only configs carry above the
    # key: it describes this conversion, which has now happened.
    lines = buf.getvalue().split("\n")
    out, skip = [], 0
    for line in lines:
        if line.strip().startswith(_MARKER):
            skip = _MARKER_LINES
        if skip:
            skip -= 1
            continue
        out.append(line)
    with open(p["cfg_path"], "w") as fh:
        fh.write("\n".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--repo-prefix", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    plans, missing, presplit = [], [], []
    for c in a.configs:
        p = plan(c, a.repo_prefix)
        if p is None:
            continue
        if "missing" in p:
            missing.append(p)
            continue
        if "presplit" in p:
            presplit.append(p)
            continue
        plans.append(p)

    by_params = {}
    for p in plans:
        if p.get("json") is not None:
            by_params.setdefault(p["params_path"], []).append(p)
    for path, group in by_params.items():
        edits = [dict(g["param_edits"]) for g in group]
        if any(e != edits[0] for e in edits[1:]):
            raise SystemExit(
                f"{path} is shared by "
                f"{[str(g['cfg_path']) for g in group]}, whose JSONs ask for "
                f"different seeds; convert them by hand."
            )

    for p in plans:
        what = (
            "drop key"
            if p.get("json") is None
            else (
                f"{p['n_fits']} fit(s) from {p['json']}: "
                f"{len(p['param_edits'])} param edit(s) "
                f"{sorted(p['param_edits'])}, "
                f"{len(p['mask_edits'])} mask(s), "
                f"seed_polish {'true' if p['polish'] else 'unchanged'}"
                + (
                    f"; user starts kept: {p['skipped']}"
                    if p["skipped"]
                    else ""
                )
                + (f"; {'; '.join(p['notes'])}" if p["notes"] else "")
            )
        )
        print(f"CONVERT {p['cfg_path']}: {what}")
    for m in missing:
        print(
            f"MISSING {m['cfg_path']}: JSON {m['missing']} absent; untouched"
        )
    for m in presplit:
        print(
            f"PRESPLIT {m['cfg_path']}: pre-v0.1.0 lens-block schema, does "
            f"not build; untouched"
        )
    if a.dry_run:
        return 0
    for path, group in by_params.items():
        apply_params(path, group[0]["param_edits"])
    for p in plans:
        apply_config(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
