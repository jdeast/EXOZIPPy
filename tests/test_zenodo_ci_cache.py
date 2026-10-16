"""Tests for scripts/zenodo_ci_cache.py, CI's Zenodo download-cache helper."""

import importlib.util
import sys
from pathlib import Path

from exozippy.utilities import zenodo_assets

# Loaded by path: scripts/ is not a package, and putting it on sys.path would
# make every script importable as a top-level module.
_SPEC = importlib.util.spec_from_file_location(
    "zenodo_ci_cache",
    Path(__file__).resolve().parents[1] / "scripts" / "zenodo_ci_cache.py",
)
ci_cache = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ci_cache
_SPEC.loader.exec_module(ci_cache)


def test_prune_keeps_every_current_pin_and_drops_everything_else(tmp_path):
    """
    Given a cache directory holding a current entry, a superseded one, a
    lock file and a stray partial download,
    When it is pruned,
    Then only the entry a current pin names is left -- so the entry CI saves
    under a new key does not carry superseded versions forward.
    """
    # ARRANGE
    rec = zenodo_assets.record("claret_ld_tables")
    filename, pin = sorted(rec.files.items())[0]
    current = tmp_path / f"{pin.md5}-{filename}"
    stale = tmp_path / f"{'0' * 32}-{filename}"
    lock = tmp_path / f"{current.name}.lock"
    part = tmp_path / f"{filename}.abc123.part"
    for path in (current, stale, lock, part):
        path.write_bytes(b"x")

    # ACT
    removed = ci_cache.prune(tmp_path)

    # ASSERT
    assert sorted(p.name for p in tmp_path.iterdir()) == [current.name]
    assert sorted(removed) == sorted([stale, lock, part])


def test_every_pinned_file_has_a_current_entry_name():
    """
    Given the registry,
    When the names prune keeps are listed,
    Then there is one per pinned file, spelled as zenodo.py names an entry.
    """
    from exozippy.utilities import zenodo

    names = ci_cache.current_entry_names()
    for name, rec in zenodo_assets.RECORDS.items():
        for filename in rec.files:
            meta = zenodo_assets.file_pin(name, filename)
            entry = zenodo._entry_path(Path("/c"), filename, meta)
            assert entry.name in names
    assert len(names) == sum(
        len(r.files) for r in zenodo_assets.RECORDS.values()
    )
