"""No large data file may be tracked under src/exozippy/models.

Everything under src/ that git tracks ships in the wheel (poetry builds from
what git does not ignore). The NextGen BC tables reached ~96 MB tracked there
and took the wheel to ~134 MB, over PyPI's 100 MB per-file default (PR #349);
they moved to Zenodo (models/NextGen/bc_tables.py), as the NextGen spectra
and the MIST EEP grid had before them. The .gitignore entry stops the tables
by NAME; this stops the next large model file by SIZE, whatever it is called.

The limit is on the size git stores (the blob in the index), so an untracked
or ignored file in the working tree -- a fetched table, say -- is not counted,
and nothing has to be on disk for the check to run.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_GUARDED = "src/exozippy/models"

# 1 MB. The largest tracked file under models/ after the move is ~0.1 MB (a
# MIST grid yaml); a model asset bigger than this belongs on Zenodo behind a
# pinned manifest, not in every install.
_LIMIT_BYTES = 1_000_000


def _tracked_blob_sizes(subdir):
    """{path: size in bytes} of every file git tracks under `subdir`."""
    listing = subprocess.run(
        ["git", "ls-files", "-s", "-z", "--", subdir],
        cwd=_REPO,
        check=True,
        capture_output=True,
    ).stdout
    entries = [e for e in listing.split(b"\0") if e]
    # "<mode> <sha> <stage>\t<path>"; batch-check answers in input order.
    pairs = []
    for entry in entries:
        meta, path = entry.split(b"\t", 1)
        pairs.append((meta.split()[1].decode(), path.decode()))
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectsize)"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        input="\n".join(sha for sha, _ in pairs).encode(),
    ).stdout.decode()
    sizes = dict(zip((p for _, p in pairs), map(int, batch.split())))
    assert len(sizes) == len(pairs)
    return sizes


@pytest.mark.skipif(
    shutil.which("git") is None or not (_REPO / ".git").exists(),
    reason="not a git checkout (an sdist or an installed copy has no index)",
)
def test_no_tracked_file_under_models_exceeds_the_size_limit():
    """
    Given the files git tracks under src/exozippy/models,
    When their stored sizes are compared to the limit,
    Then none exceeds it.
    """
    sizes = _tracked_blob_sizes(_GUARDED)
    assert sizes, f"git tracks nothing under {_GUARDED}?"

    too_big = {p: s for p, s in sizes.items() if s > _LIMIT_BYTES}

    assert not too_big, (
        f"tracked file(s) over {_LIMIT_BYTES} bytes under {_GUARDED}: "
        f"{too_big}. Large model data ships to every user in the wheel; "
        f"publish it on Zenodo behind a pinned manifest instead (see "
        f"models/NextGen/bc_tables.py and utilities/zenodo.py), and "
        f"git-ignore it."
    )
