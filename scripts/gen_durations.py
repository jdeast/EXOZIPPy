"""Build tests/durations.json from a `pytest --durations=0` transcript.

Per FILE, not per test, because --dist loadfile schedules whole files: a
per-test breakdown would be more data with no more usable signal.
"""

import collections
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?)::(\S+)\s*$")

src = Path(sys.argv[1])
out = Path(sys.argv[2])

# Test ids naming an example directory that is not in the repository. A
# developer's untracked examples/<name>/ is collected by
# test_examples_prepare locally and never on CI, so counting its cost would
# skew the shard balance for everyone else. Matched on the parametrize id.
_LOCAL_ONLY = ("ob09020",)

per_file = collections.Counter()
skipped = collections.Counter()
for line in src.read_text().splitlines():
    m = LINE.match(line.strip())
    if not m:
        continue
    if any(tag in m[4] for tag in _LOCAL_ONLY):
        skipped[m[3]] += float(m[1])
        continue
    per_file[m[3]] += float(m[1])

# Store basenames: the shard split matches on them, and an absolute path from
# whoever generated the file would be useless everywhere else.
durations = {Path(f).name: round(v, 2) for f, v in sorted(per_file.items())}

payload = {
    "_excluded_local_only": {
        Path(f).name: round(v, 2) for f, v in sorted(skipped.items())
    },
    "_comment": (
        "Per-file worker-seconds, for balancing the CI shard split. "
        "Regenerate with scripts/gen_durations.py -- see "
        "docs/testing-cache.md. Numbers are RELATIVE weights; absolute "
        "values depend on the machine, and only their ratios matter."
    ),
    "_generated_from": "pytest -q -n6 --dist loadfile --durations=0",
    "_measured_on": "2026-08-25",
    "_total_worker_seconds": round(sum(durations.values()), 1),
    "_file_count": len(durations),
    "durations": durations,
}

out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
print(
    f"wrote {out}: {len(durations)} files, "
    f"{payload['_total_worker_seconds']:.0f} worker-seconds"
)
top = sorted(durations.items(), key=lambda kv: -kv[1])[:5]
for name, v in top:
    print(f"   {v:7.1f}s  {name}")
