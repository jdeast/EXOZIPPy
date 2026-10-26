#!/bin/bash
#SBATCH --job-name=rmdc26_sub
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/n/holylabs/jjohnson_lab/Lab/rmdc26_report/submission_v2/logs/submission_%j.log
#
# The RMDC26 submission from the campaign's MANIFEST, in ONE command, as a batch job
# (never the login node; sapphire,shared because the test partition was
# mostly drained on 2026-10-26; 64G = 3x the measured MaxRSS of 18.7 GB):
#
#   sbatch examples/RMDC26/run_submission.sbatch [extra 01_evidence args]
#
# after the campaign side has run sed/verdict.py -> sed/make_manifest.py ->
# sed/make_notes.py.  Stage 1 (01_evidence.py --manifest) reuses every
# evidence record whose trace is unchanged and recomputes the rest, with
# WORKERS fits in parallel on disjoint core slices; stage 3
# (03_submission.py) rebuilds submission/ from scratch and validates it
# with microlens-submit's own code (04_validate.py).
#
# RMDC26_REPORT  output root  (default .../rmdc26_report/submission_v2)
# WORKERS        parallel fits (default 6)
# EVIDENCE_TREE  the exozippy tree that evaluates logp for the bridge.
#                Default: the tree that SAMPLED the SED-tied fits
#                (wt/master1021 = b1b41d8e), so the bridge integrates the
#                sampled posterior exactly.  Master differs from it by #478
#                (observer at the light-reception instant), which moves lp
#                by ~0.01 nat; THIS tree (master) is checked against every
#                stored lp separately (lpcheck_master/), as the confirmation.
set -uo pipefail
source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh
conda activate exozippy
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# sbatch copies the script: fall back to the submit directory's copy.
[ -f "$HERE/01_evidence.py" ] || HERE=${SLURM_SUBMIT_DIR:-.}/examples/RMDC26
[ -f "$HERE/01_evidence.py" ] || HERE=${SLURM_SUBMIT_DIR:-.}
export RMDC26_REPORT=${RMDC26_REPORT:-/n/holylabs/jjohnson_lab/Lab/rmdc26_report/submission_v2}
THIS_TREE=$(cd "$HERE/../.." && pwd)
EVIDENCE_TREE=${EVIDENCE_TREE:-/n/holylabs/jjohnson_lab/Lab/wt/master1021}
# Never the .pth's main checkout: always an explicit tree.
export PYTHONPATH=$EVIDENCE_TREE/src
export OMP_NUM_THREADS=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID:-x}"
mkdir -p "$RMDC26_REPORT/logs"
# microlens-submit's models need pydantic + psutil (absent from the env).
DEPS=$RMDC26_REPORT/deps
python -c "import sys; sys.path.insert(0, '$DEPS'); import pydantic, psutil" 2>/dev/null \
  || pip install --quiet --target "$DEPS" pydantic psutil
echo "evidence tree: $EVIDENCE_TREE ($(git -C $EVIDENCE_TREE rev-parse --short HEAD))  this tree: $THIS_TREE ($(git -C $THIS_TREE rev-parse --short HEAD))  out: $RMDC26_REPORT  started $(date -Is)"
t0=$(date +%s)
python "$HERE/01_evidence.py" --manifest --workers "${WORKERS:-6}" "$@"
rc1=$?
t1=$(date +%s)
# lp reproduction by THIS tree (master) on every submitted fit, 50 draws
# each, 8 at a time.  Reused while the trace is older than the result.
LPM=$RMDC26_REPORT/lpcheck_master
mkdir -p "$LPM"
python -c "
import json, sys
d = json.load(open('$RMDC26_REPORT/evidence/manifest_resolved.json'))
for r in d['submitted']:
    if r.get('fit'):
        print(r['fit'], r['config'], r['trace'])
" | while read FIT CFG TR; do
  [ "$LPM/$FIT.json" -nt "$TR" ] || echo "$FIT $CFG"
done | xargs -r -P 8 -n 2 sh -c 'PYTHONPATH='"$THIS_TREE"'/src python '"$HERE"'/lpcheck.py "$1" --work '"$RMDC26_REPORT"'/work/lpcheck_master/$0 --out '"$LPM"'/$0.json > '"$LPM"'/$0.log 2>&1'
t2=$(date +%s)
echo "master lpcheck: $((t2 - t1)) s"
python "$HERE/03_submission.py"
rc3=$?
echo "stage 1 exit $rc1 ($((t1 - t0)) s), stage 3 exit $rc3 ($(( $(date +%s) - t2 )) s), total $(( $(date +%s) - t0 )) s, finished $(date -Is)"
exit $(( rc1 || rc3 ))
