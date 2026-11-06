#!/bin/bash
#SBATCH --job-name=rmdc26_ev
#SBATCH --partition=test
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/jjohnson_lab/Lab/rmdc26_report/logs/evidence_%j.log
#
# Stage 1 of the RMDC26 report on the TEST partition (never the login
# node).  Arguments are passed straight to 01_evidence.py:
#
#   sbatch run_evidence.sbatch 94 173
#   sbatch run_evidence.sbatch --events-file events.txt
#
# The test partition allows 112 cores and 5 jobs per user, so at most three
# of these at a time.  Set RMDC26_REPORT to redirect the output root.
set -uo pipefail
source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh
conda activate exozippy
HERE=/n/holylabs/jjohnson_lab/Lab/wt/rmdc26-report/examples/RMDC26
# Import THIS tree, never the .pth's main checkout.
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/src
export OMP_NUM_THREADS=1
# Node-local compile dir: $HOME is NFS and its flock drops mid-compile.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID:-x}"
echo "tree: $PYTHONPATH  cores: $(python -c "import os; print(len(os.sched_getaffinity(0)))")  started $(date -Is)"
python "$HERE/01_evidence.py" "$@"
rc=$?
echo "finished $(date -Is) exit $rc"
exit $rc
