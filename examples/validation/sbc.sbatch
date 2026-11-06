#!/bin/bash
# Simulation-based calibration (review 7.16.4) on Slurm.
#
#   OUT=/path/to/output MODEL=rv_circ sbatch --array=0 sbc.sbatch prior
#   OUT=/path/to/output MODEL=rv_circ sbatch --array=0-29 sbc.sbatch run
#   OUT=/path/to/output MODEL=rv_circ sbatch --array=0 sbc.sbatch analyze
#
# `run` task k fits replicates [k*CHUNK, (k+1)*CHUNK).  MAXTIME (seconds) must
# stay below the --time limit: no replicate is started after it, and Slurm's
# SIGTERM (sent 120 s early, --signal) is forwarded to python, which stops
# without writing a partial row.  Rows already written are skipped on resubmit.
#SBATCH -p sapphire,shared
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 03:00:00
#SBATCH --signal=B:TERM@120
#SBATCH -o %x.%A_%a.log
set -u
: "${OUT:?set OUT}" "${MODEL:?set MODEL}"
CHUNK=${CHUNK:-10}
MAXTIME=${MAXTIME:-9000}
# Slurm runs a spooled copy of this file, so locate the tree from the
# submit directory (submit from examples/validation) unless WT is set.
WT=${WT:-$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)}
source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh
conda activate exozippy
export PYTHONPATH=$WT/src
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SCRIPT=$WT/examples/validation/sbc.py
case "$1" in
  prior)   args=(prior --model "$MODEL" --outdir "$OUT" --n "${NPRIOR:-300}" --cores 4) ;;
  run)     args=(run --model "$MODEL" --outdir "$OUT" --start $(( SLURM_ARRAY_TASK_ID * CHUNK )) --count "$CHUNK" --maxtime "$MAXTIME") ;;
  analyze) args=(analyze --model "$MODEL" --outdir "$OUT") ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
python "$SCRIPT" "${args[@]}" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM
wait $child
status=$?
# a trapped TERM interrupts the first wait; wait again for the real exit
wait $child 2>/dev/null
exit $status
