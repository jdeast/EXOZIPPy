#!/bin/bash
#SBATCH -p test
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH -o /n/holylabs/jjohnson_lab/Lab/rmdc26_report/logs/lpcheck_%j.log
# usage: sbatch lpcheck.sbatch <tree> <tag> <fit_dir> [...]
# e.g.   sbatch lpcheck.sbatch /n/holylabs/jjohnson_lab/Lab/wt/hogg467 hogg467 \
#            /n/holylabs/jjohnson_lab/Lab/RMDC26_run/events/094
# Runs lpcheck.py under <tree>/src for each fit (test partition only).
source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh
conda activate exozippy
export OMP_NUM_THREADS=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_$SLURM_JOB_ID"
TREE=$1; TAG=$2; shift 2
export PYTHONPATH=$TREE/src
R=/n/holylabs/jjohnson_lab/Lab/rmdc26_report
for FD in "$@"; do
  python /n/holylabs/jjohnson_lab/Lab/wt/rmdc26-report/examples/RMDC26/lpcheck.py "$FD" \
    --work $R/work/lpcheck_$TAG --out $R/lpcheck/$(basename $FD)_$TAG.json
done
