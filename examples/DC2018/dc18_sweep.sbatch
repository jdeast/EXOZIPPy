#!/bin/bash
# SLURM counterpart to dc18_sweep.job, which is SGE.  Every .job file in
# this repo is Grid Engine; this is the same run for an sbatch cluster, and
# is kept beside it deliberately so users on either scheduler have a
# working example rather than a translation exercise.  The SGE-to-SLURM
# mapping, the measured memory figures behind --mem, and a first-hour
# checklist for a new machine are in notes/canon.txt.
#
#   sbatch --array=1-6 dc18_sweep.sbatch
#
#SBATCH --job-name=dc18swp
#SBATCH --partition=shared
#SBATCH --array=1-6
#
# ONE NODE, MANY THREADS.  PTDE/ptde_async is a shared-memory sampler with
# no MPI path (review 8.6.2 shelved multi-node PTDE on purpose), so the
# whole fit must land on one host.  --ntasks=64 would be wrong in a way
# that still "works": SLURM may spread 64 independent tasks across nodes,
# and each one would run a complete redundant fit.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#
# --mem is enforced on RSS via cgroups, so it is budgeted off maxrss, not
# maxvmem.  Measured on a live six-event DC2018 sweep at 50,000 draws x 64
# cores: maxrss 51 GB for five of the six events and 201 GB for the heavily
# blended one (062, source flux 0.5% of the blend).  The headroom above
# that is for WRAP-UP, which is where this model has OOM-killed three times
# (exit 137 at 613-705 GB vmem, long after sampling finished) -- and a
# wrap-up death costs every DERIVED parameter, because the trace that gets
# saved first carries only the sampled ones (review 2.6.14).
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=sweep/dc18_sweep.%A_%a.log

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate exozippy

# PyTensor's C-module cache locks with fcntl.flock(), and ~/.pytensor on an
# NFS home can lose advisory locks mid-compile (ENOLCK/ESTALE).  Point it at
# the job's local scratch.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor"

# NOT optional.  PTDE already runs one chain per core; letting BLAS thread
# on top oversubscribes the node by the BLAS thread count and slows every
# chain at once.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$HOME/python/EXOZIPPy/examples/DC2018"

ev=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sweep/events.txt)
if [ -z "$ev" ]; then
    echo "No event for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "event $ev   commit $(git -C "$HOME/python/EXOZIPPy" rev-parse --short HEAD)   $(date)"

# Fill sampler.cores from what SLURM actually granted.  EXOZIPPy respects
# the user's `cores:` and deliberately does not probe the scheduler (review
# 2.4.11 -- most machines have none), so the job has to do it, and a config
# still claiming 64 after being granted 32 oversubscribes the node 2x.
# dc18_set_cores.py writes a per-job COPY: the committed configs are read by
# neighbouring array tasks, and what ran stays reproducible from the
# committed file plus $SLURM_CPUS_PER_TASK.
cfg="${TMPDIR:-/tmp}/DC2018_${ev}_$$.yaml"
python dc18_set_cores.py "sweep/$ev/DC2018_$ev.yaml" "$cfg" \
    --cores "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset}"

exozippy "$cfg"
echo "exit $?   $(date)"
