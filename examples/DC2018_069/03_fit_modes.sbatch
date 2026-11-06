#!/bin/bash
# Stage 3: one PTDE run per candidate mode, as a Slurm array over the lines
# of $DC069_RUNS/modes/configs.txt (02_seeds.py writes it).
#
#   cd examples/DC2018_069
#   n=$(wc -l < /n/holylabs/jjohnson_lab/Lab/dc069_runs/modes/configs.txt)
#   sbatch --array=1-$n 03_fit_modes.sbatch
#
# Settings are PROVISIONAL (README, "Run settings"): n_temps auto, tune 300,
# draws 3000, sampler.maxtime 23400 s = 6.5 h, under this 8 h wall.  The
# 1.5 h between maxtime and the wall is for the wrap-up (tables, plots,
# mode pass), and --signal=B:TERM@1800 is a second guard behind maxtime:
# without it Slurm signals only AT the wall and SIGKILLs moments later, so
# the sampler never reaches its save path (pilot 194 lost 48 h that way).
#
# One node, 64 threads: PTDE is shared-memory with no MPI path.  sapphire,
# not shared: shared nodes cannot grant 64 cores with 256 G (run1.sbatch in
# the DC2018 launch tooling records the probe).
#
#SBATCH --job-name=dc069
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@1800
#SBATCH --output=/n/holylabs/jjohnson_lab/Lab/dc069_runs/logs/%x.%A_%a.log

set -euo pipefail

# The repo this example lives in.  sbatch copies the script to a spool
# directory, so it cannot locate itself; submit from examples/DC2018_069.
EX="${SLURM_SUBMIT_DIR:?submit from examples/DC2018_069}"
ROOT="$(cd "$EX/../.." && pwd)"
RUNS="${DC069_RUNS:-/n/holylabs/jjohnson_lab/Lab/dc069_runs}"

source /n/holylabs/jjohnson_lab/Lab/miniconda3/etc/profile.d/conda.sh
conda activate exozippy
export PYTHONPATH="$ROOT/src"
# PyTensor's compile cache on node-local scratch: NFS loses flock()s.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor"
# PTDE runs one chain per core; BLAS threads on top oversubscribe the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

src_cfg=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$RUNS/modes/configs.txt")
[ -n "$src_cfg" ] || { echo "no config for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID}  cfg $src_cfg  commit $(git -C "$ROOT" rev-parse --short HEAD)  $(date)"
python -c "import exozippy, os; print('exozippy from', os.path.dirname(exozippy.__file__))"

# sampler.cores from what Slurm granted, on a per-job COPY of the config
# (the written config stays what 02_seeds.py produced).
cfg="${TMPDIR:-/tmp}/dc069_${SLURM_ARRAY_TASK_ID:-1}_$$.yaml"
python "$ROOT/examples/DC2018/dc18_set_cores.py" "$src_cfg" "$cfg" \
    --cores "${SLURM_CPUS_PER_TASK:?}"
cd "$(dirname "$src_cfg")"

# Background + forwarded SIGTERM: --signal=B:TERM reaches this shell, not a
# foreground child; `wait` is repeated while the child saves.
exozippy "$cfg" &
child=$!
trap 'echo "SIGTERM -> forwarding to exozippy ($child)"; kill -TERM "$child" 2>/dev/null' TERM
rc=0
wait "$child" || rc=$?
while kill -0 "$child" 2>/dev/null; do
    rc=0
    wait "$child" || rc=$?
done
echo "exit $rc   $(date)"
exit "$rc"
