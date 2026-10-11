#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=bfdm-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chenyangkang24@outlook.com
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --exclude=cn103,cn003,cn067,cn037

set -eo pipefail

species="${1:?Provide an encoded species name}"
tune_mode="${2:-true}"
WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$WORK_DIR/log_files"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${species}.log"
exec > "$LOG_FILE" 2>&1

finish() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        echo "Job completed successfully."
    else
        echo "Error: job failed with exit code $status."
    fi
    echo "End time: $(date)"
    exit "$status"
}
trap finish EXIT

echo "Running on node: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-30}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Start time: $(date)"
echo "Species: $species"
echo "Tuning: $tune_mode"
echo "Working directory: $WORK_DIR"
cd "$WORK_DIR"

[[ -f run_model_test.py ]] || {
    echo "Missing run_model_test.py"
    exit 1
}
[[ "$tune_mode" == true || "$tune_mode" == false ]] || {
    echo "Tuning must be true or false."
    exit 1
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=4

echo "Starting Python..."
/taiga/illinois/aces/nres/vandoren/yangkang/micromamba/envs/calc_asso/bin/python -u run_model_test.py "$species" "$tune_mode" \
    --ridge 100 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-30}" \
    --ensemble-fold 50 \
    --min-ensemble-required 45 \
    --prediction-batch-size 100000
