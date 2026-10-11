#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=bfdm-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chenyangkang24@outlook.com
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --exclude=cn103,cn003,cn067,cn037

set -eo pipefail

case_name="${1:-result_pressure}"
workers="${2:-1,4,8,16,30}"
prediction_rows="${3:-3000000}"
output_mode="${4:-separate}"
ensemble_count="${5:-50}"
repetitions="${6:-2}"
WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$WORK_DIR/log_files"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prediction_memory_${case_name}_${SLURM_JOB_ID:-local}.log"
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
echo "Case: $case_name; workers: $workers; prediction rows: $prediction_rows"
echo "Working directory: $WORK_DIR"
cd "$WORK_DIR"

[[ -f scripts/06_prediction_stress/run_stress.py ]] || {
    echo "Submit from the stemflow repository root."
    exit 1
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=4
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$WORK_DIR/.cache/matplotlib"

options=()
if [ "$output_mode" = "separate" ]; then
    options+=(--separate)
elif [ "$output_mode" != "mean" ]; then
    echo "Fourth argument must be separate or mean."
    exit 1
fi

[[ -f stemflow/lazyloading/shared_prediction.py ]] || {
    echo "Missing the memory-mapped prediction implementation."
    exit 1
}

echo "Output mode: $output_mode; ensembles: $ensemble_count; repetitions: $repetitions"
echo "Starting Python..."
/taiga/illinois/aces/nres/vandoren/yangkang/micromamba/envs/calc_asso/bin/python -u \
    scripts/06_prediction_stress/run_stress.py \
    --cases "$case_name" --strategies baseline,memmap,memmap_threads \
    --jobs "$workers" --rows "$prediction_rows" --ensembles "$ensemble_count" --repeat "$repetitions" "${options[@]}" \
    --memory-gib 64 --timeout 1800 \
    --output "data/06_prediction_stress/hpc_memory_${case_name}_${SLURM_JOB_ID:-local}"
