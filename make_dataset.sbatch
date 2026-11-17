#!/bin/bash
# Generate a synthetic facesim dataset on SMU M3.
#
# Structure: one job, one node, many cores. facesim.generate already runs scans
# in parallel (--workers), so the job just asks Slurm for N cores and passes N
# to --workers. No job array needed until you outgrow a single 128-core node.
#
# Submit (real run):   sbatch make_dataset.sbatch
# Submit (quick test): IDENTITIES=4 SCANS=3 ATTACKS=1 NAME=test \
#                      sbatch -p dev -c 8 --mem=16G -t 0:20:00 make_dataset.sbatch
# (Options given to sbatch on the command line override the #SBATCH lines below.)

#SBATCH -A slurm_account_name        # <-- your Slurm account from ColdFront
#SBATCH -J facesim
#SBATCH -o facesim_%j.out            # log file, %j = job ID
#SBATCH -p standard-s                # CPU nodes, 24 h limit (dev = 2 h, for tests)
#SBATCH -t 0-02:00:00                # wall time; runs are resumable if it runs out
#SBATCH -N 1
#SBATCH -c 64                        # cores = number of parallel scan workers
#SBATCH --mem=96G                    # check MaxRSS after the test run and adjust
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@smu.edu     # remove one '#' to get emails

# ---- dataset settings: edit here or override at submit time -----------------
IDENTITIES=${IDENTITIES:-500}        # people
SCANS=${SCANS:-20}                   # live scans per person
ATTACKS=${ATTACKS:-2}                # printed-photo spoofs per person
SEED=${SEED:-0}                      # same seed + settings = same dataset
NAME=${NAME:-synth_v1}
CODE_DIR=${CODE_DIR:-$HOME/generate_data}   # folder holding facesim/ and models/
OUT=${OUT:-${SCRATCH:-/scratch/users/$USER}/facesim/$NAME}
EXTRA=${EXTRA:-}                     # e.g. EXTRA="--phase-steps 4 --sampling my.json"
WORKERS=${SLURM_CPUS_PER_TASK:-4}

# ---- environment -------------------------------------------------------------
module purge
module load miniforge/24.11.2-1
mamba activate facesim

set -euo pipefail

# One thread per worker process. Without this, NumPy/SciPy in each of the 64
# workers can start its own thread pool and the cores get oversubscribed.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1            # log lines appear in facesim_<jobid>.out right away

cd "$CODE_DIR"
mkdir -p "$OUT"

MASKS=()
if [ -f models/FLAME_masks.pkl ]; then
    MASKS=(--masks models/FLAME_masks.pkl)
fi

echo "job $SLURM_JOB_ID on $(hostname): $IDENTITIES ids x ($SCANS scans + $ATTACKS attacks), $WORKERS workers"
echo "output: $OUT"
start=$(date +%s)

# --workers must be passed explicitly: the default uses os.cpu_count(), which on
# a compute node is all 128 cores, not the cores Slurm gave this job.
python -m facesim.generate \
    --flame models/flame2023_Open.pkl \
    "${MASKS[@]}" \
    --out "$OUT" \
    --identities "$IDENTITIES" \
    --scans-per-identity "$SCANS" \
    --attacks-per-identity "$ATTACKS" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    $EXTRA

n=$(find "$OUT" -name '*.npz' ! -name identity.npz | wc -l)
echo "done in $(( $(date +%s) - start )) s: $n scans, $(du -sh "$OUT" | cut -f1) in $OUT"
