#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=compression_benchmark
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.schlack@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# --- Hardcoded Paths ---
DATA_DIR="/data/cluster/users/schlack/data/sparse_scenes"
OUTPUT_DIR="/data/cluster/users/schlack/data/sparse_scenes_compressed"
COMPRESSION_FORMATS="sog spz cply"

# --- Logging Setup ---
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/benchmark_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "Starting Compression Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Input: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

# --- Main Loop ---
# We bind the parent directory so the container can reach both Input and Output dirs
BIND_PATH="/data/cluster/users/schlack"

for tar_file in "$DATA_DIR"/*.tar; do
    # Check if file exists
    [ -e "$tar_file" ] || continue
    
    FILENAME=$(basename "$tar_file")
    echo "Processing: $FILENAME"

    # Run Apptainer using ./image.sif directly
    apptainer exec --nv --bind "$BIND_PATH" ./image.sif \
        uv run src/compression_round_tripping/run_benchmark_compression.py \
        --source "$tar_file" \
        --output_dir "$OUTPUT_DIR" \
        --compression_formats "$COMPRESSION_FORMATS"
        
    echo "Finished: $FILENAME"
    echo "----------------------------------------------------------------"
done

echo "Benchmark Complete."