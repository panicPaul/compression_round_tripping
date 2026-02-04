#!/bin/bash
#SBATCH --job-name=compression_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Hardcoded Paths
DATA_DIR="/data/cluster/users/schlack/data/sparse_scenes/"
OUTPUT_DIR="/data/cluster/users/schlack/data/sparse_scenes_compressed/"
COMPRESSION_FORMATS="sog spz cply"

# Ensure output directory and logs exist
mkdir -p "$OUTPUT_DIR/logs"

# Redirect stdout and stderr to a log file in the output directory
LOG_FILE="$OUTPUT_DIR/logs/benchmark_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "Running compression benchmark"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Input Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "================================================================"

# Loop over all .tar files in the data directory
for tar_file in "$DATA_DIR"/*.tar; do
    [ -e "$tar_file" ] || continue
    
    echo "\n----------------------------------------------------------------"
    echo "Processing archive: $tar_file"
    echo "----------------------------------------------------------------\n"
    
    uv run src/compression_round_tripping/run_benchmark_compression.py \
        --source "$tar_file" \
        --output_dir "$OUTPUT_DIR" \
        --compression_formats "$COMPRESSION_FORMATS"
done
