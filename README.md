# Compression Round Tripping

Tooling for benchmarking 3D Gaussian Splatting compression methods (SPZ, SOG, Compressed PLY). 
It supports round-trip compression (original -> compressed -> restored) to evaluate size reduction and restoration quality, collecting detailed statistics (compression time, GPU usage, etc.).

## Prerequisites

This project relies on external tools for certain compression formats.

### 1. Install `splat-transform` (Required for .sog and .cply)
You need `npm` installed.

```bash
npm install -g splat-transform
```

### 2. Python Environment
This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

### Optional: Apptainer Container
We also provide an apptainer image, to build it do the following:

```bash
apptainer build --fakeroot --force image.sif apptainer_image.def
```

## Usage

### Single File Round-Trip
Use `main.py` to compress and restore a single `.ply` file.

```bash
# Get help
uv run python -m compression_round_tripping.main --help

# Example: Compress to SPZ and restore
uv run python -m compression_round_tripping.main \
  --input-file data/point_cloud.ply \
  --compression-format spz \
  --compressed-file output/compressed.spz \
  --decompressed-file output/restored.ply
```

### Batch Benchmarking
Use `run_benchmark_compression.py` to process an entire directory structure. It handles both loose `.ply` files and `.tar` archives containing scenes.

The script mirrors the source directory structure into the output directory.

```bash
# Get help
uv run python -m compression_round_tripping.run_benchmark_compression --help

# Example: Benchmark both SOG and SPZ on a dataset
uv run python -m compression_round_tripping.run_benchmark_compression \
  --source-dir /path/to/sparse_scenes \
  --output-dir /path/to/results \
  --compression-formats sog spz
```

## Output
The tools generate a `compression_stats.json` file alongside the restored point cloud, correctly aggregating results for multiple formats if run sequentially.

**Example Structure:**
```
output_dir/
  └── scene_name/
      ├── point_cloud.ply                  # Original (if copied)
      ├── compression_stats.json           # Stats for all formats
      ├── compressed/
      │   ├── point_cloud.spz
      │   └── point_cloud.sog
      └── decompressed/
          ├── point_cloud_spz.ply
          └── point_cloud_sog.ply
```
