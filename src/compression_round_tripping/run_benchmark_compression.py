"""Run compression / decompression loop for a whole directory of ply files."""

import json
import logging
import shutil
import tarfile
from pathlib import Path

import tyro
from beartype import beartype
from tqdm import tqdm

from compression_round_tripping.main import EligibleCompressionFormats, round_trip_compression

logger = logging.getLogger(__name__)


@beartype
def run_benchmark(
    source: Path,
    output_dir: Path,
    compression_formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool = False,
    use_cpu: bool = False,
    keep_extracted: bool = False,
) -> None:
    """Run compression benchmark on a directory or tar file."""
    if not source.exists():
        msg = f"Source {source} does not exist."
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine staging directory
    staging_dir = _setup_staging_dir(source, output_dir, overwrite=overwrite)
    staging_dir_name = staging_dir.name

    # Process files within staging_dir
    ply_files = list(staging_dir.rglob("point_cloud.ply"))
    if not ply_files:
        logger.warning("No point_cloud.ply files found in %s", staging_dir)
    else:
        for ply_path in tqdm(ply_files, desc="Processing Scenes"):
            _process_scene(
                ply_path,
                compression_formats,
                overwrite=overwrite,
                use_cpu=use_cpu,
            )

    # Re-compress to tar
    final_tar_path = output_dir / f"{staging_dir_name}.tar"
    logger.info("Creating final archive %s", final_tar_path)
    with tarfile.open(final_tar_path, "w") as tar:
        tar.add(staging_dir, arcname=staging_dir_name)

    # Cleanup staging if requested (default to True implicitly via keep_extracted=False)
    if not keep_extracted:
        logger.info("Removing staging directory %s", staging_dir)
        shutil.rmtree(staging_dir)


def _setup_staging_dir(source: Path, output_dir: Path, *, overwrite: bool) -> Path:
    """Prepare the staging directory by extracting tar or copying directory."""
    if source.is_file() and source.suffix == ".tar":
        staging_dir_name = source.stem
        staging_dir = output_dir / staging_dir_name

        if staging_dir.exists():
            if overwrite:
                shutil.rmtree(staging_dir)
            else:
                logger.info("Skipping existing staging dir %s", staging_dir)

        if not staging_dir.exists():
            logger.info("Extracting %s to %s", source, staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(source, "r") as tar:
                tar.extractall(path=staging_dir, filter="data")
        return staging_dir

    if source.is_dir():
        staging_dir_name = source.name
        staging_dir = output_dir / staging_dir_name
        if staging_dir.exists():
            if overwrite:
                shutil.rmtree(staging_dir)
            else:
                logger.info("Skipping copy for existing staging dir %s", staging_dir)

        if not staging_dir.exists():
            logger.info("Copying %s to %s", source, staging_dir)
            shutil.copytree(source, staging_dir)
        return staging_dir

    msg = "Source must be a .tar file or a directory."
    raise ValueError(msg)


def _process_scene(
    source_ply: Path,
    formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool,
    use_cpu: bool,
) -> None:
    """Run compression loop for a given PLY scene."""
    ply_dir = source_ply.parent
    stats_file = ply_dir / "compression_stats.json"

    # Ensure clean output dirs
    compressed_dir = ply_dir / "compressed"
    decompressed_dir = ply_dir / "decompressed"
    compressed_dir.mkdir(exist_ok=True)
    decompressed_dir.mkdir(exist_ok=True)

    for fmt in formats:
        final_compressed_file = compressed_dir / f"point_cloud.{fmt}"
        final_decompressed_file = decompressed_dir / f"point_cloud_{fmt}.ply"

        if final_compressed_file.exists() and not overwrite:
            logger.info("Skipping existing %s", final_compressed_file)
            continue

        try:
            # Run round trip (generates stats internally and writes to JSON)
            round_trip_compression(
                input_file=source_ply,
                compression_format=fmt,
                compressed_file=final_compressed_file,
                decompressed_file=final_decompressed_file,
                overwrite=overwrite,
                use_cpu=use_cpu,
            )

            # Merge Stats
            # Stats are generated at decompressed_file name based json
            generated_stats_path = final_decompressed_file.with_name(
                f"{final_decompressed_file.stem}_compression_statistics.json"
            )

            if generated_stats_path.exists():
                new_stats_data = {}
                if stats_file.exists():
                    try:
                        with stats_file.open("r") as f:
                            new_stats_data = json.load(f)
                    except Exception:
                        pass

                with generated_stats_path.open("r") as f:
                    current_stats = json.load(f)

                new_stats_data.update(current_stats)

                with stats_file.open("w") as f:
                    json.dump(new_stats_data, f, indent=4)

                generated_stats_path.unlink()

        except Exception:
            logger.exception("Failed to process format %s for %s", fmt, source_ply)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_benchmark)
