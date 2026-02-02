"""Run compression / decompression loop for a whole directory of ply files."""

import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import tyro
from beartype import beartype

from compression_round_tripping.main import EligibleCompressionFormats, round_trip_compression

logger = logging.getLogger(__name__)


@beartype
def run_benchmark(
    source_dir: Path,
    output_dir: Path,
    compression_formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool = False,
    use_cpu: bool = False,
) -> None:
    """Run compression benchmark on a directory."""
    if not source_dir.exists():
        msg = f"Source directory {source_dir} does not exist."
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = list(source_dir.rglob("*.tar"))
    for tar_path in tar_files:
        _process_tar(
            tar_path,
            source_dir,
            output_dir,
            compression_formats,
            overwrite=overwrite,
            use_cpu=use_cpu,
        )

    ply_files = list(source_dir.rglob("point_cloud.ply"))
    for ply_path in ply_files:
        _process_ply(
            ply_path,
            source_dir,
            output_dir,
            compression_formats,
            overwrite=overwrite,
            use_cpu=use_cpu,
        )


def _process_tar(
    tar_path: Path,
    source_root: Path,
    output_root: Path,
    formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool,
    use_cpu: bool,
) -> None:
    rel_path = tar_path.relative_to(source_root)
    dest_tar_path = output_root / rel_path

    if dest_tar_path.exists() and not overwrite:
        logger.info("Skipping existing %s", dest_tar_path)
        return

    logger.info("Processing tar: %s", tar_path)
    dest_tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_dir, filter="data")

        ply_files = list(temp_dir.rglob("point_cloud.ply"))

        if not ply_files:
            logger.warning("No point_cloud.ply found in %s", tar_path)
            return

        for source_ply in ply_files:
            _process_ply_file_common(source_ply, formats, overwrite=overwrite, use_cpu=use_cpu)

        # Repackage to destination tar
        logger.info("Repacking to %s...", dest_tar_path)
        with tarfile.open(dest_tar_path, "w") as tar:
            tar.add(temp_dir, arcname="")


def _process_ply(
    ply_path: Path,
    source_root: Path,
    output_root: Path,
    formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool,
    use_cpu: bool,
) -> None:
    rel_path = ply_path.relative_to(source_root)
    dest_ply_path = output_root / rel_path

    if dest_ply_path.exists() and not overwrite:
        # Check output existence more robustly?
        pass

    logger.info("Processing loose ply: %s", ply_path)
    dest_ply_path.parent.mkdir(parents=True, exist_ok=True)

    if not dest_ply_path.exists() or overwrite:
        shutil.copy2(ply_path, dest_ply_path)

    _process_ply_file_common(dest_ply_path, formats, overwrite=overwrite, use_cpu=use_cpu)


def _process_ply_file_common(
    source_ply: Path,
    formats: list[EligibleCompressionFormats],
    *,
    overwrite: bool,
    use_cpu: bool,
) -> None:
    """Run compression loop for a given PLY file (in-place/temp processing)."""
    ply_dir = source_ply.parent
    stats_target = ply_dir / "compression_stats.json"

    for fmt in formats:
        # Define strict output paths
        compressed_dir = ply_dir / "compressed"
        decompressed_dir = ply_dir / "decompressed"
        compressed_dir.mkdir(exist_ok=True)
        decompressed_dir.mkdir(exist_ok=True)

        final_compressed_file = compressed_dir / f"point_cloud.{fmt}"
        final_decompressed_file = decompressed_dir / f"point_cloud_{fmt}.ply"

        logger.info("  Running %s...", fmt)

        round_trip_compression(
            input_file=source_ply,
            compression_format=fmt,
            compressed_file=final_compressed_file,
            decompressed_file=final_decompressed_file,
            overwrite=overwrite,
            use_cpu=use_cpu,
        )

        # Merge Stats
        # Stats are generated at restored_file name based json
        generated_stats_path = final_decompressed_file.with_name(
            f"{final_decompressed_file.stem}_compression_statistics.json"
        )

        if generated_stats_path.exists():
            new_stats_data = {}
            if stats_target.exists():
                try:
                    with stats_target.open("r") as f:
                        new_stats_data = json.load(f)
                except Exception:
                    pass

            with generated_stats_path.open("r") as f:
                current_stats = json.load(f)

            new_stats_data.update(current_stats)

            with stats_target.open("w") as f:
                json.dump(new_stats_data, f, indent=4)

            generated_stats_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_benchmark)
