"""Small script to test compression round tripping."""

import json
import subprocess
import time
from pathlib import Path
from typing import Literal, get_args

import spz
import tyro
from beartype import beartype
from pydantic import BaseModel, field_serializer

EligibleFileFormats = Literal["spz", "sog", "ply", "cply"]
EligibleCompressionFormats = Literal["sog", "spz", "cply"]


class Arguments(BaseModel):
    """Arguments for the script.

    Args:
        input_file: Path to the input file.
        intermediate_directory: Path to the intermediate directory. Must not include a file that
            has the same name as the input file.
        output_directory: Path to the output directory. Must not include any file name that has
            the same name as the input file.
        compression_format: Compression format to use.
        overwrite: Whether to overwrite the output file if it already exists.
    """

    input_file: Path
    intermediate_directory: Path
    output_directory: Path
    compression_format: EligibleCompressionFormats
    overwrite: bool = False


class CompressionStatistics(BaseModel):
    """Statistics for the compression."""

    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_time_seconds: float
    compression_format: EligibleCompressionFormats
    input_file: Path
    output_file: Path

    def __str__(self) -> str:
        """Print the statistics in a pretty format."""
        return (
            f"Original size: {self.original_size_mb:.2f} MB\n"
            f"Compressed size: {self.compressed_size_mb:.2f} MB\n"
            f"Compression ratio: {self.compression_ratio:.2f}\n"
            f"Compression time: {self.compression_time_seconds:.2f} seconds\n"
            f"Compression format: {self.compression_format}\n"
            f"Input file: {self.input_file}\n"
            f"Output file: {self.output_file}"
        )

    @field_serializer("input_file", "output_file")
    def path_serializer(self, v: Path) -> str:
        """Serializes paths to absolute paths in string format."""
        return str(v.absolute())


class SpzOptions(BaseModel):
    """Options for SPZ compression."""

    trained_with_anti_aliasing: bool = False


def _file_names_sanity_check(
    input_file: Path,
    output_file: Path,
    input_format: EligibleFileFormats,
    output_format: EligibleFileFormats,
    *,
    overwrite: bool = False,
) -> None:
    """Check if the file conversion is correct."""
    if input_file.name == output_file.name:
        msg = "Input and output file names must be different."
        raise ValueError(msg)
    if input_file.suffix != f".{input_format}":
        msg = f"Input file must have extension .{input_format}."
        raise ValueError(msg)
    if output_file.suffix != f".{output_format}":
        msg = f"Output file must have extension .{output_format}."
        raise ValueError(msg)
    if output_file.exists() and not overwrite:
        msg = "Output file already exists."
        raise ValueError(msg)
    if output_file.exists() and overwrite:
        output_file.unlink()
    if not input_file.exists():
        msg = "Input file does not exist."
        raise ValueError(msg)


def compress_sog(
    input_file: Path, output_file: Path, *, overwrite: bool = False, use_cpu: bool = False
) -> None:
    """Compress a file using SOG compression."""
    _file_names_sanity_check(input_file, output_file, "ply", "sog", overwrite=overwrite)
    command = ["splat-transform", str(input_file), str(output_file)]
    if use_cpu:
        command.append("-g")
        command.append("cpu")
    subprocess.run(command, check=True)


def decompress_sog(input_file: Path, output_file: Path, *, overwrite: bool = False) -> None:
    """Decompress a file using SOG compression."""
    _file_names_sanity_check(input_file, output_file, "sog", "ply", overwrite=overwrite)
    command = ["splat-transform", str(input_file), str(output_file)]
    subprocess.run(command, check=True)


def compress_cply(input_file: Path, output_file: Path, *, overwrite: bool = False) -> None:
    """Compress a file using compressed ply compression.

    Quantizes chunks of 256 primitives at a time.
    """
    _file_names_sanity_check(input_file, output_file, "ply", "cply", overwrite=overwrite)
    temp_output_file = output_file.with_suffix(".compressed.ply")
    command = ["splat-transform", str(input_file), "--morton-order", str(temp_output_file)]
    subprocess.run(command, check=True)
    temp_output_file.rename(output_file)


def decompress_cply(input_file: Path, output_file: Path, *, overwrite: bool = False) -> None:
    """Decompress a file using compressed ply compression."""
    _file_names_sanity_check(input_file, output_file, "cply", "ply", overwrite=overwrite)
    temp_input_file = input_file.with_suffix(".compressed.ply")
    input_file.rename(temp_input_file)
    command = ["splat-transform", str(temp_input_file), str(output_file)]
    subprocess.run(command, check=True)
    temp_input_file.rename(input_file)


def compress_spz(input_file: Path, output_file: Path, *, overwrite: bool = False) -> None:
    """Compress a file using SPZ compression."""
    _file_names_sanity_check(input_file, output_file, "ply", "spz", overwrite=overwrite)
    unpack_options = spz.UnpackOptions()
    # unpack_options.to_coord = spz.CoordinateSystem.RDF
    splats = spz.load_splat_from_ply(str(input_file), unpack_options)
    pack_options = spz.PackOptions()  # only saves the coordinate system
    # pack_options.from_coord = spz.CoordinateSystem.RDF
    spz.save_spz(splats, pack_options, str(output_file))


def decompress_spz(input_file: Path, output_file: Path, *, overwrite: bool = False) -> None:
    """Decompress a file using SPZ compression."""
    _file_names_sanity_check(input_file, output_file, "spz", "ply", overwrite=overwrite)

    spz_splats = spz.load_spz(str(input_file))
    pack_options = spz.PackOptions()
    spz.save_splat_to_ply(spz_splats, pack_options, str(output_file))


@beartype
def round_trip_compression(
    input_file: Path,
    compression_format: EligibleCompressionFormats,
    *,
    overwrite: bool = False,
    use_cpu: bool = False,
) -> CompressionStatistics:
    """Compress and decompress a file using SOG compression."""
    # TODO: fix local GPU support (is it just 5090 Shenanigans?)
    intermediate_file = input_file.with_suffix(f".{compression_format}")
    output_file = input_file.with_name(f"{input_file.stem}_round_trip.ply")

    # compress the file
    start_time = time.time()
    match compression_format:
        case "sog":
            compress_sog(input_file, intermediate_file, overwrite=overwrite, use_cpu=use_cpu)
        case "spz":
            compress_spz(input_file, intermediate_file, overwrite=overwrite)
        case "cply":
            compress_cply(input_file, intermediate_file, overwrite=overwrite)
    end_time = time.time()

    # decompress the file
    match compression_format:
        case "sog":
            decompress_sog(intermediate_file, output_file, overwrite=overwrite)
        case "spz":
            decompress_spz(intermediate_file, output_file, overwrite=overwrite)
        case "cply":
            decompress_cply(intermediate_file, output_file, overwrite=overwrite)
    compression_statistics = CompressionStatistics(
        original_size_mb=input_file.stat().st_size / 1024 / 1024,
        compressed_size_mb=intermediate_file.stat().st_size / 1024 / 1024,
        compression_ratio=input_file.stat().st_size / intermediate_file.stat().st_size,
        compression_time_seconds=end_time - start_time,
        compression_format=compression_format,
        input_file=input_file,
        output_file=output_file,
    )

    # serialize the statistics to a json file
    compression_statistics_path = output_file.with_name(
        f"{output_file.stem}_compression_statistics.json"
    )
    statistics_dict = {}
    if compression_statistics_path.exists():
        with compression_statistics_path.open("r") as f:
            try:
                statistics_dict = json.load(f)
            except json.JSONDecodeError as err:
                if not overwrite:
                    msg = f"Corrupted stats file: {compression_statistics_path}"
                    raise ValueError(msg) from err
                statistics_dict = {}

        eligible_formats = get_args(EligibleCompressionFormats)
        keys_to_delete = []

        for key, value in statistics_dict.items():
            is_valid_key = key in eligible_formats
            is_valid_value = True
            try:
                CompressionStatistics.model_validate(value)
            except Exception:
                is_valid_value = False

            if not (is_valid_key and is_valid_value):
                if overwrite:
                    keys_to_delete.append(key)
                else:
                    msg = f"Invalid statistic for key '{key}' in {compression_statistics_path}"
                    raise ValueError(msg)

        for key in keys_to_delete:
            del statistics_dict[key]

    # overwrite the statistic regardless of the overwrite flag
    statistics_dict[compression_format] = compression_statistics.model_dump()

    # dump back into a single json file
    with compression_statistics_path.open("w") as f:
        json.dump(statistics_dict, f, indent=4)

    return compression_statistics


if __name__ == "__main__":
    tyro.cli(round_trip_compression)
