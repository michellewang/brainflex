#!/usr/bin/env python3

"""Export tabular data from a Nipoppy study for a Fed-BioMed node."""

import argparse
import json
import sys
from pathlib import Path

MIN_NIPOPPY_VERSION = "0.4.3"
FNAME_CSV = "dataset.csv"
FNAME_JSON = "dataset_info.json"
FEDBIOMED_DATA_TYPE = "csv"
FEDBIOMED_MIN_LENGTH = 3  # for tags, name and description

try:
    from nipoppy import NipoppyDataRetriever
except ImportError:
    sys.exit(
        "Error: could not import NipoppyDataRetriever. Install a compatible version of "
        f"Nipoppy, e.g. pip install 'nipoppy>={MIN_NIPOPPY_VERSION}'"
    )


def _parse_derivative(value: str) -> tuple[str, str, str]:
    # maxsplit so that the filepath pattern can contain colons
    derivative = tuple(value.split(":", 2))
    if len(derivative) != 3 or not all(derivative):
        raise argparse.ArgumentTypeError(
            f"expected PIPELINE_NAME:PIPELINE_VERSION:FILEPATH_PATTERN, got {value!r}"
        )
    return derivative


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "nipoppy_root", type=Path, help="Root directory of the Nipoppy study."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write the CSV and JSON files to."
    )
    parser.add_argument(
        "--phenotype",
        dest="phenotypes",
        action="append",
        default=[],
        metavar="TERMURL",
        help=(
            "Neurobagel TermURL of a phenotypic column in tabular/harmonized.tsv"
            " (e.g. nb:Age). Can be repeated."
        ),
    )
    parser.add_argument(
        "--derivative",
        dest="derivatives",
        action="append",
        default=[],
        type=_parse_derivative,
        metavar="PIPELINE_NAME:PIPELINE_VERSION:FILEPATH_PATTERN",
        help=(
            "Derivative TSV file to include, where FILEPATH_PATTERN is a glob pattern"
            " relative to the pipeline's derivatives directory"
            " (e.g. freesurfer:7.3.2:idp/*-aseg-volume.tsv). Can be repeated."
        ),
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        required=True,
        help="Fed-BioMed dataset tag. Can be repeated.",
    )
    parser.add_argument(
        "--name",
        help="Fed-BioMed dataset name. Default: name of the Nipoppy root directory.",
    )
    parser.add_argument("--description", help="Fed-BioMed dataset description.")
    return parser


def nipoppy2fedbiomed(
    nipoppy_root: Path,
    output_dir: Path,
    phenotypes: list[str],
    derivatives: list[tuple[str, str, str]],
    tags: list[str],
    name: str | None = None,
    description: str | None = None,
) -> tuple[Path, Path]:
    """Write the CSV dataset and its Fed-BioMed description.

    Returns the paths to the CSV and JSON files.
    """
    if not (phenotypes or derivatives):
        raise ValueError("At least one phenotype or derivative must be specified")
    nipoppy_root = nipoppy_root.resolve()
    name = name or nipoppy_root.name
    if description is None:
        sources = [*phenotypes, *(" ".join(derivative) for derivative in derivatives)]
        description = f"Data from Nipoppy study {nipoppy_root}: {', '.join(sources)}"

    data_retriever = NipoppyDataRetriever(nipoppy_root)
    df = data_retriever.get_tabular_data(
        phenotypes=phenotypes or None,
        derivatives=[tuple(derivative) for derivative in derivatives] or None,
    )
    n_rows = len(df)
    df = df.dropna(axis="index", how="any")
    if df.empty:
        raise ValueError(f"No rows without missing values in {nipoppy_root}")
    if n_dropped := n_rows - len(df):
        print(f"Dropped {n_dropped} rows with missing values")

    output_dir.mkdir(parents=True, exist_ok=True)
    fpath_csv = output_dir / FNAME_CSV
    df.to_csv(fpath_csv, index=False)
    print(f"Wrote {len(df)} rows to {fpath_csv}")

    fpath_json = output_dir / FNAME_JSON
    fpath_json.write_text(
        json.dumps(
            {
                "path": str(fpath_csv.resolve()),
                "data_type": FEDBIOMED_DATA_TYPE,
                "name": name,
                "description": description,
                # Fed-BioMed splits this string on commas
                "tags": ",".join(tags),
            },
            indent=4,
        )
        + "\n"
    )
    print(f"Wrote data info to {fpath_json}")
    return fpath_csv, fpath_json


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    try:
        nipoppy2fedbiomed(
            args.nipoppy_root,
            args.output_dir,
            phenotypes=args.phenotypes,
            derivatives=args.derivatives,
            tags=args.tags,
            name=args.name,
            description=args.description,
        )
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
