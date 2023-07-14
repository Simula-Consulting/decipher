"""Convenience script for caching data to parquet."""
from pathlib import Path
from typing import Annotated

import typer
from decipher.data import DataManager

app = typer.Typer()


@app.command()
def cache_data(
    screening_data: Annotated[
        Path,
        typer.Argument(
            help="Screening data CSV (`lp_pid_fuzzy.csv').", exists=True, dir_okay=False
        ),
    ],
    dob_data: Annotated[
        Path,
        typer.Argument(
            help="Date of birth CSV (`Folkereg_PID_fuzzy.csv`).",
            exists=True,
            dir_okay=False,
        ),
    ],
    parquet_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory where the cache should be stored.", file_okay=False
        ),
    ],
):
    """Convert CSV data files to parquet files."""

    # Create the output path, with parents as necessary. It is OK if it exists from before.
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if list(parquet_dir.iterdir()) and not typer.confirm(
        "The output directory is not empty, do you want to proceed?"
    ):  # Not empty
        raise typer.Abort()

    # Read in from CSV
    data_manager = DataManager.read_from_csv(screening_data, dob_data, read_hpv=True)
    # Store to Parquet
    data_manager.save_to_parquet(parquet_dir, engine="pyarrow")
    print(f"Successfully stored parquet files in {parquet_dir}")


if __name__ == "__main__":
    app()
