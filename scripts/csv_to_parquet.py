#!/usr/bin/env python3
"""
Convert a large CSV file to Parquet format using chunked processing.
This avoids loading the entire CSV into memory at once.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def csv_to_parquet(csv_path: str, chunksize: int = 500_000) -> None:
    """Convert CSV file to Parquet format using chunked reading."""
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if csv_file.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file: {csv_path}")

    parquet_file = csv_file.with_suffix(".parquet")

    print(f"Converting CSV -> Parquet")
    print(f"  CSV:     {csv_file}")
    print(f"  Parquet: {parquet_file}")
    print(f"  Chunksize: {chunksize} rows")

    reader = pd.read_csv(csv_file, chunksize=chunksize)
    writer = None
    total_rows = 0

    try:
        for i, chunk in enumerate(reader):
            # Optional: adjust dtypes to save space if you want
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                # Initialize writer on first chunk
                writer = pq.ParquetWriter(
                    parquet_file,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            total_rows += len(chunk)
            print(f"  Wrote chunk {i + 1}, rows so far: {total_rows}")

    finally:
        if writer is not None:
            writer.close()

    print(f"Done. Total rows written: {total_rows}")
    print(f"Parquet saved to: {parquet_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV file to Parquet format (chunked)"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file to convert",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Number of rows per chunk when reading the CSV",
    )

    args = parser.parse_args()
    csv_to_parquet(args.csv_file, chunksize=args.chunksize)


if __name__ == "__main__":
    main()