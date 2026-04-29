from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
PREPARED_DIR = PROJECT_DIR / "prepared"


def convert_measurements(
    source: Path = DATASET_DIR / "priebehy.parquet",
    target: Path = PREPARED_DIR / "priebehy_spark.parquet",
) -> Path:
    """Convert nanosecond timestamp parquet to microseconds readable by Spark 3.5."""
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None

    try:
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            fields = []
            for field in table.schema:
                if field.name == "t_utc":
                    fields.append(pa.field("t_utc", pa.timestamp("us", tz="UTC")))
                else:
                    fields.append(field)
            table = table.cast(pa.schema(fields))
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return target


def convert_faults(
    source: Path = DATASET_DIR / "Poruchy.xlsx",
    target_faults: Path = PREPARED_DIR / "poruchy.csv",
    target_types: Path = PREPARED_DIR / "typy_poruch.csv",
) -> tuple[Path, Path]:
    if target_faults.exists() and target_types.exists():
        return target_faults, target_types

    target_faults.parent.mkdir(parents=True, exist_ok=True)
    faults = pd.read_excel(source, sheet_name="Evidencia poruch")
    fault_types = pd.read_excel(source, sheet_name="Typy poruch")

    faults = faults.rename(
        columns={
            "EIC kód": "eic",
            "Dátum vzniku poruchy": "fault_start",
            "Ukončenie poruchy": "fault_end",
            "Typ poruchy": "fault_type",
        }
    )
    fault_types = fault_types.rename(
        columns={
            "Číselník": "fault_type",
            "Popis poruchy": "fault_description",
        }
    )

    faults.to_csv(target_faults, index=False, encoding="utf-8")
    fault_types.to_csv(target_types, index=False, encoding="utf-8")
    return target_faults, target_types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recreate prepared inputs")
    args = parser.parse_args()

    if args.force and PREPARED_DIR.exists():
        for path in [
            PREPARED_DIR / "priebehy_spark.parquet",
            PREPARED_DIR / "poruchy.csv",
            PREPARED_DIR / "typy_poruch.csv",
        ]:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                path.rmdir()

    measurements = convert_measurements()
    faults, types = convert_faults()
    print(f"Prepared measurements: {measurements}")
    print(f"Prepared faults:       {faults}")
    print(f"Prepared fault types:  {types}")


if __name__ == "__main__":
    main()
