from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from .config import (
    RAW_DIR,
    METADATA_DIR,
    PIPELINE_VERSION,
)

from .hashing import sha256_file


IGNORED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


def make_source_id(
    source_hash: str
) -> str:

    return (
        f"SAR-{source_hash[:16]}"
    )


def discover_source_files(
    root: Path = RAW_DIR
) -> list[Path]:

    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory "
            f"not found: {root}"
        )

    files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.name in IGNORED_NAMES:
            continue

        files.append(path)

    return sorted(files)


def inventory_file(
    path: Path,
    raw_root: Path = RAW_DIR
) -> dict:

    source_hash = sha256_file(path)

    relative_path = (
        path.relative_to(raw_root)
    )

    stat = path.stat()

    return {
        "source_id":
            make_source_id(
                source_hash
            ),

        "original_file":
            path.name,

        "relative_path":
            relative_path.as_posix(),

        "file_size_bytes":
            stat.st_size,

        "source_hash_sha256":
            source_hash,

        "pipeline_version":
            PIPELINE_VERSION,

        "inventory_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


def error_record(
    path: Path,
    exception: Exception,
    raw_root: Path = RAW_DIR
) -> dict:

    try:

        relative_path = (
            path.relative_to(
                raw_root
            ).as_posix()
        )

    except ValueError:

        relative_path = str(path)

    return {
        "relative_path":
            relative_path,

        "error_type":
            type(
                exception
            ).__name__,

        "error_message":
            str(exception),

        "pipeline_version":
            PIPELINE_VERSION,

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


def build_inventory(
    raw_root: Path = RAW_DIR
):

    files = discover_source_files(
        raw_root
    )

    records = []
    errors = []

    print(
        f"Files discovered: "
        f"{len(files)}"
    )

    for index, path in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{path.name}"
        )

        try:

            records.append(
                inventory_file(
                    path,
                    raw_root
                )
            )

        except Exception as exc:

            errors.append(
                error_record(
                    path,
                    exc,
                    raw_root
                )
            )

    return (
        pd.DataFrame(records),
        pd.DataFrame(errors)
    )


def validate_inventory(
    inventory_df: pd.DataFrame
):

    if inventory_df.empty:
        return inventory_df

    inventory_df = (
        inventory_df.copy()
    )

    inventory_df[
        "duplicate_hash"
    ] = (
        inventory_df
        .duplicated(
            subset=(
                "source_hash_sha256"
            ),
            keep=False
        )
    )

    return inventory_df


def save_inventory(
    inventory_df,
    errors_df
):

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        METADATA_DIR
        / "manifest.parquet"
    )

    inventory_df.to_parquet(
        manifest_path,
        index=False
    )

    if not errors_df.empty:

        errors_df.to_csv(
            METADATA_DIR
            / "ingestion_errors.csv",
            index=False,
            encoding="utf-8"
        )

    print()
    print(
        "Inventory completed."
    )

    print(
        f"Valid files: "
        f"{len(inventory_df)}"
    )

    print(
        f"Errors: "
        f"{len(errors_df)}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )


def run_inventory():

    inventory_df, errors_df = (
        build_inventory()
    )

    inventory_df = (
        validate_inventory(
            inventory_df
        )
    )

    save_inventory(
        inventory_df,
        errors_df
    )

    return (
        inventory_df,
        errors_df
    )

