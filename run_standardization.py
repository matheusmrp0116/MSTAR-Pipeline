from pathlib import Path

import pandas as pd

from src.mstar_pipeline.config import (
    RAW_DIR,
    METADATA_DIR,
)

from src.mstar_pipeline.standardize import (
    standardize_observation,
    result_to_dict,
)


def run_standardization():

    manifest_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    output_path = (
        METADATA_DIR
        / "standardization_manifest.parquet"
    )

    error_path = (
        METADATA_DIR
        / "standardization_errors.csv"
    )

    manifest = pd.read_parquet(
        manifest_path
    )

    observations = manifest[
        manifest["data_role"]
        == "observation"
    ].copy()

    total = len(observations)

    print(
        f"Observations to standardize: {total}"
    )

    records = []
    errors = []

    for position, (_, row) in enumerate(
        observations.iterrows(),
        start=1,
    ):

        source_id = row[
            "source_id"
        ]

        relative_path = row[
            "relative_path"
        ]

        source_hash = row[
            "source_hash_sha256"
        ]

        source_path = (
            RAW_DIR
            / relative_path
        )

        print(
            f"[{position}/{total}] "
            f"{source_id}"
        )

        try:

            result = standardize_observation(
                source_path=source_path,
                source_id=source_id,
                source_hash_sha256=source_hash,
            )

            record = result_to_dict(
                result
            )

            record[
                "relative_path"
            ] = relative_path

            records.append(
                record
            )

        except Exception as exc:

            error_record = {
                "source_id":
                    source_id,

                "relative_path":
                    relative_path,

                "status":
                    "error",

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            }

            records.append(
                error_record
            )

            errors.append(
                error_record
            )

    # ------------------------------------------
    # Standardization manifest
    # ------------------------------------------

    standardized = pd.DataFrame(
        records
    )

    standardized.to_parquet(
        output_path,
        index=False,
    )

    # ------------------------------------------
    # Error report
    # ------------------------------------------

    errors_df = pd.DataFrame(
        errors
    )

    if not errors_df.empty:

        errors_df.to_csv(
            error_path,
            index=False,
        )

    elif error_path.exists():

        error_path.unlink()

    # ------------------------------------------
    # Summary
    # ------------------------------------------

    print()
    print(
        "=== STANDARDIZATION SUMMARY ==="
    )

    print(
        "Input observations:",
        total
    )

    print(
        "Success:",
        (
            standardized["status"]
            == "success"
        ).sum()
    )

    print(
        "Errors:",
        (
            standardized["status"]
            == "error"
        ).sum()
    )

    successful = standardized[
        standardized["status"]
        == "success"
    ]

    print()
    print(
        "=== DIMENSIONS ==="
    )

    print(
        successful[
            ["rows", "columns"]
        ]
        .value_counts()
    )

    print()
    print(
        "=== RELOAD VALIDATION ==="
    )

    print(
        "Magnitude exact:",
        successful[
            "validation_equal_magnitude"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "Phase exact:",
        successful[
            "validation_equal_phase"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print(
        f"Manifest: {output_path}"
    )

    return standardized


if __name__ == "__main__":

    run_standardization()