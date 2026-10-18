from pathlib import Path

import pandas as pd

from src.mstar_pipeline.config import (
    RAW_DIR,
    METADATA_DIR,
)

from src.mstar_pipeline.signal import (
    validate_exact_layout,
    diagnose_endianness,
)


def run_endianness_collection_diagnostic():

    manifest_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    output_path = (
        METADATA_DIR
        / "endianness_diagnostic.parquet"
    )

    manifest = pd.read_parquet(
        manifest_path
    )

    # Apenas observações Phoenix confirmadas
    observations = manifest[
        manifest["data_role"]
        == "observation"
    ].copy()

    print(
        "Phoenix observations:",
        len(observations)
    )

    records = []

    total = len(observations)

    for position, (_, row) in enumerate(
        observations.iterrows(),
        start=1,
    ):

        relative_path = row[
            "relative_path"
        ]

        source_path = (
            RAW_DIR
            / relative_path
        )

        print(
            f"[{position}/{total}] "
            f"{relative_path}"
        )

        try:

            layout = validate_exact_layout(
                source_path
            )

            for plane_index in range(
                layout.plane_count
            ):

                diagnostics = (
                    diagnose_endianness(
                        source_path,
                        layout,
                        plane_index=plane_index,
                    )
                )

                for diagnostic in diagnostics:

                    records.append(
                        {
                            "source_id":
                                row["source_id"],

                            "relative_path":
                                relative_path,

                            "rows":
                                layout.rows,

                            "columns":
                                layout.columns,

                            "plane_index":
                                plane_index,

                            "endian":
                                diagnostic.endian,

                            "dtype":
                                diagnostic.dtype,

                            "sample_count":
                                diagnostic.sample_count,

                            "finite_ratio":
                                diagnostic.finite_ratio,

                            "nonnegative_ratio":
                                diagnostic.nonnegative_ratio,

                            "zero_ratio":
                                diagnostic.zero_ratio,

                            "very_small_ratio":
                                diagnostic.very_small_ratio,

                            "very_large_ratio":
                                diagnostic.very_large_ratio,

                            "minimum":
                                diagnostic.minimum,

                            "maximum":
                                diagnostic.maximum,

                            "median":
                                diagnostic.median,

                            "median_abs":
                                diagnostic.median_abs,

                            "p99_abs":
                                diagnostic.p99_abs,

                            "diagnostic_status":
                                "success",
                        }
                    )

        except Exception as exc:

            records.append(
                {
                    "source_id":
                        row["source_id"],

                    "relative_path":
                        relative_path,

                    "diagnostic_status":
                        "error",

                    "error_type":
                        type(exc).__name__,

                    "error_message":
                        str(exc),
                }
            )

    diagnostic = pd.DataFrame(
        records
    )

    diagnostic.to_parquet(
        output_path,
        index=False
    )

    print()
    print(
        "=== DIAGNOSTIC STATUS ==="
    )

    print(
        diagnostic[
            "diagnostic_status"
        ].value_counts(
            dropna=False
        )
    )

    successful = diagnostic[
        diagnostic["diagnostic_status"]
        == "success"
    ].copy()

    print()
    print(
        "=== RECORDS BY ENDIAN / PLANE ==="
    )

    print(
        successful.groupby(
            [
                "plane_index",
                "endian",
            ]
        ).size()
    )

    print()
    print(
        "=== AGGREGATED DIAGNOSTICS ==="
    )

    summary = (
        successful
        .groupby(
            [
                "plane_index",
                "endian",
            ]
        )
        .agg(
            files=(
                "source_id",
                "nunique",
            ),
            mean_finite_ratio=(
                "finite_ratio",
                "mean",
            ),
            min_finite_ratio=(
                "finite_ratio",
                "min",
            ),
            mean_nonnegative_ratio=(
                "nonnegative_ratio",
                "mean",
            ),
            mean_very_small_ratio=(
                "very_small_ratio",
                "mean",
            ),
            mean_very_large_ratio=(
                "very_large_ratio",
                "mean",
            ),
            max_p99_abs=(
                "p99_abs",
                "max",
            ),
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        f"Output: {output_path}"
    )

    return diagnostic


if __name__ == "__main__":

    run_endianness_collection_diagnostic()