from pathlib import Path

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
)

from src.mstar_pipeline.complex_signal import (
    validate_complex_roundtrip,
)


def main():

    catalog_path = (
        METADATA_DIR
        / "dataset_catalog.parquet"
    )

    catalog = pd.read_parquet(
        catalog_path
    )

    catalog = (
        catalog
        .sort_values("source_id")
        .reset_index(drop=True)
    )

    total = len(catalog)

    print(
        "Observations:",
        total
    )

    records = []

    for position, row in catalog.iterrows():

        source_id = row[
            "source_id"
        ]

        standardized_path = Path(
            row[
                "standardized_path"
            ]
        )

        print(
            f"[{position + 1}/{total}] "
            f"{source_id}"
        )

        try:

            with np.load(
                standardized_path,
                allow_pickle=False,
            ) as data:

                magnitude = data[
                    "magnitude"
                ]

                phase = data[
                    "phase"
                ]

            validation = (
                validate_complex_roundtrip(
                    magnitude,
                    phase,
                )
            )

            validation.update(
                {
                    "source_id":
                        source_id,

                    "rows":
                        int(
                            row["rows"]
                        ),

                    "columns":
                        int(
                            row["columns"]
                        ),

                    "status":
                        "success",
                }
            )

            records.append(
                validation
            )

        except Exception as exc:

            records.append(
                {
                    "source_id":
                        source_id,

                    "rows":
                        row["rows"],

                    "columns":
                        row["columns"],

                    "status":
                        "error",

                    "error_type":
                        type(exc).__name__,

                    "error_message":
                        str(exc),
                }
            )

    results = pd.DataFrame(
        records
    )

    output_path = (
        METADATA_DIR
        / "complex_roundtrip_validation.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    successful = results[
        results["status"]
        == "success"
    ]

    errors = results[
        results["status"]
        == "error"
    ]

    print()
    print(
        "=== COMPLEX ROUNDTRIP VALIDATION ==="
    )

    print(
        "Total:",
        total
    )

    print(
        "Success:",
        len(successful)
    )

    print(
        "Errors:",
        len(errors)
    )

    print()
    print(
        "Complex dtype:"
    )

    print(
        successful[
            "complex_dtype"
        ].value_counts()
    )

    print()
    print(
        "=== MAGNITUDE ERROR ==="
    )

    print(
        "Maximum:",
        successful[
            "magnitude_max_abs_error"
        ].max()
    )

    print(
        "Mean:",
        successful[
            "magnitude_mean_abs_error"
        ].mean()
    )

    print()
    print(
        "=== CIRCULAR PHASE ERROR ==="
    )

    print(
        "Maximum:",
        successful[
            "phase_max_circular_error"
        ].max()
    )

    print(
        "Mean:",
        successful[
            "phase_mean_circular_error"
        ].mean()
    )

    print()
    print(
        "=== PHASE PIXELS ==="
    )

    total_valid_phase_pixels = int(
        successful[
            "phase_valid_pixels"
        ].sum()
    )

    total_undefined_phase_pixels = int(
        successful[
            "phase_undefined_pixels"
        ].sum()
    )

    total_phase_pixels = (
        total_valid_phase_pixels
        + total_undefined_phase_pixels
    )

    print(
        "Valid phase pixels:",
        total_valid_phase_pixels
    )

    print(
        "Undefined phase pixels "
        "(magnitude == 0):",
        total_undefined_phase_pixels
    )

    if total_phase_pixels > 0:

        undefined_percentage = (
            100.0
            * total_undefined_phase_pixels
            / total_phase_pixels
        )

    print(
        "Undefined phase percentage:",
        f"{undefined_percentage:.6f}%"
    )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()