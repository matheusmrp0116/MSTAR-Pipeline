from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
    SPECKLE_DIR,
)

from src.mstar_pipeline.derived_speckle import (
    generate_derived_container,
)


# ==================================================
# TEST CONFIGURATION
# ==================================================

REALIZATIONS = 10

CONDITIONS = (
    ("K_N01", 1),
    ("K_N02", 2),
    ("K_N05", 5),
    ("FDS", None),
)


# ==================================================
# TEST PARENT SELECTION
# ==================================================

def select_test_parents(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministically select one parent observation
    from every image dimension available in the
    dataset catalog.

    No predefined image dimensions are assumed.

    For each distinct (rows, columns) combination,
    the observation with the first source_id in
    deterministic alphabetical order is selected.
    """

    required_columns = {
        "source_id",
        "standardized_id",
        "standardized_path",
        "rows",
        "columns",
    }

    missing_columns = (
        required_columns
        - set(catalog.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset catalog is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if catalog.empty:
        raise ValueError(
            "Dataset catalog contains no observations."
        )

    valid_catalog = catalog[
        catalog["rows"].notna()
        & catalog["columns"].notna()
    ].copy()

    if valid_catalog.empty:
        raise ValueError(
            "Dataset catalog contains no observations "
            "with valid rows and columns."
        )

    selected = []

    dimensions = (
        valid_catalog[
            ["rows", "columns"]
        ]
        .drop_duplicates()
        .sort_values(
            ["rows", "columns"]
        )
    )

    for _, dimension in dimensions.iterrows():

        rows = dimension["rows"]
        columns = dimension["columns"]

        subset = valid_catalog[
            (valid_catalog["rows"] == rows)
            & (valid_catalog["columns"] == columns)
        ].copy()

        parent = (
            subset
            .sort_values("source_id")
            .head(1)
        )

        if not parent.empty:
            selected.append(
                parent
            )

    if not selected:
        raise ValueError(
            "No parent observations could be selected "
            "for the derived container test."
        )

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    return (
        result
        .sort_values(
            [
                "rows",
                "columns",
                "source_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "=== DERIVED SPECKLE CONTAINER TEST ==="
    )

    print(
        "realizations per container:",
        REALIZATIONS
    )

    print()

    # --------------------------------------------------
    # Catalog
    # --------------------------------------------------

    catalog_path = (
        METADATA_DIR
        / "dataset_catalog.parquet"
    )

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Dataset catalog not found: "
            f"{catalog_path}"
        )

    catalog = pd.read_parquet(
        catalog_path
    )

    parents = select_test_parents(
        catalog
    )

    print(
        "Dimensions detected:",
        len(parents)
    )

    print()

    print(
        "Selected parents:"
    )

    print(
        parents[
            [
                "source_id",
                "rows",
                "columns",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    # --------------------------------------------------
    # IMPORTANT:
    # Use a separate test directory.
    # Do not mix with final derived files.
    # --------------------------------------------------

    test_root = (
        SPECKLE_DIR.parent
        / "speckle_container_test"
    )

    test_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    container_records = []
    logical_records = []
    errors = []

    expected_containers = (
        len(parents)
        * len(CONDITIONS)
    )

    expected_realizations = (
        expected_containers
        * REALIZATIONS
    )

    job = 0

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    for _, row in parents.iterrows():

        for (
            condition_id,
            n_scatterers,
        ) in CONDITIONS:

            job += 1

            print(
                f"[{job}/{expected_containers}] "
                f"{row['source_id']} "
                f"{row['rows']}x{row['columns']} "
                f"{condition_id}"
            )

            try:

                (
                    container_result,
                    realization_records,
                ) = generate_derived_container(
                    parent_source_id=
                        row["source_id"],

                    parent_standardized_id=
                        row["standardized_id"],

                    standardized_path=
                        Path(
                            row[
                                "standardized_path"
                            ]
                        ),

                    output_root=
                        test_root,

                    condition_id=
                        condition_id,

                    n_scatterers=
                        n_scatterers,

                    realization_count=
                        REALIZATIONS,
                )

                container_records.append(
                    asdict(
                        container_result
                    )
                )

                logical_records.extend(
                    realization_records
                )

            except Exception as exc:

                errors.append(
                    {
                        "parent_source_id":
                            row["source_id"],

                        "rows":
                            int(
                                row["rows"]
                            ),

                        "columns":
                            int(
                                row["columns"]
                            ),

                        "condition_id":
                            condition_id,

                        "error_type":
                            type(exc).__name__,

                        "error_message":
                            str(exc),
                    }
                )

    # --------------------------------------------------
    # Tables
    # --------------------------------------------------

    containers = pd.DataFrame(
        container_records
    )

    realizations = pd.DataFrame(
        logical_records
    )

    # --------------------------------------------------
    # Save test metadata
    # --------------------------------------------------

    container_manifest_path = (
        METADATA_DIR
        / "derived_container_test_manifest.parquet"
    )

    realization_manifest_path = (
        METADATA_DIR
        / "derived_container_test_realizations.parquet"
    )

    containers.to_parquet(
        container_manifest_path,
        index=False,
    )

    realizations.to_parquet(
        realization_manifest_path,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()

    print(
        "=== CONTAINER TEST SUMMARY ==="
    )

    print(
        "Dimensions tested:",
        len(parents)
    )

    print(
        "Expected containers:",
        expected_containers
    )

    print(
        "Generated containers:",
        len(containers)
    )

    print(
        "Expected logical realizations:",
        expected_realizations
    )

    print(
        "Logical realization records:",
        len(realizations)
    )

    print(
        "Errors:",
        len(errors)
    )

    if not containers.empty:

        print()

        print(
            "=== DIMENSIONS ==="
        )

        print(
            containers[
                [
                    "rows",
                    "columns",
                ]
            ]
            .value_counts()
            .sort_index()
        )

        print()

        print(
            "=== CONDITIONS ==="
        )

        print(
            containers[
                "condition_id"
            ]
            .value_counts()
            .sort_index()
        )

        print()

        print(
            "=== CONTAINER IDS ==="
        )

        print(
            "Records:",
            len(containers)
        )

        print(
            "Unique IDs:",
            containers[
                "container_id"
            ].nunique()
        )

        print()

        print(
            "=== LOGICAL DERIVED IDS ==="
        )

        print(
            "Records:",
            len(realizations)
        )

        print(
            "Unique IDs:",
            realizations[
                "derived_id"
            ].nunique()
        )

        print()

        print(
            "=== CONTAINER VALIDATION ==="
        )

        validation_columns = (
            "magnitude_finite",
            "phase_finite",
            "zero_pixels_preserved",
            "reload_magnitude_equal",
            "reload_phase_equal",
            "reload_seeds_equal",
            "reload_realization_ids_equal",
            "deterministic_reproduction",
        )

        for column in validation_columns:

            print()

            print(
                column
            )

            print(
                containers[
                    column
                ].value_counts(
                    dropna=False
                )
            )

    # --------------------------------------------------
    # Validate logical indexing
    # --------------------------------------------------

    if not realizations.empty:

        print()

        print(
            "=== ARRAY INDEX VALIDATION ==="
        )

        index_check = (
            realizations
            .groupby(
                "container_id"
            )[
                "array_index"
            ]
            .apply(
                lambda values:
                    sorted(
                        values.tolist()
                    )
                    == list(
                        range(
                            REALIZATIONS
                        )
                    )
            )
        )

        print(
            index_check.value_counts(
                dropna=False
            )
        )

        print()

        print(
            "=== REALIZATION ID VALIDATION ==="
        )

        realization_check = (
            realizations
            .groupby(
                "container_id"
            )[
                "realization_id"
            ]
            .apply(
                lambda values:
                    sorted(
                        values.tolist()
                    )
                    == list(
                        range(
                            1,
                            REALIZATIONS + 1,
                        )
                    )
            )
        )

        print(
            realization_check.value_counts(
                dropna=False
            )
        )

    # --------------------------------------------------
    # Physical files
    # --------------------------------------------------

    files = list(
        test_root.rglob(
            "*.npz"
        )
    )

    total_bytes = sum(
        file.stat().st_size
        for file in files
    )

    print()

    print(
        "=== PHYSICAL STORAGE ==="
    )

    print(
        "NPZ files:",
        len(files)
    )

    print(
        "Total MB:",
        round(
            total_bytes
            / (1024 ** 2),
            3,
        )
    )

    if files:

        print(
            "Average KB/container:",
            round(
                (
                    total_bytes
                    / len(files)
                )
                / 1024,
                3,
            )
        )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    if errors:

        error_path = (
            METADATA_DIR
            / "derived_container_test_errors.csv"
        )

        pd.DataFrame(
            errors
        ).to_csv(
            error_path,
            index=False,
        )

        print()

        print(
            f"Errors saved to: "
            f"{error_path}"
        )

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------

    print()

    print(
        f"Container manifest: "
        f"{container_manifest_path}"
    )

    print(
        f"Logical manifest: "
        f"{realization_manifest_path}"
    )


if __name__ == "__main__":
    main()