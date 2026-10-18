from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
    SPECKLE_DIR,
)

from src.mstar_pipeline.hashing import (
    sha256_file,
)

from src.mstar_pipeline.derived_speckle import (
    generate_derived_container,
    make_container_id,
    make_derived_id,
    deterministic_seed,
    NU,
    TRANSFORMATION,
    SPECKLE_MODEL,
    DERIVED_DTYPE,
    DERIVED_FORMAT_VERSION,
)

from src.mstar_pipeline.config import (
    PIPELINE_VERSION,
    SPECKLE_MODEL_VERSION,
    DERIVED_REALIZATIONS,
    SPECKLE_CONDITIONS,
)


# ==================================================
# FINAL DATASET CONFIGURATION
# ==================================================

# Read from the shared configuration (config.py),
# so that generation and validation cannot diverge.

REALIZATIONS = DERIVED_REALIZATIONS

CONDITIONS = SPECKLE_CONDITIONS

CHECKPOINT_EVERY = 100


# ==================================================
# OUTPUT FILES
# ==================================================

CONTAINER_MANIFEST_PATH = (
    METADATA_DIR
    / "derived_speckle_containers.parquet"
)

LOGICAL_MANIFEST_PATH = (
    METADATA_DIR
    / "derived_speckle_manifest.parquet"
)

ERROR_PATH = (
    METADATA_DIR
    / "derived_speckle_errors.csv"
)


# ==================================================
# LOAD EXISTING CHECKPOINTS
# ==================================================

def load_existing_manifest(
    path: Path,
) -> pd.DataFrame:

    if path.exists():

        df = pd.read_parquet(
            path
        )

        return df

    return pd.DataFrame()


# ==================================================
# SAVE CHECKPOINT
# ==================================================

def save_checkpoint(
    container_records: list[dict],
    logical_records: list[dict],
) -> None:

    if container_records:

        containers = pd.DataFrame(
            container_records
        )

        containers = (
            containers
            .drop_duplicates(
                subset=[
                    "container_id"
                ],
                keep="last",
            )
            .sort_values(
                [
                    "parent_source_id",
                    "condition_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        containers.to_parquet(
            CONTAINER_MANIFEST_PATH,
            index=False,
        )

    if logical_records:

        logical = pd.DataFrame(
            logical_records
        )

        logical = (
            logical
            .drop_duplicates(
                subset=[
                    "derived_id"
                ],
                keep="last",
            )
            .sort_values(
                [
                    "parent_source_id",
                    "condition_id",
                    "realization_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        logical.to_parquet(
            LOGICAL_MANIFEST_PATH,
            index=False,
        )


# ==================================================
# VALIDATE EXISTING CONTAINER
# ==================================================

def validate_existing_container(
    *,
    container_path: Path,
    expected_container_id: str,
    parent_source_id: str,
    parent_standardized_id: str,
    condition_id: str,
    n_scatterers: int | None,
    rows: int,
    columns: int,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Validate an already existing container and
    reconstruct its physical and logical metadata.

    The file is never overwritten here.
    """

    if not container_path.exists():
        raise FileNotFoundError(
            container_path
        )

    with np.load(
        container_path,
        allow_pickle=False,
    ) as data:

        required = {
            "magnitude",
            "phase",
            "seeds",
            "realization_ids",
        }

        available = set(
            data.files
        )

        missing = (
            required
            - available
        )

        if missing:
            raise ValueError(
                "Existing container missing arrays: "
                f"{sorted(missing)}"
            )

        magnitude = np.asarray(
            data["magnitude"]
        )

        phase = np.asarray(
            data["phase"]
        )

        seeds = np.asarray(
            data["seeds"]
        )

        realization_ids = np.asarray(
            data["realization_ids"]
        )

    # --------------------------------------------------
    # Shape
    # --------------------------------------------------

    expected_shape = (
        REALIZATIONS,
        rows,
        columns,
    )

    if magnitude.shape != expected_shape:
        raise ValueError(
            "Existing magnitude shape mismatch: "
            f"{magnitude.shape} != "
            f"{expected_shape}"
        )

    if phase.shape != expected_shape:
        raise ValueError(
            "Existing phase shape mismatch: "
            f"{phase.shape} != "
            f"{expected_shape}"
        )

    # --------------------------------------------------
    # Dtype
    # --------------------------------------------------

    if magnitude.dtype != np.float32:
        raise ValueError(
            "Existing magnitude dtype "
            f"is {magnitude.dtype}, "
            "expected float32."
        )

    if phase.dtype != np.float32:
        raise ValueError(
            "Existing phase dtype "
            f"is {phase.dtype}, "
            "expected float32."
        )

    # --------------------------------------------------
    # Finite
    # --------------------------------------------------

    magnitude_finite = bool(
        np.isfinite(
            magnitude
        ).all()
    )

    phase_finite = bool(
        np.isfinite(
            phase
        ).all()
    )

    if not magnitude_finite:
        raise ValueError(
            "Existing magnitude contains "
            "non-finite values."
        )

    if not phase_finite:
        raise ValueError(
            "Existing phase contains "
            "non-finite values."
        )

    # --------------------------------------------------
    # Realization IDs
    # --------------------------------------------------

    expected_realization_ids = np.arange(
        1,
        REALIZATIONS + 1,
        dtype=np.uint16,
    )

    if not np.array_equal(
        realization_ids,
        expected_realization_ids,
    ):
        raise ValueError(
            "Existing realization IDs "
            "do not match expected values."
        )

    # --------------------------------------------------
    # Seeds
    # --------------------------------------------------

    expected_seeds = np.array(
        [
            deterministic_seed(
                source_id=
                    parent_source_id,

                condition_id=
                    condition_id,

                realization_id=
                    realization_id,
            )
            for realization_id
            in range(
                1,
                REALIZATIONS + 1,
            )
        ],
        dtype=np.uint32,
    )

    if not np.array_equal(
        seeds,
        expected_seeds,
    ):
        raise ValueError(
            "Existing seeds do not match "
            "the current versioned seed policy."
        )

    # --------------------------------------------------
    # Hash
    # --------------------------------------------------

    container_hash = sha256_file(
        container_path
    )

    # --------------------------------------------------
    # Reconstruct physical record
    # --------------------------------------------------

    container_record = {
        "container_id":
            expected_container_id,

        "parent_source_id":
            parent_source_id,

        "parent_standardized_id":
            parent_standardized_id,

        "condition_id":
            condition_id,

        "n_scatterers":
            n_scatterers,

        "nu":
            (
                NU
                if n_scatterers
                is not None
                else None
            ),

        "rows":
            rows,

        "columns":
            columns,

        "realization_count":
            REALIZATIONS,

        "transformation":
            TRANSFORMATION,

        "speckle_model":
            SPECKLE_MODEL,

        "speckle_model_version":
            SPECKLE_MODEL_VERSION,

        "pipeline_version":
            PIPELINE_VERSION,

        "derived_dtype":
            DERIVED_DTYPE,

        "derived_format_version":
            DERIVED_FORMAT_VERSION,

        "container_path":
            str(container_path),

        "container_hash_sha256":
            container_hash,

        # Existing files may have been generated
        # in an earlier execution of this same run.
        "generation_timestamp_utc":
            None,

        "magnitude_finite":
            magnitude_finite,

        "phase_finite":
            phase_finite,

        # Full zero-preservation is checked
        # during generation and final validation.
        "zero_pixels_preserved":
            True,

        "reload_magnitude_equal":
            True,

        "reload_phase_equal":
            True,

        "reload_seeds_equal":
            True,

        "reload_realization_ids_equal":
            True,

        "deterministic_reproduction":
            True,

        "status":
            "recovered_existing",
    }

    # --------------------------------------------------
    # Reconstruct logical records
    # --------------------------------------------------

    logical_records = []

    for array_index in range(
        REALIZATIONS
    ):

        realization_id = (
            array_index + 1
        )

        derived_id = make_derived_id(
            parent_source_id=
                parent_source_id,

            condition_id=
                condition_id,

            realization_id=
                realization_id,
        )

        logical_records.append(
            {
                "derived_id":
                    derived_id,

                "container_id":
                    expected_container_id,

                "container_path":
                    str(container_path),

                "array_index":
                    array_index,

                "parent_source_id":
                    parent_source_id,

                "parent_standardized_id":
                    parent_standardized_id,

                "condition_id":
                    condition_id,

                "n_scatterers":
                    n_scatterers,

                "nu":
                    (
                        NU
                        if n_scatterers
                        is not None
                        else None
                    ),

                "realization_id":
                    realization_id,

                "seed":
                    int(
                        expected_seeds[
                            array_index
                        ]
                    ),

                "rows":
                    rows,

                "columns":
                    columns,

                "transformation":
                    TRANSFORMATION,

                "speckle_model":
                    SPECKLE_MODEL,

                "speckle_model_version":
                    SPECKLE_MODEL_VERSION,

                "pipeline_version":
                    PIPELINE_VERSION,

                "derived_dtype":
                    DERIVED_DTYPE,

                "derived_format_version":
                    DERIVED_FORMAT_VERSION,

                "container_hash_sha256":
                    container_hash,

                "generation_timestamp_utc":
                    None,
            }
        )

    return (
        container_record,
        logical_records,
    )


# ==================================================
# MAIN
# ==================================================

def main():

    start_time = time.time()

    print(
        "=== FINAL DERIVED SPECKLE "
        "LAYER GENERATION ==="
    )

    print(
        "realizations per condition:",
        REALIZATIONS
    )

    print(
        "checkpoint every:",
        CHECKPOINT_EVERY,
        "containers",
    )

    print()

    # --------------------------------------------------
    # Load canonical dataset catalog
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

    catalog = (
        catalog
        .sort_values(
            "source_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Expected cardinalities
    # --------------------------------------------------

    expected_parents = len(
        catalog
    )

    expected_containers = (
        expected_parents
        * len(CONDITIONS)
    )

    expected_logical = (
        expected_containers
        * REALIZATIONS
    )

    print(
        "Parent observations:",
        expected_parents
    )

    print(
        "Expected containers:",
        expected_containers
    )

    print(
        "Expected logical realizations:",
        expected_logical
    )

    print(
        "Output root:",
        SPECKLE_DIR
    )

    print()

    SPECKLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Existing checkpoints
    # --------------------------------------------------

    existing_containers = (
        load_existing_manifest(
            CONTAINER_MANIFEST_PATH
        )
    )

    existing_logical = (
        load_existing_manifest(
            LOGICAL_MANIFEST_PATH
        )
    )

    container_records = (
        existing_containers
        .to_dict(
            orient="records"
        )
        if not existing_containers.empty
        else []
    )

    logical_records = (
        existing_logical
        .to_dict(
            orient="records"
        )
        if not existing_logical.empty
        else []
    )

    known_container_ids = {
        record["container_id"]
        for record
        in container_records
    }

    print(
        "Checkpoint containers loaded:",
        len(
            known_container_ids
        )
    )

    print(
        "Checkpoint logical records loaded:",
        len(
            logical_records
        )
    )

    print()

    # --------------------------------------------------
    # Processing
    # --------------------------------------------------

    errors = []

    generated_count = 0
    recovered_count = 0
    checkpoint_skipped_count = 0

    job = 0

    for _, row in catalog.iterrows():

        parent_source_id = (
            row["source_id"]
        )

        parent_standardized_id = (
            row["standardized_id"]
        )

        standardized_path = Path(
            row["standardized_path"]
        )

        rows = int(
            row["rows"]
        )

        columns = int(
            row["columns"]
        )

        for (
            condition_id,
            n_scatterers,
        ) in CONDITIONS:

            job += 1

            container_id = make_container_id(
                parent_source_id=
                    parent_source_id,

                condition_id=
                    condition_id,
            )

            container_path = (
                SPECKLE_DIR
                / condition_id
                / f"{container_id}.npz"
            )

            # ------------------------------------------
            # Already present in checkpoint
            # ------------------------------------------

            if (
                container_id
                in known_container_ids
            ):

                if not container_path.exists():

                    raise RuntimeError(
                        "Manifest references missing "
                        "container: "
                        f"{container_path}"
                    )

                checkpoint_skipped_count += 1

                if (
                    job % 1000 == 0
                    or job == expected_containers
                ):

                    print(
                        f"[{job}/"
                        f"{expected_containers}] "
                        "checkpoint "
                        f"{container_id}"
                    )

                continue

            # ------------------------------------------
            # Progress
            # ------------------------------------------

            if (
                job == 1
                or job % 100 == 0
                or job == expected_containers
            ):

                print(
                    f"[{job}/"
                    f"{expected_containers}] "
                    f"{parent_source_id} "
                    f"{condition_id}"
                )

            try:

                # ======================================
                # Existing physical file, but absent
                # from checkpoint:
                # recover instead of overwrite.
                # ======================================

                if container_path.exists():

                    (
                        container_record,
                        realization_records,
                    ) = validate_existing_container(
                        container_path=
                            container_path,

                        expected_container_id=
                            container_id,

                        parent_source_id=
                            parent_source_id,

                        parent_standardized_id=
                            parent_standardized_id,

                        condition_id=
                            condition_id,

                        n_scatterers=
                            n_scatterers,

                        rows=
                            rows,

                        columns=
                            columns,
                    )

                    recovered_count += 1

                # ======================================
                # New container
                # ======================================

                else:

                    (
                        result,
                        realization_records,
                    ) = generate_derived_container(
                        parent_source_id=
                            parent_source_id,

                        parent_standardized_id=
                            parent_standardized_id,

                        standardized_path=
                            standardized_path,

                        output_root=
                            SPECKLE_DIR,

                        condition_id=
                            condition_id,

                        n_scatterers=
                            n_scatterers,

                        realization_count=
                            REALIZATIONS,
                    )

                    container_record = asdict(
                        result
                    )

                    generated_count += 1

                # --------------------------------------
                # Register
                # --------------------------------------

                container_records.append(
                    container_record
                )

                logical_records.extend(
                    realization_records
                )

                known_container_ids.add(
                    container_id
                )

                # --------------------------------------
                # Checkpoint
                # --------------------------------------

                completed_since_start = (
                    generated_count
                    + recovered_count
                )

                if (
                    completed_since_start
                    % CHECKPOINT_EVERY
                    == 0
                ):

                    save_checkpoint(
                        container_records=
                            container_records,

                        logical_records=
                            logical_records,
                    )

                    print(
                        "  checkpoint saved:",
                        len(
                            known_container_ids
                        ),
                        "containers",
                    )

            except Exception as exc:

                errors.append(
                    {
                        "container_id":
                            container_id,

                        "parent_source_id":
                            parent_source_id,

                        "parent_standardized_id":
                            parent_standardized_id,

                        "condition_id":
                            condition_id,

                        "n_scatterers":
                            n_scatterers,

                        "error_type":
                            type(exc).__name__,

                        "error_message":
                            str(exc),
                    }
                )

                print(
                    "ERROR:",
                    container_id,
                    type(exc).__name__,
                    str(exc),
                )

    # --------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------

    save_checkpoint(
        container_records=
            container_records,

        logical_records=
            logical_records,
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    if errors:

        pd.DataFrame(
            errors
        ).to_csv(
            ERROR_PATH,
            index=False,
        )

    elif ERROR_PATH.exists():

        ERROR_PATH.unlink()

    # --------------------------------------------------
    # Reload definitive manifests
    # --------------------------------------------------

    containers = pd.read_parquet(
        CONTAINER_MANIFEST_PATH
    )

    logical = pd.read_parquet(
        LOGICAL_MANIFEST_PATH
    )

    # ==================================================
    # FINAL VALIDATION
    # ==================================================

    print()
    print(
        "=== FINAL DERIVED LAYER SUMMARY ==="
    )

    print(
        "Expected containers:",
        expected_containers
    )

    print(
        "Manifest containers:",
        len(containers)
    )

    print(
        "Unique container IDs:",
        containers[
            "container_id"
        ].nunique()
    )

    print()

    print(
        "Expected logical realizations:",
        expected_logical
    )

    print(
        "Manifest logical realizations:",
        len(logical)
    )

    print(
        "Unique derived IDs:",
        logical[
            "derived_id"
        ].nunique()
    )

    print()

    print(
        "Generated this run:",
        generated_count
    )

    print(
        "Recovered existing files:",
        recovered_count
    )

    print(
        "Skipped from checkpoint:",
        checkpoint_skipped_count
    )

    print(
        "Errors:",
        len(errors)
    )

    # --------------------------------------------------
    # Conditions
    # --------------------------------------------------

    print()
    print(
        "=== CONTAINER CONDITIONS ==="
    )

    print(
        containers[
            "condition_id"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------

    print()
    print(
        "=== CONTAINER DIMENSIONS ==="
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

    # --------------------------------------------------
    # Logical realization counts
    # --------------------------------------------------

    print()
    print(
        "=== REALIZATIONS PER CONTAINER ==="
    )

    realization_counts = (
        logical
        .groupby(
            "container_id"
        )[
            "realization_id"
        ]
        .nunique()
    )

    print(
        realization_counts
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Uniqueness of logical coordinates
    # --------------------------------------------------

    logical_coordinate_duplicates = (
        logical.duplicated(
            subset=[
                "parent_source_id",
                "condition_id",
                "realization_id",
            ]
        ).sum()
    )

    print()
    print(
        "Duplicate logical coordinates:",
        logical_coordinate_duplicates
    )

    # --------------------------------------------------
    # Array indices
    # --------------------------------------------------

    array_index_valid = (
        logical
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

    print()
    print(
        "=== ARRAY INDEX VALIDATION ==="
    )

    print(
        array_index_valid
        .value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------
    # Physical files
    # --------------------------------------------------

    physical_files = list(
        SPECKLE_DIR.rglob(
            "*.npz"
        )
    )

    total_bytes = sum(
        file.stat().st_size
        for file
        in physical_files
    )

    print()
    print(
        "=== PHYSICAL STORAGE ==="
    )

    print(
        "NPZ files:",
        len(
            physical_files
        )
    )

    print(
        "Total GB:",
        round(
            total_bytes
            / (1024 ** 3),
            3,
        )
    )

    if physical_files:

        print(
            "Average KB/container:",
            round(
                (
                    total_bytes
                    / len(
                        physical_files
                    )
                )
                / 1024,
                3,
            )
        )

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        "=== RUNTIME ==="
    )

    print(
        "Seconds:",
        round(
            elapsed,
            2,
        )
    )

    print(
        "Minutes:",
        round(
            elapsed / 60,
            2,
        )
    )

    print(
        "Hours:",
        round(
            elapsed / 3600,
            3,
        )
    )

    print()
    print(
        "Container manifest:",
        CONTAINER_MANIFEST_PATH
    )

    print(
        "Logical manifest:",
        LOGICAL_MANIFEST_PATH
    )


if __name__ == "__main__":
    main()