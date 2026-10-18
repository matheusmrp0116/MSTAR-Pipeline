from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
)

from src.mstar_pipeline.complex_signal import (
    magnitude_phase_to_complex,
)

from src.mstar_pipeline.scattering import (
    generate_coherent_scatterer_field,
    generate_fully_developed_speckle,
)


# ==================================================
# PILOT CONFIGURATION
# ==================================================

NU = 1.0

N_VALUES = (
    1,
    2,
    5,
)

REALIZATIONS = 30

# Maximum number of observations selected from
# each image dimension found in the catalog.
SAMPLE_PER_DIMENSION = 10

BASE_SEED = 20260929


# ==================================================
# DETERMINISTIC SEED
# ==================================================

def deterministic_seed(
    source_id: str,
    condition_id: str,
    realization_id: int,
) -> int:
    """
    Generate a deterministic uint32 seed from
    source, condition, realization and base seed.

    The result does not depend on processing order.
    """

    key = (
        f"{BASE_SEED}|"
        f"{source_id}|"
        f"{condition_id}|"
        f"{realization_id}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


# ==================================================
# PILOT SAMPLE
# ==================================================

def select_pilot_sample(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministically select up to SAMPLE_PER_DIMENSION
    observations from every image dimension available
    in the dataset catalog.

    No predefined image dimensions are assumed.

    For each distinct (rows, columns) combination:
        - observations are sorted by source_id;
        - up to SAMPLE_PER_DIMENSION observations
          are selected;
        - if fewer observations are available, all
          available observations are selected.

    This makes the pilot independent of the dimensions
    present in a particular dataset.
    """

    required_columns = {
        "source_id",
        "rows",
        "columns",
        "standardized_path",
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

        subset = (
            subset
            .sort_values("source_id")
            .head(SAMPLE_PER_DIMENSION)
        )

        if not subset.empty:
            selected.append(
                subset
            )

    if not selected:
        raise ValueError(
            "No observations could be selected "
            "for the integration pilot."
        )

    pilot = pd.concat(
        selected,
        ignore_index=True,
    )

    return (
        pilot
        .sort_values(
            [
                "rows",
                "columns",
                "source_id",
            ]
        )
        .reset_index(drop=True)
    )


# ==================================================
# SPECKLE GENERATION
# ==================================================

def generate_speckle(
    shape: tuple[int, int],
    condition_id: str,
    n_scatterers: int | None,
    seed: int,
) -> np.ndarray:
    """
    Generate one speckle realization.
    """

    rng = np.random.default_rng(
        seed
    )

    if condition_id == "FDS":

        return (
            generate_fully_developed_speckle(
                size=shape,
                rng=rng,
            )
        )

    if n_scatterers is None:
        raise ValueError(
            "n_scatterers cannot be None "
            "for finite-scatterer conditions."
        )

    return (
        generate_coherent_scatterer_field(
            size=shape,
            n_scatterers=n_scatterers,
            nu=NU,
            rng=rng,
        )
    )


# ==================================================
# CONDITION VALIDATION
# ==================================================

def validate_condition(
    source_id: str,
    baseline_complex: np.ndarray,
    condition_id: str,
    n_scatterers: int | None,
) -> dict:
    """
    Apply multiple independent speckle realizations
    to one baseline observation and evaluate
    preservation of expected intensity.
    """

    baseline_complex = np.asarray(
        baseline_complex,
        dtype=np.complex64,
    )

    baseline_intensity = (
        np.abs(
            baseline_complex
        ).astype(np.float64)
        ** 2
    )

    shape = baseline_complex.shape

    positive_mask = (
        baseline_intensity > 0.0
    )

    zero_mask = (
        ~positive_mask
    )

    # Accumulate derived intensity across realizations.
    intensity_sum = np.zeros(
        shape,
        dtype=np.float64,
    )

    # Global power ratios for each realization.
    realization_power_ratios = []

    # Reproducibility check for first realization.
    first_seed = None
    first_speckle = None

    for realization_id in range(
        1,
        REALIZATIONS + 1,
    ):

        seed = deterministic_seed(
            source_id=source_id,
            condition_id=condition_id,
            realization_id=realization_id,
        )

        speckle = generate_speckle(
            shape=shape,
            condition_id=condition_id,
            n_scatterers=n_scatterers,
            seed=seed,
        )

        derived_complex = (
            baseline_complex
            * speckle
        )

        derived_intensity = (
            np.abs(
                derived_complex
            ).astype(np.float64)
            ** 2
        )

        intensity_sum += (
            derived_intensity
        )

        baseline_total_power = float(
            np.sum(
                baseline_intensity
            )
        )

        derived_total_power = float(
            np.sum(
                derived_intensity
            )
        )

        if baseline_total_power > 0.0:

            power_ratio = (
                derived_total_power
                / baseline_total_power
            )

        else:

            power_ratio = np.nan

        realization_power_ratios.append(
            power_ratio
        )

        if realization_id == 1:

            first_seed = seed

            first_speckle = (
                speckle.copy()
            )

    # --------------------------------------------------
    # Mean intensity across realizations
    # --------------------------------------------------

    mean_derived_intensity = (
        intensity_sum
        / REALIZATIONS
    )

    # --------------------------------------------------
    # Pixelwise expected-intensity preservation
    #
    # Evaluate only where baseline intensity > 0.
    # --------------------------------------------------

    if positive_mask.any():

        baseline_positive = (
            baseline_intensity[
                positive_mask
            ]
        )

        derived_positive = (
            mean_derived_intensity[
                positive_mask
            ]
        )

        absolute_error = np.abs(
            derived_positive
            - baseline_positive
        )

        relative_error = (
            absolute_error
            / baseline_positive
        )

        mean_relative_pixel_error = float(
            np.mean(
                relative_error
            )
        )

        median_relative_pixel_error = float(
            np.median(
                relative_error
            )
        )

        p95_relative_pixel_error = float(
            np.quantile(
                relative_error,
                0.95,
            )
        )

        max_relative_pixel_error = float(
            np.max(
                relative_error
            )
        )

    else:

        mean_relative_pixel_error = np.nan
        median_relative_pixel_error = np.nan
        p95_relative_pixel_error = np.nan
        max_relative_pixel_error = np.nan

    # --------------------------------------------------
    # Zero-magnitude preservation
    # --------------------------------------------------

    if zero_mask.any():

        zero_pixels_preserved = bool(
            np.all(
                mean_derived_intensity[
                    zero_mask
                ]
                == 0.0
            )
        )

    else:

        zero_pixels_preserved = True

    # --------------------------------------------------
    # Reproducibility:
    # regenerate realization 1 from same seed
    # --------------------------------------------------

    repeated_speckle = generate_speckle(
        shape=shape,
        condition_id=condition_id,
        n_scatterers=n_scatterers,
        seed=first_seed,
    )

    deterministic_reproduction = bool(
        np.array_equal(
            first_speckle,
            repeated_speckle,
        )
    )

    # --------------------------------------------------
    # Global power statistics
    # --------------------------------------------------

    power_ratios = np.asarray(
        realization_power_ratios,
        dtype=np.float64,
    )

    mean_power_ratio = float(
        np.nanmean(
            power_ratios
        )
    )

    std_power_ratio = float(
        np.nanstd(
            power_ratios
        )
    )

    min_power_ratio = float(
        np.nanmin(
            power_ratios
        )
    )

    max_power_ratio = float(
        np.nanmax(
            power_ratios
        )
    )

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    return {
        "source_id":
            source_id,

        "rows":
            int(shape[0]),

        "columns":
            int(shape[1]),

        "condition_id":
            condition_id,

        "n_scatterers":
            (
                n_scatterers
                if n_scatterers is not None
                else np.nan
            ),

        "nu":
            (
                NU
                if n_scatterers is not None
                else np.nan
            ),

        "realizations":
            REALIZATIONS,

        "positive_pixels":
            int(
                positive_mask.sum()
            ),

        "zero_pixels":
            int(
                zero_mask.sum()
            ),

        "mean_power_ratio":
            mean_power_ratio,

        "std_power_ratio":
            std_power_ratio,

        "min_power_ratio":
            min_power_ratio,

        "max_power_ratio":
            max_power_ratio,

        "mean_relative_pixel_error":
            mean_relative_pixel_error,

        "median_relative_pixel_error":
            median_relative_pixel_error,

        "p95_relative_pixel_error":
            p95_relative_pixel_error,

        "max_relative_pixel_error":
            max_relative_pixel_error,

        "zero_pixels_preserved":
            zero_pixels_preserved,

        "deterministic_reproduction":
            deterministic_reproduction,
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "=== MSTAR SPECKLE INTEGRATION PILOT ==="
    )

    print(
        "nu:",
        NU
    )

    print(
        "realizations per condition:",
        REALIZATIONS
    )

    print(
        "maximum observations per dimension:",
        SAMPLE_PER_DIMENSION
    )

    print()

    # --------------------------------------------------
    # Load catalog
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

    # --------------------------------------------------
    # Select deterministic pilot sample
    # --------------------------------------------------

    pilot = select_pilot_sample(
        catalog
    )

    dimensions_detected = (
        pilot[
            ["rows", "columns"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Pilot observations:",
        len(pilot)
    )

    print(
        "Dimensions detected in pilot:",
        dimensions_detected
    )

    print()

    print(
        "Pilot dimensions:"
    )

    print(
        pilot[
            ["rows", "columns"]
        ]
        .value_counts()
        .sort_index()
    )

    print()

    # --------------------------------------------------
    # Conditions
    # --------------------------------------------------

    conditions = [
        (
            "K_N01",
            1,
        ),
        (
            "K_N02",
            2,
        ),
        (
            "K_N05",
            5,
        ),
        (
            "FDS",
            None,
        ),
    ]

    records = []

    total_jobs = (
        len(pilot)
        * len(conditions)
    )

    job = 0

    # --------------------------------------------------
    # Process observations
    # --------------------------------------------------

    for _, row in pilot.iterrows():

        source_id = row[
            "source_id"
        ]

        standardized_path = Path(
            row[
                "standardized_path"
            ]
        )

        if not standardized_path.exists():
            raise FileNotFoundError(
                f"Standardized file not found: "
                f"{standardized_path}"
            )

        # ----------------------------------------------
        # Load standardized magnitude + phase
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Reconstruct complex baseline
        # ----------------------------------------------

        baseline_complex = (
            magnitude_phase_to_complex(
                magnitude,
                phase,
            )
        )

        # ----------------------------------------------
        # Conditions
        # ----------------------------------------------

        for (
            condition_id,
            n_scatterers,
        ) in conditions:

            job += 1

            print(
                f"[{job}/{total_jobs}] "
                f"{source_id} "
                f"{condition_id}"
            )

            try:

                result = validate_condition(
                    source_id=
                        source_id,

                    baseline_complex=
                        baseline_complex,

                    condition_id=
                        condition_id,

                    n_scatterers=
                        n_scatterers,
                )

                result[
                    "status"
                ] = "success"

            except Exception as exc:

                result = {
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

                    "condition_id":
                        condition_id,

                    "n_scatterers":
                        (
                            n_scatterers
                            if n_scatterers
                            is not None
                            else np.nan
                        ),

                    "status":
                        "error",

                    "error_type":
                        type(exc).__name__,

                    "error_message":
                        str(exc),
                }

            records.append(
                result
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = pd.DataFrame(
        records
    )

    output_path = (
        METADATA_DIR
        / "mstar_speckle_integration_pilot.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # Success / errors
    # --------------------------------------------------

    successful = results[
        results["status"]
        == "success"
    ].copy()

    errors = results[
        results["status"]
        == "error"
    ].copy()

    print()

    print(
        "=== PILOT SUMMARY ==="
    )

    print(
        "Jobs:",
        len(results)
    )

    print(
        "Success:",
        len(successful)
    )

    print(
        "Errors:",
        len(errors)
    )

    if successful.empty:

        print()

        print(
            "No successful pilot results."
        )

        print(
            f"Output: {output_path}"
        )

        return

    # --------------------------------------------------
    # Global power preservation
    # --------------------------------------------------

    print()

    print(
        "=== POWER PRESERVATION ==="
    )

    power_summary = (
        successful
        .groupby(
            "condition_id"
        )
        .agg(
            observations=(
                "source_id",
                "count",
            ),

            mean_power_ratio=(
                "mean_power_ratio",
                "mean",
            ),

            std_between_observations=(
                "mean_power_ratio",
                "std",
            ),

            mean_realization_std=(
                "std_power_ratio",
                "mean",
            ),

            minimum_observed_ratio=(
                "min_power_ratio",
                "min",
            ),

            maximum_observed_ratio=(
                "max_power_ratio",
                "max",
            ),
        )
    )

    print(
        power_summary
    )

    # --------------------------------------------------
    # Pixelwise expected-intensity preservation
    # --------------------------------------------------

    print()

    print(
        "=== PIXELWISE EXPECTED INTENSITY ==="
    )

    pixel_summary = (
        successful
        .groupby(
            "condition_id"
        )
        .agg(
            mean_relative_error=(
                "mean_relative_pixel_error",
                "mean",
            ),

            median_relative_error=(
                "median_relative_pixel_error",
                "median",
            ),

            mean_p95_relative_error=(
                "p95_relative_pixel_error",
                "mean",
            ),

            maximum_relative_error=(
                "max_relative_pixel_error",
                "max",
            ),
        )
    )

    print(
        pixel_summary
    )

    # --------------------------------------------------
    # Zero pixels
    # --------------------------------------------------

    print()

    print(
        "=== ZERO PIXEL PRESERVATION ==="
    )

    print(
        successful[
            "zero_pixels_preserved"
        ].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------
    # Deterministic reproducibility
    # --------------------------------------------------

    print()

    print(
        "=== DETERMINISTIC REPRODUCTION ==="
    )

    print(
        successful[
            "deterministic_reproduction"
        ].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    if not errors.empty:

        error_path = (
            METADATA_DIR
            / "mstar_speckle_integration_pilot_errors.csv"
        )

        errors.to_csv(
            error_path,
            index=False,
        )

        print()

        print(
            f"Error report: {error_path}"
        )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print()

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()