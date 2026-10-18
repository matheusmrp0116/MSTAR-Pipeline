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
# CONFIGURATION
# ==================================================

NU = 1.0

REALIZATION_LEVELS = (
    10,
    30,
    100,
    300,
    1000,
)

MAX_REALIZATIONS = max(
    REALIZATION_LEVELS
)

BASE_SEED = 20260929

CONDITIONS = (
    ("K_N01", 1),
    ("K_N02", 2),
    ("K_N05", 5),
    ("FDS", None),
)


# ==================================================
# DETERMINISTIC SEED
# ==================================================

def deterministic_seed(
    source_id: str,
    condition_id: str,
    realization_id: int,
) -> int:
    """
    Generate a deterministic uint32 seed.

    The same source, condition and realization
    always generate the same stochastic field.
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
# VALIDATION OBSERVATION SELECTION
# ==================================================

def select_validation_observations(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministically select one observation from every
    image dimension available in the dataset catalog.

    No predefined image dimensions are assumed.

    For each distinct (rows, columns) combination, the
    observation with the first source_id in deterministic
    alphabetical order is selected.
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

        observation = (
            subset
            .sort_values("source_id")
            .head(1)
        )

        if not observation.empty:
            selected.append(
                observation
            )

    if not selected:
        raise ValueError(
            "No observations could be selected "
            "for convergence validation."
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
# SPECKLE GENERATION
# ==================================================

def generate_speckle(
    shape: tuple[int, int],
    condition_id: str,
    n_scatterers: int | None,
    seed: int,
) -> np.ndarray:
    """
    Generate one deterministic speckle realization
    for the requested condition.
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
# ERROR METRICS
# ==================================================

def compute_metrics(
    baseline_intensity: np.ndarray,
    mean_derived_intensity: np.ndarray,
    realization_count: int,
) -> dict:
    """
    Compare Monte Carlo mean intensity against
    the original baseline intensity.
    """

    positive_mask = (
        baseline_intensity > 0.0
    )

    zero_mask = (
        ~positive_mask
    )

    if not positive_mask.any():
        raise ValueError(
            "Baseline contains no positive "
            "intensity pixels."
        )

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

    # --------------------------------------------------
    # Global power
    # --------------------------------------------------

    baseline_power = float(
        np.sum(
            baseline_intensity
        )
    )

    derived_power = float(
        np.sum(
            mean_derived_intensity
        )
    )

    power_ratio = (
        derived_power
        / baseline_power
    )

    # --------------------------------------------------
    # Zero-pixel preservation
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
    # Metrics
    # --------------------------------------------------

    return {
        "realizations":
            realization_count,

        "sqrt_realizations":
            float(
                np.sqrt(
                    realization_count
                )
            ),

        "inverse_sqrt_realizations":
            float(
                1.0
                / np.sqrt(
                    realization_count
                )
            ),

        "power_ratio":
            power_ratio,

        "power_relative_error":
            abs(
                power_ratio
                - 1.0
            ),

        "mean_relative_pixel_error":
            float(
                np.mean(
                    relative_error
                )
            ),

        "median_relative_pixel_error":
            float(
                np.median(
                    relative_error
                )
            ),

        "p95_relative_pixel_error":
            float(
                np.quantile(
                    relative_error,
                    0.95,
                )
            ),

        "max_relative_pixel_error":
            float(
                np.max(
                    relative_error
                )
            ),

        "mean_absolute_pixel_error":
            float(
                np.mean(
                    absolute_error
                )
            ),

        "positive_pixels":
            int(
                positive_mask.sum()
            ),

        "zero_pixels":
            int(
                zero_mask.sum()
            ),

        "zero_pixels_preserved":
            zero_pixels_preserved,
    }


# ==================================================
# CONDITION CONVERGENCE
# ==================================================

def evaluate_condition(
    source_id: str,
    baseline_complex: np.ndarray,
    condition_id: str,
    n_scatterers: int | None,
) -> list[dict]:
    """
    Generate realizations sequentially and evaluate
    convergence at R = 10, 30, 100, 300 and 1000.

    The same first R realizations are used at every
    checkpoint, avoiding independent Monte Carlo
    experiments for each R.
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

    cumulative_intensity = np.zeros(
        shape,
        dtype=np.float64,
    )

    records = []

    checkpoints = set(
        REALIZATION_LEVELS
    )

    for realization_id in range(
        1,
        MAX_REALIZATIONS + 1,
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

        cumulative_intensity += (
            derived_intensity
        )

        if realization_id in checkpoints:

            mean_derived_intensity = (
                cumulative_intensity
                / realization_id
            )

            metrics = compute_metrics(
                baseline_intensity=
                    baseline_intensity,

                mean_derived_intensity=
                    mean_derived_intensity,

                realization_count=
                    realization_id,
            )

            metrics.update(
                {
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
                            if n_scatterers
                            is not None
                            else np.nan
                        ),

                    "nu":
                        (
                            NU
                            if n_scatterers
                            is not None
                            else np.nan
                        ),
                }
            )

            records.append(
                metrics
            )

    return records


# ==================================================
# EMPIRICAL CONVERGENCE RATE
# ==================================================

def estimate_convergence_rate(
    group: pd.DataFrame,
    metric: str,
) -> float:
    """
    Estimate slope beta in:

        error ~ R^beta

    using log-log linear regression.

    Monte Carlo convergence suggests:

        beta approximately -0.5
    """

    x = np.log(
        group[
            "realizations"
        ].to_numpy(
            dtype=np.float64
        )
    )

    y_raw = group[
        metric
    ].to_numpy(
        dtype=np.float64
    )

    valid = (
        np.isfinite(y_raw)
        & (y_raw > 0.0)
    )

    if valid.sum() < 2:
        return np.nan

    y = np.log(
        y_raw[
            valid
        ]
    )

    x = x[
        valid
    ]

    slope, _ = np.polyfit(
        x,
        y,
        deg=1,
    )

    return float(
        slope
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "=== SPECKLE MONTE CARLO "
        "CONVERGENCE VALIDATION ==="
    )

    print(
        "nu:",
        NU
    )

    print(
        "realization levels:",
        REALIZATION_LEVELS
    )

    print(
        "maximum realizations:",
        MAX_REALIZATIONS
    )

    print()

    # --------------------------------------------------
    # Load dataset catalog
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
    # Select one observation per available dimension
    # --------------------------------------------------

    observations = (
        select_validation_observations(
            catalog
        )
    )

    print(
        "Dimensions detected:",
        len(observations)
    )

    print()

    print(
        "Selected observations:"
    )

    print(
        observations[
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

    records = []

    total_jobs = (
        len(observations)
        * len(CONDITIONS)
    )

    job = 0

    # --------------------------------------------------
    # Process observations
    # --------------------------------------------------

    for _, row in observations.iterrows():

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
        # Load standardized data
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
        # Complex reconstruction
        # ----------------------------------------------

        baseline_complex = (
            magnitude_phase_to_complex(
                magnitude,
                phase,
            )
        )

        # ----------------------------------------------
        # Evaluate conditions
        # ----------------------------------------------

        for (
            condition_id,
            n_scatterers,
        ) in CONDITIONS:

            job += 1

            print(
                f"[{job}/{total_jobs}] "
                f"{source_id} "
                f"{row['rows']}x{row['columns']} "
                f"{condition_id}"
            )

            condition_records = (
                evaluate_condition(
                    source_id=
                        source_id,

                    baseline_complex=
                        baseline_complex,

                    condition_id=
                        condition_id,

                    n_scatterers=
                        n_scatterers,
                )
            )

            records.extend(
                condition_records
            )

    # ==================================================
    # RESULTS
    # ==================================================

    results = pd.DataFrame(
        records
    )

    if results.empty:
        raise RuntimeError(
            "Convergence validation produced no results."
        )

    results = (
        results
        .sort_values(
            [
                "rows",
                "columns",
                "source_id",
                "condition_id",
                "realizations",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_path = (
        METADATA_DIR
        / "speckle_convergence_validation.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    # ==================================================
    # SUMMARY 1
    # Pixelwise convergence
    # ==================================================

    print()

    print(
        "=== PIXELWISE CONVERGENCE ==="
    )

    pixel_table = results[
        [
            "source_id",
            "rows",
            "columns",
            "condition_id",
            "realizations",
            "mean_relative_pixel_error",
            "median_relative_pixel_error",
            "p95_relative_pixel_error",
        ]
    ]

    print(
        pixel_table.to_string(
            index=False
        )
    )

    # ==================================================
    # SUMMARY 2
    # Power convergence
    # ==================================================

    print()

    print(
        "=== POWER CONVERGENCE ==="
    )

    power_table = results[
        [
            "source_id",
            "rows",
            "columns",
            "condition_id",
            "realizations",
            "power_ratio",
            "power_relative_error",
        ]
    ]

    print(
        power_table.to_string(
            index=False
        )
    )

    # ==================================================
    # SUMMARY 3
    # Zero pixels
    # ==================================================

    print()

    print(
        "=== ZERO PIXEL PRESERVATION ==="
    )

    print(
        results[
            "zero_pixels_preserved"
        ].value_counts(
            dropna=False
        )
    )

    # ==================================================
    # SUMMARY 4
    # Convergence slopes
    # ==================================================

    slope_records = []

    for (
        source_id,
        rows,
        columns,
        condition_id,
    ), group in results.groupby(
        [
            "source_id",
            "rows",
            "columns",
            "condition_id",
        ]
    ):

        pixel_slope = (
            estimate_convergence_rate(
                group=group,
                metric=
                    "mean_relative_pixel_error",
            )
        )

        power_slope = (
            estimate_convergence_rate(
                group=group,
                metric=
                    "power_relative_error",
            )
        )

        slope_records.append(
            {
                "source_id":
                    source_id,

                "rows":
                    rows,

                "columns":
                    columns,

                "condition_id":
                    condition_id,

                "pixel_error_slope":
                    pixel_slope,

                "power_error_slope":
                    power_slope,
            }
        )

    slopes = pd.DataFrame(
        slope_records
    )

    slope_path = (
        METADATA_DIR
        / "speckle_convergence_slopes.parquet"
    )

    slopes.to_parquet(
        slope_path,
        index=False,
    )

    print()

    print(
        "=== EMPIRICAL CONVERGENCE SLOPES ==="
    )

    print(
        slopes[
            [
                "source_id",
                "rows",
                "columns",
                "condition_id",
                "pixel_error_slope",
                "power_error_slope",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Reference Monte Carlo slope: -0.5"
    )

    # ==================================================
    # OUTPUT
    # ==================================================

    print()

    print(
        f"Validation output: {output_path}"
    )

    print(
        f"Slope output: {slope_path}"
    )


if __name__ == "__main__":
    main()