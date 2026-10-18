import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
)

from src.mstar_pipeline.scattering import (
    empirical_normalized_moment,
    generate_coherent_scatterer_field,
    generate_fully_developed_speckle,
    intensity_from_complex,
    model_M,
    theoretical_fds_normalized_moment,
    theoretical_normalized_moment,
)


# ==================================================
# CALIBRATION CONFIGURATION
# ==================================================

# Reference value used in the initial calibration.
NU = 1.0

# Finite equivalent-scatterer conditions.
N_VALUES = (
    1,
    2,
    5,
)

# Normalized intensity moments to evaluate.
MOMENT_ORDERS = (
    1,
    2,
    3,
    4,
)

# Number of independent synthetic samples
# generated for each condition.
SAMPLE_SIZE = 1_000_000

# Fixed base seed for reproducibility.
BASE_SEED = 20260929


# ==================================================
# AUXILIARY FUNCTIONS
# ==================================================

def relative_error(
    empirical: float,
    theoretical: float,
) -> float:
    """
    Compute relative error between empirical
    and theoretical values.
    """

    if theoretical == 0.0:
        return np.nan

    return abs(
        empirical
        - theoretical
    ) / abs(
        theoretical
    )


def validate_condition(
    condition_id: str,
    signal: np.ndarray,
    n_scatterers: int | None,
    nu: float | None,
) -> list[dict]:
    """
    Compute empirical intensity statistics
    and compare normalized moments against
    their theoretical values.
    """

    # --------------------------------------------------
    # Intensity
    # --------------------------------------------------

    intensity = (
        intensity_from_complex(
            signal
        )
    )

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    mean_intensity = float(
        np.mean(
            intensity
        )
    )

    variance_intensity = float(
        np.var(
            intensity
        )
    )

    std_intensity = float(
        np.std(
            intensity
        )
    )

    coefficient_of_variation = (
        std_intensity
        / mean_intensity
    )

    # --------------------------------------------------
    # Normalized moments
    # --------------------------------------------------

    records = []

    for order in MOMENT_ORDERS:

        empirical = (
            empirical_normalized_moment(
                intensity,
                order,
            )
        )

        # Fully developed speckle reference
        if n_scatterers is None:

            theoretical = (
                theoretical_fds_normalized_moment(
                    order
                )
            )

            M = np.nan

        # Finite coherent-scatterer model
        else:

            theoretical = (
                theoretical_normalized_moment(
                    order=order,
                    n_scatterers=n_scatterers,
                    nu=nu,
                )
            )

            M = model_M(
                n_scatterers,
                nu,
            )

        error = relative_error(
            empirical,
            theoretical,
        )

        records.append(
            {
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
                        nu
                        if nu is not None
                        else np.nan
                    ),

                "M":
                    M,

                "sample_size":
                    len(
                        intensity
                    ),

                "mean_intensity":
                    mean_intensity,

                "variance_intensity":
                    variance_intensity,

                "coefficient_of_variation":
                    coefficient_of_variation,

                "moment_order":
                    order,

                "empirical_normalized_moment":
                    empirical,

                "theoretical_normalized_moment":
                    theoretical,

                "relative_error":
                    error,
            }
        )

    return records


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "=== COHERENT SCATTERER "
        "MODEL VALIDATION ==="
    )

    print(
        "nu:",
        NU
    )

    print(
        "sample size:",
        SAMPLE_SIZE
    )

    print(
        "base seed:",
        BASE_SEED
    )

    print()

    records = []

    # ==================================================
    # FINITE-SCATTERER CONDITIONS
    # ==================================================

    for index, n_scatterers in enumerate(
        N_VALUES,
        start=1,
    ):

        condition_id = (
            f"K_N{n_scatterers:02d}"
        )

        seed = (
            BASE_SEED
            + index
        )

        rng = np.random.default_rng(
            seed
        )

        print(
            "Generating:",
            condition_id,
            "| seed:",
            seed,
        )

        signal = (
            generate_coherent_scatterer_field(
                size=SAMPLE_SIZE,
                n_scatterers=n_scatterers,
                nu=NU,
                rng=rng,
            )
        )

        condition_records = (
            validate_condition(
                condition_id=
                    condition_id,

                signal=
                    signal,

                n_scatterers=
                    n_scatterers,

                nu=
                    NU,
            )
        )

        records.extend(
            condition_records
        )

    # ==================================================
    # FULLY DEVELOPED SPECKLE REFERENCE
    # ==================================================

    fds_seed = (
        BASE_SEED
        + 100
    )

    rng = np.random.default_rng(
        fds_seed
    )

    print(
        "Generating: FDS",
        "| seed:",
        fds_seed,
    )

    signal = (
        generate_fully_developed_speckle(
            size=SAMPLE_SIZE,
            rng=rng,
        )
    )

    records.extend(
        validate_condition(
            condition_id="FDS",
            signal=signal,
            n_scatterers=None,
            nu=None,
        )
    )

    # ==================================================
    # CREATE RESULTS TABLE
    # ==================================================

    results = pd.DataFrame(
        records
    )

    # --------------------------------------------------
    # Save validation artifact
    # --------------------------------------------------

    output_path = (
        METADATA_DIR
        / "scattering_model_validation.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print()
    print(
        "=== MEAN INTENSITY ==="
    )

    mean_summary = (
        results[
            [
                "condition_id",
                "mean_intensity",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        mean_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Variance and coefficient of variation
    # --------------------------------------------------

    print()
    print(
        "=== INTENSITY VARIANCE ==="
    )

    variance_summary = (
        results[
            [
                "condition_id",
                "variance_intensity",
                "coefficient_of_variation",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        variance_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Normalized moments
    # --------------------------------------------------

    print()
    print(
        "=== NORMALIZED MOMENTS ==="
    )

    moment_summary = (
        results[
            [
                "condition_id",
                "moment_order",
                "empirical_normalized_moment",
                "theoretical_normalized_moment",
                "relative_error",
            ]
        ]
        .copy()
    )

    print(
        moment_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Maximum relative error by condition
    # --------------------------------------------------

    print()
    print(
        "=== MAXIMUM RELATIVE ERROR ==="
    )

    maximum_errors = (
        results
        .groupby(
            "condition_id"
        )[
            "relative_error"
        ]
        .max()
    )

    print(
        maximum_errors
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print()
    print(
        f"Output: {output_path}"
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()