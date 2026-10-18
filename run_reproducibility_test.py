from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    RAW_DIR,
    METADATA_DIR,
    SOURCE_ENDIAN,
    MAGNITUDE_PLANE_INDEX,
    PHASE_PLANE_INDEX,
)

from src.mstar_pipeline.hashing import sha256_file

from src.mstar_pipeline.signal import (
    validate_exact_layout,
    read_float32_plane,
)


def select_test_sample(manifest):

    observations = manifest[
        manifest["data_role"]
        == "observation"
    ].copy()

    observations = observations.sort_values(
        "source_id"
    )

    sample_128 = (
        observations[
            (observations["rows"] == 128)
            & (observations["columns"] == 128)
        ]
        .head(10)
    )

    sample_54 = (
        observations[
            (observations["rows"] == 54)
            & (observations["columns"] == 54)
        ]
        .head(10)
    )

    return pd.concat(
        [sample_128, sample_54],
        ignore_index=True,
    )


def regenerate_arrays(source_path):

    layout = validate_exact_layout(
        source_path
    )

    magnitude = read_float32_plane(
        source_path,
        layout,
        plane_index=MAGNITUDE_PLANE_INDEX,
        endian=SOURCE_ENDIAN,
    )

    phase = read_float32_plane(
        source_path,
        layout,
        plane_index=PHASE_PLANE_INDEX,
        endian=SOURCE_ENDIAN,
    )

    magnitude = np.asarray(
        magnitude,
        dtype=np.float32,
    )

    phase = np.asarray(
        phase,
        dtype=np.float32,
    )

    return magnitude, phase


def main():

    enriched_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    standardized_manifest_path = (
        METADATA_DIR
        / "standardization_manifest.parquet"
    )

    enriched = pd.read_parquet(
        enriched_path
    )

    standardized = pd.read_parquet(
        standardized_manifest_path
    )

    # ------------------------------------------
    # Incorporar dimensões para seleção
    # ------------------------------------------

    sample_source = enriched.merge(
        standardized[
            [
                "source_id",
                "rows",
                "columns",
            ]
        ],
        on="source_id",
        how="left",
    )

    sample = (
        sample_source[
            sample_source["data_role"]
            == "observation"
        ]
        .sort_values("source_id")
        .reset_index(drop=True)
    )

    results = []

    print(
        "Observations under reproducibility test:",
        len(sample)
    )

    for position, row in sample.iterrows():

        source_id = row[
            "source_id"
        ]

        relative_path = row[
            "relative_path"
        ]

        source_path = (
            RAW_DIR
            / relative_path
        )

        std_row = standardized[
            standardized["source_id"]
            == source_id
        ].iloc[0]

        standardized_path = Path(
            std_row["standardized_path"]
        )

        print(
            f"[{position + 1}/{len(sample)}] "
            f"{source_id}"
        )

        # --------------------------------------
        # 1. RAW integrity
        # --------------------------------------

        current_source_hash = sha256_file(
            source_path
        )

        source_hash_equal = (
            current_source_hash
            == row["source_hash_sha256"]
        )

        # --------------------------------------
        # 2. Regenerar arrays a partir do RAW
        # --------------------------------------

        regenerated_magnitude, regenerated_phase = (
            regenerate_arrays(
                source_path
            )
        )

        # --------------------------------------
        # 3. Ler Standardized Layer existente
        # --------------------------------------

        with np.load(
            standardized_path,
            allow_pickle=False,
        ) as existing:

            existing_magnitude = (
                existing["magnitude"]
            )

            existing_phase = (
                existing["phase"]
            )

        magnitude_equal = np.array_equal(
            regenerated_magnitude,
            existing_magnitude,
            equal_nan=True,
        )

        phase_equal = np.array_equal(
            regenerated_phase,
            existing_phase,
            equal_nan=True,
        )

        # --------------------------------------
        # 4. Regenerar NPZ temporariamente
        # --------------------------------------

        with tempfile.TemporaryDirectory() as tmp:

            temporary_path = (
                Path(tmp)
                / f"{source_id}_std.npz"
            )

            np.savez(
                temporary_path,
                magnitude=regenerated_magnitude,
                phase=regenerated_phase,
            )

            regenerated_hash = sha256_file(
                temporary_path
            )

        artifact_hash_equal = (
            regenerated_hash
            == std_row[
                "standardized_hash_sha256"
            ]
        )

        results.append(
            {
                "source_id":
                    source_id,

                "rows":
                    std_row["rows"],

                "columns":
                    std_row["columns"],

                "source_hash_equal":
                    source_hash_equal,

                "magnitude_equal":
                    magnitude_equal,

                "phase_equal":
                    phase_equal,

                "artifact_hash_equal":
                    artifact_hash_equal,
            }
        )

    results = pd.DataFrame(
        results
    )

    print()
    print(
        "=== REPRODUCIBILITY TEST ==="
    )

    for column in [
        "source_hash_equal",
        "magnitude_equal",
        "phase_equal",
        "artifact_hash_equal",
    ]:

        print()
        print(column)

        print(
            results[column]
            .value_counts(
                dropna=False
            )
        )

    output_path = (
        METADATA_DIR
        / "reproducibility_validation.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()