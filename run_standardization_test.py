from pathlib import Path

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    RAW_DIR,
    METADATA_DIR,
)

from src.mstar_pipeline.standardize import (
    standardize_observation,
)


def main():

    manifest_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    manifest = pd.read_parquet(
        manifest_path
    )

    observations = manifest[
        manifest["data_role"]
        == "observation"
    ].copy()

    if observations.empty:
        raise RuntimeError(
            "No observations found."
        )

    # ------------------------------------------
    # Testar somente uma observação
    # ------------------------------------------

    row = observations.iloc[0]

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
        "=== SOURCE ==="
    )

    print(
        "Source ID:",
        source_id
    )

    print(
        "Relative path:",
        relative_path
    )

    print(
        "Source SHA-256:",
        source_hash
    )

    # ------------------------------------------
    # Standardizar
    # ------------------------------------------

    result = standardize_observation(
        source_path=source_path,
        source_id=source_id,
        source_hash_sha256=source_hash,
    )

    print()
    print(
        "=== STANDARDIZATION RESULT ==="
    )

    for key, value in vars(
        result
    ).items():

        print(
            f"{key}: {value}"
        )

    # ------------------------------------------
    # Inspecionar o NPZ produzido
    # ------------------------------------------

    print()
    print(
        "=== RELOAD INSPECTION ==="
    )

    with np.load(
        result.standardized_path,
        allow_pickle=False,
    ) as data:

        print(
            "Arrays:",
            data.files
        )

        magnitude = data[
            "magnitude"
        ]

        phase = data[
            "phase"
        ]

        print(
            "Magnitude shape:",
            magnitude.shape
        )

        print(
            "Magnitude dtype:",
            magnitude.dtype
        )

        print(
            "Magnitude min:",
            magnitude.min()
        )

        print(
            "Magnitude max:",
            magnitude.max()
        )

        print(
            "Phase shape:",
            phase.shape
        )

        print(
            "Phase dtype:",
            phase.dtype
        )

        print(
            "Phase min:",
            phase.min()
        )

        print(
            "Phase max:",
            phase.max()
        )


if __name__ == "__main__":
    main()