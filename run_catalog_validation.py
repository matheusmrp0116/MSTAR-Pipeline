from pathlib import Path

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
)

from src.mstar_pipeline.catalog import (
    build_catalog,
    save_catalog,
)


def main():

    catalog = build_catalog()

    output_path = save_catalog(
        catalog
    )

    checks = {}

    # ------------------------------------------
    # Cardinalidade
    # ------------------------------------------

    # Os valores esperados são derivados do próprio
    # dataset (manifest_enriched), e não fixados no
    # código. Cada observação Phoenix reconhecida deve
    # corresponder a exatamente um registro do catálogo.

    enriched = pd.read_parquet(
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    expected_observations = enriched[
        enriched["data_role"]
        == "observation"
    ].copy()

    expected_record_count = len(
        expected_observations
    )

    checks["record_count"] = (
        len(catalog)
        == expected_record_count
    )

    checks["source_id_coverage"] = (
        set(catalog["source_id"])
        == set(
            expected_observations[
                "source_id"
            ]
        )
    )

    checks["unique_source_id"] = (
        catalog["source_id"].nunique()
        == len(catalog)
    )

    checks["unique_standardized_id"] = (
        catalog["standardized_id"].nunique()
        == len(catalog)
    )

    # ------------------------------------------
    # Campos essenciais
    # ------------------------------------------

    required_columns = [
        "source_id",
        "relative_path",
        "source_hash_sha256",
        "header_valid",
        "rows",
        "columns",
        "standardized_id",
        "standardized_path",
        "standardized_hash_sha256",
        "source_endian",
        "standardized_dtype",
        "pipeline_version",
        "standardized_format_version",
    ]

    checks["required_fields_complete"] = (
        catalog[
            required_columns
        ]
        .notna()
        .all()
        .all()
    )

    # ------------------------------------------
    # Estados esperados
    # ------------------------------------------

    checks["headers_valid"] = (
        catalog["header_valid"]
        .fillna(False)
        .all()
    )

    checks["standardization_success"] = (
        catalog["standardization_status"]
        .eq("success")
        .all()
    )

    checks["source_endian_big"] = (
        catalog["source_endian"]
        .eq("big")
        .all()
    )

    checks["standardized_float32"] = (
        catalog["standardized_dtype"]
        .eq("float32")
        .all()
    )

    # ------------------------------------------
    # Dimensões
    # ------------------------------------------

    # Nenhuma dimensão é pré-definida. As dimensões
    # esperadas de cada observação são obtidas dos
    # seus próprios metadados (Phoenix header) e
    # conferidas com:
    #   (a) as dimensões registradas no catálogo;
    #   (b) o shape real dos arrays padronizados.

    # Dimensões positivas e inteiras
    rows_numeric = pd.to_numeric(
        catalog["rows"],
        errors="coerce",
    )

    columns_numeric = pd.to_numeric(
        catalog["columns"],
        errors="coerce",
    )

    checks["dimensions_positive"] = bool(
        (
            rows_numeric.notna()
            & columns_numeric.notna()
            & (rows_numeric > 0)
            & (columns_numeric > 0)
            & (rows_numeric % 1 == 0)
            & (columns_numeric % 1 == 0)
        ).all()
    )

    # (a) Catálogo x Phoenix header
    header_dimensions = (
        expected_observations[
            [
                "source_id",
                "number_of_rows",
                "number_of_columns",
            ]
        ]
        if {
            "number_of_rows",
            "number_of_columns",
        } <= set(expected_observations.columns)
        else None
    )

    if header_dimensions is None:

        checks["dimensions_match_header"] = False

    else:

        dimension_check = catalog[
            ["source_id", "rows", "columns"]
        ].merge(
            header_dimensions,
            on="source_id",
            how="left",
            validate="one_to_one",
        )

        checks["dimensions_match_header"] = bool(
            (
                pd.to_numeric(
                    dimension_check["rows"],
                    errors="coerce",
                )
                == pd.to_numeric(
                    dimension_check["number_of_rows"],
                    errors="coerce",
                )
            ).all()
            and
            (
                pd.to_numeric(
                    dimension_check["columns"],
                    errors="coerce",
                )
                == pd.to_numeric(
                    dimension_check["number_of_columns"],
                    errors="coerce",
                )
            ).all()
        )

    # (b) Catálogo x shape real dos arrays
    shape_mismatches = []

    for _, row in catalog.iterrows():

        try:

            expected_shape = (
                int(row["rows"]),
                int(row["columns"]),
            )

            with np.load(
                Path(row["standardized_path"]),
                allow_pickle=False,
            ) as data:

                magnitude_shape = (
                    data["magnitude"].shape
                )

                phase_shape = (
                    data["phase"].shape
                )

            if (
                magnitude_shape != expected_shape
                or phase_shape != expected_shape
            ):
                shape_mismatches.append(
                    row["source_id"]
                )

        except Exception:

            shape_mismatches.append(
                row["source_id"]
            )

    checks["array_shapes_match_metadata"] = (
        len(shape_mismatches) == 0
    )

    # ------------------------------------------
    # V4 incorporada ao catálogo
    # ------------------------------------------

    for column in [
        "source_hash_equal",
        "magnitude_equal",
        "phase_equal",
        "artifact_hash_equal",
    ]:

        checks[column] = (
            catalog[column]
            .fillna(False)
            .all()
        )

    # ------------------------------------------
    # Resultado V5
    # ------------------------------------------

    validation = pd.DataFrame(
        [
            {
                "check": key,
                "passed": value,
            }
            for key, value
            in checks.items()
        ]
    )

    validation_path = (
        METADATA_DIR
        / "catalog_validation.parquet"
    )

    validation.to_parquet(
        validation_path,
        index=False,
    )

    print(
        "=== DATASET CATALOG ==="
    )

    print(
        "Records:",
        len(catalog)
    )

    print(
        "Expected records "
        "(observations in manifest_enriched):",
        expected_record_count
    )

    print(
        "Unique source IDs:",
        catalog["source_id"].nunique()
    )

    print(
        "Unique standardized IDs:",
        catalog[
            "standardized_id"
        ].nunique()
    )

    print()
    print(
        "=== DIMENSIONS ==="
    )

    print(
        catalog[
            ["rows", "columns"]
        ]
        .value_counts()
    )

    print()
    print(
        "=== V5 CATALOG VALIDATION ==="
    )

    print(
        validation.to_string(
            index=False
        )
    )

    print()

    all_passed = validation[
        "passed"
    ].all()

    print(
        "V5 passed:",
        all_passed
    )

    if shape_mismatches:

        print()

        print(
            "Observations with array shape "
            "mismatch:",
            len(shape_mismatches)
        )

        print(
            shape_mismatches[:20]
        )

    print()
    print(
        f"Catalog: {output_path}"
    )

    print(
        f"Validation: {validation_path}"
    )


if __name__ == "__main__":
    main()