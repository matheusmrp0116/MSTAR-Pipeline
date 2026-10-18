from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import METADATA_DIR


def build_catalog() -> pd.DataFrame:
    """
    Constrói o catálogo consolidado do dataset.

    Uma linha representa uma observação.

    Fontes:
    - manifest_enriched:
        identificação, origem e metadados Phoenix;
    - standardization_manifest:
        artefato padronizado e proveniência;
    - reproducibility_validation:
        resultados da validação de reprodução.
    """

    enriched_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    standardized_path = (
        METADATA_DIR
        / "standardization_manifest.parquet"
    )

    reproducibility_path = (
        METADATA_DIR
        / "reproducibility_validation.parquet"
    )

    enriched = pd.read_parquet(
        enriched_path
    )

    standardized = pd.read_parquet(
        standardized_path
    )

    reproducibility = pd.read_parquet(
        reproducibility_path
    )

    # ------------------------------------------
    # 1. Apenas observações válidas
    # ------------------------------------------

    observations = enriched[
        enriched["data_role"]
        == "observation"
    ].copy()

    # ------------------------------------------
    # 2. Verificar unicidade
    # ------------------------------------------

    for name, dataframe in [
        ("observations", observations),
        ("standardized", standardized),
        ("reproducibility", reproducibility),
    ]:

        if dataframe[
            "source_id"
        ].duplicated().any():

            raise ValueError(
                f"Duplicate source_id in {name}."
            )

    # ------------------------------------------
    # 3. Selecionar campos da Raw/Phoenix Layer
    # ------------------------------------------

    observation_columns = [
        "source_id",
        "original_file",
        "relative_path",
        "file_size_bytes",
        "source_hash_sha256",
        "data_role",
        "header_parse_status",
        "header_valid",
        "header_issues",
        "rows",
        "columns",
    ]

    # Mantém somente colunas realmente existentes.
    observation_columns = [
        column
        for column in observation_columns
        if column in observations.columns
    ]

    obs = observations[
        observation_columns
    ].copy()

    # ------------------------------------------
    # 4. Campos da Standardized Layer
    #
    # rows/columns e pipeline_version passam
    # a vir desta tabela.
    # ------------------------------------------

    standardized_columns = [
        "source_id",
        "standardized_id",
        "standardized_path",
        "standardized_hash_sha256",
        "source_endian",
        "standardized_dtype",
        "magnitude_plane_index",
        "phase_plane_index",
        "pipeline_version",
        "standardized_format_version",
        "validation_equal_magnitude",
        "validation_equal_phase",
        "status",
    ]

    # Se rows/columns não vieram do enriched,
    # trazemos da padronização.
    if "rows" not in obs.columns:
        standardized_columns.append(
            "rows"
        )

    if "columns" not in obs.columns:
        standardized_columns.append(
            "columns"
        )

    std = standardized[
        standardized_columns
    ].copy()

    std = std.rename(
        columns={
            "status":
                "standardization_status"
        }
    )

    # ------------------------------------------
    # 5. Resultados V4
    # ------------------------------------------

    reproducibility_columns = [
        "source_id",
        "source_hash_equal",
        "magnitude_equal",
        "phase_equal",
        "artifact_hash_equal",
    ]

    repro = reproducibility[
        reproducibility_columns
    ].copy()

    # ------------------------------------------
    # 6. Merge 1:1
    # ------------------------------------------

    catalog = obs.merge(
        std,
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    catalog = catalog.merge(
        repro,
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------
    # 7. Ordenação determinística
    # ------------------------------------------

    catalog = (
        catalog
        .sort_values(
            "source_id"
        )
        .reset_index(
            drop=True
        )
    )

    return catalog


def save_catalog(
    catalog: pd.DataFrame,
) -> Path:

    output_path = (
        METADATA_DIR
        / "dataset_catalog.parquet"
    )

    catalog.to_parquet(
        output_path,
        index=False,
    )

    return output_path