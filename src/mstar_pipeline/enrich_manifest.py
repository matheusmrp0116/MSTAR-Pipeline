from pathlib import Path

import pandas as pd

from .config import (
    RAW_DIR,
    METADATA_DIR
)

from .phoenix import (
    inspect_phoenix_file
)

AUXILIARY_EXTENSIONS = {
    ".gif",
    ".txt",
    ".htm",
    ".html",
}


def classify_source_file(path: Path) -> str:
    """
    Classifica o papel do arquivo na Raw Layer.

    Retorna:
        "auxiliary" para arquivos auxiliares conhecidos;
        "observation_candidate" para arquivos que devem
        ser submetidos ao parser Phoenix.
    """

    suffix = path.suffix.lower()

    if suffix in AUXILIARY_EXTENSIONS:
        return "auxiliary"

    return "observation_candidate"

def enrich_manifest():

    manifest_path = (
        METADATA_DIR
        / "manifest.parquet"
    )

    manifest = pd.read_parquet(
        manifest_path
    )

    enriched_records = []
    errors = []

    total = len(manifest)

    for position, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):

        relative_path = row[
            "relative_path"
        ]

        source_path = (
            RAW_DIR
            / relative_path
        )

        print(
            f"[{position}/{total}] "
            f"{relative_path}"
        )

        base_record = row.to_dict()

        # ---------------------------------
        # 1. Classificação do arquivo
        # ---------------------------------

        data_role = classify_source_file(
            source_path
        )

        base_record[
            "data_role"
        ] = data_role

        # ---------------------------------
        # 2. Arquivo auxiliar
        # ---------------------------------

        if data_role == "auxiliary":

            base_record[
                "phoenix_candidate"
            ] = False

            base_record[
                "header_parse_status"
            ] = "not_applicable"

            base_record[
                "header_valid"
            ] = False

            base_record[
                "header_issues"
            ] = None

            enriched_records.append(
                base_record
            )

            continue

        # ---------------------------------
        # 3. Candidato Phoenix
        # ---------------------------------

        base_record[
            "phoenix_candidate"
        ] = True

        try:

            phoenix_data = (
                inspect_phoenix_file(
                    source_path
                )
            )

            base_record.update(
                phoenix_data
            )

            base_record[
                "header_parse_status"
            ] = "success"

            # Se o parsing foi bem-sucedido,
            # confirmamos que é observação.
            base_record[
                "data_role"
            ] = "observation"

        except Exception as exc:

            base_record[
                "header_parse_status"
            ] = "error"

            base_record[
                "header_valid"
            ] = False

            base_record[
                "header_issues"
            ] = str(exc)

            # Não classificamos como auxiliary,
            # pois era candidato Phoenix e falhou.
            base_record[
                "data_role"
            ] = "unresolved"

            errors.append(
                {
                    "source_id":
                        row["source_id"],

                    "relative_path":
                        relative_path,

                    "error_type":
                        type(exc).__name__,

                    "error_message":
                        str(exc),
                }
            )

        enriched_records.append(
            base_record
        )

    enriched = pd.DataFrame(
        enriched_records
    )

    return (
        enriched,
        pd.DataFrame(errors)
    )

def save_enriched_manifest(
    enriched,
    errors
):

    output_path = (
        METADATA_DIR
        / "manifest_enriched.parquet"
    )

    enriched.to_parquet(
        output_path,
        index=False
    )

    error_path = (
        METADATA_DIR
        / "phoenix_parse_errors.csv"
    )

    if not errors.empty:

        errors.to_csv(
            error_path,
            index=False
        )

    elif error_path.exists():

        # Remove relatório antigo para evitar
        # confusão com execuções anteriores.
        error_path.unlink()

    print()
    print(
        "Phoenix parsing completed."
    )

    print(
        f"Raw Layer records: "
        f"{len(enriched)}"
    )

    print(
        "Observations:",
        (
            enriched["data_role"]
            == "observation"
        ).sum()
    )

    print(
        "Auxiliary files:",
        (
            enriched["data_role"]
            == "auxiliary"
        ).sum()
    )

    print(
        "Unresolved files:",
        (
            enriched["data_role"]
            == "unresolved"
        ).sum()
    )

    print(
        "Valid headers:",
        enriched[
            "header_valid"
        ].fillna(False).sum()
    )

    print(
        "Parsing errors:",
        len(errors)
    )

    print(
        f"Output: {output_path}"
    )

def run_header_enrichment():

    enriched, errors = (
        enrich_manifest()
    )

    save_enriched_manifest(
        enriched,
        errors
    )

    return enriched, errors