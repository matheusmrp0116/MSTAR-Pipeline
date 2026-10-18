from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .config import (
    STANDARDIZED_DIR,
    SOURCE_ENDIAN,
    MAGNITUDE_PLANE_INDEX,
    PHASE_PLANE_INDEX,
    PIPELINE_VERSION,
    STANDARDIZED_FORMAT_VERSION,
)

from .hashing import sha256_file

from .signal import (
    validate_exact_layout,
    read_float32_plane,
)


@dataclass
class StandardizationResult:
    source_id: str
    source_path: str

    standardized_id: str
    standardized_path: str

    rows: int
    columns: int

    source_endian: str
    standardized_dtype: str

    magnitude_plane_index: int
    phase_plane_index: int

    source_hash_sha256: str
    standardized_hash_sha256: str

    pipeline_version: str
    standardized_format_version: str

    validation_equal_magnitude: bool
    validation_equal_phase: bool

    status: str


def standardize_observation(
    source_path: str | Path,
    source_id: str,
    source_hash_sha256: str,
) -> StandardizationResult:
    """
    Decodifica uma observação Phoenix validada e
    cria um artefato padronizado NumPy (.npz).

    A padronização:
    - preserva as dimensões originais;
    - preserva magnitude e phase;
    - converte os valores para np.float32 nativo;
    - não normaliza;
    - não redimensiona;
    - não aplica transformação log;
    - não converte para 8 bits.
    """

    source_path = Path(source_path)

    # ------------------------------------------
    # 1. Validar layout binário
    # ------------------------------------------

    layout = validate_exact_layout(
        source_path
    )

    if layout.plane_count != 2:
        raise ValueError(
            "Expected exactly two source planes, "
            f"found {layout.plane_count}."
        )

    # ------------------------------------------
    # 2. Decodificar magnitude
    # ------------------------------------------

    magnitude = read_float32_plane(
        source_path,
        layout,
        plane_index=MAGNITUDE_PLANE_INDEX,
        endian=SOURCE_ENDIAN,
    )

    # ------------------------------------------
    # 3. Decodificar phase
    # ------------------------------------------

    phase = read_float32_plane(
        source_path,
        layout,
        plane_index=PHASE_PLANE_INDEX,
        endian=SOURCE_ENDIAN,
    )

    # ------------------------------------------
    # 4. Converter explicitamente para
    #    float32 nativo
    # ------------------------------------------

    magnitude = np.asarray(
        magnitude,
        dtype=np.float32,
    )

    phase = np.asarray(
        phase,
        dtype=np.float32,
    )

    # ------------------------------------------
    # 5. Validações antes da escrita
    # ------------------------------------------

    expected_shape = (
        layout.rows,
        layout.columns,
    )

    if magnitude.shape != expected_shape:
        raise ValueError(
            "Unexpected magnitude shape: "
            f"{magnitude.shape}"
        )

    if phase.shape != expected_shape:
        raise ValueError(
            "Unexpected phase shape: "
            f"{phase.shape}"
        )

    if not np.isfinite(magnitude).all():
        raise ValueError(
            "Magnitude contains non-finite values."
        )

    if not np.isfinite(phase).all():
        raise ValueError(
            "Phase contains non-finite values."
        )

    # ------------------------------------------
    # 6. Criar identificador padronizado
    # ------------------------------------------

    standardized_id = (
        f"{source_id}_std"
    )

    STANDARDIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    standardized_path = (
        STANDARDIZED_DIR
        / f"{standardized_id}.npz"
    )

    # ------------------------------------------
    # 7. Salvar artefato
    # ------------------------------------------

    np.savez(
        standardized_path,
        magnitude=magnitude,
        phase=phase,
    )

    # ------------------------------------------
    # 8. Hash do artefato padronizado
    # ------------------------------------------

    standardized_hash = sha256_file(
        standardized_path
    )

    # ------------------------------------------
    # 9. Reabrir o arquivo
    # ------------------------------------------

    with np.load(
        standardized_path,
        allow_pickle=False,
    ) as loaded:

        loaded_magnitude = (
            loaded["magnitude"]
        )

        loaded_phase = (
            loaded["phase"]
        )

    # ------------------------------------------
    # 10. Comparação exata
    # ------------------------------------------

    equal_magnitude = np.array_equal(
        magnitude,
        loaded_magnitude,
        equal_nan=True,
    )

    equal_phase = np.array_equal(
        phase,
        loaded_phase,
        equal_nan=True,
    )

    if not (
        equal_magnitude
        and equal_phase
    ):
        raise ValueError(
            "Reload validation failed."
        )

    # ------------------------------------------
    # 11. Resultado/proveniência
    # ------------------------------------------

    result = StandardizationResult(
        source_id=source_id,

        source_path=str(
            source_path
        ),

        standardized_id=standardized_id,

        standardized_path=str(
            standardized_path
        ),

        rows=layout.rows,
        columns=layout.columns,

        source_endian=SOURCE_ENDIAN,

        standardized_dtype=str(
            magnitude.dtype
        ),

        magnitude_plane_index=(
            MAGNITUDE_PLANE_INDEX
        ),

        phase_plane_index=(
            PHASE_PLANE_INDEX
        ),

        source_hash_sha256=(
            source_hash_sha256
        ),

        standardized_hash_sha256=(
            standardized_hash
        ),

        pipeline_version=(
            PIPELINE_VERSION
        ),

        standardized_format_version=(
            STANDARDIZED_FORMAT_VERSION
        ),

        validation_equal_magnitude=(
            equal_magnitude
        ),

        validation_equal_phase=(
            equal_phase
        ),

        status="success",
    )

    return result


def result_to_dict(
    result: StandardizationResult,
) -> dict:

    return asdict(result)