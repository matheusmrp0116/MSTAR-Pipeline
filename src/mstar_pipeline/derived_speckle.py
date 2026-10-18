from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from src.mstar_pipeline.config import (
    PIPELINE_VERSION,
    SPECKLE_MODEL_VERSION,
)

from src.mstar_pipeline.hashing import (
    sha256_file,
)

from src.mstar_pipeline.complex_signal import (
    magnitude_phase_to_complex,
)

from src.mstar_pipeline.scattering import (
    generate_coherent_scatterer_field,
    generate_fully_developed_speckle,
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

BASE_SEED = 20260929
NU = 1.0

TRANSFORMATION = "complex_multiplicative"

SPECKLE_MODEL = "coherent_equivalent_scatterer"

DERIVED_DTYPE = "float32"
DERIVED_FORMAT_VERSION = "2.0"


# ==================================================
# RESULT
# ==================================================

@dataclass
class DerivedContainerResult:

    container_id: str

    parent_source_id: str
    parent_standardized_id: str

    condition_id: str

    n_scatterers: int | None
    nu: float | None

    rows: int
    columns: int

    realization_count: int

    transformation: str

    speckle_model: str
    speckle_model_version: str

    pipeline_version: str

    derived_dtype: str
    derived_format_version: str

    container_path: str
    container_hash_sha256: str

    generation_timestamp_utc: str

    magnitude_finite: bool
    phase_finite: bool

    zero_pixels_preserved: bool

    reload_magnitude_equal: bool
    reload_phase_equal: bool
    reload_seeds_equal: bool
    reload_realization_ids_equal: bool

    deterministic_reproduction: bool

    status: str


# ==================================================
# DETERMINISTIC IDENTIFIERS
# ==================================================

def deterministic_seed(
    source_id: str,
    condition_id: str,
    realization_id: int,
) -> int:
    """
    Deterministic uint32 seed.

    Includes model and pipeline versions so that
    stochastic identity is explicitly versioned.
    """

    key = (
        f"{BASE_SEED}|"
        f"{source_id}|"
        f"{condition_id}|"
        f"{realization_id}|"
        f"{SPECKLE_MODEL_VERSION}|"
        f"{PIPELINE_VERSION}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def make_container_id(
    parent_source_id: str,
    condition_id: str,
) -> str:
    """
    Stable identifier for one
    parent x condition container.
    """

    key = (
        f"{parent_source_id}|"
        f"{condition_id}|"
        f"{SPECKLE_MODEL_VERSION}|"
        f"{PIPELINE_VERSION}|"
        f"{DERIVED_FORMAT_VERSION}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return f"SPKC-{digest[:16]}"


def make_derived_id(
    parent_source_id: str,
    condition_id: str,
    realization_id: int,
) -> str:
    """
    Stable identifier for one logical
    derived realization.
    """

    key = (
        f"{parent_source_id}|"
        f"{condition_id}|"
        f"{realization_id}|"
        f"{SPECKLE_MODEL_VERSION}|"
        f"{PIPELINE_VERSION}|"
        f"{DERIVED_FORMAT_VERSION}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return f"SPK-{digest[:16]}"


# ==================================================
# SPECKLE GENERATION
# ==================================================

def generate_speckle_field(
    shape: tuple[int, int],
    condition_id: str,
    n_scatterers: int | None,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(
        seed
    )

    if condition_id == "FDS":

        return generate_fully_developed_speckle(
            size=shape,
            rng=rng,
        )

    if n_scatterers is None:
        raise ValueError(
            "n_scatterers is required for "
            "finite-scatterer conditions."
        )

    return generate_coherent_scatterer_field(
        size=shape,
        n_scatterers=n_scatterers,
        nu=NU,
        rng=rng,
    )


# ==================================================
# COMPLEX -> CANONICAL
# ==================================================

def complex_to_canonical(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    magnitude = np.abs(
        signal
    ).astype(
        np.float32
    )

    phase = np.mod(
        np.angle(signal),
        2.0 * np.pi,
    ).astype(
        np.float32
    )

    # Phase is undefined when magnitude == 0.
    # Store zero as deterministic representation.
    phase[
        magnitude == 0.0
    ] = 0.0

    return magnitude, phase


# ==================================================
# GENERATE ONE CONTAINER
# ==================================================

def generate_derived_container(
    *,
    parent_source_id: str,
    parent_standardized_id: str,
    standardized_path: Path,
    output_root: Path,
    condition_id: str,
    n_scatterers: int | None,
    realization_count: int,
) -> tuple[
    DerivedContainerResult,
    list[dict],
]:

    if realization_count < 1:
        raise ValueError(
            "realization_count must be >= 1."
        )

    standardized_path = Path(
        standardized_path
    )

    output_root = Path(
        output_root
    )

    # --------------------------------------------------
    # Load parent
    # --------------------------------------------------

    with np.load(
        standardized_path,
        allow_pickle=False,
    ) as data:

        magnitude = np.asarray(
            data["magnitude"],
            dtype=np.float32,
        )

        phase = np.asarray(
            data["phase"],
            dtype=np.float32,
        )

    if magnitude.shape != phase.shape:
        raise ValueError(
            "Magnitude and phase shapes differ."
        )

    if magnitude.ndim != 2:
        raise ValueError(
            "Expected a 2-D observation."
        )

    if not np.isfinite(
        magnitude
    ).all():
        raise ValueError(
            "Parent magnitude contains "
            "non-finite values."
        )

    if not np.isfinite(
        phase
    ).all():
        raise ValueError(
            "Parent phase contains "
            "non-finite values."
        )

    rows, columns = magnitude.shape

    baseline_zero_mask = (
        magnitude == 0.0
    )

    # --------------------------------------------------
    # Complex parent
    # --------------------------------------------------

    baseline_complex = (
        magnitude_phase_to_complex(
            magnitude,
            phase,
        )
    )

    # --------------------------------------------------
    # Allocate container arrays
    # --------------------------------------------------

    magnitude_stack = np.empty(
        (
            realization_count,
            rows,
            columns,
        ),
        dtype=np.float32,
    )

    phase_stack = np.empty(
        (
            realization_count,
            rows,
            columns,
        ),
        dtype=np.float32,
    )

    seeds = np.empty(
        realization_count,
        dtype=np.uint32,
    )

    realization_ids = np.arange(
        1,
        realization_count + 1,
        dtype=np.uint16,
    )

    # --------------------------------------------------
    # Generate realizations
    # --------------------------------------------------

    logical_records = []

    for array_index in range(
        realization_count
    ):

        realization_id = (
            array_index + 1
        )

        seed = deterministic_seed(
            source_id=
                parent_source_id,

            condition_id=
                condition_id,

            realization_id=
                realization_id,
        )

        seeds[
            array_index
        ] = seed

        speckle = generate_speckle_field(
            shape=(
                rows,
                columns,
            ),

            condition_id=
                condition_id,

            n_scatterers=
                n_scatterers,

            seed=
                seed,
        )

        derived_complex = (
            baseline_complex
            * speckle
        )

        (
            derived_magnitude,
            derived_phase,
        ) = complex_to_canonical(
            derived_complex
        )

        magnitude_stack[
            array_index
        ] = derived_magnitude

        phase_stack[
            array_index
        ] = derived_phase

    # --------------------------------------------------
    # Numeric validation
    # --------------------------------------------------

    magnitude_finite = bool(
        np.isfinite(
            magnitude_stack
        ).all()
    )

    phase_finite = bool(
        np.isfinite(
            phase_stack
        ).all()
    )

    if not magnitude_finite:
        raise ValueError(
            "Derived magnitude contains "
            "non-finite values."
        )

    if not phase_finite:
        raise ValueError(
            "Derived phase contains "
            "non-finite values."
        )

    # --------------------------------------------------
    # Zero preservation
    # --------------------------------------------------

    if baseline_zero_mask.any():

        zero_pixels_preserved = bool(
            np.all(
                magnitude_stack[
                    :,
                    baseline_zero_mask,
                ]
                == 0.0
            )
        )

    else:
        zero_pixels_preserved = True

    if not zero_pixels_preserved:
        raise ValueError(
            "Zero baseline pixels were not "
            "preserved."
        )

    # --------------------------------------------------
    # Container identity/path
    # --------------------------------------------------

    container_id = make_container_id(
        parent_source_id=
            parent_source_id,

        condition_id=
            condition_id,
    )

    condition_dir = (
        output_root
        / condition_id
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    container_path = (
        condition_dir
        / f"{container_id}.npz"
    )

    # --------------------------------------------------
    # Atomic write
    # --------------------------------------------------

    temporary_path = (
        condition_dir
        / f"{container_id}.tmp.npz"
    )

    np.savez(
        temporary_path,
        magnitude=magnitude_stack,
        phase=phase_stack,
        seeds=seeds,
        realization_ids=realization_ids,
    )

    temporary_path.replace(
        container_path
    )

    # --------------------------------------------------
    # Reload validation
    # --------------------------------------------------

    with np.load(
        container_path,
        allow_pickle=False,
    ) as check:

        reload_magnitude = (
            check["magnitude"]
        )

        reload_phase = (
            check["phase"]
        )

        reload_seeds = (
            check["seeds"]
        )

        reload_realization_ids = (
            check["realization_ids"]
        )

    reload_magnitude_equal = bool(
        np.array_equal(
            magnitude_stack,
            reload_magnitude,
        )
    )

    reload_phase_equal = bool(
        np.array_equal(
            phase_stack,
            reload_phase,
        )
    )

    reload_seeds_equal = bool(
        np.array_equal(
            seeds,
            reload_seeds,
        )
    )

    reload_realization_ids_equal = bool(
        np.array_equal(
            realization_ids,
            reload_realization_ids,
        )
    )

    if not all(
        (
            reload_magnitude_equal,
            reload_phase_equal,
            reload_seeds_equal,
            reload_realization_ids_equal,
        )
    ):
        raise ValueError(
            "Container reload validation failed."
        )

    # --------------------------------------------------
    # Deterministic regeneration check
    # --------------------------------------------------

    validation_seed = int(
        seeds[0]
    )

    speckle_a = (
        generate_speckle_field(
            shape=(
                rows,
                columns,
            ),

            condition_id=
                condition_id,

            n_scatterers=
                n_scatterers,

            seed=
                validation_seed,
        )
    )

    speckle_b = (
        generate_speckle_field(
            shape=(
                rows,
                columns,
            ),

            condition_id=
                condition_id,

            n_scatterers=
                n_scatterers,

            seed=
                validation_seed,
        )
    )

    deterministic_reproduction = bool(
        np.array_equal(
            speckle_a,
            speckle_b,
        )
    )

    if not deterministic_reproduction:
        raise ValueError(
            "Deterministic regeneration failed."
        )

    # --------------------------------------------------
    # Hash
    # --------------------------------------------------

    container_hash = sha256_file(
        container_path
    )

    timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------
    # Logical realization records
    # --------------------------------------------------

    for array_index in range(
        realization_count
    ):

        realization_id = int(
            realization_ids[
                array_index
            ]
        )

        seed = int(
            seeds[
                array_index
            ]
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
                    container_id,

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
                    seed,

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
                    timestamp,
            }
        )

    # --------------------------------------------------
    # Container record
    # --------------------------------------------------

    result = DerivedContainerResult(
        container_id=
            container_id,

        parent_source_id=
            parent_source_id,

        parent_standardized_id=
            parent_standardized_id,

        condition_id=
            condition_id,

        n_scatterers=
            n_scatterers,

        nu=(
            NU
            if n_scatterers
            is not None
            else None
        ),

        rows=
            rows,

        columns=
            columns,

        realization_count=
            realization_count,

        transformation=
            TRANSFORMATION,

        speckle_model=
            SPECKLE_MODEL,

        speckle_model_version=
            SPECKLE_MODEL_VERSION,

        pipeline_version=
            PIPELINE_VERSION,

        derived_dtype=
            DERIVED_DTYPE,

        derived_format_version=
            DERIVED_FORMAT_VERSION,

        container_path=
            str(container_path),

        container_hash_sha256=
            container_hash,

        generation_timestamp_utc=
            timestamp,

        magnitude_finite=
            magnitude_finite,

        phase_finite=
            phase_finite,

        zero_pixels_preserved=
            zero_pixels_preserved,

        reload_magnitude_equal=
            reload_magnitude_equal,

        reload_phase_equal=
            reload_phase_equal,

        reload_seeds_equal=
            reload_seeds_equal,

        reload_realization_ids_equal=
            reload_realization_ids_equal,

        deterministic_reproduction=
            deterministic_reproduction,

        status=
            "success",
    )

    return result, logical_records