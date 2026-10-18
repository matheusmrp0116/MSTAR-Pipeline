from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.mstar_pipeline.config import (
    METADATA_DIR,
    DERIVED_REALIZATIONS,
    SPECKLE_CONDITIONS,
)

from src.mstar_pipeline.hashing import (
    sha256_file,
)

from src.mstar_pipeline.complex_signal import (
    magnitude_phase_to_complex,
)

from src.mstar_pipeline.derived_speckle import (
    deterministic_seed,
    generate_speckle_field,
    complex_to_canonical,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Read from the shared configuration (config.py),
# the same source used by the Derived Layer generation.

REALIZATIONS = DERIVED_REALIZATIONS

EXPECTED_CONDITION_IDS = {
    condition_id
    for condition_id, _ in SPECKLE_CONDITIONS
}

EXPECTED_CONDITIONS = len(
    EXPECTED_CONDITION_IDS
)

EXPECTED_ARRAY_INDICES = list(
    range(REALIZATIONS)
)

EXPECTED_REALIZATION_IDS = list(
    range(1, REALIZATIONS + 1)
)

# D4:
# Select up to five parents from every image dimension
# available in the dataset.
#
# Three realizations per container:
# r001, r005, r010.

D4_PARENTS_PER_DIMENSION = 5

# First, middle and last realization of each container.
# Derived from REALIZATIONS; for 10 realizations this
# yields (0, 4, 9), i.e. r001, r005, r010.

D4_ARRAY_INDICES = tuple(
    sorted(
        {
            0,
            (REALIZATIONS - 1) // 2,
            REALIZATIONS - 1,
        }
    )
)


# ============================================================
# INPUT PATHS
# ============================================================

CATALOG_PATH = (
    METADATA_DIR
    / "dataset_catalog.parquet"
)

CONTAINER_MANIFEST_PATH = (
    METADATA_DIR
    / "derived_speckle_containers.parquet"
)

LOGICAL_MANIFEST_PATH = (
    METADATA_DIR
    / "derived_speckle_manifest.parquet"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

D1_PATH = (
    METADATA_DIR
    / "derived_validation_d1_structure.parquet"
)

D2_PATH = (
    METADATA_DIR
    / "derived_validation_d2_integrity.parquet"
)

D3_PATH = (
    METADATA_DIR
    / "derived_validation_d3_content.parquet"
)

D4_PATH = (
    METADATA_DIR
    / "derived_validation_d4_reproducibility.parquet"
)

SUMMARY_PATH = (
    METADATA_DIR
    / "derived_layer_validation_summary.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def validation_record(
    block: str,
    test: str,
    passed: bool,
    observed=None,
    expected=None,
) -> dict:

    return {
        "block": block,
        "test": test,
        "passed": bool(passed),
        "observed": (
            None
            if observed is None
            else str(observed)
        ),
        "expected": (
            None
            if expected is None
            else str(expected)
        ),
    }


def load_inputs():

    required = (
        CATALOG_PATH,
        CONTAINER_MANIFEST_PATH,
        LOGICAL_MANIFEST_PATH,
    )

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required input not found: "
                f"{path}"
            )

    catalog = pd.read_parquet(
        CATALOG_PATH
    )

    containers = pd.read_parquet(
        CONTAINER_MANIFEST_PATH
    )

    logical = pd.read_parquet(
        LOGICAL_MANIFEST_PATH
    )

    if catalog.empty:
        raise ValueError(
            "Dataset catalog contains no observations."
        )

    if containers.empty:
        raise ValueError(
            "Derived container manifest contains no records."
        )

    if logical.empty:
        raise ValueError(
            "Derived logical manifest contains no records."
        )

    return (
        catalog,
        containers,
        logical,
    )


# ============================================================
# D1 — CARDINALITY AND STRUCTURE
# ============================================================

def validate_d1(
    catalog: pd.DataFrame,
    containers: pd.DataFrame,
    logical: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:

    records = []

    # --------------------------------------------------------
    # Expected cardinalities derived from current catalog
    # --------------------------------------------------------

    expected_parents = len(
        catalog
    )

    expected_containers = (
        expected_parents
        * EXPECTED_CONDITIONS
    )

    expected_logical = (
        expected_containers
        * REALIZATIONS
    )

    # --------------------------------------------------------
    # Global cardinality
    # --------------------------------------------------------

    records.append(
        validation_record(
            "D1",
            "parent_count",
            len(catalog)
            == expected_parents,
            len(catalog),
            expected_parents,
        )
    )

    records.append(
        validation_record(
            "D1",
            "container_count",
            len(containers)
            == expected_containers,
            len(containers),
            expected_containers,
        )
    )

    records.append(
        validation_record(
            "D1",
            "logical_record_count",
            len(logical)
            == expected_logical,
            len(logical),
            expected_logical,
        )
    )

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    unique_container_count = (
        containers[
            "container_id"
        ].nunique()
    )

    records.append(
        validation_record(
            "D1",
            "unique_container_id",
            unique_container_count
            == expected_containers,
            unique_container_count,
            expected_containers,
        )
    )

    unique_derived_count = (
        logical[
            "derived_id"
        ].nunique()
    )

    records.append(
        validation_record(
            "D1",
            "unique_derived_id",
            unique_derived_count
            == expected_logical,
            unique_derived_count,
            expected_logical,
        )
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    observed_conditions = set(
        containers[
            "condition_id"
        ].unique()
    )

    records.append(
        validation_record(
            "D1",
            "condition_set",
            observed_conditions
            == EXPECTED_CONDITION_IDS,
            sorted(
                observed_conditions
            ),
            sorted(
                EXPECTED_CONDITION_IDS
            ),
        )
    )

    condition_counts = (
        containers[
            "condition_id"
        ]
        .value_counts()
        .to_dict()
    )

    condition_counts_valid = all(
        condition_counts.get(
            condition,
            0,
        )
        == expected_parents

        for condition
        in EXPECTED_CONDITION_IDS
    )

    records.append(
        validation_record(
            "D1",
            "containers_per_condition",
            condition_counts_valid,
            condition_counts,
            (
                f"{expected_parents} "
                "per condition"
            ),
        )
    )

    # --------------------------------------------------------
    # Conditions per parent
    # --------------------------------------------------------

    conditions_per_parent = (
        containers
        .groupby(
            "parent_source_id"
        )[
            "condition_id"
        ]
        .nunique()
    )

    conditions_per_parent_valid = bool(
        len(conditions_per_parent)
        == expected_parents
        and
        (
            conditions_per_parent
            == EXPECTED_CONDITIONS
        ).all()
    )

    records.append(
        validation_record(
            "D1",
            "conditions_per_parent",
            conditions_per_parent_valid,
            conditions_per_parent
            .value_counts()
            .to_dict(),
            (
                f"{EXPECTED_CONDITIONS} "
                "conditions for every parent"
            ),
        )
    )

    # --------------------------------------------------------
    # Logical realizations per container
    # --------------------------------------------------------

    logical_per_container = (
        logical
        .groupby(
            "container_id"
        )
        .size()
    )

    realizations_per_container_valid = bool(
        len(logical_per_container)
        == expected_containers
        and
        (
            logical_per_container
            == REALIZATIONS
        ).all()
    )

    records.append(
        validation_record(
            "D1",
            "realizations_per_container",
            realizations_per_container_valid,
            logical_per_container
            .value_counts()
            .to_dict(),
            REALIZATIONS,
        )
    )

    # --------------------------------------------------------
    # Array indices
    # --------------------------------------------------------

    array_index_valid = (
        logical
        .groupby(
            "container_id"
        )[
            "array_index"
        ]
        .apply(
            lambda values:
                sorted(
                    values.astype(
                        int
                    ).tolist()
                )
                == EXPECTED_ARRAY_INDICES
        )
    )

    records.append(
        validation_record(
            "D1",
            "array_indices",
            bool(
                len(array_index_valid)
                == expected_containers
                and
                array_index_valid.all()
            ),
            array_index_valid
            .value_counts()
            .to_dict(),
            EXPECTED_ARRAY_INDICES,
        )
    )

    # --------------------------------------------------------
    # Realization IDs
    # --------------------------------------------------------

    realization_id_valid = (
        logical
        .groupby(
            "container_id"
        )[
            "realization_id"
        ]
        .apply(
            lambda values:
                sorted(
                    values.astype(
                        int
                    ).tolist()
                )
                == EXPECTED_REALIZATION_IDS
        )
    )

    records.append(
        validation_record(
            "D1",
            "realization_ids",
            bool(
                len(realization_id_valid)
                == expected_containers
                and
                realization_id_valid.all()
            ),
            realization_id_valid
            .value_counts()
            .to_dict(),
            EXPECTED_REALIZATION_IDS,
        )
    )

    # --------------------------------------------------------
    # Logical coordinate uniqueness
    # --------------------------------------------------------

    duplicate_coordinates = int(
        logical.duplicated(
            subset=[
                "parent_source_id",
                "condition_id",
                "realization_id",
            ]
        ).sum()
    )

    records.append(
        validation_record(
            "D1",
            "logical_coordinate_uniqueness",
            duplicate_coordinates == 0,
            duplicate_coordinates,
            0,
        )
    )

    # --------------------------------------------------------
    # Parent coverage
    # --------------------------------------------------------

    catalog_parent_ids = set(
        catalog[
            "source_id"
        ]
    )

    container_parent_ids = set(
        containers[
            "parent_source_id"
        ]
    )

    records.append(
        validation_record(
            "D1",
            "parent_coverage",
            container_parent_ids
            == catalog_parent_ids,
            len(
                container_parent_ids
            ),
            len(
                catalog_parent_ids
            ),
        )
    )

    # --------------------------------------------------------
    # Every logical container exists in physical manifest
    # --------------------------------------------------------

    physical_ids = set(
        containers[
            "container_id"
        ]
    )

    logical_ids = set(
        logical[
            "container_id"
        ]
    )

    records.append(
        validation_record(
            "D1",
            "logical_to_physical_container_reference",
            logical_ids
            == physical_ids,
            len(
                logical_ids
            ),
            len(
                physical_ids
            ),
        )
    )

    # --------------------------------------------------------
    # Dimension distribution derived from current catalog
    # --------------------------------------------------------

    catalog_dimension_counts = (
        catalog[
            [
                "rows",
                "columns",
            ]
        ]
        .value_counts()
        .to_dict()
    )

    expected_dimensions = {
        (
            int(rows),
            int(columns),
        ):
            int(count)
            * EXPECTED_CONDITIONS

        for (
            rows,
            columns,
        ), count
        in catalog_dimension_counts.items()
    }

    dimension_counts = (
        containers[
            [
                "rows",
                "columns",
            ]
        ]
        .value_counts()
        .to_dict()
    )

    records.append(
        validation_record(
            "D1",
            "dimension_distribution",
            dimension_counts
            == expected_dimensions,
            dimension_counts,
            expected_dimensions,
        )
    )

    # --------------------------------------------------------
    # Final D1 result
    # --------------------------------------------------------

    result = pd.DataFrame(
        records
    )

    passed = bool(
        result[
            "passed"
        ].all()
    )

    result.to_parquet(
        D1_PATH,
        index=False,
    )

    return result, passed


# ============================================================
# D2 — PHYSICAL INTEGRITY
# ============================================================

def validate_d2(
    containers: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:

    records = []

    total = len(
        containers
    )

    for position, (
        _,
        row,
    ) in enumerate(
        containers.iterrows(),
        start=1,
    ):

        container_path = Path(
            row[
                "container_path"
            ]
        )

        exists = (
            container_path.exists()
        )

        observed_hash = None
        hash_equal = False
        file_size_bytes = None

        error = None

        if exists:

            try:

                file_size_bytes = (
                    container_path
                    .stat()
                    .st_size
                )

                observed_hash = (
                    sha256_file(
                        container_path
                    )
                )

                hash_equal = (
                    observed_hash
                    == row[
                        "container_hash_sha256"
                    ]
                )

            except Exception as exc:

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        records.append(
            {
                "container_id":
                    row[
                        "container_id"
                    ],

                "parent_source_id":
                    row[
                        "parent_source_id"
                    ],

                "condition_id":
                    row[
                        "condition_id"
                    ],

                "container_path":
                    str(
                        container_path
                    ),

                "exists":
                    bool(
                        exists
                    ),

                "file_size_bytes":
                    file_size_bytes,

                "expected_hash_sha256":
                    row[
                        "container_hash_sha256"
                    ],

                "observed_hash_sha256":
                    observed_hash,

                "hash_equal":
                    bool(
                        hash_equal
                    ),

                "error":
                    error,
            }
        )

        if (
            position == 1
            or position % 1000 == 0
            or position == total
        ):

            print(
                f"D2 hash validation: "
                f"{position}/{total}"
            )

    result = pd.DataFrame(
        records
    )

    passed = bool(
        result[
            "exists"
        ].all()
        and
        result[
            "hash_equal"
        ].all()
        and
        result[
            "error"
        ].isna().all()
    )

    result.to_parquet(
        D2_PATH,
        index=False,
    )

    return result, passed


# ============================================================
# D3 — CONTAINER CONTENT
# ============================================================

def validate_d3(
    containers: pd.DataFrame,
    logical: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:

    records = []

    logical_groups = {
        container_id:
            group.sort_values(
                "array_index"
            )

        for (
            container_id,
            group,
        )
        in logical.groupby(
            "container_id"
        )
    }

    total = len(
        containers
    )

    for position, (
        _,
        row,
    ) in enumerate(
        containers.iterrows(),
        start=1,
    ):

        container_id = (
            row[
                "container_id"
            ]
        )

        container_path = Path(
            row[
                "container_path"
            ]
        )

        expected_shape = (
            REALIZATIONS,
            int(
                row["rows"]
            ),
            int(
                row["columns"]
            ),
        )

        record = {
            "container_id":
                container_id,

            "parent_source_id":
                row[
                    "parent_source_id"
                ],

            "condition_id":
                row[
                    "condition_id"
                ],

            "required_arrays":
                False,

            "magnitude_shape_equal":
                False,

            "phase_shape_equal":
                False,

            "magnitude_float32":
                False,

            "phase_float32":
                False,

            "magnitude_finite":
                False,

            "phase_finite":
                False,

            "realization_ids_equal":
                False,

            "seeds_equal_manifest":
                False,

            "seeds_equal_policy":
                False,

            "logical_record_count_equal":
                False,

            "error":
                None,
        }

        try:

            group = logical_groups[
                container_id
            ]

            record[
                "logical_record_count_equal"
            ] = (
                len(group)
                == REALIZATIONS
            )

            with np.load(
                container_path,
                allow_pickle=False,
            ) as data:

                required = {
                    "magnitude",
                    "phase",
                    "seeds",
                    "realization_ids",
                }

                available = set(
                    data.files
                )

                record[
                    "required_arrays"
                ] = (
                    required
                    <= available
                )

                if not record[
                    "required_arrays"
                ]:

                    missing = (
                        required
                        - available
                    )

                    raise ValueError(
                        "Missing arrays: "
                        f"{sorted(missing)}"
                    )

                magnitude = np.asarray(
                    data[
                        "magnitude"
                    ]
                )

                phase = np.asarray(
                    data[
                        "phase"
                    ]
                )

                seeds = np.asarray(
                    data[
                        "seeds"
                    ]
                )

                realization_ids = (
                    np.asarray(
                        data[
                            "realization_ids"
                        ]
                    )
                )

            # --------------------------------------------
            # Shapes
            # --------------------------------------------

            record[
                "magnitude_shape_equal"
            ] = (
                magnitude.shape
                == expected_shape
            )

            record[
                "phase_shape_equal"
            ] = (
                phase.shape
                == expected_shape
            )

            # --------------------------------------------
            # Dtypes
            # --------------------------------------------

            record[
                "magnitude_float32"
            ] = (
                magnitude.dtype
                == np.float32
            )

            record[
                "phase_float32"
            ] = (
                phase.dtype
                == np.float32
            )

            # --------------------------------------------
            # Finiteness
            # --------------------------------------------

            record[
                "magnitude_finite"
            ] = bool(
                np.isfinite(
                    magnitude
                ).all()
            )

            record[
                "phase_finite"
            ] = bool(
                np.isfinite(
                    phase
                ).all()
            )

            # --------------------------------------------
            # Realization IDs
            # --------------------------------------------

            expected_ids = np.arange(
                1,
                REALIZATIONS + 1,
                dtype=np.uint16,
            )

            record[
                "realization_ids_equal"
            ] = bool(
                np.array_equal(
                    realization_ids,
                    expected_ids,
                )
            )

            # --------------------------------------------
            # Seeds vs logical manifest
            # --------------------------------------------

            manifest_seeds = (
                group[
                    "seed"
                ]
                .astype(
                    np.uint32
                )
                .to_numpy()
            )

            record[
                "seeds_equal_manifest"
            ] = bool(
                np.array_equal(
                    seeds,
                    manifest_seeds,
                )
            )

            # --------------------------------------------
            # Seeds vs deterministic policy
            # --------------------------------------------

            expected_seeds = np.array(
                [
                    deterministic_seed(
                        source_id=
                            row[
                                "parent_source_id"
                            ],

                        condition_id=
                            row[
                                "condition_id"
                            ],

                        realization_id=
                            realization_id,
                    )

                    for realization_id
                    in EXPECTED_REALIZATION_IDS
                ],
                dtype=np.uint32,
            )

            record[
                "seeds_equal_policy"
            ] = bool(
                np.array_equal(
                    seeds,
                    expected_seeds,
                )
            )

        except Exception as exc:

            record[
                "error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        records.append(
            record
        )

        if (
            position == 1
            or position % 1000 == 0
            or position == total
        ):

            print(
                f"D3 content validation: "
                f"{position}/{total}"
            )

    result = pd.DataFrame(
        records
    )

    validation_columns = [
        "required_arrays",
        "magnitude_shape_equal",
        "phase_shape_equal",
        "magnitude_float32",
        "phase_float32",
        "magnitude_finite",
        "phase_finite",
        "realization_ids_equal",
        "seeds_equal_manifest",
        "seeds_equal_policy",
        "logical_record_count_equal",
    ]

    passed = bool(
        result[
            validation_columns
        ].all().all()
        and
        result[
            "error"
        ].isna().all()
    )

    result.to_parquet(
        D3_PATH,
        index=False,
    )

    return result, passed


# ============================================================
# D4 — INDEPENDENT DETERMINISTIC REPRODUCTION
# ============================================================

def select_d4_parents(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministically select up to
    D4_PARENTS_PER_DIMENSION parents from every
    image dimension available in the dataset.

    No predefined dimensions are assumed.

    If a dimension contains fewer than the configured
    number of parents, all available parents from that
    dimension are selected.
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

        subset = (
            valid_catalog[
                (valid_catalog["rows"] == rows)
                & (valid_catalog["columns"] == columns)
            ]
            .sort_values(
                "source_id"
            )
            .head(
                D4_PARENTS_PER_DIMENSION
            )
            .copy()
        )

        if not subset.empty:
            selected.append(
                subset
            )

    if not selected:
        raise ValueError(
            "No parents could be selected "
            "for D4 validation."
        )

    return (
        pd.concat(
            selected,
            ignore_index=True,
        )
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


def validate_d4(
    catalog: pd.DataFrame,
    containers: pd.DataFrame,
    logical: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:

    selected_parents = (
        select_d4_parents(
            catalog
        )
    )

    selected_source_ids = set(
        selected_parents[
            "source_id"
        ]
    )

    selected_containers = (
        containers[
            containers[
                "parent_source_id"
            ].isin(
                selected_source_ids
            )
        ]
        .sort_values(
            [
                "rows",
                "columns",
                "parent_source_id",
                "condition_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_selected_containers = (
        len(selected_parents)
        * EXPECTED_CONDITIONS
    )

    if (
        len(
            selected_containers
        )
        != expected_selected_containers
    ):

        raise ValueError(
            "Unexpected D4 container count: "
            f"{len(selected_containers)} "
            f"!= "
            f"{expected_selected_containers}"
        )

    catalog_by_source = (
        catalog
        .set_index(
            "source_id",
            drop=False,
        )
    )

    logical_lookup = (
        logical
        .set_index(
            [
                "container_id",
                "array_index",
            ],
            drop=False,
        )
    )

    records = []

    total_tests = (
        len(
            selected_containers
        )
        * len(
            D4_ARRAY_INDICES
        )
    )

    test_number = 0

    for _, container_row in (
        selected_containers.iterrows()
    ):

        parent_source_id = (
            container_row[
                "parent_source_id"
            ]
        )

        condition_id = (
            container_row[
                "condition_id"
            ]
        )

        container_id = (
            container_row[
                "container_id"
            ]
        )

        parent = (
            catalog_by_source.loc[
                parent_source_id
            ]
        )

        # ----------------------------------------------------
        # Load parent canonical standardized representation
        # ----------------------------------------------------

        with np.load(
            Path(
                parent[
                    "standardized_path"
                ]
            ),
            allow_pickle=False,
        ) as data:

            parent_magnitude = (
                np.asarray(
                    data[
                        "magnitude"
                    ],
                    dtype=np.float32,
                )
            )

            parent_phase = (
                np.asarray(
                    data[
                        "phase"
                    ],
                    dtype=np.float32,
                )
            )

        baseline_complex = (
            magnitude_phase_to_complex(
                parent_magnitude,
                parent_phase,
            )
        )

        # ----------------------------------------------------
        # Load derived container
        # ----------------------------------------------------

        with np.load(
            Path(
                container_row[
                    "container_path"
                ]
            ),
            allow_pickle=False,
        ) as data:

            stored_magnitude = (
                np.asarray(
                    data[
                        "magnitude"
                    ]
                )
            )

            stored_phase = (
                np.asarray(
                    data[
                        "phase"
                    ]
                )
            )

            stored_seeds = (
                np.asarray(
                    data[
                        "seeds"
                    ]
                )
            )

        # ----------------------------------------------------
        # Selected realizations
        # ----------------------------------------------------

        for array_index in (
            D4_ARRAY_INDICES
        ):

            test_number += 1

            realization_id = (
                array_index + 1
            )

            logical_row = (
                logical_lookup.loc[
                    (
                        container_id,
                        array_index,
                    )
                ]
            )

            expected_seed = (
                deterministic_seed(
                    source_id=
                        parent_source_id,

                    condition_id=
                        condition_id,

                    realization_id=
                        realization_id,
                )
            )

            stored_seed = int(
                stored_seeds[
                    array_index
                ]
            )

            manifest_seed = int(
                logical_row[
                    "seed"
                ]
            )

            seed_equal = (
                stored_seed
                == expected_seed
                == manifest_seed
            )

            # --------------------------------------------
            # Recreate stochastic field independently
            # --------------------------------------------

            n_scatterers = (
                None
                if condition_id
                == "FDS"
                else int(
                    container_row[
                        "n_scatterers"
                    ]
                )
            )

            speckle = (
                generate_speckle_field(
                    shape=
                        parent_magnitude.shape,

                    condition_id=
                        condition_id,

                    n_scatterers=
                        n_scatterers,

                    seed=
                        expected_seed,
                )
            )

            reproduced_complex = (
                baseline_complex
                * speckle
            )

            (
                reproduced_magnitude,
                reproduced_phase,
            ) = complex_to_canonical(
                reproduced_complex
            )

            magnitude_equal = bool(
                np.array_equal(
                    reproduced_magnitude,
                    stored_magnitude[
                        array_index
                    ],
                )
            )

            phase_equal = bool(
                np.array_equal(
                    reproduced_phase,
                    stored_phase[
                        array_index
                    ],
                )
            )

            # Exact equality is expected here.
            # Error metrics are retained as
            # diagnostic evidence.

            magnitude_max_abs_error = float(
                np.max(
                    np.abs(
                        reproduced_magnitude
                        - stored_magnitude[
                            array_index
                        ]
                    )
                )
            )

            phase_max_abs_error = float(
                np.max(
                    np.abs(
                        reproduced_phase
                        - stored_phase[
                            array_index
                        ]
                    )
                )
            )

            records.append(
                {
                    "container_id":
                        container_id,

                    "parent_source_id":
                        parent_source_id,

                    "condition_id":
                        condition_id,

                    "rows":
                        int(
                            container_row[
                                "rows"
                            ]
                        ),

                    "columns":
                        int(
                            container_row[
                                "columns"
                            ]
                        ),

                    "array_index":
                        array_index,

                    "realization_id":
                        realization_id,

                    "expected_seed":
                        expected_seed,

                    "stored_seed":
                        stored_seed,

                    "manifest_seed":
                        manifest_seed,

                    "seed_equal":
                        bool(
                            seed_equal
                        ),

                    "magnitude_equal":
                        magnitude_equal,

                    "phase_equal":
                        phase_equal,

                    "magnitude_max_abs_error":
                        magnitude_max_abs_error,

                    "phase_max_abs_error":
                        phase_max_abs_error,
                }
            )

            if (
                test_number == 1
                or test_number % 20 == 0
                or test_number == total_tests
            ):

                print(
                    "D4 deterministic "
                    "reproduction: "
                    f"{test_number}/"
                    f"{total_tests}"
                )

    result = pd.DataFrame(
        records
    )

    passed = bool(
        len(result)
        == total_tests
        and
        result[
            "seed_equal"
        ].all()
        and
        result[
            "magnitude_equal"
        ].all()
        and
        result[
            "phase_equal"
        ].all()
    )

    result.to_parquet(
        D4_PATH,
        index=False,
    )

    return result, passed


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print(
        "=== INDEPENDENT DERIVED "
        "LAYER VALIDATION ==="
    )

    print()

    (
        catalog,
        containers,
        logical,
    ) = load_inputs()

    # --------------------------------------------------------
    # Dataset characteristics detected dynamically
    # --------------------------------------------------------

    detected_dimensions = (
        catalog[
            ["rows", "columns"]
        ]
        .value_counts()
        .sort_index()
    )

    expected_parents = len(
        catalog
    )

    expected_containers = (
        expected_parents
        * EXPECTED_CONDITIONS
    )

    expected_logical = (
        expected_containers
        * REALIZATIONS
    )

    print(
        "Parents detected:",
        expected_parents
    )

    print(
        "Dimensions detected:"
    )

    print(
        detected_dimensions
    )

    print(
        "Expected containers:",
        expected_containers
    )

    print(
        "Expected logical realizations:",
        expected_logical
    )

    print()

    # ========================================================
    # D1
    # ========================================================

    print(
        "=== D1 — CARDINALITY "
        "AND STRUCTURE ==="
    )

    d1, d1_passed = (
        validate_d1(
            catalog,
            containers,
            logical,
        )
    )

    print(
        d1[
            [
                "test",
                "passed",
                "observed",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "D1 passed:",
        d1_passed
    )

    # ========================================================
    # D2
    # ========================================================

    print()

    print(
        "=== D2 — PHYSICAL "
        "INTEGRITY ==="
    )

    d2, d2_passed = (
        validate_d2(
            containers
        )
    )

    print()

    print(
        "Files existing:",
        int(
            d2[
                "exists"
            ].sum()
        ),
        "/",
        len(d2),
    )

    print(
        "Hashes equal:",
        int(
            d2[
                "hash_equal"
            ].sum()
        ),
        "/",
        len(d2),
    )

    print(
        "D2 passed:",
        d2_passed
    )

    # ========================================================
    # D3
    # ========================================================

    print()

    print(
        "=== D3 — CONTAINER "
        "CONTENT ==="
    )

    d3, d3_passed = (
        validate_d3(
            containers,
            logical,
        )
    )

    d3_validation_columns = [
        "required_arrays",
        "magnitude_shape_equal",
        "phase_shape_equal",
        "magnitude_float32",
        "phase_float32",
        "magnitude_finite",
        "phase_finite",
        "realization_ids_equal",
        "seeds_equal_manifest",
        "seeds_equal_policy",
        "logical_record_count_equal",
    ]

    print()

    for column in (
        d3_validation_columns
    ):

        print(
            column,
            ":",
            int(
                d3[
                    column
                ].sum()
            ),
            "/",
            len(d3),
        )

    print(
        "D3 passed:",
        d3_passed
    )

    # ========================================================
    # D4
    # ========================================================

    print()

    print(
        "=== D4 — INDEPENDENT "
        "REPRODUCTION ==="
    )

    d4, d4_passed = (
        validate_d4(
            catalog,
            containers,
            logical,
        )
    )

    print()

    print(
        "Tests:",
        len(d4)
    )

    print(
        "Seed equality:",
        int(
            d4[
                "seed_equal"
            ].sum()
        ),
        "/",
        len(d4),
    )

    print(
        "Magnitude exact:",
        int(
            d4[
                "magnitude_equal"
            ].sum()
        ),
        "/",
        len(d4),
    )

    print(
        "Phase exact:",
        int(
            d4[
                "phase_equal"
            ].sum()
        ),
        "/",
        len(d4),
    )

    print(
        "Maximum magnitude error:",
        d4[
            "magnitude_max_abs_error"
        ].max()
    )

    print(
        "Maximum phase error:",
        d4[
            "phase_max_abs_error"
        ].max()
    )

    print(
        "D4 passed:",
        d4_passed
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    overall_passed = bool(
        d1_passed
        and d2_passed
        and d3_passed
        and d4_passed
    )

    summary = pd.DataFrame(
        [
            {
                "validation":
                    "D1_structure",

                "passed":
                    d1_passed,
            },
            {
                "validation":
                    "D2_physical_integrity",

                "passed":
                    d2_passed,
            },
            {
                "validation":
                    "D3_container_content",

                "passed":
                    d3_passed,
            },
            {
                "validation":
                    "D4_independent_reproduction",

                "passed":
                    d4_passed,
            },
            {
                "validation":
                    "OVERALL",

                "passed":
                    overall_passed,
            },
        ]
    )

    summary.to_parquet(
        SUMMARY_PATH,
        index=False,
    )

    elapsed = (
        time.time()
        - start
    )

    print()

    print(
        "=== DERIVED LAYER "
        "VALIDATION SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Validation runtime "
        "(minutes):",
        round(
            elapsed / 60,
            2,
        )
    )

    print()

    print(
        "D1:",
        D1_PATH
    )

    print(
        "D2:",
        D2_PATH
    )

    print(
        "D3:",
        D3_PATH
    )

    print(
        "D4:",
        D4_PATH
    )

    print(
        "Summary:",
        SUMMARY_PATH
    )

    if not overall_passed:

        raise RuntimeError(
            "Derived Layer validation "
            "did not pass all D1-D4 checks."
        )


if __name__ == "__main__":
    main()