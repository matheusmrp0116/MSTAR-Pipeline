from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / "mstar"

STANDARDIZED_DIR = DATA_DIR / "standardized"

METADATA_DIR = DATA_DIR / "metadata"


PIPELINE_VERSION = "0.1.0"


# --------------------------------------------------
# Validated MSTAR source specification
# --------------------------------------------------

SOURCE_ENDIAN = "big"
SOURCE_DTYPE = ">f4"

MAGNITUDE_PLANE_INDEX = 0
PHASE_PLANE_INDEX = 1

STANDARDIZED_FORMAT_VERSION = "1.0"

DERIVED_DIR = DATA_DIR / "derived"

SPECKLE_DIR = (
    DERIVED_DIR
    / "speckle"
)

SPECKLE_MODEL_VERSION = "1.0"


# --------------------------------------------------
# Derived Layer configuration
#
# Single source of truth shared by the Derived Layer
# generation (run_derived_speckle_generation.py) and
# its independent validation (run_derived_layer_validation.py).
# --------------------------------------------------

# Number of stochastic realizations stored in each
# parent x condition container.
DERIVED_REALIZATIONS = 10

# Stochastic conditions: (condition_id, n_scatterers).
# n_scatterers = None denotes the fully developed
# speckle (FDS) reference condition.
SPECKLE_CONDITIONS = (
    ("K_N01", 1),
    ("K_N02", 2),
    ("K_N05", 5),
    ("FDS", None),
)
