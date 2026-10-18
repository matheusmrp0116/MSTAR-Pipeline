from pathlib import Path

import numpy as np

from src.mstar_pipeline.signal import (
    inspect_binary_layout,
    validate_exact_layout,
    diagnose_endianness,
    read_float32_plane,
)


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw")


# ============================================================
# AUTOMATIC TEST-FILE SELECTION
# ============================================================

def find_first_valid_mstar_file(root: Path) -> Path:
    """
    Recursively searches the Raw Layer and returns the first file
    that can be validated as an MSTAR/Phoenix observation.

    Files are inspected in deterministic alphabetical order.
    Auxiliary files and files without a valid signal layout are
    automatically ignored.
    """

    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {root.resolve()}"
        )

    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
        ),
        key=lambda path: str(path).lower(),
    )

    if not files:
        raise FileNotFoundError(
            f"No files were found under: {root.resolve()}"
        )

    for candidate in files:
        try:
            validate_exact_layout(candidate)
            return candidate
        except Exception:
            # Auxiliary files or files without a valid
            # MSTAR/Phoenix signal layout are skipped.
            continue

    raise FileNotFoundError(
        "No valid MSTAR/Phoenix observation was found under: "
        f"{root.resolve()}"
    )


# ============================================================
# SELECT TEST FILE
# ============================================================

path = find_first_valid_mstar_file(RAW_DIR)

print("=== TEST FILE ===")
print("Selected file:", path)
print("Absolute path:", path.resolve())


# ============================================================
# INSPECT HEADER AND BINARY LAYOUT
# ============================================================

header, detected_end, candidates = inspect_binary_layout(path)


print("\n=== HEADER ===")

print(
    "Rows:",
    header.rows,
)

print(
    "Columns:",
    header.columns,
)

print(
    "Declared header length:",
    header.header_length,
)

print(
    "Detected end:",
    detected_end,
)

print(
    "Native header length:",
    header.native_header_length,
)

print(
    "Declared PhoenixSigSize:",
    header.signal_size,
)


# ============================================================
# DISPLAY LAYOUT CANDIDATES
# ============================================================

print("\n=== CANDIDATES ===")

for candidate in candidates:

    print()

    for key, value in vars(candidate).items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# VALIDATE EXACT BINARY LAYOUT
# ============================================================

print("\n=== VALIDATED LAYOUT ===")

layout = validate_exact_layout(path)

for key, value in vars(layout).items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# FILE SIZE CONSISTENCY
# ============================================================

file_size = path.stat().st_size

print(
    "Physical file size:",
    file_size,
)

print(
    "PhoenixSigSize:",
    header.signal_size,
)

if header.signal_size is not None:

    print(
        "File size - PhoenixSigSize:",
        file_size - header.signal_size,
    )


# ============================================================
# INITIAL ENDIANNESS DIAGNOSTIC
# ============================================================

diagnostics = diagnose_endianness(
    path,
    layout,
    plane_index=0,
)


print(
    "\n=== ENDIANNESS DIAGNOSTIC ==="
)

for diagnostic in diagnostics:

    print()

    for key, value in vars(diagnostic).items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# ENDIANNESS DIAGNOSTIC FOR ALL PLANES
# ============================================================

for plane_index in range(layout.plane_count):

    print(
        f"\n=== PLANE {plane_index} ==="
    )

    diagnostics = diagnose_endianness(
        path,
        layout,
        plane_index=plane_index,
    )

    for diagnostic in diagnostics:

        print(
            f"\nEndian: {diagnostic.endian}"
        )

        print(
            f"Finite ratio: "
            f"{diagnostic.finite_ratio}"
        )

        print(
            f"Nonnegative ratio: "
            f"{diagnostic.nonnegative_ratio}"
        )

        print(
            f"Very small ratio: "
            f"{diagnostic.very_small_ratio}"
        )

        print(
            f"Very large ratio: "
            f"{diagnostic.very_large_ratio}"
        )

        print(
            f"Min: {diagnostic.minimum}"
        )

        print(
            f"Max: {diagnostic.maximum}"
        )

        print(
            f"Median: {diagnostic.median}"
        )

        print(
            f"P99 abs: {diagnostic.p99_abs}"
        )


# ============================================================
# READ PLANE 0 AS BIG-ENDIAN FLOAT32
# ============================================================

plane_0 = read_float32_plane(
    path,
    layout,
    plane_index=0,
    endian="big",
)


print("\n=== FLOAT32 ARRAY: PLANE 0 ===")

print("Shape:", plane_0.shape)
print("dtype:", plane_0.dtype)
print("Finite:", np.isfinite(plane_0).all())
print("Min:", plane_0.min())
print("Max:", plane_0.max())
print("Mean:", plane_0.mean())
print("Median:", np.median(plane_0))


# ============================================================
# READ PLANE 1 AS BIG-ENDIAN FLOAT32
# ============================================================

plane_1 = read_float32_plane(
    path,
    layout,
    plane_index=1,
    endian="big",
)


print("\n=== FLOAT32 ARRAY: PLANE 1 ===")

print("Shape:", plane_1.shape)
print("dtype:", plane_1.dtype)
print("Finite:", np.isfinite(plane_1).all())
print("Min:", plane_1.min())
print("Max:", plane_1.max())
print("Mean:", plane_1.mean())
print("Median:", np.median(plane_1))