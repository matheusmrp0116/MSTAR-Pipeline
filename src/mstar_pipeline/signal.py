from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .phoenix import (
    PhoenixHeader,
    parse_phoenix_header,
)


FLOAT32_BYTES = 4


@dataclass
class LayoutCandidate:
    offset_name: str
    offset_bytes: int

    file_size_bytes: int
    bytes_after_offset: int

    rows: int
    columns: int
    pixels_per_plane: int

    bytes_per_plane: int
    inferred_plane_count: float

    exact_one_plane: bool
    exact_two_planes: bool

    residual_one_plane: int
    residual_two_planes: int


@dataclass
class EndianDiagnostic:
    endian: str
    dtype: str

    sample_count: int

    finite_ratio: float
    nonnegative_ratio: float
    zero_ratio: float

    very_small_ratio: float
    very_large_ratio: float

    minimum: float | None
    maximum: float | None
    median: float | None
    median_abs: float | None
    p99_abs: float | None


@dataclass
class ValidatedLayout:
    offset_bytes: int
    offset_source: str

    rows: int
    columns: int

    pixels_per_plane: int
    bytes_per_plane: int

    plane_count: int

    file_size_bytes: int
    bytes_after_offset: int

def _unique_offset_candidates(
    header: PhoenixHeader,
    detected_end: int,
) -> list[tuple[str, int]]:

    candidates = []

    if detected_end is not None:
        candidates.append(
            ("detected_header_end", int(detected_end))
        )

    if header.header_length is not None:
        candidates.append(
            (
                "declared_header_length",
                int(header.header_length),
            )
        )

    if (
        header.header_length is not None
        and header.native_header_length is not None
    ):
        candidates.append(
            (
                "declared_plus_native",
                int(header.header_length)
                + int(header.native_header_length),
            )
        )

    # Evita testar várias vezes exatamente
    # o mesmo offset com nomes diferentes.
    unique = {}
    for name, offset in candidates:
        if offset not in unique:
            unique[offset] = name

    return [
        (name, offset)
        for offset, name in unique.items()
    ]

def inspect_binary_layout(
    path: str | Path,
) -> tuple[
    PhoenixHeader,
    int,
    list[LayoutCandidate],
]:

    path = Path(path)

    header, detected_end = (
        parse_phoenix_header(path)
    )

    if header.rows is None:
        raise ValueError(
            "NumberOfRows ausente no Phoenix Header."
        )

    if header.columns is None:
        raise ValueError(
            "NumberOfColumns ausente no Phoenix Header."
        )

    rows = int(header.rows)
    columns = int(header.columns)

    if rows <= 0 or columns <= 0:
        raise ValueError(
            f"Dimensões inválidas: {rows} x {columns}"
        )

    file_size = path.stat().st_size

    pixels = rows * columns
    bytes_per_plane = pixels * FLOAT32_BYTES

    results = []

    for (
        offset_name,
        offset_bytes,
    ) in _unique_offset_candidates(
        header,
        detected_end,
    ):

        if offset_bytes < 0:
            continue

        if offset_bytes > file_size:
            continue

        available = (
            file_size - offset_bytes
        )

        inferred_planes = (
            available / bytes_per_plane
        )

        results.append(
            LayoutCandidate(
                offset_name=offset_name,
                offset_bytes=offset_bytes,

                file_size_bytes=file_size,
                bytes_after_offset=available,

                rows=rows,
                columns=columns,
                pixels_per_plane=pixels,

                bytes_per_plane=bytes_per_plane,

                inferred_plane_count=(
                    inferred_planes
                ),

                exact_one_plane=(
                    available
                    == bytes_per_plane
                ),

                exact_two_planes=(
                    available
                    == 2 * bytes_per_plane
                ),

                residual_one_plane=(
                    available
                    - bytes_per_plane
                ),

                residual_two_planes=(
                    available
                    - 2 * bytes_per_plane
                ),
            )
        )

    return (
        header,
        detected_end,
        results,
    )

def validate_exact_layout(
    path: str | Path,
) -> ValidatedLayout:

    (
        header,
        detected_end,
        candidates,
    ) = inspect_binary_layout(path)

    exact = []

    for candidate in candidates:

        if candidate.exact_one_plane:
            exact.append(
                (
                    candidate,
                    1,
                )
            )

        if candidate.exact_two_planes:
            exact.append(
                (
                    candidate,
                    2,
                )
            )

    if len(exact) == 0:
        raise ValueError(
            "Nenhum layout float32 de 1 ou 2 planos "
            "apresentou ajuste exato ao tamanho físico "
            "do arquivo."
        )

    # Como offsets numericamente iguais foram
    # deduplicados, mais de uma solução aqui
    # realmente indica ambiguidade estrutural.
    if len(exact) > 1:
        description = [
            (
                item.offset_name,
                item.offset_bytes,
                planes,
            )
            for item, planes in exact
        ]

        raise ValueError(
            "Mais de um layout apresentou ajuste "
            f"exato: {description}"
        )

    candidate, plane_count = exact[0]

    return ValidatedLayout(
        offset_bytes=(
            candidate.offset_bytes
        ),

        offset_source=(
            candidate.offset_name
        ),

        rows=candidate.rows,
        columns=candidate.columns,

        pixels_per_plane=(
            candidate.pixels_per_plane
        ),

        bytes_per_plane=(
            candidate.bytes_per_plane
        ),

        plane_count=plane_count,

        file_size_bytes=(
            candidate.file_size_bytes
        ),

        bytes_after_offset=(
            candidate.bytes_after_offset
        ),
    )

def _numeric_diagnostic(
    values: np.ndarray,
    endian: str,
    dtype_name: str,
) -> EndianDiagnostic:

    values = np.asarray(values)

    finite = np.isfinite(values)

    finite_ratio = float(
        finite.mean()
    )

    finite_values = values[
        finite
    ]

    if finite_values.size == 0:

        return EndianDiagnostic(
            endian=endian,
            dtype=dtype_name,

            sample_count=int(
                values.size
            ),

            finite_ratio=finite_ratio,
            nonnegative_ratio=0.0,
            zero_ratio=0.0,

            very_small_ratio=0.0,
            very_large_ratio=0.0,

            minimum=None,
            maximum=None,
            median=None,
            median_abs=None,
            p99_abs=None,
        )

    absolute = np.abs(
        finite_values
    )

    nonnegative_ratio = float(
        (
            finite_values >= 0
        ).mean()
    )

    zero_ratio = float(
        (
            finite_values == 0
        ).mean()
    )

    # Apenas indicadores diagnósticos.
    # Não representam limites físicos do SAR.
    very_small_ratio = float(
        (
            (absolute > 0)
            & (absolute < 1e-30)
        ).mean()
    )

    very_large_ratio = float(
        (
            absolute > 1e30
        ).mean()
    )

    return EndianDiagnostic(
        endian=endian,
        dtype=dtype_name,

        sample_count=int(
            values.size
        ),

        finite_ratio=finite_ratio,

        nonnegative_ratio=(
            nonnegative_ratio
        ),

        zero_ratio=zero_ratio,

        very_small_ratio=(
            very_small_ratio
        ),

        very_large_ratio=(
            very_large_ratio
        ),

        minimum=float(
            np.min(finite_values)
        ),

        maximum=float(
            np.max(finite_values)
        ),

        median=float(
            np.median(finite_values)
        ),

        median_abs=float(
            np.median(absolute)
        ),

        p99_abs=float(
            np.percentile(
                absolute,
                99,
            )
        ),
    )

def diagnose_endianness(
    path: str | Path,
    layout: ValidatedLayout,
    plane_index: int = 0,
) -> list[EndianDiagnostic]:

    path = Path(path)

    if plane_index < 0:
        raise ValueError(
            "plane_index deve ser >= 0."
        )

    if plane_index >= layout.plane_count:
        raise ValueError(
            "plane_index excede o número "
            "de planos validado."
        )

    plane_offset = (
        layout.offset_bytes
        + (
            plane_index
            * layout.bytes_per_plane
        )
    )

    results = []

    for (
        endian_name,
        dtype,
    ) in [
        (
            "big",
            np.dtype(">f4"),
        ),
        (
            "little",
            np.dtype("<f4"),
        ),
    ]:

        with path.open("rb") as file:

            file.seek(
                plane_offset
            )

            values = np.fromfile(
                file,
                dtype=dtype,
                count=(
                    layout.pixels_per_plane
                ),
            )

        if (
            values.size
            != layout.pixels_per_plane
        ):
            raise IOError(
                "Número inesperado de valores "
                f"lidos para {endian_name}-endian."
            )

        results.append(
            _numeric_diagnostic(
                values,
                endian=endian_name,
                dtype_name=str(dtype),
            )
        )

    return results

def read_float32_plane(
    path: str | Path,
    layout: ValidatedLayout,
    plane_index: int,
    endian: str,
) -> np.ndarray:

    path = Path(path)

    if endian == "big":
        dtype = np.dtype(">f4")

    elif endian == "little":
        dtype = np.dtype("<f4")

    else:
        raise ValueError(
            "endian deve ser 'big' ou 'little'."
        )

    if (
        plane_index < 0
        or plane_index >= layout.plane_count
    ):
        raise ValueError(
            f"plane_index inválido: {plane_index}"
        )

    plane_offset = (
        layout.offset_bytes
        + plane_index
        * layout.bytes_per_plane
    )

    with path.open("rb") as file:

        file.seek(plane_offset)

        values = np.fromfile(
            file,
            dtype=dtype,
            count=layout.pixels_per_plane,
        )

    if values.size != layout.pixels_per_plane:
        raise IOError(
            "Leitura incompleta do plano."
        )

    array = values.reshape(
        (
            layout.rows,
            layout.columns,
        )
    )

    array = array.astype(
        np.float32,
        copy=False,
    )

    return array
