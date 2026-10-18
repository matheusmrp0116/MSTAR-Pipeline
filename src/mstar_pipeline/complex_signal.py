from __future__ import annotations

import numpy as np


TWO_PI = 2.0 * np.pi


def magnitude_phase_to_complex(
    magnitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """
    Converte magnitude + phase em sinal complexo.

    S = A * exp(j * phi)

    Retorna complex64.
    """

    magnitude = np.asarray(
        magnitude,
        dtype=np.float32,
    )

    phase = np.asarray(
        phase,
        dtype=np.float32,
    )

    if magnitude.shape != phase.shape:
        raise ValueError(
            "Magnitude and phase must have the same shape."
        )

    if not np.isfinite(magnitude).all():
        raise ValueError(
            "Magnitude contains non-finite values."
        )

    if not np.isfinite(phase).all():
        raise ValueError(
            "Phase contains non-finite values."
        )

    if (magnitude < 0).any():
        raise ValueError(
            "Magnitude contains negative values."
        )

    real = (
        magnitude
        * np.cos(phase)
    )

    imag = (
        magnitude
        * np.sin(phase)
    )

    signal = (
        real
        + 1j * imag
    )

    return np.asarray(
        signal,
        dtype=np.complex64,
    )


def complex_to_magnitude_phase(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recupera magnitude e fase do sinal complexo.

    A = |S|
    phi = angle(S)

    A fase retornada por np.angle está em [-pi, pi].
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64,
    )

    if not (
        np.isfinite(signal.real).all()
        and np.isfinite(signal.imag).all()
    ):
        raise ValueError(
            "Complex signal contains non-finite values."
        )

    magnitude = np.abs(
        signal
    ).astype(
        np.float32
    )

    phase = np.angle(
        signal
    ).astype(
        np.float32
    )

    return magnitude, phase


def circular_phase_difference(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
) -> np.ndarray:
    """
    Calcula diferença angular mínima.

    Resultado no intervalo [-pi, pi].
    """

    phase_a = np.asarray(
        phase_a,
        dtype=np.float64,
    )

    phase_b = np.asarray(
        phase_b,
        dtype=np.float64,
    )

    difference = np.angle(
        np.exp(
            1j * (
                phase_a
                - phase_b
            )
        )
    )

    return difference


def validate_complex_roundtrip(
    magnitude: np.ndarray,
    phase: np.ndarray,
) -> dict:
    """
    Executa:

    magnitude + phase
          -> complex
          -> magnitude + phase

    e calcula os erros numéricos.

    A fase é avaliada apenas nos pixels
    cuja magnitude é maior que zero.
    """

    signal = magnitude_phase_to_complex(
        magnitude,
        phase,
    )

    recovered_magnitude, recovered_phase = (
        complex_to_magnitude_phase(
            signal
        )
    )

    magnitude_reference = np.asarray(
        magnitude,
        dtype=np.float32,
    )

    phase_reference = np.asarray(
        phase,
        dtype=np.float32,
    )

    magnitude_error = np.abs(
        recovered_magnitude
        - magnitude_reference
    )

    # A fase de um sinal complexo de magnitude
    # zero é matematicamente indefinida.
    phase_valid_mask = (
        magnitude_reference > 0
    )

    if phase_valid_mask.any():

        phase_error = np.abs(
            circular_phase_difference(
                recovered_phase[
                    phase_valid_mask
                ],
                phase_reference[
                    phase_valid_mask
                ],
            )
        )

    else:

        phase_error = np.array(
            [0.0],
            dtype=np.float64,
        )

    return {
        "complex_dtype":
            str(signal.dtype),

        "magnitude_max_abs_error":
            float(
                np.max(
                    magnitude_error
                )
            ),

        "magnitude_mean_abs_error":
            float(
                np.mean(
                    magnitude_error
                )
            ),

        "phase_max_circular_error":
            float(
                np.max(
                    phase_error
                )
            ),

        "phase_mean_circular_error":
            float(
                np.mean(
                    phase_error
                )
            ),

        "phase_valid_pixels":
            int(
                phase_valid_mask.sum()
            ),

        "phase_undefined_pixels":
            int(
                (~phase_valid_mask).sum()
            ),
    }