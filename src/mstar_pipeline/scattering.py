from __future__ import annotations

import math

import numpy as np


def validate_parameters(
    n_scatterers: int,
    nu: float,
) -> None:
    """
    Validate coherent-scatterer model parameters.
    """

    if not isinstance(
        n_scatterers,
        (int, np.integer),
    ):
        raise TypeError(
            "n_scatterers must be an integer."
        )

    if n_scatterers < 1:
        raise ValueError(
            "n_scatterers must be >= 1."
        )

    if not np.isfinite(nu):
        raise ValueError(
            "nu must be finite."
        )

    if nu <= -1.0:
        raise ValueError(
            "nu must be > -1."
        )


def model_M(
    n_scatterers: int,
    nu: float,
) -> float:
    """
    Di Martino et al.:

        M = N * (nu + 1)
    """

    validate_parameters(
        n_scatterers,
        nu,
    )

    return float(
        n_scatterers
        * (nu + 1.0)
    )


def generate_complex_gaussian(
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate circular complex Gaussian samples
    with unit mean intensity:

        E[|X|^2] = 1

    Real and imaginary components therefore have
    variance 1/2.
    """

    real = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(2.0),
        size=size,
    )

    imag = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(2.0),
        size=size,
    )

    signal = (
        real
        + 1j * imag
    )

    return np.asarray(
        signal,
        dtype=np.complex128,
    )


def generate_equivalent_scatterer(
    size: int | tuple[int, ...],
    n_scatterers: int,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one equivalent-scatterer contribution.

    Unit-mean reparameterization of the
    Gamma-Gaussian scale-mixture model.

    For the complete coherent sum:

        M = N * (nu + 1)

    Each scatterer's texture is:

        Z_k ~ Gamma(
            shape = nu + 1,
            scale = 1 / M
        )

    and its complex contribution is:

        Y_k = sqrt(Z_k) * X_k

    where:

        X_k ~ circular complex Gaussian
        E[|X_k|^2] = 1
    """

    M = model_M(
        n_scatterers,
        nu,
    )

    gamma_shape = (
        nu + 1.0
    )

    gamma_scale = (
        1.0 / M
    )

    texture = rng.gamma(
        shape=gamma_shape,
        scale=gamma_scale,
        size=size,
    )

    gaussian = (
        generate_complex_gaussian(
            size=size,
            rng=rng,
        )
    )

    scatterer = (
        np.sqrt(texture)
        * gaussian
    )

    return np.asarray(
        scatterer,
        dtype=np.complex128,
    )


def generate_coherent_scatterer_field(
    size: int | tuple[int, ...],
    n_scatterers: int,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a coherent sum of N independent
    equivalent scatterers:

        E = sum(Y_k),  k = 1,...,N

    The parameter:

        M = N * (nu + 1)

    controls the resulting intensity statistics.

    The model is normalized so that:

        E[|E|^2] = 1
    """

    validate_parameters(
        n_scatterers,
        nu,
    )

    field = np.zeros(
        size,
        dtype=np.complex128,
    )

    for _ in range(
        n_scatterers
    ):

        field += (
            generate_equivalent_scatterer(
                size=size,
                n_scatterers=n_scatterers,
                nu=nu,
                rng=rng,
            )
        )

    return field


def generate_fully_developed_speckle(
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fully developed speckle reference.

    Circular complex Gaussian with unit
    mean intensity.

    Therefore:

        I = |E|^2

    follows an exponential distribution
    with mean 1.
    """

    return generate_complex_gaussian(
        size=size,
        rng=rng,
    )


def intensity_from_complex(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Compute intensity:

        I = |E|^2
    """

    signal = np.asarray(
        signal
    )

    intensity = (
        np.abs(signal) ** 2
    )

    return np.asarray(
        intensity,
        dtype=np.float64,
    )


def empirical_normalized_moment(
    intensity: np.ndarray,
    order: int,
) -> float:
    """
    Empirical normalized moment:

        E[I^n] / E[I]^n
    """

    if order < 1:
        raise ValueError(
            "order must be >= 1."
        )

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    if intensity.size == 0:
        raise ValueError(
            "intensity must not be empty."
        )

    if not np.isfinite(
        intensity
    ).all():
        raise ValueError(
            "intensity contains "
            "non-finite values."
        )

    mean_intensity = float(
        np.mean(
            intensity
        )
    )

    if mean_intensity <= 0.0:
        raise ValueError(
            "mean intensity must be > 0."
        )

    raw_moment = float(
        np.mean(
            intensity ** order
        )
    )

    return (
        raw_moment
        / (
            mean_intensity
            ** order
        )
    )


def theoretical_normalized_moment(
    order: int,
    n_scatterers: int,
    nu: float,
) -> float:
    """
    Theoretical normalized intensity moment
    from Di Martino et al., Eq. (15):

        m_n =
            n! * Gamma(M + n)
            -----------------
            Gamma(M) * M^n

    with:

        M = N * (nu + 1)
    """

    if order < 1:
        raise ValueError(
            "order must be >= 1."
        )

    M = model_M(
        n_scatterers,
        nu,
    )

    log_moment = (
        math.lgamma(
            order + 1
        )
        + math.lgamma(
            M + order
        )
        - math.lgamma(
            M
        )
        - order
        * math.log(
            M
        )
    )

    return float(
        math.exp(
            log_moment
        )
    )


def theoretical_fds_normalized_moment(
    order: int,
) -> float:
    """
    Fully developed speckle:

        normalized moment = n!
    """

    if order < 1:
        raise ValueError(
            "order must be >= 1."
        )

    return float(
        math.factorial(
            order
        )
    )