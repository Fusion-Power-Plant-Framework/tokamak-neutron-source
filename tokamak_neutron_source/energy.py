# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Neutron energy spectrum calculations.
"""

import logging
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from tokamak_neutron_source.energy_data import BallabioEnergySpectrum
from tokamak_neutron_source.reactions import Reactions
from tokamak_neutron_source.tools import trapezoid

logger = logging.getLogger(__name__)


class EnergySpectrumMethod(Enum):
    """Energy spectrum calculation method."""

    DATA = auto()
    BALLABIO_GAUSSIAN = auto()
    BALLABIO_M_GAUSSIAN = auto()


def energy_spectrum(
    temp_kev: float,
    reaction: Reactions,
    method: EnergySpectrumMethod = EnergySpectrumMethod.BALLABIO_M_GAUSSIAN,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate the tabulated energy spectrum of a reaction at a given ion temperature.

    Parameters
    ----------
    temp_kev:
        Ion temperature
    reaction:
        Neutronic fusion reaction
    method:
        Method to use to calculate the energy spectrum

    Returns
    -------
    energies:
        The energy bins of the probability distribution function
    probabilities:
        The probabilities
    """
    match method:
        case (
            EnergySpectrumMethod.BALLABIO_GAUSSIAN
            | EnergySpectrumMethod.BALLABIO_M_GAUSSIAN
        ):
            if reaction.ballabio_spectrum is not None:
                return _ballabio_spectrum(reaction.ballabio_spectrum, temp_kev, method)

            logger.warning(
                f"There is no Ballabio parameterisation for reaction {reaction.name}, "
                "returning energy spectrum calculated by data."
            )
            raise NotImplementedError

        case EnergySpectrumMethod.DATA:
            raise NotImplementedError


def _ballabio_spectrum(
    spectrum: BallabioEnergySpectrum, temp_kev: float, method=EnergySpectrumMethod
):
    mean_energy = spectrum.mean_energy(temp_kev)
    std_deviation = spectrum.std_deviation(temp_kev)

    match method:
        case EnergySpectrumMethod.BALLABIO_GAUSSIAN:
            return _gaussian_energy_spectrum(mean_energy, std_deviation)
        case EnergySpectrumMethod.BALLABIO_M_GAUSSIAN:
            return _modified_gaussian_energy_spectrum(mean_energy, std_deviation)


def _gaussian_energy_spectrum(
    mu: float, sigma: float
) -> tuple[npt.NDArray, npt.NDArray]:
    energy = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 1000)  # [keV]
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((energy - mu) ** 2) / (2 * sigma**2)
    )

    mask = pdf > 0
    pdf = pdf[mask]
    energy = energy[mask]
    probability = pdf / trapezoid(pdf, energy)
    return energy, probability


def _modified_gaussian_energy_spectrum(
    mu: float, sigma: float
) -> tuple[float | npt.NDArray, ...]:
    """
    Modified Gaussian spectrum (from e.g. Ballabio et al., 1998)

    Returns
    -------
    energy:
        MATTI TODO
    probablity:
        MATTI TODO

    Notes
    -----
    Eqs. 46, 47
    """
    energy = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 1000)  # [keV]

    factor = 1.0 - 1.5 * (sigma / mu) ** 2
    sqrt_factor = np.sqrt(factor)
    e_bar = mu * sqrt_factor
    sigma_sq = 4.0 / 3.0 * mu**2 * (sqrt_factor - factor)

    pdf = np.exp(-2.0 * e_bar * (np.sqrt(energy) - np.sqrt(e_bar)) ** 2 / sigma_sq)
    mask = pdf > 0
    pdf = pdf[mask]
    energy = energy[mask]
    probability = pdf / trapezoid(pdf, energy)
    return energy, probability
