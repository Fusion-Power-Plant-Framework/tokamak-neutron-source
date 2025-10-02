# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Neutron energy spectrum calculations.
"""

from enum import Enum, auto
import logging

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from tokamak_neutron_source.reactions import Reactions

logger = logging.getLogger(__name__)


class EnergySpectrumMethod(Enum):
    """Energy spectrum calculation method."""

    DATA = auto()
    BALLABIO = auto()


def energy_spectrum(
    temp_kev: float,
    reaction: Reactions,
    method: EnergySpectrumMethod = EnergySpectrumMethod.BALLABIO,
) -> tuple[npt.NDArray, npt.NDArray]:
    """ """

    match method:
        case EnergySpectrumMethod.BALLABIO:
            if reaction.ballabio_spectrum is not None:
                return _gaussian_energy_spectrum(
                    reaction.ballabio_spectrum.mean_energy(temp_kev),
                    reaction.ballabio_spectrum.std_deviation(temp_kev),
                )

            logger.warning(
                f"There is no Ballabio parameterisation for reaction {reaction.name}, "
                "returning energy spectrum calculated by data."
            )
        case EnergySpectrumMethod.DATA:
            raise NotImplementedError


def _gaussian_energy_spectrum(mu: float, sigma: float):
    energy = np.linspace(0, 20, 1000)  # [keV]
    pdf = norm.pdf(energy, loc=mu, scale=sigma)
    probability = pdf / np.trapz(pdf, energy)
    return energy, probability
