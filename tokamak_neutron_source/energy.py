# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Neutron energy spectrum calculations.
"""

from dataclasses import dataclass

import numpy.typing as npt
import numpy as np


@dataclass
class BallabioCoefficients:
    """
    Ballabio et al. fit parameterisation coefficients for Ti < 40.0 keV
    """

    a1: float
    a2: float
    a3: float
    a4: float


@dataclass
class BallabioEnergySpectrum:
    """
    Ballabio et al. fit data for relativistic fusion reaction neutron energy spectra.
    """

    """E_0"""
    energy_0: float  # [keV]

    omega_0: float  # [keV]

    varepsilon_0: float  # [kev^0.5]

    """\Delta E_{th}"""
    energy_shift: BallabioCoefficients

    """\delta_{\omega}"""
    energy_correction: BallabioCoefficients


BALLABIO_DT_NEUTRON = BallabioEnergySpectrum(
    energy_0=14021.0,
    omega_0=177.259,
    varepsilon_0=8.113e-3,
    energy_shift=BallabioCoefficients(
        a1=5.30509,
        a2=2.4736e-3,
        a3=1.84,
        a4=1.3818,
    ),
    energy_correction=BallabioCoefficients(
        a1=5.1068e-4,
        a2=7.6223e-3,
        a3=1.78,
        a4=8.7691e-5,
    ),
)

BALLABIO_DD_NEUTRON = BallabioEnergySpectrum(
    energy_0=2.4495e3,
    omega_0=82.542,
    varepsilon_0=2.149e-2,
    energy_shift=BallabioCoefficients(
        a1=4.69515,
        a2=-0.040729,
        a3=0.47,
        a4=0.81844,
    ),
    energy_correction=BallabioCoefficients(
        a1=1.7013e-3,
        a2=0.16888,
        a3=0.49,
        a4=7.9460e-4,
    ),
)


def ballabio_fit(
    temp_kev: float | npt.NDArray, data: BallabioCoefficients
) -> float | npt.NDArray:
    """

    Notes
    -----
    Valid over 0.0 to 40.0 keV
    """
    return (
        data.a1 / (1 + data.a2 * temp_kev**data.a3) * temp_kev ** (2 / 3)
        + data.a4 * temp_kev
    )
