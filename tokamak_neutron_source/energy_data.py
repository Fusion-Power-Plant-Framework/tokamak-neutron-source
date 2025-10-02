# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Neutron energy spectrum calculations.
"""

from dataclasses import dataclass

import numpy.typing as npt
import numpy as np

# 2.0 * np.sqrt(2.0 * np.log(2))
TWO_SQRT_2LN2 = 2.3548200450309493


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
    Ballabio et al. fit data for relativistic fusion reaction neutron energy Gaussian spectra.
    """

    """E_0"""
    energy_0: float  # [keV]

    omega_0: float  # [keV]

    """\Delta E_{th} coefficients"""
    energy_shift_coeffs: BallabioCoefficients

    """\delta_{\omega} coefficients"""
    width_correction_coeffs: BallabioCoefficients

    def energy_shift(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the energy shift \Delta E_{th} at a given ion temperature.
        """
        return ballabio_fit(temp_kev, self.energy_shift_coeffs)

    def width_correction(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the width correction \delta_{\omega} at a given ion temperature.
        """
        return ballabio_fit(temp_kev, self.width_correction_coeffs)

    def mean_energy(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the mean neutron energy at a given ion temperature (primary first moment: mu).
        """
        return self.energy_0 + self.energy_shift(temp_kev)

    def std_deviation(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the standard deviation of the neutron energy spectrum at a given ion
        temperature (primary second moment: sigma)
        """
        # Full width at half maximum
        w_12 = self.omega_0 * (1 + self.width_correction(temp_kev)) * np.sqrt(temp_kev)
        return w_12 / TWO_SQRT_2LN2

    def spectrum(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Modified Gaussian spectrum

        Notes
        -----
        This does not look right... but it is what the paper says.
        """
        e_mean = self.mean_energy(temp_kev)
        sigma_th = self.std_deviation(temp_kev)
        factor = 1.0 - 1.5 * (sigma_th / e_mean) ** 2
        sqrt_factor = np.sqrt(factor)
        e_bar = e_mean * sqrt_factor
        sigma_sq = 4.0 / 3.0 * e_mean**2 * (sqrt_factor - factor)
        norm = 1.0 / denominator_D(e_mean, sigma_sq**0.5)
        return norm * np.exp(
            -2.0 * e_bar * (np.sqrt(temp_kev) - np.sqrt(e_bar)) ** 2 / sigma_sq
        )


from scipy.special import erf


def denominator_D(Ebar: float, sigma: float) -> float:
    """
    Compute the denominator D for the modified Gaussian distribution. Guessing here.

        I(E) = I0 * exp( - (2*Ebar/sigma^2) * (sqrt(E) - sqrt(Ebar))^2 )

    where I0 is fixed by requiring that the total area integrates to S.

    Parameters
    ----------
    Ebar : float
        Mean energy (Ē).
    sigma : float
        Width parameter σ.

    Returns
    -------
    float
        The denominator D.
    """
    term1 = (sigma**2) / (2.0 * Ebar) * np.exp(-2.0 * (Ebar**2) / (sigma**2))
    term2 = sigma * np.sqrt(np.pi / 2.0) * (1.0 + erf(np.sqrt(2.0) * Ebar / sigma))
    return term1 + term2


BALLABIO_DT_NEUTRON = BallabioEnergySpectrum(
    energy_0=14021.0,
    omega_0=177.259,
    energy_shift_coeffs=BallabioCoefficients(
        a1=5.30509,
        a2=2.4736e-3,
        a3=1.84,
        a4=1.3818,
    ),
    width_correction_coeffs=BallabioCoefficients(
        a1=5.1068e-4,
        a2=7.6223e-3,
        a3=1.78,
        a4=8.7691e-5,
    ),
)

BALLABIO_DD_NEUTRON = BallabioEnergySpectrum(
    energy_0=2.4495e3,
    omega_0=82.542,
    energy_shift_coeffs=BallabioCoefficients(
        a1=4.69515,
        a2=-0.040729,
        a3=0.47,
        a4=0.81844,
    ),
    width_correction_coeffs=BallabioCoefficients(
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
