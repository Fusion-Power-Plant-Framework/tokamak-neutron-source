# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Neutron energy spectrum data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from tokamak_neutron_source.constants import raw_uc
from tokamak_neutron_source.error import EnergySpectrumError
from tokamak_neutron_source.tools import get_tns_path

logger = logging.getLogger(__name__)

# 2.0 * np.sqrt(2.0 * np.log(2))
TWO_SQRT_2LN2 = 2.3548200450309493


class TTNeutronEnergyDataSpectrum:
    """Fusion neutron energy data spectrum."""

    def __init__(self, file_name: str):
        """
        Parameters
        ----------
        file_name:
            Data file

        Raises
        ------
        EnergySpectrumError
            Data file path is not a file
        """
        path = get_tns_path("data/spectra")
        path = Path(path, file_name)
        if not path.is_file():
            raise EnergySpectrumError(f"Energy spectrum data file {path} is not a file!")

        file = path.as_posix()
        data = np.genfromtxt(file, comments="#")

        # The energy bins for this spectrum are hard-coded here
        # (no sense in interpolating).
        self._energy = raw_uc(data[:, 0], "MeV", "keV")

        self._min_temp = 1.0  # [keV]
        self._max_temp = 20.0  # [keV]
        temperature = np.linspace(self._min_temp, self._max_temp, 40)  # [keV]
        spectra = raw_uc(data[:, 1:], "1/MeV", "1/keV")
        self._interpolator = RegularGridInterpolator(
            (self._energy, temperature),
            spectra,
            method="linear",
            bounds_error=True,
            fill_value=np.nan,
        )

    def __call__(self, temp_kev: float) -> tuple[npt.NDArray, npt.NDArray]:
        """Get spectrum at a given temperature"""  # noqa: DOC201
        if not self._min_temp < temp_kev < self._max_temp:
            logger.warning(
                f"T-T spectral data not available at T = {temp_kev} keV, clipping back "
                f"to the bounds of {self._min_temp} <= T <= {self._max_temp} keV",
                stacklevel=2,
            )
            temp_kev = np.clip(temp_kev, self._min_temp, self._max_temp)

        return self._energy, self._interpolator((self._energy, temp_kev))


TT_N_SPECTRUM = TTNeutronEnergyDataSpectrum("T_T_spectra.txt")


@dataclass
class BallabioCoefficients:
    """
    Ballabio et al. fit parameterisation coefficients for Ti < 40.0 keV
    """

    a1: float
    a2: float
    a3: float
    a4: float

    def fit(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the value of the parameterisation at given temperature(s).

        Parameters
        ----------
        temp_kev:
            Ion temperatures at which to calculate the fit

        Returns
        -------
        :
            Values of the fit

        Notes
        -----
        Valid over 0.0 to 40.0 keV
        """
        return (
            self.a1 / (1 + self.a2 * temp_kev**self.a3) * temp_kev ** (2 / 3)
            + self.a4 * temp_kev
        )


@dataclass
class BallabioEnergySpectrum:
    """
    Ballabio et al. fit data for relativistic fusion reaction neutron energy Gaussian
    spectra.
    """

    """E_0"""
    energy_0: float  # [keV]

    omega_0: float  # [keV]

    r"""\Delta E_{th} coefficients"""
    energy_shift_coeffs: BallabioCoefficients

    r"""\delta_{\omega} coefficients"""
    width_correction_coeffs: BallabioCoefficients

    def energy_shift(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        r"""
        Calculate the energy shift \Delta E_{th} at a given ion temperature.
        """  # noqa: DOC201
        return self.energy_shift_coeffs.fit(temp_kev)

    def width_correction(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        r"""
        Calculate the width correction \delta_{\omega} at a given ion temperature.
        """  # noqa: DOC201
        return self.width_correction_coeffs.fit(temp_kev)

    def mean_energy(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the mean neutron energy at a given ion temperature
        (primary first moment: mu).
        """  # noqa: DOC201
        return self.energy_0 + self.energy_shift(temp_kev)

    def std_deviation(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the standard deviation of the neutron energy spectrum at a given ion
        temperature (primary second moment: sigma)
        """  # noqa: DOC201
        # Full width at half maximum (FWHM)
        w_12 = self.omega_0 * (1 + self.width_correction(temp_kev)) * np.sqrt(temp_kev)
        return w_12 / TWO_SQRT_2LN2


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
