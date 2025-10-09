# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Transport data structures."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import astuple, dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tokamak_neutron_source.profile import DataProfile, PlasmaProfile
from tokamak_neutron_source.tools import load_jsp

logger = logging.getLogger(__name__)


@dataclass
class FractionalFuelComposition:
    """
    Fractional fuel composition dataclass.

    Notes
    -----
    Fuel fractions are taken to be constant along the profile.
    Note that the D-He-3 reaction is aneutronic, but dilutes the fuel
    in the case that it is included in the fuel density profile.
    """

    """Deuterium fuel fraction"""
    D: float

    """Tritium fuel fraction"""
    T: float

    """Helium-3 fuel fraction"""
    He3: float = 0.0

    def __post_init__(self):
        """Force fractions to sum to 1.0."""
        if not np.equal(sum(astuple(self)), 1.0):
            norm = sum(astuple(self))
            self.D, self.T, self.He3 = self.D / norm, self.T / norm, self.He3 / norm
            logger.warning(
                f"Fuel fraction has been renormalized to: {self}",
                stacklevel=1,
            )


DT_5050_MIXTURE = FractionalFuelComposition(D=0.5, T=0.5, He3=0.0)


@dataclass
class TransportInformation:
    """Transport information."""

    deuterium_density_profile: PlasmaProfile  # [1/m^3]
    tritium_density_profile: PlasmaProfile  # [1/m^3]
    helium3_density_profile: PlasmaProfile  # [1/m^3]
    temperature_profile: PlasmaProfile  # [keV]
    rho_profile: npt.NDArray  # [0..1]

    @classmethod
    def from_profiles(
        cls,
        ion_temperature_profile: np.ndarray,
        fuel_density_profile: np.ndarray,
        rho_profile: np.ndarray,
        fuel_composition: FractionalFuelComposition = DT_5050_MIXTURE,
    ) -> TransportInformation:
        """
        Instantiate TransportInformation from profile arrays.

        Parameters
        ----------
        ion_temperature_profile:
            Ion temperature profile [keV]
        fuel_density_profile:
            Fuel density profile [1/m^3]
        rho_profile:
            Normalised radial coordinate profile
        fuel_composition
            Fractional fuel composition (constant fraction across profile)

        """  # noqa: DOC201
        return cls(
            DataProfile(
                fuel_composition.D * fuel_density_profile,
                rho_profile,
            ),
            DataProfile(
                fuel_composition.T * fuel_density_profile,
                rho_profile,
            ),
            DataProfile(
                fuel_composition.He3 * fuel_density_profile,
                rho_profile,
            ),
            DataProfile(ion_temperature_profile, rho_profile),
            np.asarray(rho_profile),
        )

    @classmethod
    def from_parameterisations(
        cls,
        ion_temperature_profile: PlasmaProfile,
        fuel_density_profile: PlasmaProfile,
        rho_profile: npt.NDArray,
        fuel_composition: FractionalFuelComposition = DT_5050_MIXTURE,
    ) -> TransportInformation:
        """
        Instantiate TransportInformation from profile parameterisations.

        Parameters
        ----------
        ion_temperature_profile:
            Ion temperature profile parameterisation
        fuel_density_profile:
            Fuel density profile parameterisation
        rho_profile:
            Noramlised radial coordinate profile
        fuel_composition
            Fractional fuel composition (constant fraction across profile)

        """  # noqa: DOC201
        d_profile = deepcopy(fuel_density_profile)
        d_profile.set_scale(fuel_composition.D)
        t_profile = deepcopy(fuel_density_profile)
        t_profile.set_scale(fuel_composition.T)
        he3_profile = deepcopy(fuel_density_profile)
        he3_profile.set_scale(fuel_composition.He3)

        return cls(
            d_profile,
            t_profile,
            he3_profile,
            ion_temperature_profile,
            rho_profile,
        )

    @classmethod
    def from_jetto(cls, jsp_file: str, frame_number: int = -1) -> TransportInformation:
        """
        Instantiate TransportInformation from JETTO file.

        Parameters
        ----------
        jsp_file:
            Path to the JETTO .jsp file
        frame_number:
            The specific time-slice of the JETTO run that we want to investigate.
            This ensures that all of the extracted quantities are describing the same
            point in time.
        """  # noqa: DOC201
        data = load_jsp(jsp_file, frame_number)

        return cls(
            DataProfile(
                data.d_density,
                data.rho,
            ),
            DataProfile(
                data.t_density,
                data.rho,
            ),
            DataProfile(
                data.he3_density,
                data.rho,
            ),
            DataProfile(data.ion_temperature, data.rho),
            data.rho,
        )

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the TransportInformation

        Returns
        -------
        f:
            Matplotlib Figure object
        ax:
            Matplotlib Axes object
        """
        f, ax = plt.subplots()

        d_d = self.deuterium_density_profile.value(self.rho_profile)
        d_t = self.tritium_density_profile.value(self.rho_profile)
        d_he3 = self.helium3_density_profile.value(self.rho_profile)

        for d, label, ls in zip(
            [d_d, d_t, d_he3], ["D", "T", "³He"], ["-.", "--", "."], strict=True
        ):
            if not np.allclose(d, 0.0):
                ax.plot(self.rho_profile, d, ls=ls, label=label)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$n$ [1/m$^{3}$]")
        ax.legend(loc="lower left")
        ax2 = ax.twinx()
        ax2.plot(
            self.rho_profile,
            self.temperature_profile.value(self.rho_profile),
            label=r"$T_{i}$",
            color="r",
        )
        ax2.set_ylabel(r"$T_{i}$ [keV]")
        ax2.legend(loc="upper right")
        plt.show()
        return f, ax
