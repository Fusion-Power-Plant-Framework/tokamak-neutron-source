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
    cumulative_neutron_rate: npt.NDArray | None = None  # [1/s]

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
    def from_jetto(
        cls, 
        jsp_file: str | Path,
        frame_number: int = -1
    ) -> TransportInformation:
        """
        Instantiate TransportInformation from jetto file.

        For details, refer to
        https://users.euro-fusion.org/pages/data-cmg/wiki/JETTO_ppfjsp.html

        Parameters
        ----------
        jsp_file: 
            Path to the JETTO .jsp file
        frame_number:
            The specific time-slice of the JETTO run that we want to investigate.
            This ensures that all of the extracted quantities are describing the same
            point in time.

        Effects
        -------
        self.time_stamps
            times when the snapshots are made [s]

        self.magnetic_flux
            magnetic flux value (Wb/2pi), used as the x-values when interpolating.

        self.ion_temperature
            ion temperature (D&T) profiles [keV]

        self.d_density
            D-density profiles [number of ions cm^-3]

        self.t_density
            T-density profiles [number of ions cm^-3]

        self.specific_fusion_rate
            (i.e. number of this fusion reaction per volume at the given psi.)
            dataclass that includes the following:
            1. specific D+T->4He+n fusion rate [cm^-3 s^-1]
            2. specific D+D->3He+n fusion rate [cm^-3 s^-1]
            3. specific D+D->T+p fusion rate [cm^-3 s^-1]
            4. specific T+T->4He+n+n fusion rate [cm^-3 s^-1]
            For reaction 3, it is Copied from reaction 2, since they are theoretically
            the same; and JETTO does not provide this data.

        self.specific_fusion_rate_contributions
            Reactions 1,2,4 listed in self.specific_fusion_rate, broken down into three
            components: thermal, beam-plasma, and RF-enhanced. The sum of these three
            components should give the total fusion rate of that fusion reaction.

        self.cumulative_fusion_rate
            C.D.F. of fusion rate rate w.r.t to closed-flux surfaces (indexed by Psi),
            summed over the entire plasma. This means that the cumulative_fusion_rate
            at Psi=x gives the total fusion rate [s^-1] of all of the volume of plasma
            where Psi>=x. In other words it gives the total fusion rate [s^-1] of the
            plasma enclosed by the closed flux surface Psi=x.

            This is cumulated from the plasma center towards the LCFS.

        self.cumulative_neutron_rate
            C.D.F. of neutron production rate w.r.t to closed-flux surfaces (indexed by
            Psi),summed over the entire plasma. This means that the
            cumulative_neutron_rate at Psi=x gives the
            total neutron production rate [s^-1] of all of the volume of plasma
            where Psi>=x. In other words it gives the
            total neutron production rate [s^-1] of the
            plasma enclosed by the closed flux surface Psi=x.

            This is cumulated from the plasma center towards the LCFS.


        Notes
        -----
        JETTO is also missing the He-3 density/ any data that can be used to calculate/
        indicate the D + 3He -> 4He + p reaction rate. This is a fundamental shortcoming
        of JETTO that we cannot solve at this stage.
        """
        from jetto_tools import binary 

        jsp = binary.read_binary_file(jsp_file)

         # time records
        time_stamps = jsp["TIME"][:, 0, 0]  # times when the snapshots are made
        frame_number = len(time_stamps) - 1 if frame_number == -1 else frame_number
        t = frame_number
        snapshot_time = time_stamps[t]

        # psi values, acting as the abscissa/x-axis for the interpolations below.
        magnetic_flux = jsp["XPSQ"][t, :]  # Sqrt(poloidal magnetic flux)
        # force magnetic flux to be nonnegative
        #magnetic_flux = np.sign(magnetic_flux.mean()) * magnetic_flux
        ## clamp every data point on the wrong side of the number line back to 0.
        #out_of_range_datapoints = np.sign(magnetic_flux) != +1
        #if out_of_range_datapoints.sum() > 1:
        #    warnings.warn(
        #        "More than one out of range datapoints, will result in degenerate "
        #        "data points after clamping back into range!",
        #        OutOfRangeWarning,
        #        stacklevel=2,
        #    )
#
        #
        #magnetic_flux[out_of_range_datapoints] = 0.0
        magnetic_flux = np.insert(magnetic_flux, 0, 0.0) 


        # Ordinate/y-values to be interpolated w.r.t. different values of psi.
        # ion temperature (D&T) profiles [keV]
        ion_temperature = np.insert(jsp["TI"][t, :] * 1e-3, 0, jsp["TI"][t, 0]*1e-3)  # [eV] -> [keV]
        # D-density profiles [number of ions m^-3]
        d_density = np.insert(jsp["NID"][t, :], 0, jsp["NID"][t, 0])
        # T-density profiles [number of ions m^-3]
        t_density = np.insert(jsp["NIT"][t, :], 0, jsp["NIT"][t, 0])
        # ion-density (of all thermalized ions) profiles [number of ions m^-3]
        ion_density = np.insert(jsp["NI"][t, :], 0, jsp["NIT"][t, 0])


        # fusion rates, looks like it only includes DT-thermal[s^-1] (C.D.F. w.r.t. psi)
        cumulative_fusion_rate = jsp["R00"][t, :]
        cumulative_fusion_rate = np.insert(cumulative_fusion_rate, 0, 0)
        # neutron production rate. [s^-1]  (C.D.F. w.r.t. psi)
        cumulative_neutron_rate = jsp["NT"][t, :]
        cumulative_neutron_rate = np.insert(cumulative_neutron_rate, 0, 0)
        # [t, :] is a C.D.F series, so [t, -1] gives the total

        return cls(
            DataProfile(
                d_density,
                magnetic_flux,
            ),
            DataProfile(
                t_density,
                magnetic_flux,
            ),
            DataProfile(
                ion_density*0.0, # JETTO does not provide He-3 density
                magnetic_flux,
            ),
            DataProfile(
                ion_temperature, 
                magnetic_flux
            ),
            np.asarray(magnetic_flux),
            np.asarray(cumulative_neutron_rate),
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
