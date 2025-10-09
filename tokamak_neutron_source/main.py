# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Neutromak user-facing API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tokamak_neutron_source.energy import EnergySpectrumMethod
from tokamak_neutron_source.error import ReactivityError, TNSError
from tokamak_neutron_source.reactions import (
    AllReactions,
    AneutronicReactions,
    Reactions,
    _parse_reaction,
)
from tokamak_neutron_source.reactivity import (
    ReactivityMethod,
    density_weighted_reactivity,
)
from tokamak_neutron_source.space import sample_space_2d

if TYPE_CHECKING:
    from openmc import IndependentSource

    from tokamak_neutron_source.flux import FluxMap
    from tokamak_neutron_source.transport import TransportInformation

logger = logging.getLogger(__name__)


CUSTOM_ORDER = [
    Reactions.D_T,
    Reactions.D_D,
    AneutronicReactions.D_D,
    Reactions.T_T,
    AneutronicReactions.D_He3,
]


def _parse_source_type(
    source_type: AllReactions | str | list[str | AllReactions] | None,
) -> list[AllReactions]:
    if source_type is None:
        source_type = [Reactions.D_D, Reactions.D_T, Reactions.T_T]
    elif isinstance(source_type, (str, AllReactions)):
        source_type = [_parse_reaction(source_type)]
    elif isinstance(source_type, list):
        source_type = [_parse_reaction(s) for s in source_type]
    else:
        raise ReactivityError(f"Unrecognised source type: {source_type}.")

    # If D-D specified, ensure both channels present
    if Reactions.D_D in source_type:
        source_type.append(AneutronicReactions.D_D)

    seen = set()
    unique = [x for x in source_type if not (x in seen or seen.add(x))]

    # Sort according to custom order, keep only subset
    ordering = {r: i for i, r in enumerate(CUSTOM_ORDER)}
    return sorted(unique, key=lambda r: ordering[r])


class TokamakNeutronSource:
    """
    Tokamak neutron source object.

    Parameters
    ----------
    transport:
        Plasma profile information and species composition.
    flux_map:
        Magneto-hydrodynamic equilibrium poloidal magnetic flux map containing LCFS
        geometry and the psi_norm inside of it.
    source_type:
        Which neutronic reaction(s) to include in the neutron source.
    cell_side_length:
        Discretisation in [m] for sampling the 2-D poloidal plane within the LCFS.
        Square cells of side length `cell_side_length` are created.
    total_fusion_power:
        If specified, will be used to normalise the source strength to a prescribed
        fusion power
    reactivity_method:
        Which method to use when calculating reactivities
    """

    def __init__(
        self,
        transport: TransportInformation,
        flux_map: FluxMap,
        source_type: AllReactions | list[AllReactions] | None = None,
        cell_side_length: float = 0.1,
        total_fusion_power: float | None = None,
        reactivity_method: ReactivityMethod = ReactivityMethod.BOSCH_HALE,
    ):
        self.source_type = _parse_source_type(source_type)

        self.xz, self.d_volume = sample_space_2d(
            flux_map.lcfs, flux_map.o_point, cell_side_length
        )
        self.dxdz = (cell_side_length, cell_side_length)
        psi_norm = flux_map.psi_norm(*self.xz.T)

        self.temperature = transport.temperature_profile.value(psi_norm)
        density_d = transport.deuterium_density_profile.value(psi_norm)
        density_t = transport.tritium_density_profile.value(psi_norm)
        density_he3 = transport.helium3_density_profile.value(psi_norm)

        # All reactions (neutronic and aneutronic)
        self.strength = {}
        for reaction in self.source_type:
            n1_n2_sigma = density_weighted_reactivity(
                self.temperature,
                density_d,
                density_t,
                density_he3,
                reaction=reaction,
                method=reactivity_method,
            )
            self.strength[reaction] = n1_n2_sigma * self.d_volume

        self.flux_map = flux_map
        self.transport = transport

        self.num_reactions_per_second = {}
        self.num_neutrons_per_second = {}
        if total_fusion_power is not None:
            self.normalise_fusion_power(total_fusion_power)

    def calculate_total_fusion_power(self) -> float:
        """
        Calculate the total fusion power from all reaction channels.

        Returns
        -------
        total_fusion_power
            The total fusion power

        Notes
        -----
        The aneutronic fusion reactions are included here.
        """
        return sum(
            np.sum(
                self.strength[reaction] * reaction.total_energy,
            )
            for reaction in self.source_type
        )

    def normalise_fusion_power(self, total_fusion_power: float):
        """
        Renormalise the source strength to match a total fusion power across all
        channels.

        Parameters
        ----------
        total_fusion_power:
            The total fusion power to normalise to [W]

        Notes
        -----
        This is done assuming the provided total fusion power is for the
        same channels as the source_type. The ratios of the strengths of
        each reaction is assumed to be same as modelled here.
        """
        actual_fusion_power = self.calculate_total_fusion_power()
        scaling_factor = total_fusion_power / actual_fusion_power
        self.num_reactions_per_second, self.num_neutrons_per_second = {}, {}
        for reaction in self.source_type:
            self.strength[reaction] *= scaling_factor
            self.num_reactions_per_second[reaction] = sum(self.strength[reaction])
            self.num_neutrons_per_second[reaction] = sum(self.strength[reaction]) * reaction.num_neutrons

    def to_openmc_source(
        self,
        energy_method: EnergySpectrumMethod = EnergySpectrumMethod.BALLABIO_M_GAUSSIAN,
    ) -> list[IndependentSource]:
        """
        Create an OpenMC tokamak neutron source.

        Parameters
        ----------
        energy_method:
            Which method to use when calculating neutron spectra

        Returns
        -------
        :
            List of native OpenMC source objects
        """
        from tokamak_neutron_source.openmc_interface import (  # noqa: PLC0415
            make_openmc_full_combined_source,
        )

        
        return make_openmc_full_combined_source(
            self.xz,
            self.dxdz,
            self.temperature,
            self.strength,
            energy_method,
        )

    def to_sdef_card(self):
        """
        Create an SDEF card which MCNP/openmc can use to make a tokamak neutron source.

        Note
        ----
        The position-dependence of neutron energies be captured by SDEF. Therefore the
        energy distribution of neutrons is averaged, and the same (frozen) distribution
        is used everywhere in the reactor.
        """
        raise NotImplementedError

    def to_h5_source(self):
        """
        Create a source in the HDF5 format such that the full distribution of neutron
        energies and position

        Returns
        -------
        :
            H5 format
        """
        raise NotImplementedError

    def plot(
        self, reactions: list[AllReactions] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the tokamak neutron source.

        Returns
        -------
        f:
            Matplotlib figure object
        ax:
            Matplotlib axes object

        Raises
        ------
        TNSError
            If the requested reactions are not in the source
        """
        if reactions is None:
            plot_reactions = self.source_type
        else:
            plot_reactions = []
            for reaction in reactions:
                if reaction in self.source_type:
                    plot_reactions.append(reaction)
                else:
                    logger.warning(
                        f"Cannot plot reaction {reaction}; it was not specified upon "
                        "instantiation."
                    )
            if len(plot_reactions) == 0:
                raise TNSError("No valid reactions to plot.")

        f, ax = plt.subplots(1, len(plot_reactions), figsize=[15, 8])
        if len(plot_reactions) == 1:
            ax = [ax]

        for axis, reaction in zip(ax, plot_reactions, strict=False):
            self.flux_map.plot(f=f, ax=axis)
            axis.set_title(f"{reaction.label} reaction")
            cm = axis.scatter(
                self.xz,
                c=self.strength[reaction],
                cmap="inferno",
            )

            # Make a colorbar axis that matches the plotting area's height
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="9%", pad=0.08)
            cb = f.colorbar(cm, cax=cax)

            # Put label on top
            cb.ax.set_title("[1/s]")
            axis.plot(self.flux_map.lcfs.x, self.flux_map.lcfs.z, color="r")
            axis.plot(self.flux_map.o_point.x, self.flux_map.o_point.z, "o", color="b")
        f.tight_layout()
        plt.show()
        return f, ax
