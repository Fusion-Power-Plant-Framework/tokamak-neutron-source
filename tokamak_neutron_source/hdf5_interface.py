# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""HDF5 source creation interface"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from tokamak_neutron_source.constants import E_DD_NEUTRON, E_DT_NEUTRON, E_TT_NEUTRON
from tokamak_neutron_source.reactions import AneutronicReactions, Reactions
from tokamak_neutron_source.reactivity import AllReactions
from tokamak_neutron_source.tools import raw_uc


@dataclass
class Histogram:
    """Histogram storage class"""

    name: str
    property: str
    type: str
    value: np.ndarray
    dimension: str | None = None
    bins: np.ndarray | None = None
    interpolation: str | None = None
    unit: str | None = None
    children: list[str] | None = None
    partner: list[str] | None = None

    def add_hist_to_file(self, file: h5py.File):
        """Add histogram to an hdf5 file"""
        group = file.create_group(self.name)

        # Attributes
        group.attrs["property"] = self.property
        group.attrs["type"] = self.type

        # Attributes (optional)
        if self.dimension is not None:
            group.attrs["dimension"] = self.dimension
        if self.unit is not None:
            group.attrs["unit"] = self.unit
        if self.interpolation is not None:
            group.attrs["interpolation"] = self.interpolation

        # Datasets
        group.create_dataset("value", data=self.value)
        if self.bins is not None:
            group.create_dataset("bins", data=self.bins)
        if self.children is not None:
            group.attrs["parent"] = "true"
            group.create_dataset("children", data=self.children)
        else:
            group.attrs["parent"] = "false"

        if self.partner is not None:
            group.attrs["married"] = "true"
            group.create_dataset("partner", data=self.partner)
        else:
            group.attrs["married"] = "false"


def write_hdf5_source(  # noqa: PLR0914
    file: str | Path,
    r_position: npt.NDArray,
    z_position: npt.NDArray,
    cell_side_length: float,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
):
    """
    Write the source (in terms of histograms) to a HDF5 file

    Parameters
    ----------
    file:
        The file name stub to which to write the *.h5 file
    r:
        Radial positions of the rings
    z:
        Vertical positions of the rings
    cell_side_length:
        side length of square source cell
    temperature:
        Ion temperatures at the rings [keV]
    strength:
        Dictionary of strengths for each reaction at the rings [arbitrary units]
    """
    r_position = raw_uc(r_position, "m", "cm")
    z_position = raw_uc(z_position, "m", "cm")
    half_width = raw_uc(cell_side_length / 2, "m", "cm")

    r_bounds = np.unique(r_position)

    histograms = {}
    reaction_energy = {
        Reactions.D_T: np.atleast1d(raw_uc(E_DT_NEUTRON, "J", "MeV")),
        Reactions.D_D: np.atleast1d(raw_uc(E_DD_NEUTRON, "J", "MeV")),
        Reactions.T_T: np.atleast1d(raw_uc(E_TT_NEUTRON, "J", "MeV")),
    }
    # TODO @je-cook: Is this needed or could we use reaction.name?
    reaction_sn = {Reactions.D_T: "DTn", Reactions.D_D: "DDn", Reactions.T_T: "TTn"}

    reaction_name = []
    # This will be the total reactivity for each reaction i.e. bin intensity for decision
    total_reactivity = []
    for reaction, react_data in strength.items():
        if reaction not in AneutronicReactions:
            r_name = reaction_sn[reaction]
            reaction_name.append(r_name)
            total_reactivity.append(react_data.sum())

            # Need to create a scalar to contain the reaction energy and particle type
            erg_name = f"var_{r_name}.erg_e"
            prt_name = f"var_{r_name}.prt_p"
            pos_r_name = f"var_{r_name}.pos_r"

            histograms[erg_name] = Histogram(
                name=erg_name,
                property="energy",
                dimension="e",
                type="scalar",
                value=reaction_energy[reaction],
                unit="MeV",
                children=[prt_name],
            )

            histograms[prt_name] = Histogram(
                name=prt_name,
                property="particle",
                dimension="p",
                type="scalar",
                value=np.atleast1d(reaction.num_neutrons),  # neutron
                children=[pos_r_name],
            )

            r_values = np.zeros(len(r_bounds))
            for i, r in enumerate(r_bounds):
                r_mask = [0 if j == r else 1 for j in r_position]
                mx_react = react_data[r_mask]
                mx_z = z_position[r_mask]

                # Sum up masked reactivity -> bin value for radius distribution
                r_values[i] = mx_react.sum()

                # Create z bin edges from masked position data
                z_bins = np.array(
                    [z - half_width for z in mx_z] + [mx_z[-1] + half_width]
                )

                # Create z histogram for particular r bin
                pos_z_name = f"var_{r_name}.pos_z[{i}]"
                histograms[pos_z_name] = Histogram(
                    name=pos_z_name,
                    property="position",
                    dimension="z",
                    type="histogram",
                    interpolation="linear",
                    bins=z_bins,
                    value=mx_react,
                    unit="cm",
                    partner=[f"erg_temp[{i}]"],
                )

                # Create partner temperature histograms
                # This is independent of reaction but needs the radial looping
                erg_t_name = f"erg_temp[{i}]"
                if erg_t_name not in histograms:
                    histograms[erg_t_name] = Histogram(
                        name=erg_t_name,
                        property="energy",
                        dimension="temp",
                        type="histogram",
                        interpolation="linear",
                        bins=z_bins,
                        value=temperature[r_mask],
                        unit="keV",
                    )

            histograms[pos_r_name] = Histogram(
                name=pos_r_name,
                property="position",
                dimension="r",
                type="histogram",
                interpolation="linear",
                bins=np.array(  # Create r bin edges from masked position data
                    [r - half_width for r in r_bounds] + [r_bounds[-1] + half_width]
                ),
                value=r_values,
                unit="cm",
                children=[f"var_{r_name}.pos_z[{r}]" for r in range(len(r_values))],
            )

    # Final histogram (and overall grandparent) is the reaction decision
    histograms["var_reaction"] = Histogram(
        name="var_reaction",
        property="variable",
        type="decision",
        value=np.array(total_reactivity),
        bins=np.arange(len(total_reactivity)),
        children=[f"var_{r}.erg_e" for r in reaction_name],
    )

    with h5py.File(f"{file}.h5", "w") as h5_file:
        for h in histograms.values():
            h.add_hist_to_file(h5_file)
