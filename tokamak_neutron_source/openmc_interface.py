# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC neutron source interface"""

from typing import Any

import numpy as np
import numpy.typing as npt
from openmc import IndependentSource
from openmc.data import combine_distributions
from openmc.stats import (
    CylindricalIndependent,
    Discrete,
    Isotropic,
    Uniform,
    Tabular,
    Mixture,
)

from tokamak_neutron_source.constants import raw_uc
from tokamak_neutron_source.energy import energy_spectrum
from tokamak_neutron_source.reactions import Reactions
from tokamak_neutron_source.reactivity import AllReactions


def get_neutron_energy(reaction: Reactions, temp_kev: float) -> Discrete:
    """
    Returns
    -------
    :
        Discrete neutron energy distribution for the given reaction.

    Raises
    ------
    ValueError
        Unsupported Reaction

    Notes
    -----
    Energies in MeV.
    Probabilities normalized.
    Temperature parameter left in for possible spectrum broadening.
    """
    energy, probability = energy_spectrum(reaction, temp_kev)
    energy_ev = raw_uc(energy, "keV", "eV")
    match reaction:
        case Reactions.D_T | Reactions.D_D:
            return Tabular(energy_ev, probability)

        case Reactions.T_T:
            # TODO: Add T-T spectral data
            # T + T → 4He + 2n
            # Neutrons have a broad spectrum, here approximated as two 2-9 MeV neutrons
            # (very simplified discrete placeholder)
            return Discrete(raw_uc(reaction.neutron_energies, "J", "eV"), [0.5, 0.5])

        case _:
            raise ValueError(f"Unsupported reaction: {reaction}")


def make_openmc_ring_source(
    r: float,
    z: float,
    energy_distribution: Any,
    strength: float,
) -> IndependentSource:
    """
    Make a single OpenMC ring source.

    Parameters
    ----------
    r:
        Radial position of the ring [m]
    z:
        Vertical position of the ring [m]
    energy_distribution:
        Neutron energy distribution
    strength:
        Strength of the source [arbitrary units]

    Returns
    -------
    :
        An OpenMC IndependentSource object, or None if strength is zero.
    """
    if strength > 0:
        return IndependentSource(
            energy=energy_distribution,
            space=CylindricalIndependent(
                r=Uniform(raw_uc(r, "m", "cm")),
                phi=Uniform(0, 2 * np.pi),
                z=Uniform(raw_uc(z, "m", "cm")),
                origin=(0.0, 0.0, 0.0),
            ),
            angle=Isotropic(),
            strength=strength,
        )
    return None


def make_openmc_full_combined_source(
    r: npt.NDArray,
    z: npt.NDArray,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
) -> IndependentSource:
    """
    Make an OpenMC source combining multiple reactions across the whole plasma.

    Parameters
    ----------
    r:
        Radial positions of the rings [m]
    z:
        Vertical positions of the rings [m]
    temperature:
        Ion temperatures at the rings [keV]
    strength:
        Dictionary of strengths for each reaction at the rings [arbitrary units]

    Returns
    -------
    :
        A list of OpenMC IndependentSource objects, one per ring.
    """
    sources = []
    # Neutronic reaction channels only
    n_strength = {k: v for k, v in strength.items() if isinstance(k, Reactions)}

    for i, (ri, zi, ti) in enumerate(zip(r, z, temperature, strict=False)):
        distributions = []
        weights = []

        for reaction, s in n_strength.items():
            if s[i] > 0.0:
                distributions.append(get_neutron_energy(reaction, ti))
                weights.append(s[i])

        total_strength = sum(weights)
        # TODO: Replace with Mixture
        distribution = combine_distributions(
            distributions,
            np.array(weights) / total_strength,
        )

        source = make_openmc_ring_source(ri, zi, distribution, total_strength)
        if source is not None:
            sources.append(source)

    return sources
