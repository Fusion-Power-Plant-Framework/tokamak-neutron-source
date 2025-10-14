# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC neutron source interface."""

from typing import Any

import numpy as np
import numpy.typing as npt
from openmc import IndependentSource
from openmc.stats import (
    CylindricalIndependent,
    Discrete,
    Isotropic,
    Mixture,
    Normal,
    Tabular,
    Uniform,
)

from tokamak_neutron_source.constants import raw_uc
from tokamak_neutron_source.energy import EnergySpectrumMethod, energy_spectrum
from tokamak_neutron_source.reactions import Reactions
from tokamak_neutron_source.reactivity import AllReactions
from tokamak_neutron_source.tools import QuietTTSpectrumWarnings


def get_neutron_energy_spectrum(
    reaction: Reactions, temp_kev: float, method: EnergySpectrumMethod
) -> Tabular | Discrete:
    """
    Get a native OpenMC neutron energy spectrum.

    Parameters
    ----------
    reaction:
        The neutronic reaction for which to retrieve the neutron spectrum
    temp_kev: float
        The ion temperature of the reactants
    method:
        Which method to use when calculating the energy spectrum

    Returns
    -------
    :
        OpenMC tabular neutron energy distribution for the given reaction.

    Notes
    -----
    Log-linear interpolation is used within OpenMC.
    """
    if (
        method is EnergySpectrumMethod.BALLABIO_GAUSSIAN
        and reaction.ballabio_spectrum is not None
    ):
        mean = reaction.ballabio_spectrum.mean_energy(temp_kev)
        std = reaction.ballabio_spectrum.std_deviation(temp_kev)
        return Normal(raw_uc(mean, "keV", "eV"), raw_uc(std, "keV", "eV"))
    energy, probability = energy_spectrum(temp_kev, reaction, method)
    energy_ev = raw_uc(energy, "keV", "eV")
    # Log-linear interpolation is not supported in OpenMC at present
    # see: https://github.com/openmc-dev/openmc/issues/2409
    return Tabular(energy_ev, probability, interpolation="linear-linear")


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
        Strength of the source [numebr of neutrons]

    Returns
    -------
    :
        An OpenMC IndependentSource object, or None if strength is zero.
    """
    if strength > 0:
        return IndependentSource(
            energy=energy_distribution,
            space=CylindricalIndependent(
                r=Discrete([raw_uc(r, "m", "cm")], [1.0]),
                phi=Uniform(0, 2 * np.pi),
                z=Discrete([raw_uc(z, "m", "cm")], [1.0]),
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
    source_rate: float,
    energy_spectrum_method: EnergySpectrumMethod,
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
        Dictionary of strengths for each reaction at the rings [neutrons]
    source_rate:
        Total source rate [neutrons/s]
    energy_spectrum_method:
        Which method to use when calculating neutron spectra

    Returns
    -------
    :
        A list of OpenMC IndependentSource objects, one per ring.
    """
    sources = []
    # Neutronic reaction channels only
    # We multiply the T-T channel by 2 because it is 2n
    n_strength = {
        reaction: rate * reaction.num_neutrons
        for reaction, rate in strength.items()
        if isinstance(reaction, Reactions)
    }

    with QuietTTSpectrumWarnings():
        for i, (ri, zi, ti) in enumerate(zip(r, z, temperature, strict=False)):
            distributions = []
            weights = []

            for reaction, s in n_strength.items():
                if s[i] > 0.0:
                    distributions.append(
                        get_neutron_energy_spectrum(reaction, ti, energy_spectrum_method)
                    )
                    weights.append(s[i])

            local_strength = sum(weights)

            distribution = Mixture(np.array(weights) / local_strength, distributions)

            source = make_openmc_ring_source(
                ri, zi, distribution, local_strength / source_rate
            )
            if source is not None:
                sources.append(source)

    return sources
