# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC neutron source interface."""

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
    Univariate,
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
    half_cell_length: float,
    energy_distribution: Univariate,
    strength: float,
) -> IndependentSource:
    """
    Make a single OpenMC ring source with a square cross-section.

    Parameters
    ----------
    r:
        Radial position of the centroid of the  3-D ring [m]
    z:
        Vertical position of the centroid of the 3-D ring [m]
    half_cell_length:
        Half the square cell length [m]
    energy_distribution:
        Neutron energy distribution
    strength:
        Strength of the source [number of neutrons]

    Returns
    -------
    :
        An OpenMC IndependentSource object, or None if strength is zero.

    Notes
    -----
    The z values within the square cell are uniform, and the r values vary
    linearly with increasing radius.
    """
    if strength > 0:
        r_in, r_out = r - half_cell_length, r + half_cell_length
        z_down, z_up = z - half_cell_length, z + half_cell_length
        r_lim_cm = raw_uc([r_in, r_out], "m", "cm")
        z_lim_cm = raw_uc([z_down, z_up], "m", "cm")
        r_lim_prob = np.array(r_lim_cm) / sum(r_lim_cm)
        return IndependentSource(
            energy=energy_distribution,
            space=CylindricalIndependent(
                r=Tabular(r_lim_cm, r_lim_prob, interpolation="linear-linear"),
                phi=Uniform(0, 2 * np.pi),
                z=Uniform(*z_lim_cm),
                origin=(0.0, 0.0, 0.0),
            ),
            angle=Isotropic(),
            strength=strength,
        )
    return None


def make_openmc_full_combined_source(  # noqa: PLR0913, PLR0917
    r: npt.NDArray,
    z: npt.NDArray,
    cell_side_length: float,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
    source_rate: float,
    energy_spectrum_method: EnergySpectrumMethod = EnergySpectrumMethod.AUTO,
) -> IndependentSource:
    """
    Make an OpenMC source combining multiple reactions across the whole plasma.

    Parameters
    ----------
    r:
        Radial positions of the rings [m]
    z:
        Vertical positions of the rings [m]
    cell_side_length:
        Radial and vertical spacings of the rings [m]
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

    l_2 = cell_side_length / 2
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
                ri,
                zi,
                l_2,
                distribution,
                local_strength / source_rate,
            )
            if source is not None:
                sources.append(source)

    return sources
