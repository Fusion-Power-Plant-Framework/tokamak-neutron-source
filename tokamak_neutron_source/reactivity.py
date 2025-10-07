# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Reactivity calculations.
"""

import logging
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from tokamak_neutron_source.constants import (
    raw_uc,
)
from tokamak_neutron_source.error import ReactivityError
from tokamak_neutron_source.reactions import (
    AllReactions,
    AneutronicReactions,
    Reactions,
    _parse_reaction,
)
from tokamak_neutron_source.reactivity_data import (
    BoschHaleCoefficients,
    ReactionCrossSection,
)
from tokamak_neutron_source.tools import trapezoid

__all__ = ["density_weighted_reactivity", "reactivity"]

logger = logging.getLogger(__name__)


class ReactivityMethod(Enum):
    """Reactivity calculation method."""

    XS = auto()
    BOSCH_HALE = auto()


def density_weighted_reactivity(
    temp_kev: float | npt.NDArray,
    density_d: float | npt.NDArray,
    density_t: float | npt.NDArray,
    density_he3: float | npt.NDArray,
    reaction: str | AllReactions = Reactions.D_T,
    method: ReactivityMethod = ReactivityMethod.BOSCH_HALE,
) -> float | npt.NDArray:
    """
    Calculate the density-weighted thermal reactivity of a fusion reaction in
    Maxwellian plasmas, \\t:math:`n_1 n_2 <\\sigma v>`.

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    density_d:
        Deuterium density [m^-3]
    density_t:
        Tritium density [m^-3]
    density_he3:
        Helium-3 density [m^-3]
    reaction:
        The fusion reaction
    method:
        The method to use when calculating the reactivity

    Returns
    -------
    :
        Density-weighted reactivity of the reaction at the specified temperature(s)
        [1/m^3/s]
    """
    reaction = _parse_reaction(reaction)

    match reaction:
        case Reactions.D_D | AneutronicReactions.D_D:
            n1_n2 = density_d * density_d / 2
        case Reactions.D_T:
            n1_n2 = density_d * density_t
        case Reactions.T_T:
            n1_n2 = density_t * density_t / 2
        case AneutronicReactions.D_He3:
            n1_n2 = density_d * density_he3
        case _:
            raise NotImplementedError(f"Reaction {reaction} not implemented.")

    return n1_n2 * reactivity(temp_kev, reaction, method)


def reactivity(
    temp_kev: float | npt.NDArray,
    reaction: str | AllReactions = Reactions.D_T,
    method: ReactivityMethod = ReactivityMethod.BOSCH_HALE,
) -> float | npt.NDArray:
    """
    Calculate the thermal reactivity of a fusion reaction in Maxwellian plasmas,
    \\t:math:`<\\sigma v>`.

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    reaction:
        The fusion reaction
    method:
        The method to use when calculating the reactivity

    Returns
    -------
    :
        Reactivity of the reaction at the specified temperature(s) [m^3/s]
    """
    reaction = _parse_reaction(reaction)

    match method:
        case ReactivityMethod.BOSCH_HALE:
            if reaction.bosch_hale_coefficients is not None:
                return _reactivity_bosch_hale(temp_kev, reaction)

            logger.warning(
                f"There is no Bosch-Hale parameterisation for reaction {reaction.name}, "
                "returning reactivity calculated by cross-section."
            )
            return _reactivity_from_xs(temp_kev, reaction.cross_section)
        case ReactivityMethod.XS:
            return _reactivity_from_xs(temp_kev, reaction.cross_section)


def _reactivity_bosch_hale(
    temp_kev: float | npt.NDArray,
    reaction: AllReactions,
) -> float | npt.NDArray:
    """
    Bosch-Hale reactivity parameterisation for Maxwellian plasmas.

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    reaction:
        The fusion reaction

    Returns
    -------
    :
        Reactivity of the reaction at the specified temperature(s) [cm^3/s]

    Raises
    ------
    ReactivityError
        If no Bosch-Hale parameterisation is available.

    Notes
    -----
    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """
    if reaction.bosch_hale_coefficients is not None:
        return _reactivity_channel_bosch_hale(temp_kev, reaction.bosch_hale_coefficients)

    raise ReactivityError(
        f"Reaction {reaction.name} does not have Bosch-Hale coefficients."
    )


def _reactivity_channel_bosch_hale(
    temp_kev: float | npt.NDArray,
    data: BoschHaleCoefficients,
) -> float | npt.NDArray:
    """
    Bosch-Hale reactivity parameterisation for a single reaction channel in
    Maxwellian plasmas.

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    data:
        The Bosch-Hale parameterisation data

    Returns
    -------
    :
        Reactivity of the reaction at the specified temperature(s) [m^3/s]
    """
    if not isinstance(temp_kev, np.ndarray):
        temp_kev = np.array([temp_kev])

    if np.min(temp_kev) < data.t_min or np.max(temp_kev) > data.t_max:
        logger.warning(
            f"The Bosch-Hale parameterisation for reaction {data.name} is only valid "
            f"between {data.t_min} and {data.t_max} keV; but out of range energies "
            f"{np.min(temp_kev)}keV -- {np.max(temp_kev)} is detected!",
            stacklevel=4,
        )

    # mask calculation to avoid division by zero
    safe = temp_kev > 0

    frac = np.zeros_like(temp_kev, dtype=float)
    theta = np.zeros_like(temp_kev, dtype=float)
    chi = np.zeros_like(temp_kev, dtype=float)
    result = np.zeros_like(temp_kev, dtype=float)

    frac[safe] = (
        temp_kev[safe]
        * (data.c[1] + temp_kev[safe] * (data.c[3] + temp_kev[safe] * data.c[5]))
        / (
            1
            + temp_kev[safe]
            * (data.c[2] + temp_kev[safe] * (data.c[4] + temp_kev[safe] * data.c[6]))
        )
    )
    theta[safe] = temp_kev[safe] / (1 - frac[safe])
    chi[safe] = (data.bg**2 / (4 * theta[safe])) ** (1 / 3)
    result[safe] = (
        1e-6
        * data.c[0]
        * theta[safe]
        * np.sqrt(chi[safe] / (data.mrc2 * temp_kev[safe] ** 3))
        * np.exp(-3 * chi[safe])
    )

    return result if result.shape != () else float(result)


def _reactivity_from_xs(
    temp_kev: float | npt.NDArray,
    xs: ReactionCrossSection,
) -> float | npt.NDArray:
    """
    Calculate the thermal reactivity of a fusion reaction in Maxwellian plasmas,
    \\t:math:`<\\sigma v>`.

    Parameters
    ----------
    temp_keV:
        Temperature [keV]
    xs:
        The reaction cross section data

    Returns
    -------
    :
        Reactivity of the reaction at the specified temperature(s) [m^3/s]
    """
    temp_kev = np.atleast_1d(temp_kev)
    temp_j = raw_uc(temp_kev[:, None], "keV", "J")

    t_grid_kev = np.logspace(0, 3, 1000)
    t_grid_j = raw_uc(t_grid_kev, "keV", "J")

    factor = 4 / np.sqrt(2 * np.pi * xs.reduced_mass) / temp_j**1.5

    integrand = factor * xs(t_grid_kev) * t_grid_j * np.exp(-t_grid_j / temp_j)

    sigma_v = trapezoid(integrand, t_grid_j, axis=1)
    return float(sigma_v) if np.isscalar(temp_kev) else sigma_v
