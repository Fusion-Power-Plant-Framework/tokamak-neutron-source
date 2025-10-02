# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fusion reactions and their data."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from tokamak_neutron_source.constants import (
    E_DD_HE3N_FUSION,
    E_DD_NEUTRON,
    E_DD_TP_FUSION,
    E_DHE3_FUSION,
    E_DT_FUSION,
    E_DT_NEUTRON,
    E_TT_FUSION,
    raw_uc,
)

from tokamak_neutron_source.energy_data import BALLABIO_DD_NEUTRON, BALLABIO_DT_NEUTRON, BallabioEnergySpectrum
from tokamak_neutron_source.error import ReactivityError
from tokamak_neutron_source.reactivity_data import (
    BoschHaleCoefficients,
    ReactionCrossSection,
    BOSCH_HALE_DD_3HEN,
    BOSCH_HALE_DD_TP,
    BOSCH_HALE_DT_4HEN,
    DD_HE3N_XS,
    DD_TP_XS,
    DHE3_HEP_XS,
    DT_XS,
    TT_XS,
)


@dataclass(frozen=True)
class ReactionData:
    """Reaction dataclass."""

    label: str
    total_energy: float
    neutron_energies: list[float]
    cross_section: ReactionCrossSection
    bosch_hale_coefficients: BoschHaleCoefficients | None
    ballabio_spectrum: BallabioEnergySpectrum | None


class ReactionEnumMixin:
    """Provides convenient accessors to the underlying ReactionData."""

    @property
    def data(self) -> ReactionData:
        return self.value

    @property
    def label(self) -> str:
        return self.value.label

    @property
    def total_energy(self) -> float:
        return self.value.total_energy

    @property
    def neutron_energies(self) -> list[float]:
        return self.value.neutron_energies

    @property
    def total_neutron_energy(self) -> float:
        return sum(self.value.neutron_energies)

    @property
    def cross_section(self) -> ReactionCrossSection:
        return self.value.cross_section

    @property
    def bosch_hale_coefficients(self) -> BoschHaleCoefficients | None:
        return self.value.bosch_hale_coefficients
    
    @property
    def ballabio_spectrum(self) -> BallabioEnergySpectrum | None:
        return self.value.ballabio_spectrum


class Reactions(ReactionEnumMixin, Enum):
    """Neutronic reaction channels."""

    D_T = ReactionData(
        label="D + T → ⁴He + n",
        total_energy=E_DT_FUSION,
        neutron_energies=[E_DT_NEUTRON],
        cross_section=DT_XS,
        bosch_hale_coefficients=BOSCH_HALE_DT_4HEN,
        ballabio_spectrum=BALLABIO_DT_NEUTRON,
    )
    D_D = ReactionData(
        label="D + D → ³He + n",
        total_energy=E_DD_HE3N_FUSION,
        neutron_energies=[E_DD_NEUTRON],
        cross_section=DD_HE3N_XS,
        bosch_hale_coefficients=BOSCH_HALE_DD_3HEN,
        ballabio_spectrum=BALLABIO_DD_NEUTRON,
    )
    T_T = ReactionData(
        label="T + T → ⁴He + 2n",
        total_energy=E_TT_FUSION,
        neutron_energies=[raw_uc(2.5, "MeV", "J"), raw_uc(9.0, "MeV", "J")],
        cross_section=TT_XS,
        bosch_hale_coefficients=None,
        ballabio_spectrum=None,
    )


class AneutronicReactions(ReactionEnumMixin, Enum):
    """Aneutronic reaction channels."""

    D_D = ReactionData(
        label="D + D → T + p",
        total_energy=E_DD_TP_FUSION,
        neutron_energies=[],  # no neutrons in aneutronic branch
        cross_section=DD_TP_XS,
        bosch_hale_coefficients=BOSCH_HALE_DD_TP,
        ballabio_spectrum=None,
    )
    D_He3 = ReactionData(
        label="D + ³He → ⁴He + p",
        total_energy=E_DHE3_FUSION,
        neutron_energies=[],
        cross_section=DHE3_HEP_XS,
        bosch_hale_coefficients=None,
        ballabio_spectrum=None,
    )


AllReactions: TypeAlias = Reactions | AneutronicReactions


def _parse_reaction(reaction: str | AllReactions) -> AllReactions:
    """
    Parse a single reaction, possibly from a string.

    Parameters
    ----------
    reaction:
        The reaction to parse

    Returns
    -------
    :
        The parsed reaction

    Notes
    -----
    Deliberate bias towards neutronic reactions (in the case of D-D).

    Raises
    ------
    ReactivityError
        If the specified reaction does not exist
    """
    if isinstance(reaction, str):
        string = reaction.replace("-", "_")
        try:
            reaction = Reactions[string]
        except KeyError:
            try:
                reaction = AneutronicReactions[string]
            except KeyError:
                raise ReactivityError(f"Unrecognised reaction: {string}") from None
        return reaction
    if isinstance(reaction, AllReactions):
        return reaction

    raise ReactivityError(f"Unrecognised reaction type: {type(reaction)}")
