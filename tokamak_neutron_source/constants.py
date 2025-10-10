# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of generic physical constants, conversions, and miscellaneous constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy.typing as npt
from periodictable import elements
from pint import Context, Quantity, UnitRegistry, set_application_registry
from pint.errors import PintError

if TYPE_CHECKING:
    from collections.abc import Callable


class TNSUnitRegistry(UnitRegistry):
    """
    TNS UnitRegistry
    Extra conversions:
    eV <-> Kelvin
    """

    def __init__(self):
        # Preprocessor replacements have spaces so
        # the units dont become prefixes or get prefixed
        # M$ makes sense if a bit non-standard
        super().__init__(
            fmt_locale="en_GB",
            preprocessors=[
                lambda x: x.replace("$", "USD "),
            ],
        )

        self._contexts_added = False

    def _add_contexts(self, contexts: list[Context] | None = None):
        """
        Add new contexts to registry
        """
        if not self._contexts_added:
            self.contexts = [
                self._energy_temperature_context(),
                self._mass_energy_context(),
            ]

            for c in self.contexts:
                self.add_context(c)

            self._contexts_added = True

        if contexts:
            for c in contexts:
                self.add_context(c)

    def enable_contexts(self, *contexts: Context, **kwargs):
        """
        Enable contexts
        """
        self._add_contexts(contexts)

        super().enable_contexts(*[*self.contexts, *contexts], **kwargs)

    def _energy_temperature_context(self):
        """
        Converter between energy and temperature
        temperature = energy / k_B

        Returns
        -------
        :
            pint context
        """
        e_to_t = Context("Energy_to_Temperature")

        t_units = "[temperature]"
        ev_units = "[energy]"

        conversion = self.Quantity("k_B")

        return self._transform(
            e_to_t,
            t_units,
            ev_units,
            lambda _, x: x * conversion,
            lambda _, x: x / conversion,
        )

    def _mass_energy_context(self):
        """
        Converter between mass and energy
        energy = mass * speed-of-light^2

        Returns
        -------
        :
            pint context
        """
        m_to_e = Context("Mass_to_Energy")

        m_units = "[mass]"
        e_units = "[energy]"

        conversion = self.Quantity("c^2")

        return self._transform(
            m_to_e,
            m_units,
            e_units,
            lambda _, x: x * conversion,
            lambda _, x: x / conversion,
        )

    @staticmethod
    def _transform(
        context: Context,
        units_from: str,
        units_to: str,
        forward_transform: Callable[[UnitRegistry, complex | Quantity], float],
        reverse_transform: Callable[[UnitRegistry, complex | Quantity], float],
    ) -> Context:
        formatters = ["{}", "{} / [time]"]

        for form in formatters:
            context.add_transformation(
                form.format(units_from),
                form.format(units_to),
                forward_transform,
            )
            context.add_transformation(
                form.format(units_to),
                form.format(units_from),
                reverse_transform,
            )

        return context


ureg = TNSUnitRegistry()
ureg.enable_contexts()
set_application_registry(ureg)

# For reference
TIME = ureg.second
LENGTH = ureg.metre
MASS = ureg.kilogram
CURRENT = ureg.ampere
TEMP = ureg.kelvin
QUANTITY = ureg.mol
ANGLE = ureg.degree
DENSITY = MASS / LENGTH**3
PART_DENSITY = LENGTH**-3
FLUX_DENSITY = LENGTH**-2 / TIME

# =============================================================================
# Physical constants
# =============================================================================

# Speed of light
C_LIGHT = ureg.Quantity("c").to_base_units().magnitude  # [m/s]

# absolute charge of an electron
ELEMENTARY_CHARGE = ureg.Quantity("e").to_base_units().magnitude  # [e]


# Avogadro's number, [1/mol] Number of particles in a mol
N_AVOGADRO = ureg.Quantity("avogadro_number").to_base_units().magnitude

# Stefan-Boltzmann constant: black-body radiation constant of proportionality
SIGMA_BOLTZMANN = ureg.Quantity("sigma").to_base_units().magnitude  # [W/(m^2.K^4)]

# Boltzmann constant kB = R/N_a
K_BOLTZMANN = ureg.Quantity("k_B").to_base_units().magnitude  # [J/K]

# Plank constant
H_PLANCK = ureg.Quantity("hbar").to_base_units().magnitude

# Electron charge, [C]
E_CHARGE = ureg.Quantity("e").to_base_units().magnitude

# neutron molar mass, [u] or [g/mol]
NEUTRON_MOLAR_MASS = (
    ureg.Quantity("m_n").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude

# proton molar mass, [u] or [g/mol]
PROTON_MOLAR_MASS = (
    ureg.Quantity("m_p").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude

# electron molar mass, [u] or [g/mol]
ELECTRON_MOLAR_MASS = (
    ureg.Quantity("m_e").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude


# electron mass [kg]
ELECTRON_MASS = ureg.Quantity("m_e").to_base_units().magnitude

# proton mass [kg]
PROTON_MASS = ureg.Quantity("m_p").to_base_units().magnitude

# Tritium molar mass,  [u] or [g/mol]
T_MOLAR_MASS = elements.isotope("T").mass

# Deuterium molar mass, [u] or [g/mol]
D_MOLAR_MASS = elements.isotope("D").mass

# Helium molar mass, [u] or [g/mol]
HE_MOLAR_MASS = elements.isotope("He").mass

# Helium-3 molar mass, [u] or [g/mol]
HE3_MOLAR_MASS = elements.isotope("3-He").mass


def units_compatible(unit_1: str, unit_2: str) -> bool:
    """
    Test if units are compatible.

    Parameters
    ----------
    unit_1:
        unit 1 string
    unit_2:
        unit 2 string

    Returns
    -------
    :
        True if compatible, False otherwise
    """
    try:
        raw_uc(1, unit_1, unit_2)
    except PintError:
        return False
    else:
        return True


ValueLikeT = TypeVar("ValueLikeT", bound=float | npt.ArrayLike)


def raw_uc(
    value: ValueLikeT,
    unit_from: str | ureg.Unit,
    unit_to: str | ureg.Unit,
) -> ValueLikeT:
    """
    Raw unit converter
    Converts a value from one unit to another

    Parameters
    ----------
    value:
        value to convert
    unit_from:
        unit to convert from
    unit_to:
        unit to convert to

    Returns
    -------
    :
        converted value
    """
    try:
        return (
            ureg.Quantity(value, ureg.Unit(unit_from)).to(ureg.Unit(unit_to)).magnitude
        )
    except ValueError:
        # Catch scales on units eg the ridculousness of this unit: 10^19/m^3
        unit_from_q = ureg.Quantity(unit_from)
        unit_to_q = ureg.Quantity(unit_to)
        return (
            ureg.Quantity(value * unit_from_q).to(unit_to_q.units).magnitude
            / unit_to_q.magnitude
        )


MOLAR_MASSES = {
    "D": D_MOLAR_MASS,
    "T": T_MOLAR_MASS,
    "He": HE_MOLAR_MASS,
    "He3": HE3_MOLAR_MASS,
    "n": NEUTRON_MOLAR_MASS,
    "p": PROTON_MOLAR_MASS,
    "e": ELECTRON_MOLAR_MASS,
}

# The energy released from a single D-T fusion reaction [J]
# 17.590466967089455 MeV
E_DT_FUSION: float = raw_uc(
    D_MOLAR_MASS + T_MOLAR_MASS - (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS),
    "amu",
    "J",
)

# The energy of the neutron produced by a single D-T fusion reaction [J]
# 14.049092569633018 MeV
E_DT_NEUTRON: float = (
    (HE_MOLAR_MASS - 2.0 * ELECTRON_MOLAR_MASS)
    / (HE_MOLAR_MASS - 2.0 * ELECTRON_MOLAR_MASS + NEUTRON_MOLAR_MASS)
    * E_DT_FUSION
)

# The energy released from a single D-D -> T + p fusion reaction [J]
# 4.032649253752984 MeV
E_DD_TP_FUSION: float = raw_uc(
    D_MOLAR_MASS
    + D_MOLAR_MASS
    - (T_MOLAR_MASS + PROTON_MOLAR_MASS + ELECTRON_MOLAR_MASS),
    "amu",
    "J",
)

# The energy released from a single D-D -> 3He + n fusion reaction [J]
# 3.268907714712766 MeV
E_DD_HE3N_FUSION: float = raw_uc(
    D_MOLAR_MASS + D_MOLAR_MASS - (HE3_MOLAR_MASS + NEUTRON_MOLAR_MASS),
    "amu",
    "J",
)

# The energy of the neutron produced by a single D-D -> 3He + n fusion reaction [J]
# 2.449433879692669 MeV
E_DD_NEUTRON: float = (
    (HE3_MOLAR_MASS - 2.0 * ELECTRON_MOLAR_MASS)
    / (HE3_MOLAR_MASS - 2.0 * ELECTRON_MOLAR_MASS + NEUTRON_MOLAR_MASS)
    * E_DD_HE3N_FUSION
)

# The energy released from a single T-T fusion reaction [J]
# 11.333235772236486 MeV
E_TT_FUSION: float = raw_uc(
    T_MOLAR_MASS + T_MOLAR_MASS - (HE_MOLAR_MASS + 2 * NEUTRON_MOLAR_MASS),
    "amu",
    "J",
)

# The Q value of the t + t -> 5He + n reaction, which is an intermediate state of the
# t + t -> 4He + n + n reaction.
_TT_TO_5HE_Q_VALUE = raw_uc(10597, "keV", "J")
_5HE_TO_4HE_Q_VALUE = E_TT_FUSION - _TT_TO_5HE_Q_VALUE
E_TT_NEUTRON: float = (_TT_TO_5HE_Q_VALUE * 5 / 6 + _5HE_TO_4HE_Q_VALUE * 4 / 5) / 2


# The energy released from a single D-3He fusion reaction [J]
# 18.354208506129673 MeV
E_DHE3_FUSION = raw_uc(
    D_MOLAR_MASS
    + HE3_MOLAR_MASS
    - (HE_MOLAR_MASS + PROTON_MOLAR_MASS + ELECTRON_MOLAR_MASS),
    "amu",
    "J",
)
