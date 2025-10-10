# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reactivity data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from tokamak_neutron_source.constants import MOLAR_MASSES, raw_uc
from tokamak_neutron_source.error import ReactivityError
from tokamak_neutron_source.tools import get_tns_path


class ReactionCrossSection:
    """Fusion reaction cross-section."""

    def __init__(self, file_name: str):
        """
        Parameters
        ----------
        file_name:
            Cross-sectional data file (ENDF format)

        Raises
        ------
        ReactivityError
            Data file path is not a file
        """
        path = get_tns_path("data/cross_sections")
        path = Path(path, file_name)
        if not path.is_file():
            raise ReactivityError(f"Cross-section data file {path} is not a file!")

        file = path.as_posix()
        # Read in the cross section (in barn) as a function of energy (MeV).
        energy, sigma = np.genfromtxt(file, comments="#", skip_footer=2, unpack=True)

        split = file_name.split(".", maxsplit=1)[0].split("_")
        collider, target = split[:2]
        self.name = f"{collider} + {target} -> {split[2]} + {split[3]}"

        mass_1, mass_2 = MOLAR_MASSES[collider], MOLAR_MASSES[target]

        self.reduced_mass = raw_uc(mass_1 * mass_2 / (mass_1 + mass_2), "amu", "kg")

        # Convert to center of mass frame
        # NOTE MC: Choice of target/collider thing makes Bosch-Hale line up...
        energy *= mass_2 / (mass_1 + mass_2)

        # Convert to kev / m^2
        self._cross_section = interp1d(energy * 1e3, sigma * 1e-28)

    def __call__(self, temp_kev: float | npt.NDArray) -> float | npt.NDArray:
        """Get cross section at a give temperature"""  # noqa: DOC201
        return self._cross_section(temp_kev)


# BoschHale model coefficients
@dataclass
class BoschHaleCoefficients:
    """
    Bosch-Hale parameterisation dataclass.

    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """

    name: str
    t_min: float  # [keV]
    t_max: float  # [keV]
    bg: float  # [keV**0.5]
    mrc2: float  # [keV]
    c: npt.NDArray


TT_XS = ReactionCrossSection("T_T_He_2n.txt")
DT_XS = ReactionCrossSection("D_T_He_n.txt")
DD_TP_XS = ReactionCrossSection("D_D_T_p.txt")
DD_HE3N_XS = ReactionCrossSection("D_D_He3_n.txt")
DHE3_HEP_XS = ReactionCrossSection("D_He3_He_p.txt")


# Bosch-Hale parameterisation data for the reaction:  D + T --> 4He + n
BOSCH_HALE_DT_4HEN = BoschHaleCoefficients(
    name="D + T --> 4He + n",
    t_min=0.2,
    t_max=100.0,
    bg=34.3827,
    mrc2=1.124656e6,
    c=np.array(
        [
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        ],
    ),
)

# Bosch-Hale parameterisation data for the reaction: D + D --> 3He + n
BOSCH_HALE_DD_3HEN = BoschHaleCoefficients(
    name="D + D --> 3He + n",
    t_min=0.2,
    t_max=100.0,
    bg=31.3970,
    mrc2=0.937814e6,
    c=np.array(
        [
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        ],
    ),
)

# Bosch-Hale parameterisation data for the reaction: D + D --> T + p
BOSCH_HALE_DD_TP = BoschHaleCoefficients(
    name="D + D --> T + p",
    t_min=0.2,
    t_max=100.0,
    bg=31.3970,
    mrc2=0.937814e6,
    c=np.array(
        [
            5.65718e-12,
            3.41267e-3,
            1.99167e-3,
            0.0,
            1.05060e-5,
            0.0,
            0.0,
        ],
    ),
)
