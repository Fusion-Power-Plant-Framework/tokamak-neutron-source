# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""MCNP neutron source (SDEF) interface"""

import logging
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tokamak_neutron_source.energy import EnergySpectrumMethod, energy_spectrum
from tokamak_neutron_source.reactions import AneutronicReactions, Reactions
from tokamak_neutron_source.reactivity import AllReactions
from tokamak_neutron_source.tools import load_citation, raw_uc

logger = logging.getLogger(__name__)


def write_mcnp_sdef_source(
    file: str | Path,
    r: npt.NDArray,
    z: npt.NDArray,
    cell_side_length: float,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
):
    """
    Write an MCNP SDEF source for a ring source at (r,z).

    Parameters
    ----------
    file:
        The file name stub to which to write the SDEF source
    r:
        Radial positions of the rings [m]
    z:
        Vertical positions of the rings [m]
    cell_side_length:
        side length of square source cell
    temperature:
        Ion temperatures at the rings [keV]
    strength:
        Dictionary of strengths for each reaction at the rings [arbitrary units]

    Notes
    -----
    Only Neutronic reactions are written to SDEF file Aneutronic reactions are ignored.
    The radial distribution bouldaries and probabilities are set to the SI3 and SP3 cards
    The DS4 card is used as the dependent distribution numbers
    for the vertical distributions

    """
    r = raw_uc(r, "m", "cm")
    z = raw_uc(z, "m", "cm")
    dr = raw_uc(cell_side_length, "m", "cm")

    # Half widths of 'cells'
    drad = dzed = dr / 2

    offset = 5  # First 5 distribution are reserved

    for reaction, r_data in strength.items():
        if reaction not in AneutronicReactions:
            short_react = re.findall(r"[DT]", reaction.label)
            file_name = f"{file}.{short_react[0]}{short_react[1]}"

            header = sdef_header(reaction, r_data, temperature)

            # Calculate the radial boundaries based on the ring centres
            # and 'cell width' (dr)
            r_bounds = np.unique(r) - drad
            r_bounds = np.append(r_bounds, r_bounds[-1] + dr)  # Add the last boundary

            # Identify where radial position changes
            # and therefore the range of each vertical distribution
            z_ints = np.concatenate([[-1], np.nonzero(r[1:] != r[:-1])[0], [len(r) - 1]])

            si_card = _text_wrap(
                "SI3 H " + " ".join(f"{ri:.5e}" for ri in r_bounds), new_lines=1
            )
            sp_card = _text_wrap(
                f"SP3 D {0.0:.5e} "
                + " ".join(
                    f"{np.sum(r_data[z_ints[i] + 1 : z_ints[i + 1] + 1]):.5e}"
                    for i in range(len(z_ints) - 1)
                ),
                new_lines=1,
            )

            ds_card = _text_wrap(
                "DS4 S "
                + " ".join(f"{i:d}" for i in range(offset, offset + len(r_bounds) - 1)),
                new_lines=1,
            )

            with open(file_name, "w") as sdef_file:
                sdef_file.write(
                    f"{header}{si_card}{sp_card}{ds_card}"
                    "C\nC 3. Neutron Emission Probability - Vertical Distribution\nC\n"
                )
                for si_card, sp_card in _si_sp_vertical_dist_cards(
                    offset, z, z_ints, dzed, r_data
                ):
                    sdef_file.write(f"{si_card}{sp_card}")

        else:
            logger.info(f"Skipping reaction {reaction.label} for MCNP SDEF source.")


def _si_sp_vertical_dist_cards(
    offset, z, z_ints, dzed, r_data
) -> Iterator[tuple[str, str]]:
    """Create the vertical distribution for each radius

    Notes
    -----
    Set to the SI and SP cards listed on the DS4 card.
    The SI card contains the vertical distribution bin boundaries.
    The SP card contains the vertical distribution probabilities.

    Yields
    ------
    :
        SI card
    :
        SP card

    """
    for i in range(len(z_ints) - 1):
        indent_offset = len(str(i + offset)) + 5

        zs, zf = z_ints[i] + 1, z_ints[i + 1] + 1
        z_i = " ".join(f"{zi:.5e}" for zi in z[zs:zf] - dzed)
        si_card = _text_wrap(
            f"SI{i + offset} H {z_i} {z[z_ints[i + 1]] + dzed:.5e}",
            indent=indent_offset,
            new_lines=1,
        )

        rd = " ".join(f"{s:.5e}" for s in r_data[zs:zf])
        sp_card = _text_wrap(
            f"SP{i + offset} D {0.0:.5e} {rd}", indent=indent_offset, new_lines=1
        )
        yield si_card, sp_card


def sdef_header(
    reaction: Reactions, reaction_data: npt.NDArray, temperature: float
) -> str:
    """Create SDEF file header

    Parameters
    ----------
    reaction:
        Reaction to be created
    reaction_data:
        strength of source
    temperature:
        Ion temperature

    Notes
    -----
    For DT and DD reactions MCNP's built-in gaussian spectrums are used
    For TT reactions the tabulated data is used

    """  # noqa: DOC201
    strength = sum(reaction_data)
    ion_temp = mean_ion_temp(reaction_data, temperature)

    # Read authors and git address from CITATION.cff
    citation = load_citation()
    authors = "\n".join(
        (
            f"C    {author.get('given-names', '')}"
            f" {author.get('family-names', '')},"
            f" {author.get('affiliation', '')}"
        )
        for author in citation.get("authors", [])
    )
    gitaddr = citation.get("repository-code", "")

    if reaction in {Reactions.D_T, Reactions.D_D}:
        # -1 for D-T and -2 for D-D
        reaction_data = (
            f"SP2 -4 {raw_uc(ion_temp, 'keV', 'MeV'):5e} "
            f"{-1 if reaction == Reactions.D_T else -2}\n"
        )

    elif reaction == Reactions.T_T:
        energies, probabilities = energy_spectrum(
            ion_temp, reaction, EnergySpectrumMethod.DATA
        )
        reaction_data = "SI2 H " + _text_wrap(
            f"{0.0:.5e} " + " ".join(f"{e:.5e}" for e in raw_uc(energies, "keV", "MeV"))
        )
        reaction_data += "SP2 D " + _text_wrap(
            f"{0.0:.5e} " + " ".join(f"{p:.5e}" for p in probabilities)
        )

    return f"""C ============================
C SDEF Card for Tokamak Neutron Source generated by:
C {gitaddr}
C ============================
C
C ============================
C Authors:
{authors}
C ============================
C
C ============================
C Method:
C 1. Create a cylinder that encloses the entire torus.
C 2. Then slice the cylinder along the R-axis.
C 3. Finally, define the vertical distribution, assuming rotational symmetry.
C ============================
C
C ============================
C Reaction channel: {reaction.label}
C Total source neutrons: {strength:5e} n/s
C ============================
C
C 1. Neutron Emission Probability - Set up cylindrical source
C
sdef erg=d2 par=1 wgt=1
      pos = 0 0 0    $ Center = origin
      axs = 0 0 1    $ Cylinder points along the Z axis
      rad = d3       $ radial distribution defined by distribution 3
      ext = frad d4  $ extent distribution defined by distribution 4 which is dependent on distribution rad
{reaction_data}C
C 2. Neutron Emission Probability - Radial Distribution
C
"""  # noqa: E501


def mean_ion_temp(strength: npt.NDArray, temperature: npt.NDArray) -> float:
    """Calculate the strength-weighted mean ion temperature."""  # noqa: DOC201
    return np.sum(strength * temperature) / np.sum(strength)


def _text_wrap(
    long_line: str, indent: int = 6, max_length: int = 80, new_lines: int = 1
) -> str:
    """
    Break lines such that they're never longer than max_length characters.

    Parameters
    ----------
    long_line:
        A string that requires to be broken down.

    Returns
    -------
    :
        Same string broken into multiple lines with a trailing newline character

    Raises
    ------
    ValueError
        < 6 spaces indented
    """
    # validate indent number matches MCNP syntax.
    if indent < 6:  # noqa: PLR2004
        raise ValueError(
            "MCNP input file interpret 5 or fewer indents as a new line, "
            "rather than a continued line broken from the previous.",
        )
    nl = "\n"
    line = textwrap.indent(
        nl.join(textwrap.wrap(long_line, width=max_length)), " " * indent
    ).strip()
    return f"{line}{nl * new_lines}"
