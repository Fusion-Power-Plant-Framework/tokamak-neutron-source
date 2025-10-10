# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""MCNP neutron source (SDEF) interface"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
import yaml
import re

from tokamak_neutron_source.energy import EnergySpectrumMethod, energy_spectrum
from tokamak_neutron_source.reactions import Reactions, AneutronicReactions
from tokamak_neutron_source.reactivity import AllReactions
from tokamak_neutron_source.tools import get_tns_path, load_citation, raw_uc

def write_mcnp_sdef_source(
    file: str | Path,
    r: npt.NDArray,
    z: npt.NDArray,
    dr: float,
    dz: float,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
    energy_spectrum_method: EnergySpectrumMethod,
    
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
    temperature:
        Ion temperatures at the rings [keV]
    strength:
        Dictionary of strengths for each reaction at the rings [arbitrary units]
    energy_spectrum_method:
        Which method to use when calculating neutron spectra

    Returns
    -------
    :
        None
    """

    # Convert to cm
    r = raw_uc(r, "m", "cm")
    z = raw_uc(z, "m", "cm")
    dr = raw_uc(dr, "m", "cm")
    dz = raw_uc(dz, "m", "cm")

    # Half widths of 'cells'
    drad = dr / 2
    dzed = dz / 2

    # For each non-anetronic reaction, write a separate SDEF file
    for reaction in strength:

        if not reaction in AneutronicReactions:

            # Create file name
            short_react = re.findall("[DT]", reaction.label)
            file_name = file + "." + short_react[0] + short_react[1]

            # Open the file
            with open(file_name, "w") as sdef_file:
                
                # Write the SDEF header information
                sdef_file.write(
                    sdef_header(
                        reaction, 
                        sum(strength[reaction]), 
                        mean_ion_temp(strength[reaction],temperature)
                        )
                    )

                OFFSET = 5  # First 5 distribution are reserved


                # Calculate the radial boundaries based on the ring centres and the 'cell width' (dr)
                r_bounds = np.unique(r)-drad
                r_bounds = np.append(r_bounds, r_bounds[-1]+dr)  # Add the last boundary

                # Identify indices where radial position changes (to identify the range of each vertical distribution)
                z_ints = np.where(r[1:] != r[:-1])[0]
                z_ints = np.insert(z_ints, 0, -1)       # Add the first index
                z_ints = np.append(z_ints, len(r)-1)    # Add the last index


                # Write the radial distribution bin boundaries (which we set to the SI3 and SP3 cards)
                si_card = "SI3 H " + " ".join(f"{ri:.5e}" for ri in r_bounds) + "\n"
                # Write the radial distribution probabilities (these are the sum of the vertical distribution strengths for a given radial bin)
                sp_card = "SP3 D " + f"{0.0:.5e} " + " ".join(f"{np.sum(strength[reaction][z_ints[i]+1:z_ints[i+1]+1]):.5e}" for i in range(len(z_ints)-1)) + "\n"

                # We use the DS4 card as the dependent distribution numbers for the vertical distributions
                ds_card = "DS4 S " + " ".join(f"{i+OFFSET:d}" for i in range(len(r_bounds)-1)) + "\n"
                
                # Write the radial distribution to the sdef file
                sdef_file.write(insert_linebreak(si_card))
                sdef_file.write(insert_linebreak(sp_card))
                sdef_file.write(insert_linebreak(ds_card))

                sdef_file.write("C\n")
                sdef_file.write("C 3. Neutron Emission Probability - Vertical Distribution\n")
                sdef_file.write("C\n")

                # Write the vertical distribution for each radius (which we set to the SI and SP cards listed on the DS4 card)
                for i in range(len(z_ints)-1):
                    # Write the vertical distribution bin boundaries (which we set to the SI card)
                    si_card = "SI" + str(i+OFFSET) + " H " + " ".join(f"{zi:.5e}" for zi in z[z_ints[i]+1:z_ints[i+1]+1]-dzed) + " " + f"{z[z_ints[i+1]]+dzed:.5e}" + "\n"
                    # Write the vertical distribution probabilities (which we set to the SP card)
                    sp_card = "SP" + str(i+OFFSET) + " D " + f"{0.0:.5e} " + " ".join(f"{s:.5e}" for s in strength[reaction][z_ints[i]+1:z_ints[i+1]+1]) + "\n"
                    
                    # Write the vertical distribution to the sdef file
                    sdef_file.write(insert_linebreak(si_card, indent=len(str(i + OFFSET)) + 5))
                    sdef_file.write(insert_linebreak(sp_card, indent=len(str(i + OFFSET)) + 5))

        else:
            print(f"Skipping reaction {reaction.label} for MCNP SDEF source.")




def sdef_header(reaction: Reactions, strength: float, ion_temp: float):
                #source_strength: float) -> str:
    
    # Read authors and git address from CITATION.cff
    citation = load_citation()
    authors = citation.get('authors', [])
    gitaddr = citation.get('repository-code', '')

    # Create the SDEF header
    header = """C ============================
C SDEF Card for Tokamak Neutron Source generated by:
"""
    header += "C {}\n".format(gitaddr)
    header += """C ============================
C
C ============================
C Authors:
"""
    for author in authors:
        header += "C    {} {}, {}\n".format(author.get('given-names', ''),author.get('family-names',''),author.get('affiliation',''))
    header += """C ============================
C 
C ============================
C Method:
C 1. Create a cylinder that encloses the entire torus.
C 2. Then slice the cylinder along the R-axis.
C 3. Finally, define the vertical distribution, assuming rotational symmetry.
C ============================
C
C ============================
"""
    header += "C Reaction channel: {}\n".format(reaction.label)
    header += "C Total source neutrons: {:5e} n/s\n".format(strength)

    header +="""C ============================
C
C 1. Neutron Emission Probability - Set up cylindrical source
C
sdef erg=d2 par=1 wgt=1
      pos = 0 0 0    $ Center = origin
      axs = 0 0 1    $ Cylinder points along the Z axis
      rad = d3       $ radial distribution defined by distribution 3
      ext = frad d4  $ extent distribution defined by distribution 4 which is dependent on distribution rad
"""
    
    # Neutron energy distribution (use MCNP's built-in gaussian spectrum for D-T and D-D reactions)
    if reaction == Reactions.D_T or reaction == Reactions.D_D:
        header += "SP2 -4 {:5e} {}\n".format(
                                    raw_uc(ion_temp, "keV", "MeV"),                           # convert to MeV from keV
                                    -1 if reaction == Reactions.D_T else -2  # -1 for D-T and -2 for D-D
                                    )

    # Neutron energy distribution (use tabulated data for T-T reaction)
    elif reaction == Reactions.T_T:
        energies, probabilities = energy_spectrum(ion_temp, reaction, EnergySpectrumMethod.DATA)
        header += "SI2 H " + insert_linebreak(f"{0.0:.5e} " + " ".join(f"{e:.5e}" for e in raw_uc(energies, "keV", "MeV")))
        header += "SP2 D " + insert_linebreak(f"{0.0:.5e} " + " ".join(f"{p:.5e}" for p in probabilities))

    header +="""C
C 2. Neutron Emission Probability - Radial Distribution
C
"""
    # return header
    return(header)


def mean_ion_temp(strength: npt.NDArray, temperature: npt.NDArray) -> float:
    """Calculate the strength-weighted mean ion temperature."""
    return np.sum(strength * temperature) / np.sum(strength)


def insert_linebreak(long_line: str, indent: int = 6) -> str:
    """
    Break lines such that they're never longer than MAXLENGTH characters.

    Parameters
    ----------
    long_line:
        A string that requires to be broken down.

    Returns
    -------
    string
        Same string broken into multiple lines with a trailing newline character
    """
    MAXLENGTH = 80

    # validate indent number matches MCNP syntax.
    if indent < 6:  # noqa: PLR2004
        raise SyntaxError(
            "MCNP input file interpret 5 or fewer indents as a new line, "
            "rather than a continued line broken from the previous.",
        )
    # Use regex to break line up.

    word_list = long_line.rstrip().split()
    lines = [
        word_list[0],
    ]
    for word in word_list[1:]:
        extended_line = lines[-1] + " " + word
        if len(extended_line) > MAXLENGTH:
            lines.append(" " * indent + word)
        else:
            lines[-1] = extended_line
    return "\n".join(lines) + "\n"