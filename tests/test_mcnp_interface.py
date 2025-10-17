# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""MCNP neutron source (SDEF) interface tester"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy import typing as npt

from tokamak_neutron_source import (
    FluxConvention,
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.profile import ParabolicPedestalProfile
from tokamak_neutron_source.reactions import Reactions


def make_source(source_name: str):
    match source_name:
        case "STEP_jetto":
            return TokamakNeutronSource(
                transport=TransportInformation.from_jetto(
                    "tests/test_data/STEP_jetto.jsp"
                ),
                flux_map=FluxMap.from_eqdsk(
                    "tests/test_data/STEP_jetto.eqdsk_out",
                    flux_convention=FluxConvention.SQRT,
                ),
                cell_side_length=0.3,
            )
        case "eqref_OOB":
            return TokamakNeutronSource(
                transport=TransportInformation.from_parameterisations(
                    ion_temperature_profile=ParabolicPedestalProfile(
                        25.0, 5.0, 0.1, 1.45, 2.0, 0.95
                    ),
                    fuel_density_profile=ParabolicPedestalProfile(
                        0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95
                    ),
                    rho_profile=np.linspace(0, 1, 30),
                    fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
                ),
                flux_map=FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json"),
                cell_side_length=0.3,
            )


SP_4_SCHEME = {Reactions.D_T: -1, Reactions.D_D: -2}


@pytest.mark.parametrize("source_name", ["STEP_jetto", "eqref_OOB"])
def test_sdef(source_name: str, tmp_path: Path) -> dict[str, str]:
    source = make_source(source_name)
    source.to_sdef_card(tmp_path / "test_sdef")

    for reaction in source.source_type:
        if reaction not in Reactions:
            continue
        reactant = re.findall(r"[DT]", reaction.label)
        local_file_name = "test_sdef." + reactant[0] + reactant[1]
        with open(tmp_path / local_file_name) as sdef_file:
            sdef_text = sdef_file.readlines()

        # Test the "Total source neutrons:" line
        source_neutrons_line, sdef_text = scroll_and_get_next_data_line(
            sdef_text, "C Total source neutrons:"
        )
        num_neutrons = float(source_neutrons_line[:-4].split()[-1])
        assert np.isclose(
            num_neutrons, source.num_neutrons_per_second[reaction], atol=0, rtol=1e-6
        ), "Matching number of neutrons"

        # Test sdef
        sdef, sdef_text = scroll_and_get_next_data_line(sdef_text, "sdef")
        assert bool(sdef), "SDEF file line must exist"

        # Test neutron energy distribution
        if local_file_name.endswith("TT"):
            _, sp, sdef_text = get_next_si_and_sp(sdef_text, 2)
        else:
            energy_dist_str, sdef_text = scroll_and_get_next_data_line(sdef_text, "SP2")
            sp = tokenize(energy_dist_str)
            assert int(sp.dist_type) == -4, "-4 for fusion neutron source of DD/DT type."
            ion_temp_mev = sp.data[0]
            assert 1e-3 <= ion_temp_mev <= 0.1, (
                "Between 1 to 100 keV of ion-temp expected."
            )
            assert sp.data[-1] == SP_4_SCHEME[reaction], "-1 for DT, -2 for DD."
        assert sp.def_number == 2

        # Test radial distribution
        _, sp, sdef_text = get_next_si_and_sp(sdef_text, 3)

        # get list of vertical distributions
        ds4, sdef_text = scroll_and_get_next_data_line(sdef_text, "DS4")
        ds4 = tokenize(ds4, dtype=int)
        assert ds4.dist_type == "S"
        for i in ds4.data:
            _, sp, sdef_text = get_next_si_and_sp(sdef_text, i)

        # Test end of file
        assert not scroll_and_get_next_data_line(sdef_text, "SP")[0], "EoF expected"


def scroll_and_get_next_data_line(
    text_list: list[str], match_string: str
) -> tuple[str, list[str]]:
    """
    Search the content of the file until the a row matching the match_string appears.
    Then, split just before this matched line.

    Returns
    -------
    :
        The matching data line, shrunken into a single string.
    text_list:
        the remaining text after that line, returned as a list of strings.
    """
    for i, line in enumerate(text_list):
        if line.startswith(match_string):
            j = i + 1
            while j < len(text_list) and text_list[j].startswith(" "):
                j += 1
            return " ".join(line.lstrip() for line in text_list[i:j]), text_list[j:]
    return "", []


MNEMONIC_TRANSLATION_DICT = {
    "SI": "source information",
    "SP": "source probability",
    "DS": "dependent source",
}
MNEMONIC_EXPLANATION_DICT = {
    "SI": "x-axis of a probability density function",
    "SP": "y-axis of a probability density function",
    "DS": "dependent source",
}


@dataclass
class MCNPDefinitionLine:
    """Data provided by one line in an MCNP input file."""

    mnemonic: str
    def_number: int
    dist_type: str | int
    data: npt.NDArray

    @property
    def _type(self):
        return MNEMONIC_TRANSLATION_DICT[self.mnemonic]

    @property
    def explanation(self):
        return MNEMONIC_EXPLANATION_DICT[self.mnemonic]


def tokenize(line: str, dtype=float):
    """Parse a line of MCNP input file.

    Returns
    -------
    :
        An instance of MCNPDefinitionLine
    """
    data = line.split()
    line_type, dist_num = data[0][:2], int(data[0][2:])
    if "." in data[1]:
        return MCNPDefinitionLine(
            line_type,
            dist_num,
            None,
            np.array(data[1:], dtype=dtype),
        )
    return MCNPDefinitionLine(
        line_type,
        dist_num,
        data[1],
        np.array(data[2:], dtype=dtype),
    )


def get_next_si_and_sp(
    text_list: list[str], i: int
) -> tuple[MCNPDefinitionLine, MCNPDefinitionLine, list[str]]:
    """
    Get the matching SIn and SPn lines, and the rest of the SDEF file below that.

    Returns
    -------
    si:
        The SIn line.
    sp:
        The SPn line.
    remaining_text_list:
        text, returned as a list.
    """
    si, remaining_texts = scroll_and_get_next_data_line(text_list, f"SI{i}")
    sp, remaining_texts = scroll_and_get_next_data_line(remaining_texts, f"SP{i}")
    si, sp = tokenize(si), tokenize(sp)
    assert (si.dist_type, sp.dist_type) in set(("H", "D"), ("A", None)), (
        "Either histogramic or tabular distribution expected."
    )
    assert len(si.data) == len(sp.data), "Equal number of x- and y-values expected."
    return si, sp, remaining_texts
