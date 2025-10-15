# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Example Reading from JETTO files"""
# %%

from tokamak_neutron_source import (
    FluxConvention,
    FluxMap,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.reactions import Reactions

# %% [markdown]
# # JETTO Source
# These JETTO data files are available here:
# https://github.com/ukaea/openstep
#
# A copy is available in the `tests/test_data` folder

# %%

source = TokamakNeutronSource(
    transport=TransportInformation.from_jetto("tests/test_data/STEP_jetto.jsp"),
    flux_map=FluxMap.from_eqdsk(
        "tests/test_data/STEP_jetto.eqdsk_out", flux_convention=FluxConvention.SQRT
    ),
    source_type=[Reactions.D_T, Reactions.D_D],
    cell_side_length=0.05,
)

# Print the calculated total neutron rate from the ion density and temperature profiles
print("Total D-D source neutrons:", sum(source.strength[Reactions.D_D]))

print("Calculated total source fusion power: ", source.calculate_total_fusion_power())

source.plot()
