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

"""Example reading from eqdsk file"""
# %%

import numpy as np

from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.profile import ParabolicPedestalProfile

# %% [markdown]
# # Creation from an EQDSK file.

# %%
temperature_profile = ParabolicPedestalProfile(25.0, 5.0, 0.1, 1.45, 2.0, 0.95)  # [keV]
density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
rho_profile = np.linspace(0, 1, 30)

source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=temperature_profile,
        fuel_density_profile=density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json"),
    cell_side_length=0.05,
)
f, ax = source.plot()

# %% [markdown]
# # Normalising fusion power
# We can calculate the total fusion power from the reactions specified.
#
# However, this may not correspond to some reference value we obtain from
# another code. There are many reasons why this may be the case:
# * Different reactivity parameterisations or XS data
# * Different interpolation of poloidal magnetic flux
# * Limitations of cell-based discretisation (vs flux surface integrals)
#
# We can correct for this:

# %%
print(f"Total fusion power: {source.calculate_total_fusion_power() / 1e9} GW")
source.normalise_fusion_power(2.2e9)
print(f"Total fusion power: {source.calculate_total_fusion_power() / 1e9} GW")
