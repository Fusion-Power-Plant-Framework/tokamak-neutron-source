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

"""Example using the Fausser Flux surface parameterisation"""
# %%

import numpy as np

from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.flux import FausserFluxSurface, LCFSInformation
from tokamak_neutron_source.profile import ParabolicPedestalProfile
from tokamak_neutron_source.reactions import Reactions

# %% [markdown]
# # Fausser Source
# %%
temperature_profile = ParabolicPedestalProfile(30.0, 5, 0.1, 1.45, 2.0, 0.95)  # [keV]
density_profile = ParabolicPedestalProfile(1e20, 2e19, 0.5e16, 1.0, 2.0, 0.95)
rho_profile = np.linspace(0, 1, 500)

source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=temperature_profile,
        fuel_density_profile=density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=FluxMap.from_parameterisation(
        FausserFluxSurface(
            LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 0.5),
        ),
        rho_profile=rho_profile,
    ),
    source_type=[Reactions.D_T, Reactions.D_D],
    cell_side_length=0.1,
)
f, ax = source.plot()
