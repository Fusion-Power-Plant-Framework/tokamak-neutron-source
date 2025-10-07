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

import numpy as np

from tokamak_neutron_source import (
    FluxConvention,
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.reactions import Reactions

# %% [markdown]
# # JETTO Source
# %%
eqdsk_file = "tests/test_data/jetto_600_100000.eqdsk"
rho_profile = np.linspace(0, 1, 30)

# fmt:off
t_profile = 2.0 * np.array([
    15.2, 15.0, 14.8, 14.5, 14.0, 13.6, 13.1, 12.7, 12.0, 11.3,
    10.5, 9.7, 8.9, 8.1, 7.3, 6.5, 5.6, 4.8, 3.9, 3.0,
    2.2, 1.6, 1.1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0
])
n_profile = np.array([
    9.6, 9.7, 9.6, 9.5, 9.5, 9.4, 9.3, 9.3, 9.2, 9.1,
    9.0, 8.9, 8.8, 8.6, 8.4, 7.9, 7.2, 6.5, 5.5, 4.4,
    3.5, 2.7, 2.0, 1.4, 0.9, 0.6, 0.35, 0.2, 0.1, 0.0
])
# fmt:on

source = TokamakNeutronSource(
    transport=TransportInformation.from_profiles(
        ion_temperature_profile=t_profile,
        fuel_density_profile=n_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=FluxMap.from_eqdsk(eqdsk_file),
    source_type=[Reactions.D_T, Reactions.D_D],
    cell_side_length=0.1,
)
source.plot()


# %% [markdown]
# We can also change the flux convention to be in-line with the transport solver.

# %%
source = TokamakNeutronSource(
    transport=TransportInformation.from_profiles(
        ion_temperature_profile=t_profile,
        fuel_density_profile=n_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=FluxMap.from_eqdsk(
        eqdsk_file, flux_convention=FluxConvention.SQRT
    ),
    source_type=[Reactions.D_T, Reactions.D_D],
    cell_side_length=0.1,
)
source.plot()
