# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Example Reading from JETTO files"""

# %%
from pathlib import Path
import re

import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

from tokamak_neutron_source import (
    FluxConvention,
    FluxMap,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.reactions import Reactions

# %% [markdown]
# # Load an arbitrary source.
# For this example the same source as from_jetto.ex.py is used. One can equally use the
# source from from_eqdsk.ex.py.

# %%
CELL_SIDE_LENGTH = 0.3  # [m]
source = TokamakNeutronSource(
    transport=TransportInformation.from_jetto("tests/test_data/STEP_jetto.jsp"),
    flux_map=FluxMap.from_eqdsk(
        "tests/test_data/STEP_jetto.eqdsk_out", flux_convention=FluxConvention.SQRT
    ),
    source_type=[Reactions.D_D, Reactions.D_T, Reactions.T_T],
    cell_side_length=0.3,
)
source.normalise_fusion_power(2.2E9)
sdef_path_root = Path("examples/example_sdef")
openmc_source = source.to_sdef_card(sdef_path_root)

# %% [markdown]
# ## Read the SDEF files created.
# This should create as many SDEF files as there are reactions; in the case, 3 files are
# made. The user can use these 3 files sequentially in 3 respective simulations,
# multiply each simulation's tally result by the **total number of neutrons** generated
# by that reaction, and then sum them together to obtain the desired quantity (e.g.
# neutron damage, heating, etc.)
# 
# ### How to get the **total number of neutrons** from each reaction
# Note the line printed in the header section of each file that says "Total source
# neutrons". This records the number of neutrons emitted per second through that reaction
# for a reactor that is operating at the specified fusion power of 2.2E9 W.

# %%
for reaction in Reactions:
    reactant = re.findall(r"[DT]", reaction.label)
    with open(sdef_path_root.as_posix()+"."+reactant[0]+reactant[1], "r") as sdef:
        print(sdef.read())


# %% [markdown]
# ## Clean up the example SDEF files created.

# %%
for reaction in Reactions:
    reactant = re.findall(r"[DT]", reaction.label)
    Path(sdef_path_root.as_posix()+"."+reactant[0]+reactant[1]).unlink(missing_ok=True)
