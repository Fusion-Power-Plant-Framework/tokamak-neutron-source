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

"""Reactivities."""
# %%

import matplotlib.pyplot as plt
import numpy as np

from tokamak_neutron_source.reactions import AneutronicReactions, Reactions
from tokamak_neutron_source.reactivity import (
    ReactivityMethod,
    reactivity,
)

# %% [markdown]
# # Reactivities
# Here we show the reactivities of the various fusion reactions implemented in
# `tokamak_neutron_source`.

# We can also compare the two available methods:
# * Cross-section integration (Maxwellian assumption)
# * Bosch-Hale parameterisations
#
# The Bosch-Hale parameterisations for reactivity are used by default,
# as they are ubiquitous in fusion research.
# However, they are not available for all reactions, and have limited validity
# above > 100 keV.
# %%

temperature = np.logspace(0, 3, 1000)

f, ax = plt.subplots(figsize=[12, 10])

xs_reactivities = {}
for reaction in [
    Reactions.D_T,
    Reactions.D_D,
    AneutronicReactions.D_D,
    Reactions.T_T,
    AneutronicReactions.D_He3,
]:
    xs_reactivities[reaction] = reactivity(
        temperature, reaction, method=ReactivityMethod.XS
    )
    ax.loglog(
        temperature, xs_reactivities[reaction], lw=1.5, label=f"{reaction.label} XS"
    )

bh_reactivities = {}
for reaction in [Reactions.D_T, Reactions.D_D, AneutronicReactions.D_D]:
    bh_reactivities[reaction] = reactivity(
        temperature, reaction, method=ReactivityMethod.BOSCH_HALE
    )
    ax.loglog(
        temperature,
        bh_reactivities[reaction],
        "--",
        lw=1.5,
        label=f"{reaction.label} Bosch-Hale",
    )

ax.set_ylim([1e-27, 2e-21])
xticks = np.array([1, 10, 100, 1000])
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])
ax.grid(visible=True, which="both", axis="both")
ax.set_xlabel("Temperature [keV]")
ax.set_ylabel(r"$<\sigma v>$ [m$^3$/s]")
ax.legend()
plt.show()
f.savefig("reactivities.svg", dpi=600, format="svg")
