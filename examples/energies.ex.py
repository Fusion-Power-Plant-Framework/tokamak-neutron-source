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

"""Neutron energies."""
# %%

import matplotlib.pyplot as plt

from tokamak_neutron_source import Reactions
from tokamak_neutron_source.energy import EnergySpectrumMethod, energy_spectrum

# %% [markdown]
# # Neutron energy spectra

# %%

_f, ax = plt.subplots()

for reaction, color in zip(
    [Reactions.D_D, Reactions.D_T, Reactions.T_T], ["r", "g", "b"]
):
    for temperature, ls in zip([10.0, 20.0], ["-.", "-"]):
        e, pdf = energy_spectrum(temperature, reaction)
        ax.plot(
            e,
            pdf / max(pdf),
            color=color,
            ls=ls,
            label=f"{reaction.label}, T = {temperature} keV",
        )
ax.set_xlabel(r"$E_{n}$ [keV]")
ax.set_ylabel("[a. u.]")
ax.legend()
plt.show()


# %% [markdown]
# # Comparison between normal and modified Gaussian distributions from Ballabio et al.
#

# %%

temperature = 20.0  # [keV]

for reaction in [Reactions.D_D, Reactions.D_T]:
    energy1, g_pdf = energy_spectrum(
        temperature, reaction, method=EnergySpectrumMethod.BALLABIO_GAUSSIAN
    )
    energy2, mg_pdf = energy_spectrum(
        temperature, reaction, method=EnergySpectrumMethod.BALLABIO_M_GAUSSIAN
    )
    _f, ax = plt.subplots()
    ax.semilogy(energy1, g_pdf, label=f"{reaction.name} Gaussian")
    ax.semilogy(energy2, mg_pdf, label=f"{reaction.name} modified Gaussian")
    ax.set_xlabel(r"$E_{n}$ [keV]")
    ax.set_ylabel("[a. u.]")
    ax.legend()
    plt.show()
