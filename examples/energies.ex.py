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
# Here we look in detail at the neutron energy spectra for the D-T, D-D, and T-T
# fusion reactions.
#
# The D-T and D-D spectra are calculated following
# [Ballabio et al.'s](https://iopscience.iop.org/article/10.1088/0029-5515/38/11/310)
# parameterisations.
#
# The T-T spectra are interpolated from data produced by
# [Appelbe and Chittenden](https://www.sciencedirect.com/science/article/abs/pii/S1574181816300295).

# %%

_f, ax = plt.subplots()

for reaction, color in zip(
    [Reactions.D_D, Reactions.D_T, Reactions.T_T], ["r", "g", "b"], strict=False
):
    for temperature, ls in zip([10.0, 20.0], ["-.", "-"], strict=False):
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
# Here we attempt to recreate Fig. 5 of Ballabio et al

# %%

temperature = 20.0
energy, prob = energy_spectrum(temperature, Reactions.D_T)

_f, ax = plt.subplots()
ax.semilogy(energy, prob)
ax.set_xlim([12e3, 17e3])
ax.set_ylim([10e-11, 10e-3])
ax.set_title("D-T neutron spectrum at 20 keV")
ax.set_xlabel("En [keV]")
ax.set_ylabel("[a. u.]")
plt.show()


# %% [markdown]
# # Comparison between normal and modified Gaussian distributions from Ballabio et al.
# The differences are subtle, default is to use the modified Gaussian as specified
# in the paper.

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
    ax.set_title(f"{reaction.name} neutron energy spectrum at 20 keV")
    ax.semilogy(energy1, g_pdf, label=f"{reaction.name} Gaussian")
    ax.semilogy(energy2, mg_pdf, label=f"{reaction.name} modified Gaussian")
    ax.set_xlabel(r"$E_{n}$ [keV]")
    ax.set_ylabel("[a. u.]")
    ax.legend()
    plt.show()


# %% [markdown]
# Here we recreate Fig 1. of Appelbe and Chittenden
# %%
temperatures = [1.0, 5.0, 10.0, 20.0]

f, ax = plt.subplots()
for temp in temperatures:
    energy, intensity = energy_spectrum(temp, Reactions.T_T, EnergySpectrumMethod.DATA)
    ax.plot(energy, intensity / max(intensity), label=f"Ti = {temp} keV")
ax.legend()
ax.set_title("T-T neutron energy spectrum")
ax.set_xlim([1e3, 1e4])
ax.set_ylim([0, 1])
ax.set_xlabel(r"$E_n$ [keV]")
ax.set_ylabel("[a. u.]")
plt.show()
