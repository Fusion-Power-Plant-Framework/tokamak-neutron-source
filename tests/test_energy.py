# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tokamak_neutron_source.energy import (
    _gaussian_energy_spectrum,
    _modified_gaussian_energy_spectrum,
    energy_spectrum,
)
from tokamak_neutron_source.energy_data import (
    BALLABIO_DD_NEUTRON,
    BALLABIO_DT_NEUTRON,
    TT_N_SPECTRUM,
)
from tokamak_neutron_source.reactions import Reactions


class TestEnergyShift:
    """Values graphically determined from Ballabio et al., 1998"""

    @pytest.mark.parametrize(
        ("reaction_spectrum", "temperature", "expected"),
        [
            (BALLABIO_DT_NEUTRON, 0.0, 0.0),
            (BALLABIO_DD_NEUTRON, 0.0, 0.0),
            (BALLABIO_DT_NEUTRON, 20.0, 52.0),
            (BALLABIO_DD_NEUTRON, 20.0, 58.0),
        ],
    )
    def test_energy_shift(self, reaction_spectrum, temperature, expected):
        e_shift = reaction_spectrum.energy_shift(temperature)
        assert np.isclose(e_shift, expected, rtol=1e-2, atol=0.0)


def test_ballabio_dt_spectra_methods():
    """
    Comparison of normal vs modified Gaussian with Ballabio
    """
    T = 20
    mu = Reactions.D_T.ballabio_spectrum.mean_energy(T)
    sigma = Reactions.D_T.ballabio_spectrum.std_deviation(T)

    energy1, g_pdf = _gaussian_energy_spectrum(mu, sigma)
    energy2, mg_pdf = _modified_gaussian_energy_spectrum(mu, sigma)
    _f, ax = plt.subplots()
    ax.plot(energy1, g_pdf, label="Gaussian")
    ax.plot(energy2, mg_pdf, label="Modified Gaussian")
    ax.legend()
    ax.set_xlim([12.0e3, 16.0e3])
    plt.show()


def test_dt_sprectum_plot():
    """
    Attempt to match Fig 5 of Ballabio et al., 1998. Frustratingly don't know what
    the arbitrary units are.
    """
    s = BALLABIO_DT_NEUTRON
    ti = 20.0

    mu = s.mean_energy(ti)
    sigma = s.std_deviation(ti)

    energy = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 1000)
    energy, prob = _modified_gaussian_energy_spectrum(mu, sigma)

    _f, ax = plt.subplots()
    ax.semilogy(energy, prob)

    ax.set_xlim([12e3, 17e3])
    ax.set_ylim([10e-11, 10e-3])
    ax.set_xlabel("En [keV]")
    ax.set_ylabel("[a. u.]")
    plt.show()


def test_dt_shift_plot():
    """
    Attempt to match Fig 4 of Ballabio et al., 1998
    """
    s = BALLABIO_DT_NEUTRON
    _f, ax = plt.subplots()
    t = np.linspace(0, 20, 1000)  # [keV]
    delta_e_th = s.energy_shift(t)
    ax.plot(t, delta_e_th)
    ax.set_xlim([0, 20.0])
    ax.set_ylim([0, 80.0])
    ax.set_xlabel("Ti [keV]")

    ax.set_ylabel(r"$\Delta E_{th}$ [keV]")
    plt.show()


def test_tt_spectra_plot():
    """
    Attempt to match Fig 1 of Appelbe and Chittenden, 2016
    """
    d = TT_N_SPECTRUM
    t = [1.0, 5.0, 10.0, 20.0]

    f, ax = plt.subplots()
    for temp in t:
        energy, intensity = d(temp)
        ax.plot(energy, intensity, label=f"Ti = {temp} keV")
    ax.legend()
    ax.set_title("T-T neutron energy spectrum")
    ax.set_xlabel(r"$E_n$ [MeV]")
    ax.set_ylabel("Intensity (arb. units)")
    plt.show()
