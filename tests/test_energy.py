# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from tokamak_neutron_source.energy import energy_spectrum
from tokamak_neutron_source.energy_data import (
    BALLABIO_DD_NEUTRON,
    BALLABIO_DT_NEUTRON,
)
from tokamak_neutron_source.reactions import Reactions
from tokamak_neutron_source.tools import trapezoid


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


@pytest.mark.parametrize("reaction", [Reactions.D_D, Reactions.D_T, Reactions.T_T])
def test_normalised_spectra(reaction):
    energy, prob = energy_spectrum(20.0, reaction)

    assert np.isclose(trapezoid(prob, energy), 1.0)
