# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from tokamak_neutron_source.profile import (
    DataProfile,
    ParabolicPedestalProfile,
    ParabolicProfile,
)


@pytest.mark.parametrize(
    "profile",
    [
        DataProfile(np.linspace(10, 1, 30), np.linspace(0, 1, 30)),
        ParabolicPedestalProfile(10, 1, 0.1, 1, 2, 0.8),
        ParabolicProfile(10, 2, 2),
    ],
)
def test_scale(profile):
    value = profile.value(0.5)
    profile.set_scale(2.0)
    new_value = profile.value(0.5)
    assert np.isclose(new_value, 2.0 * value)


@pytest.mark.parametrize(
    "profile",
    [
        DataProfile(np.linspace(10, 1, 30), np.linspace(0, 1, 30)),
        ParabolicPedestalProfile(10, 1, 0.1, 1, 2, 0.8),
        ParabolicProfile(10, 2, 2),
    ],
)
def test_float_array(profile):
    value = profile.value([0.5, 0.6])
    profile.set_scale(2.0)
    new_value = profile.value([0.5, 0.6])
    assert np.allclose(new_value, 2.0 * value)


@pytest.mark.parametrize(
    "profile",
    [
        DataProfile(np.linspace(10, 1, 30), np.linspace(0, 1, 30)),
        ParabolicPedestalProfile(10, 1, 0.1, 1, 2, 0.8),
        ParabolicProfile(10, 2, 2),
    ],
)
def test_rho_bounds_clipping(profile):
    value = profile.value([-0.0001, 1.0001])
    bound_value = profile.value([0.0, 1.0])
    assert np.allclose(value, bound_value)


def test_parabolic_no_pedestal():
    pp = ParabolicProfile(10, 1.0, 1.0)
    ppp = ParabolicPedestalProfile(10, 0.0, 0.0, 1.0, 1.0, 1.0)
    rho = np.linspace(0, 1, 20)
    pp_values = pp.value(rho)
    ppp_values = ppp.value(rho)
    assert np.allclose(pp_values, ppp_values)
