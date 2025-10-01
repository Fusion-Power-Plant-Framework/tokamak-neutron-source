# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from dataclasses import astuple

import pytest

from tokamak_neutron_source.transport import FractionalFuelComposition


@pytest.mark.parametrize(
    ("f_d", "f_t"),
    [
        (0.5, 0.5),
        (0.1, 0.9),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_ff_composition(f_d, f_t):
    ffc = FractionalFuelComposition(f_d, f_t)
    assert f_d == ffc.D
    assert f_t == ffc.T


@pytest.mark.parametrize(
    ("f_d", "f_t"),
    [
        (0.5, 0.6),
        (0.1, 0.19),
        (0.0, 1.1),
        (1.00001, 0.0),
    ],
)
def test_ffc_fail(f_d, f_t):
    assert sum(astuple(FractionalFuelComposition(f_d, f_t))) == 1.0
