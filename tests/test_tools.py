# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy

import numpy as np
import pytest
from eqdsk import EQDSKInterface

from tokamak_neutron_source.tools import get_centroid_2d, load_eqdsk


class TestGetCentroid:
    def test_simple(self):
        x = np.array([0, 2, 2, 0, 0])
        y = np.array([0, 0, 2, 2, 0])
        xc, yc = get_centroid_2d(x, y)
        assert np.isclose(xc, 1)
        assert np.isclose(yc, 1)
        xc, yc = get_centroid_2d(np.array(x[::-1]), np.array(y[::-1]))
        assert np.isclose(xc, 1)
        assert np.isclose(yc, 1)

    def test_negative(self):
        x = np.array([0, -2, -2, 0, 0])
        y = np.array([0, 0, -2, -2, 0])
        xc, yc = get_centroid_2d(x, y)
        assert np.isclose(xc, -1)
        assert np.isclose(yc, -1)
        xc, yc = get_centroid_2d(np.array(x[::-1]), np.array(y[::-1]))
        assert np.isclose(xc, -1)
        assert np.isclose(yc, -1)


class TestLoadEQDSK:
    eq = EQDSKInterface.from_file("tests/test_data/eqref_OOB.json", from_cocos=7)

    @pytest.mark.parametrize(
        "cocos", [1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    def test_load_psi(self, cocos):
        neq = deepcopy(self.eq).to_cocos(cocos)
        eq = load_eqdsk(neq)
        assert eq.psimag > eq.psibdry
