# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from eqdsk import EQDSKInterface

from tokamak_neutron_source.tools import (
    get_centroid_2d,
    load_citation,
    load_eqdsk,
    load_jsp,
)


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


TEST_DATA = Path(__file__).parent / "test_data"


class TestLoadJSP:
    path = Path(TEST_DATA, "STEP_jetto.jsp").as_posix()

    @pytest.mark.parametrize("bad_frame", [-10, 1e6])
    def test_error_on_bad_frame(self, bad_frame):
        pytest.importorskip("jetto_tools")
        with pytest.raises(ValueError, match="frame number"):
            load_jsp(self.path, bad_frame)

    def test_array_lengths(self):
        pytest.importorskip("jetto_tools")
        data = load_jsp(self.path)
        sizes = [
            data.rho.size,
            data.d_density.size,
            data.t_density.size,
            data.he3_density.size,
            data.ion_temperature.size,
        ]
        assert all(x == sizes[0] for x in sizes[1:])


@pytest.mark.xfail(reason="actions doesnt follow symlinks")
def test_load_citation():
    out = load_citation()
    assert out["licence"] == "LGPL-2.1-or-later"
