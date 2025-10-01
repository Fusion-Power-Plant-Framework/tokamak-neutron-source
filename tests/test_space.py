# SPDX-FileCopyrightText: 2021-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from tokamak_neutron_source.flux import ClosedFluxSurface, FluxPoint
from tokamak_neutron_source.space import sample_space_2d
from tokamak_neutron_source.tools import load_eqdsk


@pytest.mark.parametrize("cell_side_length", [0.1, 0.05, 0.025])
def test_sample_space_2d_simple(cell_side_length):
    lcfs = ClosedFluxSurface(x=[4, 8, 8, 4, 4], z=[-2, -2, 2, 2, -2])
    o_point = FluxPoint(*lcfs.center_of_mass, 0.0)

    x, z, dv = sample_space_2d(lcfs, o_point, cell_side_length)
    assert np.all((x >= 4) & (x <= 8))
    assert np.all((z >= -2) & (z <= 2))
    true_volume = lcfs.volume
    sampled_volume = np.sum(dv)
    assert np.isclose(true_volume, sampled_volume, rtol=2e-2, atol=1e-12)


@pytest.mark.parametrize("cell_side_length", [0.1, 0.05, 0.025])
def test_sample_space_2d_odd(cell_side_length):
    lcfs = ClosedFluxSurface(x=[4, 8, 6, 4], z=[-2, -2, 1, -2])
    o_point = FluxPoint(*lcfs.center_of_mass, 0.0)

    x, z, dv = sample_space_2d(lcfs, o_point, cell_side_length)
    assert np.all((x >= 4) & (x <= 8))
    assert np.all((z >= -2) & (z <= 2))
    true_volume = lcfs.volume
    sampled_volume = np.sum(dv)
    # This is because it is an equilateral triangle and we are using squares xD
    assert np.isclose(true_volume, sampled_volume, rtol=2e-2, atol=1e-12)


@pytest.mark.parametrize("cell_side_length", [0.2, 0.1, 0.05])
def test_sample_space_2d_odd2(cell_side_length):
    lcfs = ClosedFluxSurface(x=[4, 8, 6, 4], z=[-2, -2, 0.75, -2])
    o_point = FluxPoint(*lcfs.center_of_mass, 0.0)

    x, z, dv = sample_space_2d(lcfs, o_point, cell_side_length)
    assert np.all((x >= 4) & (x <= 8))
    assert np.all((z >= -2) & (z <= 2))
    true_volume = lcfs.volume
    sampled_volume = np.sum(dv)
    assert np.isclose(true_volume, sampled_volume, rtol=3e-2, atol=1e-12)


@pytest.mark.parametrize("cell_side_length", [0.2, 0.1, 0.05])
def test_sample_space_2d_real(cell_side_length):
    eq = load_eqdsk("tests/test_data/eqref_OOB.json")
    lcfs = ClosedFluxSurface(eq.xbdry, eq.zbdry)
    o_point = FluxPoint(eq.xmag, eq.zmag, eq.psimag)
    _, _, dv = sample_space_2d(lcfs, o_point, cell_side_length)
    true_volume = lcfs.volume
    sampled_volume = np.sum(dv)
    assert np.isclose(true_volume, sampled_volume, rtol=2e-3, atol=1e-12)
