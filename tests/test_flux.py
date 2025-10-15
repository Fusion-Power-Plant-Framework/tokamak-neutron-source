# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tokamak_neutron_source.error import FluxSurfaceError
from tokamak_neutron_source.flux import (
    ClosedFluxSurface,
    FausserFluxSurface,
    FluxConvention,
    FluxMap,
    LCFSInformation,
)


def circle(r0, z0, radius, n_points):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
    r = r0 + radius * np.cos(theta)
    z = z0 + radius * np.sin(theta)
    return r, z


class TestClosedFluxSurface:
    def test_closed_error(self):
        with pytest.raises(FluxSurfaceError):
            ClosedFluxSurface(
                np.array([0.0, 0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 1.0, 2.0]),
            )

    def test_area(self):
        radius = 4.0
        x, z = circle(9.0, 0.0, radius, 500)
        surface = ClosedFluxSurface(x, z)
        area = surface.area
        true_area = np.pi * radius**2
        assert np.isclose(area, true_area, rtol=1e-4, atol=1e-8)

    def test_volume(self):
        major_radius = 9.0
        radius = 4.0
        x, z = circle(major_radius, 0.0, radius, 500)
        surface = ClosedFluxSurface(x, z)
        volume = surface.volume
        true_volume = 2 * np.pi * major_radius * np.pi * radius**2
        assert np.isclose(volume, true_volume, rtol=1e-4, atol=1e-8)


class TestLCFSInformation:
    def test_minor_radius(self):
        lcfs = LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 0.5)
        assert np.isclose(lcfs.minor_radius, 9.0 / 3.1)

    def test_shafranov_shift_error(self):
        with pytest.raises(FluxSurfaceError):
            LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 4.0)

    def test_elongation_error(self):
        with pytest.raises(FluxSurfaceError):
            LCFSInformation(9.0, 0.0, 3.1, 0.8, 0.4, 0.0)

    def test_triangularity_error(self):
        with pytest.raises(FluxSurfaceError):
            LCFSInformation(9.0, 0.0, 3.1, 1.8, -1.4, 0.5)

    def test_abs_shafranov(self):
        lcfs = LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, -0.5)
        assert lcfs.shafranov_shift == 0.5


class TestFausserFluxSurface:
    def test_volume(self):
        psi_norm = np.linspace(0, 1, 50)
        info = LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 0.5)
        parameterisation = FausserFluxSurface(info)

        volume = -np.inf
        for psi_n in psi_norm:
            s = parameterisation.flux_surface(psi_n, 100)
            assert s.volume > volume
            volume = s.volume


class TestFluxMapFromParameterisation:
    lcfs_info = LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 0.5)
    flux_map = FluxMap.from_parameterisation(
        FausserFluxSurface(lcfs_info),
        np.linspace(0, 1, 30),
        100,
    )

    def test_flux_surface_single(self):
        psi_norm = np.linspace(0, 1, 5)
        volume = -np.inf
        for psi_n in psi_norm:
            s = self.flux_map.get_flux_surface(psi_n)
            assert s.volume > volume
            volume = s.volume

    def test_flux_surface_array(self):
        psi_norm = np.linspace(0, 1, 5)
        fs = self.flux_map.get_flux_surface(psi_norm)
        areas = [s.area for s in fs]
        assert np.all(np.diff(areas) >= 0)

    def test_volume(self):
        psi_norm = np.linspace(0, 1, 5)
        volume = self.flux_map.volume(psi_norm)
        assert np.all(np.diff(volume) >= 0)

    def test_psi_norm(self):
        x_mag = self.flux_map.o_point.x
        z_mag = self.flux_map.o_point.z
        lcfs = self.flux_map.get_flux_surface(1.0)
        idx = np.where(abs(lcfs.z - z_mag) < 0.1)[0]
        idx = idx[np.argmax(abs(lcfs.x[idx] - x_mag))]

        z = z_mag * np.ones(30)
        x = np.linspace(x_mag, lcfs.x[idx], 30)
        psi_norm = self.flux_map.psi_norm(x, z)
        assert np.all(np.diff(psi_norm) >= 0)

    def test_psi_norm_core(self):
        x_mag = self.flux_map.o_point.x
        z_mag = self.flux_map.o_point.z
        assert np.isclose(
            self.flux_map.psi_norm(x_mag, z_mag), 0.0, rtol=0.0, atol=1e-14
        )

    def test_plot(self):
        _f, ax = self.flux_map.plot()
        assert isinstance(ax, plt.Axes)


TEST_DATA = Path(__file__).parent / "test_data"


# Fixture yields one FluxMap per eqdsk_name
@pytest.fixture(
    scope="class",
    params=[
        Path(TEST_DATA, "DN-DEMO_eqref.json").as_posix(),
        Path(TEST_DATA, "eqref_OOB.json").as_posix(),
        Path(TEST_DATA, "jetto_600_100000.eqdsk").as_posix(),
    ],
)
def flux_map(request):
    if request.param.startswith("jetto"):
        convention = FluxConvention.SQRT
    else:
        convention = FluxConvention.LINEAR
    return FluxMap.from_eqdsk(request.param, flux_convention=convention)


@pytest.mark.usefixtures("flux_map")
class TestFluxMapFromEQDSK:
    def test_flux_surface_single(self, flux_map):
        psi_norm = np.linspace(0, 1, 5)
        volume = -np.inf
        for psi_n in psi_norm:
            s = flux_map.get_flux_surface(psi_n)
            assert s.volume > volume
            volume = s.volume

    def test_flux_surface_array(self, flux_map):
        psi_norm = np.linspace(0, 1, 5)
        fs = flux_map.get_flux_surface(psi_norm)
        areas = [s.area for s in fs]
        assert np.all(np.diff(areas) >= 0)

    def test_volume(self, flux_map):
        psi_norm = np.linspace(0, 1, 5)
        volume = flux_map.volume(psi_norm)
        assert np.all(np.diff(volume) >= 0)

    def test_psi_norm(self, flux_map):
        x_mag = flux_map.o_point.x
        z_mag = flux_map.o_point.z
        lcfs = flux_map.get_flux_surface(1.0)
        idx = np.where(abs(lcfs.z - z_mag) < 0.1)[0]
        idx = idx[np.argmax(abs(lcfs.x[idx] - x_mag))]

        z = z_mag * np.ones(30)
        x = np.linspace(x_mag, lcfs.x[idx], 30)
        psi_norm = flux_map.psi_norm(x, z)
        assert np.all(np.diff(psi_norm) >= 0)

    def test_psi_norm_core(self, flux_map):
        x_mag = flux_map.o_point.x
        z_mag = flux_map.o_point.z
        psi_norm = flux_map.psi_norm(x_mag, z_mag)
        assert np.isclose(psi_norm, 0.0, rtol=0.0, atol=3e-4)

    def test_plot(self, flux_map):
        _f, ax = flux_map.plot()
        assert isinstance(ax, plt.Axes)

    def test_interpolate_fs(self, flux_map):
        fs = flux_map.get_flux_surface(0.6)
        fs_interp = flux_map.get_flux_surface(0.6, 123)

        assert np.isclose(fs.area, fs_interp.area, rtol=1e-3)
