# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pytest
import numpy as np
from numpy import typing as npt
import openmc
import matplotlib.pyplot as plt

from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.constants import raw_uc
from tokamak_neutron_source.profile import ParabolicPedestalProfile

def extract_spacing(coordinate_array: npt.NDArray) -> float:
    return np.unique(np.diff((np.unique(coordinate_array))))[-1]

def make_universe_box(
    z_min: float, z_max: float, r_max: float,
) -> openmc.Cell:
    """Box up the universe in a cylinder (including top and bottom).

    Parameters
    ----------
    z_min:
        minimum z coordinate of the source
    z_max:
        maximum z coordinate of the source
    r_max:
        maximum r coordinate of the source
    
    Returns
    -------
    universe_cell
        An openmc.Cell that contains the entire source.
    """
    bottom = openmc.ZPlane(
        raw_uc(z_min, "m", "cm"),
        boundary_type="vacuum",
        surface_id=1,
        name="Universe bottom",
    )
    top = openmc.ZPlane(
        raw_uc(z_max, "m", "cm"),
        boundary_type="vacuum",
        surface_id=2,
        name="Universe top",
    )
    universe_cylinder = openmc.ZCylinder(
        r=raw_uc(r_max, "m", "cm"),
        surface_id=3,
        boundary_type="vacuum",
        name="Max radius of Universe",
    )
    return openmc.Cell(
        region=-top & +bottom & -universe_cylinder,
        fill=None,
        name="source cell"
    )

@dataclass
class OpenMCTrack:
    position: float
    direction: float
    energy: float  # eV
    time: float
    wgt: float
    cell_id: int
    cell_instance: int
    mat_id: int

    @property
    def position_cylindrical(self):
        return xyz_to_rphiz(*self.position)

    @property
    def direction_spherical(self):
        r, phi, z = xyz_to_rphiz(*self.direction)
        theta = np.atan2(r, z)
        return theta, phi


def xyz_to_rphiz(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.atan2(x,y)
    return r, phi, z

@pytest.mark.integration
class OpenMCSimulation:
    """A simple openmc simulation to create the particle tracks data. """
    temperature_profile = ParabolicPedestalProfile(25.0, 5.0, 0.1, 1.45, 2.0, 0.95)  # [keV]
    density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
    rho_profile = np.linspace(0, 1, 30)

    flux_map = FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json")
    source = TokamakNeutronSource(
        transport=TransportInformation.from_parameterisations(
            ion_temperature_profile=temperature_profile,
            fuel_density_profile=density_profile,
            rho_profile=rho_profile,
            fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
        ),
        flux_map=flux_map,
        cell_side_length=0.05,
    )
    # f, ax = source.plot()
    print(f"Total fusion power: {source.calculate_total_fusion_power() / 1e9} GW")
    source.normalise_fusion_power(2.2e9)
    print(f"Total fusion power: {source.calculate_total_fusion_power() / 1e9} GW")

    universe = openmc.Universe()
    dx, dz = extract_spacing(source.xz[:, 0]), extract_spacing(source.xz[:, 1])
    source_cell = make_universe_box(min(source.xz[:, 1])-dz, max(source.xz[:, 1])+dz, max(source.xz[:, 0])+dx)
    universe.add_cell(source_cell)
    geometry = openmc.Geometry(universe)

    # run an empty simulation
    settings = openmc.Settings(batches=1, run_mode="fixed source")
    openmc_source = source.to_openmc_source()
    settings.source = openmc_source
    settings.particles = settings.max_tracks = 10000
    materials = openmc.Materials()
    materials.cross_sections = "tests/test_data/cross_section.xml"
    # exporting to xml
    print("Openmc simulation exporting and started")
    geometry.export_to_xml()
    settings.export_to_xml()
    materials.export_to_xml()
    openmc.run(tracks=True)
    print("Openmc simulation completed.")
    tracks = openmc.Tracks("tracks.h5")
    Path("tracks.h5").unlink(missing_ok=True)
    Path("summary.h5").unlink(missing_ok=True)
    Path(f"statepoint.{settings.batches}.h5").unlink(missing_ok=True)
    Path("geometry.xml").unlink(missing_ok=True)
    Path("settings.xml").unlink(missing_ok=True)
    Path("materials.xml").unlink(missing_ok=True)
    # Should take about 2 minutes and 10 seconds for cell_side_length=0.05
    # (len(source)==17374).
    # Expected Leakage fraction = 1.0 since all neutrons should leave the source
    # cell (made of vacuum) without interacting with anything.
    locations, directions, energies = [], [], []
    print("Processing the track info into a useful format.")
    for ptrac in tracks:
        # particle_tracks should have len==1 since there shouldn't be any splitting
        # (lacking any obstacles in the simulation)
        start_state = OpenMCTrack(*ptrac.particle_tracks[0].states[0])
        # end_state = OpenMCTrack(*ptrac.particle_tracks[0].states[1])
        locations.append(start_state.position_cylindrical)
        directions.append(start_state.direction_spherical)
        energies.append(start_state.energy)
    locations, directions, energies = np.array(locations), np.array(directions), np.array(energies)

    @staticmethod
    def assert_is_uniform(array: npt.NDArray, known_range:Optional[tuple[float, float]]=None):
        if known_range:
            assert known_range[0]<=array.min()
            assert array.max()<=known_range[1]
        counts, bins = np.histogram(array, range=known_range)
        avg = counts.mean()
        # poisson distribution
        assert np.isclose(counts, avg, rtol=0, atol=3.5*np.sqrt(avg)).all(), "This test (3.5 sigmas) has a false positive failure rate of 0.046% per comparison."
        # 3.5 sigma should be enough.

    @staticmethod
    def assert_is_cosine(array: npt.NDArray):
        """
        Confirm the theta part of the spherical coordinate of an isotropic direction
        distribution follows a cosine curve.
        """
        counts, bins = np.histogram(array, bins=100, range=(-np.pi/2, np.pi/2))
        class_mark = bins[:-1] + np.diff(bins)/2
        cosine_dist = np.cos(class_mark)
        integral_area = 2
        scale_factor = counts.sum()/cosine_dist.sum()
        assert np.isclose(counts, cosine_dist*scale_factor, rtol=0, atol=3.5*np.sqrt(counts)).all()

    def test_location(self):
        """Confirm the sources are distributed uniformly in phi and according to the
        required distribution poloidally."""
        r, phi, z = self.locations.T
        self.assert_is_uniform(phi, (-np.pi, np.pi))
        plt.scatter(r/100, z/100, alpha=0.1, marker="o", s=0.5)
        plt.xlabel("r (m)"), plt.ylabel("z (m)")
        plt.title("Neutron generation positions\n(poloidal view)\nEach dot is a neutron emitted")
        o_point, lcfs = source.flux_map.o_point, source.flux_map.lcfs
        plt.scatter(o_point.x, o_point.z, label="o-point", facecolors='none', edgecolor="C1")
        plt.plot(lcfs.x, lcfs.z, label="LCFS")
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.show()

    def test_isotropic(self):
        """Confirm the sources are emitting neutrons isotropically."""
        dir_theta, dir_phi = self.directions.T
        self.assert_is_cosine(dir_theta)
        self.assert_is_uniform(dir_phi, (-np.pi, np.pi))

    def test_power_equal(self):
        """Confirm total power is close enough to the desired value."""
        mean_eV = self.energies.mean()
        assert np.isclose(mean_eV, 14E6, rtol=0, atol=0.1E6)

    def test_spectrum_at_known_temp(self):
        """Confirm neutron spectrum is as expected."""
        counts, bins = np.histogram(self.energies, bins=5000)
        class_mark = bins[:-1] + np.diff(bins)/2
        upper_E = class_mark[class_mark>10E6]
        upper_counts = counts[class_mark>10E6]
        upper_mean = (upper_E * upper_counts).sum()/upper_counts.sum()
        # Not enough particles to see the smaller DD peak.
        # assert np.isclose(lower_E[np.argmax(lower_counts)], 2.45E6, atol=0.05E6)
        assert np.isclose(upper_mean, 14.08E6, atol=0.1E6)
        plt.hist(self.energies, bins=5000)
        plt.title("Neutron spectrum across the entire reactor")
        plt.show()
