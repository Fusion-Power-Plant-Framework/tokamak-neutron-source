# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy import typing as npt

from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.constants import raw_uc
from tokamak_neutron_source.energy import EnergySpectrumMethod
from tokamak_neutron_source.profile import ParabolicPedestalProfile
from tokamak_neutron_source.reactions import _APPROX_NEUTRON_ENERGY, Reactions

if TYPE_CHECKING:
    import openmc

CELL_SIDE_LENGTH = 0.05


def make_universe_cylinder(
    z_min: float,
    z_max: float,
    r_max: float,
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
    import openmc  # noqa: PLC0415

    bottom = openmc.ZPlane(
        raw_uc(z_min, "m", "cm"),
        boundary_type="vacuum",
        name="Universe bottom",
    )
    top = openmc.ZPlane(
        raw_uc(z_max, "m", "cm"),
        boundary_type="vacuum",
        name="Universe top",
    )
    universe_cylinder = openmc.ZCylinder(
        r=raw_uc(r_max, "m", "cm"),
        boundary_type="vacuum",
        name="Max radius of Universe",
    )
    return openmc.Cell(
        region=-top & +bottom & -universe_cylinder, fill=None, name="source cell"
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
    def position_cylindrical(self) -> tuple[np.float64, np.float64, np.float64]:
        return xyz_to_rphiz(*self.position)

    @property
    def direction_spherical(self) -> tuple[np.float64, np.float64]:
        r, phi, z = xyz_to_rphiz(*self.direction)
        theta = np.atan2(z, r)
        return theta, phi


@dataclass
class OpenMCSimulatedSourceParticles:
    source: openmc.Source
    locations: npt.NDArray
    directions: npt.NDArray
    energies: npt.NDArray


def xyz_to_rphiz(x, y, z) -> tuple[np.float64, np.float64, np.float64]:
    r = np.sqrt(x**2 + y**2)
    phi = np.atan2(x, y)
    return r, phi, z


def run_openmc_sim(source, tmp_path, method) -> openmc.Tracks:
    # run an empty simulation
    import openmc  # noqa: PLC0415

    geometry = source_creation(source)
    settings = openmc.Settings(
        batches=1,
        run_mode="fixed source",
        output={"path": tmp_path.as_posix(), "summary": False},
    )
    settings.seed = 1
    settings.source = (
        source.to_openmc_source(method) if method else source.to_openmc_source()
    )
    settings.particles = settings.max_tracks = 1000
    materials = openmc.Materials()
    materials.cross_sections = "tests/test_data/cross_section.xml"
    # exporting to xml
    geometry.export_to_xml(tmp_path / "geometry.xml")
    settings.export_to_xml(tmp_path / "settings.xml")
    materials.export_to_xml(tmp_path / "materials.xml")
    openmc.run(cwd=tmp_path.as_posix(), tracks=True)
    return openmc.Tracks(tmp_path / "tracks.h5")


def source_creation(source) -> openmc.Geometry:
    """Make the openmc universe

    Returns
    -------
    geometry:
        an openmc.Geometry that contains only one cell (the source cell, which spans
        the entire universe).
    """
    import openmc  # noqa: PLC0415

    dx, dz = CELL_SIDE_LENGTH, CELL_SIDE_LENGTH
    source_cell = make_universe_cylinder(
        min(source.z) - dz, max(source.z) + dz, max(source.x) + dx
    )
    universe = openmc.Universe(cells=[source_cell])
    return openmc.Geometry(universe)


@pytest.fixture(scope="module", autouse=True)
def omc_path(tmp_path_factory):
    return tmp_path_factory.mktemp("openmc_output")


@pytest.fixture(
    scope="module",
    params=[
        (None, None),
        (Reactions.D_D, EnergySpectrumMethod.BALLABIO_GAUSSIAN),
        (Reactions.D_D, EnergySpectrumMethod.BALLABIO_M_GAUSSIAN),
        (Reactions.D_T, EnergySpectrumMethod.BALLABIO_GAUSSIAN),
        (Reactions.D_T, EnergySpectrumMethod.BALLABIO_M_GAUSSIAN),
        (Reactions.T_T, EnergySpectrumMethod.DATA),
    ],
)
def run_sim_and_track_particles(request, omc_path):
    """Run a simulation and get all of the particle tracks out of it.

    Returns
    -------
    source:
        TokamakNeutronSource
    locations:
        The location of every particle, stored as an array of shape (N, 3), where
        the final axis stores the location in cylindrical coordinates. [cm]
    directions:
        The direction of every particle, stored as an array of shape (N, 2), where
        the final axis stores the location in spherical coordinates. [cm]
    energies:
        The energy of every particle, stored as an array of shape (N,), where the
        final axis stores the energy in [eV].
    """
    source_type, method = request.param
    temperature_profile = ParabolicPedestalProfile(
        25.0, 5.0, 0.1, 1.45, 2.0, 0.95
    )  # [keV]
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
        source_type=source_type,
        cell_side_length=CELL_SIDE_LENGTH,
        total_fusion_power=2.2e9,
    )
    tracks = run_openmc_sim(source, omc_path, method)

    # Should take about <1 minutes per 5000 particles.
    # Expected leakage fraction = 1.0 since all neutrons should leave the source
    # cell (made of vacuum) without interacting with anything.
    locations, directions, energies = [], [], []
    for ptrac in tracks:
        # particle_tracks should have len==1 since there shouldn't be any splitting
        # (lacking any obstacles in the simulation)
        start_state = OpenMCTrack(*ptrac.particle_tracks[0].states[0])
        locations.append(start_state.position_cylindrical)
        directions.append(start_state.direction_spherical)
        energies.append(start_state.energy)
    return OpenMCSimulatedSourceParticles(
        source, np.array(locations), np.array(directions), np.array(energies)
    )


@pytest.mark.integration
class TestOpenMCSimulation:
    """Testing the particle tracks data."""

    @staticmethod
    def assert_is_uniform(
        array: npt.NDArray, known_range: tuple[float, float] | None = None
    ):
        if known_range:
            assert known_range[0] <= array.min()
            assert array.max() <= known_range[1]
        counts, _ = np.histogram(array, range=known_range)
        avg = counts.mean()
        # Close enough to a Poisson distribution, so sigma = sqrt(count)
        assert np.isclose(counts, avg, rtol=0, atol=3.5 * np.sqrt(avg)).all(), (
            "This test (3.5 sigmas) has a false negative/failure rate "
            "of 0.046% per comparison."
        )
        # 3.5 sigma should be enough.

    @staticmethod
    def assert_is_cosine(array: npt.NDArray):
        """
        Confirm the theta part of the spherical coordinate of an isotropic direction
        distribution follows a cosine curve.
        """
        counts, bins = np.histogram(array, bins=50, range=(-np.pi / 2, np.pi / 2))
        lower_bound, upper_bound = bins[:-1], bins[1:]
        expected_pdf = np.sin(upper_bound) - np.sin(lower_bound)  # CDF of cos is sin.
        scale_factor = (
            counts.sum() / expected_pdf.sum()
        )  # expected_pdf should sum to 2.0
        assert np.isclose(
            counts,
            expected_pdf * scale_factor,
            rtol=0,
            atol=3.5 * np.sqrt(np.clip(counts, 1, np.inf)),
        ).all(), (
            "This test (4 sigma) has a false negative/failure rate "
            "of 0.0063% per comparison."
        )

    def test_location(self, run_sim_and_track_particles):
        """Confirm the sources are distributed uniformly in phi and according to the
        required distribution poloidally.
        """
        sim = run_sim_and_track_particles
        r, phi, z = sim.locations.T
        self.assert_is_uniform(phi, (-np.pi, np.pi))
        _f, (ax1, ax2) = plt.subplots(2)
        ax1.scatter(r / 100, z / 100, alpha=0.1, marker="o", s=0.5)
        ax1.set_xlabel("r (m)")
        ax1.set_ylabel("z (m)")
        ax1.set_title(
            "Neutron generation positions\n(poloidal view)"
            "\nEach dot is a neutron emitted"
        )
        o_point, lcfs = sim.source.flux_map.o_point, sim.source.flux_map.lcfs
        ax2.scatter(
            o_point.x, o_point.z, label="o-point", facecolors="none", edgecolor="C1"
        )
        ax2.plot(lcfs.x, lcfs.z, label="LCFS")
        ax2.legend()
        ax2.set_aspect("equal")

    def test_isotropic(self, run_sim_and_track_particles):
        """Confirm the sources are emitting neutrons isotropically."""
        sim = run_sim_and_track_particles
        dir_theta, dir_phi = sim.directions.T
        self.assert_is_cosine(dir_theta)
        self.assert_is_uniform(dir_phi, (-np.pi, np.pi))

    def test_energy(self, run_sim_and_track_particles):
        sim = run_sim_and_track_particles
        reaction_neutron_counter = sum(
            reaction.num_neutrons for reaction in sim.source.source_type
        )
        # Plot the neutron spectrum for when there are multiple types of reactions.
        _f, ax = plt.subplots()
        ax.hist(sim.energies, bins=500)
        ax.set_title("Neutron spectrum across the entire tokamak")
        if reaction_neutron_counter > 1:
            return
        reaction = sim.source.source_type[0]

        # calculate obtained value.
        avg_neutron_energy = raw_uc(sim.energies.mean(), "eV", "J")
        assert np.isclose(
            avg_neutron_energy,
            _APPROX_NEUTRON_ENERGY[reaction],
            rtol=raw_uc(0.1, "MeV", "J"),
        )

        desired_neutron_power = sum(
            _APPROX_NEUTRON_ENERGY[rx] * sim.source.num_reactions_per_second.get(rx, 0.0)
            for rx in sim.source.source_type
        )
        n_per_second = sum(sim.source.num_neutrons_per_second.values())
        neutron_power = avg_neutron_energy * n_per_second
        assert np.isclose(neutron_power, desired_neutron_power, atol=0, rtol=0.03)
