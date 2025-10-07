from pathlib import Path

import pytest
import numpy as np
from numpy import typing as npt
import openmc

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

@pytest.mark.integration
class OpenMCSimulation:
    """A simple openmc simulation to create the particle track data. """
    temperature_profile = ParabolicPedestalProfile(25.0, 5.0, 0.1, 1.45, 2.0, 0.95)  # [keV]
    density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
    rho_profile = np.linspace(0, 1, 30)

    source = TokamakNeutronSource(
        transport=TransportInformation.from_parameterisations(
            ion_temperature_profile=temperature_profile,
            fuel_density_profile=density_profile,
            rho_profile=rho_profile,
            fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
        ),
        flux_map=FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json"),
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
    settings.source = source.to_openmc_source()
    settings.particles = settings.max_tracks = len(settings.source)*10
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

    def test_power_equal(self):
        self.tracks
        power = sum([])
        assert np.isclose(power, 2.2e9, rtol=0.01, atol=0.0)
        

    def test_num_neutrons_equal(self):
        expected_number_of_neutrons
        assert np.isclose(
            len(self.tracks),
            expected_number_of_neutrons,
            rtol=0,
            atol=len(settings.source)/2,
        )

    def test_spectrum_at_known_temp(self):
        assert False