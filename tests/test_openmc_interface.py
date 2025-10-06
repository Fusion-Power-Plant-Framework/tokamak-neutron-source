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

# %% [markdown]
# # Creation from an EQDSK file.

# %%
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
dx, dz = extract_spacing(source.x), extract_spacing(source.z)
source_cell = make_universe_box(max(source.z)-dz, min(source.z)+dz, max(source.x)+dx)
universe.add_cell(source_cell)

# run an empty simulation
settings = openmc.Settings(batches=3, particles=10000, source=source.to_openmc_source())
openmc.run()