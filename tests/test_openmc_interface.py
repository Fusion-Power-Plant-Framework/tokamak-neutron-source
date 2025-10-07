# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest
from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.energy import EnergySpectrumMethod

temperature_profile = ParabolicPedestalProfile(25.0, 5.0, 0.1, 1.45, 2.0, 0.95)  # [keV]
density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
rho_profile = 

flux_map = FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json")
source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=temperature_profile,
        fuel_density_profile=density_profile,
        rho_profile=np.linspace(0, 1, 30),
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=flux_map,
    cell_side_length=0.05,
)

@pytest.mark.parametrize("source":[source])
def test_openmc_source_conversion(source: TokamakNeutronSource):
    """
    Check that neutrons are produced in the correct locations.
    """
    openmc_source = source.to_openmc_source()
    lcfs = source.flux_map.lcfs
    lower_lim_r, upper_lim_r = min(lcfs.x), max(lcfs.x)
    lower_lim_z, upper_lim_z = min(lcfs.z), max(lcfs.z)
    min_r, max_r, min_z, max_z = [], [], [], []
    for src in openmc_source:
        min_r.append(src.space.r.x.min()/100), max_r.append(src.space.r.x.max()/100)
        min_z.append(src.space.z.x.min()/100), max_z.append(src.space.z.x.max()/100)
    assert lower_lim_r<=min(min_r) and max(max_r)<=upper_lim_r, "Radii must lie in range."
    assert lower_lim_z<=min(min_z) and max(max_z)<=upper_lim_z, "Height must lie in range."

@pytest.mark.parametrize("source":[source])
def test_source_defined_energies(source: TokamakNeutronSource):
    """
    Check, for each of the energy spectrum methods, that the energy of the produced
    neutrons are approximately sensible.
    """
    for method in EnergySpectrumMethod:
        openmc_source = source.to_openmc_source(method)
        min_E, max_E = [], []
        for src in openmc_source:
            for dist in src.energy.distribution:
                min_E.append(dist.x.min())
                max_E.append(dist.x.max())
            # assert # test should only be implemented for known sources, hence it's currently commented out.
        assert 0.0<=min(min_E) and max(max_E)<=15E6


@pytest.mark.parametrize("source":[source])
def test_source_defined_intensities(self):
    """
    Check that the intensities are defined correctly for the openmc sources.
    """
