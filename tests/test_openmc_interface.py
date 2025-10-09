# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest
import numpy as np
from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.profile import ParabolicPedestalProfile
from tokamak_neutron_source import Reactions
from tokamak_neutron_source.energy import EnergySpectrumMethod

ion_temperature_profile = ParabolicPedestalProfile(25.0, 5.0, 0.1, 1.45, 2.0, 0.95)  # [keV]
fuel_density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
rho_profile = np.linspace(0, 1, 30)
flux_map = FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json")
dd_source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=ion_temperature_profile,
        fuel_density_profile=fuel_density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=1.0, T=0.0),
    ),
    flux_map=flux_map,
    cell_side_length=0.2,
)
dt_source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=ion_temperature_profile,
        fuel_density_profile=fuel_density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    flux_map=flux_map,
    cell_side_length=0.2,
)
tt_source = TokamakNeutronSource(
    transport=TransportInformation.from_parameterisations(
        ion_temperature_profile=ion_temperature_profile,
        fuel_density_profile=fuel_density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.0, T=1.0),
    ),
    flux_map=flux_map,
    cell_side_length=0.2,
)

@pytest.mark.parametrize("source", [dt_source])
def test_openmc_source_conversion(source: TokamakNeutronSource):
    """
    Check that neutrons are produced in the correct locations.
    """
    openmc_source = source.to_openmc_source()
    lcfs = source.flux_map.lcfs
    lower_lim_r, upper_lim_r = min(lcfs.x), max(lcfs.x)
    lower_lim_z, upper_lim_z = min(lcfs.z), max(lcfs.z)
    min_r, max_r, min_z, max_z = [], [], [], []
    dx, dz = source.dxdz
    for src in openmc_source:
        min_r.append(src.space.r.x.min()/100), max_r.append(src.space.r.x.max()/100)
        min_z.append(src.space.z.a/100), max_z.append(src.space.z.b/100)
    assert lower_lim_r-dx<=min(min_r) and max(max_r)<=upper_lim_r+dx, "Radii must lie in range."
    assert lower_lim_z-dz<=min(min_z) and max(max_z)<=upper_lim_z+dz, "Height must lie in range."

@pytest.mark.parametrize(
    ("source", "method"),
    [
    [dd_source,EnergySpectrumMethod.BALLABIO_GAUSSIAN],
    [dd_source,EnergySpectrumMethod.BALLABIO_M_GAUSSIAN],
    [dt_source,EnergySpectrumMethod.BALLABIO_GAUSSIAN],
    [dt_source,EnergySpectrumMethod.BALLABIO_M_GAUSSIAN],
    [tt_source,EnergySpectrumMethod.DATA],
    ]
)
def test_source_defined_energies(source: TokamakNeutronSource, method: EnergySpectrumMethod):
    """
    Check, for each of the energy spectrum methods, that the energy of the produced
    neutrons are approximately sensible.
    """
    openmc_source = source.to_openmc_source(method)
    min_E, max_E = [], []
    for src in openmc_source:
        for dist in src.energy.distribution:
            min_E.append(dist.x.min())
            max_E.append(dist.x.max())
        # assert # test should only be implemented for known sources, hence it's currently commented out.
    assert 0.0<=min(min_E) and max(max_E)<=17E6


@pytest.mark.parametrize("source", [dt_source])
def test_source_defined_intensities(source: TokamakNeutronSource):
    """
    Check that the intensities are defined correctly for the openmc sources.
    """
    source.normalise_fusion_power(2.2e9)
    openmc_source = source.to_openmc_source()
    desired_intensities = np.sum([source.strength[rx] for rx in Reactions], axis=0)
    openmc_source_intensities = np.array([src.strength for src in openmc_source])
    scale_factor = desired_intensities.sum()/openmc_source_intensities.sum()
    assert np.isclose(openmc_source_intensities*scale_factor, desired_intensities, atol=0.0, rtol=1E-12).all()
    