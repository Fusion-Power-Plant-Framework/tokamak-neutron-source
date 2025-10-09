# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from tokamak_neutron_source import (
    FluxMap,
    FractionalFuelComposition,
    Reactions,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.energy import EnergySpectrumMethod
from tokamak_neutron_source.profile import ParabolicPedestalProfile

ion_temperature_profile = ParabolicPedestalProfile(
    25.0, 5.0, 0.1, 1.45, 2.0, 0.95
)  # [keV]
fuel_density_profile = ParabolicPedestalProfile(0.8e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
rho_profile = np.linspace(0, 1, 30)
flux_map = FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json")

dd_comp = {"D": 1.0, "T": 0.0}
dt_comp = {"D": 0.5, "T": 0.5}
tt_comp = {"D": 0.0, "T": 1.0}

CELL_SIDE_LENGTH = 0.2
def make_source(composition_dict):
    return TokamakNeutronSource(
        transport=TransportInformation.from_parameterisations(
            ion_temperature_profile=ion_temperature_profile,
            fuel_density_profile=fuel_density_profile,
            rho_profile=rho_profile,
            fuel_composition=FractionalFuelComposition(**composition_dict),
        ),
        flux_map=flux_map,
        cell_side_length=CELL_SIDE_LENGTH,
    )

@pytest.mark.parametrize("composition_dict", [dt_comp])
def test_openmc_source_conversion(composition_dict: dict):
    """
    Check that neutrons are produced in the correct locations.
    """
    source = make_source(composition_dict)
    openmc_source = source.to_openmc_source()
    lcfs = source.flux_map.lcfs
    lower_lim_r, upper_lim_r = min(lcfs.x), max(lcfs.x)
    lower_lim_z, upper_lim_z = min(lcfs.z), max(lcfs.z)
    min_r, max_r, min_z, max_z = [], [], [], []
    dx, dz = CELL_SIDE_LENGTH, CELL_SIDE_LENGTH
    for src in openmc_source:
        min_r.append(src.space.r.x.min() / 100), max_r.append(src.space.r.x.max() / 100)
        min_z.append(src.space.z.x.min() / 100), max_z.append(src.space.z.x.max() / 100)
    assert lower_lim_r - dx <= min(min_r), "Sensible minimum radius"
    assert max(max_r) <= upper_lim_r + dx, "Sensible maximum radius"
    assert lower_lim_z - dz <= min(min_z), "Sensible minimum height"
    assert max(max_z) <= upper_lim_z + dz, "Sensible maximum height"


@pytest.mark.parametrize(
    ("composition_dict", "method"),
    [
        (dd_comp, EnergySpectrumMethod.BALLABIO_GAUSSIAN),
        (dd_comp, EnergySpectrumMethod.BALLABIO_M_GAUSSIAN),
        (dt_comp, EnergySpectrumMethod.BALLABIO_GAUSSIAN),
        (dt_comp, EnergySpectrumMethod.BALLABIO_M_GAUSSIAN),
        (tt_comp, EnergySpectrumMethod.DATA),
    ],
)
def test_source_defined_energies(composition_dict: dict, method: EnergySpectrumMethod):
    """
    Check, for each of the energy spectrum methods, that the energy of the produced
    neutrons are approximately sensible.
    """
    source = make_source(composition_dict)
    openmc_source = source.to_openmc_source(method)
    min_E, max_E = [], []
    for src in openmc_source:
        for dist in src.energy.distribution:
            min_E.append(dist.x.min())
            max_E.append(dist.x.max())
    assert min(min_E) >= 0.0, "Sensible minimum energy."
    assert max(max_E) <= 17e6, "Sensible maximum energy."


@pytest.mark.parametrize("composition_dict", [dt_comp])
def test_source_defined_intensities(composition_dict: dict):
    """
    Check that the intensities are defined correctly for the openmc sources.
    """
    source = make_source(composition_dict)
    source.normalise_fusion_power(2.2e9)
    openmc_source = source.to_openmc_source()
    desired_intensities = np.sum(
        [source.strength[rx] * rx.num_neutrons for rx in Reactions], axis=0
    )
    openmc_source_intensities = np.array([src.strength for src in openmc_source])
    scale_factor = desired_intensities.sum() / openmc_source_intensities.sum()
    assert np.isclose(
        openmc_source_intensities * scale_factor,
        desired_intensities,
        atol=0.0,
        rtol=1e-12,
    ).all()
