# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path

import numpy as np
import pytest
from eqdsk import EQDSKInterface

from tokamak_neutron_source.error import ReactivityError
from tokamak_neutron_source.flux import (
    FausserFluxSurface,
    FluxConvention,
    FluxMap,
    LCFSInformation,
    SauterFluxSurface,
)
from tokamak_neutron_source.main import TokamakNeutronSource, _parse_source_type
from tokamak_neutron_source.profile import ParabolicPedestalProfile
from tokamak_neutron_source.reactions import AneutronicReactions, Reactions
from tokamak_neutron_source.reactivity import (
    ReactivityMethod,
)
from tokamak_neutron_source.tools import load_jsp
from tokamak_neutron_source.transport import (
    FractionalFuelComposition,
    TransportInformation,
)

temperature_profile = ParabolicPedestalProfile(30.0, 5, 0.1, 1.45, 2.0, 0.95)  # [keV]
density_profile = ParabolicPedestalProfile(1e20, 0.5e19, 0.5e17, 1.0, 2.0, 0.95)
rho_profile = np.linspace(0, 1, 30)

t_profile_data = np.linspace(30, 0.0, 30)
n_profile_data = np.linspace(1e19, 1e18, 30)

flux_maps = [
    FluxMap.from_parameterisation(
        FausserFluxSurface(
            LCFSInformation(9.0, 0.0, 3.1, 1.8, 0.4, 0.5),
        ),
        rho_profile=rho_profile,
    ),
    FluxMap.from_eqdsk("tests/test_data/eqref_OOB.json"),
    FluxMap.from_eqdsk(
        EQDSKInterface.from_file("tests/test_data/eqref_OOB.json", no_cocos=True)
    ),
]

transport_infos = [
    TransportInformation.from_parameterisations(
        ion_temperature_profile=temperature_profile,
        fuel_density_profile=density_profile,
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    ),
    TransportInformation.from_profiles(
        t_profile_data,
        n_profile_data,
        rho_profile,
        FractionalFuelComposition(D=0.5, T=0.5),
    ),
]

source_types = [
    None,
    Reactions.D_T,
    "D-T",
    ["D-D", Reactions.D_T],
    [Reactions.D_D, Reactions.D_T],
]


@pytest.mark.parametrize("transport", transport_infos)
@pytest.mark.parametrize("flux_map", flux_maps)
@pytest.mark.parametrize("source_type", source_types)
@pytest.mark.parametrize(
    "reactivity_method", [ReactivityMethod.BOSCH_HALE, ReactivityMethod.XS]
)
def test_main(transport, flux_map, source_type, reactivity_method):
    source = TokamakNeutronSource(
        transport, flux_map, source_type, reactivity_method=reactivity_method
    )
    source.normalise_fusion_power(1e9)
    np.testing.assert_allclose(source.calculate_total_fusion_power(), 1e9, rtol=3e-3)


class TestParseSourceType:
    @pytest.mark.parametrize(
        "reactions",
        [
            [Reactions.D_D, Reactions.T_T, Reactions.D_T],
            ["D-D", "T_T", "D-T"],
            ["D_D", AneutronicReactions.D_He3],
        ],
    )
    def test_dd_channels(self, reactions):
        new_reactions = _parse_source_type(reactions)
        assert len(new_reactions) == len(reactions) + 1

    def test_dd_channels_unchanged(self):
        reactions = [Reactions.D_D, AneutronicReactions.D_D]
        new_reactions = _parse_source_type(reactions)
        assert sorted(new_reactions, key=lambda x: x.label) == sorted(
            reactions, key=lambda x: x.label
        )

    def test_no_duplicates(self):
        assert len(_parse_source_type(["D-T", "D-T"])) == 1

    @pytest.mark.parametrize("reactions", ["None", "D-X", [None]])
    def test_fail(self, reactions):
        with pytest.raises(ReactivityError):
            _parse_source_type(reactions)


class TestPROCESSFusionBenchmark:
    """
    Benchmark to PROCESS version v3.2.0-52-gd3768e97f large tokamak solution values.
    """

    temperature_profile = ParabolicPedestalProfile(
        25.718, 5.5, 0.1, 1.45, 2.0, 0.94
    )  # [keV]
    temperature_profile.set_scale(1.0)
    density_profile = ParabolicPedestalProfile(
        1.042e20, 6.214e19, 3.655e19, 1.0, 2.0, 0.94
    )
    density_profile.set_scale(6.358 / 7.922)
    rho_profile = np.linspace(0, 1, 500)  # Default PROCESS discretisation

    def make_source(self, reaction):
        return TokamakNeutronSource(
            transport=TransportInformation.from_parameterisations(
                ion_temperature_profile=self.temperature_profile,
                fuel_density_profile=self.density_profile,
                rho_profile=self.rho_profile,
                fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
            ),
            flux_map=FluxMap.from_parameterisation(
                SauterFluxSurface(
                    LCFSInformation(
                        8.0, 0.0, 3.0, 1.85, 0.5, shafranov_shift=0.0, squareness=0.0
                    ),
                ),
                n_points=100,
                rho_profile=self.rho_profile,
                flux_convention=FluxConvention.LINEAR,
            ),
            source_type=reaction,
            cell_side_length=0.1,
            reactivity_method=ReactivityMethod.BOSCH_HALE,
        )

    @pytest.mark.parametrize(
        ("reaction", "expected_mw"),
        [
            (Reactions.D_T, 1637.79),
            (Reactions.D_D, 1.97),
        ],
    )
    def test_fusion_power(self, reaction, expected_mw):
        source = self.make_source(reaction)
        total_fusion_power_mw = source.calculate_total_fusion_power() / 1e6
        assert np.isclose(total_fusion_power_mw, expected_mw, rtol=1.5e-2, atol=0.0)

    def test_DD_source_T_rate(self):
        source = self.make_source(Reactions.D_D)
        assert source.source_rate > 0.0
        assert source.source_T_rate == 0.0

    def test_source_rates(self):
        source = self.make_source([Reactions.D_T, Reactions.D_D])
        dt_source_t_rate = source.source_T_rate
        assert source.source_rate > 0
        assert dt_source_t_rate < source.source_rate

        source = self.make_source([Reactions.D_T, Reactions.D_D, Reactions.T_T])
        assert source.source_rate > 0
        assert source.source_T_rate < source.source_rate
        assert dt_source_t_rate < source.source_T_rate

    def test_TT_source_rate(self):
        source = self.make_source(Reactions.T_T)
        # NOTE: T-T consumes 2 tritons but also produces 2 neutrons, so these
        # should be equal
        assert source.source_rate > 0
        assert np.isclose(source.source_T_rate, source.source_rate, rtol=0.0, atol=1e-16)


TEST_DATA = Path(__file__).parent / "test_data"


class TestJETTOFusionBenchmark:
    jsp_path = Path(TEST_DATA, "STEP_jetto.jsp").as_posix()
    eqdsk_path = path = Path(TEST_DATA, "STEP_jetto.eqdsk_out").as_posix()

    def test_source_power(self):
        pytest.importorskip("jetto_tools")
        data = load_jsp(self.jsp_path)
        source = TokamakNeutronSource(
            TransportInformation.from_jetto(self.jsp_path),
            FluxMap.from_eqdsk(self.eqdsk_path, flux_convention=FluxConvention.SQRT),
            source_type=[Reactions.D_T],
            cell_side_length=0.05,
        )

        dt_rate = sum(source.strength[Reactions.D_T])
        jetto_dt_rate = float(data.dt_neutron_rate)

        assert np.isclose(dt_rate, jetto_dt_rate, rtol=5e-3, atol=0.0)
