# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_neutron_source.error import ReactivityError
from tokamak_neutron_source.reactions import AneutronicReactions, Reactions
from tokamak_neutron_source.reactivity import (
    ReactivityMethod,
    reactivity,
)
from tokamak_neutron_source.reactivity_data import ReactionCrossSection
from tokamak_neutron_source.tools import get_tns_path


@pytest.fixture
def _xfail_DD_He3n_erratum_erratum(request):
    """
    As far as I can tell, there is either something wrong with the parameterisation,
    or more likely with the data presented in:

    H.-S. Bosch and G.M. Hale 1993 Nucl. Fusion 33 1919
    """
    t = request.getfixturevalue("temp_kev")
    if t == pytest.approx(1.3, rel=0, abs=np.finfo(float).eps):
        request.node.add_marker(pytest.mark.xfail(reason="Error in erratum data?"))


class TestReactivity:
    """
    H.-S. Bosch and G.M. Hale 1993 Nucl. Fusion 33 1919
    """

    path = "tests/test_data"
    filename = "reactivity_Bosch_Hale_1993.json"
    with open(Path(path, filename)) as file:
        data = json.load(file)

    temp = np.array(data["temperature_kev"])
    sv_DT = np.array(data["sv_DT_m3s"])
    sv_DD_He3p = np.array(data["sv_DD_He3p_m3s"])
    sv_DD_Tp = np.array(data["sv_DD_Tp_m3s"])

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DT])
    def test_Bosch_Hale_DT(self, temp_kev, sigmav):
        result = reactivity(temp_kev, Reactions.D_T)
        np.testing.assert_allclose(result, sigmav, rtol=0.0025, atol=0)

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DD_He3p])
    @pytest.mark.usefixtures("_xfail_DD_He3n_erratum_erratum")
    def test_Bosch_Hale_DD_He3n(self, temp_kev, sigmav):
        result = reactivity(temp_kev, Reactions.D_D)
        np.testing.assert_allclose(result, sigmav, rtol=0.003, atol=0)

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DD_Tp])
    def test_Bosch_Hale_DD_Tp(self, temp_kev, sigmav):
        result = reactivity(temp_kev, AneutronicReactions.D_D)
        np.testing.assert_allclose(result, sigmav, rtol=0.0035, atol=0)


class TestCrossSections:
    data_dir = get_tns_path("data/cross_sections")

    @pytest.mark.parametrize(
        "file_name", [p.name for p in data_dir.rglob("*.txt") if p.is_file()]
    )
    def test_files(self, file_name):
        ReactionCrossSection(file_name)

    def test_file_error(self):
        with pytest.raises(ReactivityError):
            ReactionCrossSection("dud")

    def test_sigmav_dt_comparison(self):
        t = np.linspace(0.2, 100, 100)
        sv_bh = reactivity(t, Reactions.D_T)
        sv_xs = reactivity(t, Reactions.D_T, method=ReactivityMethod.XS)
        assert max((sv_bh - sv_xs) / sv_bh) < 0.0068

    def test_sigmav_dd_comparison(self):
        t = np.linspace(0.2, 100, 100)
        sv_bh = reactivity(t, Reactions.D_D)
        sv_xs = reactivity(t, Reactions.D_D, method=ReactivityMethod.XS)
        assert max((sv_bh - sv_xs) / sv_bh) < 0.0246

    def test_sigmav_ddip_comparison(self):
        t = np.linspace(0.2, 100, 100)
        sv_bh = reactivity(t, AneutronicReactions.D_D)
        sv_xs = reactivity(t, AneutronicReactions.D_D, method=ReactivityMethod.XS)
        assert max((sv_bh - sv_xs) / sv_bh) < 0.0187
