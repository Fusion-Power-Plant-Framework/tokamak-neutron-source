# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest


@pytest.mark.integration
def test_dummy_integration():
    # REMOVE this test once we have a better integration test
    import openmc  # noqa: F401, PLC0415
