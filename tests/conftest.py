# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Used by pytest for configuration like adding command line options.
"""

import matplotlib as mpl


def pytest_addoption(parser):
    """
    Adds a custom command line option to pytest.
    """
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="only run integration tests",
    )
    parser.addoption(
        "--plotting-on",
        action="store_true",
        default=False,
        help="switch on interactive plotting in tests",
    )


def pytest_configure(config):
    """Configures pytest"""
    options = {"integration": config.option.integration}

    if not config.option.plotting_on:
        # We're not displaying plots so use a display-less backend
        mpl.use("Agg")
    config.option.markexpr = config.getoption(
        "markexpr",
        " and ".join([
            name if value else f"not {name}" for name, value in options.items()
        ]),
    )
    if not config.option.markexpr:
        config.option.markexpr = " and ".join([
            name if value else f"not {name}" for name, value in options.items()
        ])
