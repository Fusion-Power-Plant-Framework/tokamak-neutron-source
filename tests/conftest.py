# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Used by pytest for configuration like adding command line options.
"""

import matplotlib as mpl
import pytest


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


@pytest.fixture(autouse=True)
def _plot_show_and_close(request):
    """Fixture to show and close plots

    Notes
    -----
    Does not do anything if testclass marked with 'classplot'
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cls = request.node.getparent(pytest.Class)

    if cls and "classplot" in cls.keywords:
        yield
    else:
        yield
        clstitle = "" if cls is None else cls.name
        for fig in list(map(plt.figure, plt.get_fignums())):
            fig.suptitle(
                f"{fig.get_suptitle()} {clstitle}::"
                f"{request.node.getparent(pytest.Function).name}"
            )
        plt.show()
        plt.close()


@pytest.fixture(scope="class", autouse=True)
def _plot_show_and_close_class(request):
    """Fixture to show and close plots for marked classes

    Notes
    -----
    Only shows and closes figures on classes marked with 'classplot'
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "classplot" in request.keywords:
        yield
        clstitle = request.node.getparent(pytest.Class).name

        for fig in list(map(plt.figure, plt.get_fignums())):
            fig.suptitle(f"{fig.get_suptitle()} {clstitle}")
        plt.show()
        plt.close()
    else:
        yield
