# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes
"""

from textwrap import dedent, fill


class TNSError(Exception):
    """
    Base exception class. Sub-class from this for module level Errors.
    """

    def __str__(self) -> str:
        """
        Prettier handling of the Exception strings.

        Returns
        -------
        :
            The formatted exception string.
        """
        return fill(dedent(self.args[0]))


class ReactivityError(TNSError):
    """Reactivity error class"""


class EnergySpectrumError(TNSError):
    """Energy spectrum error class"""


class FluxSurfaceError(TNSError):
    """Flux surface error class"""
