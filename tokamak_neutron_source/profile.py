# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Simple 1-D profile structures and parameterisations"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


class PlasmaProfile(ABC):
    """Abstract base class for plasma profiles."""

    def __init__(self, scale: float = 1.0):
        self._scale = scale

    @abstractmethod
    def value(self, rho: float | Iterable) -> float | np.ndarray:
        """Calculate the value of the profile at given value(s) of rho."""

    def set_scale(self, scale: float = 1.0):
        """Set the scale of the rho value. Multiplies existing scale."""
        self._scale *= scale


class DataProfile(PlasmaProfile):
    """
    Plasma profile from data.

    Notes
    -----
    Normalised radius conventions are not enforced here.
    Linear interpolation is used.
    """

    def __init__(self, values: npt.NDArray, rho: npt.NDArray, *, scale: float = 1.0):
        super().__init__(scale)
        self._values = values
        self._rho = rho
        self._interpolator = interp1d(self._rho, self._values, kind="linear")

    def value(self, rho: float | Iterable) -> float | np.ndarray:
        """Calculate the value of the profile at given value(s) of rho."""  # noqa: DOC201
        rho = np.clip(rho, 0.0, 1.0)
        return self._scale * self._interpolator(rho)


class ParabolicProfile(PlasmaProfile):
    """
    Parabolic plasma profile parameterisation defined e.g. in
    :doi:`Lao 1985 <10.1088/0029-5515/25/11/007>`.

    Parameters
    ----------
    core_value:
        Value of the profile at rho = 0
    alpha:
        Alpha exponent
    beta:
        Beta exponent
    """

    def __init__(self, core_value: float, alpha: float, beta: float):
        super().__init__()
        self.core_value = core_value
        self.alpha = alpha
        self.beta = beta

    def value(self, rho):
        """
        Parabolic with pedestal profile function

        Parameters
        ----------
        rho:
            Values of the normalised radius at which to calculate the profile
            value

        Returns
        -------
        values:
            The value(s) of the profile at rho.

        Notes
        -----
        \t:math:`g(\rho)=(1-\rho^{\beta})^{\alpha}`
        """
        rho = np.clip(rho, 0, 1)
        # sign tweak needed to avoid runtime warnings in np
        f = 1 - np.sign(rho) * np.abs(rho) ** self.beta
        values = self.core_value * np.sign(f) * (np.abs(f)) ** self.alpha
        return self._scale * values


class ParabolicPedestalProfile(PlasmaProfile):
    """
    Parabolic pedestal plasma profile parameterisation.

    Parameters
    ----------
    rho_ped:
        Pedestal normalised radius
    core_value:
        Value of the profile at rho = 0
    ped_value:
        Value of the profile at rho = rho_ped
    sep_value:
        Value of the profile at rho = 1
    alpha:
        Alpha exponent
    beta:
        Beta exponent
    rho_ped:
        Normalised radius of the pedestal
    """

    def __init__(
        self,
        core_value: float,
        ped_value: float,
        sep_value: float,
        alpha: float,
        beta: float,
        rho_ped: float,
    ):
        super().__init__()
        self.core_value = core_value
        self.ped_value = ped_value
        self.sep_value = sep_value
        self.alpha = alpha
        self.beta = beta
        self.rho_ped = np.clip(rho_ped, 0, 1)

    def value(self, rho: float | Iterable) -> float | Iterable:
        """
        Parabolic with pedestal profile function

        Parameters
        ----------
        rho:
            Values of the normalised radius at which to calculate the profile
            value

        Returns
        -------
        values:
            The value(s) of the profile at rho.
        """
        rho = np.asarray(rho, dtype=float)
        rho = np.clip(rho, 0.0, 1.0)

        rho_rho_ped_beta = (rho / self.rho_ped) ** self.beta
        # Clip to avoid small zeros (-EPS)
        term = np.clip(1.0 - rho_rho_ped_beta, 0.0, 1.0)

        # Core region: rho < rho_ped
        core_part = (
            self.ped_value + (self.core_value - self.ped_value) * term**self.alpha
        )

        # Pedestal region: rho >= rho_ped
        if self.rho_ped == 1.0:
            ped_part = np.full_like(rho, self.sep_value)
        else:
            ped_part = self.sep_value + (self.ped_value - self.sep_value) * (1 - rho) / (
                1 - self.rho_ped
            )

        # Combine regions
        values = np.where(rho < self.rho_ped, core_part, ped_part)
        values *= self._scale

        # Return scalar if input was scalar
        return values.item() if np.isscalar(rho) or values.shape == () else values
