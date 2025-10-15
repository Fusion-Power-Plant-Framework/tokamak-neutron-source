# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Flux information."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from contourpy import LineType, contour_generator
from eqdsk import EQDSKInterface
from scipy.interpolate import CloughTocher2DInterpolator, RectBivariateSpline, interp1d

from tokamak_neutron_source.error import FluxSurfaceError
from tokamak_neutron_source.tools import get_area_2d, get_centroid_2d, load_eqdsk

__all__ = [
    "ClosedFluxSurface",
    "FausserFluxSurface",
    "FluxConvention",
    "FluxMap",
    "FluxPoint",
    "FluxSurfaceParameterisation",
    "SauterFluxSurface",
]


@dataclass
class FluxPoint:
    """Single poloidal magnetic flux point."""

    x: float
    z: float
    psi: float


def is_closed(x: np.ndarray, z: np.ndarray, abs_tol: float = 1e-12) -> bool:
    return np.isclose(x[0], x[-1], rtol=0.0, atol=abs_tol) and np.isclose(
        z[0],
        z[-1],
        rtol=0.0,
        atol=abs_tol,
    )


class ClosedFluxSurface:
    """Closed poloidal magnetic flux surface."""

    def __init__(self, x: np.ndarray, z: np.ndarray):
        if not is_closed(x, z):
            raise FluxSurfaceError("This is not a closed flux surface.")
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)

    @cached_property
    def center_of_mass(self) -> tuple[float, float]:
        """
        Centre of mass of the ClosedFluxSurface.

        Returns
        -------
        com:
            ClosedFluxSurface center of mass
        """
        return get_centroid_2d(self.x, self.z)

    @cached_property
    def area(self) -> float:
        """
        Enclosed area of the ClosedFluxSurface.

        Returns
        -------
        area:
            ClosedFluxSurface enclosed poloidal area
        """
        return get_area_2d(self.x, self.z)

    @cached_property
    def volume(self) -> float:
        """
        Volume of the ClosedFluxSurface.

        Returns
        -------
        volume:
            ClosedFluxSurface enclosed volume.
        """
        return 2 * np.pi * self.area * self.center_of_mass[0]


class FluxRing(ClosedFluxSurface):
    """Flux ring."""

    def __init__(self, x: float, z: float):
        self.x = np.array([x])
        self.z = np.array([z])

    @property
    def center_of_mass(self) -> tuple[float, float]:
        """
        Centre of mass of the FluxRing.

        Returns
        -------
        com:
            FluxRing center of mass
        """
        return self.x, self.z

    @property
    def area(self) -> float:
        """
        Enclosed area of the FluxRing.

        Returns
        -------
        area:
            FluxRing enclosed poloidal area
        """
        return 0.0

    @property
    def volume(self) -> float:
        """
        Volume of the FluxRing.

        Returns
        -------
        volume:
            FluxRing enclosed volume.
        """
        return 0.0


@dataclass
class LCFSInformation:
    """Last closed flux surface parameterisation information."""

    """Plasma geometric major radius [m]"""
    r_0: float

    """Plasma geometric vertical height [m]"""
    z_0: float

    """Plasma geometric aspect ratio [m]"""
    aspect_ratio: float

    """Plasma elongation"""
    kappa: float

    """Plasma triangularity"""
    delta: float

    """Plasma radial Shafranov shift (maximum) [m]"""
    shafranov_shift: float

    """Plasma squareness"""
    squareness: float = 0.0

    # Derived attribute, set in __post_init__
    minor_radius: float = field(init=False)

    def __post_init__(self):
        # --- checks ---
        if self.r_0 <= 0:
            raise FluxSurfaceError(f"Major radius r_0 must be > 0, got {self.r_0}")
        if self.aspect_ratio <= 1:
            raise FluxSurfaceError(f"Aspect ratio must be > 1, got {self.aspect_ratio}")
        if self.kappa < 1:
            raise FluxSurfaceError(f"Elongation kappa must be >= 1, got {self.kappa}")
        if not -1 <= self.delta <= 1:
            raise FluxSurfaceError(
                f"Triangularity delta must be between -1 and 1, got {self.delta}",
            )

        self.minor_radius = self.r_0 / self.aspect_ratio
        if abs(self.shafranov_shift) >= self.minor_radius:
            raise FluxSurfaceError(
                f"Shafranov shift {self.shafranov_shift} must be smaller than minor "
                f"radius {self.minor_radius}",
            )
        self.shafranov_shift = abs(self.shafranov_shift)


def get_contours(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    array: npt.NDArray[np.float64],
    value: float,
) -> list[npt.NDArray[np.float64]]:
    """
    Get the contours of a value in continuous array.

    Parameters
    ----------
    x:
        The x value array
    z:
        The z value array
    array:
        The value array
    value: f
        The value of the desired contour in the array

    Returns
    -------
    :
        The list of arrays of value contour(s) in the array
    """
    con_gen = contour_generator(
        x,
        z,
        array,
        name="mpl2014",
        line_type=LineType.SeparateCode,
    )
    return con_gen.lines(value)[0]


def find_flux_surf(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi_norm: npt.NDArray[np.float64],
    psi_norm_value: float,
    o_point: FluxPoint,
) -> npt.NDArray[np.float64]:
    r"""
    Picks a flux surface with a normalised psi norm relative to the separatrix.
    Uses least squares to retain only the most appropriate flux surface. This
    is taken to be the surface whose geometric centre is closest to the O-point.

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi_norm:
        The normalised poloidal magnetic flux map [-]
    psi_norm_value:
        The normalised psi value of the desired flux surface [-]
    o_point:
        O-points to use to calculate psinorm

    Returns
    -------
    :
        The flux surface coordinate array


    Raises
    ------
    FluxSurfaceError
        No flux surface found at psi_norm

    Notes
    -----
    \t:math:`{\\Psi}_{N} = {\\psi}_{O}-N({\\psi}_{O}-{\\psi}_{X})`

    Uses matplotlib hacks to pick contour surfaces on psi(X, Z).
    """

    def _f_min(x_opt: npt.NDArray, z_opt: npt.NDArray) -> float:
        """
        Error function for point clusters relative to base O-point.

        Returns
        -------
        error:
            Error for the point cluster
        """
        return np.sum(
            (np.mean(x_opt) - o_point.x) ** 2 + (np.mean(z_opt) - o_point.z) ** 2,
        )

    psi_surfs = get_contours(x, z, psi_norm, psi_norm_value)

    if not psi_surfs:
        raise FluxSurfaceError(
            f"No flux surface found for psi_norm = {psi_norm_value:.4f}",
        )

    # Choose the most logical flux surface
    err = [_f_min(*group.T) for group in psi_surfs]
    return psi_surfs[np.argmin(err)].T


def interpolate_flux_surface(
    x: npt.NDArray,
    z: npt.NDArray,
    n_points: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    seg_lengths = np.hypot(np.diff(x), np.diff(z))
    s = np.concatenate(([0], np.cumsum(seg_lengths)))

    fx = interp1d(s, x, kind="linear")
    fz = interp1d(s, z, kind="linear")

    s_uniform = np.linspace(0, s[-1], n_points, endpoint=False)  # n unique points
    x_new = fx(s_uniform)
    z_new = fz(s_uniform)
    x_new = np.append(x_new, x_new[0])
    z_new = np.append(z_new, z_new[0])
    return x_new, z_new


class FluxSurfaceParameterisation(ABC):
    """Abstract base class for flux surface parameterisation models."""

    def __init__(self, lcfs_info: LCFSInformation):
        self.lcfs_info = lcfs_info

    def flux_surface(
        self,
        psi_norm: float | Iterable,
        n_points: int = 100,
    ) -> ClosedFluxSurface | list[ClosedFluxSurface]:
        """
        Return a ClosedFluxSurface for given psi_norm.

        Parameters
        ----------
        psi_norm:
            Normalised flux
        n_points:
            Number of points per flux surface

        Returns
        -------
        flux_surface:
            The flux surface at a given psi_norm
        """
        if isinstance(psi_norm, Iterable):
            return [self._single_flux_surface(p, n_points) for p in psi_norm]
        return self._single_flux_surface(psi_norm, n_points)

    @abstractmethod
    def _single_flux_surface(
        self,
        psi_norm: float | Iterable,
        n_points: int = 100,
    ) -> ClosedFluxSurface | list[ClosedFluxSurface]:
        """Return a ClosedFluxSurface for given psi_norm."""


class FausserFluxSurface(FluxSurfaceParameterisation):
    """Fausser et al. flux surface parameterisation."""

    def _single_flux_surface(
        self,
        psi_norm: float,
        n_points: int = 100,
    ) -> ClosedFluxSurface:
        a = psi_norm * self.lcfs_info.r_0 / self.lcfs_info.aspect_ratio
        alpha = np.linspace(0, 2 * np.pi, n_points)
        r = (
            self.lcfs_info.r_0
            + a * np.cos(alpha + self.lcfs_info.delta * np.sin(alpha))
            + abs(self.lcfs_info.shafranov_shift) * (1 - psi_norm**2)
        )
        z = self.lcfs_info.z_0 + a * self.lcfs_info.kappa * np.sin(alpha)
        return ClosedFluxSurface(r, z)


class SauterFluxSurface(FluxSurfaceParameterisation):
    """Sauter et al. flux surface parameterisation."""

    def _single_flux_surface(self, psi_norm, n_points=100):
        a = psi_norm * self.lcfs_info.minor_radius
        alpha = np.linspace(0, 2 * np.pi, n_points)
        r = (
            self.lcfs_info.r_0
            + a * np.cos(alpha + self.lcfs_info.delta * np.sin(alpha))
            - self.lcfs_info.squareness * np.sin(2 * alpha)
        )
        z = self.lcfs_info.z_0 + self.lcfs_info.kappa * a * np.sin(
            alpha + self.lcfs_info.squareness * np.sin(2 * alpha)
        )
        return ClosedFluxSurface(r, z)


class FluxConvention(Enum):
    """Flux normalisation convention."""

    """Linear flux normalisation"""
    LINEAR = auto()

    """Square-root flux normalisation"""
    SQRT = auto()


def normalise_psi(
    psi: np.ndarray,
    axis_psi: float,
    boundary_psi: float,
    convention: FluxConvention = FluxConvention.LINEAR,
) -> np.ndarray:
    """
    Normalise flux for a given convention.

    Parameters
    ----------
    psi:
        Flux
    axis_psi:
        Magnetic axis flux
    boundary_psi:
        LCFS flux
    convention:
        Normalised flux convention

    Returns
    -------
    psi_norm:
        Normalised flux
    """
    psi_norm = (axis_psi - psi) / (axis_psi - boundary_psi)
    psi_norm = np.clip(psi_norm, 0.0, None)
    if convention == FluxConvention.SQRT:
        psi_norm = np.sqrt(psi_norm)
    return psi_norm


class FluxInterpolator(ABC):
    """Abstract base class for FluxInterpolators."""

    def __init__(
        self,
        x: npt.NDArray,
        z: npt.NDArray,
        psi_norm: npt.NDArray,
        o_point: FluxPoint,
    ):
        self.x = x
        self.z = z
        self._psi_norm = psi_norm
        self._o_point = o_point

    @abstractmethod
    def psi_norm(self, x: float, z: float) -> float:
        """
        Get the normalised flux at a given point.

        Parameters
        ----------
        x:
            Radial coordinate
        z:
            Vertical coordinate

        Returns
        -------
        psi_norm:
            Normalised flux at the point
        """
        raise NotImplementedError

    @abstractmethod
    def get_flux_surface(
        self,
        psi_norm: float,
        n_points: int | None = None,
    ) -> ClosedFluxSurface:
        """
        Get a flux surface at a given normalised flux.

        Parameters
        ----------
        psi_norm:
            Normalised flux

        Returns
        -------
        flux_surface:
            Flux surface at the normalised flux
        """
        raise NotImplementedError

    @abstractmethod
    def plot_normalised_flux(
        self,
        ax: plt.Axes,
        levels: np.ndarray | None = None,
    ) -> plt.Axes:
        """
        Plot the normalised flux.

        Parameters
        ----------
        ax:
            matplotlib Axes object to use
        levels:
            Normalised psi contour levels to plot

        Returns
        -------
        ax:
            updated matplotlib Axes
        """
        raise NotImplementedError


class EQDSKFluxInterpolator(FluxInterpolator):
    """EQDSK FluxInterpolator."""

    def __init__(
        self,
        x: npt.NDArray,
        z: npt.NDArray,
        psi_norm: npt.NDArray,
        o_point: FluxPoint,
    ):
        super().__init__(x, z, psi_norm, o_point)
        self._psi_norm_func = RectBivariateSpline(
            self.x[:, 0],
            self.z[0, :],
            self._psi_norm,
        )

    def psi_norm(self, x: float, z: float) -> float:
        """
        Get the normalised flux at a given point.

        Parameters
        ----------
        x:
            Radial coordinate
        z:
            Vertical coordinate

        Returns
        -------
        psi_norm:
            Normalised flux at the point
        """
        return self._psi_norm_func(x, z, grid=False)

    def get_flux_surface(
        self,
        psi_norm: float,
        n_points: int | None = None,
    ) -> ClosedFluxSurface:
        """
        Get a flux surface at a given normalised flux.

        Parameters
        ----------
        psi_norm:
            Normalised flux

        Returns
        -------
        flux_surface:
            Flux surface at the normalised flux
        """
        psi_norm = np.clip(psi_norm, 0.0, 1.0)
        if psi_norm == 0.0:
            return FluxRing(self._o_point.x, self._o_point.z)

        x, z = find_flux_surf(self.x, self.z, self._psi_norm, psi_norm, self._o_point)
        if n_points is None:
            return ClosedFluxSurface(x, z)
        return ClosedFluxSurface(*interpolate_flux_surface(x, z, n_points))

    def plot_normalised_flux(
        self,
        ax: plt.Axes,
        levels: npt.NDArray | None = None,
    ) -> plt.Axes:
        """
        Plot the normalised flux.

        Parameters
        ----------
        ax:
            matplotlib Axes object to use
        levels:
            Normalised psi contour levels to plot

        Returns
        -------
        ax:
            updated matplotlib Axes
        """
        if levels is None:
            levels = np.linspace(0, 1, 10)
        ax.contour(self.x, self.z, self._psi_norm, levels=levels, cmap="viridis")
        ax.plot(self._o_point.x, self._o_point.z, marker="o", color="b")
        return ax


class ParameterisationInterpolator(FluxInterpolator):
    """FluxInterpolator from a flux surface parameterisation."""

    def __init__(
        self,
        parameterisation: FluxSurfaceParameterisation,
        o_point: FluxPoint,
        rho_profile: np.ndarray,
        n_points: int,
        flux_convention: FluxConvention = FluxConvention.LINEAR,
    ):
        self._parameterisation = parameterisation
        self._n_points = n_points

        # Treat the core differently
        rho_profile = rho_profile[rho_profile > 0.0]

        flux_surfaces = self._parameterisation.flux_surface(rho_profile, self._n_points)

        x = np.concatenate([f.x for f in flux_surfaces])
        z = np.concatenate([f.z for f in flux_surfaces])

        # NOTE: Here we need to assign values to psi, and follow the convention that
        # the maximum psi is at the magnetic axis. We arbitrarily assign psi a value
        # of 1.0 at the axis, and 0.0 at the edge.
        # This is not to be confused with the normalised psi, which follows the reverse
        # trend.
        psi = np.concatenate([[1 - rho] * n_points for rho in rho_profile])

        # Prepend the core
        x = np.concatenate([[o_point.x], x])
        z = np.concatenate([[o_point.z], z])
        psi = np.concatenate([[1.0], psi])

        psi_norm = normalise_psi(psi, 1.0, 0.0, flux_convention)

        super().__init__(x, z, psi_norm, o_point)
        self._psi_norm_func = CloughTocher2DInterpolator(
            np.column_stack((x, z)),
            psi_norm,
            fill_value=1.0,
        )

    def psi_norm(self, x: float, z: float) -> float:
        """
        Get the normalised flux at a given point.

        Parameters
        ----------
        x:
            Radial coordinate
        z:
            Vertical coordinate

        Returns
        -------
        psi_norm:
            Normalised flux at the point
        """
        return self._psi_norm_func(x, z)

    def get_flux_surface(
        self,
        psi_norm: float,
        n_points: int | None = None,
    ) -> ClosedFluxSurface:
        """
        Get a flux surface at a given normalised flux.

        Parameters
        ----------
        psi_norm:
            Normalised flux

        Returns
        -------
        flux_surface:
            Flux surface at the normalised flux
        """
        if n_points is None:
            n_points = self._n_points
        return self._parameterisation.flux_surface(psi_norm, n_points)

    def plot_normalised_flux(
        self,
        ax: plt.Axes,
        levels: npt.NDArray | None = None,
    ) -> plt.Axes:
        ax.tricontour(self.x, self.z, self._psi_norm, levels=levels)
        return ax


@dataclass
class FluxMap:
    """Magneto-hydrodynamic equilibrium poloidal magnetic flux map."""

    """Last closed flux surface"""
    lcfs: ClosedFluxSurface

    """Magnetic axis point"""
    o_point: FluxPoint

    """Flux interpolator object"""
    interpolator: FluxInterpolator

    @classmethod
    def from_eqdsk(
        cls,
        file_name: str | EQDSKInterface,
        flux_convention: FluxConvention = FluxConvention.LINEAR,
    ):
        """
        Initialise a FluxMap from an EQDSK.

        Parameters
        ----------
        file_name:
            EQDSK file name (or the EQDSKInterface)
        flux_convention:
            Flux normalisation convention

        Returns
        -------
        flux_map:
            FluxMap from the EQDSK
        """
        eq = load_eqdsk(file_name)
        x, z = np.meshgrid(eq.x, eq.z, indexing="ij")

        lcfs = ClosedFluxSurface(eq.xbdry, eq.zbdry)
        o_point = FluxPoint(eq.xmag, eq.zmag, eq.psimag)
        # This is for the case where an Equilibrium is provided via EQDSK,
        # and we assume that the provided LCFS is "true".
        psi_func = RectBivariateSpline(
            x[:, 0],
            z[0, :],
            eq.psi,
        )
        # We renormalise our interpolated psi to match the LCFS in the
        # EQDSK. This will ensure that the reconstructed LCFS is
        # almost perfectly indentical (and still closed).
        boundary_psi = np.min(psi_func(lcfs.x, lcfs.z, grid=False))
        psi_norm = normalise_psi(eq.psi, eq.psimag, boundary_psi, flux_convention)
        interpolator = EQDSKFluxInterpolator(x, z, psi_norm, o_point)

        return cls(lcfs, o_point, interpolator)

    @classmethod
    def from_parameterisation(
        cls,
        parameterisation: FluxSurfaceParameterisation,
        rho_profile: np.ndarray,
        n_points: int = 100,
        flux_convention: FluxConvention = FluxConvention.LINEAR,
    ):
        """
        Initialise a FluxMap from a FluxSurfaceParameterisation.

        Parameters
        ----------
        parameterisation:
            FluxSurfaceParameteristion to use
        rho_profile:
            Normalised radius array to use
        n_points:
            Number of points per flux surface
        flux_convention:
            Flux normalisation convention

        Returns
        -------
        flux_map:
            FluxMap from the parameterisation
        """
        lcfs = parameterisation.flux_surface(1.0, n_points)
        o_point = FluxPoint(
            parameterisation.lcfs_info.r_0 + parameterisation.lcfs_info.shafranov_shift,
            0.0,
            1.0,
        )
        interpolator = ParameterisationInterpolator(
            parameterisation,
            o_point,
            rho_profile,
            n_points,
            flux_convention,
        )
        return cls(lcfs, o_point, interpolator)

    def psi_norm(self, x: float, z: float) -> float:
        """
        Get the normalised flux at a given point.

        Parameters
        ----------
        x:
            Radial coordinate
        z:
            Vertical coordinate

        Returns
        -------
        psi_norm:
            Normalised flux at the point
        """
        return self.interpolator.psi_norm(x, z)

    def get_flux_surface(
        self,
        psi_norm: float,
        n_points: int | None = None,
    ) -> ClosedFluxSurface | list[ClosedFluxSurface]:
        """
        Get a flux surface at a given normalised flux.

        Parameters
        ----------
        psi_norm:
            Normalised flux

        Returns
        -------
        flux_surface:
            Flux surface at the normalised flux
        """
        if isinstance(psi_norm, Iterable):
            return [self.interpolator.get_flux_surface(p, n_points) for p in psi_norm]
        return self.interpolator.get_flux_surface(psi_norm, n_points)

    def volume(self, psi_norm: float) -> float | list[float]:
        """
        Get the volume at a given normalised flux.

        Parameters
        ----------
        psi_norm:
            Normalised flux

        Returns
        -------
        volume:
            Volume at the normalised flux
        """
        fs = self.get_flux_surface(psi_norm)
        if isinstance(fs, Iterable):
            return [s.volume for s in fs]
        return fs.volume

    def plot(
        self,
        f: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        levels: npt.NDArray | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the FluxMap.

        Parameters
        ----------
        f:
            Matplotlib figure
        ax:
            Matplotlib axes
        levels:
            Normalised psi contour levels to plot

        Returns
        -------
        f:
            Matplotlib figure
        ax:
            Matplotlib axes
        """
        if ax is None:
            f, ax = plt.subplots()
        if f is None:
            f = ax.get_figure()
        self.interpolator.plot_normalised_flux(ax, levels=levels)
        ax.plot(self.lcfs.x, self.lcfs.z, color="r", lw=2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("equal")
        return f, ax
