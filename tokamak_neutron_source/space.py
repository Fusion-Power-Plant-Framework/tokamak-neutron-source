# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Space sampling"""

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path

from tokamak_neutron_source.flux import ClosedFluxSurface, FluxPoint


def sample_space_2d(
    lcfs: ClosedFluxSurface,
    o_point: FluxPoint,
    cell_side_length: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Sample the 2-D poloidal plane within the LCFS.

    Parameters
    ----------
    lcfs:
        Last closed flux surface
    o_point:
        O-point location
    cell_side_length:
        Side length of square cells [m]

    Returns
    -------
    x:
        Radial coordinates of sampled points [m]
    z:
        Vertical coordinates of sampled points [m]
    d_volume:
        Volumes of cells centred at points [m^3]

    Notes
    -----
    Creates points at the centres of square cells of fixed size
    (cell_side_length by cell_side_length). Only cells whose centres fall
    inside the LCFS polygon are kept.
    Cells are positioned such that the  centre of one cell lies on
    the O-point.
    """
    # Get bounding box around LCFS (+ offset)
    polygon_path = Path(np.c_[lcfs.x, lcfs.z])
    off = 2.0 * cell_side_length  # Just to be sure
    x_min, x_max = np.min(lcfs.x) - off, np.max(lcfs.x) + off
    z_min, z_max = np.min(lcfs.z) - off, np.max(lcfs.z) + off

    # Shift grid so O-point lies on a cell center
    dx = (o_point.x - x_min) % cell_side_length - 0.5 * cell_side_length
    dz = (o_point.z - z_min) % cell_side_length - 0.5 * cell_side_length

    # Construct grid
    x_centers = np.arange(x_min + dx, x_max, cell_side_length) + 0.5 * cell_side_length
    z_centers = np.arange(z_min + dz, z_max, cell_side_length) + 0.5 * cell_side_length
    x, z = np.meshgrid(x_centers, z_centers, indexing="ij")
    points = np.c_[x.ravel(), z.ravel()]

    # Mask by LCFS polygon
    inside = polygon_path.contains_points(points)
    points = points[inside]

    # Volumes: toroidal rotation of each square cell
    d_volume = 2 * np.pi * points[:, 0] * cell_side_length**2

    return points[:, 0], points[:, 1], d_volume
