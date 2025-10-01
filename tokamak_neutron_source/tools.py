# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools."""

import os
from pathlib import Path

import numba as nb
import numpy as np
from eqdsk import EQDSKInterface


def _get_relpath(folder: str | Path, subfolder: str) -> Path:
    path = Path(folder, subfolder)
    if path.is_dir():
        return path
    raise ValueError(f"{path} Not a valid folder.")


def get_tns_root() -> str:
    """
    Get the tokamak_neutron_source root install folder.

    Returns
    -------
    :
        The full path to the tokamak_neutron_source root folder, e.g.:
            '/home/user/code/tokamak_neutron_source'
    """
    import tokamak_neutron_source  # noqa: PLC0415

    path = next(iter(tokamak_neutron_source.__path__))
    return os.path.split(path)[0]


def get_tns_path(path: str = "", subfolder: str = "tokamak_neutron_source") -> Path:
    """
    Get a tns path of a module subfolder. Defaults to root folder.

    Parameters
    ----------
    path:
        The desired path from which to create a full path
    subfolder:
        The subfolder (from the tokamak_neutron_source root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'

    Returns
    -------
    :
        The full path to the desired `path` in the subfolder specified
    """
    root = get_tns_root()
    if "egg" in root:
        return Path(f"/{subfolder}")

    path = path.replace("/", os.sep)
    main_path = _get_relpath(root, subfolder)
    return Path(_get_relpath(main_path, path))


def load_eqdsk(file: str | EQDSKInterface) -> EQDSKInterface:
    """
    Load an EQDSK file.

    Parameters
    ----------
    file_name:
        The path to the EQDSK file.

    Returns
    -------
    :
        The EQDSKInterface object.

    Notes
    -----
    Enforces the local convention that psi on axis is higher than
    psi on the boundary. This way, we do not need to ask the user
    what COCOS convention they are using.

    The actual values of psi are irrelevant here, and may be changed
    to enforce this convention.
    """
    eq = EQDSKInterface.from_file(file, no_cocos=True) if isinstance(file, str) else file

    if eq.psimag < eq.psibdry:
        offset = eq.psimag
        eq.psi = offset - eq.psi
        eq.psibdry = offset - eq.psibdry
        eq.psimag = 0.0
    return eq


@nb.jit(cache=True, nopython=True)
def check_ccw(x: np.ndarray, z: np.ndarray) -> bool:
    """
    Check that a set of x, z coordinates are counter-clockwise.

    Parameters
    ----------
    x:
        The x coordinates of the polygon
    z:
        The z coordinates of the polygon

    Returns
    -------
    :
        True if polygon counterclockwise
    """
    a = 0
    for n in range(len(x) - 1):
        a += (x[n + 1] - x[n]) * (z[n + 1] + z[n])
    return a < 0


@nb.jit(cache=True, nopython=True)
def get_area_2d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x:
        The first set of coordinates [m]
    y:
        The second set of coordinates [m]

    Returns
    -------
    :
        The area of the polygon [m^2]
    """
    # No np.roll in numba
    x = np.ascontiguousarray(x.astype(np.float64))
    y = np.ascontiguousarray(y.astype(np.float64))
    x1 = np.append(x[-1], x[:-1])
    y1 = np.append(y[-1], y[:-1])
    return 0.5 * np.abs(np.dot(x, y1) - np.dot(y, x1))


@nb.jit(cache=True, nopython=True)
def get_centroid_2d(x: np.ndarray, z: np.ndarray) -> list[float]:
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x:
        x coordinates of the coordinates to calculate on
    z:
        z coordinates of the coordinates to calculate on

    Returns
    -------
    :
        The x, z coordinates of the centroid [m]
    """
    if not check_ccw(x, z):
        x = np.ascontiguousarray(x[::-1])
        z = np.ascontiguousarray(z[::-1])
    area = get_area_2d(x, z)

    cx, cz = 0, 0
    for i in range(len(x) - 1):
        a = x[i] * z[i + 1] - x[i + 1] * z[i]
        cx += (x[i] + x[i + 1]) * a
        cz += (z[i] + z[i + 1]) * a

    if area != 0:
        # Zero division protection
        cx /= 6 * area
        cz /= 6 * area

    return [cx, cz]
