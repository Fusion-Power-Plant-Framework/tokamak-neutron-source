# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import numpy.typing as npt
import yaml
from eqdsk import EQDSKInterface

from tokamak_neutron_source.constants import raw_uc


def _get_relpath(folder: str | Path, subfolder: str) -> Path:
    path = Path(folder, subfolder)
    if path.is_dir():
        return path
    raise ValueError(f"{path} Not a valid folder.")


trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # noqa: NPY201


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
    file:
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


@dataclass
class SimpleJETTOOutput:
    """Dataclass for a simplified subset of JETTO output at a single timestamp."""

    """Normalised rho coordinate profile: sqrt(poloidal flux)"""
    rho: npt.NDArray

    """Ion temperature profile [keV]"""
    ion_temperature: npt.NDArray

    """Deuterium density profile [1/m^3]"""
    d_density: npt.NDArray

    """Tritium density profile [1/m^3]"""
    t_density: npt.NDArray

    """Helium-3 density profile [1/m^3]"""
    he3_density: npt.NDArray

    """D-T neutron rate [1/s]"""
    dt_neutron_rate: float


def load_jsp(file: str | Path, frame_number: int = -1) -> SimpleJETTOOutput:
    """
    Load a JETTO JSP binary file.

    Parameters
    ----------
    file:
        File to read
    frame_number:
        Frame number to read

    Returns
    -------
    :
        Simplified JETTO output

    Raises
    ------
    ValueError
        If the specified frame number is invalid.

    Notes
    -----
    For details, refer to
    https://users.euro-fusion.org/pages/data-cmg/wiki/JETTO_ppfjsp.html

    The core values (rho = 0.0) are not provided by JETTO. Here we extrapolate
    them.

    JETTO presently does not provide Helium-3 densities. These are taken to
    be 0.0.

    JETTO presently does not provide D-D fusion power or reaction rates, or
    some files may potentially do some but only for one of the channels.
    """
    from jetto_tools.binary import read_binary_file  # noqa: PLC0415

    jsp = read_binary_file(file)

    time_stamps = jsp["TIME"][:, 0, 0]  # times when the snapshots are made

    if frame_number < -1 or frame_number > len(time_stamps) - 1:
        raise ValueError(f"This JETTO file does not have a frame number: {frame_number}")
    t_index = len(time_stamps) - 1 if frame_number == -1 else frame_number

    rho = jsp["XPSQ"][t_index, :]  # Sqrt(poloidal magnetic flux)
    ion_temperature = jsp["TI"][t_index, :]
    d_density = jsp["NID"][t_index, :]
    t_density = jsp["NIT"][t_index, :]
    he3_density = np.zeros_like(rho)  # JETTO does not provide 3-He density

    # Here we treat the core, as JETTO at present does not provide data at rho = 0.0
    rho = np.insert(rho, 0, 0.0)
    ion_temperature = np.insert(ion_temperature, 0, ion_temperature[0])
    d_density = np.insert(d_density, 0, d_density[0])
    t_density = np.insert(t_density, 0, t_density[0])
    he3_density = np.insert(he3_density, 0, he3_density[0])

    ion_temperature = raw_uc(ion_temperature, "eV", "keV")

    # Cumulative vectors for fusion power and neutron rate
    dt_neutron_rate = jsp["R00"][t_index, -1]

    return SimpleJETTOOutput(
        rho=rho,
        ion_temperature=ion_temperature,
        d_density=d_density,
        t_density=t_density,
        he3_density=he3_density,
        dt_neutron_rate=dt_neutron_rate,
    )


def load_citation() -> dict:
    """
    Load the CITATION.cff file.

    Returns
    -------
    :
        The contents of the CITATION.cff file as a dictionary.
    """
    with open(get_tns_path("data") / "CITATION.cff") as citation_file:
        return yaml.safe_load(citation_file)


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


class WarningFilter:
    """
    Filters away duplicate log messages.
    Source: https://stackoverflow.com/questions/31953272/60462619#60462619
    """

    def __init__(self, *names: str):
        self.msgs = set()
        self.loggers = [logging.getLogger(name) for name in names]

    def filter(self, record):  # noqa: D102
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):  # noqa: D105
        for logger in self.loggers:
            logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D105
        for logger in self.loggers:
            logger.removeFilter(self)


class QuietTTSpectrumWarnings(WarningFilter):
    """Filter away all duplicate warnings from the energy and energy_data module."""

    def __init__(self):
        super().__init__(
            "tokamak_neutron_source.energy", "tokamak_neutron_source.energy_data"
        )
