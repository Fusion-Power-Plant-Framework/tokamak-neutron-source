# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-FileCopyrightText: 2024-present Tokamak Neutron Source Maintainers
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Example Reading from JETTO files"""

# %%
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from numpy import typing as npt

from tokamak_neutron_source import (
    FluxConvention,
    FluxMap,
    TokamakNeutronSource,
    TransportInformation,
)
from tokamak_neutron_source.constants import raw_uc

# %% [markdown]
# ## Load an arbitrary source.
# For this example the same source as from_jetto.ex.py is used. One can equally use the
# source from from_eqdsk.ex.py.

# %%
CELL_SIDE_LENGTH = 0.3  # [m]
source = TokamakNeutronSource(
    transport=TransportInformation.from_jetto("tests/test_data/STEP_jetto.jsp"),
    flux_map=FluxMap.from_eqdsk(
        "tests/test_data/STEP_jetto.eqdsk_out", flux_convention=FluxConvention.SQRT
    ),
    cell_side_length=0.3,
)
openmc_source = source.to_openmc_source()

# %% [markdown]
# ## Define location for temporary files to be saved to.

# %%
examples_path = Path("examples/").resolve()
geometry_path = examples_path / "geometry.xml"
materials_path = examples_path / "materials.xml"
settings_path = examples_path / "settings.xml"
tracks_path = examples_path / "tracks.h5"

# %% [markdown]
# ## Set up the geometry

# %%
bot_z = raw_uc(source.z.min() - CELL_SIDE_LENGTH / 2, "m", "cm")
top_z = raw_uc(source.z.max() + CELL_SIDE_LENGTH / 2, "m", "cm")
radius = raw_uc(source.x.max() + CELL_SIDE_LENGTH / 2, "m", "cm")
bot = openmc.ZPlane(bot_z, boundary_type="vacuum")
top = openmc.ZPlane(top_z, boundary_type="vacuum")
cyl = openmc.ZCylinder(r=radius, boundary_type="vacuum")
source_cell = openmc.Cell(region=+bot & -top & -cyl, fill=None, name="source cell")
universe = openmc.Universe(cells=[source_cell])
geometry = openmc.Geometry(universe)
geometry.export_to_xml(geometry_path)

# %% [markdown]
# ## Set up the materials

# %%
materials = openmc.Materials()
materials.cross_sections = "tests/test_data/cross_section.xml"
materials.export_to_xml(materials_path)

# %% [markdown]
# ## Set simulation settings

# %%
NUM_BATCHES = 1
settings = openmc.Settings(
    batches=NUM_BATCHES,
    run_mode="fixed source",
    output={"path": examples_path.as_posix(), "summary": False},
)
settings.seed = 1
settings.source = openmc_source
settings.particles = settings.max_tracks = 1000
settings.export_to_xml(settings_path)

# %% [markdown]
# ## Run the simulation and save the results

# %%
openmc.run(cwd=examples_path.as_posix(), tracks=True)
tracks = openmc.Tracks(tracks_path)

# %% [markdown]
# ## Delete the temporarily created xml input files

# %%
geometry_path.unlink(missing_ok=True)
materials_path.unlink(missing_ok=True)
settings_path.unlink(missing_ok=True)
Path(examples_path / f"statepoint.{NUM_BATCHES}.h5").unlink(missing_ok=True)
tracks_path.unlink(missing_ok=True)


# %% [markdown]
# ## Making sense of the results
# `openmc.Tracks` stores an iterable of openmc.Track, each of which has a property
# particle_tracks, which records the particle's position, direction, and energy etc.
#
# particle_tracks would have length >1 if there are more than one track per source
# particle, but due to the lack of obstacles in this simulation,
# len(particle_tracks) always ==1.


# %%
@dataclass
class OpenMCTrack:
    """Extract the state into a more readable format."""

    position: float
    direction: float
    energy: float  # eV
    time: float
    wgt: float
    cell_id: int
    cell_instance: int
    mat_id: int

    @property
    def position_cylindrical(self) -> tuple[np.float64, np.float64, np.float64]:
        """Turn position into cylindrical coordinate for ease of plotting."""
        return xyz_to_rphiz(*self.position)

    @property
    def direction_spherical(self) -> tuple[np.float64, np.float64]:
        """
        Condense thes direction (3D) into a spherical (2D) coordinate representation
        """
        r, phi, z = xyz_to_rphiz(*self.direction)
        theta = np.atan2(z, r)
        return theta, phi


@dataclass
class OpenMCSimulatedSourceParticles:
    """
    Important information about where the particle was born and how it was travelling.
    """

    source: openmc.Source
    locations: npt.NDArray
    directions: npt.NDArray
    energies: npt.NDArray


def xyz_to_rphiz(x, y, z) -> tuple[np.float64, np.float64, np.float64]:
    """Convert cartesian into cylindrical coordinates."""
    r = np.sqrt(x**2 + y**2)
    phi = np.atan2(x, y)
    return r, phi, z


locations, directions, energies = [], [], []
for ptrac in tracks:
    start_state = OpenMCTrack(*ptrac.particle_tracks[0].states[0])
    locations.append(start_state.position_cylindrical)
    directions.append(start_state.direction_spherical)
    energies.append(start_state.energy)
locations = np.array(locations)
directions = np.array(directions)
energies = np.array(energies)

# %% [markdown]
# ## Plot the location where neutrons are emitted

# %%
ax = plt.axes()
r, phi, z = locations.T
ax.scatter(r / 100, z / 100, alpha=0.3, marker="o", s=0.5)
ax.set_xlabel("r (m)")
ax.set_ylabel("z (m)")
ax.set_title(
    "Neutron generation positions\n(poloidal view)\nEach dot is a neutron emitted"
)
o_point, lcfs = source.flux_map.o_point, source.flux_map.lcfs
ax.scatter(o_point.x, o_point.z, label="o-point", facecolors="none", edgecolor="C1")
ax.plot(lcfs.x, lcfs.z, label="LCFS")
ax.legend()
ax.set_aspect("equal")
plt.show()
