# Introduction

The `tokamak_neutron_source` package provides a method of generating neutron sources for tokamak fusion device.

Neutron sources can be generated from EQDSK files and custom profiles for export to neutron transport codes.

# Installation

We don't try to manage the installation of your neutronics codes. We recommend you install your neutronics code first. If you are using `tokamak-neutron-source` to create an OpenMC source a simple install of OpenMC using `conda` can be done with:

```bash
conda install -c conda-forge 'openmc>=0.15.0'
```

To install the latest release of `tokamak-neutron-source`

```bash
pip install tokamak-neutron-source
```

## Contents
- [Theory](theory.md)
- [Plasma Profiles](profiles.md)
- [Equilibrium Information](equilibrium.md)
- [Units and Conventions](units.md)
- [Supported Neutron Transport Codes](codes.md)
