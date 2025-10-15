[![DOI](https://zenodo.org/badge/1067705955.svg)](https://zenodo.org/badge/latestdoi/1067705955)

# Introduction

`tokamak-neutron-source` is a package that provides a flexible and high-fidelity fusion neutron source for tokamaks in OpenMC and other Monte Carlo radiation transport codes.

# Installation

We don't try to manage the installation of your neutronics codes. We recommend you install your neutronics code first. If you are using `tokamak-neutron-source` to create an OpenMC source you can create a simple install of OpenMC using `conda` with:

```bash
conda install -c conda-forge 'openmc>=0.15.0'
```

To install the latest release of `tokamak-neutron-source`

```bash
pip install tokamak-neutron-source
```

# Inputs

A tokamak neutron source can be created by specifing the plasma ion density and temperature profiles, and a description of the plasma magneto-hydrodynamic equilibrium.

Profiles can be specified in terms of arrays or as typical parameterisations, such as a parabolic-pedestal parameterisation.

Equilibrium information can be specified via an EQDSK file or as a parameterisation, such as the one found in [Fausser et al., 2012](https://www.sciencedirect.com/science/article/abs/pii/S0920379612000853).

# Outputs

A source object can be used to create an idiomatic source for use in [OpenMC](https://openmc.org/) or exported as an sdef or h5 file for use in OpenMC and [MCNP6](https://mcnp.lanl.gov/).

A neutron source from some typical parameterised profiles and a Fausser flux surface parameterisation:

![](documentation/source/fausser_source.svg)

A neutron source from some arbitrary profiles and a free-boundary equilibrium:

![](documentation/source/eqdsk_source.svg)
