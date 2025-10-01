# Supported Neutron Transport Codes

## Exporting a neutron source

`tokamak_neutron_source` is designed to be able to create neutron sources for use in any neutron transport code. At present, interfaces for two codes are supported:

### OpenMC

`tokamak_neutron_source` is able to export a neutron source in native OpenMC format, via the specification of a list of `openmc.IndependentSource`. When creating the source, the default OpenMC units are used (eV and cm). Exercise caution if you have configured OpenMC to operate with a different set of units. This functionality requires you to have OpenMC installed in the same environment.

If multiple neutronic fusion reactions are specified, the energy distributions are combined into a single source at each point.

### MCNP-6

TODO: Add documentation when this functionality exists.
