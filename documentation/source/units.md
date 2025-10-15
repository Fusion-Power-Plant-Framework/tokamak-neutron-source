# Units and conventions

## Units

All length quantities are units of `[m]`, and all temperature / energy quantities are in `[keV]`. For time, we use `[s]`.

## Conventions

### Source strength

Following a common practice in neutronics, the fixed source strength is normalised to 1 `[neutrons/s]`. Tallies can then be multiplied by the `source_rate` to obtain total values, as and when required.

If you are using a "(n,Xt)" tally to calculate TBR, note that the definition of TBR is relative to the number of tritons consumed by the plasma, not the total number of fusion reactions.

To correctly scale your "(n,Xt)" tally in [1/particles], you should scale by:
    `tbr *= source_rate / source_T_rate`

to obtain the correct TBR. This is of course only relevant if you specify reactions in addition to the D-T reaction.

Note that the source strengths are not quantised; i.e., the rates are floats, not integers.

### Equilibrium coordinate

For the `FluxMap`, some description of the plasma magneto-hydrodynamic equilibrium is required. This is specified as the spatial distribution of poloidal magnetic flux, $\psi$, in the 2-D poloidal plane $(x, z)$. Here we use the normalised flux coordinate, $\psi_n$, which we define as being 0.0 at the magnetic axis, and 1.0 at the edge. The corresponding equilibrium normalised radial coordinate, $\rho$, commonly used in the description of plasma profiles, follows the same trend but is sometimes defined as being non-dimensional in another base unit: `[m]` instead of `[V.s]` or `[V.s/rad]`. The two are equivalent, and are effectively flux surface ``labels''; they uniquely describe a 2-D poloidal flux surface. We use $\psi_n$ when referring to a quantity in the 2-D poloidal plane, and $\rho$ when referring to a quantity in a 1-D profile.

Regardless of whether the `FluxMap` is provided via an EQDSK file or a parameterisation, the convention for the calculation of the normalised coordinate should be specified correctly. This can be done using the `FluxConvention` enum. Two conventions are available for the specification for the normalised poloidal magnetic flux ($\psi_n$):

* `FluxConvention.LINEAR`: $\psi_n = \dfrac{\psi_{a} - \psi}{\psi_{a} - \psi_{b}}$
* `FluxConvention.SQRT`: $\psi_n = \sqrt{\dfrac{\psi_{a} - \psi}{\psi_{a} - \psi_{b}}}$

In other words, when specifying a plasma profile values with some $\rho$ coordinate convention, the flux map should be initialised with the corresponding convention in $\psi_n$.

Note that if an equilibrium is loaded from a file, the COCOS convention is made irrelevant here: we treat $\psi$ such that it complies with the inner workings of the code.
