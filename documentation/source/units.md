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

For the `FluxMap`, some description of the plasma magneto-hydrodynamic equilibrium is required. Regardless of whether this is provided via an EQDSK file or a parameterisation, the convention for the calculation of the normalised coordinate, should be specified correctly. This can be done using the `FluxConvention` enum. Here we use the normalised flux coordinate, $\psi_n$, which we define as being 1.0 at the magnetic axis, and 0.0 at the edge. The corresponding equilibrium radial coordinate, $\rho$, follows the opposite trend; 0.0 at the magnetic axis, and 1.0 at the edge.

Two methods are available for the specification for the poloidal magnetic flux ($\psi$):

* `FluxConvention.LINEAR`: $\psi_n = \dfrac{\psi_{a} - \psi}{\psi_{a} - \psi_{b}}$
* `FluxConvention.SQRT`: $\psi_n = \sqrt{\dfrac{\psi_{a} - \psi}{\psi_{a} - \psi_{b}}}$

Note that if an equilibrium is loaded from a file, the COCOS convention is made irrelevant here: we treat the flux map such that it complies with the inner workings of the code.
