# Dora Maira vertical weak-zone example

`DoraMairaVerticalWeakZone.jl` is a compact 2-D, particle-in-cell DYREL model inspired by
Schmalholz & Schenker (2016), *Exhumation of the Dora Maira ultrahigh-pressure unit by
buoyant uprise within a low-viscosity mantle oblique-slip shear zone*,
[doi:10.1111/ter.12227](https://doi.org/10.1111/ter.12227).

It contains a buoyant elliptical UHP crustal body inside a dense, low-viscosity mantle shear
zone. The requested variant starts with that zone vertical and centred at `x = 0`; the paper's
conceptual model instead uses an inclined weak layer. All model boundaries are free slip.

Run the demonstration from the repository root:

```sh
julia --project=. --startup-file=no SubductionShearZone/DoraMairaVerticalWeakZone.jl
```

Set `const isCUDA = false` at the top of the script for CPU (Threads), or `true` for CUDA,
as in the other JustRelax miniapps. Restart Julia when switching backends.

By default, every plotted Stokes solve runs the DYREL adjoint. Its objective is the mean vertical
velocity of the current UHP phase, so the observation follows the body as its particles move and
deform. A separate 3×3 figure shows the relative sensitivities `p ∂J/∂p` to density, the
dislocation-creep prefactor `A`, and stress exponent `n` for all three phases. Phase transport
between time steps is intentionally not differentiated, so each result is the sensitivity of the
instantaneous flow problem at that step, rather than a gradient through the full history.

The material values are illustrative, not a calibrated recreation of the paper.  In particular,
the setup preserves the paper's density/viscosity contrast mechanism while making the weak-zone
orientation explicit and easy to change.
