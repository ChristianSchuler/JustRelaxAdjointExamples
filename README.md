# Dora Maira vertical weak-zone example

`DoraMairaVerticalWeakZone.jl` is a compact 2-D, particle-in-cell DYREL model inspired by
Schmalholz & Schenker (2016), *Exhumation of the Dora Maira ultrahigh-pressure unit by
buoyant uprise within a low-viscosity mantle oblique-slip shear zone*,
[doi:10.1111/ter.12227](https://doi.org/10.1111/ter.12227).

It contains a buoyant elliptical UHP crustal body inside a dense, low-viscosity mantle shear
zone.  The requested variant starts with that zone vertical and centred at `x = 0`; the paper's
conceptual model instead uses an inclined weak layer.  Periodic horizontal boundaries and
oppositely moving top and bottom walls impose 2-D simple shear.  This represents the in-plane
component of oblique slip; a true strike-slip component requires a 3-D model.

Run the demonstration from the repository root:

```sh
julia --project=miniapps --startup-file=no JustRelaxAdjointExamples/DoraMairaVerticalWeakZone.jl
```

The final Stokes solve optionally runs the DYREL adjoint.  Its objective is the sum of vertical
velocity in a small box around the initial UHP body, and the saved figure shows the spatial
sensitivity to the weak-zone viscosity.  Phase transport between time steps is intentionally not
differentiated, so this is a sensitivity of the final instantaneous flow problem, not a gradient
through the full exhumation history.

The material values are illustrative, not a calibrated recreation of the paper.  In particular,
the setup preserves the paper's density/viscosity contrast mechanism while making the weak-zone
orientation explicit and easy to change.
