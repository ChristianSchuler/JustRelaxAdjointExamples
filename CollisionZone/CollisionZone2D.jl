# 2-D visco-elasto-plastic continental collision
#
# This is a deliberately small forward model for developing an adjoint-ready
# collision-zone inverse problem. Two continental lithospheres converge across a
# weak upper-crustal suture. The model resolves layered material strength,
# elastic stress storage, and Drucker-Prager yielding, but keeps the geometry
# and temperature fixed. That makes the first parameter studies inexpensive and
# gives a well-defined target for an adjoint: e.g. minimise the misfit to an
# observed surface velocity or strain-rate profile by varying suture cohesion,
# lower-crust viscosity, or mantle-lithosphere viscosity.
#
# A closely related 2-D thermomechanical collision setup is:
# Liao, J. & Gerya, T. (2017), "Partitioning of crustal shortening during
# continental collision: 2-D thermomechanical modeling", JGR: Solid Earth,
# 122, 592-606. https://doi.org/10.1002/2016JB013398
#
# Run with this repository's environment:
#   julia CollisionZone/CollisionZone2D.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using GeoParams
using JustPIC
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend = JustRelax.CPUBackend
const pic_backend = JustPIC.CPU
@init_parallel_stencil(Threads, Float64, 2)

const YEAR = 365.25 * 24.0 * 3600.0

"""Fill the cell-centre, vertex and velocity-node phase ratios for a layered collision cross-section."""
function initialise_collision_phases!(phase_ratios, xci, xvi)
    @parallel_indices (i, j) function assign_phase!(phases, x, y)
        # Phase 1 is a 20 km wide inherited weak suture in the upper crust.
        phase = if y[j] > -35e3 && abs(x[i]) < 10e3
            1
        elseif y[j] > -35e3
            2 # upper crust
        elseif y[j] > -80e3
            3 # lower crust
        elseif y[j] > -180e3
            4 # mantle lithosphere
        else
            5 # asthenosphere
        end

        @index phases[1, i, j] = phase == 1 ? 1.0 : 0.0
        @index phases[2, i, j] = phase == 2 ? 1.0 : 0.0
        @index phases[3, i, j] = phase == 3 ? 1.0 : 0.0
        @index phases[4, i, j] = phase == 4 ? 1.0 : 0.0
        @index phases[5, i, j] = phase == 5 ? 1.0 : 0.0
        return nothing
    end

    # The CellArrays have the grid's shape; the phase index is the cell dimension.
    @parallel (@idx size(phase_ratios.center)) assign_phase!(phase_ratios.center, xci...)
    @parallel (@idx size(phase_ratios.vertex)) assign_phase!(phase_ratios.vertex, xvi...)
    @parallel (@idx size(phase_ratios.Vx)) assign_phase!(phase_ratios.Vx, xvi[1], xci[2])
    @parallel (@idx size(phase_ratios.Vy)) assign_phase!(phase_ratios.Vy, xci[1], xvi[2])
    return nothing
end

function collision_rheology()
    # Regularised plasticity keeps the forward problem differentiable enough for
    # the first adjoint experiments. Treat these values as priors, not estimates.
    function material(phase, density, viscosity, shear_modulus, cohesion, friction)
        elasticity = ConstantElasticity(; G = shear_modulus, Kb = 1.0e11)
        plasticity = DruckerPrager_regularised(;
            C = cohesion,
            ϕ = friction,
            Ψ = 0.0,
            η_vp = 1.0e19,
        )
        return SetMaterialParams(;
            Phase = phase,
            Density = ConstantDensity(; ρ = density),
            Gravity = ConstantGravity(; g = 9.81),
            CompositeRheology = CompositeRheology((
                LinearViscous(; η = viscosity), elasticity, plasticity,
            )),
            Elasticity = elasticity,
        )
    end

    return (
        material(1, 2700.0, 1.0e20, 2.0e10, 5.0e6, 10.0),  # weak suture
        material(2, 2750.0, 1.0e22, 3.0e10, 2.0e7, 30.0),  # upper crust
        material(3, 2900.0, 3.0e20, 3.0e10, 2.0e7, 15.0),  # lower crust
        material(4, 3300.0, 1.0e23, 6.0e10, 5.0e7, 30.0),  # lithosphere
        material(5, 3300.0, 1.0e20, 6.0e10, 1.0e9, 0.0),   # asthenosphere
    )
end

function plot_collision_state(stokes, phase_ratios, grid, figpath)
    (; xci) = grid
    # each cell holds a vector of phase fractions; show the dominant phase
    phases = [argmax(p) for p in Array(phase_ratios.center)]
    log_viscosity = log10.(Array(stokes.viscosity.η_vep))
    log_strain_rate = log10.(max.(Array(stokes.ε.II), eps()))

    fig = Figure(size = (1400, 850))
    panels = (
        (phases, "phase"),
        (log_viscosity, "log₁₀ effective viscosity [Pa s]"),
        (log_strain_rate, "log₁₀ strain rate [s⁻¹]"),
        (Array(stokes.EII_pl), "accumulated plastic strain"),
    )
    for (k, (field, title)) in enumerate(panels)
        # each panel takes two layout columns: the axis and its colorbar
        row, column = k <= 2 ? (1, 2k - 1) : (2, 2(k - 2) - 1)
        axis = Axis(fig[row, column], aspect = DataAspect(), title = title,
            xlabel = "x [km]", ylabel = "depth [km]")
        image = heatmap!(axis, xci[1] ./ 1e3, xci[2] ./ 1e3, field)
        Colorbar(fig[row, column + 1], image)
    end
    save(figpath, fig)
end

const phase_names = (:suture, :upper_crust, :lower_crust, :mantle_lithosphere, :asthenosphere)

function save_collision_vtk!(vtk_dir, step, time, grid, stokes, ρg, phase_ratios)
    mkpath(vtk_dir)
    time_Myr = time / (1.0e6 * YEAR)
    Vx_v = @zeros(grid.ni .+ 1...)
    Vy_v = @zeros(grid.ni .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    data_v = (; Vx = Array(Vx_v), Vy = Array(Vy_v))
    phase_center = Array(phase_ratios.center)
    fractions = NamedTuple(
        Symbol(name, "_fraction") => getindex.(phase_center, p) for (p, name) in enumerate(phase_names)
    )
    data_c = (
        ; pressure = Array(stokes.P),
        viscosity = Array(stokes.viscosity.η),
        effective_viscosity = Array(stokes.viscosity.η_vep),
        density = Array(ρg[2]) ./ 9.81,
        stress_invariant = Array(stokes.τ.II),
        strain_rate_invariant = Array(stokes.ε.II),
        plastic_strain_rate_invariant = Array(stokes.ε_pl.II),
        accumulated_plastic_strain = Array(stokes.EII_pl),
        phase = [argmax(p) for p in phase_center],
        fractions...,
    )
    save_vtk(
        joinpath(vtk_dir, "collision_" * lpad(step, 6, '0')), grid.xvi, grid.xci, data_v, data_c,
        (Array(Vx_v), Array(Vy_v)); t = time_Myr, pvd = joinpath(vtk_dir, "collision"),
    )
    return nothing
end

function main(;
        # 250 × 2 kyr = 0.5 Myr ≈ 10 Maxwell times of the mantle lithosphere (η/G ≈ 53 kyr)
        nx = 192, ny = 96, nsteps = 250, nout = 25,
        output = joinpath(@__DIR__, "CollisionZone2D.png"),
        vtk_dir = joinpath(@__DIR__, "vtk_collision"),
    )
    nout ≥ 1 || throw(ArgumentError("nout must be positive"))
    # The 800 x 240 km box is a 2-D cross-section normal to the suture.
    li = (800e3, 240e3)
    origin = (-400e3, -240e3)
    ni = (nx, ny)
    di = li ./ ni
    grid = Geometry(ni, li; origin)
    (; xci, xvi) = grid

    rheology = collision_rheology()
    phase_ratios = PhaseRatios(pic_backend, length(rheology), ni)
    initialise_collision_phases!(phase_ratios, xci, xvi)

    # There is no air, so the whole domain is rock.
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, 0)

    stokes = StokesArrays(backend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    dt = 2.0e3 * YEAR
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)

    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    compute_lithostatic_pressure!(stokes.P, ρg[2], di[2], igg)
    viscosity_cutoff = (1.0e18, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff)

    # Move both plates together at 2 cm/yr. The normal velocity on the left and right
    # walls is prescribed; free slip lets the material slide along every wall, and the
    # free-slip top and bottom allow vertical accommodation of shortening.
    half_convergence = 0.01 / YEAR
    stokes.V.Vx .= PTArray(backend)([
        -half_convergence * x / (li[1] / 2) for x in xvi[1], _ in 1:(ny + 2)
    ])
    stokes.V.Vy .= 0.0
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; γfact = 20.0, ϵ = 1.0e-5)
    for step in 1:nsteps
        solve_VariationalDYREL!(
            stokes, nothing, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 50_000,
            total_iterMax = 100_000,
            nout = 100,
            rel_drop = 1.0e-2,
            viscosity_relaxation = 1.0e-2,
            viscosity_cutoff,
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        println("step $step/$nsteps: t = $(step * dt / (1e6 * YEAR)) Myr")
        (step % nout == 0 || step == nsteps) &&
            save_collision_vtk!(vtk_dir, step, step * dt, grid, stokes, ρg, phase_ratios)
    end

    plot_collision_state(stokes, phase_ratios, grid, output)
    return nothing
end

if !@isdefined(NO_AUTORUN)
    Base.invokelatest(main)
end
